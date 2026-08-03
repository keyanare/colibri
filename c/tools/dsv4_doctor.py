#!/usr/bin/env python3
"""Can this machine run DeepSeek-V4-Flash? — a readiness check for `deepseek_v4`.

    python3 tools/dsv4_doctor.py                    # check the machine alone
    python3 tools/dsv4_doctor.py --model /nvme/dsv4 # also check the checkpoint
    python3 tools/dsv4_doctor.py --bench            # measure real disk read speed

Answers three questions, in the order they actually block you:

  1. Does the checkpoint FIT?      ~149 GiB, and it cannot be compressed further
                                   (the experts are already 4.25 bits/weight).
  2. Does the resident set FIT?    the dense weights must stay in RAM; what is
                                   left over becomes the expert cache.
  3. How SLOW will it be?          this engine is I/O-bound. The estimate below
                                   comes from your measured RAM and disk, not
                                   from a table.

Exit status is 0 when the model can run, 1 when something blocks it. Warnings
alone do not fail: a machine that will be slow is still a machine that runs.

NOTE. `deepseek_v4` has never been run against the real checkpoint (see
docs/deepseek_v4.md). This script checks whether your hardware COULD host it;
it does not certify that the engine is correct.
"""
import argparse
import json
import os
import platform
import shutil
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
CDIR = os.path.dirname(HERE)
sys.path.insert(0, CDIR)

try:
    from resource_plan import memory_available, physical_cpu_count
except Exception:                                    # keep working standalone
    def memory_available():
        try:
            import re
            from pathlib import Path
            t = Path("/proc/meminfo").read_text()
            return int(re.search(r"MemAvailable:\s+(\d+)", t).group(1)) * 1024
        except Exception:
            return 0
    def physical_cpu_count():
        return os.cpu_count() or 1

GB, GIB = 1e9, float(1 << 30)


def _sz(n):
    return f"{n/GB:.1f} GB" if n >= GB else f"{n/1e6:.0f} MB"

# Reference geometry, from the published config.json. Used when no --model is
# given; every number below is recomputed from the real config when one is.
REF = dict(n_layers=43, dim=4096, n_heads=64, head_dim=512, q_lora_rank=1024,
           o_lora_rank=1024, o_groups=8, n_routed_experts=256,
           num_experts_per_tok=6, moe_intermediate_size=2048,
           n_shared_experts=1, vocab_size=129280, hc_mult=4,
           index_n_heads=64, index_head_dim=128, sliding_window=128,
           compress_ratios=[0, 0] + [4, 128] * 20 + [4, 0])

OK, WARN, FAIL = "ok", "warn", "FAIL"
_MARK = {OK: "  [ ok ]", WARN: "  [warn]", FAIL: "  [FAIL]"}


class Report:
    def __init__(self):
        self.rows, self.failed = [], False

    def add(self, level, what, detail):
        self.rows.append((level, what, detail))
        if level == FAIL:
            self.failed = True

    def section(self, title):
        self.rows.append((None, title, None))

    def render(self):
        for level, what, detail in self.rows:
            if level is None:
                print(f"\n{what}")
                continue
            print(f"{_MARK[level]} {what}")
            if detail:
                for line in detail.splitlines():
                    print(f"         {line}")


# ---------------------------------------------------------------- model math

# config.json spells several of these differently from the short names used
# above. Without the alias map the merge below silently keeps the reference
# defaults -- which happen to be right for the published checkpoint and wrong
# for anything else, i.e. a bug that hides exactly where it is least welcome.
ALIASES = {"hidden_size": "dim", "num_hidden_layers": "n_layers",
           "num_attention_heads": "n_heads"}


def load_cfg(model_dir):
    if not model_dir:
        return dict(REF), None
    path = os.path.join(model_dir, "config.json")
    if not os.path.exists(path):
        return dict(REF), f"{path} not found — using the published reference geometry"
    with open(path, encoding="utf-8") as f:
        cfg = json.load(f)
    merged = dict(REF)
    for k, v in cfg.items():
        key = ALIASES.get(k, k)
        if key in merged:
            merged[key] = v
    if len(merged["compress_ratios"]) < merged["n_layers"]:
        return merged, (f"compress_ratios has {len(merged['compress_ratios'])} entries "
                        f"but num_hidden_layers={merged['n_layers']}")
    return merged, None


def expert_bytes(c):
    """One routed expert on disk: w1+w3+w2 as mxfp4 nibbles + ue8m0 scales."""
    MI, D = c["moe_intermediate_size"], c["dim"]
    return (2 * (MI * ((D + 1) // 2) + MI * ((D + 31) // 32))
            + (D * ((MI + 1) // 2) + D * ((MI + 31) // 32)))


def unquantized_params(c):
    """Dense parameters the checkpoint ships UNQUANTIZED: the lm head and the
    compressor projections. bf16 on disk, expanded to f32 in RAM, so the two
    accountings differ by 2x and neither the fp8 nor the small-tensor bucket
    fits. Both were costed as fp8 until the published checkpoint was read --
    together ~0.9B parameters, i.e. 0.9 GB assumed against 3.5 GB resident."""
    return c["vocab_size"] * c["dim"] + compressor_params(c)


def compressor_params(c):
    """Parameters in the compressor projections: wkv + wgate, for the attention
    compressor and the indexer's own.

    They get their own function because they are the one dense pair in this
    model that ships UNQUANTIZED -- bf16 on disk, expanded to f32 in RAM -- so
    disk and resident disagree about them by 2x and neither bucket above fits.
    At ~390M parameters that is the difference between a 0.4 GB line item and a
    1.6 GB one, which is not a rounding error on a 16 GB machine."""
    D, HD = c["dim"], c["head_dim"]
    n = 0
    for r in c["compress_ratios"][:c["n_layers"]]:
        if not r:
            continue
        n += 2 * ((2 if r == 4 else 1) * HD * D)
        if r == 4:
            n += 2 * (2 * c["index_head_dim"] * D)
    return n


def resident_bytes(c, max_seq=8192):
    """What the engine must hold in RAM before any expert is cached.

    Mirrors deepseek_v4.c: quantized dense weights are fp8 (1 byte/param), the
    unquantized ones and the small tensors it keeps as f32 are 4, embed.weight
    is NOT resident (one row is read per token), and the KV/compressed caches
    scale with max_seq."""
    D, L = c["dim"], c["n_layers"]
    HD, H = c["head_dim"], c["n_heads"]
    hs = H * HD
    hcmix = (2 + c["hc_mult"]) * c["hc_mult"]
    ratios = c["compress_ratios"][:L]

    fp8 = 0
    fp8 += L * (c["q_lora_rank"] * D + hs * c["q_lora_rank"] + HD * D
                + c["o_groups"] * c["o_lora_rank"] * (hs // c["o_groups"])
                + D * c["o_groups"] * c["o_lora_rank"])
    MI = c["moe_intermediate_size"]
    fp8 += L * c["n_shared_experts"] * (2 * MI * D + D * MI)       # shared expert
    for r in ratios:
        if r == 4:
            fp8 += c["index_n_heads"] * c["index_head_dim"] * c["q_lora_rank"]

    f32 = unquantized_params(c)                                     # lm head + compressors
    f32 += L * 2 * hcmix * c["hc_mult"] * D                         # hc_*_fn
    f32 += L * c["n_routed_experts"] * D                            # routers
    f32 += L * (2 * D + c["q_lora_rank"] + HD)                      # norms
    f32 += sum(c["index_n_heads"] * D for r in ratios if r == 4)    # weights_proj

    caches = 0
    for r in ratios:
        rows = c["sliding_window"] + (max_seq // r + 1 if r else 0)
        caches += rows * HD * 4
        if r == 4:
            caches += (max_seq // 4 + 1) * c["index_head_dim"] * 4
    return fp8 + f32 * 4 + caches


def disk_bytes(c):
    """Whole checkpoint: experts dominate at 97%."""
    L = c["n_layers"]
    # resident_bytes holds the unquantized dense weights as f32 because that is
    # what the engine expands them to; on disk they are bf16, hence -2/param.
    return int(L * c["n_routed_experts"] * expert_bytes(c)
               + resident_bytes(c, max_seq=0) - 2 * unquantized_params(c)
               + c["vocab_size"] * c["dim"] * 2)


# ---------------------------------------------------------------- host probes

def total_ram():
    if sys.platform == "darwin":
        try:
            return int(subprocess.check_output(["sysctl", "-n", "hw.memsize"]))
        except Exception:
            return 0
    try:
        import re
        from pathlib import Path
        t = Path("/proc/meminfo").read_text()
        return int(re.search(r"MemTotal:\s+(\d+)", t).group(1)) * 1024
    except Exception:
        pass
    if sys.platform == "win32":
        try:
            import ctypes
            class MS(ctypes.Structure):
                _fields_ = [("dwLength", ctypes.c_ulong), ("dwMemoryLoad", ctypes.c_ulong),
                            ("ullTotalPhys", ctypes.c_ulonglong), ("ullAvailPhys", ctypes.c_ulonglong),
                            ("ullTotalPageFile", ctypes.c_ulonglong), ("ullAvailPageFile", ctypes.c_ulonglong),
                            ("ullTotalVirtual", ctypes.c_ulonglong), ("ullAvailVirtual", ctypes.c_ulonglong),
                            ("ullAvailExtendedVirtual", ctypes.c_ulonglong)]
            m = MS(); m.dwLength = ctypes.sizeof(MS)
            ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(m))
            return int(m.ullTotalPhys)
        except Exception:
            return 0
    return 0


def cpu_features():
    """Returns (name, set-of-flags). Flags are lowercase, best-effort."""
    name, flags = platform.processor() or platform.machine(), set()
    if sys.platform == "darwin":
        try:
            name = subprocess.check_output(
                ["sysctl", "-n", "machdep.cpu.brand_string"], text=True).strip()
        except Exception:
            pass
        if platform.machine() == "arm64":
            flags.add("neon")
        else:
            try:
                f = subprocess.check_output(
                    ["sysctl", "-n", "machdep.cpu.features",
                     "machdep.cpu.leaf7_features"], text=True).lower()
                flags.update(f.split())
            except Exception:
                pass
    else:
        try:
            from pathlib import Path
            for line in Path("/proc/cpuinfo").read_text().splitlines():
                if line.startswith("model name") and ":" in line:
                    name = line.split(":", 1)[1].strip()
                if line.startswith("flags") or line.startswith("Features"):
                    flags.update(line.split(":", 1)[1].split())
        except Exception:
            pass
        if platform.machine() in ("aarch64", "arm64"):
            flags.add("neon")
    return name, flags


def measure_disk(path, mb=512):
    """Read throughput on the volume that will hold the model.

    Writes a temp file, drops what cache it can, reads it back. Not a
    benchmark-grade number, but far better than assuming."""
    tmp = os.path.join(path, ".dsv4_doctor_bench.tmp")
    buf = os.urandom(1 << 20)
    try:
        with open(tmp, "wb") as f:
            for _ in range(mb):
                f.write(buf)
            f.flush()
            os.fsync(f.fileno())
        # Bypass the page cache, or this measures RAM. `purge` needs sudo and
        # silently does nothing without it, which is how the first version of
        # this reported 11.9 GB/s on a 5 GB/s SSD.
        got = 0
        with open(tmp, "rb") as f:
            fd = f.fileno()
            if sys.platform == "darwin":
                try:
                    import fcntl
                    fcntl.fcntl(fd, 48, 1)          # F_NOCACHE
                except Exception:
                    pass
            elif hasattr(os, "posix_fadvise"):
                try:
                    os.posix_fadvise(fd, 0, mb << 20, os.POSIX_FADV_DONTNEED)
                except OSError:
                    pass
            t0 = time.monotonic()
            while True:
                b = f.read(1 << 22)
                if not b:
                    break
                got += len(b)
            dt = time.monotonic() - t0
        return got / dt if dt > 0 else 0.0
    except OSError:
        return 0.0
    finally:
        try:
            os.unlink(tmp)
        except OSError:
            pass


# ---------------------------------------------------------------- checks

def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", help="checkpoint directory (also validates it)")
    ap.add_argument("--target", help="where the model WILL live, if not --model")
    ap.add_argument("--bench", action="store_true",
                    help="measure disk read speed instead of assuming")
    ap.add_argument("--expert-gb", type=float, default=None,
                    help="the budget you intend to pass to the engine")
    ap.add_argument("--max-seq", type=int, default=8192)
    a = ap.parse_args()

    cfg, cfg_note = load_cfg(a.model)
    r = Report()
    eb = expert_bytes(cfg)
    resident = resident_bytes(cfg, a.max_seq)
    need_disk = disk_bytes(cfg)
    per_token = cfg["num_experts_per_tok"] * cfg["n_layers"]

    print("DeepSeek-V4-Flash readiness check")
    print(f"  {cfg['n_layers']} layers, {cfg['n_routed_experts']} experts/layer, "
          f"top-{cfg['num_experts_per_tok']}, one expert = {eb/GIB*1024:.1f} MiB")
    if cfg_note:
        print(f"  note: {cfg_note}")

    # ---- disk
    r.section("Disk")
    where = a.model or a.target or os.getcwd()
    try:
        du = shutil.disk_usage(where)
    except OSError as e:
        r.add(FAIL, f"cannot stat {where}", str(e))
        du = None
    if du:
        free_gib = du.free / GIB
        need_gib = need_disk / GIB
        detail = (f"checkpoint needs ~{need_gib:.0f} GiB ({need_disk/GB:.0f} GB)\n"
                  f"free on {where}: {free_gib:.0f} GiB")
        if a.model:
            detail += "\n(the model already lives here; free space is what is LEFT)"
            r.add(OK if free_gib > 5 else WARN, "space on the model volume", detail)
        elif free_gib >= need_gib * 1.05:
            r.add(OK, "enough space for the checkpoint", detail)
        elif free_gib >= need_gib:
            r.add(WARN, "space is tight", detail + "\nno headroom for anything else")
        else:
            r.add(FAIL, "not enough space",
                  detail + f"\nshort by {need_gib - free_gib:.0f} GiB. The experts are "
                           "already 4.25 bits/weight — this cannot be shrunk.")

    # ---- ram
    r.section("Memory")
    tot, avail = total_ram(), memory_available()
    res_gb = resident / GB
    detail = (f"engine resident set: ~{res_gb:.1f} GB (dense weights + caches at "
              f"--max-seq {a.max_seq})\n"
              f"RAM total: {tot/GB:.1f} GB" + (f", available now: {avail/GB:.1f} GB" if avail else ""))
    # CAPACITY decides pass/fail; what is free RIGHT NOW is a separate, softer
    # signal. Failing because a browser is open would be wrong -- the user can
    # close it; failing because the machine has 8 GB of RAM would not.
    capacity = tot * 0.80 - resident          # 20% for the OS and page cache
    if tot == 0:
        r.add(WARN, "could not read total RAM", detail)
        capacity = 0
    elif capacity < 0:
        r.add(FAIL, "not enough RAM for the resident set", detail +
              f"\nthe dense weights alone need ~{res_gb:.1f} GB and must stay resident.\n"
              "Lower --max-seq to shrink the caches, but the weights are fixed.")
    elif capacity < 2 * GB:
        r.add(WARN, "RAM is very tight", detail +
              f"\n~{capacity/GB:.1f} GB would be left for the expert cache: every token misses.")
    else:
        r.add(OK, "resident set fits", detail +
              f"\n~{capacity/GB:.1f} GB available for the expert cache")
    if avail and tot and avail < resident + 1 * GB:
        r.add(WARN, "not enough free RAM right now",
              f"{avail/GB:.1f} GB free, the engine needs ~{res_gb:.1f} GB resident.\n"
              "Close other applications before starting, or it will swap.")

    budget = a.expert_gb if a.expert_gb is not None else max(0.5, min(capacity / GB * 0.6, 24.0))
    slots = int(budget * GB / eb)
    per_layer = slots / cfg["n_layers"]
    hit = min(0.95, per_layer / cfg["n_routed_experts"])
    r.add(OK if per_layer >= 1 else WARN, f"expert cache at --expert-gb {budget:.0f}",
          f"{slots} slots = {slots/cfg['n_layers']:.0f} experts/layer of {cfg['n_routed_experts']}\n"
          f"a token routes {per_token} experts ({per_token*eb/GB:.1f} GB); "
          f"expected hit rate ~{hit*100:.0f}%")

    # ---- cpu
    r.section("CPU")
    name, flags = cpu_features()
    phys = physical_cpu_count()
    logical = os.cpu_count() or phys
    r.add(OK, f"{name}", f"{phys} physical cores, {logical} logical")
    if "avx2" in flags:
        r.add(OK, "AVX2 present", "matmul_mxfp4 uses its vector path (build with ARCH=native)")
    elif "neon" in flags:
        r.add(OK, "NEON present", "matmul_mxfp4 uses its NEON path (I%32==0, every DSV4 expert)")
    else:
        r.add(WARN, "no AVX2/NEON detected", "expert matmuls will run scalar")
    omp_ok = True
    if sys.platform == "darwin":
        has_omp = bool(shutil.which("brew")) and os.path.exists(
            os.path.join(subprocess.run(["brew", "--prefix", "libomp"], capture_output=True,
                                        text=True).stdout.strip() or "/nonexistent", "lib"))
        # WARN, not FAIL: single-threaded is slow, not impossible, and this
        # script promises that only real blockers change the exit status.
        r.add(OK if has_omp else WARN, "OpenMP runtime",
              "libomp found — the build will be multi-threaded" if has_omp else
              "libomp NOT installed: the engine builds SINGLE-THREADED and is roughly\n"
              f"{phys}x slower. Fix: brew install libomp")
        omp_ok = has_omp

    # ---- speed
    r.section("Expected speed")
    assumed = 5.0 * GB if sys.platform == "darwin" else 2.0 * GB
    if a.bench and du:
        got_bw = measure_disk(where)
        if got_bw > 8 * GB:
            # The benchmark file was just written, so its pages are resident;
            # F_NOCACHE stops FUTURE caching but does not evict. Rather than
            # report a RAM speed as a disk speed, say so and estimate with the
            # conservative figure -- an estimate built on 15 GB/s would promise
            # a throughput no SSD in this class delivers.
            bw, src = assumed, (f"measured {got_bw/GB:.1f} GB/s, which is page cache, "
                                f"not disk — estimating with {assumed/GB:.0f} GB/s")
        elif got_bw > 0:
            bw, src = got_bw, "measured"
        else:
            bw, src = assumed, "assumed (measurement failed)"
    else:
        bw = assumed
        src = "assumed (pass --bench to measure)"
    miss = per_token * (1 - hit)
    io_s = miss * eb / bw if bw else 0
    # ~3.5G weights/s/core scalar, ~12G with AVX2; 13.3B active params per token
    active = 13.3e9
    threads = phys if omp_ok else 1
    rate = (12e9 if "avx2" in flags else 3.5e9) * threads
    cpu_s = active / rate
    total_s = io_s + cpu_s
    r.add(OK, "estimate",
          f"disk: {bw/GB:.1f} GB/s ({src})\n"
          f"per token: ~{miss:.0f} expert misses = {miss*eb/GB:.1f} GB read -> {io_s:.1f} s\n"
          f"compute:   ~{cpu_s:.1f} s ({threads} thread{'s' if threads > 1 else ''})\n"
          f"TOTAL:     ~{total_s:.1f} s/token  ({1/total_s:.2f} tok/s)")
    if total_s > 10:
        r.add(WARN, "this will be very slow", "usable for experiments, not for chat")

    # ---- checkpoint
    if a.model:
        r.section("Checkpoint")
        for f in ("config.json",):
            r.add(OK if os.path.exists(os.path.join(a.model, f)) else FAIL,
                  f, None)
        tok = os.path.join(a.model, "tokenizer.json")
        r.add(OK if os.path.exists(tok) else FAIL, "tokenizer.json",
              None if os.path.exists(tok) else
              "the published checkpoint ships none — install one with\n"
              "  python3 tools/dsv4_tokenizer.py <model_dir> --from <tokenizer.json>")
        shards = [f for f in os.listdir(a.model) if f.endswith(".safetensors")]
        got = sum(os.path.getsize(os.path.join(a.model, f)) for f in shards)
        r.add(OK if got >= need_disk * 0.95 else WARN,
              f"{len(shards)} safetensors shards",
              f"{_sz(got)} present, expected ~{_sz(need_disk)}"
              + ("" if got >= need_disk * 0.95 else "\ndownload may be incomplete"))

    r.render()
    print()
    if r.failed:
        print("VERDICT: blocked — fix the [FAIL] items above.")
        return 1
    print(f"VERDICT: this machine can run it, at roughly {total_s:.0f} s/token.")
    print("Note: the engine has never been run against the real checkpoint;")
    print("this checks the hardware, not the engine. See docs/deepseek_v4.md.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
