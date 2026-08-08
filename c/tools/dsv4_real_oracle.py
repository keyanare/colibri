#!/usr/bin/env python3
"""Token-exact oracle for the DeepSeek-V4 engine on the REAL checkpoint.

    python3 c/tools/dsv4_real_oracle.py /path/to/dsv4 --ids 1,2,3 --ngen 2 \
        --out ref_real.json
    ./c/deepseek_v4 /path/to/dsv4 "1,2,3" --ids --ngen 2 \
        --dump-logits got.json --quiet
    python3 c/tools/dsv4_real_oracle.py /path/to/dsv4 --check got.json \
        --ref ref_real.json

The synthetic fixture (tools/make_tiny_dsv4.py) proves the engine is
self-consistent with a second implementation of the same reading. This tool is
the same RefModel, but pointed at the real `-0731` checkpoint: it reads the 48
safetensors shards, decodes the on-disk dtypes with the SAME byte semantics the
engine reads (e4m3 + ue8m0 per 128x128, mxfp4 nibbles + ue8m0 per 32, bf16, f32,
int tables), and runs the independent forward pass.

WHAT THIS PROVES. That the engine's decode and compute reproduce an independent
float64 reference on the real weight BYTES. It catches byte-layout mistakes,
nibble-ordering errors, scale-sidecar misreads and routing bugs that random
synthetic bytes cannot exercise. It is still the same reading of
`inference/model.py` as the engine, so a shared misreading passes both -- the
torch reference (torch+tilelang+CUDA, 155 GB in GPU memory) remains the only
implementation-independent oracle.

MEMORY. The full model is ~155 GB on disk; this tool never materialises it.
Dense tensors are decoded on demand and cached under a byte budget; routed
experts are decoded transiently (never cached) exactly like the engine streams
them; `head.weight` is decoded once and pinned; `embed.weight` rows are decoded
per token. Peak is roughly budget + head (2.1 GB) + working set.

The reference forward is float64 in places and per-token in pure numpy, so this
is SLOW (minutes per token on the 43-layer model). Use a short prompt.
"""
import argparse
import json
import os
import struct
import sys

import numpy as np

from make_tiny_dsv4 import (E4M3, MX4, RefModel, bf16_decode, ue8m0)


class RealStore:
    """Lazy decoder over the checkpoint's shards.

    `store[name]` returns the tensor as float32 (or int32 for the tid2eid hash
    tables), reading the correct shard on first use. Dense tensors are cached
    under a byte budget with LRU eviction; expert tensors are decoded fresh on
    every access (they are the streaming part of this model, like in the
    engine); `head.weight` is pinned; `embed.weight` rows are decoded on
    demand so the 2.1 GB embedding is never materialised as a whole.
    """

    def __init__(self, model_dir, budget=6.0e9):
        self.dir = model_dir
        self.budget = budget
        index = os.path.join(model_dir, "model.safetensors.index.json")
        if os.path.exists(index):
            with open(index) as f:
                self.weight_map = json.load(f)["weight_map"]
        else:
            # Single-file checkpoint (e.g. the synthetic fixture): every tensor
            # lives in model.safetensors. Building the map from its header also
            # doubles as a self-test of the offset arithmetic against a known
            # generator.
            with open(os.path.join(model_dir, "model.safetensors"), "rb") as f:
                n = struct.unpack("<Q", f.read(8))[0]
                hdr = json.loads(f.read(n))
            self.weight_map = {name: "model.safetensors" for name in hdr
                               if name != "__metadata__"}
        self._headers = {}        # shard file -> (header dict, header byte len)
        self._cache = {}          # name -> decoded ndarray (dense, cached)
        self._order = []          # LRU order of cached names
        self._bytes = 0           # cached decoded bytes
        self._pinned = set()      # never evict
        self._head = None         # pinned head.weight
        self._load_pin("head.weight")

    # ---- shard I/O -----------------------------------------------------

    def _shard_info(self, name):
        """(header dict, data base offset) for the shard holding `name`."""
        shard = self.weight_map[name]
        if shard not in self._headers:
            with open(os.path.join(self.dir, shard), "rb") as f:
                n = struct.unpack("<Q", f.read(8))[0]
                self._headers[shard] = (json.loads(f.read(n)), n)
        hdr, n = self._headers[shard]
        return hdr, 8 + n

    def _header(self, name):
        return self._shard_info(name)[0][name]

    def _raw(self, name):
        """Read the tensor's bytes. safetensors `data_offsets` are relative to
        the start of the DATA section, which follows the 8-byte length word and
        the JSON header -- both must be skipped."""
        meta = self._header(name)
        off0, off1 = meta["data_offsets"]
        _, base = self._shard_info(name)
        with open(os.path.join(self.dir, self.weight_map[name]), "rb") as f:
            f.seek(base + off0)
            return f.read(off1 - off0)

    @staticmethod
    def _scale_name(name):
        """The checkpoint names a sidecar `<stem>.scale`, REPLACING the
        `.weight` suffix. Same spelling as dsv4_scale_name."""
        return (name[:-len(".weight")] if name.endswith(".weight") else name) + ".scale"

    # ---- decode --------------------------------------------------------

    def _decode(self, name):
        meta = self._header(name)
        dt, shape = meta["dtype"], tuple(meta["shape"])
        raw = self._raw(name)
        if dt == "F32":
            return np.frombuffer(raw, dtype="<f4").reshape(shape).astype(np.float32)
        if dt == "BF16":
            return bf16_decode(raw, int(np.prod(shape))).reshape(shape)
        if dt == "F8_E4M3":
            # e4m3 weight bytes + ue8m0 scale, one per 128x128 block.
            O, I = shape
            b = np.frombuffer(raw, dtype=np.uint8).reshape(O, I)
            emeta = self._header(self._scale_name(name))
            e = np.frombuffer(self._raw(self._scale_name(name)),
                              dtype=np.uint8).reshape(emeta["shape"])
            s = ue8m0(e)
            w = E4M3[b] * np.repeat(np.repeat(s, 128, 0), 128, 1)[:O, :I]
            return w.astype(np.float32)
        if dt in ("U8", "I8"):
            # mxfp4 expert: nibbles packed along I (low nibble = even column),
            # ue8m0 scale one per 32 columns. Same layout as the fixture.
            O, Ip = shape
            I = 2 * Ip
            packed = np.frombuffer(raw, dtype=np.uint8).reshape(O, Ip)
            lo, hi = packed & 0x0F, packed >> 4
            vals = np.empty((O, I), dtype=np.float32)
            vals[:, 0::2] = MX4[lo]
            vals[:, 1::2] = MX4[hi]
            emeta = self._header(self._scale_name(name))
            e = np.frombuffer(self._raw(self._scale_name(name)),
                              dtype=np.uint8).reshape(emeta["shape"])
            s = ue8m0(e)
            w = vals * np.repeat(s, 32, axis=1)[:, :I]
            return w.astype(np.float32)
        if dt in ("I32", "I64"):
            # hash-routing table. The engine narrows I64 -> I32 at load; do the
            # same, so the oracle's routing equals the engine's routing.
            if dt == "I64":
                return np.frombuffer(raw, dtype="<i8").reshape(shape).astype(np.int32)
            return np.frombuffer(raw, dtype="<i4").reshape(shape).astype(np.int32)
        raise ValueError(f"dsv4_real_oracle: cannot decode {name} dtype {dt}")

    # ---- caching -------------------------------------------------------

    def _load_pin(self, name):
        self._head = self._decode(name)
        self._pinned.add(name)

    def _evict(self):
        while self._bytes > self.budget and self._order:
            victim = self._order.pop(0)
            if victim in self._pinned:
                continue
            self._bytes -= self._cache[victim].nbytes
            del self._cache[victim]

    def _cache_set(self, name, v):
        if name in self._cache:
            return v
        if self._cache.get(name) is not None:
            return v
        self._cache[name] = v
        self._order.append(name)
        self._bytes += v.nbytes
        self._evict()
        return v

    # ---- dict-like access used by RefModel ----------------------------

    def __contains__(self, name):
        return name in self.weight_map

    def __getitem__(self, name):
        if name == "head.weight":
            return self._head
        if name == "embed.weight":
            return _EmbedRows(self)
        if name in self._cache:
            self._order.remove(name)
            self._order.append(name)
            return self._cache[name]
        v = self._decode(name)
        if ".experts." in name:
            return v  # streamed, never cached
        return self._cache_set(name, v)

    def get(self, name, default=None):
        try:
            return self[name]
        except KeyError:
            return default

    def keys(self):
        return self.weight_map.keys()

    @property
    def values(self):
        """RefModel reads `g.values`; expose the store itself."""
        return self


class _EmbedRows:
    """`embed.weight[i]` decodes just row i (4096 bf16 -> f32), so the 1.06 GB
    embedding tensor is never read wholesale."""

    def __init__(self, store):
        self._store = store

    def __getitem__(self, i):
        name = "embed.weight"
        meta = self._store._header(name)
        off0, _ = meta["data_offsets"]
        row = meta["shape"][1]          # dim, not hardcoded
        start = off0 + i * row * 2
        _, base = self._store._shard_info(name)
        with open(os.path.join(self._store.dir,
                               self._store.weight_map[name]), "rb") as f:
            f.seek(base + start)
            raw = f.read(row * 2)
        u = np.frombuffer(raw, dtype=np.uint16).astype(np.uint32) << 16
        return u.view(np.float32).astype(np.float32)


# ---------------------------------------------------------------- driver

def run_reference(model_dir, ids, ngen, out_path=None, budget=6.0e9):
    with open(os.path.join(model_dir, "config.json")) as f:
        cfg = json.load(f)
    store = RealStore(model_dir, budget)
    ref = RefModel(cfg, store)

    steps, pos, tok = [], 0, ids[0]
    for pos in range(len(ids) - 1):
        ref.forward(ids[pos], pos)
        tok = ids[pos + 1]
    pos = len(ids) - 1
    for _ in range(ngen):
        lg = ref.forward(tok, pos)
        steps.append({"pos": pos, "in": int(tok),
                      "logits": [float(z) for z in lg]})
        tok = int(np.argmax(lg))
        pos += 1

    payload = {"ids": ids, "ngen": ngen, "steps": steps}
    if out_path:
        with open(out_path, "w") as f:
            json.dump(payload, f)
    return payload


def check(got_path, ref_path, tol):
    got = json.load(open(got_path))["steps"]
    ref = json.load(open(ref_path))["steps"]
    if len(got) != len(ref):
        print(f"FAIL: engine produced {len(got)} steps, reference has {len(ref)}")
        return 1
    worst, bad = 0.0, 0
    for i, (gs, rs) in enumerate(zip(got, ref)):
        gv = np.array(gs["logits"], dtype=np.float64)
        rv = np.array(rs["logits"], dtype=np.float64)
        den = np.linalg.norm(rv)
        rel = np.linalg.norm(gv - rv) / (den if den > 0 else 1.0)
        worst = max(worst, rel)
        ga, ra = int(np.argmax(gv)), int(np.argmax(rv))
        flag = ""
        if rel > tol or ga != ra:
            bad += 1
            flag = "  <-- MISMATCH"
        print(f"  step {i} pos={rs['pos']} in={rs['in']}: rel-L2 {rel:.3e}  "
              f"argmax engine={ga} ref={ra}{flag}")
    print(f"  worst rel-L2 {worst:.3e} (tolerance {tol:.1e})")
    if bad:
        print(f"FAIL: {bad} step(s) disagree")
        return 1
    print("dsv4 real-weight oracle: ok")
    return 0


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("model_dir")
    ap.add_argument("--ids", default="100,101,102,103")
    ap.add_argument("--ngen", type=int, default=2)
    ap.add_argument("--out", default="ref_real.json")
    ap.add_argument("--check", metavar="GOT_JSON")
    ap.add_argument("--ref", default="ref_real.json")
    ap.add_argument("--tol", type=float, default=5e-3)
    ap.add_argument("--budget", type=float, default=6.0e9,
                    help="decoded-dense cache budget in bytes")
    a = ap.parse_args()
    ids = [int(t) for t in a.ids.split(",") if t.strip()]
    if not ids:
        sys.exit("no ids given")

    if a.check:
        if not os.path.exists(a.ref):
            print(f"computing reference into {a.ref} ...", file=sys.stderr)
            run_reference(a.model_dir, ids, a.ngen, a.ref, a.budget)
        return check(a.check, a.ref, a.tol)

    run_reference(a.model_dir, ids, a.ngen, a.out, a.budget)
    print(f"wrote {a.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
