#!/usr/bin/env python3
"""Tiny synthetic DeepSeek-V4-Flash checkpoint + an independent numpy oracle.

    python3 tools/make_tiny_dsv4.py tiny_dsv4          # generate
    ./deepseek_v4 tiny_dsv4 "3,7,1,5,2,9" --ids \\
        --ngen 4 --dump-logits tiny_dsv4/got.json --quiet
    python3 tools/make_tiny_dsv4.py tiny_dsv4 --check tiny_dsv4/got.json

Same shape as tools/make_tiny_inkling.py's role in CI: a randomly initialised
model small enough to run in milliseconds, in EXACTLY the on-disk format the
real checkpoint uses, plus a reference forward pass the C engine must
reproduce.

WHAT THIS PROVES, AND WHAT IT DOES NOT. It proves the engine is self-consistent
with a second, independent implementation of the same reading: shapes, tensor
wiring, the compressor's cross-token state machine, the KV ring buffer, the
index-set assembly, expert streaming and eviction, and every dequant path. That
is where implementation bugs live, and none of it needs the 160 GB download.

It does NOT prove the transcription matches DeepSeek. Both this reference and
the C were written by reading `inference/model.py`; a misreading shared by both
passes silently. Only a token-exact oracle against the real weights closes
that, and this is not it.

FORMAT NOTE. Quantized tensors are generated as RANDOM BYTES rather than by
quantizing random floats. The bytes are the ground truth: both sides decode the
same bytes with the same tables, so the fixture tests the decode-and-compute
path without also testing an encoder nobody ships. NaN codes (e4m3 0x7F/0xFF,
ue8m0 0xFF) are excluded so a single NaN cannot mask a real mismatch.
"""
import argparse, json, os, struct, sys

try:
    import numpy as np
except ImportError:
    sys.exit("numpy is required: pip install numpy")

# ---------------------------------------------------------------- dtypes

def e4m3_table():
    """256-entry e4m3 decode, mirroring quant.h's E4M3_LUT (OCP E4M3-FN:
    exp==0xF is not infinity; only mant==0x7 at exp==0xF is NaN)."""
    t = np.zeros(256, dtype=np.float32)
    for b in range(256):
        sign, exp, mant = (b >> 7) & 1, (b >> 3) & 0xF, b & 0x7
        if exp == 0xF and mant == 0x7:
            t[b] = np.nan
        elif exp == 0:
            t[b] = mant * (1.0 / 8.0) * 2.0 ** (1 - 7)
        else:
            t[b] = (1.0 + mant * (1.0 / 8.0)) * 2.0 ** (exp - 7)
        if sign:
            t[b] = -t[b]
    return t

E4M3 = e4m3_table()
# e2m1 doubled-value LUT, matching quant.h's mx4_lut ordering
MX4 = np.array([0, .5, 1, 1.5, 2, 3, 4, 6, -0., -.5, -1, -1.5, -2, -3, -4, -6],
               dtype=np.float32)

def ue8m0(e):
    e = np.asarray(e, dtype=np.int32)
    return np.where(e == 255, np.nan, np.ldexp(1.0, e - 127)).astype(np.float32)

def bf16_bytes(a):
    """f32 -> bf16 bytes (truncate, matching what a checkpoint stores)."""
    u = a.astype(np.float32).view(np.uint32)
    return (u >> 16).astype(np.uint16).tobytes()

def bf16_decode(raw, n):
    u = np.frombuffer(raw, dtype=np.uint16, count=n).astype(np.uint32) << 16
    return u.view(np.float32).copy()

# ---------------------------------------------------------------- generator

class Gen:
    """Deterministic byte generator + the decoded value each tensor stands for."""
    def __init__(self, seed=1234):
        self.rng = np.random.default_rng(seed)
        self.tensors = {}     # name -> (dtype_str, shape, bytes)
        self.values  = {}     # name -> decoded float32 ndarray (the oracle's view)

    def _put(self, name, dtype, shape, raw):
        self.tensors[name] = (dtype, list(shape), raw)

    @staticmethod
    def _scale_name(name):
        """The checkpoint names the sidecar `<stem>.scale`, REPLACING the
        `.weight` suffix rather than appending to it. Mirrors dsv4_scale_name
        in dsv4_model.h -- the fixture is only a gate if it spells names the
        way the real container does."""
        return (name[:-len(".weight")] if name.endswith(".weight") else name) + ".scale"

    def plain(self, name, shape, scale=1.0):
        v = (self.rng.standard_normal(shape) * scale).astype(np.float32)
        raw = bf16_bytes(v)
        # round-trip so the oracle sees exactly what the engine will read
        v = bf16_decode(raw, int(np.prod(shape))).reshape(shape)
        self._put(name, "BF16", shape, raw)
        self.values[name] = v
        return v

    def fp8(self, name, O, I):
        """e4m3 weights + ue8m0 scales, one per 128x128 block."""
        b = self.rng.integers(0, 256, size=(O, I), dtype=np.uint8)
        b[(b == 0x7F) | (b == 0xFF)] = 0x10          # drop NaN codes
        nbO, nbI = (O + 127) // 128, (I + 127) // 128
        e = self.rng.integers(120, 135, size=(nbO, nbI), dtype=np.uint8)
        # F8_E4M3, not U8: the published checkpoint declares its fp8 weights
        # that way, and the fixture exists to be the real on-disk format. The
        # bytes are identical either way -- what this exercises is st.h's dtype
        # table, which had no entry for it until a real checkpoint was read.
        self._put(name, "F8_E4M3", (O, I), b.tobytes())
        self._put(self._scale_name(name), "F8_E8M0", (nbO, nbI), e.tobytes())
        s = ue8m0(e)
        w = E4M3[b] * np.repeat(np.repeat(s, 128, 0), 128, 1)[:O, :I]
        self.values[name] = w.astype(np.float32)
        return self.values[name]

    def fp4(self, name, O, I):
        """mxfp4: e2m1 nibbles (low nibble = even column) + ue8m0 per 32."""
        assert I % 2 == 0
        packed = self.rng.integers(0, 256, size=(O, I // 2), dtype=np.uint8)
        ng = (I + 31) // 32
        e = self.rng.integers(122, 133, size=(O, ng), dtype=np.uint8)
        self._put(name, "U8", (O, I // 2), packed.tobytes())
        self._put(self._scale_name(name), "F8_E8M0", (O, ng), e.tobytes())
        lo, hi = packed & 0x0F, packed >> 4
        vals = np.empty((O, I), dtype=np.float32)
        vals[:, 0::2] = MX4[lo]          # low nibble = EVEN column
        vals[:, 1::2] = MX4[hi]
        s = ue8m0(e)
        # NO extra factor here: quant.h's scalar path is mx4_lut[nibble] *
        # 2^(e-127) with mx4_lut holding the TRUE e2m1 values {0,.5,1,...}.
        # (Its AVX2 path uses a doubled int8 LUT and folds a compensating 0.5
        # into the scale -- the two agree; only the doubled one needs the half.)
        w = vals * np.repeat(s, 32, axis=1)[:, :I]
        self.values[name] = w.astype(np.float32)
        return self.values[name]

    def i32(self, name, arr):
        arr = arr.astype(np.int32)
        self._put(name, "I32", arr.shape, arr.tobytes())
        self.values[name] = arr
        return arr

    def write(self, path):
        header, off = {}, 0
        for name, (dtype, shape, raw) in self.tensors.items():
            header[name] = {"dtype": dtype, "shape": shape,
                            "data_offsets": [off, off + len(raw)]}
            off += len(raw)
        hj = json.dumps(header, separators=(",", ":")).encode()
        with open(path, "wb") as f:
            f.write(struct.pack("<Q", len(hj)))
            f.write(hj)
            for _, (_, _, raw) in self.tensors.items():
                f.write(raw)

# ---------------------------------------------------------------- config

def tiny_config():
    """Small enough to run instantly, wide enough to exercise every branch:
    layer 0 is ratio-0 (sliding window only) AND hash-routed; layer 1 is
    ratio-4, so it owns a compressor, an indexer with its own compressor, and
    score-based noaux_tc routing."""
    return {
        "hidden_size": 64, "num_hidden_layers": 2, "num_attention_heads": 2,
        "head_dim": 32, "qk_rope_head_dim": 8,
        "q_lora_rank": 16, "o_lora_rank": 16, "o_groups": 2,
        "n_routed_experts": 8, "num_experts_per_tok": 2,
        "moe_intermediate_size": 32, "n_shared_experts": 1,
        "vocab_size": 64, "num_hash_layers": 1,
        "index_n_heads": 2, "index_head_dim": 16, "index_topk": 4,
        "hc_mult": 4, "hc_sinkhorn_iters": 20, "hc_eps": 1e-6,
        "sliding_window": 8, "max_position_embeddings": 4096,
        "rms_norm_eps": 1e-6, "norm_topk_prob": True,
        "routed_scaling_factor": 1.5, "swiglu_limit": 10.0,
        "scoring_func": "sqrtsoftplus", "topk_method": "noaux_tc",
        "rope_theta": 10000, "compress_rope_theta": 160000,
        "num_nextn_predict_layers": 0,
        "rope_scaling": {"beta_fast": 32, "beta_slow": 1, "factor": 16,
                         "original_max_position_embeddings": 512, "type": "yarn"},
        "compress_ratios": [0, 4],
        "quantization_config": {"activation_scheme": "dynamic", "fmt": "e4m3",
                                "quant_method": "fp8", "scale_fmt": "ue8m0",
                                "weight_block_size": [128, 128]},
        "model_type": "deepseek_v4", "architectures": ["DeepseekV4ForCausalLM"],
        "eos_token_id": 1, "bos_token_id": 0,
    }

def coff(ratio):  return 2 if ratio == 4 else 1

def build(outdir, seed=1234):
    cfg = tiny_config()
    os.makedirs(outdir, exist_ok=True)
    with open(os.path.join(outdir, "config.json"), "w") as f:
        json.dump(cfg, f, indent=1)

    g = Gen(seed)
    D, HC = cfg["hidden_size"], cfg["hc_mult"]
    H, HD = cfg["num_attention_heads"], cfg["head_dim"]
    hs, mix = H * HD, (2 + HC) * HC
    QR, OR, OG = cfg["q_lora_rank"], cfg["o_lora_rank"], cfg["o_groups"]
    E, K, MI = cfg["n_routed_experts"], cfg["num_experts_per_tok"], cfg["moe_intermediate_size"]
    V = cfg["vocab_size"]

    g.plain("embed.weight", (V, D))
    g.plain("norm.weight", (D,), scale=0.2)
    # bf16: the checkpoint ships the output projection unquantized, with no
    # head.scale beside it -- the same class as embed.weight.
    g.plain("head.weight", (V, D), scale=0.05)
    g.plain("hc_head_fn", (HC, HC * D), scale=0.1)
    g.plain("hc_head_base", (HC,), scale=0.5)
    g.plain("hc_head_scale", (1,), scale=0.5)

    def compressor(prefix, ratio, hd):
        c = coff(ratio)
        g.plain(f"{prefix}.ape", (ratio, c * hd), scale=0.3)
        # bf16, not fp8: the compressor projections are the one dense pair in
        # this model that ships unquantized -- no .scale sidecar exists for
        # them in the checkpoint.
        g.plain(f"{prefix}.wkv.weight", (c * hd, D), scale=0.1)
        g.plain(f"{prefix}.wgate.weight", (c * hd, D), scale=0.1)
        g.plain(f"{prefix}.norm.weight", (hd,), scale=0.2)

    for l, ratio in enumerate(cfg["compress_ratios"][:cfg["num_hidden_layers"]]):
        g.plain(f"layers.{l}.hc_attn_fn", (mix, HC * D), scale=0.1)
        g.plain(f"layers.{l}.hc_attn_base", (mix,), scale=0.5)
        g.plain(f"layers.{l}.hc_attn_scale", (3,), scale=0.5)
        g.plain(f"layers.{l}.hc_ffn_fn", (mix, HC * D), scale=0.1)
        g.plain(f"layers.{l}.hc_ffn_base", (mix,), scale=0.5)
        g.plain(f"layers.{l}.hc_ffn_scale", (3,), scale=0.5)
        g.plain(f"layers.{l}.attn_norm.weight", (D,), scale=0.2)
        g.plain(f"layers.{l}.ffn_norm.weight", (D,), scale=0.2)

        g.plain(f"layers.{l}.attn.attn_sink", (H,), scale=0.5)
        g.fp8(f"layers.{l}.attn.wq_a.weight", QR, D)
        g.plain(f"layers.{l}.attn.q_norm.weight", (QR,), scale=0.2)
        g.fp8(f"layers.{l}.attn.wq_b.weight", hs, QR)
        g.fp8(f"layers.{l}.attn.wkv.weight", HD, D)
        g.plain(f"layers.{l}.attn.kv_norm.weight", (HD,), scale=0.2)
        g.fp8(f"layers.{l}.attn.wo_a.weight", OG * OR, hs // OG)
        g.fp8(f"layers.{l}.attn.wo_b.weight", D, OG * OR)

        if ratio:
            compressor(f"layers.{l}.attn.compressor", ratio, HD)
            if ratio == 4:
                g.fp8(f"layers.{l}.attn.indexer.wq_b.weight",
                      cfg["index_n_heads"] * cfg["index_head_dim"], QR)
                g.plain(f"layers.{l}.attn.indexer.weights_proj.weight",
                        (cfg["index_n_heads"], D), scale=0.3)
                compressor(f"layers.{l}.attn.indexer.compressor", 4, cfg["index_head_dim"])

        g.plain(f"layers.{l}.ffn.gate.weight", (E, D), scale=0.3)
        if l < cfg["num_hash_layers"]:
            rng = np.random.default_rng(seed + 100 + l)
            g.i32(f"layers.{l}.ffn.gate.tid2eid", rng.integers(0, E, size=(V, K)))
        else:
            g.plain(f"layers.{l}.ffn.gate.bias", (E,), scale=0.3)
        for e in range(E):
            g.fp4(f"layers.{l}.ffn.experts.{e}.w1.weight", MI, D)
            g.fp4(f"layers.{l}.ffn.experts.{e}.w3.weight", MI, D)
            g.fp4(f"layers.{l}.ffn.experts.{e}.w2.weight", D, MI)
        g.fp8(f"layers.{l}.ffn.shared_experts.w1.weight", MI, D)
        g.fp8(f"layers.{l}.ffn.shared_experts.w3.weight", MI, D)
        g.fp8(f"layers.{l}.ffn.shared_experts.w2.weight", D, MI)

    g.write(os.path.join(outdir, "model.safetensors"))
    return cfg, g

# ---------------------------------------------------------------- reference

def rope_freqs(dim, base, original_seq_len, factor, beta_fast, beta_slow):
    import math
    f = 1.0 / (base ** (np.arange(0, dim, 2, dtype=np.float64) / dim))
    if original_seq_len > 0:
        lg = 2.0 * math.log(base)
        dlo = dim * math.log(original_seq_len / (beta_fast * 2 * math.pi)) / lg
        dhi = dim * math.log(original_seq_len / (beta_slow * 2 * math.pi)) / lg
        low, high = max(math.floor(dlo), 0), min(math.ceil(dhi), dim - 1)
        if low == high:
            high += 0.001
        t = np.clip((np.arange(dim // 2) - low) / (high - low), 0, 1)
        smooth = 1.0 - t
        f = f / factor * (1 - smooth) + f * smooth
    return f.astype(np.float32)

def rope_apply(v, pos, freqs, inverse=False):
    v = v.copy()
    n = len(freqs)
    ang = pos * freqs.astype(np.float64)
    c, s = np.cos(ang), np.sin(ang)
    if inverse:
        s = -s
    a, b = v[0:2 * n:2].astype(np.float64), v[1:2 * n:2].astype(np.float64)
    v[0:2 * n:2] = (a * c - b * s).astype(np.float32)
    v[1:2 * n:2] = (a * s + b * c).astype(np.float32)
    return v

def rmsnorm(x, w, eps):
    r = 1.0 / np.sqrt(np.mean(x.astype(np.float64) ** 2) + eps)
    return (x * r * (w if w is not None else 1.0)).astype(np.float32)

def sinkhorn(comb, iters, eps):
    c = comb.astype(np.float64)
    c = np.exp(c - c.max(axis=1, keepdims=True))
    c = c / c.sum(axis=1, keepdims=True) + eps
    c = c / (c.sum(axis=0, keepdims=True) + eps)
    for _ in range(iters - 1):
        c = c / (c.sum(axis=1, keepdims=True) + eps)
        c = c / (c.sum(axis=0, keepdims=True) + eps)
    return c

def hc_split(mixes, scale, base, hc, iters, eps):
    sig = lambda z: 1.0 / (1.0 + np.exp(-z))
    pre = sig(mixes[:hc] * scale[0] + base[:hc]) + eps
    post = 2.0 * sig(mixes[hc:2 * hc] * scale[1] + base[hc:2 * hc])
    comb = (mixes[2 * hc:] * scale[2] + base[2 * hc:]).reshape(hc, hc)
    return pre, post, sinkhorn(comb, iters, eps)

def hc_pre(x, fn, scale, base, hc, dim, iters, eps, norm_eps):
    flat = x.reshape(-1).astype(np.float64)
    rsq = 1.0 / np.sqrt(np.mean(flat ** 2) + norm_eps)
    mixes = (fn.astype(np.float64) @ flat) * rsq
    pre, post, comb = hc_split(mixes, scale, base, hc, iters, eps)
    y = (pre[:, None] * x).sum(axis=0)
    return y.astype(np.float32), post, comb

def hc_post(sub, resid, post, comb, hc, dim):
    out = post[:, None] * sub[None, :].astype(np.float64)
    out = out + np.einsum("jk,jd->kd", comb, resid.astype(np.float64))
    return out.astype(np.float32)

def hc_collapse(x, fn, scale, base, eps, norm_eps):
    flat = x.reshape(-1).astype(np.float64)
    rsq = 1.0 / np.sqrt(np.mean(flat ** 2) + norm_eps)
    m = (fn.astype(np.float64) @ flat) * rsq
    pre = 1.0 / (1.0 + np.exp(-(m * scale + base))) + eps
    return (pre[:, None] * x).sum(axis=0).astype(np.float32)

def sqrtsoftplus(z):
    z = np.asarray(z, dtype=np.float64)
    sp = np.where(z > 20, z, np.log1p(np.exp(np.minimum(z, 20))))
    return np.sqrt(sp)

def route(logits, bias, topk, norm, scale):
    sc = sqrtsoftplus(logits)
    sel = sc + (bias if bias is not None else 0.0)
    idx = list(np.argsort(-sel, kind="stable")[:topk])
    w = sc[idx]
    if norm:
        w = w / w.sum()
    return idx, (w * scale)

def swiglu(g, u, limit):
    g, u = g.astype(np.float64).copy(), u.astype(np.float64).copy()
    if limit > 0:
        u = np.clip(u, -limit, limit)
        g = np.minimum(g, limit)
    return ((g / (1.0 + np.exp(-g))) * u).astype(np.float32)

def compress_pool(kv, score, ratio, D):
    s = score.astype(np.float64)
    mx = s.max(axis=0, keepdims=True)
    e = np.where(np.isneginf(s), 0.0, np.exp(s - np.where(np.isneginf(mx), 0.0, mx)))
    den = e.sum(axis=0, keepdims=True)
    w = np.divide(e, den, out=np.zeros_like(e), where=den > 0)
    return (kv.astype(np.float64) * w).sum(axis=0).astype(np.float32)

def sparse_attn(q, kv, idx, sink, scale):
    H, Dh = q.shape
    out = np.zeros((H, Dh), dtype=np.float64)
    for h in range(H):
        valid = [p for p in idx if 0 <= p < kv.shape[0]]
        if not valid:
            continue
        s = np.array([np.dot(q[h].astype(np.float64), kv[p].astype(np.float64)) * scale
                      for p in valid])
        mx = s.max()
        e = np.exp(s - mx)
        den = e.sum() + np.exp(sink[h] - mx)
        out[h] = (e[:, None] * kv[valid].astype(np.float64)).sum(axis=0) / den
    return out.astype(np.float32)

class RefModel:
    """Independent forward pass over the generated weights."""
    def __init__(self, cfg, g):
        self.c, self.v = cfg, g.values
        c = cfg
        self.state = []
        for l, ratio in enumerate(c["compress_ratios"][:c["num_hidden_layers"]]):
            st = {"ratio": ratio,
                  "kv_ring": np.zeros((c["sliding_window"], c["head_dim"]), np.float32),
                  "comp": [], "icomp": [],
                  "cstate": None, "istate": None}
            if ratio:
                cf = coff(ratio)
                # score_state starts at -inf, matching model.py's own
                # register_buffer and the C engine: an unfilled slot must
                # contribute ZERO softmax weight, not exp(0). Initialising it
                # to zeros silently inflates the denominator of every pooled
                # latent in the first group.
                st["cstate"] = [np.zeros((cf * ratio, cf * c["head_dim"]), np.float32),
                                np.full((cf * ratio, cf * c["head_dim"]), -np.inf, np.float32)]
                if ratio == 4:
                    ih = c["index_head_dim"]
                    st["istate"] = [np.zeros((2 * 4, 2 * ih), np.float32),
                                    np.full((2 * 4, 2 * ih), -np.inf, np.float32)]
            self.state.append(st)
            if ratio:
                self.freqs_c = rope_freqs(c["qk_rope_head_dim"], c["compress_rope_theta"],
                                          c["rope_scaling"]["original_max_position_embeddings"],
                                          c["rope_scaling"]["factor"],
                                          c["rope_scaling"]["beta_fast"],
                                          c["rope_scaling"]["beta_slow"])
        self.freqs = {}
        for l, ratio in enumerate(c["compress_ratios"][:c["num_hidden_layers"]]):
            if ratio:
                self.freqs[l] = rope_freqs(c["qk_rope_head_dim"], c["compress_rope_theta"],
                                           c["rope_scaling"]["original_max_position_embeddings"],
                                           c["rope_scaling"]["factor"],
                                           c["rope_scaling"]["beta_fast"],
                                           c["rope_scaling"]["beta_slow"])
            else:
                self.freqs[l] = rope_freqs(c["qk_rope_head_dim"], c["rope_theta"],
                                           0, c["rope_scaling"]["factor"],
                                           c["rope_scaling"]["beta_fast"],
                                           c["rope_scaling"]["beta_slow"])

    def compressor_step(self, l, key, prefix, ratio, hd, x, pos):
        c, v = self.c, self.v
        st = self.state[l]
        cf = coff(ratio)
        wide = cf * hd
        kvs, scs = st["cstate"] if key == "comp" else st["istate"]
        kv = v[f"{prefix}.wkv.weight"] @ x
        sc = v[f"{prefix}.wgate.weight"] @ x
        slot = pos % ratio
        sc = sc + v[f"{prefix}.ape"][slot]
        row = ratio + slot if cf == 2 else slot
        kvs[row], scs[row] = kv, sc
        if (pos + 1) % ratio:
            return
        if cf == 2:
            pk = np.concatenate([kvs[:ratio, :hd], kvs[ratio:, hd:]], axis=0)
            ps = np.concatenate([scs[:ratio, :hd], scs[ratio:, hd:]], axis=0)
            out = compress_pool(pk, ps, 2 * ratio, hd)
            kvs[:ratio], scs[:ratio] = kvs[ratio:].copy(), scs[ratio:].copy()
        else:
            out = compress_pool(kvs[:ratio], scs[:ratio], ratio, hd)
        out = rmsnorm(out, v[f"{prefix}.norm.weight"], c["rms_norm_eps"])
        rd = c["qk_rope_head_dim"]
        out[hd - rd:] = rope_apply(out[hd - rd:], pos + 1 - ratio, self.freqs[l])
        st[key].append(out.copy())

    def forward(self, token_id, pos):
        c, v = self.c, self.v
        HC, D = c["hc_mult"], c["hidden_size"]
        H, HD, rd = c["num_attention_heads"], c["head_dim"], c["qk_rope_head_dim"]
        hs, OG, OR = H * HD, c["o_groups"], c["o_lora_rank"]
        x = np.repeat(v["embed.weight"][token_id][None, :], HC, axis=0).astype(np.float32)

        for l in range(c["num_hidden_layers"]):
            st = self.state[l]
            ratio = st["ratio"]
            p = f"layers.{l}"
            resid = x.copy()
            h, post, comb = hc_pre(x, v[f"{p}.hc_attn_fn"], v[f"{p}.hc_attn_scale"],
                                   v[f"{p}.hc_attn_base"], HC, D,
                                   c["hc_sinkhorn_iters"], c["hc_eps"], c["rms_norm_eps"])
            h = rmsnorm(h, v[f"{p}.attn_norm.weight"], c["rms_norm_eps"])

            qr = rmsnorm(v[f"{p}.attn.wq_a.weight"] @ h, v[f"{p}.attn.q_norm.weight"],
                         c["rms_norm_eps"])
            q = (v[f"{p}.attn.wq_b.weight"] @ qr).reshape(H, HD)
            for hh in range(H):
                q[hh] = q[hh] / np.sqrt(np.mean(q[hh].astype(np.float64) ** 2) + c["rms_norm_eps"])
                q[hh, HD - rd:] = rope_apply(q[hh, HD - rd:], pos, self.freqs[l])

            kv = rmsnorm(v[f"{p}.attn.wkv.weight"] @ h, v[f"{p}.attn.kv_norm.weight"],
                         c["rms_norm_eps"])
            kv[HD - rd:] = rope_apply(kv[HD - rd:], pos, self.freqs[l])
            st["kv_ring"][pos % c["sliding_window"]] = kv

            idx = [pp % c["sliding_window"]
                   for pp in range(max(0, pos - c["sliding_window"] + 1), pos + 1)]
            cache = st["kv_ring"]
            if ratio:
                self.compressor_step(l, "comp", f"{p}.attn.compressor", ratio, HD, h, pos)
                if ratio == 4:
                    self.compressor_step(l, "icomp", f"{p}.attn.indexer.compressor",
                                         4, c["index_head_dim"], h, pos)
                base = c["sliding_window"]
                if ratio == 4 and st["icomp"]:
                    ih, IH = c["index_head_dim"], c["index_n_heads"]
                    iq = (v[f"{p}.attn.indexer.wq_b.weight"] @ qr).reshape(IH, ih)
                    for hh in range(IH):
                        iq[hh, ih - rd:] = rope_apply(iq[hh, ih - rd:], pos, self.freqs[l])
                    wts = (v[f"{p}.attn.indexer.weights_proj.weight"] @ h) * \
                          (1.0 / np.sqrt(ih)) * (1.0 / np.sqrt(IH))
                    ck = np.stack(st["icomp"])
                    sc = np.zeros(len(ck))
                    for t in range(len(ck)):
                        dots = iq.astype(np.float64) @ ck[t].astype(np.float64)
                        sc[t] = (np.maximum(dots, 0) * wts).sum()
                    topk = min(c["index_topk"], len(ck))
                    order = list(np.argsort(-sc, kind="stable")[:topk])
                    idx = idx + [base + int(t) for t in order]
                elif st["comp"]:
                    idx = idx + [base + t for t in range(len(st["comp"]))]
                if st["comp"]:
                    cache = np.concatenate([st["kv_ring"], np.stack(st["comp"])], axis=0)
                else:
                    cache = st["kv_ring"]

            o = sparse_attn(q, cache, idx, v[f"{p}.attn.attn_sink"], 1.0 / np.sqrt(HD))
            for hh in range(H):
                o[hh, HD - rd:] = rope_apply(o[hh, HD - rd:], pos, self.freqs[l], inverse=True)

            gin = hs // OG
            wa = v[f"{p}.attn.wo_a.weight"].reshape(OG, OR, gin)
            flat = o.reshape(-1)
            proj = np.concatenate([wa[gg] @ flat[gg * gin:(gg + 1) * gin] for gg in range(OG)])
            attn_out = v[f"{p}.attn.wo_b.weight"] @ proj
            x = hc_post(attn_out, resid, post, comb, HC, D)

            resid = x.copy()
            h, post, comb = hc_pre(x, v[f"{p}.hc_ffn_fn"], v[f"{p}.hc_ffn_scale"],
                                   v[f"{p}.hc_ffn_base"], HC, D,
                                   c["hc_sinkhorn_iters"], c["hc_eps"], c["rms_norm_eps"])
            h = rmsnorm(h, v[f"{p}.ffn_norm.weight"], c["rms_norm_eps"])

            logits = v[f"{p}.ffn.gate.weight"] @ h
            if l < c["num_hash_layers"]:
                sel = list(v[f"{p}.ffn.gate.tid2eid"][token_id])
                w = sqrtsoftplus(logits[sel])
                if c["norm_topk_prob"]:
                    w = w / w.sum()
                w = w * c["routed_scaling_factor"]
            else:
                sel, w = route(logits, v[f"{p}.ffn.gate.bias"],
                               c["num_experts_per_tok"], c["norm_topk_prob"],
                               c["routed_scaling_factor"])
            out = np.zeros(D, dtype=np.float64)
            for i, e in enumerate(sel):
                gg = v[f"{p}.ffn.experts.{e}.w1.weight"] @ h
                uu = v[f"{p}.ffn.experts.{e}.w3.weight"] @ h
                act = swiglu(gg, uu, c["swiglu_limit"]) * w[i]
                out += v[f"{p}.ffn.experts.{e}.w2.weight"] @ act
            gg = v[f"{p}.ffn.shared_experts.w1.weight"] @ h
            uu = v[f"{p}.ffn.shared_experts.w3.weight"] @ h
            out += v[f"{p}.ffn.shared_experts.w2.weight"] @ swiglu(gg, uu, c["swiglu_limit"])
            x = hc_post(out.astype(np.float32), resid, post, comb, HC, D)

        y = hc_collapse(x, v["hc_head_fn"], float(v["hc_head_scale"][0]),
                        v["hc_head_base"], c["hc_eps"], c["rms_norm_eps"])
        y = rmsnorm(y, v["norm.weight"], c["rms_norm_eps"])
        return (v["head.weight"] @ y).astype(np.float32)

# ---------------------------------------------------------------- driver

DEFAULT_IDS = [3, 7, 1, 5, 2, 9, 4, 11]

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("outdir")
    ap.add_argument("--seed", type=int, default=1234)
    ap.add_argument("--ngen", type=int, default=4)
    ap.add_argument("--ids", default=",".join(map(str, DEFAULT_IDS)))
    ap.add_argument("--check", metavar="GOT_JSON",
                    help="compare an engine --dump-logits file against the reference")
    ap.add_argument("--tol", type=float, default=2e-3,
                    help="max relative L2 per step (f32 vs f64 accumulation)")
    a = ap.parse_args()
    ids = [int(t) for t in a.ids.split(",") if t.strip()]

    cfg, g = build(a.outdir, a.seed)
    ref = RefModel(cfg, g)

    steps, pos, tok = [], 0, ids[0]
    for pos in range(len(ids) - 1):
        ref.forward(ids[pos], pos)
        tok = ids[pos + 1]
    pos = len(ids) - 1
    for _ in range(a.ngen):
        lg = ref.forward(tok, pos)
        steps.append({"pos": pos, "in": int(tok), "logits": [float(z) for z in lg]})
        tok = int(np.argmax(lg))
        pos += 1

    refpath = os.path.join(a.outdir, "ref_dsv4.json")
    with open(refpath, "w") as f:
        json.dump({"ids": ids, "ngen": a.ngen, "steps": steps}, f)

    if not a.check:
        ntens = len(g.tensors)
        print(f"wrote {a.outdir}/config.json, model.safetensors ({ntens} tensors), ref_dsv4.json")
        print(f"  run: ./deepseek_v4 {a.outdir} \"{a.ids}\" --ids --ngen {a.ngen} "
              f"--dump-logits {a.outdir}/got.json --quiet")
        print(f"  then: python3 tools/make_tiny_dsv4.py {a.outdir} --check {a.outdir}/got.json")
        return 0

    got = json.load(open(a.check))["steps"]
    if len(got) != len(steps):
        print(f"FAIL: engine produced {len(got)} steps, reference has {len(steps)}")
        return 1
    worst, bad = 0.0, 0
    for i, (gs, rs) in enumerate(zip(got, steps)):
        gv = np.array(gs["logits"], dtype=np.float64)
        rv = np.array(rs["logits"], dtype=np.float64)
        den = np.linalg.norm(rv)
        rel = np.linalg.norm(gv - rv) / (den if den > 0 else 1.0)
        worst = max(worst, rel)
        ga, ra = int(np.argmax(gv)), int(np.argmax(rv))
        flag = ""
        if rel > a.tol or ga != ra:
            bad += 1
            flag = "  <-- MISMATCH"
        print(f"  step {i} pos={rs['pos']} in={rs['in']}: rel-L2 {rel:.3e}  "
              f"argmax engine={ga} ref={ra}{flag}")
    print(f"  worst rel-L2 {worst:.3e} (tolerance {a.tol:.1e})")
    if bad:
        print(f"FAIL: {bad} step(s) disagree")
        return 1
    print("dsv4 tiny fixture: ok")
    return 0

if __name__ == "__main__":
    sys.exit(main())
