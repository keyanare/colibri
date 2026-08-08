# DeepSeek-V4-Flash on colibrì

An engine for [DeepSeek-V4-Flash](https://huggingface.co/deepseek-ai/DeepSeek-V4-Flash)
— 284B parameters, 13B active, 1M context — streaming from disk in pure C.

> **STATUS: UNVERIFIED AGAINST THE MODEL.** Every line here was written from
> DeepSeek's published reference implementation (`inference/model.py`,
> `inference/kernel.py`) and `config.json`, without ever loading the 160 GB
> checkpoint. The compute primitives, the tensor manifest, the tokenizer and
> the full forward pass all have tests — but the only thing that closes the gap
> between "self-consistent" and "correct" is a token-exact oracle on real
> weights, and that has not been run.
>
> **It now runs on the real `-0731` checkpoint.** `--preflight` matches
> 34223/34223 names and shapes, and short prompts produce coherent continuation
> (12 tokens on an Apple M5, 10 on an AMD Ryzen 7 H 255 under WSL). **That is
> not correctness.** Neither run passes the 128-token sliding window, so the
> CSA/HCA long-context paths have never been exercised on real weights.
> Nothing has been compared against a reference implementation on real
> weights: a subtly wrong compressed path, a tie broken the other way in the
> indexer's top-512, or an off-by-one in the compressor's prefill would all
> still read as fluent text. The token-exact oracle remains the only thing
> that would close it.

---

## Why this model is a good fit

97.4% of the parameters are routed experts that a token never touches: 6 of 256
fire per layer. That is exactly the shape colibrì exists for — the model is
1000× bigger than the RAM it runs in, and the difference lives on an SSD.

Three properties are unusual even by MoE standards:

- **The KV cache is tiny.** K and V are the *same* 512-dim vector per position,
  shared by all 64 heads, and most positions are compressed 4× or 128× before
  they are stored. A million-token context costs tens of megabytes per layer,
  not tens of gigabytes.
- **Three layers need no routing prediction at all.** The first 3 of 43 layers
  pick experts by a table lookup on the *input token id* (`tid2eid`), so their
  working set is known before the forward pass begins. Nothing else in this
  repository offers exact prefetch.
- **The experts are QAT-trained MXFP4**, and `quant.h` already had a CPU kernel
  for that layout (written for Kimi K3). They stream byte-for-byte from the
  original checkpoint — no conversion step, and no re-encode that would add
  error to weights already trained in their target format.

---

## Architecture, in four pieces

| | what it is | where |
|---|---|---|
| **mHC** | The residual stream is **four** vectors, not one. Each sublayer is entered by collapsing them to one and left by expanding back, mixed by a doubly stochastic matrix from 20 Sinkhorn iterations. | `dsv4_hc_pre` / `dsv4_hc_post` / `dsv4_sinkhorn` |
| **Shared-KV attention with sinks** | One 512-dim latent per position serves every head as both key *and* value. A learnable per-head sink logit joins the softmax denominator. The output must be **de-rotated**, because the value side carries RoPE. | `dsv4_sparse_attn` |
| **CSA / HCA** | Every layer attends to a 128-token sliding window. Layers with `compress_ratio > 0` also attend to compressed positions — one latent per 4 tokens (CSA, with a learned top-512 indexer) or per 128 (HCA, all of them). Their `wkv`/`wgate` projections are the **one dense pair in the model that ships unquantized** — bf16, no `.scale` sidecar. | `dsv4_compress_pool`, `dsv4_index_score` |
| **Hash + noaux_tc routing** | `sqrt(softplus(x))` scoring; the bias steers *selection* only, never the returned weights. The first 3 layers bypass scoring entirely via `tid2eid`. | `dsv4_route`, `dsv4_route_hash` |

Two RoPE tables per model: compressed layers use `compress_rope_theta` (160000)
**with** YaRN; sliding-window-only layers use `rope_theta` (10000) with YaRN
**disabled**. Sharing one table between them is a silent correctness bug.

---

## Files

```
c/dsv4.h                        compute primitives (unit tested)
c/dsv4_model.h                  config parsing + tensor manifest (unit tested)
c/deepseek_v4.c                 the engine
c/tok_unicode_dsv4.h            generated \p{P} \p{S} \p{M} tables
c/tools/make_tiny_dsv4.py       synthetic checkpoint + numpy oracle
c/tools/dsv4_real_oracle.py     numpy oracle on the REAL checkpoint (reads the 48 shards)
c/tools/dsv4_tokenizer.py       vocabulary validator + HF cross-check
c/tools/dsv4_doctor.py          hardware readiness check
c/tools/gen_unicode_dsv4.py     regenerates tok_unicode_dsv4.h
c/tests/test_dsv4.c             17 primitive tests
c/tests/test_dsv4_model.c       manifest / shape tests
c/tests/test_dsv4_fixture.py    end-to-end vs the numpy oracle
c/tests/test_dsv4_real_oracle.py  real-weight oracle (skips when no checkpoint)
c/tests/test_dsv4_tokenizer.py  validator tests (mostly negative)
c/tests/test_tok_dsv4.c         tok.h vs HF tokenizers harness
```

Changes to shared code: `quant.h` gained a UE8M0 block-scale decoder,
`st.h` learned the `F8_E8M0`, `I32` and `I64` dtypes and `tok.h` gained a fourth
pre-tokenizer family plus support for string-form `merges`.

`I64` is there because torch's default dtype for an index tensor is int64, so
the `tid2eid` hash tables arrive 8 bytes wide. `st.h` treats them as opaque
bytes; `w_load` settles the width and narrows, and refuses any other byte span
rather than reading a routing table half-wrong.

---

## Which checkpoint

Two exist, and they are not interchangeable byte-for-byte:

| | `DeepSeek-V4-Flash` (preview) | `DeepSeek-V4-Flash-0731` |
|---|---|---|
| shards / size | 46 / ~159.6 GB | 48 / ~166.9 GB |
| speculative head | one MTP layer | three DSpark modules, `mtp.0/1/2` |
| `compress_ratios` | 44 entries | **46** entries |
| `num_nextn_predict_layers` | 1 | still 1 — it does *not* track the module count |

The main stack — the 43 layers this engine actually builds — is the same shape
in both, and the extra `compress_ratios` entries are trailing. The engine sizes
the head from `dspark_target_layer_ids` when that key is present and from
`num_nextn_predict_layers` when it is not, and refuses a length no key in the
config accounts for rather than assuming the difference is padding. Either way
the head is not loaded; preflight reports its tensors as expected-unknown.

The tables below were measured against the preview; -0731 adds ~7 GB of
speculative weights that are never read.

What HAS been checked against the real -0731 container, from its safetensors
headers alone: the tensor names of the main stack, `compress_ratios` (it is the
`[0,0] + [4,128]*20 + [4]` interleave the manifest assumed, plus three trailing
zeros), the 41 compressor / 21 indexer layer sets, and that the dtype and byte
count of every quantized tensor reconciles exactly — 35328 expert tensors,
390 dense fp8, 35718 scale sidecars, 3 int64 `tid2eid` tables. That is names and
shapes agreeing. It says nothing about whether the weights are being *used*
correctly, which still needs the token-exact oracle.

Four conventions the derivation got wrong, all found this way:

| | assumed | actual |
|---|---|---|
| speculative head | 1 MTP layer (`num_nextn_predict_layers`) | **3 DSpark modules**, counted by `dspark_target_layer_ids` |
| scale sidecar name | `<name>.weight.scale` | **`<stem>.scale`** — the suffix is replaced, not appended (`dsv4_scale_name`) |
| compressor `wkv`/`wgate` | fp8 + sidecar | **bf16**, no sidecar |
| `head.weight` | fp8 + sidecar | **bf16**, no sidecar |

Also `st.h` had no entry for `F8_E4M3` — colibrì's own containers spell fp8
weight bytes `U8`, so nothing had ever needed one.

The sidecar name is the instructive failure: appending rather than replacing
made *every* quantized tensor report "sidecar absent" while the real sidecars
piled up in the not-covered list — 33389 scale problems from one string
operation. Each of the four surfaced as a named, specific message, which is the
argument for preflight over a crash mid-load.

## Running it

### 0. Check the machine first

```bash
python3 c/tools/dsv4_doctor.py                     # before downloading 149 GiB
python3 c/tools/dsv4_doctor.py --bench             # measure the disk too
python3 c/tools/dsv4_doctor.py --model /nvme/dsv4  # and validate the checkpoint
```

Sizes the model from its own config, checks disk, RAM, SIMD and OpenMP against
it, and estimates s/token from what it measured rather than from a table. Exit
0 means it can run; only real blockers fail, a machine that will merely be slow
still passes.

### 1. Build

```bash
make -C c deepseek_v4 ARCH=native
```

`ARCH=native` matters. On Rocket Lake / Zen it unlocks the AVX2 MXFP4 kernel;
without OpenMP the engine is single-threaded and roughly 10× slower.
On macOS: `brew install libomp` first, or the build warns and stays serial.

### 2. Tokenizer

**The published checkpoint ships no vocabulary.** Its root has `config.json`,
`generation_config.json` and the shards; `encoding/` is the chat-message layer
(`encode_messages`), not a vocab; `inference/generate.py` calls
`AutoTokenizer.from_pretrained(ckpt_path)`, which cannot succeed against the
repository as published. Supply one yourself:

```bash
python3 c/tools/dsv4_tokenizer.py /nvme/dsv4 --from /path/to/tokenizer.json
```

The tool refuses rather than installs when `vocab_size`, `bos_token_id`,
`eos_token_id`, the required special tokens, or the **pre-tokenizer family**
disagree with the model. That last check is the one that matters: `tok.h`
selects a family by substring alone, so an unrecognised pattern does not error
— it tokenizes differently.

Optional cross-check against HF `tokenizers` (needs `pip install tokenizers`):

```bash
python3 c/tools/dsv4_tokenizer.py /nvme/dsv4 --ctest c/tests/test_tok_dsv4
```

### 3. Run

```bash
./c/deepseek_v4 /nvme/dsv4 --preflight                    # do this first
./c/deepseek_v4 /nvme/dsv4 "your prompt" --expert-gb 8 --max-seq 8192 --ngen 64
```

`--preflight` walks the tensor manifest against the checkpoint using only the
safetensors headers — seconds on 160 GB, no weight byte read. It names every
missing tensor, wrong byte count and absent `.scale` sidecar, and reports
tensors it does not cover (the MTP head) as expected rather than as corruption.

| flag | meaning |
|---|---|
| `--expert-gb G` | **global** streamed-expert cache budget in GB (default 2) |
| `--max-seq N` | sizes the KV and compressed caches (default 8192) |
| `--ngen N` | tokens to generate |
| `--temp T` | 0 = greedy (default) |
| `--ids` | treat the prompt as comma-separated token ids; skips the tokenizer |
| `--dump-logits F` | write per-step logits — the oracle path |
| `--chunk N` | prefill tokens per pass of expert reads (default 32, 1 = per-token) |
| `--preflight` | check names and shapes against the manifest, read no weights |

`dsv4_doctor.py` suggests an `--expert-gb` that fits your RAM.

### 4. The real-weight oracle

```bash
# 1. engine logits on the real checkpoint (short prompt -- the reference is slow)
./c/deepseek_v4 /nvme/dsv4 "0,1,2,3,4" --ids --ngen 1 \
    --dump-logits got.json --quiet
# 2. independent numpy reference on the same prompt, then compare
python3 c/tools/dsv4_real_oracle.py /nvme/dsv4 \
    --ids 0,1,2,3,4 --ngen 1 --check got.json
```

The tool reads the 48 shards with the same byte semantics as the engine
(`data_offsets` relative to the data section, e4m3 + ue8m0 per 128×128, mxfp4
nibbles + ue8m0 per 32, bf16, i64→i32 narrowing), runs `RefModel` (the same
numpy forward as the fixture) on the real weights, and requires small rel-L2
plus identical argmax on every step. `test_dsv4_real_oracle.py` gates it and
skips loudly when the checkpoint is absent (`DSV4_REAL_CHECKPOINT`). At
~20 s/token of reference compute on the 43-layer model, keep the prompt short.

---

## Hardware

Disk is the hard requirement: **~149 GiB** for the checkpoint, and it cannot be
compressed further (the experts are already 4.25 bits/weight).

The engine is I/O-bound, and on any machine where the expert cache is smaller
than a few percent of a layer's 256 experts, essentially every token streams its
whole working set: **6 experts × 43 layers × 13.4 MB ≈ 3.4 GB read per token**.

The only real runs so far — both short prompts, both CPU-only (there is **no
CUDA or Metal backend for this engine**):

| | Apple M5 (macOS) | Lenovo Lecoo, WSL (Ryzen 7 H 255, 32 GB, NVMe) |
|---|---|---|
| resident | ~8.0 GB | ~8.0 GB |
| expert cache | 3.5 GB (`--expert-gb 3.5`) | 3.5 GB (`--expert-gb 3.5`) |
| expert kernels | NEON (I%32==0) | AVX2 |
| threads (OMP) | 4 physical of 10 logical | 8 physical of 16 logical |
| prefill | 2.0 s/token (11 tok, 21.5 s) | 2.2 s/token (11 tok, 24.5 s) |
| decode | **0.06 tok/s** (~16.4 s/token) | **0.60 tok/s** (~1.7 s/token) |
| expert load | 94.5 s (3483 misses) | 64.2 s (3483 misses) |
| prompt | 12 tokens, Pushkin | 12 tokens, Pushkin |

The two runs above are apples-to-apples: the same prompt, the same flags
(`--expert-gb 3.5 --max-seq 8192 --ngen 16 --direct`), both on cold caches.
The Mac is exactly an order of magnitude slower on decode, and both causes are
visible in the run: it decodes on 4 physical threads (the Lenovo uses 8), and
it takes **1.5× longer to read the same bytes** — 3483 expert misses on both
machines, 94.5 s of loading on the Mac vs 64.2 s on the Lenovo. Both caches
thrash identically (~46% miss), so neither run is close to the resident limit
this engine is built for; a fully resident run has not happened anywhere yet.

Prefill runs **layer-major over chunks of `--chunk` tokens** (default 32), so
each unique expert is read once per layer per chunk and applied to every token
that routed to it. Token-at-a-time is the fallback (`--chunk 1`): a token's six
experts per layer are then read for that token alone. The ceiling is 256 reads
per layer per chunk however long the chunk is, and the head is skipped for
prompt tokens. Measured prefill: 2.0 s/token (Mac) and 2.2 s/token (Lenovo),
both 12-token prompts on cold caches — nearly equal, because prefill reads each
unique expert once per layer and is dominated by the same ~3.4 GB working set on
both machines.

The batching is **bit-identical** to `--chunk 1`, on purpose: the sequential
state still advances one token at a time inside each layer, the dense matmuls
still run at S=1, and each token's expert contributions are staged and summed
in route order rather than in the expert-major order they were computed in.
`test_prefill_chunking_is_bit_identical` compares the logit dumps byte-for-byte
across chunk sizes. Batching the dense matmuls too is the next step, and it is
not free in that sense — it changes reduction order.

---

## What is actually tested

| claim | evidence |
|---|---|
| shapes and tensor wiring | the derived manifest reproduces DeepSeek's own published figures: **284.3B** parameters (published 284B), **13.3B** active (published 13B), **155 GB** on disk (published 160 GB), 97.4% in routed experts. None of those numbers were used to build the manifest. |
| the whole forward pass | a synthetic checkpoint in the real on-disk format, run through the engine and an independent numpy implementation: **rel-L2 1.0e-05**, argmax identical on every step. `make test-python` (needs numpy). |
| the whole forward pass on REAL weights | `tools/dsv4_real_oracle.py` reads the 48 shards, decodes e4m3/mxfp4/bf16 with the same byte semantics as the engine, and runs the same numpy reference on the real `-0731` checkpoint: **rel-L2 ~4e-07, argmax identical** on every step (5- and 13-token prompts). The first time engine logits have been compared to anything computed from the real weight bytes. Gated in `test_dsv4_real_oracle.py`, which skips loudly when the checkpoint is absent. |
| tokenizer | **356/356** cases match HF `tokenizers` on the real vocabulary — punctuation, contractions, CJK, Korean, Thai, Arabic, Hebrew, Cyrillic, combining marks, ZWJ emoji, CRLF, URLs, code, 200 random strings. Forced down the `cl100k` path the same vocabulary scores **341/356**, which is why the fourth family exists. |
| primitives | 18 test groups, most of them aimed at a specific way a plausible reimplementation goes wrong (untransposed `comb`, symmetric SwiGLU clamp, sink folded into the loop, per-token instead of per-dimension pooling, and the MXFP4 SIMD decode vs the undoubled e2m1 LUT). |

**What none of this proves:** that the architecture was transcribed correctly.
The engine and the numpy oracle were written from the same reading of the same
file; a shared misreading passes both. The real-weight oracle (above) closes
the byte-decode and streaming half of that gap — the engine now demonstrably
reproduces an independent float64 reference from the real weight bytes — but
not the transcription half: both sides still read `inference/model.py` the
same way, and only running DeepSeek's own torch reference (torch+tilelang+CUDA,
155 GB in GPU memory) would settle that. As a check on the decode side it has
already caught one real defect: the first cut of the oracle read safetensors
`data_offsets` relative to file start instead of to the data section, so
byte-reading bugs ARE the class this tool is built to find.

Three real defects were caught this way, which is the argument for the harness:

- the MXFP4 scalar path applies `mx4_lut[nibble] × 2^(e-127)` with **no** ×0.5
  (the doubling lives only in the AVX2 branch, compensated there);
- `score_state` must start at `-inf`, not zero, or the first compressed latent
  of every group has an inflated softmax denominator;
- `tok.h` rejected DeepSeek's string-form `merges` (`"Ġ t"`) outright.

---

## Known gaps

- **The speculative head.** Real and present in the checkpoint, not loaded: its
  tensor names are unknown (the reference inference code does not build it), so
  speculation is off. The two releases size it differently and `compress_ratios`
  is where you see that — see *Which checkpoint* above.
- **No GPU backend.**
- **Prefill is serial.** Batching the prompt needs no design change.
- **The indexer skips** the reference's Hadamard rotation and FP4 QAT
  simulation. Both are norm-preserving; skipping them perturbs which positions
  win ties at the top-512 boundary, not the mechanism.
- **`dsv4_hc_collapse` is inferred**, not transcribed: `ParallelHead`'s body was
  never read, and the form follows from the parameter shapes
  (`hc_head_fn [hc_mult, hc*dim]`, `hc_head_base [hc_mult]`, `hc_head_scale [1]`).
- **The compressor prefill path** is approximated by running the incremental
  decode branch per token, rather than the reference's bulk `start_pos == 0`
  branch with its remainder handling.

## Next, in order of value

1. ✅ **NEON `matmul_mxfp4`** — done (2026-08-03). The expert kernel was scalar on
   Apple Silicon and it is the hot loop; it now has a NEON path for `I%32==0`
   (every DSV4 expert) that decodes each nibble to its doubled int8 via
   `vqtbl1q_s8` and un-doubles through the group scale — the same shape as the
   AVX2 branch, base NEON only (no DOTPROD/i8mm), pinned by an independent
   scalar reference in `test_dsv4.c` (rel-L2 1e-5).
2. ✅ **NEON `matmul_fp8`** — done (2026-08-03). The FP8 dense kernel (attention,
   projections, head) had no SIMD at all; it now has a NEON path that breaks the
   serial scalar `acc+=` chain with four independent f32 lanes (clang cannot
   auto-vectorize the `E4M3_LUT` gather), same exact LUT so NaN propagation is
   unchanged. Benchmarked ~2.1–2.75× over the scalar `-mcpu=native` build at
   dense shapes (`tests/bench_fp8`, not a gate).
3. ✅ **Batched expert I/O** — done (2026-08-04). One `pread` per expert instead
   of six: the six byte spans (w1/w3/w2 + their ue8m0 scales) are resolved ONCE
   into an `ExRef` table at load, a cache-miss overwrites one 4K-aligned slab
   with a single coalesced pread (or, for the rare expert that straddles a shard
   boundary, six per-piece preads packed into that same slab). `O_DIRECT` via
   `--direct` routes the contiguous read through the aligned twin fd; otherwise
   a `WILLNEED` prefetch overlaps the load with compute (decode and per-chunk
   prefill, sorted by disk offset). `tools/make_tiny_dsv4.py --gap` breaks one
   expert's contiguity so the fallback is exercised; pinned by new CI tests
   (`test_dsv4_fixture.py`, both `--gap` and `--direct`).
4. ✅ **Keep the unquantized tensors bf16 in RAM.** `head.weight` alone is 1.06 GB
   on disk and 2.1 GB resident, because the PLAIN role expands everything to
   f32 at load. Together with the compressor projections that is ~3.5 GB held
   at twice the necessary width — the difference between fitting and swapping
   on a 16 GB machine. Done (2026-08-06): a new `DSV4_T_PLAIN_BF16` role keeps
   `head.weight` and the compressor `wkv`/`wgate` projections bf16 (2
   bytes/element) in RAM; `w_matmul` decodes bf16→f32 on the fly, which is
   bit-exact, so the numpy oracle in `test_dsv4_fixture.py` is unchanged and
   still passes (rel-L2 1e-5). The role refuses a tensor that is not actually
   bf16 — F16 is also 2 bytes/element but a bf16 decode of F16 bits is a
   silent misread, so the dtype is checked and named. Remaining small PLAIN
   tensors (norms, `hc_*`, `ape`, `attn_sink`, `gate_bias`) still consume
   `w->f` directly in the primitives, so they stay f32 — only matmul-consumed
   dense tensors halve.
5. ✅ **Exact prefetch on the hash layers** — done (2026-08-07). On layers
   `l < n_hash_layers` the router is a lookup, not a prediction: `tid2eid` maps
   the input token id straight to the K experts that token will use, and
   `dsv4_route_hash` picks exactly that set (weights from the gate logits, the
   SET from the table). So the working set is knowable with **zero forward
   compute** — the one place in this repository where expert placement needs no
   routing heat, no learned pin, no one-layer-ahead lookahead. `hash_prefetch`
   collects the union over a chunk (prefill, issued before the layer's attention
   sublayer) or a single token (decode, issued at `forward` entry for all hash
   layers at once) and WILLNEEDs it, so those reads overlap the dense work that
   precedes the FFN instead of starting cold after it. Exact by construction and
   a hint only — never changes what is read, so the logits are byte-identical
   (the fixture's hash-routed layer 0 exercises both paths; `[HASH]` lines under
   `DSV4_DEBUG` show the union size). `--direct` skips it like every prefetch.
6. ⏳ **DeepSeek's own torch reference on the real checkpoint.** Partially done
   (2026-08-08): `tools/dsv4_real_oracle.py` runs the independent numpy
   reference on the real weights and the engine matches to rel-L2 ~4e-07 with
   identical argmax on every step — the byte-decode and streaming halves of the
   gap are closed. What remains is the *transcription* half, which only the
   published `inference/model.py` + `kernel.py` (torch+tilelang+CUDA, needs a
   GPU machine that can hold 155 GB) can settle. Nothing else in this repository
   is implementation-independent of this fork's reading of that file.

---

*Engine and tooling are unmerged work in progress. PRs in this repository go
against `dev`, not `main` — see [CONTRIBUTING.md](../CONTRIBUTING.md).*
