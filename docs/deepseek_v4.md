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
> **`--preflight` now passes on the real `-0731` container: 34223/34223 names
> and shapes matched** (2026-08-03, four corrections deep — see *Which
> checkpoint*). That is a header-only check. **No weight byte of the real
> checkpoint has ever been read by this engine, and no forward pass has ever
> been run on it.** Everything past the manifest is still unverified.

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
c/tools/dsv4_tokenizer.py       vocabulary validator + HF cross-check
c/tools/dsv4_doctor.py          hardware readiness check
c/tools/gen_unicode_dsv4.py     regenerates tok_unicode_dsv4.h
c/tests/test_dsv4.c             17 primitive tests
c/tests/test_dsv4_model.c       manifest / shape tests
c/tests/test_dsv4_fixture.py    end-to-end vs the numpy oracle
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
| `--preflight` | check names and shapes against the manifest, read no weights |

`dsv4_doctor.py` suggests an `--expert-gb` that fits your RAM.

---

## Hardware

Disk is the hard requirement: **~149 GiB** for the checkpoint, and it cannot be
compressed further (the experts are already 4.25 bits/weight).

The engine is I/O-bound, and on any machine where the expert cache is smaller
than a few percent of a layer's 256 experts, essentially every token streams its
whole working set: **6 experts × 43 layers × 13.4 MB ≈ 3.4 GB read per token**.

| | Apple M5, 16 GB, ~5 GB/s SSD | i7-11700, 32 GB, 3 GB/s NVMe |
|---|---|---|
| resident | ~9.7 GB | ~9.7 GB |
| expert cache | 3–4 GB (~5 experts/layer) | ~20 GB (~35 experts/layer) |
| expert kernels | **scalar** (no NEON path yet) | AVX2 |
| estimate | ~1.5–3 s/token | ~1.5–3 s/token |

They land in the same place for different reasons: the Mac has a faster SSD but
a cache too small to help; the PC has a usable cache but half the bandwidth.
Neither has a GPU path — **there is no CUDA or Metal backend for this engine.**

Prefill runs one token at a time, so a 50-token prompt costs 50 forward passes
before the first generated token.

---

## What is actually tested

| claim | evidence |
|---|---|
| shapes and tensor wiring | the derived manifest reproduces DeepSeek's own published figures: **284.3B** parameters (published 284B), **13.3B** active (published 13B), **155 GB** on disk (published 160 GB), 97.4% in routed experts. None of those numbers were used to build the manifest. |
| the whole forward pass | a synthetic checkpoint in the real on-disk format, run through the engine and an independent numpy implementation: **rel-L2 1.0e-05**, argmax identical on every step. `make test-python` (needs numpy). |
| tokenizer | **356/356** cases match HF `tokenizers` on the real vocabulary — punctuation, contractions, CJK, Korean, Thai, Arabic, Hebrew, Cyrillic, combining marks, ZWJ emoji, CRLF, URLs, code, 200 random strings. Forced down the `cl100k` path the same vocabulary scores **341/356**, which is why the fourth family exists. |
| primitives | 17 test groups, most of them aimed at a specific way a plausible reimplementation goes wrong (untransposed `comb`, symmetric SwiGLU clamp, sink folded into the loop, per-token instead of per-dimension pooling). |

**What none of this proves:** that the architecture was transcribed correctly.
The engine and the numpy oracle were written from the same reading of the same
file; a shared misreading passes both.

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

1. **NEON `matmul_mxfp4`** — the expert kernel is scalar on Apple Silicon and it
   is the hot loop. `quant.h` already has NEON idioms next to it.
2. **Batched expert I/O** — one `pread` per expert instead of six, `O_DIRECT`,
   and overlap with compute. `colibri.c` and `kimi_k3.c` already do all three.
3. **Keep the unquantized tensors bf16 in RAM.** `head.weight` alone is 1.06 GB
   on disk and 2.1 GB resident, because the PLAIN role expands everything to
   f32 at load. Together with the compressor projections that is ~3.5 GB held
   at twice the necessary width — the difference between fitting and swapping
   on a 16 GB machine.
4. **Exact prefetch on the hash layers** — the one place in this repository where
   the working set is knowable in advance.
5. **A token-exact oracle** on the real checkpoint. Everything above is
   provisional until this exists.

---

*Engine and tooling are unmerged work in progress. PRs in this repository go
against `dev`, not `main` — see [CONTRIBUTING.md](../CONTRIBUTING.md).*
