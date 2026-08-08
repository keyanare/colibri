# Working in this repository

colibrì runs frontier MoE models from disk in pure C. This fork's only addition
is the DeepSeek-V4 engine, `c/deepseek_v4.c`. Everything you touch that is
`dsv4_*`, `deepseek_v4.c`, `st.h` dtypes or the shared kernels should be read
as: the fork's job is the DSV4 engine, the rest is upstream.

**Read `docs/deepseek_v4.md` before touching anything DeepSeek-V4.** It has the
architecture, the verified-vs-unverified split and the roadmap. Do not
re-derive it from the sources.

## Commands

```bash
make -C c check                 # clean + portable build + full suite; MUST be 0 warnings
make -C c test-c                # C tests only (binary gates)
make -C c test-python           # Python tests (many skip loudly without numpy)
make -C c deepseek_v4 ARCH=native
```

The engine builds to `c/deepseek_v4` (in-tree, no build dir). The C tests run
via `c/tools/run_tests.py`; they are the gates, not `./deepseek_v4` directly.

Python deps are NOT installed system-wide (Homebrew Python is PEP 668
externally-managed). The working venv for this fork lives at `/tmp/venv`
(numpy + tokenizers + pytest):

```bash
/tmp/venv/bin/python -m pytest c/tests/test_dsv4_fixture.py -q   # needs engine built first
PYTHON=/tmp/venv/bin/python make -C c test-python
```

## DeepSeek-V4 engine

- **Never run against the real checkpoint, therefore unverified.** Written from
  DeepSeek's published reference, passes a synthetic numpy oracle (rel-L2 1e-5)
  and matches HF `tokenizers` 356/356 on the real vocab — that is
  self-consistency, not correctness. Say so when it matters; never call it
  "working".
- `./c/deepseek_v4 <model_dir> --preflight` checks names/shapes against the
  real `-0731` checkpoint without reading weight bytes. Run it first on a real
  model.
- Flags: `--ids` (oracle mode, takes an id list), `--ngen N`, `--dump-logits
  <path>`, `--timing` (per-stage decode breakdown), `--quiet`, `--direct`.
- Env vars: `DSV4_POOL=n` (async expert reader pool, default 4 threads),
  `DSV4_NO_ASYNC` (disable it), `DSV4_DEBUG`, `DSV4_FALLBACK`. Prefill is
  synchronous; decode uses the pool.
- Fixture loop (generator is itself the oracle): `python3
  c/tools/make_tiny_dsv4.py <dir>`, run the engine with `--ids --dump-logits
  --quiet`, then `make_tiny_dsv4.py <dir> --check got.json`. **Bit-identical**
  is asserted — a tolerance there would mask a quiet reordering bug, don't
  relax it.
- **Real-weight oracle** (token-exact on the real checkpoint): `python3
  c/tools/dsv4_real_oracle.py <model_dir> --ids 0,1,2,3,4 --ngen 1 --check
  got.json` runs the same numpy reference on the real weight bytes (rel-L2
  ~4e-07, argmax identical). The checkpoint on this machine lives at
  `/Users/keyanare/dsv4`; `test_dsv4_real_oracle.py` runs it when present and
  skips loudly otherwise. Both this oracle and the engine read
  `inference/model.py` the same way, so it does NOT close the transcription
  gap — only DeepSeek's torch reference would.

## macOS gotchas (the machine this fork is developed on)

- Apple Silicon M5, 16 GB RAM, **4 physical P-cores**. `omp_tune.h` sizes the
  OpenMP team to physical cores and deliberately does NOT enable spin-wait —
  spinning steals cores from the disk I/O pool, which is the real bottleneck.
- `posix_fadvise(WILLNEED)` is a **no-op stub on Darwin**. The prefetch path
  goes through `st_willneed()` in `c/st.h` (fcntl F_RDADVISE on macOS,
  posix_fadvise elsewhere). If prefetch seems dead on macOS, suspect this.
- Perf context: decode on this Mac is ~25 s/token (~0.04 tok/s) and that is
  the hardware ceiling (fp8/bf16 kernels at 4 P-cores + 128-bit NEON), not a
  bug. A WSL/Lenovo box does 0.27 tok/s. Profile with sample files, not
  guesses.
- Build without Homebrew libomp is single-threaded and only warns; if a build
  suddenly ignores OpenMP pragmas, `brew install libomp` went missing.
  `ARCH=native` on arm64 uses `-mcpu`, not `-march`.

## Conventions

- **`TEST_BINS` is derived from Makefile rules.** A `tests/test_x$(EXE):` rule
  makes it a gate automatically. A harness that takes arguments must be added
  to `TEST_EXCLUDE` (`test_tok_dsv4` is one — driven by
  `tools/dsv4_tokenizer.py --ctest`) or `make test-c` fails running it bare.
- **Loaders refuse rather than guess.** `qt_resolve_fmt` in `c/colibri.c` names
  the ambiguity and exits when byte arithmetic is ambiguous. Match that
  discipline: a silent misread is worse than a loud stop.
- **Quantized weights are never re-encoded.** MXFP4 (QAT-trained) is passed
  through byte-for-byte; re-quantizing adds error.
- Engine docs live in `docs/` (per-engine files like `docs/deepseek_v4.md`);
  the root holds only cross-cutting topics.

## Git workflow

- `origin` is upstream JustVugg/colibri (read-only: push is denied). The
  working remote is **`fork`** (`git@github.com:keyanare/colibri-deepseekv4flash.git`),
  and `main` tracks `fork/main`. Push with `git push fork main`.
- Upstream takes PRs against `dev`, not `main`. This fork commits straight to
  `main`.
- Do not commit the untracked sample-profile files (`stack.txt`, `stack2.txt`).

## Where things are

| | |
|---|---|
| weight formats, the fmt registry | `docs/FORMATS.md`, `c/quant.h` |
| safetensors + dtypes (+ `st_willneed`) | `c/st.h` |
| tokenizer, 4 pre-tokenizer families | `c/tok.h` |
| DeepSeek-V4 primitives / manifest | `c/dsv4.h`, `c/dsv4_model.h` |
| DSV4 tooling / fixture+oracle | `c/tools/dsv4_*.py`, `c/tools/make_tiny_dsv4.py` |
| OpenMP sizing, never spin-wait | `c/omp_tune.h` |
