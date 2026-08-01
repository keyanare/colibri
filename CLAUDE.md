# Working in this repository

colibrì runs frontier MoE models from disk in pure C. Four engines ship
upstream (`c/colibri.c` = GLM-5.2, `kimi_k3.c`, `inkling.c`, `olmoe.c`); this
fork adds a fifth, `c/deepseek_v4.c`.

**Read `docs/deepseek_v4.md` before touching anything DeepSeek-V4.** It has the
architecture, the file map, what is verified and what is not, and the roadmap.
Do not re-derive that by reading the sources.

## Commands

```bash
make -C c check                 # clean + portable build + full suite; MUST be 0 warnings
make -C c test-c                # C tests only
make -C c test-python           # Python tests (many skip without numpy)
make -C c deepseek_v4 ARCH=native
```

Python deps are NOT installed system-wide: Homebrew's python is
externally-managed (PEP 668). Tests needing numpy or `tokenizers` skip loudly
unless run with a venv:

```bash
python3 -m venv /tmp/venv && /tmp/venv/bin/pip install numpy tokenizers
PYTHON=/tmp/venv/bin/python make -C c test-python
```

## Conventions worth knowing before you trip on them

- **`TEST_BINS` is derived from Makefile rules** (`c/Makefile`, the `TEST_RULES`
  sed). Adding a `tests/test_x$(EXE):` rule makes it a gate automatically. A
  harness that takes arguments must go in `TEST_EXCLUDE` or `make test-c` will
  run it bare and fail.
- **Engine docs live in `docs/`**, not the root. The root is for cross-cutting
  topics (`GPU_BACKENDS.md`).
- **Loaders refuse rather than guess.** `qt_resolve_fmt` in `c/colibri.c` is the
  reference: when byte arithmetic is ambiguous it names the ambiguity and exits
  instead of picking. Match that discipline in new code — a silent misread is
  worse than a loud stop.
- **Quantized weights are never re-encoded.** QAT-trained formats (MXFP4) are
  passed through byte-for-byte; re-quantizing them only adds error.
- **Upstream takes PRs against `dev`, not `main`.** This fork commits straight
  to `main` — see the memory note on the workflow.

## Where things are

| | |
|---|---|
| weight formats, the fmt registry | `docs/FORMATS.md`, `c/quant.h` |
| safetensors + dtypes | `c/st.h` |
| tokenizer, 4 pre-tokenizer families | `c/tok.h` |
| DeepSeek-V4 primitives / manifest | `c/dsv4.h`, `c/dsv4_model.h` |
| DeepSeek-V4 tooling | `c/tools/dsv4_*.py`, `c/tools/make_tiny_dsv4.py` |

## Standing caveat

`deepseek_v4` has **never been run against the real checkpoint**. It was written
from DeepSeek's published reference implementation. It passes a synthetic
fixture against an independent numpy oracle (rel-L2 1e-5) and its tokenizer
matches HF `tokenizers` 356/356 on the real vocabulary — but that is
self-consistency, not correctness. Say so when it matters; do not describe it as
working.

`./c/deepseek_v4 <model_dir> --preflight` checks a checkpoint's names and shapes
without reading weight bytes. Run it before anything else on a real model.
