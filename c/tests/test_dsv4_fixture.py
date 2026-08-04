#!/usr/bin/env python3
"""End-to-end gate for the DeepSeek-V4 engine against a synthetic checkpoint.

Generates a tiny random model in the real on-disk format (tools/make_tiny_dsv4.py),
runs `deepseek_v4` over it in --ids oracle mode, and requires the logits to match
an independent numpy forward pass. Same role as the inkling-oracle CI job, minus
the 160 GB download: it cannot prove the architecture was transcribed correctly
from DeepSeek's reference, but it does prove the C engine and a second
implementation of that same reading agree -- which is where loader, cache,
state-machine and dequant bugs show up.

Skips (loudly) when numpy is absent or the engine has not been built, matching
how the rest of this suite treats optional dependencies.
"""
import json
import os
import subprocess
import sys
import tempfile
import unittest

HERE = os.path.dirname(os.path.abspath(__file__))
CDIR = os.path.dirname(HERE)
TOOL = os.path.join(CDIR, "tools", "make_tiny_dsv4.py")
ENGINE = os.path.join(CDIR, "deepseek_v4" + (".exe" if os.name == "nt" else ""))
IDS = "3,7,1,5,2,9,4,11"
NGEN = 4

try:
    import numpy  # noqa: F401
    HAVE_NUMPY = True
except ImportError:
    HAVE_NUMPY = False


@unittest.skipUnless(HAVE_NUMPY, "numpy not installed (pip install numpy)")
@unittest.skipUnless(os.path.exists(ENGINE), "deepseek_v4 not built (make deepseek_v4)")
class TinyDsv4Oracle(unittest.TestCase):
    def test_engine_matches_numpy_reference(self):
        with tempfile.TemporaryDirectory() as d:
            gen = subprocess.run([sys.executable, TOOL, d],
                                 capture_output=True, text=True, cwd=CDIR)
            self.assertEqual(gen.returncode, 0,
                             f"fixture generation failed:\n{gen.stdout}\n{gen.stderr}")
            self.assertTrue(os.path.exists(os.path.join(d, "model.safetensors")))
            self.assertTrue(os.path.exists(os.path.join(d, "ref_dsv4.json")))

            got = os.path.join(d, "got.json")
            run = subprocess.run([ENGINE, d, IDS, "--ids", "--ngen", str(NGEN),
                                  "--dump-logits", got, "--quiet"],
                                 capture_output=True, text=True, cwd=CDIR)
            self.assertEqual(run.returncode, 0,
                             f"engine failed:\n{run.stdout}\n{run.stderr}")
            self.assertTrue(os.path.exists(got), "engine wrote no logits dump")

            chk = subprocess.run([sys.executable, TOOL, d, "--check", got],
                                 capture_output=True, text=True, cwd=CDIR)
            self.assertEqual(chk.returncode, 0,
                             f"engine disagrees with the reference:\n{chk.stdout}\n{chk.stderr}")
            self.assertIn("ok", chk.stdout)

    def test_prefill_chunking_is_bit_identical(self):
        """Batched prefill reorders WHICH TOKEN a weight is read for, and
        nothing else: sequential state still advances per token inside each
        layer, dense matmuls stay at S=1, and each token's expert contributions
        are summed in route order rather than the expert-major order they were
        computed in.

        So the logits must match to the byte across chunk sizes -- not to a
        tolerance. A tolerance here would pass a version that quietly changed
        reduction order, which is the one thing this design pays staging memory
        to avoid."""
        with tempfile.TemporaryDirectory() as d:
            subprocess.run([sys.executable, TOOL, d], capture_output=True,
                           text=True, cwd=CDIR, check=True)
            ref = None
            for chunk in ("1", "2", "5", "64"):
                out = os.path.join(d, f"got{chunk}.json")
                r = subprocess.run([ENGINE, d, IDS, "--ids", "--ngen", "2",
                                    "--chunk", chunk, "--dump-logits", out, "--quiet"],
                                   capture_output=True, text=True, cwd=CDIR)
                self.assertEqual(r.returncode, 0, r.stderr)
                with open(out, "rb") as f:
                    blob = f.read()
                if ref is None:
                    ref = blob
                    # chunk 5 does not divide the 7-token prefill: the last
                    # chunk is short, which is where an off-by-one would live
                    self.assertGreater(len(ref), 0)
                else:
                    self.assertEqual(blob, ref,
                                     f"--chunk {chunk} changed the logits")

    def test_gap_fallback_is_bit_identical(self):
        """--gap parks a PLAIN tensor inside expert 0, so its six on-disk spans
        are NOT contiguous and the loader must fall back to per-piece preads
        packed into the same 4K slab. The other experts stay contiguous, so one
        model exercises BOTH the coalesced single-pread path and the fallback.
        The engine must produce the same logits (to byte) either way -- the
        oracle reference is derived from the same on-disk gap layout."""
        with tempfile.TemporaryDirectory() as d:
            subprocess.run([sys.executable, TOOL, d, "--gap"], capture_output=True,
                           text=True, cwd=CDIR, check=True)
            got = os.path.join(d, "got.json")
            run = subprocess.run([ENGINE, d, IDS, "--ids", "--ngen", str(NGEN),
                                  "--dump-logits", got, "--quiet"],
                                 capture_output=True, text=True, cwd=CDIR)
            self.assertEqual(run.returncode, 0,
                             f"engine failed on gap fixture:\n{run.stderr}")
            # --check auto-detects the __gap tensors already on disk.
            chk = subprocess.run([sys.executable, TOOL, d, "--check", got],
                                 capture_output=True, text=True, cwd=CDIR)
            self.assertEqual(chk.returncode, 0,
                             f"gap fallback disagrees with reference:\n{chk.stdout}\n{chk.stderr}")
            self.assertIn("ok", chk.stdout)

    def test_direct_reads_match(self):
        """--direct routes expert reads through the twin O_DIRECT/F_NOCACHE fd
        with a 4K-aligned window (head slack + tail fetched buffered). That path
        must read the same bytes as the buffered one -- bypassing the page cache
        changes speed, never content. On macOS the aligned-window logic and on
        Linux O_DIRECT are both covered by the same correctness gate."""
        with tempfile.TemporaryDirectory() as d:
            subprocess.run([sys.executable, TOOL, d], capture_output=True,
                           text=True, cwd=CDIR, check=True)
            got = os.path.join(d, "got.json")
            run = subprocess.run([ENGINE, d, IDS, "--ids", "--ngen", str(NGEN),
                                  "--direct", "--dump-logits", got, "--quiet"],
                                 capture_output=True, text=True, cwd=CDIR)
            self.assertEqual(run.returncode, 0,
                             f"engine failed under --direct:\n{run.stderr}")
            chk = subprocess.run([sys.executable, TOOL, d, "--check", got],
                                 capture_output=True, text=True, cwd=CDIR)
            self.assertEqual(chk.returncode, 0,
                             f"--direct disagrees with reference:\n{chk.stdout}\n{chk.stderr}")
            self.assertIn("ok", chk.stdout)

    def test_fixture_exercises_every_layer_class(self):
        """The fixture is only a gate if it reaches the branches that differ.
        Layer 0 must be sliding-window-only AND hash-routed; layer 1 must own a
        compressor, an indexer, and score-based routing."""
        with tempfile.TemporaryDirectory() as d:
            subprocess.run([sys.executable, TOOL, d], capture_output=True,
                           text=True, cwd=CDIR, check=True)
            with open(os.path.join(d, "config.json")) as f:
                cfg = json.load(f)
            self.assertEqual(cfg["compress_ratios"][:2], [0, 4])
            self.assertEqual(cfg["num_hash_layers"], 1)
            with open(os.path.join(d, "model.safetensors"), "rb") as f:
                hdr_len = int.from_bytes(f.read(8), "little")
                names = set(json.loads(f.read(hdr_len)).keys())
            for required in ("layers.0.ffn.gate.tid2eid",        # hash routing
                             "layers.1.ffn.gate.bias",            # noaux_tc routing
                             "layers.1.attn.compressor.ape",      # CSA compressor
                             "layers.1.attn.indexer.wq_b.weight", # lightning indexer
                             "layers.1.attn.indexer.compressor.ape",
                             # the sidecar REPLACES the .weight suffix -- appending
                             # it instead is what made every quantized tensor read
                             # as "no scale" against the real container
                             "layers.0.ffn.experts.0.w1.scale",   # mxfp4 sidecar
                             "layers.0.attn.wq_a.scale",          # fp8 ue8m0 sidecar
                             "hc_head_fn"):                       # head collapse gates
                self.assertIn(required, names, f"fixture is missing {required}")
            self.assertNotIn("layers.0.attn.compressor.ape", names,
                             "layer 0 is ratio-0 and must own no compressor")


if __name__ == "__main__":
    unittest.main()
