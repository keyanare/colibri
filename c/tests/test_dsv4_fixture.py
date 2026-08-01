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
                             "layers.0.ffn.experts.0.w1.weight.scale",  # mxfp4 sidecar
                             "layers.0.attn.wq_a.weight.scale",         # fp8 ue8m0 sidecar
                             "hc_head_fn"):                       # head collapse gates
                self.assertIn(required, names, f"fixture is missing {required}")
            self.assertNotIn("layers.0.attn.compressor.ape", names,
                             "layer 0 is ratio-0 and must own no compressor")


if __name__ == "__main__":
    unittest.main()
