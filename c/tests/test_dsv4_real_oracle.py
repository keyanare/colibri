#!/usr/bin/env python3
"""Oracle on real checkpoint bytes, gated by what is present.

The real-weight oracle (tools/dsv4_real_oracle.py) reads safetensors shards and
decodes fp8/fp4/bf16 with the same byte semantics as the engine, then runs the
independent numpy reference (the same RefModel the synthetic fixture uses).
This test has two halves:

1. Always: the oracle must reproduce the synthetic fixture to rel-L2 1e-5 and
   matching argmax. The fixture generator writes the same on-disk dtypes, so
   this validates the shard-offset arithmetic and the decode path -- on a
   model small enough to run in milliseconds.

2. If DSV4_REAL_CHECKPOINT points at a real checkpoint (the fork's working copy
   lives at ~/dsv4): run engine + oracle on a SHORT prompt and require argmax
   agreement. This is the token-exact oracle on real weights the roadmap asks
   for; it skips loudly when the path is absent so CI without a 155 GB download
   still passes.
"""
import json
import os
import shutil
import subprocess
import sys
import tempfile
import unittest

HERE = os.path.dirname(os.path.abspath(__file__))
CDIR = os.path.dirname(HERE)
TOOL = os.path.join(CDIR, "tools", "dsv4_real_oracle.py")
FIX = os.path.join(CDIR, "tools", "make_tiny_dsv4.py")
ENGINE = os.path.join(CDIR, "deepseek_v4" + (".exe" if os.name == "nt" else ""))
IDS = "3,7,1,5,2,9,4,11"
NGEN = 4

try:
    import numpy  # noqa: F401
    HAVE_NUMPY = True
except ImportError:
    HAVE_NUMPY = False

REAL = os.environ.get("DSV4_REAL_CHECKPOINT") or os.path.expanduser("~/dsv4")


@unittest.skipUnless(HAVE_NUMPY, "numpy not installed (pip install numpy)")
@unittest.skipUnless(os.path.exists(ENGINE), "deepseek_v4 not built (make deepseek_v4)")
class RealWeightOracle(unittest.TestCase):
    def run_tool(self, args):
        return subprocess.run([sys.executable, TOOL] + args,
                              capture_output=True, text=True, cwd=CDIR)

    def test_oracle_reproduces_synthetic_fixture(self):
        """The fixture generator is the ground truth: same dtypes, same layout,
        known values. If the oracle's shard reading or decode drifts, it cannot
        match the generator's own values to 1e-5."""
        with tempfile.TemporaryDirectory() as d:
            gen = subprocess.run([sys.executable, FIX, d],
                                 capture_output=True, text=True, cwd=CDIR)
            self.assertEqual(gen.returncode, 0, gen.stderr)
            run = subprocess.run([ENGINE, d, IDS, "--ids", "--ngen", str(NGEN),
                                  "--dump-logits", os.path.join(d, "got.json"),
                                  "--quiet"],
                                 capture_output=True, text=True, cwd=CDIR)
            self.assertEqual(run.returncode, 0, run.stderr)
            chk = self.run_tool([d, "--ids", IDS, "--ngen", str(NGEN),
                                 "--out", os.path.join(d, "ref.json")])
            self.assertEqual(chk.returncode, 0, chk.stderr)
            cmp = self.run_tool([d, "--check", os.path.join(d, "got.json"),
                                 "--ref", os.path.join(d, "ref.json")])
            self.assertEqual(cmp.returncode, 0, cmp.stdout + cmp.stderr)
            self.assertIn("ok", cmp.stdout)

    @unittest.skipUnless(os.path.isdir(REAL),
                         f"real checkpoint not present ({REAL}) -- set "
                         "DSV4_REAL_CHECKPOINT to enable the real-weight oracle")
    def test_real_checkpoint_is_token_exact(self):
        """Engine logits vs the independent numpy reference on the REAL weights.
        rel-L2 must be small and every argmax must agree -- that is the
        token-exact oracle. Slow (the numpy reference is ~20 s/token on this
        model), so keep the prompt short."""
        d = os.path.join(tempfile.mkdtemp(), "real")
        os.makedirs(d, exist_ok=True)
        got = os.path.join(d, "got.json")
        run = subprocess.run([ENGINE, REAL, "0,1,2,3,4", "--ids", "--ngen", "1",
                              "--dump-logits", got, "--quiet"],
                             capture_output=True, text=True, cwd=CDIR)
        self.assertEqual(run.returncode, 0, run.stderr)
        cmp = self.run_tool([REAL, "--ids", "0,1,2,3,4", "--ngen", "1",
                             "--check", got, "--ref", os.path.join(d, "ref.json")])
        self.assertEqual(cmp.returncode, 0, cmp.stdout + cmp.stderr)
        self.assertIn("ok", cmp.stdout)
        shutil.rmtree(d)


if __name__ == "__main__":
    unittest.main()
