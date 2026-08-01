#!/usr/bin/env python3
"""Tests for `deepseek_v4 --preflight`.

Preflight exists so that a checkpoint mismatch costs one message instead of a
160 GB download followed by a crash mid-load. So the tests that matter are the
NEGATIVE ones: a clean fixture passing proves nothing on its own, since a
preflight that always says OK would pass it too.

Each case deliberately damages exactly one thing in a known-good synthetic
checkpoint (tools/make_tiny_dsv4.py) by rewriting the safetensors header, and
requires preflight to name it and exit non-zero. The last case is the inverse:
an unrecognised tensor must NOT fail, because the manifest deliberately does not
cover the MTP head.
"""
import json
import os
import shutil
import struct
import subprocess
import sys
import tempfile
import unittest

HERE = os.path.dirname(os.path.abspath(__file__))
CDIR = os.path.dirname(HERE)
TOOL = os.path.join(CDIR, "tools", "make_tiny_dsv4.py")
ENGINE = os.path.join(CDIR, "deepseek_v4" + (".exe" if os.name == "nt" else ""))

try:
    import numpy  # noqa: F401
    HAVE_NUMPY = True
except ImportError:
    HAVE_NUMPY = False


@unittest.skipUnless(HAVE_NUMPY, "numpy not installed (needed to build the fixture)")
@unittest.skipUnless(os.path.exists(ENGINE), "deepseek_v4 not built (make deepseek_v4)")
class Preflight(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls._tmp = tempfile.TemporaryDirectory()
        cls.good = os.path.join(cls._tmp.name, "good")
        subprocess.run([sys.executable, TOOL, cls.good], capture_output=True,
                       text=True, cwd=CDIR, check=True)
        with open(os.path.join(cls.good, "model.safetensors"), "rb") as f:
            hl = struct.unpack("<Q", f.read(8))[0]
            cls.hdr = json.loads(f.read(hl))
            cls.blob = f.read()

    @classmethod
    def tearDownClass(cls):
        cls._tmp.cleanup()

    def variant(self, name, mutate):
        """A copy of the good fixture with its header mutated in place."""
        d = os.path.join(self._tmp.name, name)
        os.makedirs(d, exist_ok=True)
        shutil.copy(os.path.join(self.good, "config.json"), os.path.join(d, "config.json"))
        hdr = json.loads(json.dumps(self.hdr))
        mutate(hdr)
        hj = json.dumps(hdr, separators=(",", ":")).encode()
        with open(os.path.join(d, "model.safetensors"), "wb") as f:
            f.write(struct.pack("<Q", len(hj)))
            f.write(hj)
            f.write(self.blob)
        return d

    def run_preflight(self, d):
        return subprocess.run([ENGINE, d, "--preflight"],
                              capture_output=True, text=True, cwd=CDIR)

    def test_clean_checkpoint_passes(self):
        r = self.run_preflight(self.good)
        self.assertEqual(r.returncode, 0, r.stderr)
        self.assertIn("PREFLIGHT OK", r.stderr)
        self.assertRegex(r.stderr, r"\d+/\d+ matched")

    def test_reads_no_weight_bytes(self):
        """The whole point: this must be a header-only pass. A fixture that
        takes measurable time here would mean the loader path leaked in."""
        r = self.run_preflight(self.good)
        self.assertIn("headers only, no weights read", r.stderr)

    def test_missing_tensor_is_named(self):
        d = self.variant("missing", lambda h: h.pop("layers.1.attn.wq_b.weight"))
        r = self.run_preflight(d)
        self.assertEqual(r.returncode, 1)
        self.assertIn("MISSING", r.stderr)
        self.assertIn("layers.1.attn.wq_b.weight", r.stderr)
        self.assertIn("PREFLIGHT FAILED", r.stderr)

    def test_wrong_size_reports_both_numbers(self):
        def half(h):
            e = h["layers.0.attn.wkv.weight"]
            a, b = e["data_offsets"]
            e["data_offsets"] = [a, a + (b - a) // 2]
        d = self.variant("size", half)
        r = self.run_preflight(d)
        self.assertEqual(r.returncode, 1)
        self.assertIn("SIZE", r.stderr)
        self.assertIn("expected", r.stderr)
        # the orphaned sidecar must NOT also surface as an unrelated problem
        self.assertNotIn("does not cover", r.stderr)

    def test_missing_scale_sidecar_is_caught(self):
        d = self.variant("scale",
                         lambda h: h.pop("layers.0.ffn.experts.3.w1.weight.scale"))
        r = self.run_preflight(d)
        self.assertEqual(r.returncode, 1)
        self.assertIn("NO SCALE", r.stderr)

    def test_unknown_tensor_is_not_an_error(self):
        """The MTP head is real but its names are unknown, so the manifest does
        not cover it. Reporting that as corruption would make a correct
        checkpoint look broken."""
        def add_mtp(h):
            h["mtp.0.attn.wq_a.weight"] = dict(h["layers.0.attn.wq_a.weight"])
        d = self.variant("extra", add_mtp)
        r = self.run_preflight(d)
        self.assertEqual(r.returncode, 0, r.stderr)
        self.assertIn("does not cover", r.stderr)
        self.assertIn("EXPECTED, not an error", r.stderr)
        self.assertIn("mtp.0.attn.wq_a.weight", r.stderr)

    def test_missing_tokenizer_is_reported_but_not_fatal(self):
        """The fixture ships no vocabulary, like the real checkpoint. That
        blocks RUNNING, not the shape check, so preflight says so and still
        exits 0 when the weights are fine."""
        r = self.run_preflight(self.good)
        self.assertEqual(r.returncode, 0)
        self.assertIn("tokenizer.json absent", r.stderr)
        self.assertIn("dsv4_tokenizer.py", r.stderr)


if __name__ == "__main__":
    unittest.main()
