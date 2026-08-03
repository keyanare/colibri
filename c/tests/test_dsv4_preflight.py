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

    def variant(self, name, mutate, extra=b""):
        """A copy of the good fixture with its header mutated in place.

        `extra` is appended after the original blob, so a mutation that needs
        MORE bytes than the tensor it replaces (a widened dtype) can point its
        data_offsets past the end of the original data."""
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
            f.write(extra)
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
                         lambda h: h.pop("layers.0.ffn.experts.3.w1.scale"))
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

    def _widen_tid2eid(self, name):
        """Rewrite layers.0.ffn.gate.tid2eid as int64 holding the same values,
        appended past the original blob. Returns the variant directory."""
        import numpy as np
        key = "layers.0.ffn.gate.tid2eid"
        e = self.hdr[key]
        a, b = e["data_offsets"]
        vals = np.frombuffer(self.blob[a:b], dtype="<i4").astype("<i8")
        wide = vals.tobytes()

        def widen(h):
            h[key]["dtype"] = "I64"
            h[key]["data_offsets"] = [len(self.blob), len(self.blob) + len(wide)]
        return self.variant(name, widen, extra=wide)

    def test_i64_index_table_is_not_a_size_error(self):
        """torch's default dtype for an index tensor is int64, so tid2eid can
        legitimately arrive 8 bytes wide. Preflight sizes it from the config, so
        without knowing that it reports a correct table as the wrong size."""
        d = self._widen_tid2eid("i64tab")
        r = self.run_preflight(d)
        self.assertEqual(r.returncode, 0, r.stderr)
        self.assertIn("PREFLIGHT OK", r.stderr)

    def test_i64_index_table_loads_to_the_same_routing(self):
        """The load path narrows int64 -> int32. This is the check that it
        narrows rather than misreads: same table, same width-independent
        logits. A 2x byte span read into a 4-byte-per-entry buffer would have
        overrun it long before the comparison."""
        d = self._widen_tid2eid("i64run")
        # the fixture's own weights, minus the header the variant rewrote
        for f in ("ref_dsv4.json",):
            src = os.path.join(self.good, f)
            if os.path.exists(src):
                shutil.copy(src, os.path.join(d, f))
        ids, ngen = "3,7,1,5,2,9,4,11", "2"
        outs = []
        for where in (self.good, d):
            dump = os.path.join(where, "logits.json")
            r = subprocess.run([ENGINE, where, ids, "--ids", "--ngen", ngen,
                                "--dump-logits", dump, "--quiet"],
                               capture_output=True, text=True, cwd=CDIR)
            self.assertEqual(r.returncode, 0, r.stderr)
            with open(dump) as f:
                outs.append(json.load(f))
        self.assertEqual(outs[0], outs[1])

    def test_u8_spelling_of_fp8_weights_still_loads(self):
        """The fixture declares fp8 weights F8_E4M3, like the real checkpoint.
        colibri's OWN containers spell the identical layout U8, and both must
        keep loading -- the byte count is the same, only the label differs."""
        def relabel(h):
            h["layers.0.attn.wkv.weight"]["dtype"] = "U8"
        d = self.variant("u8fp8", relabel)
        r = self.run_preflight(d)
        self.assertEqual(r.returncode, 0, r.stderr)
        self.assertIn("PREFLIGHT OK", r.stderr)

    def test_unsupported_dtype_names_the_tensor(self):
        """st_init walks EVERY tensor in the container, including ones the
        manifest never covers, so a dtype it cannot read must say which tensor
        and which shard. The bare 'unsupported dtype: X' it used to print sends
        the reader hunting through 34k names by hand."""
        def bad(h):
            h["mtp.0.markov.state"] = {"dtype": "U64", "shape": [4],
                                       "data_offsets": [0, 32]}
        d = self.variant("baddtype", bad)
        r = self.run_preflight(d)
        self.assertNotEqual(r.returncode, 0)
        self.assertIn("mtp.0.markov.state", r.stderr)
        self.assertIn("U64", r.stderr)
        self.assertIn("model.safetensors", r.stderr)

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
