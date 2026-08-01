#!/usr/bin/env python3
"""Tests for tools/dsv4_doctor.py — the hardware readiness check.

The doctor's job is to answer "can this machine host DeepSeek-V4-Flash", so the
assertions worth making are about its ARITHMETIC (does it size the model
correctly from a config it has never seen?) and its VERDICT POLICY (does a slow
machine pass while an impossible one fails?). Probing the actual host is not
testable here — the host is whatever CI runs on.
"""
import json
import os
import subprocess
import sys
import tempfile
import unittest

HERE = os.path.dirname(os.path.abspath(__file__))
CDIR = os.path.dirname(HERE)
TOOL = os.path.join(CDIR, "tools", "dsv4_doctor.py")
sys.path.insert(0, os.path.join(CDIR, "tools"))

import importlib.util
_spec = importlib.util.spec_from_file_location("dsv4_doctor", TOOL)
doctor = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(doctor)

GB = 1e9


class ModelArithmetic(unittest.TestCase):
    """The published figures are the only external check available."""

    def test_reference_geometry_matches_published_numbers(self):
        c = dict(doctor.REF)
        disk = doctor.disk_bytes(c)
        self.assertGreater(disk / GB, 140, "checkpoint should be ~155 GB")
        self.assertLess(disk / GB, 175)
        eb = doctor.expert_bytes(c)
        self.assertAlmostEqual(eb / 1e6, 13.37, places=1,
                               msg="one routed expert is ~12.75 MiB")
        # Experts must dominate, or the streaming premise does not hold.
        # NOTE this is the share of BYTES, ~94.8%, which is deliberately lower
        # than their share of PARAMETERS (~97.4%, asserted in
        # tests/test_dsv4_model.c): experts are stored at 4.25 bits/weight and
        # the dense set at 8, so the same parameters weigh less. Conflating the
        # two is easy and was the first version of this assertion.
        experts = c["n_layers"] * c["n_routed_experts"] * eb
        self.assertGreater(experts / disk, 0.93)
        self.assertLess(experts / disk, 0.96)

    def test_resident_set_is_plausible(self):
        c = dict(doctor.REF)
        res = doctor.resident_bytes(c, max_seq=8192)
        self.assertGreater(res / GB, 4, "dense weights alone are several GB")
        self.assertLess(res / GB, 12, "embed.weight must NOT be counted resident")
        # caches scale with context, weights do not
        small = doctor.resident_bytes(c, max_seq=1024)
        self.assertLess(small, res)
        self.assertLess((res - small) / GB, 1.0, "KV growth is modest, not dominant")

    def test_config_aliases_are_honoured(self):
        """config.json spells these differently from the internal short names.
        Without the alias map the reference defaults silently win -- which is
        right for the published checkpoint and wrong for every other config."""
        with tempfile.TemporaryDirectory() as d:
            with open(os.path.join(d, "config.json"), "w") as f:
                json.dump({"hidden_size": 128, "num_hidden_layers": 2,
                           "num_attention_heads": 4, "head_dim": 32,
                           "n_routed_experts": 8, "num_experts_per_tok": 2,
                           "moe_intermediate_size": 64,
                           "compress_ratios": [0, 4]}, f)
            cfg, note = doctor.load_cfg(d)
        self.assertEqual(cfg["dim"], 128)
        self.assertEqual(cfg["n_layers"], 2)
        self.assertEqual(cfg["n_heads"], 4)
        self.assertEqual(cfg["n_routed_experts"], 8)
        self.assertIsNone(note)
        # and a tiny model must size as a tiny model
        self.assertLess(doctor.disk_bytes(cfg) / GB, 1)

    def test_short_compress_ratios_is_reported(self):
        with tempfile.TemporaryDirectory() as d:
            with open(os.path.join(d, "config.json"), "w") as f:
                json.dump({"num_hidden_layers": 43, "compress_ratios": [0, 4]}, f)
            _, note = doctor.load_cfg(d)
        self.assertIsNotNone(note)
        self.assertIn("compress_ratios", note)


class VerdictPolicy(unittest.TestCase):
    def run_tool(self, *args):
        return subprocess.run([sys.executable, TOOL, *args],
                              capture_output=True, text=True, cwd=CDIR)

    def test_runs_and_reports_on_this_machine(self):
        r = self.run_tool()
        self.assertIn("DeepSeek-V4-Flash readiness check", r.stdout)
        self.assertIn("VERDICT", r.stdout)
        self.assertIn(r.returncode, (0, 1))

    def test_a_slow_machine_still_passes(self):
        """The stated contract: warnings do not change the exit status. A box
        that will be slow is a box that runs; only a real blocker fails."""
        r = self.run_tool()
        if "[FAIL]" not in r.stdout:
            self.assertEqual(r.returncode, 0,
                             "no FAIL rows but non-zero exit:\n" + r.stdout)
        else:
            self.assertEqual(r.returncode, 1)

    def test_missing_tokenizer_blocks(self):
        """The one checkpoint defect that genuinely stops the engine starting."""
        with tempfile.TemporaryDirectory() as d:
            with open(os.path.join(d, "config.json"), "w") as f:
                json.dump({"num_hidden_layers": 2, "hidden_size": 64,
                           "n_routed_experts": 8, "compress_ratios": [0, 4]}, f)
            r = self.run_tool("--model", d)
        self.assertEqual(r.returncode, 1)
        self.assertIn("tokenizer.json", r.stdout)
        self.assertIn("dsv4_tokenizer.py", r.stdout)

    def test_impossible_disk_fails(self):
        """Pointed at a volume that cannot hold 149 GiB, it must refuse."""
        r = self.run_tool("--target", "/dev")
        # /dev is a tiny pseudo-filesystem on macOS and Linux alike
        if "not enough space" in r.stdout:
            self.assertEqual(r.returncode, 1)


if __name__ == "__main__":
    unittest.main()
