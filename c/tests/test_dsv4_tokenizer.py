#!/usr/bin/env python3
"""Tests for tools/dsv4_tokenizer.py — the validator that stands between a
third-party vocabulary and the DeepSeek-V4 engine.

The published checkpoint ships no vocabulary, so this tool's whole job is to
REFUSE a wrong one. These tests are therefore mostly negative: each builds a
synthetic tokenizer.json with exactly one defect and requires a non-zero exit.
A validator that accepts everything would pass a happy-path test and be
worthless.

No network, no model, no numpy: everything is built in a temp dir.
"""
import json
import os
import subprocess
import sys
import tempfile
import unittest

HERE = os.path.dirname(os.path.abspath(__file__))
CDIR = os.path.dirname(HERE)
TOOL = os.path.join(CDIR, "tools", "dsv4_tokenizer.py")

SPECIALS = ["<｜begin▁of▁sentence｜>", "<｜end▁of▁sentence｜>",
            "<｜User｜>", "<｜Assistant｜>"]
# The cl100k Split regex, the family tok.h uses for a pattern with neither
# \p{Lu} nor \p{Han}.
CL100K = (r"(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}{1,3}"
          r"| ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+")
O200K = (r"(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?"
         r"[\p{Lu}\p{Lt}\p{Lm}\p{Lo}\p{M}]*[\p{Ll}\p{Lm}\p{Lo}\p{M}]+|\p{N}{1,3}|\s+")
# The REAL DeepSeek-V4-Flash main Split, verbatim from its tokenizer.json.
DSV4 = (r"[!\"#$%&'()*+,\\-./:;<=>?@\\[\\\\\\]^_`{|}~][A-Za-z]+"
        r"|[^\r\n\p{L}\p{P}\p{S}]?[\p{L}\p{M}]+"
        r"| ?[\p{P}\p{S}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+")


def make_tokenizer(vocab_size=64, pattern=CL100K, specials=None, bos=0, eos=1,
                   merges=None):
    """A minimal but structurally valid HF tokenizer.json."""
    specials = SPECIALS if specials is None else specials
    vocab, nid = {}, 0
    for s in specials:
        vocab[s] = nid
        nid += 1
    # fill the rest with byte-level singletons so ids span exactly vocab_size
    i = 0
    while nid < vocab_size:
        vocab[f"tok{i}"] = nid
        nid += 1
        i += 1
    tk = {
        "version": "1.0", "truncation": None, "padding": None,
        "added_tokens": [
            {"id": vocab[s], "content": s, "single_word": False, "lstrip": False,
             "rstrip": False, "normalized": False, "special": True}
            for s in specials
        ],
        "normalizer": None,
        "pre_tokenizer": {"type": "Sequence", "pretokenizers": [
            {"type": "Split", "pattern": {"Regex": pattern},
             "behavior": "Isolated", "invert": False},
            {"type": "ByteLevel", "add_prefix_space": False,
             "trim_offsets": True, "use_regex": False},
        ]},
        "post_processor": None,
        "decoder": {"type": "ByteLevel"},
        "model": {"type": "BPE", "dropout": None, "unk_token": None,
                  "continuing_subword_prefix": None, "end_of_word_suffix": None,
                  "fuse_unk": False, "byte_fallback": False, "ignore_merges": True,
                  "vocab": vocab, "merges": merges or []},
    }
    # keep bos/eos where the caller asked
    if bos is not None:
        tk["model"]["vocab"][SPECIALS[0]] = bos
    if eos is not None:
        tk["model"]["vocab"][SPECIALS[1]] = eos
    for a in tk["added_tokens"]:
        if a["content"] in tk["model"]["vocab"]:
            a["id"] = tk["model"]["vocab"][a["content"]]
    return tk


def make_config(vocab_size=64, bos=0, eos=1):
    return {"vocab_size": vocab_size, "bos_token_id": bos, "eos_token_id": eos,
            "hidden_size": 64, "num_hidden_layers": 2}


class Dsv4TokenizerValidator(unittest.TestCase):
    def run_tool(self, d, *args):
        return subprocess.run([sys.executable, TOOL, d, *args],
                              capture_output=True, text=True, cwd=CDIR)

    def setup_dir(self, d, cfg=None, tok=None):
        with open(os.path.join(d, "config.json"), "w") as f:
            json.dump(cfg or make_config(), f)
        src = os.path.join(d, "src.json")
        with open(src, "w", encoding="utf-8") as f:
            json.dump(tok if tok is not None else make_tokenizer(), f, ensure_ascii=False)
        return src

    def test_accepts_a_consistent_vocabulary(self):
        with tempfile.TemporaryDirectory() as d:
            src = self.setup_dir(d)
            r = self.run_tool(d, "--from", src)
            self.assertEqual(r.returncode, 0, r.stdout + r.stderr)
            self.assertTrue(os.path.exists(os.path.join(d, "tokenizer.json")))
            chk = self.run_tool(d, "--check")
            self.assertEqual(chk.returncode, 0, chk.stdout + chk.stderr)
            self.assertIn("ok", chk.stdout)

    def test_refuses_vocab_size_mismatch(self):
        with tempfile.TemporaryDirectory() as d:
            src = self.setup_dir(d, cfg=make_config(vocab_size=128),
                                 tok=make_tokenizer(vocab_size=64))
            r = self.run_tool(d, "--from", src)
            self.assertNotEqual(r.returncode, 0)
            self.assertIn("vocab_size", r.stdout + r.stderr)
            self.assertFalse(os.path.exists(os.path.join(d, "tokenizer.json")),
                             "a refused vocabulary must not be installed")

    def test_refuses_wrong_eos_id(self):
        """config.json's eos_token_id is what the engine stops on; a vocabulary
        that puts that string at a different id makes generation run past the
        end of every turn."""
        with tempfile.TemporaryDirectory() as d:
            tok = make_tokenizer(eos=7)
            src = self.setup_dir(d, cfg=make_config(eos=1), tok=tok)
            r = self.run_tool(d, "--from", src)
            self.assertNotEqual(r.returncode, 0)
            self.assertIn("eos_token_id", r.stdout + r.stderr)

    def test_refuses_missing_special(self):
        with tempfile.TemporaryDirectory() as d:
            tok = make_tokenizer(specials=SPECIALS[:3])   # no <｜Assistant｜>
            src = self.setup_dir(d, tok=tok)
            r = self.run_tool(d, "--from", src)
            self.assertNotEqual(r.returncode, 0)
            self.assertIn("Assistant", r.stdout + r.stderr)

    def test_refuses_unknown_pretokenizer_family(self):
        """The defect this tool exists for. tok.h picks a family by substring,
        so an unrecognised pattern does not fail — it tokenizes differently."""
        with tempfile.TemporaryDirectory() as d:
            weird = r"[[:alpha:]]+|[[:digit:]]+|."
            src = self.setup_dir(d, tok=make_tokenizer(pattern=weird))
            r = self.run_tool(d, "--from", src)
            self.assertNotEqual(r.returncode, 0)
            self.assertIn("family", (r.stdout + r.stderr).lower())
            # and the escape hatch is explicit, not silent
            r2 = self.run_tool(d, "--from", src, "--allow-unknown-family")
            self.assertEqual(r2.returncode, 0, r2.stdout + r2.stderr)
            self.assertIn("WARNING", r2.stdout)

    def test_reports_the_family_tok_h_will_choose(self):
        with tempfile.TemporaryDirectory() as d:
            src = self.setup_dir(d, tok=make_tokenizer(pattern=O200K))
            r = self.run_tool(d, "--from", src)
            self.assertEqual(r.returncode, 0, r.stdout + r.stderr)
            self.assertIn("o200k", r.stdout)
        with tempfile.TemporaryDirectory() as d:
            src = self.setup_dir(d, tok=make_tokenizer(pattern=CL100K))
            r = self.run_tool(d, "--from", src)
            self.assertEqual(r.returncode, 0, r.stdout + r.stderr)
            self.assertIn("cl100k", r.stdout)

    def test_recognises_the_real_deepseek_v4_pattern(self):
        """The pattern that motivated the whole tool. It contains neither
        \\p{Lu} nor \\p{Han}, so tok.h's substring dispatch sent it to cl100k
        before the dsv4 family existed -- measured at 341/356 agreement with HF
        `tokenizers` on a real vocabulary, i.e. wrong on 4% of cases with no
        error anywhere. It must now classify as dsv4 and be accepted."""
        with tempfile.TemporaryDirectory() as d:
            src = self.setup_dir(d, tok=make_tokenizer(pattern=DSV4))
            r = self.run_tool(d, "--from", src)
            self.assertEqual(r.returncode, 0, r.stdout + r.stderr)
            self.assertIn("dsv4", r.stdout)
            self.assertNotIn("cl100k", r.stdout)

    def test_a_dsv4_pattern_is_never_mistaken_for_cl100k(self):
        """Guards the specific regression: the discriminator is the ABSENCE of
        the contraction group, so a validator using `any` over generic
        fragments passes this and a correct one does not."""
        from importlib import util as _u
        spec = _u.spec_from_file_location("dsv4tok", TOOL)
        mod = _u.module_from_spec(spec); spec.loader.exec_module(mod)
        self.assertEqual(mod.family_of([DSV4]), "dsv4")
        self.assertEqual(mod.family_of([CL100K]), "cl100k")
        self.assertEqual(mod.family_of([O200K]), "o200k")
        self.assertNotIn("'s|'t|'re", DSV4,
                         "the DeepSeek-V4 pattern has no contraction group")

    def test_refuses_non_bpe_model(self):
        with tempfile.TemporaryDirectory() as d:
            tok = make_tokenizer()
            tok["model"]["type"] = "WordPiece"
            src = self.setup_dir(d, tok=tok)
            r = self.run_tool(d, "--from", src)
            self.assertNotEqual(r.returncode, 0)
            self.assertIn("BPE", r.stdout + r.stderr)

    def test_does_not_overwrite_without_force(self):
        with tempfile.TemporaryDirectory() as d:
            src = self.setup_dir(d)
            self.assertEqual(self.run_tool(d, "--from", src).returncode, 0)
            again = self.run_tool(d, "--from", src)
            self.assertNotEqual(again.returncode, 0)
            self.assertIn("--force", again.stdout + again.stderr)
            self.assertEqual(self.run_tool(d, "--from", src, "--force").returncode, 0)


class Dsv4TokenizerLoadsInTokH(unittest.TestCase):
    """The generated file must actually load in tok.h, not merely satisfy the
    validator. Uses the C harness when it has been built."""
    HARNESS = os.path.join(CDIR, "tests",
                           "test_tok_dsv4" + (".exe" if os.name == "nt" else ""))

    @unittest.skipUnless(os.path.exists(HARNESS), "test_tok_dsv4 not built")
    def test_tok_h_loads_and_reports_family(self):
        with tempfile.TemporaryDirectory() as d:
            with open(os.path.join(d, "config.json"), "w") as f:
                json.dump(make_config(), f)
            path = os.path.join(d, "tokenizer.json")
            with open(path, "w", encoding="utf-8") as f:
                json.dump(make_tokenizer(pattern=O200K), f, ensure_ascii=False)
            cases = os.path.join(d, "cases.txt")
            with open(cases, "w", encoding="utf-8") as f:
                f.write("tok0\t4\n")          # a single known singleton
            r = subprocess.run([self.HARNESS, path, cases],
                               capture_output=True, text=True, cwd=CDIR)
            self.assertIn("tok.h family: o200k", r.stderr,
                          f"harness output:\n{r.stdout}\n{r.stderr}")


if __name__ == "__main__":
    unittest.main()
