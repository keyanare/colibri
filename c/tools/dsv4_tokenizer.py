#!/usr/bin/env python3
"""Prepare and validate a tokenizer.json for the DeepSeek-V4-Flash engine.

    python3 tools/dsv4_tokenizer.py <model_dir> --from <source_tokenizer.json>
    python3 tools/dsv4_tokenizer.py <model_dir> --check
    python3 tools/dsv4_tokenizer.py <model_dir> --ctest tests/test_tok_dsv4

WHY THIS TAKES A SOURCE FILE INSTEAD OF KNOWING THE ANSWER
----------------------------------------------------------
The published DeepSeek-V4-Flash checkpoint ships NO vocabulary. Its root has
config.json, generation_config.json and 46 weight shards; `encoding/` is the
chat-message layer (encode_messages / parse_message_from_completion_text) and
defines special-token STRINGS but no ids and no vocab; `inference/generate.py`
calls `AutoTokenizer.from_pretrained(ckpt_path)`, which cannot succeed against
that repository as published.

Circumstantially the vocabulary looks like DeepSeek-V3's: vocab_size 129280,
bos_token_id 0, eos_token_id 1, and the same `<｜begin▁of▁sentence｜>` /
`<｜User｜>` / `<｜Assistant｜>` family. That is a strong hint and NOT a
verified fact, so this tool does not hardcode it. You point `--from` at a
vocabulary; the tool checks it against the model's own config.json and refuses
by name when something disagrees. If DeepSeek later ships the real file, the
same command works unchanged.

WHAT IS CHECKED
---------------
  * vocab_size, bos_token_id and eos_token_id against config.json
  * every special token `encoding/README.md` names is present AND in range
  * the pre-tokenizer Split pattern maps to a family tok.h actually implements
    (cl100k / o200k / kimi / dsv4) -- REFUSING otherwise rather than letting tok.h's
    `strstr` sniffing silently pick the wrong one. That sniffing (tok.h:181)
    keys on `\\p{Lu}` and `\\p{Han}` alone, so an unknown third pattern does not
    fail loudly on its own; it just tokenizes differently, which is the kind of
    bug that shows up as mildly degraded output a long way downstream.
"""
import argparse, json, os, subprocess, sys

# From encoding/README.md. The chat layer emits these verbatim, so a vocabulary
# that lacks any of them cannot round-trip a conversation.
REQUIRED_SPECIALS = [
    "<｜begin▁of▁sentence｜>", "<｜end▁of▁sentence｜>",
    "<｜User｜>", "<｜Assistant｜>",
]
# Named by encoding/README.md but only used by specific task modes; absence is
# reported, not fatal.
OPTIONAL_SPECIALS = [
    "<think>", "</think>", "<｜latest_reminder｜>", "｜DSML｜",
    "<｜action｜>", "<｜title｜>", "<｜query｜>", "<｜authority｜>",
    "<｜domain｜>", "<｜extracted_url｜>", "<｜read_url｜>",
]

def die(msg):
    print(f"dsv4_tokenizer: {msg}", file=sys.stderr)
    sys.exit(1)

def load_json(path, what):
    if not os.path.exists(path):
        die(f"{what} not found: {path}")
    with open(path, encoding="utf-8") as f:
        return json.load(f)

def pretok_patterns(tk):
    """Every Split regex in the pre_tokenizer, in order."""
    pt = tk.get("pre_tokenizer")
    if pt is None:
        return []
    seq = pt.get("pretokenizers", [pt]) if pt.get("type") == "Sequence" else [pt]
    out = []
    for p in seq:
        if p.get("type") == "Split":
            pat = p.get("pattern", {})
            r = pat.get("Regex") if isinstance(pat, dict) else None
            if r:
                out.append(r)
    return out

def family_of(patterns):
    """Mirror tok.h's dispatch EXACTLY (tok.h:181-190) and say what it will pick.

    tok.h decides by substring: `\\p{Han}` -> kimi, else `\\p{Lu}` -> o200k,
    else cl100k. Reproducing that here rather than inventing a better rule is
    the point: this tool must predict what the engine will actually do."""
    joined = "".join(patterns)
    # dsv4 is checked FIRST because tok.h checks it last but its markers are
    # mutually exclusive with the others; keeping the two in agreement is what
    # this function is for.
    if "\\p{P}\\p{S}" in joined or "\u4e00-\u9fa5" in joined:
        return "dsv4"
    if "\\p{Han}" in joined:
        return "kimi"
    if "\\p{Lu}" in joined:
        return "o200k"
    return "cl100k"

# Fragments that must ALL be present for a pattern to really be the family
# tok.h's substring dispatch would assign it to.
#
# `all`, not `any` -- and this is not a stylistic preference. An earlier
# revision used `any` over generic fragments like "\\p{N}", which every
# byte-level BPE pattern in existence contains; the real DeepSeek-V4 pattern
# sailed through it on the first try and would have been shipped as cl100k.
# The contraction group is the discriminator worth leaning on: ALL THREE
# families tok.h implements (cl100k, o200k, kimi) carry `(?i:'s|'t|...)`, and
# a pattern without it is not any of them no matter what else it contains.
KNOWN_FRAGMENTS = {
    "cl100k": ["'s|'t|'re|'ve|'m|'ll|'d", "\\p{N}{1,3}"],
    "o200k":  ["'s|'t|'re|'ve|'m|'ll|'d", "\\p{Lu}", "\\p{Ll}"],
    "kimi":   ["'s|'t|'re|'ve|'m|'ll|'d", "\\p{Han}"],
    # DeepSeek-V4 is the one family with NO contraction group: "don't" splits
    # via the punctuation-then-ASCII-letters branch instead. Its markers are the
    # \p{P}\p{S} class and the literal CJK ranges.
    "dsv4":   ["\\p{P}\\p{S}", "\\p{L}\\p{M}"],
}

def analyse(tk, cfg, strict_family):
    errs, warns, notes = [], [], []
    model = tk.get("model", {})
    if model.get("type") != "BPE":
        errs.append(f"model.type is {model.get('type')!r}, tok.h only implements BPE")
    vocab = model.get("vocab") or {}
    merges = model.get("merges") or []
    added = tk.get("added_tokens") or []

    ids = set(vocab.values())
    for a in added:
        ids.add(a["id"])
    n_ids = (max(ids) + 1) if ids else 0

    want_v = cfg.get("vocab_size")
    if want_v is not None and n_ids != want_v:
        errs.append(f"vocabulary spans {n_ids} ids but config.json says vocab_size={want_v}")

    by_content = dict(vocab)
    for a in added:
        by_content[a["content"]] = a["id"]

    for name, key in (("bos_token_id", "<｜begin▁of▁sentence｜>"),
                      ("eos_token_id", "<｜end▁of▁sentence｜>")):
        want = cfg.get(name)
        got = by_content.get(key)
        if want is None:
            continue
        if got is None:
            errs.append(f"{key!r} absent, but config.json sets {name}={want}")
        elif got != want:
            errs.append(f"{key!r} has id {got}, but config.json sets {name}={want}")

    for s in REQUIRED_SPECIALS:
        if s not in by_content:
            errs.append(f"required special token missing: {s!r}")
        elif want_v is not None and by_content[s] >= want_v:
            errs.append(f"{s!r} has id {by_content[s]} >= vocab_size {want_v}: "
                        "the model has no embedding row for it")
    missing_opt = [s for s in OPTIONAL_SPECIALS if s not in by_content]
    if missing_opt:
        warns.append("optional special tokens absent (some chat modes will not "
                     "round-trip): " + ", ".join(repr(s) for s in missing_opt))

    pats = pretok_patterns(tk)
    fam = family_of(pats)
    joined = "".join(pats)
    recognised = all(f in joined for f in KNOWN_FRAGMENTS[fam])
    notes.append(f"pre-tokenizer family tok.h will select: {fam}")
    if not pats:
        errs.append("no Split regex in pre_tokenizer: tok.h's pretok_chunk needs one")
    elif not recognised:
        msg = (f"the Split pattern does not look like tok.h's {fam} family. "
               f"tok.h dispatches on substrings alone, so it WILL run the {fam} "
               f"rules over a pattern that is not {fam} -- silently different "
               f"tokenization, not an error. Pattern: {joined[:160]!r}")
        (errs if strict_family else warns).append(msg)

    if not merges:
        notes.append("model.merges is empty: tok.h will use rank-BPE mode "
                     "(merge the adjacent pair with the lowest vocab id), which "
                     "is exact for tiktoken-derived vocabularies")
    return errs, warns, notes, n_ids, len(added), len(merges)

def cmd_build(args, cfg):
    src = load_json(args.source, "source tokenizer")
    errs, warns, notes, n_ids, n_added, n_merges = analyse(src, cfg, not args.allow_unknown_family)
    for n in notes:
        print(f"  note:    {n}")
    for w in warns:
        print(f"  WARNING: {w}")
    for e in errs:
        print(f"  ERROR:   {e}")
    print(f"  vocabulary: {n_ids} ids, {n_added} added tokens, {n_merges} merges")
    if errs:
        die(f"{len(errs)} problem(s); refusing to write a tokenizer.json that "
            "would tokenize differently from the model that was trained")
    out = os.path.join(args.model_dir, "tokenizer.json")
    if os.path.exists(out) and not args.force:
        die(f"{out} already exists (use --force to overwrite)")
    with open(out, "w", encoding="utf-8") as f:
        json.dump(src, f, ensure_ascii=False)
    print(f"  wrote {out}")
    return 0

def cmd_check(args, cfg):
    tk = load_json(os.path.join(args.model_dir, "tokenizer.json"), "tokenizer.json")
    errs, warns, notes, n_ids, n_added, n_merges = analyse(tk, cfg, not args.allow_unknown_family)
    for n in notes:
        print(f"  note:    {n}")
    for w in warns:
        print(f"  WARNING: {w}")
    for e in errs:
        print(f"  ERROR:   {e}")
    print(f"  vocabulary: {n_ids} ids, {n_added} added tokens, {n_merges} merges")
    if errs:
        die(f"{len(errs)} problem(s)")
    print("  tokenizer.json: ok")
    return 0

def cmd_ctest(args, cfg):
    """Cross-check tok.h against HF `tokenizers` on the model's own vocabulary.

    The same discipline as tools/k3_tokenizer.py --ctest: the C tokenizer is
    only trustworthy if a reference implementation agrees on real text, and the
    corpus deliberately includes the shapes that break naive pre-tokenizers."""
    try:
        from tokenizers import Tokenizer
    except ImportError:
        die("--ctest needs the `tokenizers` package (pip install tokenizers)")
    path = os.path.join(args.model_dir, "tokenizer.json")
    ref = Tokenizer.from_file(path)
    cases = [
        "Hello, world!",
        "  leading and   repeated   spaces",
        "don't can't it's we've I'll they'd",
        "1234567890 42 007",
        "line one\nline two\r\nline three\n\n\n",
        "snake_case CamelCase SCREAMING_CASE kebab-case",
        "def main():\n    return {'a': 1}\n",
        "中文汉字测试",
        "日本語のカタカナとひらがな",
        "한국어 테스트",
        "emoji 🐦 🚀 and combining é ẹ̀",
        "mixed中英文abc123混排",
        "<｜User｜>hi<｜Assistant｜>hello<｜end▁of▁sentence｜>",
        "https://example.com/a/b?c=d&e=f",
        "\t\ttabs\tand\ttabs",
    ]
    corpus = os.path.join(args.model_dir, "dsv4_tok_cases.txt")
    with open(corpus, "w", encoding="utf-8") as f:
        for t in cases:
            ids = ref.encode(t).ids
            f.write(t.replace("\\", "\\\\").replace("\n", "\\n").replace("\r", "\\r")
                    + "\t" + ",".join(map(str, ids)) + "\n")
    print(f"  wrote {corpus} ({len(cases)} cases from HF tokenizers)")
    r = subprocess.run([args.ctest, path, corpus], capture_output=True, text=True)
    sys.stdout.write(r.stdout)
    sys.stderr.write(r.stderr)
    if r.returncode != 0:
        die("tok.h disagrees with the reference tokenizer -- see the cases above")
    print("  ctest: tok.h matches HF tokenizers on all cases")
    return 0

def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("model_dir")
    ap.add_argument("--from", dest="source", help="source tokenizer.json to validate and install")
    ap.add_argument("--check", action="store_true", help="validate the tokenizer.json already in model_dir")
    ap.add_argument("--ctest", metavar="BINARY", help="cross-check tok.h against HF tokenizers")
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--allow-unknown-family", action="store_true",
                    help="downgrade the unknown-pre-tokenizer refusal to a warning "
                         "(you are asserting tok.h's family rules fit this pattern)")
    a = ap.parse_args()
    cfg = load_json(os.path.join(a.model_dir, "config.json"), "config.json")
    if a.source:
        return cmd_build(a, cfg)
    if a.ctest:
        return cmd_ctest(a, cfg)
    if a.check:
        return cmd_check(a, cfg)
    ap.error("one of --from, --check or --ctest is required")

if __name__ == "__main__":
    sys.exit(main())
