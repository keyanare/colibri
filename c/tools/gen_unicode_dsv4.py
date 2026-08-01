"""Genera tok_unicode_dsv4.h: le classi Unicode che il pre-tokenizer di
DeepSeek-V4-Flash usa e che tok_unicode.h / tok_unicode_o200k.h non coprono.

  - \\p{P}  punteggiatura (categoria Unicode che inizia per 'P')
  - \\p{S}  simboli      (categoria che inizia per 'S')  -> emesso come uni_Sym,
            NON uni_S: quel nome in tok_unicode.h e' gia' il WHITESPACE.
  - \\p{M}  marks        (categoria che inizia per 'M')

Same range-array + binary-search shape as tools/gen_unicode.py, so the two
headers compose. Run once:  python3 tools/gen_unicode_dsv4.py > tok_unicode_dsv4.h
"""
import sys, unicodedata

def cat(cp):
    try:
        return unicodedata.category(chr(cp))
    except ValueError:
        return "Cn"

def ranges(pred):
    out, lo = [], None
    for cp in range(0x110000):
        if 0xD800 <= cp <= 0xDFFF:          # surrogati: mai
            if lo is not None:
                out.append((lo, cp - 1)); lo = None
            continue
        if pred(cp):
            if lo is None:
                lo = cp
        else:
            if lo is not None:
                out.append((lo, cp - 1)); lo = None
    if lo is not None:
        out.append((lo, 0x10FFFF))
    return out

def emit(name, rs):
    print(f"static const uint32_t {name}[][2] = {{")
    for i in range(0, len(rs), 6):
        row = "".join(f"{{0x{a:X},0x{b:X}}}," for a, b in rs[i:i + 6])
        print("    " + row)
    print("};")
    print(f"static const int {name}_n = {len(rs)};")
    print()

def main():
    print("/* GENERATO da tools/gen_unicode_dsv4.py — non modificare a mano.")
    print(" * Classi Unicode del pre-tokenizer DeepSeek-V4-Flash: \\p{P}, \\p{S}, \\p{M}.")
    print(" * uni_Sym e' \\p{S} (simboli); uni_S in tok_unicode.h e' il whitespace. */")
    print("#ifndef TOK_UNICODE_DSV4_H")
    print("#define TOK_UNICODE_DSV4_H")
    print()
    emit("uni_P",   ranges(lambda c: cat(c).startswith("P")))
    emit("uni_Sym", ranges(lambda c: cat(c).startswith("S")))
    emit("uni_M",   ranges(lambda c: cat(c).startswith("M")))
    print("static inline int is_P(uint32_t c){ return uni_in(uni_P,uni_P_n,c); }")
    print("static inline int is_Sym(uint32_t c){ return uni_in(uni_Sym,uni_Sym_n,c); }")
    print("static inline int is_M(uint32_t c){ return uni_in(uni_M,uni_M_n,c); }")
    print()
    print("#endif")

if __name__ == "__main__":
    main()
