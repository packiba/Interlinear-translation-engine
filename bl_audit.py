#!/usr/bin/env python3
"""Аудит готовых final_aligned_book*.json (любая пара языков, без GPU/модели).

    python bl_audit.py final_aligned_book-ru-bg.json [другие ...] [--chapters]

Показывает то, что видно по самому тексту пар: «мега-пары» (приклеенные блоки
без пары), резкие расхождения длин, пары, начинающиеся со строчной буквы
(потерянное начало предложения), главы с аномальным отношением длин."""
import json, re, sys, statistics as st

LEAD = " –—―-«„\"“'([.…¿¡"


def lr(p):
    a, b = len(p["src"]), len(p["tgt"])
    return min(a, b) / max(1, max(a, b))


def lower_start(s):
    s = s.lstrip(LEAD)
    return bool(s) and s[0].isalpha() and s[0].islower()


def audit(path, per_chapter=False):
    d = json.load(open(path, encoding="utf-8"))
    chs = d["chapters"]
    P = [p for c in chs for p in c["pairs"]]
    big = [p for p in P if max(len(p["src"]), len(p["tgt"])) > 1500]
    print(f"\n=== {path}\n  глав {len(chs)}, пар {len(P)}, "
          f"медиана len_ratio {st.median(lr(p) for p in P):.2f}")
    print(f"  пар длиннее 1500 символов (признак приклеенных блоков): {len(big)}"
          + (f", максимум {max(max(len(p['src']), len(p['tgt'])) for p in big)} симв." if big else ""))
    print(f"  пар с len_ratio < 0.25: {sum(lr(p) < .25 for p in P)} "
          f"({sum(lr(p) < .25 for p in P) / len(P):.1%})")
    ls, lt = sum(lower_start(p['src']) for p in P), sum(lower_start(p['tgt']) for p in P)
    print(f"  пар, начинающихся со строчной: src {ls}, tgt {lt}"
          + ("   <- похоже на обрезанные начала предложений" if max(ls, lt) > 0.01 * len(P) else ""))
    tot = sum(len(p['tgt']) for p in P) / max(1, sum(len(p['src']) for p in P))
    for i, c in enumerate(chs, 1):
        cs = sum(len(p['src']) for p in c['pairs']); ct = sum(len(p['tgt']) for p in c['pairs'])
        r = ct / max(1, cs)
        flag = not 0.75 * tot <= r <= 1.25 * tot
        bad = sum(lr(p) < .25 for p in c['pairs'])
        if per_chapter or flag or bad > 0.08 * len(c['pairs']):
            print(f"  {'⚠' if flag or bad > 0.08 * len(c['pairs']) else ' '} гл.{i:>2} "
                  f"{c['title_src'][:28]!r} ↔ {c['title_tgt'][:28]!r}: пар {len(c['pairs'])}, "
                  f"tgt/src={r:.2f}, len_ratio<.25: {bad}")


if __name__ == "__main__":
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    if not args:
        sys.exit(__doc__)
    for f in args:
        audit(f, "--chapters" in sys.argv)
