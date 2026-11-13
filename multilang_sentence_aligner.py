#!/usr/bin/env python3
"""
Многоязычное выравнивание без опорного языка.

1. Каждая пара языков выравнивается напрямую (bl_sentence_aligner), для N языков
   это N*(N-1)/2 запусков со своими рабочими папками.
2. Каждая найденная пара находится в тексте обеих книг (позиции символов без
   пробелов) и связывает отрезок одного языка с отрезком другого.
3. Связи объединяются по всем языкам: предложения, связанные хотя бы цепочкой,
   попадают в одну строку. Строки, которые пересекаются или идут в разном
   порядке в разных языках, сливаются, поэтому порядок во всех колонках один.

Использование:
  python multilang_sentence_aligner.py ru.json en.json fr.json es.json -o out.json
Код языка берётся из meta.language. Порядок книг влияет только на оглавление:
главы и их заголовки берутся из первой книги, на сами строки это не влияет.
"""

import sys
import re
import json
import bisect
import argparse
import subprocess
from collections import defaultdict
from itertools import combinations
from pathlib import Path

for _stream in (sys.stdout, sys.stderr):
    try:
        if _stream.encoding and "utf-8" not in _stream.encoding.lower():
            _stream.reconfigure(encoding="utf-8", errors="replace")
    except (AttributeError, ValueError):
        pass

HERE = Path(__file__).resolve().parent


def squash(s):
    return re.sub(r"\s+", "", s)


def load_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


class Axis:
    """Текст книги без пробелов (для поиска пар) и связь с исходным текстом."""

    def __init__(self, book):
        sq, idx, orig, self.spans = [], [], [], {}
        for ch in book["chapters"]:
            start = len(sq)
            for para in ch.get("paragraphs", []):
                for c in str(para).strip():
                    if not c.isspace():
                        sq.append(c)
                        idx.append(len(orig))
                    orig.append(c)
                orig.append(" ")
            self.spans[ch["chapter_no"]] = (start, len(sq))
        self.text, self.idx, self.orig = "".join(sq), idx, "".join(orig)

    def slice(self, s, e):
        return self.orig[self.idx[s]:self.idx[e - 1] + 1].strip()


def run_pair_aligner(a_path, b_path, work_dir, out_path, force):
    work_dir.mkdir(parents=True, exist_ok=True)
    if out_path.exists() and not force:
        print(f"  [кэш] {out_path.name}")
        return
    cmd = [sys.executable, str(HERE / "bl_sentence_aligner.py"),
           str(a_path), str(b_path), "-o", str(out_path), "--work-dir", str(work_dir)]
    if force:
        cmd.append("--force")
    print(f"  {out_path.stem}")
    res = subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8")
    if res.returncode != 0:
        raise RuntimeError(f"aligner failed for {out_path.stem}:\n{res.stdout[-1500:]}\n{res.stderr[-1500:]}")


def locate_beads(work_dir, a, b, axes):
    """Пары одного запуска -> связи (язык, начало, конец) <-> (язык, начало, конец) и описания глав."""
    beads, units = [], []
    for f in sorted(work_dir.glob("chapter_*.json")):
        rec = json.loads(f.read_text(encoding="utf-8"))
        units.append({"source_chapters": rec["source_chapters"], "title_tgt": rec.get("title_tgt", "")})
        lo = {a: min(axes[a].spans[c][0] for c in rec["source_chapters"]),
              b: min(axes[b].spans[c][0] for c in rec["target_chapters"])}
        hi = {a: max(axes[a].spans[c][1] for c in rec["source_chapters"]),
              b: max(axes[b].spans[c][1] for c in rec["target_chapters"])}
        cur = dict(lo)
        for p in rec["pairs"]:
            ka, kb = squash(p["src"]), squash(p["tgt"])
            if not ka or not kb:
                continue
            sa = axes[a].text.find(ka, cur[a], hi[a])
            sb = axes[b].text.find(kb, cur[b], hi[b])
            if sa < 0 or sb < 0:
                raise RuntimeError(f"{f}: пара не найдена в тексте: {p['src'][:50]!r} / {p['tgt'][:50]!r}")
            cur[a], cur[b] = sa + len(ka), sb + len(kb)
            beads.append(((a, sa, cur[a]), (b, sb, cur[b])))
    return beads, units


def group_beads(beads, lengths):
    """Связи -> упорядоченные группы {язык: (начало, конец)} и число вынужденных слияний."""
    # Разные запуски режут один язык по-разному: перекрывающиеся отрезки — одна единица.
    ivs = defaultdict(set)
    for x, y in beads:
        ivs[x[0]].add(x)
        ivs[y[0]].add(y)
    unit_of, uid = {}, 0
    for lang, items in ivs.items():
        end = cur = None
        for iv in sorted(items, key=lambda t: (t[1], t[2])):
            if cur is None or iv[1] >= end:
                uid += 1
                cur, end = uid, iv[2]
            else:
                end = max(end, iv[2])
            unit_of[iv] = cur

    parent = list(range(uid + 1))

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    for x, y in beads:
        parent[find(unit_of[x])] = find(unit_of[y])

    comps = defaultdict(dict)
    for iv, u in unit_of.items():
        lang, s, e = iv
        ext = comps[find(u)]
        old = ext.get(lang)
        ext[lang] = (s, e) if old is None else (min(old[0], s), max(old[1], e))

    def key(ext):
        return sum(s / lengths[l] for l, (s, _) in ext.items()) / len(ext)

    stack, forced = [], 0
    for ext in sorted(comps.values(), key=key):
        cur = dict(ext)
        while True:
            low, pending = None, set(cur)
            for k in range(len(stack) - 1, -1, -1):
                for lang in list(pending):
                    if lang in stack[k]:
                        if stack[k][lang][1] > cur[lang][0]:
                            low = k
                        else:
                            pending.discard(lang)
                if not pending:
                    break
            if low is None:
                break
            for g in stack[low:]:
                for lang, (s, e) in g.items():
                    o = cur.get(lang)
                    cur[lang] = (s, e) if o is None else (min(o[0], s), max(o[1], e))
            del stack[low:]
            forced += 1
        stack.append(cur)
    return stack, forced


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("books", nargs="+", help="JSON книг (минимум две)")
    ap.add_argument("-o", "--output", default="multilang_aligned_sentences.json")
    ap.add_argument("--work-root", default="", help="папка для промежуточных файлов (по умолчанию рядом с -o)")
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--drop", action="append", default=[], metavar="LANG:N,N",
                    help="выбросить из книги служебные главы (chapter_no), например en:1,30,31")
    args = ap.parse_args()
    if len(args.books) < 2:
        raise SystemExit("нужно минимум две книги")

    drop = {}
    for item in args.drop:
        lang, _, nums = item.partition(":")
        drop.setdefault(lang, set()).update(int(n) for n in nums.split(",") if n.strip())

    out = Path(args.output)
    root = Path(args.work_root) if args.work_root else out.parent

    paths, books = {}, {}
    for p in args.books:
        book = load_json(p)
        lang = book.get("meta", {}).get("language")
        if not lang:
            raise SystemExit(f"{p}: в meta нет language")
        if lang in drop:
            book["chapters"] = [c for c in book["chapters"] if c["chapter_no"] not in drop[lang]]
            p = root / f"clean_{lang}.json"
            p.write_text(json.dumps(book, ensure_ascii=False), encoding="utf-8")
        paths[lang], books[lang] = Path(p), book
    langs = list(books)
    ref = langs[0]
    axes = {l: Axis(b) for l, b in books.items()}

    print(f"[1/3] Выравнивание пар ({len(langs) * (len(langs) - 1) // 2})")
    for a, b in combinations(langs, 2):
        run_pair_aligner(paths[a], paths[b], root / f"work_{a}_{b}", root / f"pair_{a}_{b}.json", args.force)

    print("[2/3] Привязка пар к тексту книг")
    beads, ref_units = [], {}
    for a, b in combinations(langs, 2):
        found, units = locate_beads(root / f"work_{a}_{b}", a, b, axes)
        beads += found
        if a == ref:
            ref_units[b] = units
        cov = sum(x[2] - x[1] for x, _ in found) / max(1, len(axes[a].text)) * 100
        print(f"  {a}-{b}: {len(found)} пар, покрыто {cov:.1f}% текста {a}")

    print("[3/3] Слияние связей")
    groups, forced = group_beads(beads, {l: len(ax.text) for l, ax in axes.items()})
    print(f"  строк {len(groups)}, вынужденных слияний (пересечения, разный порядок): {forced}")

    ref_chapters = sorted((s, e, no) for no, (s, e) in axes[ref].spans.items())
    starts = [c[0] for c in ref_chapters]
    chapters, by_no = [], {}
    last_ci = 0
    for g in groups:
        if ref in g:
            last_ci = max(0, bisect.bisect_right(starts, g[ref][0]) - 1)
        pair = {l: axes[l].slice(*g[l]) for l in langs if l in g}
        by_no.setdefault(ref_chapters[last_ci][2], []).append(pair)

    for ch in books[ref]["chapters"]:
        rows = by_no.get(ch["chapter_no"])
        if not rows:
            continue
        titles = {ref: str(ch.get("title", "")).strip()}
        for lang, units in ref_units.items():
            for u in units:
                # Заголовок другого языка берём, только если он похож на заголовок:
                # парсер иногда принимает за него начало абзаца.
                if (u["source_chapters"] == [ch["chapter_no"]]
                        and 0 < len(u["title_tgt"]) <= 30 and " · " not in u["title_tgt"]):
                    titles[lang] = u["title_tgt"]
        chapters.append({"chapter_no": len(chapters) + 1, "title": titles[ref],
                         "titles": titles, "multilingual_pairs": rows})

    total = sum(len(c["multilingual_pairs"]) for c in chapters)
    complete = sum(1 for c in chapters for p in c["multilingual_pairs"] if len(p) == len(langs))
    print(f"  глав {len(chapters)}, строк {total}, из них со всеми {len(langs)} языками: {complete}")
    out.write_text(json.dumps({
        "version": "multilingual-aligned",
        "languages": langs,
        "n_chapters": len(chapters),
        "chapters": chapters,
    }, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[OK] {out}")


if __name__ == "__main__":
    main()
