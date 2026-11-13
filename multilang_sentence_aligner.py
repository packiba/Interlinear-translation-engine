#!/usr/bin/env python3
"""
Многоязычное выравнивание: русский текст — якорь.

1. Для каждого языка запускается bl_sentence_aligner (src=RU, tgt=язык) со своей
   рабочей папкой; он сам сопоставляет главы и выравнивает предложения.
2. Каждая пара находится в русском тексте книги (глобальные позиции символов).
3. Строкой общей таблицы становится отрезок русского текста между «разрезами»,
   которые не пересекает ни одна пара ни одного языка. Так во всех колонках
   строка — это один и тот же фрагмент книги.

Использование:
  python multilang_sentence_aligner.py ru.json en.json fr.json es.json -o out.json
Первый файл — якорный язык; код языка берётся из meta.language файла.
"""

import sys
import re
import json
import argparse
import subprocess
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


def run_pair_aligner(ru_path, tgt_path, work_dir, out_path, force):
    work_dir.mkdir(parents=True, exist_ok=True)
    if out_path.exists() and not force:
        print(f"  [кэш] {out_path.name}")
        return
    cmd = [sys.executable, str(HERE / "bl_sentence_aligner.py"),
           str(ru_path), str(tgt_path), "-o", str(out_path), "--work-dir", str(work_dir)]
    if force:
        cmd.append("--force")
    print("  " + " ".join(cmd[2:]))
    res = subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8")
    if res.returncode != 0:
        raise RuntimeError(f"aligner failed for {tgt_path}:\n{res.stdout[-1500:]}\n{res.stderr[-1500:]}")


def ru_coordinates(ru_book):
    """-> ({chapter_no: (start, end)}, весь русский текст без пробелов)."""
    spans, text, pos = {}, [], 0
    for ch in ru_book["chapters"]:
        t = squash("".join(str(p) for p in ch.get("paragraphs", [])))
        spans[ch["chapter_no"]] = (pos, pos + len(t))
        text.append(t)
        pos += len(t)
    return spans, "".join(text)


def locate_segments(work_dir, spans, ru_text):
    """Читает отчёты глав -> сегменты (start, end, ru_text, tgt_text) и описание единиц."""
    segs, units = [], []
    for f in sorted(work_dir.glob("chapter_*.json")):
        rec = json.loads(f.read_text(encoding="utf-8"))
        src_chs = rec["source_chapters"]
        lo = min(spans[c][0] for c in src_chs)
        hi = max(spans[c][1] for c in src_chs)
        cursor = lo
        units.append({"source_chapters": src_chs, "title_tgt": rec.get("title_tgt", "")})
        for p in rec["pairs"]:
            key = squash(p["src"])
            if not key:
                continue
            at = ru_text.find(key, cursor, hi)
            if at < 0:
                raise RuntimeError(f"{f.name}: русская пара не найдена в тексте: {p['src'][:60]!r}")
            segs.append((at, at + len(key), p["src"], p["tgt"]))
            cursor = at + len(key)
    segs.sort(key=lambda s: (s[0], s[1]))
    return segs, units


def merge(spans, per_lang):
    """per_lang: {lang: [(start, end, ru_text, tgt_text)]} -> строки с общими границами."""
    points = set()
    for a, b in spans.values():
        points.update((a, b))
    for segs in per_lang.values():
        for s, e, *_ in segs:
            points.update((s, e))
    points = sorted(points)

    # Разрез недопустим, если его пересекает пара любого языка.
    banned = set()
    for segs in per_lang.values():
        for s, e, *_ in segs:
            banned.update(p for p in points if s < p < e)
    cuts = [p for p in points if p not in banned]

    rows = []
    for a, b in zip(cuts, cuts[1:]):
        row, ru_parts = {}, []
        for lang, segs in per_lang.items():
            inside = [sg for sg in segs if sg[0] >= a and sg[1] <= b]
            if inside:
                row[lang] = " ".join(sg[3] for sg in inside).strip()
                if not ru_parts:
                    ru_parts = [sg[2] for sg in inside]
        if ru_parts:
            rows.append((a, " ".join(ru_parts).strip(), row))
    return rows


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("books", nargs="+", help="JSON книг; первый — русский (якорь)")
    ap.add_argument("-o", "--output", default="multilang_aligned_sentences.json")
    ap.add_argument("--work-root", default="", help="папка для промежуточных файлов (по умолчанию рядом с -o)")
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--drop", action="append", default=[], metavar="LANG:N,N",
                    help="выбросить из книги служебные главы (chapter_no), например en:1,30,31")
    args = ap.parse_args()

    drop = {}
    for item in args.drop:
        lang, _, nums = item.partition(":")
        drop.setdefault(lang, set()).update(int(n) for n in nums.split(",") if n.strip())

    out = Path(args.output)
    root = Path(args.work_root) if args.work_root else out.parent

    def prepare(path):
        """Возвращает (язык, путь к книге без выброшенных глав)."""
        book = load_json(path)
        lang = book.get("meta", {}).get("language")
        if not lang:
            raise SystemExit(f"{path}: в meta нет language")
        if lang in drop:
            book["chapters"] = [c for c in book["chapters"] if c["chapter_no"] not in drop[lang]]
            path = root / f"clean_{lang}.json"
            path.write_text(json.dumps(book, ensure_ascii=False), encoding="utf-8")
        return lang, Path(path), book

    ru_lang, ru_path, ru_book = prepare(args.books[0])
    others = [prepare(p)[:2] for p in args.books[1:]]
    print(f"[1/3] Выравнивание пар {ru_lang}-X")
    for lang, path in others:
        run_pair_aligner(ru_path, path, root / f"work_{lang}", root / f"pair_{ru_lang}_{lang}.json", args.force)

    print("[2/3] Привязка пар к русскому тексту")
    spans, ru_text = ru_coordinates(ru_book)
    per_lang, unit_info = {}, {}
    for lang, _ in others:
        segs, units = locate_segments(root / f"work_{lang}", spans, ru_text)
        per_lang[lang], unit_info[lang] = segs, units
        covered = sum(e - s for s, e, *_ in segs)
        print(f"  {lang}: {len(segs)} пар, покрыто {covered / max(1, len(ru_text)) * 100:.1f}% русского текста")

    print("[3/3] Слияние")
    rows = merge(spans, per_lang)
    langs = [ru_lang] + [l for l, _ in others]

    chapters = []
    for ch in ru_book["chapters"]:
        a, b = spans[ch["chapter_no"]]
        mine = [(ru, row) for start, ru, row in rows if a <= start < b]
        if not mine:
            continue
        titles = {ru_lang: str(ch.get("title", "")).strip()}
        for lang, units in unit_info.items():
            for u in units:
                # Заголовок другого языка берём, только если он похож на заголовок:
                # парсер иногда принимает за него начало абзаца.
                if (u["source_chapters"] == [ch["chapter_no"]]
                        and 0 < len(u["title_tgt"]) <= 30 and " · " not in u["title_tgt"]):
                    titles[lang] = u["title_tgt"]
        pairs = [{ru_lang: ru, **row} for ru, row in mine]
        chapters.append({"chapter_no": len(chapters) + 1, "title": titles[ru_lang],
                         "titles": titles, "multilingual_pairs": pairs})

    total = sum(len(c["multilingual_pairs"]) for c in chapters)
    complete = sum(1 for c in chapters for p in c["multilingual_pairs"] if len(p) == len(langs))
    print(f"  глав {len(chapters)}, строк {total}, из них со всеми {len(langs)} языками: {complete}")
    out.write_text(json.dumps({
        "version": "2-multilingual-aligned",
        "languages": langs,
        "n_chapters": len(chapters),
        "chapters": chapters,
    }, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[OK] {out}")


if __name__ == "__main__":
    main()
