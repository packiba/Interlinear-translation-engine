#!/usr/bin/env python3
"""
Текстовый файл (TXT) -> JSON книги в той же схеме, что даёт bl_book_parser.py (для PDF).

  python bl_text_parser.py book.txt -o book.json --language ru
  python bl_text_parser.py ru.txt -o ru.json --language ru --from-bilingual

Режимы:
  обычный            абзацы разделены пустой строкой (или каждая строка — абзац).
  --from-bilingual   файл — параллельный текст, где абзацы разных языков чередуются
                     (английский, русский, английский, русский …), иногда прямо внутри
                     одной строки. Из него берётся только язык --language: каждый
                     «кусок» текста в его письменности — один абзац. Короткие вкрапления
                     другой письменности внутри предложения (термины, названия) остаются.

Главы начинаются с абзаца-заголовка вида «Глава 12», «ГЛАВА 12», «Chapter 12», «Пролог»,
«Эпилог» (--chapter-re заменяет шаблон). Текст до первого заголовка — глава «Начало».
Сноски переводчика в конце файла (строки «1», затем текст сноски, …) отбрасываются.
"""

import sys
import re
import json
import argparse
from pathlib import Path

for _stream in (sys.stdout, sys.stderr):
    try:
        if _stream.encoding and "utf-8" not in _stream.encoding.lower():
            _stream.reconfigure(encoding="utf-8", errors="replace")
    except (AttributeError, ValueError):
        pass

DEFAULT_CHAPTER_RE = (r"^\s*(?:(?:глава|часть|chapter|part|capítulo|chapitre|kapitel)\s+[\wIVXLC]+\.?"
                      r"|пролог|эпилог|prologue|epilogue|prólogo|epílogo|prolog|epilog)\s*$")
TERMINAL = ".!?…»”\")"
OPENERS = re.compile(r"\s[\"“«„'(\[—–-]+\s*$")      # пробел + открывающие знаки в конце куска
# заголовок на чужом языке («CHAPTER 45») — всегда граница, а не вкрапление в предложение
HEADING_O = re.compile(r"(?:^|\s)(?:chapter\s*\d+|prologue|epilogue|fact:?)\s*$", re.I)
CHAPTER_PAT = re.compile(DEFAULT_CHAPTER_RE, re.I)


def read_text(path):
    raw = Path(path).read_bytes()
    for enc in ("utf-8-sig", "cp1251"):
        try:
            text = raw.decode(enc)
            break
        except UnicodeDecodeError:
            continue
    else:
        text = raw.decode("utf-8", errors="replace")
    return text.replace("\r\n", "\n").replace("\r", "\n")


def script_of(ch, target):
    """'T' — буква нужного письма, 'O' — буква другого письма, None — не буква."""
    cyr = "Ѐ" <= ch <= "ӿ"
    lat = ch.isascii() and ch.isalpha()
    if not (cyr or lat):
        return None
    want_cyr = target == "cyr"
    return "T" if (cyr == want_cyr) else "O"


def runs_of_script(text, target):
    """Куски текста по письменности: [(kind, text)], kind 'T' (нужный язык) или 'O' (чужой)."""
    runs, cur, buf = [], None, []
    for ch in text:
        k = script_of(ch, target)
        if k and cur and k != cur:
            piece = "".join(buf)
            # открывающая кавычка или тире перед первой буквой нового языка принадлежит ему
            m = OPENERS.search(piece)
            carry = ""
            if m:
                carry = m.group().lstrip()
                piece = piece[:m.start()] + " "
            runs.append((cur, piece))
            buf = list(carry)
        if k:
            cur = k
        buf.append(ch)
    runs.append((cur or "T", "".join(buf)))
    return runs


def clean(s):
    s = re.sub(r"(?<=[^\W\d_])\d{1,3}(?=[\s,.;:)!?—-]|$)", "", s)      # номера сносок: «Приорат1 Сиона»
    return re.sub(r"\s+", " ", s).strip()


def bilingual_paragraphs(text, language):
    """Абзацы нужного языка из параллельного текста (см. описание режима)."""
    target = "cyr" if language in {"ru", "bg", "uk", "sr", "mk", "be"} else "lat"
    runs = runs_of_script(text, target)
    paras, i = [], 0
    while i < len(runs):
        kind, chunk = runs[i]
        if kind != "T":
            i += 1
            continue
        cur = chunk
        # короткая вставка чужой письменности посреди предложения — часть этого же абзаца
        while i + 2 < len(runs) and runs[i + 1][0] == "O" and runs[i + 2][0] == "T":
            foreign = runs[i + 1][1]
            heading = HEADING_O.search(foreign)
            if heading:                  # «… "S-O-F-I-A". CHAPTER 78»: до заголовка — вкрапление, дальше граница
                before = foreign[:heading.start()]
                if before.strip() and len(before.split()) <= 3 and cur.rstrip()[-1:] not in TERMINAL:
                    cur += before
                break
            if (CHAPTER_PAT.match(cur.strip()) or len(foreign.split()) > 3
                    or cur.rstrip()[-1:] in TERMINAL):
                break
            cur += foreign + runs[i + 2][1]
            i += 2
        paras.append(cur)
        i += 1
    return [p for p in (clean(p) for p in paras) if sum(c.isalpha() for c in p) >= 2]


def plain_paragraphs(text):
    blocks = [b for b in re.split(r"\n\s*\n", text) if b.strip()]
    if len(blocks) < 5:                               # нет пустых строк: абзац = строка
        blocks = [l for l in text.splitlines() if l.strip()]
    return [re.sub(r"\s+", " ", b).strip() for b in blocks]


def cut_notes(text):
    """Отбрасывает список сносок в конце: строки «1», текст, пустая строка, «2», текст …"""
    last = None
    for m in re.finditer(r"\n[ \t]*1[ \t]*\n(?=[^\n]+\n[ \t]*\n[ \t]*2[ \t]*\n)", text):
        last = m
    if last and last.start() > 0.7 * len(text):
        return text[:last.start()], len(text) - last.start()
    return text, 0


def split_chapters(paragraphs, chapter_re):
    pat = re.compile(chapter_re, re.I)
    chapters, cur = [], {"title": "Начало", "paragraphs": []}
    for p in paragraphs:
        if len(p) <= 40 and pat.match(p):
            if cur["paragraphs"] or chapters:
                chapters.append(cur)
            cur = {"title": p.strip(" ."), "paragraphs": []}
        else:
            cur["paragraphs"].append(p)
    chapters.append(cur)
    return [c for c in chapters if c["paragraphs"]]


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("txt")
    ap.add_argument("-o", "--output", default="")
    ap.add_argument("--language", required=True, help="язык текста (ru, en, bg …)")
    ap.add_argument("--title", default="")
    ap.add_argument("--author", default="")
    ap.add_argument("--from-bilingual", action="store_true", help="взять только --language из параллельного текста")
    ap.add_argument("--chapter-re", default=DEFAULT_CHAPTER_RE)
    args = ap.parse_args()

    text = read_text(args.txt)
    text, notes_chars = cut_notes(text)
    paragraphs = bilingual_paragraphs(text, args.language) if args.from_bilingual else plain_paragraphs(text)
    chapters = split_chapters(paragraphs, args.chapter_re)

    out = {
        "meta": {"title": args.title or Path(args.txt).stem, "author": args.author, "language": args.language,
                 "source_file": Path(args.txt).name, "pages": 0},
        "structure": [],
        "chapters": [{
            "chapter_no": i, "title": c["title"], "page_start": 0, "page_end": 0, "toc_level": 1,
            "source": "text", "parents": [], "dateline": "", "epigraph": "", "epigraph_source": "",
            "paragraphs": c["paragraphs"], "text": "\n\n".join(c["paragraphs"]),
        } for i, c in enumerate(chapters, 1)],
        "diagnostics": {"strategy": "text", "bilingual": args.from_bilingual, "notes_chars_cut": notes_chars},
    }
    dest = Path(args.output) if args.output else Path(args.txt).with_suffix(".json")
    dest.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    total = sum(len(p) for c in chapters for p in c["paragraphs"])
    print(f"{dest}: глав {len(chapters)}, абзацев {sum(len(c['paragraphs']) for c in chapters)}, "
          f"символов {total}, сноски отброшены: {notes_chars} симв.")
    for c in chapters[:3] + chapters[-2:]:
        print(f"  {c['title']!r}: {len(c['paragraphs'])} абз. | {c['paragraphs'][0][:70]!r}")


if __name__ == "__main__":
    main()
