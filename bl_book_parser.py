#!/usr/bin/env python3
"""
bl_book_parser.py — Этап 1: PDF -> JSON книги (главы + абзацы) для bl_sentence_aligner.py.

Универсальный парсер художественных книг в PDF (calibre/FB2/EPUB-конвертации,
Prince, Word, pypdf-склейки). Сам выбирает стратегию разметки глав:

  1. Встроенное оглавление (TOC) — после чистки мусора (Annotation, титульные
     страницы, сноски в конце книги, «Допълнителна информация» и т.п.),
     выбора уровня глав (подразделы «1», «2» внутри «Глава N» вливаются в главу)
     и проверки, что TOC покрывает текст книги.
  2. Заголовки по вёрстке — крупный/жирный/центрированный короткий текст,
     ключевые слова (Глава/Част/Chapter/Capítulo/Том/Book…), номера, римские цифры.
  3. Разрывы секций — глава начинается с новой страницы с увеличенным верхним
     отступом (книги без заголовков глав, например главы с эпиграфами).
  4. Если ничего не найдено — вся книга одной главой.

Чистка текста: колонтитулы и номера страниц, надстрочные ссылки на сноски,
блоки сносок «[1] … ↑», страницы примечаний в конце книги, переносы слов,
склейка разбитых по словам строк (выключка), разделители «* * *».

Использование:
    python bl_book_parser.py book.pdf [-o book.json] [--strategy auto|toc|headings|breaks]
                                      [--keep-extras] [--keep-separators] [-v]
    python bl_book_parser.py pdf/*.pdf --out-dir parsed/

Выход (совместим с align_json_chapters.py и bl_sentence_aligner.py):
    {meta: {title, author, language, source_file, pages},
     structure: [{level, title, page, kind}],
     chapters: [{chapter_no, title, page_start, page_end, toc_level, source, parents,
                 dateline, epigraph, epigraph_source, paragraphs, text}],
     diagnostics: {...}}
"""

import re
import sys
import json
import argparse
import statistics
import unicodedata
from pathlib import Path
from collections import Counter

import fitz  # PyMuPDF

for _stream in (sys.stdout, sys.stderr):
    try:
        if _stream.encoding and "utf-8" not in _stream.encoding.lower():
            _stream.reconfigure(encoding="utf-8", errors="replace")
    except (AttributeError, ValueError):
        pass

VERBOSE = 1


def _log(msg, level=1):
    if VERBOSE >= level:
        print(msg, flush=True)


# ---------------------------------------------------------------------------
# Словари и регулярные выражения
# ---------------------------------------------------------------------------
# Кириллические двойники латинских букв, встречающиеся в «римских» цифрах (ХІV и т.п.)
_HOMO = str.maketrans({"Х": "X", "х": "x", "І": "I", "і": "i", "Ӏ": "I", "С": "C", "с": "c",
                       "М": "M", "м": "m", "Ⅰ": "I", "Ⅴ": "V", "Ⅹ": "X"})

ROMAN = r"[ivxlcdm]{1,8}"
NUM_WORDS = (
    # ru
    r"перв\w*|втор\w*|трет\w*|четв[её]рт\w*|пят\w*|шест\w*|седьм\w*|восьм\w*|девят\w*|десят\w*|"
    r"одиннадцат\w*|двенадцат\w*|тринадцат\w*|четырнадцат\w*|пятнадцат\w*|шестнадцат\w*|"
    r"семнадцат\w*|восемнадцат\w*|девятнадцат\w*|двадцат\w*|тридцат\w*|сорок\w*|"
    r"последн\w*|заключительн\w*|"
    # bg
    r"първ\w*|четвърт\w*|седм\w*|осм\w*|единадесет\w*|дванадесет\w*|тринадесет\w*|"
    r"четиринадесет\w*|петнадесет\w*|шестнадесет\w*|седемнадесет\w*|осемнадесет\w*|"
    r"деветнадесет\w*|двадесет\w*|тридесет\w*|\w+йсет\w*|\w+десет\w*|пет\w*|шест\w*|девет\w*|десет\w*|"
    # en
    r"one|two|three|four|five|six|seven|eight|nine|ten|eleven|twelve|thirteen|fourteen|"
    r"fifteen|sixteen|seventeen|eighteen|nineteen|twenty\w*|thirty\w*|first|second|third|"
    r"fourth|fifth|sixth|seventh|eighth|ninth|tenth|last|final|"
    # es / de / fr / it
    r"primer\w*|segund\w*|tercer\w*|cuart\w*|quint\w*|sext\w*|s[ée]ptim\w*|octav\w*|noven\w*|"
    r"d[ée]cim\w*|uno|dos|tres|cuatro|cinco|seis|siete|ocho|nueve|diez|"
    r"erste\w*|zweite\w*|dritte\w*|premi[eè]re?|deuxi[eè]me|troisi[eè]me|prim[oa]|second[oa]|terz[oa]"
)
_NW = rf"(?:{NUM_WORDS})"
NUM = rf"(?:\d{{1,4}}(?:-?[а-яa-z]{{1,3}})?|{ROMAN}|{_NW}(?:[\s-]+(?:и\s+|and\s+|y\s+|e\s+)?{_NW}){{0,3}})"

VOLUME_KW = r"том|книга|book|volume|tome|libro|livre|buch|band"
PART_KW = r"част|часть|part|parte|partie|teil|раздел"
CHAPTER_KW = r"глава|chapter|cap[íi]tulo|kapitel|chapitre|capitolo|rozdział|kapitola"
SPECIAL_CH = (r"пролог|епилог|эпилог|prologue?|epilogue?|pr[óo]logo|ep[íi]logo|"
              r"интерлюдия|interlude|intermedio|послесловие|послеслов\w*|afterword|"
              r"предисловие|предговор|foreword|preface|prefacio|introduction|въведение|введение|"
              r"от автора|from the author|бележка на автора|author's note|"
              r"заключение|conclusion|финал|finale")

RE_VOLUME = re.compile(rf"^(?:{VOLUME_KW})\b\.?\s*{NUM}?\b|^{NUM}\s+(?:{VOLUME_KW})\b", re.I)
RE_PART = re.compile(rf"^(?:{PART_KW})\b\.?\s*(?:{NUM})?\b|^{NUM}\s+(?:{PART_KW})\b", re.I)
RE_CHAPTER = re.compile(rf"^(?:{CHAPTER_KW})\b\.?\s*(?:{NUM})?\b|^{NUM}\s+(?:{CHAPTER_KW})\b", re.I)
RE_SPECIAL = re.compile(rf"^(?:{SPECIAL_CH})\b", re.I)
RE_BARE_NUM = re.compile(rf"^(?:\d{{1,4}}|{ROMAN})\s*[.:)]?$", re.I)
RE_NUM_TITLE = re.compile(rf"^(?:\d{{1,3}}|{ROMAN})\s*[.:)]?\s+\S", re.I)       # «1. Прошлое», «IV. Въпроси»
RE_KW_CHAPTER = re.compile(rf"\b(?:{CHAPTER_KW})\b", re.I)

FRONT_JUNK = re.compile(
    r"^(?:annotation|анотация|аннотация|заглавна страница|title page|title|cover|корица|обложка|"
    r"copyright(?: page)?|авторско право|dedication|посвещение|посвящение|contents|"
    r"table of contents|съдържание|содержание|оглавление|índice|indice|description|описание|"
    r"back cover|информация за текста|допълнителна информация|заслуги|credits|"
    r"about the author|за автора|об авторе|also by .*|other books by .*|acknowledg\w*|"
    r"благодарност\w*|uncopyright|colophon|imprint|licen[cs]e|"
    r"информация о книге|выходные данные|praise for .*|epigraph)\.?$", re.I)
BACK_CUT = re.compile(
    r"^(?:notes|endnotes|footnotes|примечания|бележки|комментарии|коментари|"
    r"печатные варианты.*|указатель.*|index|обзор содержания.*|"
    r"редакционные пояснения.*|предисловие к .*томам.*|иллюстрации|"
    r"допълнителна информация|заслуги|credits|uncopyright|colophon)\.?$", re.I)
SEPARATOR = re.compile(r"^[\s*•·∙●○◦~#=_—–\-⁂✻✱✶❖◆◇♦□■]{1,20}$")
FOOTNOTE_START = re.compile(r"^\s*(?:\[\d{1,4}\]|\{\d{1,4}\})")
INLINE_NOTE = re.compile(r"\s?(?:\[\d{1,4}\]|\{\d{1,4}\})")
PAGE_NUM = re.compile(r"^(?:[-–—]?\s*\d{1,4}\s*[-–—]?|\d{1,4}\s*/\s*\d{1,4}|"
                      rf"(?:page|стр\.?|с\.)\s*\d{{1,4}})$", re.I)
TERMINAL = ".!?…:;»\"”’)]*"


_ROMAN_TOKEN = re.compile(r"(?<!\w)[IVXLCDMivxlcdmХхІіСсМм]+(?!\w)")


def fix_roman(s):
    """Кириллические Х/І/С/М внутри «римских» чисел -> латиница (только целые токены)."""
    return _ROMAN_TOKEN.sub(lambda m: m.group(0).translate(_HOMO), s)


def norm(s):
    """Нормализация для сравнения заголовков: регистр, ё, кириллические двойники, без пунктуации."""
    s = fix_roman(unicodedata.normalize("NFKC", str(s))).lower().replace("ё", "е")
    s = "".join(ch for ch in unicodedata.normalize("NFKD", s) if not unicodedata.combining(ch))
    return re.sub(r"[\W_]+", "", s)


_RV = {"i": 1, "v": 5, "x": 10, "l": 50, "c": 100, "d": 500, "m": 1000}


def _num_value(title):
    """Значение голого номера «12», «XIV.», «ХІV» -> int | None."""
    t = fix_roman(title).strip(" .:)").lower()
    if t.isdigit():
        return int(t)
    if t and all(ch in _RV for ch in t):
        tot = 0
        for k, ch in enumerate(t):
            v = _RV[ch]
            tot += -v if k + 1 < len(t) and _RV[t[k + 1]] > v else v
        return tot
    return None


def heading_class(title):
    """'volume' | 'part' | 'chapter' | 'numtitle' («1. Название») | 'sub' (голый номер) |
    'special' (пролог, эпилог…) | None."""
    t = re.sub(r"\s+", " ", fix_roman(unicodedata.normalize("NFKC", title))).strip()
    t = t.strip(" .:-–—")
    if not t:
        return None
    if RE_BARE_NUM.match(t):
        return "sub"
    if RE_VOLUME.match(t):
        return "volume"
    if RE_PART.match(t):
        return "part"
    if RE_CHAPTER.match(t):
        return "chapter"
    if RE_SPECIAL.match(t):
        return "special"
    if RE_NUM_TITLE.match(t):
        return "numtitle"
    # «Първа глава», «Третья часть», «Книга первая»
    head = " ".join(t.split()[:3])
    if re.search(rf"^{NUM}\s+(?:{CHAPTER_KW})\b", head, re.I):
        return "chapter"
    if re.search(rf"^{NUM}\s+(?:{PART_KW})\b", head, re.I):
        return "part"
    return None


# ---------------------------------------------------------------------------
# 1. Извлечение строк
# ---------------------------------------------------------------------------
class Row:
    __slots__ = ("page", "x0", "y0", "x1", "y1", "text", "size", "bold", "italic",
                 "kind", "pw", "ph")

    def __init__(self, page, bbox, text, size, bold, italic, pw, ph):
        self.page = page
        self.x0, self.y0, self.x1, self.y1 = bbox
        self.text = text
        self.size = size
        self.bold = bold
        self.italic = italic
        self.kind = "body"  # body | furniture | title | footnote | drop
        self.pw, self.ph = pw, ph

    def __repr__(self):
        return f"<p{self.page} y={self.y0:.0f} x={self.x0:.0f} sz={self.size:.1f} {self.kind} {self.text[:40]!r}>"


_BOLD_RE = re.compile(r"bold|black|heavy|semibold|demi", re.I)
_ITAL_RE = re.compile(r"italic|oblique|[-,_]it$|[-,_]it[-,]", re.I)


def _span_style(s):
    f = s.get("font", "")
    bold = bool(s["flags"] & 16) or bool(_BOLD_RE.search(f))
    italic = bool(s["flags"] & 2) or bool(_ITAL_RE.search(f))
    return bold, italic


_NOTE_REF = re.compile(r"\[?\d{1,4}\]?|\{\d{1,4}\}|note\s*\d{1,4}|\*{1,3}", re.I)


def extract_rows(doc):
    """Строки страниц: строки MuPDF одной базовой линии склеиваются (выключка по словам),
    надстрочные ссылки на сноски удаляются."""
    pages = []
    removed_refs = 0
    flags = fitz.TEXT_PRESERVE_LIGATURES | fitz.TEXT_PRESERVE_WHITESPACE | fitz.TEXT_MEDIABOX_CLIP
    for pno in range(len(doc)):
        page = doc[pno]
        pw, ph = page.rect.width, page.rect.height
        rows = []
        cur = None
        try:
            d = page.get_text("dict", flags=flags)
        except Exception:
            pages.append(rows)
            continue
        for b in d["blocks"]:
            if b.get("type") != 0:
                continue
            for ln in b["lines"]:
                if abs(ln.get("dir", (1, 0))[1]) > 0.2:   # вертикальный/повёрнутый текст
                    continue
                spans = [s for s in ln["spans"] if s["text"]]
                if not spans:
                    continue
                mx = max(s["size"] for s in spans)
                kept = []
                for si, s in enumerate(spans):
                    t = s["text"].strip()
                    # мелкий номер в начале строки — маркер тела сноски внизу страницы: оставить
                    leading = not any(x["text"].strip() for x in spans[:si])
                    if t and s["size"] < 0.86 * mx and len(t) <= 10 and not leading and \
                            ((s["flags"] & 1) or _NOTE_REF.fullmatch(t)):
                        removed_refs += 1
                        continue
                    kept.append(s)
                if not kept or not "".join(s["text"] for s in kept).strip():
                    continue
                text = "".join(s["text"] for s in kept).replace("­\n", "").replace("\n", " ")
                chars = sum(len(s["text"].strip()) for s in kept) or 1
                size = Counter()
                bc = ic = 0
                for s in kept:
                    n = len(s["text"].strip())
                    size[round(s["size"] * 2) / 2] += n
                    b_, i_ = _span_style(s)
                    bc += n * b_
                    ic += n * i_
                x0 = ln["bbox"][0] if kept[0] is spans[0] else kept[0]["bbox"][0]
                bbox = (x0, min(s["bbox"][1] for s in kept), kept[-1]["bbox"][2], max(s["bbox"][3] for s in kept))
                r = Row(pno, bbox, text, size.most_common(1)[0][0], bc / chars >= 0.7, ic / chars >= 0.7, pw, ph)
                # склейка с предыдущей строкой той же базовой линии
                if cur is not None:
                    h = max(1.0, min(cur.y1 - cur.y0, r.y1 - r.y0))
                    same_line = abs(cur.y1 - r.y1) < 0.4 * h and abs(cur.y0 - r.y0) < 0.6 * h
                    if same_line and -2 <= r.x0 - cur.x1 < 0.25 * pw:
                        sep = "" if (cur.text.endswith(" ") or r.text.startswith(" ")
                                     or r.x0 - cur.x1 < 0.12 * r.size) else " "
                        if len(r.text) > len(cur.text):
                            cur.size = r.size
                        cur.text += sep + r.text
                        cur.x1 = max(cur.x1, r.x1)
                        cur.y0, cur.y1 = min(cur.y0, r.y0), max(cur.y1, r.y1)
                        cur.bold = cur.bold and r.bold
                        cur.italic = cur.italic and r.italic
                        continue
                cur = r
                rows.append(r)
        for r in rows:
            r.text = re.sub(r"[ \t     ]+", " ", r.text).strip()
        pages.append([r for r in rows if r.text])
    return pages, removed_refs


# ---------------------------------------------------------------------------
# 2. Статистика вёрстки
# ---------------------------------------------------------------------------
class Layout:
    pass


def _mode(vals, default=0.0):
    vals = list(vals)
    if not vals:
        return default
    return Counter(vals).most_common(1)[0][0]


def analyze_layout(pages):
    L = Layout()
    sizes = Counter()
    for rows in pages:
        for r in rows:
            sizes[r.size] += len(r.text)
    L.body_size = sizes.most_common(1)[0][0] if sizes else 12.0
    body = [r for rows in pages for r in rows if abs(r.size - L.body_size) <= 0.6]
    L.left = _mode(round(r.x0) for r in body) if body else 72
    xs = sorted(r.x1 for r in body)
    L.right = xs[int(0.9 * (len(xs) - 1))] if xs else 500
    L.width = max(50.0, L.right - L.left)
    long_rows = [r for r in body if r.x1 - r.x0 > 0.6 * L.width]
    # Отступ первой строки абзаца
    offs = Counter()
    for r in body:
        d = round(r.x0 - L.left)
        if 4 <= d <= 0.2 * L.width:
            offs[d] += 1
    if offs:
        ind, cnt = offs.most_common(1)[0]
        L.indent = float(ind)
        L.indent_ratio = cnt / max(1, len(body))
    else:
        L.indent, L.indent_ratio = 0.0, 0.0
    L.uses_indent = L.indent_ratio >= 0.04
    # левое поле каждой страницы (по длинным строкам) — для зеркальных полей
    L.page_left = {}
    for pno, rows in enumerate(pages):
        xs = [round(r.x0) for r in rows if abs(r.size - L.body_size) <= 0.6 and r.x1 - r.x0 > 0.6 * L.width]
        if len(xs) >= 3:
            L.page_left[pno] = _mode(xs)
    if not L.uses_indent:
        L.indent = max(L.indent, 2 * L.body_size)
    # Межстрочный шаг
    adv = []
    for rows in pages:
        prev = None
        for r in rows:
            if prev is not None and abs(r.size - L.body_size) <= 0.6 and abs(prev.size - L.body_size) <= 0.6:
                d = r.y0 - prev.y0
                if 0.5 * L.body_size < d < 2.5 * L.body_size:
                    adv.append(round(d, 1))
            prev = r
    L.adv = statistics.median(adv) if adv else 1.2 * L.body_size
    # Верх/низ полосы набора (по длинным строкам)
    if long_rows:
        ys0 = sorted(r.y0 for r in long_rows)
        ys1 = sorted(r.y1 for r in long_rows)
        L.top = ys0[int(0.01 * (len(ys0) - 1))]
        L.bottom = ys1[int(0.99 * (len(ys1) - 1))]
    else:
        L.top, L.bottom = 0, 10 ** 6
    L.first_mode = _mode((round(rows[0].y0) for rows in pages if rows), L.top)
    # Перенос слов в документе?
    ends = sum(1 for r in body if re.search(r"\w[-­]$", r.text))
    L.hyphenated = ends / max(1, len(body)) > 0.01
    return L


# ---------------------------------------------------------------------------
# 3. Колонтитулы, номера страниц, сноски
# ---------------------------------------------------------------------------
def mark_furniture(pages, L):
    n_pages = len(pages)
    sig_count = Counter()
    cand = []
    for rows in pages:
        seen = set()
        for r in rows:
            zone = None
            if r.y1 <= L.top + 1 or r.y1 < 0.06 * r.ph:
                zone = "top"
            elif r.y0 >= L.bottom - 1 or r.y0 > 0.94 * r.ph:
                zone = "bot"
            if zone is None or len(r.text) > 90:
                continue
            sig = (zone, round(r.y0 / 6), re.sub(r"\d+", "#", r.text.lower()))
            cand.append((r, sig))
            if sig not in seen:
                sig_count[sig] += 1
                seen.add(sig)
    removed = 0
    thr = max(3, 0.03 * n_pages)
    for r, sig in cand:
        if PAGE_NUM.match(r.text.strip()) or sig_count[sig] >= thr:
            r.kind = "furniture"
            removed += 1
    # номер страницы последней строкой неполной страницы (вне зоны колонтитула):
    # число должно совпадать с номером страницы с тем же сдвигом, что у найденных колонтитулов
    offs = Counter(int(r.text) - (r.page + 1) for r, _ in cand if r.kind == "furniture" and r.text.isdigit())
    off = offs.most_common(1)[0][0] if offs else 0
    for pno, rows in enumerate(pages):
        body = [r for r in rows if r.kind == "body"]
        if body:
            t = body[-1].text.strip()
            if t.isdigit() and len(t) <= 4 and abs(int(t) - (pno + 1 + off)) <= 1:
                body[-1].kind = "furniture"
                removed += 1
    return removed


def mark_footnotes(pages, L):
    """Блоки сносок «[1] текст … ↑» (chitanka/calibre) внутри текста."""
    removed = 0
    has_arrows = sum(1 for rows in pages for r in rows if r.text.endswith("↑")) >= 3
    in_note = False
    for rows in pages:
        for r in rows:
            if r.kind != "body":
                continue
            t = r.text
            if FOOTNOTE_START.match(t) and (r.x0 - L.left > 0.4 * L.indent or has_arrows):
                in_note = True
            if in_note:
                r.kind = "footnote"
                removed += 1
                if not has_arrows or t.endswith("↑"):
                    in_note = False
                continue
            if t == "↑":
                r.kind = "footnote"
    # Сноски внизу страницы (Word/типографская вёрстка): хвост страницы, который
    # начинается с маркера «1 текст» / «* текст» и набран не крупнее основного текста.
    marker = re.compile(r"^(?:\d{1,3}|\*{1,3}|\[\d{1,3}\])\s+\S")
    for rows in pages:
        body = [r for r in rows if r.kind == "body"]
        k = len(body)
        while k > 0 and len(body) - k < 15:
            r = body[k - 1]
            if r.size > L.body_size + 0.5 or r.y0 < 0.5 * r.ph:
                break
            k -= 1
            if marker.match(r.text) and (r.size < 0.92 * L.body_size or r.bold or
                                         (k > 0 and r.y0 - body[k - 1].y1 > 0.6 * L.adv)):
                for x in body[k:]:
                    if x.size <= L.body_size + 0.5:
                        x.kind = "footnote"
                        removed += 1
                # выше могут быть ещё сноски того же блока
                while k > 0 and marker.match(body[k - 1].text) and body[k - 1].size <= L.body_size + 0.5 \
                        and (body[k - 1].size < 0.92 * L.body_size or body[k - 1].bold):
                    k -= 1
                    body[k].kind = "footnote"
                    removed += 1
                break
            if r.size >= 0.92 * L.body_size and not r.bold:
                break      # обычная строка основного текста — сносок нет
    return removed


# ---------------------------------------------------------------------------
# 4. Заголовки по вёрстке
# ---------------------------------------------------------------------------
def _is_centered(r, L):
    c = (r.x0 + r.x1) / 2
    mid = (L.left + L.right) / 2
    return abs(c - mid) < 0.06 * L.width and (r.x1 - r.x0) < 0.75 * L.width and r.x0 - L.left > 0.08 * L.width


def find_layout_headings(pages, L):
    """Группы заголовочных строк: [{rows, title, size, cls, page}]."""
    groups = []
    for pno, rows in enumerate(pages):
        body = [r for r in rows if r.kind == "body"]
        # страница оглавления (много коротких номерных строк) — не источник заголовков
        numbered = sum(1 for r in body if len(r.text) < 60 and heading_class(r.text) in ("chapter", "part", "sub", "volume"))
        if numbered >= 6 and numbered >= 0.4 * len(body):
            continue
        cur = None
        for k, r in enumerate(body):
            t = r.text
            prev = body[k - 1] if k else None
            nxt = body[k + 1] if k + 1 < len(body) else None
            gap_before = (r.y0 - prev.y1) if prev else 99
            gap_after = (nxt.y0 - r.y1) if nxt else 99
            isolated = gap_before > 0.45 * (L.adv - L.body_size) + 2 and gap_after > 0.45 * (L.adv - L.body_size) + 2
            big = r.size >= 1.12 * L.body_size
            bold = r.bold and r.size >= 0.97 * L.body_size
            cls = heading_class(t)
            short = len(t) <= 160 and not t.endswith(",")
            dialog = bool(re.match(r"^[–—―-]\s", t))
            is_h = False
            if short and not dialog:
                if big:
                    is_h = True
                elif bold and len(t) <= 80 and (cls or isolated and (gap_before > 1.5 * L.adv or prev is None)):
                    is_h = True
                elif cls and cls != "numtitle" and len(t) <= 60 and isolated and \
                        (_is_centered(r, L) or (cls != "sub" and gap_before > 1.5 * L.adv)):
                    is_h = True
            if not is_h:
                cur = None
                continue
            if (cur is not None and cur["rows"][-1] is prev and gap_before < 2.2 * L.adv
                    and abs(r.size - cur["size"]) < 0.25 * cur["size"]
                    and not (cls == "sub" and cur["cls"] in ("chapter", "sub", "part", "volume", "numtitle"))
                    and cur["cls"] != "sub"):
                cur["rows"].append(r)
                cur["title"] += " " + t
                if cur["cls"] in (None, "special") and cls and cls != "sub":
                    cur["cls"] = cls
                continue
            cur = {"rows": [r], "title": t, "size": r.size, "cls": cls, "page": pno}
            groups.append(cur)
    # длинные «заголовки» (абзацы, набранные крупно) — это текст
    out = []
    for g in groups:
        if len(g["rows"]) > 4 or len(g["title"]) > 200:
            continue
        g["title"] = re.sub(r"\s+", " ", g["title"]).strip()
        out.append(g)
    return out


# ---------------------------------------------------------------------------
# 5. Структура: записи-заголовки -> главы
# ---------------------------------------------------------------------------
class Entry:
    def __init__(self, level, title, pos, title_rows=(), source="", cls=None, page=0):
        self.level = level
        self.title = title
        self.pos = pos            # индекс первой строки в общем списке строк
        self.title_rows = list(title_rows)
        self.source = source
        self.cls = cls if cls is not None else heading_class(title)
        self.page = page
        self.junk = False

    def __repr__(self):
        return f"<E L{self.level} p{self.page} {self.cls} {self.title[:40]!r}>"


def _flatten(pages):
    flat = [r for rows in pages for r in rows]
    index = {id(r): i for i, r in enumerate(flat)}
    return flat, index


def _match_title_rows(rows, title, max_rows=6):
    """Ищет на странице строки, из которых складывается заголовок. -> (start_idx, n_rows) | None."""
    nt = norm(title)
    if not nt:
        return None
    for k, r in enumerate(rows):
        if r.kind != "body":
            continue
        acc = ""
        for m in range(k, min(len(rows), k + max_rows)):
            if rows[m].kind != "body":
                break
            acc += norm(rows[m].text)
            if not acc:
                continue
            if acc == nt:
                return k, m - k + 1
            if len(nt) >= 12 and acc.startswith(nt):
                return k, m - k + 1
            if len(nt) >= 40 and acc.startswith(nt[:40]):       # TOC обрезал длинный заголовок
                return k, m - k + 1
            if nt.startswith(acc):
                continue
            break
    return None


_TITLE_NUM_PREFIX = re.compile(r"^([IVXLC]{1,7}|\d{1,3})[.)]?\s+(\S.*)$")
_PROSE_LEAD = re.compile(r"^[–—―«„“\"'(]")


def is_prose_title(title):
    """Закладка TOC, в которой лежит начало текста главы, а не заголовок: в некоторых
    изданиях оглавление строится из первых строк глав (реплика, предложение)."""
    if heading_class(title) in ("chapter", "part", "volume", "special"):
        return False        # «Глава I. Тук се разказва…» — длинный, но настоящий заголовок
    body = _TITLE_NUM_PREFIX.sub(r"\2", title.strip())
    words = len(body.split())
    if _PROSE_LEAD.match(body) or len(body) > 90:
        return True
    return (words >= 5 and body[-1:] in ".!…") or (words >= 3 and body[-1:] == ":")


def toc_entries(doc, pages, flat, index, L, meta_title):
    toc = doc.get_toc(simple=False)
    if not toc:
        return [], {"toc_entries": 0}
    n = len(doc)
    raw = []
    for it in toc:
        level, title, page = it[0], str(it[1]).strip(), it[2]
        dest = it[3] if len(it) > 3 and isinstance(it[3], dict) else {}
        if page < 1 or page > n:
            continue
        y = None
        to = dest.get("to")
        if dest.get("kind") == fitz.LINK_GOTO and to is not None:
            try:
                y = float(to.y)
            except Exception:
                y = None
        raw.append([level, re.sub(r"\s+", " ", title), page - 1, y])
    info = {"toc_entries": len(raw)}
    if not raw:
        return [], info

    def span(i):
        nxt = raw[i + 1][2] if i + 1 < len(raw) else n
        return nxt - raw[i][2]

    # --- хвост сносок: подряд идущие записи по одной странице в конце книги
    k = len(raw)
    while k > 0 and (BACK_CUT.match(raw[k - 1][1]) or FRONT_JUNK.match(raw[k - 1][1])):
        k -= 1
    j = k
    while j > 0 and span(j - 1) <= 1:
        j -= 1
    run = k - j
    numeric = sum(1 for e in raw[j:k] if re.fullmatch(r"(?:note\s*)?\d{1,4}\.?", e[1], re.I))
    if run >= 8 and (numeric >= 0.5 * run or run >= 20):
        info["notes_pages_from"] = raw[j][2] + 1
        info["notes_entries"] = run
        for r in flat[index[id(pages[raw[j][2]][0])] if pages[raw[j][2]] else len(flat):]:
            if r.kind == "body":
                r.kind = "drop"
        raw = raw[:j]
    # всё после первой служебной записи во второй половине книги
    for i, e in enumerate(raw):
        if BACK_CUT.match(e[1]) and e[2] > 0.5 * n:
            info["back_matter_from"] = f"{e[1]} (стр. {e[2] + 1})"
            first = next((r for q in range(e[2], n) for r in pages[q]), None)
            if first is not None:
                for r in flat[index[id(first)]:]:
                    if r.kind == "body":
                        r.kind = "drop"
            raw = raw[:i]
            break
    entries = []
    found = 0
    last_pos = 0
    for level, title, p, y in raw:
        # заголовок ищем на целевой странице и в начале двух следующих (TOC calibre
        # нередко указывает на страницу раньше); не раньше предыдущей записи
        m, rows = None, pages[p]
        for q in range(p, min(n, p + 3)):
            cand = [r for r in pages[q] if r.kind == "body" and index[id(r)] >= last_pos]
            if q > p:
                cand = cand[:8]
            mm = _match_title_rows(cand, title)
            if mm is not None:
                p, rows, m = q, cand, mm
                break
        trows = []
        if m is not None:
            s, cnt = m
            found += 1
            pos = index[id(rows[s])]
            trows = rows[s:s + cnt]
            if len(norm("".join(r.text for r in trows))) > 1.6 * len(norm(title)) + 12:
                trows = []  # «заголовок» TOC — это начало абзаца: текст не трогаем
            elif trows and is_prose_title(title) and not "".join(r.text for r in trows).isupper():
                # Текст остаётся в главе; заголовком считаем только номер, если он есть.
                nm = _TITLE_NUM_PREFIX.match(title.strip())
                keep = []
                if nm:
                    if norm(trows[0].text) == norm(nm.group(1)):
                        keep = trows[:1]
                    else:
                        fm = _TITLE_NUM_PREFIX.match(trows[0].text.strip())
                        if fm and norm(fm.group(1)) == norm(nm.group(1)):
                            trows[0].text = fm.group(2)   # номер в одной строке с текстом
                title = nm.group(1).rstrip(".)") if nm else ""
                trows = keep
        else:
            body = [r for r in rows if r.kind == "body" and (y is None or r.y1 > y + 1)]
            if body:
                pos = index[id(body[0])]
            else:
                nxt = next((r for q in range(p + 1, n) for r in pages[q] if r.kind == "body"), None)
                pos = index[id(nxt)] if nxt is not None else len(flat)
        pos = max(pos, last_pos)
        last_pos = pos
        e = Entry(level, title, pos, trows, "toc", page=p)
        e.junk = bool(FRONT_JUNK.match(title)) or bool(meta_title and norm(title) == norm(meta_title))
        entries.append(e)
    info["toc_titles_found"] = f"{found}/{len(raw)}"
    return entries, info


def choose_chapter_level(entries):
    """Самый глубокий уровень, который не является «подразделами» (голые номера внутри «Глава N»)."""
    levels = sorted({e.level for e in entries if not e.junk})
    if not levels:
        return None
    for lv in reversed(levels):
        es = [e for e in entries if e.level == lv and not e.junk]
        bare = sum(1 for e in es if e.cls == "sub")
        if bare < 0.6 * len(es) or lv == levels[0]:
            return lv
        parents = []
        for i, e in enumerate(entries):
            if e.level == lv and not e.junk:
                for q in range(i - 1, -1, -1):
                    if entries[q].level < lv:
                        parents.append(entries[q])
                        break
        chap_parents = sum(1 for p in parents if p.cls in ("chapter", "special", "numtitle"))
        if chap_parents < 0.5 * max(1, len(parents)):
            return lv          # «I», «II» внутри «Част първа» — это главы
    return levels[0]


def build_chapters(entries, flat, chapter_level, meta_title, keep_extras, source):
    """Секции между записями уровня <= chapter_level. Записи глубже — внутренние
    подзаголовки (их строки удаляются). Родители-контейнеры дают parents."""
    for e in entries:
        for r in e.title_rows:
            r.kind = "title"
    # Раздел, все подпункты которого — голые номера («Последние записи» > 1, 2, 3),
    # и который сам не «Часть/Том», — это одна глава с подразделами.
    absorbed = set()
    for i, e in enumerate(entries):
        if e.level >= chapter_level or e.junk or e.cls in ("part", "volume"):
            continue
        kids = []
        for q in range(i + 1, len(entries)):
            if entries[q].level <= e.level:
                break
            kids.append(entries[q])
        if kids and all(k.cls == "sub" for k in kids):
            absorbed.update(id(k) for k in kids)
    bounds = [e for e in entries if e.level <= chapter_level and id(e) not in absorbed]
    chapters = []
    stack = []   # контейнеры (части/тома)
    pending = []
    mt = norm(meta_title or "")
    for i, e in enumerate(bounds):
        end = bounds[i + 1].pos if i + 1 < len(bounds) else len(flat)
        rows = flat[e.pos:end]
        nxt = bounds[i + 1] if i + 1 < len(bounds) else None
        is_container = e.level < chapter_level and nxt is not None and nxt.level > e.level
        while stack and stack[-1].level >= e.level:
            stack.pop()
        if is_container:
            if not e.junk and not (mt and mt in norm(e.title)):
                stack.append(e)
            pending.extend(rows)      # вводная часть раздела (эпиграф) -> в первую главу
            continue
        if e.junk and not keep_extras:
            size = sum(len(r.text) for r in rows if r.kind == "body")
            # служебное, только если оно небольшое: под «заголовком = названию книги»
            # иногда лежит целая повесть
            limit = 15000 if FRONT_JUNK.match(e.title) else 3000
            if size < limit:
                pending = []
                continue
        chapters.append({
            "title": e.title,
            "rows": pending + list(rows),
            "level": e.level,
            "parents": [{"title": p.title, "level": p.level} for p in stack],
            "source": e.source or source,
        })
        pending = []
    if pending and chapters:
        chapters[-1]["rows"].extend(pending)
    return chapters


def layout_entries(pages, flat, index, L, meta_title):
    groups = find_layout_headings(pages, L)
    info = {"layout_headings": len(groups)}
    if not groups:
        return [], info
    # уровни: том=1, часть=2, глава=3, подраздел=4; безымянные — по кеглю
    ch_sizes = [g["size"] for g in groups if g["cls"] in ("chapter", "special", "sub", "numtitle")]
    part_sizes = [g["size"] for g in groups if g["cls"] in ("part", "volume")]
    ch_size = statistics.median(ch_sizes) if ch_sizes else None
    entries = []
    for g in groups:
        cls = g["cls"]
        if cls == "volume":
            lv = 1
        elif cls == "part":
            lv = 2
        elif cls in ("chapter", "special", "numtitle"):
            lv = 3
        elif cls == "sub":
            lv = 4
        elif ch_size is not None and g["size"] <= ch_size * 1.05:
            lv = 3
        elif part_sizes and g["size"] >= min(part_sizes) * 0.95:
            lv = 2
        elif ch_size is not None:
            lv = 2
        else:
            lv = 3
        e = Entry(lv, g["title"], index[id(g["rows"][0])], g["rows"], "headings", cls, g["page"])
        e.junk = bool(FRONT_JUNK.match(g["title"])) or bool(meta_title and norm(g["title"]) == norm(meta_title))
        entries.append(e)
    # голые номера должны идти по порядку (1, 2, 3 …): буквы словаря «М», «С», «Х»
    # или случайные «I» в тексте — не заголовки
    kept, last_val = [], None
    for e in entries:
        if e.cls == "sub":
            val = _num_value(e.title)
            ok = val is not None and (val <= 2 or (last_val is not None and 0 < val - last_val <= 2))
            if not ok:
                continue
            last_val = val
        kept.append(e)
    entries = kept
    # голые номера: подразделы внутри «Глава N», иначе — сами главы
    last_ch = None
    for e in entries:
        if e.level == 3:
            last_ch = e
        elif e.level < 3:
            last_ch = None
        elif e.level == 4 and (last_ch is None or last_ch.cls not in ("chapter",)
                               or not RE_KW_CHAPTER.search(last_ch.title)):
            e.level, e.cls = 3, "chapter"
    # служебный хвост
    n = len(pages)
    for i, e in enumerate(entries):
        if BACK_CUT.match(e.title) and e.page > 0.5 * n and e.level <= 3:
            info["back_matter_from"] = f"{e.title} (стр. {e.page + 1})"
            for r in flat[e.pos:]:
                if r.kind == "body":
                    r.kind = "drop"
            entries = entries[:i]
            break
    return entries, info


def break_entries(pages, index, L):
    """Начала секций по разрывам страниц: страница начинается заметно ниже обычного,
    а предыдущая закончилась раньше обычного."""
    out = []
    prev_last = None
    first_mode = _mode((round(b[0].y0) for b in ([r for r in rows if r.kind == "body"] for rows in pages) if b),
                       L.top)
    for p in range(len(pages)):
        body = [r for r in pages[p] if r.kind == "body"]
        if not body:
            prev_last = None
            continue
        first = body[0]
        prev_short = prev_last is None or prev_last < L.bottom - 2.5 * L.adv
        if p > 0 and first.y0 >= first_mode + 1.2 * L.adv and prev_short:
            out.append(Entry(3, "", index[id(first)], [], "breaks", "chapter", p))
        prev_last = body[-1].y1
    return out


# ---------------------------------------------------------------------------
# 6. Абзацы
# ---------------------------------------------------------------------------
def build_vocab(rows):
    vocab = Counter()
    for r in rows:
        for w in re.findall(r"\w+(?:-\w+)*", r.text.lower()):
            vocab[w] += 1
    return vocab


def _join(a, b, L, vocab):
    if a.endswith("­"):
        return a[:-1] + b
    m = re.search(r"(\w+)-$", a)
    if m and b[:1].isalpha() and b[:1].islower():
        left = m.group(1).lower()
        right = re.match(r"\w+", b)
        right = right.group(0).lower() if right else ""
        joined, hyph = left + right, left + "-" + right
        if vocab.get(joined, 0) > vocab.get(hyph, 0):
            return a[:-1] + b
        if vocab.get(hyph, 0) > 0:
            return a + b
        return (a[:-1] + b) if L.hyphenated else (a + b)
    if m and b[:1].isalpha():           # «Фейд-» + «Раута» — дефис внутри составного имени
        return a + b
    return a + " " + b


_OPENERS = "–—―-«„\"“'‘¿¡(["


def rows_to_paragraphs(rows, L, vocab, keep_separators=False):
    paras = []
    cur = ""
    prev = None
    stats = Counter()
    for r in rows:
        if r.kind != "body":
            if r.kind == "title" and cur:
                paras.append(cur)
                cur, prev = "", None
            continue
        new = prev is None
        if not new:
            # x относительно левого поля своей страницы (зеркальные поля в Word)
            x, px = r.x0 - L.page_left.get(r.page, L.left), prev.x0 - L.page_left.get(prev.page, L.left)
            same_page = prev.page == r.page
            short_prev = prev.x1 < L.right - 0.1 * L.width
            ends_sent = prev.text.rstrip()[-1:] in TERMINAL
            starts_sent = r.text[:1].isupper() or r.text[:1] in _OPENERS
            if same_page and (r.y0 - prev.y0 > 1.5 * L.adv or r.y0 < prev.y0 - 2):
                new = True                       # пустая строка между абзацами / новая колонка
            elif x > px + 0.5 * L.indent and not prev.text.endswith(("-", "­")):
                new = True                       # красная строка (отступ вправо относительно предыдущей)
            elif short_prev and ends_sent and starts_sent:
                new = True                       # короткая строка с концом предложения = конец абзаца
            elif L.uses_indent and x > 0.5 * L.indent and abs(x - px) <= 3 and ends_sent and starts_sent:
                new = True                       # две красные строки подряд: однострочная реплика
            elif abs(r.size - prev.size) > 1.5:
                new = True
        if new and cur:
            paras.append(cur)
            cur = ""
        cur = _join(cur, r.text, L, vocab) if cur else r.text
        prev = r
    if cur:
        paras.append(cur)
    out = []
    for p in paras:
        p = INLINE_NOTE.sub("", p)
        p = p.replace("↑", "").replace("­", "")
        p = re.sub(r"\s+", " ", p).strip()
        if not p:
            continue
        if FOOTNOTE_START.match(p):
            stats["footnote_paras"] += 1
            continue
        if SEPARATOR.match(p):
            stats["separators"] += 1
            if keep_separators:
                out.append(p)
            continue
        if p.isdigit() and len(p) <= 4:
            continue
        out.append(p)
    return out, stats


# ---------------------------------------------------------------------------
# 7. Метаданные
# ---------------------------------------------------------------------------
def detect_language(sample):
    s = sample.lower()
    cyr = len(re.findall(r"[а-яёъьѣ]", s))
    lat = len(re.findall(r"[a-zà-ÿñ]", s))
    if cyr > lat:
        ru = len(re.findall(r"[ыэё]", s))
        bg = len(re.findall(r"ъ", s))
        uk = len(re.findall(r"[їієґ]", s))
        if uk > ru and uk > bg:
            return "uk"
        return "ru" if ru >= bg * 0.5 else "bg"
    scores = {
        "es": len(re.findall(r"[ñ¿¡]|\b(?:el|los|las|que|una|del|por)\b", s)),
        "de": len(re.findall(r"[äöüß]|\b(?:und|der|die|das|nicht|ist)\b", s)),
        "fr": len(re.findall(r"[çœ]|\b(?:les|des|une|est|pas|qui)\b", s)),
        "it": len(re.findall(r"\b(?:il|della|che|non|sono|gli)\b", s)),
        "en": len(re.findall(r"\b(?:the|and|of|to|was|he|she)\b", s)),
    }
    return max(scores, key=scores.get) if any(scores.values()) else "und"


def build_meta(doc, path, chapters):
    md = doc.metadata or {}
    title = (md.get("title") or "").strip()
    author = (md.get("author") or "").strip()
    if not title or title.lower() in ("cover", "untitled", "unknown") or "фондов" in title.lower() \
            or title.lower().endswith((".doc", ".docx", ".pdf")):
        title = Path(path).stem
    if author.lower() in ("unknown", "неизвестен", "неизвестный"):
        author = ""
    sample = " ".join(p for ch in chapters[:5] for p in ch["paragraphs"][:40])[:20000]
    return {"title": title, "author": author, "language": detect_language(sample),
            "source_file": Path(path).name, "pages": len(doc)}


# ---------------------------------------------------------------------------
# 8. Главная функция
# ---------------------------------------------------------------------------
_GLUED_REF = re.compile(r"(?<=[a-zà-ÿа-яёъ.,;:!?…»”\"’)])(\d{1,3})(?=[\s.,;:!?…»”\"’)]|$)")


def strip_glued_refs(chapters, diag):
    """Ссылки на сноски, набранные обычными цифрами вплотную к слову («Русо1 и»,
    «Fragola!“5.»). Удаляются, только если таких мест много и номера идут по порядку."""
    nums = [int(m.group(1)) for c in chapters for p in c["paragraphs"] for m in _GLUED_REF.finditer(p)]
    if len(nums) < 5:
        return
    seq = sum(1 for a, b in zip(nums, nums[1:]) if b == a + 1 or b == 1)
    if seq < 0.5 * (len(nums) - 1):
        return
    for c in chapters:
        c["paragraphs"] = [_GLUED_REF.sub("", p) for p in c["paragraphs"]]
    diag["glued_refs_removed"] = len(nums)


def _total_chars(chapters):
    return sum(len(p) for c in chapters for p in c["paragraphs"])


def _structure(entries, ch_level):
    return [{"level": e.level, "title": e.title, "page": e.page + 1,
             "kind": "junk" if e.junk else ("chapter" if e.level == ch_level else
                                            "container" if e.level < ch_level else "sub")}
            for e in entries]


def parse_pdf(pdf_path, strategy="auto", keep_extras=False, keep_separators=False, min_chapter_chars=150):
    doc = fitz.open(pdf_path)
    pages, refs = extract_rows(doc)
    L = analyze_layout(pages)
    diag = {"body_font": L.body_size, "left_margin": round(L.left, 1), "indent": L.indent,
            "uses_indent": L.uses_indent, "line_advance": L.adv, "hyphenated": L.hyphenated,
            "footnote_refs_removed": refs}
    diag["furniture_removed"] = mark_furniture(pages, L)
    diag["footnote_lines_removed"] = mark_footnotes(pages, L)
    flat, index = _flatten(pages)
    vocab = build_vocab([r for r in flat if r.kind == "body"])
    meta_title = (doc.metadata or {}).get("title", "") or ""
    body_total = sum(len(r.text) for r in flat if r.kind == "body")

    def assemble(entries, ch_level, source):
        chs = build_chapters(entries, flat, ch_level, meta_title, keep_extras, source)
        out = []
        for c in chs:
            paras, st = rows_to_paragraphs(c["rows"], L, vocab, keep_separators)
            for k, v in st.items():
                diag[k] = diag.get(k, 0) + v
            body_rows = [r for r in c["rows"] if r.kind in ("body", "title")]
            out.append({**c, "paragraphs": paras,
                        "page_start": (body_rows[0].page + 1) if body_rows else 0,
                        "page_end": (body_rows[-1].page + 1) if body_rows else 0})
        return out

    def reset_marks(kinds=("title",)):
        for r in flat:
            if r.kind in kinds:
                r.kind = "body"

    chapters, used, structure = [], None, []
    # --- 1. TOC
    if strategy in ("auto", "toc"):
        entries, info = toc_entries(doc, pages, flat, index, L, meta_title)
        diag.update(info)
        real = [e for e in entries if not e.junk]
        if real:
            lv = choose_chapter_level(entries)
            chs = assemble(entries, lv, "toc")
            n_ok = sum(1 for c in chs if _total_chars([c]) >= min_chapter_chars)
            cover = _total_chars(chs) / max(1, body_total)
            diag["toc_chapter_level"] = lv
            diag["toc_coverage"] = round(cover, 3)
            if strategy == "toc" or (n_ok >= 3 and cover >= 0.5):
                chapters, used = chs, "toc"
                structure = _structure(entries, lv)
            else:
                reset_marks(("title", "drop"))
                _log(f"  TOC отклонён: глав {n_ok}, покрытие {cover:.0%}", 2)
        elif entries:
            reset_marks(("title", "drop"))
    # --- 2. Заголовки по вёрстке (+ 3. разрывы страниц, если глав-заголовков нет)
    if not chapters and strategy in ("auto", "headings", "breaks"):
        entries, info = layout_entries(pages, flat, index, L, meta_title) if strategy != "breaks" else ([], {})
        diag.update(info)
        n_ch = sum(1 for e in entries if e.level == 3 and not e.junk)
        if n_ch < 3:
            brk = break_entries(pages, index, L)
            known = {e.page for e in entries}
            brk = [b for b in brk if b.page not in known and b.page - 1 not in known]
            if len(brk) >= 3:
                used = "headings+breaks" if entries else "breaks"
                entries = sorted(entries + brk, key=lambda e: e.pos)
                diag["section_breaks"] = len(brk)
        if entries:
            levels = sorted({e.level for e in entries if not e.junk})
            ch_level = 3 if any(e.level == 3 for e in entries) else (levels[-1] if levels else 3)
            first = min(e.pos for e in entries)
            pre = sum(len(r.text) for r in flat[:first] if r.kind == "body")
            if pre > 3000 or keep_extras:          # длинный текст до первого заголовка
                entries.insert(0, Entry(ch_level, "", 0, [], "headings", "special", 0))
            chs = assemble(entries, ch_level, "headings")
            if sum(1 for c in chs if _total_chars([c]) >= min_chapter_chars) >= 2:
                chapters = chs
                used = used or "headings"
                structure = _structure(entries, ch_level)
            else:
                reset_marks(("title", "drop"))
    # --- 4. Вся книга — одна глава
    if not chapters:
        rows = [r for r in flat if r.kind == "body"]
        paras, _ = rows_to_paragraphs(rows, L, vocab, keep_separators)
        chapters = [{"title": Path(pdf_path).stem, "paragraphs": paras, "level": 1, "parents": [],
                     "source": "single", "page_start": 1, "page_end": len(doc)}]
        used = "single"

    strip_glued_refs(chapters, diag)
    # пустые/крошечные главы (титульные листы, страницы частей) — убрать
    final = []
    for c in chapters:
        if _total_chars([c]) < min_chapter_chars and not keep_extras:
            diag["tiny_chapters_dropped"] = diag.get("tiny_chapters_dropped", 0) + 1
            continue
        final.append(c)
    out_chapters = []
    for i, c in enumerate(final, 1):
        out_chapters.append({
            "chapter_no": i,
            "title": c["title"],
            "page_start": c["page_start"],
            "page_end": c["page_end"],
            "toc_level": c.get("level", 0),
            "source": c.get("source", used),
            "parents": c.get("parents", []),
            "dateline": "",
            "epigraph": "",
            "epigraph_source": "",
            "paragraphs": c["paragraphs"],
            "text": "\n\n".join(c["paragraphs"]),
        })
    meta = build_meta(doc, pdf_path, out_chapters)
    diag["strategy"] = used
    diag["chapters"] = len(out_chapters)
    diag["paragraphs"] = sum(len(c["paragraphs"]) for c in out_chapters)
    diag["chars"] = _total_chars(out_chapters)
    diag["body_chars_total"] = body_total
    return {"meta": meta, "structure": structure, "chapters": out_chapters, "diagnostics": diag}


def parse_pdf_to_json(pdf_path, output_json_path, expected_chapters=None, **kw):
    """Совместимость со старым API (run-parcer.py)."""
    data = parse_pdf(pdf_path, **kw)
    Path(output_json_path).write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
    d = data["diagnostics"]
    msg = (f"Сохранён: {output_json_path} — глав {d['chapters']}, абзацев {d['paragraphs']}, "
           f"символов {d['chars']}, стратегия {d['strategy']}")
    if expected_chapters and abs(d["chapters"] - expected_chapters) > max(2, 0.15 * expected_chapters):
        msg += f"  [!] ожидалось ~{expected_chapters}"
    print(msg)
    return data


def _summary(data, show=12):
    m, d = data["meta"], data["diagnostics"]
    print(f"  {m['title']!r} / {m['author']!r} [{m['language']}], стр. {m['pages']}")
    print(f"  стратегия: {d['strategy']}, глав: {d['chapters']}, абзацев: {d['paragraphs']}, "
          f"символов: {d['chars']} из {d['body_chars_total']}")
    for c in data["chapters"][:show]:
        par = " / ".join(p["title"] for p in c["parents"])
        print(f"    {c['chapter_no']:>3}. [{c['page_start']}-{c['page_end']}] "
              f"{(par[:40] + ' > ') if par else ''}{c['title'][:60]!r} ({len(c['paragraphs'])} абз.)")
    if len(data["chapters"]) > show:
        print(f"    … ещё {len(data['chapters']) - show}")


def main():
    global VERBOSE
    ap = argparse.ArgumentParser(description="PDF -> JSON книги (главы и абзацы)")
    ap.add_argument("pdf", nargs="+")
    ap.add_argument("-o", "--output", help="выходной JSON (для одного PDF)")
    ap.add_argument("--out-dir", help="каталог для JSON (для нескольких PDF)")
    ap.add_argument("--strategy", default="auto", choices=["auto", "toc", "headings", "breaks"])
    ap.add_argument("--keep-extras", action="store_true",
                    help="не выбрасывать служебные разделы (Annotation, титульные, мелкие главы)")
    ap.add_argument("--keep-separators", action="store_true", help="сохранять разделители «* * *»")
    ap.add_argument("-v", "--verbose", action="count", default=1)
    ap.add_argument("-q", "--quiet", action="store_true")
    a = ap.parse_args()
    VERBOSE = 0 if a.quiet else a.verbose
    for f in a.pdf:
        if a.output and len(a.pdf) == 1:
            out = Path(a.output)
        else:
            out = Path(a.out_dir or Path(f).parent) / (Path(f).stem + ".json")
        out.parent.mkdir(parents=True, exist_ok=True)
        print(f"== {f}")
        data = parse_pdf(f, a.strategy, a.keep_extras, a.keep_separators)
        out.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
        if VERBOSE:
            _summary(data, 12 if VERBOSE < 2 else 10 ** 6)
        print(f"  -> {out}")


if __name__ == "__main__":
    main()
