#!/usr/bin/env python3
"""
Поиск изданий одной книги на разных языках: только свободный доступ, только текстовые форматы.

По названию и автору:
1. Wikidata даёт произведение, его названия на всех языках, автора и год его смерти,
   а также страницы Wikisource.
2. Ищем издания в Wikisource, Project Gutenberg, Internet Archive и (для болгарского) на
   chitanka.info.
3. Оставляем только свободные издания в текстовых форматах (TXT, EPUB, FB2, HTML, вики-текст,
   OCR-текст) и печатаем, на каких языках сколько найдено. Ничего не скачивается.

Правила поиска (--languages, по умолчанию ru,en,bg,es,fr):
  - сначала ищется русский язык; если издания на нём нет, остальные языки не ищутся
    (код возврата 2);
  - затем по важности английский, болгарский, испанский, французский; другие языки не нужны.

Что считается свободным:
  - Wikisource и Project Gutenberg: размещают только общественное достояние;
  - Internet Archive: издание вышло до 1930 года и не закрыто для выдачи.
Перевод может быть защищён отдельно от оригинала; для старых изданий (до 1930) это почти
не встречается, для более новых результат нужно проверять вручную.

Пример:
  python find_book_editions.py --title "Дети капитана Гранта" --author "Жюль Верн" --title-lang ru
"""

import sys
import re
import json
import html
import time
import argparse
import unicodedata
import urllib.error
import urllib.parse
import urllib.request
import threading
import xml.etree.ElementTree as ET
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor

for _stream in (sys.stdout, sys.stderr):
    try:
        if _stream.encoding and "utf-8" not in _stream.encoding.lower():
            _stream.reconfigure(encoding="utf-8", errors="replace")
    except (AttributeError, ValueError):
        pass

UA = "book-edition-finder/0.1 (personal research project)"
FREE_BEFORE_YEAR = 1930           # издания до этого года — общественное достояние в США
STOP = {"the", "a", "an", "of", "and", "le", "la", "les", "l", "de", "du", "des", "el", "los", "las",
        "un", "una", "der", "die", "das", "und", "il", "lo", "i", "и", "в", "на"}
LANG_ALIASES = {
    "fre": "fr", "fra": "fr", "french": "fr", "eng": "en", "english": "en", "rus": "ru", "russian": "ru",
    "ger": "de", "deu": "de", "german": "de", "spa": "es", "spanish": "es", "ita": "it", "italian": "it",
    "bul": "bg", "bulgarian": "bg", "pol": "pl", "polish": "pl", "por": "pt", "portuguese": "pt",
    "dut": "nl", "nld": "nl", "dutch": "nl", "swe": "sv", "swedish": "sv", "fin": "fi", "finnish": "fi",
    "cze": "cs", "ces": "cs", "czech": "cs", "lat": "la", "latin": "la", "chi": "zh", "chinese": "zh",
    "jpn": "ja", "japanese": "ja", "hun": "hu", "hungarian": "hu", "dan": "da", "danish": "da",
    "nor": "no", "norwegian": "no", "ukr": "uk", "ukrainian": "uk", "srp": "sr", "serbian": "sr",
    "hrv": "hr", "croatian": "hr", "tur": "tr", "turkish": "tr", "gre": "el", "ell": "el", "greek": "el",
    "heb": "he", "hebrew": "he", "ara": "ar", "arabic": "ar", "epo": "eo", "esperanto": "eo",
    "cat": "ca", "catalan": "ca", "rum": "ro", "ron": "ro", "romanian": "ro", "slv": "sl", "slk": "sk",
}


MIN_GAP = 0.4                      # секунд между любыми двумя запросами (вежливость к серверам)
_LOCK = threading.Lock()
_LAST = [0.0]


def _throttle():
    with _LOCK:
        wait = MIN_GAP - (time.time() - _LAST[0])
        if wait > 0:
            time.sleep(wait)
        _LAST[0] = time.time()


def http_get(url, timeout=40, tries=5):
    last = None
    for i in range(tries):
        _throttle()
        try:
            req = urllib.request.Request(url, headers={"User-Agent": UA})
            with urllib.request.urlopen(req, timeout=timeout) as r:
                return r.read()
        except urllib.error.HTTPError as exc:
            last = exc
            if exc.code not in (429, 502, 503, 504):
                break                                   # 404 и прочее повторять бессмысленно
            retry = exc.headers.get("Retry-After", "")
            time.sleep(min(float(retry) if retry.isdigit() else 5 * (i + 1), 60))
        except Exception as exc:                        # сеть: повторяем, потом сообщаем
            last = exc
            time.sleep(1.5 * (i + 1))
    raise RuntimeError(f"{last} ({url[:70]})")


def get_json(url):
    return json.loads(http_get(url).decode("utf-8"))


def norm(s):
    s = unicodedata.normalize("NFKD", str(s).lower().replace("ё", "е"))
    s = "".join(c for c in s if not unicodedata.combining(c))
    return re.sub(r"[^\w]+", " ", s).strip()


def tokens(s):
    return [t for t in norm(s).split() if t not in STOP and len(t) > 1]


def lang_code(value):
    v = (value[0] if isinstance(value, list) and value else value) or ""
    v = str(v).strip().lower()
    return LANG_ALIASES.get(v, v if 2 <= len(v) <= 3 else "und")


def title_matches(candidate, labels):
    """Название издания совпадает хотя бы с одним из известных названий произведения."""
    c = set(tokens(candidate))
    if len(c) < 2:
        return False
    for label in labels:
        l = set(tokens(label))
        if len(l) < 2:
            continue
        if len(c & l) / min(len(c), len(l)) >= 0.7:
            return True
    return False


def title_close(candidate, label):
    """Строгое сравнение (когда автора проверить нечем): наборы слов почти совпадают."""
    c, l = set(tokens(candidate)), set(tokens(label))
    if not c or not l:
        return False
    return len(c ^ l) <= 1 or len(c & l) / max(len(c), len(l)) >= 0.8


FAILED = [0]      # число запросов, которые не удались (результат может быть неполным)


def pmap(fn, items, workers=3):
    """Параллельный map для сетевых запросов; при ошибке элемент даёт None."""
    def safe(x):
        try:
            return fn(x)
        except Exception as exc:
            FAILED[0] += 1
            FAILED.append(str(exc)[:160])
            return None
    with ThreadPoolExecutor(max_workers=workers) as ex:
        return list(ex.map(safe, items))


def author_matches(names, author_labels):
    surnames = {t for lab in author_labels for t in tokens(lab)[-1:]}
    toks = {t for n in names for t in tokens(n)}
    return bool(surnames & toks)


def latin_surname(work):
    latin = [a for a in work["author_labels"] if re.fullmatch(r"[\w\s.\-'’]+", a, re.ASCII)]
    return tokens(latin[0])[-1] if latin and tokens(latin[0]) else ""


# ---------------------------------------------------------------------------
# Wikidata
# ---------------------------------------------------------------------------
def wd_api(**params):
    params["format"] = "json"
    return get_json("https://www.wikidata.org/w/api.php?" + urllib.parse.urlencode(params))


def death_year(entity):
    try:
        t = entity["claims"]["P570"][0]["mainsnak"]["datavalue"]["value"]["time"]
        return int(t[1:5]) * (-1 if t[0] == "-" else 1)
    except (KeyError, IndexError, ValueError):
        return None


def resolve_work(title, title_lang, author):
    found = wd_api(action="wbsearchentities", search=title, language=title_lang, uselang=title_lang,
                   type="item", limit=10).get("search", [])
    if not found:
        raise SystemExit(f"В Wikidata не найдено: {title!r}")
    works = wd_api(action="wbgetentities", ids="|".join(f["id"] for f in found),
                   props="labels|claims|sitelinks")["entities"]
    best = None
    for qid, ent in works.items():
        author_ids = [c["mainsnak"]["datavalue"]["value"]["id"]
                      for c in ent.get("claims", {}).get("P50", []) if "datavalue" in c["mainsnak"]]
        if not author_ids:
            continue
        authors = wd_api(action="wbgetentities", ids="|".join(author_ids), props="labels|claims")["entities"]
        for aid, aent in authors.items():
            alabels = [v["value"] for v in aent.get("labels", {}).values()]
            if tokens(author) and author_matches([author], alabels):
                score = len(ent.get("sitelinks", {}))
                if best is None or score > best[0]:
                    best = (score, qid, ent, aid, aent)
    if best is None:
        raise SystemExit(f"В Wikidata не найдено произведение {title!r} автора {author!r}")
    _, qid, ent, aid, aent = best
    return {
        "work_id": qid, "author_id": aid,
        "labels": {k: v["value"] for k, v in ent.get("labels", {}).items()},
        "author_labels": [v["value"] for v in aent.get("labels", {}).values()],
        "author_death_year": death_year(aent),
        "sitelinks": ent.get("sitelinks", {}),
    }


# ---------------------------------------------------------------------------
# Источники
# ---------------------------------------------------------------------------
def from_wikisource(work):
    out = []
    for key, link in work["sitelinks"].items():
        if key.endswith("wikisource") and key != "wikisource":
            lang = key[:-len("wikisource")].replace("_", "-")
            url = f"https://{lang}.wikisource.org/wiki/" + urllib.parse.quote(link["title"].replace(" ", "_"))
            out.append({"source": "wikisource", "lang": lang_code(lang), "title": link["title"],
                        "url": url, "format": "HTML/вики-текст",
                        "free": "Wikisource: только общественное достояние"})
    # Wikidata связывает с Wikisource не все страницы: ищем по названию в каждом языковом разделе.
    have = {e["lang"] for e in out}
    try:   # реально существующие языковые разделы Wikisource (один запрос)
        matrix = get_json("https://www.wikidata.org/w/api.php?action=sitematrix&smtype=language"
                          "&smlangprop=code%7Csite&smsiteprop=code&format=json")["sitematrix"]
        existing = {v["code"] for k, v in matrix.items() if k.isdigit()
                    and any(s.get("code") == "wikisource" for s in v.get("site", []))}
    except (RuntimeError, KeyError, ValueError):
        existing = set(work["labels"])
    tasks = [(lang, label) for lang, label in work["labels"].items()
             if lang in existing and lang_code(lang) not in have]

    def search(task):
        lang, label = task
        q = urllib.parse.urlencode({"action": "query", "list": "search", "srsearch": label,
                                    "srnamespace": 0, "srlimit": 8, "format": "json"})
        hits = json.loads(http_get(f"https://{lang}.wikisource.org/w/api.php?{q}", timeout=25, tries=3)
                          .decode("utf-8"))["query"]["search"]
        pages = {}
        for h in hits:
            # подстраницы («…/Глава 1») и главы с номером на конце («… 04») — части одной книги
            base = re.sub(r"[\s_]+\d{1,3}$", "", h["title"].split("/")[0])
            m = re.search(r"\((.*?)\)", base)
            if not title_close(re.sub(r"\(.*?\)", "", base), label):
                continue
            if m and not author_matches([m.group(1)], work["author_labels"]):
                continue
            pages.setdefault(base, h["title"])
        return [{"source": "wikisource", "lang": lang_code(lang), "title": base,
                 "url": f"https://{lang}.wikisource.org/wiki/" + urllib.parse.quote(base.replace(" ", "_")),
                 "format": "HTML/вики-текст", "free": "Wikisource: только общественное достояние"}
                for base in pages]

    for res in pmap(search, tasks):
        out += res or []
    return out


def from_gutenberg(work, max_queries=16):
    ns = {"rdf": "http://www.w3.org/1999/02/22-rdf-syntax-ns#", "dcterms": "http://purl.org/dc/terms/",
          "pgterms": "http://www.gutenberg.org/2009/pgterms/"}
    surname = latin_surname(work)
    labels = list(dict.fromkeys(norm(v) for v in work["labels"].values()))[:max_queries]
    all_labels = list(work["labels"].values())

    def search(label):
        q = urllib.parse.quote(f"{surname} {label}".strip())
        xml = http_get(f"https://www.gutenberg.org/ebooks/search.opds/?query={q}", tries=2).decode("utf-8", "replace")
        return re.findall(r"/ebooks/(\d+)\.opds", xml)

    ids = []
    for found in pmap(search, labels):
        for i in found or []:
            if i not in ids:
                ids.append(i)

    def edition(i):
        root = ET.fromstring(http_get(f"https://www.gutenberg.org/cache/epub/{i}/pg{i}.rdf", tries=2))
        ebook = root.find("pgterms:ebook", ns)
        if ebook is None:
            return None
        title = ebook.findtext("dcterms:title", namespaces=ns) or ""
        creators = [n.text or "" for n in ebook.iterfind(".//dcterms:creator//pgterms:name", ns)]
        langs = [v.text for v in ebook.iterfind("dcterms:language//rdf:value", ns)]
        if not (title_matches(title, all_labels) and author_matches(creators, work["author_labels"])):
            return None
        files = {}
        for f in ebook.iterfind("dcterms:hasFormat/pgterms:file", ns):
            mime = f.findtext("dcterms:format//rdf:value", namespaces=ns) or ""
            about = f.get("{%s}about" % ns["rdf"], "")
            if mime.startswith("text/plain"):
                files.setdefault("TXT", about)
            elif mime.startswith("application/epub"):
                files.setdefault("EPUB", about)
            elif mime.startswith("text/html"):
                files.setdefault("HTML", about)
        if not files:
            return None
        fmt = "TXT" if "TXT" in files else next(iter(files))
        return {"source": "gutenberg", "lang": lang_code(langs), "title": title.replace("\n", " "),
                "url": files[fmt], "format": "/".join(files),
                "free": "Project Gutenberg: общественное достояние",
                "page": f"https://www.gutenberg.org/ebooks/{i}"}

    return [e for e in pmap(edition, ids[:80]) if e]


def from_internet_archive(work, max_labels=12, per_label=10):
    surname = latin_surname(work)
    all_labels = list(work["labels"].values())
    queries = []
    for label in list(dict.fromkeys(all_labels))[:max_labels]:
        words = " ".join(tokens(label))
        if len(words.split()) >= 2:
            queries.append(words)

    def search(words):
        url = ("https://archive.org/advancedsearch.php?" + urllib.parse.urlencode(
            [("q", f"title:({words}) AND mediatype:texts"), ("fl[]", "identifier"),
             ("rows", per_label), ("output", "json")]))
        return [d["identifier"] for d in get_json(url)["response"]["docs"]]

    ids = []
    for found in pmap(search, queries):
        for i in found or []:
            if i not in ids:
                ids.append(i)

    def edition(i):
        meta = get_json(f"https://archive.org/metadata/{i}")
        md = meta.get("metadata", {})
        title = md.get("title", "")
        title = " ".join(title) if isinstance(title, list) else title
        creators = md.get("creator", [])
        creators = creators if isinstance(creators, list) else [creators]
        author_ok = (not creators or author_matches(creators, work["author_labels"])
                     or (surname and surname in norm(" ".join(creators))))
        if not (title_matches(title, all_labels) and author_ok):
            return None
        if str(md.get("access-restricted-item", "")).lower() == "true":
            return None
        m = re.search(r"\d{4}", str(md.get("date", "") or md.get("year", "")))
        year = int(m.group()) if m else None
        if year is None or year >= FREE_BEFORE_YEAR:
            return None
        files = {f.get("format"): f["name"] for f in meta.get("files", []) if f.get("name")}
        fmt = next((k for k in ("EPUB", "Text", "DjVuTXT") if k in files), None)
        if fmt is None:
            return None
        return {"source": "internet_archive", "lang": lang_code(md.get("language")), "title": title,
                "url": f"https://archive.org/download/{i}/{urllib.parse.quote(files[fmt])}",
                "format": {"Text": "TXT", "DjVuTXT": "OCR-текст"}.get(fmt, fmt),
                "free": f"Internet Archive: издание {year} года", "page": f"https://archive.org/details/{i}"}

    return [e for e in pmap(edition, ids[:60]) if e]


def from_chitanka(work):
    """chitanka.info — свободная болгарская библиотека (ссылки на EPUB, FB2, TXT)."""
    bg = work["labels"].get("bg")
    if not bg:
        return []
    page = re.sub(r"\s+", " ", http_get("https://chitanka.info/search?q=" + urllib.parse.quote(bg))
                  .decode("utf-8", "replace"))
    all_labels = list(work["labels"].values())

    def clean(x):
        return html.unescape(re.sub(r"<[^>]+>", "", x)).strip()

    found = {}
    for marker in ('<article class="media book-media"', '<dl class="text-entity"'):   # сначала книги, потом тексты
        for chunk in page.split(marker)[1:]:
            chunk = chunk[:6000]
            t = re.search(r'<i(?: itemprop="name")?>([^<]+)</i>', chunk)
            a = re.search(r'itemprop="author".*?itemprop="name"[^>]*>(.*?)</a>', chunk)
            p = re.search(r'href="(/(?:book|text)/\d+-[\w-]+)"', chunk)
            links = re.findall(r'href="(/(?:book|text)/\d+-[\w-]+\.(?:epub|fb2\.zip|txt\.zip))"', chunk)
            if not (t and a and p and links):
                continue
            title, author = clean(t.group(1)), clean(a.group(1))
            if not (title_matches(title, all_labels) and author_matches([author], work["author_labels"])):
                continue
            by_ext = {l.split(".", 1)[1].upper().replace("FB2.ZIP", "FB2").replace("TXT.ZIP", "TXT"): l for l in links}
            best = by_ext.get("EPUB") or by_ext.get("FB2") or next(iter(by_ext.values()))
            found.setdefault(norm(title), {
                "source": "chitanka", "lang": "bg", "title": title,
                "url": "https://chitanka.info" + best, "format": "/".join(by_ext),
                "free": "Читанка: свободная библиотека", "page": "https://chitanka.info" + p.group(1)})
    return list(found.values())


def restricted(work, wanted, keep_label_lang):
    """Копия произведения только с названиями и страницами Wikisource нужных языков."""
    w = dict(work)
    w["labels"] = {k: v for k, v in work["labels"].items() if lang_code(k) in wanted or k == keep_label_lang}
    w["sitelinks"] = {k: v for k, v in work["sitelinks"].items()
                      if k.endswith("wikisource") and lang_code(k[:-len("wikisource")].replace("_", "-")) in wanted}
    return w


def dedupe(editions):
    seen, out = set(), []
    for e in editions:
        key = (e["source"], e["url"])
        if key not in seen:
            seen.add(key)
            out.append(e)
    return out


# ---------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("--title", required=True, help="название книги (на любом языке)")
    ap.add_argument("--author", required=True, help="автор (на любом языке)")
    ap.add_argument("--title-lang", default="ru", help="язык введённого названия (по умолчанию ru)")
    ap.add_argument("--json", default="", help="сохранить отчёт в JSON")
    ap.add_argument("--no-ia", action="store_true", help="не искать в Internet Archive (быстрее)")
    ap.add_argument("--languages", default="ru,en,bg,es,fr",
                    help="языки по важности; первый обязателен: если на нём издания нет, остальные не ищутся")
    args = ap.parse_args()
    priority = [l.strip().lower() for l in args.languages.split(",") if l.strip()]
    must, others = priority[0], priority[1:]

    print(f"[1/3] Wikidata: {args.title!r} / {args.author!r}")
    work = resolve_work(args.title, args.title_lang, args.author)
    dy = work["author_death_year"]
    print(f"  произведение {work['work_id']}, автор {work['author_id']}, умер: {dy}, "
          f"названий на языках: {len(work['labels'])}")
    if dy is None or dy > 1954:
        print(f"  ВНИМАНИЕ: автор умер в {dy} году (или год неизвестен) — оригинал может быть защищён "
              f"авторским правом, находки нужно проверять вручную.")

    def search(wanted):
        w = restricted(work, wanted, args.title_lang)
        sources = [("Wikisource", from_wikisource), ("Project Gutenberg", from_gutenberg)]
        if not args.no_ia:
            sources.append(("Internet Archive", from_internet_archive))
        if "bg" in wanted:
            sources.append(("Читанка", from_chitanka))
        found = []
        for name, fn in sources:
            before = FAILED[0]
            got = [e for e in fn(w) if e["lang"] in wanted]
            failed = FAILED[0] - before
            print(f"  {name}: {len(got)}"
                  + (f"  (не удалось запросов: {failed}; например: {FAILED[-1]})" if failed else ""))
            found += got
        return dedupe(found)

    print(f"[2/3] Язык «{must}» (обязателен)")
    editions = search({must})
    if not editions:
        print(f"\nИздания на языке «{must}» в свободном доступе в текстовых форматах не найдено, "
              f"остальные языки не ищем.")
        if FAILED[0]:
            print(f"  ВНИМАНИЕ: не удалось запросов: {FAILED[0]}, результат может быть неполным")
        return 2
    print(f"[3/3] Остальные языки: {', '.join(others)}")
    editions += search(set(others))

    by_lang = defaultdict(list)
    for e in editions:
        by_lang[e["lang"]].append(e)

    print(f"\nИтого: {len(editions)} изданий (свободный доступ, текстовые форматы)")
    print("  по языкам: " + ", ".join(f"{l} {len(by_lang[l])}" for l in priority))
    if FAILED[0]:
        print(f"  ВНИМАНИЕ: не удалось запросов: {FAILED[0]}, результат может быть неполным")
    for lang in priority:
        items = by_lang[lang]
        srcs = defaultdict(int)
        for e in items:
            srcs[e["source"]] += 1
        print(f"\n  {lang}: {len(items)}" + ("  (" + ", ".join(f"{k} {v}" for k, v in sorted(srcs.items())) + ")"
                                            if items else ""))
        for e in items:
            print(f"     [{e['format']}] {e['title'][:60]} — {e.get('page', e['url'])}")
    if args.json:
        with open(args.json, "w", encoding="utf-8") as f:
            json.dump({"work": {k: work[k] for k in ("work_id", "author_id", "author_death_year")},
                       "editions": editions}, f, ensure_ascii=False, indent=2)
        print(f"\nОтчёт: {args.json}")


if __name__ == "__main__":
    sys.exit(main() or 0)
