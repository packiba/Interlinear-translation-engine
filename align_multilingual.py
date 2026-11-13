#!/usr/bin/env python3
"""
Многоязычное выравнивание: главы и предложения одновременно для N языков.
"""

import sys
import json
import argparse
from pathlib import Path
from typing import Dict, List, Tuple

for _stream in (sys.stdout, sys.stderr):
    try:
        if _stream.encoding and "utf-8" not in _stream.encoding.lower():
            _stream.reconfigure(encoding="utf-8", errors="replace")
    except (AttributeError, ValueError):
        pass


def load_book(path: str) -> Dict:
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    if "chapters" not in data or not isinstance(data["chapters"], list):
        raise ValueError(f"File {path} missing chapters array")
    return data


def find_common_chapters(books: Dict[str, Dict]) -> List[Tuple[str, int]]:
    """
    Найти общие главы по максимальному пересечению.
    Возвращает список (глава_RU, индекс_RU) для максимально общего набора.
    """
    ru_book = books["ru"]
    ru_chapters = ru_book["chapters"]
    n_ru = len(ru_chapters)

    print(f"[*] Анализ структуры:")
    for lang, book in sorted(books.items()):
        print(f"    {lang.upper()}: {len(book['chapters'])} глав")

    max_common = min(len(book["chapters"]) for book in books.values())
    print(f"[*] Максимум общих глав: {max_common}")

    # Используем первые N-1 глав (исключаем предисловия/благодарности)
    common = []
    for i in range(max_common):
        common.append((f"ch_{i+1}", i))

    return common


def align_chapters(books: Dict[str, Dict], common_indices: List[Tuple[str, int]]) -> Dict:
    """
    Выровнять главы для всех языков.
    Возвращает структуру с главами {lang: [выровненные_главы]}
    """
    result = {}

    for lang, book in books.items():
        aligned_chapters = []
        for ch_id, idx in common_indices:
            if idx < len(book["chapters"]):
                chapter = book["chapters"][idx]
                # Стандартизируем формат
                aligned_chapters.append({
                    "chapter_no": idx + 1,
                    "title": str(chapter.get("title", "")).strip(),
                    "paragraphs": chapter.get("paragraphs", []),
                    "text": "\n\n".join(str(p).strip()
                           for p in chapter.get("paragraphs", []) if str(p).strip()),
                })
        result[lang] = aligned_chapters

    return result


def main():
    parser = argparse.ArgumentParser(description="Многоязычное выравнивание книг")
    parser.add_argument("ru", help="RU JSON")
    parser.add_argument("en", help="EN JSON")
    parser.add_argument("fr", help="FR JSON")
    parser.add_argument("es", help="ES JSON")
    parser.add_argument("-o", "--output", default="multilang_aligned.json",
                        help="Output JSON")

    args = parser.parse_args()

    books = {
        "ru": load_book(args.ru),
        "en": load_book(args.en),
        "fr": load_book(args.fr),
        "es": load_book(args.es),
    }

    print("[1/3] Анализ структуры...")
    common = find_common_chapters(books)
    print(f"      Использованы первые {len(common)} глав\n")

    print("[2/3] Выравнивание глав...")
    aligned = align_chapters(books, common)
    for lang, chaps in aligned.items():
        print(f"      {lang.upper()}: {len(chaps)} выровненных глав")

    print("\n[3/3] Сохранение результата...")
    output = {
        "version": "2-multilingual",
        "languages": ["ru", "en", "fr", "es"],
        "n_chapters": len(aligned["ru"]),
        "chapters": []
    }

    for ch_idx in range(len(aligned["ru"])):
        ch = {
            "chapter_no": ch_idx + 1,
            "languages": {}
        }
        for lang in ["ru", "en", "fr", "es"]:
            if ch_idx < len(aligned[lang]):
                ch["languages"][lang] = aligned[lang][ch_idx]
        output["chapters"].append(ch)

    Path(args.output).write_text(json.dumps(output, ensure_ascii=False, indent=2),
                                 encoding="utf-8")
    print(f"[OK] Многоязычная структура: {args.output}")


if __name__ == "__main__":
    main()
