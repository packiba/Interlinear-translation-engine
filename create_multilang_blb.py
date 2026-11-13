#!/usr/bin/env python3
"""
Создание многоязычного BLB файла из многоязычного выравнивания.
"""

import sys
import json
import base64
import argparse
from pathlib import Path

import fitz

for _stream in (sys.stdout, sys.stderr):
    try:
        if _stream.encoding and "utf-8" not in _stream.encoding.lower():
            _stream.reconfigure(encoding="utf-8", errors="replace")
    except (AttributeError, ValueError):
        pass


def extract_cover_from_pdf(pdf_path, page_num=0, dpi=150):
    doc = fitz.open(pdf_path)
    page = doc[page_num]
    pix = page.get_pixmap(matrix=fitz.Matrix(dpi/72, dpi/72))
    image_data = pix.tobytes(output="png")
    return {
        "mime": "image/png",
        "width": pix.width,
        "height": pix.height,
        "data": image_data
    }


def create_multilang_blb(multilang_json, ru_pdf_path, output_path):
    with open(multilang_json, 'r', encoding='utf-8') as f:
        multilang_data = json.load(f)

    print("[*] Извлечение обложки...")
    cover_info = extract_cover_from_pdf(ru_pdf_path, page_num=0, dpi=150)
    cover_base64 = base64.b64encode(cover_info["data"]).decode('utf-8')

    print("[*] Создание многоязычного BLB...")
    blb = {
        "blb_version": 2,
        "format": "multilingual",
        "book": {
            "title": "Маленький принц",
            "author": "Антуан де Сент Экзюпери",
            "languages": ["ru", "en", "fr", "es"],
            "lang_names": {
                "ru": "Русский",
                "en": "English",
                "fr": "Français",
                "es": "Español"
            }
        },
        "cover": {
            "mime": "image/png",
            "width": cover_info["width"],
            "height": cover_info["height"],
            "data_base64": cover_base64
        },
        "chapters": multilang_data["chapters"]
    }

    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(blb, f, ensure_ascii=False, indent=2)

    print(f"[OK] Многоязычный BLB: {output_path}")
    print(f"    Версия BLB: {blb['blb_version']}")
    print(f"    Формат: {blb['format']}")
    print(f"    Языки: {', '.join(blb['book']['languages'])}")
    print(f"    Глав: {len(blb['chapters'])}")

    total_pairs = sum(len(ch['multilingual_pairs']) for ch in blb['chapters'])
    print(f"    Многоязычных пар: {total_pairs}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Создание многоязычного BLB")
    parser.add_argument("multilang_json", help="Многоязычное выравнивание JSON")
    parser.add_argument("ru_pdf", help="PDF источника (для обложки)")
    parser.add_argument("-o", "--output", default="multilang.blb", help="Output BLB file")

    args = parser.parse_args()

    create_multilang_blb(args.multilang_json, args.ru_pdf, args.output)
