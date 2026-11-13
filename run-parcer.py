from pathlib import Path

from bl_book_parser import parse_pdf_to_json

PDF_DIR = Path("pdf")
languages = ["bg", "ru", "es", "en"]

for lang in languages:
    pdf_path = PDF_DIR / f"11-22-63-{lang}.pdf"
    json_path = f"11-22-63-{lang}.json"

    # Парсим PDF и сохраняем первичный JSON (главы + абзацы для bl_sentence_aligner.py)
    parse_pdf_to_json(str(pdf_path), json_path, expected_chapters=31)
