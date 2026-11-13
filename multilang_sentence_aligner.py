#!/usr/bin/env python3
"""
Многоязычное выравнивание предложений для варианта 2.
Использует текущий алгоритм bl_sentence_aligner для получения попарных выравниваний,
затем объединяет их в единую структуру.
"""

import sys
import json
import subprocess
import tempfile
from pathlib import Path
from typing import Dict, List, Tuple

for _stream in (sys.stdout, sys.stderr):
    try:
        if _stream.encoding and "utf-8" not in _stream.encoding.lower():
            _stream.reconfigure(encoding="utf-8", errors="replace")
    except (AttributeError, ValueError):
        pass


def create_bilingual_json(multilang: Dict, lang1: str, lang2: str) -> Tuple[str, str]:
    """Создать временные JSON для двух языков в формате для bl_sentence_aligner."""
    lang1_data = {
        "meta": multilang.get("meta", {}),
        "chapters": []
    }
    lang2_data = {
        "meta": multilang.get("meta", {}),
        "chapters": []
    }

    for ch in multilang["chapters"]:
        if lang1 in ch["languages"] and lang2 in ch["languages"]:
            lang1_data["chapters"].append(ch["languages"][lang1])
            lang2_data["chapters"].append(ch["languages"][lang2])

    f1 = tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False, encoding='utf-8')
    f2 = tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False, encoding='utf-8')

    json.dump(lang1_data, f1, ensure_ascii=False)
    json.dump(lang2_data, f2, ensure_ascii=False)

    f1.close()
    f2.close()

    return f1.name, f2.name


def run_pairwise_alignment(lang1_file: str, lang2_file: str, lang1: str, lang2: str) -> Dict:
    """Запустить bl_sentence_aligner для пары языков."""
    output_file = f"temp_alignment_{lang1}_{lang2}.json"

    cmd = [
        sys.executable, "bl_sentence_aligner.py",
        lang1_file, lang2_file,
        "-o", output_file
    ]

    print(f"  Выравнивание {lang1.upper()}-{lang2.upper()}...")
    result = subprocess.run(cmd, capture_output=True, text=True)

    if result.returncode != 0:
        print(f"    ⚠ Ошибка при выравнивании: {result.stderr}")
        return {"chapters": []}

    with open(output_file, 'r', encoding='utf-8') as f:
        data = json.load(f)

    Path(output_file).unlink()
    return data


def merge_pairwise_alignments(multilang: Dict, alignments: Dict[str, Dict]) -> Dict:
    """
    Объединить попарные выравнивания в единую многоязычную структуру.
    
    Структура выравнивания по главам:
    - Для каждой главы есть пары RU-EN, RU-FR, RU-ES
    - Объединяем их в одну главу со всеми предложениями всех языков
    """
    result = {
        "version": "2-multilingual-aligned",
        "languages": ["ru", "en", "fr", "es"],
        "n_chapters": len(multilang["chapters"]),
        "chapters": []
    }

    for ch_idx in range(len(multilang["chapters"])):
        ch_merged = {
            "chapter_no": ch_idx + 1,
            "title": multilang["chapters"][ch_idx]["languages"]["ru"].get("title", ""),
            "multilingual_pairs": []
        }

        # Получить пары RU-EN, RU-FR, RU-ES для этой главы
        ru_en_pairs = []
        ru_fr_pairs = []
        ru_es_pairs = []

        if "ru-en" in alignments and ch_idx < len(alignments["ru-en"].get("chapters", [])):
            ru_en_pairs = alignments["ru-en"]["chapters"][ch_idx].get("pairs", [])
        if "ru-fr" in alignments and ch_idx < len(alignments["ru-fr"].get("chapters", [])):
            ru_fr_pairs = alignments["ru-fr"]["chapters"][ch_idx].get("pairs", [])
        if "ru-es" in alignments and ch_idx < len(alignments["ru-es"].get("chapters", [])):
            ru_es_pairs = alignments["ru-es"]["chapters"][ch_idx].get("pairs", [])

        # Объединить всё в одну структуру
        # Каждая строка: (RU, EN, FR, ES)
        max_pairs = max(len(ru_en_pairs), len(ru_fr_pairs), len(ru_es_pairs))

        for i in range(max_pairs):
            multilang_pair = {
                "ru": "",
                "en": "",
                "fr": "",
                "es": ""
            }

            if i < len(ru_en_pairs):
                multilang_pair["ru"] = ru_en_pairs[i].get("src", "")
                multilang_pair["en"] = ru_en_pairs[i].get("tgt", "")

            if i < len(ru_fr_pairs):
                # ru уже задан, не перезаписываем
                multilang_pair["fr"] = ru_fr_pairs[i].get("tgt", "")

            if i < len(ru_es_pairs):
                # ru уже задан, не перезаписываем
                multilang_pair["es"] = ru_es_pairs[i].get("tgt", "")

            ch_merged["multilingual_pairs"].append(multilang_pair)

        result["chapters"].append(ch_merged)

    return result


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Многоязычное выравнивание предложений")
    parser.add_argument("multilang_aligned", help="Выровненная многоязычная структура глав")
    parser.add_argument("-o", "--output", default="multilang_aligned_sentences.json",
                        help="Output JSON")

    args = parser.parse_args()

    print(f"[*] Загрузка {args.multilang_aligned}...")
    with open(args.multilang_aligned, 'r', encoding='utf-8') as f:
        multilang = json.load(f)

    print(f"[*] Выравнивание предложений для {multilang['n_chapters']} глав...")
    print()

    alignments = {}

    # Создать попарные выравнивания RU-EN, RU-FR, RU-ES
    for lang in ["en", "fr", "es"]:
        lang1_file, lang2_file = create_bilingual_json(multilang, "ru", lang)

        alignment = run_pairwise_alignment(lang1_file, lang2_file, "ru", lang)
        alignments[f"ru-{lang}"] = alignment

        Path(lang1_file).unlink()
        Path(lang2_file).unlink()

    print()
    print("[*] Объединение попарных выравниваний...")
    multilang_result = merge_pairwise_alignments(multilang, alignments)

    print(f"[*] Сохранение результата...")
    with open(args.output, 'w', encoding='utf-8') as f:
        json.dump(multilang_result, f, ensure_ascii=False, indent=2)

    print(f"[OK] Многоязычное выравнивание: {args.output}")
    print(f"    Версия: {multilang_result['version']}")
    print(f"    Глав: {multilang_result['n_chapters']}")


if __name__ == "__main__":
    main()
