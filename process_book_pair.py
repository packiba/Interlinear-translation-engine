#!/usr/bin/env python3
"""
Автоматический конвейер: папка с двумя PDF -> BLB файл
Использование: python process_book_pair.py /path/to/folder/with/pdfs
Все промежуточные файлы остаются в папке, готовый BLB тоже там же.
"""

import sys
import json
import subprocess
from pathlib import Path

for _stream in (sys.stdout, sys.stderr):
    try:
        if _stream.encoding and "utf-8" not in _stream.encoding.lower():
            _stream.reconfigure(encoding="utf-8", errors="replace")
    except (AttributeError, ValueError):
        pass


def find_pdf_files(folder):
    """Находит PDF файлы в папке (сортирует по размеру - больший = русский)"""
    pdfs = sorted(folder.glob("*.pdf"), key=lambda x: x.stat().st_size, reverse=True)
    if len(pdfs) < 2:
        print(f"ERROR: Found only {len(pdfs)} PDF file(s) in {folder}")
        print("Expected at least 2 PDF files")
        return None, None
    
    # Предполагаем: больший файл = исходный язык (обычно русский), меньший = целевой
    src_pdf = pdfs[0]  # Больший
    tgt_pdf = pdfs[1]  # Меньший
    
    return src_pdf, tgt_pdf


def run_cmd(cmd, description, cwd):
    """Выполняет команду в указанной папке"""
    print(f"\n[*] {description}...")
    result = subprocess.run(
        cmd,
        shell=True,
        cwd=cwd,
        capture_output=True,
        text=True
    )
    
    if result.returncode != 0:
        print(f"ERROR: {description} failed!")
        print(f"STDERR: {result.stderr}")
        return False
    
    return True


def main():
    if len(sys.argv) < 2:
        print("Usage: python process_book_pair.py /path/to/folder/with/pdfs [--keep-intermediate]")
        print("\nExample:")
        print("  python process_book_pair.py D:\my_books\book1")
        print("\nThe script will:")
        print("  1. Find two PDF files in the folder")
        print("  2. Parse them to JSON")
        print("  3. Align chapters")
        print("  4. Run GPU sentence alignment")
        print("  5. Create BLB file")
        print("  6. Clean up intermediate files (unless --keep-intermediate is used)")
        return 1
    
    folder = Path(sys.argv[1]).resolve()
    keep_intermediate = "--keep-intermediate" in sys.argv
    
    if not folder.exists():
        print(f"ERROR: Folder not found: {folder}")
        return 1
    
    # Находим PDF файлы
    src_pdf, tgt_pdf = find_pdf_files(folder)
    if src_pdf is None:
        return 1
    
    print("=" * 70)
    print("BILINGUAL BOOK PIPELINE - AUTOMATIC MODE")
    print("=" * 70)
    print(f"\nFolder: {folder}")
    print(f"\nDetected PDFs:")
    print(f"  Source: {src_pdf.name} ({src_pdf.stat().st_size / 1024 / 1024:.1f} MB)")
    print(f"  Target: {tgt_pdf.name} ({tgt_pdf.stat().st_size / 1024 / 1024:.1f} MB)")
    
    # Генерируем имена файлов
    prefix = folder.name.lower().replace(" ", "_")
    src_json = folder / f"{prefix}_src.json"
    tgt_json = folder / f"{prefix}_tgt.json"
    src_aligned = folder / f"{prefix}_src_aligned.json"
    tgt_aligned = folder / f"{prefix}_tgt_aligned.json"
    aligned_sents = folder / f"{prefix}_aligned_sentences.json"
    output_blb = folder / f"{prefix}_final.blb"
    
    # Папка парсера (где находятся скрипты)
    script_dir = Path(__file__).parent
    
    # ЭТАП 1: Парсинг
    if not run_cmd(
        f'python "{script_dir / "bl_book_parser.py"}" "{src_pdf}" -o "{src_json}" -v',
        "Parsing source PDF",
        folder
    ):
        return 1
    
    if not run_cmd(
        f'python "{script_dir / "bl_book_parser.py"}" "{tgt_pdf}" -o "{tgt_json}" -v',
        "Parsing target PDF",
        folder
    ):
        return 1
    
    # ЭТАП 2: Выравнивание по главам
    if not run_cmd(
        f'python "{script_dir / "align_json_chapters.py"}" "{src_json}" "{tgt_json}" -o1 "{src_aligned}" -o2 "{tgt_aligned}"',
        "Aligning chapters",
        folder
    ):
        return 1
    
    # ЭТАП 3: GPU выравнивание предложений
    if not run_cmd(
        f'python "{script_dir / "bl_sentence_aligner.py"}" "{src_aligned}" "{tgt_aligned}" -o "{aligned_sents}"',
        "Aligning sentences (GPU)",
        folder
    ):
        return 1
    
    # ЭТАП 4: Создание BLB
    if not run_cmd(
        f'python "{script_dir / "create_blb.py"}" "{aligned_sents}" "{src_json}" "{src_pdf}" --tgt-json "{tgt_json}" -o "{output_blb}"',
        "Creating BLB file",
        folder
    ):
        return 1
    
    # Очистка промежуточных файлов
    if not keep_intermediate:
        print("\n[*] Cleaning up intermediate files...")
        for f in [src_json, tgt_json, src_aligned, tgt_aligned, aligned_sents]:
            if f.exists():
                f.unlink()
                print(f"    Removed: {f.name}")
    
    # Финальный отчёт
    print("\n" + "=" * 70)
    print("SUCCESS!")
    print("=" * 70)
    
    with open(output_blb, 'r', encoding='utf-8') as f:
        blb = json.load(f)
    
    print(f"\nOutput file: {output_blb.name}")
    print(f"Size: {output_blb.stat().st_size / 1024 / 1024:.1f} MB")
    print(f"\nBook info:")
    print(f"  Chapters: {len(blb['chapters'])}")
    print(f"  Sentence pairs: {sum(len(c['pairs']) for c in blb['chapters'])}")
    print(f"  Languages: {blb['book']['lang_src']} <-> {blb['book']['lang_tgt']}")
    
    if not keep_intermediate:
        print(f"\nIntermediate files: CLEANED")
    else:
        print(f"\nIntermediate files: KEPT (use --keep-intermediate to keep them)")
    
    print(f"\nReady for Billing Reader!")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
