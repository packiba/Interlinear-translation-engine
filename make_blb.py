#!/usr/bin/env python3
"""
One-command pipeline: PDF + PDF → BLB file.
Usage: python make_blb.py src.pdf tgt.pdf -o book.blb
"""

import json
import sys
import argparse
import subprocess
import tempfile
from pathlib import Path

for _stream in (sys.stdout, sys.stderr):
    try:
        if _stream.encoding and "utf-8" not in _stream.encoding.lower():
            _stream.reconfigure(encoding="utf-8", errors="replace")
    except (AttributeError, ValueError):
        pass


def run_command(cmd, description):
    print(f"[*] {description}...", flush=True)
    result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
    if result.returncode != 0:
        print(f"ERROR: {description} failed")
        print(f"STDOUT: {result.stdout}")
        print(f"STDERR: {result.stderr}")
        return False
    return True


def main():
    parser = argparse.ArgumentParser(description="Create BLB from two PDFs")
    parser.add_argument("src_pdf", help="Source PDF")
    parser.add_argument("tgt_pdf", help="Target PDF")
    parser.add_argument("-o", "--output", default="book.blb", help="Output BLB file")
    parser.add_argument("--keep", action="store_true", help="Keep intermediate files")
    
    args = parser.parse_args()
    
    src_pdf = Path(args.src_pdf).resolve()
    tgt_pdf = Path(args.tgt_pdf).resolve()
    output_blb = Path(args.output).resolve()
    
    if not src_pdf.exists() or not tgt_pdf.exists():
        print("ERROR: PDF files not found")
        return 1
    
    # Create temporary directory for intermediate files
    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)
        
        src_json = tmpdir / "src.json"
        tgt_json = tmpdir / "tgt.json"
        src_aligned = tmpdir / "src_aligned.json"
        tgt_aligned = tmpdir / "tgt_aligned.json"
        aligned_sents = tmpdir / "aligned_sentences.json"
        
        print(f"Pipeline: {src_pdf.name} + {tgt_pdf.name} -> {output_blb.name}")
        print()
        
        # Step 1: Parse PDFs
        if not run_command(f'python bl_book_parser.py "{src_pdf}" -o "{src_json}"', "Parsing source PDF"):
            return 1
        if not run_command(f'python bl_book_parser.py "{tgt_pdf}" -o "{tgt_json}"', "Parsing target PDF"):
            return 1
        
        # Step 2: Align chapters
        if not run_command(
            f'python align_json_chapters.py "{src_json}" "{tgt_json}" -o1 "{src_aligned}" -o2 "{tgt_aligned}"',
            "Aligning chapters"):
            return 1
        
        # Step 3: Align sentences (GPU)
        if not run_command(
            f'python bl_sentence_aligner.py "{src_aligned}" "{tgt_aligned}" -o "{aligned_sents}"',
            "Aligning sentences (GPU)"):
            return 1
        
        # Step 4: Create BLB
        if not run_command(
            f'python create_blb.py "{aligned_sents}" "{src_json}" "{src_pdf}" --tgt-json "{tgt_json}" -o "{output_blb}"',
            "Creating BLB file"):
            return 1
        
        print()
        print(f"[OK] BLB file created: {output_blb}")
        print(f"     Size: {output_blb.stat().st_size / 1024 / 1024:.1f} MB")
        
        # Copy intermediate files if requested
        if args.keep:
            for f in [src_json, tgt_json, src_aligned, tgt_aligned, aligned_sents]:
                import shutil
                dest = output_blb.parent / f.name
                shutil.copy2(f, dest)
                print(f"     Kept: {dest.name}")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
