#!/usr/bin/env python3
import json
import base64
import argparse
import sys
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


def create_blb(aligned_json_path, src_json_path, tgt_json_path, src_pdf_path, output_path):
    with open(aligned_json_path, 'r', encoding='utf-8') as f:
        aligned_data = json.load(f)
    
    # Try to load source JSON for metadata
    try:
        with open(src_json_path, 'r', encoding='utf-8') as f:
            src_data = json.load(f)
            src_meta = src_data.get("meta", {})
    except:
        src_meta = {}
    
    print("Extracting cover...")
    cover_info = extract_cover_from_pdf(src_pdf_path, page_num=0, dpi=150)
    cover_base64 = base64.b64encode(cover_info["data"]).decode('utf-8')
    
    blb = {
        "blb_version": 1,
        "book": {
            "title_src": src_meta.get("title", "Unknown"),
            "title_tgt": src_meta.get("title", "Unknown"),
            "author": src_meta.get("author", "Unknown"),
            "lang_src": src_meta.get("language", "ru"),
            "lang_tgt": "bg"
        },
        "cover": {
            "mime": "image/png",
            "width": cover_info["width"],
            "height": cover_info["height"],
            "data_base64": cover_base64
        },
        "chapters": aligned_data.get("chapters", [])
    }
    
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(blb, f, ensure_ascii=False)
    
    print(f"[OK] BLB file created: {output_path}")
    print(f"  Chapters: {len(blb['chapters'])}")
    print(f"  Cover: {cover_info['width']}x{cover_info['height']} PNG")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Create BLB file")
    parser.add_argument("aligned_json", help="JSON with aligned sentences")
    parser.add_argument("src_json", help="Source parsed JSON (for metadata)")
    parser.add_argument("src_pdf", help="Source PDF (for cover)")
    parser.add_argument("-o", "--output", default="book.blb", help="Output BLB file")
    
    args = parser.parse_args()
    
    create_blb(args.aligned_json, args.src_json, None, args.src_pdf, args.output)
