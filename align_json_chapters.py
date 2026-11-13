#!/usr/bin/env python3
"""
align_json_chapters.py — Выравнивание отпарсенных книг по главам.
"""

import sys
import json
import argparse
from pathlib import Path

for _stream in (sys.stdout, sys.stderr):
    try:
        if _stream.encoding and "utf-8" not in _stream.encoding.lower():
            _stream.reconfigure(encoding="utf-8", errors="replace")
    except (AttributeError, ValueError):
        pass


def load_parsed_json(path):
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    if "chapters" not in data or not isinstance(data["chapters"], list):
        raise ValueError(f"File {path} missing chapters array")
    return data


def format_chapter(chapter_no, raw_chapter):
    paragraphs = [str(p).strip() for p in raw_chapter.get("paragraphs", []) if str(p).strip()]
    
    return {
        "chapter_no": chapter_no,
        "title": str(raw_chapter.get("title", "")).strip(),
        "page_start": raw_chapter.get("page_start", 0),
        "page_end": raw_chapter.get("page_end", 0),
        "toc_level": raw_chapter.get("toc_level", 0),
        "source": raw_chapter.get("source", "aligned"),
        "parents": raw_chapter.get("parents", []),
        "dateline": raw_chapter.get("dateline", ""),
        "epigraph": raw_chapter.get("epigraph", ""),
        "epigraph_source": raw_chapter.get("epigraph_source", ""),
        "paragraphs": paragraphs,
        "text": "\n\n".join(paragraphs).strip()
    }


def align_chapters(src_data, tgt_data):
    src_ch = src_data["chapters"]
    tgt_ch = tgt_data["chapters"]

    out_src_chapters = []
    out_tgt_chapters = []

    min_len = min(len(src_ch), len(tgt_ch))
    
    if len(src_ch) != len(tgt_ch):
        print(f"[WARN] Chapter count mismatch!")
        print(f"       Source: {len(src_ch)} | Target: {len(tgt_ch)}")
        print(f"       Aligning first {min_len} chapters.")

    for idx in range(min_len):
        ch_num = idx + 1
        out_src_chapters.append(format_chapter(ch_num, src_ch[idx]))
        out_tgt_chapters.append(format_chapter(ch_num, tgt_ch[idx]))

    src_result = {
        "meta": src_data.get("meta", {}),
        "structure": src_data.get("structure", []),
        "chapters": out_src_chapters,
        "diagnostics": src_data.get("diagnostics", {})
    }

    tgt_result = {
        "meta": tgt_data.get("meta", {}),
        "structure": tgt_data.get("structure", []),
        "chapters": out_tgt_chapters,
        "diagnostics": tgt_data.get("diagnostics", {})
    }

    return src_result, tgt_result


def main():
    parser = argparse.ArgumentParser(description="Align JSON books by chapters")
    parser.add_argument("src_json", help="Source book JSON")
    parser.add_argument("tgt_json", help="Target book JSON")
    parser.add_argument("-o1", "--out-src", default="src_aligned.json", help="Output source")
    parser.add_argument("-o2", "--out-tgt", default="tgt_aligned.json", help="Output target")
    
    args = parser.parse_args()

    print(f"Loading {args.src_json} and {args.tgt_json}...")
    src_data = load_parsed_json(args.src_json)
    tgt_data = load_parsed_json(args.tgt_json)

    src_aligned, tgt_aligned = align_chapters(src_data, tgt_data)

    Path(args.out_src).write_text(json.dumps(src_aligned, ensure_ascii=False, indent=2), encoding="utf-8")
    Path(args.out_tgt).write_text(json.dumps(tgt_aligned, ensure_ascii=False, indent=2), encoding="utf-8")

    print("\nOK! Files created for bl_sentence_aligner:")
    print(f"  1. {args.out_src} ({len(src_aligned['chapters'])} chapters)")
    print(f"  2. {args.out_tgt} ({len(tgt_aligned['chapters'])} chapters)")


if __name__ == "__main__":
    main()
