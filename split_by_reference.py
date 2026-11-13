#!/usr/bin/env python3
"""
Делит книгу без разметки глав на главы по эталонной книге на другом языке.

Бывает, что в издании нет ни оглавления, ни номеров глав (сплошной текст), и парсер
возвращает одну главу. Выравниватель такую главу с сотней глав другого издания не
сопоставит. Здесь начало каждой главы эталона ищется среди абзацев целевой книги по смыслу
(LaBSE); найденные места должны идти по порядку, а положение подсказывает доля объёма.
По ним целевая книга режется на главы, заголовки берутся у эталона.

  python split_by_reference.py ru.json en.json -o en_chapters.json --skip-ref-chapters 1

Печатает сходство найденных начал и отношение объёмов глав: резкие выбросы означают,
что границу нашли неверно.
"""

import sys
import json
import argparse
from pathlib import Path

import numpy as np

for _stream in (sys.stdout, sys.stderr):
    try:
        if _stream.encoding and "utf-8" not in _stream.encoding.lower():
            _stream.reconfigure(encoding="utf-8", errors="replace")
    except (AttributeError, ValueError):
        pass

LOW_SIM = 0.45              # ниже — начало главы найдено неуверенно
RATIO_RANGE = (0.55, 1.8)   # допустимое отношение объёма главы цели к эталону


def load(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def head_text(paras, n=2, limit=600):
    return " ".join(paras[:n])[:limit]


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("reference", help="разобранная книга с главами (эталон)")
    ap.add_argument("target", help="разобранная книга без глав")
    ap.add_argument("-o", "--output", required=True)
    ap.add_argument("--skip-ref-chapters", default="",
                    help="chapter_no эталона, которых нет в цели (титул и т.п.): 1,2")
    ap.add_argument("--alpha", type=float, default=1.0, help="вес подсказки по доле объёма")
    ap.add_argument("--model", default="sentence-transformers/LaBSE")
    args = ap.parse_args()

    skip = {int(x) for x in args.skip_ref_chapters.split(",") if x.strip()}
    ref, tgt = load(args.reference), load(args.target)
    ref_chs = [c for c in ref["chapters"] if c["chapter_no"] not in skip and c["paragraphs"]]
    paras = [p for c in tgt["chapters"] for p in c["paragraphs"]]
    K, P = len(ref_chs), len(paras)
    print(f"эталон: {K} глав, цель: {P} абзацев")

    from sentence_transformers import SentenceTransformer
    import torch
    model = SentenceTransformer(args.model, device="cuda" if torch.cuda.is_available() else "cpu")

    def enc(texts):
        return model.encode(texts, normalize_embeddings=True, batch_size=64, show_progress_bar=False)

    queries = enc([head_text(c["paragraphs"]) for c in ref_chs])
    windows = enc([head_text(paras[j:j + 2]) for j in range(P)])
    sim = queries @ windows.T                                         # K x P

    ref_len = np.array([sum(len(p) for p in c["paragraphs"]) for c in ref_chs], dtype=float)
    expected = np.cumsum(np.concatenate([[0.0], ref_len]))[:-1] / ref_len.sum()
    para_len = np.array([len(p) for p in paras], dtype=float)
    share = np.cumsum(np.concatenate([[0.0], para_len]))[:-1] / para_len.sum()

    # динамическое программирование по строго возрастающим позициям
    score = sim - args.alpha * np.abs(share[None, :] - expected[:, None])
    dp = np.full((K, P), -1e9)
    parent = np.zeros((K, P), dtype=int)
    dp[0] = score[0]
    idx = np.arange(P)
    for k in range(1, K):
        run = np.maximum.accumulate(dp[k - 1])
        arg = np.maximum.accumulate(np.where(dp[k - 1] == run, idx, 0))
        dp[k, 1:] = score[k, 1:] + run[:-1]
        parent[k, 1:] = arg[:-1]
    pos = [int(np.argmax(dp[K - 1]))]
    for k in range(K - 1, 0, -1):
        pos.append(int(parent[k, pos[-1]]))
    pos = pos[::-1]

    picked = np.array([sim[k, p] for k, p in enumerate(pos)])
    ends = pos[1:] + [P]
    ratios = np.array([para_len[pos[k]:ends[k]].sum() / max(1.0, ref_len[k]) for k in range(K)])
    low = [k for k in range(K) if picked[k] < LOW_SIM]
    odd = [k for k in range(K) if not RATIO_RANGE[0] <= ratios[k] <= RATIO_RANGE[1]]
    print(f"сходство начал: среднее {picked.mean():.3f}, минимум {picked.min():.3f}, ниже {LOW_SIM}: {len(low)}")
    print(f"отношение объёмов цель/эталон: медиана {np.median(ratios):.2f}, вне {RATIO_RANGE}: {len(odd)}")
    for k in sorted(set(low) | set(odd))[:12]:
        print(f"   глава {ref_chs[k]['title']!r}: sim {picked[k]:.2f}, объём x{ratios[k]:.2f}, абзац {pos[k]}")

    chapters = []

    def add(title, chunk):
        chapters.append({
            "chapter_no": len(chapters) + 1, "title": title, "page_start": 0, "page_end": 0, "toc_level": 1,
            "source": "by-reference", "parents": [], "dateline": "", "epigraph": "", "epigraph_source": "",
            "paragraphs": chunk, "text": "\n\n".join(chunk)})

    if pos[0] > 0:
        add("Начало", paras[:pos[0]])
    for k in range(K):
        add(ref_chs[k]["title"], paras[pos[k]:ends[k]])
    out = {"meta": tgt.get("meta", {}), "structure": [], "chapters": chapters,
           "diagnostics": {"strategy": "by-reference", "reference": Path(args.reference).name,
                           "mean_sim": float(picked.mean()), "low_sim": len(low), "odd_size": len(odd)}}
    Path(args.output).write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[OK] {args.output}: {len(chapters)} глав")


if __name__ == "__main__":
    main()
