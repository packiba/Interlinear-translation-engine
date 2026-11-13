#!/usr/bin/env python3

"""
JSON -> aligned bilingual book JSON. Stage 2 of the pipeline.

Input: two book-parser JSON files.
Output: exact final schema requested by the user:
{meta: {...}, chapters: [{pair_num, title_src, title_tgt, pairs: [{src,tgt}, ...]}]}

Embeddings and SaT are explicitly forced to CUDA. NumPy DP/QC remains on CPU;
this is intentional because the dense neural work is the expensive GPU part.
"""


import os
import shutil
import subprocess
import time
os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")
os.environ.setdefault("TRANSFORMERS_NO_ADVISORY_WARNINGS", "1")
os.environ.setdefault("TRANSFORMERS_NO_TF", "1")
os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")

import re
import sys
import json
import copy
import difflib
import unicodedata
import statistics
import inspect
from pathlib import Path
from collections import Counter, defaultdict

import numpy as np

try:
    from bertalign import Bertalign
except Exception:            # без bertalign работает движок "dp"
    Bertalign = None

try:
    from sentence_transformers import SentenceTransformer
except ImportError:
    SentenceTransformer = None

# Заглушить шум библиотек (transformers/hf) — оставить только свой прогресс.
import logging
logging.disable(logging.WARNING)
try:
    from transformers import logging as hf_logging
    hf_logging.set_verbosity_error()
except Exception:
    pass

for _stream in (sys.stdout, sys.stderr):
    try:
        if _stream.encoding and "utf-8" not in _stream.encoding.lower():
            _stream.reconfigure(encoding="utf-8", errors="replace")
    except (AttributeError, ValueError):
        pass

def convert_numpy(obj):
    if isinstance(obj, np.generic):
        return obj.item()
    raise TypeError



def _norm_text(s):
    s = str(s).lower()
    s = _fold_latin(s)
    s = re.sub(r"[^\w\s]|_", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s

# Эмбеддинги и SaT
# ------------------------------------------------------------
# Модель эмбеддингов: LaBSE по умолчанию. Для проверки альтернатив
#   set BL_EMBED_MODEL=BAAI/bge-m3
# (модели с префиксами запросов, напр. e5, требуют BL_EMBED_PREFIX; внутри
# Bertalign префикс не применяется, поэтому для него лучше bge-m3/LaBSE).
EMBED_MODEL_NAME = os.getenv("BL_EMBED_MODEL", "sentence-transformers/LaBSE")
EMBED_PREFIX = os.getenv("BL_EMBED_PREFIX", "")
# Движки-кандидаты для выравнивания главы: "bertalign" и/или "dp".
ENGINES = tuple(e for e in os.getenv("BL_ENGINES", "dp")
                .replace(" ", "").split(",") if e)

# Подробность логов: BL_VERBOSE=0 тихо, 1 основное (по умолчанию),
# 2 подробно (каждый кандидат, окно переалайнинга, худшие пары).
LOG_LEVEL = int(os.getenv("BL_VERBOSE", "1"))


def _log(level, msg=""):
    if LOG_LEVEL >= level:
        print(msg, flush=True)


def _dt(t0):
    return f"{time.time() - t0:.1f}с"


def _preview(s, n=70):
    s = " ".join(str(s).split())
    return s if len(s) <= n else s[:n - 1] + "…"


# Счётчики работы разбивщика предложений (для логов).
_SPLIT_STATS = Counter()

# Загрузчики модели эмбеддингов/SaT-сплиттера и _encode() определены один раз,
# ниже, в блоке "Explicit GPU configuration" — пайплайн работает только на CUDA.

# ------------------------------------------------------------
# Разбивка на предложения с учётом диалога (п.1)
# ------------------------------------------------------------
_LINE_SEPS_RE = re.compile(r"[\u000b\u000c\u001c-\u001e\u0085\u2028\u2029\r\n]+")
# Реплика внутри абзаца: после конца предложения идёт тире и заглавная.
_TURN_CAND_RE = re.compile(
    r'(?<=[.!?…»”"])\s+(?=[–—―]\s?[«„"“¿¡]*([^\W\d_]))')


def _turn_split(text):
    """Режет абзац перед репликой «— Заглавная...». Заглавность проверяется
    через str.isupper(), поэтому работает для кириллицы, латиницы с диакритикой
    (—Él, —Ñoño), испанских «—¿Qué...?». Строчная после тире — это
    авторская ремарка («—¿Vamos? —preguntó»), её не отделяем."""
    out, last = [], 0
    for m in _TURN_CAND_RE.finditer(text):
        if m.group(1).isupper():
            out.append(text[last:m.start()])
            last = m.end()
    out.append(text[last:])
    return out
_INNER_END_RE = re.compile(r"[.!?…]['»”\"]?\s+\S")
_LEAD_JUNK = " –—―-«„\"“'([.…"


def _starts_lower(s):
    s = s.lstrip(_LEAD_JUNK)
    return bool(s) and s[0].islower()


def _is_stub(s):
    return not re.search(r"\w", s)


def _merge_fragments(sents):
    """Предложение, начинающееся со строчной буквы (в т.ч. после тире/кавычки)
    или состоящее из одних знаков, — огрызок: приклеиваем к предыдущему."""
    out = []
    for s in sents:
        s = s.strip()
        if not s:
            continue
        if out and (_is_stub(s) or _starts_lower(s)):
            out[-1] = out[-1] + " " + s
        else:
            out.append(s)
    return out


def split_sentences_smart(text, base_split):
    """Один абзац -> предложения: реплики режутся детерминированно, остальное
    отдаётся base_split (SaT). Все переводы строк/разделители (в т.ч. U+2028),
    из-за которых bertalign.splitlines() дробил бы предложения, убираются."""
    text = _LINE_SEPS_RE.sub(" ", text)
    text = re.sub(r"\s+", " ", text).strip()
    if not text:
        return []
    sents = []
    chunks = _turn_split(text)
    _SPLIT_STATS["turn_splits"] += max(0, len(chunks) - 1)
    for chunk in chunks:
        chunk = chunk.strip()
        if not chunk:
            continue
        if _INNER_END_RE.search(chunk):
            _SPLIT_STATS["base_calls"] += 1
            parts = [p.strip() for p in (base_split(chunk) or [chunk])]
            sents.extend(p for p in parts if p)
        else:
            _SPLIT_STATS["base_skipped"] += 1
            sents.append(chunk)       # внутри нет конца предложения
    merged = _merge_fragments(sents)
    _SPLIT_STATS["fragment_merges"] += len(sents) - len(merged)
    return merged


def split_paragraph_sentences(paras, base_split):
    """Абзацы -> (предложения, флаги «первое предложение абзаца»)."""
    sents, starts = [], []
    for p in paras:
        ss = split_sentences_smart(p, base_split)
        for k, s in enumerate(ss):
            sents.append(s)
            starts.append(k == 0)
    return sents, starts


def _squash(s):
    return re.sub(r"\s+", "", s)


def _lead_lower(s):
    s = s.lstrip(_LEAD_JUNK + "¿¡")
    return bool(s) and s[0].isalpha() and s[0].islower()


def _ends_sentence(s):
    return bool(re.search(r"[.!?…»”\"“)]\s*$", s))


def check_coverage(paras, sents):
    """Сверка «что было в абзацах» == «что ушло в предложения» (без пробелов).
    Возвращает (ok, |разница символов|)."""
    a, b = _squash("".join(paras)), _squash("".join(sents))
    return (a == b), abs(len(a) - len(b))


# ------------------------------------------------------------
# Якоря, не зависящие от эмбеддингов и от пары языков (п.4)
# ------------------------------------------------------------
# Имя собственное в кириллице и в латинице приводится к одному «скелету»:
# транслитерация -> классы согласных (Гарри / Harry / Хари -> "kr"). Поэтому
# якоря работают для любой пары скриптов (ru-bg, ru-en, ru-es, bg-en, en-es...).
_TOKEN_RE = re.compile(r"[^\W\d_]+")
_NUM_RE = re.compile(r"\d+")
ANCHOR_MAX_DF = 15     # слишком частые якоря (Артур) места не определяют
_CYR2LAT = {
    "а": "a", "б": "b", "в": "v", "г": "g", "д": "d", "е": "e", "ё": "e",
    "ж": "zh", "з": "z", "и": "i", "й": "", "к": "k", "л": "l", "м": "m",
    "н": "n", "о": "o", "п": "p", "р": "r", "с": "s", "т": "t", "у": "u",
    "ф": "f", "х": "h", "ц": "ts", "ч": "ch", "ш": "sh", "щ": "sht",
    "ъ": "", "ы": "i", "ь": "", "э": "e", "ю": "ju", "я": "ja", "ѝ": "i",
    "і": "i", "ї": "ji", "є": "je", "ґ": "g", "ў": "u", "ј": "", "љ": "l",
    "њ": "n", "ћ": "c", "ђ": "d", "џ": "dzh",
}
# классы согласных: близкие по звучанию при транслитерации сливаются
_CONS_CLASS = {
    "b": "p", "p": "p", "f": "p", "v": "p", "w": "p",
    "g": "k", "k": "k", "c": "k", "q": "k", "x": "k", "h": "k",
    "d": "t", "t": "t", "s": "s", "z": "s",
    "l": "l", "m": "m", "n": "n", "r": "r",
}


def _fold_latin(word):
    """Убирает диакритику: Ramón -> Ramon, Łódź -> Lodz (где возможно)."""
    w = unicodedata.normalize("NFKD", word)
    return "".join(ch for ch in w if not unicodedata.combining(ch))


def _skeleton(word):
    """Согласный скелет слова в любой письменности: Монтанелли / Montanelli ->
    «mtnl», Гарри / Harry / Хари -> «kr», Сейди / Sadie -> «s t»."""
    lat = []
    for ch in word.lower():
        lat.append(_CYR2LAT.get(ch, ch))
    lat = _fold_latin("".join(lat))
    sk = "".join(_CONS_CLASS.get(c, "") for c in lat)
    return re.sub(r"(.)\1+", r"\1", sk)


def _is_latin_word(w):
    return all(ch.isascii() or "LATIN" in unicodedata.name(ch, "") for ch in w)


def extract_anchors(text):
    keys = set()
    for w in _NUM_RE.findall(text):
        keys.add("n:" + w)
    toks = list(_TOKEN_RE.finditer(text))
    n_cyr = sum(1 for m in toks if "\u0400" <= m.group()[0] <= "\u04ff")
    for m in toks:
        w = m.group()
        if len(w) < 3:
            continue
        lat = _is_latin_word(w)
        j = m.start() - 1
        while j >= 0 and text[j].isspace():
            j -= 1
        mid = j >= 0 and (text[j].isalnum() or text[j] in ",;:")
        if lat and len(w) >= 6:
            keys.add("l:" + _fold_latin(w).lower())      # редкие слова/бренды
        # имя: заглавное слово не в начале предложения; либо латинское слово
        # внутри кириллического предложения (бренд, название)
        if (w[0].isupper() and mid) or (lat and n_cyr > len(toks) // 2):
            sk = _skeleton(w)
            if len(sk) >= 2:
                keys.add("p:" + sk[:3])   # 3 согласных терпят падежные окончания
    return keys


def anchor_overlap(a, b):
    """Доля общих якорей относительно меньшего набора, 0..1."""
    if not a or not b:
        return 0.0
    return len(a & b) / max(1, min(len(a), len(b)))


def build_anchor_matrix(src_sents, tgt_sents, max_df=ANCHOR_MAX_DF):
    """M[i, j] = число общих редких якорей предложений src i и tgt j."""
    n, m = len(src_sents), len(tgt_sents)
    A_s = [extract_anchors(s) for s in src_sents]
    A_t = [extract_anchors(s) for s in tgt_sents]
    df_s = Counter(k for a in A_s for k in a)
    df_t = Counter(k for a in A_t for k in a)
    inv_t = defaultdict(list)
    for j, a in enumerate(A_t):
        for k in a:
            inv_t[k].append(j)
    M = np.zeros((n, m), dtype=np.float32)
    for i, a in enumerate(A_s):
        for k in a:
            if df_s[k] <= max_df and 0 < df_t.get(k, 0) <= max_df:
                M[i, inv_t[k]] += 1.0
    na_s = np.array([sum(1 for k in a if df_s[k] <= max_df) for a in A_s], float)
    na_t = np.array([sum(1 for k in a if df_t[k] <= max_df) for a in A_t], float)
    return M, na_s, na_t, A_s, A_t


# ------------------------------------------------------------
# Целевая функция и глобальный DP (п.3, п.4)
# ------------------------------------------------------------
# Стоимость сегмента (m предложений src : k предложений tgt), в «символах»:
#   W * ((1 - sim_eff) + len_w*dev + merge_pen*(m+k-2) + para_w*x)
# W — средняя длина сегмента в символах src-масштаба (длиннее сегмент —
# весомее ошибка; сумма W по выравниванию не зависит от нарезки, поэтому
# число сегментов не поощряется), sim_eff = cos + anchor_w*бонус_якорей,
# dev in [0,1) — симметричное расхождение длины, x = -1/0/+1: границы абзаца
# совпали на обеих сторонах / ни на одной / только на одной.
# Пропуск предложения (skip) стоит skip_d*W_предложения; skip_d < типичного
# (1-sim) мусорной пары, так что DP не клеит насильно предложения без пары.
DP_PARAMS = {
    "len_w": 0.30, "merge_pen": 0.13, "skip_d": 0.80,
    "anchor_w": 0.12, "para_w": 0.04, "min_w": 12.0,
    "max_side": 5, "max_total": 7,
}
_INF = 1e18


def _unit_rows(V):
    n = np.linalg.norm(V, axis=1, keepdims=True)
    return V / np.maximum(n, 1e-9)


class AlignCtx:
    """Префиксные суммы эмбеддингов/длин/якорей для быстрого расчёта стоимости
    любого сегмента и DP по произвольному окну [i0,i1) x [j0,j1)."""

    def __init__(self, src_sents, tgt_sents, src_emb, tgt_emb,
                 src_pstart=None, tgt_pstart=None, params=None, use_anchors=True):
        self.src_sents, self.tgt_sents = list(src_sents), list(tgt_sents)
        n, m = len(self.src_sents), len(self.tgt_sents)
        self.n, self.m = n, m
        self.p = {**DP_PARAMS, **(params or {})}
        d = src_emb.shape[1]
        self.CEs = np.zeros((n + 1, d)); self.CEs[1:] = np.cumsum(src_emb, axis=0)
        self.CEt = np.zeros((m + 1, d)); self.CEt[1:] = np.cumsum(tgt_emb, axis=0)
        self.Ls = np.array([len(s) for s in self.src_sents], float)
        self.Lt = np.array([len(s) for s in self.tgt_sents], float)
        self.CLs = np.concatenate([[0.0], np.cumsum(self.Ls)])
        self.CLt = np.concatenate([[0.0], np.cumsum(self.Lt)])
        ts, tt = self.CLs[-1], self.CLt[-1]
        own = float(tt / ts) if ts > 0 and tt > 0 else 1.0
        # Коэффициент длин tgt/src. Если в главе есть текст без пары (пролог,
        # которого нет в другом издании; лишняя глава), собственный коэффициент
        # искажён и портит штрафы за длину во ВСЕЙ главе. Тогда берём
        # коэффициент книги (ratio_prior = медиана по хорошо сошедшимся главам).
        prior = self.p.get("ratio_prior")
        self.ratio_own = own
        self.ratio = prior if (prior and not 0.88 * prior <= own <= 1.12 * prior) else own
        self.ps = (np.asarray(src_pstart, bool) if src_pstart is not None
                   else np.zeros(n, bool))
        self.pt = (np.asarray(tgt_pstart, bool) if tgt_pstart is not None
                   else np.zeros(m, bool))
        self.P = None
        self.A_s = [set() for _ in range(n)]
        self.A_t = [set() for _ in range(m)]
        self.CNs = np.zeros(n + 1); self.CNt = np.zeros(m + 1)
        if use_anchors and n and m:
            M, na_s, na_t, self.A_s, self.A_t = build_anchor_matrix(
                self.src_sents, self.tgt_sents)
            P = np.zeros((n + 1, m + 1))
            P[1:, 1:] = M.cumsum(axis=0).cumsum(axis=1)
            self.P = P
            self.CNs[1:] = np.cumsum(na_s)
            self.CNt[1:] = np.cumsum(na_t)

    # ---- скалярные стоимости ------------------------------------------
    def skip_cost_src(self, i0, i1):
        return self.p["skip_d"] * 0.5 * max(self.p["min_w"], self.CLs[i1] - self.CLs[i0])

    def skip_cost_tgt(self, j0, j1):
        return self.p["skip_d"] * 0.5 * max(
            self.p["min_w"], (self.CLt[j1] - self.CLt[j0]) / self.ratio)

    def seg_sim(self, i0, i1, j0, j1):
        sv = self.CEs[i1] - self.CEs[i0]
        tv = self.CEt[j1] - self.CEt[j0]
        ns, nt = np.linalg.norm(sv), np.linalg.norm(tv)
        return float(sv @ tv / max(ns * nt, 1e-9))

    def seg_anchor_bonus(self, i0, i1, j0, j1):
        if self.P is None:
            return 0.0
        cnt = (self.P[i1, j1] - self.P[i0, j1] - self.P[i1, j0] + self.P[i0, j0])
        den = max(1.0, min(self.CNs[i1] - self.CNs[i0], self.CNt[j1] - self.CNt[j0]))
        return float(min(1.0, cnt / den))

    def seg_cost(self, i0, i1, j0, j1):
        di, dj = i1 - i0, j1 - j0
        if di == 0 and dj == 0:
            return 0.0
        if di == 0:
            return self.skip_cost_tgt(j0, j1)
        if dj == 0:
            return self.skip_cost_src(i0, i1)
        p, r = self.p, self.ratio
        sim = self.seg_sim(i0, i1, j0, j1)
        cs = self.CLs[i1] - self.CLs[i0]
        ct = self.CLt[j1] - self.CLt[j0]
        W = max(p["min_w"], 0.5 * (cs + ct / r))
        pred = cs * r
        dev = abs(ct - pred) / max(pred, ct, 1.0)
        sim_eff = sim + p["anchor_w"] * self.seg_anchor_bonus(i0, i1, j0, j1)
        sx, tx = bool(self.ps[i0]), bool(self.pt[j0])
        x = -1.0 if (sx and tx) else (1.0 if (sx or tx) else 0.0)
        d = ((1.0 - sim_eff) + p["len_w"] * dev
             + p["merge_pen"] * (di + dj - 2) + p["para_w"] * x)
        return W * d

    def total_cost(self, segs):
        return float(sum(self.seg_cost(*s) for s in segs))


def dp_align_range(ctx, i0, i1, j0, j1, allow_skip=True, progress=False):
    """Оптимальное монотонное выравнивание окна [i0,i1) x [j0,j1) -> список
    сегментов (a, b, c, d). При allow_skip возможны односторонние сегменты
    (вызывающий приклеивает их через attach_one_sided)."""
    n, m = i1 - i0, j1 - j0
    if n == 0 or m == 0:
        return [(i0, i1, j0, j1)] if (n or m) else []
    p, r = ctx.p, ctx.ratio
    K, Tmax = p["max_side"], p["max_total"]
    moves = [(a, b) for a in range(1, K + 1) for b in range(1, K + 1)
             if a + b <= Tmax and min(a, b) <= 2 and a <= n and b <= m]
    # нормированные суммы эмбеддингов tgt для каждого dj: (m+1-dj, d)
    CEt_w = ctx.CEt[j0:j1 + 1]
    TVn = {}
    for dj in {b for _, b in moves}:
        TVn[dj] = _unit_rows(CEt_w[dj:] - CEt_w[:m + 1 - dj])

    dp = np.full((n + 1, m + 1), _INF)
    bdi = np.zeros((n + 1, m + 1), np.int8)
    bdj = np.zeros((n + 1, m + 1), np.int8)
    skip_s = p["skip_d"] * 0.5 * np.maximum(p["min_w"], ctx.Ls[i0:i1])
    skip_t = p["skip_d"] * 0.5 * np.maximum(p["min_w"], ctx.Lt[j0:j1] / r)
    S = np.concatenate([[0.0], np.cumsum(skip_t)])
    use_anch = ctx.P is not None

    t_dp = time.time()
    next_mark = 0.25
    for i in range(n + 1):
        if progress and n >= 150 and i / n >= next_mark:
            _log(2, f"      DP {int(next_mark * 100)}%  (строка {i}/{n}, "
                    f"{_dt(t_dp)})")
            next_mark += 0.25
        gi = i0 + i
        row = np.full(m + 1, _INF)
        ri = np.zeros(m + 1, np.int8)
        rj = np.zeros(m + 1, np.int8)
        if i == 0:
            row[0] = 0.0
        else:
            if allow_skip:
                row = dp[i - 1] + skip_s[i - 1]
                ri[:] = 1
            for di, dj in moves:
                if di > i:
                    continue
                sv = ctx.CEs[gi] - ctx.CEs[gi - di]
                sv = sv / max(np.linalg.norm(sv), 1e-9)
                sim = TVn[dj] @ sv                          # j = dj..m
                cs = ctx.CLs[gi] - ctx.CLs[gi - di]
                ct = ctx.CLt[j0 + dj:j1 + 1] - ctx.CLt[j0:j1 + 1 - dj]
                if use_anch:
                    P = ctx.P
                    cnt = (P[gi, j0 + dj:j1 + 1] - P[gi - di, j0 + dj:j1 + 1]
                           - P[gi, j0:j1 + 1 - dj] + P[gi - di, j0:j1 + 1 - dj])
                    na_s = ctx.CNs[gi] - ctx.CNs[gi - di]
                    na_t = ctx.CNt[j0 + dj:j1 + 1] - ctx.CNt[j0:j1 + 1 - dj]
                    den = np.maximum(1.0, np.minimum(na_s, na_t))
                    sim = sim + p["anchor_w"] * np.minimum(1.0, cnt / den)
                W = np.maximum(p["min_w"], 0.5 * (cs + ct / r))
                pred = cs * r
                dev = np.abs(ct - pred) / np.maximum(np.maximum(pred, ct), 1.0)
                pt_start = ctx.pt[j0:j1 + 1 - dj]
                if ctx.ps[gi - di]:
                    x = np.where(pt_start, -1.0, 1.0)
                else:
                    x = np.where(pt_start, 1.0, 0.0)
                d = ((1.0 - sim) + p["len_w"] * dev
                     + p["merge_pen"] * (di + dj - 2) + p["para_w"] * x)
                cand = dp[i - di, 0:m + 1 - dj] + W * d
                view = row[dj:]
                better = cand < view
                if better.any():
                    view[better] = cand[better]
                    ri[dj:][better] = di
                    rj[dj:][better] = dj
        if allow_skip:
            g = np.minimum.accumulate(row - S)
            new = g + S
            skipped = new < row - 1e-9
            if skipped.any():
                row = np.where(skipped, new, row)
                ri[skipped] = 0
                rj[skipped] = 1
        dp[i] = row
        bdi[i] = ri
        bdj[i] = rj

    if dp[n, m] >= _INF / 2:
        return [(i0, i1, j0, j1)]
    segs = []
    i, j = n, m
    while i > 0 or j > 0:
        di, dj = int(bdi[i, j]), int(bdj[i, j])
        if di == 0 and dj == 0:            # недостижимо: страховка
            return [(i0, i1, j0, j1)]
        segs.append((i0 + i - di, i0 + i, j0 + j - dj, j0 + j))
        i -= di
        j -= dj
    segs.reverse()
    return segs


ATTACH_MAX_SENTS = int(os.getenv("BL_ATTACH_MAX_SENTS", "2"))
ATTACH_MAX_CHARS = int(os.getenv("BL_ATTACH_MAX_CHARS", "500"))


def _one_sided(s):
    return s[1] == s[0] or s[3] == s[2]


def attach_one_sided(segs, ctx=None, max_sents=None, max_chars=None):
    """Куски без пары (skip в DP).

    v1 приклеивал ВСЕ такие куски к соседней паре. Если в одном издании есть
    текст, которого нет в другом (пролог, лишняя глава, послесловие), весь этот
    блок оказывался в одной «паре» (в ваших файлах — до 61 000 символов против
    одного предложения). Теперь приклеиваем только короткие хвостики (до
    max_sents предложений и max_chars символов с каждой стороны); крупные
    блоки остаются односторонними сегментами и уходят в отчёт unaligned.
    BL_ATTACH_MAX_SENTS=999999 BL_ATTACH_MAX_CHARS=999999999 возвращает v1."""
    max_sents = ATTACH_MAX_SENTS if max_sents is None else max_sents
    max_chars = ATTACH_MAX_CHARS if max_chars is None else max_chars
    runs = []                                  # ("pair"|"gap", [a, b, c, d])
    for a, b, c, d in segs:
        if b == a and d == c:
            continue
        if _one_sided((a, b, c, d)):
            if runs and runs[-1][0] == "gap":  # подряд идущие skip -> один блок
                runs[-1][1][1], runs[-1][1][3] = b, d
            else:
                runs.append(("gap", [a, b, c, d]))
        else:
            runs.append(("pair", [a, b, c, d]))

    def small(g):
        a, b, c, d = g
        if b - a > max_sents or d - c > max_sents:
            return False
        if ctx is not None and (ctx.CLs[b] - ctx.CLs[a] > max_chars
                                or ctx.CLt[d] - ctx.CLt[c] > max_chars):
            return False
        return True

    out, fwd = [], None
    for kind, g in runs:
        if kind == "pair":
            if fwd is not None:                # хвостик в начале -> к первой паре
                g[0], g[2] = fwd[0], fwd[2]
                fwd = None
            out.append(g)
        elif small(g):
            if out and not _one_sided(out[-1]):
                out[-1][1], out[-1][3] = g[1], g[3]
            else:
                fwd = g
        else:                                  # крупный блок: оставляем как есть
            a, b, c, d = g
            if b > a:
                out.append([a, b, c, c])
            if d > c:
                out.append([b, b, c, d])
    if fwd is not None:                        # весь юнит — один маленький блок
        out.append(list(fwd))
    return [tuple(x) for x in out]


def count_one_sided(segs):
    return sum(1 for a, b, c, d in segs if b == a or d == c)


def check_tiling(segs, n, m):
    """Сегменты идут подряд, без дыр и перекрытий, и покрывают всё."""
    ci = cj = 0
    for a, b, c, d in segs:
        if a != ci or c != cj or b < a or d < c:
            return False
        ci, cj = b, d
    return ci == n and cj == m


# ------------------------------------------------------------
# Кандидаты: Bertalign на всю главу и глобальный DP
# ------------------------------------------------------------
DEFAULT_BERTALIGN_PARAMS = {
    "max_align": 5, "top_k": 3, "win": 5, "skip": -0.1,
    "margin": True, "len_penalty": True,
}

# Альтернативные пресеты Bertalign, которые пробуются по очереди, если
# результат с DEFAULT_BERTALIGN_PARAMS получается низкого качества.
PARAM_VARIANTS = [
    {},                                              # дефолт как есть
    {"win": 8, "max_align": 6},                       # шире окно поиска
    {"win": 3, "skip": -0.2, "max_align": 3},         # уже окно, строже штраф
    {"len_penalty": False, "skip": -0.05},            # без штрафа за длину
    {"top_k": 5, "max_align": 8, "win": 8},           # сильное 1-ко-многим
]


def _bertalign_beads(src_sents, tgt_sents, bt_params=None):
    """Bertalign на всю главу. -> список бусин (idx_src, idx_tgt) или None."""
    if Bertalign is None:
        return None
    kw = {**DEFAULT_BERTALIGN_PARAMS, **(bt_params or {}), "is_split": True}
    try:
        if (_BT_ENCODER is not None
                and "model" in inspect.signature(Bertalign.__init__).parameters):
            kw["model"] = _BT_ENCODER
    except (TypeError, ValueError):
        pass
    try:
        with open(os.devnull, "w", encoding="utf-8") as devnull:
            sys.stdout = devnull
            try:
                al = Bertalign("\n".join(src_sents), "\n".join(tgt_sents), **kw)
                al.align_sents()
            finally:
                sys.stdout = sys.__stdout__
    except Exception:
        return None
    if len(al.src_sents) != len(src_sents) or len(al.tgt_sents) != len(tgt_sents):
        return None                        # индексы не совпали с нашими
    return [([int(x) for x in b[0]], [int(x) for x in b[1]]) for b in al.result]


def beads_to_segments(ctx, beads):
    """Бусины Bertalign -> сегменты (a,b,c,d). M:N-бусины дробятся нашим DP
    без skip; одно/односторонние оставляем как есть (attach — снаружи)."""
    segs = []
    ci = cj = 0
    for si, ti in beads:
        if (si and si != list(range(ci, ci + len(si)))) or \
           (ti and ti != list(range(cj, cj + len(ti)))):
            return None
        i1, j1 = ci + len(si), cj + len(ti)
        if len(si) > 1 and len(ti) > 1 and len(si) + len(ti) > 3:
            segs.extend(dp_align_range(ctx, ci, i1, cj, j1, allow_skip=False))
        else:
            segs.append((ci, i1, cj, j1))
        ci, cj = i1, j1
    if ci != ctx.n or cj != ctx.m:
        return None
    return segs


# ------------------------------------------------------------
# Контроль качества (п.7)
# ------------------------------------------------------------
QC_SIM_THRESHOLD = 0.5        # мягкий порог sim (для коротких пар ниже)
QC_SIM_HARD = 0.30            # ниже этого sim пара подозрительна всегда
QC_LEN_RATIO_THRESHOLD = 0.25  # min(len)/max(len) ниже — структурная ошибка
QC_LEN_RATIO_SOFT = 0.55      # при sim < порога флаг только если и длины разошлись
QC_SHORT_CHARS = 40           # «короткая» пара: LaBSE на них шумит
QC_SHORT_SIM_RELAX = 0.08
QC_ANCHOR_W = 0.15            # вклад совпавших якорей в sim при оценке
QC_BAD_RATIO_OK = 0.1         # если доля плохих пар <= этого, дальше не перебираем


def _suspicion(sim, lr, ls, lt, anc=0.0, sim_thr=QC_SIM_THRESHOLD,
               lr_thr=QC_LEN_RATIO_THRESHOLD):
    """-> (bad, reason). Низкий sim сам по себе (при нормальных длинах) —
    вольный перевод, не ошибка; флаг только если sim совсем низкий, либо
    низкий sim сочетается с расхождением длин, либо длины разошлись
    настолько, что это почти наверняка неверная граница."""
    sim_adj = sim + QC_ANCHOR_W * anc
    short = max(ls, lt) < QC_SHORT_CHARS
    thr = sim_thr - (QC_SHORT_SIM_RELAX if short else 0.0)
    reasons = []
    if sim_adj < QC_SIM_HARD:
        reasons.append("low_sim")
    elif sim_adj < thr and lr < QC_LEN_RATIO_SOFT:
        reasons.append("low_sim_len")
    if lr < lr_thr and not (short and sim_adj >= thr):
        reasons.append("len_ratio")
    return bool(reasons), "+".join(reasons)


def _pair_similarity(pairs, model, max_chars=1000):
    if not pairs:
        return np.zeros(0)
    emb_src = _encode(model, [p["src"][:max_chars] for p in pairs])
    emb_tgt = _encode(model, [p["tgt"][:max_chars] for p in pairs])
    return np.sum(emb_src * emb_tgt, axis=1)


def _pair_similarity_ctx(ctx, segs, pairs, model, max_chars=1000):
    """Как _pair_similarity, но для сегментов 1:1 берёт уже посчитанный на
    шаге эмбеддинга предложений вектор (ctx.CEs/CEt — точная нормированная
    эмбеддинга того же самого предложения), а не гоняет текст пары через
    энкодер заново. На реальных книгах большинство пар — 1:1, так что это
    примерно вдвое сокращает число вызовов _encode на финальном QC-шаге.
    Заново кодируются только пары со слиянием (1:N/N:1/N:M), где эмбеддинг
    склеенного текста не равен сумме эмбеддингов отдельных предложений."""
    n = len(pairs)
    sims = np.zeros(n, dtype=np.float32)
    merged_idx = []
    for i, (a, b, c, d) in enumerate(segs):
        if b - a == 1 and d - c == 1:
            sv = ctx.CEs[b] - ctx.CEs[a]
            tv = ctx.CEt[d] - ctx.CEt[c]
            ns, nt = np.linalg.norm(sv), np.linalg.norm(tv)
            sims[i] = float(sv @ tv / max(ns * nt, 1e-9))
        else:
            merged_idx.append(i)
    if merged_idx:
        sub = [pairs[i] for i in merged_idx]
        emb_src = _encode(model, [p["src"][:max_chars] for p in sub])
        emb_tgt = _encode(model, [p["tgt"][:max_chars] for p in sub])
        sub_sims = np.sum(emb_src * emb_tgt, axis=1)
        for i, s in zip(merged_idx, sub_sims):
            sims[i] = s
    return sims


def _length_ratio(pairs):
    ratios = []
    for p in pairs:
        ls, lt = len(p["src"]), len(p["tgt"])
        ratios.append(0.0 if (ls == 0 or lt == 0) else min(ls, lt) / max(ls, lt))
    return np.array(ratios)


def assess_pairs_quality(pairs, model, sim_threshold=QC_SIM_THRESHOLD,
                          len_ratio_threshold=QC_LEN_RATIO_THRESHOLD,
                          max_chars=1000, ctx=None, segs=None):
    """Оценка пар {"src","tgt"}: cos эмбеддингов текста пары, отношение длин,
    совпадение якорей. Возвращает (per_pair, stats); per_pair[i] =
    {"sim","len_ratio","anchor","bad","reason"}.
    Если переданы ctx и segs (сегменты, из которых собраны pairs), для пар
    1:1 используются уже посчитанные эмбеддинги предложений — без повторного
    кодирования (см. _pair_similarity_ctx)."""
    if not pairs:
        return [], {"avg_sim": 1.0, "bad_ratio": 0.0, "n_bad": 0, "n": 0}
    if ctx is not None and segs is not None:
        sims = _pair_similarity_ctx(ctx, segs, pairs, model, max_chars)
    else:
        sims = _pair_similarity(pairs, model, max_chars)
    lens = _length_ratio(pairs)
    per_pair, n_bad = [], 0
    prev = None
    for p, sim, lr in zip(pairs, sims, lens):
        anc = anchor_overlap(extract_anchors(p["src"]), extract_anchors(p["tgt"]))
        bad, reason = _suspicion(float(sim), float(lr), len(p["src"]), len(p["tgt"]),
                                 anc, sim_threshold, len_ratio_threshold)
        # Пара начинается со строчной, а предыдущая закончила предложение —
        # начало предложения потеряно (чаще всего ещё в парсере: колонтитул,
        # перенос, сноска) либо разбивка рассекла его посередине.
        for side in ("src", "tgt"):
            if _lead_lower(p[side]) and (prev is None or _ends_sentence(prev[side])):
                bad, reason = True, (reason + "+" if reason else "") + "frag_" + side
        prev = p
        n_bad += int(bad)
        per_pair.append({"sim": float(sim), "len_ratio": float(lr),
                         "anchor": float(anc), "bad": bad, "reason": reason})
    n = len(pairs)
    stats = {"avg_sim": float(np.mean(sims)), "bad_ratio": n_bad / n,
             "n_bad": n_bad, "n": n}
    return per_pair, stats


def run_lengths(flags):
    """Для каждой позиции — длина серии подряд идущих True, в которую она
    входит (0 для False)."""
    out = [0] * len(flags)
    i = 0
    while i < len(flags):
        if flags[i]:
            j = i
            while j + 1 < len(flags) and flags[j + 1]:
                j += 1
            for k in range(i, j + 1):
                out[k] = j - i + 1
            i = j + 1
        else:
            i += 1
    return out


def fast_quality(ctx, segs):
    """Дешёвая оценка пар по средним эмбеддингам предложений (без повторного
    encode): нужна для поиска окон и для защиты refine.
    -> список {"sim","len_ratio","bad","good","reason"}."""
    out = []
    for a, b, c, d in segs:
        sim = ctx.seg_sim(a, b, c, d)
        ls = ctx.CLs[b] - ctx.CLs[a]
        lt = ctx.CLt[d] - ctx.CLt[c]
        lr = 0.0 if (ls == 0 or lt == 0) else min(ls, lt) / max(ls, lt)
        As = set().union(*ctx.A_s[a:b]) if b > a else set()
        At = set().union(*ctx.A_t[c:d]) if d > c else set()
        bad, reason = _suspicion(sim, lr, ls, lt, anchor_overlap(As, At))
        good = (not bad) and sim >= 0.6 and lr >= 0.6 and max(ls, lt) >= 25
        out.append({"sim": sim, "len_ratio": lr, "bad": bad, "good": good,
                    "reason": reason})
    return out


# ------------------------------------------------------------
# Дрейф и окна переалайнинга (п.5, п.6)
# ------------------------------------------------------------
DRIFT_MIN_CHARS = 400     # абсолютный порог накопленной разницы длин
DRIFT_REL = 0.03          # ... и доля от длины src главы (берётся большее)
DRIFT_LOW_FRAC = 0.3      # «вернулись к нулю»: |D| <= порог * эта доля
DRIFT_MAX_PAIRS = 80      # реальный дрейф компактен (в v2: 11-63 пар); более
                          # широкий «горб» — медленная разница в стиле перевода
REFINE_PASSES = 2
REFINE_MAX_EXPAND = 40    # на сколько пар максимум расширять окно серии
REFINE_ANCHOR_RUN = 2     # окно должно упираться в N подряд надёжных пар
REFINE_EPS = 0.005        # требуемое улучшение стоимости (доля от W окна)


def detect_drift(len_src, len_tgt):
    """Накопленная разница D_k = sum(len_tgt - ratio*len_src) по парам.
    Отрезки, где |D| превышает порог, ограничены ближайшими «возвратами к
    нулю» (|D| <= порог*DRIFT_LOW_FRAC) — там выравнивание заведомо
    синхронно. -> список окон [a, b) в индексах пар."""
    s = np.asarray(len_src, float)
    t = np.asarray(len_tgt, float)
    n = len(s)
    if n == 0 or s.sum() <= 0:
        return []
    ratio = t.sum() / s.sum()
    D = np.concatenate([[0.0], np.cumsum(t - ratio * s)])
    thr = max(DRIFT_MIN_CHARS, DRIFT_REL * s.sum())
    low = thr * DRIFT_LOW_FRAC
    wins = []
    b = 0
    while b <= n:
        if abs(D[b]) > thr:
            a = b
            while a > 0 and abs(D[a]) > low:
                a -= 1
            e = b
            while e < n and abs(D[e]) > low:
                e += 1
            if e - a <= DRIFT_MAX_PAIRS:
                wins.append((a, e))
            else:
                _log(2, f"    участок |D|>порога на парах {a}-{e} слишком широкий "
                        f"({e - a} > {DRIFT_MAX_PAIRS}) — похоже на разницу стиля, "
                        f"а не дрейф; пропущен")
            b = e + 1
        else:
            b += 1
    return wins


def _expand_to_anchors(good, a, b, need=REFINE_ANCHOR_RUN, max_expand=REFINE_MAX_EXPAND):
    """Расширяет [a,b) до N подряд надёжных пар с каждой стороны."""
    n = len(good)
    k, cnt, steps = a, 0, 0
    while k > 0 and steps < max_expand:
        k -= 1
        steps += 1
        cnt = cnt + 1 if good[k] else 0
        if cnt >= need:
            break
    a2 = k
    k, cnt, steps = b, 0, 0
    while k < n and steps < max_expand:
        cnt = cnt + 1 if good[k] else 0
        k += 1
        steps += 1
        if cnt >= need:
            break
    return a2, k


def _merge_windows(wins):
    """wins: [(a, b, метка)] -> объединённые [(a, b, "метка1+метка2")]."""
    out = []
    for a, b, tag in sorted(wins):
        if out and a <= out[-1][1]:
            pa, pb, ptag = out[-1]
            tags = ptag.split("+")
            if tag not in tags:
                tags.append(tag)
            out[-1] = (pa, max(pb, b), "+".join(tags))
        else:
            out.append((a, b, tag))
    return out


def find_suspect_windows(ctx, segs, q=None):
    """Окна [a,b,метка) для переалайнинга: (1) участки дрейфа по длинам
    (метка "drift"); (2) серии >=2 подозрительных пар подряд и одиночные
    структурные ошибки (len_ratio) — расширенные до надёжных пар-якорей
    (метка "run")."""
    q = q or fast_quality(ctx, segs)
    n = len(segs)
    ls = [ctx.CLs[b] - ctx.CLs[a] for a, b, c, d in segs]
    lt = [ctx.CLt[d] - ctx.CLt[c] for a, b, c, d in segs]
    drift = detect_drift(ls, lt)
    wins = [(max(0, a - 1), min(n, b + 1), "drift") for a, b in drift]
    flags = [x["bad"] for x in q]
    runs = run_lengths(flags)
    good = [x["good"] for x in q]
    n_runs = 0
    i = 0
    while i < n:
        if flags[i]:
            j = i
            while j + 1 < n and flags[j + 1]:
                j += 1
            structural = any("len_ratio" in q[k]["reason"] for k in range(i, j + 1))
            if runs[i] >= 2 or structural:
                a2, b2 = _expand_to_anchors(good, i, j + 1)
                wins.append((a2, b2, "run"))
                n_runs += 1
            i = j + 1
        else:
            i += 1
    _log(2, f"    поиск окон: плохих пар {sum(flags)}/{n}, серий-кандидатов "
            f"{n_runs}, участков дрейфа {len(drift)}")
    return _merge_windows(wins)


def try_realign(ctx, segs, a, b, tag=""):
    """Переалайнить пары [a,b) одним окном; принять только если стоимость
    упала и число плохих пар не выросло. -> (новые сегменты | None, info)."""
    t0 = time.time()
    old = segs[a:b]
    i0, i1 = old[0][0], old[-1][1]
    j0, j1 = old[0][2], old[-1][3]
    info = {"pair_start": a, "pair_end": b, "source": tag, "n_old": len(old),
            "sent_src": i1 - i0, "sent_tgt": j1 - j0, "accepted": False}
    if i1 == i0 or j1 == j0:
        info["status"] = "one_sided_window"
        return None, info
    raw = dp_align_range(ctx, i0, i1, j0, j1, allow_skip=True)
    info["skips_attached"] = count_one_sided(raw)
    new = attach_one_sided(raw, ctx)
    if not (new and new[0][0] == i0 and new[-1][1] == i1
            and new[0][2] == j0 and new[-1][3] == j1):
        info["status"] = "tiling_failed"
        return None, info
    cost_old = ctx.total_cost(old)
    cost_new = ctx.total_cost(new)
    W = max(1.0, ctx.CLs[i1] - ctx.CLs[i0])
    bad_old = sum(x["bad"] for x in fast_quality(ctx, old))
    bad_new = sum(x["bad"] for x in fast_quality(ctx, new))
    info.update({"n_new": len(new), "cost_old": round(cost_old / W, 4),
                 "cost_new": round(cost_new / W, 4),
                 "bad_old": bad_old, "bad_new": bad_new,
                 "seconds": round(time.time() - t0, 2)})
    if cost_new < cost_old - REFINE_EPS * W and bad_new <= bad_old:
        info["accepted"] = True
        info["status"] = "refined"
        return new, info
    if cost_new >= cost_old - REFINE_EPS * W:
        info["status"] = "rejected_cost_not_better"
    else:
        info["status"] = "rejected_more_bad_pairs"
    return None, info


def _log_window(info):
    mark = "✓ принято " if info["accepted"] else "✗ отклонено"
    if "cost_old" not in info:
        _log(1, f"    окно пар {info['pair_start']}-{info['pair_end']} "
                f"[{info['source']}]: {info['status']}")
        return
    _log(1, f"    {mark} окно пар {info['pair_start']}-{info['pair_end']} "
            f"[{info['source']}] ({info['sent_src']}:{info['sent_tgt']} предл.) "
            f"пар {info['n_old']}→{info['n_new']} | "
            f"стоимость {info['cost_old']:.3f}→{info['cost_new']:.3f} | "
            f"плохих {info['bad_old']}→{info['bad_new']} | "
            f"{info['seconds']}с"
            + ("" if info["accepted"] else f" ({info['status']})"))


def refine_alignment(ctx, segs, passes=REFINE_PASSES):
    """Итеративно переалайнивает подозрительные окна с защитой (try_realign).
    -> (сегменты, логи окон)."""
    segs = list(segs)
    logs = []
    for ps in range(1, passes + 1):
        q = fast_quality(ctx, segs)
        wins = find_suspect_windows(ctx, segs, q)
        if not wins:
            _log(1, f"  refine, проход {ps}: подозрительных окон нет")
            break
        _log(1, f"  refine, проход {ps}: окон {len(wins)}, "
                f"плохих пар {sum(x['bad'] for x in q)}/{len(segs)}")
        changed = False
        for a, b, tag in reversed(wins):
            new, info = try_realign(ctx, segs, a, b, tag)
            info["pass"] = ps
            logs.append(info)
            _log_window(info)
            if new is not None:
                segs[a:b] = new
                changed = True
        if not changed:
            _log(1, "  refine: ни одно окно не улучшило результат — стоп")
            break
    n_acc = sum(1 for x in logs if x["accepted"])
    _log(1, f"  refine итог: окон {len(logs)}, принято {n_acc}, "
            f"отклонено {len(logs) - n_acc}")
    return segs, logs


# ------------------------------------------------------------
# Выравнивание одной единицы (главы) целиком
# ------------------------------------------------------------
def _cand_summary(ctx, segs):
    q = fast_quality(ctx, segs)
    n_bad = sum(x["bad"] for x in q)
    return {"segments": len(segs), "bad": n_bad,
            "bad_ratio": n_bad / max(1, len(segs)),
            "cost": ctx.total_cost(segs) / max(1.0, ctx.CLs[-1]),
            "avg_sim": float(np.mean([x["sim"] for x in q])) if q else 1.0}


def build_candidates(ctx, src_sents, tgt_sents, engines, bt_variants):
    """-> (список (имя, сегменты), число прогнанных вариантов Bertalign)."""
    cands = []
    tried = 0
    if "dp" in engines:
        t0 = time.time()
        _log(1, f"  кандидат dp: глобальный DP {ctx.n}x{ctx.m} предложений...")
        raw = dp_align_range(ctx, 0, ctx.n, 0, ctx.m, True, progress=True)
        n_skip = count_one_sided(raw)
        segs = attach_one_sided(raw, ctx)
        if check_tiling(segs, ctx.n, ctx.m):
            cands.append(("dp", segs))
            c = _cand_summary(ctx, segs)
            _log(1, f"    dp: пар {c['segments']}, односторонних приклеено {n_skip}, "
                    f"плохих {c['bad']} ({c['bad_ratio'] * 100:.1f}%), "
                    f"avg_sim={c['avg_sim']:.3f}, cost={c['cost']:.4f} [{_dt(t0)}]")
        else:
            _log(1, "    dp: нарушено покрытие — кандидат отброшен")
    if "bertalign" in engines:
        if Bertalign is None:
            _log(1, "  кандидат bertalign: пакет bertalign не установлен — пропуск")
        else:
            for k, variant in enumerate(bt_variants):
                t0 = time.time()
                desc = ", ".join(f"{a}={b}" for a, b in variant.items()) or "дефолт"
                _log(1, f"  кандидат bertalign#{k} ({desc})...")
                beads = _bertalign_beads(src_sents, tgt_sents, variant)
                tried += 1
                if not beads:
                    _log(1, f"    bertalign#{k}: не удалось (ошибка или несовпадение "
                            f"числа предложений) [{_dt(t0)}]")
                    continue
                n_mn = sum(1 for si, ti in beads if len(si) > 1 and len(ti) > 1)
                n_one = sum(1 for si, ti in beads if not si or not ti)
                segs = beads_to_segments(ctx, beads)
                if not segs:
                    _log(1, f"    bertalign#{k}: бусины не сложились в покрытие")
                    continue
                segs = attach_one_sided(segs, ctx)
                if not check_tiling(segs, ctx.n, ctx.m):
                    _log(1, f"    bertalign#{k}: нарушено покрытие — отброшен")
                    continue
                cands.append((f"bertalign#{k}", segs))
                c = _cand_summary(ctx, segs)
                _log(1, f"    bertalign#{k}: бусин {len(beads)} (M:N {n_mn}, "
                        f"односторонних {n_one}) → пар {c['segments']}, "
                        f"плохих {c['bad']} ({c['bad_ratio'] * 100:.1f}%), "
                        f"avg_sim={c['avg_sim']:.3f}, cost={c['cost']:.4f} "
                        f"[{_dt(t0)}]")
                if c["bad_ratio"] <= QC_BAD_RATIO_OK:
                    if k + 1 < len(bt_variants):
                        _log(1, f"    качество достаточно (плохих ≤ "
                                f"{QC_BAD_RATIO_OK * 100:.0f}%) — остальные "
                                f"варианты Bertalign не нужны")
                    break
    return cands, tried


def _log_worst_pairs(pairs, per_pair, top=5):
    idx = [i for i, pp in enumerate(per_pair) if pp["bad"]]
    idx.sort(key=lambda i: (per_pair[i]["sim"] + 0.5 * per_pair[i]["len_ratio"]))
    for i in idx[:top]:
        pp = per_pair[i]
        _log(2, f"      #{i} sim={pp['sim']:.2f} lr={pp['len_ratio']:.2f} "
                f"anc={pp['anchor']:.2f} [{pp['reason']}] серия={pp.get('run_len', 0)}")
        _log(2, f"         RU: {_preview(pairs[i]['src'])}")
        _log(2, f"         BG: {_preview(pairs[i]['tgt'])}")


def align_unit(paras_src, paras_tgt, base_split, model, engines=None,
               bt_variants=None, params=None):
    """Выравнивание всей единицы (главы) на уровне предложений.
    -> (pairs, per_pair, stats, info); pairs — [{"src","tgt"}]."""
    t_all = time.time()
    engines = tuple(engines) if engines else ENGINES
    bt_variants = PARAM_VARIANTS if bt_variants is None else bt_variants

    # --- 1. разбивка на предложения --------------------------------------
    t0 = time.time()
    _SPLIT_STATS.clear()
    src_sents, src_ps = split_paragraph_sentences(paras_src, base_split)
    st_src = dict(_SPLIT_STATS)
    _SPLIT_STATS.clear()
    tgt_sents, tgt_ps = split_paragraph_sentences(paras_tgt, base_split)
    st_tgt = dict(_SPLIT_STATS)
    _log(1, f"  [1/5] предложения: RU {len(paras_src)} абз. → {len(src_sents)} предл., "
            f"BG {len(paras_tgt)} абз. → {len(tgt_sents)} предл. [{_dt(t0)}]")
    _log(1, f"        реплик выделено: RU {st_src.get('turn_splits', 0)}, "
            f"BG {st_tgt.get('turn_splits', 0)} | огрызков приклеено: "
            f"RU {st_src.get('fragment_merges', 0)}, BG {st_tgt.get('fragment_merges', 0)} "
            f"| вызовов SaT: RU {st_src.get('base_calls', 0)} "
            f"(пропущено {st_src.get('base_skipped', 0)}), "
            f"BG {st_tgt.get('base_calls', 0)} (пропущено {st_tgt.get('base_skipped', 0)})")

    info = {"n_sent_src": len(src_sents), "n_sent_tgt": len(tgt_sents),
            "engine": None, "candidates": {}, "windows": [],
            "bertalign_variants_tried": 0,
            "split_stats": {"src": st_src, "tgt": st_tgt}}
    ok_s, lost_s = check_coverage(paras_src, src_sents)
    ok_t, lost_t = check_coverage(paras_tgt, tgt_sents)
    info["coverage_ok"] = bool(ok_s and ok_t)
    info["coverage_diff_chars"] = {"src": lost_s, "tgt": lost_t}
    if info["coverage_ok"]:
        _log(1, "        покрытие текста при разбивке: OK (потерь символов нет)")
    else:
        _log(1, f"        ⚠ покрытие текста при разбивке нарушено: расхождение "
                f"символов RU {lost_s}, BG {lost_t}")
    if not src_sents or not tgt_sents:
        _log(1, "  нет предложений на одной из сторон — пропуск")
        return [], [], {"avg_sim": 1.0, "bad_ratio": 0.0, "n_bad": 0, "n": 0}, info

    # --- 2. эмбеддинги и якоря ---------------------------------------------
    t0 = time.time()
    E_s, E_t = _encode(model, src_sents), _encode(model, tgt_sents)
    _log(1, f"  [2/5] эмбеддинги: {len(src_sents) + len(tgt_sents)} предложений "
            f"[{_dt(t0)}]")
    t0 = time.time()
    ctx = AlignCtx(src_sents, tgt_sents, E_s, E_t, src_ps, tgt_ps, params)
    n_anch = int((ctx.P[-1, -1] if ctx.P is not None else 0))
    _log(1, f"        якоря (латиница/числа/имена): совпадений RU↔BG в матрице "
            f"{n_anch}, коэффициент длин BG/RU={ctx.ratio:.3f}, абзацных границ "
            f"RU {int(ctx.ps.sum())} / BG {int(ctx.pt.sum())} [{_dt(t0)}]")

    # --- 3. кандидаты ------------------------------------------------------
    _log(1, f"  [3/5] кандидаты выравнивания (движки: {', '.join(engines)})")
    cands, tried = build_candidates(ctx, src_sents, tgt_sents, engines, bt_variants)
    info["bertalign_variants_tried"] = tried
    if not cands:
        _log(1, "    ⚠ ни один кандидат не построен — вся глава одной парой")
        cands = [("single", [(0, ctx.n, 0, ctx.m)])]
    scored = [(ctx.total_cost(s) / max(1.0, ctx.CLs[-1]), name, s) for name, s in cands]
    scored.sort(key=lambda x: x[0])
    info["candidates"] = {name: round(c, 4) for c, name, _ in scored}
    _, best_name, segs = scored[0]
    info["engine"] = best_name
    info["initial_segments"] = len(segs)
    ranking = ", ".join(f"{name}={c:.4f}" for c, name, _ in scored)
    _log(1, f"        выбран «{best_name}» (стоимость на символ: {ranking})")

    # --- 4. refine -----------------------------------------------------------
    _log(1, "  [4/5] проверка дрейфа и переалайнинг подозрительных окон")
    ls = [ctx.CLs[b] - ctx.CLs[a] for a, b, c, d in segs]
    lt = [ctx.CLt[d] - ctx.CLt[c] for a, b, c, d in segs]
    ratio = ctx.ratio
    D = np.cumsum(np.array(lt) - ratio * np.array(ls)) if segs else np.zeros(1)
    _log(1, f"        накопленная разница длин BG-k·RU: max |D|={np.abs(D).max():.0f} симв., "
            f"порог дрейфа={max(DRIFT_MIN_CHARS, DRIFT_REL * ctx.CLs[-1]):.0f}")
    segs, logs = refine_alignment(ctx, segs)
    info["windows"] = logs
    if not check_tiling(segs, ctx.n, ctx.m):
        raise RuntimeError("нарушено покрытие предложений после refine")

    # --- 5. итоговый QC ---------------------------------------------------------
    t0 = time.time()
    # Односторонние сегменты — текст без пары в другом издании. В pairs они не
    # попадают (раньше приклеивались к соседней паре и портили её), но не
    # теряются: уходят в info["unaligned"] -> отчёт *_unaligned.json.
    unaligned, segs_al = [], []
    for sg in segs:
        a, b, c, d = sg
        if _one_sided(sg):
            unaligned.append({
                "after_pair": len(segs_al) - 1,
                "n_src": b - a, "n_tgt": d - c,
                "chars_src": int(ctx.CLs[b] - ctx.CLs[a]),
                "chars_tgt": int(ctx.CLt[d] - ctx.CLt[c]),
                "src": " ".join(src_sents[a:b]).strip(),
                "tgt": " ".join(tgt_sents[c:d]).strip()})
        else:
            segs_al.append(sg)
    info["unaligned"] = unaligned
    if unaligned:
        cs = sum(u["chars_src"] for u in unaligned)
        ct = sum(u["chars_tgt"] for u in unaligned)
        _log(1, f"        ⚠ без пары: блоков {len(unaligned)}, символов "
                f"src {cs} ({cs / max(1.0, ctx.CLs[-1]) * 100:.1f}%) / "
                f"tgt {ct} ({ct / max(1.0, ctx.CLt[-1]) * 100:.1f}%) — "
                f"в pairs не включены, см. unaligned-отчёт")
    segs = segs_al
    pairs = [{"src": " ".join(src_sents[a:b]).strip(),
              "tgt": " ".join(tgt_sents[c:d]).strip()} for a, b, c, d in segs]
    # Баланс текста: всё, что вошло в предложения, должно оказаться либо в
    # парах, либо в unaligned. Расхождение = баг выравнивателя (не парсера).
    kept_s = sum(len(_squash(p["src"])) for p in pairs) + \
        sum(len(_squash(u["src"])) for u in unaligned)
    kept_t = sum(len(_squash(p["tgt"])) for p in pairs) + \
        sum(len(_squash(u["tgt"])) for u in unaligned)
    want_s = len(_squash("".join(src_sents)))
    want_t = len(_squash("".join(tgt_sents)))
    info["conservation_ok"] = (kept_s == want_s and kept_t == want_t)
    if not info["conservation_ok"]:
        _log(1, f"        ✗ БАЛАНС ТЕКСТА нарушен: src {kept_s}/{want_s}, "
                f"tgt {kept_t}/{want_t}")
    per_pair, stats = assess_pairs_quality(pairs, model, ctx=ctx, segs=segs)
    runs = run_lengths([pp["bad"] for pp in per_pair])
    for pp, rl in zip(per_pair, runs):
        pp["run_len"] = rl
    reasons = Counter(r for pp in per_pair if pp["bad"] for r in pp["reason"].split("+"))
    n_in_runs = sum(1 for r in runs if r >= 3)
    _log(1, f"  [5/5] итоговый QC: пар {len(pairs)}, avg_sim={stats['avg_sim']:.3f}, "
            f"плохих {stats['n_bad']} ({stats['bad_ratio'] * 100:.1f}%) [{_dt(t0)}]")
    if stats["n_bad"]:
        _log(1, f"        причины: " + ", ".join(f"{k}={v}" for k, v in reasons.most_common())
                + f" | в сериях ≥3 подряд: {n_in_runs}, одиночных: "
                  f"{sum(1 for r in runs if r == 1)}")
        _log_worst_pairs(pairs, per_pair)
    _log(1, f"  единица выровнена за {_dt(t_all)} (движок «{best_name}»)")
    return pairs, per_pair, stats, info


# ---------------------------------------------------------------------------
# Explicit GPU configuration
# ---------------------------------------------------------------------------
try:
    import torch
except ImportError as exc:
    raise RuntimeError("PyTorch is required for GPU alignment. Install a CUDA build of torch.") from exc

if not torch.cuda.is_available():
    raise RuntimeError(
        "CUDA GPU is required for this script. torch.cuda.is_available() is False. "
        "Install a CUDA-enabled PyTorch build and verify with nvidia-smi."
    )

GPU_DEVICE = os.getenv("BL_CUDA_DEVICE", "cuda:0")
GPU_BATCH_SIZE = int(os.getenv("BL_GPU_BATCH_SIZE", "32"))
GPU_FP16 = os.getenv("BL_GPU_FP16", "1").lower() not in {"0", "false", "no", "off"}
SAT_MODEL_NAME = os.getenv("BL_SAT_MODEL", "sat-3l-sm")

try:
    torch.cuda.get_device_properties(GPU_DEVICE)
except Exception as exc:
    raise RuntimeError(f"CUDA device {GPU_DEVICE!r} is not available: {exc}") from exc


_EMBED_MODEL = None
_BT_ENCODER = None
_SAT_MODEL = None
CURRENT_CHAPTER_NO = "?"


def get_embed_model():
    global _EMBED_MODEL
    if _EMBED_MODEL is None:
        if SentenceTransformer is None:
            raise ImportError("sentence-transformers is required")
        print(f"[GPU] loading embedding model {EMBED_MODEL_NAME} on {GPU_DEVICE}")
        kwargs = {"device": GPU_DEVICE}
        if GPU_FP16:
            kwargs["model_kwargs"] = {"torch_dtype": "float16"}
        _EMBED_MODEL = SentenceTransformer(EMBED_MODEL_NAME, **kwargs)
        actual = str(_EMBED_MODEL.device)
        if not actual.startswith("cuda"):
            raise RuntimeError(f"Embedding model fell back to {actual}; CUDA is required.")
        print(f"[GPU] embeddings device: {actual}")
    return _EMBED_MODEL


def _encode(model, texts):
    texts = list(texts)
    if not texts:
        return np.zeros((0, 768), dtype=np.float32)
    if EMBED_PREFIX:
        texts = [EMBED_PREFIX + t for t in texts]
    arr = model.encode(
        texts,
        batch_size=GPU_BATCH_SIZE,
        show_progress_bar=False,
        normalize_embeddings=True,
        convert_to_numpy=True,
        device=GPU_DEVICE,
    )
    return np.asarray(arr, dtype=np.float32)


def get_sat_splitter(model_name=SAT_MODEL_NAME):
    global _SAT_MODEL
    if _SAT_MODEL is None:
        from wtpsplit import SaT
        print(f"[GPU] loading SaT model {model_name} on {GPU_DEVICE}")
        _SAT_MODEL = SaT(model_name)
        if GPU_FP16:
            _SAT_MODEL.half()
        _SAT_MODEL.to(GPU_DEVICE)
        print("[GPU] SaT device: cuda")
    def split(text):
        return [s.strip() for s in _SAT_MODEL.split(text, batch_size=GPU_BATCH_SIZE) if s.strip()]
    return split


# ---------------------------------------------------------------------------
# JSON input / chapter matching
# ---------------------------------------------------------------------------
def load_book(path):
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    chapters = data.get("chapters")
    if not isinstance(chapters, list) or not chapters:
        raise ValueError(f"{path}: missing non-empty 'chapters' list")
    for i, ch in enumerate(chapters, 1):
        if not isinstance(ch, dict):
            raise ValueError(f"{path}: chapters[{i}] is not an object")
        if not isinstance(ch.get("paragraphs"), list):
            raise ValueError(f"{path}: chapters[{i}] has no 'paragraphs' list")
    return data


def _chapter_chars(ch):
    return max(1, sum(len(str(p)) for p in ch.get("paragraphs", [])))


def _title_sim_simple(a, b):
    na, nb = _norm_text(a), _norm_text(b)
    if not na or not nb:
        return 0.0
    if na == nb:
        return 1.0
    seq = difflib.SequenceMatcher(None, na, nb).ratio()
    wa, wb = set(na.split()), set(nb.split())
    jac = len(wa & wb) / max(1, len(wa | wb))
    return max(seq, jac)


_GENERIC_TITLE_RE = re.compile(
    r"^\s*(?:unit|chapter|part|section|глава|част|часть|раздел|книга|book|capitulo|parte|kapitel|teil|chapitre|partie|capitolo)?\s*(?:\d{1,3}|[ivxlc]{1,7})\s*[.:)]?\s*$",
    re.I)


def _titles_informative(chapters):
    """False when titles are just numbers / 'Unit N' / 'Глава N': such titles say nothing
    about which chapter pairs with which (and restart in every part), so the matcher must
    not score them - otherwise merging chapters into big groups looks 'cheaper'."""
    ts = [str(c.get("title", "")).strip() for c in chapters]
    named = sum(1 for t in ts if t and not _GENERIC_TITLE_RE.match(_norm_text(t) or t))
    return named >= 0.5 * max(1, len(ts))


def _part_key(ch):
    par = ch.get("parents") or []
    return str(par[-1].get("title", "")) if par and isinstance(par[-1], dict) else ""


def _split_by_parts(chapters):
    """[(start, end)] runs of consecutive chapters that share the same part; None if no parts."""
    keys = [_part_key(c) for c in chapters]
    if not any(keys):
        return None
    runs, st = [], 0
    for i in range(1, len(keys) + 1):
        if i == len(keys) or keys[i] != keys[st]:
            runs.append((st, i))
            st = i
    return runs


CH_SAMPLE_PARAS = int(os.getenv("BL_CH_SAMPLE", "64"))   # абзацев на главу для центроида
CH_SKIP_BASE = float(os.getenv("BL_CH_SKIP", "0.70"))    # цена «у главы нет пары»
CH_W_SEM, CH_W_LEN, CH_W_MERGE = 1.0, 1.0, 0.10

_ROMAN = {"i": 1, "v": 5, "x": 10, "l": 50, "c": 100}


def _roman_to_int(w):
    tot = 0
    for k, ch in enumerate(w):
        v = _ROMAN[ch]
        tot += -v if k + 1 < len(w) and _ROMAN[w[k + 1]] > v else v
    return tot


def _title_number(title):
    """Номер главы из заголовка: «Глава 12», «Chapter 12», «CAPÍTULO 12», «XII».
    Пролог -> 0. Заголовки словами («Двайсет и втора глава») -> None: их
    сопоставляет содержание, а не номер."""
    t = _norm_text(title)
    if re.search(r"пролог|prolog|prologue", t):
        return 0
    m = re.search(r"\d+", t)
    if m:
        return int(m.group())
    for w in t.split():
        if re.fullmatch(r"[ivxlc]{1,7}", w):
            return _roman_to_int(w)
    return None


def _chapter_centroid(model, ch, k=CH_SAMPLE_PARAS):
    paras = [str(x).strip() for x in ch.get("paragraphs", []) if str(x).strip()]
    if not paras:
        return None
    if len(paras) > k:
        idx = np.linspace(0, len(paras) - 1, k).round().astype(int)
        paras = [paras[i] for i in idx]
    return _encode(model, [x[:400] for x in paras]).mean(axis=0)


def _centered_centroids(model, chs):
    """Центроиды глав, центрированные внутри книги: общий «стиль книги» вычтен,
    остаётся то, чем главы отличаются друг от друга. Без этого cos между любыми
    двумя главами одной книги ~0.9 и верную пару от неверной не отличить."""
    C = [_chapter_centroid(model, c) for c in chs]
    d = next((v.shape[0] for v in C if v is not None), 768)
    M = np.stack([v if v is not None else np.zeros(d, np.float32) for v in C])
    M = M - M.mean(axis=0, keepdims=True)
    return _unit_rows(M)


def _match_dp(su, tu, E1, E2, use_titles, C1=None, C2=None, ratio=1.0, max_group=3):
    """Монотонное сопоставление глав группами a:b (a,b <= max_group) с возможностью
    пропустить главу целиком (1:0 / 0:1). Цена группы: расхождение содержания
    (1 - cos центроидов), расхождение длин (с поправкой на коэффициент книги),
    штраф за склейку, мелкая поправка за совпадение/несовпадение номеров."""
    n, m = len(su), len(tu)
    INF = 1e18
    chars_s = np.array([x["_chars"] for x in su], float)
    chars_t = np.array([x["_chars"] for x in tu], float)
    med = float(np.median(np.concatenate([chars_s, chars_t]))) if (n + m) else 1.0
    nums_s = [_title_number(x.get("title", "")) for x in su]
    nums_t = [_title_number(x.get("title", "")) for x in tu]
    dp = np.full((n + 1, m + 1), INF)
    prev = [[None] * (m + 1) for _ in range(n + 1)]
    dp[0, 0] = 0.0
    w_len = CH_W_LEN if C1 is not None else 1.5

    def skip_cost(chars):
        return CH_SKIP_BASE + 0.3 * min(2.0, chars / max(med, 1.0))

    def group_cost(i, j, a, b):
        ls, lt = chars_s[i:i + a].sum(), chars_t[j:j + b].sum()
        pred = ls * ratio
        lr = min(pred, lt) / max(pred, lt, 1.0)
        # ниже 0.6 штраф за длину удваивается: склеить главу с «чужой» глубокой
        # разницей размеров не должно быть дешевле, чем признать её лишней
        c = w_len * ((1 - lr) + max(0.0, 0.6 - lr)) + CH_W_MERGE * (a + b - 2)
        if C1 is not None and C2 is not None:
            sv = (C1[i:i + a] * chars_s[i:i + a, None]).sum(axis=0)
            tv = (C2[j:j + b] * chars_t[j:j + b, None]).sum(axis=0)
            sem = float(sv @ tv / max(np.linalg.norm(sv) * np.linalg.norm(tv), 1e-9))
            c += CH_W_SEM * (1 - sem)
        if use_titles:
            st = " · ".join(x.get("title", "") for x in su[i:i + a])
            tt = " · ".join(x.get("title", "") for x in tu[j:j + b])
            lex = _title_sim_simple(st, tt)
            sem_t = 0.0
            if E1 is not None and E2 is not None:
                sv = np.mean(E1[i:i + a], axis=0); tv = np.mean(E2[j:j + b], axis=0)
                sem_t = float(np.dot(sv, tv) / max(np.linalg.norm(sv) * np.linalg.norm(tv), 1e-9))
            c += 0.4 * (1 - max(lex, sem_t))
        ns, nt = nums_s[i:i + a], nums_t[j:j + b]
        if None not in ns and None not in nt:
            if sorted(ns) == sorted(nt):
                c -= 0.15
            elif not set(ns) & set(nt):
                c += 0.15
        return c

    for i in range(n + 1):
        for j in range(m + 1):
            if dp[i, j] >= INF:
                continue
            if i < n:
                c = dp[i, j] + skip_cost(chars_s[i])
                if c < dp[i + 1, j]:
                    dp[i + 1, j] = c; prev[i + 1][j] = (i, j)
            if j < m:
                c = dp[i, j] + skip_cost(chars_t[j])
                if c < dp[i, j + 1]:
                    dp[i, j + 1] = c; prev[i][j + 1] = (i, j)
            for a in range(1, min(max_group, n - i) + 1):
                for b in range(1, min(max_group, m - j) + 1):
                    if a + b > max_group + 1:
                        continue
                    c = dp[i, j] + group_cost(i, j, a, b)
                    if c < dp[i + a, j + b]:
                        dp[i + a, j + b] = c; prev[i + a][j + b] = (i, j)
    groups, i, j = [], n, m
    while i or j:
        pi, pj = prev[i][j]
        groups.append((pi, i, pj, j))
        i, j = pi, pj
    groups.reverse()
    return groups


def _group_ratio_median(su, tu, groups):
    rs = []
    for a, b, c, d in groups:
        if b > a and d > c:
            ls = sum(x["_chars"] for x in su[a:b]); lt = sum(x["_chars"] for x in tu[c:d])
            if ls and lt:
                rs.append(lt / ls)
    return float(np.median(rs)) if rs else 1.0


def _match_two_pass(su, tu, E1, E2, use_titles, C1, C2):
    ts = sum(x["_chars"] for x in su); tt = sum(x["_chars"] for x in tu)
    ratio = tt / max(ts, 1)
    groups = _match_dp(su, tu, E1, E2, use_titles, C1, C2, ratio)
    r2 = _group_ratio_median(su, tu, groups)
    if abs(r2 / ratio - 1) > 0.03:             # уточняем коэффициент по сошедшимся главам
        groups = _match_dp(su, tu, E1, E2, use_titles, C1, C2, r2)
    return groups


def _log_chapter_groups(su, tu, groups, C1, C2):
    _log(1, "  сопоставление глав (длины tgt/src, sem = сходство содержания "
            "по центрированным центроидам):")
    for a, b, c, d in groups:
        st = " · ".join(str(x.get("title", "")) for x in su[a:b]) or "—"
        tt = " · ".join(str(x.get("title", "")) for x in tu[c:d]) or "—"
        if b > a and d > c:
            ls = sum(x["_chars"] for x in su[a:b]); lt = sum(x["_chars"] for x in tu[c:d])
            extra = f"len={lt / ls:.2f}"
            flag = not 0.6 <= lt / ls <= 1.6
            if C1 is not None:
                sv = (C1[a:b] * np.array([x["_chars"] for x in su[a:b]])[:, None]).sum(0)
                tv = (C2[c:d] * np.array([x["_chars"] for x in tu[c:d]])[:, None]).sum(0)
                sem = float(sv @ tv / max(np.linalg.norm(sv) * np.linalg.norm(tv), 1e-9))
                extra += f" sem={sem:.2f}"
                flag = flag or sem < 0.2
            _log(1, f"    {'⚠' if flag else ' '} {st[:34]!r} ↔ {tt[:34]!r}  {extra}")
        else:
            side = "src" if b > a else "tgt"
            _log(1, f"    ✗ БЕЗ ПАРЫ ({side}): {(st if b > a else tt)[:60]!r}")


def match_chapters(src, tgt, model):
    su = [dict(ch, _chars=_chapter_chars(ch)) for ch in src["chapters"]]
    tu = [dict(ch, _chars=_chapter_chars(ch)) for ch in tgt["chapters"]]
    use_titles = _titles_informative(su) and _titles_informative(tu)
    E1 = E2 = None
    if use_titles:
        try:
            E1 = _encode(model, [x.get("title", "") for x in su])
            E2 = _encode(model, [x.get("title", "") for x in tu])
        except Exception as exc:
            print(f"[warn] semantic chapter-title matching failed: {exc}")
    else:
        print("[info] chapter titles are generic (numbers / 'Unit N'): matching by "
              "content, numbers, size and order")
    C1 = C2 = None
    try:
        C1 = _centered_centroids(model, src["chapters"])
        C2 = _centered_centroids(model, tgt["chapters"])
    except Exception as exc:
        print(f"[warn] content-based chapter matching failed ({exc}); using size only")

    # Parts (Книга/Part/Часть...) are hard boundaries: a chapter group never crosses one.
    rs, rt = _split_by_parts(su), _split_by_parts(tu)
    if rs and rt and len(rs) == len(rt) and len(rs) >= 2:
        ok = True
        for (a0, a1), (b0, b1) in zip(rs, rt):
            ls = sum(x["_chars"] for x in su[a0:a1]); lt = sum(x["_chars"] for x in tu[b0:b1])
            if not 0.5 <= ls / lt <= 2.0:
                ok = False
        if ok:
            groups = []
            for (a0, a1), (b0, b1) in zip(rs, rt):
                sub = _match_two_pass(su[a0:a1], tu[b0:b1],
                                      E1[a0:a1] if E1 is not None else None,
                                      E2[b0:b1] if E2 is not None else None, use_titles,
                                      C1[a0:a1] if C1 is not None else None,
                                      C2[b0:b1] if C2 is not None else None)
                groups += [(a0 + p, a0 + q, b0 + r, b0 + t) for p, q, r, t in sub]
            print(f"[info] chapters matched inside {len(rs)} parts: {len(groups)} groups")
            _log_chapter_groups(su, tu, groups, C1, C2)
            return groups
        print("[warn] part structure differs between the books - falling back to global matching")

    groups = _match_two_pass(su, tu, E1, E2, use_titles, C1, C2)
    _log_chapter_groups(su, tu, groups, C1, C2)
    return groups


def build_alignment_plan(src, tgt, model):
    """-> (plan, unmatched). unmatched — главы, у которых в другом издании нет
    пары (пролог, примечания, послесловие): они не выравниваются, а попадают в
    отчёт, вместо того чтобы склеиваться с чужой главой."""
    groups = match_chapters(src, tgt, model)
    plan, unmatched = [], []
    for (a, b, c, d) in groups:
        su, tu = src["chapters"][a:b], tgt["chapters"][c:d]
        if not su or not tu:
            side, chs = ("src", su) if su else ("tgt", tu)
            unmatched.append({
                "side": side,
                "titles": [x.get("title", "") for x in chs],
                "chapter_nos": [x.get("chapter_no") for x in chs],
                "chars": sum(_chapter_chars(x) for x in chs)})
            continue
        plan.append({
            "chapter_no": len(plan) + 1,
            "title_src": " · ".join(x.get("title", "") for x in su),
            "title_tgt": " · ".join(x.get("title", "") for x in tu),
            "paragraphs_src": [p for x in su for p in x.get("paragraphs", []) if str(p).strip()],
            "paragraphs_tgt": [p for x in tu for p in x.get("paragraphs", []) if str(p).strip()],
            "source_chapters": [x.get("chapter_no") for x in su],
            "target_chapters": [x.get("chapter_no") for x in tu],
        })
    if not plan:
        raise RuntimeError("Unable to match chapters")
    rs = []
    for u in plan:
        ls = sum(len(str(p)) for p in u["paragraphs_src"])
        lt = sum(len(str(p)) for p in u["paragraphs_tgt"])
        if ls and lt:
            rs.append(lt / ls)
    good = [r for r in rs if 0.6 <= r <= 1.6]
    prior = float(np.median(good)) if good else None
    for u in plan:
        u["ratio_prior"] = prior
    return plan, unmatched


def _meta_for_output(src, tgt, override=""):
    sm = src.get("meta", {})
    tm = tgt.get("meta", {})
    meta = {
        "author_src": str(sm.get("author", "")),
        "title_src": str(sm.get("title", "")),
        "author_tgt": str(tm.get("author", "")),
        "title_tgt": str(tm.get("title", "")),
    }
    if override:
        for item in override.split(";"):
            if "=" in item:
                k, v = item.split("=", 1)
                if k.strip() in meta and v.strip():
                    meta[k.strip()] = v.strip()
    return meta


def _write_json(path, payload):
    Path(path).write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def _final_payload(meta, chapters):
    return {"meta": meta, "chapters": chapters}


# ---------------------------------------------------------------------------
# Optional OpenCode semantic repair for suspicious windows.
# ---------------------------------------------------------------------------
OPENCODE_LLM_ENABLED = os.getenv("OPENCODE_LLM_ENABLED", "0").lower() not in {"0", "false", "no", "off"}
OPENCODE_MODEL = os.getenv("OPENCODE_MODEL", "").strip()
OPENCODE_TIMEOUT = int(os.getenv("OPENCODE_TIMEOUT", "180"))
OPENCODE_CONTEXT_PAIRS = int(os.getenv("OPENCODE_LLM_CONTEXT_PAIRS", "2"))
OPENCODE_MAX_PAIRS = int(os.getenv("OPENCODE_LLM_MAX_PAIRS", "7"))
OPENCODE_RETRIES = int(os.getenv("OPENCODE_LLM_RETRIES", "1"))


def _find_opencode_command():
    for name in ("opencode", "opencode.cmd", "opencode.exe"):
        path = shutil.which(name)
        if path:
            return path
    return None


def _extract_json_from_llm(raw):
    if not raw:
        return None
    decoder = json.JSONDecoder()
    for pos, ch in enumerate(raw):
        if ch not in "{[":
            continue
        try:
            value, _ = decoder.raw_decode(raw[pos:])
            if isinstance(value, dict):
                return value
        except json.JSONDecodeError:
            pass
    return None


def _run_opencode_prompt(prompt):
    cmd = _find_opencode_command()
    if not cmd:
        return None, "opencode_not_found"
    args = [cmd, "run"]
    if OPENCODE_MODEL:
        args += ["--model", OPENCODE_MODEL]
    args += ["--format", "default", prompt]
    try:
        proc = subprocess.run(args, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True,
                              encoding="utf-8", errors="replace", timeout=OPENCODE_TIMEOUT,
                              cwd=os.getcwd())
    except subprocess.TimeoutExpired:
        return None, "timeout"
    except Exception as exc:
        return None, f"exception:{type(exc).__name__}:{exc}"
    if proc.returncode != 0:
        return None, f"exit_{proc.returncode}:{(proc.stderr or '').strip()[:300]}"
    return (proc.stdout or "").strip(), "ok"


def _validate_llm_alignment(payload, n_src, n_tgt):
    if not isinstance(payload, dict) or not isinstance(payload.get("pairs"), list) or not payload["pairs"]:
        return False, "missing_pairs", []
    used_s, used_t, out = [], [], []
    for item in payload["pairs"]:
        if not isinstance(item, dict) or not isinstance(item.get("src"), list) or not isinstance(item.get("tgt"), list):
            return False, "bad_pair", []
        si, ti = item["src"], item["tgt"]
        if not si or not ti or not all(isinstance(x, int) and not isinstance(x, bool) for x in si + ti):
            return False, "bad_indices", []
        if any(x < 0 or x >= n_src for x in si) or any(x < 0 or x >= n_tgt for x in ti):
            return False, "index_out_of_range", []
        si, ti = sorted(si), sorted(ti)
        if set(si) & set(used_s) or set(ti) & set(used_t):
            return False, "duplicate_indices", []
        used_s.extend(si); used_t.extend(ti); out.append({"src": si, "tgt": ti})
    if sorted(used_s) != list(range(n_src)) or sorted(used_t) != list(range(n_tgt)):
        return False, "not_full_coverage", []
    ps = pt = -1
    for item in out:
        if min(item["src"]) <= ps or min(item["tgt"]) <= pt:
            return False, "non_monotonic", []
        ps, pt = max(item["src"]), max(item["tgt"])
    return True, "ok", out


def _opencode_realign_segment(src_sents, tgt_sents, chapter_no, pair_start, retry=0):
    src_block = "\n".join(f"S{i}: {s}" for i, s in enumerate(src_sents))
    tgt_block = "\n".join(f"T{i}: {s}" for i, s in enumerate(tgt_sents))
    prompt = f"""
You are repairing sentence alignment for a bilingual Russian -> Bulgarian book.
Chapter: {chapter_no}
Problem-window pair index: {pair_start}

Do NOT translate, rewrite, omit, invent, or reorder text.
Return ONLY JSON.
Every S index and every T index must be used exactly once.
Alignment must be monotonic.
Allowed groupings: 1:1, 1:N, N:1, N:M.

Format:
{{"pairs":[{{"src":[0],"tgt":[0]}},{{"src":[1,2],"tgt":[1]}}, ...]}}

RU:
{src_block}

BG:
{tgt_block}
""".strip()
    raw, status = _run_opencode_prompt(prompt)
    if status != "ok":
        return None, status
    payload = _extract_json_from_llm(raw)
    if payload is None:
        return None, "invalid_json"
    ok, reason, mapping = _validate_llm_alignment(payload, len(src_sents), len(tgt_sents))
    if not ok:
        return None, reason
    segs = []
    for item in mapping:
        segs.append((item["src"][0], item["src"][-1] + 1, item["tgt"][0], item["tgt"][-1] + 1))
    return segs, "ok"


def opencode_refine_suspect(ctx, segs, src_sents, tgt_sents, chapter_no):
    if not OPENCODE_LLM_ENABLED:
        return segs, []
    q = fast_quality(ctx, segs)
    bad = [i for i, x in enumerate(q) if x["bad"]]
    if not bad:
        return segs, []
    # Build windows around bad pair groups.
    intervals = []
    for idx in bad:
        lo = max(0, idx - OPENCODE_CONTEXT_PAIRS)
        hi = min(len(segs), idx + OPENCODE_CONTEXT_PAIRS + 1)
        if not intervals or lo > intervals[-1][1]:
            intervals.append([lo, hi])
        else:
            intervals[-1][1] = max(intervals[-1][1], hi)
    windows = []
    for lo, hi in intervals:
        cur = lo
        while cur < hi:
            nxt = min(cur + OPENCODE_MAX_PAIRS, hi)
            windows.append((cur, nxt)); cur = nxt

    logs = []
    out = list(segs)
    for a, b in reversed(windows):
        old = out[a:b]
        i0, i1 = old[0][0], old[-1][1]
        j0, j1 = old[0][2], old[-1][3]
        src_window, tgt_window = src_sents[i0:i1], tgt_sents[j0:j1]
        result = None; status = None
        for attempt in range(OPENCODE_RETRIES + 1):
            result, status = _opencode_realign_segment(src_window, tgt_window, chapter_no, a, attempt)
            if result is not None:
                break
        log = {"pair_start": a, "pair_end": b, "status": status, "accepted": False}
        if result is None:
            logs.append(log); continue
        new = [(i0 + x0, i0 + x1, j0 + y0, j0 + y1) for x0, x1, y0, y1 in result]
        local = [(x0 - i0, x1 - i0, y0 - j0, y1 - j0) for x0, x1, y0, y1 in new]
        if not new or new[0][0] != i0 or new[-1][1] != i1 or new[0][2] != j0 or new[-1][3] != j1 or not check_tiling(local, i1 - i0, j1 - j0):
            log["status"] = "tiling_failed"; logs.append(log); continue
        cost_old = ctx.total_cost(old); cost_new = ctx.total_cost(new)
        bad_old = sum(x["bad"] for x in q[a:b])
        bad_new = sum(x["bad"] for x in fast_quality(ctx, new))
        W = max(1.0, ctx.CLs[i1] - ctx.CLs[i0])
        if cost_new < cost_old - REFINE_EPS * W and bad_new <= bad_old:
            out[a:b] = new
            log.update({"accepted": True, "cost_old": cost_old / W, "cost_new": cost_new / W,
                        "bad_old": bad_old, "bad_new": bad_new})
        else:
            log.update({"status": "rejected_guard", "cost_old": cost_old / W,
                        "cost_new": cost_new / W, "bad_old": bad_old, "bad_new": bad_new})
        logs.append(log)
    return out, logs


# ---------------------------------------------------------------------------
# Per-chapter runner / final output
# ---------------------------------------------------------------------------
def align_plan_unit(unit, splitter, model):
    global CURRENT_CHAPTER_NO
    CURRENT_CHAPTER_NO = unit["chapter_no"]
    pairs, per_pair, stats, info = align_unit(
        unit["paragraphs_src"], unit["paragraphs_tgt"], splitter, model,
        engines=ENGINES,
        bt_variants=None,
        params={"ratio_prior": unit.get("ratio_prior")},
    )
    return pairs, per_pair, stats, info


def unit_signature(unit):
    """Identity of an alignment unit's INPUT. Cached chapter_NNN.json files are keyed by number
    only, so after the chapter grouping changes 'chapter_005.json' would silently be reused for
    a different piece of text."""
    import hashlib
    h = hashlib.sha1()
    h.update(b"aligner-v2")            # кэш v1 нельзя использовать с v2
    for key in ("paragraphs_src", "paragraphs_tgt"):
        for para in unit.get(key, []):
            h.update(str(para).encode("utf-8")); h.update(b"\x00")
        h.update(b"\x01")
    return h.hexdigest()[:16]


def write_unit_report(work_dir, unit, pairs, per_pair, stats, info):
    work_dir.mkdir(parents=True, exist_ok=True)
    flagged = []
    for i, (p, q) in enumerate(zip(pairs, per_pair)):
        if q["bad"]:
            flagged.append({"pair_idx": i, "src": p["src"], "tgt": p["tgt"],
                            "sim": round(q["sim"], 4), "len_ratio": round(q["len_ratio"], 4),
                            "anchor": round(q["anchor"], 4), "reason": q["reason"],
                            "run_len": q.get("run_len", 1)})
    payload = {
        "chapter_no": unit["chapter_no"],
        "unit_signature": unit_signature(unit),
        "title_src": unit["title_src"],
        "title_tgt": unit["title_tgt"],
        "source_chapters": unit["source_chapters"],
        "target_chapters": unit["target_chapters"],
        "pairs": pairs,
        "stats": stats,
        "info": info,
        "flagged_pairs": flagged,
    }
    _write_json(work_dir / f"chapter_{unit['chapter_no']:03d}.json", payload)
    return flagged


def main():
    import argparse
    p = argparse.ArgumentParser(description="JSON -> GPU sentence alignment -> final JSON")
    p.add_argument("src_json")
    p.add_argument("tgt_json")
    p.add_argument("-o", "--output", default="final_aligned_book.json")
    p.add_argument("--work-dir", default="alignment_work")
    p.add_argument("--meta-override", default="")
    p.add_argument("--units", default="all", help="all | N | A-B")
    p.add_argument("--force", action="store_true", help="recompute units even if cached reports exist")
    args = p.parse_args()

    print("=" * 76)
    print("BL SENTENCE ALIGNER v2")
    print(f"SOURCE: {args.src_json}")
    print(f"TARGET: {args.tgt_json}")
    print(f"EMBED:  {EMBED_MODEL_NAME}")
    print(f"DEVICE: {GPU_DEVICE}")
    print(f"ENGINES:{', '.join(ENGINES)}")
    print("=" * 76)

    src = load_book(args.src_json)
    tgt = load_book(args.tgt_json)
    meta = _meta_for_output(src, tgt, args.meta_override)

    model = get_embed_model()
    splitter = get_sat_splitter()
    plan, unmatched = build_alignment_plan(src, tgt, model)
    print(f"\nAlignment units: {len(plan)}")
    if unmatched:
        print(f"[warn] глав без пары: {len(unmatched)} (не выравниваются, см. отчёт):")
        for u in unmatched:
            print(f"   {u['side']}: {' · '.join(map(str, u['titles']))[:70]!r} "
                  f"({u['chars']} симв.)")
    for u in plan[:20]:
        print(f"  {u['chapter_no']:>3}: {u['title_src'][:55]!r} <-> {u['title_tgt'][:55]!r}")

    work_dir = Path(args.work_dir)
    work_dir.mkdir(parents=True, exist_ok=True)
    existing = {
        int(m.group(1)) for f in work_dir.iterdir()
        if f.is_file() and (m := re.match(r"chapter_(\d+)\.json$", f.name))
    }
    # Drop cached reports that were produced for a different input (other grouping / other text).
    sigs = {u["chapter_no"]: unit_signature(u) for u in plan}
    for no in sorted(existing):
        try:
            rec_sig = json.loads((work_dir / f"chapter_{no:03d}.json").read_text(encoding="utf-8")).get("unit_signature")
        except Exception:
            rec_sig = None
        if sigs.get(no) != rec_sig:
            existing.discard(no)
            print(f"[cache] chapter_{no:03d}.json belongs to a different input - will be recomputed")
    if args.units == "all":
        selected = {u["chapter_no"] for u in plan} if args.force else {u["chapter_no"] for u in plan if u["chapter_no"] not in existing}
    elif re.fullmatch(r"\d+", args.units):
        k = int(args.units); selected = {u["chapter_no"] for u in plan[:k]}
    elif re.fullmatch(r"\d+\s*-\s*\d+", args.units):
        lo, hi = sorted(map(int, re.split(r"\s*-\s*", args.units))); selected = {u["chapter_no"] for u in plan if lo <= u["chapter_no"] <= hi}
    else:
        raise ValueError(f"Invalid --units: {args.units}")

    final_chapters = []
    all_flagged = []
    all_unaligned = []
    for unit in plan:
        no = unit["chapter_no"]
        if no not in selected and no not in existing:
            continue
        if no in existing and no not in selected:
            rec = json.loads((work_dir / f"chapter_{no:03d}.json").read_text(encoding="utf-8"))
            pairs = [{"src": p["src"], "tgt": p["tgt"]} for p in rec.get("pairs", [])]
        elif no in existing and no in selected:
            # Reuse a completed chapter only when --units does not explicitly
            # select it; selected units are recomputed to make debugging easy.
            t0 = time.time()
            pairs, per_pair, stats, info = align_plan_unit(unit, splitter, model)
            # Optional LLM refinement works at sentence-level through ctx; leave
            # disabled by default unless OPENCODE_LLM_ENABLED=1.
            write_unit_report(work_dir, unit, pairs, per_pair, stats, info)
            print(f"[chapter {no}] {len(pairs)} pairs in {_dt(t0)}")
        else:
            t0 = time.time()
            pairs, per_pair, stats, info = align_plan_unit(unit, splitter, model)
            write_unit_report(work_dir, unit, pairs, per_pair, stats, info)
            print(f"[chapter {no}] {len(pairs)} pairs in {_dt(t0)}")

        final_chapters.append({
            "pair_num": no,
            "title_src": unit["title_src"],
            "title_tgt": unit["title_tgt"],
            "pairs": pairs,
        })

        report_path = work_dir / f"chapter_{no:03d}.json"
        if report_path.exists():
            try:
                rec = json.loads(report_path.read_text(encoding="utf-8"))
                all_flagged.extend({"chapter_no": no, **x} for x in rec.get("flagged_pairs", []))
                all_unaligned.extend({"chapter_no": no, **x}
                                     for x in rec.get("info", {}).get("unaligned", []))
            except Exception:
                pass

    # Any missing unit is a hard error: we never silently write a partial final book.
    done = {c["pair_num"] for c in final_chapters}
    missing = [u["chapter_no"] for u in plan if u["chapter_no"] not in done]
    if missing:
        raise RuntimeError(f"Missing alignment units: {missing}. Refusing to write partial final JSON.")
    final_chapters.sort(key=lambda x: x["pair_num"])
    for i, ch in enumerate(final_chapters, 1):
        ch["pair_num"] = i

    final = _final_payload(meta, final_chapters)
    _write_json(args.output, final)
    review_path = Path(args.output).with_name(Path(args.output).stem + "_review_needed.json")
    if all_flagged:
        _write_json(review_path, all_flagged)
    elif review_path.exists():
        review_path.unlink()

    side_path = Path(args.output).with_name(Path(args.output).stem + "_unaligned.json")
    if all_unaligned or unmatched:
        _write_json(side_path, {"unmatched_chapters": unmatched,
                                "unaligned_blocks": all_unaligned})
    elif side_path.exists():
        side_path.unlink()

    print("\n" + "=" * 76)
    print(f"DONE: {args.output}")
    if all_unaligned or unmatched:
        print(f"unaligned={side_path} | блоков без пары: {len(all_unaligned)}, "
              f"глав без пары: {len(unmatched)}")
    print(f"chapters={len(final_chapters)} | pairs={sum(len(c['pairs']) for c in final_chapters)}")
    if all_flagged:
        print(f"review={review_path} | flagged={len(all_flagged)}")
    else:
        print("review: no flagged pairs")


if __name__ == "__main__":
    main()
