"""Stage 7 (export): build the course deliverable files from aligned pairs.

Produces the three files the course spec asks for:
  {prefix}_raw.txt       - raw OCR concatenated (tap4 + tap5 + tap6)
  {prefix}_parallel.tsv  - [pair_id]\\t[han_sentence]\\t[viet_sentence]\\t[sino]
  {prefix}_parallel.xlsx - same columns in Excel

Reads only pairs.jsonl (align output) and the raw OCR txt files. Prefix is set
via HVB_DELIVERABLE_PREFIX (mã số sinh viên).

Optional Sino-Viet phonetic filter (HVB_MIN_SINO, default 0 = off): drops pairs
whose Han→Sino-Viet pronunciation has <threshold lexical overlap with the
Việt side. Effective at removing semantic-embedding false positives on short
title/header pairs. See scripts/rescore_sino.py.
"""
from __future__ import annotations

import csv
import json
import os
import re
import sys
import unicodedata
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from src.utils.config import (  # noqa: E402
    DELIVERABLE_PREFIX,
    DELIVERABLE_RAW,
    DELIVERABLE_TSV,
    DELIVERABLE_XLSX,
    PAIRS_JSONL,
    VI_OCR_RAW_DIR,
)

# Drop degenerate pairs whose Hán or Việt side exceeds this many characters.
# Median sentence is ~25 (Hán) / ~107 (Việt) chars; anything in the thousands is
# a Vecalign range-merge artifact (non-monotonic order), not a real sentence.
# Also keeps every cell under Excel's 32,767-char limit. Set 0 to disable.
MAX_PAIR_CHARS = int(os.environ.get("HVB_MAX_PAIR_CHARS", "2000"))

# Minimum Sino-Viet phonetic precision for a pair to be kept. 0 = off.
# Default 0.15 drops ~5% of pairs (short titles, publisher names, page
# furniture bleed — vecalign false positives where semantic embedding
# happens to land similar-length short strings). Override via HVB_MIN_SINO=0
# to disable, or HVB_MIN_SINO=0.30 for high-precision subset.
MIN_SINO = float(os.environ.get("HVB_MIN_SINO", "0.15"))

# Length ratio guardrails (viet_chars / han_chars). Bertalign m-n beads can
# emit heavy asymmetric merges; spot-checked pairs outside [0.5, 8.0] are
# almost always bad matches. Set both to 0 to disable.
MIN_LEN_RATIO = float(os.environ.get("HVB_MIN_LEN_RATIO", "0.5"))
MAX_LEN_RATIO = float(os.environ.get("HVB_MAX_LEN_RATIO", "8.0"))

# Rescue policy: if embedder cosine is high, don't drop a pair for weak sino or
# out-of-range length ratio. Prevents false positives from proxy filters when
# semantic embedding is confident (e.g. Hán 千載 → Việt "nghìn năm" — no
# Sino-Viet syllable match but clearly correct translation).
# Only applies when score is cosine similarity (bertalign). Set to 1.1 to disable.
SINO_RESCUE_COS = float(os.environ.get("HVB_SINO_RESCUE_COS", "0.55"))
RATIO_RESCUE_COS = float(os.environ.get("HVB_RATIO_RESCUE_COS", "0.60"))
_SCORE_IS_SIM = os.environ.get("HVB_ALIGNER", "vecalign").lower() == "bertalign"

_CONVERTER = None
_TOK_RE = re.compile(r"[A-Za-zÀ-ỹ]+")


def _strip_diacritics(s: str) -> str:
    s = s.lower()
    s = unicodedata.normalize("NFKD", s)
    return "".join(c for c in s if not unicodedata.combining(c))


def _sino_precision(han: str, viet: str) -> float:
    """Han→Sino-Viet pronunciation overlap with Việt side. 0..1."""
    global _CONVERTER
    if _CONVERTER is None:
        try:
            from cn2vn import converter as _c
            _CONVERTER = _c
        except ImportError:
            return 1.0
    sino = _CONVERTER.cn2vn(han)
    toks = _TOK_RE.findall(sino)
    viet_flat = _strip_diacritics(viet)
    ge3 = [_strip_diacritics(t) for t in toks if len(_strip_diacritics(t)) >= 3]
    if not ge3:
        return 0.0
    matched = sum(1 for t in ge3 if t in viet_flat)
    return matched / len(ge3)

# One physical line per sentence: collapse internal newlines so the TSV/XLSX
# stay strictly one record per row.
def _flatten(text: str) -> str:
    return " ".join(text.split())


def load_pairs() -> list[tuple[int, str, str, float]]:
    if not PAIRS_JSONL.exists():
        raise SystemExit(f"Missing {PAIRS_JSONL}. Run the align stage first.")
    rows: list[tuple[int, str, str, float]] = []
    dropped_empty = 0
    dropped_long = 0
    dropped_sino = 0
    dropped_ratio = 0
    rescued_ratio = 0
    rescued_sino = 0
    pair_id = 1
    for line in PAIRS_JSONL.open(encoding="utf-8"):
        line = line.strip()
        if not line:
            continue
        p = json.loads(line)
        han = _flatten(p.get("src", ""))
        viet = _flatten(p.get("tgt", ""))
        if not han or not viet:
            dropped_empty += 1
            continue
        if MAX_PAIR_CHARS and (len(han) > MAX_PAIR_CHARS or len(viet) > MAX_PAIR_CHARS):
            dropped_long += 1
            continue
        cos = float(p.get("score", 0.0)) if _SCORE_IS_SIM else 0.0
        if MIN_LEN_RATIO > 0 or MAX_LEN_RATIO > 0:
            ratio = len(viet) / max(1, len(han))
            if (MIN_LEN_RATIO > 0 and ratio < MIN_LEN_RATIO) or \
                    (MAX_LEN_RATIO > 0 and ratio > MAX_LEN_RATIO):
                if _SCORE_IS_SIM and cos >= RATIO_RESCUE_COS:
                    rescued_ratio += 1
                else:
                    dropped_ratio += 1
                    continue
        sino = _sino_precision(han, viet) if MIN_SINO > 0 else 0.0
        if MIN_SINO > 0 and sino < MIN_SINO:
            if _SCORE_IS_SIM and cos >= SINO_RESCUE_COS:
                rescued_sino += 1
            else:
                dropped_sino += 1
                continue
        rows.append((pair_id, han, viet, sino))
        pair_id += 1
    if not rows:
        raise SystemExit(f"No usable pairs in {PAIRS_JSONL}.")
    print(
        f"  pairs kept={len(rows):,} "
        f"dropped(empty={dropped_empty}, >{MAX_PAIR_CHARS}chars={dropped_long}"
        + (f", ratio∉[{MIN_LEN_RATIO},{MAX_LEN_RATIO}]={dropped_ratio}"
           if (MIN_LEN_RATIO > 0 or MAX_LEN_RATIO > 0) else "")
        + (f", sino<{MIN_SINO}={dropped_sino}" if MIN_SINO > 0 else "")
        + ")"
    )
    if _SCORE_IS_SIM and (rescued_ratio or rescued_sino):
        print(
            f"  rescued by cosine: ratio={rescued_ratio} (cos≥{RATIO_RESCUE_COS}), "
            f"sino={rescued_sino} (cos≥{SINO_RESCUE_COS})"
        )
    return rows


def write_raw() -> None:
    """Concatenate the per-tap raw OCR into one {prefix}_raw.txt."""
    taps = sorted(
        f for f in VI_OCR_RAW_DIR.glob("tap*.txt") if "_page_" not in f.name
    )
    if not taps:
        print(f"  WARN: no per-tap OCR in {VI_OCR_RAW_DIR}, skipping raw.txt")
        return
    parts = []
    for t in taps:
        parts.append(f"===== {t.stem} =====")
        parts.append(t.read_text(encoding="utf-8").rstrip("\n"))
    DELIVERABLE_RAW.write_text("\n\n".join(parts) + "\n", encoding="utf-8")
    print(f"  raw   -> {DELIVERABLE_RAW} ({len(taps)} tập)")


def write_tsv(rows: list[tuple[int, str, str, float]]) -> None:
    with DELIVERABLE_TSV.open("w", encoding="utf-8", newline="") as f:
        w = csv.writer(f, delimiter="\t", quoting=csv.QUOTE_MINIMAL)
        w.writerow(["pair_id", "han_sentence", "viet_sentence", "sino"])
        w.writerows(rows)
    print(f"  tsv   -> {DELIVERABLE_TSV} ({len(rows):,} pairs)")


def write_xlsx(rows: list[tuple[int, str, str, float]]) -> None:
    try:
        import pandas as pd
    except ImportError:
        raise SystemExit("pandas required for xlsx export (uv add pandas openpyxl).")
    df = pd.DataFrame(
        rows, columns=["pair_id", "han_sentence", "viet_sentence", "sino"]
    )
    try:
        df.to_excel(DELIVERABLE_XLSX, index=False, engine="openpyxl")
    except ImportError:
        raise SystemExit("openpyxl required for xlsx export (uv add openpyxl).")
    print(f"  xlsx  -> {DELIVERABLE_XLSX} ({len(rows):,} pairs)")


def main() -> None:
    DELIVERABLE_RAW.parent.mkdir(parents=True, exist_ok=True)
    print(f"Exporting deliverable (prefix='{DELIVERABLE_PREFIX}')")
    rows = load_pairs()
    write_raw()
    write_tsv(rows)
    write_xlsx(rows)


if __name__ == "__main__":
    main()
