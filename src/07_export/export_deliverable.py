"""Stage 7 (export): build per-tap deliverable files from aligned pairs.

Produces separate output for each Vietnamese volume (tap):
  {prefix}_{tap}_raw.txt       - raw OCR for that tap
  {prefix}_{tap}_parallel.tsv  - [pair_id]\\t[han_sentence]\\t[viet_sentence]\\t[sino]
  {prefix}_{tap}_parallel.xlsx - same columns in Excel

Reads pairs.jsonl (align output, now with tap field) and per-page raw OCR files.
Prefix is set via HVB_DELIVERABLE_PREFIX (mã số sinh viên).
Only outputs Vietnamese taps (tap4, tap5, tap6).

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

# Ultra-short Hán filter: drop Hán <=4 chars (punctuator noise fragments like
# 夫。撫。三。). These are almost always false alignments with score ~0.5.
# Set to 0 to disable. Default 4 catches ~87 noise pairs in Phase 1 data.
MIN_HAN_LEN = int(os.environ.get("HVB_MIN_HAN_LEN", "4"))

# Page-margin artifact filter: drop Việt starting with `1)`, `2)`, `(1)`, `[1]`,
# etc. (docx/OCR page furniture). These have score ~0.5 and are not real
# translations. Set to 0 to disable. Default catches ~119 artifacts in Phase 1.
NUM_MARKER_RE = re.compile(r"^[\(\[]?\d")

# Rescue policy: if embedder cosine is high, don't drop a pair for weak sino or
# out-of-range length ratio. Prevents false positives from proxy filters when
# semantic embedding is confident (e.g. Hán 千載 → Việt "nghìn năm" — no
# Sino-Viet syllable match but clearly correct translation).
# Only applies when score is cosine similarity (bertalign). Set to 1.1 to disable.
SINO_RESCUE_COS = float(os.environ.get("HVB_SINO_RESCUE_COS", "0.55"))
RATIO_RESCUE_COS = float(os.environ.get("HVB_RATIO_RESCUE_COS", "0.60"))

# Low-confidence outlier filter: drops short Hán with huge Việt explanation
# but zero phonetic overlap. Targets genuinely noisy pairs from list-bracket
# splitting while keeping legitimate semantic translations (high cosine).
# Condition: ratio > 10 AND han_len < 10 AND sino < 0.3
# Set to 0 to disable. Default catches ~17 noisy pairs (0.05% of corpus).
DROP_LOW_CONF_OUTLIERS = bool(os.environ.get("HVB_DROP_LOW_CONF", "1"))

# Hard ceiling on length ratio (never rescues): drops extreme outliers
# (e.g. Hán 15 chars → Việt 1461 chars = 97x ratio). Even high-cosine
# semantic pairs drop if ratio exceeds this. Set to 0 to disable.
# Default 15 catches remaining 28 extreme cases.
MAX_EXTREME_RATIO = float(os.environ.get("HVB_MAX_EXTREME_RATIO", "15.0"))
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


def load_pairs() -> dict[str | None, list[tuple[int, str, str, float]]]:
    """Load pairs grouped by tap (Vietnamese volume).

    Returns dict[tap_name, list of (pair_id, han, viet, sino)].
    tap_name = 'tap4', 'tap5', etc. or None for pairs without tap info.
    """
    if not PAIRS_JSONL.exists():
        raise SystemExit(f"Missing {PAIRS_JSONL}. Run the align stage first.")
    rows_by_tap: dict[str | None, list[tuple[int, str, str, float]]] = {}
    dropped_empty = 0
    dropped_long = 0
    dropped_sino = 0
    dropped_ratio = 0
    dropped_low_conf_outlier = 0
    dropped_extreme_ratio = 0
    dropped_han_short = 0
    dropped_num_marker = 0
    rescued_ratio = 0
    rescued_sino = 0
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
        ratio = len(viet) / max(1, len(han))
        sino = _sino_precision(han, viet) if (MIN_SINO > 0 or DROP_LOW_CONF_OUTLIERS) else 0.0
        if DROP_LOW_CONF_OUTLIERS and ratio > 10 and len(han) < 10 and sino < 0.3:
            dropped_low_conf_outlier += 1
            continue
        if MAX_EXTREME_RATIO > 0 and ratio > MAX_EXTREME_RATIO:
            dropped_extreme_ratio += 1
            continue
        if MIN_SINO > 0 and sino == 0.0:
            sino = _sino_precision(han, viet)
        if MIN_SINO > 0 and sino < MIN_SINO:
            if _SCORE_IS_SIM and cos >= SINO_RESCUE_COS:
                rescued_sino += 1
            else:
                dropped_sino += 1
                continue
        if MIN_HAN_LEN > 0 and len(han) <= MIN_HAN_LEN:
            dropped_han_short += 1
            continue
        if NUM_MARKER_RE and NUM_MARKER_RE.match(viet.lstrip()):
            dropped_num_marker += 1
            continue
        tap = p.get("tap")  # Group by tap (Vietnamese volume)
        if tap not in rows_by_tap:
            rows_by_tap[tap] = []
        rows_by_tap[tap].append((han, viet, sino))

    if not rows_by_tap:
        raise SystemExit(f"No usable pairs in {PAIRS_JSONL}.")

    # Assign pair_ids per tap
    for tap in rows_by_tap:
        rows_by_tap[tap] = [(i + 1, han, viet, sino) for i, (han, viet, sino) in enumerate(rows_by_tap[tap])]

    total_pairs = sum(len(rows) for rows in rows_by_tap.values())
    print(
        f"  pairs kept={total_pairs:,} by tap={list(rows_by_tap.keys())} "
        f"dropped(empty={dropped_empty}, >{MAX_PAIR_CHARS}chars={dropped_long}"
        + (f", ratio∉[{MIN_LEN_RATIO},{MAX_LEN_RATIO}]={dropped_ratio}"
           if (MIN_LEN_RATIO > 0 or MAX_LEN_RATIO > 0) else "")
        + (f", low_conf_outlier={dropped_low_conf_outlier}" if DROP_LOW_CONF_OUTLIERS else "")
        + (f", extreme_ratio>{MAX_EXTREME_RATIO}={dropped_extreme_ratio}" if MAX_EXTREME_RATIO > 0 else "")
        + (f", sino<{MIN_SINO}={dropped_sino}" if MIN_SINO > 0 else "")
        + (f", han<={MIN_HAN_LEN}chars={dropped_han_short}" if MIN_HAN_LEN > 0 else "")
        + (f", num_marker={dropped_num_marker}" if NUM_MARKER_RE else "")
        + ")"
    )
    if _SCORE_IS_SIM and (rescued_ratio or rescued_sino):
        print(
            f"  rescued by cosine: ratio={rescued_ratio} (cos≥{RATIO_RESCUE_COS}), "
            f"sino={rescued_sino} (cos≥{SINO_RESCUE_COS})"
        )
    return rows_by_tap


def write_raw(tap: str | None) -> None:
    """Write raw OCR for a specific Vietnamese tap."""
    if not tap:
        print(f"  WARN: skipping raw.txt for pairs without tap info")
        return
    raw_file = DELIVERABLE_RAW.parent / f"{DELIVERABLE_PREFIX}_{tap}_raw.txt"
    # Concatenate per-page OCR for this tap
    pages = sorted(VI_OCR_RAW_DIR.glob(f"{tap}_page_*.txt")) if VI_OCR_RAW_DIR.exists() else []
    if not pages:
        # Fallback to combined tap file
        combined = VI_OCR_RAW_DIR / f"{tap}.txt" if VI_OCR_RAW_DIR.exists() else None
        if combined and combined.exists():
            raw_file.write_text(combined.read_text(encoding="utf-8"), encoding="utf-8")
            print(f"  raw   -> {raw_file}")
        return
    parts = []
    for p in pages:
        parts.append(p.read_text(encoding="utf-8").rstrip("\n"))
    raw_file.write_text("\n\n".join(parts) + "\n", encoding="utf-8")
    print(f"  raw   -> {raw_file} ({len(pages)} pages)")


def write_tsv(tap: str | None, rows: list[tuple[int, str, str, float]]) -> None:
    if not tap:
        print(f"  WARN: skipping TSV for pairs without tap info")
        return
    tsv_file = DELIVERABLE_TSV.parent / f"{DELIVERABLE_PREFIX}_{tap}_parallel.tsv"
    with tsv_file.open("w", encoding="utf-8", newline="") as f:
        w = csv.writer(f, delimiter="\t", quoting=csv.QUOTE_MINIMAL)
        w.writerow(["pair_id", "han_sentence", "viet_sentence", "sino"])
        w.writerows(rows)
    print(f"  tsv   -> {tsv_file} ({len(rows):,} pairs)")


def write_xlsx(tap: str | None, rows: list[tuple[int, str, str, float]]) -> None:
    if not tap:
        print(f"  WARN: skipping XLSX for pairs without tap info")
        return
    try:
        import pandas as pd
    except ImportError:
        raise SystemExit("pandas required for xlsx export (uv add pandas openpyxl).")
    xlsx_file = DELIVERABLE_XLSX.parent / f"{DELIVERABLE_PREFIX}_{tap}_parallel.xlsx"
    df = pd.DataFrame(
        rows, columns=["pair_id", "han_sentence", "viet_sentence", "sino"]
    )
    try:
        df.to_excel(xlsx_file, index=False, engine="openpyxl")
    except ImportError:
        raise SystemExit("openpyxl required for xlsx export (uv add openpyxl).")
    print(f"  xlsx  -> {xlsx_file} ({len(rows):,} pairs)")


def main() -> None:
    DELIVERABLE_RAW.parent.mkdir(parents=True, exist_ok=True)
    print(f"Exporting deliverable per-tap (prefix='{DELIVERABLE_PREFIX}')")
    rows_by_tap = load_pairs()
    for tap in sorted(rows_by_tap.keys()):
        print(f"\n{tap}:")
        write_raw(tap)
        write_tsv(tap, rows_by_tap[tap])
        write_xlsx(tap, rows_by_tap[tap])


if __name__ == "__main__":
    main()
