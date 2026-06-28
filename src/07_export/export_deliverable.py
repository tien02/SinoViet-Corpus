"""Stage 7 (export): build the course deliverable files from aligned pairs.

Produces the three files the course spec asks for:
  {prefix}_raw.txt       - raw OCR concatenated (tap4 + tap5 + tap6)
  {prefix}_parallel.tsv  - [pair_id]\\t[han_sentence]\\t[viet_sentence]
  {prefix}_parallel.xlsx - same three columns in Excel

Reads only pairs.jsonl (align output) and the raw OCR txt files. Prefix is set
via HVB_DELIVERABLE_PREFIX (mã số sinh viên).
"""
from __future__ import annotations

import csv
import json
import os
import sys
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

# One physical line per sentence: collapse internal newlines so the TSV/XLSX
# stay strictly one record per row.
def _flatten(text: str) -> str:
    return " ".join(text.split())


def load_pairs() -> list[tuple[int, str, str]]:
    if not PAIRS_JSONL.exists():
        raise SystemExit(f"Missing {PAIRS_JSONL}. Run the align stage first.")
    rows: list[tuple[int, str, str]] = []
    dropped_empty = 0
    dropped_long = 0
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
        rows.append((pair_id, han, viet))
        pair_id += 1
    if not rows:
        raise SystemExit(f"No usable pairs in {PAIRS_JSONL}.")
    print(
        f"  pairs kept={len(rows):,} "
        f"dropped(empty={dropped_empty}, >{MAX_PAIR_CHARS}chars={dropped_long})"
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


def write_tsv(rows: list[tuple[int, str, str]]) -> None:
    with DELIVERABLE_TSV.open("w", encoding="utf-8", newline="") as f:
        w = csv.writer(f, delimiter="\t", quoting=csv.QUOTE_MINIMAL)
        w.writerow(["pair_id", "han_sentence", "viet_sentence"])
        w.writerows(rows)
    print(f"  tsv   -> {DELIVERABLE_TSV} ({len(rows):,} pairs)")


def write_xlsx(rows: list[tuple[int, str, str]]) -> None:
    try:
        import pandas as pd
    except ImportError:
        raise SystemExit("pandas required for xlsx export (uv add pandas openpyxl).")
    df = pd.DataFrame(rows, columns=["pair_id", "han_sentence", "viet_sentence"])
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
