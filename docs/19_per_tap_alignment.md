# Per-Tap Alignment Update

**Date:** 2026-07-23  
**Status:** Implementation Complete  
**Scope:** Changed alignment & export to generate separate deliverables per Vietnamese volume (tap4, tap5, tap6)

## Problem

Previous alignment produced combined outputs across all Vietnamese volumes:
- Single `{prefix}_raw.txt` (all taps merged)
- Single `{prefix}_parallel.tsv` (all pairs mixed)
- Single `{prefix}_parallel.xlsx` (all pairs mixed)

Teacher requirement: **Separate deliverables per work/tap.**

## Solution

### 1. Stage 5: Alignment (vecalign_runner.py)

**Change:** Propagate `tap` field through alignment pipeline.

- Modified `prep_vecalign_input()` to capture `tap` metadata from VI_SENT JSONL
- Added `is_vi` parameter to track which side we're processing
- Updated `main()` to extract first `tgt_idx`'s `tap` and add to pairs object

**Result:** pairs.jsonl now includes `tap` field:
```json
{
  "src_idx": [42],
  "tgt_idx": [58],
  "src": "紹治四年三月十一日",
  "tgt": "Năm Thiệu Trị thứ tư, ngày 11 tháng 3",
  "score": 0.823,
  "tap": "tap4"
}
```

### 2. Stage 7: Export (export_deliverable.py)

**Change:** Group pairs by tap, generate separate outputs.

**Modified functions:**
- `load_pairs()` → returns `dict[tap_name, list of (pair_id, han, viet, sino)]`
- `write_raw(tap)` → writes `{prefix}_{tap}_raw.txt`
- `write_tsv(tap, rows)` → writes `{prefix}_{tap}_parallel.tsv`
- `write_xlsx(tap, rows)` → writes `{prefix}_{tap}_parallel.xlsx`
- `main()` → iterates over taps, calls write functions per-tap

**Output structure:**
```
data/final/
├── {prefix}_tap4_raw.txt          # Raw OCR concatenated for tap4
├── {prefix}_tap4_parallel.tsv     # 1 row per pair (pair_id, han, viet, sino)
├── {prefix}_tap4_parallel.xlsx    # Same in Excel
├── {prefix}_tap5_raw.txt
├── {prefix}_tap5_parallel.tsv
├── {prefix}_tap5_parallel.xlsx
├── {prefix}_tap6_raw.txt
├── {prefix}_tap6_parallel.tsv
└── {prefix}_tap6_parallel.xlsx
```

### 3. Documentation

Updated:
- **docs/02_data.md**: Stage 5 & 7 schema (added tap field, per-tap outputs)
- **docs/03_pipeline.md**: Added detailed Stage 5 & 7 descriptions for per-tap flow

## Data Format

### Input: pairs.jsonl (Stage 5 output)

```json
{"src_idx": [0], "tgt_idx": [5], "src": "Han", "tgt": "Viet", "score": 0.85, "tap": "tap4"}
{"src_idx": [1], "tgt_idx": [6], "src": "Han", "tgt": "Viet", "score": 0.92, "tap": "tap5"}
```

### Output: TSV/XLSX (Stage 7 output)

Example `{prefix}_tap4_parallel.tsv`:
```
pair_id	han_sentence	viet_sentence	sino
1	紹治四年三月十一日	Năm Thiệu Trị thứ tư, ngày 11 tháng 3	thieu tri
2	上諭	Chỉ dụ	thuong nho
3	...	...	...
```

### Output: Raw OCR (Stage 7 output)

`{prefix}_tap4_raw.txt` = concatenated raw OCR text for all pages of tap4.

## Backward Compatibility

- **pairs.jsonl schema:** Added optional `tap` field. Old code reading without `tap` still works.
- **export_deliverable.py:** Completely replaced with new per-tap logic. Old single-file output no longer generated.

## Testing

✓ Python syntax validation passed for:
- `src/05_align/vecalign_runner.py`
- `src/07_export/export_deliverable.py`

To run:
```bash
./scripts/run_pipeline.sh align    # Stage 5: add tap field to pairs
./scripts/run_pipeline.sh export   # Stage 7: generate per-tap deliverables
```

## Files Modified

1. `src/05_align/vecalign_runner.py` — propagate tap through alignment
2. `src/07_export/export_deliverable.py` — group pairs by tap, separate outputs
3. `docs/02_data.md` — document tap field & per-tap output schema
4. `docs/03_pipeline.md` — document per-tap stages 5 & 7

## Notes

- Only outputs Vietnamese taps (tap4, tap5, tap6). Han side has no tap.
- pair_id resets per-tap (1-indexed within each tap).
- Raw OCR concatenated from per-page files for each tap.
- Filtering (Sino-Viet precision, length ratio) applied before pair_id assignment.
