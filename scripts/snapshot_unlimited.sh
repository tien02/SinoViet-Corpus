#!/usr/bin/env bash
# Snapshot prior Unlimited-OCR run: rename artifacts with `_unlimited` suffix
# so a new backend (e.g. PaddleOCR-VL) writes to default paths without
# colliding. Idempotent: skips moves whose source is missing or target exists.
# Hán-side artifacts + prep checkpoints untouched.
#
# Usage:
#   ./scripts/snapshot_unlimited.sh
#   HVB_DELIVERABLE_PREFIX=21127001 ./scripts/snapshot_unlimited.sh
#   ./scripts/snapshot_unlimited.sh --dry-run
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

DRY=0
if [ "${1:-}" = "--dry-run" ]; then
    DRY=1
    echo "[snapshot_unlimited] DRY RUN — no files will be moved"
fi

SUFFIX="_unlimited"
PREFIX="${HVB_DELIVERABLE_PREFIX:-hvb}"

move() {
    local src="$1" dst="$2"
    if [ ! -e "$src" ]; then
        echo "  skip (missing): $src"
        return
    fi
    if [ -e "$dst" ]; then
        echo "  skip (target exists): $dst"
        return
    fi
    echo "  $src -> $dst"
    [ "$DRY" -eq 1 ] || mv "$src" "$dst"
}

remove() {
    local p="$1"
    if [ -e "$p" ]; then
        echo "  rm $p"
        [ "$DRY" -eq 1 ] || rm -f "$p"
    fi
}

echo "[snapshot_unlimited] interim OCR + downstream"
move "data/interim/vi_ocr_raw"           "data/interim/vi_ocr_raw${SUFFIX}"
move "data/interim/vi_ocr_corrected"     "data/interim/vi_ocr_corrected${SUFFIX}"
move "data/interim/vi_sentences.jsonl"   "data/interim/vi_sentences${SUFFIX}.jsonl"
move "data/interim/vi_embeds.npy"        "data/interim/vi_embeds${SUFFIX}.npy"

echo "[snapshot_unlimited] aligned"
move "data/aligned/pairs.jsonl"          "data/aligned/pairs${SUFFIX}.jsonl"

echo "[snapshot_unlimited] final deliverables (prefix=${PREFIX})"
move "data/final/${PREFIX}_raw.txt"       "data/final/${PREFIX}_raw${SUFFIX}.txt"
move "data/final/${PREFIX}_parallel.tsv"  "data/final/${PREFIX}_parallel${SUFFIX}.tsv"
move "data/final/${PREFIX}_parallel.xlsx" "data/final/${PREFIX}_parallel${SUFFIX}.xlsx"

echo "[snapshot_unlimited] invalidating downstream checkpoints"
CKPT="data/interim/.checkpoint"
for f in ocr llm_correct llm_correct_skip split_vi labse_embed vecalign export_deliverable; do
    remove "$CKPT/$f"
done

echo
echo "[snapshot_unlimited] done."
