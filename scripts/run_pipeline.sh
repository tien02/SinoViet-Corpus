#!/usr/bin/env bash
# HVB pipeline runner with checkpointing.
# Usage: ./scripts/run_pipeline.sh [stage_name]
#   stage_name in: prep, ocr, split, embed, align, ner, eval, all (default: all)
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

STAGE="${1:-all}"
CKPT="$ROOT/data/interim/.checkpoint"
mkdir -p "$(dirname "$CKPT")"

run() {
    local name="$1"; shift
    local cmd="$1"; shift
    if [ -f "$CKPT/$name" ] && [ "$STAGE" != "all" ] && [ "$STAGE" != "$name" ]; then
        echo "[skip] $name (checkpoint)"
        return
    fi
    echo "=== $name ==="
    $cmd
    touch "$CKPT/$name"
}

case "$STAGE" in
    prep)
        run normalize_han "uv run python -m src.01_prep.normalize_han"
        run pdf_to_images "uv run python -m src.01_prep.pdf_to_images"
        ;;
    ocr)
        run paddle_ocr "uv run python -m src.02_ocr.paddle_ocr"
        run llm_correct "uv run python -m src.02_ocr.llm_correct"
        ;;
    split)
        run split_han "uv run python -m src.03_split.split_han"
        run split_vi "uv run python -m src.03_split.split_vi"
        ;;
    embed)
        run labse_embed "uv run python -m src.04_embed.labse_embed"
        ;;
    align)
        run vecalign "uv run python -m src.05_align.vecalign_runner"
        ;;
    ner)
        run ner_han "uv run python -m src.06_ner.ner_han"
        run ner_vi "uv run python -m src.06_ner.ner_vi"
        run ner_bridge "uv run python -m src.06_ner.ner_bridge"
        ;;
    eval)
        run auto_metrics "uv run python -m src.07_eval.auto_metrics"
        run flores_sanity "uv run python -m src.07_eval.flores_sanity"
        run round_trip "uv run python -m src.07_eval.round_trip"
        run holdout_mt "uv run python -m src.07_eval.holdout_mt"
        run llm_ensemble "uv run python -m src.07_eval.llm_ensemble_judge"
        run export_corpus "uv run python -m src.07_eval.export_corpus"
        ;;
    all)
        run normalize_han "uv run python -m src.01_prep.normalize_han"
        run pdf_to_images "uv run python -m src.01_prep.pdf_to_images"
        run paddle_ocr "uv run python -m src.02_ocr.paddle_ocr"
        run llm_correct "uv run python -m src.02_ocr.llm_correct"
        run split_han "uv run python -m src.03_split.split_han"
        run split_vi "uv run python -m src.03_split.split_vi"
        run labse_embed "uv run python -m src.04_embed.labse_embed"
        run vecalign "uv run python -m src.05_align.vecalign_runner"
        run ner_han "uv run python -m src.06_ner.ner_han"
        run ner_vi "uv run python -m src.06_ner.ner_vi"
        run ner_bridge "uv run python -m src.06_ner.ner_bridge"
        run auto_metrics "uv run python -m src.07_eval.auto_metrics"
        run flores_sanity "uv run python -m src.07_eval.flores_sanity"
        run round_trip "uv run python -m src.07_eval.round_trip"
        run holdout_mt "uv run python -m src.07_eval.holdout_mt"
        run llm_ensemble "uv run python -m src.07_eval.llm_ensemble_judge"
        run export_corpus "uv run python -m src.07_eval.export_corpus"
        ;;
    *)
        echo "Unknown stage: $STAGE"
        echo "Usage: $0 {prep|ocr|split|embed|align|ner|eval|all}"
        exit 1
        ;;
esac

echo "=== STAGE $STAGE DONE ==="
