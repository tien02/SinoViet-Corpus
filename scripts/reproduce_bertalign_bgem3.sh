#!/usr/bin/env bash
# Reproduce the Bertalign + BGE-M3 pipeline (best config, 2026-07-07).
#
# Assumes prep/ocr/split stages already done (checkpoints in data/interim/.checkpoint).
# Runs: align (embed + bertalign + rerank) -> export (with length filter).
#
# Output:
#   data/aligned/pairs.jsonl              (5,917 aligned pairs, similarity scores)
#   data/aligned/pairs_reranked.jsonl     (enriched with sino + combined)
#   data/final/hvb_parallel.tsv           (5,826 filtered pairs)
#   data/final/hvb_parallel.xlsx          (same, Excel)
#
# Metrics: sino avg 0.500, length outlier 1.2%, 98.5% 1-N beads.
#
# Usage:
#   ./scripts/reproduce_bertalign_bgem3.sh          # align + export
#   ./scripts/reproduce_bertalign_bgem3.sh export   # export only

set -euo pipefail
cd "$(dirname "$0")/.."

STAGE="${1:-all}"

# Ensure bertalign encoder has HVB_EMBED_MAX_SEQ / HVB_EMBED_FP16 patch
# (external/bertalign is git-clone, not tracked in this repo).
BERTALIGN_ENC="external/bertalign/bertalign/encoder.py"
if [ -f "$BERTALIGN_ENC" ] && ! grep -q "HVB_EMBED_MAX_SEQ" "$BERTALIGN_ENC"; then
    echo "[patch] Applying scripts/patches/bertalign_encoder.py -> $BERTALIGN_ENC"
    cp scripts/patches/bertalign_encoder.py "$BERTALIGN_ENC"
fi

# Aligner + embedder
export HVB_ALIGNER=bertalign
export HVB_EMBED_MODEL=BAAI/bge-m3

# GPU embed settings (fp16 + cap seq len -> 12x speedup vs default)
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export HVB_EMBED_BATCH="${HVB_EMBED_BATCH:-64}"
export HVB_EMBED_MAX_SEQ="${HVB_EMBED_MAX_SEQ:-256}"
export HVB_EMBED_FP16=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# Bertalign score = similarity (higher=better). Keep pairs >= 0.5.
export ALIGN_MIN_SCORE="${ALIGN_MIN_SCORE:-0.5}"

# Export-stage filters
export HVB_MIN_SINO="${HVB_MIN_SINO:-0.15}"
export HVB_MIN_LEN_RATIO="${HVB_MIN_LEN_RATIO:-0.5}"
export HVB_MAX_LEN_RATIO="${HVB_MAX_LEN_RATIO:-8.0}"
export HVB_MAX_PAIR_CHARS="${HVB_MAX_PAIR_CHARS:-2000}"

if [ "$STAGE" = "all" ] || [ "$STAGE" = "align" ]; then
    rm -f data/interim/.checkpoint/vecalign \
          data/interim/.checkpoint/rerank \
          data/interim/.checkpoint/export_deliverable
    ./scripts/run_pipeline.sh align
fi

if [ "$STAGE" = "all" ] || [ "$STAGE" = "export" ]; then
    rm -f data/interim/.checkpoint/export_deliverable
    ./scripts/run_pipeline.sh export
fi

echo ""
echo "=== DONE ==="
echo "  pairs:       data/aligned/pairs.jsonl"
echo "  reranked:    data/aligned/pairs_reranked.jsonl"
echo "  deliverable: data/final/hvb_parallel.{tsv,xlsx}"
