#!/usr/bin/env bash
# HVB pipeline runner with checkpointing.
# Usage: ./scripts/run_pipeline.sh [stage_name]
#   stage_name in: prep, ocr, split, embed, align, export, all (default: all)
#   'all' runs the full pipeline through export (the course deliverable).
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

# Auto-load Unlimited-OCR endpoints written by start_unlimited_ocr.sh.
# Lets `ocr` and `all` stages run seamlessly after container startup
# without manual `export UNLIMITED_OCR_BASE_URLS=…`.
if [ -f "$ROOT/.unlimited_ocr.env" ]; then
    # shellcheck disable=SC1091
    source "$ROOT/.unlimited_ocr.env"
fi

STAGE="${1:-all}"

# Subset-aware checkpoint dir: matches INTERIM path used by Python stages.
# HVB_SUBSET=N → data/interim_subN/.checkpoint
# HVB_SUBSET=N HVB_SUBSET_OFFSET=K → data/interim_subN_offK/.checkpoint
SUBSET_TAG=""
if [ -n "${HVB_SUBSET:-}" ] && [ "${HVB_SUBSET:-0}" != "0" ]; then
    SUBSET_TAG="_sub${HVB_SUBSET}"
    if [ -n "${HVB_SUBSET_OFFSET:-}" ] && [ "${HVB_SUBSET_OFFSET:-0}" != "0" ]; then
        SUBSET_TAG="${SUBSET_TAG}_off${HVB_SUBSET_OFFSET}"
    fi
fi
CKPT="$ROOT/data/interim${SUBSET_TAG}/.checkpoint"
mkdir -p "$CKPT"

# LLM post-correction is OPTIONAL. Default: skip (copy raw OCR through).
# Set HVB_RUN_LLM_CORRECT=1 to enable vLLM correction (requires vLLM docker).
RUN_LLM_CORRECT="${HVB_RUN_LLM_CORRECT:-0}"

# GPUs to shard PaddleOCR across (comma-separated CUDA device ids).
# All 3 docs' pages are sharded round-robin over these GPUs for load balance.
OCR_GPUS="${HVB_OCR_GPUS:-0,1}"

# Run PaddleOCR sharded across OCR_GPUS (one worker per GPU), then combine.
paddle_ocr_parallel() {
    IFS=',' read -ra gpus <<< "$OCR_GPUS"
    local n="${#gpus[@]}"
    echo "Sharding OCR across $n GPU(s): $OCR_GPUS"
    local pids=()
    for i in "${!gpus[@]}"; do
        CUDA_VISIBLE_DEVICES="${gpus[$i]}" \
            uv run python -m src.02_ocr.paddle_ocr --shard "$i" --num-shards "$n" &
        pids+=("$!")
    done
    local rc=0
    for pid in "${pids[@]}"; do
        wait "$pid" || rc=1
    done
    [ "$rc" -eq 0 ] || { echo "OCR worker failed"; return 1; }
    uv run python -m src.02_ocr.paddle_ocr --combine
}

# Run Baidu Unlimited-OCR via vLLM. Multi-GPU: one container per GPU, requests
# round-robin across endpoints. Higher quality than PaddleOCR on Vietnamese
# historical scans. Endpoints come from UNLIMITED_OCR_BASE_URLS (comma-sep),
# UNLIMITED_OCR_BASE_URL, or default http://localhost:8002/v1.
unlimited_ocr_run() {
    local urls="${UNLIMITED_OCR_BASE_URLS:-${UNLIMITED_OCR_BASE_URL:-http://localhost:8002/v1}}"
    IFS=',' read -ra endpoints <<< "$urls"
    local fail=0
    for u in "${endpoints[@]}"; do
        u="${u// /}"  # strip whitespace
        if ! curl -fsS "$u/models" >/dev/null 2>&1; then
            echo "ERROR: Unlimited-OCR endpoint not reachable: $u"
            fail=1
        else
            echo "  ok: $u"
        fi
    done
    if [ "$fail" -ne 0 ]; then
        echo
        echo "Start containers with:  ./scripts/start_unlimited_ocr.sh"
        echo "  (defaults to HVB_OCR_GPUS=0,1 — one container per GPU)"
        return 1
    fi
    uv run python -m src.02_ocr.unlimited_ocr
}

# OCR backend selector: HVB_OCR_BACKEND=unlimited (default) | paddle
OCR_BACKEND="${HVB_OCR_BACKEND:-unlimited}"
run_ocr_backend() {
    case "$OCR_BACKEND" in
        unlimited) unlimited_ocr_run ;;
        paddle)    paddle_ocr_parallel ;;
        *) echo "Unknown HVB_OCR_BACKEND=$OCR_BACKEND (use: unlimited|paddle)"; return 1 ;;
    esac
}

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
        run ocr run_ocr_backend
        if [ "$RUN_LLM_CORRECT" = "1" ]; then
            run llm_correct "uv run python -m src.02_ocr.llm_correct"
        else
            echo "[skip] llm_correct (set HVB_RUN_LLM_CORRECT=1 to enable vLLM post-fix)"
            HVB_SKIP_LLM_CORRECT=1 run llm_correct_skip "uv run python -m src.02_ocr.llm_correct"
        fi
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
    export)
        run export_deliverable "uv run python -m src.07_export.export_deliverable"
        ;;
    all)
        run normalize_han "uv run python -m src.01_prep.normalize_han"
        run pdf_to_images "uv run python -m src.01_prep.pdf_to_images"
        run ocr run_ocr_backend
        if [ "$RUN_LLM_CORRECT" = "1" ]; then
            run llm_correct "uv run python -m src.02_ocr.llm_correct"
        else
            echo "[skip] llm_correct (set HVB_RUN_LLM_CORRECT=1 to enable vLLM post-fix)"
            HVB_SKIP_LLM_CORRECT=1 run llm_correct_skip "uv run python -m src.02_ocr.llm_correct"
        fi
        run split_han "uv run python -m src.03_split.split_han"
        run split_vi "uv run python -m src.03_split.split_vi"
        run labse_embed "uv run python -m src.04_embed.labse_embed"
        run vecalign "uv run python -m src.05_align.vecalign_runner"
        run export_deliverable "uv run python -m src.07_export.export_deliverable"
        ;;
    *)
        echo "Unknown stage: $STAGE"
        echo "Usage: $0 {prep|ocr|split|embed|align|export|all}"
        exit 1
        ;;
esac

echo "=== STAGE $STAGE DONE ==="
