#!/usr/bin/env bash
# Setup HVB project: uv venv + vLLM docker (Qwen2.5-7B-Instruct)
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

VLLM_MODEL="${VLLM_MODEL:-Qwen/Qwen2.5-7B-Instruct}"
VLLM_PORT="${VLLM_PORT:-8000}"

echo "=== 1. uv venv ==="
uv venv --python 3.11

echo "=== 2. uv sync (deps) ==="
uv sync

echo "=== 3. System packages (pdf2image needs poppler) ==="
if ! command -v pdfinfo >/dev/null 2>&1; then
  echo "poppler-utils missing. Install with: sudo apt install poppler-utils"
fi

echo "=== 4. Clone Vecalign ==="
mkdir -p external
if [ ! -d external/vecalign ]; then
  git clone https://github.com/thompsonb/vecalign.git external/vecalign
fi

echo "=== 5. vLLM docker ($VLLM_MODEL on :$VLLM_PORT) ==="
if ! docker ps --format '{{.Names}}' | grep -q '^vllm$'; then
  docker run -d --name vllm \
    --gpus=all \
    -p "${VLLM_PORT}:8000" \
    -v vllm:/root/.cache/huggingface \
    --restart unless-stopped \
    vllm/vllm-openai:latest \
    --model "$VLLM_MODEL" \
    --gpu-memory-utilization 0.9 \
    --max-model-len 4096 \
    --dtype half
  echo "Waiting 30s for vLLM to load model + weights..."
  sleep 30
else
  echo "vllm container already running"
fi

echo "=== 6. vLLM health check ==="
for i in 1 2 3 4 5 6; do
  if curl -fsS "http://localhost:${VLLM_PORT}/v1/models" >/dev/null 2>&1; then
    echo "vLLM ready at http://localhost:${VLLM_PORT}/v1"
    break
  fi
  echo "vLLM not ready (attempt $i/6), waiting 15s..."
  sleep 15
done

echo "=== DONE ==="
echo "Activate: source .venv/bin/activate"
echo "Skip LLM correct: HVB_SKIP_LLM_CORRECT=1 ./scripts/run_pipeline.sh ocr"
