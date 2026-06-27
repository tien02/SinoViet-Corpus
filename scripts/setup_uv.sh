#!/usr/bin/env bash
# Setup HVB project: uv venv + Ollama docker + models
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

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

echo "=== 5. Ollama docker ==="
if ! docker ps --format '{{.Names}}' | grep -q '^ollama$'; then
  docker run -d --name ollama \
    --gpus=all \
    -p 11434:11434 \
    -v ollama:/root/.ollama \
    --restart unless-stopped \
    ollama/ollama:latest
  echo "Waiting 10s for ollama to start..."
  sleep 10
else
  echo "ollama container already running"
fi

echo "=== 6. Pull LLM models ==="
for model in qwen2.5:7b seallm:7b; do
  echo "Pulling $model..."
  docker exec ollama ollama pull "$model"
done

echo "=== DONE ==="
echo "Activate: source .venv/bin/activate"
