#!/usr/bin/env bash
# === HVB Mighty Setup ===
# One-button bootstrap for Hán-Việt Parallel Corpus pipeline.
#
# Checks + installs:
#   1. Python 3.11, uv, git, git-lfs, docker, poppler-utils
#   2. Project venv + deps via uv sync
#   3. external/vecalign clone (thompsonb/vecalign)
#   4. vLLM docker container serving Qwen2.5-7B-Instruct on :8000
#   5. NVIDIA GPU sanity check
#
# Usage:
#   ./scripts/setup.sh           # full bootstrap
#   ./scripts/setup.sh --check   # pre-flight only, no installs
#
# Env overrides:
#   VLLM_MODEL   (default Qwen/Qwen2.5-7B-Instruct)
#   VLLM_PORT    (default 8000)
#
# Safe to re-run — skips already-done steps.

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

CHECK_ONLY=0
[ "${1:-}" = "--check" ] && CHECK_ONLY=1

VLLM_MODEL="${VLLM_MODEL:-Qwen/Qwen2.5-7B-Instruct}"
VLLM_PORT="${VLLM_PORT:-8000}"

log() { printf "\033[1;36m=== %s ===\033[0m\n" "$*"; }
ok()  { printf "\033[1;32m  ✓ %s\033[0m\n" "$*"; }
warn(){ printf "\033[1;33m  ! %s\033[0m\n" "$*"; }
err() { printf "\033[1;31m  ✗ %s\033[0m\n" "$*" >&2; }
die() { err "$*"; exit 1; }

# ---------- 1. Pre-flight: Python ----------
log "Python 3.11"
if command -v python3.11 >/dev/null 2>&1; then
  ok "python3.11: $(python3.11 --version)"
else
  warn "python3.11 missing"
  if [ "$CHECK_ONLY" = "0" ]; then
    echo "Install: sudo apt install python3.11 python3.11-venv"
    die "Re-run after installing python3.11"
  fi
fi

# ---------- 2. uv ----------
log "uv package manager"
if command -v uv >/dev/null 2>&1; then
  ok "uv: $(uv --version)"
else
  warn "uv missing"
  if [ "$CHECK_ONLY" = "0" ]; then
    curl -LsSf https://astral.sh/uv/install.sh | sh
    # shellcheck disable=SC1091
    source "$HOME/.local/bin/env" 2>/dev/null || true
    command -v uv >/dev/null 2>&1 || die "uv install failed — restart shell or add ~/.local/bin to PATH"
    ok "uv installed: $(uv --version)"
  fi
fi

# ---------- 3. git + git-lfs ----------
log "git + git-lfs"
command -v git >/dev/null 2>&1 || die "git missing — sudo apt install git"
ok "git: $(git --version)"

if ! git lfs version >/dev/null 2>&1; then
  warn "git-lfs missing"
  if [ "$CHECK_ONLY" = "0" ]; then
    sudo apt update && sudo apt install -y git-lfs
    git lfs install
    ok "git-lfs: $(git lfs version)"
  fi
else
  ok "git-lfs: $(git lfs version)"
fi

# ---------- 4. docker ----------
log "Docker"
if ! command -v docker >/dev/null 2>&1; then
  die "docker missing — install: https://docs.docker.com/engine/install/"
fi
docker info >/dev/null 2>&1 || die "docker daemon not running — sudo systemctl start docker"
ok "docker: $(docker --version)"

# ---------- 5. poppler (pdf2image dep) ----------
log "poppler-utils"
if command -v pdfinfo >/dev/null 2>&1; then
  ok "poppler: $(pdfinfo -v 2>&1 | head -1)"
else
  warn "poppler-utils missing"
  if [ "$CHECK_ONLY" = "0" ]; then
    sudo apt update && sudo apt install -y poppler-utils
    ok "poppler-utils installed"
  fi
fi

# ---------- 6. NVIDIA GPU ----------
log "NVIDIA GPU"
if command -v nvidia-smi >/dev/null 2>&1; then
  ok "GPU: $(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader | head -1)"
else
  warn "nvidia-smi missing — pipeline will fall back to CPU (very slow)"
fi

[ "$CHECK_ONLY" = "1" ] && { echo ""; ok "Pre-flight complete."; exit 0; }

# ---------- 7. uv venv + sync deps ----------
log "Python venv + deps (uv sync)"
uv venv --python 3.11
uv sync
ok "venv ready at .venv/"

# ---------- 8. Clone vecalign ----------
log "Vecalign (thompsonb/vecalign)"
mkdir -p external
if [ ! -d external/vecalign/.git ]; then
  rm -rf external/vecalign
  git clone https://github.com/thompsonb/vecalign.git external/vecalign
  ok "vecalign cloned"
else
  ok "vecalign already cloned"
fi

# ---------- 9. vLLM docker ----------
# vLLM serves an OpenAI-compatible API on :8000 with PagedAttention
# (5-10x faster than Ollama). One model per container.
log "vLLM docker container ($VLLM_MODEL on :$VLLM_PORT)"
if docker ps --format '{{.Names}}' | grep -q '^vllm$'; then
  ok "vllm container running"
elif docker ps -a --format '{{.Names}}' | grep -q '^vllm$'; then
  docker start vllm
  ok "vllm container started (was stopped)"
else
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
  echo "  waiting 30s for vLLM to load model + weights..."
  sleep 30
  ok "vllm container created + started"
fi

# ---------- 10. Health-check vLLM endpoint ----------
log "vLLM health check"
for i in 1 2 3 4 5 6; do
  if curl -fsS "http://localhost:${VLLM_PORT}/v1/models" >/dev/null 2>&1; then
    ok "vLLM ready at http://localhost:${VLLM_PORT}/v1"
    break
  fi
  warn "vLLM not ready (attempt $i/6), waiting 15s..."
  sleep 15
done

# ---------- DONE ----------
echo ""
log "Setup complete"
cat <<EOF

vLLM serving: $VLLM_MODEL on http://localhost:${VLLM_PORT}/v1
Override via: VLLM_MODEL=... VLLM_PORT=... ./scripts/setup.sh

Next steps:
  # Smoke test (10 pages, ~5 min):
  HVB_SUBSET=10 HVB_SUBSET_OFFSET=50 ./scripts/run_pipeline.sh prep
  HVB_SUBSET=10 HVB_SUBSET_OFFSET=50 ./scripts/run_pipeline.sh ocr
  HVB_SUBSET=10 HVB_SUBSET_OFFSET=50 ./scripts/run_pipeline.sh split
  HVB_SUBSET=10 HVB_SUBSET_OFFSET=50 ./scripts/run_pipeline.sh embed
  HVB_SUBSET=10 HVB_SUBSET_OFFSET=50 ./scripts/run_pipeline.sh align

  # Skip LLM post-correction (copy raw OCR through):
  HVB_SKIP_LLM_CORRECT=1 HVB_SUBSET=10 HVB_SUBSET_OFFSET=50 ./scripts/run_pipeline.sh ocr

  # Full pipeline (2-3 days GPU):
  ./scripts/run_pipeline.sh all

Activate venv for non-scripted work:
  source .venv/bin/activate

Docs:
  docs/01_setup.md   — detailed install + troubleshooting
  docs/03_pipeline.md — stage-by-stage explanation
EOF
