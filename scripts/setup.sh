#!/usr/bin/env bash
# === HVB Setup ===
# Assumes deps already installed (python3.11, uv, git, git-lfs, docker, poppler-utils).
# This script only:
#   1. Pre-flight checks (warn only, no sudo install)
#   2. uv venv + uv sync
#   3. Clone vecalign if missing
#   4. Start vLLM docker container on :8000
#   5. Health-check endpoint
#
# Usage:
#   ./scripts/setup.sh           # full
#   ./scripts/setup.sh --check   # pre-flight only
#
# Env overrides:
#   VLLM_MODEL   (default Qwen/Qwen2.5-7B-Instruct)
#   VLLM_PORT    (default 8000)

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

# ---------- 1. Pre-flight (warn only, never install) ----------
log "Pre-flight checks"

command -v python3.11 >/dev/null 2>&1 && ok "python3.11: $(python3.11 --version)" || warn "python3.11 missing — install manually"
command -v uv >/dev/null 2>&1 && ok "uv: $(uv --version)" || warn "uv missing — install manually: curl -LsSf https://astral.sh/uv/install.sh | sh"
command -v git >/dev/null 2>&1 && ok "git: $(git --version)" || warn "git missing"
git lfs version >/dev/null 2>&1 && ok "git-lfs: $(git lfs version 2>&1 | head -1)" || warn "git-lfs missing"
command -v docker >/dev/null 2>&1 && ok "docker: $(docker --version)" || die "docker missing — install manually"
docker info >/dev/null 2>&1 || die "docker daemon not running — sudo systemctl start docker"
command -v pdfinfo >/dev/null 2>&1 && ok "poppler: $(pdfinfo -v 2>&1 | head -1)" || warn "poppler-utils missing — needed by pdf2image"
command -v nvidia-smi >/dev/null 2>&1 && ok "GPU: $(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader | head -1)" || warn "no NVIDIA GPU — CPU fallback will be slow"

[ "$CHECK_ONLY" = "1" ] && { echo ""; ok "Pre-flight complete."; exit 0; }

# ---------- 2. uv venv + sync deps ----------
log "Python venv + deps (uv sync)"
uv venv --python 3.11
uv sync
ok "venv ready at .venv/"

# ---------- 3. Clone vecalign ----------
log "Vecalign (thompsonb/vecalign)"
mkdir -p external
if [ ! -d external/vecalign/.git ]; then
  rm -rf external/vecalign
  git clone https://github.com/thompsonb/vecalign.git external/vecalign
  ok "vecalign cloned"
else
  ok "vecalign already cloned"
fi

# ---------- 4. vLLM docker ----------
# OpenAI-compatible API on :8000 with PagedAttention.
# One model per container. Weights download on first start (~5GB).
log "vLLM docker ($VLLM_MODEL on :$VLLM_PORT)"
if docker ps --format '{{.Names}}' | grep -q '^vllm$'; then
  ok "vllm container already running"
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
  ok "vllm container created + started (first-run weight download ~10-20 min)"
fi

# ---------- 5. Health-check vLLM endpoint ----------
log "vLLM health check (curl /v1/models)"
READY=0
for i in $(seq 1 40); do
  if curl -fsS "http://localhost:${VLLM_PORT}/v1/models" >/dev/null 2>&1; then
    ok "vLLM ready at http://localhost:${VLLM_PORT}/v1"
    READY=1
    break
  fi
  warn "vLLM not ready (attempt $i/40), waiting 30s..."
  sleep 30
done
if [ "$READY" = "0" ]; then
  warn "vLLM still loading after 20 min — check: docker logs vllm --tail 100"
  warn "Look for: 'Application startup complete'."
fi

# ---------- DONE ----------
echo ""
log "Setup complete"
cat <<EOF

vLLM serving: $VLLM_MODEL on http://localhost:${VLLM_PORT}/v1
Override: VLLM_MODEL=... VLLM_PORT=... ./scripts/setup.sh

Next:
  # Smoke test (10 pages):
  HVB_SUBSET=10 HVB_SUBSET_OFFSET=50 ./scripts/run_pipeline.sh prep
  HVB_SUBSET=10 HVB_SUBSET_OFFSET=50 ./scripts/run_pipeline.sh ocr
  HVB_SUBSET=10 HVB_SUBSET_OFFSET=50 ./scripts/run_pipeline.sh split
  HVB_SUBSET=10 HVB_SUBSET_OFFSET=50 ./scripts/run_pipeline.sh embed
  HVB_SUBSET=10 HVB_SUBSET_OFFSET=50 ./scripts/run_pipeline.sh align

  # Skip LLM correct (fast smoke on clean OCR):
  HVB_SKIP_LLM_CORRECT=1 HVB_SUBSET=10 HVB_SUBSET_OFFSET=50 ./scripts/run_pipeline.sh ocr

  # Full pipeline (2-3 days GPU):
  ./scripts/run_pipeline.sh all

Docs: docs/01_setup.md, docs/03_pipeline.md
EOF
