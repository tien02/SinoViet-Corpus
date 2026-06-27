#!/usr/bin/env bash
# === HVB Mighty Setup ===
# One-button bootstrap for Hán-Việt Parallel Corpus pipeline.
#
# Checks + installs:
#   1. Python 3.11, uv, git, git-lfs, docker, poppler-utils
#   2. Project venv + deps via uv sync
#   3. external/vecalign clone (thompsonb/vecalign)
#   4. Ollama docker container + qwen2.5:7b + seallm:7b
#   5. NVIDIA GPU sanity check
#
# Usage:
#   ./scripts/setup.sh           # full bootstrap
#   ./scripts/setup.sh --check   # pre-flight only, no installs
#
# Safe to re-run — skips already-done steps.

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

CHECK_ONLY=0
[ "${1:-}" = "--check" ] && CHECK_ONLY=1

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

# ---------- 9. Ollama docker ----------
log "Ollama docker container"
if docker ps --format '{{.Names}}' | grep -q '^ollama$'; then
  ok "ollama container running"
elif docker ps -a --format '{{.Names}}' | grep -q '^ollama$'; then
  docker start ollama
  ok "ollama container started (was stopped)"
else
  docker run -d --name ollama \
    --gpus=all \
    -p 11434:11434 \
    -v ollama:/root/.ollama \
    --restart unless-stopped \
    ollama/ollama:latest
  echo "  waiting 10s for ollama to start..."
  sleep 10
  ok "ollama container created + started"
fi

# ---------- 10. Pull LLM models ----------
log "Pull LLM models (~5GB total)"
for model in qwen2.5:7b seallm:7b; do
  if docker exec ollama ollama list | awk '{print $1}' | grep -qx "$model"; then
    ok "$model already pulled"
  else
    echo "  pulling $model..."
    docker exec ollama ollama pull "$model"
    ok "$model pulled"
  fi
done

# ---------- DONE ----------
echo ""
log "Setup complete"
cat <<'EOF'

Next steps:
  # Smoke test (10 pages, ~5 min):
  HVB_SUBSET=10 HVB_SUBSET_OFFSET=50 ./scripts/run_pipeline.sh prep
  HVB_SUBSET=10 HVB_SUBSET_OFFSET=50 ./scripts/run_pipeline.sh ocr
  HVB_SUBSET=10 HVB_SUBSET_OFFSET=50 ./scripts/run_pipeline.sh split
  HVB_SUBSET=10 HVB_SUBSET_OFFSET=50 ./scripts/run_pipeline.sh embed
  HVB_SUBSET=10 HVB_SUBSET_OFFSET=50 ./scripts/run_pipeline.sh align

  # Full pipeline (3-5 days GPU):
  ./scripts/run_pipeline.sh all

Activate venv for non-scripted work:
  source .venv/bin/activate

Docs:
  docs/01_setup.md   — detailed install + troubleshooting
  docs/03_pipeline.md — stage-by-stage explanation
EOF
