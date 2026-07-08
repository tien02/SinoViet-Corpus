"""Central config: paths, model IDs, device, batch sizes."""
from __future__ import annotations

import os
import warnings
from pathlib import Path

# Silence pynvml→nvidia-ml-py redirector FutureWarning (torch CUDA init triggers it).
warnings.filterwarnings("ignore", message="The pynvml package is deprecated.*")

ROOT = Path(__file__).resolve().parents[2]

DATA = ROOT / "data"
RAW = DATA / "raw"

# Subset mode: HVB_SUBSET=N selects N pages of tap4 + limited Hán chars.
# HVB_SUBSET_OFFSET=K skips first K pages (avoid cover/blank front matter).
# Outputs go to data/{interim,aligned,final}_subN[_offK]/ to keep full-run clean.
SUBSET_N = int(os.environ.get("HVB_SUBSET", "0") or "0")
SUBSET_OFFSET = int(os.environ.get("HVB_SUBSET_OFFSET", "0") or "0")
SUBSET_HAN_CHARS = 500 * SUBSET_N  # rough: 500 Han chars per Viet page
SUBSET_HAN_OFFSET = 500 * SUBSET_OFFSET

if SUBSET_N > 0:
    tag = f"sub{SUBSET_N}" if SUBSET_OFFSET == 0 else f"sub{SUBSET_N}_off{SUBSET_OFFSET}"
    INTERIM = DATA / f"interim_{tag}"
    ALIGNED = DATA / f"aligned_{tag}"
    FINAL = DATA / f"final_{tag}"
else:
    INTERIM = DATA / "interim"
    ALIGNED = DATA / "aligned"
    FINAL = DATA / "final"

GOLD = DATA / "gold"

HAN_TXT = RAW / "Đại Nam Thực Lục - 大南寔錄_full.txt"

# Hán line slice — restrict Hán TXT to lines covering only Việt Tập 4-6.
# Verified via title-page markers (Quyển CXVII/CLXXVII/Đệ Tam Kỷ):
#   Tập 4 = Kỷ 2 Q.117 (line 50073) → Q.176 end (line 80943)
#   Tập 5 = Kỷ 2 Q.177 (line 81590) → end Kỷ 2 (line 107115)
#   Tập 6 = Kỷ 3 Q.1 (line 107116) → end Kỷ 3 (line 138723)
# Set both to 0 to disable slicing (use full Hán text).
HAN_LINE_START = int(os.environ.get("HVB_HAN_LINE_START", "50073"))
HAN_LINE_END = int(os.environ.get("HVB_HAN_LINE_END", "138723"))
VI_PDF_TAP4 = RAW / "Đại Nam Thực Lục tập 4 - Quốc Sử Quán Triều Nguyễn.pdf"
VI_PDF_TAP5 = RAW / "Đại Nam Thực Lục tập 5 - Quốc Sử Quán Triều Nguyễn.pdf"
VI_PDF_TAP6 = RAW / "Đại Nam Thực Lục tập 6 - Quốc Sử Quán Triều Nguyễn.pdf"
VI_PDFS = [VI_PDF_TAP4, VI_PDF_TAP5, VI_PDF_TAP6]
if SUBSET_N > 0:
    VI_PDFS = [VI_PDF_TAP4]

HAN_CLEAN = INTERIM / "han_clean.txt"
# Punctuation-restored Han (guwen-biaodian pass on han_clean).
# When present, split_han reads this instead of HAN_CLEAN — classical Chinese
# without terminal punct otherwise falls into 200-char chunk fallback,
# destroying semantics.
HAN_PUNCT = INTERIM / "han_punctuated.txt"
HAN_PUNCT_MODEL = os.environ.get(
    "HVB_HAN_PUNCT_MODEL", "raynardj/classical-chinese-punctuation-guwen-biaodian"
)
HAN_PUNCT_WINDOW = int(os.environ.get("HVB_HAN_PUNCT_WINDOW", "300"))
HAN_PUNCT_OVERLAP = int(os.environ.get("HVB_HAN_PUNCT_OVERLAP", "50"))
HAN_PUNCT_BATCH = int(os.environ.get("HVB_HAN_PUNCT_BATCH", "16"))
HAN_PUNCT_MAX_NEW = int(os.environ.get("HVB_HAN_PUNCT_MAX_NEW", "512"))
VI_PAGES_DIR = INTERIM / "vi_pages"
VI_OCR_RAW_DIR = INTERIM / "vi_ocr_raw"
VI_OCR_CORRECTED_DIR = INTERIM / "vi_ocr_corrected"
HAN_SENT = INTERIM / "han_sentences.jsonl"
VI_SENT = INTERIM / "vi_sentences.jsonl"
HAN_EMBEDS = INTERIM / "han_embeds.npy"
VI_EMBEDS = INTERIM / "vi_embeds.npy"

PAIRS_JSONL = Path(os.environ["HVB_PAIRS_OUT"]) if os.environ.get("HVB_PAIRS_OUT") else ALIGNED / "pairs.jsonl"

# Course deliverable files: {prefix}_raw.txt, {prefix}_parallel.tsv/.xlsx.
# Set prefix to your mã số sinh viên, e.g. HVB_DELIVERABLE_PREFIX=21127001.
DELIVERABLE_PREFIX = os.environ.get("HVB_DELIVERABLE_PREFIX", "hvb")
DELIVERABLE_RAW = FINAL / f"{DELIVERABLE_PREFIX}_raw.txt"
DELIVERABLE_TSV = FINAL / f"{DELIVERABLE_PREFIX}_parallel.tsv"
DELIVERABLE_XLSX = FINAL / f"{DELIVERABLE_PREFIX}_parallel.xlsx"

OCR_GOLD = GOLD / "ocr_gold"

import torch  # noqa: E402

CUDA_VISIBLE = torch.cuda.device_count()
DEVICE = "cuda" if CUDA_VISIBLE > 0 else "cpu"

PADDLE_LANG = "vi"
PADDLE_USE_GPU = CUDA_VISIBLE > 0
OCR_DPI = 300

LABSE_MODEL = "sentence-transformers/LaBSE"  # legacy alias
# Sentence-embedding backbone. BGE-M3 (BAAI, 2024) — 568M params, 1024-dim,
# stronger CJK + low-resource than LaBSE. Drop-in via sentence-transformers.
# Override with HVB_EMBED_MODEL (e.g. sentence-transformers/LaBSE).
EMBED_MODEL = os.environ.get("HVB_EMBED_MODEL", "BAAI/bge-m3")
EMBED_MAX_SEQ = int(os.environ.get("HVB_EMBED_MAX_SEQ", "512"))
EMBED_BATCH = int(os.environ.get("HVB_EMBED_BATCH", "32"))

VECALIGN_REPO = ROOT / "external" / "vecalign"
BERTALIGN_REPO = ROOT / "external" / "bertalign"
# Aligner backend: 'vecalign' (default, monotonic DP) | 'bertalign' (two-pass,
# handles non-monotonic drift).
ALIGNER = os.environ.get("HVB_ALIGNER", "vecalign")
ALIGN_MIN_SCORE = float(os.environ.get("ALIGN_MIN_SCORE", "0.5"))
# Vecalign emits raw cosine DISTANCE (lower = better; see
# external/vecalign/dp_utils.py:282). Filter keeps pairs with
# distance < ALIGN_MAX_DIST in pairs.jsonl; the rest goes to
# pairs_review.jsonl for manual audit / QA back-translation.
ALIGN_MAX_DIST = float(os.environ.get("ALIGN_MAX_DIST", "0.7"))
PAIRS_REVIEW_JSONL = ALIGNED / "pairs_review.jsonl"

# LLM backend for optional OCR post-correction (Stage 2b): vLLM, OpenAI-compatible.
# Qwen2.5-7B-Instruct. Set HVB_SKIP_LLM_CORRECT=1 to copy raw OCR → corrected and
# skip the vLLM call (when vLLM unavailable or for fast smoke tests).
VLLM_BASE_URL = os.environ.get("VLLM_BASE_URL", "http://localhost:8001/v1")
VLLM_API_KEY = os.environ.get("VLLM_API_KEY", "dummy")  # vLLM ignores key, OpenAI client requires non-empty
VLLM_MODEL = os.environ.get("VLLM_MODEL", "Qwen/Qwen2.5-7B-Instruct")
LLM_TIMEOUT = 180
LLM_TEMPERATURE = 0.2
LLM_MAX_TOKENS = 4096
SKIP_LLM_CORRECT = bool(os.environ.get("HVB_SKIP_LLM_CORRECT", ""))

PADDLE_BATCH = 8

# Unlimited-OCR (Baidu) via vLLM — replacement for PaddleOCR.
# Served on separate port (8002) so Qwen LLM-correct container (8001) can coexist.
# Recipe: https://recipes.vllm.ai/baidu/Unlimited-OCR
#
# Multi-GPU speedup: start one vLLM container per GPU (see
# scripts/start_unlimited_ocr.sh) and list endpoints in UNLIMITED_OCR_BASE_URLS
# (comma-separated). The OCR stage round-robins requests across endpoints.
UNLIMITED_OCR_MODEL = os.environ.get("UNLIMITED_OCR_MODEL", "baidu/Unlimited-OCR")
UNLIMITED_OCR_BASE_URL = os.environ.get(
    "UNLIMITED_OCR_BASE_URL", "http://localhost:8002/v1"
)
# Optional: comma-separated list of endpoints for round-robin across containers.
# If set, overrides UNLIMITED_OCR_BASE_URL.
UNLIMITED_OCR_BASE_URLS = [
    u.strip()
    for u in os.environ.get("UNLIMITED_OCR_BASE_URLS", "").split(",")
    if u.strip()
]
UNLIMITED_OCR_API_KEY = os.environ.get("UNLIMITED_OCR_API_KEY", "EMPTY")
UNLIMITED_OCR_TIMEOUT = int(os.environ.get("UNLIMITED_OCR_TIMEOUT", "3600"))
UNLIMITED_OCR_MAX_TOKENS = int(os.environ.get("UNLIMITED_OCR_MAX_TOKENS", "4096"))
UNLIMITED_OCR_NGRAM_SIZE = int(os.environ.get("UNLIMITED_OCR_NGRAM_SIZE", "35"))
UNLIMITED_OCR_WINDOW_SIZE = int(os.environ.get("UNLIMITED_OCR_WINDOW_SIZE", "128"))
# Per-endpoint concurrency. With 2 endpoints on 2 GPUs, effective parallelism = 2*BATCH.
# Default 16 saturates one RTX 3060 with max-model-len 8192.
UNLIMITED_OCR_BATCH = int(os.environ.get("UNLIMITED_OCR_BATCH", "16"))


def ensure_dirs() -> None:
    for d in [
        RAW, INTERIM, ALIGNED, GOLD, FINAL,
        VI_PAGES_DIR, VI_OCR_RAW_DIR, VI_OCR_CORRECTED_DIR,
        OCR_GOLD,
    ]:
        d.mkdir(parents=True, exist_ok=True)


if __name__ == "__main__":
    ensure_dirs()
    print(f"ROOT={ROOT}")
    print(f"DEVICE={DEVICE} CUDA_VISIBLE={CUDA_VISIBLE}")
    print(f"HAN_TXT exists: {HAN_TXT.exists()}")
    for p in VI_PDFS:
        print(f"  {p.name}: {p.exists()}")
