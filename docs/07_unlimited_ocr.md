# Stage 2 — Unlimited-OCR (Baidu, via vLLM)

OCR backend for Vietnamese scans of *Đại Nam Thực Lục* (tập 4-6). Replaces
PaddleOCR as the default. Pipeline auto-loads endpoints from
`.unlimited_ocr.env` written by `scripts/start_unlimited_ocr.sh`, so OCR →
split → embed → align runs without manual env exports.

---

## 1. Why Unlimited-OCR (not PaddleOCR, Qwen3-VL, Qwen2.5-Omni, dots.mocr, Qwen3.5-OCR)

### Decision

Default backend: **`baidu/Unlimited-OCR`** served by vLLM, one container per GPU.

### Comparison

| Model                  | Family / vision tower      | Strengths                                                   | Weaknesses on our data                                                                                  | Verdict       |
|------------------------|----------------------------|-------------------------------------------------------------|---------------------------------------------------------------------------------------------------------|---------------|
| **Unlimited-OCR**      | DeepSeek-OCR lineage, SAM-ViT-B + CLIP-L | Document-grounded heads, ngram repetition guard, Vietnamese diacritics preserved, long pages in single shot (8K ctx), vLLM continuous batching | n-gram logits processor needs `--logits_processors` flag; 7-8 GB VRAM per container | **Chosen**    |
| PaddleOCR (PP-OCRv4 vi) | CNN detector + CRNN recog  | CPU-friendly fallback; PP-Structure for layout              | Diacritic loss on faded 1960s offset print; per-line CER spikes on Nôm-mixed pages; needs LLM post-fix to be usable | Fallback (`HVB_OCR_BACKEND=paddle`) |
| Qwen3-VL-2B            | Qwen3-VL                   | Permissive license, fast                                    | 2B reasons poorly on dense classical Vietnamese; hallucinates whole columns when text is faint           | Tested, dropped |
| Qwen2.5-Omni-7B        | Qwen2.5-Omni multimodal    | Strong general VLM                                          | Omni heads pulled toward dialogue/caption style; misses page-edge rubrics; no grounding tokens           | Tested, dropped |
| Qwen3.5-9B-Instruct    | text-only                  | n/a                                                         | text-only, cannot consume images                                                                       | Tested, dropped |
| Qwen3.5-OCR            | OCR-specialized            | OCR-tuned                                                   | Weights gated / not yet released as a stable vLLM checkpoint at our cut-off                              | Tested, dropped |
| dots.mocr              | rednote-hilab              | Strong on Chinese                                           | Latin diacritic vocabulary thin; Vietnamese-specific tones drop                                          | Tested, dropped |

A/B probe scripts were removed from the tree after the decision; recover via
`git log -- src/02_ocr/` if needed.

### Concrete reasons Unlimited-OCR wins for this corpus

1. **Built for document parsing, not chat.** Prompt `"<image>document parsing."`
   emits `<|ref|>…<|/ref|>` text blocks + `<|det|>` bounding boxes. Layout
   stays intact across multi-column pages. Stripped in
   `src/02_ocr/unlimited_ocr.py::clean_output`.
2. **Diacritics survive.** SAM-ViT-B detail tower resolves tone marks (ấ ầ ẩ ẫ
   ậ) on degraded scans where PaddleOCR drops or substitutes them. Direct
   impact on LaBSE similarity → fewer alignment misses in `vecalign`.
3. **Ngram repetition guard.** vLLM extra arg
   `vllm_xargs={"ngram_size": 35, "window_size": 128}` kills the "looped
   line" failure mode that plagues every other open VLM on dense classical
   text. Configured in `config.py`.
4. **Single-shot whole-page OCR.** 8K ctx (`--max-model-len 8192`) handles a
   full page without a slicing strategy. PaddleOCR needs per-line crops; Omni
   drops layout.
5. **vLLM PagedAttention + continuous batching.** Saturates RTX 3060 12 GB
   with concurrent requests far better than ad-hoc HF generate loops.
6. **Per-GPU container scaling.** Stateless OpenAI-compatible endpoint →
   linear scale-out: 1 container per GPU, round-robin in
   `unlimited_ocr.py::ocr_image_round_robin`.

### When to fall back to PaddleOCR

- No internet to pull `baidu/Unlimited-OCR` weights (~7 GB).
- VRAM < 10 GB per GPU.
- Smoke test before docker is up. Run with `HVB_OCR_BACKEND=paddle`.

---

## 2. Setup (2× GPU, RTX 3060 12 GB each)

### One-shot

```bash
# Start one vLLM container per GPU, wait for /models, write .unlimited_ocr.env.
./scripts/start_unlimited_ocr.sh                  # defaults: HVB_OCR_GPUS=0,1
# or pin specific GPUs:
HVB_OCR_GPUS=0,1 ./scripts/start_unlimited_ocr.sh
```

Result:

- `unlimited-ocr-gpu0` on host port `8002` → container `:8000`.
- `unlimited-ocr-gpu1` on host port `8102` → container `:8000`.
- Weights cached in named docker volumes `unlimited-ocr-gpu0`,
  `unlimited-ocr-gpu1` (`/root/.cache/huggingface`). First run pulls ~7 GB
  per GPU (≈10-20 min on a fast link).
- `.unlimited_ocr.env` written in repo root:

```bash
# Auto-generated by start_unlimited_ocr.sh — do not edit.
# Endpoints for Baidu Unlimited-OCR vLLM containers.
export UNLIMITED_OCR_BASE_URLS="http://localhost:8002/v1,http://localhost:8102/v1"
export HVB_OCR_BACKEND=unlimited
```

`run_pipeline.sh` sources this file automatically — no manual export needed.

### Tunables (env vars at startup)

| Var                       | Default | Purpose                                                |
|---------------------------|---------|--------------------------------------------------------|
| `HVB_OCR_GPUS`            | `0,1`   | Comma-separated CUDA ids → one container per id        |
| `HVB_OCR_MEM_UTIL`        | `0.92`  | vLLM `--gpu-memory-utilization`                        |
| `HVB_OCR_MAX_MODEL_LEN`   | `8192`  | vLLM `--max-model-len` (input + output token cap)      |
| `HF_TOKEN`                | (empty) | passed if model gated                                  |

### Tunables (env vars at OCR runtime)

Set before `./scripts/run_pipeline.sh ocr` or `all`, or persist by editing
`src/utils/config.py`:

| Var                            | Default                          | Purpose                                                                  |
|--------------------------------|----------------------------------|--------------------------------------------------------------------------|
| `UNLIMITED_OCR_BASE_URLS`      | written by startup script        | Comma-separated endpoints; round-robined                                |
| `UNLIMITED_OCR_BATCH`          | `16`                             | Concurrent requests **per endpoint**. Effective parallelism = N × BATCH |
| `UNLIMITED_OCR_MAX_TOKENS`     | `4096`                           | Output cap per page                                                     |
| `UNLIMITED_OCR_NGRAM_SIZE`     | `35`                             | Repetition guard window                                                 |
| `UNLIMITED_OCR_WINDOW_SIZE`    | `128`                            | Repetition guard lookback                                               |
| `UNLIMITED_OCR_TIMEOUT`        | `3600`                           | Per-request timeout (sec)                                               |
| `UNLIMITED_OCR_MODEL`          | `baidu/Unlimited-OCR`            | Model id passed to vLLM                                                 |

### Health check

```bash
source .unlimited_ocr.env
for u in ${UNLIMITED_OCR_BASE_URLS//,/ }; do curl -fsS "$u/models" && echo; done
```

### Logs / tear-down

```bash
docker logs -f unlimited-ocr-gpu0
docker logs -f unlimited-ocr-gpu1
docker rm -f unlimited-ocr-gpu0 unlimited-ocr-gpu1   # stop and remove
# keep weights:   docker volume ls | grep unlimited-ocr
# wipe weights:   docker volume rm unlimited-ocr-gpu0 unlimited-ocr-gpu1
```

---

## 3. Run — seamless with downstream stages

After startup, pipeline runs end-to-end without re-exporting anything:

```bash
# Full pipeline (prep → ocr → split → embed → align → export)
./scripts/run_pipeline.sh all

# Or step-by-step
./scripts/run_pipeline.sh prep
./scripts/run_pipeline.sh ocr      # uses Unlimited-OCR by default
./scripts/run_pipeline.sh split
./scripts/run_pipeline.sh embed
./scripts/run_pipeline.sh align    # vecalign — reads OCR-derived sentences
./scripts/run_pipeline.sh export
```

`run_pipeline.sh` at startup:

1. Sources `.unlimited_ocr.env` if present → loads `UNLIMITED_OCR_BASE_URLS`
   and `HVB_OCR_BACKEND=unlimited`.
2. On `ocr` stage, calls `unlimited_ocr_run()`, which `curl`-pings every
   endpoint in `UNLIMITED_OCR_BASE_URLS` and fails fast if any is down.
3. On success, calls `python -m src.02_ocr.unlimited_ocr`. That module
   builds one OpenAI client per endpoint, dispatches pages via
   `ThreadPoolExecutor(max_workers = N_endpoints × UNLIMITED_OCR_BATCH)`,
   and round-robins each page to a starting endpoint. On per-endpoint
   failure, it walks to the next endpoint before counting the page failed.
4. Per-page outputs land in `data/interim/vi_ocr_raw/{tap}_page_{NNNN}.txt`
   and combined `{tap}.txt`. Resumable: pages with an existing output file
   are skipped.
5. LLM post-correction is off by default (`HVB_RUN_LLM_CORRECT=0`). Raw
   Unlimited-OCR output is high enough quality to feed `split_vi` directly
   without Qwen post-fix. Enable with `HVB_RUN_LLM_CORRECT=1` if a gold-set
   CER measurement says otherwise.

Align stage reads `data/interim/vi_sentences.jsonl` produced by `split_vi`,
which reads from `data/interim/vi_ocr_corrected/` (auto-populated by the
`llm_correct` stub when LLM correction is skipped). No manual hand-off
between OCR and align.

---

## 4. Throughput sizing — 2× RTX 3060 12 GB

- One vLLM container per GPU, each at `--gpu-memory-utilization 0.92` with
  `--max-model-len 8192`.
- `UNLIMITED_OCR_BATCH=16` per endpoint → effective parallelism **32**.
- Rough rate observed on a 300-DPI Vietnamese page: ≈1.5-3 s/page/endpoint
  at batch 16. With 2 endpoints: ≈0.8-1.5 s/page wall-clock.
- 3 242 pages (`tap4+5+6`) → expect ~45-80 minutes wall-clock once weights
  are warm.

If batch 16 OOMs:

```bash
HVB_OCR_MEM_UTIL=0.85 ./scripts/start_unlimited_ocr.sh
UNLIMITED_OCR_BATCH=8 ./scripts/run_pipeline.sh ocr
```

If only 1 GPU available:

```bash
HVB_OCR_GPUS=0 ./scripts/start_unlimited_ocr.sh
./scripts/run_pipeline.sh ocr
```

---

## 5. Troubleshooting

| Symptom                                                                 | Cause / fix                                                                                                                       |
|-------------------------------------------------------------------------|-----------------------------------------------------------------------------------------------------------------------------------|
| `ERROR: Unlimited-OCR endpoint not reachable: http://localhost:8002/v1` | Container not up. `docker ps`, then `./scripts/start_unlimited_ocr.sh`.                                                           |
| Container exits immediately                                             | `docker logs unlimited-ocr-gpu0`. Common: `--logits_processors` path mismatch in older vLLM image. Pin tag `vllm/vllm-openai:unlimited-ocr`. |
| Output has `<\|det\|>…<\|/det\|>` boxes left in text                    | `clean_output` regex bypassed — check that `skip_special_tokens=False` is reaching vLLM (it must be, for cleanup to work).         |
| Repeated line loops on faded pages                                      | Lower `UNLIMITED_OCR_NGRAM_SIZE` (e.g. 20) or raise `UNLIMITED_OCR_WINDOW_SIZE` (e.g. 192).                                       |
| OOM at vLLM startup                                                     | Lower `HVB_OCR_MEM_UTIL=0.85` and/or `HVB_OCR_MAX_MODEL_LEN=6144`.                                                                |
| Some pages still empty after run                                        | Failures reported at end of run as `FAIL <page>`. Delete the page's output file and re-run `ocr` stage — resumable.               |
| Want to A/B vs PaddleOCR                                                | `HVB_OCR_BACKEND=paddle ./scripts/run_pipeline.sh ocr` (uses `paddle_ocr_parallel` sharded across `HVB_OCR_GPUS`).                |
