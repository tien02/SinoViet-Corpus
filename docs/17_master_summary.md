# HVB Pipeline — Master Summary (2026-07-21)

Complete build history of Hán-Việt parallel corpus from Đại Nam Thực Lục.
Covers all iterations from initial commit (3c1ef89) to dual-filter quality control (b7904c9).

**Final deliverable:** 33,424 aligned pairs (TSV + XLSX)
**Reproduce:** `./scripts/reproduce_bertalign_bgem3.sh`

---

## Timeline of Major Iterations

| Date | Commit | Milestone | Pairs | Cosine Mean |
|------|--------|-----------|------:|------------:|
| 2026-06 | 3c1ef89 | Initial pipeline (PaddleOCR + Vecalign + LaBSE) | ~3,000 | n/a |
| 2026-06 | 5b3cbd8 | Scope refinement (drop NER + eval stages) | ~3,000 | n/a |
| 2026-06 | 5cbc706 | Switch OCR → Baidu Unlimited-OCR (vLLM) | ~5,000 | n/a |
| 2026-07 | ef4b2af | **Bertalign + BGE-M3** (replace Vecalign+LaBSE) | 5,917 | 0.559 |
| 2026-07 | 6a027dd | **Guwen-biaodian Hán punctuator** (5.3x pairs) | 30,959 | 0.621 |
| 2026-07 | 22ed238 | **`.docx` clean Việt source** (bypass OCR) | 32,665 | 0.628 |
| 2026-07 | 5409608 | MAX_EXTREME_RATIO hard ceiling | 32,665 | 0.628 |
| 2026-07 | fa3b567 | **Dual-filter QC** (surgical + hard ceiling) | **33,424** | **0.624** |

---

## Final Pipeline Architecture

```
┌────────────────────────────────────────────────────────────────────┐
│                  HVB PIPELINE (FINAL — 2026-07-21)                 │
└────────────────────────────────────────────────────────────────────┘

 data/raw/                                   data/raw/
 Đại Nam Thực Lục.txt                       dai-nam-thuc-luc-tap{4,5,6}.docx
 (Wiki文库, 233K lines)                       (clean digitized, 3 volumes)
       │                                              │
       ▼                                              ▼
 ┌──────────────────────┐                  ┌──────────────────────┐
 │ STAGE 1a: normalize  │                  │ STAGE 1b: docx_load  │
 │ normalize_han.py     │                  │ docx_to_vi_sentences │
 │                      │                  │                      │
 │ • strip wiki markers │                  │ • python-docx parse  │
 │ • full→halfwidth     │                  │ • underthesea split  │
 │ • slice 50073-138723 │                  │ • MIN_SENT_LEN=8     │
 └──────────────────────┘                  └──────────────────────┘
       │                                              │
       ▼                                              ▼
 han_clean.txt                              vi_sentences.jsonl
 (181 paragraphs,                           (58,032 sentences)
  ~zero 。)                                         │
       │                                              │
       ▼                                              │
 ┌──────────────────────────────────────────────────┐│
 │ STAGE 2-pre: han_punctuate (NEW 2026-07-08)      ││
 │ han_punctuate.py                                 ││
 │                                                  ││
 │ Model: raynardj/classical-chinese-               ││
 │        punctuation-guwen-biaodian                ││
 │ (BERT token-classifier, 21 labels, fp16)         ││
 │                                                  ││
 │ • sliding window w=300, overlap=50               ││
 │ • +99,753 。 ！ ？ ； inserted                    ││
 └──────────────────────────────────────────────────┘│
       │                                              │
       ▼                                              │
 han_punctuated.txt                                  │
       │                                              │
       ▼                                              │
 ┌──────────────────────┐                            │
 │ STAGE 2: split_han   │                            │
 │ split_han.py         │                            │
 │                      │                            │
 │ • split 。！？；     │                            │
 │ • protect 〈…〉 ann │                            │
 │ • MIN_HAN_LEN=15     │                            │
 │   merge short frags  │                            │
 └──────────────────────┘                            │
       │                                              │
       ▼                                              │
 han_sentences.jsonl                                 │
 (49,927 sentences)                                  │
       │                                              │
       └──────────────────┬───────────────────────────┘
                          ▼
                   ┌──────────────────────┐
                   │ STAGE 3: embed       │
                   │ labse_embed.py       │
                   │                      │
                   │ Model: BAAI/bge-m3   │
                   │ (568M params, fp16,  │
                   │  dim=1024, max_seq=  │
                   │  256, batch=64)      │
                   └──────────────────────┘
                          │
                          ▼
                   {han,vi}_embeds.npy
                   (49,927 + 58,032 = 108K vectors)
                          │
                          ▼
                   ┌──────────────────────────────────┐
                   │ STAGE 4: align                   │
                   │ bertalign_runner.py              │
                   │                                  │
                   │ Bertalign 2-pass DP              │
                   │ + BGE-M3 encoder                 │
                   │ max_align=5, top_k=3, win=5,     │
                   │ skip=-0.1, margin=True,          │
                   │ len_penalty=True                 │
                   │ filter cosine ≥ 0.5              │
                   └──────────────────────────────────┘
                          │
                          ▼
                   pairs.jsonl (35,622 raw pairs)
                          │
                          ▼
                   ┌──────────────────────────────────┐
                   │ STAGE 5: rerank                  │
                   │ scripts/rerank_combined.py       │
                   │                                  │
                   │ • cn2vn Sino-Viet phonetic       │
                   │ • combined score = sino + cos    │
                   └──────────────────────────────────┘
                          │
                          ▼
                   pairs_reranked.jsonl
                          │
                          ▼
                   ┌──────────────────────────────────┐
                   │ STAGE 6: export (FINAL DUAL-FLT) │
                   │ export_deliverable.py            │
                   │                                  │
                   │ Drop rules:                      │
                   │ • > 2000 chars (Excel limit)     │
                   │ • ratio ∉ [0.5, 8.0]             │
                   │ • sino < 0.15                    │
                   │ • NEW: low_conf (ratio>10 ∧      │
                   │        han<10 ∧ sino<0.3)        │
                   │ • NEW: extreme ratio > 15.0      │
                   │                                  │
                   │ Rescue (bertalign only):         │
                   │ • ratio outlier + cos ≥ 0.60     │
                   │ • sino weak + cos ≥ 0.55         │
                   └──────────────────────────────────┘
                          │
                          ▼
         ┌─────────────────────────────────────┐
         │           data/final/               │
         │  hvb_parallel.tsv   33,424 pairs    │
         │  hvb_parallel.xlsx  33,424 pairs    │
         │  hvb_raw.txt        3-tập OCR concat│
         │                                     │
         │  Schema: pair_id ⇥ han ⇥ viet ⇥    │
         │          sino                       │
         │  Cosine mean: 0.624                 │
         │  Sino mean: 0.454                   │
         │  Max ratio: 14.94                   │
         └─────────────────────────────────────┘
```

---

## Phase 1: Foundation (Initial Pipeline)

**Commits:** 3c1ef89 → 5b3cbd8

### Original Stack
- OCR: PaddleOCR (Vietnamese model)
- Embedder: LaBSE
- Aligner: Vecalign (monotonic DP)
- Output: ~3,000 pairs

### Scope Decision (5b3cbd8)
Refactored pipeline to course deliverable scope:
- Removed NER stage (out of scope for alignment task)
- Removed 5-pillar automated eval (kept sino phonetic proxy only)
- Output schema locked: `pair_id ⇥ han ⇥ viet ⇥ sino`

---

## Phase 2: OCR Upgrade (Unlimited-OCR via vLLM)

**Commits:** 21cee27 → 5cbc706 → 07c81b0

### Switch Rationale
- PaddleOCR diacritic loss heavy on century-old Quốc Ngữ scans
- Solution: Baidu Unlimited-OCR (VLM, vLLM-served)
- Setup: Docker container `vllm/vllm-openai:latest` at `http://localhost:8001/v1`
- Model: `baidu/Unlimited-OCR` recipe from recipes.vllm.ai

### vLLM Choice Justification
- PagedAttention → 5-10x faster than Ollama
- OpenAI-compatible API (drop-in for existing OCR code)
- Local GPU (2x RTX 3060 12GB), no cloud dependency
- Auto-restart via `--restart unless-stopped`

---

## Phase 3: Alignment Revolution (Bertalign + BGE-M3)

**Commits:** 1a1d441 → ef4b2af → 4fbb3e0 → 78f9387

### Why Bertalign Over Vecalign
| Aspect | Vecalign | Bertalign |
|--------|----------|-----------|
| Algorithm | Monotonic DP | 2-pass DP + anchor |
| Score | Cosine distance (low=better) | Cosine similarity (high=better) |
| Bead shapes | 1-1, 1-N, N-1, N-N | Same, but unpinned |
| Score range | 0.0-1.0 distance | 0.0-1.0 similarity |
| Filter threshold | `ALIGN_MIN_SCORE=0.5` (dist) | `ALIGN_MIN_SCORE=0.5` (sim) |

**Result on same data:** 5,917 pairs (Bertalign) vs 7,043 (Vecalign), but sino mean 0.500 vs 0.275 = **+82% phonetic precision**.

### Why BGE-M3 Over LaBSE
- BGE-M3: 568M params (BAAI 2024), strong CJK + multilingual
- LaBSE: ~140M params, weaker on Classical Chinese
- Both pipelines measured: BGE-M3 consistently higher cosine on semantic translations

### Optional LLM Post-Correction
- Model: `Qwen/Qwen2.5-7B-Instruct` (vLLM-served)
- Purpose: Fix OCR diacritics before splitting
- Bypass: `HVB_SKIP_LLM_CORRECT=1` (default in production — `.docx` source made this redundant)

---

## Phase 4: Hán Punctuation Restoration (5.3x Growth)

**Commit:** 6a027dd → docs/13

### Problem
Wiki文库 source shipped as 181 chapter-blocks avg 8,317 chars each. **176/181 blocks had zero terminal punctuation**. Old `split_han` chunked at 200 chars → crossed 3-5 topics per chunk → BGE-M3 topical dilution → cosine capped ~0.77.

### Solution
- Model: `raynardj/classical-chinese-punctuation-guwen-biaodian`
- BERT token-classifier (21 labels), trained on 四庫全書
- Sliding window (300 chars, overlap 50)
- Center-vote merge for overlapping windows
- Runtime: ~30s on RTX 3060 fp16

### Post-Processing
Model over-inserts 。 after common characters (議, 賞, 嗣).
- Raw: 80,440 sentences median 14 chars (10k below 5 chars)
- After `_merge_short(MIN_HAN_LEN=15)`: 49,927 sentences median 25 chars

### Impact (Same Embedder + Aligner)

| Metric | 200-char chunker | guwen-biaodian | Δ |
|--------|-----------------:|---------------:|---|
| Hán sentences | 7,048 | **49,927** | 7.1x |
| Aligned pairs | 5,917 | **33,221** | **5.6x** |
| Cosine mean | 0.559 | **0.621** | +11% |
| Cosine max | 0.768 | **0.879** | +14% |
| Cosine ≥ 0.7 | 22 (0.4%) | **4,777 (14.4%)** | 217x |
| 1-1 beads | 55 (0.9%) | **15,549 (46.8%)** | 283x |
| Delivered pairs | 5,856 | **30,959** | **5.3x** |

---

## Phase 5: `.docx` Clean Việt Source (Bypass OCR)

**Commit:** 22ed238 → docs/15

### Source Files
```
data/raw/dai-nam-thuc-luc-tap04.docx   1.6M
data/raw/dai-nam-thuc-luc-tap05.docx   1.2M
data/raw/dai-nam-thuc-luc-tap06.docx   1.5M
```

### Rationale
PaddleOCR-VL-1.6 over-segmented: 66,615 Việt sentences inflated by OCR noise, page-break cuts, layout fragments. Clean `.docx` (digitized from same 3 PDFs) has 12.9% fewer sentences but more *alignable* content.

### Quantitative Comparison

| Metric | PaddleOCR-VL-1.6 | `.docx` clean | Δ |
|--------|------------------:|--------------:|---|
| Việt sentences | 66,615 | 58,032 | -12.9% |
| Delivered TSV pairs | 30,959 | **32,665** | **+5.5%** |
| Cosine mean | 0.621 | **0.628** | +1.1% |
| Sino mean | 0.484 | 0.471 | -2.7% |

### Why Sino Dipped
Clean prose uses more semantic dịch thuần Việt (`千載` → "nghìn năm" not "thiên tải"). Less phonetic overlap, same or better translation. Sino dip = expected byproduct of cleaner source.

---

## Phase 6: Dual-Filter Quality Control (Final Refinement)

**Commits:** 5409608 → fa3b567 → b7904c9 → docs/16

### Problem Identified
After fragment-split enabled, max ratio jumped to 97.4 (bracket-splitting artifacts):
- List brackets `〈一。...二。...〉` decomposed into separate sentences
- Created false alignments: 15-char Hán → 1,461-char Việt (97x ratio)

### Single-Filter Approach Failed
- `MAX_LEN_RATIO=8.0` alone: dropped high-cosine semantic translations (false negatives)
- Remove upper bound: 97x artifacts passed (too permissive)
- **Solution:** Two complementary filters targeting different error modes

### Filter 1: Surgical Low-Confidence (fa3b567)

```python
if DROP_LOW_CONF_OUTLIERS and ratio > 10 and len(han) < 10 and sino < 0.3:
    dropped_low_conf_outlier += 1
    continue
```

**Logic:** 3-way AND catches bracket-splitting noise while preserving high-cosine semantic translations.

**Example kept (rescued by cosine):**
```
千載 → "nghìn năm"
Cosine: 0.75 (high)
Sino: 0.0 (zero phonetic overlap — semantic translation)
Ratio: 1.67 (normal)
Han_len: 2 (very short)
```

**Example dropped:**
```
[15-char Hán fragment] → [1,461-char Việt explanation]
Cosine: 0.48 (marginal)
Sino: 0.0
Ratio: 97.4 (extreme)
```

**Result:** 17 pairs removed

### Filter 2: Hard Ceiling (5409608)

```python
if MAX_EXTREME_RATIO > 0 and ratio > MAX_EXTREME_RATIO:  # default 15.0
    dropped_extreme_ratio += 1
    continue
```

**Logic:** Structural safety valve. Never rescues, even on cosine=1.0.

**Rationale:** Ratios > 15.0 indicate non-monotonic alignment (Vecalign range-merge artifacts or data corruption).

**Result:** 25 pairs removed

### Final Metrics

| Signal | Mean | Median | Min | Max |
|--------|-----:|-------:|----:|----:|
| Cosine similarity | **0.624** | 0.616 | 0.501 | 0.899 |
| Sino precision | 0.454 | 0.429 | 0.0 | 1.0 |
| Length ratio | 3.74 | 3.41 | 0.5 | 14.94 |

**Filter summary:**
- 35,622 raw pairs (Bertalign output)
- -17 surgical low-confidence
- -25 extreme ratio ceiling
- = **33,424 delivered pairs**

---

## Architectural Decisions Locked

1. **Semantic-first philosophy**
   - Cosine similarity (BGE-M3 confidence) = primary signal
   - Sino precision (cn2vn phonetic) = secondary proxy
   - Length ratio = structural sanity check
   - High cosine overrides weak sino/ratio (bertalign rescue)

2. **Bertalign + BGE-M3 stack**
   - 2-pass DP handles local non-monotonicity (page ordering drift)
   - Cosine score field enables rescue logic
   - BGE-M3 fp16 on RTX 3060 (1024-dim, max_seq=256, batch=64)

3. **Dual-filter quality control**
   - Surgical (empirical, 3-way AND): catches bracket artifacts
   - Hard ceiling (structural, never rescues): catches data corruption
   - Single threshold insufficient (either too strict or too lax)

4. **Local embeddings, no cloud LLM**
   - All inference on local GPU (2x RTX 3060 12GB)
   - vLLM Docker for OCR + optional LLM correct
   - Reproducible: no API keys, no rate limits

5. **Fragment splitting retained**
   - List bracket decomposition is a feature, not a bug
   - Outliers managed via dual-filter, not by disabling splitting

---

## Model Provenance

| Stage | Model | Size | Source |
|-------|-------|-----:|--------|
| Hán punctuator | `raynardj/classical-chinese-punctuation-guwen-biaodian` | 110M | HF (trained on 四庫全書) |
| Embedder | `BAAI/bge-m3` | 568M | BAAI 2024 |
| Aligner | Bertalign (custom encoder patch) | — | external/bertalign/ git clone |
| OCR (default) | `baidu/Unlimited-OCR` | VLM | recipes.vllm.ai |
| OCR (current prod) | `.docx` direct digitization | — | user-provided |
| Sino proxy | `cn2vn` | rule-based | PyPI |
| LLM correct (optional) | `Qwen/Qwen2.5-7B-Instruct` | 7B | vLLM docker |

---

## Reproducibility

**Single-shot entry point:**
```bash
./scripts/reproduce_bertalign_bgem3.sh
```

**Stage-by-stage:**
```bash
./scripts/run_pipeline.sh prep     # normalize Hán + PDF/docx load
./scripts/run_pipeline.sh split    # han_punctuate + split_han + split_vi
./scripts/run_pipeline.sh embed    # BGE-M3 embeddings
./scripts/run_pipeline.sh align    # Bertalign 2-pass DP
HVB_ALIGNER=bertalign ./scripts/run_pipeline.sh export
```

**Force full rerun:**
```bash
rm -rf data/interim/.checkpoint/{han_punctuate,split_han,split_vi,labse_embed,bertalign,export_deliverable}
./scripts/reproduce_bertalign_bgem3.sh
```

**Key env vars (defaults in reproduce script):**
```bash
HVB_ALIGNER=bertalign
HVB_EMBED_MODEL=BAAI/bge-m3
HVB_EMBED_MAX_SEQ=256
HVB_EMBED_BATCH=64
HVB_EMBED_FP16=1
HVB_HAN_PUNCT_MODEL=raynardj/classical-chinese-punctuation-guwen-biaodian
HVB_HAN_PUNCT_WINDOW=300
HVB_HAN_PUNCT_OVERLAP=50
HVB_MIN_HAN_LEN=15
ALIGN_MIN_SCORE=0.5
HVB_MIN_SINO=0.15
HVB_MIN_LEN_RATIO=0.5
HVB_MAX_LEN_RATIO=8.0
HVB_SINO_RESCUE_COS=0.55
HVB_RATIO_RESCUE_COS=0.60
HVB_MAX_PAIR_CHARS=2000
HVB_DROP_LOW_CONF=1
HVB_MAX_EXTREME_RATIO=15.0
```

---

## Final Deliverable

```
data/final/hvb_parallel.tsv    33,424 pairs   ⭐ course submission
data/final/hvb_parallel.xlsx   33,424 pairs   ⭐ Excel copy
data/final/hvb_raw.txt         3-tập raw OCR concat
```

**Schema:** `pair_id \t han_sentence \t viet_sentence \t sino`

**Quality metrics:**
- Cosine mean: 0.624
- Sino mean: 0.454 (intentionally lower — semantic translations retained)
- Length ratio max: 14.94 (within hard ceiling)
- 1-1 beads: 46.8% (clean DP alignment)
- Hán coverage: 90.2%

---

## Limitations Carried Forward

### 1. No Ground Truth Labels
No expert Hán-Nôm verification. All metrics are proxy:
- Cosine: circular (uses pipeline embedder)
- Sino: phonetic overlap only (doesn't measure semantic correctness)
- Length ratio: structural sanity check

Cannot compute true precision/recall. Quality via convergent evidence across proxies.

### 2. BGE-M3 Out-of-Distribution
Trained on contemporary text → Classical Chinese + century-old Quốc Ngữ remain OOD. Cosine IQR narrow (~0.06) → weak discrimination between good/bad pairs by cosine alone.

**Future fix:** Fine-tune BGE-M3 on top-band pairs (open work).

### 3. Sino Proxy False Negatives
Semantic translations (`千載` → "nghìn năm") don't create phonetic overlap. ~5-10% of dropped pairs likely valid. Mitigated by rescue logic (cos ≥ 0.55).

### 4. Monotonic Order Assumption
Bertalign 2-pass handles local non-monotonicity, but assumes roughly monotonic order between Hán TXT and Việt PDF/docx. Page order verified at Hán slice boundaries (L50073, L81586, L107115, L138723).

---

## Open Work (Future Iterations)

1. **Mini gold set (100-200 pairs)** — manual Hán-Việt expert labels for real precision/recall. Effort thấp, impact cao.

2. **Chapter-anchor pre-alignment** — split both sides by `Quyển` heading before Bertalign. Reduces drift, may rescue borderline pairs.

3. **Fine-tune BGE-M3** on top-band pairs → boost OOD performance on Classical Chinese ↔ century-old Quốc Ngữ.

4. **Round-trip translation eval** — vLLM Qwen2.5-7B: Hán → Việt (LLM dịch) → chrF/BLEU vs corpus → alternative ground truth.

---

## Documentation Index

| File | Content |
|------|---------|
| `docs/00_problem.md` | Problem statement, rationale |
| `docs/01_setup.md` | Setup guide |
| `docs/02_data.md` | Data schemas |
| `docs/03_pipeline.md` | Original pipeline (pre-punctuator) |
| `docs/05_troubleshooting.md` | Common errors |
| `docs/06_extend.md` | Extension patterns |
| `docs/07_unlimited_ocr.md` | Unlimited-OCR setup |
| `docs/08_results.md` | Initial Vecalign results (legacy) |
| `docs/09_han_pipeline.md` | Original Hán chunker (superseded) |
| `docs/10_fail_cases.md` | Failure case studies |
| `docs/11_current_issues.md` | Open issues |
| `docs/12_final_results.md` | Bertalign baseline 5,856 pairs (superseded) |
| `docs/13_han_punctuator.md` | Guwen-biaodian impact (5.3x growth) |
| `docs/14_pipeline_diagram.md` | ASCII pipeline diagram |
| `docs/15_docx_source_results.md` | `.docx` source switch (+5.5%) |
| `docs/16_final_results_2026-07-21.md` | Dual-filter QC final (33,424 pairs) |
| **`docs/17_master_summary.md`** | **This document — full history** |

---

## Summary

**33,424 pairs** delivered through 6 phases of refinement:

1. **Foundation:** Initial PaddleOCR + Vecalign + LaBSE
2. **OCR upgrade:** Unlimited-OCR via vLLM
3. **Alignment revolution:** Bertalign + BGE-M3
4. **Hán punctuation:** Guwen-biaodian (5.3x growth)
5. **Clean Việt source:** `.docx` bypass OCR (+5.5%)
6. **Dual-filter QC:** Surgical + hard ceiling (final)

**Philosophy:** Semantic > phonetic; structural safety; transparent filters; reproducible pipeline.
