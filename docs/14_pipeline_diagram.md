# Pipeline Diagram — HVB (2026-07-08)

Full stage-by-stage flow of the Hán-Việt alignment pipeline.
Post-punctuator config (docs/13). Supersedes older diagrams in docs/03.

## Models in use

| Stage | Model | Purpose | Config key |
|-------|-------|---------|------------|
| OCR (default) | `baidu/Unlimited-OCR` (vLLM) | Vietnamese scan → text | `UNLIMITED_OCR_MODEL` |
| OCR (alt) | PaddleOCR (`vi`) | Vietnamese fallback OCR | `PADDLE_LANG` |
| LLM post-correct (optional) | `Qwen/Qwen2.5-7B-Instruct` (vLLM :8001) | Fix OCR diacritics | `VLLM_MODEL` |
| **Han punctuator (new)** | `raynardj/classical-chinese-punctuation-guwen-biaodian` | Restore 。 ！ ？ ； ， on Wenyan | `HAN_PUNCT_MODEL` |
| Vi sentence split | `underthesea` sent_tokenize | Vietnamese sent boundary | — |
| Embeddings | `BAAI/bge-m3` (dim=1024, fp16) | Han + Vi joint semantic space | `EMBED_MODEL` |
| Aligner | Bertalign 2-pass DP + BGE-M3 encoder | Cross-lingual sentence alignment | `HVB_ALIGNER=bertalign` |
| Sino proxy | `cn2vn` (rule-based) | Hán-Việt phonetic match | — |

## ASCII diagram

```
┌────────────────────────────────────────────────────────────────────────┐
│                       HVB PIPELINE (2026-07-08)                        │
└────────────────────────────────────────────────────────────────────────┘

  data/raw/                                    data/raw/
  Đại Nam Thực Lục.txt                         Đại Nam Thực Lục.pdf (Tập 4-6)
  (Wikisource, 233k lines)                     (3,242 pages)
        │                                              │
        │                                              │
        ▼                                              ▼
  ╔═══════════════════════╗                    ╔═══════════════════════╗
  ║ STAGE 1a: normalize   ║                    ║ STAGE 1b: pdf→images  ║
  ║ src/01_prep/          ║                    ║ src/01_prep/          ║
  ║ normalize_han.py      ║                    ║ pdf_to_images.py      ║
  ║                       ║                    ║ (pdf2image, 300 DPI)  ║
  ║ • strip wiki markers  ║                    ╚═══════════════════════╝
  ║ • fullwidth→halfwidth ║                              │
  ║ • slice lines 50073-  ║                              ▼
  ║   138723 (Tập 4-6)    ║                    data/interim/vi_pages/
  ╚═══════════════════════╝                    tap{4,5,6}_page_NNNN.png
        │                                              │
        ▼                                              ▼
  data/interim/                                ╔═══════════════════════╗
  han_clean.txt                                ║ STAGE 2: OCR          ║
  (181 paragraphs,                             ║ src/02_ocr/           ║
   avg 8,317 chars,                            ║ unlimited_ocr.py      ║
   ~zero 。)                                   ║ Model: baidu/         ║
        │                                      ║   Unlimited-OCR       ║
        │                                      ║ (vLLM, 2 GPUs)        ║
        │                                      ║                       ║
        │                                      ║ [optional LLM correct ║
        │                                      ║  via Qwen2.5-7B vLLM] ║
        │                                      ╚═══════════════════════╝
        │                                              │
        │                                              ▼
        │                                      data/interim/vi_ocr_raw/
        │                                      tap{N}_page_NNNN.txt
        │                                              │
        ▼                                              │
  ╔═══════════════════════╗ ★ NEW 2026-07-08           │
  ║ STAGE 3a-pre:         ║                            │
  ║ han_punctuate         ║                            │
  ║ src/03_split/         ║                            │
  ║ han_punctuate.py      ║                            │
  ║                       ║                            │
  ║ Model: raynardj/      ║                            │
  ║ classical-chinese-    ║                            │
  ║ punctuation-guwen-    ║                            │
  ║ biaodian (BERT tok-   ║                            │
  ║ classification, 21    ║                            │
  ║ labels, fp16 GPU)     ║                            │
  ║                       ║                            │
  ║ • window=300 chars    ║                            │
  ║ • overlap=50 chars    ║                            │
  ║ • merges via nearest- ║                            │
  ║   center vote         ║                            │
  ║                       ║                            │
  ║ Inserts 。，：；？！   ║                            │
  ║ +99,753 terminators   ║                            │
  ╚═══════════════════════╝                            │
        │                                              │
        ▼                                              │
  data/interim/                                        │
  han_punctuated.txt                                   │
        │                                              │
        ▼                                              ▼
  ╔═══════════════════════╗                    ╔═══════════════════════╗
  ║ STAGE 3a: split_han   ║                    ║ STAGE 3b: split_vi    ║
  ║ src/03_split/         ║                    ║ src/03_split/         ║
  ║ split_han.py          ║                    ║ split_vi.py           ║
  ║                       ║                    ║ Model: underthesea    ║
  ║ • split on 。！？；    ║                    ║   sent_tokenize       ║
  ║ • protect 〈…〉ann.   ║                    ║ • MIN_SENT_LEN=8      ║
  ║ • MIN_HAN_LEN=15      ║                    ║ • rejoin block cuts   ║
  ║   merge short frags   ║                    ╚═══════════════════════╝
  ╚═══════════════════════╝                            │
        │                                              │
        ▼                                              ▼
  data/interim/                                data/interim/
  han_sentences.jsonl                          vi_sentences.jsonl
  49,927 sents, med 25c                        63,401 sents, med 95c
        │                                              │
        └──────────────────┬───────────────────────────┘
                           ▼
                    ╔═══════════════════════╗
                    ║ STAGE 4: embed        ║
                    ║ src/04_embed/         ║
                    ║ labse_embed.py        ║
                    ║                       ║
                    ║ Model: BAAI/bge-m3    ║
                    ║ (FlagEmbedding, fp16, ║
                    ║  dim=1024, max_seq=   ║
                    ║  256, batch=64)       ║
                    ╚═══════════════════════╝
                           │
                           ▼
                    data/interim/
                    {han,vi}_embeds.npy
                           │
                           ▼
                    ╔═══════════════════════╗
                    ║ STAGE 5: align        ║
                    ║ src/05_align/         ║
                    ║ bertalign_runner.py   ║
                    ║                       ║
                    ║ Bertalign 2-pass DP   ║
                    ║ + BGE-M3 encoder      ║
                    ║ max_align=5, top_k=3, ║
                    ║ win=5, skip=-0.1,     ║
                    ║ margin=True,          ║
                    ║ len_penalty=True      ║
                    ║ filter score≥0.5      ║
                    ╚═══════════════════════╝
                           │
                           ▼
                    data/aligned/
                    pairs.jsonl
                    33,221 pairs
                    cos mean 0.621
                    max 0.879
                    46.8% 1-1 beads
                           │
                           ▼
                    ╔═══════════════════════╗
                    ║ STAGE 6: rerank       ║
                    ║ scripts/rerank_       ║
                    ║ combined.py           ║
                    ║                       ║
                    ║ Model: cn2vn          ║
                    ║ (rule-based Sino-Viet)║
                    ║ • add sino score      ║
                    ║ • combined = 0.5·cos  ║
                    ║              + 0.5·sino║
                    ║ • rescue pairs_review ║
                    ║   with sino≥0.4       ║
                    ╚═══════════════════════╝
                           │
                           ▼
                    data/aligned/
                    pairs_reranked.jsonl
                    33,221 enriched
                           │
                           ▼
                    ╔═══════════════════════╗
                    ║ STAGE 7: export       ║
                    ║ src/07_export/        ║
                    ║ export_deliverable.py ║
                    ║                       ║
                    ║ Drop rules:           ║
                    ║ • > 2000 chars   → 12 ║
                    ║ • V/H ratio      →810 ║
                    ║   ∉[0.5, 8.0]         ║
                    ║ • sino < 0.15   →1440 ║
                    ║                       ║
                    ║ Rescue (bertalign):   ║
                    ║ • ratio + cos≥0.60    ║
                    ║ • sino  + cos≥0.55    ║
                    ╚═══════════════════════╝
                           │
                           ▼
        ┌─────────────────────────────────────┐
        │           data/final/               │
        │  ┌───────────────────────────────┐  │
        │  │ hvb_parallel.tsv              │  │
        │  │ 30,959 pairs                  │  │
        │  │ pair_id ⇥ han ⇥ viet ⇥ sino  │  │
        │  ├───────────────────────────────┤  │
        │  │ hvb_parallel.xlsx (Excel)    │  │
        │  │ hvb_raw.txt (concat OCR)     │  │
        │  └───────────────────────────────┘  │
        │  90.2% Han coverage                 │
        └─────────────────────────────────────┘
```

## Checkpoint files

`data/interim/.checkpoint/{stage_name}` guard re-runs.

Force full rerun:
```bash
rm data/interim/.checkpoint/{han_punctuate,split_han,split_vi,labse_embed,vecalign,rerank,export_deliverable}
./scripts/reproduce_bertalign_bgem3.sh
```

## Model provenance

- **baidu/Unlimited-OCR** — Baidu, 2024. VLM for scanned Vietnamese pages. Recipe: recipes.vllm.ai/baidu/Unlimited-OCR.
- **Qwen/Qwen2.5-7B-Instruct** — Alibaba, 2024. Post-correct OCR diacritics. Only used when `HVB_RUN_LLM_CORRECT=1`.
- **raynardj/classical-chinese-punctuation-guwen-biaodian** — HuggingFace, BERT token-classifier, trained on 四庫全書 (Ming/Qing classical corpus). 21 labels (0=O, 1=。, 2=，, ..., 20=】).
- **underthesea** ≥6.8 — Vietnamese NLP toolkit.
- **BAAI/bge-m3** — BAAI, 2024. 568M params, 1024-dim. CJK + low-resource multilingual encoder. Chosen over LaBSE for classical Chinese support.
- **cn2vn** ≥0.2 — Rule-based Sino-Vietnamese phonetic converter.

## References

- Config: `src/utils/config.py`
- Stage code: `src/{01_prep,02_ocr,03_split,04_embed,05_align,07_export}/`
- Runner: `scripts/run_pipeline.sh`
- Reproduce: `scripts/reproduce_bertalign_bgem3.sh`
- Results: `docs/13_han_punctuator.md`
