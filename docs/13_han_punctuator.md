# Hán punctuation restoration (guwen-biaodian) — 2026-07-08

Pre-split stage `han_punctuate` restores 。 ！ ？ ； ， ： 、 on Classical
Chinese before `split_han` runs. Fixes retrieval collapse caused by
Wikisource's punctuation-free chapter blocks.

Model: **raynardj/classical-chinese-punctuation-guwen-biaodian**
(BertForTokenClassification, 21 labels, trained on 四庫全書 punctuation
restoration). HuggingFace, ~110M params, fp16-safe on RTX 3060.

## Motivation

Wikisource Đại Nam Thực Lục ships as 181 chapter-blocks avg 8,317 chars
each. **176 / 181 have zero terminal punctuation**. Intra-block `\n` =
word-wrap at ~17 chars, not sentence boundary.

Old `split_han._greedy_merge_lines(CHUNK_TARGET=200)` sliced blocks at
arbitrary 200-char boundaries crossing 3–5 topics per chunk. BGE-M3
vector of a mixed-topic chunk → topical dilution → cosine capped ~0.77
vs Vi single-topic sentences.

## Pipeline

```
han_clean.txt (181 paragraphs, ~8k chars each, ~zero 。)
  ↓ han_punctuate  (guwen-biaodian, sliding window w=300, overlap=50)
han_punctuated.txt (99,753 terminators inserted)
  ↓ split_han  (ZH_TERM_RE splits on 。！？；)
han_sentences.jsonl  (49,927 sentences, avg 33 chars)
```

Downstream (embed → bertalign → export) unchanged.

## Config knobs

Added to `src/utils/config.py`:

| Constant | Env override | Default | Effect |
|----------|--------------|---------|--------|
| `HAN_PUNCT` | — | `data/interim/han_punctuated.txt` | Output path |
| `HAN_PUNCT_MODEL` | `HVB_HAN_PUNCT_MODEL` | `raynardj/classical-chinese-punctuation-guwen-biaodian` | HF model id |
| `HAN_PUNCT_WINDOW` | `HVB_HAN_PUNCT_WINDOW` | `300` | Chars per inference window |
| `HAN_PUNCT_OVERLAP` | `HVB_HAN_PUNCT_OVERLAP` | `50` | Overlap chars between windows |
| `HAN_PUNCT_BATCH` | `HVB_HAN_PUNCT_BATCH` | `16` | Windows per GPU batch |
| `HAN_PUNCT_MAX_NEW` | `HVB_HAN_PUNCT_MAX_NEW` | `512` | Reserved for seq2seq fallback |
| `MIN_HAN_LEN` (split_han) | `HVB_MIN_HAN_LEN` | `15` | Merge Han fragments below this into next sentence |

## Idempotency

`han_punctuate.main()` skips when `HAN_PUNCT.mtime > HAN_CLEAN.mtime`.
Delete `data/interim/han_punctuated.txt` + `data/interim/.checkpoint/han_punctuate`
to force rerun.

## Overlap-merge strategy

Per-window prediction returns per-char labels via `offset_mapping`.
Merge: each global char position takes the label from the window whose
center is closest. Interior beats edge → inference stability up.

## Post-split short-fragment merge

Model over-inserts 。 after common characters (議, 賞, 嗣). Raw output:
80,440 sentences median 14 chars, 10k below 5 chars — noise.

`split_han._merge_short(sents, MIN_HAN_LEN=15)` merges any sentence below
threshold forward into next. Trailing tail merges backward. Preserves
monotonic order.

Result: 49,927 sentences median 25 chars, minimum 15.

## Runtime cost

RTX 3060, batch 16, seq 300, fp16:

```
paragraphs: 181
throughput: ~6.2 para/s
wall time:  ~30s
```

Sample output:

```
IN : 又以丁艱亦有輕重而限內槩不預朝賀殊未分別準禮部議嗣凡內外官員有三年喪...
OUT: 又以丁艱亦有輕重，而限內槩不預朝賀，殊未分別。準禮部議。嗣。凡內外官員有三年喪，...
```

Some false 。 (after 議 / 賞 / 嗣). Post-merge threshold hides them —
BGE-M3 sees 15+ char units, not 3-char noise fragments.

## Impact on retrieval

Same corpus, embedder (BGE-M3), aligner (Bertalign). Only Han splitter
changed:

### Aligned pairs (Bertalign kept ≥ 0.5)

| Metric | 200-char chunker (baseline) | guwen-biaodian (new) | Δ |
|--------|----------------------------:|---------------------:|---|
| Han sentences | 7,048 | **49,927** | 7.1x |
| Han:Vi sentence ratio | 1 : 9.0 | **1 : 1.27** | — |
| Aligned pairs kept (≥0.5) | 5,917 | **33,221** | **5.6x** |
| Cosine mean | 0.559 | **0.621** | +11% |
| Cosine max | 0.768 | **0.879** | +14% |
| Cosine ≥ 0.6 | ~866 (14.6%) | **19,121 (57.6%)** | 22x |
| Cosine ≥ 0.7 | 22 (0.4%) | **4,777 (14.4%)** | 217x |
| Cosine ≥ 0.8 | 0 | **201 (0.6%)** | ∞ |
| 1-1 beads | 55 (0.9%) | **15,549 (46.8%)** | 283x |
| 1-N beads | 5,827 (98.5%) | 9,029 (27.2%) | — |
| N-1 beads | 27 | 6,032 | — |
| N-N beads | 8 | 2,611 | — |
| Max bead size | 4 (pinned, DP capped) | 4 (unpinned, mean 1.4) | — |
| Sino mean | 0.500 | 0.467 | -6.6% |
| Combined mean | n/a | 0.544 | — |

### Delivered pairs (post rerank + export filters)

| Metric | Baseline | New | Δ |
|--------|---------:|----:|---|
| Delivered pairs | 5,856 | **30,959** | **5.3x** |
| Han coverage | 84.7% (5,969/7,048) | **90.2% (45,030/49,927)** | +5.5pp |
| Drops (>2000 chars) | 13 | 12 | — |
| Drops (ratio ∉ [0.5, 8.0]) | 36 | 810 | 22x (proportional) |
| Drops (sino < 0.15) | 12 | 1,440 | 120x (proportional) |
| Han chars med | 208 | 36 | — |
| Viet chars med | 632 | 163 | — |
| V/H char ratio mean | 3.38 | 4.62 | +37% |
| Delivered sino mean | 0.500 | 0.484 | -3% |

**Interpretation**:
- Delivered pairs 5.3x baseline. Han coverage 90% of the punctuated
  sentence pool.
- Sino mean drops slightly because many new 1-1 short beads contain few
  name tokens (sino saturates on long records with many proper nouns).
- Ratio outlier + sino drops scale with total pair count → filter
  effectiveness unchanged in relative terms.

### Deliverable files (data/final/)

```
hvb_parallel.tsv   30,959 pairs   (pair_id ⇥ han ⇥ viet ⇥ sino)
hvb_parallel.xlsx  30,959 pairs   (Excel copy)
hvb_raw.txt        3 tập raw OCR concat
```

Two structural wins:

1. **DP no longer capped.** Old 1:9 ratio forced Bertalign DP into
   max_align=5 pinned beads on 98.5% of records. New 1:1.27 ratio →
   1-1 clean beads dominate.
2. **Score distribution widens.** Real translations reach 0.8+ for the
   first time; 14% of pairs cross 0.7 vs 0.4% before.

## Known limitations

- **False 。 after nominal endings** (議, 賞, 嗣). Noise below
  MIN_HAN_LEN=15 gets merged, but slightly reshuffles sentence
  boundaries vs true 記錄 structure.
- **No confidence threshold** on predictions — raw argmax. A
  `logit_threshold` env would let us keep only high-confidence
  terminators (bigger sentences, cleaner but coarser).
- **Model out-of-domain risk**: trained on 四庫全書 (Ming/Qing
  classical). Đại Nam Thực Lục (Nguyen dynasty, same register) is close
  but not identical. Not measured with gold.

## Reproduce

```bash
# Force full rerun from split onward
rm -f data/interim/.checkpoint/{han_punctuate,split_han,split_vi,labse_embed,vecalign,rerank,export_deliverable}
rm -f data/interim/han_punctuated.txt data/interim/han_sentences.jsonl
rm -f data/interim/{han,vi}_embeds.npy data/aligned/pairs.jsonl

./scripts/run_pipeline.sh split
HVB_ALIGNER=bertalign HVB_EMBED_MODEL=BAAI/bge-m3 \
  HVB_EMBED_MAX_SEQ=256 HVB_EMBED_BATCH=64 HVB_EMBED_FP16=1 \
  ALIGN_MIN_SCORE=0.5 CUDA_VISIBLE_DEVICES=1 \
  ./scripts/run_pipeline.sh embed
./scripts/run_pipeline.sh align
./scripts/run_pipeline.sh export
```

Or single-shot:

```bash
./scripts/reproduce_bertalign_bgem3.sh
```

## Files

- `src/03_split/han_punctuate.py` — new module
- `src/03_split/split_han.py` — added `_merge_short`, prefers `HAN_PUNCT`
- `src/utils/config.py:53-64` — HAN_PUNCT + tuning constants
- `scripts/run_pipeline.sh:132,155` — new `han_punctuate` stage in split
- `pyproject.toml` — added `FlagEmbedding` (required by BGE-M3 loader in
  `labse_embed.py`; runtime-installed prior)

## Related docs

- `docs/09_han_pipeline.md` — original 200-char chunker rationale
  (superseded but kept for history).
- `docs/12_final_results.md` — production results after this fix.
