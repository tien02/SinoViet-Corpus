# Evaluation — Sino-Viet Phonetic Proxy

Đánh giá chất lượng aligned corpus HVB bằng phonetic precision proxy. Không có ground truth — chỉ có tín hiệu bên ngoài từ Sino-Vietnamese pronunciation mapping.

## Why manual evaluation is not available

Để có ground truth precision (đúng/sai từng cặp), cần expert annotate 100-200 pairs. Không có nguồn Hán-Nôm expert cho project hiện tại. Do đó chỉ dùng proxy signals:

| Signal | Source | Trust |
|--------|--------|-------|
| Vecalign / Bertalign score | Aligner's own confidence | LOW — self-reported, circular |
| BERTScore / LaBSE cosine | Embedding similarity (same as aligner optimizes for) | LOW — circular |
| **Sino-Viet phonetic precision** (`sino`) | `cn2vn` 13898 Han→Viet mappings (external) | MEDIUM — real signal but still a proxy |

## Phonetic proxy (`scripts/rescore_sino.py`)

Sino precision = số Hán character có对应的 Sino-Viet reading trong Việt translation / tổng số Hán character.

```bash
uv run python scripts/rescore_sino.py
# prints score-band histogram on data/aligned/pairs_reranked.jsonl
```

Bands:

| sino range | Typical content | Trust |
|------------|-----------------|-------|
| 0.0–0.15   | publisher names, page numbers, vecalign range-merge garbage | drop |
| 0.15–0.40  | partial matches, mixed sentences | review |
| 0.40–0.70  | mostly correct sentences w/ some proper nouns | keep |
| 0.70–1.00  | name lists, official titles, dates | very keep |

Export filter `HVB_MIN_SINO=0.15` drops the first band by default. Bump to `0.30` cho high-precision subset, `0` to disable.

## How to decide if a change is an improvement

Without manual labels, use these signals:

1. **Sino-correct correlation** (rescore_sino.py output): higher delta = better proxy
2. **Delivered pair count** (after sino filter): more pairs = better coverage
3. **Fewer low-sino pairs in high-scoring band**: fewer false positives

Trade-off: Sino proxy fails on pure-Vietnamese sentences (paraphrase, idioms, Buddhist-Sanskrit borrowings) → low `sino` despite correct translation. Manual labels needed for those cases.

## Sino-Viet rescue (stage 5b)

`scripts/rerank_combined.py` rescues pairs vecalign rejected when sino precision strong (default `>= 0.40`). Runs after vecalign, before export. Backs up `pairs.jsonl` → `pairs_pre_rerank.jsonl.bak`, writes merged+rescued output sorted by `combined` score descending.

Wired into `scripts/run_pipeline.sh` `align` and `all` stages.
