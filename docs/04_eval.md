# Evaluation Methodology

Đánh giá chất lượng aligned corpus HVB — manual stratified sample + optional phonetic proxy.

> **Scope note (2026-07):** The earlier 5-pillar automated eval (COMET, FLORES,
> round-trip, hold-out MT, LLM-judge) was removed — it was never wired up and
> the auto-metrics were all self-reported (no ground truth). What's left is
> honest: a small manually-labeled sample plus two signals computed on the
> full pairs file. This file documents the new reality.

## Why not trust self-reported metrics

- **Vecalign / Bertalign score** = the aligner's own confidence. If the
  embedder mis-encodes a sentence, the aligner can still report high
  confidence on a wrong pair. Not a quality signal, only a rank signal.
- **BERTScore / LaBSE cosine** on aligned pairs = "are the two sides
  semantically similar", which is exactly what the aligner optimised for.
  Circular.
- **Sino-Viet phonetic precision** (`sino` column in deliverable) = a real
  *external* signal — it comes from `cn2vn`'s 13898 Han→Viet mappings, not
  from the aligner. High `sino` on a pair = the Việt side contains the
  actual Sino-Viet readings of the Han characters. Strong
  translation-correctness proxy for this specific domain (Sino-Vietnamese
  is how Classical Chinese was historically rendered into Vietnamese).
  But it's still a proxy, not ground truth — short name lists inflate it.

The only honest ground truth is **a human reading the pair and deciding
correct/incorrect**. The framework below makes that tractable.

## Manual eval framework

Two scripts, one workflow:

### 1. Build the sample — `scripts/build_eval_sample.py`

Stratified 100-pair sample from `data/aligned/pairs_reranked.jsonl`. Strata
by `combined` score (0.5·dist_sim + 0.5·sino):

| Stratum | Combined range | Sample size | What it tests |
|---------|----------------|-------------|---------------|
| high    | ≥ 0.80         | 30          | Should be ~all correct |
| mid     | 0.55–0.80      | 40          | Mostly correct, some noise |
| low     | < 0.55         | 30          | Mostly noise / hard cases |

```bash
uv run python scripts/build_eval_sample.py
# -> data/eval/eval_sample_100.csv
```

CSV columns: `pair_id, stratum, han, viet, sino, combined, label, notes`
(`label` and `notes` empty — annotator fills).

### 2. Label it

Open `data/eval/eval_sample_100.csv` in any spreadsheet. For each row:

- `label = 1` if `viet` is a faithful translation of `han`
- `label = 0` if not (wrong sentence, partial match, garbage, etc.)
- `notes` — optional, one-line reason for 0s

Takes ~45 minutes for 100 pairs once you get going. Tolerance for ambiguous
cases: label 1 if the *core meaning* is preserved, even if the translation
is liberal.

### 3. Score it — `scripts/score_eval.py`

```bash
uv run python scripts/score_eval.py
# -> per-stratum precision + bootstrap 95% CI + sino-correct correlation
# optional JSON: --json data/eval/eval_summary.json
```

Output looks like:

```
labeled: 100 / 100 (0 still unlabeled)

  high  n= 30  correct= 28  precision=0.933  CI95=[0.825, 1.000]
  mid   n= 40  correct= 30  precision=0.750  CI95=[0.625, 0.875]
  low   n= 30  correct=  8  precision=0.267  CI95=[0.133, 0.433]

  ALL   n=100  correct= 66  precision=0.660  CI95=[0.560, 0.750]

  sino mean: correct=0.580  wrong=0.230  delta=+0.350
```

## How to decide if a change is an improvement

This is the question the manual eval framework exists to answer.

**Rule:** A change is an improvement iff the eval-sample precision goes up
**outside the bootstrap CI** of the previous run, on the same sample.

Workflow when you change anything in the align/embed/rerank pipeline:

1. Re-run the affected stages (embed, align, rerank, export).
2. Re-build the eval sample **with the same `--seed`** so you label the
   same 100 pairs (only their `combined` score changes — pair_ids stay
   stable because they come from `pairs_reranked.jsonl` row order).
3. Re-label. Many cells won't change; copy labels from the prior CSV for
   pairs whose `han`+`viet` are identical.
4. Run `score_eval.py`. Compare:
   - overall precision + CI to prior run
   - per-stratum precision (a change that lifts the *mid* stratum is more
     interesting than one that lifts *high* — *high* is already saturated)

**Tie-breakers when precision is similar:**
- Prefer the run with **fewer rescued pairs** (lower reliance on the
  Sino-Viet heuristic = more semantic-aligned = more robust).
- Prefer the run with **higher sino-correct delta** (the phonetic proxy
  correlates better with manual labels → more trustworthy as a filter).
- Prefer the run that drops more `low`-stratum pairs at export time
  (precision via filtering, not via rescue).

**What does NOT count as improvement:**
- Higher vecalign/bertalign score on its own (self-reported).
- Higher mean LaBSE cosine on its own (circular).
- More total pairs in `pairs.jsonl` on its own (could be rescued noise).
- A bigger model that takes longer to run, unless it moves precision.

## Phonetic proxy (full-corpus, no manual labeling)

When you don't have labels yet, `sino` precision (computed on every pair,
output as a column in the deliverable TSV/XLSX) is the next-best signal:

```bash
uv run python scripts/rescore_sino.py
# prints score-band histogram on data/aligned/pairs.jsonl
```

Bands worth watching:

| sino range | Typical content | Trust |
|------------|-----------------|-------|
| 0.0–0.15   | publisher names, page numbers, vecalign range-merge garbage | drop |
| 0.15–0.40  | partial matches, mixed sentences | review |
| 0.40–0.70  | mostly correct sentences w/ some proper nouns | keep |
| 0.70–1.00  | name lists, official titles, dates | very keep |

The export filter `HVB_MIN_SINO=0.15` drops the first band by default.
Bump to `0.30` for a high-precision subset, `0` to disable.

## Sino-Viet rescue (stage 5b)

`scripts/rerank_combined.py` rescues pairs vecalign rejected when their
sino precision is strong (default `>= 0.40`). Runs after vecalign, before
export. Backs up `pairs.jsonl` → `pairs_pre_rerank.jsonl.bak`, writes
merged+rescued output sorted by `combined` score descending.

Wired into `scripts/run_pipeline.sh` `align` and `all` stages.

## Why this beats auto-eval for this corpus

- **Domain mismatch:** FLORES-200 (modern zh-vi news) cannot validate a
  Classical Chinese → Vietnamese royal annals corpus. They share writing
  system, not vocabulary.
- **No reference translation:** round-trip MT eval requires a MarianMT
  fine-tuned on Classical Chinese → Vietnamese, which doesn't exist.
  Round-trip via Qwen is a hallucination check, not a faithfulness check.
- **LLM judge bias:** a single Qwen-7B judge agrees with itself, not with
  humans, on this domain. Krippendorff α needs ≥2 independent judges.
- **Cheap ground truth:** 100 manually-labeled pairs gives a 95% CI of
  ±~10pp on precision. Enough to compare two runs. 200 pairs gives ±7pp.
  Diminishing returns past that.

## Limitations of this framework

- **Sample size:** 100 pairs, ±10pp CI. Detects big moves (>15pp), misses
  small ones.
- **Stratification drift:** if the score distribution shifts a lot between
  runs, the same 100 sample positions cover different parts of the curve.
  Re-stratify from scratch if `combined` distribution shifts >0.1 mean.
- **Annotator drift:** label the entire CSV in one sitting to stay
  consistent. Spread across days → noise.
- **Sino-Viet proxy fails on pure-Vietnamese sentences:** paraphrase
  translations, idioms, and Buddhist-Sanskrit borrowings all have low `sino`
  despite being correct translations. Manual labels are the only check
  there.
