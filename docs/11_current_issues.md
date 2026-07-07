# Current Issues (2026-07-07)

Pipeline status after BGE-M3 + rerank + sino filter + split fix.
Delivered: **11 678 pairs** (sino≥0.15 filter on 44 718 aligned).

## Issue 1 — Embedder fails on Classical→Quốc ngữ translation

**Severity:** Critical — root cause of poor alignment quality.

**Symptom:** 99.95% of aligned pairs in `low` band (combined<0.55).
Top-scoring pairs are garbage:

```
combined=0.723 sino=1.00
  H: 子攸,廕恩騎尉。 (Tử Du, ấm ân kỵ uý — name + title)
  V: Nhân sắc dụ: Từ Nghệ An ra Bắc, phàm có những quan to ở Kinh...
```

**Cause:** BGE-M3 (and LaBSE) do not capture Classical Chinese →
Quốc ngữ translation equivalence. Both embedders were trained on
modern multilingual data; Han-Viet classical diglossia is
out-of-distribution.

**Evidence:**
- BGE-M3 top pair distance score = 0.54 (similarity), unrelated content
- LaBSE top pair distance score = 1.21, equally unrelated content
- Both embedders fail in different ways → not embedder-specific

**Candidate fixes:**
1. Cross-lingual encoder fine-tuned on Classical→Quốc ngữ (may not exist)
2. Multilingual MT model embeddings (Qwen2.5-7B, NLLB)
3. Sino-primary alignment (replace cosine distance with sino precision)
4. Iterative coarse-to-fine (chapter-heading anchor → within-chapter align)

---

## Issue 2 — Sino proxy false positives on short Hán + long Việt

**Severity:** High — corrupts `combined` ranking and rescue logic.

**Symptom:** Short Hán (2-5 chars) saturates sino=1.0 because every
character finds a Sino-Viet reading *somewhere* in a long Việt sentence.

**Example:**
```
H: 子情。                    ← 2 Hán chars
S: Tử Tình                  ← 2 Sino-Viet readings
V: Tình hình gian khổ...     ← long Việt, contains "Tình"
sino = 2/2 = 1.00            ← false positive
```

**Impact:**
- `combined = 0.5·dist + 0.5·sino` saturates for short Hán
- Rescue (`sino≥0.40`) pulls wrong pairs from review pool
- Top of `pairs_reranked.jsonl` dominated by short-Hán false positives

**Candidate fixes:**
1. Length-normalize sino: `matches / max(len_han, len_viet_sino_syllables)`
2. Require dense match (consecutive syllables), not sparse
3. Cap sino contribution when `len(han) < N` characters
4. Weight by Việt sentence length inverse

---

## Issue 3 — Vecalign score semantics unclear across embedders

**Severity:** Medium — complicates threshold tuning.

**Symptom:**
- BGE-M3 vecalign outputs `score=0.54` (looks like similarity ∈ [0,1])
- LaBSE vecalign outputs `score=1.21` (looks like similarity > 1.0)
- Same `vecalign.py` code, different embedder → different scale

**Impact:** `score≥0.5` threshold not portable. Cannot A/B compare
runs across embedders without normalization.

**Candidate fixes:**
1. Standardize to similarity ∈ [0,1] in `vecalign_runner.py`
2. Document score formula in code
3. Add unit test verifying score range

---

## Issue 4 — Hán split still has 7 oversized sentences (>2000 chars)

**Severity:** Low — affects 0.01% of sentences.

**Symptom:** Max 4868 chars, 7 sentences >2000 chars remain after
split fix.

**Cause:** These are single-line long paragraphs in raw text with
NO internal newlines to split on. The fallback `_greedy_merge_lines`
requires newlines.

**Candidate fixes:**
1. Hard-split oversized blocks at terminators regardless of merge
2. Sentence-aware transformer splitter (e.g. PySBD with Chinese rules)
3. Manual annotation of these 7 sentences as atomic

---

## Issue 5 — No ground truth / no expert labels

**Severity:** High — blocks honest evaluation.

**Symptom:** Cannot compute precision. Only have sino proxy.

**Cause:** No Hán-Nôm expert available for project.

**Impact:**
- Cannot definitively rank two runs beyond raw counts
- Sino proxy is the only signal, and it has known failure modes
  (Issue 2)
- Cannot validate that changes are improvements

**Candidate fixes:**
1. Find expert annotator (academic collaborator, paid service)
2. Bootstrap weak labels from existing partial translations
3. Use cross-lingual retrieval on a held-out subset as proxy
4. Accept sino-only evaluation with explicit caveats

---

## Issue 6 — Việt OCR page order unverified

**Severity:** Unknown — could be high if violated.

**Symptom:** Vecalign assumes monotonic alignment. If 3 tập PDFs
were scanned/OCR'd out of order relative to Hán text, alignment
breaks silently.

**Status:** Unverified. No spot-check done.

**Candidate fixes:**
1. Chapter-heading anchor match between Hán and Việt before vecalign
2. Spot-check page 1, 100, 500, 1000 of each tập — compare headings
3. Compute vecalign score distribution per page; flag outliers

---

## Issue 7 — Deliverable coverage low

**Severity:** Medium — limits usefulness of corpus.

**Number:**
```
55 105 Hán sentences (input)
     ↓ vecalign + rerank
44 718 aligned pairs      (81% sentence coverage)
     ↓ sino≥0.15 export filter
11 678 delivered pairs    (21% sentence coverage)
```

**Trade-off:** Lower `HVB_MIN_SINO` threshold → more pairs but more
noise. Higher threshold → higher precision but less coverage.

**Candidate fixes:**
1. Fix Issue 1 (embedder) → more pairs naturally pass filter
2. Lower threshold to 0.10 with explicit "low-confidence" warning
3. Ship two deliverables: high-precision (sino≥0.40) + high-recall (sino≥0.10)

---

## Priority order

| # | Issue | Effort | Impact |
|---|-------|--------|--------|
| 1 | Issue 2: Fix sino proxy normalization | Low | High — removes false positives |
| 2 | Issue 6: Verify Việt page order | Low | Unknown — could be huge |
| 3 | Issue 1: Sino-primary alignment experiment | Medium | High — bypass embedder entirely |
| 4 | Issue 1: Try Qwen/NLLB embeddings | Medium | Unknown |
| 5 | Issue 3: Standardize vecalign score | Low | Medium — tooling |
| 6 | Issue 4: 7 oversized sentences | Low | Low |
| 7 | Issue 5: Find expert annotator | High | Critical — unlocks honest eval |
| 8 | Issue 7: Two-tier deliverable | Low | Medium — useful now |

---

## Reproducing current state

```bash
# Counts
wc -l data/interim/han_sentences.jsonl           # 55105
wc -l data/interim/vi_sentences.jsonl            # 63401
wc -l data/aligned/pairs_reranked.jsonl          # 44718
wc -l data/final/hvb_parallel.tsv                # 11678 (no header)

# Sino distribution
uv run python scripts/rescore_sino.py

# Top pairs inspection
python3 -c "
import json
pairs = [json.loads(l) for l in open('data/aligned/pairs_reranked.jsonl') if l.startswith('{')]
pairs.sort(key=lambda p: -p['combined'])
for p in pairs[:5]:
    print(f'combined={p[\"combined\"]:.3f} sino={p[\"sino\"]:.2f}')
    print(f'  H: {p[\"src\"][:80]}')
    print(f'  V: {p[\"tgt\"][:80]}')
"
```
