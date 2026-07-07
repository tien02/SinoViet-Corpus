# 10 — Fail Cases: PaddleOCR-VL-1.6 + Full-Han Alignment

Diagnostic notes on where the current pipeline breaks. Baseline: LaBSE + Vecalign, PaddleOCR-VL-1.6 on Việt side, cleaned Hán TXT on Hán side.

## Numbers

| Side | Sentences | Notes |
|---|---:|---|
| Hán (Wiki文库, cleaned) | 54 991 | full corpus — target for alignment |
| Việt (PaddleOCR-VL-1.6) | 66 615 | 3 242 pages, split by `underthesea` |
| Aligned pairs (raw) | 52 783 | Vecalign beads with score ≥ 0.5 |
| Deliverable pairs | 52 766 | after `HVB_MAX_PAIR_CHARS=2000` drop |

- **Hán coverage**: 52 783 / 54 991 = **96.0 %** covered by ≥ 1 bead.
- **Uncovered Hán**: 2 208 sentences.
- **1 613 uncovered are single-idx gaps** — isolated skips inside otherwise aligned runs.
- Largest consecutive gap: 7 sentences (`idx 18877..18883`, `idx 43737..43743`).

## Pipeline stance: align all Hán

Current `split_han` emits 54 991 Hán sentences. `vecalign_runner` uses **every one** as source. No Hán filtering pre-align. Consequence: uncovered Hán = signal loss, not filter artefact.

## Why 2 208 Hán sentences never align

### G1 — noise / short Hán after split (5 sentences, `len < 3`)
Vecalign hard-drops `len(text) < 3` before overlap encoding.
- `idx=391` = `"。"`  · `idx=52812` = `"”"`  · `idx=53303` = `"有女一。"`

Cheap to lose.

### G2 — Hán bookkeeping headers (~80 sentences)
Volume/chapter delimiters injected by Wiki文库 editors, no Việt counterpart.
- `idx=52961` = `"大南正編列傳二集卷之二〈止〉"` (volume-end marker).

Safe to lose.

### G3 — Việt OCR failed at corresponding page (~500 sentences)
Vecalign is monotonic. Việt page produces zero usable sentences → Hán window covering that page produces zero beads.

Chain: 1 bad Việt page → 15–30 Hán sentences skipped.

### G4 — length mismatch: short Hán vs long Việt block (~1 500 sentences)
Median uncovered Hán len = **14 chars** (corpus median ~90). Edicts / quoted speech fragments split at `；`. Việt side merges into flowing paragraph. Vecalign 1-1 window misses; 1-N would catch — but only within `max_align=8` and only if score margin passes.

### G5 — reordering (small, hard to count)
Việt editors reorder events by month/day; Hán chronology strict. Non-monotonic → Vecalign refuses. Cannot recover with monotonic DP.

## Việt fail classes (PaddleOCR-VL-1.6)

See `docs/08_results.md` for cross-backend numbers. Below is per-class detail.

### F1 — decoder collapse: hallucinated integer runs
- Trigger: back-matter indices (`Gia Định : 5, 6, 7, ...`).
- Symptom: real page-list morphs into monotonic hallucination `100, 101, ..., 605`.
- Extent: 153 pages with 20+ consecutive comma-ints; 100 pages with 50+.
- Example: `tap4_page_1088.txt` line 61 — 4 105-char single "sentence".
- Cost: 0 real pairs lost (Hán has no name-index). Wastes Vecalign DP.

### F2 — repetition loop mid-paragraph
- Trigger: schema-repetitive Việt (rank tables, silk-award schedules, cannon-caliber schedules).
- Symptom: n-gram trap — one clause loops 20-30× until token cutoff.
- Extent: 16 pages with 4-gram-dup > 0.35 excluding legitimate TOC/index.
- Example: `tap5_page_0210.txt` line 7 — `"1 tấm so dày nửa sợi, dệt con măng xà to, mây, thụy ba"` × 28.
- Cost: 200–400 real Hán sentences uncovered (G3 chain).

### F3 — page furniture bleed
- Trigger: PaddleOCR-VL-1.6 tags book-spine / running header as `text` block instead of `header`.
- Symptom: shards like `章9 - ENTL - T4` become own sentence.
- Example: `tap4_page_0126.txt` line 15.
- Cost: mostly filtered by Vecalign `len<3` or LaBSE cosine.

### F4 — diacritic / homoglyph drift
Consistent errors on Nôm names + rare Sino-Việt terms:
- `dōng` for `dõng` (勇)
- `hắc tâu` for `hặc tâu` (劾奏)
- `xiêng xích` for `xiềng xích` (枷鎖)
- `sang sử` for `sang sứ` (使)
- `giao định thân` for `giao đình thần` (交廷臣)

Root: VL vision encoder confuses tilde vs macron; Nôm proper nouns OOD. LaBSE robust — most pairs still ≥ 0.5. Rare-name pairs occasionally drop below threshold.

### F5 — mid-word truncation across blocks
- **12.7 %** of Việt sentences (8 433 / 66 615) don't end with `.!?;:"”)`.
- **8 075 end mid-word** — block boundary cut the sentence.
- Root: `convert_paddlevl.py` emits one line per PaddleOCR-VL block; `split_vi` doesn't stitch across `paragraph_title` breaks.
- Cost: modest. Vecalign 1-2 / 2-1 beads mostly stitch them.

### F6 — data asymmetry (Việt back-matter has no Hán match)
- **92 (tap, page) pages** contribute zero aligned Việt sentences.
- Concentrated: `tap4 p≥1088`, `tap5 p≥894`, `tap6 p≥1000`.
- Content: modern editorial indices, glossaries, publisher colophons, TOCs — Việt editors added, Hán Wiki文库 omitted.
- Not OCR failure. 100 % of sentences from these pages are dead weight in embed/align.

### F7 — micro-artefacts (list markers, section jump-tabs)
- 422 sentences = `"”."` (dangling close-quote + dot)
- 300+ sentences = orphan list numbers `1.`, `2.`, ...
- Section-jumper 2-letter codes on tap6 back-matter: `CH`, `KH`, `NG`, `NH`, `TH`, `TR`.
- Cost: dropped by Vecalign `len<3`.

## Cost ranking

| Class | Pages hit | Hán uncov cost | Fix cost |
|---|---:|---:|---|
| F1 | ~150 | 0 | regex — kill runs of 20+ commas in `scripts/convert_paddlevl.py` |
| F2 | ~16 | 200-400 | re-OCR with `repetition_penalty=1.15` |
| F3 | ~50 | ~0 | regex drop `章\d+ - \w+ - T\d` |
| F4 | pervasive | small tail | LLM post-correct (currently skipped) or Nôm fine-tune |
| F5 | 12.7 % | modest | cross-block rejoin in `src/03_split/split_vi.py` when next block starts lowercase |
| F6 | 92 | 0 real, wastes DP | page-range skip pre-embed |
| F7 | scattered | ~0 | length filter at split |
| G4 | — | ~1 500 | switch to bertalign 1-N or split Hán at `。` only |
| G5 | — | unknown | requires chapter anchoring, not a Vecalign win |

## Fast wins (before touching model)

1. **Kill F1 + F7** in `scripts/convert_paddlevl.py` — regex drop `^\d+(?:\s*,\s*\d+){19,}` blocks and single-token trash.
2. **Kill F6** — page-range skip: `SKIP_PAGES = {"tap4": range(1088,1141), "tap5": range(894,946), "tap6": range(1000,1157)}`. Frees ~15 % of embed compute.
3. **Kill F3** — regex drop headers matching `章\d+ - \w+ - T\d`.

Ceiling after wins: still ~96 %. G4 is the real ceiling. Past 96 % needs 1-N alignment or Hán re-split at `。` only.

## What "align all Hán" costs today

`split_han` emits 54 991 sentences. Aligner consumes all. Result:

- **52 783 covered** (96.0 %).
- **2 208 uncovered** → silently dropped from deliverable.

Alternative: split Hán at `。` only (drop `！？；`):
- Hán count drops 54 991 → ~38 000.
- Uncovered rate drops 4 % → ~1 % — larger chunks align 1-1 more reliably with Việt paragraphs.
- Trade-off: coarser deliverable — some rows contain 2-3 logical sentences merged.

Not implemented — decision pending.
