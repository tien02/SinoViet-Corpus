# Hán normalization + sentence-split pipeline

Stage 1a (`normalize_han`) → Stage 3a (`split_han`).

Consumes: `data/raw/Đại Nam Thực Lục - 大南寔錄_full.txt` (Wiki文库 digitized,
UTF-8, ~4.9M chars, 233k lines).

Produces:
- `data/interim/han_clean.txt` (normalized, paragraph-preserving)
- `data/interim/han_sentences.jsonl` (per-line `{idx:int, text:str}`)

---

## 1. Raw shape

- Full-width CJK punctuation: `。！？；，：、` (U+3001–U+FF)
- Wiki markers: `姊妹计划:`, `## …` section headers, `【 … 】` block markers
- Paragraph breaks: blank lines (`\n{2,}`) between passages
- **Zero-terminator blocks**: imperial edicts / decrees reproduced without
  punctuation — 378 such paragraphs in this corpus, median 8 568 chars

---

## 2. `normalize_han` (Stage 1a)

**File:** `src/01_prep/normalize_han.py`

### Steps (in order)

1. `BRACKET_RE` — strip `【…】` wiki block markers
2. `WIKI_HEADER_RE` — strip `姊妹计划:`, `#`/`##` headers, `数据项`,
   `\s*---+\s*` rules
3. `FULLWIDTH_RE` — map U+FF01..U+FF5E → U+21..U+7E (full-width Latin +
   ASCII punctuation to half-width). CJK-block punctuation (`。、〈〉「」『』`)
   is outside this range and untouched.
4. `MULTI_NEWLINE_RE` — collapse `\n{3,}` → `\n\n`
5. **Paragraph-preserving line collapse** — split on `\n{2,}`, strip empty
   lines inside each paragraph, rejoin lines within a paragraph with `\n`,
   rejoin paragraphs with `\n\n`

### Why paragraph-preserving

Pre-fix: `"\n".join(ln for ln in lines if ln)` collapsed `\n\n` to `\n`.
Every paragraph break was lost — `split_han` then processed the whole
4.9M-char file as **one** paragraph. Combined with the terminator bug
(§3), one "sentence" grew to **989 086 chars** — an entire imperial edict
swallowed whole.

Post-fix: 633 paragraphs preserved.

### Full-width → half-width bug (pre-fix)

`FULLWIDTH_RE = re.compile(r"[！-～]")` covers U+FF01..U+FF5E. That range
**includes** the full-width sentence terminators `！？；` — they get
converted to ASCII `!?;` here in Stage 1a. `split_han`'s original terminator
regex only matched `[。！？；]` (full-width), so post-normalize, three of
four sentence terminators were invisible. The fix is Stage 3a's regex now
accepts both forms — see §3.

### Verify

```bash
uv run python -m src.01_prep.normalize_han
head -c 500 data/interim/han_clean.txt
grep -c "^$" data/interim/han_clean.txt    # > 0: blank lines preserved
```

---

## 3. `split_han` (Stage 3a)

**File:** `src/03_split/split_han.py`

### Pipeline per paragraph

1. **Protect annotations**: nested `〈…〉`, `「…」`, `『…』`, `（…）` blocks
   are extracted, replaced with `__ANNn__` placeholders. Prevents split on
   terminators embedded inside annotations (e.g. era-name glosses
   `〈清道光七年〉`).
2. **Split on terminators**: `ZH_TERM_RE = r"([。！？；!?;])"` — both
   full-width AND half-width forms (`FULLWIDTH_RE` in Stage 1a converts
   `！？；` → `!?;`; `。` is U+3002, outside the fullwidth range).
3. **Merge terminator back** to preceding content so sentence keeps its
   terminal punctuation.
4. **Restore annotations**: `__ANNn__` → original bracketed text.
5. **Fallback for zero-terminator blocks** — see §4.

### Terminator regex — both forms

```python
ZH_TERM_RE = re.compile(r"([。！？；!?;])")
```

Full-width `！？；` kept for corpora where `FULLWIDTH_RE` did not run;
ASCII `!?;` matched for the post-normalize case. Redundant-but-safe.

---

## 4. Zero-terminator fallback

**Problem:** 378 paragraphs carry ZERO terminal punctuation. Median 8 568
chars, max 24 476. Imperial edicts / decrees reproduced from stele or
palace-archive style. Without a fallback, each becomes ONE sentence — 349
of 378 exceed the export cap `HVB_MAX_PAIR_CHARS = 2000` and get dropped
at `export_deliverable`. Net: ~91 % of edict content lost.

**Constraints on the fallback:**

- Vecalign is **monotonic 1-to-N**. Ratio Hán:Việt sentence count matters —
  if Hán is 4× Việt count, DP degrades.
- Line-by-line split (`s.split("\n")`) on a 8 568-char block yields
  ~120 tiny 4–5-char fragments. Ratio blows up (251k Hán : 66k Việt = 3.8:1).
- Whole-block atomic is too coarse.

**Solution:** `_greedy_merge_lines(text) -> list[str]` — split on `\n`, greedily
concatenate lines until running buffer reaches `CHUNK_TARGET = 200` chars,
emit chunk, reset. Preserves original line order (monotonic-safe).

**Trigger:**

```python
if (
    len(s) > MAX_LEN            # 2000 chars — matches export cap
    and "\n" in s
    and not any(c in s for c in TERM_CHARS)
):
    sentences.extend(_greedy_merge_lines(s))
```

Only fires when the segment is oversized AND has no terminator AND has
newlines to split on. Normal punctuated paragraphs and annotation-heavy
long sentences (e.g. 4 851-char casualty lists inside `〈…〉`) are left
atomic.

---

## 5. Measured impact

Corpus: `Đại Nam Thực Lục - 大南寔錄_full.txt` (4.9M chars, 233k lines).

| Metric                               | Pre-fix    | Post-fix   |
|--------------------------------------|-----------:|-----------:|
| `han_clean.txt` paragraphs (`\n\n`)  | 1          | 633        |
| Terminator hits (`。！？；!?;`)       | 42 045     | 45 825     |
| Zero-terminator paragraphs           | n/a        | 378        |
| `han_sentences.jsonl` count          | 34 609     | **54 991** |
| Sentence median length               | 25 chars   | 33 chars   |
| Sentence max length                  | 989 086    | **13 163** |
| Sentences over 2 000 chars           | 24         | **12**     |
| Aligned pairs (Vecalign ≥ 0.5)       | 33 718     | **52 710** |
| Deliverable pairs (post export)      | 33 692     | **52 693** |
| Export drops (`>HVB_MAX_PAIR_CHARS`) | 26         | 17         |

Yield: **52 693 pairs from ~54 991 Hán sentences = 95.8 %** land in the
parallel record. Prior yield was 97.3 % of a much smaller sentence pool —
the absolute pair count is what matters for the deliverable.

---

## 6. Reproducing

```bash
rm -f data/interim/.checkpoint/normalize_han
rm -f data/interim/.checkpoint/split_han
rm -f data/interim/.checkpoint/labse_embed
rm -f data/interim/.checkpoint/vecalign
rm -f data/interim/.checkpoint/export_deliverable

./scripts/run_pipeline.sh prep     # normalize_han
./scripts/run_pipeline.sh split    # split_han (+ split_vi)
./scripts/run_pipeline.sh embed
./scripts/run_pipeline.sh align
./scripts/run_pipeline.sh export
```

Post-run verify:

```bash
wc -l data/interim/han_sentences.jsonl
python3 -c "import json; L=[len(json.loads(l)['text']) for l in open('data/interim/han_sentences.jsonl')]; L.sort(); print('median', L[len(L)//2], 'max', L[-1])"
wc -l data/aligned/pairs.jsonl
wc -l data/final/hvb_parallel.tsv
```

---

## 7. Tuning knobs (`src/03_split/split_han.py`)

| Constant       | Default | Effect if increased                      |
|----------------|--------:|------------------------------------------|
| `MAX_LEN`      | 2000    | Fewer fallback fires, more oversized sentences reach export → more drops |
| `CHUNK_TARGET` | 200     | Fewer, longer chunks; ratio Hán:Việt shifts lower; risk of oversized units |

Kept as module constants, not env vars — structural corpus properties,
not per-run preferences.

---

## 8. Known residuals

- **~15 000 sentences with < 10 chars.** Many are 4-char classical
  couplets or fragments; not filtered because underthesea's Việt split
  also emits short units and Vecalign uses embedding similarity, not
  length parity. Drop threshold could be added post-align if warranted.
- **6-to-12 sentences still exceed 2 000 chars** — annotation-heavy blocks
  (e.g. 275-name casualty lists inside `〈…〉`). Left atomic because
  splitting inside protected annotations would break era-name / official-
  title semantics used by LaBSE.
- **`paragraphs: 1` output from Stage 3a on pre-fix `han_clean.txt`** is
  the smoking gun for the paragraph-collapse bug. Post-fix prints
  `paragraphs: 633`.
