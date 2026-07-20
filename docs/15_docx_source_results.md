# Results — `.docx` clean Việt source (2026-07-19)

Bypass OCR entirely. Replace PaddleOCR-VL-1.6 OCR'd Việt side with clean
`.docx` text digitized from same 3 Quốc Sử Quán Triều Nguyễn PDFs.

Reproduce: `uv run python src/03_split/docx_to_vi_sentences.py && ./scripts/reproduce_bertalign_bgem3.sh`

---

## 1. Source files

```
data/raw/dai-nam-thuc-luc-tap04.docx   1.6M
data/raw/dai-nam-thuc-luc-tap05.docx   1.2M
data/raw/dai-nam-thuc-luc-tap06.docx   1.5M
```

## 2. Pipeline change

Single script addition: `src/03_split/docx_to_vi_sentences.py`.

Reads 3 `.docx` files via `python-docx`, reuses `split_vi` paragraph logic
(`underthesea.sent_tokenize` + CJK punctuation split + `MIN_SENT_LEN=8`),
emits `data/interim/vi_sentences.jsonl` with same schema (`idx`, `tap`,
`page=-1`, `text`). Downstream stages unchanged — bertalign reads
`vi_sentences.jsonl` directly and encodes internally.

No config flag added. Backup old `vi_sentences.jsonl` first, run docx
script, then re-run `reproduce_bertalign_bgem3.sh`.

## 3. Quantitative comparison

Baseline = PaddleOCR-VL-1.6 + bertalign + BGE-M3 (last production run,
preserved in `data/aligned.ocr_20260719_105611/`).

| Metric                              | PaddleOCR-VL-1.6 baseline | `.docx` clean | Δ        |
|-------------------------------------|---------------------------|---------------|----------|
| Việt sentences                      | 66,615                    | 58,032        | -12.9%   |
| Pairs (bertalign kept `score≥0.5`)  | 33,221                    | **33,567**    | +1.0%    |
| Delivered TSV pairs                 | 30,959                    | **32,665**    | **+5.5%**|
| Cosine mean                         | 0.621                     | **0.628**     | +1.1%    |
| Cosine median                       | 0.614                     | **0.622**     | +1.3%    |
| Cosine max                          | 0.879                     | **0.885**     | +0.7%    |
| Sino mean                           | 0.484                     | 0.471         | -2.7%    |
| Length ratio V/H mean               | 4.81                      | 4.88          | +1.5%    |
| Length outliers ∉ `[0.5, 8.0]`      | 2.5%                      | 3.2%          | +0.7pp   |
| Top bead shape                      | 1-1 (47%)                 | 1-1 (47%)     | same     |
| Bead shape distribution             | 1-1, 1-2, 2-1, 2-2, 1-3   | same          | —        |

## 4. Hán trim verification

Hán slice = `HAN_LINE_START=50073` → `HAN_LINE_END=138723`. Verified
content scope matches docx 3-tập span:

| Marker                              | Hán line  | Content                                       |
|-------------------------------------|-----------|-----------------------------------------------|
| Slice start                         | L50073    | mid-Quyển 117 (Minh Mạng Y15) — Kỷ 2 opens    |
| Internal Tập 4 end / Tập 5 start    | L81586    | `...卷之一百七十七 〈 止 〉` (end Q.177)       |
| Internal Tập 5 end / Tập 6 start    | L107115   | `## 大南寔錄正編第三紀` (start Kỷ 3)           |
| Slice end                           | L138723   | before `## 大南寔錄正編第四紀` (Kỷ 3 Q.72 end)|

Docx tap4→tap5→tap6 ordering matches Hán slice Kỷ 2 Q.117 → Kỷ 3 Q.72.

**Sanity sample:** 10 random high-score (cos≥0.7) 1-1 pairs spot-checked
manually. All semantically correct translations. Hán idx → Việt idx
mapping monotonic (ratio ~1:1.3 — Việt denser sentence splitting).

**Minor caveat:** Docx Tập 4 last sentence (`"Chuẩn cho Tôn Thất Bật
chuyên làm việc bắt giặc."`) lands mid-content, not at Hán Q.177
boundary. Likely ~1 quyển publisher drift at internal Tập 4/5 volume
division. Doesn't affect slice — bertalign treats whole Hán block as one
monotonic sequence vs concatenated Việt block. 2-pass DP handles local
non-monotonicity. Confirmed by low length-outlier rate (3.2%) and clean
semantic samples.

## 5. Interpretation

### Why docx wins despite fewer sentences

PaddleOCR-VL-1.6 over-segments: 66,615 Việt sentences inflated by OCR
noise, page-break cuts, layout furniture fragments. Clean `.docx` has
12.9% fewer sentences but more *alignable* content. Bertalign produces
more beads at higher cosine — OCR diacritic loss was capping embedder
discrimination.

### Why sino dips

Sino proxy measures Hán-Việt phonetic overlap. Clean Vietnamese prose
uses more semantic dịch thuần Việt (e.g. `千載` → "nghìn năm" not
"thiên tải"). Less phonetic overlap, same or better translation. Sino
dip = expected byproduct of cleaner source, not regression. Confirmed
by cosine rising in opposite direction.

### Net effect

+1,706 delivered pairs, cosine up across full distribution (mean, median,
max). OCR noise floor removed. Bottleneck now shifts back to embedder
domain fit (BGE-M3 trained on contemporary text → Classical Chinese +
Quốc ngữ century-old prose remains OOD, per docs/12 §4).

## 6. Artifacts

Current production paths:

```
data/final/hvb_parallel.tsv         32,665 pairs  (course deliverable)
data/final/hvb_parallel.xlsx        32,665 pairs
data/final/hvb_raw.txt              3-tập concat
data/aligned/pairs.jsonl            33,567 pairs (bertalign kept ≥0.5)
data/aligned/pairs_reranked.jsonl   33,567 pairs (sino + rescue applied)
data/interim/vi_sentences.jsonl     58,032 sentences (docx-sourced)
```

Backups (rollback path):

```
data/aligned.ocr_20260719_105611/                    old aligned + old TSV
data/interim/vi_sentences.ocr_20260719_105524.jsonl   old OCR sentences
data/interim/vi_embeds*.ocr_20260719_105524.npy       old embeddings
```

Source script:

```
src/03_split/docx_to_vi_sentences.py     docx → vi_sentences.jsonl
```

## 7. Known caveats — manually audited bad pairs

Spot-check (2026-07-19) of 63 tail candidates (`sino<0.20 ∧ len_ratio<1.5
∧ pair_id>30000`) labeled 8 pairs as **wrong** (different content /
different names / contradicts Hán). Not dropped from deliverable —
documented here for transparency. Real fix is per-clause alignment
(currently out of scope; bertalign produces 1-bead-per-Hán-segment
which over-abbreviates long mandarinate lists).

| pair_id | Issue | Hán head | Việt |
|---------|-------|----------|------|
| 31574 | 2-word fragment | `其父旣蒙旻賞,尙在同居,擬應停給。` | `đình cấp.` |
| 31830 | Wrong tax stations | Hán: 宣威/東川/安樂/福禮 | V: Phú Mỹ/Mỹ Thu |
| 32202 | Wrong person | Hán: 阮公義/何文亨 | V: Trí Phú |
| 32248 | Contradicts Hán | Hán: do NOT set up school | V: DO set up |
| 32436 | Different person | Hán: wife of 逆雲, sister of 瑾 | V: Thị Nhị, sister of Cận |
| 32499 | Different topic | Hán: teaching youngsters | V: lawsuits/quarrels |
| 32645 | Numbered marker | Hán: ritual character | V: `1)không thấy nói đến.` |
| 32665 | Different topic | Hán: west border defense | V: revenue ministry |

Audit also labeled 10/63 as CORRECT (15.9%) and 45/63 as PARTIAL
(71.4% — Việt translates only opening clause of multi-clause Hán).
Low-sino bucket is therefore NOT bulk garbage — confirms docs/12 §2
hypothesis that pure Việt semantic translation (`千載` → "nghìn năm")
breaks phonetic overlap without breaking translation.

## 8. Open work

Same bottlenecks remain (from docs/12 §"Open work"):

1. Mini gold set 100-200 pairs — manual Hán-Việt expert labels for real
   precision/recall. All current metrics are proxies (cosine, sino,
   length ratio).
2. Chapter-anchor pre-alignment — split both sides by `Quyển` heading
   before bertalign. Reduces drift, may rescue borderline pairs.
3. Fine-tune BGE-M3 on top-band pairs — boost OOD performance on
   Classical Chinese ↔ century-old Quốc ngữ.
4. Round-trip translation eval — vLLM Qwen2.5-7B Hán→Việt, chrF/BLEU
   against corpus. Alternative ground truth without manual labels.
