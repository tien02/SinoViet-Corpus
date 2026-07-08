# Final Results — Bertalign + BGE-M3 pipeline (2026-07-07)

**⚠ SUPERSEDED 2026-07-08**: Guwen-biaodian punctuator (docs/13) delivers
30,959 pairs (5.3x baseline) with cosine mean 0.621, max 0.879. Numbers
below reflect the pre-punctuator pipeline (200-char chunker). Retained
for ablation/history.

Production pipeline for the Đại Nam Thực Lục Hán-Việt parallel corpus.
Supersedes the Vecalign baseline in `docs/08_results.md`.

Reproduce: `./scripts/reproduce_bertalign_bgem3.sh`

---

## Final deliverable

```
data/final/hvb_parallel.tsv    5,856 pairs   ⭐ course submission
data/final/hvb_parallel.xlsx   5,856 pairs   ⭐ Excel copy
data/final/hvb_raw.txt         3 tập raw OCR concat
```

Schema: `pair_id \t han_sentence \t viet_sentence \t sino`.

---

## Pipeline

```
Hán TXT (Wiki文库, sliced Tập 4-6, lines 50073–138723)
  ↓ normalize + slice
7,048 Hán sentences

Việt PDF (3 Tập, PaddleOCR default)
  ↓ split (underthesea + regex fallback)
63,401 Việt sentences

Both sides
  ↓ Bertalign (BGE-M3 fp16, max_seq=256, batch=64, GPU)
    - encoder-side overlap embeddings
    - two-pass DP anchor + fill
    - filter score (cosine sim) ≥ 0.5
5,917 aligned pairs

  ↓ Rerank (add sino proxy + combined score)
5,917 enriched pairs

  ↓ Export filters + cosine rescue
5,856 delivered pairs
```

---

## Quantitative results

### Coverage

| Metric | Value |
|--------|-------|
| Hán sentences (sliced Tập 4-6) | 7,048 |
| Việt sentences | 63,401 |
| Aligned pairs (bertalign kept ≥0.5) | 5,917 |
| Delivered pairs (after filters + rescue) | 5,856 |
| Hán coverage | 84.7% (5,969 unique Hán idx in beads) |

### Score distributions (n=5,917)

| Signal | Mean | Median | p25 | p75 | Min | Max |
|--------|------|--------|-----|-----|-----|-----|
| Cosine similarity | 0.559 | 0.553 | 0.530 | 0.581 | 0.500 | 0.768 |
| Sino avg (independent) | 0.500 | 0.494 | 0.410 | 0.585 | 0.000 | 0.980 |
| Length ratio (V/H chars) | 3.38 | 3.11 | 2.50 | 3.93 | 0.23 | 27.1 |

### Alignment shape (Bertalign beads)

| Shape | Count | % |
|-------|-------|---|
| 1-1 | 55 | 0.9% |
| **1-N (1 Hán → N Việt)** | **5,827** | **98.5%** |
| N-1 | 27 | 0.5% |
| N-N | 8 | 0.1% |

Reflects corpus reality: câu Hán cổ súc tích → nhiều câu Việt dịch dài.

### Length ratio outlier

`< 0.5 hoặc > 8.0`: **1.2%** (70 pairs, down from 45.7% in Vecalign baseline).

---

## Ablation (2×2)

| Aligner \ Embedder | Vecalign (monotonic DP) | Bertalign (2-pass) |
|--------------------|-------------------------|--------------------|
| BGE-M3 | 7,043 pairs — sino 0.275, outlier 45.7% | **5,917 pairs — sino 0.500, outlier 1.2%** ⭐ |
| Qwen3-Embedding-0.6B | (not run) | 6,953 pairs — sino 0.472 (no filter) |

**Aligner delta (BGE-M3 fixed):** sino 0.275 → 0.500 = **+82%**. Length outlier 45.7% → 1.2% = **-97%**.

**Embedder delta (Bertalign fixed, both no-filter):** BGE-M3 sino 0.500 vs Qwen 0.472. Gap ~5%, not decisive. Qwen threshold scale is different (its p90 = 0.461 vs BGE-M3 p90 = 0.612) — score not portable across embedders.

Conclusion: **aligner is the bottleneck, not embedder.**

---

## Filter + rescue policy

Three filters + two rescue conditions at export stage (`src/07_export/export_deliverable.py`):

```
Drop:
  len(han) > 2000 or len(viet) > 2000     → 13 drops (Excel + real range-merge junk)
  ratio ∉ [0.5, 8.0]                       → 36 real drops + 17 rescued
  sino < 0.15                              → 12 real drops + 14 rescued

Rescue (only when HVB_ALIGNER=bertalign, cosine sim available):
  ratio outlier BUT cosine ≥ 0.60          → +17 pairs
  sino < 0.15 BUT cosine ≥ 0.55            → +14 pairs
```

Rescue captured 30 real translations lost to proxy limits (semantic dịch such as
`千載` → "nghìn năm" that break Sino-Viet phonetic overlap).

---

## Reproducibility

Files documenting the config:

```
scripts/reproduce_bertalign_bgem3.sh     # one-shot entry
scripts/patches/bertalign_encoder.py     # bertalign encoder overlay (fp16 + max_seq)
scripts/rerank_combined.py               # dual-mode score dispatch
src/07_export/export_deliverable.py      # filters + rescue
src/05_align/bertalign_runner.py         # bertalign wrapper
src/utils/config.py                      # HVB_HAN_LINE_START/END, HVB_PAIRS_OUT
```

Key env vars (defaults in reproduce script):

```
HVB_ALIGNER=bertalign
HVB_EMBED_MODEL=BAAI/bge-m3
HVB_EMBED_MAX_SEQ=256
HVB_EMBED_BATCH=64
HVB_EMBED_FP16=1
ALIGN_MIN_SCORE=0.5
HVB_MIN_SINO=0.15
HVB_MIN_LEN_RATIO=0.5
HVB_MAX_LEN_RATIO=8.0
HVB_SINO_RESCUE_COS=0.55
HVB_RATIO_RESCUE_COS=0.60
HVB_MAX_PAIR_CHARS=2000
```

---

## Limitations

### 1. No ground truth alignment labels

Không có Hán-Nôm expert xác nhận cặp đúng/sai. Mọi metric hiện tại là **proxy**:
- Cosine similarity — circular với embedder chính pipeline dùng.
- Sino precision — chỉ đo overlap từ vựng Hán-Việt, không đo ngữ nghĩa/ngữ pháp.
- Length ratio — chỉ sanity structural.

Không tính được **precision/recall thật**. Kết luận chất lượng chỉ dựa vào **convergent evidence** giữa các proxy độc lập.

### 2. Sino proxy false negatives

Dịch nghĩa (semantic translation) không tạo overlap Hán-Việt phonetic:

```
故千載之下,可以求千載之前
→ "nghìn năm về sau có thể tìm biết công việc nghìn năm về trước"
   (千載 → "nghìn năm" thuần Việt, không phải "thiên tải")
```

Rescue policy giảm impact nhưng không loại hoàn toàn. Ước tính **~5-10% false negatives** trong dropped set.

### 3. Sino proxy false positives với short Hán

Câu Hán ngắn (2-5 ký tự) dễ đạt sino=1.0 do mọi âm tự có Hán-Việt reading ngẫu nhiên xuất hiện trong Việt dài. Slice + split hiện tại tránh được bẫy này (98.5% Hán ≥ 20 ký tự), nhưng vẫn tồn tại ở đuôi phân phối.

### 4. Embedder yếu ở Hán cổ ↔ Quốc ngữ

BGE-M3 huấn luyện trên corpus đương đại → **out-of-distribution** với Classical Chinese + Quốc ngữ dịch thế kỷ 20. Cosine mean 0.559 (thấp), IQR 0.067 (hẹp) → embedder không phân biệt tốt cặp tốt/xấu bằng cosine alone. Confirmed by Qwen ablation — tương đương với BGE-M3, cùng ceiling.

**Fix chưa thực hiện:** fine-tune BGE-M3 trên gold pairs hoặc dùng encoder chuyên biệt cho Classical Chinese.

### 5. Vecalign score semantics khác Bertalign

Vecalign xuất **cosine distance** (thấp = tốt), Bertalign xuất **cosine similarity** (cao = tốt). Ngưỡng không portable → phải chỉnh `ALIGN_MIN_SCORE` per aligner. Reproduce script đã document.

### 6. Việt PDF page order chưa verify

Bertalign vẫn giả định monotonic order trong 2-pass. Nếu 3 Tập PDF được scan/OCR lệch thứ tự so với Hán TXT → alignment lệch nhưng score vẫn cao. Chưa có spot-check chapter-heading giữa 2 phía.

### 7. Bertalign encoder patched externally

`external/bertalign/` là git clone, không tracked trong repo chính. Patch nằm ở `scripts/patches/bertalign_encoder.py` và auto-apply bởi reproduce script. Cần chạy setup + reproduce script thay vì `git clone bertalign` trực tiếp.

### 8. Deliverable coverage tương đối thấp

5,856 delivered pairs / 7,048 Hán sentences = **83% coverage**. Còn 17% câu Hán không có bản dịch reliable — có thể do:
- Việt OCR bỏ sót câu (page ordering issue).
- Bertalign emit deletion beads (bỏ câu Hán không match Việt nào).
- Filter cuối cắt bớt (13 long + 36 ratio + 12 sino).

Không có metric nào bắt được cái nào đóng góp lớn nhất.

---

## Open work

Xếp theo effort/impact (từ `docs/11_current_issues.md`):

1. **Mini gold set 100-200 cặp** — spot-check bằng người biết Hán-Việt → tính precision/recall thật. **Effort thấp, impact cao.**
2. **Chapter-anchor pre-alignment** — split cả 2 phía theo tiêu đề chương trước Bertalign → giảm drift.
3. **Fine-tune BGE-M3** trên 493 top-band pairs làm positive → boost embedder trong domain.
4. **Round-trip translation eval** — dùng vLLM Qwen2.5-7B: Hán → Việt (LLM dịch) → chrF/BLEU với Việt corpus → alternative ground truth.
5. **N-1 bead review** — 27 N-1 pairs cần spot-check thủ công (50% real / 50% fail estimated).

---

## Reference — history of runs

Tracked backup files in `data/aligned/`:

```
pairs.vecalign_bgem3.jsonl         7,043  Vecalign+BGE-M3 baseline (sliced)
pairs.bertalign_bgem3.jsonl        5,917  Bertalign+BGE-M3 (⭐ current prod)
pairs.bertalign_qwen.jsonl           266  Bertalign+Qwen kept ≥0.5
pairs.bertalign_qwen_full.jsonl    6,953  Bertalign+Qwen no filter (ablation)
```
