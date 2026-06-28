# Bài toán, Pipeline, và Chiến lược đánh giá

Tài liệu giới thiệu (read this first): định nghĩa bài toán, lý do từng
bước pipeline, và cách đánh giá corpus khi **không có gold standard**.

---

## 1. Bài toán

### Đây là bài toán gì?

**Cross-lingual sentence alignment on historical documents** — dóng hàng
câu giữa văn bản gốc Hán-Nôm (Classical Chinese) với bản dịch Việt ngữ
hiện đại của cùng một tác phẩm.

Thuộc lớp bài toán **Bitext extraction / Parallel corpus mining** (như
Vecalign, Bleualign, Hunalign) nhưng có thêm 3 đặc thù:

| Đặc thù | Mô tả |
|---------|-------|
| **Domain** | Văn cổ — từ vựng archaic, văn phong biên niên sử, nhiều nhân danh/địa danh/niên hiệu |
| **Noisy input** | Một bên (Hán) đã số hóa sạch; bên kia (Việt) phải OCR từ scan chất lượng kém |
| **No gold** | Không có cặp gold alignment; đánh giá phải auto |

### Input

```
Hán side (đã số hóa sẵn):
  data/raw/Đại Nam Thực Lục - 大南寔錄_full.txt
  - 233,493 dòng / 4.9M chars
  - Nguồn: Wiki文库 (số hóa từ in bản)

Việt side (chưa số hóa):
  data/raw/Đại Nam Thực Lục tập {4,5,6}.pdf
  - 3,242 trang scan @ 300 DPI
  - Nguồn: Quốc Sử Quán Triều Nguyễn (bản in 1961–1970s)
```

Hai bên là **cùng tác phẩm** (Đại Nam Thực Lục — biên niên sử hoàng gia
nhà Nguyễn, thế kỷ 19) nhưng phân chia theo cấu trúc khác nhau, không
cùng số chương, và chất lượng digitization khác nhau.

### Output

```jsonl
// data/final/hvb_corpus.jsonl — 1 cặp/dòng
{
  "src": "初置泰康營,時占城國王婆抋侵富安",
  "tgt": "Mới đặt dinh Thái Khang, bấy giờ vua Chiêm Thành là Bà Trĩ xâm phạm Phú An.",
  "score": 0.83,
  "src_idx": [42],
  "tgt_idx": [187],
  "entities": [
    {"han": "泰康", "vi": "Thái Khang", "type": "LOC"},
    {"han": "富安", "vi": "Phú An", "type": "LOC"}
  ]
}

// data/final/eval/*.json — báo cáo đánh giá
//   auto_metrics.json   : LaBSE/COMET/BERTScore/BLEU/chrF
//   flores_sanity.json  : precision@1 trên FLORES-200 zh-vi
//   round_trip.json     : chrF/BLEU round-trip Việt→Hán
//   holdout_mt.json     : BLEU/COMET trên hold-out 20%
//   llm_ensemble.json   : rubric scores + Krippendorff α
```

**Use cases downstream:** SMT/NMT training cho văn cổ, từ điển học thuật,
cross-lingual NER historical, contrastive linguistics Hán-Việt.

---

## 2. Pipeline — tại sao từng bước?

```
[Hán TXT] → Normalize ──┐
                         ├─→ Sent Split ─┐
[3 PDFs Việt] → PNG → OCR (PaddleOCR + LLM) ─→ Sent Split ─┤
                                                            ├─→ LaBSE Embed
                                                            │     ↓
                                                            │  Vecalign
                                                            │     ↓
                                                            │  pairs.jsonl
                                                            │     ↓
                                                            ├─→ NER bridge ─→ entities
                                                            │     ↓
                                                            └─→ Export + Eval
```

| Stage | Làm gì | Tại sao |
|-------|--------|---------|
| **1. prep** | Normalize Hán TXT (strip Wiki header, fix punctuation fullwidth→halfwidth); PDF→PNG @ 300 DPI | OCR cần ảnh sạch; split cần text normalize |
| **2. OCR** | PaddleOCR (PPOCRv5 vietnamese) → text thô; vLLM LLM (Qwen2.5-7B-Instruct) fix lỗi OCR (font cổ, Nôm, dấu câu) | Scan 1961 chất lượng thấp, PaddleOCR đơn độc ~10–15% CER; LLM post-fix giảm xuống ~5% |
| **3. split** | Tách câu: Hán theo `。！？` + HanLP; Việt theo Underthesea `sent_tokenize` | Vecalign cần đơn vị câu, không phải đoạn |
| **4. embed** | LaBSE (sentence-transformers) tạo 768-dim vector cho mỗi câu | Vecalign dùng cosine similarity để quyết align; LaBSE cross-lingual tốt cho cả Hán lẫn Việt |
| **5. align** | Vecalign dynamic programming trên similarity matrix | Tìm monotonic alignment tối ưu giữa 2 chuỗi câu; DP O(n·m) cho phép 1-many và many-1 (xử lý câu dài/ngắn không khớp 1-1) |
| **6. NER** | HanLP NER bên Hán + PhoBERT/Underthesea NER bên Việt; bridge matching qua Sino-Vietnamese transliteration | Entity (nhân vật, địa danh, niên hiệu) là anchor cross-lingual mạnh — nếu 2 câu align mà có entity cùng ref, độ tin cậy cao |
| **7. eval** | 5 trụ auto (xem §3) + export corpus | Không có gold alignment nên không tính precision/recall trực tiếp |

**Sắp xếp lý do:**
- OCR trước split vì split cần text sạch
- Embed trước align vì align cần similarity
- NER sau align vì NER dùng output pairs để validate
- Eval cuối vì cần corpus hoàn chỉnh

---

## 3. Đánh giá — không có gold, làm sao?

**Vấn đề:** Đánh giá chất lượng alignment cần ground truth — cặp câu do
con người annotate đúng/sai. Nhưng:
- Không có annotated corpus cho Đại Nam Thực Lục
- Tự annotate 200+ cặp tốn 2 annotators × 1 tuần, chưa kể κ inter-annotator
- Domain văn cổ khó thuê annotator có chuyên môn

**Giải pháp: full-auto 5 trụ** — kết hợp nhiều proxy metric, mỗi metric
đo một khía cạnh khác, cross-validate lẫn nhau:

### Trụ 7a — Auto metrics trực tiếp
- **LaBSE cosine**: mean similarity của các cặp aligned. Cao = 2 câu gần nghĩa.
- **COMET-QE-22**: reference-free quality estimation (Unbabel), đánh giá MT
- **BERTScore**: similarity âm vị trên mBERT
- **BLEU/chrF++** bi-directional: dịch Hán→Việt + Việt→Hán rồi so với reference
- **Hạn chế**: metric tự tham chiếu chính nó, không phát hiện systematic bias

### Trụ 7b — External benchmark sanity-check
- **FLORES-200 zh-vi** (2009 câu, modern Chinese)
- **OPUS zh-vi** (news/web, modern)
- Chạy **full pipeline Stage 4–5** trên FLORES (đã có gold alignment)
- **Kỳ vọng**: precision@1 > 0.95, COMET > 0.4
- **Mục đích**: validate pipeline chạy đúng (LaBSE + Vecalign cho kết quả hợp lý trên data đã biết good)
- **KHÔNG claim đánh giá domain**: FLORES modern, văn cổ khác hoàn toàn — pass FLORES ≠ pass Đại Nam Thực Lục

### Trụ 7c — Round-trip consistency
- Dịch ngược Việt → Hán bằng local LLM (Qwen2.5)
- So sánh Hán gốc vs Hán round-trip: chrF++, BLEU, BERTScore
- **Ý nghĩa**: nếu alignment đúng, dịch ngược phải gần với Hán gốc
- **Hạn chế**: LLM yếu cho văn cổ → chrF kỳ vọng 0.4 (không 0.7)
- 500 cặp stratified (3 bucket theo LaBSE: low < 0.5 < mid < 0.7 < high)

### Trụ 7d — Internal hold-out MT
- Chia aligned corpus 80/20 train/dev
- Train mini-NMT: Transformer-base (MarianMT Helsinki-zh-vi pretrain)
- Evaluate trên 20% hold-out: BLEU, COMET, chrF++
- **Ý nghĩa**: alignment tốt ↔ MT train tốt ↔ BLEU cao. **Domain-match thật sự.**
- **Hạn chế**: cần ≥ 5000 cặp aligned (Vecalign yield phụ thuộc noise)
- Tự skip nếu < 5000 pairs (check `HOLDOUT_MIN_PAIRS` trong config)

### Trụ 7e — LLM ensemble judge
- 1 LLM (Qwen2.5-7B-Instruct) chấm 1–5 mỗi cặp trên 5 rubric — single rater nên Krippendorff α undefined; eval chỉ report mean-only:
  - Adequacy (đầy đủ ý)
  - Fluency (thông suốt tự nhiên)
  - Alignment (có thực sự là dịch của nhau)
  - Fidelity (trung thực nguồn)
  - Terminology (đúng thuật ngữ lịch sử)
- Cross-model **Krippendorff's α** (thay Cohen's κ cho ≥ 2 raters) — chỉ khả thi khi `LLM_MODELS` có ≥ 2 model; mặc định 1 model → α = None, report mean-only
- **Ý nghĩa**: nếu ≥ 2 LLM độc lập đồng thuận, độ tin cậy cao hơn 1 LLM
- **Hạn chế**: LLM share biases → α target 0.5 (không 0.7 như human κ); với 1 model, không có độ đo đồng thuận giữa các rater
- 500 cặp stratified

### Trụ 7f — NER-Bridge coverage
- Tỉ lệ cặp aligned có ≥ 1 entity match cross-lingual (Han-Viet transliteration)
- **Ý nghĩa**: entity là anchor mạnh; coverage cao = alignment đúng nhiều
- **Target**: > 50% cặp có entity bridge

### Tổng hợp

| Metric | Target | Tools |
|--------|--------|-------|
| LaBSE cosine mean | > 0.6 | sentence-transformers |
| COMET-QE mean | > 0.5 | unmt/comet-qe-22 |
| FLORES precision@1 | > 0.95 (sanity) | vecalign + flores devtest |
| Round-trip chrF | > 0.40 | sacrebleu + vLLM |
| Hold-out MT BLEU | > 15 (skip nếu < 5000 pairs) | transformers + sacrebleu |
| Krippendorff α (LLM) | ≥ 0.5 (chỉ khi ≥ 2 model) | krippendorff + vLLM |
| NER-Bridge coverage | > 50% | custom matcher |

**Nếu bất kỳ metric nào trượt target**, xem `docs/05_troubleshooting.md`
cho diagnostic patterns.

### Tại sao KHÔNG dùng phương pháp khác?

| Phương pháp | Lý do không dùng |
|-------------|------------------|
| Human annotation | Tốn thời gian + khó tìm annotator chuyên văn cổ; dự án full-auto |
| So với bản dịch đã có | Không tồn tại — Đại Nam Thực Lục chưa có parallel corpus công bố |
| MT zero-shot (GPT-4o) | Cloud API cấm (CLAUDE.md); model lạ domain văn cổ sẽ yếu |
| BLEU vs NIST test set | Test set modern zh-vi (WMT), không represent văn cổ |

---

## 4. Giả định + giới hạn

1. **Vecalign giả định monotonic alignment** — Hán và Việt theo cùng thứ tự
   thời gian. Nếu sai, chunk theo chapter heading trước Stage 5.
2. **FLORES/OPUS modern zh-vi** — chỉ sanity pipeline, không phải domain eval.
3. **LLM 7B yếu cho văn cổ** — kỳ vọng residual OCR error ~10% sau LLM fix.
4. **Round-trip Việt→Hán khó hơn Hán→Việt** — chrF 0.4 là tốt.
5. **Hold-out MT cần ≥ 5000 pairs** — phụ thuộc Vecalign yield.
6. **LLM ensemble α** kỳ vọng 0.5 (LLMs share bias), không 0.7 như human.

---

## 5. Tài liệu liên quan

- [`docs/01_setup.md`](01_setup.md) — Cài đặt môi trường chi tiết
- [`docs/02_data.md`](02_data.md) — Spec input + output schema
- [`docs/03_pipeline.md`](03_pipeline.md) — Stage code flow
- [`docs/04_eval.md`](04_eval.md) — Eval methodology chi tiết
- [`docs/05_troubleshooting.md`](05_troubleshooting.md) — Lỗi thường gặp
- [`docs/06_extend.md`](06_extend.md) — Mở rộng (thêm PDF, đổi model, custom eval)
