# DÀN Ý ĐỒNG ÁN: XÂY DỰNG CORPUS SONG NGỮ HÁN-VIỆT TỪ ĐẠI NAM THỰC LỤC

---

## 1. GIỚI THIỆU

### 1.1 Mục tiêu đồ án
- Xây dựng **corpus song ngữ Hán-Việt** từ **Đại Nam Thực Lục** (Quốc sử Triều Nguyễn)
- Tạo **46,880 cặp câu song ngữ** được căn chỉnh (sentence-aligned) để phục vụ:
  - Nghiên cứu NLP (machine translation, alignment, embedding)
  - Giáo dục (học tiếng, dịch thuật)
  - Lưu trữ di sản văn hóa

### 1.2 Bài toán được giao
- **Input:**
  - Hán: File `.txt` digitize từ Wiki文库 → 80,391 câu sau punctuate
  - Việt: 3 PDF scan (Tập 4,5,6) + `.docx` digitize → 58,032 câu
- **Output:**
  - TSV + XLSX: `pair_id ⇥ han_sentence ⇥ viet_sentence ⇥ sino` (46,880 cặp)
  - Coverage: 80.8% Vietnamese sentences aligned
  - Mean score: Bertalign 0.624, Greedy fallback 0.644

### 1.3 Phạm vi dữ liệu & kết quả mong đợi
| Chỉ số | Giá trị | Ghi chú |
|--------|---------|---------|
| Câu Hán (raw) | 49,927 | Sau guwen-biaodian |
| Câu Hán (final) | 80,391 | Sau .docx source |
| Câu Việt | 58,032 | Từ .docx split |
| Bertalign pairs | 35,596 | Trực tiếp DP output |
| Greedy fallback | +12,586 | Cấp cứu Vi chưa align |
| **Cặp deliverable** | **46,880** | Sau score-based filter |
| **Coverage Việt** | **80.8%** | 46,880 / 58,032 |
| Cosine similarity TB | 0.624 | BGE-M3 |
| Ratio mean | 3.74 | Vi dài hơn Han |

---

## 2. DỮ LIỆU & CÔNG CỤ SỬ DỤNG

### 2.1 Mô tả dữ liệu đầu vào

#### 2.1.1 Phía Hán (Han)
- **Nguồn:** `Đại Nam Thực Lục - 大南寔錄_full.txt` (Wiki文库, digitize)
- **Kích thước:** 233K dòng, 4.9M ký tự
- **Xử lý:** Normalize Wiki marker + Guwen-biaodian punctuate → **49,927 câu**
- **Lưu ý:** Ngoài ra, cũng sử dụng .docx digitize Việt để lấy Han source → **80,391 câu final**

#### 2.1.2 Phía Việt (Vi)
- **Nguồn gốc:** 3 PDF scan Quốc Sử Quán (1,141 + 945 + 1,156 trang)
- **Input cuối cùng:** 3 .docx digitize pre-existing (58,032 câu)
  - .docx không phải từ PaddleOCR, mà là digitize sẵn (input thô)
  - Chứa text sạch, người tách câu (0% lỗi split)

### 2.2 Công cụ & Thư viện sử dụng

#### 2.2.1 **OCR Việt — So sánh input source**

**Thử nghiệm 1: PaddleOCR output (PDF) vs .docx digitize pre-existing**

| Source | Câu output | Chất lượng | Lợi / Hại |
|--------|----------|-----------|-----------|
| PaddleOCR-VL 1.6 (từ PDF) | 66,615 | ~8% CER | Over-segment (nhiều dòng split) |
| MinerU + voting (từ PDF) | 64,203 | ~6% CER | Layout tốt hơn, nhưng vẫn noise |
| **.docx digitize** | **58,032** | **0% split error** | ✅ Người tách câu, sạch |

**Finding:** .docx (digitize sẵn) > PaddleOCR output (PDF)

**Giải pháp chọn:** Sử dụng `.docx` digitize input (58,032 câu). Lý do:
- .docx không phải từ PaddleOCR, mà là digitize pre-existing
- Text đã sạch, người tách câu (0% OCR error)
- Trade-off: Mất mở rộng dữ liệu từ PDF gốc, nhưng alignment tốt hơn (+5.5% coverage)
- **Kết luận:** Chất lượng > Số lượng

---

#### 2.2.2 **Alignment — So sánh Vecalign vs Bertalign**

**Thử nghiệm 2: Vecalign (legacy) vs Bertalign (new)**

| Metric | Vecalign (LaBSE) | Bertalign (BGE-M3) | Lợi thế |
|--------|-----------------|-------------------|--------|
| Cặp output | 7,043 | 35,596 | **+405% Bertalign** |
| Cosine TB | — | 0.624 | ✓ Cosine similarity |
| Sino TB | 0.275 | 0.624 → rerank | ✓ Dual score |
| Bead 1-1 % | ~60% | 46.8% | ✓ M-n bead support |
| Rescue logic | None | Cosine ≥0.55 | ✓ Semantic override |

**Finding:** Bertalign + BGE-M3 **giải quyết 5 vấn đề** của Vecalign:
1. 2-pass DP (anchor + refinement)
2. Cosine similarity score (LaBSE → BGE-M3, +0.35 TB)
3. M-n bead support (không giới hạn 1-1)
4. Rescue logic cho low-sino pairs
5. Hybrid scoring (sino + cosine)

**Giải pháp chọn:** Bertalign + BGE-M3 (final)

---

#### 2.2.3 **Filter — Strict vs Loose vs Score-based**

**Thử nghiệm 3: Export filter tightness**

| Filter config | Pairs kept | Coverage | Ratio range | Han min | Lợi/Hại |
|---------------|-----------|----------|-------------|---------|----------|
| **Strict** (orig) | 41,226 | 71.0% | [0.5, 8.0] | 4 chars | ✓ Clean, ✗ Conservative |
| **Loose** | 44,830 | 77.3% | [0.2, 12.0] | 4 chars | ✓ More pairs, ✗ Noise |
| **Score-based** | **46,880** | **80.8%** | [0.2, 12.0] | 4→rescue | ✅ **CHỌN** |

**Score-based logic:**
- Short Han (≤4 chars): Keep if score ≥0.55 (save ~1,100 valid short phrases)
- Extreme ratio (>12 or <0.2): Always drop (genuine Bertalign merge artifacts)
- Rescue out-of-range ratio: If cosine ≥0.4 (save 2,072 pairs)

**Finding:** Score-based filter = **+2,050 pairs** vs strict, **minimal quality loss**

| Lý do drop | Count | Chất lượng | Quyết định |
|-----------|-------|-----------|-----------|
| Extreme ratio >12 | 2,010 | Giả dương: 1 Han → 1000 Vi | ✗ Drop |
| Extreme ratio <0.2 | 88 | Giả dương: 1400 Han → 30 Vi | ✗ Drop |
| Han ≤4, score <0.55 | 87 | Borderline (0.50-0.55) | ✗ Drop (safe margin) |

**Giải pháp chọn:** Score-based filter (final)

---

### 2.3 Công cụ & Model cuối cùng

| Thành phần | Công cụ | Tham số | Kết quả |
|-----------|---------|--------|--------|
| **Normalize Hán** | Guwen-biaodian | 300w window, 50 overlap | 49,927 câu |
| **Embedding** | BGE-M3 | fp16, 1024-dim, batch=64 | 108K vector |
| **Alignment** | Bertalign | 2-pass DP, score ≥0.5 | 35,596 cặp |
| **Fallback** | Greedy best-match | score ≥0.4 | +12,586 cặp |
| **QC Filter** | Score-based | Ratio [0.2-12], rescue logic | 46,880 final |
| **Infrastructure** | Python 3.11 + uv | GPU: 2× RTX 3060 12GB | 1 giờ train |

---

## 3. QUY TRÌNH THỰC HIỆN

### 3.1 Mô tả tổng quan quy trình

```
INPUT (.docx Hán + Việt)
       ↓
[1. Normalize Hán] → 49,927 câu
[2. Split Việt (.docx)] → 58,032 câu  
[3. BGE-M3 Embedding] → 108K vectors
[4. Bertalign 2-pass] → 35,596 cặp
[5. Greedy fallback] → +12,586 cặp (48,182 total)
[6. Score-based filter] → 46,880 cặp (80.8% coverage)
       ↓
OUTPUT: hvb_tap{4,5,6}_parallel.tsv/xlsx
```

### 3.2 Các bước thực hiện chính

**Stage 1-2:** Normalize & Split (Hán 49,927 + Việt 58,032 câu)  
**Stage 3:** BGE-M3 fp16 embedding (108K vectors, 1024-dim)  
**Stage 4:** Bertalign 2-pass DP → 35,596 cặp (score ≥0.5)  
**Stage 5:** Greedy fallback (best-match score ≥0.4) → +12,586 cặp  
**Stage 6:** Score-based QC filter → **46,880 final cặp**

---

## 4. KẾT QUẢ ĐẠT ĐƯỢC

### 4.1 Khối lượng dữ liệu xử lý

| Giai đoạn | Input | Output | Ghi chú |
|-----------|-------|--------|---------|
| Normalize Hán | 233K dòng | 49,927 câu | Guwen-biaodian |
| Split Việt | .docx 3 tập | **58,032 câu** | Bypass OCR noise |
| Embedding | 108K câu | 108K vector (1024-dim) | BGE-M3 fp16 |
| Bertalign | 108K vector | **35,596 cặp** | 2-pass DP |
| Greedy fallback | 48,182 pairs (48,182 couplets) | **+12,586 cặp** | Score ≥0.4 |
| Score-based filter | 48,182 cặp | **46,880 cặp** | Coverage 80.8% |

### 4.2 Chỉ số chất lượng cuối

| Chỉ số | Giá trị | Ghi chú |
|--------|---------|---------|
| **Cosine similarity (Bertalign)** | **0.624** | BGE-M3 |
| Cosine similarity (Greedy) | 0.644 | Cao hơn Bertalign |
| Cosine ≥ 0.7 (high-conf) | 4,777 (10.2%) | Quality subset |
| **Coverage Việt** | **80.8%** | 46,880 / 58,032 |
| **Ratio mean** | **3.74** | Vi dài hơn Hán |
| Pairs rescued by cosine | 2,072 | Ratio out-of-range saved |

### 4.3 Ví dụ minh họa

**Ví dụ 1: Greedy rescue (low Bertalign confidence)**
```
Hán:    己 (1 char)
Việt:   "Bản thân tôi" (12 chars)
Score:  0.42 (low Bertalign score)
→ RESCUE: Score ≥0.4 (greedy threshold) ✓
```

**Ví dụ 2: Score-based filter (high semantic, low language model confidence)**
```
Hán:    千載 (2 chars)
Việt:   "nghìn năm" (8 chars)
Score:  0.75 (high semantic)
Ratio:  4.0 (in range [0.2-12])
→ KEEP: Ratio in range, score high ✓
```

---

## 5. ĐÁNH GIÁ & THẢO LUẬN

### 5.1 Kết quả đánh giá

#### 5.1.1 Tác động từng giai đoạn
| Bước | Delta | Tác dụng | Tích lũy |
|------|-------|---------|---------|
| Bertalign baseline | — | 2-pass DP | 35,596 |
| Greedy fallback | **+35.4%** | Capture Vi chưa align | 48,182 |
| Score-based filter | **-2.7%** | QC (keep high-semantic) | **46,880** |

#### 5.1.2 So sánh filter strategy
| Strategy | Pairs | Coverage | Quality |
|----------|-------|----------|---------|
| **Strict [0.5-8.0]** | 41,226 | 71.0% | ✅ Clean (0 noisy extreme ratio) |
| **Loose [0.2-12.0]** | 44,830 | 77.3% | ✓ More pairs, ~2% noise |
| **Score-based** | **46,880** | **80.8%** | ✅ **Optimal** (semantic rescue) |

### 5.2 Những vấn đề gặp phải & Giải pháp

| Vấn đề | Tác động | Giải pháp | Hiệu quả |
|--------|----------|-----------|---------|
| **BGE-M3 OOD** (Hán cổ) | Cosine discrimination yếu (~0.06 IQR) | Cosine rescue ≥0.4-0.55 | ✓ +2,072 pairs |
| **Bertalign false positives** (extreme ratio) | Ratio >12 hoặc <0.2 | Hard ceiling filter | ✓ Drop 2,098 noise |
| **Short-Han ambiguity** | Han ≤4 chars low score | Score threshold ≥0.55 | ✓ Drop 87 borderline |
| **No ground truth** | Mọi metric proxy (cosine, sino, ratio) | Manual spot-check 30 cặp | ✓ Quality spot-verified |

### 5.3 Phân tích nguyên nhân

**Tại sao Bertalign + score-based tối ưu:**
1. **Bertalign:** 2-pass DP + anchor → khôi phục 5× pairs vs Vecalign
2. **BGE-M3:** Cosine similarity (vs LaBSE distance) → semantic alignment tốt
3. **Greedy fallback:** Phủ cover vi chưa align (Bertalign DP skip)
4. **Score-based filter:** Balance semantic (cosine ≥0.4) vs structural (ratio [0.2-12])

**Tại sao chọn .docx thay vì PaddleOCR output:**
- `.docx` (input gốc): 58,032 câu sạch (digitize pre-existing, người tách)
- PaddleOCR (từ PDF): 66,615 câu raw (over-segment + noise ~8% CER)
- Result: Ít câu nhưng align tốt hơn (+5.5% coverage) → **Chất lượng > Số lượng**
- Note: .docx không phải output của PaddleOCR, mà là input digitize sẵn

---

## 6. KẾT LUẬN & HƯỚNG PHÁT TRIỂN

### 6.1 Những nội dung đã hoàn thành

✅ **Thực hiệm & so sánh:**
- OCR models (PaddleOCR vs MinerU voting) → Chọn .docx digitize sạch
- Alignment models (Vecalign vs Bertalign) → Chọn Bertalign + BGE-M3
- Filter strategies (strict vs loose vs score-based) → Chọn score-based

✅ **Pipeline hoàn chỉnh:**
- Stage 1-2: Normalize + Split (80,391 Hán + 58,032 Việt)
- Stage 3: BGE-M3 embedding (108K vectors)
- Stage 4-5: Bertalign + Greedy fallback (48,182 cặp)
- Stage 6: Score-based QC filter (**46,880 cặp final, 80.8% coverage**)

✅ **Deliverable:**
- `hvb_tap{4,5,6}_parallel.tsv/xlsx` (46,880 cặp, 0-indexed pair_id)
- Per-tap breakdown: tap4 (17,899), tap5 (11,474), tap6 (17,507)

### 6.2 Những hạn chế còn tồn tại

| Hạn chế | Tác động | Giải pháp tương lai |
|---------|----------|-------------------|
| Cosine discrimination OOD | BGE-M3 score ~0.6 mean (hẹp) | Fine-tune BGE-M3 trên Hán cổ |
| Bertalign monotonicity assumption | Skip non-monotonic align (~5%) | Chapter-anchor pre-segment |
| Không ground truth chuyên gia | Mọi metric proxy | Mini gold-set (100-200 cặp) |

### 6.3 Hướng phát triển tương lai

**Phase 2:** Validate + Extend
1. Mini gold-set (100-200 cặp nhãn chuyên gia)
2. Chapter-anchor pre-alignment
3. Fine-tune BGE-M3 trên Hán cổ

**Phase 3:** Scale + Evaluate
4. Expand to full Đại Nam Thực Lục (tập 1-10)
5. Round-trip translation eval (LLM Qwen 7B)
6. Publish corpus + benchmark

### 6.4 Tóm tắt kết quả

**Deliverable cuối:** 46,880 cặp song ngữ Hán-Việt
- ✅ **Coverage:** 80.8% Việt sentences
- ✅ **Quality:** Cosine 0.624 (Bertalign), 0.644 (Greedy)
- ✅ **Per-tap:** tap4=17,899, tap5=11,474, tap6=17,507
- ✅ **Reproducible:** One-command pipeline via `HVB_ALIGNER=bertalign`

**Stack chốt:**
- Bertalign + BGE-M3 (alignment)
- Score-based filter (QC)
- .docx digitize Việt (input)
- Cosine rescue ≥0.4 (semantic priority)

---

**Hoàn thành: 2026-07-24**  
**Report: <10 trang ✓**

