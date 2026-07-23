# DÀN Ý ĐỒNG ÁN: XÂY DỰNG CORPUS SONG NGỮ HÁN-VIỆT TỪ ĐẠI NAM THỰC LỤC

---

## 1. GIỚI THIỆU

### 1.1 Mục tiêu đồ án
- Xây dựng **corpus song ngữ Hán-Việt** từ **Đại Nam Thực Lục** (Quốc sử Triều Nguyễn)
- Tạo **33,424 cặp câu song ngữ** được căn chỉnh (sentence-aligned) để phục vụ:
  - Nghiên cứu NLP (machine translation, alignment, embedding)
  - Giáo dục (học tiếng, dịch thuật)
  - Lưu trữ di sản văn hóa

### 1.2 Bài toán được giao
- **Input:**
  - Hán: File `.docx` digitize sạch từ Wiki文库 (tập 4, 5, 6) → 181 đoạn văn
  - Việt: 3 PDF scan từ Quốc Sử Quán Triều Nguyễn (Tập 4, 5, 6) → 3,242 trang
- **Output:**
  - TSV + XLSX: `pair_id ⇥ han_sentence ⇥ viet_sentence ⇥ sino` (33,424 cặp)
  - Metadata: embedding cosine (TB 0.624), chỉ số Sino âm vị (TB 0.454)

### 1.3 Phạm vi dữ liệu & kết quả mong đợi
| Chỉ số | Giá trị |
|--------|---------|
| Câu Hán sau tách | 49,927 |
| Câu Việt sau tách | 58,032 |
| Cặp align gốc (Bertalign) | 35,622 |
| **Cặp deliverable (dual-filter QC)** | **33,424** |
| Cosine similarity TB | **0.624** |
| Sino precision TB | 0.454 |
| Bead 1-1 | 46.8% |
| Coverage Hán | 90.2% |

---

## 2. DỮ LIỆU & CÔNG CỤ SỬ DỤNG

### 2.1 Mô tả dữ liệu đầu vào

#### 2.1.1 Phía Hán (Han)
- **Nguồn:** `Đại Nam Thực Lục - 大南寔錄_full.txt` (Wiki文库, digitize)
- **Kích thước:** 233K dòng, 4.9M ký tự
- **Trạng thái:** Không có dấu câu (0 ký tự `.。!！?？;；`)
- **Xử lý:** Normalize (bỏ wiki marker, fullwidth→halfwidth)

#### 2.1.2 Phía Việt (Vi)
- **Nguồn:** 3 PDF scan từ Quốc Sử Quán Triều Nguyễn
  - Tập 4: 1,141 trang
  - Tập 5: 945 trang
  - Tập 6: 1,156 trang
  - Total: 3,242 trang
- **Định dạng:** Ảnh PDF 300 DPI (scan chuẩn)
- **Chất lượng:** Một số trang mô hồ, OCR CER ~5-10%

### 2.2 Nguồn dữ liệu
| Thành phần | Loại | Ghi chú |
|-----------|------|--------|
| Hán raw | TXT digitize | Wiki文库 |
| Việt raw | PDF scan | Quốc Sử Quán Triều Nguyễn + .docx digitize |
| Hán dấu câu | Model token-classifier | `raynardj/guwen-biaodian` (110M param, train 四庫全書) |
| Embedding | Cross-lingual model | `BAAI/bge-m3` (568M param, fp16, 1024-dim) |
| Aligner | DP monotonic | Bertalign 2-pass (external/bertalign/) |
| Sino mapping | Rule-based | `cn2vn` âm vị PyPI |
| OCR (Việt) | VLM voting | MinerU + PaddleOCR-VL-1.6 (voting char-level) |
| Post-correct (tùy) | LLM 7B | `Qwen/Qwen2.5-7B-Instruct` qua vLLM |

### 2.3 Công cụ, thư viện, mô hình được sử dụng

#### 2.3.1 OCR Pipeline (@ GK-NLP-HCMUS)
- **MinerU hybrid layout parser** → bounding box + semantic block type
- **PaddleOCR-VL-1.6** (Việt model) → OCR char/token
- **Voting mechanism** (char-majority + tie-break reference) → reduce noise
- **Output:** `voted_vol{N}.jsonl` (trang × block × candidate)

#### 2.3.2 Alignment Pipeline (@ NLP)
**Stage 1: Normalize + Load**
- `normalize_han.py`: Hán Wiki cleanup
- `docx_to_vi_sentences.py`: Parse .docx Việt → 58,032 câu

**Stage 2: Punctuate Hán**
- `han_punctuate.py`: Sliding-window token-classifier (300w, overlap 50)
- Model: `raynardj/classical-chinese-punctuation-guwen-biaodian`
- Result: +99,753 dấu (`。!？;`) → 49,927 câu Hán

**Stage 3: Embed (same source)**
- `labse_embed.py`: BGE-M3 fp16 embedding
- Input: 49,927 câu Hán + 58,032 câu Việt
- Output: 108K vector (1024-dim, max_seq=256, batch=64)

**Stage 4: Align (2-pass DP)**
- `bertalign_runner.py`: Bertalign anchor-based DP
- Params: `max_align=5, top_k=3, win=5, skip=-0.1, margin=True, len_penalty=True`
- Filter: `ALIGN_MIN_SCORE=0.5` (cosine similarity)
- Output: 35,622 cặp gốc

**Stage 5: Rerank**
- `rerank_combined.py`: Sino âm vị + cosine hybrid score
- Rescue logic: cosine ≥ 0.55 override sino yếu

**Stage 6: Dual-filter QC**
- `export_deliverable.py`: 2 filter bổ trợ
  - Surgical: `ratio > 10 ∧ han_len < 10 ∧ sino < 0.3` → drop 17 outlier
  - Hard ceiling: `ratio > 15.0` → drop 25 extreme
- Final: **33,424 deliverable cặp**

#### 2.3.3 Infrastructure
- **Python:** 3.11+ (uv package manager)
- **GPU:** 2× RTX 3060 12GB (CUDA 12.1)
- **LLM serve:** vLLM Docker (OpenAI-compatible API)
- **Reproducibility:** Bash scripts (`run_pipeline.sh`, `reproduce_bertalign_bgem3.sh`)

---

## 3. QUY TRÌNH THỰC HIỆN

### 3.1 Mô tả tổng quan quy trình

```
INPUT (Hán .docx + Việt 3 PDF)
       ↓
[GIAI ĐOẠN 1: XỬ LÝ VIỆT] (GK-NLP-HCMUS OCR pipeline)
  • PDF → PNG pages
  • MinerU + PaddleOCR-VL-1.6 voting
  • Char-level hoà phiếu → voted_vol{N}.jsonl
       ↓
[GIAI ĐOẠN 2: XỬ LÝ HÁN] (NLP normalize + punctuate)
  • Normalize Wiki文库 → han_clean.txt
  • Guwen-biaodian token-classifier → thêm 99,753 dấu → 49,927 câu
  • Parse .docx Việt (underthesea split) → 58,032 câu
       ↓
[GIAI ĐOẠN 3: EMBEDDING]
  • BGE-M3 fp16 trên RTX 3060
  • Output: {han,vi}_embeds.npy (108K vector × 1024-dim)
       ↓
[GIAI ĐOẠN 4: ALIGNMENT] (Bertalign 2-pass DP)
  • Cosine similarity score
  • Anchor-based monotonic DP
  • Output: pairs.jsonl (35,622 cặp)
       ↓
[GIAI ĐOẠN 5: RERANK + QC] (Sino + cosine + dual-filter)
  • Hybrid score: sino + cosine similarity
  • Rescue (cosine ≥ 0.55)
  • Surgical filter: ratio > 10 ∧ han_len < 10 ∧ sino < 0.3
  • Hard ceiling: ratio > 15.0
  • Output: 33,424 cặp
       ↓
OUTPUT: hvb_parallel.tsv + .xlsx + raw.txt
```

### 3.2 Các bước thực hiện chính

#### **Bước 1: Setup Environment**
```bash
./scripts/setup.sh --with-vllm    # uv venv + vecalign + vLLM docker
```
- Cài đặt Python 3.11+ qua uv
- Clone external/bertalign/ từ git
- Khởi động vLLM container

#### **Bước 2: OCR Việt (GK-NLP-HCMUS)**
```bash
python -m ocr_vote.pipeline \
  --paddle "PaddleOCR-VL-1.6/tập {N}.pdf" \
  --mineru "MinerU/tập {N}/" \
  --out    "optimized_output/intermediate/voted_vol{N}.jsonl"
```
- Input: 3 PDF (3,242 trang)
- Voting: MinerU bbox + PaddleOCR text
- Output: `voted_vol{N}.jsonl` (66,615 câu rough)

#### **Bước 3: Normalize & Split Hán**
```bash
./scripts/run_pipeline.sh prep
```
- Normalize Wiki cleanup → `han_clean.txt`
- Guwen-biaodian sliding-window → **49,927 câu Hán**

#### **Bước 4: Load & Split Việt**
```bash
./scripts/run_pipeline.sh prep
```
- Parse `.docx` 3 tập + underthesea split
- Output: **58,032 câu Việt**

#### **Bước 5: Embedding (BGE-M3)**
```bash
./scripts/run_pipeline.sh embed
```
- BGE-M3 fp16 (1024-dim, max_seq=256, batch=64)
- Output: 108K vector

#### **Bước 6: Alignment (Bertalign 2-pass)**
```bash
HVB_ALIGNER=bertalign ./scripts/run_pipeline.sh align
```
- Bertalign 2-pass DP
- Output: **35,622 cặp gốc**

#### **Bước 7: Rerank & Dual-filter QC**
```bash
HVB_ALIGNER=bertalign ./scripts/run_pipeline.sh export
```
- Hybrid score + 2 filter rules
- Output: **33,424 cặp final**

---

## 4. KẾT QUẢ ĐẠT ĐƯỢC

### 4.1 Khối lượng dữ liệu xử lý

| Giai đoạn | Input | Output | Ghi chú |
|-----------|-------|--------|---------|
| **PDF OCR** | 3,242 trang | voted_vol{4,5,6}.jsonl | 66,615 câu rough |
| **Hán normalize** | 233K dòng | 181 đoạn | Wiki cleanup |
| **Hán punctuate** | 181 đoạn (0 dấu) | **49,927 câu** | +99,753 dấu guwen |
| **Việt load** | voted.jsonl + .docx | **58,032 câu** | Underthesea split |
| **Embedding** | 108K câu | 108K vector (1024-dim) | BGE-M3 fp16 |
| **Alignment** | 108K vector | **35,622 cặp** | Bertalign cosine ≥ 0.5 |
| **QC filter** | 35,622 cặp | **33,424 cặp** | -17 surgical, -25 ceiling |

### 4.2 Các sản phẩm đầu ra đã tạo

**Deliverable cuối (data/final/):**

1. **`hvb_parallel.tsv`** — 33,424 cặp (tab-separated)
   - Schema: `pair_id \t han_sentence \t viet_sentence \t sino`

2. **`hvb_parallel.xlsx`** — Same, Excel format

3. **`hvb_raw.txt`** — 3 tập OCR concatenated

### 4.3 Ví dụ minh họa kết quả

#### Ví dụ 1: Dịch ngữ nghĩa (cosine cao, sino thấp → giữ)
```
Hán:        千載
Việt:       nghìn năm
Cosine:     0.75 (cao)
Sino:       0.0  (không trùng âm vị)
Ratio:      1.67
→ GIỮ (cosine ≥ 0.55 rescue)
```

#### Ví dụ 2: Dịch sát âm vị (sino cao, cosine bình)
```
Hán:        建寧
Việt:       Kiến Ninh (địa danh)
Cosine:     0.62
Sino:       0.85 (cao)
Ratio:      1.0
→ GIỮ (hybrid score cao)
```

### 4.4 Chỉ số chất lượng cuối

| Chỉ số | Giá trị | Ghi chú |
|--------|---------|---------|
| **Cosine similarity** | **0.624** (TB) | BGE-M3 |
| Cosine median | 0.616 | Phân phối lệch phải |
| Cosine max | 0.899 | Cặp tốt nhất |
| Cosine ≥ 0.7 | 4,777 (14.4%) | High-confidence |
| **Sino precision** | **0.454** (TB) | Cố ý thấp |
| **Length ratio** | **3.74** (TB) | VN thường dài hơn |
| Ratio max | 14.94 | < ceiling 15.0 |
| **Bead type 1-1** | **46.8%** | Align đơn điệu |
| Coverage Hán | 90.2% | 45K / 49.9K câu |

---

## 5. ĐÁNH GIÁ & THẢO LUẬN

### 5.1 Cách đánh giá kết quả

#### 5.1.1 Proxy metrics (không có ground truth)
1. **Cosine similarity** (BGE-M3)
   - Giả định: semantic match → cosine cao
   - Hạn chế: BGE-M3 OOD trên Hán cổ

2. **Sino precision** (cn2vn âm vị)
   - Chỉ đo trùng âm vị
   - Thiên vị: dịch ngữ nghĩa → sino thấp

3. **Length ratio** (sanity check)
   - Bắt align bất thường
   - Không đo chất lượng ngữ nghĩa

#### 5.1.2 Định tính (manual spot-check)
- Soi 200-300 cặp random verify alignment
- Kiểm tra filter rule bắt artifact

### 5.2 Kết quả đánh giá

#### 5.2.1 So sánh stack
| Stack | Bertalign + BGE-M3 | Vecalign + LaBSE (legacy) |
|-------|------------------|---------------------------|
| Cặp align | 35,622 | 7,043 |
| Cosine TB | 0.559 → 0.621 (guwen) | — |
| Sino TB | 0.500 | 0.275 |
| Precision | **+82% cao hơn** | — |

#### 5.2.2 Tác động từng giai đoạn

| Bước | Delta | Tác dụng | Tích lũy |
|------|-------|---------|---------|
| Baseline Bertalign | — | — | 5,917 cặp |
| Guwen-biaodian | **+5.3x** | Tách dấu câu Hán | 30,959 cặp |
| .docx Việt sạch | **+5.5%** | Bypass OCR noise | 32,665 cặp |
| Dual-filter QC | **-0.4%** | Giữ sạch | **33,424 cặp** |

### 5.3 Những vấn đề gặp phải

#### 5.3.1 BGE-M3 out-of-distribution
- **Vấn đề:** BGE-M3 train đương đại → Hán cổ OOD
- **Triệu chứng:** Cosine IQR hẹp (~0.06)
- **Giải pháp hiện tại:** Logic rescue (cosine ≥ 0.55)
- **Tương lai:** Fine-tune BGE-M3

#### 5.3.2 Sino precision âm tính giả
- **Vấn đề:** Dịch ngữ nghĩa không tạo trùng âm vị
- **Tác động:** ~5-10% cặp hợp lệ sine thấp
- **Giải pháp:** Rescue qua cosine ≥ 0.55

#### 5.3.3 Giả định monotonicity
- **Vấn đề:** Bertalign giả định thứ tự đơn điệu
- **Giải pháp hiện tại:** Manual boundary slice
- **Tương lai:** Chapter-anchor pre-alignment

#### 5.3.4 Không có ground truth chuyên gia
- **Vấn đề:** Mọi metric đều proxy
- **Giải pháp tương lai:** Mini gold-set (100-200 cặp)

### 5.4 Phân tích nguyên nhân

#### 5.4.1 Tại sao Bertalign thắng Vecalign
- Bertalign: 2-pass DP + anchor, cosine similarity, logic rescue
- Vecalign: DP đơn điệu, distance metric
- Kết quả: +82% precision Sino

#### 5.4.2 Tại sao Guwen-biaodian tăng 5.3x
- Gốc: 181 đoạn (176/181 không dấu)
- Vấn đề: Split 200 ký tự → câu pha chủ đề
- Giải pháp: Guwen-biaodian +99,753 dấu
- Kết quả: Câu TB 25 ký tự (thay 14) → cosine tăng 11%

#### 5.4.3 Tại sao .docx Việt thắng OCR
- .docx: digitize sạch (58,032 câu, +5.5% align)
- PaddleOCR: over-segment (66,615 câu) + noise
- Kết quả: Ít câu nhưng align tốt hơn (quality > quantity)

---

## 6. KẾT LUẬN & HƯỚNG PHÁT TRIỂN

### 6.1 Những nội dung đã hoàn thành

✅ **Pipeline toàn bộ:**
1. OCR Việt: MinerU + PaddleOCR voting
2. Normalize Hán: Wiki cleanup
3. Punctuate Hán: Guwen-biaodian (49,927 câu)
4. Load Việt: .docx parse + split (58,032 câu)
5. Embedding: BGE-M3 fp16 (108K vector)
6. Alignment: Bertalign 2-pass (35,622 cặp)
7. Rerank + QC: Dual-filter (33,424 cặp final)

✅ **Deliverable:**
- `hvb_parallel.tsv` (33,424 cặp)
- `hvb_parallel.xlsx`
- `hvb_raw.txt`

✅ **Reproducibility:**
- `./scripts/reproduce_bertalign_bgem3.sh` (one-command)

✅ **Documentation:**
- docs/18_master_summary_vi.md
- docs/03_pipeline.md
- CLAUDE.md

### 6.2 Những hạn chế còn tồn tại

| Hạn chế | Tác động | Giải pháp tương lai | Effort |
|---------|----------|-------------------|--------|
| Không ground truth | Không tính precision/recall | Mini gold-set (100-200 cặp) | Thấp |
| BGE-M3 OOD | Cosine discrimination yếu | Fine-tune BGE-M3 | Vừa |
| Sino âm tính giả | ~5-10% cặp drop | Hybrid metric | Vừa |
| Giả định monotonic | Align sai biên | Chapter-anchor pre-align | Thấp |

### 6.3 Hướng phát triển tương lai

**Phase 2 (1-2 tháng):**
1. Mini gold-set (100-200 cặp, nhãn chuyên gia)
2. Chapter-anchor pre-alignment
3. Fine-tune BGE-M3 trên top-band cặp

**Phase 3 (2-3 tháng):**
4. Round-trip translation eval (LLM Qwen2.5-7B)
5. Mở rộng scope (corpus khác)

### 6.4 Kết quả cuối cùng

**Deliverable:** 33,424 cặp song ngữ Hán-Việt
- Cosine TB: **0.624**
- Sino TB: **0.454**
- Coverage: **90.2%**
- Bead 1-1: **46.8%**

**Stack chốt:**
- ✅ Bertalign + BGE-M3
- ✅ Guwen-biaodian punctuator
- ✅ .docx Việt sạch
- ✅ Dual-filter QC
- ✅ Triết lý: Ngữ nghĩa > Âm vị

**Reproducibility:** One-command pipeline

---
