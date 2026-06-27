# Evaluation Methodology

Đánh giá chất lượng corpus HVB — full-auto, 5 trụ.

## Triết lý

Không dùng human annotation (user preference). Thay vào đó, cross-validate bằng 5 phương pháp độc lập:

1. **Auto metrics** — proxy chất lượng MT (COMET, BERTScore, BLEU)
2. **External benchmark** — sanity check pipeline trên data đã biết (FLORES-200)
3. **Round-trip consistency** — dịch ngược + compare, đo thông tin bảo toàn
4. **Internal hold-out MT** — train mini-NMT, eval trên hold-out
5. **LLM ensemble judge** — multi-LLM scoring + agreement

## 5 trụ — chi tiết

### 7a. Auto Metrics (`auto_metrics.py`)

**Mục đích:** Đo trực tiếp chất lượng từng cặp aligned.

**Metrics:**

| Metric | Model | Score range | Target |
|--------|-------|-------------|--------|
| **LaBSE cosine** | sentence-transformers/LaBSE | [-1, 1] | mean > 0.6 |
| **COMET-QE-22** | unmt/comet-qe-22 | [-1, 1] | mean > 0.5 |
| **BERTScore F1** | bert-score (mBERT) | [0, 1] | mean > 0.7 |
| **BLEU zh→vi** | sacrebleu | [0, 100] | > 10 |
| **chrF++ zh→vi** | sacrebleu | [0, 100] | > 35 |
| **BLEU vi→zh** | sacrebleu | [0, 100] | > 10 |

**Lưu ý:** BLEU/chrF trên cặp aligned không phải đánh giá MT thật (không có separate reference). Chỉ proxy cho lexical overlap. COMET-QE-22 là reference-free QE model — tin cậy hơn.

**Output:** `data/final/eval/auto_metrics.json`

### 7b. FLORES-200 Sanity Check (`flores_sanity.py`)

**Mục đích:** Verify pipeline chạy đúng trên data đã biết. **KHÔNG phải đánh giá domain HVB** (FLORES là modern Chinese news/wiki, không phải Hán cổ).

**Pipeline test:**
1. Load FLORES-200 devtest zh-vi (2009 pairs)
2. Embed zh + vi bằng LaBSE
3. Tính similarity matrix `e1 @ e2.T`
4. **Precision@1**: với mỗi zh row, check argmax có phải matching vi row không
5. **Diagonal cosine**: mean cosine trên diagonal (matched pairs)
6. COMET-QE trên 500 pairs đầu

**Targets (domain match chỉ zh-vi modern):**
- Precision@1 > 0.95 (pipeline chạy đúng)
- Diagonal cosine mean > 0.80
- COMET-QE > 0.40

**Output:** `data/final/eval/flores_sanity.json`

**Interpretation:** Nếu FLORES pass + HVB COMET low → chất lượng HVB thấp do domain khó, không phải do pipeline bug.

### 7c. Round-trip Consistency (`round_trip.py`)

**Mục đích:** Đo % thông tin bảo toàn qua dịch ngược.

**Pipeline:**
1. Sample 500 cặp stratified by LaBSE score (150 low / 200 mid / 150 high)
2. Với mỗi cặp: dùng LLM (Qwen2.5:7b) dịch Viet → Han
3. So sánh Han original vs Han round-trip

**Metrics:**
- **chrF++**: lexical overlap với n-gram F
- **BLEU**: chuẩn MT metric
- **BERTScore F1**: semantic overlap

**Targets:**
- chrF > 0.40 (information preserved > 40%)
- BLEU > 10
- BERTScore F1 > 0.5

**Lưu ý:** Round-trip không perfect do:
- LLM dịch Han cổ không tốt bằng Han hiện đại
- 1 câu có nhiều bản dịch Hán cổ tương đương

**Output:** `data/final/eval/round_trip.json` (summary + 500 pairs chi tiết)

### 7d. Internal Hold-out MT (`holdout_mt.py`)

**Mục đích:** Domain-match thực sự — alignment tốt ↔ MT train tốt ↔ BLEU hold-out cao.

**Pipeline:**
1. Load tất cả aligned pairs
2. **Skip** nếu < `HOLDOUT_MIN_PAIRS` (5000) — model không converge
3. Shuffle (seed 42), split 80/20 train/test
4. Fine-tune `Helsinki-NLP/opus-mt-zh-vi` (MarianMT) trên train
5. Predict trên test, compute BLEU + chrF

**Hyperparams:**
- Epochs: 3
- Batch: 16 (`MT_BATCH`)
- Max len: 256 tokens
- fp16 nếu CUDA

**Targets:**
- BLEU > 15 (rough baseline cho zh-vi MarianMT fine-tuned)
- chrF > 40

**Output:** `data/final/eval/holdout_mt.json` hoặc `{"skipped": true, "reason": "..."}`.

### 7e. LLM Ensemble Judge (`llm_ensemble_judge.py`)

**Mục đích:** Thay human bằng LLM ensemble, đo agreement (Krippendorff α).

**Pipeline:**
1. Sample 500 cặp stratified by LaBSE score
2. Mỗi LLM (`qwen2.5:7b`, `seallm:7b`) chấm 5 tiêu chí 1-5:
   - **Adequacy**: đầy đủ ý
   - **Fluency**: thông suất ngữ pháp
   - **Alignment**: cặp đúng
   - **Fidelity**: trung thực nguyên bản
   - **Terminology**: thuật ngữ lịch sử
3. Compute **Krippendorff α** (ordinal) cho mỗi criterion qua 2 LLMs
4. Mean scores per model

**Prompt template:** JSON output mode (`format: "json"`), temperature 0.1.

**Targets:**
- Krippendorff α ≥ 0.5 (lower bar than human κ 0.7 do LLM share biases)
- Mean scores > 3.5 cho mỗi criterion

**Output:** `data/final/eval/llm_ensemble_judge.json`

### 7f. NER-Bridge F1 (`ner_bridge.py`)

**Mục đích:** Validate alignment qua entity matching.

**Pipeline:**
1. Run NER cho Han (HanLP) + Viet (Underthesea)
2. Với mỗi aligned pair: collect entities từ src + tgt
3. Transliterate Han entity → Sino-Viet approximate
4. Match nếu substring overlap
5. **Coverage** = pairs có match / pairs có entity

**Target:** Coverage > 50% (không 100% do transliteration approx + NER miss).

**Output:** `data/final/eval/entities.jsonl` + log coverage.

## Overall targets

| Metric | Target | Tool | Ý nghĩa |
|--------|--------|------|---------|
| LaBSE cosine mean | > 0.6 | sentence-transformers | Semantic alignment |
| COMET-QE mean | > 0.5 | unmt/comet-qe-22 | Reference-free MT quality |
| FLORES precision@1 | > 0.95 | sacrebleu + LaBSE | Pipeline sanity |
| Round-trip chrF | > 0.40 | sacrebleu | Information preservation |
| Hold-out BLEU | > 15 | sacrebleu + MarianMT | Domain match |
| Krippendorff α | ≥ 0.5 | krippendorff | LLM judge agreement |
| NER-Bridge coverage | > 50% | custom | Entity cross-lingual |

## Phân tích lỗi (Error Analysis)

Sau khi eval xong, phân tích:

```bash
# Pairs thấp (< 0.5) — alignment fail
jq -c 'select(.labse_score < 0.5)' data/final/hvb_corpus.jsonl | head -20

# Pairs dài异常 (> 500 chars src) — split fail
jq -c 'select((.src | length) > 1500)' data/final/hvb_corpus.jsonl | head -5

# Pairs có entity nhưng không match — terminology gap
jq -c 'select(.entities | length == 0) | {src, tgt}' data/final/hvb_corpus.jsonl | head -20

# Score distribution
jq -r '.labse_score' data/final/hvb_corpus.jsonl | sort -n | awk '
  BEGIN {b0=0; b1=0; b2=0; b3=0; b4=0}
  {if ($1 < 0.2) b0++; else if ($1 < 0.5) b1++; else if ($1 < 0.7) b2++; else if ($1 < 0.9) b3++; else b4++}
  END {print "<0.2:", b0; print "0.2-0.5:", b1; print "0.5-0.7:", b2; print "0.7-0.9:", b3; print ">0.9:", b4}'
```

## Dashboard report

Tạo báo cáo tổng hợp `data/final/eval/REPORT.md` (script mẫu ngắn):

```bash
{
  echo "=== HVB Evaluation Report ==="
  echo ""
  echo "## Auto Metrics"
  jq -r '"- LaBSE cosine mean: \(.labse_cosine.mean)\n- COMET-QE mean: \(.comet_qe.mean)"' data/final/eval/auto_metrics.json
  echo ""
  echo "## FLORES Sanity"
  jq -r '"- Precision@1: \(.precision_at_1)\n- Diagonal cosine: \(.diagonal_cosine_mean)"' data/final/eval/flores_sanity.json
  echo ""
  echo "## Round-trip"
  jq -r '"- chrF: \(.summary.chrf)\n- BLEU: \(.summary.bleu)"' data/final/eval/round_trip.json
  echo ""
  echo "## Hold-out MT"
  jq -r 'if .skipped then "- SKIPPED: \(.reason)" else "- BLEU: \(.bleu)\n- chrF: \(.chrf)" end' data/final/eval/holdout_mt.json
  echo ""
  echo "## LLM Ensemble"
  jq -r '.summary | "- Adequacy (qwen): \(.mean_per_model["qwen2.5:7b"].adequacy)"' data/final/eval/llm_ensemble_judge.json
} > data/final/eval/REPORT.md
cat data/final/eval/REPORT.md
```

## Hạn chế acknowledged

- **FLORES = modern zh-vi**: chỉ sanity check, không claim đánh giá domain
- **Round-trip**: LLM dịch Han cổ kém → chrF thấp hơn kỳ vọng thực
- **Hold-out MT**: cần >= 5000 pairs, model MarianMT pretrained có thể chưa hiểu Hán cổ tốt
- **LLM ensemble**: α target 0.5 (không 0.7 như human), LLMs share biases
- **NER-Bridge**: transliteration approx, miss nhiều entity thực tế
