# Troubleshooting

Lỗi thường gặp + cách fix.

## OCR errors

### PaddleOCR OOM

**Triệu chứng:**
```
RuntimeError: CUDA out of memory. Tried to allocate 2.00 GiB
```

**Fix:**
- Giảm `PADDLE_BATCH` trong `src/utils/config.py` (mặc định 8 → 4 hoặc 2)
- Disable GPU tạm thời: `PADDLE_USE_GPU = False` (chậm 5-10x)
- Restart kernel/session để clear VRAM

### PaddleOCR trả text rỗng

**Triệu chứng:** File `.txt` output rỗng hoặc chỉ có newline.

**Nguyên nhân:**
- Page scan mờ/nhỏ
- Page chứa hình ảnh không có text (trang bìa, minh họa)
- Page rotated 90/180 độ

**Fix:**
- Verify `use_angle_cls=True` trong `paddle_ocr.py`
- Inspect PNG gốc: kiểm tra `data/interim/vi_pages/tap4_0042.png`
- Nếu mờ: re-render ở DPI cao hơn (`OCR_DPI = 400`)

### CER > 15% trên gold pages

**Check:**
1. So sánh `data/interim/vi_ocr_raw/tap4_page_0042.txt` vs `data/gold/ocr_gold/tap4_page_0042.txt`
2. Phân loại lỗi: Nôm chars / dấu câu / dấu thanh / common confusions (vd `0` vs `o`)
3. Nếu Nôm nhiều → cần TrOCR fine-tune

**Fix tăng chất lượng:**
- TrOCR fine-tune: tạo `src/02_ocr/trocr_finetune.py` train trên 50-100 gold pages
- Đổi sang Gemini/GPT-4o Vision OCR (API) nếu local fail
- Post-process mạnh hơn: tăng LLM chunk overlap, multi-shot prompt

## LLM/vLLM errors

### Connection refused

```
httpx.ConnectError: [Errno 111] Connection refused
```

**Check:**
```bash
docker ps | grep vllm
curl http://localhost:8001/v1/models
```

**Fix:**
```bash
docker start vllm
sleep 15
curl http://localhost:8001/v1/models
```

### Model not found

```
openai.NotFoundError: 404 The model `Qwen/Qwen2.5-7B-Instruct` does not exist
```
hoặc vLLM log trong `docker logs vllm`:
```
ERROR: Model Qwen/Qwen2.5-7B-Instruct not loadable
```

**Fix:**
```bash
docker logs vllm --tail 50
# vLLM tự download weights lần đầu; nếu network fail, restart:
docker restart vllm
```

### LLM timeout (> 180s)

**Nguyên nhân:** Chunk quá dài, hoặc VRAM chia sẻ với stage khác.

**Fix:**
- Giảm chunk size trong `llm_correct.py`: `chunk_text(text, max_chars=300)`
- Tăng timeout: `LLM_TIMEOUT = 300` trong config
- Stop các process GPU khác khi chạy LLM

### LLM hallucination (sửa text quá tay)

**Triệu chứng:** Output LLM khác hoàn toàn input, thêm info không có.

**Fix:**
- Lower temperature: `options={"temperature": 0.1}` trong `llm_correct.py`
- Few-shot prompt với ví dụ cụ thể
- Đổi sang model lớn hơn (qwen2.5:14b nếu VRAM đủ)

## Sentence split errors

### Han sentence quá dài (> 500 chars)

**Nguyên nhân:** Missing terminator `。` trong text.

**Check:**
```bash
jq -r 'select((.text | length) > 500) | .text' data/interim/han_sentences.jsonl | head -5
```

**Fix:** Thêm custom terminator vào `split_han.py` `ZH_TERM_RE` (vd thêm `\n` nếu structured).

### Việt sentence quá ngắn (< 3 chars)

**Nguyên nhân:** OCR broken thành nhiều mảnh nhỏ.

**Fix:** Tăng filter trong `split_vi.py`: `if len(s) >= 5` thay vì 2.

### Underthesea crash

```
ModuleNotFoundError: No module named 'underthesea'
```

**Fix:** `uv add underthesea` hoặc check venv activated.

## Vecalign errors

### Vecalign subprocess fail

```
FileNotFoundError: external/vecalign/vecalign.py
```

**Fix:**
```bash
git clone https://github.com/neulab/vecalign.git external/vecalign
```

### Embeddings shape mismatch

```
AssertionError: src embeddings 1000 != sentences 1200
```

**Nguyên nhân:** Stage 4 chạy trước khi Stage 3 xong, hoặc sentences JSONL bị append.

**Fix:** Xóa checkpoint của 2 stage liên quan, re-run từ Stage 3.

### Vecalign chạy > 4 giờ

**Bottleneck:** Dynamic programming O(n*m) với n,m > 100K.

**Fix:** Chunk theo chapter/section trước (cắt corpus thành 10 phần, chạy song song).

## Eval errors

### COMET download fail

```
ConnectionError: https://huggingface.co/unmt/comet-qe-22
```

**Fix:** Pre-download:
```bash
uv run python -c "
from comet import download_model
download_model('unmt/comet-qe-22')
"
```

### BERTScore lang error

```
ValueError: Unsupported language: zh
```

**Fix:** BERTScore cần tự detect model. Force model trong `auto_metrics.py`:
```python
from bert_score import score
P, R, F1 = score(src, tgt, model_type="bert-base-multilingual-cased", verbose=False)
```

### Hold-out MT không converge

**Triệu chứng:** BLEU hold-out < 5.

**Nguyên nhân:** < 5000 pairs hoặc data noise quá cao.

**Fix:**
- Skip hold-out (cho phép `{"skipped": true}`)
- Lấy thêm data: add tập 1-3 PDFs (nếu có)
- Filter aligned pairs: chỉ giữ `score > 0.7`

### LLM ensemble JSON parse fail

**Triệu chứng:** `all_scores[model]` chứa `{}` hoặc `{"error": ...}`.

**Fix:** Improve prompt trong `llm_ensemble_judge.py`:
- Explicit JSON schema trong prompt
- Retry mechanism (chưa implement, có thể thêm try-except + retry)

## Performance issues

### Stage chạy chậm bất thường

**Check:**
- GPU utilization: `nvidia-smi -l 1`
- CPU: `htop`
- Disk I/O: `iostat -x 1`

**Common:**
- DataLoader bottleneck → tăng `num_workers`
- Disk full → check `df -h`
- Swap thrashing → RAM thiếu

### PDF → PNG chậm

**Fix:**
- Tăng `thread_count` trong `pdf_to_images.py` (mặc định 4 → 8)
- Lower DPI nếu OCR chấp nhận (`OCR_DPI = 200`)

## Data integrity

### Han TXT không phải UTF-8

**Check:**
```bash
file data/raw/*.txt
iconv -f UTF-8 -t UTF-8 data/raw/*.txt > /dev/null && echo "OK"
```

**Fix:** Convert encoding nếu cần:
```bash
iconv -f GB18030 -t UTF-8 data/raw/han_raw.txt > data/raw/han.txt
```

### Filename encoding issue

**Triệu chứng:** `FileNotFoundError` cho path có dấu Việt.

**Fix:**
- Luôn dùng absolute paths
- Trong shell: quote tất cả paths `"path/with spaces/"`
- Trong Python: dùng `pathlib.Path`, không string concat

### JSONL corrupted (truncated line)

**Check:**
```bash
jq -c '.' data/interim/han_sentences.jsonl > /dev/null
```

**Fix:** Drop last line nếu truncate:
```bash
head -n -1 data/interim/han_sentences.jsonl > /tmp/clean.jsonl
mv /tmp/clean.jsonl data/interim/han_sentences.jsonl
```

## Checkpoint issues

### Stage skip khi không nên

**Nguyên nhân:** Checkpoint file exist từ run trước.

**Fix:** Xóa checkpoint file tương ứng (`data/interim/.checkpoint/{stage_name}`) rồi re-run stage.

### Stage không skip khi nên

**Nguyên nhân:** Hook bắt file tên khác với stage name.

**Check:** Tên stage trong `run_pipeline.sh` case statement phải match `run` first arg.

## vLLM docker issues

### Container restart loop

```bash
docker logs vllm --tail 50
```

**Common:**
- VRAM thiếu cho model → giảm `--gpu-memory-utilization` hoặc chọn model nhỏ hơn
- Permission volume → fix ownership bằng `chown`

### Lost weights sau reboot

**Nguyên nhân:** Volume bị clear.

**Fix:**
- vLLM tự re-download weights HuggingFace lần đầu start: `docker restart vllm` rồi đợi log "Application startup complete"
- Hoặc persist volume: `-v vllm:/root/.cache/huggingface` (mặc định trong `setup.sh --with-vllm`)

## Getting help

- Check log stage: redirect output `./scripts/run_pipeline.sh prep 2>&1 | tee prep.log`
- Re-run với verbose: thêm `set -x` vào script
- Verify dependencies: `uv run python -c "import paddleocr, sentence_transformers, hanlp, underthesea; print('OK')"`
