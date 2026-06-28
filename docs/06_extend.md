# Extend Pipeline

Patterns mở rộng: thêm PDF, đổi model, custom eval.

## Thêm PDF mới

### Thêm tập 1/2/3 của Đại Nam Thực Lục

**Bước 1:** Copy PDF vào `data/raw/`:
```bash
cp /path/to/"Đại Nam Thực Lục tập 1.pdf" data/raw/
```

**Bước 2:** Update `src/utils/config.py`:
```python
VI_PDF_TAP1 = RAW / "Đại Nam Thực Lục tập 1.pdf"
VI_PDFS = [VI_PDF_TAP1, VI_PDF_TAP4, VI_PDF_TAP5, VI_PDF_TAP6]
```

**Bước 3:** Xóa checkpoint stages liên quan + re-run:
```bash
rm data/interim/.checkpoint/pdf_to_images
rm data/interim/.checkpoint/paddle_ocr
rm data/interim/.checkpoint/llm_correct
rm data/interim/.checkpoint/split_vi
./scripts/run_pipeline.sh ocr
./scripts/run_pipeline.sh split
```

Stages embed/align/ner/eval cũng cần re-run nếu muốn include new data.

### Đổi sang source khác (vd: Đại Nam Liệt Truyện)

**Bước 1:** Đặt Han TXT mới vào `data/raw/`, edit `HAN_TXT` path trong config.

**Bước 2:** Adjust `tap_key()` logic trong `pdf_to_images.py` để match filename pattern mới (regex theo tên thay vì hardcode "tập 4/5/6").

**Bước 3:** Nếu Han source không phải Wiki文库 format, bỏ/quan lại `normalize_han.py` cho phù hợp (vd: không có `姊妹计划` header).

## Đổi models

### Đổi LLM (vLLM)

**Bước 1:** Khởi động lại container vLLM với model mới (vLLM tự download weights):
```bash
docker rm -f vllm
docker run -d --name vllm --gpus=all -p 8000:8000 \
  -v vllm:/root/.cache/huggingface \
  vllm/vllm-openai:latest \
  --model Qwen/Qwen2.5-14B-Instruct \
  --gpu-memory-utilization 0.9 --max-model-len 4096 --dtype half
# 14B cần >= 24GB VRAM; đợi log "Application startup complete"
```

**Bước 2:** Update `src/utils/config.py`:
```python
VLLM_MODEL = "Qwen/Qwen2.5-14B-Instruct"  # 14B cần >= 24GB VRAM
LLM_MODELS = [VLLM_MODEL]  # backward-compat alias
```

**Bước 3:** Re-run Stage 2b + Stage 7e:
```bash
rm data/interim/.checkpoint/llm_correct
./scripts/run_pipeline.sh ocr
```

### Đổi embedding model

**Bước 1:** Test model mới (vd: LASER):
```python
from sentence_transformers import SentenceTransformer
model = SentenceTransformer("facebook/laser-base")
```

**Bước 2:** Update `LABSE_MODEL` trong config:
```python
LABSE_MODEL = "facebook/laser-base"
```

**Bước 3:** Re-run Stage 4 + Stage 5:
```bash
rm data/interim/.checkpoint/labse_embed
rm data/interim/.checkpoint/vecalign
./scripts/run_pipeline.sh embed
./scripts/run_pipeline.sh align
```

**Lưu ý:** Vecalign expect LaBSE format. Nếu model mới khác dimension (vd LASER 1024), cần adjust `--embeddings-format` trong `vecalign_runner.py` hoặc export đúng format.

### Đổi NER model

**HanLP** → **CKIP**:
```python
# Trong ner_han.py
# Thay:
# hanlp_ner = hanlp.load(...)
# Bằng:
from ckip_transformers.nlp import CkipNerDriver
ner = CkipNerDriver(model="bert-base")
```

**Underthesea** → **PhoBERT fine-tuned**:
```python
# Trong ner_vi.py
from transformers import AutoTokenizer, AutoModelForTokenClassification, pipeline
tok = AutoTokenizer.from_pretrained("NlpHUST/ner-vietnamese-se-lstm")
model = AutoModelForTokenClassification.from_pretrained("NlpHUST/ner-vietnamese-se-lstm")
ner_pipe = pipeline("ner", model=model, tokenizer=tok, aggregation_strategy="simple")
ents = ner_pipe(text)
```

## Custom eval rubric

### Thêm criterion mới (vd: "Style preservation")

**Bước 1:** Update `LLM_JUDGE_RUBRIC` trong config:
```python
LLM_JUDGE_RUBRIC = [
    "adequacy", "fluency", "alignment", "fidelity",
    "terminology", "style_preservation",
]
```

**Bước 2:** Update prompt template trong `llm_ensemble_judge.py` để include criterion mới.

**Bước 3:** Re-run Stage 7e:
```bash
./scripts/run_pipeline.sh eval
```

### Thêm metric mới (vd: TER)

**Bước 1:** Add dependency: `uv add pyter3`.

**Bước 2:** Update `src/07_eval/auto_metrics.py`:
```python
def ter_score(pairs):
    import pyter
    scores = [pyter.ter(p["tgt"].split(), p["src"].split()) for p in pairs]
    return {"mean": statistics.mean(scores), "median": statistics.median(scores)}
```

**Bước 3:** Thêm vào `main()` result dict.

## Custom OCR post-process

### Sửa lỗi cụ thể (vd: "ℓ" → "l")

Thêm rule-based fix trong `src/02_ocr/llm_correct.py`:
```python
RULES = {
    "ℓ": "l",
    "ñ": "n",
}

def rule_fix(text: str) -> str:
    for wrong, right in RULES.items():
        text = text.replace(wrong, right)
    return text

# Apply trước LLM:
text = rule_fix(raw_text)
```

### Thêm spell-check dictionary

```python
# Trong llm_correct.py
from symspellpy import SymSpell, Verbosity

spell = SymSpell(max_dictionary_edit_distance=2)
spell.load_dictionary("data/gold/viet_dict.txt", term_index=0, count_index=1)

def correct_word(w: str) -> str:
    suggestions = spell.lookup(w, Verbosity.CLOSEST, max_edit_distance=2)
    return suggestions[0].term if suggestions else w
```

## Custom alignment strategy

### Chunk theo chapter trước

Vecalign giả định monotonic. Nếu Han TXT và Viet PDF thứ tự khác, chunk:

```python
# Trong vecalign_runner.py
def chunk_by_chapter(han_text: str, vi_text: str) -> list[tuple[str, str]]:
    """Chunk theo chapter heading như 大南寔錄前編, 正編."""
    chapters = []
    # Detect heading pattern, split both sides
    return chapters
```

### Try different alignment algorithm

Vecalign tốt nhưng có alternatives:
- **BERTalign** (`bertalign` PyPI): BERTScore-based
- **FastAlign** (`fastalign`): IBM Model 2, faster
- **Gargantua** (LLM-based): for hard cases

Implement wrapper tương tự `vecalign_runner.py`.

## Add interactive UI

### Streamlit corpus browser

```python
# src/ui/browser.py
import streamlit as st
import json

st.title("HVB Corpus Browser")

pairs = [json.loads(l) for l in open("data/final/hvb_corpus.jsonl")]
score_threshold = st.slider("Min score", 0.0, 1.0, 0.5)

for p in pairs:
    if p["labse_score"] >= score_threshold:
        st.write(f"**Score:** {p['labse_score']:.3f}")
        st.write(f"**Han:** {p['src']}")
        st.write(f"**Viet:** {p['tgt']}")
        if p.get("entities"):
            st.write("**Entities:**", p["entities"])
        st.divider()
```

Run: `uv run streamlit run src/ui/browser.py`

## Add evaluation dashboard

### HTML report generator

```python
# src/eval/report_html.py
import json
from jinja2 import Template

template = Template("""
<!DOCTYPE html>
<html>
<head><title>HVB Eval Report</title></head>
<body>
  <h1>HVB Evaluation Dashboard</h1>
  <h2>Auto Metrics</h2>
  <p>LaBSE cosine mean: {{ auto.labse_cosine.mean }}</p>
  <p>COMET-QE mean: {{ auto.comet_qe.mean }}</p>
  <h2>FLORES Sanity</h2>
  <p>Precision@1: {{ flores.precision_at_1 }}</p>
</body>
</html>
""")

auto = json.load(open("data/final/eval/auto_metrics.json"))
flores = json.load(open("data/final/eval/flores_sanity.json"))
html = template.render(auto=auto, flores=flores)
open("data/final/eval/report.html", "w").write(html)
```

## CI/CD hooks

### Pre-commit check

`.pre-commit-config.yaml`:
```yaml
repos:
  - repo: https://github.com/astral-sh/ruff-pre-commit
    rev: v0.6.0
    hooks:
      - id: ruff
        args: [--fix]
      - id: ruff-format
```

### Test smoke test

`tests/test_smoke.py`:
```python
def test_imports():
    import paddleocr
    import sentence_transformers
    import hanlp
    import underthesea
    from src.utils import config

def test_data_raw_exists():
    from src.utils.config import HAN_TXT, VI_PDFS
    assert HAN_TXT.exists()
    for pdf in VI_PDFS:
        assert pdf.exists(), f"Missing {pdf}"
```

Run: `uv run pytest tests/`

## Performance scaling

### Multi-GPU OCR

```python
# Trong paddle_ocr.py
import torch
def main():
    n_gpus = torch.cuda.device_count()
    pages_per_gpu = len(pages) // n_gpus
    for gpu_id in range(n_gpus):
        start = gpu_id * pages_per_gpu
        end = start + pages_per_gpu
        # Run OCR với CUDA_VISIBLE_DEVICES=gpu_id
```

### Batch LLM correction

```python
# Trong llm_correct.py - batch 5 chunks per call
def correct_batch(client, model, chunks):
    resp = client.chat(
        model=model,
        messages=[{"role": "user", "content": prompt_for_batch(chunks)}],
    )
    # Parse batch response
```

## Add new language pair

Pipeline có thể adapt cho song ngữ khác (vd Hán-Anh, Việt-Anh):

1. Replace Han TXT + Viet PDF với source mới
2. Đổi NER models (`ner_han.py` dùng HanLP không fit cho English)
3. Update `LLM_JUDGE_RUBRIC` nếu cần
4. FLORES-200 có sẵn cho ~200 language pairs — change `flores_sanity.py` để match
