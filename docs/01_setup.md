# Setup Environment

Hướng dẫn cài đặt chi tiết cho HVB pipeline.

## Yêu cầu hệ thống

### Hardware
- **GPU NVIDIA**: tối thiểu 8GB VRAM (test trên 2x RTX 3060 12GB)
- **RAM**: 16GB+
- **Disk**: 10GB cho code+deps, 5GB cho images trung gian, 2GB cho models

### Software

| Tool | Version | Mục đích |
|------|---------|---------|
| Python | 3.11 (3.10-3.12 OK) | Runtime |
| uv | 0.10+ | Package manager |
| Docker | 24+ | Chạy Ollama |
| NVIDIA Driver | 535+ | CUDA 12.1 |
| poppler-utils | any | pdf2image |
| git | any | Clone vecalign |
| qpdf hoặc pdftk | any | Subset PDF cho smoke test |

## Cài đặt từng bước

### 1. System packages (Ubuntu/Debian)

```bash
sudo apt update
sudo apt install -y poppler-utils qpdf git build-essential
```

### 2. uv (Astral package manager)

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
uv --version  # phải >= 0.10
```

### 3. NVIDIA + CUDA

```bash
nvidia-smi
nvidia-smi | grep "CUDA Version"
```

Nếu CUDA < 12.1, update driver hoặc sửa `pyproject.toml` sang `cu118` index.

### 4. Docker + Ollama

```bash
curl -fsSL https://get.docker.com | sh
sudo usermod -aG docker $USER
# Logout + login lại để group có hiệu lực
docker run hello-world
```

### 5. Project setup

```bash
cd /home/tienda/WorkSpace/HCMUS/NLP

# Chạy script setup (tự động):
# - uv venv + uv sync
# - git clone vecalign
# - docker run ollama
# - ollama pull qwen2.5:7b seallm:7b
./scripts/setup_uv.sh
```

## Verify cài đặt

```bash
# 1. Python venv
source .venv/bin/activate
python -c "import torch; print(torch.cuda.is_available(), torch.cuda.device_count())"
# Expected: True 2

# 2. PaddleOCR
python -c "from paddleocr import PaddleOCR; print('OK')"

# 3. Ollama
curl http://localhost:11434/api/tags | jq '.models[].name'
# Expected: ["qwen2.5:7b", "seallm:7b"]

# 4. Vecalign
ls external/vecalign/vecalign.py

# 5. pdf2image backend
which pdfinfo pdftoppm

# 6. Config check
uv run python -m src.utils.config
# Expected: ROOT path, DEVICE=cuda, tất cả paths exists
```

## Cấu hình optional

### Đổi models LLM

Sửa `src/utils/config.py`:
```python
LLM_MODELS = ["qwen2.5:14b", "seallm:7b"]  # 14B cần >= 16GB VRAM
```

Pull thêm model:
```bash
docker exec ollama ollama pull qwen2.5:14b
```

### Đổi batch size (VRAM tight)

Sửa `src/utils/config.py`:
```python
EMBED_BATCH = 32   # mặc định 64, giảm nếu OOM
MT_BATCH = 8       # mặc định 16
PADDLE_BATCH = 4   # mặc định 8
```

### Giới hạn GPU cho Ollama

Setup mặc định dùng tất cả GPU. Giới hạn: edit `scripts/setup_uv.sh` flag `--gpus` theo nhu cầu.

## Troubleshooting setup

### uv sync fail: paddle wheel không tìm thấy

```
error: couldn't find paddlepaddle-gpu in https://www.paddlepaddle.org.cn/packages/stable/cu118/
```

**Fix**: Check CUDA version của driver. Sửa index URL trong `pyproject.toml` sang version match (cu118 / cu120 / cu121).

### Ollama: connection refused

```bash
docker ps | grep ollama
docker start ollama
sleep 5
curl http://localhost:11434/api/tags
```

### Docker: could not select device driver

```bash
sudo apt install -y nvidia-container-toolkit
sudo systemctl restart docker
```

### PyTorch không thấy GPU

```bash
python -c "import torch; print(torch.version.cuda)"
# Nếu None hoặc cũ, force reinstall qua uv
uv pip install --reinstall torch --index-url https://download.pytorch.org/whl/cu121
```

### Hết disk space

Cleanup an toàn (đọc doc trước khi chạy):
- `docker system prune -a` — gỡ Docker images không dùng
- `uv cache clean` — xóa uv cache
- HF cache: xóa thủ công thư mục cache HuggingFace (sẽ re-download khi cần)

## Next steps

- Quay lại [`README.md`](../README.md)
- Đọc [`02_data.md`](02_data.md) — spec input data
- Đọc [`03_pipeline.md`](03_pipeline.md) — pipeline chi tiết
