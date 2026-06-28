# Documentation — HVB Pipeline

Hướng dẫn chi tiết cho pipeline Hán-Việt Đại Nam Thực Lục.

## Index

| Doc | Nội dung |
|-----|----------|
| [00_problem.md](00_problem.md) | **Đọc đầu tiên** — bài toán, pipeline lý do, chiến lược đánh giá không-gold |
| [01_setup.md](01_setup.md) | Cài đặt môi trường: uv, vLLM docker, system deps, verification |
| [02_data.md](02_data.md) | Spec input data, định dạng output, schema JSONL |
| [03_pipeline.md](03_pipeline.md) | Chi tiết từng stage: code flow, params, outputs |
| [04_eval.md](04_eval.md) | Methodology đánh giá 5 trụ, metric definitions, targets |
| [05_troubleshooting.md](05_troubleshooting.md) | Lỗi thường gặp + cách fix |
| [06_extend.md](06_extend.md) | Mở rộng: thêm PDF, đổi LLM, custom eval rubric |

## Đọc theo thứ tự

**Lần đầu chạy project:**
1. `00_problem.md` → hiểu bài toán + chiến lược đánh giá
2. `01_setup.md` → cài môi trường
3. `02_data.md` → xác nhận input data đúng format
4. `03_pipeline.md` → hiểu pipeline
5. Chạy: `./scripts/run_pipeline.sh prep` rồi inspect output trước khi next stage

**Khi gặp lỗi:**
- `05_troubleshooting.md` trước
- Check `data/interim/.checkpoint/` xem stage nào đã xong

**Khi muốn mở rộng:**
- `06_extend.md` cho patterns thêm data, đổi model

## Quick links

- Root README: [`../README.md`](../README.md)
- Config paths: [`../src/utils/config.py`](../src/utils/config.py)
- Deps: [`../pyproject.toml`](../pyproject.toml)
- Runner: [`../scripts/run_pipeline.sh`](../scripts/run_pipeline.sh)
