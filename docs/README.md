# Documentation — HVB Pipeline

Hướng dẫn chi tiết cho pipeline Hán-Việt Đại Nam Thực Lục.

## Index

| Doc | Nội dung |
|-----|----------|
| [00_problem.md](00_problem.md) | **Đọc đầu tiên** — bài toán, pipeline lý do, chiến lược đánh giá không-gold |
| [01_setup.md](01_setup.md) | Cài đặt môi trường: uv, vLLM docker, system deps, verification |
| [02_data.md](02_data.md) | Spec input data, định dạng output, schema JSONL |
| [03_pipeline.md](03_pipeline.md) | Chi tiết từng stage: code flow, params, outputs |
| [04_eval.md](04_eval.md) | Sino-Viet phonetic proxy, no manual labels available, sino bands + rescue |
| [05_troubleshooting.md](05_troubleshooting.md) | Lỗi thường gặp + cách fix |
| [06_extend.md](06_extend.md) | Mở rộng: thêm PDF, đổi LLM, custom eval rubric |
| [07_unlimited_ocr.md](07_unlimited_ocr.md) | Stage 2 — Unlimited-OCR (vLLM, 2-GPU): tại sao chọn, setup, run seamless với align |
| [08_results.md](08_results.md) | Số liệu thực tế: Hán items, PaddleOCR vs Unlimited-OCR vs PaddleOCR-VL-1.6 (pages, chars, sentences, aligned pairs, deliverable) |
| [09_han_pipeline.md](09_han_pipeline.md) | Chi tiết Hán normalize + split: bug full-width→ASCII, paragraph preservation, zero-terminator edict fallback, measured impact (34K → 52K aligned pairs) |
| [10_fail_cases.md](10_fail_cases.md) | Fail cases PaddleOCR-VL-1.6 + full-Hán alignment: 2 208 Hán uncovered, decoder collapse, repetition loop, back-matter asymmetry, cost ranking, fast wins |
| [11_current_issues.md](11_current_issues.md) | Current pipeline issues (2026-07-07): embedder fails on Han-Viet, sino proxy saturation, vecalign score scale, coverage 21%, priority order |
| [12_final_results.md](12_final_results.md) | Bertalign + BGE-M3 baseline results (2026-07-07) — pre-punctuator |
| [13_han_punctuator.md](13_han_punctuator.md) | **guwen-biaodian punctuator (2026-07-08)** — 30,959 delivered pairs (5.3x baseline), cosine max 0.77→0.88, 1-1 beads 0.9%→47%. Supersedes ch.9 200-char chunker + ch.12 baseline |
| [14_pipeline_diagram.md](14_pipeline_diagram.md) | ASCII pipeline diagram + full model registry (OCR, LLM, punctuator, embedder, aligner, sino) |

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
