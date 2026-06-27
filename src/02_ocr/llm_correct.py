"""Stage 2b: LLM post-correction of OCR via Ollama docker.

Chunks raw OCR by ~500 chars, asks Qwen2.5/SeaLLM to fix common OCR errors
(font Nôm, dấu câu cổ, broken Vietnamese diacritics).
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from src.utils.config import (  # noqa: E402
    LLM_MODELS,
    LLM_TIMEOUT,
    OLLAMA_HOST,
    VI_OCR_CORRECTED_DIR,
    VI_OCR_RAW_DIR,
)

PROMPT_TMPL = """Bạn là chuyên gia hiệu đính văn bản lịch sử Việt Nam (Đại Nam Thực Lục).
Sửa lỗi OCR trong đoạn sau. Lỗi phổ biến:
- Ký tự Nôm/Hán bị nhận sai (vd: "ℓ" -> "l", "0" -> "o")
- Dấu câu cổ bị mất (。、；)
- Dấu thanh Việt Nam bị sai (vd: "hoo" -> "họa")
- Tên riêng vua quan, địa danh: giữ nguyên nếu đúng

Chỉ trả về văn bản đã sửa, không giải thích.

Đoạn cần sửa:
---
{chunk}
---
"""


def chunk_text(text: str, max_chars: int = 500) -> list[str]:
    paras = text.split("\n\n")
    out, cur = [], ""
    for p in paras:
        if len(cur) + len(p) + 2 > max_chars and cur:
            out.append(cur)
            cur = p
        else:
            cur = f"{cur}\n\n{p}" if cur else p
    if cur:
        out.append(cur)
    return out


def correct_chunk(client, model: str, chunk: str) -> str:
    resp = client.chat(
        model=model,
        messages=[{"role": "user", "content": PROMPT_TMPL.format(chunk=chunk)}],
        options={"temperature": 0.2, "num_ctx": 4096},
    )
    return resp["message"]["content"].strip()


def correct_file(raw_path: Path, model: str) -> str:
    import ollama

    client = ollama.Client(host=OLLAMA_HOST)
    text = raw_path.read_text(encoding="utf-8")
    chunks = chunk_text(text)
    fixed = []
    from tqdm import tqdm
    for c in tqdm(chunks, desc=f"{raw_path.stem}({model})", leave=False):
        try:
            fixed.append(correct_chunk(client, model, c))
        except Exception as e:
            print(f"  WARN chunk failed ({e}), keep raw")
            fixed.append(c)
    return "\n\n".join(fixed)


def main(model: str = LLM_MODELS[0]) -> None:
    VI_OCR_CORRECTED_DIR.mkdir(parents=True, exist_ok=True)
    raw_files = sorted(VI_OCR_RAW_DIR.glob("tap*.txt"))
    if not raw_files:
        raise SystemExit(f"No raw OCR txt in {VI_OCR_RAW_DIR}. Run paddle_ocr first.")
    print(f"Correct {len(raw_files)} files with {model}")
    for rf in raw_files:
        out = VI_OCR_CORRECTED_DIR / rf.name
        corrected = correct_file(rf, model)
        out.write_text(corrected, encoding="utf-8")
        print(f"  {rf.name} -> {out}")


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=LLM_MODELS[0], choices=LLM_MODELS)
    args = ap.parse_args()
    main(args.model)
