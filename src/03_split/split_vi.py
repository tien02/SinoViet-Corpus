"""Stage 3b: Viet sentence split (van co).

Underthesea sent_tokenize + custom rules.
Output: data/interim/vi_sentences.jsonl with {"idx", "tap", "page", "text"}.
"""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from src.utils.config import (  # noqa: E402
    VI_OCR_CORRECTED_DIR,
    VI_OCR_RAW_DIR,
    VI_SENT,
)


def underthesea_split(text: str) -> list[str]:
    try:
        from underthesea import sent_tokenize
        return sent_tokenize(text)
    except Exception:
        return re.split(r"(?<=[\.\!\?\:\n])\s+", text)


def split_vi_paragraph(para: str) -> list[str]:
    para = para.strip()
    if not para:
        return []
    chunks = re.split(r"(?<=[。；！？])\s+", para)
    sentences = []
    for ch in chunks:
        ch = ch.strip()
        if not ch:
            continue
        for s in underthesea_split(ch):
            s = s.strip()
            if len(s) >= 2:
                sentences.append(s)
    return sentences


def main() -> None:
    # Prefer LLM-corrected pages; fall back to raw OCR when correction is
    # skipped/not run (LLM post-correction is optional in this pipeline).
    src_dir = VI_OCR_CORRECTED_DIR
    per_page_files = (
        sorted(src_dir.glob("tap*_page_*.txt")) if src_dir.exists() else []
    )
    if not per_page_files:
        src_dir = VI_OCR_RAW_DIR
        per_page_files = (
            sorted(src_dir.glob("tap*_page_*.txt")) if src_dir.exists() else []
        )
    if not per_page_files:
        raise SystemExit(
            f"No per-page OCR files in {VI_OCR_CORRECTED_DIR} or {VI_OCR_RAW_DIR}. "
            "Run paddle_ocr (and optionally llm_correct) first."
        )
    print(f"split_vi: reading {len(per_page_files)} pages from {src_dir.name}")

    VI_SENT.parent.mkdir(parents=True, exist_ok=True)
    idx = 0
    with VI_SENT.open("w", encoding="utf-8") as fout:
        for pf in per_page_files:
            stem = pf.stem
            try:
                tap, _, page_num = stem.split("_")
                page = int(page_num)
            except ValueError:
                tap, page = stem, -1
            text = pf.read_text(encoding="utf-8")
            paras = re.split(r"\n{2,}", text)
            for para in paras:
                for s in split_vi_paragraph(para):
                    fout.write(json.dumps(
                        {"idx": idx, "tap": tap, "page": page, "text": s},
                        ensure_ascii=False,
                    ) + "\n")
                    idx += 1

    if idx == 0:
        raise SystemExit(
            f"split_vi produced 0 sentences from {src_dir} — OCR output is empty?"
        )
    print(f"sentences: {idx:,}")
    print(f"output: {VI_SENT}")


if __name__ == "__main__":
    main()
