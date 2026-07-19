"""Stage 3c: Viet sentence split from clean .docx (bypass OCR).

Reads dai-nam-thuc-luc-tap0{4,5,6}.docx and emits vi_sentences.jsonl
with the same schema as split_vi.py: {"idx", "tap", "page", "text"}.
page=-1 because .docx has no page markers.

Activate by running this script directly. Backup data/interim/vi_sentences.jsonl
first — overwrites in place.
"""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from src.utils.config import RAW, VI_SENT  # noqa: E402

MIN_SENT_LEN = 8

DOCX_FILES = {
    "tap4": RAW / "dai-nam-thuc-luc-tap04.docx",
    "tap5": RAW / "dai-nam-thuc-luc-tap05.docx",
    "tap6": RAW / "dai-nam-thuc-luc-tap06.docx",
}


def underthesea_split(text: str) -> list[str]:
    try:
        from underthesea import sent_tokenize
        return sent_tokenize(text)
    except Exception:
        return re.split(r"(?<=[\.\!\?])\s+", text)


def split_vi_paragraph(para: str) -> list[str]:
    para = para.strip()
    if not para:
        return []
    chunks = re.split(r"(?<=[。！？])\s+", para)
    sentences: list[str] = []
    for ch in chunks:
        ch = ch.strip()
        if not ch:
            continue
        for s in underthesea_split(ch):
            s = s.strip()
            if len(s) >= MIN_SENT_LEN:
                sentences.append(s)
    return sentences


def main() -> None:
    from docx import Document

    VI_SENT.parent.mkdir(parents=True, exist_ok=True)
    idx = 0
    per_tap_counts: dict[str, int] = {}
    with VI_SENT.open("w", encoding="utf-8") as fout:
        for tap, path in DOCX_FILES.items():
            if not path.exists():
                raise SystemExit(f"Missing: {path}")
            doc = Document(str(path))
            count = 0
            for para in doc.paragraphs:
                text = para.text.strip()
                if not text:
                    continue
                for s in split_vi_paragraph(text):
                    fout.write(json.dumps(
                        {"idx": idx, "tap": tap, "page": -1, "text": s},
                        ensure_ascii=False,
                    ) + "\n")
                    idx += 1
                    count += 1
            per_tap_counts[tap] = count
            print(f"{tap}: {count:,} sentences from {path.name}")

    if idx == 0:
        raise SystemExit("docx_to_vi_sentences produced 0 sentences — check .docx contents")

    print(f"total: {idx:,}")
    print(f"output: {VI_SENT}")


if __name__ == "__main__":
    main()
