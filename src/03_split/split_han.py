"""Stage 3a: Han sentence split (Classical Chinese).

HanLP sentence_split + custom rules for terminal punctuation and annotation brackets.
Output: data/interim/han_sentences.jsonl with {"idx", "text"}.
"""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from src.utils.config import HAN_CLEAN, HAN_SENT  # noqa: E402

ZH_TERM_RE = re.compile(r"([。！？；])")
ANNOT_OPEN = "〈「『（"
ANNOT_CLOSE = "〉」』）"


def split_classical(text: str) -> list[str]:
    """Split on terminal punctuation but keep annotations intact."""
    protected = []
    out_chars = []
    i = 0
    while i < len(text):
        c = text[i]
        if c in ANNOT_OPEN:
            close = ANNOT_CLOSE[ANNOT_OPEN.index(c)]
            depth = 1
            j = i + 1
            while j < len(text) and depth > 0:
                if text[j] == c:
                    depth += 1
                elif text[j] == close:
                    depth -= 1
                j += 1
            placeholder = f"__ANN{len(protected)}__"
            protected.append(text[i:j])
            out_chars.append(placeholder)
            i = j
        else:
            out_chars.append(c)
            i += 1
    prot_text = "".join(out_chars)

    parts = ZH_TERM_RE.split(prot_text)
    merged = []
    i = 0
    while i < len(parts):
        if i + 1 < len(parts) and ZH_TERM_RE.fullmatch(parts[i + 1]):
            merged.append(parts[i] + parts[i + 1])
            i += 2
        else:
            if parts[i].strip():
                merged.append(parts[i])
            i += 1

    sentences = []
    for s in merged:
        s = s.strip()
        if not s:
            continue
        for k, ann in enumerate(protected):
            s = s.replace(f"__ANN{k}__", ann)
        sentences.append(s)
    return sentences


def main() -> None:
    if not HAN_CLEAN.exists():
        raise SystemExit(f"Run normalize_han first. Missing: {HAN_CLEAN}")
    text = HAN_CLEAN.read_text(encoding="utf-8")
    paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]
    sentences = []
    for para in paragraphs:
        sentences.extend(split_classical(para))

    HAN_SENT.parent.mkdir(parents=True, exist_ok=True)
    with HAN_SENT.open("w", encoding="utf-8") as f:
        for i, s in enumerate(sentences):
            f.write(json.dumps({"idx": i, "text": s}, ensure_ascii=False) + "\n")

    lens = [len(s) for s in sentences]
    print(f"paragraphs: {len(paragraphs):,}")
    print(f"sentences: {len(sentences):,}")
    print(f"avg len: {sum(lens) / len(lens):.1f} chars")
    print(f"output: {HAN_SENT}")


if __name__ == "__main__":
    main()
