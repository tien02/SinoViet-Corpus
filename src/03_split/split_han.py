"""Stage 3a: Han sentence split (Classical Chinese).

Regex split on terminal punctuation + custom rules for annotation brackets.
Output: data/interim/han_sentences.jsonl with {"idx", "text"}.
"""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from src.utils.config import HAN_CLEAN, HAN_PUNCT, HAN_SENT  # noqa: E402

ZH_TERM_RE = re.compile(r"([。！？；!?;])")
TERM_CHARS = "。！？；!?;"
ANNOT_OPEN = "〈「『（"
ANNOT_CLOSE = "〉」』）"

# Fallback for imperial-edict / decree blocks with ZERO terminal punctuation
# (378 such paragraphs in the corpus, median 8 568 chars). Splits the block
# on `\n` and greedily re-merges consecutive lines into sentence-sized
# chunks — line-by-line over-fragments (Vecalign's ratio to Vi side blows
# up), whole-block atomically forces oversized drops at export.
MAX_LEN = 2000
CHUNK_TARGET = 200
# Post-punctuator merge: guwen-biaodian over-predicts 。 after common
# characters (議, 賞, 嗣), yielding many <10-char fragments that lack
# enough semantic signal for BGE-M3. Merge fragments below MIN_HAN_LEN
# into the following sentence (or previous, at paragraph end).
MIN_HAN_LEN = int(__import__("os").environ.get("HVB_MIN_HAN_LEN", "15"))


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
        # Fall back to newline split when block is oversized AND either:
        # (a) has zero terminators (zero-terminator edict blocks), or
        # (b) has very few terminators relative to size (paragraph-merged
        # blocks with 1 stray 。+ many newlines), or
        # (c) is exceptionally large (>10K chars).
        # Prior bug: condition required zero terminators; 1 stray 。in a
        # 13K-char paragraph disabled the fallback.
        if len(s) > MAX_LEN and "\n" in s:
            term_count = sum(1 for c in s if c in TERM_CHARS)
            if term_count < 5 or len(s) > 10000:
                sentences.extend(_greedy_merge_lines(s))
            else:
                sentences.append(s)
        else:
            sentences.append(s)
    return sentences


def _merge_short(sents: list[str], min_len: int) -> list[str]:
    """Merge sentences below min_len forward into the next sentence.

    Trailing short fragment (paragraph tail) merges backward. Preserves
    monotonic order — critical for bertalign DP.
    """
    if not sents:
        return sents
    out: list[str] = []
    buf = ""
    for s in sents:
        if buf:
            s = buf + s
            buf = ""
        if len(s) < min_len:
            buf = s
        else:
            out.append(s)
    if buf:
        if out:
            out[-1] = out[-1] + buf
        else:
            out.append(buf)
    return out


def _greedy_merge_lines(text: str) -> list[str]:
    """Group consecutive non-empty lines into ~CHUNK_TARGET-char chunks.

    Emits when running buffer exceeds target OR at end. Preserves original
    line order — critical for monotonic alignment.
    """
    chunks: list[str] = []
    buf: list[str] = []
    buf_len = 0
    for line in text.split("\n"):
        line = line.strip()
        if not line:
            continue
        buf.append(line)
        buf_len += len(line)
        if buf_len >= CHUNK_TARGET:
            chunks.append("".join(buf))
            buf = []
            buf_len = 0
    if buf:
        chunks.append("".join(buf))
    return chunks


def main() -> None:
    # Prefer punctuation-restored file when present. Raw HAN_CLEAN lacks
    # terminal punct → CHUNK_TARGET fallback slices arbitrarily. Punctuated
    # file lets ZH_TERM_RE split at real record boundaries.
    if HAN_PUNCT.exists():
        src_path = HAN_PUNCT
    elif HAN_CLEAN.exists():
        src_path = HAN_CLEAN
        print(f"[warn] {HAN_PUNCT} missing — falling back to {HAN_CLEAN}. "
              f"Run han_punctuate first for better splits.")
    else:
        raise SystemExit(f"Run normalize_han first. Missing: {HAN_CLEAN}")
    print(f"reading: {src_path}")
    text = src_path.read_text(encoding="utf-8")
    paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]
    sentences = []
    for para in paragraphs:
        sentences.extend(_merge_short(split_classical(para), MIN_HAN_LEN))

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
