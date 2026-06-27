"""Stage 1a: Normalize Hán TXT.

Strip Wiki文库 header markers, normalize punctuation, dedupe blank lines.
"""
from __future__ import annotations

import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from src.utils.config import (  # noqa: E402
    HAN_CLEAN,
    HAN_TXT,
    SUBSET_HAN_CHARS,
    SUBSET_HAN_OFFSET,
    SUBSET_N,
)

WIKI_HEADER_RE = re.compile(
    r"^(姊妹计划\s*[:：].*$|^数据项$|^#.*$|^##.*$|^\s*---+\s*$)",
    re.MULTILINE,
)
BRACKET_RE = re.compile(r"【[^】]*】")  # wiki block markers
MULTI_NEWLINE_RE = re.compile(r"\n{3,}")

# Normalize full-width Latin to half-width (keep CJK punctuation)
FULLWIDTH_RE = re.compile(r"[！-～]")


def _fullwidth_to_half(match: re.Match) -> str:
    ch = match.group(0)
    return chr(ord(ch) - 0xFEE0)


def normalize(text: str) -> str:
    text = BRACKET_RE.sub("", text)
    text = WIKI_HEADER_RE.sub("", text)
    text = FULLWIDTH_RE.sub(_fullwidth_to_half, text)
    text = MULTI_NEWLINE_RE.sub("\n\n", text)
    lines = [ln.strip() for ln in text.splitlines()]
    text = "\n".join(ln for ln in lines if ln)
    return text.strip() + "\n"


def main() -> None:
    if not HAN_TXT.exists():
        raise FileNotFoundError(f"Hán TXT not found: {HAN_TXT}")
    raw = HAN_TXT.read_text(encoding="utf-8")
    clean = normalize(raw)
    if SUBSET_N > 0 and SUBSET_HAN_CHARS > 0:
        clean = clean[SUBSET_HAN_OFFSET : SUBSET_HAN_OFFSET + SUBSET_HAN_CHARS]
        print(
            f"SUBSET: chars [{SUBSET_HAN_OFFSET}:{SUBSET_HAN_OFFSET + SUBSET_HAN_CHARS}] "
            f"(SUBSET_N={SUBSET_N}, OFFSET={SUBSET_HAN_OFFSET})"
        )
    HAN_CLEAN.parent.mkdir(parents=True, exist_ok=True)
    HAN_CLEAN.write_text(clean, encoding="utf-8")
    print(f"raw chars: {len(raw):,}")
    print(f"clean chars: {len(clean):,}")
    print(f"ratio: {len(clean) / len(raw):.2%}")
    print(f"output: {HAN_CLEAN}")


if __name__ == "__main__":
    main()
