"""Convert PaddleOCR-VL `parsing_res_list` JSON → per-page text files.

Input:  silver/<pdf_stem>/<pdf_stem>_<page_index>_res.json
Output: data/interim/vi_ocr_raw/tap{4,5,6}_page_{page:04d}.txt
        data/interim/vi_ocr_raw/tap{4,5,6}.txt  (combined per-tập)

Drops page furniture (header/footer/page-number/image/table blocks). Keeps
narrative text blocks + titles + footnotes. Blocks ordered by
`block_order` (None → end).
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

TAP_RE = re.compile(r"tập\s*([456])", re.IGNORECASE)
PAGE_RE = re.compile(r"_(\d+)_res\.json$")

KEEP_LABELS = {
    "text",
    "doc_title",
    "paragraph_title",
    "abstract",
    "aside_text",
    "footnote",
    "reference_content",
    "content",
    "vertical_text",
}


def pdf_stem_to_tap(stem: str) -> str | None:
    m = TAP_RE.search(stem)
    return f"tap{m.group(1)}" if m else None


def sort_key(block: dict) -> tuple[int, int]:
    order = block.get("block_order")
    if order is None:
        return (1, block.get("block_id", 0))
    return (0, order)


def render_page(data: dict) -> str:
    blocks = [b for b in data.get("parsing_res_list", []) if b.get("block_label") in KEEP_LABELS]
    blocks.sort(key=sort_key)
    chunks = []
    for b in blocks:
        content = (b.get("block_content") or "").strip()
        if content:
            chunks.append(content)
    return "\n\n".join(chunks)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--silver-dir", type=Path, required=True,
                    help="Extracted silver/ dir containing <pdf_stem>/ subdirs")
    ap.add_argument("--out-dir", type=Path, required=True,
                    help="Output vi_ocr_raw dir")
    args = ap.parse_args()

    silver: Path = args.silver_dir
    out: Path = args.out_dir
    out.mkdir(parents=True, exist_ok=True)

    per_tap_pages: dict[str, list[tuple[int, str]]] = {}
    total_json = 0
    kept_pages = 0
    for pdf_dir in sorted(silver.iterdir()):
        if not pdf_dir.is_dir():
            continue
        tap = pdf_stem_to_tap(pdf_dir.name)
        if tap is None:
            print(f"  skip (no tap match): {pdf_dir.name}", file=sys.stderr)
            continue
        for jf in sorted(pdf_dir.glob("*_res.json")):
            total_json += 1
            m = PAGE_RE.search(jf.name)
            if not m:
                continue
            page = int(m.group(1))
            with jf.open("r", encoding="utf-8") as fh:
                data = json.load(fh)
            text = render_page(data)
            page_file = out / f"{tap}_page_{page:04d}.txt"
            page_file.write_text(text, encoding="utf-8")
            per_tap_pages.setdefault(tap, []).append((page, text))
            kept_pages += 1

    print(f"json files: {total_json}, per-page txt written: {kept_pages}")

    for tap, pages in per_tap_pages.items():
        pages.sort()
        combined = "\n".join(text for _, text in pages if text)
        (out / f"{tap}.txt").write_text(combined, encoding="utf-8")
        print(f"  {tap}: {len(pages)} pages, combined chars={len(combined):,}")


if __name__ == "__main__":
    main()
