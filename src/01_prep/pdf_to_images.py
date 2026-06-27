"""Stage 1b: Convert Việt PDFs to PNG pages at 300 DPI.

Output: data/interim/vi_pages/{tap}_{page:04d}.png
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from src.utils.config import (  # noqa: E402
    OCR_DPI,
    SUBSET_N,
    SUBSET_OFFSET,
    VI_PAGES_DIR,
    VI_PDFS,
)


def tap_key(pdf_path: Path) -> str:
    """Extract tap4/tap5/tap6 from filename."""
    name = pdf_path.stem
    for k in ("tập 4", "tập 5", "tập 6", "tap 4", "tap 5", "tap 6"):
        if k in name.lower():
            return f"tap{k.split()[-1]}"
    return pdf_path.stem.replace(" ", "_")[:32]


def convert_pdf(
    pdf_path: Path,
    out_dir: Path,
    dpi: int = OCR_DPI,
    first_pages: int | None = None,
    page_offset: int = 0,
) -> int:
    from pdf2image import convert_from_path

    key = tap_key(pdf_path)
    start = page_offset + 1
    end = page_offset + first_pages if first_pages else None
    pages = convert_from_path(
        str(pdf_path), dpi=dpi, thread_count=4, first_page=start, last_page=end
    )
    n = 0
    for i, img in enumerate(pages, start=start):
        out = out_dir / f"{key}_{i:04d}.png"
        img.save(out, format="PNG")
        n += 1
        if i % 50 == 0:
            print(f"  {key} page {i}")
    return n


def main() -> None:
    VI_PAGES_DIR.mkdir(parents=True, exist_ok=True)
    total = 0
    for pdf in VI_PDFS:
        if not pdf.exists():
            print(f"WARN: missing {pdf}")
            continue
        print(f"[{tap_key(pdf)}] {pdf.name}")
        n = convert_pdf(
            pdf,
            VI_PAGES_DIR,
            OCR_DPI,
            first_pages=SUBSET_N or None,
            page_offset=SUBSET_OFFSET,
        )
        print(f"  -> {n} pages")
        total += n
    print(f"Total: {total} pages -> {VI_PAGES_DIR}")


if __name__ == "__main__":
    main()
