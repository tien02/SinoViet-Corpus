"""Stage 2a: PaddleOCR Vietnamese on all pages.

Output: data/interim/vi_ocr_raw/{tap}.txt (one line per page)
        data/interim/vi_ocr_raw/{tap}_page_{n:04d}.txt (per-page)
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from src.utils.config import (  # noqa: E402
    CUDA_VISIBLE,
    PADDLE_BATCH,
    PADDLE_LANG,
    VI_OCR_RAW_DIR,
    VI_PAGES_DIR,
)


def build_ocr():
    from paddleocr import PaddleOCR

    return PaddleOCR(
        lang=PADDLE_LANG,
        ocr_version="PP-OCRv5",
        use_textline_orientation=True,
    )


def ocr_image(ocr, img_path: Path) -> str:
    """PaddleOCR 3.x: .predict() returns list[OCRResult] with rec_texts field."""
    result = ocr.predict(str(img_path))
    lines: list[str] = []
    if result:
        for page in result:
            if not page:
                continue
            texts = None
            if isinstance(page, dict):
                texts = page.get("rec_texts")
            elif hasattr(page, "rec_texts"):
                texts = page.rec_texts
            if texts:
                lines.extend(texts)
    return "\n".join(lines)


def main() -> None:
    VI_OCR_RAW_DIR.mkdir(parents=True, exist_ok=True)
    pages = sorted(VI_PAGES_DIR.glob("*.png"))
    if not pages:
        raise SystemExit(f"No PNG pages in {VI_PAGES_DIR}. Run pdf_to_images first.")
    print(f"OCR {len(pages)} pages, GPU={CUDA_VISIBLE > 0}")

    ocr = build_ocr()
    by_tap: dict[str, list[str]] = {}

    from tqdm import tqdm
    for p in tqdm(pages, desc="OCR"):
        key = p.stem.rsplit("_", 1)[0]  # tap4_0001 -> tap4
        page_idx = int(p.stem.rsplit("_", 1)[1])
        text = ocr_image(ocr, p)
        (VI_OCR_RAW_DIR / f"{key}_page_{page_idx:04d}.txt").write_text(
            text + "\n", encoding="utf-8"
        )
        by_tap.setdefault(key, []).append((page_idx, text))

    for key, items in by_tap.items():
        items.sort()
        combined = "\n\n".join(t for _, t in items)
        (VI_OCR_RAW_DIR / f"{key}.txt").write_text(combined, encoding="utf-8")
        print(f"  {key}: {len(items)} pages -> {VI_OCR_RAW_DIR / f'{key}.txt'}")


if __name__ == "__main__":
    main()
