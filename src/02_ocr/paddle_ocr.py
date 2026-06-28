"""Stage 2a: PaddleOCR Vietnamese on all pages.

Output: data/interim/vi_ocr_raw/{tap}.txt (one line per page)
        data/interim/vi_ocr_raw/{tap}_page_{n:04d}.txt (per-page)

Multi-GPU: shard pages round-robin across workers, one worker per GPU.
    CUDA_VISIBLE_DEVICES=0 python -m src.02_ocr.paddle_ocr --shard 0 --num-shards 2 &
    CUDA_VISIBLE_DEVICES=1 python -m src.02_ocr.paddle_ocr --shard 1 --num-shards 2 &
    wait
    python -m src.02_ocr.paddle_ocr --combine   # merge per-page -> per-tap

Single-GPU (default) runs the full set then combines in one shot.
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from src.utils.config import (  # noqa: E402
    CUDA_VISIBLE,
    PADDLE_LANG,
    VI_OCR_RAW_DIR,
    VI_PAGES_DIR,
)

# Detection input cap: book scans are ~7.4 MP (300 DPI). PaddleOCR's det model
# memory scales with image area, so full-res pages spike toward the 12 GB VRAM
# ceiling and OOM. Downscaling the longest side to this many pixels cuts peak
# memory ~9x with negligible quality loss on already-large 300 DPI text.
DET_SIDE_LEN = int(os.environ.get("HVB_DET_SIDE_LEN", "1536"))


def build_ocr():
    from paddleocr import PaddleOCR

    return PaddleOCR(
        lang=PADDLE_LANG,
        ocr_version="PP-OCRv5",
        use_textline_orientation=True,
        text_det_limit_type="max",
        text_det_limit_side_len=DET_SIDE_LEN,
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


def page_out_path(img_path: Path) -> Path:
    key = img_path.stem.rsplit("_", 1)[0]  # tap4_0001 -> tap4
    page_idx = int(img_path.stem.rsplit("_", 1)[1])
    return VI_OCR_RAW_DIR / f"{key}_page_{page_idx:04d}.txt"


def run_ocr(shard: int, num_shards: int) -> None:
    """OCR this worker's slice of pages (round-robin), skipping done pages."""
    VI_OCR_RAW_DIR.mkdir(parents=True, exist_ok=True)
    pages = sorted(VI_PAGES_DIR.glob("*.png"))
    if not pages:
        raise SystemExit(f"No PNG pages in {VI_PAGES_DIR}. Run pdf_to_images first.")

    mine = pages[shard::num_shards]
    todo = [p for p in mine if not page_out_path(p).exists()]
    tag = f"shard {shard}/{num_shards}"
    print(
        f"[{tag}] {len(mine)} pages assigned, {len(todo)} to do "
        f"({len(mine) - len(todo)} cached), GPU={CUDA_VISIBLE > 0}"
    )
    if not todo:
        return

    ocr = build_ocr()
    from tqdm import tqdm

    for p in tqdm(todo, desc=f"OCR[{tag}]"):
        text = ocr_image(ocr, p)
        page_out_path(p).write_text(text + "\n", encoding="utf-8")


def combine() -> None:
    """Merge per-page txt files into per-tap combined txt files."""
    VI_OCR_RAW_DIR.mkdir(parents=True, exist_ok=True)
    by_tap: dict[str, list[tuple[int, str]]] = {}
    for f in sorted(VI_OCR_RAW_DIR.glob("*_page_*.txt")):
        # tap4_page_0001 -> key=tap4, idx=1
        key, _, idx = f.stem.rsplit("_", 2)
        text = f.read_text(encoding="utf-8").rstrip("\n")
        by_tap.setdefault(key, []).append((int(idx), text))

    if not by_tap:
        raise SystemExit(f"No per-page OCR files in {VI_OCR_RAW_DIR}. Run OCR first.")

    for key, items in by_tap.items():
        items.sort()
        combined = "\n\n".join(t for _, t in items)
        (VI_OCR_RAW_DIR / f"{key}.txt").write_text(combined, encoding="utf-8")
        print(f"  {key}: {len(items)} pages -> {VI_OCR_RAW_DIR / f'{key}.txt'}")


def main() -> None:
    ap = argparse.ArgumentParser(description="PaddleOCR Vietnamese pages (multi-GPU shardable)")
    ap.add_argument("--shard", type=int, default=0, help="this worker's index (0-based)")
    ap.add_argument("--num-shards", type=int, default=1, help="total number of workers")
    ap.add_argument("--combine", action="store_true", help="merge per-page txt into per-tap txt")
    args = ap.parse_args()

    if args.combine:
        combine()
        return

    if not (0 <= args.shard < args.num_shards):
        raise SystemExit(f"--shard must be in [0, {args.num_shards}), got {args.shard}")

    run_ocr(args.shard, args.num_shards)

    # Single-worker run combines immediately; multi-worker defers to --combine.
    if args.num_shards == 1:
        combine()


if __name__ == "__main__":
    main()
