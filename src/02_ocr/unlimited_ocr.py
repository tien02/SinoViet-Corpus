"""Stage 2a (alt): Baidu Unlimited-OCR via vLLM — replacement for PaddleOCR.

Calls vLLM OpenAI-compatible endpoint serving `baidu/Unlimited-OCR` (DeepSeek-OCR
lineage, SAM-ViT-B + CLIP-L vision stack). Higher quality than PaddleOCR on
Vietnamese historical scans; outputs clean text after stripping grounding tokens.

Output: data/interim/vi_ocr_raw/{tap}_page_{n:04d}.txt (per-page)
        data/interim/vi_ocr_raw/{tap}.txt           (combined)

Single vLLM endpoint serves one GPU; throughput scaled via concurrent requests
inside one worker (UNLIMITED_OCR_BATCH). Multi-container sharding is TODO.

Recipe: https://recipes.vllm.ai/baidu/Unlimited-OCR
"""
from __future__ import annotations

import argparse
import base64
import re
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from src.utils.config import (  # noqa: E402
    UNLIMITED_OCR_API_KEY,
    UNLIMITED_OCR_BASE_URL,
    UNLIMITED_OCR_BASE_URLS,
    UNLIMITED_OCR_BATCH,
    UNLIMITED_OCR_MAX_TOKENS,
    UNLIMITED_OCR_MODEL,
    UNLIMITED_OCR_NGRAM_SIZE,
    UNLIMITED_OCR_TIMEOUT,
    UNLIMITED_OCR_WINDOW_SIZE,
    VI_OCR_RAW_DIR,
    VI_PAGES_DIR,
)

# Endpoints we round-robin across. Falls back to single URL if no list configured.
ENDPOINTS = UNLIMITED_OCR_BASE_URLS or [UNLIMITED_OCR_BASE_URL]

# Strip <|det|>...</|det|> coordinate boxes; unwrap <|ref|>...</|ref|> text.
DET_RE = re.compile(r"<\|det\|>.*?<\|/det\|>", re.DOTALL)
REF_OPEN_RE = re.compile(r"<\|ref\|>")
REF_CLOSE_RE = re.compile(r"<\|/ref\|>")
MULTIBLANK_RE = re.compile(r"\n{3,}")


def make_client(base_url: str):
    """Build an OpenAI client bound to a specific vLLM endpoint."""
    from openai import OpenAI

    return OpenAI(
        base_url=base_url,
        api_key=UNLIMITED_OCR_API_KEY,
        timeout=UNLIMITED_OCR_TIMEOUT,
    )


def encode_image_b64(img_path: Path) -> str:
    """Read image, return base64 data URL."""
    suffix = img_path.suffix.lower().lstrip(".")
    mime = {"png": "image/png", "jpg": "image/jpeg", "jpeg": "image/jpeg"}.get(
        suffix, "image/png"
    )
    b64 = base64.b64encode(img_path.read_bytes()).decode("ascii")
    return f"data:{mime};base64,{b64}"


def clean_output(raw: str) -> str:
    """Strip grounding tokens, normalize whitespace."""
    if not raw:
        return ""
    out = DET_RE.sub("", raw)
    out = REF_OPEN_RE.sub("", out)
    out = REF_CLOSE_RE.sub("", out)
    out = "\n".join(line.rstrip() for line in out.splitlines())
    out = MULTIBLANK_RE.sub("\n\n", out)
    return out.strip()


def ocr_image(client, img_path: Path) -> str:
    """Single-image OCR via Unlimited-OCR chat completions."""
    data_url = encode_image_b64(img_path)
    resp = client.chat.completions.create(
        model=UNLIMITED_OCR_MODEL,
        messages=[
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "<image>document parsing."},
                    {"type": "image_url", "image_url": {"url": data_url}},
                ],
            }
        ],
        max_tokens=UNLIMITED_OCR_MAX_TOKENS,
        temperature=0.0,
        extra_body={
            "skip_special_tokens": False,
            "vllm_xargs": {
                "ngram_size": UNLIMITED_OCR_NGRAM_SIZE,
                "window_size": UNLIMITED_OCR_WINDOW_SIZE,
            },
        },
    )
    return clean_output(resp.choices[0].message.content or "")


# Per-endpoint retry wrapper: try each endpoint in sequence until success.
def ocr_image_round_robin(clients, img_path: Path, start_idx: int) -> tuple[str, str]:
    """Try endpoint at start_idx first; on failure, walk the list.

    Returns (text, endpoint_url_used).
    """
    last_err: Exception | None = None
    n = len(clients)
    for off in range(n):
        idx = (start_idx + off) % n
        url, client = clients[idx]
        try:
            return ocr_image(client, img_path), url
        except Exception as e:  # noqa: BLE001
            last_err = e
    raise RuntimeError(f"All endpoints failed for {img_path.name}: {last_err}")


def page_out_path(img_path: Path) -> Path:
    key = img_path.stem.rsplit("_", 1)[0]  # tap4_0001 -> tap4
    page_idx = int(img_path.stem.rsplit("_", 1)[1])
    return VI_OCR_RAW_DIR / f"{key}_page_{page_idx:04d}.txt"


def run_ocr(limit: int | None = None) -> None:
    """OCR all (or first N) pages concurrently across all endpoints."""
    VI_OCR_RAW_DIR.mkdir(parents=True, exist_ok=True)
    pages = sorted(VI_PAGES_DIR.glob("*.png"))
    if not pages:
        raise SystemExit(f"No PNG pages in {VI_PAGES_DIR}. Run pdf_to_images first.")

    todo = [p for p in pages if not page_out_path(p).exists()]
    if limit:
        todo = todo[:limit]
    print(
        f"{len(pages)} pages total, {len(todo)} to do "
        f"({len(pages) - len(todo)} cached)"
    )
    print(
        f"  endpoints: {len(ENDPOINTS)} | per-endpoint batch: {UNLIMITED_OCR_BATCH} "
        f"| effective parallelism: {len(ENDPOINTS) * UNLIMITED_OCR_BATCH}"
    )
    if not todo:
        return

    # Build one OpenAI client per endpoint. Clients are stateless beyond a connection
    # pool; round-robin via the page's index so load is balanced.
    clients = [(url, make_client(url)) for url in ENDPOINTS]
    from tqdm import tqdm

    failures: list[tuple[Path, str]] = []
    total_workers = len(ENDPOINTS) * UNLIMITED_OCR_BATCH
    with ThreadPoolExecutor(max_workers=total_workers) as pool, tqdm(
        total=len(todo), desc="Unlimited-OCR"
    ) as bar:
        future_to_path = {
            pool.submit(ocr_image_round_robin, clients, p, i % len(clients)): p
            for i, p in enumerate(todo)
        }
        for fut in as_completed(future_to_path):
            p = future_to_path[fut]
            try:
                text, _used_url = fut.result()
                page_out_path(p).write_text(text + "\n", encoding="utf-8")
            except Exception as e:
                failures.append((p, str(e)))
            bar.update(1)
            bar.set_postfix(fail=len(failures))

    for p, err in failures:
        print(f"  FAIL {p.name}: {err}")


def combine() -> None:
    """Merge per-page txt files into per-tap combined txt files."""
    VI_OCR_RAW_DIR.mkdir(parents=True, exist_ok=True)
    by_tap: dict[str, list[tuple[int, str]]] = {}
    for f in sorted(VI_OCR_RAW_DIR.glob("*_page_*.txt")):
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
    ap = argparse.ArgumentParser(
        description="Baidu Unlimited-OCR via vLLM (replacement for PaddleOCR)"
    )
    ap.add_argument("--limit", type=int, default=None, help="OCR only first N pages")
    ap.add_argument("--combine", action="store_true", help="merge per-page -> per-tap")
    ap.add_argument(
        "--no-combine", action="store_true", help="skip auto-combine after run"
    )
    args = ap.parse_args()

    if args.combine:
        combine()
        return

    run_ocr(limit=args.limit)
    if not args.no_combine:
        combine()


if __name__ == "__main__":
    main()
