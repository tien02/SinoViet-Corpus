"""Stage 3a-pre: Restore punctuation on Classical Chinese Han text.

Wikisource Đại Nam Thực Lục lacks terminal punctuation inside chapters
(176/181 paragraphs have zero 。！？；). Without boundaries, split_han
falls back to arbitrary 200-char slicing that crosses topics and
destroys BGE-M3 retrieval signal.

Fix: run raynardj/classical-chinese-punctuation-guwen-biaodian
(BertForTokenClassification, 21 labels — 0=O, 1=。, 2=，, ..., trained on
四庫全書 punctuation restoration) as a per-token classifier over sliding
windows, insert the predicted punct after each char, then let split_han
split on the restored terminators.

Input:  HAN_CLEAN   (data/interim/han_clean.txt)
Output: HAN_PUNCT   (data/interim/han_punctuated.txt)

Idempotent: skips if HAN_PUNCT is newer than HAN_CLEAN.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from src.utils.config import (  # noqa: E402
    DEVICE,
    HAN_CLEAN,
    HAN_PUNCT,
    HAN_PUNCT_BATCH,
    HAN_PUNCT_MODEL,
    HAN_PUNCT_OVERLAP,
    HAN_PUNCT_WINDOW,
)

TERM_CHARS = "。！？；"
# Only keep informative punct in output. Skip decorative quote/bracket
# labels — noisy on Wikisource text without paired glyphs upstream.
KEEP_LABELS = {"。", "，", "：", "；", "？", "！", "、"}


def _paragraph_windows(text: str, window: int, overlap: int) -> list[tuple[int, str]]:
    """Return (offset, substring) windows over a stripped-newline paragraph.

    Overlap gives the model context on both ends; downstream merge picks
    labels from the window whose center is closest to each char position.
    """
    text = text.replace("\n", "")
    if len(text) <= window:
        return [(0, text)]
    step = max(1, window - overlap)
    out: list[tuple[int, str]] = []
    i = 0
    while i < len(text):
        out.append((i, text[i : i + window]))
        if i + window >= len(text):
            break
        i += step
    return out


def _load_model():
    from transformers import AutoModelForTokenClassification, AutoTokenizer
    import torch

    tokenizer = AutoTokenizer.from_pretrained(HAN_PUNCT_MODEL)
    model = AutoModelForTokenClassification.from_pretrained(HAN_PUNCT_MODEL)
    device = DEVICE if DEVICE.startswith("cuda") else "cpu"
    if device.startswith("cuda"):
        model = model.to(device).half()
    else:
        model = model.to(device)
    model.eval()
    id2label = model.config.id2label
    return tokenizer, model, device, torch, id2label


def _predict_batch_labels(
    tokenizer, model, device, torch, id2label, texts: list[str]
) -> list[list[str]]:
    """Batched per-char labels for a list of window texts."""
    enc = tokenizer(
        texts,
        return_tensors="pt",
        return_offsets_mapping=True,
        truncation=True,
        padding=True,
        max_length=HAN_PUNCT_WINDOW + 8,
    )
    offsets_batch = enc.pop("offset_mapping").tolist()
    enc = {k: v.to(device) for k, v in enc.items()}
    with torch.no_grad():
        logits = model(**enc).logits
    pred_batch = logits.argmax(-1).tolist()

    results: list[list[str]] = []
    for text, offsets, pred_ids in zip(texts, offsets_batch, pred_batch):
        labels = ["O"] * len(text)
        for (start, end), pid in zip(offsets, pred_ids):
            if start == end:
                continue
            pos = end - 1
            if 0 <= pos < len(text):
                labels[pos] = id2label[pid]
        results.append(labels)
    return results


def _merge_window_labels(
    text_len: int,
    windows: list[tuple[int, str]],
    per_window_labels: list[list[str]],
) -> list[str]:
    """Vote per char: prefer label from the window whose center is
    closest to that char position (interior beats edge)."""
    merged: list[str] = ["O"] * text_len
    best_dist: list[float] = [float("inf")] * text_len
    for (offset, wtext), labels in zip(windows, per_window_labels):
        center = offset + len(wtext) / 2
        for local_i, lab in enumerate(labels):
            global_i = offset + local_i
            if global_i >= text_len:
                break
            d = abs(global_i - center)
            if d < best_dist[global_i]:
                merged[global_i] = lab
                best_dist[global_i] = d
    return merged


def punctuate_paragraph(tokenizer, model, device, torch, id2label, para: str) -> str:
    windows = _paragraph_windows(para, HAN_PUNCT_WINDOW, HAN_PUNCT_OVERLAP)
    flat_text = para.replace("\n", "")
    per_window_labels: list[list[str]] = []
    for i in range(0, len(windows), HAN_PUNCT_BATCH):
        batch_texts = [wtext for _, wtext in windows[i : i + HAN_PUNCT_BATCH]]
        per_window_labels.extend(
            _predict_batch_labels(
                tokenizer, model, device, torch, id2label, batch_texts
            )
        )
    merged = _merge_window_labels(len(flat_text), windows, per_window_labels)

    out_chars: list[str] = []
    for ch, lab in zip(flat_text, merged):
        out_chars.append(ch)
        if lab in KEEP_LABELS:
            out_chars.append(lab)
    return "".join(out_chars)


def main() -> None:
    if not HAN_CLEAN.exists():
        raise SystemExit(f"Missing {HAN_CLEAN}. Run normalize_han first.")
    if HAN_PUNCT.exists() and HAN_PUNCT.stat().st_mtime > HAN_CLEAN.stat().st_mtime:
        print(f"[skip] {HAN_PUNCT} newer than {HAN_CLEAN}")
        return

    raw = HAN_CLEAN.read_text(encoding="utf-8")
    paragraphs = [p for p in raw.split("\n\n") if p.strip()]
    print(f"paragraphs: {len(paragraphs):,}")
    print(f"model: {HAN_PUNCT_MODEL}")
    print(f"window={HAN_PUNCT_WINDOW} overlap={HAN_PUNCT_OVERLAP} batch={HAN_PUNCT_BATCH}")

    tokenizer, model, device, torch, id2label = _load_model()
    print(f"device: {device}")

    from time import time

    out_paras: list[str] = []
    t0 = time()
    for i, para in enumerate(paragraphs, 1):
        out_paras.append(
            punctuate_paragraph(tokenizer, model, device, torch, id2label, para)
        )
        if i % 5 == 0 or i == len(paragraphs):
            elapsed = time() - t0
            rate = i / elapsed if elapsed else 0
            eta = (len(paragraphs) - i) / rate if rate else 0
            print(f"  [{i}/{len(paragraphs)}] {rate:.2f} para/s eta={eta:.0f}s")

    HAN_PUNCT.parent.mkdir(parents=True, exist_ok=True)
    HAN_PUNCT.write_text("\n\n".join(out_paras) + "\n", encoding="utf-8")

    joined = "\n\n".join(out_paras)
    inserted = sum(1 for c in joined if c in TERM_CHARS)
    print(f"in chars:  {len(raw):,}")
    print(f"out chars: {len(joined):,}")
    print(f"terminators inserted: {inserted:,}")
    print(f"output: {HAN_PUNCT}")


if __name__ == "__main__":
    main()
