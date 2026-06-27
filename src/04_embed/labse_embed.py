"""Stage 4: LaBSE embed both sides on GPU.

Output: data/interim/{han,vi}_embeds.npy (N x 768 float32)
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from src.utils.config import (  # noqa: E402
    DEVICE,
    EMBED_BATCH,
    HAN_EMBEDS,
    HAN_SENT,
    LABSE_MODEL,
    VI_EMBEDS,
    VI_SENT,
)


def load_sentences(path: Path) -> list[str]:
    out = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            obj = json.loads(line)
            out.append(obj["text"])
    return out


def embed_texts(texts: list[str], model_name: str = LABSE_MODEL) -> np.ndarray:
    from sentence_transformers import SentenceTransformer
    model = SentenceTransformer(model_name, device=DEVICE)
    model.max_seq_length = 256
    embs = model.encode(
        texts,
        batch_size=EMBED_BATCH,
        convert_to_numpy=True,
        normalize_embeddings=True,
        show_progress_bar=True,
    )
    return embs.astype(np.float32)


def main() -> None:
    for label, sent_path, out_path in [
        ("han", HAN_SENT, HAN_EMBEDS),
        ("vi", VI_SENT, VI_EMBEDS),
    ]:
        if not sent_path.exists():
            raise SystemExit(f"Run split_{label} first. Missing: {sent_path}")
        sents = load_sentences(sent_path)
        print(f"[{label}] {len(sents):,} sentences -> embed ({DEVICE})")
        embs = embed_texts(sents)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        np.save(out_path, embs)
        print(f"  shape={embs.shape} dtype={embs.dtype}")
        print(f"  -> {out_path}")


if __name__ == "__main__":
    main()
