"""Stage 4: sentence-embed both sides on GPU.

Default backbone: BAAI/bge-m3 (1024-dim). Override via HVB_EMBED_MODEL.

Output: data/interim/{han,vi}_embeds.npy (N x D float32, D=1024 for BGE-M3)
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
    EMBED_MAX_SEQ,
    EMBED_MODEL,
    HAN_EMBEDS,
    HAN_SENT,
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


def embed_texts(texts: list[str], model_name: str = EMBED_MODEL) -> np.ndarray:
    import os
    from sentence_transformers import SentenceTransformer
    # use_safetensors=True bypasses torch<2.6 CVE-2025-32434 hard-block on
    # .bin checkpoints. LaBSE + GTE + E5 all ship safetensors variants.
    # trust_remote_code required by GTE-family + Qwen-embeddings; opt-in
    # via HVB_TRUST_REMOTE_CODE=1 — executes arbitrary HF-hub Python.
    trust = bool(os.environ.get("HVB_TRUST_REMOTE_CODE", ""))
    kwargs = {"use_safetensors": True}
    st_kwargs = {"device": DEVICE, "model_kwargs": kwargs}
    if trust:
        st_kwargs["trust_remote_code"] = True
    model = SentenceTransformer(model_name, **st_kwargs)
    model.max_seq_length = EMBED_MAX_SEQ
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
