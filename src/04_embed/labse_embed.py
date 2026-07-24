"""Stage 4: sentence-embed both sides on GPU.

Default backbone: BAAI/bge-m3 (1024-dim). Override via HVB_EMBED_MODEL.

BGE-M3 is loaded via FlagEmbedding (official package) because the model
shipped only pytorch_model.bin locally, and SentenceTransformer 5.x fails
to load .bin under torch<2.6 (CVE-2025-32434) + has incompatible Pooling
module config. FlagEmbedding loads the .bin directly without ST.

Output: data/interim/{han,vi}_embeds.npy (N x D float32, D=1024 for BGE-M3)
"""
from __future__ import annotations

import json
import os
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
    out: list[str] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            obj = json.loads(line)
            out.append(obj["text"])
    return out


def embed_texts(texts: list[str], model_name: str = EMBED_MODEL) -> np.ndarray:
    if "bge-m3" in model_name.lower():
        return _embed_bgem3(texts, model_name)
    return _embed_sentence_transformer(texts, model_name)


def _embed_bgem3(texts: list[str], model_name: str) -> np.ndarray:
    """BGE-M3 via FlagEmbedding. Vectors are pre-normalized (norm ≈ 1.0)."""
    from FlagEmbedding import BGEM3FlagModel

    snap = _resolve_local_snapshot(model_name)
    device = "cuda" if DEVICE.startswith("cuda") else "cpu"
    idx = 0
    if DEVICE.startswith("cuda") and ":" in DEVICE:
        idx = int(DEVICE.split(":")[1])
    model = BGEM3FlagModel(
        snap or model_name,
        use_fp16=device == "cuda",
        device=device if device == "cpu" else idx,
    )
    n = len(texts)
    out: list[np.ndarray] = []
    for i in range(0, n, EMBED_BATCH):
        chunk = texts[i : i + EMBED_BATCH]
        r = model.encode(
            chunk,
            batch_size=len(chunk),
            max_length=EMBED_MAX_SEQ,
            return_dense=True,
            return_sparse=False,
            return_colbert_vecs=False,
        )
        out.append(np.asarray(r["dense_vecs"], dtype=np.float32))
    embs = np.concatenate(out, axis=0)
    # Guard: re-normalize in case fp16 rounding left norms slightly off 1.
    norms = np.linalg.norm(embs, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return embs / norms


def _resolve_local_snapshot(model_name: str) -> str | None:
    cache = Path.home() / ".cache/huggingface/hub"
    repo_dir = cache / model_name.replace("/", "--")
    if not (repo_dir / "snapshots").exists():
        return None
    snaps = list((repo_dir / "snapshots").iterdir())
    return str(snaps[0]) if snaps else None


def _embed_sentence_transformer(
    texts: list[str], model_name: str
) -> np.ndarray:
    """LaBSE / other ST models."""
    import os
    from sentence_transformers import SentenceTransformer

    trust = bool(os.environ.get("HVB_TRUST_REMOTE_CODE", ""))
    st_kwargs: dict = {"device": DEVICE}
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


def _embed_qwen3(texts: list[str], model_name: str) -> np.ndarray:
    """Qwen3-Embedding (0.6B/4B/8B) via sentence-transformers.

    Requires transformers>=4.51, sentence-transformers>=2.7.
    Instruction-aware; for bitext mining we encode symmetrically (no query
    prompt) — both Hán and Việt treated as documents. Asymmetric
    (prompt_name='query' on one side) is a tunable for later.
    """
    from sentence_transformers import SentenceTransformer

    model = SentenceTransformer(
        model_name,
        device=DEVICE,
        model_kwargs={"attn_implementation": "eager"},
    )
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
    side = os.environ.get("HVB_EMBED_SIDE", "both").lower()
    jobs = [
        ("han", HAN_SENT, HAN_EMBEDS),
        ("vi", VI_SENT, VI_EMBEDS),
    ]
    if side in ("han", "vi"):
        jobs = [(l, s, o) for l, s, o in jobs if l == side]
    elif side != "both":
        raise SystemExit(f"HVB_EMBED_SIDE must be han|vi|both, got: {side}")
    for label, sent_path, out_path in jobs:
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
