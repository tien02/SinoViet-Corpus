"""Stage 7b: FLORES-200 zh-vi sanity check.

Sanity-check pipeline: run LaBSE embed on FLORES devtest zh-vi,
expect alignment precision@1 > 0.95 + COMET > 0.4.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from src.utils.config import DEVICE, FLORES_DIR, LABSE_MODEL, FINAL  # noqa: E402

OUT = FINAL / "eval" / "flores_sanity.json"


def load_flores() -> tuple[list[str], list[str]]:
    zh_file = FLORES_DIR / "zho_Hans.devtest"
    vi_file = FLORES_DIR / "vie_Latn.devtest"
    if not zh_file.exists():
        try:
            from datasets import load_dataset
            ds = load_dataset("facebook/flores", "zho_Hans_vie_Latn", split="devtest")
            zh = [x["sentence_zho_Hans"] for x in ds]
            vi = [x["sentence_vie_Latn"] for x in ds]
            return zh, vi
        except Exception as e:
            raise SystemExit(
                f"FLORES not cached and HF load failed: {e}. "
                "Download from https://github.com/facebookresearch/flores"
            )
    return (
        zh_file.read_text(encoding="utf-8").splitlines(),
        vi_file.read_text(encoding="utf-8").splitlines(),
    )


def main() -> None:
    zh, vi = load_flores()
    print(f"FLORES devtest zh={len(zh)} vi={len(vi)}")
    from sentence_transformers import SentenceTransformer
    import numpy as np
    model = SentenceTransformer(LABSE_MODEL, device=DEVICE)
    e1 = model.encode(zh, batch_size=64, normalize_embeddings=True, show_progress_bar=False)
    e2 = model.encode(vi, batch_size=64, normalize_embeddings=True, show_progress_bar=False)
    sim = e1 @ e2.T
    n = min(len(zh), len(vi))
    diagonal = sim[np.arange(n), np.arange(n)]
    argmax = sim.argmax(axis=1)
    precision_at_1 = float((argmax[:n] == np.arange(n)).mean())

    comet_score = None
    try:
        from comet import download_model, load_from_checkpoint
        mp = download_model("unmt/comet-qe-22")
        m = load_from_checkpoint(mp)
        data = [{"src": z, "mt": v} for z, v in zip(zh[:500], vi[:500])]
        preds = m.predict(data, batch_size=16, gpus=1 if DEVICE == "cuda" else 0)
        comet_score = sum(preds["scores"]) / len(preds["scores"])
    except Exception as e:
        print(f"COMET skipped: {e}")

    result = {
        "n_pairs": n,
        "diagonal_cosine_mean": float(diagonal.mean()),
        "diagonal_cosine_median": float(np.median(diagonal)),
        "precision_at_1": precision_at_1,
        "comet_qe_mean_first500": comet_score,
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(result, indent=2, ensure_ascii=False))
    print(f"-> {OUT}")


if __name__ == "__main__":
    main()
