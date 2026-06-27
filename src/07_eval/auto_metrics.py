"""Stage 7a: Auto metrics on aligned pairs.

LaBSE cosine mean, COMET-QE-22, BERTScore, chrF/BLEU (bi-directional pseudo).
Output: data/final/eval/auto_metrics.json
"""
from __future__ import annotations

import json
import statistics
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from src.utils.config import (  # noqa: E402
    COMET_MODEL,
    DEVICE,
    FINAL,
    LABSE_MODEL,
    PAIRS_JSONL,
)

OUT = FINAL / "eval" / "auto_metrics.json"


def load_pairs() -> list[dict]:
    out = []
    with PAIRS_JSONL.open(encoding="utf-8") as f:
        for line in f:
            out.append(json.loads(line))
    return out


def labse_cosine(pairs: list[dict]) -> list[float]:
    from sentence_transformers import SentenceTransformer
    import numpy as np
    model = SentenceTransformer(LABSE_MODEL, device=DEVICE)
    src = [p["src"] for p in pairs]
    tgt = [p["tgt"] for p in pairs]
    e1 = model.encode(src, batch_size=64, normalize_embeddings=True, show_progress_bar=False)
    e2 = model.encode(tgt, batch_size=64, normalize_embeddings=True, show_progress_bar=False)
    return [float(np.dot(a, b)) for a, b in zip(e1, e2)]


def comet_qe(pairs: list[dict]) -> list[float]:
    from comet import download_model, load_from_checkpoint
    model_path = download_model(COMET_MODEL)
    model = load_from_checkpoint(model_path)
    data = [{"src": p["src"], "mt": p["tgt"]} for p in pairs]
    predictions = model.predict(data, batch_size=16, gpus=1 if DEVICE == "cuda" else 0)
    return [float(s) for s in predictions["scores"]]


def bertscore(pairs: list[dict]) -> dict:
    from bert_score import score
    src = [p["src"] for p in pairs]
    tgt = [p["tgt"] for p in pairs]
    P, R, F1 = score(src, tgt, lang="zh", verbose=False, device=DEVICE)
    return {
        "precision_mean": float(P.mean()),
        "recall_mean": float(R.mean()),
        "f1_mean": float(F1.mean()),
    }


def bleu_chrf(pairs: list[dict]) -> dict:
    import sacrebleu
    src = [p["src"] for p in pairs]
    tgt = [p["tgt"] for p in pairs]
    bleu_zh2vi = sacrebleu.corpus_bleu(tgt, [src]).score
    bleu_vi2zh = sacrebleu.corpus_bleu(src, [tgt]).score
    chrf_zh2vi = sacrebleu.corpus_chrf(tgt, [src]).score
    chrf_vi2zh = sacrebleu.corpus_chrf(src, [tgt]).score
    return {
        "bleu_zh2vi": bleu_zh2vi,
        "bleu_vi2zh": bleu_vi2zh,
        "chrf_zh2vi": chrf_zh2vi,
        "chrf_vi2zh": chrf_vi2zh,
    }


def main() -> None:
    if not PAIRS_JSONL.exists():
        raise SystemExit(f"Run vecalign_runner first. Missing: {PAIRS_JSONL}")
    pairs = load_pairs()
    print(f"pairs: {len(pairs):,}")

    print("LaBSE cosine...")
    cos = labse_cosine(pairs)
    print("COMET-QE-22...")
    comet_scores = comet_qe(pairs)
    print("BERTScore...")
    bs = bertscore(pairs)
    print("BLEU/chrF...")
    bc = bleu_chrf(pairs)

    result = {
        "n_pairs": len(pairs),
        "labse_cosine": {
            "mean": statistics.mean(cos),
            "median": statistics.median(cos),
            "stdev": statistics.pstdev(cos),
        },
        "comet_qe": {
            "mean": statistics.mean(comet_scores),
            "median": statistics.median(comet_scores),
        },
        "bertscore": bs,
        "bleu_chrf": bc,
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(result, indent=2, ensure_ascii=False))
    print(f"-> {OUT}")


if __name__ == "__main__":
    main()
