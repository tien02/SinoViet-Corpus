"""Stage 7c: Round-trip consistency.

Viet -> Han via LLM, compare to original Han. chrF++/BLEU/BERTScore.
Sample 500 pairs stratified by LaBSE score.
"""
from __future__ import annotations

import json
import random
import statistics
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from src.utils.config import (  # noqa: E402
    DEVICE,
    EVAL_SAMPLE,
    FINAL,
    LLM_MODELS,
    LLM_TIMEOUT,
    OLLAMA_HOST,
    PAIRS_JSONL,
)

OUT = FINAL / "eval" / "round_trip.json"
PROMPT = """Dich cau Viet sau sang Han van (Classical Chinese).
Giu phong cach Dai Nam Thuc Luc. Chi tra ve ban dich, khong giai thich.

Viet: {vi}
Han:"""


def stratified_sample(pairs: list[dict], n: int) -> list[dict]:
    pairs_sorted = sorted(pairs, key=lambda p: p["score"])
    third = len(pairs_sorted) // 3
    buckets = [
        pairs_sorted[:third],
        pairs_sorted[third:2 * third],
        pairs_sorted[2 * third:],
    ]
    out = []
    for b in buckets:
        if not b:
            continue
        out.extend(random.sample(b, min(n // 3, len(b))))
    return out[:n]


def translate_vi_to_han(client, model: str, vi: str) -> str:
    resp = client.chat(
        model=model,
        messages=[{"role": "user", "content": PROMPT.format(vi=vi)}],
        options={"temperature": 0.2, "num_ctx": 2048},
        timeout=LLM_TIMEOUT,
    )
    return resp["message"]["content"].strip()


def main(model: str = LLM_MODELS[0]) -> None:
    if not PAIRS_JSONL.exists():
        raise SystemExit(f"Run vecalign_runner first. Missing: {PAIRS_JSONL}")
    pairs = [json.loads(l) for l in PAIRS_JSONL.open(encoding="utf-8")]
    sample = stratified_sample(pairs, EVAL_SAMPLE)
    print(f"sample: {len(sample)} pairs ({model})")

    import ollama
    client = ollama.Client(host=OLLAMA_HOST)

    from tqdm import tqdm
    results = []
    for p in tqdm(sample, desc="round-trip"):
        try:
            rt = translate_vi_to_han(client, model, p["tgt"])
        except Exception as e:
            rt = ""
            err = str(e)
        else:
            err = None
        results.append({
            "src_orig": p["src"],
            "vi": p["tgt"],
            "han_roundtrip": rt,
            "labse_score": p["score"],
            **({"error": err} if err else {}),
        })

    import sacrebleu
    refs = [r["src_orig"] for r in results]
    hyps = [r["han_roundtrip"] for r in results]
    chrf = sacrebleu.corpus_chrf(hyps, [refs]).score
    bleu = sacrebleu.corpus_bleu(hyps, [refs]).score
    bs = {}
    try:
        from bert_score import score as bs_score
        P, R, F1 = bs_score(hyps, refs, lang="zh", verbose=False, device=DEVICE)
        bs = {"f1": float(F1.mean())}
    except Exception as e:
        print(f"BERTScore skipped: {e}")

    summary = {
        "model": model,
        "n_sample": len(results),
        "chrf": chrf,
        "bleu": bleu,
        "bertscore": bs,
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(
        json.dumps({"summary": summary, "pairs": results}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    print(f"-> {OUT}")


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=LLM_MODELS[0])
    args = ap.parse_args()
    main(args.model)
