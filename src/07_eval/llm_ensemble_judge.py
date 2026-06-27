"""Stage 7e: LLM ensemble judge.

Each LLM (qwen2.5:7b, seallm:7b) scores 500 pairs on 5 criteria (1-5).
Compute Krippendorff alpha for cross-LLM agreement.
"""
from __future__ import annotations

import json
import random
import statistics
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from src.utils.config import (  # noqa: E402
    EVAL_SAMPLE,
    FINAL,
    LLM_JUDGE_RUBRIC,
    LLM_MODELS,
    LLM_TIMEOUT,
    OLLAMA_HOST,
    PAIRS_JSONL,
)

OUT = FINAL / "eval" / "llm_ensemble_judge.json"
PROMPT = """Danh gia cap Han-Viet sau theo 5 tieu chi (1-5, 5 = xuat sac).
Chi tra ve JSON, khong giai thich.

Han: {src}
Viet: {tgt}

Tieu chi:
- adequacy: du y nghia
- fluency: thong suat ngu phap
- alignment: cap dung
- fidelity: trung thuc nguyen ban
- terminology: dung thuat ngu lich su

JSON:
{{"adequacy": N, "fluency": N, "alignment": N, "fidelity": N, "terminology": N}}
"""


def load_pairs() -> list[dict]:
    return [json.loads(l) for l in PAIRS_JSONL.open(encoding="utf-8")]


def stratified_sample(pairs: list[dict], n: int) -> list[dict]:
    pairs_sorted = sorted(pairs, key=lambda p: p["score"])
    third = len(pairs_sorted) // 3
    buckets = [pairs_sorted[:third], pairs_sorted[third:2*third], pairs_sorted[2*third:]]
    out = []
    for b in buckets:
        if b:
            out.extend(random.sample(b, min(n // 3, len(b))))
    return out[:n]


def parse_judge_response(text: str) -> dict:
    import re
    m = re.search(r"\{[^}]+\}", text, re.DOTALL)
    if not m:
        return {}
    try:
        obj = json.loads(m.group(0).replace("'", '"'))
        return {k: int(v) for k, v in obj.items() if isinstance(v, (int, float, str)) and str(v).strip().isdigit()}
    except Exception:
        return {}


def judge_one(client, model: str, src: str, tgt: str) -> dict:
    resp = client.chat(
        model=model,
        messages=[{"role": "user", "content": PROMPT.format(src=src, tgt=tgt)}],
        options={"temperature": 0.1, "num_ctx": 2048, "format": "json"},
        timeout=LLM_TIMEOUT,
    )
    return parse_judge_response(resp["message"]["content"])


def main() -> None:
    if not PAIRS_JSONL.exists():
        raise SystemExit(f"Run vecalign_runner first. Missing: {PAIRS_JSONL}")
    pairs = load_pairs()
    random.seed(42)
    sample = stratified_sample(pairs, EVAL_SAMPLE)
    print(f"sample: {len(sample)} pairs x {len(LLM_MODELS)} LLMs")

    import ollama
    from tqdm import tqdm
    client = ollama.Client(host=OLLAMA_HOST)

    all_scores: dict[str, list[dict]] = {m: [] for m in LLM_MODELS}
    for model in LLM_MODELS:
        for p in tqdm(sample, desc=model):
            try:
                scores = judge_one(client, model, p["src"], p["tgt"])
            except Exception as e:
                scores = {"error": str(e)}
            all_scores[model].append(scores)

    # Krippendorff alpha per criterion
    import krippendorff
    alpha_per = {}
    for crit in LLM_JUDGE_RUBRIC:
        rows = []
        for model in LLM_MODELS:
            row = [all_scores[model][i].get(crit, None) for i in range(len(sample))]
            rows.append(row)
        try:
            alpha = krippendorff.alpha(reliability_data=rows, level_of_measurement="ordinal")
        except Exception:
            alpha = None
        alpha_per[crit] = alpha

    # Mean scores per model
    means_per = {}
    for model in LLM_MODELS:
        means = {}
        for crit in LLM_JUDGE_RUBRIC:
            vals = [s.get(crit) for s in all_scores[model] if isinstance(s.get(crit), (int, float))]
            if vals:
                means[crit] = statistics.mean(vals)
        means_per[model] = means

    result = {
        "n_sample": len(sample),
        "models": LLM_MODELS,
        "krippendorff_alpha": alpha_per,
        "mean_per_model": means_per,
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(
        json.dumps({"summary": result, "scores": all_scores}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(result, indent=2, ensure_ascii=False))
    print(f"-> {OUT}")


if __name__ == "__main__":
    main()
