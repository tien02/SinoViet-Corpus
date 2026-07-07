"""Build stratified 100-pair eval sample for manual labeling.

Picks 100 pairs stratified by combined-score bands so manual review covers
both high-confidence and borderline pairs. Output is a CSV the annotator
fills in with a 0/1 "is this pair a correct translation" label.

Strata (by `combined` score, fallback to (1 - score) if missing):
  high (>=0.80):    30 pairs — should be almost all correct
  mid  (0.55-0.80): 40 pairs — majority correct, some noise
  low  (<0.55):     30 pairs — should be mostly noise / hard cases

Output: data/eval/eval_sample_100.csv
  columns: pair_id, stratum, han, viet, sino, combined, label, notes
  (label and notes empty — annotator fills)

Usage:
  uv run python scripts/build_eval_sample.py
  uv run python scripts/build_eval_sample.py --n 200
  uv run python scripts/build_eval_sample.py --seed 42
"""
from __future__ import annotations

import argparse
import csv
import json
import random
from pathlib import Path


def _combined_score(p: dict) -> float:
    if "combined" in p:
        return float(p["combined"])
    return max(0.0, 1.0 - float(p.get("score", 1.0)))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--pairs",
        type=Path,
        default=Path("data/aligned/pairs_reranked.jsonl"),
        help="input reranked pairs (default pairs_reranked.jsonl)",
    )
    ap.add_argument(
        "--out",
        type=Path,
        default=Path("data/eval/eval_sample_100.csv"),
    )
    ap.add_argument("--n", type=int, default=100)
    ap.add_argument("--seed", type=int, default=2026)
    ap.add_argument(
        "--high-min", type=float, default=0.80,
        help="lower bound of 'high' stratum",
    )
    ap.add_argument(
        "--mid-min", type=float, default=0.55,
        help="lower bound of 'mid' stratum (upper bound of 'low')",
    )
    args = ap.parse_args()

    if not args.pairs.exists():
        raise SystemExit(
            f"missing {args.pairs}. Run align + rerank first."
        )

    pairs = []
    with args.pairs.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            obj["_combined"] = _combined_score(obj)
            pairs.append(obj)

    high = [p for p in pairs if p["_combined"] >= args.high_min]
    mid = [
        p for p in pairs
        if args.mid_min <= p["_combined"] < args.high_min
    ]
    low = [p for p in pairs if p["_combined"] < args.mid_min]

    random.seed(args.seed)
    n_high = min(30, len(high))
    n_mid = min(40, len(mid))
    n_low = min(30, len(low))
    remaining = args.n - (n_high + n_mid + n_low)
    for bucket, attr in ((high, "n_high"), (mid, "n_mid"), (low, "n_low")):
        if remaining <= 0:
            break
        cur = {"n_high": n_high, "n_mid": n_mid, "n_low": n_low}[attr]
        extra = min(remaining, max(0, len(bucket) - cur))
        if attr == "n_high":
            n_high += extra
        elif attr == "n_mid":
            n_mid += extra
        else:
            n_low += extra
        remaining -= extra

    picks_high = random.sample(high, n_high) if n_high else []
    picks_mid = random.sample(mid, n_mid) if n_mid else []
    picks_low = random.sample(low, n_low) if n_low else []

    print(
        f"pool: high={len(high)} mid={len(mid)} low={len(low)} "
        f"-> sampled high={n_high} mid={n_mid} low={n_low}"
    )

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(
            ["pair_id", "stratum", "han", "viet", "sino", "combined", "label", "notes"]
        )
        pid = 1
        for stratum, picks in (
            ("high", picks_high),
            ("mid", picks_mid),
            ("low", picks_low),
        ):
            for p in picks:
                w.writerow([
                    pid,
                    stratum,
                    p.get("src", ""),
                    p.get("tgt", ""),
                    f"{p.get('sino', 0.0):.3f}",
                    f"{p['_combined']:.3f}",
                    "",
                    "",
                ])
                pid += 1
    print(f"wrote {args.out} ({pid - 1} pairs)")
    print(
        "\nLabel each row: 1 = correct translation, 0 = wrong. "
        "Then run scripts/score_eval.py to compute precision per stratum."
    )


if __name__ == "__main__":
    main()
