"""Score a labeled eval sample. Computes precision per stratum + overall.

Reads annotated CSV from scripts/build_eval_sample.py (label column filled
with 0/1), reports precision per stratum (high/mid/low), overall, and a
simple bootstrapped 95% CI on overall precision.

Usage:
  uv run python scripts/score_eval.py
  uv run python scripts/score_eval.py --csv custom.csv
  uv run python scripts/score_eval.py --csv annotated.csv --json out.json
"""
from __future__ import annotations

import argparse
import csv
import json
import random
import statistics
import sys
from pathlib import Path


def _load(csv_path: Path) -> list[dict]:
    rows = []
    with csv_path.open(encoding="utf-8") as f:
        for r in csv.DictReader(f):
            label = (r.get("label") or "").strip()
            if label not in {"0", "1"}:
                continue
            r["label"] = int(label)
            r["combined"] = float(r.get("combined") or 0.0)
            r["sino"] = float(r.get("sino") or 0.0)
            rows.append(r)
    return rows


def _bootstrap_ci(
    labels: list[int], n_boot: int = 2000, seed: int = 2026
) -> tuple[float, float]:
    if not labels:
        return (0.0, 0.0)
    rng = random.Random(seed)
    n = len(labels)
    means = []
    for _ in range(n_boot):
        sample = [labels[rng.randrange(n)] for _ in range(n)]
        means.append(sum(sample) / n)
    means.sort()
    lo = means[int(0.025 * n_boot)]
    hi = means[int(0.975 * n_boot)]
    return (lo, hi)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--csv",
        type=Path,
        default=Path("data/eval/eval_sample_100.csv"),
    )
    ap.add_argument(
        "--json",
        type=Path,
        default=None,
        help="optional path to write JSON summary",
    )
    ap.add_argument("--bootstrap", type=int, default=2000)
    args = ap.parse_args()

    if not args.csv.exists():
        sys.exit(f"missing {args.csv}")

    rows = _load(args.csv)
    if not rows:
        sys.exit(
            f"no rows with label in 0/1 found in {args.csv}. "
            "Fill the label column first."
        )
    total = len(rows)
    unlabeled_count = 0
    with args.csv.open(encoding="utf-8") as f:
        for r in csv.DictReader(f):
            if (r.get("label") or "").strip() not in {"0", "1"}:
                unlabeled_count += 1

    by_stratum: dict[str, list[int]] = {"high": [], "mid": [], "low": []}
    for r in rows:
        s = r.get("stratum", "mid")
        by_stratum.setdefault(s, []).append(r["label"])

    print(f"=== eval: {args.csv} ===")
    print(
        f"labeled: {total} / {total + unlabeled_count} "
        f"({unlabeled_count} still unlabeled)"
    )
    print()
    summary: dict = {}
    for s, labels in by_stratum.items():
        if not labels:
            continue
        n = len(labels)
        correct = sum(labels)
        prec = correct / n
        ci = _bootstrap_ci(labels, args.bootstrap)
        print(
            f"  {s:5s}  n={n:3d}  correct={correct:3d}  "
            f"precision={prec:.3f}  CI95=[{ci[0]:.3f}, {ci[1]:.3f}]"
        )
        summary[s] = {
            "n": n,
            "correct": correct,
            "precision": prec,
            "ci95": list(ci),
        }

    overall_labels = [r["label"] for r in rows]
    op = sum(overall_labels) / len(overall_labels)
    oci = _bootstrap_ci(overall_labels, args.bootstrap)
    print()
    print(
        f"  ALL   n={total:3d}  correct={sum(overall_labels):3d}  "
        f"precision={op:.3f}  CI95=[{oci[0]:.3f}, {oci[1]:.3f}]"
    )
    summary["overall"] = {
        "n": total,
        "correct": sum(overall_labels),
        "precision": op,
        "ci95": list(oci),
    }

    sino_correct = [r["sino"] for r in rows if r["label"] == 1]
    sino_wrong = [r["sino"] for r in rows if r["label"] == 0]
    if sino_correct and sino_wrong:
        m1 = statistics.mean(sino_correct)
        m0 = statistics.mean(sino_wrong)
        print()
        print(
            f"  sino mean: correct={m1:.3f}  wrong={m0:.3f}  "
            f"delta={m1 - m0:+.3f}"
        )
        summary["sino_mean_correct"] = m1
        summary["sino_mean_wrong"] = m0

    if args.json:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(
            json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        print(f"\nwrote {args.json}")


if __name__ == "__main__":
    main()
