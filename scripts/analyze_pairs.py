#!/usr/bin/env python
"""Analyze pairs.jsonl: score distribution, sino precision per tap."""
import json
import sys
from pathlib import Path
from collections import defaultdict
import statistics

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.utils.config import PAIRS_JSONL


def _sino_precision(han: str, viet: str) -> float:
    """Han→Sino-Viet pronunciation overlap with Việt side. 0..1."""
    try:
        from cn2vn import converter
        sino = converter.cn2vn(han)
        import re, unicodedata
        toks = re.findall(r"[A-Za-zÀ-ỹ]+", sino)
        viet_flat = unicodedata.normalize("NFKD", viet.lower())
        viet_flat = "".join(c for c in viet_flat if not unicodedata.combining(c))
        ge3 = [
            "".join(c for c in unicodedata.normalize("NFKD", t.lower()) if not unicodedata.combining(c))
            for t in toks if len("".join(c for c in unicodedata.normalize("NFKD", t.lower()) if not unicodedata.combining(c))) >= 3
        ]
        if not ge3:
            return 0.0
        matched = sum(1 for t in ge3 if t in viet_flat)
        return matched / len(ge3)
    except ImportError:
        return 0.0


def main():
    if not PAIRS_JSONL.exists():
        print(f"Error: {PAIRS_JSONL} not found. Run align stage first.")
        return

    stats_by_tap = defaultdict(lambda: {"scores": [], "sinos": [], "count": 0})

    with PAIRS_JSONL.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            p = json.loads(line)
            tap = p.get("tap", "unknown")
            score = float(p.get("score", 0.0))
            han = p.get("src", "")
            viet = p.get("tgt", "")

            sino = _sino_precision(han, viet)
            stats_by_tap[tap]["scores"].append(score)
            stats_by_tap[tap]["sinos"].append(sino)
            stats_by_tap[tap]["count"] += 1

    print("\n=== Pairs Analysis ===\n")
    print(f"{'Tap':<8} {'Count':<8} {'Score (cosine)':<50} {'Sino Precision':<30}")
    print("-" * 96)

    total_pairs = 0
    for tap in sorted(stats_by_tap.keys()):
        data = stats_by_tap[tap]
        scores = data["scores"]
        sinos = data["sinos"]
        total_pairs += data["count"]

        score_min = min(scores) if scores else 0
        score_max = max(scores) if scores else 0
        score_mean = statistics.mean(scores) if scores else 0
        score_median = statistics.median(scores) if scores else 0

        sino_min = min(sinos) if sinos else 0
        sino_max = max(sinos) if sinos else 0
        sino_mean = statistics.mean(sinos) if sinos else 0

        score_str = f"min={score_min:.3f} mean={score_mean:.3f} max={score_max:.3f} med={score_median:.3f}"
        sino_str = f"min={sino_min:.3f} mean={sino_mean:.3f} max={sino_max:.3f}"

        print(f"{tap:<8} {data['count']:<8} {score_str:<50} {sino_str:<30}")

    print("-" * 96)
    print(f"{'TOTAL':<8} {total_pairs:<8}")
    print()


if __name__ == "__main__":
    main()
