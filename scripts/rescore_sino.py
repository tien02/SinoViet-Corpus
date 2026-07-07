"""Prototype: Sino-Viet pronunciation reranker.

Reads pairs.jsonl, converts each Han sentence to Sino-Viet pronunciation
via cn2vn, then computes lexical overlap with the Việt side. Reports
distribution of phonetic scores and how they correlate with vecalign
distance.

Goal: test paper's claim (HCMUS 2026, Si-Dien Dinh) that Sino-Viet
phrase matching beats pure LaBSE/Bertalign on Han-Viet classical text.

Usage:
    uv run python scripts/rescore_sino.py [--in data/aligned/pairs.jsonl]
"""
from __future__ import annotations

import argparse
import json
import re
import sys
import unicodedata
from pathlib import Path

from cn2vn import converter


def strip_diacritics(s: str) -> str:
    """Lowercase + strip Vietnamese diacritics → ASCII (for fuzzy match)."""
    s = s.lower()
    s = unicodedata.normalize("NFKD", s)
    return "".join(c for c in s if not unicodedata.combining(c))


def sino_score(han: str, viet: str) -> dict:
    """Compute phonetic overlap between Han (via Sino-Viet) and Việt."""
    sino = converter.cn2vn(han)
    sino_tokens = re.findall(r"[A-Za-zÀ-ỹ]+", sino)
    if not sino_tokens:
        return {"sino": sino, "tokens": 0, "matched": 0, "precision": 0.0}

    viet_flat = strip_diacritics(viet)
    tokens_ge3 = 0
    matched = 0
    for tok in sino_tokens:
        t = strip_diacritics(tok)
        if len(t) < 3:
            continue
        tokens_ge3 += 1
        if t in viet_flat:
            matched += 1
    precision = matched / tokens_ge3 if tokens_ge3 else 0.0
    return {
        "sino": sino,
        "tokens": tokens_ge3,
        "matched": matched,
        "precision": precision,
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="inp", default="data/aligned/pairs.jsonl")
    ap.add_argument("--n", type=int, default=0, help="sample N pairs (0=all)")
    args = ap.parse_args()

    inp = Path(args.inp)
    if not inp.exists():
        sys.exit(f"missing {inp}")

    pairs = [json.loads(l) for l in inp.open(encoding="utf-8")]
    if args.n:
        pairs = pairs[: args.n]

    print(f"loaded {len(pairs):,} pairs from {inp}")
    bins = [(0.0, 0.3), (0.3, 0.5), (0.5, 0.7), (0.7, 0.9), (0.9, 1.1)]
    bin_stats = {b: [0, 0.0] for b in bins}
    samples_per_bin = {b: None for b in bins}

    enriched = []
    for p in pairs:
        s = sino_score(p["src"], p["tgt"])
        d = p["score"]
        for lo, hi in bins:
            if lo <= d < hi:
                bin_stats[(lo, hi)][0] += 1
                bin_stats[(lo, hi)][1] += s["precision"]
                if samples_per_bin[(lo, hi)] is None and s["tokens"] >= 3:
                    samples_per_bin[(lo, hi)] = (p, s)
                break
        enriched.append({**p, "sino": s})

    print(f"{'dist bin':<14} {'n':>6} {'avg sino_prec':>14}")
    for (lo, hi), (n, sum_p) in bin_stats.items():
        avg = sum_p / n if n else 0
        print(f"  [{lo:.1f},{hi:.1f})  {n:>6}  {avg:>14.3f}")
        if samples_per_bin[(lo, hi)]:
            p, s = samples_per_bin[(lo, hi)]
            han_short = p["src"][:50] + ("..." if len(p["src"]) > 50 else "")
            viet_short = p["tgt"][:60] + ("..." if len(p["tgt"]) > 60 else "")
            print(f"      H: {han_short}")
            print(f"      S: {s['sino'][:80]}")
            print(f"      V: {viet_short}")
            print(f"      match: {s['matched']}/{s['tokens']} = {s['precision']:.2f}")

    outp = inp.with_name(inp.stem + "_sino.jsonl")
    with outp.open("w", encoding="utf-8") as f:
        for p in enriched:
            f.write(json.dumps(p, ensure_ascii=False) + "\n")
    print(f"\nwrote enriched pairs to {outp}")


if __name__ == "__main__":
    main()
