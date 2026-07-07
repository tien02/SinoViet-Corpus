"""Combined reranker: vecalign distance + Sino-Viet phonetic precision.

Rescues pairs vecalign rejected (in pairs_review.jsonl) when their
Sino-Viet pronunciation strongly overlaps with the Việt side — mainly
effective for proper-noun lists (names, places, ranks) where semantic
embedding is weak but lexical Sino-Viet match is strong.

Reads:
  - data/aligned/pairs.jsonl         (kept, dist < ALIGN_MAX_DIST)
  - data/aligned/pairs_review.jsonl  (borderline / rejected, dist >= max)

Writes:
  - data/aligned/pairs_reranked.jsonl
    fields: src_idx, tgt_idx, src, tgt, score (vecalign dist),
            sino (precision 0..1), combined (0..1, higher=better),
            rescued (bool)

Pipeline integration (--in-place):
  - Backs up pairs.jsonl → pairs_pre_rerank.jsonl.bak
  - Writes merged (kept + rescued) → pairs.jsonl (sorted by combined desc)
  - Keeps pairs_reranked.jsonl as full enriched view (incl. combined field)
  - Review pool pairs NOT rescued stay in pairs_review.jsonl

Combined formula:
  norm_sim = 1 - score            # vecalign cosine distance to similarity
  combined = w_dist * norm_sim + w_sino * sino
"""
from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import sys
import unicodedata
from pathlib import Path

from cn2vn import converter


def strip_diacritics(s: str) -> str:
    s = s.lower()
    s = unicodedata.normalize("NFKD", s)
    return "".join(c for c in s if not unicodedata.combining(c))


_TOK_RE = re.compile(r"[A-Za-zÀ-ỹ]+")


def sino_precision(han: str, viet: str) -> tuple[float, int, int]:
    sino = converter.cn2vn(han)
    toks = _TOK_RE.findall(sino)
    viet_flat = strip_diacritics(viet)
    ge3 = [strip_diacritics(t) for t in toks if len(strip_diacritics(t)) >= 3]
    if not ge3:
        return 0.0, 0, 0
    matched = sum(1 for t in ge3 if t in viet_flat)
    return matched / len(ge3), matched, len(ge3)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--keep", type=float,
        default=float(os.environ.get("HVB_RERANK_MIN_SINO", "0.40")),
        help="min sino precision for rescue (default 0.40)",
    )
    ap.add_argument("--w-sino", type=float, default=0.5)
    ap.add_argument("--w-dist", type=float, default=0.5)
    ap.add_argument(
        "--in-place", action="store_true",
        help="backup pairs.jsonl and replace with reranked output",
    )
    ap.add_argument(
        "--aligned-dir", type=Path, default=Path("data/aligned"),
    )
    args = ap.parse_args()

    kept_in = args.aligned_dir / "pairs.jsonl"
    review_in = args.aligned_dir / "pairs_review.jsonl"
    if not kept_in.exists():
        sys.exit(f"missing {kept_in}")
    outp = args.aligned_dir / "pairs_reranked.jsonl"

    kept_pairs = [json.loads(l) for l in kept_in.open(encoding="utf-8")]
    review_pairs = (
        [json.loads(l) for l in review_in.open(encoding="utf-8")]
        if review_in.exists()
        else []
    )
    print(f"kept: {len(kept_pairs):,}  review: {len(review_pairs):,}")

    # Bertalign score = cosine similarity (higher=better); Vecalign = distance.
    aligner = os.environ.get("HVB_ALIGNER", "vecalign").lower()
    score_is_sim = aligner == "bertalign"

    def enrich(p: dict, rescued: bool) -> dict:
        prec, m, t = sino_precision(p["src"], p["tgt"])
        norm_sim = max(0.0, min(1.0, p["score"] if score_is_sim else 1.0 - p["score"]))
        combined = args.w_dist * norm_sim + args.w_sino * prec
        return {
            "src_idx": p["src_idx"],
            "tgt_idx": p["tgt_idx"],
            "src": p["src"],
            "tgt": p["tgt"],
            "score": p["score"],
            "sino": prec,
            "sino_match": f"{m}/{t}",
            "combined": combined,
            "rescued": rescued,
        }

    enriched = [enrich(p, False) for p in kept_pairs]
    rescued_enriched = [enrich(p, True) for p in review_pairs]

    rescue_candidates = [
        p for p in rescued_enriched if p["sino"] >= args.keep
    ]
    print(f"rescue candidates (sino>={args.keep}): {len(rescue_candidates)}")

    final = enriched + rescue_candidates
    final.sort(key=lambda x: x["combined"], reverse=True)

    rescued_count = len(rescue_candidates)
    print(
        f"final kept: {len(final):,}  "
        f"(vecalign kept: {len(enriched):,}, rescued: {rescued_count})"
    )

    with outp.open("w", encoding="utf-8") as f:
        for p in final:
            f.write(json.dumps(p, ensure_ascii=False) + "\n")
    print(f"wrote {outp}")

    if args.in_place:
        bak = kept_in.with_suffix(".jsonl.bak")
        # Don't overwrite existing backup — preserve original vecalign output.
        if not bak.exists():
            shutil.copy2(kept_in, bak)
            print(f"backup: {bak}")
        else:
            print(f"backup exists (preserved): {bak}")
        # Export reads pairs.jsonl. Strip enriched fields so export schema
        # stays clean (just src_idx/tgt_idx/src/tgt/score).
        with kept_in.open("w", encoding="utf-8") as f:
            for p in final:
                row = {
                    "src_idx": p["src_idx"],
                    "tgt_idx": p["tgt_idx"],
                    "src": p["src"],
                    "tgt": p["tgt"],
                    "score": p["score"],
                }
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
        print(f"replaced {kept_in} with reranked output")

    if rescue_candidates:
        print("\n--- top 5 rescued examples ---")
        for p in sorted(
            rescue_candidates, key=lambda x: x["combined"], reverse=True
        )[:5]:
            print(
                f"combined={p['combined']:.3f} sino={p['sino']:.2f} "
                f"dist={p['score']:.3f}"
            )
            print(f"  H: {p['src'][:80]}")
            print(f"  S: {converter.cn2vn(p['src'])[:100]}")
            print(f"  V: {p['tgt'][:80]}")


if __name__ == "__main__":
    main()
