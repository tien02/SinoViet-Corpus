"""Stage 8: Export final HVB corpus.

Merge pairs + entities into final JSONL with all metadata.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from src.utils.config import (  # noqa: E402
    ENTITIES_JSONL,
    FINAL_CORPUS,
    PAIRS_JSONL,
)


def main() -> None:
    if not PAIRS_JSONL.exists():
        raise SystemExit(f"Missing {PAIRS_JSONL}")
    ents_by_key = {}
    if ENTITIES_JSONL.exists():
        for line in ENTITIES_JSONL.open(encoding="utf-8"):
            o = json.loads(line)
            ents_by_key[(tuple(o["src_idx"]), tuple(o["tgt_idx"]))] = o

    FINAL_CORPUS.parent.mkdir(parents=True, exist_ok=True)
    n = 0
    with PAIRS_JSONL.open(encoding="utf-8") as fin, FINAL_CORPUS.open("w", encoding="utf-8") as fout:
        for line in fin:
            p = json.loads(line)
            key = (tuple(p["src_idx"]), tuple(p["tgt_idx"]))
            ent = ents_by_key.get(key, {})
            rec = {
                "src": p["src"],
                "tgt": p["tgt"],
                "src_idx": p["src_idx"],
                "tgt_idx": p["tgt_idx"],
                "labse_score": p["score"],
                "entities": ent.get("matches", []),
            }
            fout.write(json.dumps(rec, ensure_ascii=False) + "\n")
            n += 1
    print(f"corpus: {n:,} pairs -> {FINAL_CORPUS}")


if __name__ == "__main__":
    main()
