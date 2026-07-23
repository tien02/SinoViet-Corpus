#!/usr/bin/env python
"""Add tap field to existing pairs.jsonl from vi_sentences.jsonl"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.utils.config import VI_SENT, PAIRS_JSONL

def main():
    # Load vi_sentences to build vi_idx -> tap mapping
    vi_tap_map = {}
    with VI_SENT.open(encoding="utf-8") as f:
        for line in f:
            obj = json.loads(line)
            vi_idx = obj["idx"]
            tap = obj.get("tap", "unknown")
            vi_tap_map[vi_idx] = tap

    print(f"Loaded {len(vi_tap_map)} vi_sentences with tap info")

    # Add tap to pairs
    output_path = PAIRS_JSONL
    with PAIRS_JSONL.open(encoding="utf-8") as fin:
        lines = [line.strip() for line in fin if line.strip()]

    updated = 0
    with output_path.open("w", encoding="utf-8") as fout:
        for line in lines:
            p = json.loads(line)
            tgt_idxs = p.get("tgt_idx", [])
            if tgt_idxs and tgt_idxs[0] in vi_tap_map:
                p["tap"] = vi_tap_map[tgt_idxs[0]]
                updated += 1
            fout.write(json.dumps(p, ensure_ascii=False) + "\n")

    print(f"Updated {updated} pairs with tap field")
    print(f"Output: {output_path}")

if __name__ == "__main__":
    main()
