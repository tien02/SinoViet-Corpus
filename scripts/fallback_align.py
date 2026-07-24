#!/usr/bin/env python
"""Fallback greedy alignment for unaligned Vietnamese sentences."""
import json
import os
import numpy as np
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.utils.config import PAIRS_JSONL, ALIGNED, INTERIM

print("=== Fallback Greedy Alignment ===\n")

# Use canonical symlinks (config-driven) so stage reruns stay in sync.
vi_sent_path = INTERIM / "vi_sentences.jsonl"
han_sent_path = INTERIM / "han_sentences.jsonl"
vi_embed_path = INTERIM / "vi_embeds.npy"
han_embed_path = INTERIM / "han_embeds.npy"

# Load embeddings (proper npy, no allow_pickle needed)
print("Loading embeddings...")
vi_embeds = np.load(str(vi_embed_path)).astype(np.float32)
han_embeds = np.load(str(han_embed_path)).astype(np.float32)
print(f"  Vi: {vi_embeds.shape}")
print(f"  Han: {han_embeds.shape}")

# Load sentences
print("Loading sentences...")
vi_sentences = {}
with vi_sent_path.open() as f:
    for line in f:
        obj = json.loads(line)
        vi_sentences[obj["idx"]] = obj

han_sentences = {}
with han_sent_path.open() as f:
    for line in f:
        obj = json.loads(line)
        han_sentences[obj["idx"]] = obj

print(f"  Vi: {len(vi_sentences)}")
print(f"  Han: {len(han_sentences)}")

# Load existing pairs
print(f"Loading existing pairs from {PAIRS_JSONL}...")
aligned_vi_indices = set()
existing_pairs = []
with PAIRS_JSONL.open() as f:
    for line in f:
        p = json.loads(line)
        existing_pairs.append(p)
        for tgt_idx in p.get("tgt_idx", []):
            aligned_vi_indices.add(tgt_idx)

print(f"  Existing pairs: {len(existing_pairs)}")
print(f"  Aligned Vi indices: {len(aligned_vi_indices)}")

# Find unaligned Vi
unaligned_vi = sorted(set(vi_sentences.keys()) - aligned_vi_indices)
print(f"\nUnaligned Vi sentences: {len(unaligned_vi)}")

# Greedy alignment for unaligned Vi
print("\nRunning greedy alignment...")
SIM_THRESHOLD = float(os.environ.get("HVB_GREEDY_MIN_SIM", "0.4"))
# Precompute norms once (embeds may not be perfectly normalized after fp32 round-trip).
han_norms = np.linalg.norm(han_embeds, axis=1)
han_unit = han_embeds / (han_norms[:, None] + 1e-8)

new_pairs = []
BATCH = 256
for i in range(0, len(unaligned_vi), BATCH):
    batch_idx = unaligned_vi[i:i+BATCH]
    if (i // BATCH) % 10 == 0:
        print(f"  {i}/{len(unaligned_vi)}...")
    vi_batch = vi_embeds[batch_idx]  # (B, 1024)
    vi_norms = np.linalg.norm(vi_batch, axis=1, keepdims=True)
    vi_unit = vi_batch / (vi_norms + 1e-8)
    sims = vi_unit @ han_unit.T  # (B, N_han)
    best = sims.argmax(axis=1)
    best_sim = sims.max(axis=1)
    for j, vi_idx in enumerate(batch_idx):
        s = float(best_sim[j])
        if s < SIM_THRESHOLD:
            continue
        best_han_idx = int(best[j])
        vi_obj = vi_sentences[vi_idx]
        han_obj = han_sentences[best_han_idx]
        # Score = cosine SIMILARITY (matches bertalign convention so
        # downstream export treats greedy pairs via _SCORE_IS_SIM rescue paths).
        new_pair = {
            "src_idx": [best_han_idx],
            "tgt_idx": [vi_idx],
            "src": han_obj["text"],
            "tgt": vi_obj["text"],
            "score": s,
            "tap": vi_obj["tap"],
            "method": "greedy",
        }
        new_pairs.append(new_pair)

print(f"\nCreated {len(new_pairs)} greedy pairs (sim >= {SIM_THRESHOLD})")

# Backup existing pairs then overwrite pairs.jsonl with combined set so
# downstream export reads a single canonical file.
import shutil
backup_file = ALIGNED / "pairs.bertalign_pre_greedy.jsonl"
shutil.copy(PAIRS_JSONL, backup_file)
print(f"backup: {backup_file}")

print(f"Writing {len(existing_pairs) + len(new_pairs)} total pairs to {PAIRS_JSONL}...")
with PAIRS_JSONL.open("w", encoding="utf-8") as f:
    for p in existing_pairs:
        f.write(json.dumps(p, ensure_ascii=False) + "\n")
    for p in new_pairs:
        f.write(json.dumps(p, ensure_ascii=False) + "\n")

total_pairs = len(existing_pairs) + len(new_pairs)
coverage = total_pairs / len(vi_sentences) * 100
print(f"\nComplete!")
print(f"  Bertalign pairs: {len(existing_pairs):,}")
print(f"  Greedy pairs:    {len(new_pairs):,}")
print(f"  Total pairs:     {total_pairs:,}")
print(f"  Vi coverage:     {total_pairs:,} / {len(vi_sentences):,} ({coverage:.1f}%)")
print(f"  Output: {PAIRS_JSONL}")
