"""Stage 5: Vecalign sentence alignment.

Uses external/vecalign (thompsonb/vecalign). Pre-computed LaBSE embeddings
fed via --src_embed/--tgt_embed flags (LASER raw float32 format).

Output: data/aligned/pairs.jsonl with {"src_idx", "tgt_idx", "src", "tgt", "score"}.
"""
from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from src.utils.config import (  # noqa: E402
    ALIGN_MIN_SCORE,
    HAN_EMBEDS,
    HAN_SENT,
    PAIRS_JSONL,
    VECALIGN_REPO,
    VI_EMBEDS,
    VI_SENT,
)


def prep_vecalign_input(
    sent_path: Path, tmp_file: Path
) -> tuple[int, list[int], list[str]]:
    """Write vecalign-format file (1 unique sentence per line, len >= 3).

    Returns (n_written, kept_indices, unique_sentences).
    kept_indices maps new line idx -> original sentence idx in jsonl.
    unique_sentences is the deduped list (indexed by new line idx).
    """
    seen: set[str] = set()
    kept_indices: list[int] = []
    unique_sentences: list[str] = []
    with tmp_file.open("w", encoding="utf-8") as fout, sent_path.open(
        "r", encoding="utf-8"
    ) as fin:
        for orig_idx, line in enumerate(fin):
            obj = json.loads(line)
            text = obj["text"].replace("\n", " ").replace("\r", " ").strip()
            if len(text) < 3:
                continue
            if text in seen:
                continue
            seen.add(text)
            fout.write(text + "\n")
            kept_indices.append(orig_idx)
            unique_sentences.append(text)
    return len(kept_indices), kept_indices, unique_sentences


def parse_vecalign_output(out_path: Path) -> list[dict]:
    """Vecalign output: '[src_idx]:[tgt_idx]:score' (thompsonb/vecalign format).

    idx field can be empty `[]`, single `[N]`, or range `[N--M]`.
    """
    pairs = []
    line_re = re.compile(r"^\[([0-9\-]*)\]:\[([0-9\-]*)\]:([\d.]+)")
    for line in out_path.open("r", encoding="utf-8"):
        m = line_re.match(line)
        if not m:
            continue
        pairs.append(
            {
                "src_idx_raw": m.group(1),
                "tgt_idx_raw": m.group(2),
                "score": float(m.group(3)),
            }
        )
    return pairs


def expand_indices(idx_str: str) -> list[int]:
    """'[0--1]' inner '0--1' -> [0, 1]; '5' -> [5]; '' -> []."""
    if not idx_str:
        return []
    if "--" in idx_str:
        a, b = idx_str.split("--")
        return list(range(int(a), int(b) + 1))
    return [int(idx_str)]


def main() -> None:
    if not VECALIGN_REPO.exists():
        raise SystemExit(
            f"Vecalign repo missing: {VECALIGN_REPO}. "
            "Run: git clone https://github.com/thompsonb/vecalign.git external/vecalign"
        )
    for p in [HAN_SENT, VI_SENT, HAN_EMBEDS, VI_EMBEDS]:
        if not p.exists():
            raise SystemExit(f"Missing dependency: {p}. Run upstream stages first.")

    tmp_dir = PAIRS_JSONL.parent / "_vecalign_tmp"
    tmp_dir.mkdir(parents=True, exist_ok=True)
    src_file = tmp_dir / "han.txt"
    tgt_file = tmp_dir / "vi.txt"

    src_n, src_kept, src_unique = prep_vecalign_input(HAN_SENT, src_file)
    tgt_n, tgt_kept, tgt_unique = prep_vecalign_input(VI_SENT, tgt_file)
    print(f"src={src_n:,} tgt={tgt_n:,}")

    # Slice embeddings to match deduped files (vecalign assumes 1:1 line:embed).
    src_raw = tmp_dir / "han_embeds.f32"
    tgt_raw = tmp_dir / "vi_embeds.f32"
    han_arr = np.load(HAN_EMBEDS)
    vi_arr = np.load(VI_EMBEDS)
    if han_arr.shape[0] != len(src_kept) or vi_arr.shape[0] != len(tgt_kept):
        print(
            f"WARN: embed rows (han={han_arr.shape[0]}, vi={vi_arr.shape[0]}) "
            f"!= kept (han={len(src_kept)}, vi={len(tgt_kept)}). "
            "Re-run Stage 4 to regenerate embeddings."
        )
    han_arr[src_kept].astype(np.float32).tofile(src_raw)
    vi_arr[tgt_kept].astype(np.float32).tofile(tgt_raw)

    out_file = tmp_dir / "alignment_output.txt"
    cmd = [
        sys.executable,
        str(VECALIGN_REPO / "vecalign.py"),
        "-s",
        str(src_file),
        "-t",
        str(tgt_file),
        "--src_embed",
        str(src_file),
        str(src_raw),
        "--tgt_embed",
        str(tgt_file),
        str(tgt_raw),
        "--print_aligned_text",
    ]
    print(" ".join(cmd))
    proc = subprocess.run(cmd, check=True, capture_output=True, text=True)
    out_file.write_text(proc.stdout, encoding="utf-8")

    raw_pairs = parse_vecalign_output(out_file)
    PAIRS_JSONL.parent.mkdir(parents=True, exist_ok=True)
    kept = 0
    with PAIRS_JSONL.open("w", encoding="utf-8") as fout:
        for p in raw_pairs:
            if p["score"] < ALIGN_MIN_SCORE:
                continue
            src_lines = expand_indices(p["src_idx_raw"])
            tgt_lines = expand_indices(p["tgt_idx_raw"])
            src_text = " ".join(
                src_unique[i] for i in src_lines if 0 <= i < len(src_unique)
            )
            tgt_text = " ".join(
                tgt_unique[i] for i in tgt_lines if 0 <= i < len(tgt_unique)
            )
            # Map deduped line idx back to original jsonl idx.
            src_orig = [src_kept[i] for i in src_lines if 0 <= i < len(src_kept)]
            tgt_orig = [tgt_kept[i] for i in tgt_lines if 0 <= i < len(tgt_kept)]
            fout.write(
                json.dumps(
                    {
                        "src_idx": src_orig,
                        "tgt_idx": tgt_orig,
                        "src": src_text,
                        "tgt": tgt_text,
                        "score": p["score"],
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )
            kept += 1

    if os.environ.get("HVB_KEEP_TMP"):
        print(f"KEEP tmp_dir: {tmp_dir}")
    else:
        shutil.rmtree(tmp_dir, ignore_errors=True)
    print(f"kept pairs (score>={ALIGN_MIN_SCORE}): {kept:,}")
    print(f"output: {PAIRS_JSONL}")


if __name__ == "__main__":
    main()
