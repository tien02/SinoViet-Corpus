"""Stage 5 (alt): BertAlign sentence alignment.

Uses external/bertalign (bfsujason/bertalign). Two-pass DP: pass-1 finds
1-1 anchor points, pass-2 fills in 1-N / N-1 / N-N inside anchor windows.
Better recall than Vecalign on drift-prone corpora.

Sentences already split by Stage 3 (split_han + split_vi). We monkey-patch
the bertalign package to (a) reuse `EMBED_MODEL` (LaBSE by default) and
(b) bypass `googletrans` lang detection (fixed zh + vi).

Output: data/aligned/pairs.jsonl (same schema as vecalign_runner).
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from src.utils.config import (  # noqa: E402
    ALIGN_MIN_SCORE,
    BERTALIGN_REPO,
    DEVICE,
    EMBED_MODEL,
    HAN_SENT,
    PAIRS_JSONL,
    VI_SENT,
)


def _install_bertalign_patches() -> None:
    """Insert BERTALIGN_REPO on sys.path + stub optional deps."""
    if not BERTALIGN_REPO.exists():
        raise SystemExit(
            f"Bertalign repo missing: {BERTALIGN_REPO}. "
            "Run: git clone https://github.com/bfsujason/bertalign.git external/bertalign"
        )
    sys.path.insert(0, str(BERTALIGN_REPO))

    # Stub googletrans + sentence_splitter so bertalign.utils imports succeed
    # without pulling either dep. detect_lang monkey-patched in main().
    # Model selection handled by vendored bertalign/__init__.py reading
    # HVB_EMBED_MODEL env var.
    import types
    for name, attrs in [
        ("googletrans", {"Translator": type("Translator", (), {})}),
        ("sentence_splitter", {
            "SentenceSplitter": type("SentenceSplitter", (), {
                "__init__": lambda self, language=None: None,
                "split": lambda self, text: text.splitlines(),
            })
        }),
    ]:
        if name not in sys.modules:
            mod = types.ModuleType(name)
            for k, v in attrs.items():
                setattr(mod, k, v)
            sys.modules[name] = mod


def _load_bertalign_class():
    """Import Bertalign class. Fix LANG.ISO if needed."""
    import bertalign  # triggers HVBEncoder(EMBED_MODEL) via patched module
    import bertalign.aligner as _al
    from bertalign.utils import LANG

    if "zh" not in LANG.ISO:
        LANG.ISO["zh"] = "Chinese"
    if "vi" not in LANG.ISO:
        LANG.ISO["vi"] = "Vietnamese"

    return bertalign, _al


def _extract_pairs(aligner, han_sents, vi_sents):
    """Turn Bertalign `.result` beads into pair records with cosine scores.

    result: list of ([src_lines], [tgt_lines]) — either list may be empty
    (deletion / insertion). Skip empty-side beads (no alignment info).
    """
    src_vecs = aligner.src_vecs[0]  # (N_src, D) single-sentence rows
    tgt_vecs = aligner.tgt_vecs[0]
    pairs = []
    for bead in aligner.result:
        src_idxs, tgt_idxs = list(bead[0]), list(bead[1])
        if not src_idxs or not tgt_idxs:
            continue  # 0-1 or 1-0 = deletion / insertion; skip
        src_pool = src_vecs[src_idxs].mean(axis=0)
        tgt_pool = tgt_vecs[tgt_idxs].mean(axis=0)
        # Both L2-normalized → dot product = cosine similarity
        score = float(np.dot(src_pool, tgt_pool))
        pairs.append({
            "src_idx": list(map(int, src_idxs)),
            "tgt_idx": list(map(int, tgt_idxs)),
            "src": " ".join(han_sents[i] for i in src_idxs),
            "tgt": " ".join(vi_sents[i] for i in tgt_idxs),
            "score": score,
        })
    return pairs


def main() -> None:
    import os
    for p in [HAN_SENT, VI_SENT]:
        if not p.exists():
            raise SystemExit(f"Missing dependency: {p}. Run upstream stages first.")

    # Ensure vendored bertalign/__init__.py picks up the same model as
    # the rest of the pipeline (its default "LaBSE" would silently diverge).
    os.environ["HVB_EMBED_MODEL"] = EMBED_MODEL

    _install_bertalign_patches()
    bertalign, _al = _load_bertalign_class()
    Bertalign = bertalign.Bertalign

    han_sents = [json.loads(l)["text"].replace("\n", " ").strip() for l in HAN_SENT.open()]
    vi_sents = [json.loads(l)["text"].replace("\n", " ").strip() for l in VI_SENT.open()]
    han_sents = [s for s in han_sents if s]
    vi_sents = [s for s in vi_sents if s]
    print(f"src={len(han_sents):,} tgt={len(vi_sents):,}")

    src_txt = "\n".join(han_sents)
    tgt_txt = "\n".join(vi_sents)

    # Bertalign init calls detect_lang(src) then detect_lang(tgt). Fix both.
    _state = [0]
    def _fixed_detect(_text):
        _state[0] += 1
        return "zh" if _state[0] == 1 else "vi"
    _al.detect_lang = _fixed_detect

    print(f"embedding + aligning with {EMBED_MODEL} ...")
    aligner = Bertalign(
        src_txt, tgt_txt,
        max_align=5,
        top_k=3,
        win=5,
        skip=-0.1,
        margin=True,
        len_penalty=True,
        is_split=True,
    )
    aligner.align_sents()

    pairs = _extract_pairs(aligner, han_sents, vi_sents)
    kept = 0
    PAIRS_JSONL.parent.mkdir(parents=True, exist_ok=True)
    with PAIRS_JSONL.open("w", encoding="utf-8") as fout:
        for p in pairs:
            if p["score"] < ALIGN_MIN_SCORE:
                continue
            fout.write(json.dumps(p, ensure_ascii=False) + "\n")
            kept += 1
    print(f"beads: {len(pairs):,}, kept (score>={ALIGN_MIN_SCORE}): {kept:,}")
    print(f"output: {PAIRS_JSONL}")


if __name__ == "__main__":
    main()
