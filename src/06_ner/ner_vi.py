"""Stage 6b: Việt NER via PhoBERT/Underthesea.

Extract PER/LOC/ORG entities from each Việt sentence.
Output: data/aligned/entities_vi.jsonl
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from src.utils.config import VI_SENT  # noqa: E402

OUT = VI_SENT.parent / "entities_vi.jsonl"


def underthesea_ner(text: str) -> list[dict]:
    from underthesea import ner
    ents = ner(text)
    out = []
    for e in ents:
        if len(e) >= 3:
            out.append({"text": e[0], "label": e[3] if len(e) > 3 else e[2]})
    return out


def main() -> None:
    if not VI_SENT.exists():
        raise SystemExit(f"Run split_vi first. Missing: {VI_SENT}")

    OUT.parent.mkdir(parents=True, exist_ok=True)
    from tqdm import tqdm
    with VI_SENT.open("r", encoding="utf-8") as fin, OUT.open("w", encoding="utf-8") as fout:
        for i, line in enumerate(tqdm(list(fin), desc="Vi NER")):
            obj = json.loads(line)
            text = obj["text"]
            try:
                ents = underthesea_ner(text)
            except Exception as e:
                ents = []
                err = str(e)
            else:
                err = None
            fout.write(json.dumps({
                "idx": obj["idx"],
                "tap": obj.get("tap"),
                "page": obj.get("page"),
                "text": text,
                "entities": ents,
                **({"error": err} if err else {}),
            }, ensure_ascii=False) + "\n")
    print(f"output: {OUT}")


if __name__ == "__main__":
    main()
