"""Stage 6a: Hán NER via HanLP.

Extract PER/LOC/ORG/TIME entities from each Hán sentence.
Output: data/aligned/entities_han.jsonl
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from src.utils.config import HAN_SENT  # noqa: E402

OUT = HAN_SENT.parent / "entities_han.jsonl"
LABEL_MAP = {"PER": "PERSON", "LOC": "LOCATION", "ORG": "ORG", "TIME": "TIME"}


def build_pipeline():
    import hanlp
    return hanlp.load(
        hanlp.pretrained.mtl.CLOSE_TOK_POS_NER_SRL_DEP_SDP_CON_ELECTRA_BASE_ZH,
        tasks=["ner"],
    )


def main() -> None:
    if not HAN_SENT.exists():
        raise SystemExit(f"Run split_han first. Missing: {HAN_SENT}")
    hanlp_ner = build_pipeline()

    sentences = []
    with HAN_SENT.open("r", encoding="utf-8") as f:
        for line in f:
            obj = json.loads(line)
            sentences.append(obj["text"])

    OUT.parent.mkdir(parents=True, exist_ok=True)
    from tqdm import tqdm
    with OUT.open("w", encoding="utf-8") as fout:
        for i, text in enumerate(tqdm(sentences, desc="Han NER")):
            try:
                doc = hanlp_ner(text)
                ner_result = doc.get("ner", doc.get("ner/msra", []))
                entities = []
                if ner_result:
                    for ent in ner_result:
                        if len(ent) >= 2:
                            surface = ent[0] if isinstance(ent[0], str) else "".join(ent[0])
                            label = ent[1] if isinstance(ent[1], str) else ent[1][0]
                            entities.append({"text": surface, "label": LABEL_MAP.get(label, label)})
                fout.write(json.dumps({
                    "idx": i,
                    "text": text,
                    "entities": entities,
                }, ensure_ascii=False) + "\n")
            except Exception as e:
                fout.write(json.dumps({"idx": i, "text": text, "entities": [], "error": str(e)}, ensure_ascii=False) + "\n")
    print(f"output: {OUT}")


if __name__ == "__main__":
    main()
