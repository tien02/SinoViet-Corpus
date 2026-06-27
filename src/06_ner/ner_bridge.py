"""Stage 6c: Cross-lingual NER bridge — match entities between Han-Viet aligned pairs.

Compute NER-Bridge F1: how many aligned pairs have matching entities
(same transliteration / translation across src/tgt).

Output: data/aligned/entities.jsonl with matched pairs.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from src.utils.config import ENTITIES_JSONL, PAIRS_JSONL  # noqa: E402

HAN_ENT = PAIRS_JSONL.parent.parent / "interim" / "entities_han.jsonl"
VI_ENT = PAIRS_JSONL.parent.parent / "interim" / "entities_vi.jsonl"

# Common Han-Viet character mappings for name matching
HANVIET_MAP = {
    "阮": "Nguyễn", "黎": "Lê", "陳": "Trần", "李": "Lý", "鄭": "Trịnh",
    "莫": "Mạc", "胡": "Hồ", "楊": "Dương", "丁": "Đinh", "前": "Tiền",
    "後": "Hậu", "上": "Thượng", "下": "Hạ", "大": "Đại", "南": "Nam",
    "北": "Bắc", "東": "Đông", "西": "Tây", "中": "Trung", "國": "Quốc",
    "王": "Vương", "皇": "Hoàng", "帝": "Đế", "公": "Công", "侯": "Hầu",
    "伯": "Bá", "子": "Tử", "男": "Nam", "嘉": "Gia", "隆": "Long",
    "明": "Minh", "命": "Mệnh", "元": "Nguyên", "年": "Niên", "月": "Nguyệt",
    "日": "Nhật", "時": "Thời", "順": "Thuận", "化": "Hóa", "昇": "Thăng",
    "龍": "Long", "城": "Thành", "都": "Đô", "府": "Phủ", "縣": "Huyện",
    "州": "Châu", "社": "Xã", "村": "Thôn", "江": "Giang", "山": "Sơn",
    "河": "Hà", "海": "Hải", "湖": "Hồ", "池": "Điềm", "井": "Tỉnh",
}


def transliterate_hanviets(text: str) -> str:
    """Approximate Sino-Vietnamese reading."""
    return "".join(HANVIET_MAP.get(c, c) for c in text)


def normalize(s: str) -> str:
    import unicodedata
    s = unicodedata.normalize("NFC", s).lower()
    return "".join(c for c in s if c.isalnum())


def match_entities(han_ents: list[dict], vi_ents: list[dict]) -> list[tuple[str, str, float]]:
    """Match by approximate transliteration. Returns list of (han, vi, score)."""
    matches = []
    for h in han_ents:
        h_text = h.get("text", "")
        if not h_text:
            continue
        h_norm = normalize(transliterate_hanviets(h_text))
        for v in vi_ents:
            v_text = v.get("text", "")
            if not v_text:
                continue
            v_norm = normalize(v_text)
            # Partial match scoring
            if not h_norm or not v_norm:
                continue
            if h_norm in v_norm or v_norm in h_norm:
                matches.append((h_text, v_text, 1.0))
            elif len(h_norm) > 2 and h_norm[:3] == v_norm[:3]:
                matches.append((h_text, v_text, 0.5))
    return matches


def main() -> None:
    if not PAIRS_JSONL.exists():
        raise SystemExit(f"Run vecalign_runner first. Missing: {PAIRS_JSONL}")
    if not HAN_ENT.exists() or not VI_ENT.exists():
        raise SystemExit(f"Missing NER outputs. Run ner_han + ner_vi first.")

    # Index NER by sentence idx
    han_by_idx = {}
    with HAN_ENT.open(encoding="utf-8") as f:
        for line in f:
            o = json.loads(line)
            han_by_idx[o["idx"]] = o.get("entities", [])
    vi_by_idx = {}
    with VI_ENT.open(encoding="utf-8") as f:
        for line in f:
            o = json.loads(line)
            vi_by_idx[o["idx"]] = o.get("entities", [])

    n_total, n_matched = 0, 0
    ENTITIES_JSONL.parent.mkdir(parents=True, exist_ok=True)
    with PAIRS_JSONL.open(encoding="utf-8") as f, ENTITIES_JSONL.open("w", encoding="utf-8") as fout:
        for line in f:
            pair = json.loads(line)
            src_idxs = pair["src_idx"]
            tgt_idxs = pair["tgt_idx"]
            han_ents = [e for i in src_idxs for e in han_by_idx.get(i, [])]
            vi_ents = [e for i in tgt_idxs for e in vi_by_idx.get(i, [])]
            if han_ents or vi_ents:
                n_total += 1
                matches = match_entities(han_ents, vi_ents)
                if matches:
                    n_matched += 1
                fout.write(json.dumps({
                    "src_idx": src_idxs,
                    "tgt_idx": tgt_idxs,
                    "han_entities": han_ents,
                    "vi_entities": vi_ents,
                    "matches": [{"han": m[0], "vi": m[1], "score": m[2]} for m in matches],
                }, ensure_ascii=False) + "\n")

    coverage = (n_matched / n_total) if n_total else 0.0
    print(f"pairs with entities: {n_total:,}")
    print(f"pairs with >=1 match: {n_matched:,}")
    print(f"NER-Bridge coverage: {coverage:.1%}")
    print(f"output: {ENTITIES_JSONL}")


if __name__ == "__main__":
    main()
