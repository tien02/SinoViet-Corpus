"""Stage 7d: Internal hold-out MT.

Split aligned pairs 80/20, fine-tune Helsinki-zh-vi MarianMT, eval on hold-out.
Skips if pairs < HOLDOUT_MIN_PAIRS.
"""
from __future__ import annotations

import json
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from src.utils.config import (  # noqa: E402
    DEVICE,
    FINAL,
    HOLDOUT_MIN_PAIRS,
    HOLDOUT_RATIO,
    MT_BATCH,
    PAIRS_JSONL,
)

OUT = FINAL / "eval" / "holdout_mt.json"
MODEL_NAME = "Helsinki-NLP/opus-mt-zh-vi"


def load_pairs() -> list[dict]:
    return [json.loads(l) for l in PAIRS_JSONL.open(encoding="utf-8")]


def main() -> None:
    if not PAIRS_JSONL.exists():
        raise SystemExit(f"Run vecalign_runner first. Missing: {PAIRS_JSONL}")
    pairs = load_pairs()
    if len(pairs) < HOLDOUT_MIN_PAIRS:
        print(f"Skip hold-out MT: pairs={len(pairs)} < {HOLDOUT_MIN_PAIRS}")
        OUT.parent.mkdir(parents=True, exist_ok=True)
        OUT.write_text(json.dumps({
            "skipped": True,
            "reason": f"insufficient pairs ({len(pairs)} < {HOLDOUT_MIN_PAIRS})",
        }, indent=2), encoding="utf-8")
        return

    random.seed(42)
    random.shuffle(pairs)
    n_test = int(len(pairs) * HOLDOUT_RATIO)
    test = pairs[:n_test]
    train = pairs[n_test:]
    print(f"train={len(train)} test={len(test)}")

    from datasets import Dataset
    from transformers import (
        AutoModelForSeq2SeqLM,
        AutoTokenizer,
        DataCollatorForSeq2Seq,
        Seq2SeqTrainer,
        Seq2SeqTrainingArguments,
    )
    import numpy as np
    import sacrebleu

    tok = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_NAME).to(DEVICE)

    def preprocess(ex):
        m = tok(ex["src"], max_length=256, truncation=True, padding="max_length")
        lbl = tok(ex["tgt"], max_length=256, truncation=True, padding="max_length")
        m["labels"] = lbl["input_ids"]
        return m

    train_ds = Dataset.from_dict({"src": [p["src"] for p in train], "tgt": [p["tgt"] for p in train]})
    test_ds = Dataset.from_dict({"src": [p["src"] for p in test], "tgt": [p["tgt"] for p in test]})
    train_ds = train_ds.map(preprocess, remove_columns=["src", "tgt"], batched=True)
    test_ds = test_ds.map(preprocess, remove_columns=["src", "tgt"], batched=True)

    args = Seq2SeqTrainingArguments(
        output_dir=str(FINAL / "_mt_ckpt"),
        num_train_epochs=3,
        per_device_train_batch_size=MT_BATCH,
        per_device_eval_batch_size=MT_BATCH,
        predict_with_generate=True,
        eval_strategy="epoch",
        save_strategy="no",
        logging_steps=50,
        fp16=DEVICE == "cuda",
    )
    collator = DataCollatorForSeq2Seq(tok, model=model)

    trainer = Seq2SeqTrainer(
        model=model,
        args=args,
        train_dataset=train_ds,
        eval_dataset=test_ds,
        data_collator=collator,
        tokenizer=tok,
    )
    trainer.train()

    preds = trainer.predict(test_ds)
    pred_ids = preds.predictions
    if isinstance(pred_ids, tuple):
        pred_ids = pred_ids[0]
    pred_ids = np.clip(pred_ids, 0, tok.vocab_size - 1)
    hyps = tok.batch_decode(pred_ids, skip_special_tokens=True)
    refs = [p["tgt"] for p in test]
    bleu = sacrebleu.corpus_bleu(hyps, [refs]).score
    chrf = sacrebleu.corpus_chrf(hyps, [refs]).score

    result = {
        "n_train": len(train),
        "n_test": len(test),
        "bleu": bleu,
        "chrf": chrf,
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(result, indent=2, ensure_ascii=False))
    print(f"-> {OUT}")


if __name__ == "__main__":
    main()
