#!/usr/bin/env python3
"""Aggregate-only independent label and length audit for private-vote text."""

import argparse
import json
from collections import Counter
from pathlib import Path


MODEL = "textattack/distilbert-base-uncased-ag-news"
REVISION = "52ee64de95f38323f136c6f6b05e1af7c433417e"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--jsonl", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    rows = [json.loads(line) for line in args.jsonl.read_text().splitlines() if line.strip()]
    labels = [int(row["label"]) - 1 for row in rows]
    if len(rows) not in (16, 64) or Counter(labels) != {i: len(rows) // 4 for i in range(4)}:
        raise ValueError("unexpected AG News pilot class balance")
    import torch
    from transformers import AutoModelForSequenceClassification, AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(MODEL, revision=REVISION, local_files_only=True)
    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL, revision=REVISION, local_files_only=True).eval()
    torch.set_num_threads(4)
    predictions, lengths = [], []
    for offset in range(0, len(rows), 16):
        texts = [row["text"] for row in rows[offset:offset + 16]]
        lengths += [len(ids) for ids in tokenizer(texts, add_special_tokens=True)["input_ids"]]
        tokens = tokenizer(texts, padding=True, truncation=True,
                           max_length=64, return_tensors="pt")
        with torch.inference_mode():
            predictions += model(**tokens).logits.argmax(dim=-1).tolist()
    matrix = [[0] * 4 for _ in range(4)]
    for label, prediction in zip(labels, predictions):
        matrix[label][prediction] += 1
    report = {"records": len(rows), "class_balance": dict(Counter(str(label + 1) for label in labels)),
              "independent_classifier": MODEL, "classifier_revision": REVISION,
              "max_length": 64,
              "label_agreement": sum(a == b for a, b in zip(labels, predictions)) / len(rows),
              "fraction_exceeding_64_tokens": sum(n > 64 for n in lengths) / len(rows),
              "min_tokens": min(lengths), "max_tokens": max(lengths),
              "confusion_requested_by_predicted": matrix,
              "scope": "aggregate_proxy_only_no_private_text"}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps(report, sort_keys=True))


if __name__ == "__main__":
    main()
