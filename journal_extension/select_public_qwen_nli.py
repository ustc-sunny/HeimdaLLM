#!/usr/bin/env python3
"""Select Qwen AG News guidance with a frozen public NLI model (Non-DP)."""

import argparse
import hashlib
import json
import os
from collections import Counter
from pathlib import Path

from select_agnews_dp_public import (
    DESCRIPTIONS, MODEL, REVISION, TEMPLATE, score_texts,
)


LABELS = ("1", "2", "3", "4")
CLIENTS = (1, 21)


def digest(path):
    value = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            value.update(chunk)
    return value.hexdigest()


def read_pool(path, expected_sha256):
    if digest(path) != expected_sha256:
        raise ValueError("candidate pool hash differs from fixed public pool")
    rows = [json.loads(line) for line in path.read_text().splitlines()]
    if len(rows) != 128 or Counter(str(row["label"]) for row in rows) != {
        label: 32 for label in LABELS
    }:
        raise ValueError("expected exactly 32 candidates for each of four labels")
    if any(not isinstance(row["text"], str) or not row["text"].strip()
           for row in rows):
        raise ValueError("empty candidate text")
    normalized = [" ".join(row["text"].lower().split()) for row in rows]
    if len(set(normalized)) != len(rows):
        raise ValueError("duplicate candidate text")
    return rows


def choose(rows, scores):
    if len(rows) != len(scores):
        raise ValueError("candidate and score counts differ")
    selected = []
    ranks = {}
    for label in LABELS:
        indices = [i for i, row in enumerate(rows) if str(row["label"]) == label]
        ordered = sorted(indices, key=lambda i: (-scores[i][int(label) - 1], i))
        # Interleave the top 16 so both fixed client slots receive comparable quality.
        for slot, client_id in enumerate(CLIENTS):
            for i in ordered[slot:16:2]:
                selected.append({"client_id": client_id, "label": label,
                                 "text": rows[i]["text"]})
        ranks[label] = {"selected_indices": ordered[:16],
                        "selected_target_score_mean": sum(
                            scores[i][int(label) - 1] for i in ordered[:16]) / 16,
                        "rejected_target_score_mean": sum(
                            scores[i][int(label) - 1] for i in ordered[16:]) / 16}
    if len(selected) != 64 or Counter(
        (row["client_id"], row["label"]) for row in selected
    ) != {(client, label): 8 for client in CLIENTS for label in LABELS}:
        raise AssertionError("selected output is not balanced")
    return selected, ranks


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidates-jsonl", required=True, type=Path)
    parser.add_argument("--expected-candidate-sha256", required=True)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--batch-size", type=int, default=8)
    args = parser.parse_args()
    if args.output_dir.exists():
        parser.error("output directory already exists")
    if args.batch_size < 1:
        parser.error("batch size must be positive")
    rows = read_pool(args.candidates_jsonl, args.expected_candidate_sha256)

    import torch
    from transformers import AutoModelForSequenceClassification, AutoTokenizer
    torch.set_num_threads(min(8, torch.get_num_threads()))
    tokenizer = AutoTokenizer.from_pretrained(
        MODEL, revision=REVISION, local_files_only=True, trust_remote_code=False)
    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL, revision=REVISION, local_files_only=True, trust_remote_code=False).eval()
    entailment = model.config.label2id.get("entailment")
    if entailment is None:
        entailment = model.config.label2id.get("ENTAILMENT")
    if entailment is None:
        raise ValueError("NLI entailment label not found")
    scores = score_texts([row["text"] for row in rows], tokenizer, model,
                         torch, "cpu", args.batch_size, entailment)
    selected, ranks = choose(rows, scores)

    args.output_dir.mkdir(parents=True, mode=0o700)
    os.chmod(args.output_dir, 0o700)
    output = args.output_dir / "synthetic.jsonl"
    with output.open("w") as handle:
        for row in selected:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    os.chmod(output, 0o600)
    manifest = {"schema_version": 1, "status": "complete", "privacy": "Non-DP",
                "selection": "frozen_public_bart_mnli_top16_interleaved_v1",
                "model": MODEL, "model_revision": REVISION,
                "hypothesis_template": TEMPLATE, "descriptions": list(DESCRIPTIONS),
                "candidate_sha256": args.expected_candidate_sha256,
                "selected_sha256": digest(output), "selected_records": len(selected),
                "label_counts": dict(Counter(row["label"] for row in selected)),
                "client_counts": dict(Counter(str(row["client_id"]) for row in selected)),
                "private_records_read": False, "official_test_used": False,
                "device": "cpu", "rank_summary": ranks}
    (args.output_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"status": "complete", "selected_sha256": digest(output),
                      "selection": manifest["selection"]}), flush=True)


if __name__ == "__main__":
    main()
