#!/usr/bin/env python3
"""Select balanced Qwen candidates by a public deterministic rule (Non-DP)."""

import argparse
import hashlib
import json
import os
from collections import Counter
from pathlib import Path


LABELS = ("1", "2", "3", "4")
CLIENTS = (1, 21)


def digest(path):
    value = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            value.update(chunk)
    return value.hexdigest()


def select(pool, seed, per_client_label=8):
    if seed < 0:
        raise ValueError("seed must be nonnegative")
    groups = {label: [] for label in LABELS}
    seen = set()
    for line in pool.read_text().splitlines():
        row = json.loads(line)
        label, content = str(row["label"]), row["text"]
        if label not in groups or not isinstance(content, str) or not content.strip():
            raise ValueError("invalid public candidate row")
        key = " ".join(content.lower().split())
        if key in seen:
            raise ValueError("duplicate public candidate text")
        seen.add(key)
        groups[label].append(content)
    if {label: len(texts) for label, texts in groups.items()} != {label: 32 for label in LABELS}:
        raise ValueError("expected exactly 32 public candidates per label")
    chosen = []
    for label in LABELS:
        texts = groups[label]
        order = sorted(range(len(texts)), key=lambda index: hashlib.sha256(
            ("public_index_selection_v1|%d|%s|%d" % (seed, label, index)).encode()).digest())
        for slot, client_id in enumerate(CLIENTS):
            for index in order[slot * per_client_label:(slot + 1) * per_client_label]:
                chosen.append({"client_id": client_id, "label": label, "text": texts[index]})
    if len(chosen) != len(CLIENTS) * len(LABELS) * per_client_label:
        raise ValueError("candidate pool insufficient for disjoint selection")
    return chosen


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidates-jsonl", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--expected-candidate-sha256", required=True)
    parser.add_argument("--seed", type=int, default=57)
    args = parser.parse_args()
    if args.output_dir.exists():
        parser.error("output directory already exists")
    if digest(args.candidates_jsonl) != args.expected_candidate_sha256:
        raise ValueError("candidate pool hash differs from fixed protocol")
    selected = select(args.candidates_jsonl, args.seed)
    args.output_dir.mkdir(parents=True, mode=0o700)
    os.chmod(args.output_dir, 0o700)
    output = args.output_dir / "synthetic.jsonl"
    with output.open("w") as handle:
        for row in selected:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    os.chmod(output, 0o600)
    manifest = {"schema_version": 1, "status": "complete", "privacy": "Non-DP",
                "selection": "public_sha256_index_order_v1_no_private_records",
                "seed": args.seed, "candidate_sha256": args.expected_candidate_sha256,
                "selected_sha256": digest(output), "selected_records": len(selected),
                "label_counts": dict(Counter(row["label"] for row in selected)),
                "client_counts": dict(Counter(str(row["client_id"]) for row in selected)),
                "private_records_read": False, "official_test_used": False}
    (args.output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"status": "complete", "selected_sha256": manifest["selected_sha256"]}), flush=True)


if __name__ == "__main__":
    main()
