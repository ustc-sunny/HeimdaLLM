#!/usr/bin/env python3
"""Create a deterministic label-shuffled synthetic negative control."""

from __future__ import print_function

import argparse
from collections import Counter
import hashlib
import json
import os
from pathlib import Path
import random


def sha256_file(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_json_values(values):
    """Hash a value sequence without depending on JSONL key ordering."""
    digest = hashlib.sha256()
    for value in values:
        encoded = json.dumps(
            value, ensure_ascii=True, sort_keys=True, separators=(",", ":")
        ).encode("utf-8")
        digest.update(len(encoded).to_bytes(8, byteorder="big"))
        digest.update(encoded)
    return digest.hexdigest()


def atomic_jsonl(path, rows):
    temporary = Path(str(path) + ".tmp")
    if temporary.exists():
        temporary.unlink()
    with temporary.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=True, sort_keys=True) + "\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(str(temporary), str(path))


def atomic_json(path, payload):
    temporary = Path(str(path) + ".tmp")
    if temporary.exists():
        temporary.unlink()
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=True, indent=2, sort_keys=True)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(str(temporary), str(path))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--manifest-out", required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--min-label-agreement", type=float, default=0.0)
    parser.add_argument("--max-label-agreement", type=float, default=0.60)
    parser.add_argument("--attempts", type=int, default=1000)
    args = parser.parse_args()

    input_path = Path(args.input).expanduser().resolve()
    output_path = Path(args.output).expanduser().resolve()
    manifest_path = Path(args.manifest_out).expanduser().resolve()
    if not input_path.is_file():
        raise FileNotFoundError(str(input_path))
    if output_path.exists() or manifest_path.exists():
        raise FileExistsError("refusing to overwrite shuffled-control artifacts")
    if not (
        0.0 <= args.min_label_agreement
        <= args.max_label_agreement < 1.0
    ) or args.attempts <= 0:
        raise ValueError("invalid agreement threshold or attempt count")

    rows = []
    with input_path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            row = json.loads(line)
            if not isinstance(row.get("text"), str) or not row["text"].strip():
                raise ValueError("invalid text at input line %d" % line_number)
            row["label"] = str(row.get("label"))
            rows.append(row)
    if len(rows) < 2:
        raise ValueError("at least two synthetic rows are required")
    original = [row["label"] for row in rows]
    histogram = Counter(original)
    if len(histogram) < 2:
        raise ValueError("label-shuffled control requires at least two labels")

    rng = random.Random(args.seed)
    best = None
    best_agreement = None
    target_agreement = (
        args.min_label_agreement + args.max_label_agreement
    ) / 2.0
    attempts_used = 0
    for attempt in range(1, args.attempts + 1):
        candidate = list(original)
        rng.shuffle(candidate)
        agreement = sum(
            left == right for left, right in zip(original, candidate)
        ) / float(len(original))
        if best_agreement is None or abs(agreement - target_agreement) < abs(
            best_agreement - target_agreement
        ):
            best = candidate
            best_agreement = agreement
        if args.min_label_agreement <= agreement <= args.max_label_agreement:
            best = candidate
            best_agreement = agreement
            attempts_used = attempt
            break
    if attempts_used == 0:
        raise RuntimeError(
            "could not obtain label agreement in [%.3f, %.3f]; best was %.3f"
            % (
                args.min_label_agreement,
                args.max_label_agreement,
                best_agreement,
            )
        )
    if Counter(best) != histogram:
        raise AssertionError("label shuffle changed the label histogram")

    shuffled_rows = []
    for row, shuffled_label in zip(rows, best):
        updated = dict(row)
        updated["label"] = shuffled_label
        shuffled_rows.append(updated)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    atomic_jsonl(output_path, shuffled_rows)
    payload = {
        "schema_version": 1,
        "status": "complete",
        "seed": args.seed,
        "records": len(rows),
        "label_counts": dict(sorted(histogram.items())),
        "label_agreement_fraction": best_agreement,
        "min_label_agreement": args.min_label_agreement,
        "max_label_agreement": args.max_label_agreement,
        "attempts_used": attempts_used,
        "text_sequence_sha256": sha256_json_values(
            [row["text"] for row in rows]
        ),
        "text_multiset_sha256": sha256_json_values(sorted(
            row["text"] for row in rows
        )),
        "original_label_sequence_sha256": sha256_json_values(original),
        "shuffled_label_sequence_sha256": sha256_json_values(best),
        "input": str(input_path),
        "input_sha256": sha256_file(input_path),
        "output": str(output_path),
        "output_sha256": sha256_file(output_path),
    }
    atomic_json(manifest_path, payload)
    print(json.dumps(payload, sort_keys=True))


if __name__ == "__main__":
    main()
