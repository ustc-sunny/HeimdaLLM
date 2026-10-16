#!/usr/bin/env python3
"""Postprocess each existing DP-selected AG News buffer with a fixed label cycle.

No private vote or new randomizer is accessed. Outputs are restricted local
artifacts for the paired SHUFFLED_DP_LABEL control; never overwrite an input.
"""

import argparse
import hashlib
import json
import os
from collections import Counter
from pathlib import Path


TRIALS = {57: "D01", 58: "D02", 59: "D03"}
CYCLE = {"1": "2", "2": "3", "3": "4", "4": "1"}


def digest(path):
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def read_rows(path):
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()
            if line.strip()]


def prepare_one(source_root, output_root, seed):
    trial = TRIALS[seed]
    source = source_root / trial
    manifest = json.loads((source / "release_manifest.json").read_text())
    inp = source / "synthetic.jsonl"
    if manifest["trial_id"] != trial or manifest["status"] != "released":
        raise ValueError("source is not the prespecified released trial")
    if digest(inp) != manifest["selected_sha256"]:
        raise ValueError("source release SHA256 mismatch")
    rows = read_rows(inp)
    if len(rows) != 64 or len({r["text"] for r in rows}) != 64:
        raise ValueError("expected 64 unique public texts")
    before = Counter(str(r["label"]) for r in rows)
    if before != Counter({key: 16 for key in CYCLE}):
        raise ValueError("expected 16 records in each of four labels")
    slots = Counter((str(r["client_id"]), str(r["label"])) for r in rows)
    if set(client for client, _ in slots) != {"1", "21"} or any(
            slots[(client, label)] != 8 for client in ("1", "21") for label in CYCLE):
        raise ValueError("expected eight slots per client and class")
    shuffled = [dict(row, label=CYCLE[str(row["label"])]) for row in rows]
    if [r["text"] for r in rows] != [r["text"] for r in shuffled]:
        raise AssertionError("text sequence changed")
    if Counter(str(r["label"]) for r in shuffled) != before:
        raise AssertionError("global label balance changed")
    dest = output_root / ("seed_%d" % seed)
    if dest.exists():
        raise FileExistsError(str(dest))
    dest.mkdir(mode=0o700, parents=True)
    os.chmod(dest, 0o700)
    out = dest / "synthetic.jsonl"
    with os.fdopen(os.open(str(out), os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600),
                   "w", encoding="utf-8") as handle:
        for row in shuffled:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
    validation = {
        "arm": "SHUFFLED_DP_LABEL", "fl_seed": seed, "source_trial": trial,
        "source_sha256": manifest["selected_sha256"], "postprocessed_sha256": digest(out),
        "label_permutation": CYCLE, "texts_in_original_order_identical": True,
        "original_and_new_label_histogram": dict(sorted(before.items())),
        "per_client_per_original_class_slots": 8, "unique_texts": 64,
        "new_private_query": False, "new_dp_release": False,
        "downstream_fl_transcript_dp": False,
    }
    receipt = dest / "validation.json"
    with os.fdopen(os.open(str(receipt), os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600),
                   "w", encoding="utf-8") as handle:
        json.dump(validation, handle, indent=2, sort_keys=True)
        handle.write("\n")
    return validation


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--release-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args()
    if args.output_root.exists():
        raise FileExistsError("output root already exists: %s" % args.output_root)
    args.output_root.mkdir(mode=0o700, parents=True)
    os.chmod(args.output_root, 0o700)
    for seed in TRIALS:
        result = prepare_one(args.release_root, args.output_root, seed)
        print(json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()
