#!/usr/bin/env python3
"""Replay a Qwen candidate pool with no private-record access (Non-DP)."""

import argparse
import json
import os
from pathlib import Path
from types import SimpleNamespace

from augpe_nondp_pilot import LABELS, generate_candidates, gpu_preflight, sha256


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--generator-model", required=True)
    parser.add_argument("--task-tokenizer", required=True)
    parser.add_argument("--expected-sha256", required=True)
    parser.add_argument("--gpu", type=int, default=2)
    parser.add_argument("--seed", type=int, default=57)
    parser.add_argument("--candidates-per-label", type=int, default=32)
    args = parser.parse_args()
    if args.output_dir.exists():
        parser.error("output directory already exists")
    preflight = gpu_preflight(args.gpu, 14000)
    from transformers import AutoTokenizer
    task_tokenizer = AutoTokenizer.from_pretrained(args.task_tokenizer, local_files_only=True)
    generation_args = SimpleNamespace(generator_model=args.generator_model, gpu=args.gpu,
                                      seed=args.seed, candidates_per_label=args.candidates_per_label)
    candidates = generate_candidates(generation_args, task_tokenizer, forbidden=set())
    args.output_dir.mkdir(parents=True, mode=0o700)
    os.chmod(args.output_dir, 0o700)
    pool = args.output_dir / "candidates.jsonl"
    with pool.open("w") as handle:
        for label in LABELS:
            for candidate in candidates[label]:
                handle.write(json.dumps({"label": label, "text": candidate}, ensure_ascii=False) + "\n")
    os.chmod(pool, 0o600)
    actual = sha256(pool)
    manifest = {"schema_version": 1, "status": "verified" if actual == args.expected_sha256 else "mismatch",
                "method": "public_only_qwen_replay_no_private_source_access", "seed": args.seed,
                "candidate_records": sum(map(len, candidates.values())),
                "candidates_per_label": args.candidates_per_label,
                "candidate_sha256": actual, "expected_sha256": args.expected_sha256,
                "gpu_preflight": preflight, "private_records_read": False,
                "official_test_used": False}
    (args.output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"status": manifest["status"], "candidate_sha256": actual}), flush=True)
    if actual != args.expected_sha256:
        raise ValueError("public-only replay differs from frozen candidate pool")


if __name__ == "__main__":
    main()
