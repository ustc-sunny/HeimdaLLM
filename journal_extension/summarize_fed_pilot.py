#!/usr/bin/env python3
"""Parse a HeimdaLLM FedFwd log and write arm-level reproducibility files."""

from __future__ import print_function

import argparse
import ast
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import re
import sys


EVAL_PATTERN = re.compile(r"eval_model\(\):\s*(\{.*\})\s*$")
FATAL_PATTERNS = (
    "Traceback (most recent call last)",
    "CUDA out of memory",
    "MPI_ABORT was invoked",
    "Primary job  terminated normally, but 1 process returned",
)


def sha256_file(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def parse_artifacts(values):
    artifacts = {}
    for value in values:
        if "=" not in value:
            raise ValueError("--artifact must have NAME=PATH form: %s" % value)
        name, raw_path = value.split("=", 1)
        if not name or not raw_path:
            raise ValueError("--artifact must have NAME=PATH form: %s" % value)
        path = Path(raw_path)
        record = {"path": str(path), "exists": path.is_file()}
        if path.is_file():
            record["bytes"] = path.stat().st_size
            record["sha256"] = sha256_file(path)
        artifacts[name] = record
    return artifacts


def expected_eval_rounds(rounds, eval_every, pre_update_eval=False):
    values = [-1] if pre_update_eval else []
    values.extend(range(0, rounds, eval_every))
    final_round = rounds - 1
    if final_round not in values:
        values.append(final_round)
    return values


def parse_metrics(log_text):
    metrics = []
    rejected = []
    for line_number, line in enumerate(log_text.splitlines(), start=1):
        match = EVAL_PATTERN.search(line)
        if not match:
            continue
        try:
            record = ast.literal_eval(match.group(1))
        except (SyntaxError, ValueError) as error:
            rejected.append({"line": line_number, "error": str(error)})
            continue
        if not isinstance(record, dict) or "acc" not in record or "eval_loss" not in record:
            rejected.append({"line": line_number, "error": "missing acc/eval_loss"})
            continue
        cleaned = {}
        for key, value in record.items():
            if hasattr(value, "item"):
                value = value.item()
            cleaned[str(key)] = value
        cleaned["log_line"] = line_number
        metrics.append(cleaned)
    return metrics, rejected


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--log", required=True)
    parser.add_argument("--metrics-out", required=True)
    parser.add_argument("--manifest-out", required=True)
    parser.add_argument("--command-file", required=True)
    parser.add_argument(
        "--condition", required=True,
        choices=[
            "no_cloud", "client_syn", "public_syn", "same_source_real_matched",
            "real_matched", "shuffled_label",
        ],
    )
    parser.add_argument("--cloud-source", required=True)
    parser.add_argument("--evaluation-mode", required=True,
                        choices=["dev", "final-test"])
    parser.add_argument("--stage", required=True)
    parser.add_argument("--alpha", required=True, type=float)
    parser.add_argument("--seed", required=True, type=int)
    parser.add_argument("--rounds", required=True, type=int)
    parser.add_argument("--eval-every", required=True, type=int)
    parser.add_argument("--pre-update-eval", action="store_true")
    parser.add_argument("--logical-clients", required=True, type=int)
    parser.add_argument("--clients-per-round", required=True, type=int)
    parser.add_argument("--mpi-workers", required=True, type=int)
    parser.add_argument("--exit-code", required=True, type=int)
    parser.add_argument("--artifact", action="append", default=[])
    parser.add_argument("--require-complete", action="store_true")
    args = parser.parse_args()

    log_path = Path(args.log)
    log_text = log_path.read_text(encoding="utf-8", errors="replace")
    evaluations, rejected = parse_metrics(log_text)
    completion_lines = [
        line_number for line_number, line in enumerate(log_text.splitlines(), start=1)
        if "training is finished" in line
    ]
    expected_rounds = expected_eval_rounds(
        args.rounds, args.eval_every, args.pre_update_eval
    )
    for index, evaluation in enumerate(evaluations):
        evaluation["evaluation_index"] = index
        evaluation["round"] = expected_rounds[index] if index < len(expected_rounds) else None

    fatal_matches = [pattern for pattern in FATAL_PATTERNS if pattern in log_text]
    checks = {
        "exit_code_zero": args.exit_code == 0,
        "single_completion_marker_present": len(completion_lines) == 1,
        "no_fatal_pattern": not fatal_matches,
        "all_expected_evaluations_present": len(evaluations) == len(expected_rounds),
        "final_round_evaluation_present": bool(evaluations)
        and evaluations[-1].get("round") == args.rounds - 1,
        "completion_follows_final_evaluation": bool(evaluations)
        and len(completion_lines) == 1
        and completion_lines[0] > evaluations[-1]["log_line"],
    }
    complete = all(checks.values())
    metrics_document = {
        "schema_version": 1,
        "status": "complete" if complete else "failed",
        "condition": args.condition,
        "cloud_source": args.cloud_source,
        "evaluation_mode": args.evaluation_mode,
        "seed": args.seed,
        "expected_eval_rounds": expected_rounds,
        "evaluations": evaluations,
        "final": evaluations[-1] if evaluations else None,
        "pre_update": (
            evaluations[0]
            if args.pre_update_eval and evaluations
            else None
        ),
        "rejected_metric_lines": rejected,
        "checks": checks,
        "fatal_matches": fatal_matches,
    }

    command_path = Path(args.command_file)
    manifest = {
        "schema_version": 1,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "status": metrics_document["status"],
        "stage": args.stage,
        "condition": args.condition,
        "cloud_source": args.cloud_source,
        "evaluation_mode": args.evaluation_mode,
        "alpha": args.alpha,
        "seed": args.seed,
        "rounds": args.rounds,
        "eval_every": args.eval_every,
        "pre_update_evaluation": args.pre_update_eval,
        "logical_clients": args.logical_clients,
        "clients_per_round": args.clients_per_round,
        "mpi_workers": args.mpi_workers,
        "mpi_processes": args.mpi_workers + 2,
        "exit_code": args.exit_code,
        "command_file": str(command_path),
        "command_sha256": sha256_file(command_path),
        "log": str(log_path),
        "log_sha256": sha256_file(log_path),
        "artifacts": parse_artifacts(args.artifact),
        "baseline_semantics": {
            "no_cloud": (
                "alpha=1 makes the cloud-guidance coefficient zero in every round; "
                "the same cloud artifact remains recorded for provenance and the "
                "synchronization barrier remains matched, while cloud BP is skipped."
            ),
            "client_syn": (
                "alpha=0.5 mixes client ZOO directions with true-NonDP "
                "client-synthetic guidance."
            ),
            "public_syn": (
                "alpha=0.5 uses synthetic guidance generated directly by the "
                "public pretrained language model and semantic category prompts; "
                "the generator receives no client-local LoRA update and no private "
                "record is used for generator training."
            ),
            "same_source_real_matched": (
                "alpha=0.5 uses a non-deployable oracle made from the same source "
                "clients' private train records, matched to the synthetic total, "
                "label counts, and label sequence."
            ),
            "real_matched": (
                "alpha=0.5 uses disjoint reserved-client real guidance with the "
                "same total and per-label cloud budget as client-synthetic guidance."
            ),
            "shuffled_label": (
                "alpha=0.5 uses the identical synthetic text sequence and label "
                "histogram after deterministic label permutation."
            ),
        }[args.condition],
        "evaluation_semantics": (
            "Rank 1 retains the real SST-2 data and fixed pilot partition; rank 0 "
            "alone switches to the arm-specific cloud H5. Evaluation mode is %s."
            " Logical round -1 is evaluated before any client update when "
            "pre_update_evaluation is true."
            % args.evaluation_mode
        ),
        "checks": checks,
    }

    metrics_path = Path(args.metrics_out)
    manifest_path = Path(args.manifest_out)
    metrics_path.write_text(
        json.dumps(metrics_document, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps({"status": metrics_document["status"], "checks": checks}, sort_keys=True))
    if args.require_complete and not complete:
        return 2
    return 0


if __name__ == "__main__":
    sys.exit(main())
