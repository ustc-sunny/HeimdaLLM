#!/usr/bin/env python3
"""Summarize completed DP arms without copying private staging or raw logs."""

import argparse
import csv
import hashlib
import json
import statistics
from datetime import datetime, timezone
from pathlib import Path


def load(path):
    with path.open(encoding="utf-8") as handle:
        return json.load(handle)


def write_atomic(path, value):
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(value, encoding="utf-8")
    temporary.replace(path)


def stats(values):
    return {
        "n": len(values),
        "mean": statistics.mean(values),
        "sample_sd": statistics.stdev(values) if len(values) > 1 else None,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-root", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--run-pattern", default="ustc_agnews_dp_eps{epsilon}_20260928")
    parser.add_argument("--epsilons", default="8,4,2,1")
    parser.add_argument("--seeds", default="57,58,59")
    parser.add_argument("--rounds", type=int, default=50)
    parser.add_argument("--reference-curves")
    args = parser.parse_args()
    root = Path(args.results_root).expanduser().resolve()
    output = Path(args.output_dir).expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    epsilons = args.epsilons.split(",")
    seeds = [int(value) for value in args.seeds.split(",")]
    reference = {}
    if args.reference_curves:
        with Path(args.reference_curves).open(newline="", encoding="utf-8") as handle:
            for row in csv.DictReader(handle):
                reference[(int(row["seed"]), row["arm"], int(row["round"]))] = float(row["accuracy"])

    runs = []
    missing = []
    curves = []
    expected_rounds = list(range(-1, args.rounds))
    for epsilon in epsilons:
        run_id = args.run_pattern.format(epsilon=epsilon)
        for seed in seeds:
            base = root / run_id / ("seed_%d" % seed)
            metrics_path = base / "arms/client_syn/metrics.json"
            generator_path = base / "synthetic/generated/manifest.json"
            validation_path = base / "synthetic/generated/validation_manifest.json"
            if not all(path.is_file() for path in (metrics_path, generator_path, validation_path)):
                missing.append({"epsilon": epsilon, "seed": seed, "run_id": run_id})
                continue
            metrics = load(metrics_path)
            generator = load(generator_path)
            validation = load(validation_path)
            if metrics.get("status") != "complete" or not all(metrics.get("checks", {}).values()):
                missing.append({"epsilon": epsilon, "seed": seed, "run_id": run_id,
                                "reason": "arm_incomplete"})
                continue
            if [row["round"] for row in metrics["evaluations"]] != expected_rounds:
                raise ValueError("unexpected evaluation trajectory for %s seed %d" % (run_id, seed))
            if validation.get("status") != "complete" or not all(validation.get("checks", {}).values()):
                raise ValueError("DP release validation failed: %s" % validation_path)
            privacy = generator["privacy"]
            if abs(float(privacy["target_epsilon"]) - float(epsilon)) > 1e-12:
                raise ValueError("target epsilon mismatch: %s" % generator_path)
            if privacy["achieved_epsilon_max"] > float(epsilon) + 0.011:
                raise ValueError("privacy target exceeded: %s" % generator_path)
            if privacy.get("dp_randomness") != "fresh_private_entropy_not_derived_from_public_seed":
                raise ValueError("DP randomness provenance is missing: %s" % generator_path)
            trajectory = [row for row in metrics["evaluations"] if row["round"] >= 0]
            record = {
                "run_id": run_id,
                "seed": seed,
                "epsilon_target": float(epsilon),
                "epsilon_achieved": privacy["achieved_epsilon_max"],
                "delta": privacy["delta"],
                "noise_multipliers": privacy["noise_multipliers"],
                "pre_update_accuracy": metrics["pre_update"]["acc"],
                "final_accuracy": metrics["final"]["acc"],
                "final_macro_f1": metrics["final"]["macro_f1"],
                "final_loss": metrics["final"]["eval_loss"],
                "best_accuracy": max(row["acc"] for row in trajectory),
                "mean_trajectory_accuracy": statistics.mean(row["acc"] for row in trajectory),
                "metrics_sha256": hashlib.sha256(metrics_path.read_bytes()).hexdigest(),
                "generator_manifest_sha256": hashlib.sha256(generator_path.read_bytes()).hexdigest(),
            }
            for arm in ("client_syn", "public_syn", "no_cloud"):
                key = (seed, arm, args.rounds - 1)
                if key in reference:
                    record["difference_vs_v3_%s_pp" % arm] = 100.0 * (record["final_accuracy"] - reference[key])
            runs.append(record)
            for row in metrics["evaluations"]:
                curves.append({
                    "epsilon_target": epsilon, "seed": seed, "round": row["round"],
                    "accuracy": row["acc"], "macro_f1": row["macro_f1"],
                    "loss": row["eval_loss"],
                })

    aggregate = {}
    for epsilon in epsilons:
        selected = [run for run in runs if run["epsilon_target"] == float(epsilon)]
        if selected:
            aggregate[epsilon] = {
                name: stats([run[name] for run in selected])
                for name in ("final_accuracy", "final_macro_f1", "final_loss",
                             "best_accuracy", "mean_trajectory_accuracy")
            }
    document = {
        "schema_version": 1,
        "status": "complete" if not missing else "partial",
        "updated_at_utc": datetime.now(timezone.utc).isoformat(),
        "scope": "Record-level DP generator output; downstream FL updates are Non-DP.",
        "hardware": "USTC shared physical GPU 1, NVIDIA A40",
        "runs_completed": len(runs),
        "runs_expected": len(epsilons) * len(seeds),
        "runs": runs,
        "missing": missing,
        "aggregate": aggregate,
        "comparison_limitations": [
            "Three seeds do not establish statistical significance.",
            "DP generation uses float32 and private randomness; v3 comparisons are descriptive.",
            "Shared GPU timing is excluded from method performance claims.",
        ],
    }
    write_atomic(output / "summary.json", json.dumps(document, indent=2, sort_keys=True) + "\n")
    temporary = output / "curves.csv.tmp"
    with temporary.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["epsilon_target", "seed", "round", "accuracy", "macro_f1", "loss"])
        writer.writeheader()
        writer.writerows(curves)
    temporary.replace(output / "curves.csv")
    lines = [
        "# AG News DP v1 — USTC continuation", "",
        "Status: **%s**, %d/%d complete runs." % (document["status"], len(runs), document["runs_expected"]), "",
        "This run restarts the unfinished GitHub DP plan on the USTC server. The released rental instance's DP outputs were not present in GitHub and are not included.", "",
        "Privacy scope: record-level DP of the generated synthetic release. Downstream FL updates remain Non-DP. No end-to-end FL privacy claim is made.", "",
        "Protocol: epsilon targets 8/4/2/1; delta 1e-5; seeds 57/58/59; 120 records per client; two independent client LoRA adapters; five generator epochs; 32 synthetic records per class; 50 FL rounds; evaluation before training and after every round; fixed 512-example development set.", "",
        "All tasks share physical GPU 1. Runtime is not used for speed comparisons.", "",
        "| Epsilon target | Completed seeds | Final accuracy (mean ± sample SD, %) | Final macro F1 |",
        "|---:|---:|---:|---:|",
    ]
    for epsilon in epsilons:
        if epsilon not in aggregate:
            lines.append("| %s | 0/3 | Pending | Pending |" % epsilon)
            continue
        result = aggregate[epsilon]
        value = result["final_accuracy"]
        sd = "—" if value["sample_sd"] is None else "%.2f" % (100 * value["sample_sd"])
        lines.append("| %s | %d/3 | %.2f ± %s | %.4f |" % (
            epsilon, value["n"], 100 * value["mean"], sd, result["final_macro_f1"]["mean"]))
    lines.extend(["", "Per-seed observations and achieved accountant bounds are in `summary.json`. Every available evaluation is in `curves.csv`.", "",
                  "Three seeds do not establish statistical significance. Comparisons with the GitHub Non-DP v3 are descriptive because the DP generator uses float32 and fresh private randomness.", ""])
    write_atomic(output / "REPORT.md", "\n".join(lines))
    print(json.dumps({"status": document["status"], "runs_completed": len(runs), "runs_expected": document["runs_expected"], "output": str(output)}))


if __name__ == "__main__":
    main()
