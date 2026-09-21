#!/usr/bin/env python3
"""Validate and summarize all long-horizon AG News Non-DP v3 runs."""

import argparse
import csv
import json
import statistics
from pathlib import Path


SEEDS = (57, 58, 59)
ARMS = (
    "no_cloud", "client_syn", "public_syn", "same_source_real_matched",
    "real_matched", "shuffled_label",
)
CHECKPOINTS = (0, 4, 9, 24, 49)
THRESHOLDS = (0.35, 0.40, 0.45, 0.50)


def load(path):
    with Path(path).open(encoding="utf-8") as handle:
        return json.load(handle)


def mean(values):
    return statistics.mean(values)


def sample_sd(values):
    return statistics.stdev(values) if len(values) > 1 else 0.0


def first_round_at(evaluations, threshold):
    for item in evaluations:
        if item["round"] >= 0 and item["acc"] >= threshold:
            return item["round"]
    return None


def arm_summary(metrics):
    evaluations = metrics["evaluations"]
    by_round = {item["round"]: item for item in evaluations}
    expected = [-1] + list(range(50))
    if [item["round"] for item in evaluations] != expected:
        raise ValueError("evaluation trajectory is not -1,0,...,49")
    rounds = [by_round[index] for index in range(50)]
    return {
        "final": by_round[49],
        "mean_accuracy_rounds_0_49": mean(item["acc"] for item in rounds),
        "mean_macro_f1_rounds_0_49": mean(item["macro_f1"] for item in rounds),
        "mean_loss_rounds_0_49": mean(item["eval_loss"] for item in rounds),
        "checkpoints": {str(index): by_round[index] for index in CHECKPOINTS},
        "first_round_at_accuracy": {
            str(int(threshold * 100)): first_round_at(evaluations, threshold)
            for threshold in THRESHOLDS
        },
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    curves = []
    runs = {}
    public_validation = {}
    diagnostics = {}
    for seed in SEEDS:
        root = args.results_root / ("matpool_agnews_nondp_v3_seed%d" % seed)
        seed_dir = root / ("seed_%d" % seed)
        cross = load(seed_dir / "cross_arm_validation.json")
        if cross["status"] != "complete" or cross["errors"]:
            raise ValueError("cross-arm validation failed for seed %d" % seed)
        if cross["cross_arm"]["declared_arm_count"] != 6:
            raise ValueError("seed %d does not declare six arms" % seed)
        if cross["cross_arm"]["client_objective_queries_per_arm"] != 3000:
            raise ValueError("seed %d has unexpected query budget" % seed)
        public_validation[str(seed)] = load(
            seed_dir / "public_synthetic/validation_manifest.json"
        )
        if public_validation[str(seed)]["status"] != "complete":
            raise ValueError("public validation failed for seed %d" % seed)
        diagnostics[str(seed)] = load(root / "gradient_subspace_diagnostic.json")
        if diagnostics[str(seed)]["status"] != "complete":
            raise ValueError("gradient diagnostic failed for seed %d" % seed)

        runs[str(seed)] = {}
        for arm in ARMS:
            metrics = load(seed_dir / "arms" / arm / "metrics.json")
            if metrics["status"] != "complete" or not all(metrics["checks"].values()):
                raise ValueError("incomplete arm seed=%d arm=%s" % (seed, arm))
            runs[str(seed)][arm] = arm_summary(metrics)
            for item in metrics["evaluations"]:
                curves.append({
                    "seed": seed,
                    "arm": arm,
                    "round": item["round"],
                    "accuracy": item["acc"],
                    "macro_f1": item["macro_f1"],
                    "loss": item["eval_loss"],
                })

    paired = {}
    comparisons = {
        "client_minus_no_cloud": ("client_syn", "no_cloud"),
        "client_minus_public": ("client_syn", "public_syn"),
        "client_minus_shuffled": ("client_syn", "shuffled_label"),
        "public_minus_no_cloud": ("public_syn", "no_cloud"),
    }
    curve_lookup = {
        (row["seed"], row["arm"], row["round"]): row for row in curves
    }
    for name, (left, right) in comparisons.items():
        paired[name] = {}
        for seed in SEEDS:
            deltas = {
                round_index: (
                    curve_lookup[(seed, left, round_index)]["accuracy"]
                    - curve_lookup[(seed, right, round_index)]["accuracy"]
                )
                for round_index in range(50)
            }
            paired[name][str(seed)] = {
                "final_accuracy_difference_pp": 100.0 * deltas[49],
                "mean_accuracy_difference_pp_rounds_0_49": 100.0 * mean(deltas.values()),
                "early_mean_difference_pp_rounds_0_9": 100.0 * mean(
                    deltas[index] for index in range(10)
                ),
                "late_mean_difference_pp_rounds_40_49": 100.0 * mean(
                    deltas[index] for index in range(40, 50)
                ),
                "positive_round_count": sum(value > 0.0 for value in deltas.values()),
                "checkpoint_difference_pp": {
                    str(index): 100.0 * deltas[index] for index in CHECKPOINTS
                },
            }

    aggregate = {}
    for arm in ARMS:
        final_values = [runs[str(seed)][arm]["final"]["acc"] * 100 for seed in SEEDS]
        auc_values = [
            runs[str(seed)][arm]["mean_accuracy_rounds_0_49"] * 100
            for seed in SEEDS
        ]
        aggregate[arm] = {
            "final_accuracy_pct_mean": mean(final_values),
            "final_accuracy_pct_sample_sd": sample_sd(final_values),
            "mean_trajectory_accuracy_pct_mean": mean(auc_values),
            "mean_trajectory_accuracy_pct_sample_sd": sample_sd(auc_values),
        }
    paired_aggregate = {}
    for name in comparisons:
        entries = [paired[name][str(seed)] for seed in SEEDS]
        paired_aggregate[name] = {}
        for field in (
            "final_accuracy_difference_pp",
            "mean_accuracy_difference_pp_rounds_0_49",
            "early_mean_difference_pp_rounds_0_9",
            "late_mean_difference_pp_rounds_40_49",
        ):
            values = [item[field] for item in entries]
            paired_aggregate[name][field + "_mean"] = mean(values)
            paired_aggregate[name][field + "_sample_sd"] = sample_sd(values)
            paired_aggregate[name][field + "_positive_seed_count"] = sum(
                value > 0.0 for value in values
            )

    result = {
        "schema_version": 1,
        "status": "complete",
        "scope": (
            "Three seeds, fixed source/development clients, 50 rounds, Non-DP. "
            "This is not a statistical-significance, final-convergence or privacy result."
        ),
        "seeds": list(SEEDS),
        "arms": list(ARMS),
        "runs": runs,
        "paired": paired,
        "aggregate": aggregate,
        "paired_aggregate": paired_aggregate,
        "public_generator_validation": public_validation,
        "gradient_subspace_diagnostics": diagnostics,
    }
    (args.output_dir / "summary.json").write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n"
    )
    with (args.output_dir / "curves.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(
            handle, fieldnames=("seed", "arm", "round", "accuracy", "macro_f1", "loss")
        )
        writer.writeheader()
        writer.writerows(curves)

    lines = [
        "# AG News Non-DP v3 metrics", "", result["scope"], "",
        "| Seed | No guidance | Client synthetic | Public synthetic | Same-source real | Held-out real | Shuffled |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for seed in SEEDS:
        values = [runs[str(seed)][arm]["final"]["acc"] * 100 for arm in ARMS]
        lines.append("| %d | %s |" % (
            seed, " | ".join("%.4f%%" % value for value in values)
        ))
    lines.append("| Mean | %s |" % " | ".join(
        "%.4f%%" % aggregate[arm]["final_accuracy_pct_mean"] for arm in ARMS
    ))
    lines += ["", "| Comparison | Final mean (pp) | Rounds 0–49 mean (pp) | Early 0–9 (pp) | Late 40–49 (pp) |",
              "|---|---:|---:|---:|---:|"]
    for name in comparisons:
        item = paired_aggregate[name]
        lines.append(
            "| %s | %+.4f | %+.4f | %+.4f | %+.4f |" % (
                name,
                item["final_accuracy_difference_pp_mean"],
                item["mean_accuracy_difference_pp_rounds_0_49_mean"],
                item["early_mean_difference_pp_rounds_0_9_mean"],
                item["late_mean_difference_pp_rounds_40_49_mean"],
            )
        )
    lines += ["", "All 18 arms, 51 evaluations per arm and 3000 client objective queries per arm were validated.", ""]
    (args.output_dir / "METRICS.md").write_text("\n".join(lines))
    print(json.dumps(paired_aggregate, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
