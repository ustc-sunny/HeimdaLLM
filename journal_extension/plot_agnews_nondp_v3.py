#!/usr/bin/env python3
"""Create publication-style long-horizon accuracy figures from curves.csv."""

import argparse
import csv
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


LABELS = {
    "no_cloud": "No guidance",
    "client_syn": "Client-LoRA synthetic",
    "public_syn": "Public synthetic",
    "same_source_real_matched": "Same-source real",
    "real_matched": "Held-out real",
    "shuffled_label": "Shuffled-label synthetic",
}
COLORS = {
    "no_cloud": "#4d4d4d",
    "client_syn": "#0072B2",
    "public_syn": "#E69F00",
    "same_source_real_matched": "#009E73",
    "real_matched": "#56B4E9",
    "shuffled_label": "#CC79A7",
}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--curves", type=Path, required=True)
    parser.add_argument("--output-prefix", type=Path, required=True)
    args = parser.parse_args()
    data = defaultdict(dict)
    with args.curves.open(newline="") as handle:
        for row in csv.DictReader(handle):
            round_index = int(row["round"])
            if round_index >= 0:
                data[row["arm"]].setdefault(int(row["seed"]), {})[round_index] = {
                    "accuracy": 100.0 * float(row["accuracy"]),
                    "macro_f1": 100.0 * float(row["macro_f1"]),
                    "loss": float(row["loss"]),
                }
    rounds = np.arange(50)
    fig, axes = plt.subplots(2, 2, figsize=(12.0, 8.4), constrained_layout=True)
    metric_panels = (
        ("accuracy", axes[0, 0], "AG News development accuracy", "Accuracy (%)"),
        ("macro_f1", axes[0, 1], "AG News development macro-F1", "Macro-F1 (%)"),
        ("loss", axes[1, 0], "AG News development loss", "Cross-entropy loss"),
    )
    for metric, axis, title, ylabel in metric_panels:
        for arm in LABELS:
            matrix = np.asarray([
                [data[arm][seed][int(index)][metric] for index in rounds]
                for seed in sorted(data[arm])
            ])
            average = matrix.mean(axis=0)
            axis.plot(
                rounds, average, label=LABELS[arm], color=COLORS[arm], linewidth=2
            )
            axis.fill_between(
                rounds, matrix.min(axis=0), matrix.max(axis=0),
                color=COLORS[arm], alpha=0.10, linewidth=0,
            )
        axis.set_title(title)
        axis.set_xlabel("FL round")
        axis.set_ylabel(ylabel)
        axis.grid(alpha=0.25)
    axes[0, 0].legend(fontsize=8, ncol=2)

    comparisons = (
        ("client_syn", "no_cloud", "Client synthetic − no guidance", "#0072B2"),
        ("client_syn", "public_syn", "Client synthetic − public synthetic", "#D55E00"),
        ("client_syn", "shuffled_label", "Client synthetic − shuffled", "#009E73"),
    )
    for left, right, label, color in comparisons:
        matrix = np.asarray([
            [
                data[left][seed][int(index)]["accuracy"]
                - data[right][seed][int(index)]["accuracy"]
                for index in rounds
            ]
            for seed in sorted(data[left])
        ])
        axes[1, 1].plot(
            rounds, matrix.mean(axis=0), label=label, color=color, linewidth=2
        )
        axes[1, 1].fill_between(
            rounds, matrix.min(axis=0), matrix.max(axis=0),
            color=color, alpha=0.12, linewidth=0,
        )
    axes[1, 1].axhline(0.0, color="black", linewidth=1, linestyle="--")
    axes[1, 1].set_title("Paired accuracy difference")
    axes[1, 1].set_xlabel("FL round")
    axes[1, 1].set_ylabel("Difference (percentage points)")
    axes[1, 1].grid(alpha=0.25)
    axes[1, 1].legend(fontsize=8)

    args.output_prefix.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(str(args.output_prefix) + ".pdf", bbox_inches="tight")
    fig.savefig(str(args.output_prefix) + ".png", dpi=220, bbox_inches="tight")
    print(str(args.output_prefix) + ".pdf")


if __name__ == "__main__":
    main()
