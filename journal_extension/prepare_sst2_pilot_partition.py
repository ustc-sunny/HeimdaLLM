#!/usr/bin/env python3
"""Build a fixed-client SST-2 partition for the Non-DP feasibility pilot.

The legacy data manager exposes ``int(n_clients * 0.8)`` partitions to FL
workers.  This writer therefore adds the minimum number of evaluation-only
partition slots needed to expose exactly the requested logical clients.

By default, evaluation examples come from disjoint reserved clients' training
partitions and are balanced by label.  The official SST-2 test split is
available only through explicit ``final-test`` mode after configuration
selection has been locked.
"""

from __future__ import print_function

import argparse
from collections import Counter
import json
import os
from pathlib import Path
import random

import h5py
import numpy as np


def parse_client_ids(raw_value):
    values = [value.strip() for value in raw_value.split(",")]
    if not values or any(not value for value in values):
        raise ValueError("--client-ids must be a non-empty comma-separated list")
    if len(set(values)) != len(values):
        raise ValueError("--client-ids contains duplicate IDs")
    return values


def decode_json_dataset(value):
    if isinstance(value, bytes):
        value = value.decode("utf-8")
    return json.loads(value)


def decode_label(value):
    if isinstance(value, bytes):
        return value.decode("utf-8")
    if hasattr(value, "decode"):
        return value.decode("utf-8")
    return str(value)


def output_partition_count(logical_client_count):
    """Return the smallest n for which int(0.8 * n) exposes exactly K."""
    candidate = logical_client_count
    while int(candidate * 0.8) != logical_client_count:
        candidate += 1
    return candidate


def ordered_unique(values):
    seen = set()
    result = []
    for value in values:
        value = int(value)
        if value in seen:
            raise ValueError("index %d occurs more than once" % value)
        seen.add(value)
        result.append(value)
    return result


def validate_balanced_clients(data_file, train_by_client, label_vocab):
    public_labels = set(str(label) for label in label_vocab.keys())
    histograms = {}
    for output_id, indices in enumerate(train_by_client):
        histogram = Counter(
            decode_label(data_file["Y"][str(int(index))][()]) for index in indices
        )
        if set(histogram.keys()) != public_labels:
            raise ValueError(
                "source client %d does not contain every public label: %s"
                % (output_id, dict(histogram))
            )
        if len(set(histogram.values())) != 1:
            raise ValueError(
                "source client %d is not label-balanced: %s"
                % (output_id, dict(histogram))
            )
        histograms[str(output_id)] = dict(sorted(histogram.items()))
    return histograms


def ordered_labels(label_vocab):
    def sort_key(label):
        try:
            return (0, int(label_vocab[label]), str(label))
        except (TypeError, ValueError):
            return (1, str(label_vocab[label]), str(label))
    return sorted((str(label) for label in label_vocab), key=sort_key)


def label_histogram(data_file, indices):
    return dict(sorted(Counter(
        decode_label(data_file["Y"][str(int(index))][()]) for index in indices
    ).items()))


def select_balanced_dev(data_file, source_data, dev_client_ids, train_universe,
                        selected_train, label_vocab, per_label, seed):
    dev_pool = []
    for client_id in dev_client_ids:
        if client_id not in source_data:
            raise ValueError("dev source client is missing: %s" % client_id)
        indices = [int(index) for index in source_data[client_id]["train"][()]]
        if not indices:
            raise ValueError("dev source client %s has no train records" % client_id)
        if not set(indices).issubset(train_universe):
            raise ValueError("dev source client %s contains non-train indices" % client_id)
        dev_pool.extend(indices)
    ordered_unique(dev_pool)
    overlap = set(dev_pool).intersection(selected_train)
    if overlap:
        raise ValueError("FL/generator and dev indices overlap (%d records)" % len(overlap))

    labels = ordered_labels(label_vocab)
    by_label = dict((label, []) for label in labels)
    for index in dev_pool:
        label = decode_label(data_file["Y"][str(index)][()])
        if label not in by_label:
            raise ValueError("dev record has label outside label_vocab: %r" % label)
        by_label[label].append(index)
    available = dict((label, len(by_label[label])) for label in labels)
    target = min(available.values()) if per_label is None else per_label
    if target <= 0:
        raise ValueError("balanced dev quota must be positive")
    insufficient = dict(
        (label, count) for label, count in available.items() if count < target
    )
    if insufficient:
        raise ValueError(
            "reserved dev clients cannot supply %d examples per label: %s"
            % (target, insufficient)
        )
    selected = []
    for position, label in enumerate(labels):
        values = list(by_label[label])
        random.Random(seed + position).shuffle(values)
        selected.extend(values[:target])
    random.Random(seed + 1009).shuffle(selected)
    ordered_unique(selected)
    return selected, available, target


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-file", required=True, help="real SST-2 data H5")
    parser.add_argument("--partition-file", required=True, help="source partition H5")
    parser.add_argument("--partition-method", default="uniform")
    parser.add_argument("--client-ids", required=True)
    parser.add_argument("--output", required=True, help="new partition H5")
    parser.add_argument("--output-method", required=True)
    parser.add_argument("--evaluation-mode", choices=["dev", "final-test"],
                        default="dev")
    parser.add_argument("--dev-client-ids", default="90,91,92,93,94,95,96,97,98")
    parser.add_argument("--dev-per-label", type=int, default=256)
    parser.add_argument("--dev-seed", type=int, default=57)
    parser.add_argument("--require-balanced", action="store_true")
    parser.add_argument("--require-equal-train-size", action="store_true")
    args = parser.parse_args()

    client_ids = parse_client_ids(args.client_ids)
    logical_client_count = len(client_ids)
    total_partitions = output_partition_count(logical_client_count)
    output_path = Path(args.output)
    if output_path.exists():
        raise FileExistsError("refusing to overwrite existing output: %s" % output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with h5py.File(args.data_file, "r", swmr=True) as data_file, h5py.File(
        args.partition_file, "r", swmr=True
    ) as source_partition:
        if args.partition_method not in source_partition:
            raise ValueError("partition method is absent: %s" % args.partition_method)
        source_data = source_partition[args.partition_method]["partition_data"]
        missing = [client_id for client_id in client_ids if client_id not in source_data]
        if missing:
            raise ValueError("source partition is missing clients: %s" % ",".join(missing))

        attributes = decode_json_dataset(data_file["attributes"][()])
        train_universe = set(int(index) for index in attributes["train_index_list"])
        official_test_indices = ordered_unique(attributes["test_index_list"])
        test_universe = set(official_test_indices)
        if train_universe.intersection(test_universe):
            raise ValueError("real train and test attributes overlap")

        train_by_client = []
        all_selected_train = []
        for client_id in client_ids:
            indices = [
                int(index)
                for index in source_data[client_id]["train"][()]
            ]
            if not indices:
                raise ValueError("source client %s has no training examples" % client_id)
            if not set(indices).issubset(train_universe):
                raise ValueError("source client %s contains non-train indices" % client_id)
            train_by_client.append(indices)
            all_selected_train.extend(indices)
        ordered_unique(all_selected_train)
        selected_train_set = set(all_selected_train)

        train_sizes = [len(indices) for indices in train_by_client]
        if args.require_equal_train_size and len(set(train_sizes)) != 1:
            raise ValueError("selected clients have unequal train sizes: %s" % train_sizes)

        if args.require_balanced:
            label_histograms = validate_balanced_clients(
                data_file, train_by_client, attributes["label_vocab"]
            )
        else:
            label_histograms = {}

        if args.evaluation_mode == "dev":
            dev_client_ids = parse_client_ids(args.dev_client_ids)
            if set(dev_client_ids).intersection(client_ids):
                raise ValueError("FL/generator client IDs and dev client IDs overlap")
            evaluation_indices, dev_available, dev_per_label = select_balanced_dev(
                data_file,
                source_data,
                dev_client_ids,
                train_universe,
                selected_train_set,
                attributes["label_vocab"],
                args.dev_per_label,
                args.dev_seed,
            )
            evaluation_source = {
                "mode": "dev",
                "split": "reserved_client_train",
                "source_client_ids": dev_client_ids,
                "available_label_counts": dev_available,
                "selected_per_label": dev_per_label,
                "selection_seed": args.dev_seed,
            }
        else:
            evaluation_indices = official_test_indices
            evaluation_source = {
                "mode": "final-test",
                "split": "official_test",
                "source_client_ids": [],
            }
        evaluation_label_counts = label_histogram(data_file, evaluation_indices)

    evaluation_shards = np.array_split(
        np.asarray(evaluation_indices, dtype=np.int64), total_partitions
    )
    temporary_path = Path(str(output_path) + ".tmp")
    if temporary_path.exists():
        raise FileExistsError("temporary output already exists: %s" % temporary_path)
    try:
        with h5py.File(str(temporary_path), "w") as output_file:
            method = output_file.create_group(args.output_method)
            method.create_dataset("n_clients", data=total_partitions)
            method.attrs["logical_client_count"] = logical_client_count
            method.attrs["source_partition_method"] = args.partition_method
            method.attrs["source_client_ids_json"] = json.dumps(client_ids)
            method.attrs["evaluation_source_json"] = json.dumps(
                evaluation_source, sort_keys=True
            )
            partition_data = method.create_group("partition_data")
            for partition_id in range(total_partitions):
                client = partition_data.create_group(str(partition_id))
                if partition_id < logical_client_count:
                    train_indices = np.asarray(
                        train_by_client[partition_id], dtype=np.int64
                    )
                else:
                    train_indices = np.asarray([], dtype=np.int64)
                client.create_dataset("train", data=train_indices)
                client.create_dataset(
                    "test",
                    data=np.asarray(evaluation_shards[partition_id], dtype=np.int64),
                )
        os.replace(str(temporary_path), str(output_path))
    except Exception:
        if temporary_path.exists():
            temporary_path.unlink()
        raise

    report = {
        "output": str(output_path),
        "output_method": args.output_method,
        "source_client_ids": client_ids,
        "client_id_remap": {
            client_id: output_id for output_id, client_id in enumerate(client_ids)
        },
        "logical_client_count": logical_client_count,
        "reported_n_clients": total_partitions,
        "legacy_visible_clients": int(total_partitions * 0.8),
        "train_counts": train_sizes,
        "label_histograms": label_histograms,
        "evaluation": evaluation_source,
        "evaluation_count": len(evaluation_indices),
        "evaluation_label_counts": evaluation_label_counts,
    }
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
