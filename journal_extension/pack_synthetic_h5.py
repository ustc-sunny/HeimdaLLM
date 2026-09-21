#!/usr/bin/env python3
"""Pack deduplicated synthetic text into an independent FedNLP HDF5 pair.

The output contains synthetic training records only.  Its test split and every
client's test partition are empty, preventing synthetic data from becoming the
evaluation set.  ``label_vocab`` is copied verbatim from the real source H5 so
the cloud and real-data server use identical label ids.
"""

import argparse
import hashlib
import json
import os
import re
import time
import unicodedata
from collections import Counter, defaultdict
from pathlib import Path


def sha256_file(path, chunk_size=1024 * 1024):
    digest = hashlib.sha256()
    with open(str(path), "rb") as handle:
        while True:
            chunk = handle.read(chunk_size)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def json_from_h5_scalar(value):
    if isinstance(value, bytes):
        value = value.decode("utf-8")
    elif not isinstance(value, str):
        value = value.tobytes().decode("utf-8")
    return json.loads(value)


def normalize_text(text):
    text = unicodedata.normalize("NFKC", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text.casefold()


def read_and_deduplicate(path, label_vocab):
    rows = []
    seen = {}
    duplicate_count = 0
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                record = json.loads(line)
            except ValueError as error:
                raise ValueError("invalid JSON at line %d: %s" % (line_number, error))
            text = record.get("text")
            label = record.get("label")
            if not isinstance(text, str) or not text.strip():
                raise ValueError("line %d has empty/non-string text" % line_number)
            label = str(label)
            if label not in label_vocab:
                raise ValueError(
                    "line %d label %r is absent from source label_vocab"
                    % (line_number, label)
                )
            key = normalize_text(text)
            if not key:
                raise ValueError("line %d normalizes to empty text" % line_number)
            previous_label = seen.get(key)
            if previous_label is not None:
                if previous_label != label:
                    raise ValueError(
                        "the same normalized text has conflicting labels %r and %r"
                        % (previous_label, label)
                    )
                duplicate_count += 1
                continue
            seen[key] = label
            rows.append({"text": re.sub(r"\s+", " ", text).strip(), "label": label})
    if not rows:
        raise ValueError("synthetic JSONL contains no usable rows")
    return rows, duplicate_count


def ordered_labels(label_vocab):
    def key(label):
        value = label_vocab[label]
        try:
            return (0, int(value), str(label))
        except (TypeError, ValueError):
            return (1, str(value), str(label))
    return sorted((str(label) for label in label_vocab), key=key)


def assign_stratified_partitions(rows, labels, cloud_clients):
    by_label = defaultdict(list)
    for index, row in enumerate(rows):
        by_label[row["label"]].append(index)
    partitions = [[] for _ in range(cloud_clients)]
    offset = 0
    for label in labels:
        for position, index in enumerate(by_label.get(label, [])):
            partitions[(offset + position) % cloud_clients].append(index)
        offset = (offset + len(by_label.get(label, []))) % cloud_clients
    for indices in partitions:
        indices.sort()
    flattened = [index for indices in partitions for index in indices]
    if sorted(flattened) != list(range(len(rows))):
        raise AssertionError("synthetic train partition is not a one-to-one index cover")
    return partitions


def atomic_write_json(path, payload):
    temporary = path.with_name(path.name + ".tmp")
    if temporary.exists():
        temporary.unlink()
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=True, indent=2, sort_keys=True)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(str(temporary), str(path))


def write_data_h5(h5py, path, rows, attributes):
    temporary = path.with_name(path.name + ".tmp")
    if temporary.exists():
        temporary.unlink()
    try:
        with h5py.File(str(temporary), "w") as handle:
            x_group = handle.create_group("X")
            y_group = handle.create_group("Y")
            for index, row in enumerate(rows):
                x_group.create_dataset(str(index), data=row["text"].encode("utf-8"))
                y_group.create_dataset(str(index), data=row["label"].encode("utf-8"))
            handle.create_dataset(
                "attributes", data=json.dumps(attributes, ensure_ascii=True).encode("utf-8")
            )
            handle.flush()
        os.replace(str(temporary), str(path))
    except Exception:
        if temporary.exists():
            temporary.unlink()
        raise


def write_partition_h5(h5py, np, path, method, partitions):
    temporary = path.with_name(path.name + ".tmp")
    if temporary.exists():
        temporary.unlink()
    try:
        with h5py.File(str(temporary), "w") as handle:
            method_group = handle.create_group(method)
            method_group.create_dataset("n_clients", data=len(partitions))
            partition_group = method_group.create_group("partition_data")
            for client_id, indices in enumerate(partitions):
                client_group = partition_group.create_group(str(client_id))
                client_group.create_dataset(
                    "train", data=np.asarray(indices, dtype=np.int64)
                )
                client_group.create_dataset(
                    "test", data=np.asarray([], dtype=np.int64)
                )
            handle.flush()
        os.replace(str(temporary), str(path))
    except Exception:
        if temporary.exists():
            temporary.unlink()
        raise


def verify_outputs(h5py, data_path, partition_path, method, expected_rows,
                   expected_label_vocab):
    with h5py.File(str(data_path), "r", swmr=True) as data_handle:
        attributes = json_from_h5_scalar(data_handle["attributes"][()])
        if attributes.get("label_vocab") != expected_label_vocab:
            raise AssertionError("packed label_vocab differs from source label_vocab")
        train_indices = [int(value) for value in attributes.get("train_index_list", [])]
        test_indices = [int(value) for value in attributes.get("test_index_list", [])]
        if train_indices != list(range(expected_rows)):
            raise AssertionError("packed train_index_list is invalid")
        if test_indices:
            raise AssertionError("packed test_index_list must be empty")
        if len(data_handle["X"]) != expected_rows or len(data_handle["Y"]) != expected_rows:
            raise AssertionError("packed X/Y record count is invalid")
    with h5py.File(str(partition_path), "r", swmr=True) as partition_handle:
        method_group = partition_handle[method]
        n_clients = int(method_group["n_clients"][()])
        all_train = []
        all_test = []
        for client_id in range(n_clients):
            group = method_group["partition_data"][str(client_id)]
            all_train.extend(int(value) for value in group["train"][()])
            all_test.extend(int(value) for value in group["test"][()])
        if sorted(all_train) != list(range(expected_rows)) or len(all_train) != len(set(all_train)):
            raise AssertionError("packed train partitions duplicate or omit indices")
        if all_test:
            raise AssertionError("packed client test partitions must be empty")


def main():
    parser = argparse.ArgumentParser(
        description="Pack synthetic JSONL into train-only FedNLP data/partition H5 files."
    )
    parser.add_argument("--jsonl", required=True)
    parser.add_argument("--source-data-file", required=True,
                        help="Real task H5 used only to copy label_vocab/task metadata")
    parser.add_argument("--data-out", required=True)
    parser.add_argument("--partition-out", required=True)
    parser.add_argument("--partition-method", default="uniform")
    parser.add_argument("--cloud-clients", type=int, default=10)
    parser.add_argument("--manifest-out", default=None)
    parser.add_argument("--require-equal-label-counts", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    started = time.time()
    jsonl_path = Path(args.jsonl).expanduser().resolve()
    source_path = Path(args.source_data_file).expanduser().resolve()
    data_path = Path(args.data_out).expanduser().resolve()
    partition_path = Path(args.partition_out).expanduser().resolve()
    manifest_path = (Path(args.manifest_out).expanduser().resolve()
                     if args.manifest_out else Path(str(data_path) + ".manifest.json"))
    if not jsonl_path.is_file():
        raise FileNotFoundError(str(jsonl_path))
    if not source_path.is_file():
        raise FileNotFoundError(str(source_path))
    if args.cloud_clients <= 0:
        raise ValueError("--cloud-clients must be positive")
    if not args.partition_method or "/" in args.partition_method:
        raise ValueError("--partition-method must be a non-empty H5 group name")
    if data_path == partition_path:
        raise ValueError("--data-out and --partition-out must be different paths")

    # Compatible with the Python 3.7 legacy FedNLP environment.
    import h5py
    import numpy as np

    with h5py.File(str(source_path), "r", swmr=True) as source_handle:
        if "attributes" not in source_handle:
            raise ValueError("source data H5 has no attributes dataset")
        source_attributes = json_from_h5_scalar(source_handle["attributes"][()])
    label_vocab = source_attributes.get("label_vocab")
    if not isinstance(label_vocab, dict) or not label_vocab:
        raise ValueError("source attributes.label_vocab must be a non-empty object")
    rows, duplicates_dropped = read_and_deduplicate(jsonl_path, label_vocab)
    labels = ordered_labels(label_vocab)
    label_counts = dict((label, 0) for label in labels)
    label_counts.update(Counter(row["label"] for row in rows))
    if args.require_equal_label_counts and len(set(label_counts.values())) != 1:
        raise ValueError("synthetic label counts are not equal: %s" % label_counts)
    partitions = assign_stratified_partitions(rows, labels, args.cloud_clients)
    attributes = {
        "index_list": list(range(len(rows))),
        "train_index_list": list(range(len(rows))),
        "test_index_list": [],
        "label_vocab": label_vocab,
        "num_labels": source_attributes.get("num_labels", len(label_vocab)),
        "task_type": source_attributes.get("task_type", "text_classification"),
    }
    plan = {
        "schema_version": 1,
        "status": "dry_run" if args.dry_run else "complete",
        "input_jsonl": str(jsonl_path),
        "input_jsonl_sha256": sha256_file(jsonl_path),
        "source_data_file": str(source_path),
        "source_data_sha256": sha256_file(source_path),
        "label_vocab": label_vocab,
        "records_before_deduplication": len(rows) + duplicates_dropped,
        "duplicates_dropped": duplicates_dropped,
        "records": len(rows),
        "label_counts": label_counts,
        "test_records": 0,
        "partition_method": args.partition_method,
        "cloud_clients": args.cloud_clients,
        "client_train_counts": [len(indices) for indices in partitions],
        "all_train_indices_unique": True,
        "data_out": str(data_path),
        "partition_out": str(partition_path),
    }
    if args.dry_run:
        print(json.dumps(plan, ensure_ascii=True, indent=2, sort_keys=True))
        return

    for path in (data_path, partition_path, manifest_path):
        if path.exists():
            raise FileExistsError("%s already exists; choose new output paths" % path)
        path.parent.mkdir(parents=True, exist_ok=True)
    write_data_h5(h5py, data_path, rows, attributes)
    write_partition_h5(
        h5py, np, partition_path, args.partition_method, partitions
    )
    verify_outputs(
        h5py, data_path, partition_path, args.partition_method,
        len(rows), label_vocab,
    )
    plan.update({
        "data_sha256": sha256_file(data_path),
        "partition_sha256": sha256_file(partition_path),
        "manifest": str(manifest_path),
        "elapsed_seconds": round(time.time() - started, 6),
    })
    atomic_write_json(manifest_path, plan)
    print(json.dumps({
        "status": "complete",
        "data_file": str(data_path),
        "partition_file": str(partition_path),
        "manifest": str(manifest_path),
        "records": len(rows),
        "test_records": 0,
        "label_counts": label_counts,
    }, ensure_ascii=True, sort_keys=True))


if __name__ == "__main__":
    main()
