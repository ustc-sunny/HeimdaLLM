#!/usr/bin/env python3
"""Build a train-only real-data oracle matched to a synthetic JSONL budget.

The output follows the reference label sequence exactly, but every text comes
from the selected source clients' real train partitions.  Source records are
allocated as evenly as availability permits and are never duplicated.
"""

from __future__ import print_function

import argparse
from collections import Counter, defaultdict, deque
import hashlib
import json
import os
from pathlib import Path
import random
import stat


def sha256_file(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_values(values):
    digest = hashlib.sha256()
    for value in values:
        encoded = json.dumps(
            value, ensure_ascii=True, sort_keys=True, separators=(",", ":")
        ).encode("utf-8")
        digest.update(len(encoded).to_bytes(8, byteorder="big"))
        digest.update(encoded)
    return digest.hexdigest()


def decode_scalar(value):
    if isinstance(value, bytes):
        return value.decode("utf-8")
    if hasattr(value, "tobytes") and not isinstance(value, str):
        raw = value.tobytes()
        try:
            return raw.decode("utf-8")
        except UnicodeDecodeError:
            pass
    return str(value)


def json_from_h5_scalar(value):
    if isinstance(value, bytes):
        value = value.decode("utf-8")
    elif not isinstance(value, str):
        value = value.tobytes().decode("utf-8")
    return json.loads(value)


def parse_client_ids(values):
    client_ids = []
    for value in values or []:
        for item in str(value).split(","):
            item = item.strip()
            if item:
                client_ids.append(int(item))
    if not client_ids:
        raise ValueError("--client-ids must contain at least one client id")
    if len(client_ids) != len(set(client_ids)):
        raise ValueError("--client-ids contains duplicates")
    if any(client_id < 0 for client_id in client_ids):
        raise ValueError("client ids must be non-negative")
    return client_ids


def load_reference(path):
    rows = []
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            row = json.loads(line)
            text = row.get("text")
            if not isinstance(text, str) or not text.strip():
                raise ValueError("invalid reference text at line %d" % line_number)
            label = str(row.get("label"))
            rows.append({"text": text, "label": label})
    if not rows:
        raise ValueError("reference JSONL is empty")
    return rows


def atomic_write_jsonl(path, rows):
    temporary = Path(str(path) + ".tmp")
    if temporary.exists():
        temporary.unlink()
    with temporary.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=True, sort_keys=True) + "\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.chmod(str(temporary), stat.S_IRUSR | stat.S_IWUSR)
    os.replace(str(temporary), str(path))


def atomic_write_json(path, payload):
    temporary = Path(str(path) + ".tmp")
    if temporary.exists():
        temporary.unlink()
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=True, indent=2, sort_keys=True)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.chmod(str(temporary), stat.S_IRUSR | stat.S_IWUSR)
    os.replace(str(temporary), str(path))


def choose_label_records(label, target, client_ids, records_by_client, seed):
    available = sum(len(records_by_client[client_id][label]) for client_id in client_ids)
    if available < target:
        raise ValueError(
            "selected source clients have only %d train records for label %r; need %d"
            % (available, label, target)
        )

    order = list(client_ids)
    random.Random("%d:allocation:%s" % (seed, label)).shuffle(order)
    buckets = {}
    for client_id in client_ids:
        records = list(records_by_client[client_id][label])
        random.Random("%d:records:%d:%s" % (seed, client_id, label)).shuffle(records)
        buckets[client_id] = deque(records)

    selected = []
    while len(selected) < target:
        progressed = False
        for client_id in order:
            if len(selected) >= target:
                break
            if buckets[client_id]:
                selected.append(buckets[client_id].popleft())
                progressed = True
        if not progressed:
            raise AssertionError("quota allocation stopped before reaching its target")
    return selected


def main():
    parser = argparse.ArgumentParser(
        description="Create a same-source real oracle matched to a reference JSONL."
    )
    parser.add_argument("--data-file", required=True)
    parser.add_argument("--partition-file", required=True)
    parser.add_argument("--partition-method", required=True)
    parser.add_argument("--client-ids", nargs="+", required=True)
    parser.add_argument("--reference-jsonl", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--manifest-out", required=True)
    parser.add_argument("--seed", type=int, required=True)
    args = parser.parse_args()

    data_path = Path(args.data_file).expanduser().resolve()
    partition_path = Path(args.partition_file).expanduser().resolve()
    reference_path = Path(args.reference_jsonl).expanduser().resolve()
    output_path = Path(args.output).expanduser().resolve()
    manifest_path = Path(args.manifest_out).expanduser().resolve()
    client_ids = parse_client_ids(args.client_ids)
    for path in (data_path, partition_path, reference_path):
        if not path.is_file():
            raise FileNotFoundError(str(path))
    if output_path.exists() or manifest_path.exists():
        raise FileExistsError("refusing to overwrite matched-real artifacts")
    if output_path == manifest_path:
        raise ValueError("--output and --manifest-out must differ")

    reference_rows = load_reference(reference_path)
    reference_labels = [row["label"] for row in reference_rows]
    target_counts = Counter(reference_labels)

    # Imported lazily so --help and py_compile work outside the legacy H5 env.
    import h5py

    records_by_client = dict((client_id, defaultdict(list)) for client_id in client_ids)
    source_counts = {}
    index_owner = {}
    with h5py.File(str(data_path), "r", swmr=True) as data_handle, h5py.File(
        str(partition_path), "r", swmr=True
    ) as partition_handle:
        if not all(name in data_handle for name in ("attributes", "X", "Y")):
            raise ValueError("data H5 must contain attributes, X, and Y")
        attributes = json_from_h5_scalar(data_handle["attributes"][()])
        train_indices = set(int(value) for value in attributes.get("train_index_list", []))
        label_vocab = attributes.get("label_vocab")
        if not train_indices or not isinstance(label_vocab, dict) or not label_vocab:
            raise ValueError("source H5 has invalid train indices or label vocabulary")
        missing_labels = sorted(set(target_counts) - set(str(key) for key in label_vocab))
        if missing_labels:
            raise ValueError("reference labels absent from source vocabulary: %s" % missing_labels)
        if args.partition_method not in partition_handle:
            raise KeyError("partition method not found: %s" % args.partition_method)
        method_group = partition_handle[args.partition_method]
        if "partition_data" not in method_group or "n_clients" not in method_group:
            raise ValueError("partition method lacks partition_data or n_clients")
        n_clients = int(method_group["n_clients"][()])

        for client_id in client_ids:
            if client_id >= n_clients:
                raise ValueError("client %d is outside n_clients=%d" % (client_id, n_clients))
            client_key = str(client_id)
            if client_key not in method_group["partition_data"]:
                raise KeyError("partition_data is missing client %s" % client_key)
            client_group = method_group["partition_data"][client_key]
            if "train" not in client_group:
                raise KeyError("client %d lacks a train partition" % client_id)
            indices = [int(value) for value in client_group["train"][()]]
            if len(indices) != len(set(indices)):
                raise ValueError("client %d train partition contains duplicate indices" % client_id)
            source_counts[str(client_id)] = Counter()
            for index in indices:
                if index not in train_indices:
                    raise ValueError("client %d uses non-train index %d" % (client_id, index))
                previous = index_owner.get(index)
                if previous is not None:
                    raise ValueError(
                        "selected clients %d and %d share source index %d"
                        % (previous, client_id, index)
                    )
                index_owner[index] = client_id
                key = str(index)
                if key not in data_handle["X"] or key not in data_handle["Y"]:
                    raise KeyError("source data is missing sample index %d" % index)
                text = decode_scalar(data_handle["X"][key][()])
                label = decode_scalar(data_handle["Y"][key][()])
                if label not in label_vocab:
                    raise ValueError("source label %r is absent from label_vocab" % label)
                if not text.strip():
                    raise ValueError("source index %d has empty text" % index)
                records_by_client[client_id][label].append(
                    {"index": index, "client_id": client_id, "text": text, "label": label}
                )
                source_counts[str(client_id)][label] += 1

    selected_by_label = {}
    for label, target in sorted(target_counts.items()):
        selected_by_label[label] = deque(choose_label_records(
            label, target, client_ids, records_by_client, args.seed
        ))

    selected = []
    output_rows = []
    for label in reference_labels:
        record = selected_by_label[label].popleft()
        selected.append(record)
        output_rows.append({"text": record["text"], "label": record["label"]})
    if any(selected_by_label[label] for label in selected_by_label):
        raise AssertionError("selected-record queues were not exhausted")
    selected_indices = [record["index"] for record in selected]
    if len(selected_indices) != len(set(selected_indices)):
        raise AssertionError("matched real control selected a source record twice")
    if Counter(row["label"] for row in output_rows) != target_counts:
        raise AssertionError("matched real control changed the reference label counts")

    output_path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    manifest_path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    atomic_write_jsonl(output_path, output_rows)
    contribution_counts = {}
    for client_id in client_ids:
        records = [record for record in selected if record["client_id"] == client_id]
        contribution_counts[str(client_id)] = {
            "records": len(records),
            "label_counts": dict(sorted(Counter(
                record["label"] for record in records
            ).items())),
            "selected_index_sha256": sha256_values(sorted(
                record["index"] for record in records
            )),
        }
    payload = {
        "schema_version": 1,
        "status": "complete",
        "control": "same_source_real_matched",
        "deployable": False,
        "contains_private_training_text": True,
        "source_split": "train_only",
        "seed": args.seed,
        "source_client_ids": client_ids,
        "records": len(output_rows),
        "label_counts": dict(sorted(target_counts.items())),
        "label_sequence_matches_reference": [
            row["label"] for row in output_rows
        ] == reference_labels,
        "selected_indices_unique": len(selected_indices) == len(set(selected_indices)),
        "selected_indices_in_train": True,
        "selected_index_sha256": sha256_values(sorted(selected_indices)),
        "client_contributions": contribution_counts,
        "available_source_label_counts": {
            client_id: dict(sorted(counts.items()))
            for client_id, counts in sorted(source_counts.items())
        },
        "reference": {
            "path": str(reference_path),
            "sha256": sha256_file(reference_path),
            "label_sequence_sha256": sha256_values(reference_labels),
        },
        "source": {
            "data_file": str(data_path),
            "data_sha256": sha256_file(data_path),
            "partition_file": str(partition_path),
            "partition_sha256": sha256_file(partition_path),
            "partition_method": args.partition_method,
        },
        "output": {
            "path": str(output_path),
            "sha256": sha256_file(output_path),
        },
    }
    atomic_write_json(manifest_path, payload)
    print(json.dumps(payload, sort_keys=True))


if __name__ == "__main__":
    main()
