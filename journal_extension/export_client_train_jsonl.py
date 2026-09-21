#!/usr/bin/env python3
"""Export selected clients' private *training* records for an isolated generator.

This utility exists because the legacy FedNLP environment has ``h5py`` while
the causal-LM/PEFT environment may not.  It reads only ``partition_data/<id>/
train`` and writes one mode-0600 JSONL file per client.  The staging directory
contains private data and must remain local to the experiment host.
"""

import argparse
import hashlib
import json
import os
import random
import stat
import time
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


def parse_client_ids(values):
    result = []
    for value in values or []:
        for item in str(value).split(","):
            item = item.strip()
            if item:
                result.append(int(item))
    if not result:
        raise ValueError("--client-ids must contain at least one client id")
    if len(result) != len(set(result)):
        raise ValueError("--client-ids contains duplicates")
    if any(client_id < 0 for client_id in result):
        raise ValueError("client ids must be non-negative")
    return result


def json_from_h5_scalar(value):
    if isinstance(value, bytes):
        value = value.decode("utf-8")
    elif not isinstance(value, str):
        value = value.tobytes().decode("utf-8")
    return json.loads(value)


def stratified_limit(rows, limit, seed):
    """Select at most ``limit`` rows while retaining every label when possible."""
    if limit is None or len(rows) <= limit:
        return list(rows)
    if limit <= 0:
        raise ValueError("--sample-limit-per-client must be positive")

    by_label = defaultdict(list)
    for row in rows:
        by_label[row["label"]].append(row)
    rng = random.Random(seed)
    for label_rows in by_label.values():
        rng.shuffle(label_rows)

    labels = sorted(by_label)
    selected = []
    while len(selected) < limit:
        progressed = False
        for label in labels:
            if by_label[label] and len(selected) < limit:
                selected.append(by_label[label].pop())
                progressed = True
        if not progressed:
            break
    return selected


def atomic_write_jsonl(path, rows):
    temporary = path.with_name(path.name + ".tmp")
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
    temporary = path.with_name(path.name + ".tmp")
    if temporary.exists():
        temporary.unlink()
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=True, indent=2, sort_keys=True)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.chmod(str(temporary), stat.S_IRUSR | stat.S_IWUSR)
    os.replace(str(temporary), str(path))


def main():
    parser = argparse.ArgumentParser(
        description="Stage selected clients' train-only H5 records as private JSONL files."
    )
    parser.add_argument("--data-file", required=True)
    parser.add_argument("--partition-file", required=True)
    parser.add_argument("--partition-method", required=True)
    parser.add_argument("--client-ids", nargs="+", required=True,
                        help="Comma-separated and/or space-separated client ids")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--aggregate-output", default="all_clients.jsonl",
                        help="Private aggregate filename for a Real-Matched control")
    parser.add_argument("--no-aggregate-output", action="store_true",
                        help="Do not create the optional private aggregate JSONL")
    parser.add_argument("--sample-limit-per-client", type=int, default=None,
                        help="Deterministic label-stratified limit for a smoke run")
    parser.add_argument("--seed", type=int, default=57)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    started = time.time()
    data_path = Path(args.data_file).expanduser().resolve()
    partition_path = Path(args.partition_file).expanduser().resolve()
    output_dir = Path(args.output_dir).expanduser().resolve()
    client_ids = parse_client_ids(args.client_ids)
    aggregate_name = None if args.no_aggregate_output else args.aggregate_output
    if aggregate_name:
        aggregate_candidate = Path(aggregate_name)
        if aggregate_candidate.is_absolute() or len(aggregate_candidate.parts) != 1:
            raise ValueError("--aggregate-output must be a filename inside --output-dir")
        if aggregate_name == "manifest.json" or aggregate_name.startswith("client_"):
            raise ValueError("--aggregate-output conflicts with a reserved staging filename")
    if not data_path.is_file():
        raise FileNotFoundError(str(data_path))
    if not partition_path.is_file():
        raise FileNotFoundError(str(partition_path))

    # Import lazily so --help and py_compile work outside the legacy H5 env.
    import h5py

    client_rows = {}
    raw_counts = {}
    selected_index_owner = {}
    with h5py.File(str(data_path), "r", swmr=True) as data_handle, h5py.File(
            str(partition_path), "r", swmr=True) as partition_handle:
        if "attributes" not in data_handle or "X" not in data_handle or "Y" not in data_handle:
            raise ValueError("data H5 must contain attributes, X, and Y")
        attributes = json_from_h5_scalar(data_handle["attributes"][()])
        label_vocab = attributes.get("label_vocab")
        if not isinstance(label_vocab, dict) or not label_vocab:
            raise ValueError("source attributes.label_vocab must be a non-empty object")
        if args.partition_method not in partition_handle:
            raise KeyError("partition method not found: %s" % args.partition_method)
        method_group = partition_handle[args.partition_method]
        if "partition_data" not in method_group or "n_clients" not in method_group:
            raise ValueError("partition method is missing n_clients or partition_data")
        n_clients = int(method_group["n_clients"][()])
        source_train = set(int(value) for value in attributes.get("train_index_list", []))
        if not source_train:
            raise ValueError("source attributes.train_index_list is empty")

        for client_id in client_ids:
            if client_id >= n_clients:
                raise ValueError("client %d is outside n_clients=%d" % (client_id, n_clients))
            client_key = str(client_id)
            if client_key not in method_group["partition_data"]:
                raise KeyError("partition_data is missing client %s" % client_key)
            client_group = method_group["partition_data"][client_key]
            if "train" not in client_group:
                raise KeyError("client %d is missing train indices" % client_id)
            indices = [int(value) for value in client_group["train"][()]]
            if not indices:
                raise ValueError("client %d has no train records" % client_id)
            if len(indices) != len(set(indices)):
                raise ValueError("client %d train partition contains duplicate indices" % client_id)
            outside_train = [index for index in indices if index not in source_train]
            if outside_train:
                raise ValueError(
                    "client %d contains indices outside source train_index_list" % client_id
                )
            for index in indices:
                previous = selected_index_owner.get(index)
                if previous is not None:
                    raise ValueError(
                        "selected clients %d and %d share train index %d"
                        % (previous, client_id, index)
                    )
                selected_index_owner[index] = client_id

            rows = []
            for index in indices:
                key = str(index)
                if key not in data_handle["X"] or key not in data_handle["Y"]:
                    raise KeyError("source data is missing sample index %d" % index)
                text = decode_scalar(data_handle["X"][key][()])
                label = decode_scalar(data_handle["Y"][key][()])
                if label not in label_vocab:
                    raise ValueError("sample label %r is absent from source label_vocab" % label)
                if not text.strip():
                    raise ValueError("client %d contains an empty text record" % client_id)
                rows.append({"text": text, "label": label})
            raw_counts[str(client_id)] = len(rows)
            client_rows[client_id] = stratified_limit(
                rows, args.sample_limit_per_client, args.seed + client_id
            )

    source = {
        "data_file": str(data_path),
        "data_sha256": sha256_file(data_path),
        "partition_file": str(partition_path),
        "partition_sha256": sha256_file(partition_path),
        "partition_method": args.partition_method,
        "n_clients": n_clients,
        "label_vocab": label_vocab,
        "task_type": attributes.get("task_type", "text_classification"),
    }
    files = {}
    for client_id in client_ids:
        rows = client_rows[client_id]
        label_counts = dict(sorted(Counter(row["label"] for row in rows).items()))
        filename = "client_%d.jsonl" % client_id
        files[str(client_id)] = {
            "path": filename,
            "records": len(rows),
            "source_train_records": raw_counts[str(client_id)],
            "label_counts": label_counts,
        }

    manifest = {
        "schema_version": 1,
        "status": "dry_run" if args.dry_run else "complete",
        "source_split": "train_only",
        "contains_private_training_text": not args.dry_run,
        "selected_client_ids": client_ids,
        "sample_limit_per_client": args.sample_limit_per_client,
        "seed": args.seed,
        "source": source,
        "clients": files,
        "aggregate_private_file": ({
            "path": aggregate_name,
            "records": sum(len(client_rows[client_id]) for client_id in client_ids),
            "label_counts": dict(sorted(Counter(
                row["label"]
                for client_id in client_ids for row in client_rows[client_id]
            ).items())),
            "contains_private_training_text": True,
        } if aggregate_name else None),
        "elapsed_seconds": round(time.time() - started, 6),
    }
    if args.dry_run:
        print(json.dumps(manifest, ensure_ascii=True, indent=2, sort_keys=True))
        return

    output_dir.mkdir(parents=True, exist_ok=True, mode=0o700)
    os.chmod(str(output_dir), stat.S_IRWXU)
    manifest_path = output_dir / "manifest.json"
    if manifest_path.exists():
        raise FileExistsError(
            "%s already exists; use a new staging directory" % manifest_path
        )
    planned_paths = [
        output_dir / files[str(client_id)]["path"] for client_id in client_ids
    ]
    if aggregate_name:
        planned_paths.append(output_dir / aggregate_name)
    for file_path in planned_paths:
        if file_path.exists():
            raise FileExistsError("%s already exists" % file_path)
    for client_id in client_ids:
        file_path = output_dir / files[str(client_id)]["path"]
        atomic_write_jsonl(file_path, client_rows[client_id])
        files[str(client_id)]["sha256"] = sha256_file(file_path)
    if aggregate_name:
        aggregate_path = output_dir / aggregate_name
        aggregate_rows = [
            row for client_id in client_ids for row in client_rows[client_id]
        ]
        atomic_write_jsonl(aggregate_path, aggregate_rows)
        manifest["aggregate_private_file"]["sha256"] = sha256_file(aggregate_path)
    manifest["elapsed_seconds"] = round(time.time() - started, 6)
    atomic_write_json(manifest_path, manifest)
    print(json.dumps({
        "status": "complete",
        "output_dir": str(output_dir),
        "clients": len(client_ids),
        "records": sum(item["records"] for item in files.values()),
        "manifest": str(manifest_path),
    }, ensure_ascii=True, sort_keys=True))


if __name__ == "__main__":
    main()
