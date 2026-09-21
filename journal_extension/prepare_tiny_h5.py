#!/usr/bin/env python3
"""Create a tiny, schema-compatible HDF5/partition pair for a smoke run."""

import argparse
import json
from pathlib import Path

import h5py
import numpy as np


def read_jsonl(path):
    rows = []
    with open(path, "r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
    if not rows:
        raise RuntimeError("synthetic.jsonl is empty")
    return rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--jsonl", required=True)
    parser.add_argument("--data-out", required=True)
    parser.add_argument("--partition-out", required=True)
    parser.add_argument("--total-clients", type=int, default=1000)
    parser.add_argument("--active-clients", type=int, default=8)
    parser.add_argument("--reported-clients", type=int, default=None,
                        help="n_clients reported to the legacy manager; partition keys still use total-clients")
    parser.add_argument("--cloud-active-clients", type=int, default=100,
                        help="non-empty clients in the final cloud suffix")
    args = parser.parse_args()
    rows = read_jsonl(args.jsonl)
    if args.total_clients < 100 or args.active_clients > int(args.total_clients * 0.8):
        raise ValueError("total-clients must preserve the 80%% client convention")

    labels = sorted({row["label"] for row in rows})
    # Match the SST-2 loader contract: labels are stored as strings and mapped
    # to contiguous integer ids in attributes.
    label_vocab = {label: index for index, label in enumerate(labels)}
    data_out = Path(args.data_out)
    part_out = Path(args.partition_out)
    data_out.parent.mkdir(parents=True, exist_ok=True)
    part_out.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(data_out, "w") as data:
        x_group = data.create_group("X")
        y_group = data.create_group("Y")
        for index, row in enumerate(rows):
            x_group.create_dataset(str(index), data=row["text"].encode("utf-8"))
            y_group.create_dataset(str(index), data=row["label"].encode("utf-8"))
        attributes = {
            "index_list": list(range(len(rows))),
            "train_index_list": list(range(len(rows))),
            "test_index_list": list(range(len(rows))),
            "label_vocab": label_vocab,
            "num_labels": len(label_vocab),
            "task_type": "text_classification",
        }
        data.create_dataset("attributes", data=json.dumps(attributes).encode("utf-8"))

    with h5py.File(part_out, "w") as part:
        method = part.create_group("uniform_client_1000")
        method.create_dataset("n_clients", data=args.reported_clients or args.total_clients)
        partition = method.create_group("partition_data")
        # Every key exists because the legacy manager iterates all 1000 keys.
        for client_id in range(args.total_clients):
            client = partition.create_group(str(client_id))
            if client_id < args.active_clients:
                indices = np.asarray([client_id % len(rows)], dtype=np.int64)
                client.create_dataset("train", data=indices)
                client.create_dataset("test", data=indices)
            elif max(0, args.total_clients - 100) <= client_id < max(0, args.total_clients - 100) + args.cloud_active_clients:
                # The legacy cloud loader merges the final 100 client ids.
                # Populate that suffix for arbitrary smoke partition sizes.
                cloud_id = client_id - max(0, args.total_clients - 100)
                indices = np.asarray([cloud_id % len(rows)], dtype=np.int64)
                client.create_dataset("train", data=indices)
                client.create_dataset("test", data=indices)
            else:
                client.create_dataset("train", data=np.asarray([], dtype=np.int64))
                client.create_dataset("test", data=np.asarray([], dtype=np.int64))
    print(json.dumps({"rows": len(rows), "total_clients": args.total_clients, "active_clients": args.active_clients}))


if __name__ == "__main__":
    main()
