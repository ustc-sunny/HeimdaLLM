#!/usr/bin/env python3
"""Build a NEW SST-2 partition for deployment checks, not KDD reproduction.

Uses the same root-sentence/neutral-removal rule as data/raw_data_loader/SST_2.
Each of 100 clients gets 32 examples per label from the official train split.
Remaining training examples are explicitly unused. Official test is separate.
"""
import argparse
import hashlib
import json
import random
import zipfile
from pathlib import Path

import h5py
import numpy as np
from nltk.tree import Tree


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-zip", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--seed", type=int, default=20260921)
    args = parser.parse_args()
    source = Path(args.source_zip)
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    paths = [out / name for name in ("sst_2_data.h5", "sst_2_partition.h5", "manifest.json")]
    if any(path.exists() for path in paths):
        raise FileExistsError("Refusing to overwrite deployment dataset")
    rows, splits = [], {}
    with zipfile.ZipFile(source) as archive:
        for split in ("train", "test"):
            indices = []
            for line in archive.read("trees/" + split + ".txt").decode().splitlines():
                tree = Tree.fromstring(line)
                label = int(tree.label())
                if label == 2:
                    continue
                indices.append(len(rows))
                rows.append((" ".join(tree.leaves()), "negative" if label < 2 else "positive"))
            splits[split] = indices
    if (len(splits["train"]), len(splits["test"])) != (6920, 1821):
        raise ValueError("Unexpected SST-2 root-sentence counts")
    labels = {"negative": 0, "positive": 1}
    pools = {label: [i for i in splits["train"] if rows[i][1] == label] for label in labels}
    for offset, pool in enumerate(pools.values()):
        random.Random(args.seed + offset).shuffle(pool)
        if len(pool) < 3200:
            raise ValueError("Insufficient train data for deployment partition")
    clients = {}
    for client in range(100):
        indices = sum((pool[client * 32:(client + 1) * 32] for pool in pools.values()), [])
        random.Random(args.seed + 100 + client).shuffle(indices)
        clients[client] = indices
    assigned = [i for indices in clients.values() for i in indices]
    assert len(assigned) == len(set(assigned)) == 6400
    assert not set(assigned).intersection(splits["test"])
    attributes = dict(index_list=list(range(len(rows))), train_index_list=splits["train"],
                      test_index_list=splits["test"], label_vocab=labels, num_labels=2,
                      task_type="text_classification", deployment_only=True)
    with h5py.File(paths[0], "w") as data:
        x, y = data.create_group("X"), data.create_group("Y")
        for i, (text, label) in enumerate(rows):
            x.create_dataset(str(i), data=text.encode())
            y.create_dataset(str(i), data=label.encode())
        data.create_dataset("attributes", data=json.dumps(attributes).encode())
    with h5py.File(paths[1], "w") as part:
        method = part.create_group("deployment_balanced100")
        method.create_dataset("n_clients", data=100)
        groups = method.create_group("partition_data")
        for client, indices in clients.items():
            group = groups.create_group(str(client))
            group.create_dataset("train", data=np.asarray(indices, dtype=np.int64))
            group.create_dataset("test", data=np.asarray(splits["test"][client::100], dtype=np.int64))
    manifest = dict(purpose="deployment_check_only_not_original_KDD_partition",
                    source_url="https://nlp.stanford.edu/sentiment/trainDevTestTrees_PTB.zip",
                    source_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
                    seed=args.seed, train_records=6920, test_records=1821,
                    clients=100, records_per_client=64, label_vocab=labels,
                    unused_train_indices=sorted(set(splits["train"]) - set(assigned)),
                    file_sha256={p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in paths[:2]})
    paths[2].write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps({k: v for k, v in manifest.items() if k != "unused_train_indices"}, indent=2))


if __name__ == "__main__":
    main()
