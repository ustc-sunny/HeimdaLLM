#!/usr/bin/env python3
"""Local-only aggregate audit of AG News synthetic releases."""
import argparse
import collections
import hashlib
import json
import math
import re
import statistics
import tarfile
import unicodedata
from pathlib import Path


MODEL = "textattack/distilbert-base-uncased-ag-news"
REVISION = "52ee64de95f38323f136c6f6b05e1af7c433417e"
SEEDS = (57, 58, 59)
ARMS = ("zero_noise", "dp_eps8", "public")
WORD = re.compile(r"[A-Za-z0-9]+(?:'[A-Za-z0-9]+)?")


def file_sha256(path):
    result = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            result.update(chunk)
    return result.hexdigest()


def load_archive(record, directory):
    receipt = record["archive"]
    path = directory / receipt["name"]
    if file_sha256(path) != receipt["sha256"]:
        raise ValueError("archive checksum mismatch: " + path.name)
    with tarfile.open(path, "r:gz") as archive:
        matches = [n for n in archive.getnames() if n.endswith("/generated/synthetic.jsonl")]
        if len(matches) != 1:
            raise ValueError("synthetic release missing or ambiguous: " + path.name)
        raw = archive.extractfile(matches[0]).read().decode("utf-8")
    rows = [json.loads(line) for line in raw.splitlines()]
    if len(rows) != 128 or collections.Counter(str(r["label"]) for r in rows) != {
        "1": 32, "2": 32, "3": 32, "4": 32
    }:
        raise ValueError("unexpected synthetic class balance: " + path.name)
    return rows


def static_metrics(rows):
    texts = [r["text"] for r in rows]
    canonical = [" ".join(unicodedata.normalize("NFKC", t).casefold().split())
                 for t in texts]
    tokens = [WORD.findall(t.casefold()) for t in texts]
    bigrams = [pair for words in tokens for pair in zip(words, words[1:])]
    counts = [len(words) for words in tokens]
    return {
        "records": len(rows),
        "duplicates": len(rows) - len(set(canonical)),
        "mean_words": statistics.mean(counts),
        "median_words": statistics.median(counts),
        "distinct_2": len(set(bigrams)) / len(bigrams) if bigrams else 0.0,
        "prompt_echo_count": sum("category:" in t.casefold() or
                                 "news article:" in t.casefold() for t in texts),
    }


def load_dev(data_file, partition_file):
    import h5py
    with h5py.File(data_file) as data, h5py.File(partition_file) as partition:
        group = partition["pilot_uniform_2/partition_data"]
        indices = [int(i) for client in ("0", "1", "2") for i in group[client]["test"][:]]
        if len(indices) != 512 or len(set(indices)) != 512:
            raise ValueError("unexpected fixed dev partition")
        rows = [(data["X"][str(i)][()].decode("utf-8"),
                 int(data["Y"][str(i)][()].decode("ascii")) - 1) for i in indices]
    if collections.Counter(label for _, label in rows) != {0: 128, 1: 128, 2: 128, 3: 128}:
        raise ValueError("unexpected fixed dev class balance")
    return rows


def classify(texts, tokenizer, model, torch, device, batch_size):
    predictions, assigned_probabilities, entropies = [], [], []
    for offset in range(0, len(texts), batch_size):
        batch = texts[offset:offset + batch_size]
        inputs = tokenizer(batch, padding=True, truncation=True,
                           max_length=128, return_tensors="pt")
        inputs = {key: value.to(device) for key, value in inputs.items()}
        with torch.inference_mode():
            probabilities = model(**inputs).logits.softmax(dim=-1).cpu().tolist()
        for row in probabilities:
            predictions.append(max(range(4), key=lambda i: row[i]))
            assigned_probabilities.append(row)
            entropies.append(-sum(p * math.log(max(p, 1e-12)) for p in row))
    return predictions, assigned_probabilities, entropies


def agreement_metrics(rows, prediction, probabilities, entropies):
    labels = [int(r["label"]) - 1 for r in rows]
    matrix = [[0] * 4 for _ in range(4)]
    for label, predicted in zip(labels, prediction):
        matrix[label][predicted] += 1
    return {
        "label_agreement": sum(a == b for a, b in zip(labels, prediction)) / len(rows),
        "mean_requested_label_probability": statistics.mean(
            p[label] for p, label in zip(probabilities, labels)),
        "mean_prediction_entropy": statistics.mean(entropies),
        "confusion_requested_by_predicted": matrix,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bridge-summary", type=Path, required=True)
    parser.add_argument("--bridge-archives", type=Path, required=True)
    parser.add_argument("--public-summary", type=Path, required=True)
    parser.add_argument("--public-archives", type=Path, required=True)
    parser.add_argument("--data-file", type=Path)
    parser.add_argument("--partition-file", type=Path)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--static-only", action="store_true")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.batch_size < 1:
        parser.error("batch-size must be positive")
    documents = {
        "bridge": json.loads(args.bridge_summary.read_text()),
        "public": json.loads(args.public_summary.read_text()),
    }
    cases = {}
    for arm in ARMS:
        document = documents["public" if arm == "public" else "bridge"]
        directory = args.public_archives if arm == "public" else args.bridge_archives
        for seed in SEEDS:
            found = [r for r in document["runs"] if r["arm"] == arm and r["seed"] == seed]
            if len(found) != 1:
                raise ValueError("missing or duplicate run: %s %d" % (arm, seed))
            rows = load_archive(found[0], directory)
            cases[(arm, seed)] = {"rows": rows, "static": static_metrics(rows)}

    result = {"status": "complete", "scope": "local_only_aggregate_diagnostic",
              "seeds": list(SEEDS), "arms": list(ARMS), "classifier": None,
              "runs": []}
    if not args.static_only:
        if not args.data_file or not args.partition_file:
            parser.error("classifier mode requires fixed dev data and partition files")
        import torch
        from transformers import AutoModelForSequenceClassification, AutoTokenizer
        torch.set_num_threads(min(8, torch.get_num_threads()))
        tokenizer = AutoTokenizer.from_pretrained(MODEL, revision=REVISION,
                                                   trust_remote_code=False)
        model = AutoModelForSequenceClassification.from_pretrained(
            MODEL, revision=REVISION, trust_remote_code=False).to(args.device).eval()
        if model.config.num_labels != 4:
            raise ValueError("classifier must have four output classes")
        dev = load_dev(args.data_file, args.partition_file)
        dev_pred, _, _ = classify([text for text, _ in dev], tokenizer, model, torch,
                                  args.device, args.batch_size)
        dev_labels = [label for _, label in dev]
        dev_accuracy = sum(a == b for a, b in zip(dev_labels, dev_pred)) / len(dev)
        result["classifier"] = {"model": MODEL, "revision": REVISION,
                                "device": args.device, "dev_accuracy": dev_accuracy,
                                "dev_records": len(dev),
                                "valid_for_synthetic_interpretation": dev_accuracy >= 0.85}
        for case in cases.values():
            rows = case["rows"]
            pred, probs, entropy = classify([r["text"] for r in rows], tokenizer,
                                             model, torch, args.device, args.batch_size)
            case["classifier"] = agreement_metrics(rows, pred, probs, entropy)

    for arm in ARMS:
        for seed in SEEDS:
            case = cases[(arm, seed)]
            result["runs"].append({"arm": arm, "seed": seed,
                                   "static": case["static"],
                                   "classifier": case.get("classifier")})
    result["aggregate"] = {}
    for arm in ARMS:
        runs = [r for r in result["runs"] if r["arm"] == arm]
        result["aggregate"][arm] = {
            key: statistics.mean(r["static"][key] for r in runs)
            for key in ("mean_words", "distinct_2", "duplicates", "prompt_echo_count")
        }
        if not args.static_only:
            result["aggregate"][arm]["label_agreement"] = statistics.mean(
                r["classifier"]["label_agreement"] for r in runs)
            result["aggregate"][arm]["mean_requested_label_probability"] = statistics.mean(
                r["classifier"]["mean_requested_label_probability"] for r in runs)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    staging = args.output.with_suffix(args.output.suffix + ".tmp")
    staging.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    staging.replace(args.output)
    print(json.dumps({"status": "complete", "output": str(args.output),
                      "aggregate": result["aggregate"], "classifier": result["classifier"]}))


if __name__ == "__main__":
    main()
