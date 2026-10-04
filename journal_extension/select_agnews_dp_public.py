#!/usr/bin/env python3
"""Select balanced DP/public synthetic records using a public NLI model."""
import argparse
import collections
import hashlib
import json
import math
import os
from pathlib import Path

from diagnose_agnews_synthetic import load_dev


MODEL = "facebook/bart-large-mnli"
REVISION = "d7645e127eaf1aefc7862fd59a17a5aa8558b8ce"
DESCRIPTIONS = (
    "world politics and international affairs",
    "sports",
    "business and finance",
    "science and technology",
)
TEMPLATE = "This news article is about {}."
SEEDS = (57, 58, 59)
CLIENTS = ("1", "21")
LABELS = ("1", "2", "3", "4")


def digest(path):
    sha = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            sha.update(chunk)
    return sha.hexdigest()


def read_release(path):
    manifest_path = path.parent / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    if manifest["status"] != "complete" or digest(path) != manifest["synthetic_sha256"]:
        raise ValueError("source synthetic release failed hash/status check: " + str(path))
    rows = [json.loads(line) for line in path.read_text().splitlines()]
    counts = collections.Counter((str(r["client_id"]), str(r["label"])) for r in rows)
    if len(rows) != 128 or counts != {(client, label): 16 for client in CLIENTS for label in LABELS}:
        raise ValueError("source release has unexpected class/client balance")
    return rows, manifest


def score_texts(texts, tokenizer, model, torch, device, batch_size, entailment_id):
    hypotheses = [TEMPLATE.format(description) for description in DESCRIPTIONS]
    flat = [(text, hypothesis) for text in texts for hypothesis in hypotheses]
    logits = []
    for start in range(0, len(flat), batch_size):
        pairs = flat[start:start + batch_size]
        tokenized = tokenizer([p[0] for p in pairs], [p[1] for p in pairs],
                              padding=True, truncation=True, max_length=160,
                              return_tensors="pt")
        tokenized = {name: value.to(device) for name, value in tokenized.items()}
        with torch.inference_mode():
            values = model(**tokenized).logits[:, entailment_id].float().cpu().tolist()
        logits.extend(values)
    if len(logits) != 4 * len(texts):
        raise AssertionError("NLI score count mismatch")
    probabilities = []
    for start in range(0, len(logits), 4):
        group = logits[start:start + 4]
        maximum = max(group)
        exp = [math.exp(x - maximum) for x in group]
        denominator = sum(exp)
        probabilities.append([x / denominator for x in exp])
    return probabilities


def choose(rows, scores, seed, mode):
    grouped = collections.defaultdict(list)
    for position, (row, probabilities) in enumerate(zip(rows, scores)):
        key = str(row["client_id"]), str(row["label"])
        if mode == "top":
            rank = (-probabilities[int(row["label"]) - 1], position)
        elif mode == "random":
            token = "%d|%s|%s|%s|public_random_v1" % (
                seed, key[0], key[1], row["text"])
            rank = (hashlib.sha256(token.encode()).hexdigest(), position)
        else:
            raise ValueError(mode)
        grouped[key].append((rank, position))
    selected = set()
    for client in CLIENTS:
        for label in LABELS:
            group = sorted(grouped[(client, label)])
            if len(group) != 16:
                raise ValueError("expected 16 input records per client/class")
            selected.update(position for _, position in group[:8])
    if len(selected) != 64:
        raise AssertionError("selected record count mismatch")
    return sorted(selected)


def write_jsonl(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        raise FileExistsError(str(path))
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w") as output:
        for row in rows:
            output.write(json.dumps(row, ensure_ascii=True, sort_keys=True) + "\n")
        output.flush()
        os.fsync(output.fileno())
    temporary.replace(path)
    return digest(path)


def independent_agreement(rows, tokenizer, model, torch, device, batch_size):
    texts = [r["text"] for r in rows]
    predicted = []
    for start in range(0, len(texts), batch_size):
        tokenized = tokenizer(texts[start:start + batch_size], padding=True,
                              truncation=True, max_length=128, return_tensors="pt")
        tokenized = {name: value.to(device) for name, value in tokenized.items()}
        with torch.inference_mode():
            predicted.extend(model(**tokenized).logits.argmax(dim=-1).cpu().tolist())
    return sum(p == int(row["label"]) - 1 for p, row in zip(predicted, rows)) / len(rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--work-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--data-file", type=Path, required=True)
    parser.add_argument("--partition-file", type=Path, required=True)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--min-free-mib", type=int, default=12000)
    parser.add_argument("--batch-size", type=int, default=16)
    args = parser.parse_args()
    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        raise FileExistsError("selection output exists; use a fresh run directory")
    import torch
    from transformers import AutoModelForSequenceClassification, AutoTokenizer
    torch.set_num_threads(min(8, torch.get_num_threads()))
    if args.device.startswith("cuda:"):
        index = int(args.device.split(":")[1])
        torch.cuda.set_device(index)
        free, total = torch.cuda.mem_get_info(index)
        if free < args.min_free_mib * 1024 * 1024:
            raise RuntimeError("GPU free memory below required preflight threshold")
        torch.cuda.set_per_process_memory_fraction(0.20, index)
        device_snapshot = {"gpu_index": index, "free_mib_before": free // 1048576,
                           "total_mib": total // 1048576, "memory_fraction_cap": 0.20}
    else:
        device_snapshot = {"device": "cpu"}
    nli_tokenizer = AutoTokenizer.from_pretrained(MODEL, revision=REVISION,
                                                  trust_remote_code=False)
    nli = AutoModelForSequenceClassification.from_pretrained(
        MODEL, revision=REVISION, trust_remote_code=False).to(args.device).eval()
    entailment = nli.config.label2id.get("entailment")
    if entailment is None:
        entailment = nli.config.label2id.get("ENTAILMENT")
    if entailment is None:
        raise ValueError("NLI entailment class not found")
    dev = load_dev(args.data_file, args.partition_file)
    dev_scores = score_texts([text for text, _ in dev], nli_tokenizer, nli,
                             torch, args.device, args.batch_size, entailment)
    dev_accuracy = sum(max(range(4), key=lambda i: probs[i]) == label
                       for (_, label), probs in zip(dev, dev_scores)) / len(dev)
    if dev_accuracy < 0.70:
        raise RuntimeError("NLI dev accuracy below locked 70% threshold: %.4f" % dev_accuracy)
    # The independent AG News classifier is an audit only; it never chooses records.
    audit_model = "textattack/distilbert-base-uncased-ag-news"
    audit_revision = "52ee64de95f38323f136c6f6b05e1af7c433417e"
    audit_tokenizer = AutoTokenizer.from_pretrained(audit_model, revision=audit_revision,
                                                    trust_remote_code=False)
    audit = AutoModelForSequenceClassification.from_pretrained(
        audit_model, revision=audit_revision, trust_remote_code=False).to(args.device).eval()
    result = {"status": "complete", "selector_model": MODEL,
              "selector_revision": REVISION, "hypothesis_template": TEMPLATE,
              "descriptions": list(DESCRIPTIONS), "device_snapshot": device_snapshot,
              "selector_dev_accuracy": dev_accuracy, "retain_per_client_class": 8,
              "total_selected": 64, "privacy_scope": "postprocessing_of_existing_dp_release_only",
              "audit_model": audit_model, "audit_revision": audit_revision,
              "runs": []}
    base = args.work_root / "results"
    for seed in SEEDS:
        inputs = {
            "dp_eps8": base / "ustc_agnews_noise_bridge_20260930/dp_eps8" /
                       ("seed_%d" % seed) / "generated/synthetic.jsonl",
            "public": base / "ustc_agnews_matched_formal_20260929/public" /
                      ("seed_%d" % seed) / "generated/synthetic.jsonl",
        }
        for source, path in inputs.items():
            rows, manifest = read_release(path)
            if (source == "dp_eps8") != bool(manifest["is_record_level_dp"]):
                raise ValueError("source DP declaration mismatch")
            probabilities = score_texts([r["text"] for r in rows], nli_tokenizer,
                                        nli, torch, args.device,
                                        args.batch_size, entailment)
            for arm, mode in (("filtered_dp", "top"), ("random_dp", "random")) if source == "dp_eps8" else (("filtered_public", "top"),):
                positions = choose(rows, probabilities, seed, mode)
                chosen = [rows[i] for i in positions]
                destination = args.output_dir / arm / ("seed_%d.jsonl" % seed)
                checksum = write_jsonl(destination, chosen)
                result["runs"].append({"arm": arm, "seed": seed,
                    "source": source, "source_manifest_sha256": digest(path.parent / "manifest.json"),
                    "source_synthetic_sha256": digest(path), "selected_sha256": checksum,
                    "selected_records": len(chosen),
                    "selector_label_agreement": sum(
                        max(range(4), key=lambda i: probabilities[j][i]) ==
                        int(rows[j]["label"]) - 1 for j in positions) / len(chosen),
                    "independent_label_agreement": independent_agreement(
                        chosen, audit_tokenizer, audit, torch, args.device, args.batch_size)})
    destination = args.output_dir / "selection_summary.json"
    destination.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"status": result["status"], "output": str(destination),
                      "selector_dev_accuracy": dev_accuracy,
                      "runs": [{k: r[k] for k in ("arm", "seed", "independent_label_agreement")}
                               for r in result["runs"]]}))


if __name__ == "__main__":
    main()
