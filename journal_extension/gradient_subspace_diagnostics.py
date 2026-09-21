#!/usr/bin/env python3
"""Offline gradient and rank-2 client-subspace diagnostics for AG News v3.

This diagnostic uses one frozen task-model checkpoint per seed. It does not run
federated optimization and must not be interpreted as end-to-end evidence.
"""

import argparse
import hashlib
import json
import math
from pathlib import Path

import torch

import gradient_guidance_alignment as base


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_synthetic(path):
    rows = []
    with Path(path).open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                item = json.loads(line)
                rows.append({
                    "text": str(item["text"]),
                    "label": str(item["label"]),
                    "client_id": int(item["client_id"]),
                })
    if not rows:
        raise ValueError("empty synthetic file: %s" % path)
    return rows


def client_balanced(rows, client_ids, labels, per_label, seed):
    result = {}
    for offset, client_id in enumerate(client_ids):
        candidates = [row for row in rows if int(row["client_id"]) == client_id]
        result[client_id] = base.balanced_sample(
            candidates, labels, per_label, seed + offset
        )
    return result


def flatten64(gradient):
    return base.flatten(gradient).double()


def orthonormal_basis(gradients, tolerance=1e-10):
    matrix = torch.stack([flatten64(item) for item in gradients], dim=0)
    gram = matrix.mm(matrix.t())
    eigenvalues, eigenvectors = torch.linalg.eigh(gram)
    maximum = float(eigenvalues.max()) if eigenvalues.numel() else 0.0
    threshold = max(tolerance * maximum, 1e-20)
    keep = eigenvalues > threshold
    if int(keep.sum()) == 0:
        return torch.empty((matrix.shape[1], 0), dtype=torch.float64), []
    values = eigenvalues[keep]
    vectors = eigenvectors[:, keep]
    basis = matrix.t().mm(vectors).div(values.sqrt().unsqueeze(0))
    return basis, [float(value) for value in values.flip(0)]


def projection_metrics(gradients, reference):
    basis, eigenvalues = orthonormal_basis(gradients)
    target = flatten64(reference)
    target_sq = float(torch.dot(target, target))
    projected_sq = float((basis.t().mv(target) ** 2).sum()) if basis.shape[1] else 0.0
    fraction = projected_sq / target_sq if target_sq > 0.0 else 0.0
    return {
        "rank": int(basis.shape[1]),
        "gram_eigenvalues_desc": eigenvalues,
        "reference_l2": math.sqrt(target_sq),
        "projected_reference_l2_squared": projected_sq,
        "projected_reference_energy_fraction": fraction,
    }


def principal_cosines(left_gradients, right_gradients):
    left, _ = orthonormal_basis(left_gradients)
    right, _ = orthonormal_basis(right_gradients)
    if left.shape[1] == 0 or right.shape[1] == 0:
        values = []
    else:
        values = [
            float(value) for value in
            torch.linalg.svdvals(left.t().mm(right)).clamp(0.0, 1.0)
        ]
    return {
        "left_rank": int(left.shape[1]),
        "right_rank": int(right.shape[1]),
        "principal_cosines_desc": values,
        "principal_angles_degrees_asc": [
            math.degrees(math.acos(value)) for value in values
        ],
    }


def atomic_json(path, payload):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-file", required=True)
    parser.add_argument("--partition-file", required=True)
    parser.add_argument("--partition-method", required=True)
    parser.add_argument("--client-ids", required=True)
    parser.add_argument("--real-control-client-ids", required=True)
    parser.add_argument("--dev-client-ids", required=True)
    parser.add_argument("--client-synthetic", required=True)
    parser.add_argument("--public-synthetic", required=True)
    parser.add_argument("--shuffled-synthetic", required=True)
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--samples-per-label", type=int, default=32)
    parser.add_argument("--per-client-samples-per-label", type=int, default=16)
    parser.add_argument("--dev-samples-per-label", type=int, default=64)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--max-length", type=int, default=64)
    parser.add_argument("--relative-step", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()

    client_ids = base.parse_ids(args.client_ids)
    control_ids = base.parse_ids(args.real_control_client_ids)
    dev_ids = base.parse_ids(args.dev_client_ids)
    if len(client_ids) != 2:
        raise ValueError("v3 subspace diagnostic expects exactly two source clients")
    if set(client_ids) & set(control_ids + dev_ids) or set(control_ids) & set(dev_ids):
        raise ValueError("source, control and dev client IDs must be disjoint")

    attributes = base.load_attributes(args.data_file)
    label_vocab = {str(key): int(value) for key, value in attributes["label_vocab"].items()}
    labels = [item[0] for item in sorted(label_vocab.items(), key=lambda pair: pair[1])]
    train_universe = set(int(index) for index in attributes["train_index_list"])

    source_by_client = {}
    source_indices = []
    for offset, client_id in enumerate(client_ids):
        rows, indices = base.load_client_rows(
            args.data_file, args.partition_file, args.partition_method,
            [client_id], train_universe,
        )
        source_indices.extend(indices)
        source_by_client[client_id] = base.balanced_sample(
            rows, labels, args.per_client_samples_per_label,
            args.seed + 101 + offset,
        )
    control_rows, control_indices = base.load_client_rows(
        args.data_file, args.partition_file, args.partition_method,
        control_ids, train_universe,
    )
    dev_rows, dev_indices = base.load_client_rows(
        args.data_file, args.partition_file, args.partition_method,
        dev_ids, train_universe,
    )
    base.validate_disjoint_index_groups({
        "source": source_indices,
        "real_control": control_indices,
        "dev": dev_indices,
    })
    source_rows = [row for client_id in client_ids for row in source_by_client[client_id]]
    control_rows = base.balanced_sample(
        control_rows, labels, args.samples_per_label, args.seed + 211
    )
    dev_rows = base.balanced_sample(
        dev_rows, labels, args.dev_samples_per_label, args.seed + 223
    )

    client_synthetic = load_synthetic(args.client_synthetic)
    public_synthetic = load_synthetic(args.public_synthetic)
    shuffled_synthetic = load_synthetic(args.shuffled_synthetic)
    client_syn_by_client = client_balanced(
        client_synthetic, client_ids, labels,
        args.per_client_samples_per_label, args.seed + 307,
    )
    public_syn_by_client = client_balanced(
        public_synthetic, client_ids, labels,
        args.per_client_samples_per_label, args.seed + 311,
    )
    # Shuffling preserves client IDs and total class counts, but not necessarily
    # per-client class counts. Use every matched record within each client.
    shuffled_by_client = {
        client_id: [
            row for row in shuffled_synthetic
            if int(row["client_id"]) == client_id
        ]
        for client_id in client_ids
    }

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    tokenizer = base.DistilBertTokenizer.from_pretrained(
        args.model_path, do_lower_case=True, local_files_only=True
    )
    model, trainable = base.make_model(
        args.model_path, len(label_vocab), args.seed, device
    )

    row_sets = {
        "source_real": source_rows,
        "heldout_real": control_rows,
        "client_synthetic": client_synthetic,
        "public_synthetic": public_synthetic,
        "client_synthetic_shuffled": shuffled_synthetic,
        "dev": dev_rows,
    }
    gradients = {}
    losses = {}
    for name, rows in row_sets.items():
        gradients[name], losses[name] = base.mean_gradient(
            model, trainable, tokenizer, rows, label_vocab,
            args.batch_size, args.max_length, device,
        )

    per_client = {name: {} for name in (
        "source_real", "client_synthetic", "public_synthetic",
        "client_synthetic_shuffled",
    )}
    per_client_rows = {
        "source_real": source_by_client,
        "client_synthetic": client_syn_by_client,
        "public_synthetic": public_syn_by_client,
        "client_synthetic_shuffled": shuffled_by_client,
    }
    for name, groups in per_client_rows.items():
        for client_id in client_ids:
            per_client[name][client_id], _loss = base.mean_gradient(
                model, trainable, tokenizer, groups[client_id], label_vocab,
                args.batch_size, args.max_length, device,
            )

    baseline = base.evaluate(
        model, tokenizer, dev_rows, label_vocab,
        args.batch_size, args.max_length, device,
    )
    aggregate = {}
    for name in (
        "source_real", "heldout_real", "client_synthetic",
        "public_synthetic", "client_synthetic_shuffled",
    ):
        after = base.evaluate_step(
            model, trainable, tokenizer, dev_rows, label_vocab,
            gradients[name], args.relative_step, args.batch_size,
            args.max_length, device,
        )
        aggregate[name] = {
            "gradient_loss": losses[name],
            "alignment_to_source_real": base.alignment_metrics(
                gradients[name], gradients["source_real"]
            ),
            "alignment_to_dev": base.alignment_metrics(
                gradients[name], gradients["dev"]
            ),
            "dev_after_equal_norm_step": after,
            "dev_loss_decrease": baseline["loss"] - after["loss"],
            "dev_accuracy_change": after["accuracy"] - baseline["accuracy"],
        }

    subspaces = {}
    for name in (
        "source_real", "client_synthetic", "public_synthetic",
        "client_synthetic_shuffled",
    ):
        vectors = [per_client[name][client_id] for client_id in client_ids]
        subspaces[name] = {
            "projection_of_source_real": projection_metrics(
                vectors, gradients["source_real"]
            ),
            "projection_of_dev": projection_metrics(vectors, gradients["dev"]),
        }

    pairs = (
        ("client_synthetic", "source_real"),
        ("public_synthetic", "source_real"),
        ("client_synthetic_shuffled", "source_real"),
        ("client_synthetic", "public_synthetic"),
    )
    principal = {}
    for left, right in pairs:
        principal[left + "__vs__" + right] = principal_cosines(
            [per_client[left][client_id] for client_id in client_ids],
            [per_client[right][client_id] for client_id in client_ids],
        )

    correspondence = {}
    for client_id in client_ids:
        correspondence[str(client_id)] = {
            name: base.alignment_metrics(
                per_client[name][client_id], per_client["source_real"][client_id]
            )
            for name in (
                "client_synthetic", "public_synthetic",
                "client_synthetic_shuffled",
            )
        }

    result = {
        "schema_version": 1,
        "status": "complete",
        "diagnostic": "offline_gradient_and_two_client_subspace",
        "is_end_to_end_federated_evaluation": False,
        "seed": args.seed,
        "client_ids": client_ids,
        "real_control_client_ids": control_ids,
        "dev_client_ids": dev_ids,
        "label_vocab": label_vocab,
        "samples": {
            "source_real": len(source_rows),
            "heldout_real": len(control_rows),
            "client_synthetic": len(client_synthetic),
            "public_synthetic": len(public_synthetic),
            "client_synthetic_shuffled": len(shuffled_synthetic),
            "dev": len(dev_rows),
        },
        "input_sha256": {
            "data": sha256(args.data_file),
            "partition": sha256(args.partition_file),
            "client_synthetic": sha256(args.client_synthetic),
            "public_synthetic": sha256(args.public_synthetic),
            "client_synthetic_shuffled": sha256(args.shuffled_synthetic),
        },
        "index_validation": {
            "actual_real_index_groups_mutually_disjoint": True,
            "source_index_sha256": base.index_digest(source_indices),
            "control_index_sha256": base.index_digest(control_indices),
            "dev_index_sha256": base.index_digest(dev_indices),
        },
        "trainable_parameter_count": sum(
            parameter.numel() for _name, parameter in trainable
        ),
        "dev_baseline": baseline,
        "aggregate_gradient_diagnostics": aggregate,
        "two_client_subspace_diagnostics": subspaces,
        "principal_angles_between_subspaces": principal,
        "corresponding_client_gradient_alignment_to_source_real": correspondence,
        "interpretation_limits": (
            "Gradients are measured at one initial task-model checkpoint. The rank-2 "
            "subspaces span the two per-client mean gradients and are sign invariant. "
            "They do not measure multi-round convergence or DP behavior. Public virtual "
            "client IDs only match generation quotas/seeds and do not imply private-client "
            "conditioning."
        ),
    }
    atomic_json(args.output, result)
    print(json.dumps({
        "status": result["status"],
        "seed": args.seed,
        "client_syn_source_projection": subspaces["client_synthetic"]
            ["projection_of_source_real"]["projected_reference_energy_fraction"],
        "public_syn_source_projection": subspaces["public_synthetic"]
            ["projection_of_source_real"]["projected_reference_energy_fraction"],
    }, sort_keys=True))


if __name__ == "__main__":
    main()
