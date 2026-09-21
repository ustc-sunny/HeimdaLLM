#!/usr/bin/env python3
"""Measure whether client-generated text supplies a useful GGD direction.

This is an offline diagnostic.  It never releases the private comparison
records: the output contains only aggregate gradient and loss statistics.
"""

from __future__ import print_function

import argparse
import hashlib
import json
import math
import random
import statistics
from pathlib import Path

import h5py
import torch
from torch import nn
from transformers import (
    DistilBertConfig,
    DistilBertForSequenceClassification,
    DistilBertTokenizer,
)


def parse_ids(value):
    raw_items = [item.strip() for item in value.split(",")]
    if not raw_items or any(not item for item in raw_items):
        raise ValueError("client IDs must be a non-empty comma-separated list")
    result = [int(item) for item in raw_items]
    if any(item < 0 for item in result):
        raise ValueError("client IDs must be non-negative")
    if len(result) != len(set(result)):
        raise ValueError("client IDs must not contain duplicates")
    return result


def decode(value):
    if isinstance(value, bytes):
        return value.decode("utf-8")
    return str(value)


def sha256_file(path):
    digest = hashlib.sha256()
    with open(str(path), "rb") as handle:
        while True:
            chunk = handle.read(1024 * 1024)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def load_attributes(data_file):
    with h5py.File(str(data_file), "r") as handle:
        raw = handle["attributes"][()]
        attributes = json.loads(decode(raw))
    return attributes


def load_client_rows(data_file, partition_file, method, client_ids, train_universe):
    rows = []
    used_indices = []
    seen_indices = set()
    with h5py.File(str(data_file), "r") as data_handle:
        with h5py.File(str(partition_file), "r") as partition_handle:
            if method not in partition_handle:
                raise KeyError("Missing partition method: %s" % method)
            for client_id in client_ids:
                key = "%s/partition_data/%d/train" % (method, client_id)
                if key not in partition_handle:
                    raise KeyError("Missing client train partition: %s" % key)
                for raw_index in partition_handle[key][()]:
                    index = int(raw_index)
                    if index not in train_universe:
                        raise ValueError(
                            "Client group contains an index outside attributes.train_index_list"
                        )
                    if index in seen_indices:
                        raise ValueError(
                            "Client group contains duplicate/overlapping train indices"
                        )
                    seen_indices.add(index)
                    if str(index) not in data_handle["X"] or str(index) not in data_handle["Y"]:
                        raise KeyError("Data H5 is missing a selected train record")
                    text = decode(data_handle["X"][str(index)][()])
                    label = decode(data_handle["Y"][str(index)][()])
                    rows.append({"text": text, "label": label})
                    used_indices.append(index)
    if not rows:
        raise RuntimeError("No records found for clients %s" % client_ids)
    return rows, used_indices


def validate_disjoint_index_groups(groups):
    """Reject overlap between logical private/control/dev record sets."""
    names = list(groups)
    normalized = {}
    for name in names:
        values = list(groups[name])
        if len(values) != len(set(values)):
            raise ValueError("%s contains duplicate train indices" % name)
        normalized[name] = set(values)
    for left_position, left_name in enumerate(names):
        for right_name in names[left_position + 1:]:
            overlap = normalized[left_name].intersection(normalized[right_name])
            if overlap:
                raise ValueError(
                    "%s and %s overlap in %d actual train indices"
                    % (left_name, right_name, len(overlap))
                )


def load_jsonl(path):
    rows = []
    with open(str(path), "r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                record = json.loads(line)
                rows.append({"text": str(record["text"]), "label": str(record["label"])})
    if not rows:
        raise RuntimeError("No synthetic records found in %s" % path)
    return rows


def balanced_sample(rows, labels, per_label, seed):
    rng = random.Random(seed)
    selected = []
    for label in labels:
        candidates = [row for row in rows if str(row["label"]) == str(label)]
        if len(candidates) < per_label:
            raise RuntimeError(
                "Need %d records for label %r, found %d" %
                (per_label, label, len(candidates))
            )
        rng.shuffle(candidates)
        selected.extend(candidates[:per_label])
    rng.shuffle(selected)
    return selected


def index_digest(indices):
    canonical = ",".join(str(item) for item in sorted(set(indices)))
    return hashlib.sha256(canonical.encode("ascii")).hexdigest()


def make_model(model_path, num_labels, seed, device):
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    config = DistilBertConfig.from_pretrained(
        model_path, num_labels=num_labels, local_files_only=True
    )
    model = DistilBertForSequenceClassification.from_pretrained(
        model_path, config=config, local_files_only=True
    )
    adapter_config = {
        "original_ln_before": True,
        "original_ln_after": True,
        "residual_before_ln": True,
        "adapter_residual_before_ln": False,
        "ln_before": False,
        "ln_after": False,
        "mh_adapter": False,
        "output_adapter": True,
        "non_linearity": "relu",
        "reduction_factor": 16,
        "inv_adapter": None,
        "inv_adapter_reduction_factor": None,
        "cross_adapter": False,
        "leave_out": [],
    }
    model.add_adapter("alignment_adapter", config=adapter_config)
    model.train_adapter("alignment_adapter")
    # Match ForwardTextClassificationTrainer's DistilBERT topology.
    model.add_module("pre_classifier", nn.Sequential())
    model.to(device)
    model.eval()
    trainable = [(name, parameter) for name, parameter in model.named_parameters()
                 if parameter.requires_grad]
    if not trainable:
        raise RuntimeError("No trainable parameters after enabling the adapter")
    return model, trainable


def tokenize_batch(tokenizer, rows, label_vocab, max_length, device):
    encoded = tokenizer(
        [row["text"] for row in rows],
        padding=True,
        truncation=True,
        max_length=max_length,
        return_tensors="pt",
    )
    inputs = {
        key: value.to(device)
        for key, value in encoded.items()
        if key in ("input_ids", "attention_mask")
    }
    labels = torch.tensor(
        [int(label_vocab[str(row["label"])]) for row in rows],
        dtype=torch.long,
        device=device,
    )
    return inputs, labels


def mean_gradient(model, trainable, tokenizer, rows, label_vocab,
                  batch_size, max_length, device):
    model.zero_grad()
    total_loss = 0.0
    total = 0
    for start in range(0, len(rows), batch_size):
        batch = rows[start:start + batch_size]
        inputs, labels = tokenize_batch(
            tokenizer, batch, label_vocab, max_length, device
        )
        output = model(**inputs, labels=labels)
        batch_count = len(batch)
        (output.loss * batch_count).backward()
        total_loss += float(output.loss.detach()) * batch_count
        total += batch_count
    gradients = []
    for _name, parameter in trainable:
        if parameter.grad is None:
            gradients.append(torch.zeros_like(parameter, device="cpu", dtype=torch.float32))
        else:
            gradients.append(parameter.grad.detach().float().cpu() / float(total))
    model.zero_grad()
    return gradients, total_loss / float(total)


def flatten(tensors):
    return torch.cat([tensor.reshape(-1) for tensor in tensors])


def alignment_from_dot_norms(dot, candidate_norm, reference_norm):
    """Return sign-aware and rank-1-subspace alignment from scalar products."""
    denominator = candidate_norm * reference_norm
    if denominator == 0.0:
        signed_cosine = 0.0
        projected_reference_energy = 0.0
    else:
        signed_cosine = dot / denominator
        # Guard only against tiny floating-point excursions outside [-1, 1].
        signed_cosine = max(-1.0, min(1.0, signed_cosine))
        projected_reference_energy = (dot * dot) / (candidate_norm * candidate_norm)
    cosine_squared = signed_cosine * signed_cosine
    return {
        "signed_cosine": signed_cosine,
        "absolute_cosine": abs(signed_cosine),
        "cosine_squared": cosine_squared,
        # Fraction of reference-gradient squared norm captured by span(candidate).
        # This equals cosine_squared, but both names are emitted to make the
        # rank-1 perturbation interpretation explicit.
        "projected_reference_energy_fraction": cosine_squared,
        "projected_reference_energy_l2_squared": projected_reference_energy,
    }


def alignment_metrics(candidate, reference):
    candidate_flat = flatten(candidate)
    reference_flat = flatten(reference)
    candidate_norm = float(candidate_flat.norm())
    reference_norm = float(reference_flat.norm())
    dot = float(torch.dot(candidate_flat, reference_flat))
    result = alignment_from_dot_norms(
        dot, candidate_norm, reference_norm
    )
    result.update({
        "candidate_l2": candidate_norm,
        "reference_l2": reference_norm,
        "dot_product": dot,
    })
    return result


def cosine(left, right):
    return alignment_metrics(left, right)["signed_cosine"]


def sign_agreement(left, right):
    left_flat = flatten(left)
    right_flat = flatten(right)
    mask = (left_flat != 0) & (right_flat != 0)
    if int(mask.sum()) == 0:
        return 0.0, 0
    agreement = (torch.sign(left_flat[mask]) == torch.sign(right_flat[mask])).float().mean()
    return float(agreement), int(mask.sum())


def vector_norm(tensors):
    return float(flatten(tensors).norm())


def normalize_direction(tensors):
    norm = vector_norm(tensors)
    if not math.isfinite(norm) or norm <= 0.0:
        return None, norm
    return [tensor.detach().float().cpu() / norm for tensor in tensors], norm


def random_unit_direction_like(reference, generator):
    tensors = [
        torch.randn(
            tuple(tensor.shape), dtype=torch.float32, device="cpu",
            generator=generator,
        )
        for tensor in reference
    ]
    normalized, norm = normalize_direction(tensors)
    if normalized is None:
        raise RuntimeError("failed to sample a finite non-zero random direction")
    return normalized, norm


def relative_error(candidate, reference):
    delta = [left - right for left, right in zip(candidate, reference)]
    denominator = max(vector_norm(reference), 1e-12)
    return vector_norm(delta) / denominator


def evaluate(model, tokenizer, rows, label_vocab, batch_size, max_length, device):
    total_loss = 0.0
    total_correct = 0
    total = 0
    model.eval()
    with torch.no_grad():
        for start in range(0, len(rows), batch_size):
            batch = rows[start:start + batch_size]
            inputs, labels = tokenize_batch(
                tokenizer, batch, label_vocab, max_length, device
            )
            output = model(**inputs, labels=labels)
            count = len(batch)
            total_loss += float(output.loss) * count
            total_correct += int((output.logits.argmax(dim=-1) == labels).sum())
            total += count
    return {"loss": total_loss / float(total), "accuracy": total_correct / float(total)}


def evaluate_step(model, trainable, tokenizer, dev_rows, label_vocab,
                  gradient, relative_step, batch_size, max_length, device):
    originals = [parameter.detach().clone() for _name, parameter in trainable]
    parameter_norm = math.sqrt(sum(float((value.float() ** 2).sum()) for value in originals))
    gradient_norm = max(vector_norm(gradient), 1e-12)
    update_norm = relative_step * max(parameter_norm, 1.0)
    with torch.no_grad():
        for (_name, parameter), grad in zip(trainable, gradient):
            parameter.sub_(grad.to(parameter.device, dtype=parameter.dtype),
                           alpha=update_norm / gradient_norm)
    metrics = evaluate(
        model, tokenizer, dev_rows, label_vocab, batch_size, max_length, device
    )
    with torch.no_grad():
        for (_name, parameter), original in zip(trainable, originals):
            parameter.copy_(original)
    metrics["update_l2"] = update_norm
    return metrics


def rank1_zo_proxy_step(model, trainable, tokenizer, diagnostic_rows,
                        label_vocab, direction, epsilon, update_norm,
                        batch_size, max_length, device, baseline):
    """Evaluate one central-FD rank-1 estimator on one fixed diagnostic batch.

    The estimator is ``d(v) * v`` for a unit direction ``v``, where ``d(v)`` is
    a central finite-difference directional derivative.  The applied descent
    update is normalized to ``update_norm`` so guided and random directions are
    compared at exactly the same step norm.  This is an offline local proxy for
    the direction mechanism, not a federated or end-to-end HeimdaLLM run.
    """
    unit_direction, original_direction_norm = normalize_direction(direction)
    if unit_direction is None:
        return {
            "status": "zero_or_nonfinite_direction",
            "input_direction_l2": original_direction_norm,
            "finite_difference_epsilon": epsilon,
            "requested_update_l2": update_norm,
            "loss_before": baseline["loss"],
            "accuracy_before": baseline["accuracy"],
        }

    originals = [parameter.detach().clone() for _name, parameter in trainable]

    def restore_with_offset(scale):
        with torch.no_grad():
            for (_name, parameter), original, value in zip(
                    trainable, originals, unit_direction):
                parameter.copy_(original)
                if scale:
                    parameter.add_(
                        value.to(parameter.device, dtype=parameter.dtype),
                        alpha=scale,
                    )

    try:
        restore_with_offset(epsilon)
        plus = evaluate(
            model, tokenizer, diagnostic_rows, label_vocab,
            batch_size, max_length, device,
        )
        restore_with_offset(-epsilon)
        minus = evaluate(
            model, tokenizer, diagnostic_rows, label_vocab,
            batch_size, max_length, device,
        )
        directional_derivative = (plus["loss"] - minus["loss"]) / (2.0 * epsilon)
        estimator_l2 = abs(directional_derivative)

        if not math.isfinite(directional_derivative):
            status = "nonfinite_directional_derivative"
            applied_update_l2 = 0.0
            after = dict(baseline)
        elif estimator_l2 <= 1e-20:
            status = "zero_directional_derivative"
            applied_update_l2 = 0.0
            after = dict(baseline)
        else:
            # estimator / ||estimator|| = sign(d(v)) * v for unit v.
            restore_with_offset(-update_norm * directional_derivative / estimator_l2)
            after = evaluate(
                model, tokenizer, diagnostic_rows, label_vocab,
                batch_size, max_length, device,
            )
            status = "complete"
            applied_update_l2 = update_norm
    finally:
        restore_with_offset(0.0)

    return {
        "status": status,
        "input_direction_l2": original_direction_norm,
        "normalized_direction_l2": 1.0,
        "finite_difference_epsilon": epsilon,
        "loss_plus_epsilon": plus["loss"],
        "loss_minus_epsilon": minus["loss"],
        "directional_derivative": directional_derivative,
        "rank1_estimator_l2": estimator_l2,
        "requested_update_l2": update_norm,
        "applied_update_l2": applied_update_l2,
        "loss_before": baseline["loss"],
        "loss_after": after["loss"],
        "loss_decrease": baseline["loss"] - after["loss"],
        "accuracy_before": baseline["accuracy"],
        "accuracy_after": after["accuracy"],
        "accuracy_change": after["accuracy"] - baseline["accuracy"],
        "sign_flip_invariance": {
            "verified_from_shared_central_difference_evaluations": True,
            "directional_derivative_for_negated_v": -directional_derivative,
            "rank1_estimator_difference_l2": 0.0,
            "loss_after_equal_norm_step_difference": 0.0,
            "explanation": "d(-v)(-v) = d(v)v; the rank-1 estimator is sign invariant.",
        },
    }


def summarize_scalar_values(values):
    if not values:
        return {"count": 0, "mean": None, "median": None, "min": None, "max": None}
    return {
        "count": len(values),
        "mean": sum(values) / float(len(values)),
        "median": statistics.median(values),
        "min": min(values),
        "max": max(values),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-file", required=True)
    parser.add_argument("--partition-file", required=True)
    parser.add_argument("--partition-method", default="uniform")
    parser.add_argument("--client-ids", required=True,
                        help="Private clients represented by the synthetic set")
    parser.add_argument("--real-control-client-ids", required=True,
                        help="Disjoint clients for an equal-size real positive control")
    parser.add_argument("--dev-client-ids", required=True,
                        help="Disjoint clients used only for the loss-decrease diagnostic")
    parser.add_argument("--synthetic", required=True)
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--samples-per-label", type=int, default=32)
    parser.add_argument("--dev-samples-per-label", type=int, default=64)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--max-length", type=int, default=64)
    parser.add_argument("--relative-step", type=float, default=1e-3)
    parser.add_argument(
        "--zo-fd-epsilon", type=float, default=1e-3,
        help="Central finite-difference epsilon for the offline rank-1 ZO proxy",
    )
    parser.add_argument(
        "--zo-random-directions", type=int, default=8,
        help="Number of fixed-seed random matched directions in the ZO proxy",
    )
    parser.add_argument(
        "--zo-dev-batch-size", type=int, default=None,
        help="Number of fixed dev examples used by every ZO proxy direction; default: --batch-size",
    )
    parser.add_argument("--seed", type=int, default=57)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()

    if args.samples_per_label <= 0 or args.dev_samples_per_label <= 0:
        raise ValueError("sample counts must be positive")
    if args.batch_size <= 0 or args.max_length <= 0:
        raise ValueError("batch size and max length must be positive")
    if args.relative_step <= 0.0 or args.zo_fd_epsilon <= 0.0:
        raise ValueError("step size and finite-difference epsilon must be positive")
    if args.zo_random_directions <= 0:
        raise ValueError("--zo-random-directions must be positive")
    if args.zo_dev_batch_size is not None and args.zo_dev_batch_size <= 0:
        raise ValueError("--zo-dev-batch-size must be positive")

    client_ids = parse_ids(args.client_ids)
    control_ids = parse_ids(args.real_control_client_ids)
    dev_ids = parse_ids(args.dev_client_ids)
    if set(client_ids) & set(control_ids) or set(client_ids) & set(dev_ids) or set(control_ids) & set(dev_ids):
        raise ValueError("client, real-control, and dev client IDs must be disjoint")

    attributes = load_attributes(args.data_file)
    label_vocab = {str(key): int(value) for key, value in attributes["label_vocab"].items()}
    labels = [item[0] for item in sorted(label_vocab.items(), key=lambda pair: pair[1])]
    train_universe = set(int(index) for index in attributes.get("train_index_list", []))
    if not train_universe:
        raise ValueError("attributes.train_index_list is empty")

    client_rows, client_indices = load_client_rows(
        args.data_file, args.partition_file, args.partition_method, client_ids,
        train_universe,
    )
    control_rows, control_indices = load_client_rows(
        args.data_file, args.partition_file, args.partition_method, control_ids,
        train_universe,
    )
    dev_rows, dev_indices = load_client_rows(
        args.data_file, args.partition_file, args.partition_method, dev_ids,
        train_universe,
    )
    validate_disjoint_index_groups({
        "client": client_indices,
        "real_control": control_indices,
        "dev": dev_indices,
    })
    synthetic_rows = load_jsonl(args.synthetic)

    client_rows = balanced_sample(
        client_rows, labels, args.samples_per_label, args.seed + 11
    )
    control_rows = balanced_sample(
        control_rows, labels, args.samples_per_label, args.seed + 17
    )
    dev_rows = balanced_sample(
        dev_rows, labels, args.dev_samples_per_label, args.seed + 23
    )
    synthetic_rows = balanced_sample(
        synthetic_rows, labels, args.samples_per_label, args.seed + 29
    )
    shuffled_rows = [dict(row) for row in synthetic_rows]
    shuffled_labels = [row["label"] for row in shuffled_rows]
    random.Random(args.seed + 31).shuffle(shuffled_labels)
    for row, label in zip(shuffled_rows, shuffled_labels):
        row["label"] = label

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    tokenizer = DistilBertTokenizer.from_pretrained(
        args.model_path, do_lower_case=True, local_files_only=True
    )
    model, trainable = make_model(
        args.model_path, len(label_vocab), args.seed, device
    )

    datasets = {
        "client_real": client_rows,
        "real_matched": control_rows,
        "client_synthetic": synthetic_rows,
        "client_synthetic_shuffled": shuffled_rows,
        "dev": dev_rows,
    }
    gradients = {}
    gradient_losses = {}
    for name, rows in datasets.items():
        gradients[name], gradient_losses[name] = mean_gradient(
            model, trainable, tokenizer, rows, label_vocab,
            args.batch_size, args.max_length, device
        )

    baseline = evaluate(
        model, tokenizer, dev_rows, label_vocab,
        args.batch_size, args.max_length, device
    )
    diagnostic_batch_size = args.zo_dev_batch_size or args.batch_size
    diagnostic_rows = dev_rows[:min(diagnostic_batch_size, len(dev_rows))]
    diagnostic_baseline = evaluate(
        model, tokenizer, diagnostic_rows, label_vocab,
        args.batch_size, args.max_length, device,
    )
    parameter_norm = math.sqrt(sum(
        float((parameter.detach().float() ** 2).sum())
        for _name, parameter in trainable
    ))
    zo_update_norm = args.relative_step * max(parameter_norm, 1.0)

    comparisons = {}
    for name in ("client_real", "real_matched", "client_synthetic",
                 "client_synthetic_shuffled"):
        agree_client, agree_client_count = sign_agreement(
            gradients[name], gradients["client_real"]
        )
        agree_dev, agree_dev_count = sign_agreement(
            gradients[name], gradients["dev"]
        )
        after = evaluate_step(
            model, trainable, tokenizer, dev_rows, label_vocab,
            gradients[name], args.relative_step, args.batch_size,
            args.max_length, device
        )
        client_alignment = alignment_metrics(
            gradients[name], gradients["client_real"]
        )
        dev_alignment = alignment_metrics(
            gradients[name], gradients["dev"]
        )
        comparisons[name] = {
            "gradient_l2": vector_norm(gradients[name]),
            # Preserve original scalar fields for existing consumers.
            "cosine_to_client_real": client_alignment["signed_cosine"],
            "cosine_to_dev": dev_alignment["signed_cosine"],
            "absolute_cosine_to_client_real": client_alignment["absolute_cosine"],
            "absolute_cosine_to_dev": dev_alignment["absolute_cosine"],
            "cosine_squared_to_client_real": client_alignment["cosine_squared"],
            "cosine_squared_to_dev": dev_alignment["cosine_squared"],
            "projected_client_real_energy_fraction": client_alignment[
                "projected_reference_energy_fraction"
            ],
            "projected_dev_energy_fraction": dev_alignment[
                "projected_reference_energy_fraction"
            ],
            "projected_client_real_energy_l2_squared": client_alignment[
                "projected_reference_energy_l2_squared"
            ],
            "projected_dev_energy_l2_squared": dev_alignment[
                "projected_reference_energy_l2_squared"
            ],
            "alignment_to_client_real": client_alignment,
            "alignment_to_dev": dev_alignment,
            "sign_agreement_to_client_real": agree_client,
            "sign_agreement_to_client_real_nonzero": agree_client_count,
            "sign_agreement_to_dev": agree_dev,
            "sign_agreement_to_dev_nonzero": agree_dev_count,
            "relative_error_to_client_real": relative_error(
                gradients[name], gradients["client_real"]
            ),
            "dev_after_step": after,
            "dev_loss_decrease": baseline["loss"] - after["loss"],
            "dev_accuracy_change": after["accuracy"] - baseline["accuracy"],
            "rank1_zo_proxy": rank1_zo_proxy_step(
                model, trainable, tokenizer, diagnostic_rows, label_vocab,
                gradients[name], args.zo_fd_epsilon, zo_update_norm,
                args.batch_size, args.max_length, device,
                diagnostic_baseline,
            ),
        }

    # The no-argument form is CPU-backed and works with the older PyTorch used
    # by the legacy FedNLP environment.
    random_generator = torch.Generator()
    random_generator.manual_seed(args.seed + 104729)
    random_controls = []
    for direction_index in range(args.zo_random_directions):
        random_direction, raw_random_norm = random_unit_direction_like(
            gradients["client_synthetic"], random_generator
        )
        proxy = rank1_zo_proxy_step(
            model, trainable, tokenizer, diagnostic_rows, label_vocab,
            random_direction, args.zo_fd_epsilon, zo_update_norm,
            args.batch_size, args.max_length, device, diagnostic_baseline,
        )
        proxy["direction_index"] = direction_index
        proxy["raw_gaussian_direction_l2"] = raw_random_norm
        random_controls.append(proxy)

    random_loss_decreases = [
        item["loss_decrease"] for item in random_controls
        if item.get("status") == "complete"
    ]
    guided_loss_decrease = comparisons["client_synthetic"][
        "rank1_zo_proxy"
    ].get("loss_decrease")
    paired_advantages = (
        [guided_loss_decrease - value for value in random_loss_decreases]
        if guided_loss_decrease is not None else []
    )
    rank1_zo_diagnostic = {
        "name": "offline_central_fd_rank1_zo_proxy",
        "is_end_to_end_federated_evaluation": False,
        "same_fixed_dev_batch_for_all_directions": True,
        "diagnostic_dev_records": len(diagnostic_rows),
        "finite_difference_epsilon": args.zo_fd_epsilon,
        "equal_update_l2": zo_update_norm,
        "random_seed": args.seed + 104729,
        "random_direction_count_requested": args.zo_random_directions,
        "random_controls": random_controls,
        "random_loss_decrease_summary": summarize_scalar_values(
            random_loss_decreases
        ),
        "client_synthetic_loss_decrease": guided_loss_decrease,
        "client_synthetic_minus_random_loss_decrease": paired_advantages,
        "client_synthetic_minus_random_summary": summarize_scalar_values(
            paired_advantages
        ),
        "interpretation": (
            "Central finite differences estimate d(v); each equal-norm step follows "
            "the normalized rank-1 estimator d(v)*v. Positive loss_decrease means "
            "improvement on this fixed held-out diagnostic batch. This isolates a "
            "local direction mechanism and is not a complete HeimdaLLM training run."
        ),
    }

    result = {
        "status": "complete",
        "diagnostic": "offline_gradient_guidance_alignment",
        "seed": args.seed,
        "true_non_dp": True,
        "samples_per_label": args.samples_per_label,
        "dev_samples_per_label": args.dev_samples_per_label,
        "label_vocab": label_vocab,
        "client_ids": client_ids,
        "real_control_client_ids": control_ids,
        "dev_client_ids": dev_ids,
        "private_index_hashes": {
            "client": index_digest(client_indices),
            "real_control": index_digest(control_indices),
            "dev": index_digest(dev_indices),
        },
        "index_validation": {
            "all_selected_indices_in_attributes_train_index_list": True,
            "actual_index_groups_mutually_disjoint": True,
            "selected_index_counts": {
                "client": len(client_indices),
                "real_control": len(control_indices),
                "dev": len(dev_indices),
            },
        },
        "input_sha256": {
            "data": sha256_file(args.data_file),
            "partition": sha256_file(args.partition_file),
            "synthetic": sha256_file(args.synthetic),
        },
        "model_path": str(Path(args.model_path).resolve()),
        "trainable_parameter_count": sum(parameter.numel() for _name, parameter in trainable),
        "trainable_parameter_names": [name for name, _parameter in trainable],
        "gradient_set_losses": gradient_losses,
        "dev_baseline": baseline,
        "relative_step": args.relative_step,
        "rank1_zo_diagnostic": rank1_zo_diagnostic,
        "diagnostic_semantics": {
            "gradient_alignment": (
                "Signed cosine describes gradient orientation; absolute cosine and "
                "cosine_squared/projected energy describe the sign-invariant span "
                "available to a rank-1 perturbation."
            ),
            "rank1_zo_proxy": (
                "Offline central-finite-difference diagnostic on one fixed dev batch; "
                "it does not include client sampling, communication, aggregation, or "
                "multi-round optimization and must not be reported as end-to-end GGD."
            ),
            "evaluation_split": (
                "All private, real-control, and dev indices are verified to belong to "
                "attributes.train_index_list and to be mutually disjoint."
            ),
        },
        "comparisons": comparisons,
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
