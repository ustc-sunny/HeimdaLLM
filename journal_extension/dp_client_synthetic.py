#!/usr/bin/env python3
"""Generate label-conditioned synthetic text with record-level DP LoRA.

Every selected client gets a fresh base model and LoRA adapter.  Training uses
independent Poisson sampling, per-example global L2 clipping, Gaussian noise,
and Opacus' RDP accountant.  Client partitions are required to be disjoint by
the private staging exporter, so the released collection uses parallel
composition across clients.

The release contains synthetic text and public accounting metadata only.  It
does not publish private examples, per-example losses, gradient norms, clipping
rates, private label histograms, or private-dependent exact-match diagnostics.
"""

import argparse
import gc
import json
import math
import random
from collections import Counter
from pathlib import Path

import non_dp_client_synthetic as common


def allocate_public_quotas(client_ids, labels, samples_per_label, target_per_label):
    """Allocate quotas using public IDs and labels, never observed client labels."""
    quotas = {
        client_id: {label: 0 for label in labels}
        for client_id in client_ids
    }
    ordered_clients = sorted(client_ids)
    if target_per_label is None:
        for client_id in ordered_clients:
            for label in labels:
                quotas[client_id][label] = samples_per_label
        return quotas

    for label_position, label in enumerate(labels):
        quotient, remainder = divmod(target_per_label, len(ordered_clients))
        offset = label_position % len(ordered_clients)
        rotated = ordered_clients[offset:] + ordered_clients[:offset]
        extra = set(rotated[:remainder])
        for client_id in ordered_clients:
            quotas[client_id][label] = quotient + (1 if client_id in extra else 0)
    return quotas


def public_source_provenance(source):
    """Retain dataset provenance without copying private staging metadata."""
    return {
        "kind": source.get("kind"),
        "original_data_sha256": source.get("original_data_sha256"),
        "original_partition_sha256": source.get("original_partition_sha256"),
        "partition_method": source.get("partition_method"),
        "selected_client_partitions_verified_disjoint": True,
        "source_split": "train_only",
    }


def calibrate_noise(get_noise_multiplier, args, sample_rate, total_steps):
    if args.target_epsilon is not None:
        noise_multiplier = get_noise_multiplier(
            target_epsilon=args.target_epsilon,
            target_delta=args.delta,
            sample_rate=sample_rate,
            steps=total_steps,
            accountant="rdp",
            epsilon_tolerance=args.epsilon_tolerance,
        )
    else:
        noise_multiplier = args.noise_multiplier
    if not math.isfinite(noise_multiplier) or noise_multiplier <= 0.0:
        raise ValueError("calibrated noise multiplier is invalid")
    return float(noise_multiplier)


def dp_train_adapter(
        torch, model, tokenizer, rows, args, device, training_seed,
        prompt_template, RDPAccountant, get_noise_multiplier):
    """Run auditable Poisson-sampled DP-SGD over trainable LoRA parameters."""
    if len(rows) != args.records_per_client:
        raise ValueError(
            "client has %d records; --records-per-client requires exactly %d"
            % (len(rows), args.records_per_client)
        )
    encoded = common.encode_training_rows(
        tokenizer, rows, args.max_length, prompt_template, args.label_names
    )
    trainable = [parameter for parameter in model.parameters() if parameter.requires_grad]
    if not trainable:
        raise RuntimeError("LoRA created no trainable parameters")

    steps_per_epoch = int(math.ceil(float(len(encoded)) / args.batch_size))
    sample_rate = 1.0 / float(steps_per_epoch)
    total_steps = args.epochs * steps_per_epoch
    expected_batch_size = sample_rate * len(encoded)
    noise_multiplier = calibrate_noise(
        get_noise_multiplier, args, sample_rate, total_steps
    )
    accountant = RDPAccountant()
    optimizer = torch.optim.AdamW(
        trainable, lr=args.learning_rate, weight_decay=args.weight_decay
    )
    collate = common.make_collate(torch, tokenizer.pad_token_id)
    sampling_generator = torch.Generator(device="cpu")
    sampling_generator.manual_seed(training_seed)
    noise_device = str(device) if device.type == "cuda" else "cpu"
    noise_generator = torch.Generator(device=noise_device)
    noise_generator.manual_seed(common.derived_seed(training_seed, "gaussian_noise"))
    accumulators = [
        torch.zeros_like(parameter, dtype=torch.float32, device=device)
        for parameter in trainable
    ]

    model.train()
    model.config.use_cache = False
    optimizer.zero_grad(set_to_none=True)
    for _step in range(total_steps):
        selection = torch.rand(
            len(encoded), generator=sampling_generator, device="cpu"
        ) < sample_rate
        selected_indices = selection.nonzero(as_tuple=False).flatten().tolist()
        for index in selected_indices:
            batch = {
                key: value.to(device)
                for key, value in collate([encoded[index]]).items()
            }
            loss = model(**batch).loss
            if not torch.isfinite(loss):
                raise RuntimeError("DP training loss became non-finite")
            loss.backward()
            squared_norm = torch.zeros((), dtype=torch.float32, device=device)
            for parameter in trainable:
                if parameter.grad is not None:
                    squared_norm.add_(parameter.grad.detach().float().pow(2).sum())
            grad_norm = torch.sqrt(squared_norm)
            clip_factor = torch.clamp(
                args.max_grad_norm / (grad_norm + 1e-12), max=1.0
            )
            for accumulator, parameter in zip(accumulators, trainable):
                if parameter.grad is not None:
                    accumulator.add_(parameter.grad.detach().float() * clip_factor)
            optimizer.zero_grad(set_to_none=True)

        noise_std = noise_multiplier * args.max_grad_norm
        for accumulator, parameter in zip(accumulators, trainable):
            noise = torch.randn(
                accumulator.shape,
                generator=noise_generator,
                device=device,
                dtype=torch.float32,
            )
            noisy_average = (accumulator + noise * noise_std) / expected_batch_size
            parameter.grad = noisy_average.to(dtype=parameter.dtype)
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
        for accumulator in accumulators:
            accumulator.zero_()
        accountant.step(
            noise_multiplier=noise_multiplier, sample_rate=sample_rate
        )

    epsilon, best_alpha = accountant.get_privacy_spent(delta=args.delta)
    if args.target_epsilon is not None:
        allowed = args.target_epsilon + max(args.epsilon_tolerance, 1e-6)
        if epsilon > allowed:
            raise RuntimeError(
                "accounted epsilon %.8f exceeds target %.8f" % (
                    epsilon, args.target_epsilon
                )
            )
    return optimizer, {
        "status": "complete",
        "records": args.records_per_client,
        "epochs": args.epochs,
        "steps_per_epoch": steps_per_epoch,
        "optimizer_steps": total_steps,
        "sampling": "independent_poisson",
        "sample_rate": sample_rate,
        "expected_batch_size": expected_batch_size,
        "max_grad_norm": args.max_grad_norm,
        "noise_multiplier": noise_multiplier,
        "accountant": "opacus_rdp",
        "epsilon": float(epsilon),
        "delta": args.delta,
        "best_alpha": float(best_alpha),
    }


def validate_args(args):
    args.prompt_template = common.normalize_prompt_template(args.prompt_template)
    args.label_names = json.loads(args.label_names_json)
    if not isinstance(args.label_names, dict) or not args.label_names:
        raise ValueError("--label-names-json must be a nonempty public label table")
    if any(not isinstance(value, str) or not value.strip()
           for value in args.label_names.values()):
        raise ValueError("public label names must be nonempty strings")
    if args.adapter_init_seed is None:
        args.adapter_init_seed = args.seed
    for value, name in (
            (args.epochs, "epochs"),
            (args.batch_size, "batch-size"),
            (args.records_per_client, "records-per-client"),
            (args.max_length, "max-length"),
            (args.max_new_tokens, "max-new-tokens"),
            (args.generation_batch_size, "generation-batch-size"),
            (args.max_attempt_factor, "max-attempt-factor"),
            (args.lora_r, "lora-r"),
            (args.lora_alpha, "lora-alpha")):
        if value <= 0:
            raise ValueError("--%s must be positive" % name)
    if args.samples_per_label is None and args.target_per_label is None:
        args.samples_per_label = 4
    if args.samples_per_label is not None and args.samples_per_label <= 0:
        raise ValueError("--samples-per-label must be positive")
    if args.target_per_label is not None and args.target_per_label <= 0:
        raise ValueError("--target-per-label must be positive")
    if args.target_epsilon is None and args.noise_multiplier is None:
        raise ValueError("one privacy target is required")
    if args.target_epsilon is not None and args.target_epsilon <= 0.0:
        raise ValueError("--target-epsilon must be positive")
    if args.noise_multiplier is not None and args.noise_multiplier <= 0.0:
        raise ValueError("--noise-multiplier must be positive")
    if not (0.0 < args.delta < 1.0):
        raise ValueError("--delta must be in (0, 1)")
    if args.max_grad_norm <= 0.0 or args.epsilon_tolerance <= 0.0:
        raise ValueError("clipping norm and epsilon tolerance must be positive")
    if args.learning_rate <= 0.0 or args.weight_decay < 0.0:
        raise ValueError("optimizer parameters are invalid")
    if not (0.0 <= args.lora_dropout < 1.0):
        raise ValueError("--lora-dropout must be in [0, 1)")
    if not (0.0 < args.top_p <= 1.0) or args.temperature <= 0.0:
        raise ValueError("sampling parameters are invalid")
    if args.top_k < 0 or args.repetition_penalty <= 0.0:
        raise ValueError("generation parameters are invalid")
    if args.min_generated_chars <= 0 or args.min_free_mib < 0:
        raise ValueError("generation length or free-memory threshold is invalid")
    if not (0.0 < args.memory_fraction <= 1.0):
        raise ValueError("--memory-fraction must be in (0, 1]")


def build_parser():
    parser = argparse.ArgumentParser(
        description="Train a fresh record-level DP LoRA per client and generate text."
    )
    parser.add_argument("--client-json-dir", required=True)
    parser.add_argument("--client-ids", nargs="*")
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--records-per-client", type=int, required=True,
                        help="Public fixed record count required for every client")
    parser.add_argument("--seed", type=int, default=57)
    parser.add_argument("--adapter-init-seed", type=int)
    parser.add_argument("--prompt-template", default="Label: {label}\nText:\n")
    quota = parser.add_mutually_exclusive_group()
    quota.add_argument("--samples-per-label", type=int)
    quota.add_argument("--target-per-label", type=int)
    parser.add_argument("--label-names-json", required=True)
    privacy = parser.add_mutually_exclusive_group(required=True)
    privacy.add_argument("--target-epsilon", type=float)
    privacy.add_argument("--noise-multiplier", type=float)
    parser.add_argument("--delta", type=float, default=1e-5)
    parser.add_argument("--max-grad-norm", type=float, default=1.0)
    parser.add_argument("--epsilon-tolerance", type=float, default=0.01)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--learning-rate", type=float, default=5e-4)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--max-length", type=int, default=192)
    parser.add_argument("--lora-r", type=int, default=8)
    parser.add_argument("--lora-alpha", type=int, default=16)
    parser.add_argument("--lora-dropout", type=float, default=0.05)
    parser.add_argument("--lora-target-modules", default="auto")
    parser.add_argument("--generation-batch-size", type=int, default=4)
    parser.add_argument("--max-new-tokens", type=int, default=80)
    parser.add_argument("--max-attempt-factor", type=int, default=20)
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--top-p", type=float, default=0.9)
    parser.add_argument("--top-k", type=int, default=0)
    parser.add_argument("--repetition-penalty", type=float, default=1.05)
    parser.add_argument("--min-generated-chars", type=int, default=3)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--dtype", choices=["auto", "float32", "float16", "bfloat16"],
                        default="auto")
    parser.add_argument("--min-free-mib", type=int, default=3000)
    parser.add_argument("--memory-fraction", type=float, default=0.90)
    parser.add_argument("--allow-model-download", action="store_true")
    parser.add_argument("--deterministic", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser


def main():
    args = build_parser().parse_args()
    validate_args(args)
    requested_client_ids = common.parse_client_ids(args.client_ids)
    clients, client_ids, label_vocab, task_type, source = common.load_staged_clients(
        Path(args.client_json_dir), requested_client_ids
    )
    labels = common.ordered_labels(label_vocab)
    if set(args.label_names) != set(labels):
        raise ValueError("public label names must exactly cover the dataset labels")
    for client_id in client_ids:
        if len(clients[client_id]) != args.records_per_client:
            raise ValueError(
                "client %d has %d records; expected %d" % (
                    client_id, len(clients[client_id]), args.records_per_client
                )
            )
    quotas = allocate_public_quotas(
        client_ids, labels, args.samples_per_label, args.target_per_label
    )
    output_dir = Path(args.output_dir).expanduser().resolve()
    model_fingerprint = common.model_file_fingerprints(args.model_path)
    public_input = {
        "records_per_client": args.records_per_client,
        "client_count": len(client_ids),
        "client_ids": client_ids,
        "labels": labels,
        "generation_quotas": {
            str(client_id): quotas[client_id] for client_id in client_ids
        },
    }
    if args.dry_run:
        payload = {
            "schema_version": 1,
            "status": "dry_run",
            "experiment": "per_client_label_conditioned_record_dp_synthesis",
            "privacy": {
                "mechanism": "poisson_sampled_gaussian_dp_sgd",
                "accountant": "opacus_rdp",
                "target_epsilon": args.target_epsilon,
                "noise_multiplier": args.noise_multiplier,
                "delta": args.delta,
                "max_grad_norm": args.max_grad_norm,
                "unit": "one training record within one selected client",
            },
            "public_input": public_input,
            "source": public_source_provenance(source),
            "model": model_fingerprint,
        }
        output_dir.mkdir(parents=True, exist_ok=True)
        common.atomic_write_json(output_dir / "dry_run_manifest.json", payload)
        print(json.dumps(payload, ensure_ascii=True, indent=2, sort_keys=True))
        return

    output_dir.mkdir(parents=True, exist_ok=True)
    synthetic_path = output_dir / "synthetic.jsonl"
    manifest_path = output_dir / "manifest.json"
    clients_dir = output_dir / "clients"
    if synthetic_path.exists() or manifest_path.exists() or clients_dir.exists():
        raise FileExistsError(
            "output artifacts already exist in %s; use a new directory" % output_dir
        )
    clients_dir.mkdir()

    import opacus
    import peft
    import torch
    import transformers
    from opacus.accountants import RDPAccountant
    from opacus.accountants.utils import get_noise_multiplier
    from peft import LoraConfig, TaskType, get_peft_model
    from transformers import AutoModelForCausalLM, AutoTokenizer

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.deterministic:
        torch.use_deterministic_algorithms(True)
    device = common.choose_device(torch, args.device)
    dtype = common.choose_dtype(torch, args.dtype, device)
    local_files_only = not args.allow_model_download
    tokenizer = AutoTokenizer.from_pretrained(
        args.model_path, local_files_only=local_files_only
    )
    if tokenizer.eos_token is None:
        raise ValueError("causal tokenizer has no eos token")
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    aggregate_rows = []
    global_seen = set()
    client_results = {}
    resolved_targets = None
    for client_id in client_ids:
        training_seed = common.derived_seed(args.seed, "dp_training", client_id)
        random.seed(args.adapter_init_seed)
        torch.manual_seed(args.adapter_init_seed)
        if device.type == "cuda":
            torch.cuda.manual_seed_all(args.adapter_init_seed)
            torch.cuda.empty_cache()
        common.guard_cuda_memory(
            torch, device, args.min_free_mib, args.memory_fraction
        )
        base_model = AutoModelForCausalLM.from_pretrained(
            args.model_path,
            local_files_only=local_files_only,
            torch_dtype=dtype,
        )
        targets = common.select_lora_targets(base_model, args.lora_target_modules)
        if resolved_targets is None:
            resolved_targets = targets
        elif targets != resolved_targets:
            raise RuntimeError("inconsistent LoRA targets across clients")
        random.seed(args.adapter_init_seed)
        torch.manual_seed(args.adapter_init_seed)
        if device.type == "cuda":
            torch.cuda.manual_seed_all(args.adapter_init_seed)
        model = get_peft_model(base_model, LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            r=args.lora_r,
            lora_alpha=args.lora_alpha,
            lora_dropout=args.lora_dropout,
            target_modules=targets,
            bias="none",
        )).to(device)
        optimizer, accounting = dp_train_adapter(
            torch, model, tokenizer, clients[client_id], args, device,
            training_seed, args.prompt_template, RDPAccountant,
            get_noise_multiplier,
        )

        rows = []
        attempts_by_label = {}
        duplicates_by_label = {}
        for label in labels:
            generated, attempts, duplicates, private_matches = common.generate_label(
                torch, model, tokenizer, label, quotas[client_id][label], args,
                device, client_id, global_seen, set(), args.prompt_template, labels,
            )
            if private_matches != 0:
                raise AssertionError("private exact-match filtering must be disabled")
            rows.extend(generated)
            attempts_by_label[label] = attempts
            duplicates_by_label[label] = duplicates
        client_path = clients_dir / ("client_%d.jsonl" % client_id)
        common.atomic_write_jsonl(client_path, rows)
        aggregate_rows.extend(rows)
        client_results[str(client_id)] = {
            "status": "complete",
            "fresh_base_model_loaded": True,
            "fresh_lora_initialized": True,
            "generation_quotas": quotas[client_id],
            "generated_label_counts": dict(sorted(Counter(
                row["label"] for row in rows
            ).items())),
            "generation_attempts": attempts_by_label,
            "duplicates_rejected": duplicates_by_label,
            "accounting": accounting,
            "output_file": str(client_path.relative_to(output_dir)),
            "output_sha256": common.sha256_file(client_path),
        }
        del optimizer
        del model
        del base_model
        gc.collect()
        if device.type == "cuda":
            torch.cuda.empty_cache()

    normalized = [common.normalize_generated_text(row["text"]) for row in aggregate_rows]
    if len(normalized) != len(set(normalized)):
        raise AssertionError("aggregate synthetic records are not globally unique")
    expected = sum(sum(quotas[client_id].values()) for client_id in client_ids)
    if len(aggregate_rows) != expected:
        raise AssertionError("aggregate count does not match public quotas")
    aggregate_counts = dict(sorted(Counter(
        row["label"] for row in aggregate_rows
    ).items()))
    if args.target_per_label is not None:
        for label in labels:
            if aggregate_counts.get(label, 0) != args.target_per_label:
                raise AssertionError("aggregate target was not met for label %r" % label)
    common.atomic_write_jsonl(synthetic_path, aggregate_rows)

    epsilons = [
        result["accounting"]["epsilon"] for result in client_results.values()
    ]
    noise_multipliers = sorted(set(
        result["accounting"]["noise_multiplier"]
        for result in client_results.values()
    ))
    manifest = {
        "schema_version": 1,
        "status": "complete",
        "experiment": "per_client_label_conditioned_record_dp_synthesis",
        "privacy": {
            "mechanism": "poisson_sampled_gaussian_dp_sgd",
            "unit": "one training record within one selected client",
            "adjacency": "add_or_remove_one_record",
            "per_example_global_l2_clipping": True,
            "max_grad_norm": args.max_grad_norm,
            "sampling": "independent_poisson",
            "noise_added_to_clipped_gradient_sum": True,
            "noise_multipliers": noise_multipliers,
            "accountant": "opacus_rdp",
            "target_epsilon": args.target_epsilon,
            "achieved_epsilon_max": max(epsilons),
            "achieved_epsilon_min": min(epsilons),
            "delta": args.delta,
            "composition_across_clients": "parallel_disjoint_partitions",
            "release_is_postprocessing_of_dp_adapters": True,
            "secure_rng": False,
            "secure_rng_scope": "research measurement; not cryptographic deployment",
        },
        "is_record_level_dp": True,
        "client_records_used_for_model_training": True,
        "fresh_base_and_lora_per_client": True,
        "adapter_reuse_between_clients": False,
        "source_split": "train_only",
        "private_exact_match_filter": "disabled_for_dp_release",
        "private_diagnostics_released": False,
        "label_conditioning": {
            "prompt_template": args.prompt_template,
            "public_label_names": args.label_names,
            "public_fixed_generation_labels": True,
            "prompt_tokens_masked_from_training_loss": True,
        },
        "seed": args.seed,
        "shared_adapter_initialization_seed": args.adapter_init_seed,
        "public_input": public_input,
        "label_vocab": label_vocab,
        "task_type": task_type,
        "quota_mode": "aggregate_target" if args.target_per_label is not None else "per_client",
        "samples_per_label": args.samples_per_label,
        "target_per_label": args.target_per_label,
        "global_deduplication": "NFKC + whitespace collapse + casefold",
        "source": public_source_provenance(source),
        "model": model_fingerprint,
        "software": {
            "torch": torch.__version__,
            "transformers": transformers.__version__,
            "peft": peft.__version__,
            "opacus": opacus.__version__,
        },
        "training_parameters": {
            "epochs": args.epochs,
            "batch_size_target": args.batch_size,
            "learning_rate": args.learning_rate,
            "weight_decay": args.weight_decay,
            "max_length": args.max_length,
            "lora_r": args.lora_r,
            "lora_alpha": args.lora_alpha,
            "lora_dropout": args.lora_dropout,
            "lora_target_modules": resolved_targets,
        },
        "generation_parameters": {
            "batch_size": args.generation_batch_size,
            "max_new_tokens": args.max_new_tokens,
            "max_attempt_factor": args.max_attempt_factor,
            "temperature": args.temperature,
            "top_p": args.top_p,
            "top_k": args.top_k,
            "repetition_penalty": args.repetition_penalty,
            "min_generated_chars": args.min_generated_chars,
        },
        "clients": client_results,
        "synthetic_records": len(aggregate_rows),
        "synthetic_label_counts": aggregate_counts,
        "synthetic_file": synthetic_path.name,
        "synthetic_sha256": common.sha256_file(synthetic_path),
    }
    common.atomic_write_json(manifest_path, manifest)
    print(json.dumps({
        "status": "complete",
        "synthetic_file": str(synthetic_path),
        "manifest": str(manifest_path),
        "records": len(aggregate_rows),
        "label_counts": aggregate_counts,
        "epsilon": max(epsilons),
        "delta": args.delta,
    }, ensure_ascii=True, sort_keys=True))


if __name__ == "__main__":
    main()
