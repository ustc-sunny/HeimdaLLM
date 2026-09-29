#!/usr/bin/env python3
"""FP32 generator controls for DP v1; every trained control is explicitly Non-DP.

Public generation never opens the private staging directory. All arms use public
quotas and the DP-compatible public-only release filter. Private loss/gradient
diagnostics are written outside the result directory and excluded from release.
"""
import argparse
import gc
import json
import math
import random
from collections import Counter
from pathlib import Path

import non_dp_client_synthetic as common
from dp_client_synthetic import allocate_public_quotas, public_source_provenance


MODES = ("public", "ordinary", "fixed_example", "poisson", "clipped")


def train_control(torch, model, tokenizer, rows, args, device, seed):
    encoded = common.encode_training_rows(
        tokenizer, rows, args.max_length, args.prompt_template, args.label_names
    )
    parameters = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(parameters, lr=args.learning_rate,
                                  weight_decay=args.weight_decay)
    collate = common.make_collate(torch, tokenizer.pad_token_id)
    steps_per_epoch = int(math.ceil(len(encoded) / args.batch_size))
    rate = 1.0 / steps_per_epoch
    denominator = rate * len(encoded)
    sampling = torch.Generator(device="cpu").manual_seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(seed)
    sums = [torch.zeros_like(p, dtype=torch.float32) for p in parameters]
    norms, losses, sum_norms, update_norms = [], [], [], []
    clipped, empty_batches, examples = 0, 0, 0
    model.train()
    model.config.use_cache = False
    for epoch in range(args.epochs):
        order = torch.randperm(len(encoded), generator=sampling).tolist() \
            if args.mode in ("ordinary", "fixed_example") else None
        for step in range(steps_per_epoch):
            if order is None:
                selected = (torch.rand(len(encoded), generator=sampling) < rate) \
                    .nonzero(as_tuple=False).flatten().tolist()
            else:
                selected = order[step * args.batch_size:(step + 1) * args.batch_size]
            examples += len(selected)
            empty_batches += int(not selected)
            optimizer.zero_grad(set_to_none=True)
            if args.mode == "ordinary":
                batch = {k: v.to(device) for k, v in
                         collate([encoded[i] for i in selected]).items()}
                loss = model(**batch).loss
                if not torch.isfinite(loss):
                    raise FloatingPointError("ordinary loss is not finite")
                losses.append(loss.detach().item())
                loss.backward()
                sum_norm = math.sqrt(sum(p.grad.detach().float().pow(2).sum().item()
                                         for p in parameters if p.grad is not None))
            else:
                for i in selected:
                    batch = {k: v.to(device) for k, v in collate([encoded[i]]).items()}
                    loss = model(**batch).loss
                    if not torch.isfinite(loss):
                        raise FloatingPointError("per-example loss is not finite")
                    losses.append(loss.detach().item())
                    loss.backward()
                    squared = torch.zeros((), device=device, dtype=torch.float32)
                    for p in parameters:
                        if p.grad is not None:
                            squared.add_(p.grad.detach().float().pow(2).sum())
                    norm = torch.sqrt(squared)
                    norms.append(norm.item())
                    factor = torch.clamp(args.max_grad_norm / (norm + 1e-12), max=1.0) \
                        if args.mode == "clipped" else 1.0
                    clipped += int(args.mode == "clipped" and norm.item() > args.max_grad_norm)
                    for total, p in zip(sums, parameters):
                        if p.grad is not None:
                            total.add_(p.grad.detach().float() * factor)
                    optimizer.zero_grad(set_to_none=True)
                sum_norm = math.sqrt(sum(total.pow(2).sum().item() for total in sums))
                divisor = len(selected) if args.mode == "fixed_example" else denominator
                for total, p in zip(sums, parameters):
                    p.grad = (total / divisor).to(p.dtype)
            before = [p.detach().clone() for p in parameters]
            optimizer.step()
            update_norms.append(math.sqrt(sum(
                (p.detach() - old).float().pow(2).sum().item()
                for p, old in zip(parameters, before))))
            sum_norms.append(sum_norm)
            optimizer.zero_grad(set_to_none=True)
            for total in sums:
                total.zero_()
        print("client training mode=%s epoch=%d/%d complete" %
              (args.mode, epoch + 1, args.epochs), flush=True)
    diagnostics = {
        "privacy_status": "NON_DP_PRIVATE_DEBUG_DO_NOT_PUBLISH_AS_DP",
        "losses": losses, "per_example_gradient_norms": norms,
        "clipped_examples": clipped, "examples_sampled": examples,
        "empty_batches": empty_batches, "gradient_sum_norms": sum_norms,
        "optimizer_update_norms": update_norms,
    }
    return optimizer, {
        "optimizer_steps": args.epochs * steps_per_epoch,
        "steps_per_epoch": steps_per_epoch,
        "sampling": "shuffled_without_replacement" if order is not None else "independent_poisson",
        "sample_rate": None if order is not None else rate,
        "expected_batch_size": denominator,
        "loss_reduction": "mean_target_tokens" if args.mode == "ordinary" else "mean_records",
        "per_example_clipping": args.mode == "clipped",
        "max_grad_norm": args.max_grad_norm if args.mode == "clipped" else None,
        "noise_multiplier": 0.0,
        "sampling_randomness": "public_seed_for_non_dp_ablation_only",
    }, diagnostics


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=MODES, required=True)
    parser.add_argument("--client-json-dir")
    parser.add_argument("--client-ids", default="1,21")
    parser.add_argument("--private-diagnostics-dir")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--seed", type=int, default=57)
    parser.add_argument("--adapter-init-seed", type=int, default=57)
    parser.add_argument("--records-per-client", type=int, default=120)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--learning-rate", type=float, default=5e-4)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--max-grad-norm", type=float, default=1.0)
    parser.add_argument("--max-length", type=int, default=192)
    parser.add_argument("--target-per-label", type=int, default=32)
    parser.add_argument("--max-new-tokens", type=int, default=80)
    parser.add_argument("--device", default="cuda:1")
    args = parser.parse_args()
    if min(args.records_per_client, args.epochs, args.batch_size, args.max_length,
           args.target_per_label, args.max_new_tokens) <= 0 or args.max_grad_norm <= 0:
        parser.error("counts and clipping threshold must be positive")
    args.label_names = {"1": "World", "2": "Sports", "3": "Business",
                        "4": "Science and Technology"}
    args.prompt_template = "Category: {label}\nNews article:\n"
    args.lora_r, args.lora_alpha, args.lora_dropout = 8, 16, 0.05
    args.lora_target_modules = "auto"
    args.generation_batch_size, args.max_attempt_factor = 4, 20
    args.temperature, args.top_p, args.top_k = 0.8, 0.9, 0
    args.repetition_penalty, args.min_generated_chars = 1.05, 3
    args.memory_fraction, args.min_free_mib = 0.90, 3000
    output = Path(args.output_dir).resolve()
    if output.exists():
        raise FileExistsError("fresh output directory required: %s" % output)
    ids = [int(i) for i in args.client_ids.split(",")]
    if len(set(ids)) != len(ids):
        parser.error("duplicate client IDs")
    labels = list(args.label_names)
    source, clients = None, None
    private_dir = None
    if args.mode != "public":
        if not args.client_json_dir or not args.private_diagnostics_dir:
            parser.error("trained controls require staging and separate private diagnostics")
        private_dir = Path(args.private_diagnostics_dir).resolve()
        if private_dir == output or output in private_dir.parents:
            parser.error("private diagnostics must be outside released results")
        private_dir.mkdir(parents=True, exist_ok=True, mode=0o700)
        clients, loaded_ids, vocab, task, source = common.load_staged_clients(
            Path(args.client_json_dir), ids)
        if loaded_ids != ids or set(vocab) != set(labels):
            raise ValueError("staging labels/client IDs mismatch")
        if any(len(clients[i]) != args.records_per_client for i in ids):
            raise ValueError("fixed record count mismatch")
    # No staging file is opened by the public arm, including no match filter.
    import peft
    import torch
    import transformers
    from peft import LoraConfig, TaskType, get_peft_model
    from transformers import AutoModelForCausalLM, AutoTokenizer
    device = common.choose_device(torch, args.device)
    tokenizer = AutoTokenizer.from_pretrained(args.model_path, local_files_only=True)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"
    quotas = allocate_public_quotas(ids, labels, None, args.target_per_label)
    output.mkdir(parents=True)
    (output / "clients").mkdir()
    records, seen, results = [], set(), {}
    for client in ids:
        random.seed(args.adapter_init_seed)
        torch.manual_seed(args.adapter_init_seed)
        if device.type == "cuda":
            torch.cuda.manual_seed_all(args.adapter_init_seed)
        common.guard_cuda_memory(torch, device, args.min_free_mib, args.memory_fraction)
        base = AutoModelForCausalLM.from_pretrained(args.model_path,
                                                  local_files_only=True,
                                                  torch_dtype=torch.float32)
        optimizer = None
        if args.mode == "public":
            model = base.to(device)
            training = {"optimizer_steps": 0, "client_records_accessed": False}
        else:
            targets = common.select_lora_targets(base, args.lora_target_modules)
            torch.manual_seed(args.adapter_init_seed)
            if device.type == "cuda":
                torch.cuda.manual_seed_all(args.adapter_init_seed)
            model = get_peft_model(base, LoraConfig(task_type=TaskType.CAUSAL_LM,
                r=args.lora_r, lora_alpha=args.lora_alpha, lora_dropout=args.lora_dropout,
                target_modules=targets, bias="none")).to(device)
            optimizer, training, diagnostics = train_control(
                torch, model, tokenizer, clients[client], args, device,
                common.derived_seed(args.seed, "dp_training", client))
            common.atomic_write_json(private_dir / ("client_%d.json" % client), diagnostics)
        rows, attempts, duplicates = [], {}, {}
        for label in labels:
            generated, attempt, duplicate, private_match = common.generate_label(
                torch, model, tokenizer, label, quotas[client][label], args, device,
                client, seen, set(), args.prompt_template, labels)
            assert private_match == 0
            rows.extend(generated)
            attempts[label], duplicates[label] = attempt, duplicate
        path = output / "clients" / ("client_%d.jsonl" % client)
        common.atomic_write_jsonl(path, rows)
        records.extend(rows)
        results[str(client)] = {"training": training, "generation_attempts": attempts,
            "duplicates_rejected": duplicates, "output_sha256": common.sha256_file(path)}
        print("client=%d mode=%s generated=%d" % (client, args.mode, len(rows)), flush=True)
        del optimizer, model, base
        gc.collect()
        if device.type == "cuda":
            torch.cuda.empty_cache()
    if len(records) != args.target_per_label * len(labels):
        raise AssertionError("public quota mismatch")
    path = output / "synthetic.jsonl"
    common.atomic_write_jsonl(path, records)
    parameters = {k: getattr(args, k) for k in (
        "epochs", "batch_size", "learning_rate", "weight_decay", "max_length",
        "lora_r", "lora_alpha", "lora_dropout", "lora_target_modules")}
    manifest = {"schema_version": 1, "status": "complete", "mode": args.mode,
        "is_record_level_dp": False, "privacy_guarantee": "none",
        "client_records_used_for_model_training": args.mode != "public",
        "generator_reads_private_records": args.mode != "public",
        "private_exact_match_filter": "disabled_public_only_filter_all_arms",
        "private_diagnostics_released": False, "dtype": "float32", "seed": args.seed,
        "shared_adapter_initialization_seed": args.adapter_init_seed,
        "public_client_ids": ids, "generation_quotas": quotas,
        "training_parameters": parameters,
        "generation_parameters": {k: getattr(args, k) for k in (
            "generation_batch_size", "max_new_tokens", "max_attempt_factor",
            "temperature", "top_p", "top_k", "repetition_penalty", "min_generated_chars")},
        "prompt_template": args.prompt_template, "public_label_names": args.label_names,
        "source": public_source_provenance(source) if source else None,
        "model": common.model_file_fingerprints(args.model_path),
        "software": {"torch": torch.__version__, "peft": peft.__version__,
                     "transformers": transformers.__version__},
        "clients": results, "synthetic_records": len(records),
        "synthetic_label_counts": dict(Counter(row["label"] for row in records)),
        "synthetic_sha256": common.sha256_file(path)}
    common.atomic_write_json(output / "manifest.json", manifest)


if __name__ == "__main__":
    main()
