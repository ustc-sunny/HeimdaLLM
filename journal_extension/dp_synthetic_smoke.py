#!/usr/bin/env python3
"""Bounded DP-LoRA synthetic-text smoke test for HeimdaLLM+.

This intentionally uses a tiny client sample and an existing local causal LM.
It never writes private text to the output; only generated text and a manifest
are released. The per-process CUDA memory fraction is a hard safety guard.
"""

import argparse
import json
import math
import os
import random
from pathlib import Path

import torch
from peft import LoraConfig, TaskType, get_peft_model
from transformers import AutoModelForCausalLM, AutoTokenizer


def read_samples(path, limit):
    if str(path).endswith(".jsonl"):
        rows = []
        with open(path, "r", encoding="utf-8") as handle:
            for line in handle:
                if line.strip():
                    record = json.loads(line)
                    rows.append((record["text"], str(record["label"])))
                    if len(rows) >= limit:
                        break
        if not rows:
            raise RuntimeError("No samples found in JSONL file")
        return rows
    import h5py
    rows = []
    with h5py.File(path, "r", swmr=True) as handle:
        keys = sorted(handle["X"].keys(), key=int)
        for key in keys[:limit]:
            text = handle["X"][key][()].decode("utf-8")
            label = handle["Y"][key][()].decode("utf-8")
            rows.append((text, label))
    if not rows:
        raise RuntimeError("No samples found in data file")
    return rows


def clip_and_accumulate(model, tokenized, max_grad_norm, accumulator):
    model.zero_grad(set_to_none=True)
    loss = model(**tokenized, labels=tokenized["input_ids"]).loss
    loss.backward()
    trainable = [p for p in model.parameters() if p.requires_grad]
    norm_sq = sum(float((p.grad.detach().float() ** 2).sum()) for p in trainable if p.grad is not None)
    scale = min(1.0, max_grad_norm / (math.sqrt(norm_sq) + 1e-12))
    for index, parameter in enumerate(trainable):
        if parameter.grad is not None:
            accumulator[index].add_(parameter.grad.detach().float().cpu(), alpha=scale)
    model.zero_grad(set_to_none=True)
    return float(loss.detach()), math.sqrt(norm_sq), scale


def gaussian_rdp_upper_bound(steps, noise_multiplier, delta):
    """Conservative composition bound without subsampling amplification."""
    if noise_multiplier <= 0:
        return float("inf"), None
    orders = list(range(2, 65))
    epsilons = [
        (float(order) * steps / (2.0 * noise_multiplier ** 2))
        + math.log(1.0 / delta) / (order - 1.0)
        for order in orders
    ]
    best = min(zip(epsilons, orders))
    return best


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-file", required=True)
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--sample-limit", type=int, default=8)
    parser.add_argument("--clients", type=int, default=2)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--logical-batch-size", type=int, default=2)
    parser.add_argument("--max-length", type=int, default=64)
    parser.add_argument("--max-grad-norm", type=float, default=1.0)
    parser.add_argument("--noise-multiplier", type=float, default=1.0)
    parser.add_argument("--samples-per-label", type=int, default=2)
    parser.add_argument("--seed", type=int, default=57)
    parser.add_argument("--delta", type=float, default=1e-5)
    parser.add_argument("--min-free-mib", type=int, default=20000)
    parser.add_argument("--memory-fraction", type=float, default=0.60)
    args = parser.parse_args()

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for this smoke test")
    device = torch.device("cuda:0")
    free_bytes, total_bytes = torch.cuda.mem_get_info(device)
    free_mib = free_bytes // (1024 * 1024)
    if free_mib < args.min_free_mib:
        raise RuntimeError("GPU free memory below safety threshold: %s MiB" % free_mib)
    torch.cuda.set_per_process_memory_fraction(args.memory_fraction, device)
    torch.cuda.reset_peak_memory_stats(device)

    rows = read_samples(args.data_file, args.sample_limit)
    labels = sorted({label for _, label in rows})
    tokenizer = AutoTokenizer.from_pretrained(args.model_path, local_files_only=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    dtype = torch.float16 if torch.cuda.is_available() else torch.float32
    model = AutoModelForCausalLM.from_pretrained(
        args.model_path, local_files_only=True, torch_dtype=dtype
    ).to(device)
    model.config.pad_token_id = tokenizer.pad_token_id
    model = get_peft_model(model, LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=4,
        lora_alpha=8,
        lora_dropout=0.0,
        target_modules=["q_proj", "v_proj"],
        bias="none",
    )).to(device)
    model.train()
    trainable = [p for p in model.parameters() if p.requires_grad]
    if not trainable:
        raise RuntimeError("LoRA did not create trainable parameters")
    accumulator = [torch.zeros_like(p, dtype=torch.float32, device="cpu") for p in trainable]

    steps = 0
    clip_records = []
    for _ in range(args.epochs):
        for start in range(0, len(rows), args.logical_batch_size):
            batch = rows[start:start + args.logical_batch_size]
            if not batch:
                continue
            for text, _label in batch:
                tokenized = tokenizer(
                    text, return_tensors="pt", truncation=True,
                    max_length=args.max_length,
                )
                tokenized = {k: v.to(device) for k, v in tokenized.items()}
                loss, norm, scale = clip_and_accumulate(
                    model, tokenized, args.max_grad_norm, accumulator
                )
                clip_records.append({"loss": loss, "grad_norm": norm, "clip_scale": scale})
            noise_scale = args.noise_multiplier * args.max_grad_norm
            with torch.no_grad():
                for parameter, summed in zip(trainable, accumulator):
                    noisy = summed.to(device) + torch.randn_like(parameter, dtype=torch.float32) * noise_scale
                    parameter.add_((noisy / float(len(batch))).to(parameter.dtype), alpha=-1e-2)
            for tensor in accumulator:
                tensor.zero_()
            steps += 1

    model.eval()
    generated = []
    for label in labels:
        prompt = "sentiment %s review:" % label
        for offset in range(args.samples_per_label):
            torch.manual_seed(args.seed + offset + len(generated))
            inputs = tokenizer(prompt, return_tensors="pt").to(device)
            with torch.no_grad():
                output = model.generate(
                    **inputs, max_new_tokens=24, do_sample=True,
                    top_p=0.95, temperature=0.9,
                    pad_token_id=tokenizer.pad_token_id,
                )
            text = tokenizer.decode(output[0], skip_special_tokens=True).strip()
            generated.append({"text": text, "label": label})

    # Conservative Gaussian-mechanism upper-bound indicator for the smoke run.
    sample_rate = min(1.0, float(args.logical_batch_size) / max(1, len(rows)))
    approx_epsilon = (steps * sample_rate / max(args.noise_multiplier, 1e-12)) * math.sqrt(
        2.0 * math.log(1.0 / args.delta)
    )
    rdp_epsilon, rdp_order = gaussian_rdp_upper_bound(
        steps, args.noise_multiplier, args.delta
    )
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    with (out / "synthetic.jsonl").open("w", encoding="utf-8") as handle:
        for record in generated:
            handle.write(json.dumps(record, ensure_ascii=True) + "\n")
    manifest = {
        "status": "smoke_only",
        "model_path": os.path.abspath(args.model_path),
        "seed": args.seed,
        "clients": args.clients,
        "rows_used": len(rows),
        "epochs": args.epochs,
        "logical_batch_size": args.logical_batch_size,
        "physical_batch_size": 1,
        "max_length": args.max_length,
        "max_grad_norm": args.max_grad_norm,
        "noise_multiplier": args.noise_multiplier,
        "delta": args.delta,
        "steps": steps,
        "sample_rate": sample_rate,
        "epsilon_indicator": approx_epsilon,
        "epsilon_rdp_upper_bound_no_subsampling": rdp_epsilon,
        "rdp_order": rdp_order,
        "epsilon_accountant": "Gaussian RDP composition upper bound without subsampling amplification",
        "peak_allocated_mib": torch.cuda.max_memory_allocated(device) // (1024 * 1024),
        "peak_reserved_mib": torch.cuda.max_memory_reserved(device) // (1024 * 1024),
        "clip_records": clip_records,
    }
    with (out / "manifest.json").open("w", encoding="utf-8") as handle:
        json.dump(manifest, handle, indent=2, ensure_ascii=True)
    print(json.dumps(manifest, indent=2, ensure_ascii=True))


if __name__ == "__main__":
    main()
