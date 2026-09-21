#!/usr/bin/env python3
"""Generate label-conditioned synthetic text with an independent LoRA per client.

The default path consumes the train-only JSONL staging directory produced by
``export_client_train_jsonl.py``.  Direct HDF5 input is also supported when the
active Python environment provides h5py.  For every client this script reloads
the same base causal LM, initializes a new LoRA adapter, trains with ordinary
AdamW, generates that client's quota, then destroys the model before moving to
the next client.

This is a true non-DP baseline: it applies neither per-example/batch gradient
clipping nor noise.  Generated text is released; private input text is never
copied into the output directory or manifest.
"""

import argparse
import gc
import hashlib
import json
import math
import os
import random
import re
import time
import unicodedata
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


def sha256_json(payload):
    encoded = json.dumps(
        payload, ensure_ascii=True, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def atomic_write_jsonl(path, rows):
    temporary = path.with_name(path.name + ".tmp")
    if temporary.exists():
        temporary.unlink()
    with temporary.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=True, sort_keys=True) + "\n")
        handle.flush()
        os.fsync(handle.fileno())
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
    os.replace(str(temporary), str(path))


def parse_client_ids(values, required=False):
    result = []
    for value in values or []:
        for item in str(value).split(","):
            item = item.strip()
            if item:
                result.append(int(item))
    if required and not result:
        raise ValueError("--client-ids must contain at least one client id")
    if len(result) != len(set(result)):
        raise ValueError("--client-ids contains duplicates")
    if any(client_id < 0 for client_id in result):
        raise ValueError("client ids must be non-negative")
    return result


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


def json_from_h5_scalar(value):
    if isinstance(value, bytes):
        value = value.decode("utf-8")
    elif not isinstance(value, str):
        value = value.tobytes().decode("utf-8")
    return json.loads(value)


def load_jsonl(path):
    rows = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except ValueError as error:
                raise ValueError("invalid JSON at %s:%d: %s" % (path, line_number, error))
            text = row.get("text")
            label = row.get("label")
            if not isinstance(text, str) or not text.strip():
                raise ValueError("%s:%d has empty/non-string text" % (path, line_number))
            if not isinstance(label, (str, int, float, bool)):
                raise ValueError("%s:%d has an unsupported label" % (path, line_number))
            rows.append({"text": text, "label": str(label)})
    if not rows:
        raise ValueError("%s contains no records" % path)
    return rows


def path_is_within(path, directory):
    try:
        return os.path.commonpath([str(path), str(directory)]) == str(directory)
    except ValueError:
        return False


def load_staged_clients(directory, requested_client_ids):
    directory = directory.expanduser().resolve()
    manifest_path = directory / "manifest.json"
    if not manifest_path.is_file():
        raise FileNotFoundError("staging manifest is missing: %s" % manifest_path)
    with manifest_path.open("r", encoding="utf-8") as handle:
        staging_manifest = json.load(handle)
    if staging_manifest.get("status") != "complete":
        raise ValueError("staging manifest status must be complete")
    if staging_manifest.get("source_split") != "train_only":
        raise ValueError("staging manifest must declare source_split=train_only")
    manifest_ids = [int(value) for value in staging_manifest.get("selected_client_ids", [])]
    client_ids = requested_client_ids or manifest_ids
    if not client_ids:
        raise ValueError("staging manifest contains no selected clients")
    if any(client_id not in manifest_ids for client_id in client_ids):
        raise ValueError("requested client id is absent from the staging manifest")

    source = staging_manifest.get("source", {})
    label_vocab = source.get("label_vocab")
    if not isinstance(label_vocab, dict) or not label_vocab:
        raise ValueError("staging manifest is missing source.label_vocab")
    clients = {}
    input_files = {}
    client_metadata = staging_manifest.get("clients", {})
    for client_id in client_ids:
        metadata = client_metadata.get(str(client_id))
        if not isinstance(metadata, dict):
            raise ValueError("staging manifest is missing client %d" % client_id)
        relative_path = metadata.get("path")
        if not isinstance(relative_path, str):
            raise ValueError("staging path for client %d is invalid" % client_id)
        file_path = (directory / relative_path).resolve()
        if not path_is_within(file_path, directory):
            raise ValueError("staging file escapes its directory: %s" % relative_path)
        if not file_path.is_file():
            raise FileNotFoundError(str(file_path))
        actual_hash = sha256_file(file_path)
        expected_hash = metadata.get("sha256")
        if expected_hash and actual_hash != expected_hash:
            raise ValueError("staging hash mismatch for client %d" % client_id)
        rows = load_jsonl(file_path)
        if int(metadata.get("records", len(rows))) != len(rows):
            raise ValueError("staging record count mismatch for client %d" % client_id)
        unknown_labels = sorted(set(row["label"] for row in rows) - set(label_vocab))
        if unknown_labels:
            raise ValueError("client %d has labels outside label_vocab: %s" % (
                client_id, unknown_labels
            ))
        clients[client_id] = rows
        input_files[str(client_id)] = {
            "path": str(file_path),
            "sha256": actual_hash,
            "records": len(rows),
        }
    source_descriptor = {
        "kind": "staged_client_jsonl",
        "staging_manifest": str(manifest_path),
        "staging_manifest_sha256": sha256_file(manifest_path),
        "original_data_file": source.get("data_file"),
        "original_data_sha256": source.get("data_sha256"),
        "original_partition_file": source.get("partition_file"),
        "original_partition_sha256": source.get("partition_sha256"),
        "partition_method": source.get("partition_method"),
        "input_files": input_files,
    }
    return clients, client_ids, label_vocab, source.get(
        "task_type", "text_classification"
    ), source_descriptor


def load_h5_clients(data_path, partition_path, partition_method, client_ids):
    # Imported only for direct-H5 mode; the PEFT environment need not install h5py.
    import h5py

    data_path = data_path.expanduser().resolve()
    partition_path = partition_path.expanduser().resolve()
    if not data_path.is_file():
        raise FileNotFoundError(str(data_path))
    if not partition_path.is_file():
        raise FileNotFoundError(str(partition_path))
    clients = {}
    selected_owner = {}
    with h5py.File(str(data_path), "r", swmr=True) as data_handle, h5py.File(
            str(partition_path), "r", swmr=True) as partition_handle:
        attributes = json_from_h5_scalar(data_handle["attributes"][()])
        label_vocab = attributes.get("label_vocab")
        if not isinstance(label_vocab, dict) or not label_vocab:
            raise ValueError("source attributes.label_vocab must be a non-empty object")
        if partition_method not in partition_handle:
            raise KeyError("partition method not found: %s" % partition_method)
        method_group = partition_handle[partition_method]
        n_clients = int(method_group["n_clients"][()])
        source_train = set(int(value) for value in attributes.get("train_index_list", []))
        if not source_train:
            raise ValueError("source attributes.train_index_list is empty")
        for client_id in client_ids:
            if client_id >= n_clients:
                raise ValueError("client %d is outside n_clients=%d" % (client_id, n_clients))
            group = method_group["partition_data"].get(str(client_id))
            if group is None or "train" not in group:
                raise KeyError("client %d train partition is missing" % client_id)
            indices = [int(value) for value in group["train"][()]]
            if not indices or len(indices) != len(set(indices)):
                raise ValueError("client %d train indices are empty or duplicated" % client_id)
            if any(index not in source_train for index in indices):
                raise ValueError("client %d includes a non-training source index" % client_id)
            for index in indices:
                if index in selected_owner:
                    raise ValueError("selected client train partitions overlap")
                selected_owner[index] = client_id
            rows = []
            for index in indices:
                key = str(index)
                text = decode_scalar(data_handle["X"][key][()])
                label = decode_scalar(data_handle["Y"][key][()])
                if not text.strip() or label not in label_vocab:
                    raise ValueError("invalid source record for client %d" % client_id)
                rows.append({"text": text, "label": label})
            clients[client_id] = rows
        task_type = attributes.get("task_type", "text_classification")
    source_descriptor = {
        "kind": "direct_h5",
        "data_file": str(data_path),
        "data_sha256": sha256_file(data_path),
        "partition_file": str(partition_path),
        "partition_sha256": sha256_file(partition_path),
        "partition_method": partition_method,
    }
    return clients, client_ids, label_vocab, task_type, source_descriptor


def ordered_labels(label_vocab):
    def key(label):
        value = label_vocab[label]
        try:
            return (0, int(value), str(label))
        except (TypeError, ValueError):
            return (1, str(value), str(label))
    return sorted((str(label) for label in label_vocab), key=key)


def allocate_quotas(clients, client_ids, labels, samples_per_label,
                    target_per_label, require_all_labels):
    observed = {
        client_id: set(row["label"] for row in clients[client_id])
        for client_id in client_ids
    }
    if require_all_labels:
        missing = {
            str(client_id): sorted(set(labels) - observed[client_id])
            for client_id in client_ids if set(labels) - observed[client_id]
        }
        if missing:
            raise ValueError("selected clients do not observe every label: %s" % missing)

    quotas = {client_id: {label: 0 for label in labels} for client_id in client_ids}
    if target_per_label is not None:
        ordered_clients = sorted(client_ids)
        for label_position, label in enumerate(labels):
            eligible = [client_id for client_id in ordered_clients if label in observed[client_id]]
            if not eligible:
                raise ValueError("no selected client observes label %r" % label)
            quotient, remainder = divmod(target_per_label, len(eligible))
            # Rotate remainder ownership by label so small imbalances do not always
            # fall on the same client.
            rotated = eligible[label_position % len(eligible):] + eligible[:label_position % len(eligible)]
            extra = set(rotated[:remainder])
            for client_id in eligible:
                quotas[client_id][label] = quotient + (1 if client_id in extra else 0)
    else:
        for client_id in client_ids:
            for label in observed[client_id]:
                quotas[client_id][label] = samples_per_label
    return quotas, observed


def normalize_prompt_template(template):
    # Command files often pass backslash-n literally; accept that representation
    # without interpreting arbitrary Python/unicode escape sequences.
    template = template.replace("\\n", "\n").replace("\\t", "\t")
    if template.count("{label}") != 1:
        raise ValueError("--prompt-template must contain {label} exactly once")
    try:
        template.format(label="probe")
    except (KeyError, IndexError, ValueError) as error:
        raise ValueError("invalid --prompt-template: %s" % error)
    return template


def prompt_for_label(label, prompt_template, label_names=None):
    return prompt_template.format(label=(label_names or {}).get(str(label), label))


def normalize_generated_text(text):
    text = unicodedata.normalize("NFKC", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text.casefold()


def normalized_text_sha256(text):
    return hashlib.sha256(normalize_generated_text(text).encode("utf-8")).hexdigest()


def clean_continuation(text, prompt_template, labels, label_names=None):
    # Prevent a sampled second training record from entering the released item.
    markers = ["\nLabel:", "\nText:"]
    template_prefix = prompt_template.split("{label}", 1)[0].strip()
    if template_prefix:
        markers.append("\n" + template_prefix)
    for label in labels:
        rendered = prompt_for_label(label, prompt_template, label_names)
        markers.extend([rendered, "\n" + rendered])
    for marker in markers:
        if marker in text:
            text = text.split(marker, 1)[0]
    return re.sub(r"\s+", " ", text).strip()


def derived_seed(base_seed, *parts):
    material = "|".join([str(base_seed)] + [str(part) for part in parts])
    value = int(hashlib.sha256(material.encode("utf-8")).hexdigest()[:16], 16)
    return value % 2147483647


def model_file_fingerprints(model_path):
    path = Path(model_path).expanduser()
    if not path.exists():
        return {"identifier": model_path, "local_files": {}}
    path = path.resolve()
    names = [
        "config.json", "generation_config.json", "tokenizer.json",
        "tokenizer_config.json", "special_tokens_map.json",
    ]
    files = {}
    for name in names:
        candidate = path / name
        if candidate.is_file():
            files[name] = sha256_file(candidate)
    return {
        "identifier": str(path),
        "local_files": files,
        "metadata_sha256": sha256_json(files),
    }


def choose_device(torch, requested):
    if requested == "auto":
        return torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    device = torch.device(requested)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    return device


def choose_dtype(torch, requested, device):
    if requested == "auto":
        if device.type == "cuda":
            supports_bf16 = getattr(torch.cuda, "is_bf16_supported", lambda: False)()
            return torch.bfloat16 if supports_bf16 else torch.float16
        return torch.float32
    mapping = {
        "float32": torch.float32,
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
    }
    dtype = mapping[requested]
    if device.type == "cpu" and dtype == torch.float16:
        raise ValueError("float16 is unsupported for this CPU training path")
    return dtype


def guard_cuda_memory(torch, device, min_free_mib, memory_fraction):
    if device.type != "cuda":
        return None
    free_bytes, total_bytes = torch.cuda.mem_get_info(device)
    free_mib = int(free_bytes // (1024 * 1024))
    if free_mib < min_free_mib:
        raise RuntimeError(
            "GPU free memory is %d MiB, below --min-free-mib=%d"
            % (free_mib, min_free_mib)
        )
    if memory_fraction < 1.0:
        torch.cuda.set_per_process_memory_fraction(memory_fraction, device)
    return {
        "free_mib_before_load": free_mib,
        "total_mib": int(total_bytes // (1024 * 1024)),
    }


def select_lora_targets(model, requested):
    if requested != "auto":
        targets = [item.strip() for item in requested.split(",") if item.strip()]
        if not targets:
            raise ValueError("--lora-target-modules is empty")
        return targets
    module_suffixes = set(name.rsplit(".", 1)[-1] for name, _ in model.named_modules())
    if {"q_proj", "v_proj"}.issubset(module_suffixes):
        return ["q_proj", "v_proj"]
    if "c_attn" in module_suffixes:
        targets = ["c_attn"]
        if "c_proj" in module_suffixes:
            targets.append("c_proj")
        return targets
    raise ValueError(
        "could not infer LoRA targets; pass --lora-target-modules explicitly"
    )


def encode_training_rows(tokenizer, rows, max_length, prompt_template, label_names=None):
    encoded = []
    eos_id = tokenizer.eos_token_id
    if eos_id is None:
        raise ValueError("causal tokenizer has no eos_token_id")
    for row in rows:
        prompt_ids = tokenizer.encode(
            prompt_for_label(row["label"], prompt_template, label_names), add_special_tokens=True
        )
        text_ids = tokenizer.encode(row["text"].strip(), add_special_tokens=False)
        text_ids.append(eos_id)
        if len(prompt_ids) >= max_length:
            raise ValueError("--max-length is too short for the label prompt")
        text_ids = text_ids[:max_length - len(prompt_ids)]
        if not text_ids:
            raise ValueError("a training record has no target tokens after truncation")
        input_ids = prompt_ids + text_ids
        encoded.append({
            "input_ids": input_ids,
            "attention_mask": [1] * len(input_ids),
            "labels": [-100] * len(prompt_ids) + list(text_ids),
        })
    return encoded


def make_collate(torch, pad_token_id):
    def collate(features):
        max_size = max(len(feature["input_ids"]) for feature in features)
        input_ids = []
        attention_masks = []
        labels = []
        for feature in features:
            padding = max_size - len(feature["input_ids"])
            input_ids.append(feature["input_ids"] + [pad_token_id] * padding)
            attention_masks.append(feature["attention_mask"] + [0] * padding)
            labels.append(feature["labels"] + [-100] * padding)
        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "attention_mask": torch.tensor(attention_masks, dtype=torch.long),
            "labels": torch.tensor(labels, dtype=torch.long),
        }
    return collate


def train_adapter(torch, model, tokenizer, rows, args, device, training_seed,
                  prompt_template):
    encoded = encode_training_rows(
        tokenizer, rows, args.max_length, prompt_template, args.label_names
    )
    generator = torch.Generator()
    generator.manual_seed(training_seed)
    loader = torch.utils.data.DataLoader(
        encoded,
        batch_size=args.batch_size,
        shuffle=True,
        generator=generator,
        collate_fn=make_collate(torch, tokenizer.pad_token_id),
        num_workers=0,
        drop_last=False,
    )
    trainable = [parameter for parameter in model.parameters() if parameter.requires_grad]
    if not trainable:
        raise RuntimeError("LoRA created no trainable parameters")
    optimizer = torch.optim.AdamW(
        trainable, lr=args.learning_rate, weight_decay=args.weight_decay
    )
    model.train()
    model.config.use_cache = False
    losses = []
    optimizer_steps = 0
    optimizer.zero_grad(set_to_none=True)
    batches_since_step = 0
    started = time.time()
    for _epoch in range(args.epochs):
        for batch_index, batch in enumerate(loader):
            batch = {key: value.to(device) for key, value in batch.items()}
            loss = model(**batch).loss
            if not torch.isfinite(loss):
                raise RuntimeError("training loss became non-finite")
            losses.append(float(loss.detach().cpu()))
            (loss / args.gradient_accumulation_steps).backward()
            batches_since_step += 1
            is_last = batch_index + 1 == len(loader)
            if batches_since_step == args.gradient_accumulation_steps or is_last:
                # True non-DP baseline: deliberately no clip_grad_norm_ call and
                # no random noise is applied to gradients or parameters.
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
                optimizer_steps += 1
                batches_since_step = 0
    return optimizer, {
        "epochs": args.epochs,
        "examples": len(rows),
        "batches": len(losses),
        "optimizer_steps": optimizer_steps,
        "mean_loss": sum(losses) / len(losses),
        "first_loss": losses[0],
        "last_loss": losses[-1],
        "min_loss": min(losses),
        "max_loss": max(losses),
        "seconds": round(time.time() - started, 6),
    }


def generate_label(torch, model, tokenizer, label, quota, args, device,
                   client_id, global_seen, private_text_hashes,
                   prompt_template, all_labels):
    if quota <= 0:
        return [], 0, 0, 0
    prompt = prompt_for_label(label, prompt_template, args.label_names)
    accepted = []
    attempts = 0
    rejected_duplicates = 0
    rejected_private_matches = 0
    maximum_attempts = max(quota, quota * args.max_attempt_factor)
    model.eval()
    model.config.use_cache = True
    while len(accepted) < quota and attempts < maximum_attempts:
        batch_size = min(
            args.generation_batch_size,
            quota - len(accepted),
            maximum_attempts - attempts,
        )
        generation_seed = derived_seed(args.seed, client_id, label, attempts)
        random.seed(generation_seed)
        torch.manual_seed(generation_seed)
        if device.type == "cuda":
            torch.cuda.manual_seed_all(generation_seed)
        tokenized = tokenizer(
            [prompt] * batch_size, return_tensors="pt", padding=True,
            truncation=True, max_length=args.max_length,
        )
        tokenized = {key: value.to(device) for key, value in tokenized.items()}
        input_width = int(tokenized["input_ids"].shape[1])
        with torch.no_grad():
            generated = model.generate(
                **tokenized,
                max_new_tokens=args.max_new_tokens,
                do_sample=True,
                top_p=args.top_p,
                top_k=args.top_k,
                temperature=args.temperature,
                repetition_penalty=args.repetition_penalty,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
            )
        attempts += batch_size
        for sequence in generated:
            continuation = tokenizer.decode(
                sequence[input_width:], skip_special_tokens=True
            )
            text = clean_continuation(continuation, prompt_template, all_labels, args.label_names)
            normalized = normalize_generated_text(text)
            if len(text) < args.min_generated_chars or not normalized:
                continue
            if hashlib.sha256(normalized.encode("utf-8")).hexdigest() in private_text_hashes:
                rejected_private_matches += 1
                continue
            if normalized in global_seen:
                rejected_duplicates += 1
                continue
            global_seen.add(normalized)
            accepted.append({
                "client_id": client_id,
                "label": label,
                "text": text,
                "generation_seed": generation_seed,
            })
            if len(accepted) >= quota:
                break
    if len(accepted) != quota:
        raise RuntimeError(
            "client %d label %r produced %d/%d unique samples after %d attempts"
            % (client_id, label, len(accepted), quota, attempts)
        )
    return accepted, attempts, rejected_duplicates, rejected_private_matches


def validate_args(args):
    args.prompt_template = normalize_prompt_template(args.prompt_template)
    args.label_names = json.loads(args.label_names_json)
    if not isinstance(args.label_names, dict) or any(
            not isinstance(v, str) or not v.strip() for v in args.label_names.values()):
        raise ValueError("--label-names-json must be an object of nonempty names")
    if args.adapter_init_seed is None:
        args.adapter_init_seed = args.seed
    if args.epochs <= 0 or args.batch_size <= 0:
        raise ValueError("--epochs and --batch-size must be positive")
    if args.gradient_accumulation_steps <= 0:
        raise ValueError("--gradient-accumulation-steps must be positive")
    if args.max_length <= 8 or args.max_new_tokens <= 0:
        raise ValueError("sequence length arguments are invalid")
    if args.generation_batch_size <= 0 or args.max_attempt_factor <= 0:
        raise ValueError("generation batch size and attempt factor must be positive")
    if args.samples_per_label is not None and args.samples_per_label <= 0:
        raise ValueError("--samples-per-label must be positive")
    if args.target_per_label is not None and args.target_per_label <= 0:
        raise ValueError("--target-per-label must be positive")
    if args.samples_per_label is not None and args.target_per_label is not None:
        raise ValueError("choose only one of --samples-per-label and --target-per-label")
    if args.samples_per_label is None and args.target_per_label is None:
        args.samples_per_label = 4
    if args.learning_rate <= 0 or args.weight_decay < 0:
        raise ValueError("optimizer parameters are invalid")
    if args.lora_r <= 0 or args.lora_alpha <= 0 or not (0.0 <= args.lora_dropout < 1.0):
        raise ValueError("LoRA parameters are invalid")
    if args.top_k < 0 or args.repetition_penalty <= 0 or args.min_generated_chars <= 0:
        raise ValueError("generation parameters are invalid")
    if args.min_free_mib < 0:
        raise ValueError("--min-free-mib must be non-negative")
    if not (0.0 < args.memory_fraction <= 1.0):
        raise ValueError("--memory-fraction must be in (0, 1]")
    if not (0.0 < args.top_p <= 1.0) or args.temperature <= 0:
        raise ValueError("sampling parameters are invalid")


def build_parser():
    parser = argparse.ArgumentParser(
        description="Train a fresh true-non-DP LoRA per client and generate synthetic text."
    )
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--client-json-dir",
                        help="Private train-only staging directory from export_client_train_jsonl.py")
    source.add_argument("--data-file", help="Direct source H5 (requires h5py in this env)")
    parser.add_argument("--partition-file")
    parser.add_argument("--partition-method")
    parser.add_argument("--client-ids", nargs="*",
                        help="Optional staged subset, or required comma/space-separated IDs for H5")
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--seed", type=int, default=57)
    parser.add_argument("--adapter-init-seed", type=int, default=None,
                        help="Shared LoRA initialization seed for every client (default: --seed)")
    parser.add_argument("--prompt-template", default="Label: {label}\nText:\n",
                        help="Training/generation prompt containing exactly one {label}")
    quota = parser.add_mutually_exclusive_group()
    quota.add_argument("--samples-per-label", type=int, default=None,
                       help="Quota per observed label for every client (default: 4)")
    quota.add_argument("--target-per-label", type=int, default=None,
                       help="Exact aggregate quota per label, divided across eligible clients")
    parser.add_argument("--label-names-json", default="{}",
                        help="Public raw-label to category-name JSON mapping; output labels are unchanged")
    parser.add_argument("--require-all-labels", action="store_true")
    parser.add_argument(
        "--public-generator-control", action="store_true",
        help=(
            "Generate directly from the pretrained base model without constructing "
            "or training LoRA. Client records are read only for matched quotas and "
            "the existing exact-match release filter."
        ),
    )
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=1)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--max-length", type=int, default=128)
    parser.add_argument("--lora-r", type=int, default=4)
    parser.add_argument("--lora-alpha", type=int, default=8)
    parser.add_argument("--lora-dropout", type=float, default=0.0)
    parser.add_argument("--lora-target-modules", default="auto")
    parser.add_argument("--generation-batch-size", type=int, default=4)
    parser.add_argument("--max-new-tokens", type=int, default=48)
    parser.add_argument("--max-attempt-factor", type=int, default=20)
    parser.add_argument("--temperature", type=float, default=0.9)
    parser.add_argument("--top-p", type=float, default=0.95)
    parser.add_argument("--top-k", type=int, default=50)
    parser.add_argument("--repetition-penalty", type=float, default=1.05)
    parser.add_argument("--min-generated-chars", type=int, default=3)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--dtype", choices=["auto", "float32", "float16", "bfloat16"],
                        default="auto")
    parser.add_argument("--min-free-mib", type=int, default=3000)
    parser.add_argument("--memory-fraction", type=float, default=0.90)
    parser.add_argument("--allow-model-download", action="store_true",
                        help="Allow Hugging Face network access; local-only is the default")
    parser.add_argument("--deterministic", action="store_true")
    parser.add_argument("--dry-run", action="store_true",
                        help="Validate inputs and quotas without importing torch/transformers/peft")
    return parser


def main():
    args = build_parser().parse_args()
    validate_args(args)
    started = time.time()
    requested_client_ids = parse_client_ids(args.client_ids)
    if args.client_json_dir:
        clients, client_ids, label_vocab, task_type, source_descriptor = load_staged_clients(
            Path(args.client_json_dir), requested_client_ids
        )
    else:
        if not args.partition_file or not args.partition_method:
            raise ValueError(
                "direct H5 mode requires --partition-file and --partition-method"
            )
        client_ids = parse_client_ids(args.client_ids, required=True)
        clients, client_ids, label_vocab, task_type, source_descriptor = load_h5_clients(
            Path(args.data_file), Path(args.partition_file), args.partition_method, client_ids
        )
    labels = ordered_labels(label_vocab)
    if args.label_names and set(args.label_names) != set(labels):
        raise ValueError("label names must exactly cover the source label vocabulary")
    quotas, observed = allocate_quotas(
        clients, client_ids, labels, args.samples_per_label,
        args.target_per_label, args.require_all_labels,
    )
    input_summary = {
        str(client_id): {
            "records": len(clients[client_id]),
            "label_counts": dict(sorted(Counter(
                row["label"] for row in clients[client_id]
            ).items())),
            "generation_quotas": quotas[client_id],
        }
        for client_id in client_ids
    }
    output_dir = Path(args.output_dir).expanduser().resolve()
    model_fingerprint = model_file_fingerprints(args.model_path)
    # Keep only normalized-text hashes in memory.  Neither the set nor any
    # individual hash is released in the manifest.
    private_text_hashes = set(
        normalized_text_sha256(row["text"])
        for client_id in client_ids for row in clients[client_id]
    )
    dry_run_payload = {
        "schema_version": 1,
        "status": "dry_run",
        "privacy": {
            "mechanism": "none",
            "gradient_clipping": False,
            "noise_added": False,
            "noise_multiplier": 0.0,
        },
        "public_generator_control": args.public_generator_control,
        "client_records_used_for_model_training": not args.public_generator_control,
        "fresh_base_and_lora_per_client": not args.public_generator_control,
        "shared_adapter_initialization_seed": args.adapter_init_seed,
        "label_conditioning": {
            "prompt_template": args.prompt_template,
            "public_label_names": args.label_names,
            "prompt_tokens_masked_from_training_loss": True,
        },
        "private_exact_match_filter": "NFKC + whitespace collapse + casefold + SHA-256 (in memory only)",
        "client_ids": client_ids,
        "labels": labels,
        "label_vocab": label_vocab,
        "task_type": task_type,
        "source": source_descriptor,
        "model": model_fingerprint,
        "clients": input_summary,
        "expected_synthetic_records": sum(
            sum(client_quota.values()) for client_quota in quotas.values()
        ),
    }
    if args.dry_run:
        output_dir.mkdir(parents=True, exist_ok=True)
        dry_path = output_dir / "dry_run_manifest.json"
        atomic_write_json(dry_path, dry_run_payload)
        print(json.dumps(dry_run_payload, ensure_ascii=True, indent=2, sort_keys=True))
        return

    output_dir.mkdir(parents=True, exist_ok=True)
    synthetic_path = output_dir / "synthetic.jsonl"
    manifest_path = output_dir / "manifest.json"
    clients_dir = output_dir / "clients"
    if synthetic_path.exists() or manifest_path.exists() or clients_dir.exists():
        raise FileExistsError(
            "output artifacts already exist in %s; use a new output directory" % output_dir
        )
    clients_dir.mkdir()

    # Heavy imports intentionally occur only after input validation and dry-run.
    import torch
    import transformers
    import peft
    from peft import LoraConfig, TaskType, get_peft_model
    from transformers import AutoModelForCausalLM, AutoTokenizer

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.deterministic:
        torch.use_deterministic_algorithms(True)
    device = choose_device(torch, args.device)
    dtype = choose_dtype(torch, args.dtype, device)
    local_files_only = not args.allow_model_download
    tokenizer = AutoTokenizer.from_pretrained(
        args.model_path, local_files_only=local_files_only
    )
    if tokenizer.eos_token is None:
        raise ValueError("causal tokenizer has no eos token")
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    global_seen = set()
    aggregate_rows = []
    client_results = {}
    resolved_targets = None
    aggregate_private_matches_rejected = 0
    for client_id in client_ids:
        client_quota = quotas[client_id]
        requested_records = sum(client_quota.values())
        training_seed = derived_seed(args.seed, "training", client_id)
        if requested_records == 0:
            client_results[str(client_id)] = {
                "status": "no_generation_quota",
                "adapter_initialization_seed": args.adapter_init_seed,
                "training_seed": training_seed,
                "input_records": len(clients[client_id]),
                "input_label_counts": dict(sorted(Counter(
                    row["label"] for row in clients[client_id]
                ).items())),
                "generation_quotas": client_quota,
                "generated_label_counts": {},
            }
            continue
        random.seed(training_seed)
        torch.manual_seed(training_seed)
        if device.type == "cuda":
            torch.cuda.manual_seed_all(training_seed)
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats(device)
        memory = guard_cuda_memory(
            torch, device, args.min_free_mib, args.memory_fraction
        )
        client_started = time.time()
        random.seed(args.adapter_init_seed)
        torch.manual_seed(args.adapter_init_seed)
        if device.type == "cuda":
            torch.cuda.manual_seed_all(args.adapter_init_seed)
        base_model = AutoModelForCausalLM.from_pretrained(
            args.model_path,
            local_files_only=local_files_only,
            torch_dtype=dtype,
        )
        if args.public_generator_control:
            model = base_model.to(device)
            optimizer = None
            training = {
                "status": "skipped_public_pretrained_control",
                "epochs": 0,
                "examples": 0,
                "batches": 0,
                "optimizer_steps": 0,
                "seconds": 0.0,
            }
        else:
            targets = select_lora_targets(base_model, args.lora_target_modules)
            if resolved_targets is None:
                resolved_targets = targets
            elif targets != resolved_targets:
                raise RuntimeError("inconsistent inferred LoRA targets across clients")
            # Reset immediately before adapter construction.  All clients therefore
            # start with identical LoRA parameters even if model loading consumes RNG.
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
            random.seed(training_seed)
            torch.manual_seed(training_seed)
            if device.type == "cuda":
                torch.cuda.manual_seed_all(training_seed)
            optimizer, training = train_adapter(
                torch, model, tokenizer, clients[client_id], args, device,
                training_seed, args.prompt_template,
            )
        generation_started = time.time()
        rows = []
        attempts_by_label = {}
        rejected_by_label = {}
        private_matches_by_label = {}
        for label in labels:
            generated_rows, attempts, rejected, private_matches = generate_label(
                torch, model, tokenizer, label, client_quota[label], args,
                device, client_id, global_seen, private_text_hashes,
                args.prompt_template, labels,
            )
            rows.extend(generated_rows)
            attempts_by_label[label] = attempts
            rejected_by_label[label] = rejected
            private_matches_by_label[label] = private_matches
            aggregate_private_matches_rejected += private_matches
        client_path = clients_dir / ("client_%d.jsonl" % client_id)
        atomic_write_jsonl(client_path, rows)
        aggregate_rows.extend(rows)
        result = {
            "status": "complete",
            "fresh_base_model_loaded": True,
            "fresh_lora_initialized": not args.public_generator_control,
            "client_records_used_for_model_training": not args.public_generator_control,
            "adapter_initialization_seed": args.adapter_init_seed,
            "training_seed": training_seed,
            "input_records": len(clients[client_id]),
            "input_label_counts": dict(sorted(Counter(
                row["label"] for row in clients[client_id]
            ).items())),
            "generation_quotas": client_quota,
            "generated_label_counts": dict(sorted(Counter(
                row["label"] for row in rows
            ).items())),
            "generation_attempts": attempts_by_label,
            "duplicates_rejected": rejected_by_label,
            "exact_private_matches_rejected": private_matches_by_label,
            "training": training,
            "generation_seconds": round(time.time() - generation_started, 6),
            "total_seconds": round(time.time() - client_started, 6),
            "output_file": str(client_path.relative_to(output_dir)),
            "output_sha256": sha256_file(client_path),
            "memory": memory,
        }
        if device.type == "cuda":
            result["memory"].update({
                "peak_allocated_mib": int(
                    torch.cuda.max_memory_allocated(device) // (1024 * 1024)
                ),
                "peak_reserved_mib": int(
                    torch.cuda.max_memory_reserved(device) // (1024 * 1024)
                ),
            })
        client_results[str(client_id)] = result
        if optimizer is not None:
            del optimizer
        del model
        del base_model
        gc.collect()
        if device.type == "cuda":
            torch.cuda.empty_cache()

    normalized = [normalize_generated_text(row["text"]) for row in aggregate_rows]
    if len(normalized) != len(set(normalized)):
        raise AssertionError("aggregate synthetic records are not globally unique")
    expected = sum(sum(client_quota.values()) for client_quota in quotas.values())
    if len(aggregate_rows) != expected:
        raise AssertionError("aggregate count does not match allocated quotas")
    aggregate_counts = dict(sorted(Counter(
        row["label"] for row in aggregate_rows
    ).items()))
    if args.target_per_label is not None:
        for label in labels:
            if aggregate_counts.get(label, 0) != args.target_per_label:
                raise AssertionError("aggregate target was not met for label %r" % label)
    atomic_write_jsonl(synthetic_path, aggregate_rows)

    manifest = {
        "schema_version": 1,
        "status": "complete",
        "experiment": (
            "public_pretrained_label_conditioned_synthesis_control"
            if args.public_generator_control
            else "per_client_label_conditioned_true_non_dp_synthesis"
        ),
        "privacy": {
            "mechanism": "none",
            "gradient_clipping": False,
            "noise_added": False,
            "noise_multiplier": 0.0,
            "dp_accountant": None,
        },
        "is_true_non_dp": True,
        "public_generator_control": args.public_generator_control,
        "client_records_used_for_model_training": not args.public_generator_control,
        "fresh_base_and_lora_per_client": not args.public_generator_control,
        "adapter_reuse_between_clients": False if not args.public_generator_control else None,
        "source_split": "train_only",
        "label_conditioning": {
            "prompt_template": args.prompt_template,
            "public_label_names": args.label_names,
            "prompt_tokens_masked_from_training_loss": True,
        },
        "seed": args.seed,
        "shared_adapter_initialization_seed": args.adapter_init_seed,
        "client_ids": client_ids,
        "labels": labels,
        "label_vocab": label_vocab,
        "task_type": task_type,
        "quota_mode": "aggregate_target" if args.target_per_label is not None else "per_client",
        "samples_per_label": args.samples_per_label,
        "target_per_label": args.target_per_label,
        "global_deduplication": "NFKC + whitespace collapse + casefold",
        "private_exact_match_filter": "NFKC + whitespace collapse + casefold + SHA-256 (in memory only)",
        "exact_private_matches_rejected": aggregate_private_matches_rejected,
        "source": source_descriptor,
        "model": model_fingerprint,
        "software": {
            "torch": torch.__version__,
            "transformers": transformers.__version__,
            "peft": peft.__version__,
        },
        "training_parameters": {
            "enabled": not args.public_generator_control,
            "epochs": 0 if args.public_generator_control else args.epochs,
            "batch_size": args.batch_size,
            "gradient_accumulation_steps": args.gradient_accumulation_steps,
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
        "synthetic_sha256": sha256_file(synthetic_path),
        "elapsed_seconds": round(time.time() - started, 6),
    }
    atomic_write_json(manifest_path, manifest)
    print(json.dumps({
        "status": "complete",
        "synthetic_file": str(synthetic_path),
        "manifest": str(manifest_path),
        "records": len(aggregate_rows),
        "label_counts": aggregate_counts,
    }, ensure_ascii=True, sort_keys=True))


if __name__ == "__main__":
    main()
