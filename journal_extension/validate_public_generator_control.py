#!/usr/bin/env python3
"""Validate the matched pretrained-only public synthetic control."""

import argparse
import hashlib
import json
from pathlib import Path


def load(path):
    with Path(path).open(encoding="utf-8") as handle:
        return json.load(handle)


def require(condition, message):
    if not condition:
        raise ValueError(message)


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--client-manifest", required=True)
    parser.add_argument("--public-manifest", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    client = load(args.client_manifest)
    public = load(args.public_manifest)
    require(client.get("status") == "complete", "client manifest is incomplete")
    require(public.get("status") == "complete", "public manifest is incomplete")
    require(not client.get("public_generator_control"), "client generator marked public")
    require(public.get("public_generator_control") is True, "public flag is missing")
    require(public.get("client_records_used_for_model_training") is False,
            "public generator used client records for model training")
    require(public.get("fresh_base_and_lora_per_client") is False,
            "public generator constructed LoRA")
    require(public.get("training_parameters", {}).get("enabled") is False,
            "public training parameters are enabled")
    require(public.get("training_parameters", {}).get("epochs") == 0,
            "public training epochs are nonzero")

    for client_id, item in public.get("clients", {}).items():
        training = item.get("training", {})
        require(training.get("status") == "skipped_public_pretrained_control",
                "public client %s was not marked training-skipped" % client_id)
        require(training.get("optimizer_steps") == 0,
                "public client %s has optimizer steps" % client_id)
        require(item.get("client_records_used_for_model_training") is False,
                "public client %s used private training records" % client_id)
        require(item.get("fresh_lora_initialized") is False,
                "public client %s initialized LoRA" % client_id)

    matched_fields = (
        "seed", "client_ids", "labels", "label_vocab", "quota_mode",
        "samples_per_label", "target_per_label", "generation_parameters",
        "label_conditioning", "global_deduplication", "private_exact_match_filter",
    )
    for field in matched_fields:
        require(public.get(field) == client.get(field),
                "matched field differs: %s" % field)
    require(public.get("model") == client.get("model"), "base model differs")
    require(public.get("synthetic_records") == client.get("synthetic_records"),
            "sample budget differs")
    require(public.get("synthetic_label_counts") == client.get("synthetic_label_counts"),
            "class balance differs")
    require(public.get("source") == client.get("source"), "quota/filter source differs")

    public_jsonl = Path(args.public_manifest).parent / public["synthetic_file"]
    client_jsonl = Path(args.client_manifest).parent / client["synthetic_file"]
    if not public_jsonl.is_file():
        public_jsonl = Path(public["synthetic_file"])
    if not client_jsonl.is_file():
        client_jsonl = Path(client["synthetic_file"])
    require(sha256(public_jsonl) == public["synthetic_sha256"],
            "public JSONL hash mismatch")
    require(sha256(client_jsonl) == client["synthetic_sha256"],
            "client JSONL hash mismatch")

    result = {
        "schema_version": 1,
        "status": "complete",
        "checks": {
            "pretrained_base_model_identical": True,
            "prompt_and_decoding_identical": True,
            "sample_budget_and_class_balance_identical": True,
            "filter_configuration_identical": True,
            "public_lora_not_constructed": True,
            "public_optimizer_steps_zero": True,
            "client_records_not_used_for_public_model_training": True,
            "synthetic_hashes_verified": True,
        },
        "public_exact_private_matches_rejected": public.get(
            "exact_private_matches_rejected"
        ),
        "client_exact_private_matches_rejected": client.get(
            "exact_private_matches_rejected"
        ),
        "public_synthetic_sha256": public["synthetic_sha256"],
        "client_synthetic_sha256": client["synthetic_sha256"],
    }
    output = Path(args.output)
    temporary = output.with_suffix(output.suffix + ".tmp")
    temporary.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    temporary.replace(output)
    print(json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()
