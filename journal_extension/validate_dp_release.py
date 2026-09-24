#!/usr/bin/env python3
"""Validate that a DP synthetic release has complete accounting metadata."""

import argparse
import hashlib
import json
import math
from collections import Counter
from pathlib import Path


FORBIDDEN_KEYS = {
    "clip_records",
    "clip_scale",
    "first_loss",
    "grad_norm",
    "input_files",
    "input_label_counts",
    "last_loss",
    "losses",
    "mean_loss",
    "min_loss",
    "max_loss",
    "private_text_hashes",
    "staging_manifest",
    "staging_manifest_sha256",
}


def sha256_file(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def find_forbidden(value, location="$"):
    findings = []
    if isinstance(value, dict):
        for key, child in value.items():
            child_location = "%s.%s" % (location, key)
            if key in FORBIDDEN_KEYS:
                findings.append(child_location)
            findings.extend(find_forbidden(child, child_location))
    elif isinstance(value, list):
        for index, child in enumerate(value):
            findings.extend(find_forbidden(child, "%s[%d]" % (location, index)))
    return findings


def require(condition, message):
    if not condition:
        raise ValueError(message)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--output")
    parser.add_argument("--epsilon-tolerance", type=float, default=0.011)
    args = parser.parse_args()

    manifest_path = Path(args.manifest).expanduser().resolve()
    with manifest_path.open(encoding="utf-8") as handle:
        manifest = json.load(handle)
    require(manifest.get("status") == "complete", "manifest is incomplete")
    require(manifest.get("is_record_level_dp") is True,
            "manifest does not declare record-level DP")
    require(manifest.get("private_diagnostics_released") is False,
            "private diagnostic release flag is invalid")
    require(manifest.get("private_exact_match_filter") == "disabled_for_dp_release",
            "private exact-match filtering was not disabled")
    findings = find_forbidden(manifest)
    require(not findings, "forbidden private diagnostics: %s" % findings)

    privacy = manifest.get("privacy", {})
    require(privacy.get("mechanism") == "poisson_sampled_gaussian_dp_sgd",
            "unexpected privacy mechanism")
    require(privacy.get("per_example_global_l2_clipping") is True,
            "per-example clipping is not declared")
    require(privacy.get("noise_added_to_clipped_gradient_sum") is True,
            "Gaussian noise is not declared")
    require(privacy.get("accountant") == "opacus_rdp",
            "unexpected accountant")
    require(privacy.get("composition_across_clients") == "parallel_disjoint_partitions",
            "parallel composition is not declared")
    delta = float(privacy["delta"])
    require(0.0 < delta < 1.0, "delta is invalid")
    maximum = float(privacy["achieved_epsilon_max"])
    minimum = float(privacy["achieved_epsilon_min"])
    require(math.isfinite(maximum) and maximum > 0.0, "epsilon max is invalid")
    require(math.isfinite(minimum) and 0.0 < minimum <= maximum,
            "epsilon min is invalid")
    target = privacy.get("target_epsilon")
    if target is not None:
        require(maximum <= float(target) + args.epsilon_tolerance,
                "achieved epsilon exceeds target")

    public_input = manifest.get("public_input", {})
    client_ids = [str(value) for value in public_input.get("client_ids", [])]
    clients = manifest.get("clients", {})
    require(client_ids and sorted(client_ids) == sorted(clients),
            "public client IDs and accounting entries differ")
    client_epsilons = []
    for client_id in client_ids:
        result = clients[client_id]
        require(result.get("status") == "complete",
                "client %s is incomplete" % client_id)
        accounting = result.get("accounting", {})
        require(accounting.get("sampling") == "independent_poisson",
                "client %s sampling is invalid" % client_id)
        require(accounting.get("accountant") == "opacus_rdp",
                "client %s accountant is invalid" % client_id)
        require(int(accounting.get("records", -1)) ==
                int(public_input.get("records_per_client", -2)),
                "client %s record bound differs" % client_id)
        require(float(accounting.get("max_grad_norm")) ==
                float(privacy.get("max_grad_norm")),
                "client %s clipping norm differs" % client_id)
        epsilon = float(accounting.get("epsilon"))
        require(math.isfinite(epsilon) and epsilon > 0.0,
                "client %s epsilon is invalid" % client_id)
        require(float(accounting.get("delta")) == delta,
                "client %s delta differs" % client_id)
        client_epsilons.append(epsilon)
    require(abs(max(client_epsilons) - maximum) <= 1e-12,
            "top-level epsilon max differs from clients")
    require(abs(min(client_epsilons) - minimum) <= 1e-12,
            "top-level epsilon min differs from clients")

    synthetic_path = manifest_path.parent / manifest["synthetic_file"]
    require(synthetic_path.is_file(), "synthetic JSONL is missing")
    require(sha256_file(synthetic_path) == manifest.get("synthetic_sha256"),
            "synthetic JSONL hash mismatch")
    counts = Counter()
    records = 0
    with synthetic_path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            row = json.loads(line)
            require(isinstance(row.get("text"), str) and row["text"].strip(),
                    "invalid text at line %d" % line_number)
            label = str(row.get("label"))
            require(label in manifest.get("label_vocab", {}),
                    "non-public label at line %d" % line_number)
            counts[label] += 1
            records += 1
    require(records == int(manifest.get("synthetic_records", -1)),
            "synthetic record count mismatch")
    require(dict(sorted(counts.items())) == manifest.get("synthetic_label_counts"),
            "synthetic label count mismatch")

    payload = {
        "schema_version": 1,
        "status": "complete",
        "manifest": str(manifest_path),
        "manifest_sha256": sha256_file(manifest_path),
        "synthetic_file": str(synthetic_path),
        "synthetic_sha256": sha256_file(synthetic_path),
        "clients": len(client_ids),
        "records": records,
        "epsilon_max": maximum,
        "epsilon_min": minimum,
        "delta": delta,
        "checks": {
            "accounting_complete": True,
            "client_composition_declared": True,
            "no_forbidden_private_diagnostics": True,
            "public_fixed_labels": True,
            "release_hashes_match": True,
        },
    }
    if args.output:
        output = Path(args.output).expanduser().resolve()
        common_parent = output.parent
        common_parent.mkdir(parents=True, exist_ok=True)
        temporary = Path(str(output) + ".tmp")
        with temporary.open("w", encoding="utf-8") as handle:
            json.dump(payload, handle, ensure_ascii=True, indent=2, sort_keys=True)
            handle.write("\n")
        temporary.replace(output)
    print(json.dumps(payload, ensure_ascii=True, sort_keys=True))


if __name__ == "__main__":
    main()
