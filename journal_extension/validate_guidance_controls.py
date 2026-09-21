#!/usr/bin/env python3
"""Assert equal cloud budgets and text preservation across pilot controls."""

from __future__ import print_function

import argparse
from collections import Counter
import hashlib
import json
import os
from pathlib import Path


def load_json(path):
    with path.open(encoding="utf-8") as handle:
        return json.load(handle)


def load_jsonl(path):
    rows = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                row = json.loads(line)
                rows.append((row["text"], str(row["label"])))
    return rows


def sha256_file(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_json_values(values):
    digest = hashlib.sha256()
    for value in values:
        encoded = json.dumps(
            value, ensure_ascii=True, sort_keys=True, separators=(",", ":")
        ).encode("utf-8")
        digest.update(len(encoded).to_bytes(8, byteorder="big"))
        digest.update(encoded)
    return digest.hexdigest()


def validate_pack_artifacts(name, manifest, expected_jsonl=None):
    checks = {}
    manifest_jsonl = Path(manifest["input_jsonl"]).expanduser().resolve()
    if expected_jsonl is not None and manifest_jsonl != expected_jsonl:
        raise ValueError("%s pack manifest refers to the wrong JSONL" % name)
    if not manifest_jsonl.is_file():
        raise FileNotFoundError(str(manifest_jsonl))
    actual_jsonl_hash = sha256_file(manifest_jsonl)
    if manifest.get("input_jsonl_sha256") != actual_jsonl_hash:
        raise ValueError("%s packed JSONL hash no longer matches its manifest" % name)
    checks["input_jsonl"] = {
        "path": str(manifest_jsonl), "sha256": actual_jsonl_hash,
    }
    for field, hash_field in (
            ("data_out", "data_sha256"),
            ("partition_out", "partition_sha256")):
        artifact = Path(manifest[field]).expanduser().resolve()
        if not artifact.is_file():
            raise FileNotFoundError(str(artifact))
        actual_hash = sha256_file(artifact)
        if manifest.get(hash_field) != actual_hash:
            raise ValueError(
                "%s %s hash no longer matches its manifest" % (name, field)
            )
        checks[field] = {"path": str(artifact), "sha256": actual_hash}
    return checks


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--synthetic-pack-manifest", required=True)
    parser.add_argument("--real-matched-pack-manifest", required=True)
    parser.add_argument("--shuffled-pack-manifest", required=True)
    parser.add_argument("--same-source-real-pack-manifest")
    parser.add_argument("--synthetic-jsonl", required=True)
    parser.add_argument("--shuffled-jsonl", required=True)
    parser.add_argument("--shuffle-manifest", required=True)
    parser.add_argument("--same-source-real-jsonl")
    parser.add_argument("--same-source-real-build-manifest")
    parser.add_argument("--output", required=True)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()

    paths = dict(
        (name, Path(value).expanduser().resolve())
        for name, value in vars(args).items()
        if name not in ("output", "resume") and value is not None
    )
    same_source_fields = (
        "same_source_real_pack_manifest",
        "same_source_real_jsonl",
        "same_source_real_build_manifest",
    )
    same_source_present = [field in paths for field in same_source_fields]
    if any(same_source_present) and not all(same_source_present):
        raise ValueError(
            "same-source real control requires its pack manifest, JSONL, and build manifest"
        )
    for path in paths.values():
        if not path.is_file():
            raise FileNotFoundError(str(path))
    pack_manifests = {
        "synthetic": load_json(paths["synthetic_pack_manifest"]),
        "real_matched": load_json(paths["real_matched_pack_manifest"]),
        "shuffled_label": load_json(paths["shuffled_pack_manifest"]),
    }
    if all(same_source_present):
        pack_manifests["same_source_real_matched"] = load_json(
            paths["same_source_real_pack_manifest"]
        )
    for name, manifest in pack_manifests.items():
        if manifest.get("status") != "complete":
            raise ValueError("%s pack manifest is incomplete" % name)
    reference_records = pack_manifests["synthetic"]["records"]
    reference_counts = pack_manifests["synthetic"]["label_counts"]
    for name, manifest in pack_manifests.items():
        if manifest.get("records") != reference_records:
            raise ValueError("%s cloud record budget differs from synthetic" % name)
        if manifest.get("label_counts") != reference_counts:
            raise ValueError("%s per-label budget differs from synthetic" % name)

    pack_artifacts = {
        "synthetic": validate_pack_artifacts(
            "synthetic", pack_manifests["synthetic"], paths["synthetic_jsonl"]
        ),
        "real_matched": validate_pack_artifacts(
            "real_matched", pack_manifests["real_matched"]
        ),
        "shuffled_label": validate_pack_artifacts(
            "shuffled_label", pack_manifests["shuffled_label"],
            paths["shuffled_jsonl"],
        ),
    }
    if all(same_source_present):
        pack_artifacts["same_source_real_matched"] = validate_pack_artifacts(
            "same_source_real_matched",
            pack_manifests["same_source_real_matched"],
            paths["same_source_real_jsonl"],
        )

    synthetic_rows = load_jsonl(paths["synthetic_jsonl"])
    shuffled_rows = load_jsonl(paths["shuffled_jsonl"])
    if [row[0] for row in synthetic_rows] != [row[0] for row in shuffled_rows]:
        raise ValueError("shuffled-label control changed text content or order")
    if Counter(row[1] for row in synthetic_rows) != Counter(
        row[1] for row in shuffled_rows
    ):
        raise ValueError("shuffled-label control changed the label histogram")
    same_source_label_sequence_identical = None
    if all(same_source_present):
        same_source_rows = load_jsonl(paths["same_source_real_jsonl"])
        same_source_label_sequence_identical = (
            [row[1] for row in synthetic_rows]
            == [row[1] for row in same_source_rows]
        )
        if not same_source_label_sequence_identical:
            raise ValueError(
                "same-source real control does not match the synthetic label sequence"
            )
        build_manifest = load_json(paths["same_source_real_build_manifest"])
        if build_manifest.get("status") != "complete":
            raise ValueError("same-source real build manifest is incomplete")
        if build_manifest.get("control") != "same_source_real_matched":
            raise ValueError("same-source real build manifest has the wrong control type")
        if build_manifest.get("deployable") is not False:
            raise ValueError("same-source real oracle must be marked non-deployable")
        if build_manifest.get("contains_private_training_text") is not True:
            raise ValueError("same-source real oracle must declare private training text")
        if build_manifest.get("output", {}).get("sha256") != sha256_file(
            paths["same_source_real_jsonl"]
        ):
            raise ValueError("same-source real JSONL hash differs from its build manifest")
        if build_manifest.get("reference", {}).get("sha256") != sha256_file(
            paths["synthetic_jsonl"]
        ):
            raise ValueError("same-source build manifest refers to the wrong synthetic JSONL")
    agreement = sum(
        left[1] == right[1] for left, right in zip(synthetic_rows, shuffled_rows)
    ) / float(len(synthetic_rows))
    shuffle_manifest = load_json(paths["shuffle_manifest"])
    if abs(agreement - float(shuffle_manifest["label_agreement_fraction"])) > 1e-12:
        raise ValueError("shuffle agreement differs from its manifest")
    minimum_agreement = float(shuffle_manifest.get("min_label_agreement", 0.0))
    maximum_agreement = float(shuffle_manifest.get("max_label_agreement", 1.0))
    if not minimum_agreement <= agreement <= maximum_agreement:
        raise ValueError("shuffle agreement is outside its declared interval")
    text_sequence_hash = sha256_json_values([row[0] for row in synthetic_rows])
    text_multiset_hash = sha256_json_values(sorted(
        row[0] for row in synthetic_rows
    ))
    original_label_hash = sha256_json_values([row[1] for row in synthetic_rows])
    shuffled_label_hash = sha256_json_values([row[1] for row in shuffled_rows])
    recorded_hashes = {
        "text_sequence_sha256": text_sequence_hash,
        "text_multiset_sha256": text_multiset_hash,
        "original_label_sequence_sha256": original_label_hash,
        "shuffled_label_sequence_sha256": shuffled_label_hash,
    }
    for field, expected in recorded_hashes.items():
        if shuffle_manifest.get(field) != expected:
            raise ValueError("shuffle manifest %s is invalid" % field)

    output_path = Path(args.output).expanduser().resolve()
    payload = {
        "schema_version": 1,
        "status": "complete",
        "records_per_arm": reference_records,
        "label_counts_per_arm": reference_counts,
        "all_cloud_budgets_equal": True,
        "shuffled_text_sequence_identical": True,
        "same_source_real_present": all(same_source_present),
        "same_source_real_label_sequence_identical": (
            same_source_label_sequence_identical
        ),
        "text_sequence_sha256": text_sequence_hash,
        "text_multiset_sha256": text_multiset_hash,
        "original_label_sequence_sha256": original_label_hash,
        "shuffled_label_sequence_sha256": shuffled_label_hash,
        "shuffled_label_agreement_fraction": agreement,
        "shuffled_label_agreement_interval": [
            minimum_agreement, maximum_agreement,
        ],
        "validated_pack_artifacts": pack_artifacts,
        "artifacts": dict(
            (name, {"path": str(path), "sha256": sha256_file(path)})
            for name, path in sorted(paths.items())
        ),
    }
    if output_path.exists():
        if not args.resume:
            raise FileExistsError("validation output already exists: %s" % output_path)
        existing = load_json(output_path)
        if existing != payload:
            raise ValueError("existing control validation no longer matches artifacts")
        print(json.dumps({"status": "matched", "output": str(output_path)}))
        return
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(str(output_path) + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=True, indent=2, sort_keys=True)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(str(temporary), str(output_path))
    print(json.dumps(payload, sort_keys=True))


if __name__ == "__main__":
    main()
