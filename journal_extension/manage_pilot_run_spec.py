#!/usr/bin/env python3
"""Create or strictly validate the immutable scientific specification of a run."""

from __future__ import print_function

import argparse
import hashlib
import json
import os
from pathlib import Path


def sha256_file(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def parse_pairs(values, option):
    result = {}
    for value in values:
        if "=" not in value:
            raise ValueError("%s requires NAME=VALUE: %s" % (option, value))
        key, item = value.split("=", 1)
        if not key or key in result:
            raise ValueError("empty or duplicate key for %s: %s" % (option, key))
        result[key] = item
    return result


def canonical_hash(payload):
    encoded = json.dumps(
        payload, ensure_ascii=True, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--value", action="append", default=[])
    parser.add_argument("--file", action="append", default=[])
    args = parser.parse_args()

    values = parse_pairs(args.value, "--value")
    raw_files = parse_pairs(args.file, "--file")
    files = {}
    for name, raw_path in sorted(raw_files.items()):
        path = Path(raw_path).expanduser().resolve()
        if not path.is_file():
            raise FileNotFoundError(str(path))
        files[name] = {
            "path": str(path),
            "bytes": path.stat().st_size,
            "sha256": sha256_file(path),
        }
    payload = {"schema_version": 1, "values": values, "files": files}
    document = dict(payload)
    document["spec_sha256"] = canonical_hash(payload)
    output_path = Path(args.output).expanduser().resolve()

    if args.resume:
        if not output_path.is_file():
            raise FileNotFoundError("resume requires existing run spec: %s" % output_path)
        with output_path.open(encoding="utf-8") as handle:
            existing = json.load(handle)
        existing_payload = {
            "schema_version": existing.get("schema_version"),
            "values": existing.get("values"),
            "files": existing.get("files"),
        }
        if existing.get("spec_sha256") != canonical_hash(existing_payload):
            raise ValueError("existing run spec has an invalid internal hash")
        if existing != document:
            raise ValueError(
                "resume specification mismatch: existing=%s requested=%s"
                % (existing.get("spec_sha256"), document["spec_sha256"])
            )
        print(json.dumps({"status": "matched", "spec_sha256": document["spec_sha256"]}))
        return

    if output_path.exists():
        raise FileExistsError("run spec already exists: %s" % output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(str(output_path) + ".tmp")
    if temporary.exists():
        temporary.unlink()
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(document, handle, ensure_ascii=True, indent=2, sort_keys=True)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(str(temporary), str(output_path))
    print(json.dumps({"status": "created", "spec_sha256": document["spec_sha256"]}))


if __name__ == "__main__":
    main()
