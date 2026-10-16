#!/usr/bin/env python3
"""Emit aggregate-only selection overlap from immutable Stage-1 H5 buffers."""

import hashlib
import json
import re
import unicodedata
from collections import Counter
from itertools import combinations
from pathlib import Path

import h5py


ROOT = Path("/mnt/heimdallm-stage1-20261015")
OUT = Path("/mnt/heimdallm-stage2-20261016/restricted_diagnostics/selection_aggregate.json")
POOL_SHA = "7e54c05ce2dc94d754b178218963dedfceb781c491b97ed1d9db5ab9e7d7c54f"
SOURCES = {
    "PUB_RANDOM": ROOT / "reused/public_random_selection/cloud_data.h5",
    "PUB_NLI": ROOT / "guidance/public_nli_selection/cloud_data.h5",
    "PRIVATE_ZERO": ROOT / "reused/private_vote_synthetic/cloud_data.h5",
    "DP_D01": ROOT / "guidance/private_dp_e4/D01/cloud_data.h5",
    "DP_D02": ROOT / "guidance/private_dp_e4/D02/cloud_data.h5",
    "DP_D03": ROOT / "guidance/private_dp_e4/D03/cloud_data.h5",
}


def norm(text):
    return re.sub(r"\s+", " ", unicodedata.normalize("NFKC", text)).strip().casefold()


def decoded(value):
    return value.decode("utf-8") if isinstance(value, bytes) else value.tobytes().decode("utf-8")


def sha(path):
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def main():
    pool_path = ROOT / "reused/public_pool/candidates.jsonl"
    if sha(pool_path) != POOL_SHA:
        raise RuntimeError("frozen candidate pool hash mismatch")
    pool_rows = [json.loads(line) for line in pool_path.read_text().splitlines() if line.strip()]
    lookup = {norm(row["text"]): (index, str(row["label"]))
              for index, row in enumerate(pool_rows)}
    if len(pool_rows) != 128 or len(lookup) != 128:
        raise RuntimeError("pool size/uniqueness mismatch")
    selected = {}
    source_provenance = {}
    for arm, path in SOURCES.items():
        with h5py.File(path, "r", swmr=True) as handle:
            keys = sorted(handle["X"], key=int)
            records = [(decoded(handle["X"][key][()]),
                        decoded(handle["Y"][key][()])) for key in keys]
        if len(records) != 64:
            raise RuntimeError("unexpected guidance budget for " + arm)
        ids = set()
        labels = Counter()
        for text, label in records:
            match = lookup.get(norm(text))
            if match is None or match[1] != str(label):
                raise RuntimeError("text or label does not match frozen public pool for " + arm)
            ids.add(match[0])
            labels[str(label)] += 1
        if len(ids) != 64 or labels != {str(k): 16 for k in range(1, 5)}:
            raise RuntimeError("duplicate or class imbalance for " + arm)
        selected[arm] = ids
        source_provenance[arm] = {"guidance_h5_sha256": sha(path), "count": 64,
                                  "unique_count": 64, "label_counts": dict(labels)}
    pairwise = []
    for left, right in combinations(SOURCES, 2):
        overall_intersection = len(selected[left] & selected[right])
        classes = {}
        for label in map(str, range(1, 5)):
            universe = {i for i, row in enumerate(pool_rows) if str(row["label"]) == label}
            a, b = selected[left] & universe, selected[right] & universe
            intersection = len(a & b)
            classes[label] = {"intersection": intersection,
                              "union": len(a | b), "jaccard": intersection / len(a | b)}
        pairwise.append({"left": left, "right": right,
                         "intersection": overall_intersection,
                         "union": len(selected[left] | selected[right]),
                         "jaccard": overall_intersection / len(selected[left] | selected[right]),
                         "by_class": classes})
    output = {"schema_version": 1, "pool_sha256": POOL_SHA, "pool_count": 128,
              "source_provenance": source_provenance,
              "pairwise": pairwise,
              "privacy_review": "aggregate counts only; no selected IDs, texts, vote scores, or private records",
              "semantic_duplication": "not_measured; normalized exact duplicates are zero"}
    OUT.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
    OUT.write_text(json.dumps(output, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"output": str(OUT), "arms": list(SOURCES),
                      "pairwise_count": len(pairwise), "status": "pass"}))


if __name__ == "__main__":
    main()
