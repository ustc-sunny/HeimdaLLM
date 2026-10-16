#!/usr/bin/env python3
"""Co-located CPU profile and aggregate-only private selection audit.

No selected IDs, private records, raw votes, or sampled noise are written.
The noise timing uses a fixed public zero histogram, so it creates no new
private DP release.
"""

import hashlib
import json
import os
import statistics
import sys
import time
from collections import Counter
from pathlib import Path

import psutil
import torch
from transformers import AutoModel, AutoTokenizer


STAGE1 = Path("/mnt/heimdallm-stage1-20261015")
STAGE2 = Path("/mnt/heimdallm-stage2-20261016")
REPO = Path("/root/heimdallm-stage2-20261016/HeimdaLLM")
OUT = STAGE2 / "restricted_diagnostics/selection_profile.json"
sys.path.insert(0, str(STAGE1))
sys.path.insert(0, str(REPO / "journal_extension"))

from private_vote_dp_e4 import (  # noqa: E402
    LABELS, discrete_laplace, private_histograms, public_pool, select_from_scores,
)
from non_dp_client_synthetic import load_staged_clients, normalize_generated_text  # noqa: E402
from augpe_nondp_pilot import embed  # noqa: E402


def sha(path):
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def percentile(values, q):
    ordered = sorted(values)
    return ordered[min(len(ordered) - 1, int(round(q * (len(ordered) - 1))))]


def main():
    if OUT.exists():
        raise FileExistsError(OUT)
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    torch.set_num_threads(1)
    pool_path = STAGE1 / "reused/public_pool/candidates.jsonl"
    candidates = public_pool(pool_path)
    clients, client_ids, _, _, descriptor = load_staged_clients(
        STAGE1 / "private_staging/source", [1, 21])
    model_root = STAGE1 / "models/all-MiniLM-L6-v2"
    rss_before = psutil.Process().memory_info().rss
    load_start = time.perf_counter()
    tokenizer = AutoTokenizer.from_pretrained(str(model_root), local_files_only=True)
    model = AutoModel.from_pretrained(str(model_root), local_files_only=True).eval()
    model_load_seconds = time.perf_counter() - load_start
    rss_after_load = psutil.Process().memory_info().rss
    vote_start = time.perf_counter()
    histograms = private_histograms(clients, client_ids, candidates, model, tokenizer)
    vote_seconds = time.perf_counter() - vote_start
    rss_after_vote = psutil.Process().memory_info().rss
    boundaries, ties = [], 0
    used_by_class = {label: set() for label in LABELS}
    for label in LABELS:
        for client_id in client_ids:
            counts = histograms[(client_id, label)]
            remaining = [i for i in range(32) if i not in used_by_class[label]]
            ranked = sorted(remaining, key=lambda i: (-counts[i], i))
            margin = counts[ranked[7]] - counts[ranked[8]]
            boundaries.append(margin)
            ties += int(margin == 0)
            used_by_class[label].update(ranked[:8])
    similarity_start = time.perf_counter()
    public_vectors = []
    for label in LABELS:
        public_vectors.append(embed(model, tokenizer, candidates[label]))
    vectors = torch.cat(public_vectors, dim=0)
    cosine = vectors @ vectors.T
    semantic_pairs_gt_09 = sum(
        bool(cosine[i, j] > 0.9) for i in range(128) for j in range(i + 1, 128))
    similarity_seconds = time.perf_counter() - similarity_start
    # Benchmark the reviewed noise/top-8 implementation on a PUBLIC zero
    # histogram. No private-selected output is created or shown.
    sample_times = []
    zero_scores = {(client_id, label): [0] * 32
                   for label in LABELS for client_id in client_ids}
    for _ in range(20):
        start = time.perf_counter()
        toy = {key: [value + discrete_laplace() for value in counts]
               for key, counts in zero_scores.items()}
        select_from_scores(toy, candidates, client_ids)
        sample_times.append(time.perf_counter() - start)
    pool_rows = [json.loads(line) for line in pool_path.read_text().splitlines() if line.strip()]
    lookup = {normalize_generated_text(row["text"]): index
              for index, row in enumerate(pool_rows)}
    release_path = STAGE1 / "private_staging/e2_release/D01/synthetic.jsonl"
    release_manifest = json.loads((release_path.parent / "release_manifest.json").read_text())
    if sha(release_path) != release_manifest["selected_sha256"]:
        raise RuntimeError("D01 release hash mismatch")
    selected = [json.loads(line) for line in release_path.read_text().splitlines() if line.strip()]
    if len(selected) != 64:
        raise RuntimeError("D01 release budget mismatch")
    rows = [{"candidate_id": lookup[normalize_generated_text(row["text"])],
             "label": str(row["label"]), "client_id": int(row["client_id"])}
            for row in selected]
    by_client = {str(client): [row for row in rows if row["client_id"] == client]
                 for client in client_ids}
    payloads = {client: json.dumps({"version": 1, "selected_ids": items},
                                   separators=(",", ":")).encode("utf-8")
                for client, items in by_client.items()}
    headers = {client: json.dumps({"type": "guidance_ids", "version": 1,
                                   "content_length": len(payload)},
                                  separators=(",", ":")).encode("utf-8")
               for client, payload in payloads.items()}
    result = {
        "schema_version": 1, "device": "CPU on co-located A6000 server",
        "model": str(model_root), "model_load_seconds": model_load_seconds,
        "T_encode_vote_seconds": vote_seconds,
        "T_public_semantic_similarity_audit_seconds": similarity_seconds,
        "T_noise_select_public_zero_hist_median_seconds": statistics.median(sample_times),
        "T_noise_select_public_zero_hist_p95_seconds": percentile(sample_times, 0.95),
        "noise_timing_repetitions": 20,
        "cpu_rss_before_bytes": rss_before, "cpu_rss_after_model_load_bytes": rss_after_load,
        "cpu_rss_after_vote_bytes": rss_after_vote,
        "source_manifest_sha256": descriptor["staging_manifest_sha256"],
        "pool_sha256": sha(pool_path),
        "selection_margin_8_vs_9_min": min(boundaries),
        "selection_margin_8_vs_9_median": statistics.median(boundaries),
        "selection_margin_8_vs_9_max": max(boundaries),
        "selection_boundary_ties": ties, "selection_groups": len(boundaries),
        "max_per_record_histogram_l1_contribution": 1,
        "public_semantic_duplicate_pairs_cosine_gt_0_9": semantic_pairs_gt_09,
        "public_pair_count": 128 * 127 // 2,
        "public_pool_jsonl_file_bytes": pool_path.stat().st_size,
        "reference_id_payload_bytes_per_client": {k: len(v) for k, v in payloads.items()},
        "reference_id_header_bytes_per_client": {k: len(v) for k, v in headers.items()},
        "reference_id_payload_bytes_total": sum(map(len, payloads.values())),
        "reference_id_header_bytes_total": sum(map(len, headers.values())),
        "released_full_text_jsonl_file_bytes": release_path.stat().st_size,
        "serialization_status": "measured Python JSON bytes of an explicit reference codec; no production network capture",
        "privacy": "no raw votes, selected IDs, noise draws or private text persisted; noise timing uses public zeros",
    }
    OUT.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
    OUT.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"output": str(OUT), "status": "pass",
                      "vote_seconds": vote_seconds, "boundary_ties": ties}))


if __name__ == "__main__":
    main()
