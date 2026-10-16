#!/usr/bin/env python3
"""Run one preregistered SHUFFLED_DP_LABEL endpoint on the restored A6000.

Inputs are immutable Stage-1 releases and a fixed label-cycle postprocess.
The script creates a fresh Stage-2 output directory and stops on any mismatch.
"""

import argparse
import hashlib
import json
import os
import shlex
import socket
import subprocess
from datetime import datetime, timezone
from pathlib import Path


STAGE1 = Path("/mnt/heimdallm-stage1-20261015")
STAGE2 = Path("/mnt/heimdallm-stage2-20261016")
REPO = Path("/root/heimdallm-stage2-20261016/HeimdaLLM")
DATA = Path("/root/heimdallm-work/data/fednlp_data/data_files/agnews_data.h5")
MODEL = Path("/root/heimdallm-work/models/distilbert-base-uncased")
PYTHON = Path("/root/miniconda3/envs/heimdallm-kdd/bin/python")
COMMIT = "ad325aa320789c4f59f911898033844d5b19ccec"
SOURCE_SHA = "1474dd264d535b50ad2a7b1cf714bf11e182c2fe5b07f5e982feb31a9b114616"
PARTITION_SHA = "e16e8551a87ba702cc23b4d56d57b9762b77c4fea8bf341b96d5759be7996a6c"
POOL_SHA = "7e54c05ce2dc94d754b178218963dedfceb781c491b97ed1d9db5ab9e7d7c54f"
TRIALS = {57: "D01", 58: "D02", 59: "D03"}
CYCLE = {"1": "2", "2": "3", "3": "4", "4": "1"}


def digest(path):
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def option(command, name, value):
    if name not in command:
        raise ValueError("missing template option " + name)
    command[command.index(name) + 1] = str(value)


def check_gpu():
    proc = subprocess.run(["nvidia-smi", "--query-gpu=index,name,memory.free,utilization.gpu",
                           "--format=csv,noheader,nounits"], text=True,
                          capture_output=True, check=True, timeout=15)
    rows = [line.split(",") for line in proc.stdout.splitlines()]
    gpu = next((r for r in rows if r[0].strip() == "0"), None)
    if gpu is None:
        raise RuntimeError("GPU 0 missing")
    name, free, util = gpu[1].strip(), int(gpu[2]), int(gpu[3])
    result = {"index": 0, "name": name, "free_mib": free,
              "utilization_percent": util, "minimum_free_mib": 14000,
              "maximum_utilization_percent": 20}
    if "A6000" not in name or free < 14000 or util > 20:
        raise RuntimeError("GPU preflight failed: " + json.dumps(result))
    processes = subprocess.run(["nvidia-smi", "--query-compute-apps=pid,process_name,used_gpu_memory",
                                "--format=csv,noheader,nounits"], text=True,
                               capture_output=True, check=True, timeout=15)
    if processes.stdout.strip():
        raise RuntimeError("GPU already has compute processes: " + processes.stdout.strip())
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, choices=tuple(TRIALS), required=True)
    parser.add_argument("--smoke", action="store_true",
                        help="run one audited FL round without creating a formal endpoint")
    args = parser.parse_args()
    seed, trial = args.seed, TRIALS[args.seed]
    dest = (STAGE2 / "preflight/SHUFFLED_DP_LABEL" if args.smoke else
            STAGE2 / "runs/SHUFFLED_DP_LABEL") / ("seed_%d" % seed)
    if dest.exists():
        raise FileExistsError(str(dest))
    commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=str(REPO),
                                     text=True).strip()
    tracked_changes = subprocess.check_output(["git", "diff", "--name-only"],
                                              cwd=str(REPO), text=True).strip()
    if commit != COMMIT or tracked_changes:
        raise RuntimeError("unexpected source revision or tracked changes: %s %s" %
                           (commit, tracked_changes))
    fixed = STAGE1 / "shared/fixed_partition.h5"
    pool = STAGE1 / "reused/public_pool/candidates.jsonl"
    for path, expected in ((DATA, SOURCE_SHA), (fixed, PARTITION_SHA), (pool, POOL_SHA)):
        if digest(path) != expected:
            raise RuntimeError("frozen input hash mismatch: " + str(path))
    original = STAGE1 / "private_staging/e2_release" / trial
    original_manifest = json.loads((original / "release_manifest.json").read_text())
    if digest(original / "synthetic.jsonl") != original_manifest["selected_sha256"]:
        raise RuntimeError("source DP release hash mismatch")
    selected = STAGE2 / "private_postprocess/shuffled_dp_label" / ("seed_%d" % seed)
    validation = json.loads((selected / "validation.json").read_text())
    if (validation["source_trial"] != trial or
            validation["source_sha256"] != original_manifest["selected_sha256"] or
            validation["label_permutation"] != CYCLE or
            not validation["texts_in_original_order_identical"] or
            validation["unique_texts"] != 64 or
            digest(selected / "synthetic.jsonl") != validation["postprocessed_sha256"]):
        raise RuntimeError("shuffled text/label validation mismatch")
    packed = selected / "packed"
    pack = json.loads((packed / "pack_manifest.json").read_text())
    cloud_data, cloud_part = packed / "cloud_data.h5", packed / "cloud_partition.h5"
    if (pack["records"] != 64 or pack["test_records"] != 0 or
            pack["duplicates_dropped"] != 0 or
            pack["label_counts"] != {k: 16 for k in CYCLE} or
            digest(cloud_data) != pack["data_sha256"] or
            digest(cloud_part) != pack["partition_sha256"]):
        raise RuntimeError("shuffled cloud H5 validation mismatch")
    gpu = check_gpu()
    shared = STAGE2 / "shared"
    shared.mkdir(mode=0o700, exist_ok=True)
    host = socket.gethostname()
    hostfile, mapping = shared / "mpi_host_file", shared / "gpu_mapping.yaml"
    expected_hostfile = host + " slots=3\n"
    expected_mapping = "mapping_private_vote:\n  %s: [3]\n" % host
    for path, expected in ((hostfile, expected_hostfile), (mapping, expected_mapping)):
        if path.exists() and path.read_text() != expected:
            raise RuntimeError("existing MPI mapping conflicts with host: " + str(path))
        if not path.exists():
            path.write_text(expected)
    template = shlex.split((STAGE1 / "runs/PRIVATE_DP_E4" /
                            ("seed_%d" % seed) / "train.command.txt").read_text())
    template[template.index("-m") - 1] = str(PYTHON)
    option(template, "--hostfile", hostfile)
    option(template, "--gpu_mapping_file", mapping)
    option(template, "--data_file_path", DATA)
    option(template, "--partition_file_path", fixed)
    option(template, "--cloud_data_file_path", cloud_data)
    option(template, "--cloud_partition_file_path", cloud_part)
    option(template, "--model_name", MODEL)
    option(template, "--manual_seed", seed)
    option(template, "--run_id", seed)
    option(template, "--comm_round", 1 if args.smoke else 50)
    option(template, "--output_dir", dest / "model_output")
    if args.smoke:
        template.append("--audit_updates")
    for name, expected in (("--alpha", "0.5"), ("--zo_estimator", "legacy"),
                           ("--pool_size", "1"), ("--lr", "0.01")):
        if template[template.index(name) + 1] != expected:
            raise RuntimeError("template training flag changed: " + name)
    dest.mkdir(mode=0o700, parents=True)
    spec = {"schema_version": 1, "stage": "stage2_A2_smoke" if args.smoke else "stage2_A2",
            "status": "started",
            "arm": "SHUFFLED_DP_LABEL", "seed": seed, "trial_id": trial,
            "source_commit": commit, "host": host, "gpu_preflight": gpu,
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
            "source_data_sha256": SOURCE_SHA, "fixed_partition_sha256": PARTITION_SHA,
            "candidate_pool_sha256": POOL_SHA,
            "source_dp_release_sha256": original_manifest["selected_sha256"],
            "shuffled_jsonl_sha256": validation["postprocessed_sha256"],
            "cloud_data_sha256": pack["data_sha256"],
            "cloud_partition_sha256": pack["partition_sha256"],
            "official_test_used": False, "new_dp_release": False,
            "privacy_status": "postprocessing_of_existing_guidance_release_only"}
    (dest / "run_spec.json").write_text(json.dumps(spec, indent=2, sort_keys=True) + "\n")
    (dest / "train.command.txt").write_text(shlex.join(template) + "\n")
    env = os.environ.copy()
    env.pop("CUDA_VISIBLE_DEVICES", None)
    entry = REPO / "experiments/distributed/transformer_exps/run_tc_exps"
    env.update(PYTHONPATH=":".join(map(str, (entry, REPO, REPO / "FedML"))),
               PYTHONHASHSEED=str(seed), HEIMDALLM_FAULTHANDLER="1",
               PYTHONNOUSERSITE="1", WANDB_MODE="disabled", WANDB_SILENT="true",
               HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1",
               TOKENIZERS_PARALLELISM="false", OMP_NUM_THREADS="1",
               MKL_NUM_THREADS="1", OMPI_MCA_btl_vader_single_copy_mechanism="none",
               OMPI_ALLOW_RUN_AS_ROOT="1", OMPI_ALLOW_RUN_AS_ROOT_CONFIRM="1")
    print("Starting A2 %s seed %d" % (trial, seed), flush=True)
    started = datetime.now(timezone.utc)
    with (dest / "train.log").open("w") as handle:
        proc = subprocess.run(template, cwd=str(entry), env=env, stdout=handle,
                              stderr=subprocess.STDOUT)
    (dest / "train.exit_code.txt").write_text(str(proc.returncode) + "\n")
    spec["wall_seconds"] = (datetime.now(timezone.utc) - started).total_seconds()
    if proc.returncode:
        spec["status"] = "train_failed"
        (dest / "run_spec.json").write_text(json.dumps(spec, indent=2, sort_keys=True) + "\n")
        raise RuntimeError("A2 training failed with exit code %d" % proc.returncode)
    if args.smoke:
        spec["status"] = "one_round_completed_pending_manual_operator_review"
        (dest / "run_spec.json").write_text(json.dumps(spec, indent=2, sort_keys=True) + "\n")
        print("A2 one-round smoke complete: " + str(dest), flush=True)
        return
    summary = [str(PYTHON), str(REPO / "journal_extension/summarize_fed_pilot.py"),
               "--log", str(dest / "train.log"), "--metrics-out", str(dest / "metrics.json"),
               "--manifest-out", str(dest / "manifest.json"),
               "--command-file", str(dest / "train.command.txt"),
               "--condition", "client_syn", "--cloud-source", "shuffled_dp_label_postprocess",
               "--evaluation-mode", "dev", "--stage", "stage2_a2",
               "--alpha", "0.5", "--seed", str(seed), "--rounds", "50",
               "--eval-every", "1", "--pre-update-eval", "--logical-clients", "2",
               "--clients-per-round", "2", "--mpi-workers", "1", "--exit-code", "0",
               "--artifact", "fixed_partition=" + str(fixed), "--require-complete"]
    (dest / "validate.command.txt").write_text(shlex.join(summary) + "\n")
    with (dest / "validate.log").open("w") as handle:
        valid = subprocess.run(summary, cwd=str(REPO), env=env, stdout=handle,
                               stderr=subprocess.STDOUT)
    (dest / "validate.exit_code.txt").write_text(str(valid.returncode) + "\n")
    spec["status"] = "complete" if valid.returncode == 0 else "validation_failed"
    (dest / "run_spec.json").write_text(json.dumps(spec, indent=2, sort_keys=True) + "\n")
    if valid.returncode:
        raise RuntimeError("A2 validation failed: " + str(dest))
    manifest_path = dest / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["baseline_semantics"] = (
        "alpha=0.5 uses the same 64 public texts as the paired PRIVATE_DP_E4 "
        "release, with a fixed cyclic class-label permutation. This is "
        "postprocessing of an existing guidance release and does not make "
        "the downstream FL transcript differentially private.")
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    print("A2 endpoint complete: " + str(dest), flush=True)


if __name__ == "__main__":
    main()
