#!/usr/bin/env python3
"""Run one preregistered pure-ZOO NO_GUIDANCE control in a separate clone."""

import argparse
import json
import os
import shlex
import socket
import subprocess
from datetime import datetime, timezone
from pathlib import Path

from run_stage2_a2 import check_gpu, digest, option


STAGE1 = Path("/mnt/heimdallm-stage1-20261015")
STAGE2 = Path("/mnt/heimdallm-stage2-20261016")
REPO = Path("/root/heimdallm-stage2-20261016/HeimdaLLM_A1")
DATA = Path("/root/heimdallm-work/data/fednlp_data/data_files/agnews_data.h5")
MODEL = Path("/root/heimdallm-work/models/distilbert-base-uncased")
PYTHON = Path("/root/miniconda3/envs/heimdallm-kdd/bin/python")
COMMIT = "ad325aa320789c4f59f911898033844d5b19ccec"
SOURCE_SHA = "1474dd264d535b50ad2a7b1cf714bf11e182c2fe5b07f5e982feb31a9b114616"
PARTITION_SHA = "e16e8551a87ba702cc23b4d56d57b9762b77c4fea8bf341b96d5759be7996a6c"
AFTER_SHA = {
    "forward_training/tc_transformer_trainer_distribute.py":
        "d2b346671ec1658b1d0da837dce1159ecc9e29e4d49a226bef8c0759254dc2dc",
    "experiments/distributed/transformer_exps/run_tc_exps/fedavg_main_tc.py":
        "afbc4f58b282c34b624c26878391aa13f134c996224ec15467d6c4a8aa17e6f8",
}


def remove_option(command, name):
    if name not in command:
        raise ValueError("missing cloud option " + name)
    index = command.index(name)
    del command[index:index + 2]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, choices=(57, 58, 59), required=True)
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    seed = args.seed
    dest = (STAGE2 / "preflight/NO_GUIDANCE" if args.smoke else
            STAGE2 / "runs/NO_GUIDANCE") / ("seed_%d" % seed)
    if dest.exists():
        raise FileExistsError(str(dest))
    head = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=REPO, text=True).strip()
    tracked = set(subprocess.check_output(["git", "diff", "--name-only"],
                                           cwd=REPO, text=True).splitlines())
    if head != COMMIT or tracked != set(AFTER_SHA):
        raise RuntimeError("unexpected A1 source revision or changed-file set")
    for relative, expected in AFTER_SHA.items():
        if digest(REPO / relative) != expected:
            raise RuntimeError("A1 operator hash mismatch: " + relative)
    fixed = STAGE1 / "shared/fixed_partition.h5"
    for path, expected in ((DATA, SOURCE_SHA), (fixed, PARTITION_SHA)):
        if digest(path) != expected:
            raise RuntimeError("frozen input hash mismatch: " + str(path))
    gpu = check_gpu()
    shared = STAGE2 / "shared"
    shared.mkdir(mode=0o700, exist_ok=True)
    host = socket.gethostname()
    hostfile, mapping = shared / "mpi_host_file", shared / "gpu_mapping.yaml"
    for path, expected in ((hostfile, host + " slots=3\n"),
                           (mapping, "mapping_private_vote:\n  %s: [3]\n" % host)):
        if not path.exists() or path.read_text() != expected:
            raise RuntimeError("missing or conflicting frozen MPI mapping: " + str(path))
    template = shlex.split((STAGE1 / "runs/PRIVATE_DP_E4" /
                            ("seed_%d" % seed) / "train.command.txt").read_text())
    template[template.index("-m") - 1] = str(PYTHON)
    option(template, "--hostfile", hostfile)
    option(template, "--gpu_mapping_file", mapping)
    option(template, "--data_file_path", DATA)
    option(template, "--partition_file_path", fixed)
    option(template, "--model_name", MODEL)
    option(template, "--manual_seed", seed)
    option(template, "--run_id", seed)
    option(template, "--comm_round", 1 if args.smoke else 50)
    option(template, "--output_dir", dest / "model_output")
    option(template, "--alpha", "1.0")
    option(template, "--zo_estimator", "isotropic_raw")
    for flag in ("--cloud_dataset", "--cloud_data_file_path",
                 "--cloud_partition_file_path", "--cloud_partition_method",
                 "--cloud_client_ids", "--cloud_max_seq_length"):
        remove_option(template, flag)
    if args.smoke:
        template.append("--audit_updates")
    for name, expected in (("--pool_size", "1"), ("--lr", "0.01"),
                           ("--fd_step", "0.01"), ("--beta", "1")):
        if template[template.index(name) + 1] != expected:
            raise RuntimeError("template training flag changed: " + name)
    if any(token.startswith("--cloud_") for token in template):
        raise RuntimeError("no-guidance command still contains cloud data options")
    dest.mkdir(mode=0o700, parents=True)
    spec = {"schema_version": 1,
            "stage": "stage2_A1_smoke" if args.smoke else "stage2_A1",
            "status": "started", "arm": "NO_GUIDANCE", "seed": seed,
            "source_commit": head, "modified_source_sha256": AFTER_SHA,
            "operator": "paper_eq_zoo_raw_isotropic_gaussian",
            "cloud_training_data": None, "cloud_bp_skipped": True,
            "host": host, "gpu_preflight": gpu,
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
            "source_data_sha256": SOURCE_SHA, "fixed_partition_sha256": PARTITION_SHA,
            "official_test_used": False, "new_dp_release": False,
            "privacy_status": "not_dp"}
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
    print("Starting A1 seed %d" % seed, flush=True)
    started = datetime.now(timezone.utc)
    with (dest / "train.log").open("w") as handle:
        proc = subprocess.run(template, cwd=entry, env=env, stdout=handle,
                              stderr=subprocess.STDOUT)
    (dest / "train.exit_code.txt").write_text(str(proc.returncode) + "\n")
    spec["wall_seconds"] = (datetime.now(timezone.utc) - started).total_seconds()
    if proc.returncode:
        spec["status"] = "train_failed"
        (dest / "run_spec.json").write_text(json.dumps(spec, indent=2, sort_keys=True) + "\n")
        raise RuntimeError("A1 training failed with exit code %d" % proc.returncode)
    if args.smoke:
        spec["status"] = "one_round_completed_pending_manual_operator_review"
        (dest / "run_spec.json").write_text(json.dumps(spec, indent=2, sort_keys=True) + "\n")
        print("A1 one-round smoke complete: " + str(dest), flush=True)
        return
    summary = [str(PYTHON), str(REPO / "journal_extension/summarize_fed_pilot.py"),
               "--log", str(dest / "train.log"), "--metrics-out", str(dest / "metrics.json"),
               "--manifest-out", str(dest / "manifest.json"),
               "--command-file", str(dest / "train.command.txt"),
               "--condition", "no_cloud", "--cloud-source", "none",
               "--evaluation-mode", "dev", "--stage", "stage2_a1_raw_zoo",
               "--alpha", "1.0", "--seed", str(seed), "--rounds", "50",
               "--eval-every", "1", "--pre-update-eval", "--logical-clients", "2",
               "--clients-per-round", "2", "--mpi-workers", "1", "--exit-code", "0",
               "--artifact", "fixed_partition=" + str(fixed), "--require-complete"]
    (dest / "validate.command.txt").write_text(shlex.join(summary) + "\n")
    with (dest / "validate.log").open("w") as handle:
        valid = subprocess.run(summary, cwd=REPO, env=env, stdout=handle,
                               stderr=subprocess.STDOUT)
    (dest / "validate.exit_code.txt").write_text(str(valid.returncode) + "\n")
    spec["status"] = "complete" if valid.returncode == 0 else "validation_failed"
    (dest / "run_spec.json").write_text(json.dumps(spec, indent=2, sort_keys=True) + "\n")
    if valid.returncode:
        raise RuntimeError("A1 validation failed: " + str(dest))
    manifest_path = dest / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["baseline_semantics"] = (
        "Pure ZOO with raw beta*z, z~N(0,I_d), central FD and no cloud guidance data; "
        "the cloud rank is a synchronization barrier and skips BP.")
    manifest["evaluation_semantics"] = (
        "Rank 1 evaluates the fixed 512-record real dev set; rank 0 loads no training records. "
        "Logical round -1 is the pre-update evaluation.")
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    print("A1 endpoint complete: " + str(dest), flush=True)


if __name__ == "__main__":
    main()
