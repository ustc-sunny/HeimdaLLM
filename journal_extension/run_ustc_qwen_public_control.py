#!/usr/bin/env python3
"""One-arm, guarded AG News public-selection control against the frozen GPU2 pilot."""

import csv
import json
import os
import shlex
import shutil
import tarfile
from pathlib import Path

from run_ustc_augpe_nondp_pilot import check_gpu, digest, now, replace, run, write_json


RUN_ID = "ustc_agnews_qwen_public_control_20261013"
OLD_ID = "ustc_agnews_private_vote_pilot_20261009"
CANDIDATE_SHA = "7e54c05ce2dc94d754b178218963dedfceb781c491b97ed1d9db5ab9e7d7c54f"
ARMS = ("private_vote_synthetic", "public_random_selection", "same_source_real")


def option(command, name):
    if command.count(name) != 1:
        raise ValueError("unexpected historical command option " + name)
    return command[command.index(name) + 1]


def reference_metrics(path):
    value = json.loads(path.read_text())
    if value["status"] != "complete" or not all(value["checks"].values()):
        raise ValueError("invalid reference metrics: " + str(path))
    if [point["round"] for point in value["evaluations"]] != list(range(-1, 50)):
        raise ValueError("incomplete reference trajectory: " + str(path))
    return value


def summarize(root, old, protocol):
    runs, curves = [], []
    for arm in ARMS:
        path = (root if arm == "public_random_selection" else old) / arm / "metrics.json"
        value = reference_metrics(path)
        runs.append({"arm": arm, "seed": 57, "final_accuracy": value["final"]["acc"],
                     "final_macro_f1": value["final"]["macro_f1"],
                     "late_mean_accuracy": sum(point["acc"] for point in value["evaluations"]
                                               if point["round"] >= 40) / 10,
                     "metrics_sha256": digest(path),
                     "reference_run": OLD_ID if arm != "public_random_selection" else RUN_ID})
        for point in value["evaluations"]:
            curves.append({"arm": arm, "round": point["round"],
                           "accuracy": point["acc"], "macro_f1": point["macro_f1"],
                           "loss": point["eval_loss"]})
    summary = root / "summary"
    summary.mkdir(exist_ok=True)
    payload = {"schema_version": 1, "status": "complete", "updated_at": now(),
               "runs_completed": 3, "runs_expected": 3, "runs": runs, "protocol": protocol,
               "official_test_used": False, "privacy": "Non-DP",
               "private_text_included": False}
    write_json(summary / "summary.json", payload)
    with (summary / "curves.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["arm", "round", "accuracy", "macro_f1", "loss"])
        writer.writeheader()
        writer.writerows(curves)
    score = {row["arm"]: row["final_accuracy"] for row in runs}
    lines = ["# AG News Qwen public-selection control", "",
             "Non-DP seed-57 development-set comparison; 50 rounds, 51 evaluations per arm.",
             "The private-vote and same-source-real arms are immutable references from " + OLD_ID + ".", "",
             "| Guidance | Final dev accuracy |", "|---|---:|"]
    for row in runs:
        lines.append("| %s | %.2f%% |" % (row["arm"], 100 * row["final_accuracy"]))
    lines += ["", "Public selection minus private vote: %+.2f pp." %
              (100 * (score["public_random_selection"] - score["private_vote_synthetic"])),
              "Public selection minus same-source real: %+.2f pp." %
              (100 * (score["public_random_selection"] - score["same_source_real"])), "",
              "One seed, no DP, no official test. A stronger public-only selector remains an open control."]
    (summary / "REPORT.md").write_text("\n".join(lines) + "\n")


def main():
    base = Path("/home/zzkevin/heimdallm-journal-20260928")
    repo = base / "HeimdaLLM"
    scripts = repo / "journal_extension"
    old = base / "results" / OLD_ID
    old_private = base / "private_staging" / OLD_ID
    root = base / "results" / RUN_ID
    private = base / "private_staging" / RUN_ID
    if root.exists() or private.exists():
        raise FileExistsError("run already exists; preserve previous results")
    old_summary = json.loads((old / "summary/summary.json").read_text())
    old_generator = json.loads((old_private / "generated/manifest.json").read_text())
    if old_summary["status"] != "complete" or old_generator["candidate_sha256"] != CANDIDATE_SHA:
        raise ValueError("reference pilot incomplete or candidate pool changed")
    if digest(old_private / "generated/candidates.jsonl") != CANDIDATE_SHA:
        raise ValueError("reference candidate file changed")
    reference_sha = {row["arm"]: row["metrics_sha256"] for row in old_summary["runs"]}
    for arm in ("private_vote_synthetic", "same_source_real"):
        reference_metrics(old / arm / "metrics.json")
        if digest(old / arm / "metrics.json") != reference_sha[arm]:
            raise ValueError("reference metrics hash changed: " + arm)
    old_command = shlex.split((old / "private_vote_synthetic/train.command.txt").read_text())
    for flag, expected in (("--dataset", "agnews"), ("--cloud_dataset", "agnews"),
                           ("--manual_seed", "57"), ("--comm_round", "50"),
                           ("--client_num_in_total", "2"), ("--client_num_per_round", "2"),
                           ("--alpha", "0.5"), ("--learning_rate", "0.01")):
        if option(old_command, flag) != expected:
            raise ValueError("historical task configuration differs: " + flag)
    check_gpu(2)
    root.mkdir(parents=True)
    private.mkdir(parents=True, mode=0o700)
    os.chmod(private, 0o700)
    shared = root / "shared"
    shutil.copytree(old / "shared", shared)
    for filename in ("fixed_partition.h5", "gpu_mapping.yaml", "mpi_host_file"):
        if digest(shared / filename) != digest(old / "shared" / filename):
            raise ValueError("shared task configuration changed")
    protocol = {"schema_version": 1, "run_id": RUN_ID, "seed": 57, "gpu": 2,
                "rounds": 50, "dev_records": 512, "guidance_records": 64,
                "source_clients": [1, 21], "candidate_records": 128,
                "candidate_sha256": CANDIDATE_SHA,
                "selection": "public_sha256_index_order_v1_no_private_records",
                "reference_run": OLD_ID,
                "reference_private_vote_metrics_sha256": digest(old / "private_vote_synthetic/metrics.json"),
                "reference_real_metrics_sha256": digest(old / "same_source_real/metrics.json"),
                "reference_fixed_partition_sha256": digest(old / "shared/fixed_partition.h5"),
                "runner_sha256": digest(Path(__file__)), "official_test_used": False,
                "privacy": "Non-DP"}
    write_json(root / "protocol.json", protocol)
    write_json(root / "state.json", {"status": "running", "stage": "public_pool_replay", "updated_at": now()})
    env = os.environ.copy()
    env.pop("CUDA_VISIBLE_DEVICES", None)
    env.update(PYTHONNOUSERSITE="1", HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1",
               TOKENIZERS_PARALLELISM="false", OMP_NUM_THREADS="1", MKL_NUM_THREADS="1")
    try:
        pool = private / "public_pool"
        run([base / "envs/augpe/bin/python", scripts / "replay_public_qwen_pool.py",
             "--output-dir", pool, "--generator-model", base / "models/Qwen2.5-3B-Instruct",
             "--task-tokenizer", base / "models/distilbert-base-uncased",
             "--expected-sha256", CANDIDATE_SHA, "--gpu", "2", "--seed", "57"],
            private, "replay_pool", env)
        selected = private / "public_selection"
        run([base / "envs/kdd/bin/python", scripts / "select_public_qwen_candidates.py",
             "--candidates-jsonl", pool / "candidates.jsonl", "--output-dir", selected,
             "--expected-candidate-sha256", CANDIDATE_SHA, "--seed", "57"],
            private, "select_public", env)
        run([base / "envs/augpe/bin/python", scripts / "audit_private_vote_pilot.py",
             "--jsonl", selected / "synthetic.jsonl", "--output", root / "quality.json"],
            private, "quality", env)
        path = root / "public_random_selection"
        path.mkdir()
        write_json(root / "state.json", {"status": "running", "stage": "pack", "updated_at": now()})
        run([base / "envs/kdd/bin/python", scripts / "pack_synthetic_h5.py",
             "--jsonl", selected / "synthetic.jsonl",
             "--source-data-file", base / "data/fednlp_data/data_files/agnews_data.h5",
             "--data-out", path / "cloud_data.h5", "--partition-out", path / "cloud_partition.h5",
             "--partition-method", "synthetic_cloud", "--cloud-clients", "1",
             "--require-equal-label-counts", "--manifest-out", path / "pack_manifest.json"],
            path, "pack", env)
        write_json(path / "gpu_preflight.json", check_gpu(2))
        command = old_command[:]
        for flag, value in (("--gpu_mapping_file", shared / "gpu_mapping.yaml"),
                            ("--hostfile", shared / "mpi_host_file"),
                            ("--partition_file_path", shared / "fixed_partition.h5"),
                            ("--cloud_data_file_path", path / "cloud_data.h5"),
                            ("--cloud_partition_file_path", path / "cloud_partition.h5"),
                            ("--output_dir", path / "model_output")):
            replace(command, flag, value)
        task_env = env.copy()
        entry = repo / "experiments/distributed/transformer_exps/run_tc_exps"
        task_env.update(PYTHONPATH="%s:%s:%s" % (entry, repo, repo / "FedML"),
                        PYTHONHASHSEED="57", HEIMDALLM_FAULTHANDLER="1",
                        WANDB_MODE="disabled", WANDB_SILENT="true",
                        OMPI_MCA_btl_vader_single_copy_mechanism="none")
        write_json(root / "state.json", {"status": "running", "stage": "train", "updated_at": now()})
        run(command, path, "train", task_env)
        run([base / "envs/kdd/bin/python", scripts / "summarize_fed_pilot.py",
             "--log", path / "train.log", "--metrics-out", path / "metrics.json",
             "--manifest-out", path / "manifest.json", "--command-file", path / "train.command.txt",
             "--condition", "public_syn", "--cloud-source", "public_random_selection",
             "--evaluation-mode", "dev", "--stage", "qwen_public_selection_control",
             "--alpha", "0.5", "--seed", "57", "--rounds", "50", "--eval-every", "1",
             "--pre-update-eval", "--logical-clients", "2", "--clients-per-round", "2",
             "--mpi-workers", "1", "--exit-code", "0",
             "--artifact", "protocol=" + str(root / "protocol.json"),
             "--artifact", "fixed_partition=" + str(shared / "fixed_partition.h5"),
             "--require-complete"], path, "validate", env)
        summarize(root, old, protocol)
        archive = base / "backups" / (RUN_ID + ".tar.gz")
        with tarfile.open(str(archive) + ".tmp", "w:gz") as handle:
            handle.add(path, arcname="public_random_selection")
            handle.add(shared, arcname="shared")
            handle.add(private / "public_pool", arcname="public_pool")
            handle.add(private / "public_selection", arcname="public_selection")
            handle.add(root / "protocol.json", arcname="protocol.json")
            handle.add(root / "quality.json", arcname="quality.json")
        Path(str(archive) + ".tmp").replace(archive)
        Path(str(archive) + ".sha256").write_text(digest(archive) + "  " + archive.name + "\n")
        write_json(root / "state.json", {"status": "complete", "updated_at": now()})
        print("Control complete; archive SHA-256 " + digest(archive), flush=True)
    except Exception:
        write_json(root / "state.json", {"status": "failed", "updated_at": now()})
        raise


if __name__ == "__main__":
    main()
