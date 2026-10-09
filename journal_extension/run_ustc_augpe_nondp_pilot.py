#!/usr/bin/env python3
"""Guarded, paired AG News Non-DP private-vote generation pilot on GPU2.

This intentionally uses one seed and two source clients. It is a feasibility
comparison against matched real guidance, not a KDD-scale or DP result.
"""

import argparse
import csv
import hashlib
import json
import os
import shlex
import shutil
import socket
import subprocess
import tarfile
from datetime import datetime, timezone
from pathlib import Path


ARMS = ("private_vote_synthetic", "same_source_real")


def now():
    return datetime.now(timezone.utc).isoformat()


def digest(path):
    value = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            value.update(chunk)
    return value.hexdigest()


def write_json(path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    temp.replace(path)


def run(command, directory, stage, env):
    directory.mkdir(parents=True, exist_ok=True)
    (directory / (stage + ".command.txt")).write_text(shlex.join([str(x) for x in command]) + "\n")
    print("Starting %s in %s at %s" % (stage, directory, now()), flush=True)
    with (directory / (stage + ".log")).open("w") as handle:
        result = subprocess.run([str(x) for x in command], cwd=directory,
                                stdout=handle, stderr=subprocess.STDOUT, env=env)
    (directory / (stage + ".exit_code.txt")).write_text(str(result.returncode) + "\n")
    if result.returncode:
        raise RuntimeError("%s failed with exit %d; see %s" % (stage, result.returncode, directory))


def replace(command, flag, value):
    if command.count(flag) != 1:
        raise ValueError("expected exactly one %s in historical command" % flag)
    command[command.index(flag) + 1] = str(value)


def check_gpu(index, min_free_mib=14000):
    raw = subprocess.check_output([
        "nvidia-smi", "--query-gpu=index,memory.free,memory.used,utilization.gpu",
        "--format=csv,noheader,nounits"], text=True)
    rows = [tuple(int(x.strip()) for x in line.split(",")) for line in raw.splitlines()]
    row = next((row for row in rows if row[0] == index), None)
    if row is None or row[1] < min_free_mib or row[3] > 20:
        raise RuntimeError("GPU%d is too busy: %s" % (index, row))
    return {"gpu": index, "free_mib": row[1], "used_mib": row[2],
            "utilization_percent": row[3], "checked_at": now()}


def update_summary(root, protocol):
    runs, curves = [], []
    for arm in ARMS:
        path = root / arm
        metric = path / "metrics.json"
        if not metric.is_file():
            continue
        value = json.loads(metric.read_text())
        if value["status"] != "complete" or not all(value["checks"].values()):
            raise ValueError("invalid metrics: " + str(metric))
        if [row["round"] for row in value["evaluations"]] != list(range(-1, protocol["rounds"])):
            raise ValueError("incomplete evaluation trajectory: " + str(metric))
        runs.append({"arm": arm, "seed": protocol["seed"],
                     "final_accuracy": value["final"]["acc"],
                     "final_macro_f1": value["final"]["macro_f1"],
                     "metrics_sha256": digest(metric)})
        for point in value["evaluations"]:
            curves.append({"arm": arm, "round": point["round"],
                           "accuracy": point["acc"], "macro_f1": point["macro_f1"],
                           "loss": point["eval_loss"]})
    status = "complete" if len(runs) == len(ARMS) else "partial"
    payload = {"status": status, "updated_at": now(), "runs": runs,
               "runs_completed": len(runs), "runs_expected": len(ARMS),
               "protocol": protocol, "privacy": "Non-DP; private-vote selection", "official_test_used": False,
               "private_text_included": False}
    summary = root / "summary"
    write_json(summary / "summary.json", payload)
    (summary / "curves.csv").parent.mkdir(exist_ok=True)
    with (summary / "curves.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["arm", "round", "accuracy", "macro_f1", "loss"])
        writer.writeheader()
        writer.writerows(curves)
    lines = ["# AG News training-free private-vote pilot", "",
             "Status: %d/%d arms complete; seed %d, %d FL rounds, two source clients." %
             (len(runs), len(ARMS), protocol["seed"], protocol["rounds"]), "",
             "This is a one-step Aug-PE-inspired **Non-DP** pilot, not full Aug-PE or a KDD-scale result.",
             "The official AG News test set was not used; accuracy uses the fixed 512-record dev set.", "",
             "| Guidance | Final dev accuracy |", "|---|---:|"]
    for row in runs:
        lines.append("| %s | %.2f%% |" % (row["arm"], 100 * row["final_accuracy"]))
    if len(runs) == 2:
        values = {row["arm"]: row["final_accuracy"] for row in runs}
        lines += ["", "Synthetic minus matched real: %+.2f percentage points." %
                  (100 * (values[ARMS[0]] - values[ARMS[1]]))]
    (summary / "REPORT.md").write_text("\n".join(lines) + "\n")
    return payload


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--work-root", type=Path, default=Path("/home/zzkevin/heimdallm-journal-20260928"))
    parser.add_argument("--run-id", default="ustc_agnews_private_vote_pilot_20261009")
    parser.add_argument("--seed", type=int, default=57)
    parser.add_argument("--rounds", type=int, default=50)
    parser.add_argument("--gpu", type=int, default=2)
    parser.add_argument("--candidates-per-label", type=int, default=32)
    parser.add_argument("--selected-per-client-label", type=int, default=8)
    args = parser.parse_args()
    if args.gpu not in range(4) or args.seed != 57 or args.rounds != 50:
        parser.error("the fixed pilot requires GPU 0-3, seed 57 and 50 rounds")
    if any(c not in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789._-" for c in args.run_id):
        parser.error("invalid run ID")
    base = args.work_root
    repo = base / "HeimdaLLM"
    script = repo / "journal_extension"
    root = base / "results" / args.run_id
    private = base / "private_staging" / args.run_id
    if root.exists() or private.exists():
        raise FileExistsError("pilot already exists; preserve it and use a new run ID")
    check_gpu(args.gpu)
    shared = root / "shared"
    shared.mkdir(parents=True)
    private.mkdir(parents=True, mode=0o700)
    os.chmod(private, 0o700)
    historical = base / "results/ustc_agnews_noise_bridge_20260930"
    shutil.copy2(historical / "shared/fixed_partition.h5", shared / "fixed_partition.h5")
    host = socket.gethostname().split(".")[0]
    slots = [0] * 4
    slots[args.gpu] = 3
    (shared / "gpu_mapping.yaml").write_text("mapping_private_vote:\n  %s: %s\n" % (host, json.dumps(slots)))
    (shared / "mpi_host_file").write_text("%s slots=3\n" % host)
    protocol = {"schema_version": 1, "run_id": args.run_id, "seed": args.seed,
                "rounds": args.rounds, "gpu": args.gpu, "source_clients": [1, 21],
                "source_records_per_client": 120, "dev_records": 512,
                "candidates_per_label": args.candidates_per_label,
                "selected_per_client_label": args.selected_per_client_label,
                "guidance_records": args.selected_per_client_label * 8,
                "generator": "Qwen2.5-3B-Instruct public pretrained, no fine-tuning",
                "selection": "one-step private embedding votes without noise; Non-DP",
                "fl": "KDD FedFwd DistilBERT adapter, same historical guided configuration",
                "official_test_used": False, "runner_sha256": digest(Path(__file__)),
                "generator_sha256": digest(script / "augpe_nondp_pilot.py"),
                "source_data_sha256": digest(base / "data/fednlp_data/data_files/agnews_data.h5"),
                "source_partition_sha256": digest(base / "data/fednlp_data/partition_files/agnews_partition.h5")}
    write_json(root / "protocol.json", protocol)
    update_summary(root, protocol)
    env = os.environ.copy()
    env.pop("CUDA_VISIBLE_DEVICES", None)
    env.update(PYTHONNOUSERSITE="1", HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1",
               TOKENIZERS_PARALLELISM="false", OMP_NUM_THREADS="1", MKL_NUM_THREADS="1")
    staging = private / "source"
    run([base / "envs/kdd/bin/python", script / "export_client_train_jsonl.py",
         "--data-file", base / "data/fednlp_data/data_files/agnews_data.h5",
         "--partition-file", base / "data/fednlp_data/partition_files/agnews_partition.h5",
         "--partition-method", "uniform_client_1000", "--client-ids", "1,21",
         "--output-dir", staging, "--seed", str(args.seed),
         "--sample-limit-per-client", "120", "--no-aggregate-output"], private, "export", env)
    generated = private / "generated"
    run([base / "envs/augpe/bin/python", script / "augpe_nondp_pilot.py",
         "--client-json-dir", staging, "--output-dir", generated,
         "--generator-model", base / "models/Qwen2.5-3B-Instruct",
         "--embedding-model", base / "models/all-MiniLM-L6-v2",
         "--task-tokenizer", base / "models/distilbert-base-uncased",
         "--gpu", str(args.gpu), "--seed", str(args.seed),
         "--candidates-per-label", str(args.candidates_per_label),
         "--selected-per-client-label", str(args.selected_per_client_label)],
        private, "generate", env)
    real = private / "real_matched.jsonl"
    run([base / "envs/kdd/bin/python", script / "build_matched_real_control.py",
         "--data-file", base / "data/fednlp_data/data_files/agnews_data.h5",
         "--partition-file", base / "data/fednlp_data/partition_files/agnews_partition.h5",
         "--partition-method", "uniform_client_1000", "--client-ids", "1,21",
         "--reference-jsonl", generated / "synthetic.jsonl", "--output", real,
         "--manifest-out", private / "real_matched_manifest.json", "--seed", str(args.seed)],
        private, "match_real", env)
    for arm, source in ((ARMS[0], generated / "synthetic.jsonl"), (ARMS[1], real)):
        path = root / arm
        path.mkdir()
        try:
            write_json(root / "state.json", {"status": "running", "arm": arm, "updated_at": now()})
            run([base / "envs/kdd/bin/python", script / "pack_synthetic_h5.py",
                 "--jsonl", source,
                 "--source-data-file", base / "data/fednlp_data/data_files/agnews_data.h5",
                 "--data-out", path / "cloud_data.h5", "--partition-out", path / "cloud_partition.h5",
                 "--partition-method", "synthetic_cloud", "--cloud-clients", "1",
                 "--require-equal-label-counts", "--manifest-out", path / "pack_manifest.json"],
                path, "pack", env)
            write_json(path / "gpu_preflight.json", check_gpu(args.gpu))
            template = historical / ("dp_eps8/seed_%d/train.command.txt" % args.seed)
            command = shlex.split(template.read_text())
            replace(command, "--gpu_mapping_file", shared / "gpu_mapping.yaml")
            replace(command, "--gpu_mapping_key", "mapping_private_vote")
            replace(command, "--hostfile", shared / "mpi_host_file")
            replace(command, "--partition_file_path", shared / "fixed_partition.h5")
            replace(command, "--cloud_data_file_path", path / "cloud_data.h5")
            replace(command, "--cloud_partition_file_path", path / "cloud_partition.h5")
            replace(command, "--output_dir", path / "model_output")
            task_env = env.copy()
            entry = repo / "experiments/distributed/transformer_exps/run_tc_exps"
            task_env.update(PYTHONPATH="%s:%s:%s" % (entry, repo, repo / "FedML"),
                            PYTHONHASHSEED=str(args.seed), HEIMDALLM_FAULTHANDLER="1",
                            WANDB_MODE="disabled", WANDB_SILENT="true",
                            OMPI_MCA_btl_vader_single_copy_mechanism="none")
            run(command, path, "train", task_env)
            run([base / "envs/kdd/bin/python", script / "summarize_fed_pilot.py",
                 "--log", path / "train.log", "--metrics-out", path / "metrics.json",
                 "--manifest-out", path / "manifest.json", "--command-file", path / "train.command.txt",
                 "--condition", "client_syn" if arm == ARMS[0] else "same_source_real_matched",
                 "--cloud-source", arm, "--evaluation-mode", "dev", "--stage", "private_vote_pilot",
                 "--alpha", "0.5", "--seed", str(args.seed), "--rounds", str(args.rounds),
                 "--eval-every", "1", "--pre-update-eval", "--logical-clients", "2",
                 "--clients-per-round", "2", "--mpi-workers", "1", "--exit-code", "0",
                 "--artifact", "protocol=" + str(root / "protocol.json"),
                 "--artifact", "fixed_partition=" + str(shared / "fixed_partition.h5"),
                 "--require-complete"], path, "validate", env)
            archive_dir = base / "backups"
            archive_dir.mkdir(exist_ok=True)
            name = "%s_%s.tar.gz" % (args.run_id, arm)
            archive = archive_dir / name
            with tarfile.open(str(archive) + ".tmp", "w:gz") as handle:
                handle.add(path, arcname=arm)
                handle.add(shared, arcname="shared")
                handle.add(root / "protocol.json", arcname="protocol.json")
            Path(str(archive) + ".tmp").replace(archive)
            Path(str(archive) + ".sha256").write_text(digest(archive) + "  " + name + "\n")
            update_summary(root, protocol)
        except Exception:
            write_json(root / "state.json", {"status": "failed", "arm": arm, "updated_at": now()})
            raise
    summary = update_summary(root, protocol)
    write_json(root / "state.json", {"status": summary["status"], "updated_at": now()})


if __name__ == "__main__":
    main()
