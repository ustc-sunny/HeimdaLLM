#!/usr/bin/env python3
"""Run a fixed-budget, DP-postprocessing AG News filter pilot on a guarded GPU."""
import argparse
import csv
import hashlib
import json
import os
import shlex
import shutil
import socket
import statistics
import subprocess
import tarfile
from datetime import datetime, timezone
from pathlib import Path


ARMS = ("filtered_dp", "random_dp", "filtered_public")
SEEDS = (57, 58, 59)
RUN_ID = "ustc_agnews_dp_filter_20261004"
SELECTION_CODE_REVISION = "64ccf96"


def now():
    return datetime.now(timezone.utc).isoformat()


def digest(path):
    result = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            result.update(chunk)
    return result.hexdigest()


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    temp.replace(path)


def execute(command, case, stage, env=None):
    case.mkdir(parents=True, exist_ok=True)
    (case / (stage + ".command.txt")).write_text(shlex.join([str(x) for x in command]) + "\n")
    print("Starting %s %s at %s" % (case, stage, now()), flush=True)
    with (case / (stage + ".log")).open("w") as log:
        outcome = subprocess.run([str(x) for x in command], cwd=case, env=env,
                                 stdout=log, stderr=subprocess.STDOUT)
    (case / (stage + ".exit_code.txt")).write_text(str(outcome.returncode) + "\n")
    if outcome.returncode:
        raise RuntimeError("%s failed (%d): %s" % (stage, outcome.returncode, case))


def replace_arg(tokens, flag, value):
    if tokens.count(flag) != 1:
        raise ValueError("expected exactly one " + flag)
    tokens[tokens.index(flag) + 1] = str(value)


def check_gpu(gpu, min_free_mib=14000, max_utilization=20):
    outcome = subprocess.run(["nvidia-smi", "--query-gpu=index,memory.free,memory.used,utilization.gpu",
                              "--format=csv,noheader,nounits"], capture_output=True,
                             text=True, check=True)
    rows = [[int(x.strip()) for x in line.split(",")] for line in outcome.stdout.splitlines()]
    selected = next((row for row in rows if row[0] == gpu), None)
    if selected is None:
        raise ValueError("selected GPU is absent")
    index, free, used, utilization = selected
    if free < min_free_mib or utilization > max_utilization:
        raise RuntimeError("GPU%d busy: %d MiB free, %d%% utilization" %
                           (index, free, utilization))
    return {"gpu": index, "free_mib": free, "used_mib": used,
            "utilization_percent": utilization, "checked_at": now()}


def update_summary(root, protocol):
    runs, curves, missing = [], [], []
    for case in protocol["cases"]:
        arm, seed = case["arm"], case["seed"]
        path = root / arm / ("seed_%d" % seed)
        metric = path / "metrics.json"
        receipt = path / "archive_receipt.json"
        if not metric.is_file() or not receipt.is_file():
            missing.append(case)
            continue
        value = json.loads(metric.read_text())
        if value["status"] != "complete" or not all(value["checks"].values()):
            raise ValueError("invalid run metrics: " + str(path))
        if [row["round"] for row in value["evaluations"]] != list(range(-1, 50)):
            raise ValueError("incomplete 50-round trajectory")
        runs.append({"arm": arm, "seed": seed,
                     "final_accuracy": value["final"]["acc"],
                     "final_macro_f1": value["final"]["macro_f1"],
                     "final_loss": value["final"]["eval_loss"],
                     "metrics_sha256": digest(metric),
                     "archive": json.loads(receipt.read_text())})
        for row in value["evaluations"]:
            curves.append({"arm": arm, "seed": seed, "round": row["round"],
                           "accuracy": row["acc"], "macro_f1": row["macro_f1"],
                           "loss": row["eval_loss"]})
    aggregate = {}
    for arm in ARMS:
        values = [r["final_accuracy"] for r in runs if r["arm"] == arm]
        if values:
            aggregate[arm] = {"n": len(values), "accuracy_mean": statistics.mean(values),
                              "accuracy_sample_sd": statistics.stdev(values)
                              if len(values) > 1 else None}
    summary = {"status": "complete" if not missing else "partial",
               "updated_at": now(), "runs_completed": len(runs),
               "runs_expected": len(protocol["cases"]), "missing": missing,
               "protocol": protocol, "runs": runs, "aggregate": aggregate,
               "privacy_scope": "Filtered/random DP records are postprocessing of the existing epsilon<=8, delta=1e-5 synthetic release; all downstream FL updates are Non-DP.",
               "private_text_included": False}
    dest = root / "summary"
    write_json(dest / "summary.json", summary)
    temporary = dest / "curves.csv.tmp"
    with temporary.open("w", newline="") as output:
        writer = csv.DictWriter(output, fieldnames=["arm", "seed", "round", "accuracy", "macro_f1", "loss"])
        writer.writeheader()
        writer.writerows(curves)
    temporary.replace(dest / "curves.csv")
    lines = ["# AG News DP synthetic postprocessing filter", "",
             "Status: %d/%d runs completed." % (len(runs), len(protocol["cases"])), "",
             "Only the existing DP synthetic release has record-level epsilon<=8, delta=1e-5; downstream FL is Non-DP.",
             "Official test untouched; evaluation uses the fixed 512-record development set.", "",
             "| Arm | n | Final accuracy (%) | Sample SD (pp) |",
             "|---|---:|---:|---:|"]
    for arm in ARMS:
        if arm in aggregate:
            item = aggregate[arm]
            sd = item["accuracy_sample_sd"]
            lines.append("| %s | %d | %.4f | %s |" %
                         (arm, item["n"], 100 * item["accuracy_mean"],
                          "—" if sd is None else "%.4f" % (100 * sd)))
    lines += ["", "The selector retains 8 of 16 samples per source client/class; each FL arm uses 64 samples.",
              "Random DP and filtered public are matched-budget controls.",
              "No significance or end-to-end DP claim is made.", ""]
    (dest / "REPORT.md").write_text("\n".join(lines))
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--work-root", type=Path, required=True)
    parser.add_argument("--selection-dir", type=Path, required=True)
    parser.add_argument("--gpu", type=int, default=2)
    parser.add_argument("--pilot-only", action="store_true")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    if args.gpu not in range(4):
        parser.error("GPU must be one of the server's physical devices 0-3")
    base = args.work_root
    root = base / "results" / RUN_ID
    selection = json.loads((args.selection_dir / "selection_summary.json").read_text())
    if selection["status"] != "complete" or len(selection["runs"]) != 9:
        raise ValueError("selection is incomplete")
    source_bridge = json.loads((base / "results/ustc_agnews_noise_bridge_20260930/summary/summary.json").read_text())
    if source_bridge["status"] != "complete" or source_bridge["runs_completed"] != 6:
        raise ValueError("source DP bridge is incomplete")
    cases = [{"arm": arm, "seed": seed} for seed in SEEDS for arm in ARMS]
    protocol = {"schema_version": 1, "run_id": RUN_ID, "phase": "formal_pilot_then_continuation",
                "cases": cases, "rounds": 50, "seeds": list(SEEDS), "arms": list(ARMS),
                "selected_gpu": args.gpu, "gpu_preflight_min_free_mib": 14000,
                "gpu_preflight_max_utilization_percent": 20,
                "selection_code_revision": SELECTION_CODE_REVISION,
                "runner_sha256": digest(Path(__file__)),
                "selection_summary_sha256": digest(args.selection_dir / "selection_summary.json"),
                "source_bridge_summary_sha256": digest(base / "results/ustc_agnews_noise_bridge_20260930/summary/summary.json"),
                "selection_records_per_arm": 64, "logical_clients": [1, 21],
                "dev_records": 512, "official_test_used": False,
                "privacy_scope": "record_level_dp_synthetic_release_only_for_filtered_dp_and_random_dp",
                "downstream_guided_estimator": "legacy", "downstream_guided_learning_rate": 0.01,
                "objective_queries_per_run": 3000}
    if root.exists():
        if not args.resume or json.loads((root / "protocol.json").read_text()) != protocol:
            raise ValueError("existing result root/protocol mismatch")
    else:
        shared = root / "shared"
        shared.mkdir(parents=True)
        historical = base / "results/ustc_agnews_noise_bridge_20260930/shared"
        shutil.copy2(historical / "fixed_partition.h5", shared / "fixed_partition.h5")
        host = socket.gethostname().split(".")[0]
        count = [0, 0, 0, 0]
        count[args.gpu] = 3
        (shared / "gpu_mapping.yaml").write_text("mapping_filter:\n  %s: %s\n" %
                                                  (host, json.dumps(count)))
        (shared / "mpi_host_file").write_text("%s slots=3\n" % host)
        write_json(root / "protocol.json", protocol)
    update_summary(root, protocol)
    repo = base / "HeimdaLLM"
    script = repo / "journal_extension"
    env = os.environ.copy()
    env.pop("CUDA_VISIBLE_DEVICES", None)
    entry = repo / "experiments/distributed/transformer_exps/run_tc_exps"
    env.update(PYTHONPATH="%s:%s:%s" % (entry, repo, repo / "FedML"),
               HEIMDALLM_FAULTHANDLER="1", PYTHONNOUSERSITE="1", WANDB_MODE="disabled",
               WANDB_SILENT="true", TOKENIZERS_PARALLELISM="false",
               OMP_NUM_THREADS="1", MKL_NUM_THREADS="1", HF_HUB_OFFLINE="1",
               TRANSFORMERS_OFFLINE="1", OMPI_MCA_btl_vader_single_copy_mechanism="none")
    for case in cases:
        arm, seed = case["arm"], case["seed"]
        if args.pilot_only and seed != 57:
            break
        path = root / arm / ("seed_%d" % seed)
        if (path / "archive_receipt.json").is_file():
            print("Skipping archived %s seed %d" % (arm, seed), flush=True)
            continue
        if path.exists():
            raise ValueError("incomplete case retained; inspect before retry: " + str(path))
        path.mkdir(parents=True)
        write_json(root / "state.json", {"status": "running", "arm": arm,
                                          "seed": seed, "updated_at": now()})
        try:
            preflight = check_gpu(args.gpu)
            write_json(path / "gpu_preflight.json", preflight)
            selected = next(r for r in selection["runs"] if r["arm"] == arm and r["seed"] == seed)
            source = args.selection_dir / arm / ("seed_%d.jsonl" % seed)
            if digest(source) != selected["selected_sha256"]:
                raise ValueError("selected release checksum mismatch")
            shutil.copy2(source, path / "selected.jsonl")
            execute([base / "envs/kdd/bin/python", script / "pack_synthetic_h5.py",
                     "--jsonl", path / "selected.jsonl", "--source-data-file",
                     base / "data/fednlp_data/data_files/agnews_data.h5",
                     "--data-out", path / "cloud_data.h5", "--partition-out",
                     path / "cloud_partition.h5", "--partition-method", "synthetic_cloud",
                     "--cloud-clients", "1", "--require-equal-label-counts",
                     "--manifest-out", path / "pack_manifest.json"], path, "pack", env)
            historical = base / "results/ustc_agnews_noise_bridge_20260930/dp_eps8" / ("seed_%d" % seed)
            command = shlex.split((historical / "train.command.txt").read_text())
            replace_arg(command, "--gpu_mapping_file", root / "shared/gpu_mapping.yaml")
            replace_arg(command, "--gpu_mapping_key", "mapping_filter")
            replace_arg(command, "--hostfile", root / "shared/mpi_host_file")
            replace_arg(command, "--partition_file_path", root / "shared/fixed_partition.h5")
            replace_arg(command, "--cloud_data_file_path", path / "cloud_data.h5")
            replace_arg(command, "--cloud_partition_file_path", path / "cloud_partition.h5")
            replace_arg(command, "--output_dir", path / "model_output")
            env["PYTHONHASHSEED"] = str(seed)
            execute(command, path, "train", env)
            condition = "public_syn" if arm == "filtered_public" else "client_syn"
            execute([base / "envs/kdd/bin/python", script / "summarize_fed_pilot.py",
                     "--log", path / "train.log", "--metrics-out", path / "metrics.json",
                     "--manifest-out", path / "manifest.json", "--command-file",
                     path / "train.command.txt", "--condition", condition,
                     "--cloud-source", "dp_filter_" + arm, "--evaluation-mode", "dev",
                     "--stage", "dp_filter_pilot", "--alpha", "0.5", "--seed", str(seed),
                     "--rounds", "50", "--eval-every", "1", "--pre-update-eval",
                     "--logical-clients", "2", "--clients-per-round", "2",
                     "--mpi-workers", "1", "--exit-code", "0",
                     "--artifact", "protocol=" + str(root / "protocol.json"),
                     "--artifact", "fixed_partition=" + str(root / "shared/fixed_partition.h5"),
                     "--artifact", "selection_summary=" + str(args.selection_dir / "selection_summary.json"),
                     "--require-complete"], path, "validate", env)
            archive_dir = base / "backups"
            archive_dir.mkdir(exist_ok=True)
            name = "%s_%s_seed%d.tar.gz" % (RUN_ID, arm, seed)
            archive = archive_dir / name
            temporary = archive_dir / (name + ".tmp")
            with tarfile.open(temporary, "w:gz") as output:
                output.add(path, arcname="%s/seed_%d" % (arm, seed))
                output.add(root / "shared", arcname="shared")
                output.add(root / "protocol.json", arcname="protocol.json")
            temporary.replace(archive)
            checksum = digest(archive)
            archive.with_suffix(archive.suffix + ".sha256").write_text(checksum + "  " + name + "\n")
            write_json(path / "archive_receipt.json", {"name": name, "sha256": checksum})
            summary = update_summary(root, protocol)
            print("Completed %s seed %d, progress %d/%d" %
                  (arm, seed, summary["runs_completed"], summary["runs_expected"]), flush=True)
        except Exception:
            write_json(root / "state.json", {"status": "failed", "arm": arm,
                                              "seed": seed, "updated_at": now()})
            raise
    summary = update_summary(root, protocol)
    write_json(root / "state.json", {"status": "pilot_complete" if args.pilot_only else
                                      "complete" if summary["status"] == "complete" else "partial",
                                      "updated_at": now()})


if __name__ == "__main__":
    main()
