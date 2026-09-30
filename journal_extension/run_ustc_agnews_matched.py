#!/usr/bin/env python3
"""Sequential Non-DP ablations on shared physical GPU 1, with per-run archives.

This runner never trains a DP generator or accesses the official test set.
Probes calibrate an opt-in isotropic estimator; formal arms preserve the legacy
guided estimator so generator changes can be compared to DP v1 descriptively.
"""
import argparse
import csv
import fcntl
import hashlib
import io
import json
import os
import shlex
import socket
import statistics
import subprocess
import tarfile
from datetime import datetime, timezone
from pathlib import Path

import non_dp_client_synthetic as common


SCRIPT = Path(__file__).resolve().parent
REPO = SCRIPT.parent
ARMS = ("public", "ordinary", "fixed_example", "poisson", "clipped", "no_guidance")
BRIDGE_ARMS = ("zero_noise", "dp_eps8")
PROBES = {"legacy_lr001": ("legacy", 0.01),
          "isotropic_raw_lr001": ("isotropic_raw", 0.01),
          "isotropic_lr001": ("isotropic", 0.01),
          "isotropic_lr0001": ("isotropic", 0.001),
          "isotropic_lr00001": ("isotropic", 0.0001)}


def now():
    return datetime.now(timezone.utc).isoformat()


def execute(command, directory, name, env=None, cwd=None):
    directory.mkdir(parents=True, exist_ok=True)
    (directory / (name + ".command.txt")).write_text(shlex.join(map(str, command)) + "\n")
    print("Starting %s %s at %s" % (directory.name, name, now()), flush=True)
    with (directory / (name + ".log")).open("w") as handle:
        result = subprocess.run(list(map(str, command)), stdout=handle,
                                stderr=subprocess.STDOUT, env=env, cwd=cwd)
    (directory / (name + ".exit_code.txt")).write_text(str(result.returncode) + "\n")
    if result.returncode:
        raise RuntimeError("%s failed (%d); see %s" % (name, result.returncode, directory))


def validate_zero_noise_manifest(path):
    manifest = json.loads(path.read_text())
    privacy = manifest.get("privacy", {})
    if manifest.get("status") != "complete" or manifest.get("is_record_level_dp") is not False:
        raise ValueError("zero-noise generator must be complete and labeled Non-DP")
    if privacy.get("noise_added_to_clipped_gradient_sum") is not False or \
       privacy.get("noise_multipliers") != [0.0] or \
       privacy.get("accountant") != "none_non_dp_control":
        raise ValueError("zero-noise generator privacy metadata mismatch")
    if manifest.get("synthetic_records") != 128:
        raise ValueError("zero-noise generator quota mismatch")


def publish_summary(root):
    config = json.loads((root / "protocol.json").read_text())
    bridge = any(case["arm"] in BRIDGE_ARMS for case in config["cases"])
    expected = config["cases"]
    runs, curves, missing = [], [], []
    for case in expected:
        arm, seed = case["arm"], case["seed"]
        path = root / arm / ("seed_%d" % seed)
        metric = path / "metrics.json"
        archive = path / "archive_receipt.json"
        if not metric.exists() or not archive.exists():
            missing.append(case)
            continue
        value = json.loads(metric.read_text())
        if value.get("status") != "complete" or not all(value.get("checks", {}).values()):
            missing.append(case)
            continue
        if [x["round"] for x in value["evaluations"]] != list(range(-1, config["rounds"])):
            raise ValueError("evaluation trajectory mismatch")
        if arm in BRIDGE_ARMS:
            generator_manifest = json.loads((path / "generated/manifest.json").read_text())
            if arm == "zero_noise":
                if generator_manifest.get("is_record_level_dp") is not False or \
                   generator_manifest.get("privacy", {}).get("noise_added_to_clipped_gradient_sum") is not False:
                    raise ValueError("zero-noise control was mislabeled")
            else:
                validation = json.loads((path / "dp_validation.json").read_text())
                if validation.get("status") != "complete" or not all(validation.get("checks", {}).values()):
                    raise ValueError("DP release validation missing")
        runs.append({"arm": arm, "seed": seed, "final_accuracy": value["final"]["acc"],
                     "final_macro_f1": value["final"]["macro_f1"],
                     "final_loss": value["final"]["eval_loss"],
                     "metrics_sha256": common.sha256_file(metric),
                     "archive": json.loads(archive.read_text())})
        for row in value["evaluations"]:
            curves.append({"arm": arm, "seed": seed, "round": row["round"],
                           "accuracy": row["acc"], "macro_f1": row["macro_f1"],
                           "loss": row["eval_loss"]})
    aggregate = {}
    for arm in sorted({run["arm"] for run in runs}):
        selected = [r["final_accuracy"] for r in runs if r["arm"] == arm]
        aggregate[arm] = {"n": len(selected), "accuracy_mean": statistics.mean(selected),
                          "accuracy_sample_sd": statistics.stdev(selected) if len(selected) > 1 else None}
    summary = {"status": "complete" if not missing else "partial", "updated_at": now(),
               "runs_completed": len(runs), "runs_expected": len(expected),
               "protocol": config, "runs": runs, "missing": missing, "aggregate": aggregate,
               "privacy_scope": ("Only dp_eps8 synthetic generation is record-level DP; "
                                 "zero_noise and all downstream FL updates are Non-DP."
                                 if bridge else "All newly trained generators and FL updates are NON-DP."),
               "private_diagnostics_included": False}
    destination = root / "summary"
    destination.mkdir(exist_ok=True)
    common.atomic_write_json(destination / "summary.json", summary)
    buffer = io.StringIO()
    writer = csv.DictWriter(buffer, fieldnames=["arm", "seed", "round", "accuracy", "macro_f1", "loss"])
    writer.writeheader()
    writer.writerows(curves)
    temporary = destination / "curves.csv.tmp"
    temporary.write_text(buffer.getvalue())
    temporary.replace(destination / "curves.csv")
    lines = ["# AG News same-pipeline DP noise bridge" if bridge else "# AG News matched Non-DP ablations", "",
             "Status: %d/%d runs completed.\n" % (len(runs), len(expected)),
             summary["privacy_scope"] if bridge else
             "All new generator training is Non-DP. Historical DP comparisons are descriptive.",
             "Official test untouched; only the fixed 512-record dev set is evaluated.", "",
             "| Arm | n | Final accuracy (%) | Sample SD (pp) |", "|---|---:|---:|---:|"]
    for arm, value in aggregate.items():
        sd = value["accuracy_sample_sd"]
        lines.append("| %s | %d | %.4f | %s |" % (arm, value["n"],
                     100 * value["accuracy_mean"], "—" if sd is None else "%.4f" % (100 * sd)))
    lines += ["", "Generator arms use the historical guided estimator and learning rate 0.01."]
    if not bridge:
        lines += ["The formal no-guidance arm uses an explicitly labeled isotropic covariance correction",
                  "and a probe-selected learning rate; probe tuning queries are additional and reported."]
    lines += ["No significance or end-to-end DP claim is made. Private debug logs are excluded."]
    temporary = destination / "REPORT.md.tmp"
    temporary.write_text("\n".join(lines) + "\n")
    temporary.replace(destination / "REPORT.md")
    return summary


def task_command(args, root, path, seed, cloud_data, cloud_partition, alpha, estimator, lr):
    base = args.work_root
    entry = REPO / "experiments/distributed/transformer_exps/run_tc_exps"
    env = os.environ.copy()
    env.pop("CUDA_VISIBLE_DEVICES", None)
    env.update(PYTHONPATH="%s:%s:%s" % (entry, REPO, REPO / "FedML"),
               PYTHONHASHSEED=str(seed), HEIMDALLM_FAULTHANDLER="1", PYTHONNOUSERSITE="1",
               WANDB_MODE="disabled", WANDB_SILENT="true", TOKENIZERS_PARALLELISM="false",
               OMP_NUM_THREADS="1", MKL_NUM_THREADS="1", HF_HUB_OFFLINE="1",
               TRANSFORMERS_OFFLINE="1", OMPI_MCA_btl_vader_single_copy_mechanism="none")
    command = ["timeout", "--signal=TERM", "--kill-after=60s", "240m",
        "mpirun", "--bind-to", "none", "-np", "3", "--hostfile", root / "shared/mpi_host_file",
        base / "envs/kdd/bin/python", "-m", "fedavg_main_tc",
        "--gpu_mapping_file", root / "shared/gpu_mapping.yaml", "--gpu_mapping_key", "mapping_pilot",
        "--dataset", "agnews", "--data_file_path", base / "data/fednlp_data/data_files/agnews_data.h5",
        "--partition_file_path", root / "shared/fixed_partition.h5", "--partition_method", "pilot_uniform_2",
        "--cloud_dataset", "agnews", "--cloud_data_file_path", cloud_data,
        "--cloud_partition_file_path", cloud_partition, "--cloud_partition_method", "synthetic_cloud",
        "--cloud_client_ids", "0", "--cloud_max_seq_length", "64", "--fl_algorithm", "FedFwd",
        "--model_type", "distilbert", "--model_name", base / "models/distilbert-base-uncased",
        "--do_lower_case", "True", "--train_batch_size", "8", "--eval_batch_size", "8",
        "--max_seq_length", "64", "--client_num_in_total", "2", "--client_num_per_round", "2",
        "--worker_num", "1", "--comm_round", str(args.rounds), "--frequency_of_the_test", "1",
        "--evaluate_before_training", "--manual_seed", str(seed), "--run_id", str(seed), "--epochs", "1",
        "--lr", str(lr), "--server_lr", "0.1", "--learning_rate", str(lr), "--peft_method", "adapter",
        "--use_adapter", "True", "--forward_mode", "--var_control", "--perturbation_sampling",
        "--max_var_retries", "0", "--v_num", "1", "--beta", "1", "--pool_size", "1",
        "--alpha", str(alpha), "--zo_estimator", estimator, "--fd_step", "0.01",
        "--output_dir", path / "model_output"]
    if args.phase == "probes":
        command.append("--audit_updates")
    return command, env


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--work-root", type=Path, default=Path("/home/zzkevin/heimdallm-journal-20260928"))
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--phase", choices=["probes", "formal"], required=True)
    parser.add_argument("--rounds", type=int)
    parser.add_argument("--seeds", default="57,58,59")
    parser.add_argument("--selected-baseline-lr", type=float)
    parser.add_argument("--probe-summary", type=Path)
    parser.add_argument("--arms", default=",".join(ARMS))
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--pilot-only", action="store_true",
                        help="For the DP bridge, stop after both seed-57 arms pass validation")
    parser.add_argument("--summarize-only", action="store_true")
    args = parser.parse_args()
    if not args.run_id or any(c not in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789._-" for c in args.run_id):
        parser.error("invalid run ID")
    root = args.work_root / "results" / args.run_id
    if args.summarize_only:
        publish_summary(root)
        return
    args.rounds = args.rounds or (5 if args.phase == "probes" else 50)
    seeds = [int(s) for s in args.seeds.split(",")]
    arms = args.arms.split(",")
    if args.phase == "probes":
        seeds, arms = [57], list(PROBES)
    elif any(a not in ARMS + BRIDGE_ARMS for a in arms) or \
         ("no_guidance" in arms and args.selected_baseline_lr not in (0.01, 0.001, 0.0001)):
        parser.error("formal phase requires valid arms and a probe-selected no-guidance LR")
    if args.phase == "formal" and "no_guidance" in arms and not args.probe_summary:
        parser.error("no-guidance formal arm requires the completed probe summary")
    if any(a in BRIDGE_ARMS for a in arms) and any(a not in BRIDGE_ARMS for a in arms):
        parser.error("DP noise bridge must use only its two matched arms")
    if any(a in BRIDGE_ARMS for a in arms) and arms != list(BRIDGE_ARMS):
        parser.error("DP noise bridge requires zero_noise,dp_eps8 in that order")
    if args.pilot_only and (arms != list(BRIDGE_ARMS) or seeds[0] != 57):
        parser.error("--pilot-only requires the DP bridge starting at seed 57")
    base = args.work_root
    lock_path = base / "logs/matched-gpu1.lock"
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    lock = lock_path.open("a")
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    dirty = subprocess.check_output(["git", "status", "--porcelain"], cwd=REPO, text=True)
    if dirty.strip():
        raise RuntimeError("commit the reviewed code before running")
    revision = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=REPO, text=True).strip()
    historical = base / "results/ustc_agnews_dp_eps8_20260928"
    fixed = historical / "shared/agnews_smoke_dev_fixed_clients_partition.h5"
    reference = historical / "seed_57/synthetic"
    bridge = all(a in BRIDGE_ARMS for a in arms)
    cases = ([{"arm": arm, "seed": seed} for seed in seeds for arm in arms] if bridge
             else [{"arm": arm, "seed": seed} for arm in arms for seed in seeds])
    protocol = {"schema_version": 1, "phase": args.phase, "rounds": args.rounds,
        "seeds": seeds, "cases": cases,
        "code_revision": revision, "fixed_partition_sha256": common.sha256_file(fixed),
        "source_data_sha256": common.sha256_file(base / "data/fednlp_data/data_files/agnews_data.h5"),
        "source_partition_sha256": common.sha256_file(base / "data/fednlp_data/partition_files/agnews_partition.h5"),
        "gpu": 1, "logical_clients": [1, 21], "dev_records": 512, "official_test_used": False,
        "new_dp_training": bridge, "new_privacy_guarantee": (
            "record_level_synthetic_release_only_for_dp_eps8" if bridge else "none"),
        "generation_dtype": "float32", "generation_filter": "public_only_no_private_match",
        "downstream_guided_estimator": "legacy", "downstream_guided_learning_rate": 0.01,
        "formal_no_guidance_estimator": "isotropic_dimension_over_beta_squared",
        "selected_baseline_learning_rate": args.selected_baseline_lr,
        "baseline_selection": (None if bridge else
            "max probe final dev accuracy, then min final dev loss, then smallest LR"),
        "probe_summary_sha256": common.sha256_file(args.probe_summary) if args.probe_summary else None,
        "generator_modes": {"ordinary": "shuffled batches, token-mean objective",
            "fixed_example": "same shuffled batches, record-mean objective",
            "poisson": "Poisson sampling, record mean, expected batch denominator",
            "clipped": "same Poisson draws/dropout as poisson, per-record global clipping C=1",
            "public": "no LoRA or private generator access",
            "zero_noise": "same DP training loop and Poisson/clipping, Gaussian noise omitted; Non-DP",
            "dp_eps8": "same DP training loop with Gaussian noise; record-level epsilon<=8, delta=1e-5"},
        "objective_queries_per_run": args.rounds * 60,
        "probe_extra_objective_queries": 5 * 5 * 60 if args.phase == "formal" and not bridge else None}
    if root.exists():
        if not args.resume or json.loads((root / "protocol.json").read_text()) != protocol:
            raise ValueError("existing result directory or protocol mismatch")
    else:
        (root / "shared").mkdir(parents=True)
        common.atomic_write_json(root / "protocol.json", protocol)
        (root / "shared/fixed_partition.h5").write_bytes(fixed.read_bytes())
        hostname = socket.gethostname().split(".")[0]
        (root / "shared/gpu_mapping.yaml").write_text("mapping_pilot:\n  %s: [0, 3, 0, 0]\n" % hostname)
        (root / "shared/mpi_host_file").write_text("%s slots=3\n" % hostname)
    publish_summary(root)
    environment = os.environ.copy()
    environment.pop("CUDA_VISIBLE_DEVICES", None)
    environment.update(PYTHONNOUSERSITE="1", HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1",
                       TOKENIZERS_PARALLELISM="false", OMP_NUM_THREADS="2", MKL_NUM_THREADS="2")
    private = base / "private_staging" / args.run_id
    private.mkdir(parents=True, exist_ok=True, mode=0o700)
    for case in protocol["cases"]:
        arm, seed = case["arm"], case["seed"]
        if args.pilot_only and seed != 57:
            break
        path = root / arm / ("seed_%d" % seed)
        if (path / "archive_receipt.json").exists():
            print("Skipping archived %s seed %d" % (arm, seed), flush=True)
            continue
        if path.exists():
            raise ValueError("incomplete run retained; use a fresh run ID: %s" % path)
        path.mkdir(parents=True)
        common.atomic_write_json(root / "state.json", {"status": "running", "arm": arm,
                                                       "seed": seed, "updated_at": now()})
        try:
            if args.phase == "probes" or arm == "no_guidance":
                cloud_data = reference / "agnews_client_synthetic_data.h5"
                cloud_partition = reference / "agnews_client_synthetic_partition.h5"
                estimator, lr = PROBES[arm] if args.phase == "probes" else ("isotropic", args.selected_baseline_lr)
                alpha, condition = 1.0, "no_cloud"
            else:
                if arm in BRIDGE_ARMS:
                    generator = [base / "envs/ton/bin/python", SCRIPT / "dp_client_synthetic.py",
                        "--model-path", base / "models/distilgpt2", "--output-dir", path / "generated",
                        "--seed", str(seed), "--adapter-init-seed", "57", "--device", "cuda:1",
                        "--dtype", "float32", "--records-per-client", "120", "--target-per-label", "32",
                        "--label-names-json", json.dumps({"1": "World", "2": "Sports",
                            "3": "Business", "4": "Science and Technology"}),
                        "--prompt-template", "Category: {label}\\nNews article:\\n",
                        "--epochs", "5", "--batch-size", "4", "--learning-rate", "0.0005",
                        "--weight-decay", "0.01", "--max-grad-norm", "1", "--max-length", "192", "--lora-r", "8",
                        "--lora-alpha", "16", "--lora-dropout", "0.05",
                        "--generation-batch-size", "4", "--max-new-tokens", "80",
                        "--temperature", "0.8", "--top-p", "0.9", "--top-k", "0",
                        "--repetition-penalty", "1.05", "--min-free-mib", "3000",
                        "--memory-fraction", "0.9"]
                    generator += ["--diagnostic-zero-noise"] if arm == "zero_noise" else [
                        "--target-epsilon", "8", "--delta", "1e-5"]
                else:
                    generator = [base / "envs/ton/bin/python", SCRIPT / "matched_nondp_synthetic.py",
                        "--mode", arm, "--model-path", base / "models/distilgpt2",
                        "--output-dir", path / "generated", "--seed", str(seed), "--device", "cuda:1"]
                if arm != "public":
                    staging = private / ("seed_%d" % seed)
                    if not (staging / "manifest.json").exists():
                        execute([base / "envs/kdd/bin/python", SCRIPT / "export_client_train_jsonl.py",
                            "--data-file", base / "data/fednlp_data/data_files/agnews_data.h5",
                            "--partition-file", base / "data/fednlp_data/partition_files/agnews_partition.h5",
                            "--partition-method", "uniform_client_1000", "--client-ids", "1,21",
                            "--output-dir", staging, "--seed", str(seed),
                            "--sample-limit-per-client", "120", "--no-aggregate-output"],
                            private / ("export_seed_%d" % seed), "export", environment)
                    generator += ["--client-json-dir", staging]
                    if arm not in BRIDGE_ARMS:
                        generator += ["--private-diagnostics-dir",
                                      private / "diagnostics" / arm / ("seed_%d" % seed)]
                execute(generator, path, "generate", environment)
                if arm == "zero_noise":
                    validate_zero_noise_manifest(path / "generated/manifest.json")
                if arm == "dp_eps8":
                    execute([base / "envs/ton/bin/python", SCRIPT / "validate_dp_release.py",
                        "--manifest", path / "generated/manifest.json",
                        "--output", path / "dp_validation.json"], path, "dp_validate", environment)
                cloud_data, cloud_partition = path / "cloud_data.h5", path / "cloud_partition.h5"
                execute([base / "envs/kdd/bin/python", SCRIPT / "pack_synthetic_h5.py",
                    "--jsonl", path / "generated/synthetic.jsonl",
                    "--source-data-file", base / "data/fednlp_data/data_files/agnews_data.h5",
                    "--data-out", cloud_data, "--partition-out", cloud_partition,
                    "--partition-method", "synthetic_cloud", "--cloud-clients", "1",
                    "--manifest-out", path / "pack_manifest.json"], path, "pack", environment)
                alpha, estimator, lr = 0.5, "legacy", 0.01
                condition = "public_syn" if arm == "public" else "client_syn"
            command, task_env = task_command(args, root, path, seed, cloud_data, cloud_partition,
                                             alpha, estimator, lr)
            execute(command, path, "train", task_env, cwd=path)
            validate_command = [base / "envs/kdd/bin/python", SCRIPT / "summarize_fed_pilot.py",
                "--log", path / "train.log", "--metrics-out", path / "metrics.json",
                "--manifest-out", path / "manifest.json", "--command-file", path / "train.command.txt",
                "--condition", condition, "--cloud-source", (
                    "noise_bridge_" if bridge else "matched_nondp_") + arm,
                "--evaluation-mode", "dev", "--stage", "dp_noise_bridge" if bridge else "matched_ablation",
                "--alpha", str(alpha), "--seed", str(seed), "--rounds", str(args.rounds),
                "--eval-every", "1", "--pre-update-eval", "--logical-clients", "2",
                "--clients-per-round", "2", "--mpi-workers", "1", "--exit-code", "0",
                "--artifact", "protocol=" + str(root / "protocol.json"),
                "--artifact", "fixed_partition=" + str(root / "shared/fixed_partition.h5"),
                "--require-complete"]
            if bridge:
                validate_command += ["--artifact", "generator_manifest=" + str(path / "generated/manifest.json")]
            if arm == "dp_eps8":
                validate_command += ["--artifact", "dp_validation=" + str(path / "dp_validation.json")]
            execute(validate_command, path, "validate", environment)
            archives = base / "backups"
            archives.mkdir(exist_ok=True)
            archive = archives / ("%s_%s_seed%d.tar.gz" % (args.run_id, arm, seed))
            temporary = archive.with_name(archive.name + ".tmp")
            with tarfile.open(temporary, "w:gz") as handle:
                handle.add(path, arcname=str(path.relative_to(root)))
                handle.add(root / "shared", arcname="shared")
                handle.add(root / "protocol.json", arcname="protocol.json")
            temporary.replace(archive)
            checksum = common.sha256_file(archive)
            archive.with_name(archive.name + ".sha256").write_text(checksum + "  " + archive.name + "\n")
            common.atomic_write_json(path / "archive_receipt.json", {"name": archive.name, "sha256": checksum})
            result = publish_summary(root)
            print("Completed %s seed=%d; progress=%d/%d" %
                  (arm, seed, result["runs_completed"], result["runs_expected"]), flush=True)
        except Exception:
            common.atomic_write_json(root / "state.json", {"status": "failed", "arm": arm,
                                                           "seed": seed, "updated_at": now()})
            raise
    final = publish_summary(root)
    if final["status"] != "complete" and not args.pilot_only:
        raise RuntimeError("queue ended with missing formal cases")
    common.atomic_write_json(root / "state.json", {
        "status": "complete" if final["status"] == "complete" else "pilot_complete",
        "updated_at": now()})


if __name__ == "__main__":
    main()
