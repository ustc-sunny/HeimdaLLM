#!/usr/bin/env python3
"""Guarded seed-57 Qwen public-NLI control against frozen AG News references."""

import argparse
import csv
import json
import os
import shlex
import shutil
import tarfile
from pathlib import Path

from run_ustc_augpe_nondp_pilot import check_gpu, digest, now, replace, run, write_json
from run_ustc_qwen_public_control import option, reference_metrics


BASE = Path("/home/zzkevin/heimdallm-journal-20260928")
RUN_ID = "ustc_agnews_qwen_nli_control_20261014"
PUBLIC_ID = "ustc_agnews_qwen_public_control_20261013"
PRIVATE_ID = "ustc_agnews_private_vote_pilot_20261009"
CANDIDATE_SHA = "7e54c05ce2dc94d754b178218963dedfceb781c491b97ed1d9db5ab9e7d7c54f"
ARMS = ("private_vote_synthetic", "public_random_selection",
        "public_nli_selection", "same_source_real")


def summarize(root, protocol):
    reference = {"private_vote_synthetic": BASE / "results" / PRIVATE_ID,
                 "public_random_selection": BASE / "results" / PUBLIC_ID,
                 "public_nli_selection": root,
                 "same_source_real": BASE / "results" / PRIVATE_ID}
    rows, curves = [], []
    for arm in ARMS:
        path = reference[arm] / arm / "metrics.json"
        value = reference_metrics(path)
        rows.append({"arm": arm, "seed": 57, "final_accuracy": value["final"]["acc"],
                     "final_macro_f1": value["final"]["macro_f1"],
                     "late_mean_accuracy": sum(point["acc"] for point in value["evaluations"]
                                               if point["round"] >= 40) / 10,
                     "metrics_sha256": digest(path)})
        for point in value["evaluations"]:
            curves.append({"arm": arm, "round": point["round"],
                           "accuracy": point["acc"], "macro_f1": point["macro_f1"],
                           "loss": point["eval_loss"]})
    summary = root / "summary"
    summary.mkdir(exist_ok=True)
    write_json(summary / "summary.json", {
        "schema_version": 1, "status": "complete", "updated_at": now(),
        "runs_completed": 4, "runs_expected": 4, "runs": rows,
        "protocol": protocol, "official_test_used": False,
        "privacy": "Non-DP", "private_text_included": False})
    with (summary / "curves.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=[
            "arm", "round", "accuracy", "macro_f1", "loss"])
        writer.writeheader()
        writer.writerows(curves)
    lines = ["# AG News Qwen public-NLI selection control", "",
             "Non-DP seed-57 comparison on the fixed 512-record development set.",
             "All arms use 50 FL rounds and 64 guidance records. The three reference",
             "trajectories are frozen historical runs; only public NLI selection is new.", "",
             "| Guidance | Final accuracy | Last 10-round mean |",
             "|---|---:|---:|"]
    for row in rows:
        lines.append("| %s | %.2f%% | %.2f%% |" % (
            row["arm"], 100 * row["final_accuracy"],
            100 * row["late_mean_accuracy"]))
    lines.extend(["", "One candidate pool and one FL seed; no DP or official test."])
    (summary / "REPORT.md").write_text("\n".join(lines) + "\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gpu", type=int, default=1, choices=range(4))
    args = parser.parse_args()
    repo = BASE / "HeimdaLLM"
    scripts = repo / "journal_extension"
    root = BASE / "results" / RUN_ID
    private = BASE / "private_staging" / RUN_ID
    if root.exists():
        raise FileExistsError("result directory already exists; preserve earlier output")
    selection = private / "selection"
    manifest = json.loads((selection / "manifest.json").read_text())
    if (manifest["status"] != "complete" or manifest["candidate_sha256"] != CANDIDATE_SHA
            or manifest["selected_records"] != 64 or manifest["private_records_read"]
            or digest(selection / "synthetic.jsonl") != manifest["selected_sha256"]):
        raise ValueError("public NLI selection failed fixed-pool checks")
    old = BASE / "results" / PUBLIC_ID
    old_command = shlex.split((old / "public_random_selection/train.command.txt").read_text())
    for flag, expected in (("--manual_seed", "57"), ("--comm_round", "50"),
                           ("--client_num_in_total", "2"),
                           ("--client_num_per_round", "2")):
        if option(old_command, flag) != expected:
            raise ValueError("historical downstream protocol changed: " + flag)
    preflight = check_gpu(args.gpu)

    root.mkdir(parents=True)
    shared = root / "shared"
    shared.mkdir()
    for filename in ("fixed_partition.h5", "mpi_host_file"):
        shutil.copy2(old / "shared" / filename, shared / filename)
    mapping = [0] * 4
    mapping[args.gpu] = 3
    (shared / "gpu_mapping.yaml").write_text(
        "mapping_private_vote:\n  amax: [%s]\n" % ", ".join(map(str, mapping)))
    protocol = {"schema_version": 1, "run_id": RUN_ID, "seed": 57,
                "physical_gpu": args.gpu, "rounds": 50, "dev_records": 512,
                "guidance_records": 64, "source_clients": [1, 21],
                "candidate_sha256": CANDIDATE_SHA,
                "selection": manifest["selection"],
                "selection_sha256": manifest["selected_sha256"],
                "reference_run_ids": [PRIVATE_ID, PUBLIC_ID],
                "fixed_partition_sha256": digest(shared / "fixed_partition.h5"),
                "runner_sha256": digest(Path(__file__)),
                "official_test_used": False, "privacy": "Non-DP",
                "gpu_preflight": preflight}
    write_json(root / "protocol.json", protocol)
    write_json(root / "state.json", {"status": "running", "stage": "audit",
                                      "updated_at": now()})
    env = os.environ.copy()
    env.pop("CUDA_VISIBLE_DEVICES", None)
    env.update(PYTHONNOUSERSITE="1", HF_HUB_OFFLINE="1",
               TRANSFORMERS_OFFLINE="1", TOKENIZERS_PARALLELISM="false",
               OMP_NUM_THREADS="1", MKL_NUM_THREADS="1")
    try:
        run([BASE / "envs/ton/bin/python", scripts / "audit_private_vote_pilot.py",
             "--jsonl", selection / "synthetic.jsonl", "--output", root / "quality.json"],
            private, "audit", env)
        path = root / "public_nli_selection"
        path.mkdir()
        write_json(root / "state.json", {"status": "running", "stage": "pack",
                                          "updated_at": now()})
        run([BASE / "envs/kdd/bin/python", scripts / "pack_synthetic_h5.py",
             "--jsonl", selection / "synthetic.jsonl",
             "--source-data-file", BASE / "data/fednlp_data/data_files/agnews_data.h5",
             "--data-out", path / "cloud_data.h5",
             "--partition-out", path / "cloud_partition.h5",
             "--partition-method", "synthetic_cloud", "--cloud-clients", "1",
             "--require-equal-label-counts", "--manifest-out", path / "pack_manifest.json"],
            path, "pack", env)
        write_json(path / "gpu_preflight.json", check_gpu(args.gpu))
        command = old_command[:]
        for flag, value in (("--gpu_mapping_file", shared / "gpu_mapping.yaml"),
                            ("--hostfile", shared / "mpi_host_file"),
                            ("--partition_file_path", shared / "fixed_partition.h5"),
                            ("--cloud_data_file_path", path / "cloud_data.h5"),
                            ("--cloud_partition_file_path", path / "cloud_partition.h5"),
                            ("--output_dir", path / "model_output")):
            replace(command, flag, value)
        entry = repo / "experiments/distributed/transformer_exps/run_tc_exps"
        task_env = env.copy()
        task_env.update(PYTHONPATH="%s:%s:%s" % (entry, repo, repo / "FedML"),
                        PYTHONHASHSEED="57", HEIMDALLM_FAULTHANDLER="1",
                        WANDB_MODE="disabled", WANDB_SILENT="true",
                        OMPI_MCA_btl_vader_single_copy_mechanism="none")
        write_json(root / "state.json", {"status": "running", "stage": "train",
                                          "updated_at": now()})
        run(command, path, "train", task_env)
        run([BASE / "envs/kdd/bin/python", scripts / "summarize_fed_pilot.py",
             "--log", path / "train.log", "--metrics-out", path / "metrics.json",
             "--manifest-out", path / "manifest.json",
             "--command-file", path / "train.command.txt",
             "--condition", "public_syn", "--cloud-source", "public_nli_selection",
             "--evaluation-mode", "dev", "--stage", "qwen_public_nli_control",
             "--alpha", "0.5", "--seed", "57", "--rounds", "50",
             "--eval-every", "1", "--pre-update-eval", "--logical-clients", "2",
             "--clients-per-round", "2", "--mpi-workers", "1", "--exit-code", "0",
             "--artifact", "protocol=" + str(root / "protocol.json"),
             "--artifact", "fixed_partition=" + str(shared / "fixed_partition.h5"),
             "--require-complete"], path, "validate", env)
        summarize(root, protocol)
        archive = BASE / "backups" / (RUN_ID + ".tar.gz")
        with tarfile.open(str(archive) + ".tmp", "w:gz") as handle:
            handle.add(path, arcname="public_nli_selection")
            handle.add(shared, arcname="shared")
            handle.add(selection, arcname="selection")
            handle.add(root / "protocol.json", arcname="protocol.json")
            handle.add(root / "quality.json", arcname="quality.json")
        Path(str(archive) + ".tmp").replace(archive)
        Path(str(archive) + ".sha256").write_text(digest(archive) + "  " + archive.name + "\n")
        write_json(root / "state.json", {"status": "complete", "updated_at": now()})
        print("NLI control complete; archive SHA-256 " + digest(archive), flush=True)
    except Exception:
        write_json(root / "state.json", {"status": "failed", "updated_at": now()})
        raise


if __name__ == "__main__":
    main()
