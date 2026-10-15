#!/usr/bin/env python3
"""Read-only stage-1 archive audit. No server connection, training, or private text export."""
import csv
import hashlib
import importlib.util
import io
import json
import math
import os
import shlex
import subprocess
import sys
import tarfile
from collections import Counter
from pathlib import Path

HERE = Path(__file__).resolve().parent
def locate_original_workspace():
    configured = os.environ.get("HEIMDALLM_STAGE1_BACKUP_ROOT")
    candidates = [Path(configured).expanduser()] if configured else []
    candidates.extend((HERE, *HERE.parents))
    for candidate in candidates:
        if (candidate / "HeimdaLLM-github").is_dir() and (candidate / "local_backups").is_dir():
            return candidate
    raise RuntimeError("Original local_backups are required to replay this historical audit; see LOCAL_RAW_ARCHIVE_INDEX.json")

ROOT = locate_original_workspace()
REPO = ROOT / "HeimdaLLM-github"
POOL_SHA = "7e54c05ce2dc94d754b178218963dedfceb781c491b97ed1d9db5ab9e7d7c54f"
ARCHIVES = {
    "PRIVATE_ZERO": ROOT / "local_backups/agnews_private_vote_pilot_20261009/ustc_agnews_private_vote_pilot_20261009_private_vote_synthetic.tar.gz",
    "REAL_MATCHED": ROOT / "local_backups/agnews_private_vote_pilot_20261009/ustc_agnews_private_vote_pilot_20261009_same_source_real.tar.gz",
    "PUB_RANDOM": ROOT / "local_backups/agnews_qwen_public_control_20261013/ustc_agnews_qwen_public_control_20261013.tar.gz",
}
ARMS = ("PUB_RANDOM", "PUB_NLI", "PRIVATE_ZERO", "REAL_MATCHED", "PRIVATE_DP_E4")
SEEDS = (57, 58, 59)


def sha(data):
    return hashlib.sha256(data).hexdigest()


def read_json(tf, suffix):
    names = [n for n in tf.getnames() if n.endswith(suffix)]
    if suffix == "/manifest.json" and len(names) > 1:
        names = [n for n in names if n.split("/")[0] in ("private_vote_synthetic", "same_source_real", "public_random_selection")]
    if len(names) != 1:
        raise AssertionError((suffix, names))
    return json.load(tf.extractfile(names[0]))


def write_json(name, obj):
    (HERE / name).write_text(json.dumps(obj, indent=2, ensure_ascii=False, sort_keys=True) + "\n")


def write_csv(name, fields, rows):
    with (HERE / name).open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def module_from(path):
    spec = importlib.util.spec_from_file_location("public_selector", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def main():
    if not HERE.exists():
        raise AssertionError("output root missing")
    head = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=REPO, text=True).strip()
    tree_before = subprocess.check_output(["git", "status", "--porcelain=v1"], cwd=REPO, text=True)
    records = {}
    curves = []
    classes = []
    costs = []
    pool_bytes = None
    random_selected = None
    random_manifest = None
    command_flags = {}
    fixed_partition_hashes = {}
    for arm, path in ARCHIVES.items():
        raw = path.read_bytes()
        digest = sha(raw)
        expected = (path.parent / (path.name + ".sha256")).read_text().split()[0]
        assert digest == expected, path
        with tarfile.open(fileobj=io.BytesIO(raw), mode="r:gz") as tf:
            metrics = read_json(tf, "/metrics.json")
            manifest = read_json(tf, "/manifest.json")
            pack = read_json(tf, "/pack_manifest.json")
            protocol = read_json(tf, "protocol.json")
            command_name = [n for n in tf.getnames() if n.endswith("/train.command.txt")][0]
            command = shlex.split(tf.extractfile(command_name).read().decode())
            flag_names = ("--dataset", "--partition_method", "--cloud_dataset", "--cloud_partition_method",
                "--cloud_client_ids", "--cloud_max_seq_length", "--fl_algorithm", "--model_type",
                "--do_lower_case", "--train_batch_size", "--eval_batch_size", "--max_seq_length",
                "--client_num_in_total", "--client_num_per_round", "--worker_num", "--comm_round",
                "--frequency_of_the_test", "--manual_seed", "--run_id", "--epochs", "--lr",
                "--server_lr", "--learning_rate", "--peft_method", "--use_adapter", "--max_var_retries",
                "--v_num", "--beta", "--pool_size", "--alpha", "--zo_estimator", "--fd_step")
            command_flags[arm] = {flag: command[command.index(flag)+1] for flag in flag_names}
            for flag in ("--evaluate_before_training", "--forward_mode", "--var_control", "--perturbation_sampling"):
                assert flag in command
            fixed_partition_hashes[arm] = sha(tf.extractfile("shared/fixed_partition.h5").read())
            evs = metrics["evaluations"]
            assert len(evs) == 51 and [e["round"] for e in evs] == list(range(-1, 50))
            assert metrics["status"] == manifest["status"] == "complete"
            assert all(metrics["checks"].values()) and manifest["exit_code"] == 0
            assert pack["records"] == 64 and pack["duplicates_dropped"] == 0
            assert pack["test_records"] == 0 and pack["all_train_indices_unique"]
            final = evs[-1]
            assert final["acc"] == metrics["final"]["acc"]
            assert sum(map(sum, final["confusion_matrix"])) == 512
            for e in evs:
                curves.append({"arm": arm, "fl_seed": 57, "updates_completed": e["round"] + 1,
                               "dev_accuracy_percent": 100 * e["acc"], "macro_f1": e["macro_f1"],
                               "loss": e["eval_loss"], "status": "historical_verified_locally"})
            cm = final["confusion_matrix"]
            for i in range(4):
                tp = cm[i][i]
                support = sum(cm[i]); predicted = sum(row[i] for row in cm)
                precision = tp / predicted if predicted else 0
                recall = tp / support if support else 0
                f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0
                classes.append({"arm": arm, "fl_seed": 57, "class_index": i,
                                "support": support, "precision": precision, "recall": recall, "f1": f1,
                                "confusion_row": json.dumps(cm[i], separators=(",", ":"))})
            late = sum(e["acc"] for e in evs[-10:]) / 10
            auc = (evs[0]["acc"] / 2 + sum(e["acc"] for e in evs[1:-1]) + evs[-1]["acc"] / 2) / 50
            thresholds = {}
            for t in (.5, .6, .7):
                reached = next((i for i, e in enumerate(evs) if e["acc"] >= t), None)
                thresholds[str(int(100*t))] = {"first_update": reached, "right_censored": reached is None,
                                                "queries_at_first_update": None if reached is None else reached * 60}
            records[arm] = {"archive": str(path.relative_to(ROOT)), "archive_sha256": digest,
                            "metrics_sha256": sha(tf.extractfile([n for n in tf.getnames() if n.endswith("/metrics.json")][0]).read()),
                            "final_accuracy_percent": 100 * final["acc"], "final_macro_f1": final["macro_f1"],
                            "final_loss": final["eval_loss"], "late_window_accuracy_percent": 100 * late,
                            "normalized_accuracy_auc": auc, "thresholds": thresholds,
                            "confusion_matrix": cm, "protocol": protocol, "pack": pack,
                            "command_sha256": manifest["command_sha256"],
                            "checks": metrics["checks"], "seed": metrics["seed"]}
            costs.append({"component": "cloud_data_pack", "arm": arm, "fl_seed": 57,
                          "logical_location": "server_staging", "measured_device": "unverified_historical_host",
                          "elapsed_seconds": pack["elapsed_seconds"], "peak_memory_mib": "NA",
                          "bytes_sent": "NA", "measurement_status": "historical_pack_only"})
            if arm == "PUB_RANDOM":
                pool_bytes = tf.extractfile("public_pool/candidates.jsonl").read()
                random_selected = tf.extractfile("public_selection/synthetic.jsonl").read()
                random_manifest = read_json(tf, "public_selection/manifest.json")
    assert sha(pool_bytes) == POOL_SHA
    assert len({json.dumps(v, sort_keys=True) for v in command_flags.values()}) == 1
    assert len(set(fixed_partition_hashes.values())) == 1
    pool = [json.loads(line) for line in pool_bytes.splitlines()]
    assert len(pool) == 128 and Counter(str(x["label"]) for x in pool) == {str(i): 32 for i in range(1, 5)}
    assert len({" ".join(x["text"].lower().split()) for x in pool}) == 128
    assert sha(random_selected) == random_manifest["selected_sha256"]
    nli_dir = ROOT / "local_backups/agnews_qwen_nli_control_20261014/selection"
    nli_manifest = json.loads((nli_dir / "manifest.json").read_text())
    nli_quality = json.loads((nli_dir / "quality.json").read_text())
    nli_selected = (nli_dir / "synthetic.jsonl").read_bytes()
    assert nli_manifest["candidate_sha256"] == POOL_SHA
    assert sha(nli_selected) == nli_manifest["selected_sha256"]
    nli_rows = [json.loads(line) for line in nli_selected.splitlines()]
    assert len(nli_rows) == 64 and len({x["text"] for x in nli_rows}) == 64
    expected_nli = []
    for label in ("1", "2", "3", "4"):
        ranks = nli_manifest["rank_summary"][label]["selected_indices"]
        for slot, cid in enumerate((1, 21)):
            expected_nli.extend({"client_id": cid, "label": label, "text": pool[i]["text"]} for i in ranks[slot:16:2])
    assert nli_rows == expected_nli
    assert nli_quality["records"] == 64 and nli_quality["label_agreement"] == 59 / 64
    import tempfile
    with tempfile.TemporaryDirectory() as tmp:
        public_path = Path(tmp) / "candidates.jsonl"
        public_path.write_bytes(pool_bytes)
        selector = module_from(REPO / "journal_extension/select_public_qwen_candidates.py")
        selected = selector.select(public_path, 57)
        replay = b"".join((json.dumps(r, ensure_ascii=False) + "\n").encode() for r in selected)
        assert sha(replay) == sha(random_selected)
    for arm, r in records.items():
        assert r["seed"] == 57
        assert r["protocol"]["rounds"] == 50 and r["protocol"]["dev_records"] == 512
    p = records["PRIVATE_ZERO"]; real = records["REAL_MATCHED"]; pub = records["PUB_RANDOM"]
    assert p["protocol"]["source_data_sha256"] == real["protocol"]["source_data_sha256"]
    assert p["protocol"]["source_partition_sha256"] == real["protocol"]["source_partition_sha256"]
    assert pub["protocol"]["reference_private_vote_metrics_sha256"] == p["metrics_sha256"]
    assert pub["protocol"]["reference_real_metrics_sha256"] == real["metrics_sha256"]
    fixed = {"protocol_id": "agnews_qwen_pool128_guidance64_fedfwd_legacy50_c1b238c_provisional",
             "source_commit": head, "working_tree_clean_at_audit_start": not bool(tree_before),
             "candidate_pool_sha256": POOL_SHA, "candidate_pool_rows": 128,
             "public_random_selection_sha256": random_manifest["selected_sha256"],
             "public_nli_selection_sha256": nli_manifest["selected_sha256"],
             "clients": [1, 21], "records_per_client_reported": 120,
             "guidance_slots": 64, "per_client_per_class_slots": 8, "unique_texts_historical": 64,
             "fl_seeds": list(SEEDS), "updates": 50, "evaluations": 51,
             "dev_records": 512, "official_test_allowed": False,
             "downstream": {"model": "DistilBERT adapter", "algorithm": "FedFwd",
                            "alpha": .5, "learning_rate": .01, "fd_step": .01,
                            "zo_estimator": "legacy", "pool_size": 1, "beta": 1,
                            "trainable_dimension_reported": 450340,
                            "client_fd_directions_per_update_reported": 30,
                            "client_objective_queries_per_update_reported": 60},
             "status": "provisional_historical_protocol; live source and one-step audit blocked"}
    write_json("FROZEN_PROTOCOL.json", fixed)
    run_rows = []
    metric_rows = []
    for arm in ARMS:
        for seed in SEEDS:
            r = records.get(arm) if seed == 57 else None
            status = "historical_provisional_reuse" if r else "blocked_not_started"
            run_id = f"{fixed['protocol_id']}__{arm}__s{seed}"
            row = {"run_id": run_id, "arm": arm, "fl_seed": seed,
                   "status": status, "completed_updates": 50 if r else "NA",
                   "final_dev_accuracy_percent": r["final_accuracy_percent"] if r else "NA",
                   "final_macro_f1": r["final_macro_f1"] if r else "NA",
                   "final_loss": r["final_loss"] if r else "NA",
                   "late_window_accuracy_percent": r["late_window_accuracy_percent"] if r else "NA",
                   "normalized_accuracy_auc": r["normalized_accuracy_auc"] if r else "NA",
                   "client_objective_queries_reported": 3000 if r else "NA",
                   "cloud_bp_sample_visits_reported": 3200 if r else "NA",
                   "guidance_slots": 64 if r else "NA", "unique_texts": 64 if r else "NA",
                   "source_archive": r["archive"] if r else "NA"}
            for threshold in (50,60,70):
                t = r["thresholds"][str(threshold)] if r else None
                row[f"first_update_acc_{threshold}"] = "NA" if not t or t["right_censored"] else t["first_update"]
                row[f"acc_{threshold}_right_censored"] = "NA" if not t else t["right_censored"]
            metric_rows.append(row)
            run_rows.append({"run_id": run_id, "arm": arm, "fl_seed": seed, "status": status,
                             "reused_from": r["archive"] if r else None,
                             "archive_sha256": r["archive_sha256"] if r else None,
                             "privacy_status": "non_dp" if r else "not_implemented" if arm == "PRIVATE_DP_E4" else "not_applicable_or_pending",
                             "failure_or_block_reason": None if r else "SSH closes after password; GPU and live source unverified"})
    write_csv("metrics.csv", list(metric_rows[0]), metric_rows)
    write_csv("curves.csv", list(curves[0]), curves)
    write_csv("per_class_metrics.csv", list(classes[0]), classes)
    pairs = []
    for other in ("PUB_RANDOM", "REAL_MATCHED", "PUB_NLI", "PRIVATE_DP_E4"):
        for seed in SEEDS:
            available = seed == 57 and other in records
            pairs.append({"fl_seed": seed, "reference_arm": "PRIVATE_ZERO", "comparison_arm": other,
                          "status": "historical_provisional_pair" if available else "blocked_not_started",
                          "reference_accuracy_percent": p["final_accuracy_percent"] if available else "NA",
                          "comparison_accuracy_percent": records[other]["final_accuracy_percent"] if available else "NA",
                          "reference_minus_comparison_pp": p["final_accuracy_percent"] - records[other]["final_accuracy_percent"] if available else "NA"})
    write_csv("paired_comparisons.csv", list(pairs[0]), pairs)
    for component, location in (("generate_public_pool_128", "server_public"),
       ("load_cache_pool", "client_or_server_unverified"),
       ("candidate_downlink", "server_to_client"),
       ("client_encode_vote", "client"), ("client_noise_topk", "client"),
       ("selected_id_uplink", "client_to_server"), ("cloud_bp", "cloud_server")):
        costs.append({"component":component,"arm":"NA","fl_seed":"NA",
          "logical_location":location,"measured_device":"NA","elapsed_seconds":"NA",
          "peak_memory_mib":"NA","bytes_sent":"NA","measurement_status":"not_measured"})
    write_csv("COMPONENT_COSTS.csv", list(costs[0]), costs)
    reuse = {"source_commit_local": head, "status": "provisional_pending_live_source_and_operator_check",
             "historical_runs": {a: {k: v for k,v in r.items() if k not in ("protocol", "pack", "checks", "confusion_matrix")}
                                 for a,r in records.items()},
             "validation": {"archive_sidecars": True, "all_51_evaluations_each": True,
                            "completion_flags": True, "public_random_replayed_byte_identically": True,
                            "public_nli_rank_assignment_reconstructed": True,
                            "archived_fixed_partition_sha256": next(iter(fixed_partition_hashes.values())),
                            "archived_training_flags_equal": True,
                            "source_data_and_partition_rehashed_locally": False,
                            "live_server_commit_verified": False,
                            "actual_update_operator_verified": False}}
    write_json("REUSE_MANIFEST.json", reuse)
    write_json("RUN_MANIFEST.json", {"protocol_id": fixed["protocol_id"], "runs": run_rows,
                                     "new_runs_completed": 0, "historical_provisional": 3,
                                     "blocked_not_started": 12})
    write_json("cross_arm_validation.json", {"status": "partial",
        "historical_seed57_reuse_prechecks_passed": True,
        "archived_training_flags_equal": True,
        "archived_fixed_partition_sha256": next(iter(fixed_partition_hashes.values())),
        "public_pool_sha256": POOL_SHA,
        "random_replay_sha256": sha(random_selected), "nli_selection_sha256": sha(nli_selected),
        "all_15_protocol_matched": False,
        "live_source_and_single_step_verified": False,
        "note": "No new training or GPU preflight; historical reuse remains provisional."})
    write_json("PRIVACY_LEDGER_REDACTED.json", {"scope": "synthetic_guidance_release_only",
        "release_count": 0, "completed_dp_trials": [], "actual_joint_epsilon": None,
        "planned_epsilon_per_release": 4, "planned_laplace_scale": .5,
        "implementation_review_status": "blocked", "raw_votes_or_noise_secrets_in_output": False})
    write_json("summary.json", {"target_endpoints": 15, "historical_provisional": 3,
        "new_completed": 0, "blocked_not_started": 12, "dp_completed": 0,
        "seed57_final_accuracy_percent": {a:r["final_accuracy_percent"] for a,r in records.items()},
        "seed57_private_minus_public_random_pp": p["final_accuracy_percent"]-pub["final_accuracy_percent"],
        "seed57_private_minus_real_pp": p["final_accuracy_percent"]-real["final_accuracy_percent"],
        "nli_auxiliary_label_agreement_percent": 100*nli_quality["label_agreement"],
        "nli_downstream_accuracy_percent": None})
    handoff = HERE / "paper_handoff"; handoff.mkdir(exist_ok=True)
    template = json.loads((ROOT / "tasks/06_PAPER_EVIDENCE_MANIFEST_TEMPLATE.json").read_text())
    template["kind"] = "PARTIAL_STAGE1_EVIDENCE"
    template["last_updated"] = "2026-10-14"
    template["protocol"].update({"protocol_id": fixed["protocol_id"], "source_commit": head,
        "working_tree_diff_hash": sha(tree_before.encode()), "protocol_spec_path": "../FROZEN_PROTOCOL.json",
        "pool_id": "qwen_public_pool_128_20261013", "pool_sha256": POOL_SHA,
        "source_group": "AGNews_clients_1_21", "cross_arm_validation_path": "../cross_arm_validation.json"})
    template["privacy"].update({"implementation_review_status": "blocked_no_dp_implementation_or_live_review",
        "redacted_ledger_path": "../PRIVACY_LEDGER_REDACTED.json"})
    template["runs"] = run_rows
    template["claims"] = [{"claim_id":"historical_seed57_descriptive",
       "english_text":"On one fixed AG News development split and FL seed 57, private-vote guidance reached 75.00% accuracy, versus 70.31% for public random and 72.07% for matched real guidance.",
       "status":"historical_provisional", "evidence_kind":"local_archive_recalculation",
       "source_paths":[records[a]["archive"] for a in ("PRIVATE_ZERO","PUB_RANDOM","REAL_MATCHED")],
       "source_fields":["metrics.json/evaluations[-1]/acc"],
       "protocol_ids":[fixed["protocol_id"]], "comparison_unit":"one_FL_seed",
       "limitations":["No seed replication", "Live server code and one-step audit pending", "Not a DP result"],
       "verified_by_recomputation": True, "eligible_for_paper": False,
       "recommended_manuscript_location":"experiments after full validation"}]
    template["system_measurements"].update({"component_costs_path":"../COMPONENT_COSTS.csv",
        "real_edge_device_tested":False,"controlled_timing_available":False,"speedup_claim_permitted":False})
    template["theory"].update({"implementation_map_path":"THEORY_IMPLEMENTATION_MAP.md",
        "covariance_model_matches_code":False, "privacy_proof_path":"../PRIVATE_GUIDANCE_RELEASE_SPEC.md",
        "mathematical_assumptions_verified":False,
        "conditions_or_open_items":["Sequential client disjoint selection", "Noise implementation absent", "Live update test absent"]})
    template["publication_and_handoff"].update({"results_for_paper_path":"RESULTS_FOR_PAPER.md",
        "tables_for_paper_path":"TABLES_FOR_PAPER.csv", "all_numbers_have_provenance":True,
        "contains_unprotected_private_artifacts":False,"author_review_required_before_publication":True})
    (handoff / "PAPER_EVIDENCE_MANIFEST.json").write_text(json.dumps(template,indent=2,sort_keys=True)+"\n")
    write_csv("paper_handoff/TABLES_FOR_PAPER.csv", list(metric_rows[0]), metric_rows)
    print(json.dumps({"status":"partial_blocked", "historical":3,"new":0,"blocked":12,
                      "public_random_replay":True,"nli_reconstruction":True}))


if __name__ == "__main__":
    main()
