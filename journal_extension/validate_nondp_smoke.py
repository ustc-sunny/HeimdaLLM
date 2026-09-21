#!/usr/bin/env python3
"""Validate a completed paired true-NonDP SST-2 feasibility run.

The arm summarizer intentionally remains a small, per-arm log parser.  This
validator adds the checks that require the immutable run specification or a
comparison across its declared arms, especially the fixed central-FD query budget.
"""

from __future__ import print_function

import argparse
import ast
import hashlib
import json
import math
from pathlib import Path
import re
import shlex
import sys


ARM_SPECS = {
    "no_cloud": {
        "alpha": 1.0,
        "cloud_source": "client_synthetic_nondp",
        "partition_method": "synthetic_cloud",
        "data_relative": "synthetic/sst_2_client_synthetic_data.h5",
        "partition_relative": "synthetic/sst_2_client_synthetic_partition.h5",
    },
    "client_syn": {
        "alpha": 0.5,
        "cloud_source": "client_synthetic_nondp",
        "partition_method": "synthetic_cloud",
        "data_relative": "synthetic/sst_2_client_synthetic_data.h5",
        "partition_relative": "synthetic/sst_2_client_synthetic_partition.h5",
    },
    "public_syn": {
        "alpha": 0.5,
        "cloud_source": "public_pretrained_synthetic",
        "partition_method": "public_synthetic_cloud",
        "data_relative": "public_synthetic/sst_2_public_synthetic_data.h5",
        "partition_relative": "public_synthetic/sst_2_public_synthetic_partition.h5",
    },
    "same_source_real_matched": {
        "alpha": 0.5,
        "cloud_source": "nondeployable_same_source_client_real_matched",
        "partition_method": "same_source_real_cloud",
        "data_relative": "controls/same_source_real_matched/sst_2_same_source_real_matched_data.h5",
        "partition_relative": "controls/same_source_real_matched/sst_2_same_source_real_matched_partition.h5",
    },
    "real_matched": {
        "alpha": 0.5,
        "cloud_source": "reserved_client_real_matched",
        "partition_method": "real_matched_cloud",
        "data_relative": "controls/real_matched/sst_2_real_matched_data.h5",
        "partition_relative": "controls/real_matched/sst_2_real_matched_partition.h5",
    },
    "shuffled_label": {
        "alpha": 0.5,
        "cloud_source": "client_synthetic_shuffled_labels",
        "partition_method": "shuffled_label_cloud",
        "data_relative": "controls/shuffled_label/sst_2_shuffled_label_data.h5",
        "partition_relative": "controls/shuffled_label/sst_2_shuffled_label_partition.h5",
    },
}

EVAL_PATTERN = re.compile(r"eval_model\(\):\s*(\{.*\})\s*$")
TRAIN_PATTERN = re.compile(
    r"#+training#+\s+round_id\s*=\s*(\d+)\s+data_id\s*=\s*(\d+)"
)
LOGICAL_CLIENT_PATTERN = re.compile(
    r"HEIMDALLM_LOGICAL_CLIENTS\s+round=(\d+)\s+ids=([0-9,]+)"
)
FD_PATTERN = re.compile(
    r"\[ZGR\] finite differences:\s*nonzero=(\d+)/(\d+)\s+"
    r"rate=([^\s]+)\s+mean_abs_delta=([^\s]+)\s+max_abs_delta=([^\s]+)"
)
COMPONENT_NORM_PATTERN = re.compile(
    r"\[ZGR\] component global L2:\s*raw_random=([^\s]+)\s+"
    r"local=([^\s]+)\s+cloud=([^\s]+)\s+final=([^\s]+)"
)
CLOUD_NORM_PATTERN = re.compile(
    r"\[ZGR\] cloud basis m=1:\s*raw_L2=([^\s]+)\s+"
    r"basis_L2=([^\s]+)\s+rademacher_seed=(\d+)\s+"
    r"sign=([^\s]+)\s+Vz_g_L2=([^\s]+)"
)


def require(condition, message):
    if not condition:
        raise ValueError(message)


def sha256_file(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_json(path):
    require(path.is_file(), "missing file: %s" % path)
    with path.open(encoding="utf-8") as handle:
        return json.load(handle)


def expected_eval_rounds(rounds, eval_every, pre_update_eval=False):
    values = [-1] if pre_update_eval else []
    values.extend(range(0, rounds, eval_every))
    final_round = rounds - 1
    if final_round not in values:
        values.append(final_round)
    return values


def finite_number(value, name):
    require(
        isinstance(value, (int, float)) and not isinstance(value, bool),
        "%s is not numeric: %r" % (name, value),
    )
    value = float(value)
    require(math.isfinite(value), "%s is not finite: %r" % (name, value))
    return value


def option_value(tokens, option):
    positions = [index for index, token in enumerate(tokens) if token == option]
    require(len(positions) == 1, "%s must occur exactly once in command" % option)
    position = positions[0]
    require(position + 1 < len(tokens), "%s has no command value" % option)
    return tokens[position + 1]


def parse_command(path):
    lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
    command_text = "\n".join(
        line for line in lines if not line.lstrip().startswith("#")
    )
    return shlex.split(command_text)


def parse_log(log_text):
    evaluations = []
    training_events = []
    finite_differences = []
    component_norms = []
    cloud_norms = []
    completion_lines = []
    logical_client_rounds = []
    pretrain_eval_begin_lines = []
    pretrain_eval_complete_lines = []

    for line_number, line in enumerate(log_text.splitlines(), start=1):
        if "HEIMDALLM_PRETRAIN_EVAL begin round=-1" in line:
            pretrain_eval_begin_lines.append(line_number)
        if "HEIMDALLM_PRETRAIN_EVAL complete round=-1" in line:
            pretrain_eval_complete_lines.append(line_number)

        train_match = TRAIN_PATTERN.search(line)
        if train_match:
            training_events.append({
                "line": line_number,
                "round": int(train_match.group(1)),
                "data_id": int(train_match.group(2)),
            })

        logical_client_match = LOGICAL_CLIENT_PATTERN.search(line)
        if logical_client_match:
            logical_client_rounds.append({
                "line": line_number,
                "round": int(logical_client_match.group(1)),
                "client_ids": [
                    int(value)
                    for value in logical_client_match.group(2).split(",")
                ],
            })

        eval_match = EVAL_PATTERN.search(line)
        if eval_match:
            try:
                record = ast.literal_eval(eval_match.group(1))
            except (SyntaxError, ValueError) as error:
                raise ValueError(
                    "unparseable eval metric at log line %d: %s"
                    % (line_number, error)
                )
            require(isinstance(record, dict), "eval metric is not a dictionary")
            acc = finite_number(record.get("acc"), "eval acc")
            loss = finite_number(record.get("eval_loss"), "eval loss")
            preceding_rounds = [
                event["round"] for event in training_events
                if event["line"] < line_number
            ]
            if preceding_rounds:
                inferred_round = preceding_rounds[-1]
            else:
                require(
                    pretrain_eval_begin_lines
                    and pretrain_eval_begin_lines[-1] < line_number,
                    "eval appears before training without a pre-update marker",
                )
                inferred_round = -1
            evaluations.append({
                "line": line_number,
                "inferred_round": inferred_round,
                "acc": acc,
                "eval_loss": loss,
            })

        fd_match = FD_PATTERN.search(line)
        if fd_match:
            nonzero = int(fd_match.group(1))
            total = int(fd_match.group(2))
            rate = finite_number(float(fd_match.group(3)), "FD nonzero rate")
            mean_delta = finite_number(float(fd_match.group(4)), "FD mean delta")
            max_delta = finite_number(float(fd_match.group(5)), "FD max delta")
            finite_differences.append({
                "line": line_number,
                "nonzero": nonzero,
                "total": total,
                "rate": rate,
                "mean_abs_delta": mean_delta,
                "max_abs_delta": max_delta,
            })

        component_match = COMPONENT_NORM_PATTERN.search(line)
        if component_match:
            component_norms.append({
                "line": line_number,
                "raw_random": finite_number(
                    float(component_match.group(1)), "raw random norm"
                ),
                "local": finite_number(
                    float(component_match.group(2)), "local component norm"
                ),
                "cloud": finite_number(
                    float(component_match.group(3)), "cloud component norm"
                ),
                "final": finite_number(
                    float(component_match.group(4)), "final direction norm"
                ),
            })

        cloud_match = CLOUD_NORM_PATTERN.search(line)
        if cloud_match:
            cloud_norms.append({
                "line": line_number,
                "raw": finite_number(float(cloud_match.group(1)), "cloud raw norm"),
                "basis": finite_number(
                    float(cloud_match.group(2)), "cloud basis norm"
                ),
                "seed": int(cloud_match.group(3)),
                "sign": finite_number(float(cloud_match.group(4)), "cloud sign"),
                "sampled": finite_number(
                    float(cloud_match.group(5)), "sampled cloud norm"
                ),
            })

        if "training is finished" in line:
            completion_lines.append(line_number)

    return {
        "evaluations": evaluations,
        "training_events": training_events,
        "logical_client_rounds": logical_client_rounds,
        "finite_differences": finite_differences,
        "component_norms": component_norms,
        "cloud_norms": cloud_norms,
        "completion_lines": completion_lines,
        "pretrain_eval_begin_lines": pretrain_eval_begin_lines,
        "pretrain_eval_complete_lines": pretrain_eval_complete_lines,
    }


def validate_artifacts(artifacts, hash_cache):
    require(artifacts, "manifest contains no artifacts")
    for name, record in artifacts.items():
        require(record.get("exists") is True, "artifact was absent: %s" % name)
        path = Path(record.get("path", ""))
        require(path.is_file(), "artifact is no longer present: %s" % path)
        require(path.stat().st_size == record.get("bytes"), "artifact size changed: %s" % name)
        key = str(path.resolve())
        if key not in hash_cache:
            hash_cache[key] = sha256_file(path)
        require(
            hash_cache[key] == record.get("sha256"),
            "artifact hash changed: %s" % name,
        )


def validate_training_schedule(events, rounds):
    require(events, "no client training events found")
    by_round = {}
    for event in events:
        by_round.setdefault(event["round"], []).append(event["data_id"])
    require(sorted(by_round) == list(range(rounds)), "client training rounds are incomplete")
    reference = by_round[0]
    require(reference == list(range(len(reference))), "round 0 data_id sequence is not contiguous")
    for round_index in range(rounds):
        require(
            by_round[round_index] == reference,
            "data_id schedule differs at round %d" % round_index,
        )
    return reference


def validate_logical_clients(events, rounds, total_clients, clients_per_round):
    require(len(events) == rounds, "logical-client schedule count differs from rounds")
    require(
        [event["round"] for event in events] == list(range(rounds)),
        "logical-client round sequence is incomplete",
    )
    schedule = []
    for event in events:
        client_ids = event["client_ids"]
        require(
            len(client_ids) == clients_per_round,
            "round %d logical-client count mismatch" % event["round"],
        )
        require(
            len(client_ids) == len(set(client_ids)),
            "round %d repeats a logical client" % event["round"],
        )
        require(
            all(0 <= client_id < total_clients for client_id in client_ids),
            "round %d has an out-of-range logical client" % event["round"],
        )
        if clients_per_round == total_clients:
            require(
                sorted(client_ids) == list(range(total_clients)),
                "round %d does not include every logical client" % event["round"],
            )
        schedule.append(client_ids)
    return schedule


def validate_arm(seed_dir, arm, arm_spec, run_values, expected_rounds, hash_cache):
    arm_dir = seed_dir / "arms" / arm
    metrics_path = arm_dir / "metrics.json"
    manifest_path = arm_dir / "manifest.json"
    command_path = arm_dir / "command.txt"
    log_path = arm_dir / "train.log"
    exit_code_path = arm_dir / "exit_code.txt"

    metrics = load_json(metrics_path)
    manifest = load_json(manifest_path)
    require(command_path.is_file(), "missing command file: %s" % command_path)
    require(log_path.is_file(), "missing train log: %s" % log_path)
    require(exit_code_path.is_file(), "missing exit-code file: %s" % exit_code_path)
    log_text = log_path.read_text(encoding="utf-8", errors="replace")
    parsed = parse_log(log_text)

    require(metrics.get("status") == "complete", "%s metrics are incomplete" % arm)
    require(manifest.get("status") == "complete", "%s manifest is incomplete" % arm)
    require(metrics.get("condition") == arm, "%s metrics condition mismatch" % arm)
    require(manifest.get("condition") == arm, "%s manifest condition mismatch" % arm)
    require(metrics.get("cloud_source") == arm_spec["cloud_source"], "%s cloud source mismatch" % arm)
    require(manifest.get("cloud_source") == arm_spec["cloud_source"], "%s manifest cloud source mismatch" % arm)
    require(metrics.get("evaluation_mode") == run_values["evaluation_mode"], "%s evaluation mode mismatch" % arm)
    require(manifest.get("evaluation_mode") == run_values["evaluation_mode"], "%s manifest evaluation mode mismatch" % arm)
    require(int(metrics.get("seed")) == int(run_values["seed"]), "%s metrics seed mismatch" % arm)
    require(int(manifest.get("seed")) == int(run_values["seed"]), "%s manifest seed mismatch" % arm)
    require(math.isclose(float(manifest.get("alpha")), arm_spec["alpha"]), "%s alpha mismatch" % arm)
    require(int(manifest.get("exit_code")) == 0, "%s manifest exit code is nonzero" % arm)
    require(int(exit_code_path.read_text().strip()) == 0, "%s recorded exit code is nonzero" % arm)
    require(metrics.get("checks"), "%s metrics checks are missing" % arm)
    require(manifest.get("checks"), "%s manifest checks are missing" % arm)
    require(all(value is True for value in metrics.get("checks", {}).values()), "%s metrics checks failed" % arm)
    require(all(value is True for value in manifest.get("checks", {}).values()), "%s manifest checks failed" % arm)
    require(
        manifest.get("stage") == run_values["stage"],
        "%s manifest stage mismatch" % arm,
    )
    require(int(manifest.get("rounds")) == int(run_values["rounds"]), "%s manifest rounds mismatch" % arm)
    require(int(manifest.get("eval_every")) == int(run_values["eval_every"]), "%s manifest eval frequency mismatch" % arm)
    require(
        bool(manifest.get("pre_update_evaluation"))
        == bool(run_values["pre_update_evaluation"]),
        "%s pre-update evaluation setting mismatch" % arm,
    )
    require(int(manifest.get("logical_clients")) == int(run_values["logical_clients"]), "%s logical-client count mismatch" % arm)
    require(int(manifest.get("clients_per_round")) == int(run_values["clients_per_round"]), "%s clients-per-round mismatch" % arm)
    require(int(manifest.get("mpi_workers")) == int(run_values["mpi_workers"]), "%s MPI-worker count mismatch" % arm)
    require(int(manifest.get("mpi_processes")) == int(run_values["mpi_workers"]) + 2, "%s MPI-process count mismatch" % arm)
    require(not metrics.get("fatal_matches"), "%s log contains a fatal pattern" % arm)
    require(not metrics.get("rejected_metric_lines"), "%s has rejected metric lines" % arm)

    evaluations = metrics.get("evaluations", [])
    require(metrics.get("expected_eval_rounds") == expected_rounds, "%s expected rounds mismatch" % arm)
    require(
        len(evaluations) == len(expected_rounds),
        "%s evaluation count mismatch" % arm,
    )
    require([item.get("round") for item in evaluations] == expected_rounds, "%s metric rounds mismatch" % arm)
    require(metrics.get("final") == evaluations[-1], "%s final metric mismatch" % arm)
    for index, evaluation in enumerate(evaluations):
        finite_number(evaluation.get("acc"), "%s evaluation %d acc" % (arm, index))
        finite_number(evaluation.get("eval_loss"), "%s evaluation %d loss" % (arm, index))

    parsed_evaluations = parsed["evaluations"]
    require(
        len(parsed_evaluations) == len(expected_rounds),
        "%s log evaluation count mismatch" % arm,
    )
    require(
        [item["inferred_round"] for item in parsed_evaluations] == expected_rounds,
        "%s eval lines do not follow the expected real training rounds" % arm,
    )
    for index, (logged, summarized) in enumerate(zip(parsed_evaluations, evaluations)):
        require(logged["line"] == summarized.get("log_line"), "%s eval log line mismatch" % arm)
        require(math.isclose(logged["acc"], float(summarized["acc"]), rel_tol=0.0, abs_tol=1e-12), "%s eval acc mismatch" % arm)
        require(math.isclose(logged["eval_loss"], float(summarized["eval_loss"]), rel_tol=0.0, abs_tol=1e-12), "%s eval loss mismatch" % arm)

    if run_values["pre_update_evaluation"]:
        require(
            len(parsed["pretrain_eval_begin_lines"]) == 1,
            "%s must have one pre-update evaluation begin marker" % arm,
        )
        require(
            len(parsed["pretrain_eval_complete_lines"]) == 1,
            "%s must have one pre-update evaluation completion marker" % arm,
        )
        first_training_line = parsed["training_events"][0]["line"]
        require(
            parsed["pretrain_eval_begin_lines"][0]
            < parsed_evaluations[0]["line"]
            < parsed["pretrain_eval_complete_lines"][0]
            < first_training_line,
            "%s pre-update evaluation did not finish before client training" % arm,
        )
        require(
            metrics.get("pre_update") == evaluations[0]
            and evaluations[0].get("round") == -1,
            "%s pre-update metric is missing or inconsistent" % arm,
        )
    else:
        require(
            not parsed["pretrain_eval_begin_lines"]
            and not parsed["pretrain_eval_complete_lines"],
            "%s has unexpected pre-update evaluation markers" % arm,
        )

    require(len(parsed["completion_lines"]) == 1, "%s must have one completion marker" % arm)
    require(parsed["completion_lines"][0] > parsed_evaluations[-1]["line"], "%s completion precedes final eval" % arm)

    rounds = int(run_values["rounds"])
    data_ids = validate_training_schedule(parsed["training_events"], rounds)
    logical_client_schedule = None
    if run_values["logical_client_schedule_logging"]:
        logical_client_schedule = validate_logical_clients(
            parsed["logical_client_rounds"],
            rounds,
            int(run_values["logical_clients"]),
            int(run_values["clients_per_round"]),
        )
    fd_records = parsed["finite_differences"]
    require(len(fd_records) == len(parsed["training_events"]), "%s FD summary count differs from training calls" % arm)
    for record in fd_records:
        require(record["total"] > 0, "%s has an empty FD summary" % arm)
        require(record["nonzero"] > 0, "%s has an all-zero finite-difference call" % arm)
        require(record["nonzero"] <= record["total"], "%s FD nonzero count exceeds total" % arm)
        require(
            math.isclose(
                record["rate"],
                float(record["nonzero"]) / float(record["total"]),
                rel_tol=0.0,
                abs_tol=1e-6,
            ),
            "%s FD nonzero rate is inconsistent with its counts" % arm,
        )
        require(record["mean_abs_delta"] > 0.0, "%s FD mean delta is not positive" % arm)
        require(record["max_abs_delta"] > 0.0, "%s FD max delta is not positive" % arm)

    direction_count = sum(record["total"] for record in fd_records)
    nonzero_direction_count = sum(record["nonzero"] for record in fd_records)
    zero_direction_count = direction_count - nonzero_direction_count
    require(
        float(nonzero_direction_count) / float(direction_count) >= 0.99,
        "%s has more than one percent zero finite-difference directions" % arm,
    )

    component_norms = parsed["component_norms"]
    require(len(component_norms) == len(parsed["training_events"]), "%s direction norm count differs from training calls" % arm)
    for record in component_norms:
        require(record["raw_random"] > 0.0, "%s raw direction norm is not positive" % arm)
        require(record["local"] > 0.0, "%s local direction norm is not positive" % arm)
        require(record["final"] > 0.0, "%s final direction norm is not positive" % arm)
        if arm == "no_cloud":
            require(record["cloud"] == 0.0, "no_cloud has a nonzero cloud component")
        else:
            require(record["cloud"] > 0.0, "%s cloud direction norm is not positive" % arm)

    cloud_norms = parsed["cloud_norms"]
    if arm == "no_cloud":
        require(not cloud_norms, "no_cloud unexpectedly constructed cloud directions")
        require(log_text.count("alpha=1: skip cloud BP") == rounds, "no_cloud did not skip cloud BP in every round")
        require(log_text.count("alpha=1: send an empty synchronization sentinel") == rounds, "no_cloud synchronization sentinel count mismatch")
    else:
        require(len(cloud_norms) == rounds, "%s cloud direction count mismatch" % arm)
        for round_index, record in enumerate(cloud_norms):
            require(record["raw"] > 0.0, "%s cloud raw norm is not positive" % arm)
            require(math.isclose(record["basis"], 1.0, rel_tol=1e-5, abs_tol=1e-6), "%s cloud basis is not unit norm" % arm)
            require(math.isclose(record["sampled"], 1.0, rel_tol=1e-5, abs_tol=1e-6), "%s sampled cloud direction is not unit norm" % arm)
            require(abs(record["sign"]) == 1.0, "%s cloud sign is not Rademacher" % arm)
            require(record["seed"] == int(run_values["seed"]) + round_index, "%s cloud direction seed mismatch" % arm)

    require("calculate more v" not in log_text, "%s performed an adaptive direction retry" % arm)
    command_tokens = parse_command(command_path)
    require(option_value(command_tokens, "--max_var_retries") == "0", "%s max_var_retries is not zero" % arm)
    require(option_value(command_tokens, "--v_num") == "1", "%s v_num is not one" % arm)
    require(option_value(command_tokens, "--pool_size") == "1", "%s pool_size is not one" % arm)
    require(int(option_value(command_tokens, "--comm_round")) == rounds, "%s command rounds mismatch" % arm)
    require(int(option_value(command_tokens, "--frequency_of_the_test")) == int(run_values["eval_every"]), "%s command eval frequency mismatch" % arm)
    require(
        command_tokens.count("--evaluate_before_training")
        == (1 if run_values["pre_update_evaluation"] else 0),
        "%s command pre-update evaluation flag mismatch" % arm,
    )
    require(math.isclose(float(option_value(command_tokens, "--alpha")), arm_spec["alpha"]), "%s command alpha mismatch" % arm)
    require(int(option_value(command_tokens, "--manual_seed")) == int(run_values["seed"]), "%s command seed mismatch" % arm)
    require(int(option_value(command_tokens, "--client_num_in_total")) == int(run_values["logical_clients"]), "%s command logical-client count mismatch" % arm)
    require(int(option_value(command_tokens, "--client_num_per_round")) == int(run_values["clients_per_round"]), "%s command clients-per-round mismatch" % arm)
    require(int(option_value(command_tokens, "--worker_num")) == int(run_values["mpi_workers"]), "%s command MPI-worker count mismatch" % arm)
    require(
        option_value(command_tokens, "--cloud_partition_method")
        == arm_spec["partition_method"],
        "%s command cloud partition method mismatch" % arm,
    )
    require(
        Path(option_value(command_tokens, "--cloud_data_file_path")).resolve()
        == (seed_dir / arm_spec["data_relative"]).resolve(),
        "%s command cloud data path mismatch" % arm,
    )
    require(
        Path(option_value(command_tokens, "--cloud_partition_file_path")).resolve()
        == (seed_dir / arm_spec["partition_relative"]).resolve(),
        "%s command cloud partition path mismatch" % arm,
    )

    require(sha256_file(command_path) == manifest.get("command_sha256"), "%s command hash mismatch" % arm)
    require(sha256_file(log_path) == manifest.get("log_sha256"), "%s log hash mismatch" % arm)
    validate_artifacts(manifest.get("artifacts", {}), hash_cache)

    return {
        "status": "complete",
        "final": {
            "round": evaluations[-1]["round"],
            "acc": evaluations[-1]["acc"],
            "eval_loss": evaluations[-1]["eval_loss"],
        },
        "pre_update": evaluations[0] if expected_rounds[0] == -1 else None,
        "evaluation_rounds": expected_rounds,
        "training_invocations": len(parsed["training_events"]),
        "data_ids_per_round": data_ids,
        "fd_summary_count": len(fd_records),
        "fd_directions": direction_count,
        "client_objective_queries": 2 * direction_count,
        "all_finite_differences_nonzero": zero_direction_count == 0,
        "nonzero_finite_difference_directions": nonzero_direction_count,
        "zero_finite_difference_directions": zero_direction_count,
        "finite_difference_nonzero_rate": (
            float(nonzero_direction_count) / float(direction_count)
        ),
        "component_norm_summary_count": len(component_norms),
        "cloud_direction_count": len(cloud_norms),
        "training_schedule": [
            [event["round"], event["data_id"]]
            for event in parsed["training_events"]
        ],
        "logical_client_schedule": logical_client_schedule,
        "fd_totals_by_training_invocation": [
            record["total"] for record in fd_records
        ],
        "cloud_rademacher": [
            [record["seed"], record["sign"]] for record in cloud_norms
        ],
    }


def resolve_directories(raw_run_dir, seed):
    candidate = Path(raw_run_dir).expanduser().resolve()
    if (candidate / "shared" / "run_spec.json").is_file():
        return candidate, candidate / ("seed_%d" % seed)
    if (candidate.parent / "shared" / "run_spec.json").is_file():
        require(candidate.name == "seed_%d" % seed, "seed directory name does not match --seed")
        return candidate.parent, candidate
    raise ValueError("run directory does not contain shared/run_spec.json: %s" % candidate)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", required=True,
                        help="Run root, or its seed_N directory")
    parser.add_argument("--seed", type=int, default=57)
    parser.add_argument("--output", default=None,
                        help="Default: seed_N/cross_arm_validation.json")
    args = parser.parse_args()

    errors = []
    arms = {}
    cross_arm = {}
    run_root = Path(args.run_dir).expanduser().resolve()
    seed_dir = run_root / ("seed_%d" % args.seed)
    run_spec_hash = None
    expected_rounds = None

    try:
        run_root, seed_dir = resolve_directories(args.run_dir, args.seed)
        run_spec_path = run_root / "shared" / "run_spec.json"
        run_spec = load_json(run_spec_path)
        payload = {
            "schema_version": run_spec.get("schema_version"),
            "values": run_spec.get("values"),
            "files": run_spec.get("files"),
        }
        run_spec_hash = hashlib.sha256(
            json.dumps(
                payload, ensure_ascii=True, sort_keys=True, separators=(",", ":")
            ).encode("utf-8")
        ).hexdigest()
        require(run_spec_hash == run_spec.get("spec_sha256"), "run spec internal hash mismatch")
        values = run_spec.get("values", {})
        require(
            values.get("stage") in ("smoke", "pilot"),
            "run spec stage is neither smoke nor pilot",
        )
        require(values.get("fixed_zo_query_budget") == "true", "run spec does not declare a fixed ZO query budget")
        require(values.get("downstream_max_var_retries") == "0", "run spec max_var_retries is not zero")
        require(values.get("downstream_var_control") == "true", "run spec var_control is not enabled")
        require(values.get("downstream_perturbation_sampling") == "true", "run spec perturbation_sampling is not enabled")
        require(str(args.seed) in values.get("seeds", "").split(","), "seed is absent from run spec")
        rounds = int(values["rounds"])
        eval_every = int(values["eval_every"])
        pre_update_evaluation = (
            values.get("downstream_pre_update_evaluation", "false") == "true"
        )
        expected_rounds = expected_eval_rounds(
            rounds, eval_every, pre_update_evaluation
        )
        declared_arms = []
        for arm_entry in values.get("downstream_arms", "").split(","):
            if not arm_entry:
                continue
            arm_name, separator, raw_alpha = arm_entry.partition(":")
            require(separator == ":", "invalid downstream arm declaration")
            require(arm_name in ARM_SPECS, "unknown declared arm: %s" % arm_name)
            require(arm_name not in declared_arms, "duplicate declared arm: %s" % arm_name)
            require(
                math.isclose(float(raw_alpha), ARM_SPECS[arm_name]["alpha"]),
                "declared alpha mismatch for %s" % arm_name,
            )
            declared_arms.append(arm_name)
        require("no_cloud" in declared_arms, "declared arms omit no_cloud")
        require("client_syn" in declared_arms, "declared arms omit client_syn")
        dataset = values.get("dataset", "sst_2")
        require(dataset in ("sst_2", "agnews"), "unsupported dataset in run spec")
        active_arm_specs = {name: dict(ARM_SPECS[name]) for name in declared_arms}
        for spec in active_arm_specs.values():
            for key in ("data_relative", "partition_relative"):
                spec[key] = spec[key].replace("sst_2_", dataset + "_")
        run_values = {
            "stage": values["stage"],
            "rounds": rounds,
            "eval_every": eval_every,
            "pre_update_evaluation": pre_update_evaluation,
            "logical_client_schedule_logging": (
                values.get("downstream_logical_client_schedule_logging", "false")
                == "true"
            ),
            "evaluation_mode": values["evaluation_mode"],
            "seed": args.seed,
            "logical_clients": int(values["logical_client_count"]),
            "clients_per_round": int(values["clients_per_round"]),
            "mpi_workers": int(values["mpi_workers"]),
        }

        hash_cache = {}
        for arm, arm_spec in active_arm_specs.items():
            try:
                arms[arm] = validate_arm(
                    seed_dir, arm, arm_spec, run_values,
                    expected_rounds, hash_cache,
                )
            except Exception as error:
                errors.append("%s: %s" % (arm, error))
                arms[arm] = {"status": "failed", "error": str(error)}

        if not errors:
            schedules = [arms[name]["training_schedule"] for name in declared_arms]
            fd_totals = [arms[name]["fd_totals_by_training_invocation"] for name in declared_arms]
            query_counts = [arms[name]["client_objective_queries"] for name in declared_arms]
            logical_client_schedules = [
                arms[name]["logical_client_schedule"] for name in declared_arms
            ]
            require(all(item == schedules[0] for item in schedules[1:]), "training schedule differs across arms")
            require(all(item == fd_totals[0] for item in fd_totals[1:]), "FD direction schedule differs across arms")
            require(all(item == query_counts[0] for item in query_counts[1:]), "client query budget differs across arms")
            if run_values["logical_client_schedule_logging"]:
                require(
                    all(
                        item == logical_client_schedules[0]
                        for item in logical_client_schedules[1:]
                    ),
                    "logical-client schedule differs across arms",
                )
            guided_rademacher = [
                arms[name]["cloud_rademacher"]
                for name in declared_arms if name != "no_cloud"
            ]
            require(all(item == guided_rademacher[0] for item in guided_rademacher[1:]), "paired cloud signs differ across guided arms")
            pre_update_metrics_identical = None
            if pre_update_evaluation:
                baselines = [arms[name]["pre_update"] for name in declared_arms]
                pre_update_metrics_identical = all(
                    math.isclose(
                        float(item["acc"]), float(baselines[0]["acc"]),
                        rel_tol=0.0, abs_tol=1e-12,
                    )
                    and math.isclose(
                        float(item["eval_loss"]),
                        float(baselines[0]["eval_loss"]),
                        rel_tol=0.0, abs_tol=1e-7,
                    )
                    for item in baselines[1:]
                )
                require(
                    pre_update_metrics_identical,
                    "pre-update metrics differ across paired arms",
                )
            cross_arm = {
                "all_declared_arms_complete": True,
                "declared_arm_count": len(declared_arms),
                "declared_arms": declared_arms,
                "evaluation_rounds_identical": True,
                "pre_update_metrics_identical": pre_update_metrics_identical,
                "training_schedule_identical": True,
                "logical_client_schedule_identical": (
                    True
                    if run_values["logical_client_schedule_logging"]
                    else None
                ),
                "fd_direction_schedule_identical": True,
                "fixed_client_objective_query_budget": True,
                "client_objective_queries_per_arm": query_counts[0],
                "paired_cloud_rademacher_sequence_identical": True,
            }
    except Exception as error:
        errors.append(str(error))

    status = "complete" if not errors else "failed"
    result = {
        "schema_version": 1,
        "status": status,
        "diagnostic": "nondp_paired_feasibility_cross_validation",
        "run_dir": str(run_root),
        "seed_dir": str(seed_dir),
        "seed": args.seed,
        "run_spec_sha256": run_spec_hash,
        "expected_eval_rounds": expected_rounds,
        "arms": arms,
        "cross_arm": cross_arm,
        "errors": errors,
    }
    output = (
        Path(args.output).expanduser().resolve()
        if args.output else seed_dir / "cross_arm_validation.json"
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps({"status": status, "output": str(output), "errors": errors}, sort_keys=True))
    return 0 if status == "complete" else 2


if __name__ == "__main__":
    sys.exit(main())
