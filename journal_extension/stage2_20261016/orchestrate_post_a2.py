#!/usr/bin/env python3
"""Finish the preregistered A1/B/report work after the A2 queue.

This runner performs no new DP release and will not start C or an epsilon scan.
Every complete endpoint has its own immutable output directory and exit code.
"""

import ast
import json
import math
import os
import re
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path


S1 = Path("/mnt/heimdallm-stage1-20261015")
S2 = Path("/mnt/heimdallm-stage2-20261016")
REPO = Path("/root/heimdallm-stage2-20261016/HeimdaLLM")
A1 = Path("/root/heimdallm-stage2-20261016/HeimdaLLM_A1")
PYTHON = Path("/root/miniconda3/envs/heimdallm-kdd/bin/python")
STATUS = S2 / "post_a2_orchestration.json"
PROTOCOL_SEEDS = (57, 58, 59)
BASELINE_SEMANTICS = (
    "alpha=0.5 uses the same 64 public texts as the paired PRIVATE_DP_E4 "
    "release, with a fixed cyclic class-label permutation. This is "
    "postprocessing of an existing guidance release and does not make "
    "the downstream FL transcript differentially private."
)


def update_status(**values):
    old = json.loads(STATUS.read_text()) if STATUS.exists() else {}
    old.update(values)
    old["updated_at_utc"] = datetime.now(timezone.utc).isoformat()
    STATUS.write_text(json.dumps(old, indent=2, sort_keys=True) + "\n")


def run_logged(command, path, env=None):
    if path.exists():
        raise FileExistsError(path)
    with path.open("w") as handle:
        result = subprocess.run(command, stdout=handle, stderr=subprocess.STDOUT,
                                env=env)
    if result.returncode:
        raise RuntimeError("command failed with exit %d; see %s" % (result.returncode, path))


def wait_for_a2():
    deadline = time.monotonic() + 5 * 3600
    while time.monotonic() < deadline:
        statuses = {}
        for seed in PROTOCOL_SEEDS:
            path = S2 / "runs/SHUFFLED_DP_LABEL" / ("seed_%d" % seed) / "run_spec.json"
            statuses[str(seed)] = json.loads(path.read_text())["status"] if path.exists() else "not_started"
        update_status(stage="waiting_for_A2", a2_statuses=statuses)
        if all(value == "complete" for value in statuses.values()):
            return
        if any(value.endswith("failed") for value in statuses.values()):
            raise RuntimeError("an A2 endpoint failed; no dependent work started")
        log_path = S2 / "a2_queue.log"
        if log_path.exists() and "Traceback (most recent call last)" in log_path.read_text():
            raise RuntimeError("A2 queue traceback; no dependent work started")
        time.sleep(30)
    raise TimeoutError("A2 did not finish within 5 hours")


def correct_a2_semantics():
    for seed in PROTOCOL_SEEDS:
        path = S2 / "runs/SHUFFLED_DP_LABEL" / ("seed_%d" % seed) / "manifest.json"
        doc = json.loads(path.read_text())
        if doc.get("baseline_semantics") != BASELINE_SEMANTICS:
            doc["baseline_semantics"] = BASELINE_SEMANTICS
            path.write_text(json.dumps(doc, indent=2, sort_keys=True) + "\n")
    update_status(a2_manifest_semantics_corrected=True)


def review_smoke():
    root = S2 / "preflight/NO_GUIDANCE/seed_57"
    if (root / "train.exit_code.txt").read_text().strip() != "0":
        raise RuntimeError("A1 smoke exited nonzero")
    log = (root / "train.log").read_text(errors="replace")
    required = ("parameterization=raw_isotropic_z", "pure ZOO: cloud rank has no training data",
                "skip cloud BP", "send an empty synchronization sentinel")
    if not all(part in log for part in required):
        raise RuntimeError("A1 smoke did not take the no-guidance source/operator path")
    components = re.findall(
        r"component global L2: raw_random=([0-9.eE+-]+) local=([0-9.eE+-]+) "
        r"cloud=([0-9.eE+-]+) final=([0-9.eE+-]+)", log)
    if len(components) != 15:
        raise RuntimeError("A1 smoke missing one or more raw direction audits")
    raw_norms = []
    for raw, local, cloud, final in components:
        values = tuple(map(float, (raw, local, cloud, final)))
        if (not all(math.isfinite(x) for x in values) or values[2] != 0.0 or
                not math.isclose(values[0], values[1], rel_tol=1e-5) or
                not math.isclose(values[0], values[3], rel_tol=1e-5)):
            raise RuntimeError("A1 final direction differs from raw z")
        raw_norms.append(values[0])
    if not all(550 < x < 790 for x in raw_norms):
        raise RuntimeError("raw z norm incompatible with d=450340")
    finite_diffs = re.findall(r"finite differences: nonzero=(\d+)/(\d+) rate=([0-9.]+) "
                              r"mean_abs_delta=([0-9.eE+-]+)", log)
    if (len(finite_diffs) != 15 or sum(int(x[1]) for x in finite_diffs) != 30 or
            any(int(x[0]) == 0 or float(x[3]) <= 0 for x in finite_diffs)):
        raise RuntimeError("A1 smoke finite difference is zero or incomplete")
    audits = re.findall(r"\[UPDATE_AUDIT\].*?actual_l2=([0-9.eE+-]+) "
                        r"planned_l2=([0-9.eE+-]+) changed=(\d+)/(\d+)", log)
    if len(audits) != 15:
        raise RuntimeError("A1 smoke update audit count mismatch")
    update_norms = []
    for actual, planned, changed, total in audits:
        a, p = float(actual), float(planned)
        if (not math.isfinite(a) or not math.isfinite(p) or p <= 0 or
                not math.isclose(a, p, rel_tol=1e-3, abs_tol=1e-6) or
                int(changed) <= 0 or int(total) != 450340):
            raise RuntimeError("A1 smoke actual update differs from planned update")
        update_norms.append(a)
    evaluations = [ast.literal_eval(text) for text in
                   re.findall(r"eval_model\(\): (\{[^\n]+\})", log)]
    if len(evaluations) != 2:
        raise RuntimeError("A1 smoke lacks pre/post dev evaluations")
    reference = json.loads((S1 / "runs/PRIVATE_DP_E4/seed_57/metrics.json").read_text())["pre_update"]
    initial = evaluations[0]
    if (initial["acc"] != reference["acc"] or
            initial["confusion_matrix"] != reference["confusion_matrix"] or
            not math.isclose(initial["eval_loss"], reference["eval_loss"], abs_tol=1e-6)):
        raise RuntimeError("A1 initial model/dev does not match Stage-1 seed57")
    if not math.isfinite(evaluations[-1]["eval_loss"]):
        raise RuntimeError("A1 smoke post-update dev loss is nonfinite")
    report = {"status": "pass", "raw_z_norm_min": min(raw_norms),
              "raw_z_norm_max": max(raw_norms),
              "update_l2_min": min(update_norms), "update_l2_max": max(update_norms),
              "fd_directions": 30, "objective_queries": 60,
              "preupdate_acc": initial["acc"], "postupdate_acc": evaluations[-1]["acc"],
              "preupdate_loss": initial["eval_loss"],
              "postupdate_loss": evaluations[-1]["eval_loss"],
              "notes": "automated numeric gate; inspect full log for interpretation"}
    (root / "operator_gate.json").write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    update_status(a1_smoke_gate=report)


def main():
    if STATUS.exists():
        raise FileExistsError(STATUS)
    try:
        update_status(stage="started")
        wait_for_a2()
        time.sleep(15)  # let the final MPI ranks release GPU memory
        correct_a2_semantics()
        update_status(stage="A1_smoke")
        run_logged([str(PYTHON), str(A1 / "journal_extension/stage2_20261016/run_stage2_a1.py"),
                    "--seed", "57", "--smoke"], S2 / "a1_smoke_queue.log")
        review_smoke()
        for seed in PROTOCOL_SEEDS:
            update_status(stage="A1_formal", current_seed=seed)
            run_logged([str(PYTHON), str(A1 / "journal_extension/stage2_20261016/run_stage2_a1.py"),
                        "--seed", str(seed)], S2 / ("a1_seed_%d_queue.log" % seed))
        update_status(stage="B1_round0")
        env = os.environ.copy()
        env.update(PYTHONPATH=":".join(map(str, (REPO, REPO / "FedML"))),
                   OMP_NUM_THREADS="1", MKL_NUM_THREADS="1",
                   TOKENIZERS_PARALLELISM="false", HF_HUB_OFFLINE="1",
                   TRANSFORMERS_OFFLINE="1", PYTHONNOUSERSITE="1")
        try:
            run_logged([str(PYTHON), str(REPO / "journal_extension/stage2_20261016/diagnose_round0_zgr.py")],
                       S2 / "b1_round0_queue.log", env=env)
            update_status(b1_status="complete")
        except Exception as error:
            update_status(b1_status="failed_or_blocked", b1_error=str(error))
        update_status(stage="build_report")
        run_logged([str(PYTHON), str(REPO / "journal_extension/stage2_20261016/build_stage2_report.py")],
                   S2 / "build_report.log")
        update_status(stage="complete", report_path=str(S2 / "report_public"))
    except Exception as error:
        update_status(stage="blocked", error=str(error))
        raise
    finally:
        (S2 / "resource_monitor.stop").touch()


if __name__ == "__main__":
    main()
