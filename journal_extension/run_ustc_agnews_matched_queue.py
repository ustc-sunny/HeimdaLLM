#!/usr/bin/env python3
"""Gate the formal Non-DP queue on completed, numerically valid probes."""
import argparse
import fcntl
import json
import math
import re
import subprocess
import time
from pathlib import Path

import non_dp_client_synthetic as common
from run_ustc_agnews_matched import PROBES, SCRIPT, now


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--work-root", type=Path,
                        default=Path("/home/zzkevin/heimdallm-journal-20260928"))
    parser.add_argument("--probe-run-id", default="ustc_agnews_matched_probes_v2_20260929")
    parser.add_argument("--formal-run-id", default="ustc_agnews_matched_formal_20260929")
    args = parser.parse_args()
    base = args.work_root
    lock = (base / "logs/matched-queue.lock").open("a")
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    root = base / "results" / args.probe_run_id
    deadline = time.monotonic() + 4 * 3600
    while True:
        state_file = root / "state.json"
        if state_file.exists():
            state = json.loads(state_file.read_text())
            if state["status"] == "failed":
                raise RuntimeError("probe failed; formal queue not launched")
            if state["status"] == "complete":
                break
        if time.monotonic() > deadline:
            raise TimeoutError("probe wait exceeded four hours; formal queue not launched")
        time.sleep(30)
    summary_path = root / "summary/summary.json"
    summary = json.loads(summary_path.read_text())
    if summary["status"] != "complete" or summary["runs_completed"] != len(PROBES):
        raise ValueError("probe summary is incomplete")
    private = base / "private_staging" / args.probe_run_id
    audit = {}
    for run in summary["runs"]:
        log = root / run["arm"] / "seed_57/train.log"
        matches = re.findall(
            r"\[UPDATE_AUDIT\].*actual_l2=([\deE+.-]+) planned_l2=([\deE+.-]+) changed=(\d+)/(\d+)",
            log.read_text(errors="replace"))
        if len(matches) != 5 * 15:
            raise ValueError("missing update audit steps: " + run["arm"])
        values = [(float(a), float(p), int(c), int(n)) for a, p, c, n in matches]
        if any(not math.isfinite(a) or not math.isfinite(p) or a <= 0 or p <= 0 or c <= 0
               for a, p, c, n in values):
            raise ValueError("non-finite/zero update found; formal queue not launched")
        audit[run["arm"]] = {"update_steps": len(values), "actual_l2": [v[0] for v in values],
                             "planned_l2": [v[1] for v in values], "changed": [v[2] for v in values]}
    common.atomic_write_json(private / "probe_update_audit.json", audit)
    candidates = [r for r in summary["runs"] if PROBES[r["arm"]][0] == "isotropic"]
    selected = sorted(candidates, key=lambda r: (-r["final_accuracy"], r["final_loss"],
                                                PROBES[r["arm"]][1]))[0]
    lr = PROBES[selected["arm"]][1]
    selection = {"status": "complete", "selected_arm": selected["arm"],
        "selected_learning_rate": lr, "rule": summary["protocol"]["baseline_selection"],
        "probe_summary_sha256": common.sha256_file(summary_path), "all_update_audits_passed": True,
        "extra_objective_queries": 1500, "selected_at": now(),
        "selection_split": "development_only", "official_test_used": False,
        "interpretation": "exploratory calibration; no final utility claim"}
    common.atomic_write_json(root / "summary/baseline_selection.json", selection)
    print(json.dumps(selection), flush=True)
    formal = base / "results" / args.formal_run_id
    command = [str(base / "envs/kdd/bin/python"), "-u",
        str(SCRIPT / "run_ustc_agnews_matched.py"), "--phase", "formal", "--run-id", args.formal_run_id,
        "--work-root", str(base), "--rounds", "50", "--seeds", "57,58,59",
        "--selected-baseline-lr", str(lr), "--probe-summary", str(summary_path)]
    if formal.exists():
        command.append("--resume")
    subprocess.run(command, check=True)
    print("All matched Non-DP formal conditions completed at " + now(), flush=True)


if __name__ == "__main__":
    main()
