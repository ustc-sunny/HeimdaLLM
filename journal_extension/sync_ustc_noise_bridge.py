#!/usr/bin/env python3
"""Verify noise-bridge raw archives locally and publish only curated metrics."""
import argparse
import csv
import json
import shlex
import shutil
import subprocess
import tarfile
import tempfile
import time
from pathlib import Path

from sync_ustc_agnews_results import digest, run


REPO = Path(__file__).resolve().parents[1]
RUN_ID = "ustc_agnews_noise_bridge_20260930"
REMOTE = "/home/zzkevin/heimdallm-journal-20260928"


def synchronize(args):
    options = ["-o", "BatchMode=yes", "-o", "ConnectTimeout=15",
               "-o", "ServerAliveInterval=15", "-o", "ServerAliveCountMax=2",
               "-o", "ControlPath=" + args.control_socket]
    ssh = ["ssh", *options, "-p", "1010", "zzkevin@172.16.50.129"]
    scp = ["scp", *options, "-P", "1010"]
    summary = REMOTE + "/results/" + RUN_ID + "/summary"
    run([*ssh, "test -f " + shlex.quote(summary + "/summary.json")],
        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    backup = args.backup_dir.resolve()
    backup.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="incoming-", dir=backup) as temporary:
        incoming = Path(temporary)
        for name in ("summary.json", "curves.csv", "REPORT.md"):
            run([*scp, "zzkevin@172.16.50.129:" + summary + "/" + name,
                 str(incoming / name)], stdout=subprocess.DEVNULL)
        document = json.loads((incoming / "summary.json").read_text())
        protocol = document["protocol"]
        planned = [(seed, arm) for seed in (57, 58, 59)
                   for arm in ("zero_noise", "dp_eps8")]
        if [(case["seed"], case["arm"]) for case in protocol["cases"]] != planned:
            raise ValueError("unexpected bridge case list")
        if protocol["rounds"] != 50 or document["runs_expected"] != 6:
            raise ValueError("unexpected bridge protocol")
        if len(document["runs"]) != document["runs_completed"]:
            raise ValueError("summary run count mismatch")
        with (incoming / "curves.csv").open(newline="") as handle:
            curves = list(csv.DictReader(handle))
        if len(curves) != document["runs_completed"] * 51:
            raise ValueError("summary and trajectory snapshot differ; retry")
        if len({(r["arm"], r["seed"], r["round"]) for r in curves}) != len(curves):
            raise ValueError("duplicate trajectory points")
        for record in document["runs"]:
            arm, seed = record["arm"], record["seed"]
            if (seed, arm) not in planned:
                raise ValueError("unplanned bridge result")
            receipt = record["archive"]
            name, checksum = receipt["name"], receipt["sha256"]
            if name != "%s_%s_seed%d.tar.gz" % (RUN_ID, arm, seed) or len(checksum) != 64:
                raise ValueError("archive provenance mismatch")
            local = backup / name
            if not local.exists() or digest(local) != checksum:
                pending = incoming / name
                run([*scp, "zzkevin@172.16.50.129:" + REMOTE + "/backups/" + name,
                     str(pending)], stdout=subprocess.DEVNULL)
                if digest(pending) != checksum:
                    raise ValueError("archive SHA256 mismatch")
                pending.replace(local)
            with tarfile.open(local, "r:gz") as archive:
                stem = "%s/seed_%d/" % (arm, seed)
                raw = archive.extractfile(stem + "metrics.json").read()
                generator = json.load(archive.extractfile(stem + "generated/manifest.json"))
                if arm == "dp_eps8":
                    validation = json.load(archive.extractfile(stem + "dp_validation.json"))
                    if validation["status"] != "complete" or not all(validation["checks"].values()):
                        raise ValueError("DP release validation failed")
                    if generator.get("is_record_level_dp") is not True:
                        raise ValueError("DP arm lacks its privacy declaration")
                elif generator.get("is_record_level_dp") is not False or \
                     generator["privacy"]["noise_added_to_clipped_gradient_sum"] is not False:
                    raise ValueError("zero-noise control is mislabeled")
            import hashlib
            if hashlib.sha256(raw).hexdigest() != record["metrics_sha256"]:
                raise ValueError("raw metrics hash mismatch")
            metrics = json.loads(raw)
            if metrics["status"] != "complete" or not all(metrics["checks"].values()):
                raise ValueError("raw run validation failed")
            if [r["round"] for r in metrics["evaluations"]] != list(range(-1, 50)):
                raise ValueError("raw trajectory incomplete")
            if abs(metrics["final"]["acc"] - record["final_accuracy"]) > 1e-12:
                raise ValueError("raw final accuracy mismatch")
            (backup / (name + ".sha256")).write_text(checksum + "  " + name + "\n")
        snapshot = backup / "summary"
        snapshot.mkdir(exist_ok=True)
        for name in ("summary.json", "curves.csv", "REPORT.md"):
            shutil.copy2(incoming / name, snapshot / name)
        if args.publish:
            branch = run(["git", "branch", "--show-current"], cwd=REPO,
                         capture_output=True, text=True).stdout.strip()
            if branch != "HeimdaLLM+":
                raise ValueError("publishing requires HeimdaLLM+ branch")
            target = REPO / "journal_extension/results/agnews_dp_noise_bridge_ustc"
            target.mkdir(parents=True, exist_ok=True)
            paths = []
            for name in ("summary.json", "curves.csv", "REPORT.md"):
                output = target / name
                shutil.copy2(incoming / name, output)
                paths.append(str(output.relative_to(REPO)))
            run(["git", "add", "--", *paths], cwd=REPO)
            changed = subprocess.run(["git", "diff", "--cached", "--quiet", "--", *paths], cwd=REPO)
            if changed.returncode == 1:
                run(["git", "commit", "--only", "-m",
                     "Save AG News DP noise bridge (%d/6 runs)" % document["runs_completed"],
                     "--", *paths], cwd=REPO)
            elif changed.returncode:
                raise RuntimeError("cannot inspect staged bridge results")
            run(["git", "push", "origin", "HeimdaLLM+"], cwd=REPO)
        print("Verified noise bridge: %d/6 raw archives" % document["runs_completed"], flush=True)
        return document["status"] == "complete"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--control-socket", default="/private/tmp/heimdallm-ustc-control")
    parser.add_argument("--backup-dir", type=Path,
                        default=REPO.parent / "local_backups/agnews_dp_noise_bridge_ustc_20260930")
    parser.add_argument("--publish", action="store_true")
    parser.add_argument("--watch", action="store_true")
    parser.add_argument("--interval", type=int, default=120)
    args = parser.parse_args()
    if args.interval < 30:
        parser.error("interval must be at least 30 seconds")
    while True:
        try:
            complete = synchronize(args)
        except (subprocess.CalledProcessError, ValueError, OSError, KeyError) as error:
            if not args.watch:
                raise
            print("Sync pending; retrying: %s" % error, flush=True)
            complete = False
        if complete or not args.watch:
            return
        time.sleep(args.interval)


if __name__ == "__main__":
    main()
