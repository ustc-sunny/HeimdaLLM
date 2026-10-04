#!/usr/bin/env python3
"""Back up validated DP filter runs; publish only curated curves and metrics."""
import argparse
import csv
import hashlib
import json
import shutil
import subprocess
import tarfile
import tempfile
import time
from pathlib import Path


REPO = Path(__file__).resolve().parents[1]
BASE = "/home/zzkevin/heimdallm-journal-20260928"
RUN_ID = "ustc_agnews_dp_filter_20261004"
ARMS = ("filtered_dp", "random_dp", "filtered_public")
SEEDS = (57, 58, 59)


def digest(path):
    sha = hashlib.sha256()
    with Path(path).open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            sha.update(chunk)
    return sha.hexdigest()


def call(command, **kwargs):
    return subprocess.run([str(x) for x in command], check=True, **kwargs)


def sync(args):
    options = ["-o", "BatchMode=yes", "-o", "ConnectTimeout=10",
               "-o", "ControlPath=" + args.control_socket]
    ssh = ["ssh", *options, "-p", "1010", "zzkevin@172.16.50.129"]
    scp = ["scp", *options, "-P", "1010"]
    root = BASE + "/results/" + RUN_ID
    call([*ssh, "test -f " + root + "/summary/summary.json"],
         stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    backup = args.backup_dir.resolve()
    backup.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="incoming-", dir=backup) as name:
        incoming = Path(name)
        for file in ("summary.json", "curves.csv", "REPORT.md"):
            call([*scp, "zzkevin@172.16.50.129:" + root + "/summary/" + file,
                  incoming / file], stdout=subprocess.DEVNULL)
        summary = json.loads((incoming / "summary.json").read_text())
        expected = [(seed, arm) for seed in SEEDS for arm in ARMS]
        if [(c["seed"], c["arm"]) for c in summary["protocol"]["cases"]] != expected:
            raise ValueError("case list mismatch")
        if summary["protocol"]["rounds"] != 50 or summary["runs_expected"] != 9:
            raise ValueError("protocol mismatch")
        if len(summary["runs"]) != summary["runs_completed"]:
            raise ValueError("run count mismatch")
        with (incoming / "curves.csv").open(newline="") as source:
            curves = list(csv.DictReader(source))
        if len(curves) != summary["runs_completed"] * 51:
            raise ValueError("trajectory snapshot mismatch")
        for record in summary["runs"]:
            arm, seed = record["arm"], record["seed"]
            if (seed, arm) not in expected:
                raise ValueError("unexpected run")
            receipt = record["archive"]
            file = receipt["name"]
            if file != "%s_%s_seed%d.tar.gz" % (RUN_ID, arm, seed):
                raise ValueError("archive name mismatch")
            local = backup / file
            if not local.exists() or digest(local) != receipt["sha256"]:
                temporary = incoming / file
                call([*scp, "zzkevin@172.16.50.129:" + BASE + "/backups/" + file,
                      temporary], stdout=subprocess.DEVNULL)
                if digest(temporary) != receipt["sha256"]:
                    raise ValueError("archive SHA256 mismatch")
                temporary.replace(local)
            with tarfile.open(local, "r:gz") as archive:
                stem = "%s/seed_%d/" % (arm, seed)
                raw = archive.extractfile(stem + "metrics.json").read()
                selected = archive.extractfile(stem + "selected.jsonl").read()
                manifest = json.load(archive.extractfile(stem + "pack_manifest.json"))
            if hashlib.sha256(raw).hexdigest() != record["metrics_sha256"]:
                raise ValueError("metric SHA256 mismatch")
            metric = json.loads(raw)
            if metric["status"] != "complete" or not all(metric["checks"].values()) or \
               [row["round"] for row in metric["evaluations"]] != list(range(-1, 50)):
                raise ValueError("incomplete run metric")
            if abs(metric["final"]["acc"] - record["final_accuracy"]) > 1e-12:
                raise ValueError("final accuracy mismatch")
            if manifest["records"] != 64 or manifest["label_counts"] != {str(i): 16 for i in range(1, 5)}:
                raise ValueError("selected release pack mismatch")
            selection = json.loads((backup / "selection_v1/selection_summary.json").read_text())
            chosen = next(r for r in selection["runs"] if r["arm"] == arm and r["seed"] == seed)
            if hashlib.sha256(selected).hexdigest() != chosen["selected_sha256"]:
                raise ValueError("selected input SHA256 mismatch")
            (backup / (file + ".sha256")).write_text(receipt["sha256"] + "  " + file + "\n")
        destination = backup / "downstream_summary"
        destination.mkdir(exist_ok=True)
        for file in ("summary.json", "curves.csv", "REPORT.md"):
            shutil.copy2(incoming / file, destination / file)
        finished = summary["status"] == "complete"
        if not finished:
            decision = call([*ssh, "test -f " + root + "/pilot_decision.json && cat " +
                             root + "/pilot_decision.json || true"],
                            capture_output=True, text=True).stdout.strip()
            if decision:
                parsed = json.loads(decision)
                finished = parsed["continue"] is False and summary["runs_completed"] == 3
        if args.publish and finished:
            if call(["git", "branch", "--show-current"], cwd=REPO,
                    capture_output=True, text=True).stdout.strip() != "HeimdaLLM+":
                raise ValueError("publish requires HeimdaLLM+ branch")
            target = REPO / "journal_extension/results/agnews_dp_filter_ustc"
            target.mkdir(parents=True, exist_ok=True)
            files = []
            for file in ("summary.json", "curves.csv", "REPORT.md"):
                shutil.copy2(incoming / file, target / file)
                files.append(str((target / file).relative_to(REPO)))
            call(["git", "add", "-f", "--", *files], cwd=REPO)
            changed = subprocess.run(["git", "diff", "--cached", "--quiet", "--", *files], cwd=REPO)
            if changed.returncode == 1:
                call(["git", "commit", "--only", "-m", "Save AG News DP filter pilot results",
                      "--", *files], cwd=REPO)
            elif changed.returncode:
                raise RuntimeError("cannot inspect staged results")
            call(["git", "-c", "http.proxy=socks5h://127.0.0.1:10899",
                  "push", "origin", "HeimdaLLM+"], cwd=REPO)
        print("Verified %d/9 DP filter archives; finished=%s" %
              (summary["runs_completed"], finished), flush=True)
        return finished


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--control-socket", default="/private/tmp/heimdallm-ustc-control")
    parser.add_argument("--backup-dir", type=Path,
                        default=REPO.parent / "local_backups/agnews_dp_recovery_20261004")
    parser.add_argument("--publish", action="store_true")
    parser.add_argument("--watch", action="store_true")
    parser.add_argument("--interval", type=int, default=120)
    args = parser.parse_args()
    if args.interval < 30:
        parser.error("interval must be at least 30 seconds")
    while True:
        try:
            finished = sync(args)
        except (subprocess.CalledProcessError, ValueError, OSError, KeyError) as error:
            if not args.watch:
                raise
            print("Sync pending: %s" % error, flush=True)
            finished = False
        if finished or not args.watch:
            return
        time.sleep(args.interval)


if __name__ == "__main__":
    main()
