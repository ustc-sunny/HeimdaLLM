#!/usr/bin/env python3
"""Verified local raw backups and curated GitHub snapshots for matched ablations.

Uses an existing authenticated SSH socket; contains no password. The server queue
and archives survive a local disconnect. Local/GitHub syncing needs local wake/VPN.
"""
import argparse
import csv
import io
import json
import shlex
import shutil
import subprocess
import tempfile
import tarfile
import time
from pathlib import Path

from sync_ustc_agnews_results import digest, run


REPO = Path(__file__).resolve().parents[1]
REMOTE = "/home/zzkevin/heimdallm-journal-20260928"


def synchronize(args):
    options = ["-o", "BatchMode=yes", "-o", "ConnectTimeout=15", "-o", "ServerAliveInterval=15",
               "-o", "ServerAliveCountMax=2", "-o", "ControlPath=" + args.control_socket]
    ssh = ["ssh", *options, "-p", "1010", "zzkevin@172.16.50.129"]
    scp = ["scp", *options, "-P", "1010"]
    target = REPO / "journal_extension/results/agnews_matched_ablation_ustc"
    target.mkdir(parents=True, exist_ok=True)
    backup = args.backup_dir.resolve()
    backup.mkdir(parents=True, exist_ok=True)
    published, counts, complete = [], {}, False
    for phase in ("probes", "formal"):
        identifier = "ustc_agnews_matched_probes_v2_20260929" if phase == "probes" \
            else "ustc_agnews_matched_formal_20260929"
        summary = REMOTE + "/results/" + identifier + "/summary"
        existence = subprocess.run([*ssh, "test -f " + shlex.quote(summary + "/summary.json")],
                                   stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        if existence.returncode:
            continue
        with tempfile.TemporaryDirectory(prefix="incoming-", dir=backup) as temporary:
            incoming = Path(temporary)
            for name in ("summary.json", "curves.csv", "REPORT.md"):
                run([*scp, "zzkevin@172.16.50.129:" + summary + "/" + name, str(incoming / name)],
                    stdout=subprocess.DEVNULL)
            document = json.loads((incoming / "summary.json").read_text())
            expected = 5 if phase == "probes" else 18
            rounds = 5 if phase == "probes" else 50
            if document["runs_expected"] != expected or document["protocol"]["rounds"] != rounds:
                raise ValueError("unexpected experiment protocol")
            rows = list(csv.DictReader(io.StringIO((incoming / "curves.csv").read_text())))
            if len(rows) != document["runs_completed"] * (rounds + 1):
                raise ValueError("summary/curves snapshot differs; retry later")
            if len({(r["arm"], r["seed"], r["round"]) for r in rows}) != len(rows):
                raise ValueError("duplicate trajectory rows")
            if len(document["runs"]) != document["runs_completed"]:
                raise ValueError("run count mismatch")
            allowed = {(c["arm"], c["seed"]) for c in document["protocol"]["cases"]}
            if not all((r["arm"], r["seed"]) in allowed for r in document["runs"]):
                raise ValueError("unplanned run in summary")
            for record in document["runs"]:
                archive = record["archive"]
                name, checksum = archive["name"], archive["sha256"]
                if not name.startswith(identifier + "_") or Path(name).name != name or len(checksum) != 64:
                    raise ValueError("invalid archive provenance")
                local = backup / name
                if not local.exists() or digest(local) != checksum:
                    pending = incoming / name
                    run([*scp, "zzkevin@172.16.50.129:" + REMOTE + "/backups/" + name, str(pending)],
                        stdout=subprocess.DEVNULL)
                    if digest(pending) != checksum:
                        raise ValueError("archive checksum mismatch")
                    pending.replace(local)
                member = "%s/seed_%d/metrics.json" % (record["arm"], record["seed"])
                with tarfile.open(local, "r:gz") as handle:
                    raw = handle.extractfile(member).read()
                import hashlib
                if hashlib.sha256(raw).hexdigest() != record["metrics_sha256"]:
                    raise ValueError("raw metrics hash differs from summary")
                metrics = json.loads(raw)
                if metrics["status"] != "complete" or not all(metrics["checks"].values()):
                    raise ValueError("raw run validation failed")
                if [r["round"] for r in metrics["evaluations"]] != list(range(-1, rounds)):
                    raise ValueError("raw trajectory incomplete")
                if abs(metrics["final"]["acc"] - record["final_accuracy"]) > 1e-12:
                    raise ValueError("raw final accuracy mismatch")
                (backup / (name + ".sha256")).write_text(checksum + "  " + name + "\n")
            destination = backup / phase
            destination.mkdir(exist_ok=True)
            for name in ("summary.json", "curves.csv", "REPORT.md"):
                shutil.copy2(incoming / name, destination / name)
                output = target / (phase + "_" + name)
                if args.publish:
                    shutil.copy2(incoming / name, output)
                    published.append(str(output.relative_to(REPO)))
            if phase == "probes":
                selection = summary + "/baseline_selection.json"
                exists = subprocess.run([*ssh, "test -f " + shlex.quote(selection)],
                                        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
                if exists.returncode == 0:
                    run([*scp, "zzkevin@172.16.50.129:" + selection, str(incoming / "baseline_selection.json")],
                        stdout=subprocess.DEVNULL)
                    shutil.copy2(incoming / "baseline_selection.json", destination / "baseline_selection.json")
                    if args.publish:
                        output = target / "baseline_selection.json"
                        shutil.copy2(incoming / "baseline_selection.json", output)
                        published.append(str(output.relative_to(REPO)))
            counts[phase] = document["runs_completed"]
            complete = phase == "formal" and document["status"] == "complete"
    if args.publish and published:
        branch = run(["git", "branch", "--show-current"], cwd=REPO,
                     capture_output=True, text=True).stdout.strip()
        if branch != "HeimdaLLM+":
            raise ValueError("publishing requires HeimdaLLM+ branch")
        run(["git", "add", "--", *published], cwd=REPO)
        changed = subprocess.run(["git", "diff", "--cached", "--quiet", "--", *published], cwd=REPO)
        if changed.returncode == 1:
            run(["git", "commit", "--only", "-m", "Save matched Non-DP progress (probes %d/5, formal %d/18)" %
                 (counts.get("probes", 0), counts.get("formal", 0)), "--", *published], cwd=REPO)
        elif changed.returncode:
            raise RuntimeError("cannot inspect staged changes")
        run(["git", "push", "origin", "HeimdaLLM+"], cwd=REPO)
    print("Verified matched backups: " + json.dumps(counts), flush=True)
    return complete


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--control-socket", default="/private/tmp/heimdallm-ustc-control")
    parser.add_argument("--backup-dir", type=Path,
                        default=REPO.parent / "local_backups/agnews_matched_ablation_ustc_20260929")
    parser.add_argument("--publish", action="store_true")
    parser.add_argument("--watch", action="store_true")
    parser.add_argument("--interval", type=int, default=120)
    args = parser.parse_args()
    if args.interval < 30:
        parser.error("interval must be at least 30s")
    while True:
        try:
            complete = synchronize(args)
        except (subprocess.CalledProcessError, ValueError, OSError) as error:
            if not args.watch:
                raise
            print("Sync pending; retrying: " + str(error), flush=True)
            complete = False
        if complete or not args.watch:
            return
        time.sleep(args.interval)


if __name__ == "__main__":
    main()
