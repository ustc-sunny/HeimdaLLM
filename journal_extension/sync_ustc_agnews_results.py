#!/usr/bin/env python3
"""Back up completed USTC conditions locally and optionally publish curated metrics.

Uses an already authenticated SSH ControlMaster socket; never stores a password.
The watcher requires the local computer to stay awake and its VPN to stay connected.
"""
import argparse
import csv
import hashlib
import json
import shlex
import shutil
import subprocess
import tempfile
import time
from pathlib import Path


def run(command, **kwargs):
    return subprocess.run(command, check=True, **kwargs)


def digest(path):
    value = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def synchronize(args):
    ssh_options = ["-o", "BatchMode=yes", "-o", "ConnectTimeout=15",
                   "-o", "ServerAliveInterval=15", "-o", "ServerAliveCountMax=2",
                   "-o", "ControlPath=" + args.control_socket]
    ssh = ["ssh", *ssh_options, "-p", "1010", "zzkevin@172.16.50.129"]
    scp = ["scp", *ssh_options, "-P", "1010"]
    remote = "/home/zzkevin/heimdallm-journal-20260928"
    summary = remote + "/results/agnews_dp_v1_ustc_summary"
    run([*ssh, "test -f " + shlex.quote(summary + "/summary.json")],
        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    backup = args.backup_dir.resolve()
    backup.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="incoming-", dir=backup) as temporary:
        incoming = Path(temporary)
        for name in ("summary.json", "curves.csv", "REPORT.md"):
            run([*scp, "zzkevin@172.16.50.129:" + summary + "/" + name, str(incoming / name)],
                stdout=subprocess.DEVNULL)
        document = json.loads((incoming / "summary.json").read_text())
        if document["runs_expected"] != 12 or document["runs_completed"] not in range(13):
            raise ValueError("Unexpected protocol in remote summary")
        with (incoming / "curves.csv").open(newline="") as handle:
            rows = list(csv.DictReader(handle))
        if len(rows) != document["runs_completed"] * 51:
            raise ValueError("Summary and trajectory snapshot differ; retry later")
        completed = {epsilon for epsilon, value in document["aggregate"].items()
                     if value["final_accuracy"]["n"] == 3}
        for epsilon in sorted(completed):
            if epsilon not in {"1", "2", "4", "8"}:
                raise ValueError("Unexpected epsilon")
            archive_name = "ustc_agnews_dp_eps%s_20260928.tar.gz" % epsilon
            checksum_path = incoming / (archive_name + ".sha256")
            run([*scp, "zzkevin@172.16.50.129:" + remote + "/backups/" + checksum_path.name,
                 str(checksum_path)], stdout=subprocess.DEVNULL)
            expected = checksum_path.read_text().split()[0]
            if len(expected) != 64:
                raise ValueError("Invalid archive SHA256")
            archive = backup / archive_name
            if not archive.exists() or digest(archive) != expected:
                pending = incoming / archive_name
                run([*scp, "zzkevin@172.16.50.129:" + remote + "/backups/" + archive_name,
                     str(pending)], stdout=subprocess.DEVNULL)
                if digest(pending) != expected:
                    raise ValueError("Archive checksum mismatch: " + archive_name)
                pending.replace(archive)
            (backup / checksum_path.name).write_text(expected + "  " + archive_name + "\n")
        snapshot = backup / "summary"
        snapshot.mkdir(exist_ok=True)
        for name in ("summary.json", "curves.csv", "REPORT.md"):
            shutil.copy2(incoming / name, snapshot / name)
        if args.publish:
            repo = Path(__file__).resolve().parents[1]
            branch = run(["git", "branch", "--show-current"], cwd=repo,
                         capture_output=True, text=True).stdout.strip()
            if branch != "HeimdaLLM+":
                raise ValueError("Publishing requires the HeimdaLLM+ branch")
            target = repo / "journal_extension/results/agnews_dp_v1_ustc"
            target.mkdir(parents=True, exist_ok=True)
            paths = []
            for name in ("summary.json", "curves.csv", "REPORT.md"):
                shutil.copy2(incoming / name, target / name)
                paths.append(str((target / name).relative_to(repo)))
            run(["git", "add", "--", *paths], cwd=repo)
            changed = subprocess.run(["git", "diff", "--cached", "--quiet", "--", *paths], cwd=repo)
            if changed.returncode == 1:
                run(["git", "commit", "--only", "-m",
                     "Save USTC AG News DP progress (%d/12 runs)" % document["runs_completed"],
                     "--", *paths], cwd=repo)
            elif changed.returncode:
                raise RuntimeError("Unable to inspect staged result changes")
            run(["git", "push", "origin", "HeimdaLLM+"], cwd=repo)
        print("Verified local backup: %d/12 runs; %d complete archives" %
              (document["runs_completed"], len(completed)), flush=True)
        return document["status"] == "complete"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--control-socket", default="/private/tmp/heimdallm-ustc-control")
    parser.add_argument("--backup-dir", type=Path,
                        default=Path(__file__).resolve().parents[2] / "local_backups/agnews_dp_v1_ustc_20260928")
    parser.add_argument("--publish", action="store_true")
    parser.add_argument("--watch", action="store_true")
    parser.add_argument("--interval", type=int, default=120)
    args = parser.parse_args()
    if args.interval < 30:
        parser.error("Polling interval must be at least 30 seconds")
    while True:
        try:
            complete = synchronize(args)
        except (subprocess.CalledProcessError, ValueError, OSError) as error:
            if not args.watch:
                raise
            print("Sync pending, retrying: %s" % error, flush=True)
            complete = False
        if complete or not args.watch:
            return
        time.sleep(args.interval)


if __name__ == "__main__":
    main()
