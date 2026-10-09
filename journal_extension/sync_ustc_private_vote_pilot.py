#!/usr/bin/env python3
"""Back up the guarded private-vote pilot; keep raw archives outside Git."""

import argparse
import hashlib
import json
import subprocess
import tempfile
import time
from pathlib import Path


RUN_ID = "ustc_agnews_private_vote_pilot_20261009"
ARMS = ("private_vote_synthetic", "same_source_real")
REMOTE = "zzkevin@172.16.50.129"
BASE = "/home/zzkevin/heimdallm-journal-20260928"


def sha256(path):
    result = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            result.update(chunk)
    return result.hexdigest()


def fetch(remote_path, local_path, options):
    with tempfile.NamedTemporaryFile(dir=local_path.parent, prefix="incoming-", delete=False) as temp:
        incoming = Path(temp.name)
    try:
        subprocess.run(["scp", *options, "-P", "1010", REMOTE + ":" + remote_path,
                        str(incoming)], check=True, stdout=subprocess.DEVNULL,
                       stderr=subprocess.DEVNULL)
        incoming.replace(local_path)
    finally:
        incoming.unlink(missing_ok=True)


def once(backup, options):
    backup.mkdir(parents=True, exist_ok=True)
    root = BASE + "/results/" + RUN_ID
    summary_dir = backup / "downstream_summary"
    summary_dir.mkdir(exist_ok=True)
    for name in ("summary.json", "curves.csv", "REPORT.md"):
        try:
            fetch(root + "/summary/" + name, summary_dir / name, options)
        except subprocess.CalledProcessError:
            if name == "summary.json":
                raise
    for name, remote in (("protocol.json", root + "/protocol.json"),
                         ("quality.json", root + "/quality.json"),
                         ("generator_manifest.json", BASE + "/private_staging/" + RUN_ID + "/generated/manifest.json")):
        try:
            fetch(remote, backup / name, options)
        except subprocess.CalledProcessError:
            pass
    summary = json.loads((summary_dir / "summary.json").read_text())
    if summary["protocol"]["run_id"] != RUN_ID or summary["runs_expected"] != 2:
        raise ValueError("unexpected pilot summary")
    verified = 0
    for arm in ARMS:
        name = "%s_%s.tar.gz" % (RUN_ID, arm)
        sidecar = backup / (name + ".sha256")
        try:
            fetch(BASE + "/backups/" + name + ".sha256", sidecar, options)
        except subprocess.CalledProcessError:
            continue
        expected = sidecar.read_text().split()[0]
        local = backup / name
        if not local.exists() or sha256(local) != expected:
            fetch(BASE + "/backups/" + name, local, options)
        if sha256(local) != expected:
            raise ValueError("archive SHA-256 mismatch: " + name)
        verified += 1
    complete = summary["status"] == "complete" and verified == 2
    print("Verified %d/2 pilot archives; finished=%s" % (verified, complete), flush=True)
    return complete


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--backup-dir", type=Path, required=True)
    parser.add_argument("--control-socket", default="/private/tmp/heimdallm-ustc-control")
    parser.add_argument("--watch", action="store_true")
    parser.add_argument("--interval", type=int, default=45)
    args = parser.parse_args()
    options = ["-o", "BatchMode=yes", "-o", "ConnectTimeout=10",
               "-o", "ControlPath=" + args.control_socket]
    while True:
        try:
            finished = once(args.backup_dir.resolve(), options)
            if finished or not args.watch:
                return
        except Exception as error:
            print("Sync retry: %s" % error, flush=True)
            if not args.watch:
                raise
        time.sleep(args.interval)


if __name__ == "__main__":
    main()
