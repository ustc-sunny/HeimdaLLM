#!/usr/bin/env python3
"""Back up the Qwen public-selection control; keep raw text outside Git."""

import argparse
import hashlib
import json
import subprocess
import tempfile
import time
from pathlib import Path


RUN_ID = "ustc_agnews_qwen_public_control_20261013"
REMOTE = "zzkevin@172.16.50.129"
BASE = "/home/zzkevin/heimdallm-journal-20260928"


def sha256(path):
    result = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            result.update(chunk)
    return result.hexdigest()


def fetch(remote, local, options):
    local.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=local.parent, prefix="incoming-", delete=False) as temp:
        incoming = Path(temp.name)
    try:
        subprocess.run(["scp", *options, "-P", "1010", REMOTE + ":" + remote,
                        str(incoming)], check=True, stdout=subprocess.DEVNULL,
                       stderr=subprocess.DEVNULL)
        incoming.replace(local)
    finally:
        incoming.unlink(missing_ok=True)


def once(backup, options):
    root = BASE + "/results/" + RUN_ID
    for name in ("state.json", "protocol.json", "quality.json"):
        try:
            fetch(root + "/" + name, backup / name, options)
        except subprocess.CalledProcessError:
            if name == "state.json":
                raise
    if (backup / "state.json").is_file() and json.loads((backup / "state.json").read_text())["status"] == "failed":
        raise RuntimeError("server run failed; inspect logs")
    for name in ("manifest.json",):
        for stage in ("public_pool", "public_selection"):
            try:
                fetch(BASE + "/private_staging/" + RUN_ID + "/" + stage + "/" + name,
                      backup / stage / name, options)
            except subprocess.CalledProcessError:
                pass
    for name in ("summary.json", "curves.csv", "REPORT.md"):
        try:
            fetch(root + "/summary/" + name, backup / "summary" / name, options)
        except subprocess.CalledProcessError:
            pass
    name = RUN_ID + ".tar.gz"
    sidecar = backup / (name + ".sha256")
    try:
        fetch(BASE + "/backups/" + name + ".sha256", sidecar, options)
    except subprocess.CalledProcessError:
        print("Archive not yet ready", flush=True)
        return False
    expected = sidecar.read_text().split()[0]
    archive = backup / name
    if not archive.is_file() or sha256(archive) != expected:
        fetch(BASE + "/backups/" + name, archive, options)
    if sha256(archive) != expected:
        raise ValueError("raw archive SHA-256 mismatch")
    summary = json.loads((backup / "summary/summary.json").read_text())
    state = json.loads((backup / "state.json").read_text())
    complete = summary["status"] == "complete" and summary["runs_completed"] == 3 and state["status"] == "complete"
    print("Archive verified; finished=%s" % complete, flush=True)
    return complete


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--backup-dir", required=True, type=Path)
    parser.add_argument("--control-socket", default="/private/tmp/heimdallm-ustc-control-20261013b")
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
