#!/usr/bin/env python3
"""Sample co-located Stage-2 resource use without touching training processes."""

import argparse
import csv
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path

import psutil


def sample():
    output = subprocess.check_output(
        ["nvidia-smi", "--query-gpu=memory.used,utilization.gpu",
         "--format=csv,noheader,nounits"], text=True, timeout=15)
    gpu_mem, gpu_util = [int(x.strip()) for x in output.splitlines()[0].split(",")]
    rss = []
    for process in psutil.process_iter(["name", "memory_info"]):
        try:
            name = process.info["name"] or ""
            if name.startswith("FedNLP-agnews"):
                rss.append(process.info["memory_info"].rss)
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            continue
    return gpu_mem, gpu_util, len(rss), sum(rss), max(rss, default=0)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--stop-file", type=Path, required=True)
    parser.add_argument("--interval", type=float, default=5.0)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    args.output.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
    with args.output.open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(("utc_time", "gpu0_memory_used_mib", "gpu0_utilization_percent",
                         "fednlp_process_count", "fednlp_rss_sum_bytes", "fednlp_rss_max_bytes"))
        handle.flush()
        while not args.stop_file.exists():
            stamp = datetime.now(timezone.utc).isoformat()
            try:
                writer.writerow((stamp,) + sample())
            except Exception:
                writer.writerow((stamp, "NA", "NA", "NA", "NA", "NA"))
            handle.flush()
            time.sleep(args.interval)


if __name__ == "__main__":
    main()
