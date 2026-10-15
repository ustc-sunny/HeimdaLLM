#!/usr/bin/env python3
"""Read-only readiness check for a new single-GPU AG News Stage-1 host.

This does not install software, read client records, or start a training job.
"""

import argparse
import hashlib
import json
import shutil
import subprocess
from pathlib import Path


EXPECTED = {
    "agnews_data": "1474dd264d535b50ad2a7b1cf714bf11e182c2fe5b07f5e982feb31a9b114616",
    "agnews_source_partition": "fd6196bc2518457856365ec8f8207b1eb4b278d165a829320fc5b9fdd0a67f12",
    "fixed_partition": "e16e8551a87ba702cc23b4d56d57b9762b77c4fea8bf341b96d5759be7996a6c",
    "public_pool": "7e54c05ce2dc94d754b178218963dedfceb781c491b97ed1d9db5ab9e7d7c54f",
    "public_random": "79773d11e68e9bd128b1852b6eb942debd23a3add32eb1f329685f43c65fcee3",
    "public_nli": "52810c8290a47533048d82a33bc92279c50d4df77f3628a1539031ba2905194a",
}
MIN_FREE_MIB = 14000


def digest(path):
    result = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            result.update(block)
    return result.hexdigest()


def file_check(path, expected):
    if not path.is_file():
        return {"status": "missing", "path": str(path), "expected_sha256": expected}
    actual = digest(path)
    return {"status": "pass" if actual == expected else "hash_mismatch",
            "path": str(path), "expected_sha256": expected, "actual_sha256": actual}


def model_check(path):
    files = {"config": path / "config.json",
             "tokenizer": path / "tokenizer.json",
             "tokenizer_alt": path / "vocab.txt",
             "weights": path / "pytorch_model.bin",
             "weights_alt": path / "model.safetensors"}
    complete = (files["config"].is_file() and
                (files["tokenizer"].is_file() or files["tokenizer_alt"].is_file()) and
                (files["weights"].is_file() or files["weights_alt"].is_file()))
    return {"status": "pass" if complete else "missing_files", "path": str(path)}


def python_check(path, modules):
    if not path.is_file():
        return {"status": "missing", "path": str(path)}
    script = "import importlib, json; names=%r; " % modules + \
             "print(json.dumps({n: importlib.import_module(n).__version__ if hasattr(importlib.import_module(n), '__version__') else 'installed' for n in names}))"
    try:
        process = subprocess.run([str(path), "-c", script], text=True,
                                 capture_output=True, timeout=45, check=False)
    except (OSError, subprocess.TimeoutExpired) as exc:
        return {"status": "failed", "path": str(path), "error": type(exc).__name__}
    if process.returncode:
        return {"status": "failed", "path": str(path),
                "error": process.stderr.strip().splitlines()[-1] if process.stderr.strip() else "nonzero_exit"}
    try:
        versions = json.loads(process.stdout)
    except json.JSONDecodeError:
        return {"status": "failed", "path": str(path), "error": "unexpected_import_output"}
    return {"status": "pass", "path": str(path), "versions": versions}


def gpu_check(index):
    if not shutil.which("nvidia-smi"):
        return {"status": "missing_nvidia_smi", "index": index}
    try:
        text = subprocess.check_output([
            "nvidia-smi", "--query-gpu=index,name,memory.total,memory.free,utilization.gpu",
            "--format=csv,noheader,nounits"], text=True, timeout=15)
        rows = [line.split(",") for line in text.splitlines()]
        row = next((x for x in rows if int(x[0].strip()) == index), None)
    except (OSError, subprocess.CalledProcessError, subprocess.TimeoutExpired, ValueError) as exc:
        return {"status": "failed", "index": index, "error": type(exc).__name__}
    if row is None:
        return {"status": "missing_gpu", "index": index}
    name = row[1].strip()
    total, free, utilization = (int(x.strip()) for x in row[2:5])
    return {"status": "pass" if free >= MIN_FREE_MIB and utilization <= 20 else "busy",
            "index": index, "name": name, "total_mib": total, "free_mib": free,
            "utilization_percent": utilization, "minimum_free_mib": MIN_FREE_MIB}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--work-root", type=Path, required=True,
                        help="Persistent-data root containing data/, models/, and envs/")
    parser.add_argument("--fixed-partition", type=Path,
                        help="Restricted fixed_partition.h5 restored from the checked local archive")
    parser.add_argument("--gpu-index", type=int, default=0)
    parser.add_argument("--kdd-python", type=Path,
                        help="Python interpreter for the original FedFwd downstream code")
    parser.add_argument("--ton-python", type=Path,
                        help="Separate interpreter for the public embedding/private selector tools")
    parser.add_argument("--json-out", type=Path,
                        help="Optional new report path; refuses to overwrite an existing file")
    args = parser.parse_args()
    repo = Path(__file__).resolve().parent.parent
    work = args.work_root.expanduser().resolve()
    public = repo / "journal_extension/frozen_agnews_qwen_pool_20261013"
    data = work / "data/fednlp_data"
    checks = {
        "public_pool": file_check(public / "pool/candidates.jsonl", EXPECTED["public_pool"]),
        "public_random": file_check(public / "random/synthetic.jsonl", EXPECTED["public_random"]),
        "public_nli": file_check(public / "nli/synthetic.jsonl", EXPECTED["public_nli"]),
        "agnews_data": file_check(data / "data_files/agnews_data.h5", EXPECTED["agnews_data"]),
        "agnews_source_partition": file_check(data / "partition_files/agnews_partition.h5",
                                              EXPECTED["agnews_source_partition"]),
        "fixed_partition": file_check(args.fixed_partition.expanduser().resolve()
                                      if args.fixed_partition else work / "protocol/fixed_partition.h5",
                                      EXPECTED["fixed_partition"]),
        "distilbert_model": model_check(work / "models/distilbert-base-uncased"),
        "minilm_model": model_check(work / "models/all-MiniLM-L6-v2"),
        "kdd_python": python_check(args.kdd_python or work / "envs/kdd/bin/python",
                                   ["torch", "transformers", "functorch", "mpi4py", "h5py"]),
        "ton_python": python_check(args.ton_python or work / "envs/ton/bin/python",
                                   ["torch", "transformers"]),
        "mpirun": {"status": "pass" if shutil.which("mpirun") else "missing"},
        "gpu": gpu_check(args.gpu_index),
    }
    report = {"schema_version": 1, "purpose": "single_gpu_stage1_e1_readiness",
              "work_root": str(work), "repository": str(repo), "checks": checks,
              "ready_for_e1": all(value["status"] == "pass" for value in checks.values()),
              "ready_for_e2": False,
              "e2_note": "DP private-vote selector still requires matching implementation and review"}
    content = json.dumps(report, indent=2, sort_keys=True) + "\n"
    if args.json_out:
        output = args.json_out.expanduser()
        with output.open("x") as handle:
            handle.write(content)
    print(content, end="")
    return 0 if report["ready_for_e1"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
