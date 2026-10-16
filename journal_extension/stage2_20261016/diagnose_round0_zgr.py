#!/usr/bin/env python3
"""Measure production BP/GGD directions on one hashed round-0 checkpoint.

This is a co-located, deterministic diagnostic replay, not a historical
checkpoint from Stage-1 and not a downstream FL endpoint.
"""

import argparse
import copy
import hashlib
import json
import os
import shlex
import time
from pathlib import Path

import numpy as np
import psutil
import torch

from data_manager.base_data_manager import BaseDataManager
from data_manager.text_classification_data_manager import TextClassificationDataManager
from data_preprocessing.text_classification_preprocessor import TLMPreprocessor
from experiments.distributed.transformer_exps.initializer import (
    add_federated_args, create_model, set_seed,
)
from forward_training.tc_transformer_trainer_distribute import ForwardTextClassificationTrainer
from model.transformer.model_args import ClassificationArgs
from FedML.fedml_api.distributed.fedsgd.FedSgdCloud import FedSGDCloud


STAGE1 = Path("/mnt/heimdallm-stage1-20261015")
STAGE2 = Path("/mnt/heimdallm-stage2-20261016")
OUT = STAGE2 / "restricted_diagnostics/round0_zgr"
ARMS = {
    "PUB_RANDOM": STAGE1 / "reused/public_random_selection",
    "PUB_NLI": STAGE1 / "guidance/public_nli_selection",
    "PRIVATE_ZERO": STAGE1 / "reused/private_vote_synthetic",
    "PRIVATE_DP_E4_D01": STAGE1 / "guidance/private_dp_e4/D01",
    "REAL_MATCHED": STAGE1 / "reused/same_source_real",
    "SHUFFLED_DP_LABEL": STAGE2 / "private_postprocess/shuffled_dp_label/seed_57/packed",
}


def sha(path):
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def model_hash(model):
    h = hashlib.sha256()
    count = 0
    for name, p in model.named_parameters():
        if p.requires_grad:
            h.update(name.encode())
            h.update(p.detach().cpu().contiguous().numpy().tobytes())
            count += p.numel()
    return h.hexdigest(), count


def paired_stats(left, ref):
    a = torch.cat([x.flatten() for x in left]).double()
    b = torch.cat([x.flatten() for x in ref]).double()
    an, bn = a.norm().item(), b.norm().item()
    if an == 0 or bn == 0:
        raise RuntimeError("zero cloud/reference gradient")
    cos = torch.dot(a, b).item() / (an * bn)
    threshold = max(1e-12, float(b.abs().max()) * 1e-6)
    mask = (a.abs() > threshold) & (b.abs() > threshold)
    return {"cosine": cos,
            "relative_l2_error": ((a - b).norm() / b.norm()).item(),
            "sign_agreement": (torch.sign(a[mask]) == torch.sign(b[mask])).double().mean().item()
                              if mask.any() else None,
            "sign_coordinates": int(mask.sum()),
            "reference_threshold": threshold,
            "zgr_rank1_normalized_energy_capture": cos * cos}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--preflight-only", action="store_true")
    cli = parser.parse_args()
    if OUT.exists():
        raise FileExistsError(OUT)
    if torch.cuda.device_count() != 1:
        raise RuntimeError("diagnostic requires sole A6000")
    gpu = subprocess_gpu()
    if gpu["free_mib"] < 14000 or gpu["utilization_percent"] > 20:
        raise RuntimeError("GPU busy: " + json.dumps(gpu))
    if cli.preflight_only:
        print(json.dumps({"gpu": gpu, "arms": list(ARMS)}))
        return
    for root in ARMS.values():
        if not (root / "cloud_data.h5").exists() or not (root / "cloud_partition.h5").exists():
            raise FileNotFoundError(root)
    command = shlex.split((STAGE1 / "runs/PRIVATE_DP_E4/seed_57/train.command.txt").read_text())
    args = add_federated_args(argparse.ArgumentParser()).parse_args(
        command[command.index("fedavg_main_tc") + 1:])
    args.model_name = "/root/heimdallm-work/models/distilbert-base-uncased"
    args.data_file_path = "/root/heimdallm-work/data/fednlp_data/data_files/agnews_data.h5"
    args.manual_seed = 57
    set_seed(57)
    device = torch.device("cuda:0")
    attributes = BaseDataManager.load_attributes(args.data_file_path)
    model_args = ClassificationArgs()
    model_args.model_name = args.model_name
    model_args.model_type = args.model_type
    model_args.load(model_args.model_name)
    model_args.num_labels = len(attributes["label_vocab"])
    model_args.update_from_dict({
        "fl_algorithm": args.fl_algorithm, "freeze_layers": args.freeze_layers,
        "epochs": args.epochs, "learning_rate": args.learning_rate,
        "gradient_accumulation_steps": args.gradient_accumulation_steps,
        "do_lower_case": args.do_lower_case, "manual_seed": args.manual_seed,
        "reprocess_input_data": False, "overwrite_output_dir": True,
        "max_seq_length": args.max_seq_length,
        "train_batch_size": args.train_batch_size,
        "eval_batch_size": args.eval_batch_size,
        "evaluate_during_training": False,
        "evaluate_during_training_steps": args.evaluate_during_training_steps,
        "fp16": args.fp16, "data_file_path": args.data_file_path,
        "partition_file_path": args.partition_file_path,
        "partition_method": args.partition_method, "dataset": args.dataset,
        "output_dir": str(STAGE2 / "restricted_diagnostics/round0_cache"),
        "cache_dir": str(STAGE2 / "restricted_diagnostics/round0_cache"),
        "is_debug_mode": args.is_debug_mode, "fedprox_mu": args.fedprox_mu,
        "use_adapter": args.use_adapter, "comm_round": args.comm_round,
        "peft_method": args.peft_method, "var_control": args.var_control,
        "perturbation_sampling": args.perturbation_sampling,
        "alpha": args.alpha,
    })
    model_args.config["num_labels"] = model_args.num_labels
    config, initial, tokenizer = create_model(model_args, formulation="classification")
    # Production trainer removes DistilBERT's pre_classifier (590,592 params).
    # Hash and check the actual trained architecture, not create_model's output.
    initialization_trainer = ForwardTextClassificationTrainer(
        model_args, torch.device("cpu"), initial, None, None)
    initial = initialization_trainer.model
    del initialization_trainer
    init_hash, trainable_count = model_hash(initial)
    if trainable_count != 450340:
        raise RuntimeError("unexpected trainable parameter count: %d" % trainable_count)
    OUT.mkdir(mode=0o700, parents=True)
    checkpoint = {name: p.detach().cpu().clone() for name, p in initial.named_parameters()
                  if p.requires_grad}
    torch.save(checkpoint, OUT / "round0_trainable.pt")
    if model_hash(initial)[0] != init_hash:
        raise RuntimeError("checkpoint mutated during save")
    gradients = {}
    production_directions = {}
    component_costs = {}
    provenance = {}
    trainable_names = [name for name, p in initial.named_parameters() if p.requires_grad]
    for arm, root in ARMS.items():
        set_seed(57)  # pair dropout and data-loader random streams across arms
        arm_model = copy.deepcopy(initial)
        if model_hash(arm_model)[0] != init_hash:
            raise RuntimeError("arm model is not identical to shared initialization")
        args.data_file_path = str(root / "cloud_data.h5")
        args.partition_file_path = str(root / "cloud_partition.h5")
        args.partition_method = "synthetic_cloud"
        args.cloud_client_ids = "0"
        args.cloud_client_count = None
        args.output_dir = str(OUT / "cache" / arm)
        model_args.update_from_dict({"data_file_path": args.data_file_path,
                                     "partition_file_path": args.partition_file_path,
                                     "partition_method": args.partition_method,
                                     "dataset": "agnews", "output_dir": args.output_dir,
                                     "cache_dir": str(OUT / "cache" / arm)})
        preprocessor = TLMPreprocessor(args=model_args,
                                       label_vocab=attributes["label_vocab"],
                                       tokenizer=tokenizer)
        dm = TextClassificationDataManager(args, model_args, preprocessor, 0,
                                            args.client_num_per_round)
        count, train_dl, _, _, _, _, _ = dm.load_cloud_server_client_data(process_id=0)
        if count != 64:
            raise RuntimeError("guidance count mismatch for " + arm)
        trainer = ForwardTextClassificationTrainer(model_args, device, arm_model, None, None)
        if model_hash(trainer.model)[0] != init_hash:
            raise RuntimeError("production trainer changed the common checkpoint")
        trainer.train_dl = train_dl
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.synchronize()
        rss_before = psutil.Process().memory_info().rss
        start = time.perf_counter()
        trainer.train_model_bp(device=device)
        torch.cuda.synchronize()
        bp_seconds = time.perf_counter() - start
        gradient = [g.detach().cpu().float().clone() for p, g in
                    zip(trainer.model.parameters(), trainer.grad_bp) if p.requires_grad]
        if len(gradient) != len(trainable_names) or not all(torch.isfinite(g).all() for g in gradient):
            raise RuntimeError("gradient missing/nonfinite for " + arm)
        gradients[arm] = gradient
        cloud = FedSGDCloud(None, 0, device, args, None)
        cloud.model_pool = [[g.detach() for g in trainer.grad_bp]]
        torch.cuda.synchronize()
        ggd_start = time.perf_counter()
        direction = cloud.create_perturbation()
        torch.cuda.synchronize()
        ggd_seconds = time.perf_counter() - ggd_start
        if not direction:
            raise RuntimeError("GGD missing for " + arm)
        production_directions[arm] = [g.detach().cpu().float().clone() for p, g in
                                     zip(trainer.model.parameters(), direction) if p.requires_grad]
        component_costs[arm] = {"T_cloudBP_seconds": bp_seconds,
                                "T_GGD_compute_seconds": ggd_seconds,
                                "cpu_rss_before_bytes": rss_before,
                                "cpu_rss_after_bytes": psutil.Process().memory_info().rss,
                                "gpu_peak_allocated_bytes": torch.cuda.max_memory_allocated(),
                                "gpu_peak_reserved_bytes": torch.cuda.max_memory_reserved(),
                                "device": str(device), "co_located": True}
        provenance[arm] = {"cloud_h5_sha256": sha(root / "cloud_data.h5"),
                           "cloud_partition_sha256": sha(root / "cloud_partition.h5"),
                           "records": count, "checkpoint_sha256": init_hash}
        del dm, preprocessor, trainer, cloud, direction, arm_model, train_dl
        torch.cuda.empty_cache()
        print("diagnosed " + arm, flush=True)
    ref = gradients["REAL_MATCHED"]
    overall = {}
    layers = {}
    for arm, grad in gradients.items():
        overall[arm] = paired_stats(grad, ref)
        measured_v = paired_stats(production_directions[arm], ref)
        bp_energy = overall[arm]["zgr_rank1_normalized_energy_capture"]
        v_energy = measured_v["zgr_rank1_normalized_energy_capture"]
        if abs(bp_energy - v_energy) > 1e-5:
            raise RuntimeError("production GGD disagrees with rank-1 BP span")
        overall[arm]["zgr_rank1_normalized_energy_capture"] = v_energy
        overall[arm]["production_direction_reference_cosine"] = measured_v["cosine"]
        overall[arm]["production_direction_l2"] = torch.cat(
            [g.flatten() for g in production_directions[arm]]).norm().item()
        overall[arm]["gradient_l2"] = torch.cat([g.flatten() for g in grad]).norm().item()
        groups = {}
        for name, g, r in zip(trainable_names, grad, ref):
            key = name.split(".")[0]
            groups.setdefault(key, ([], []))[0].append(g)
            groups[key][1].append(r)
        layers[arm] = {key: paired_stats(parts[0], parts[1]) for key, parts in groups.items()}
    report = {"schema_version": 1, "source_commit": "ad325aa320789c4f59f911898033844d5b19ccec",
              "round": 0, "checkpoint_kind": "deterministically_recreated_preupdate_trainable_state",
              "checkpoint_sha256": init_hash, "checkpoint_file_sha256": sha(OUT / "round0_trainable.pt"),
              "trainable_count": trainable_count, "model_name": args.model_name,
              "initialization": "create_model_then_production_trainer_pre_classifier_removal",
              "energy_capture_measurement": "actual_create_perturbation_output_checked_against_BP_span",
              "dropout_seed": 57, "reference": "REAL_MATCHED_restricted_real_guidance",
              "production_bp_operator": "ForwardTextClassificationTrainer.train_model_bp",
              "production_ggd_operator": "FedSGDCloud.create_perturbation_pool_size_1",
              "historical_round0_ggd_bitwise_reproduction": False,
              "overall": overall, "layer_groups": layers,
              "component_costs": component_costs, "provenance": provenance,
              "privacy": "aggregate metrics only; no raw gradients or private records in JSON"}
    (OUT / "diagnostics.json").write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"output": str(OUT / "diagnostics.json"),
                      "checkpoint_sha256": init_hash, "arms": list(overall)}))


def subprocess_gpu():
    import subprocess
    output = subprocess.check_output(
        ["nvidia-smi", "--query-gpu=name,memory.free,utilization.gpu",
         "--format=csv,noheader,nounits"], text=True, timeout=15).splitlines()[0]
    name, free, util = [x.strip() for x in output.split(",")]
    if "A6000" not in name:
        raise RuntimeError("unexpected GPU: " + name)
    return {"name": name, "free_mib": int(free), "utilization_percent": int(util)}


if __name__ == "__main__":
    main()
