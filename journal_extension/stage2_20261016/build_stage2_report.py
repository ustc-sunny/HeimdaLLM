#!/usr/bin/env python3
"""Build aggregate-only Stage-2 evidence after all six matched controls finish."""

import argparse
import csv
import hashlib
import json
import math
import re
import statistics
from collections import defaultdict
from pathlib import Path


S1 = Path("/mnt/heimdallm-stage1-20261015")
S2 = Path("/mnt/heimdallm-stage2-20261016")
OUT = S2 / "report_public"
SEEDS = (57, 58, 59)
NEW_ARMS = ("NO_GUIDANCE", "SHUFFLED_DP_LABEL")
COMPARISONS = (("PRIVATE_DP_E4", "NO_GUIDANCE"),
               ("PRIVATE_DP_E4", "SHUFFLED_DP_LABEL"),
               ("PUB_NLI", "NO_GUIDANCE"),
               ("REAL_MATCHED", "NO_GUIDANCE"))


def sha(path):
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def read_csv(path):
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def write_csv(path, rows, fields):
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def metric_summary(metrics):
    evaluations = metrics["evaluations"]
    if (metrics["status"] != "complete" or len(evaluations) != 51 or
            [item["round"] for item in evaluations] != list(range(-1, 50)) or
            not all(metrics["checks"].values())):
        raise RuntimeError("endpoint does not have 51 complete dev evaluations")
    accuracy = [item["acc"] for item in evaluations]
    return {"final_accuracy_percent": 100 * accuracy[-1],
            "final_macro_f1": evaluations[-1]["macro_f1"],
            "final_loss": evaluations[-1]["eval_loss"],
            "late10_accuracy_percent": 10 * sum(accuracy[-10:]),
            "normalized_accuracy_auc": (0.5 * accuracy[0] +
                                        sum(accuracy[1:-1]) + 0.5 * accuracy[-1]) / 50}


def count_queries(log_path):
    content = log_path.read_text(errors="replace")
    updates = content.count("aggregate time cost")
    directions = sum(int(total) for total in
                     re.findall(r"\[ZGR\] finite differences: nonzero=\d+/(\d+)", content))
    if updates != 750 or directions != 1500:
        raise RuntimeError("wrong update/query budget in " + str(log_path))
    return updates, directions, 2 * directions


def fmt(value, digits=2):
    return ("%+." + str(digits) + "f") % value


def main():
    global OUT
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, default=OUT)
    OUT = parser.parse_args().output_dir
    if OUT.exists():
        raise FileExistsError(OUT)
    temp = OUT.with_name(OUT.name + ".tmp")
    if temp.exists():
        raise FileExistsError(temp)
    old_rows = read_csv(S1 / "report/metrics.csv")
    if len(old_rows) != 15:
        raise RuntimeError("Stage-1 baseline matrix is not 15/15")
    runs = {}
    manifest = []
    metric_rows = []
    curves = []
    all_rows = [(row["arm"], int(row["fl_seed"]), Path(row["run_path"]),
                 "stage1_reuse", row) for row in old_rows]
    all_rows += [(arm, seed, S2 / "runs" / arm / ("seed_%d" % seed),
                  "stage2_new", {}) for arm in NEW_ARMS for seed in SEEDS]
    for arm, seed, directory, origin, stage1_row in all_rows:
        key = (arm, seed)
        if key in runs:
            raise RuntimeError("duplicate endpoint")
        metrics_path = directory / "metrics.json"
        run_metrics = json.loads(metrics_path.read_text())
        summary = metric_summary(run_metrics)
        if origin == "stage2_new":
            spec = json.loads((directory / "run_spec.json").read_text())
            if spec["status"] != "complete" or spec["official_test_used"] or spec["seed"] != seed:
                raise RuntimeError("Stage-2 source/validation mismatch for " + str(key))
            updates, directions, queries = count_queries(directory / "train.log")
            wall_seconds = spec["wall_seconds"]
        else:
            if stage1_row["status"] != "complete":
                raise RuntimeError("Stage-1 reused endpoint is incomplete")
            updates = int(stage1_row["server_inner_updates_completed"])
            directions = int(stage1_row["client_fd_directions_reported"])
            queries = int(stage1_row["client_objective_queries_reported"])
            wall_seconds = stage1_row["training_wall_seconds"] or "NA"
        if (updates, directions, queries) != (750, 1500, 3000):
            raise RuntimeError("budget mismatch for " + str(key))
        runs[key] = {"metrics": run_metrics, "summary": summary, "origin": origin,
                     "path": directory, "wall_seconds": wall_seconds}
        manifest.append({"arm": arm, "fl_seed": seed, "status": "complete",
                         "origin": origin, "server_run_path": str(directory),
                         "metrics_sha256": sha(metrics_path),
                         "source_commit": "ad325aa320789c4f59f911898033844d5b19ccec",
                         "rounds": 50, "server_updates": updates,
                         "fd_directions": directions, "objective_queries": queries,
                         "official_test_used": False, "wall_seconds": wall_seconds})
        metric_rows.append({"arm": arm, "fl_seed": seed, **summary,
                            "rounds": 50, "objective_queries": 3000,
                            "dev_records": 512, "origin": origin})
        for evaluation in run_metrics["evaluations"]:
            curves.append({"arm": arm, "fl_seed": seed,
                           "fl_rounds_completed": evaluation["round"] + 1,
                           "accuracy_percent": 100 * evaluation["acc"],
                           "macro_f1": evaluation["macro_f1"],
                           "loss": evaluation["eval_loss"]})
    for seed in SEEDS:
        baseline = runs[("PRIVATE_DP_E4", seed)]["metrics"]["pre_update"]
        for arm in NEW_ARMS:
            observed = runs[(arm, seed)]["metrics"]["pre_update"]
            for key in ("acc", "macro_f1", "eval_loss", "confusion_matrix"):
                matched = (math.isclose(observed[key], baseline[key], abs_tol=1e-6)
                           if key in ("macro_f1", "eval_loss") else
                           observed[key] == baseline[key])
                if not matched:
                    raise RuntimeError("initial model/dev mismatch %s seed%d %s" % (arm, seed, key))
    pairs = []
    for left, right in COMPARISONS:
        for seed in SEEDS:
            a, b = runs[(left, seed)]["summary"], runs[(right, seed)]["summary"]
            pairs.append({"left_arm": left, "right_arm": right, "fl_seed": seed,
                          "final_accuracy_difference_pp":
                              a["final_accuracy_percent"] - b["final_accuracy_percent"],
                          "late10_accuracy_difference_pp":
                              a["late10_accuracy_percent"] - b["late10_accuracy_percent"],
                          "auc_difference_pp":
                              100 * (a["normalized_accuracy_auc"] - b["normalized_accuracy_auc"]),
                          "final_macro_f1_difference": a["final_macro_f1"] - b["final_macro_f1"],
                          "final_loss_difference": a["final_loss"] - b["final_loss"]})
    temp.mkdir(mode=0o700)
    write_csv(temp / "RUN_MANIFEST.csv", manifest, list(manifest[0]))
    write_csv(temp / "MATCHED_METRICS.csv", metric_rows, list(metric_rows[0]))
    write_csv(temp / "PAIRED_DIFFERENCES.csv", pairs, list(pairs[0]))
    write_csv(temp / "CURVES.csv", curves, list(curves[0]))
    by_arm = defaultdict(list)
    for row in metric_rows:
        by_arm[row["arm"]].append(row["final_accuracy_percent"])
    arm_lines = ["| Arm | seeds 57/58/59 final dev accuracy (%) | mean (%) |",
                 "|---|---|---:|"]
    for arm in ("NO_GUIDANCE", "PUB_RANDOM", "PUB_NLI", "PRIVATE_ZERO",
                "PRIVATE_DP_E4", "SHUFFLED_DP_LABEL", "REAL_MATCHED"):
        values = by_arm[arm]
        arm_lines.append("| %s | %s | %.2f |" %
                         (arm, "/".join("%.2f" % v for v in values), statistics.mean(values)))
    pair_means = {(l, r): statistics.mean(p["final_accuracy_difference_pp"] for p in pairs
                                          if p["left_arm"] == l and p["right_arm"] == r)
                  for l, r in COMPARISONS}
    (temp / "A_MATCHED_CONTROLS.md").write_text(
        "# Stage-2 A：匹配对照\n\n"
        "固定 128 公共候选、2 客户端、64 指导槽、512 开发集、50 轮、3,000 客户端目标查询/运行；"
        "同一 seed 的预更新指标逐项一致。仅报告开发集，官方测试集未用。\n\n" +
        "\n".join(arm_lines) + "\n\n" +
        "配对平均最终准确率差（百分点）：DP−NO %s，DP−错标签 %s，NLI−NO %s，REAL−NO %s。"
        "逐 seed、最后十轮、AUC、macro-F1 和 loss 见 CSV。三次 FL seed 是重复单位，51 个评估点不是独立重复；"
        "不据此声称统计显著。\n\n"
        "NO_GUIDANCE 采用论文 Eq. (ZOO) 的原始 z ∼ N(0,I_d)，云端不读指导训练数据且跳过 BP；"
        "guided 臂采用 α=0.5 的单位量级混合方向。二者查询半径随方向范数不同，此算子差异须作为"
        "解释限制，不能把 normalized `alpha=1` 历史基线混入此表。\n" %
        tuple(fmt(pair_means[key]) for key in COMPARISONS))

    selection = json.loads((S2 / "restricted_diagnostics/selection_aggregate.json").read_text())
    profile = json.loads((S2 / "restricted_diagnostics/selection_profile.json").read_text())
    interesting = {(p["left"], p["right"]): p for p in selection["pairwise"]}
    (temp / "SELECTION_AUDIT.md").write_text(
        "# Stage-2 B3：候选选择审计\n\n"
        "所有 6 个已保存选择缓冲均为冻结公开池中的 64 个唯一文本、四类各 16。"
        "PRIVATE_ZERO 与 DP D01/D02/D03 的交集分别为 %d/%d/%d 条，"
        "与 PUB_NLI 的交集为 %d 条；逐类 Jaccard 保存在受限分析 JSON，公开报告只给汇总。"
        "客户端×类别 8 个 top-8 边界中 %d 个第 8/9 名并列（中位票差 %.1f），"
        "这是冻结候选和固定 staged 域上的一次审计，未做新 ε 扫描。"
        "公开池的 8,128 个成对余弦中 >0.9 的对数为 %d（MiniLM 同机 CPU 编码）；"
        "该阈值不等于下游任务质量。原始票数和私有选择 ID 未进入公开报告。\n" %
        (interesting[("PRIVATE_ZERO", "DP_D01")]["intersection"],
         interesting[("PRIVATE_ZERO", "DP_D02")]["intersection"],
         interesting[("PRIVATE_ZERO", "DP_D03")]["intersection"],
         interesting[("PUB_NLI", "PRIVATE_ZERO")]["intersection"],
         profile["selection_boundary_ties"], profile["selection_margin_8_vs_9_median"],
         profile["public_semantic_duplicate_pairs_cosine_gt_0_9"]))

    diagnostic_path = S2 / "restricted_diagnostics/round0_zgr/diagnostics.json"
    diagnostic = json.loads(diagnostic_path.read_text()) if diagnostic_path.exists() else None
    if diagnostic:
        public_diagnostic = {key: diagnostic[key] for key in (
            "schema_version", "round", "checkpoint_kind", "checkpoint_sha256",
            "trainable_count", "initialization", "energy_capture_measurement",
            "production_bp_operator", "production_ggd_operator",
            "historical_round0_ggd_bitwise_reproduction", "overall", "layer_groups",
            "component_costs", "privacy")}
        (temp / "ZGR_DIAGNOSTICS.json").write_text(
            json.dumps(public_diagnostic, indent=2, sort_keys=True) + "\n")
        lines = ["# Stage-2 B1：共同 round-0 初始化的实际 BP/GGD 诊断", "",
                 "从 seed57、相同 adapter 初始化重建并逐臂核对 trainable 参数 SHA256；"
                 "调用正式 `train_model_bp` 与 `FedSGDCloud.create_perturbation`。"
                 "这是受控的 round-0 重建，不是历史 Stage-1 的比特级 GGD 复现。",
                 "", "| 指导源 | 相对 REAL_MATCHED cosine | 相对 L2 误差 | sign agreement | rank-1 能量捕获 |",
                 "|---|---:|---:|---:|---:|"]
        for arm, stats in diagnostic["overall"].items():
            lines.append("| %s | %.4f | %.4f | %s | %.4f |" %
                         (arm, stats["cosine"], stats["relative_l2_error"],
                          "%.4f" % stats["sign_agreement"] if stats["sign_agreement"] is not None else "NA",
                          stats["zgr_rank1_normalized_energy_capture"]))
        lines += ["", "当前 production GGD 的 pool_size=1，因此 V 为该 BP 梯度归一化形成的 rank-1 方向；"
                  "能量捕获由 actual create_perturbation 返回方向计算，并核验与 BP cos² 一致；不是未测的高秩子空间。"
                  "没有中期共同 checkpoint，round10/25/49 不报告。原始梯度与私有记录只留受限目录。"]
        (temp / "ACTUAL_ZGR_DIAGNOSTICS.md").write_text("\n".join(lines) + "\n")
    else:
        (temp / "ACTUAL_ZGR_DIAGNOSTICS.md").write_text(
            "# Stage-2 B1\n\n共同 round-0 BP/GGD 诊断尚未完成；不能报告方向余弦或子空间捕获。\n")

    cost_rows = [
        {"component": "MiniLM_model_load", "arm": "shared", "device": "co_located_CPU",
         "seconds": profile["model_load_seconds"], "measurement_status": "measured_one_run"},
        {"component": "client_encode_vote", "arm": "shared", "device": "co_located_CPU",
         "seconds": profile["T_encode_vote_seconds"], "measurement_status": "measured_one_run"},
        {"component": "noise_select_public_zero_hist_median", "arm": "shared", "device": "co_located_CPU",
         "seconds": profile["T_noise_select_public_zero_hist_median_seconds"],
         "measurement_status": "measured_20_reps_public_input"},
    ]
    if diagnostic:
        for arm, costs in diagnostic["component_costs"].items():
            for component, field in (("cloud_BP", "T_cloudBP_seconds"),
                                     ("GGD_compute", "T_GGD_compute_seconds")):
                cost_rows.append({"component": component, "arm": arm,
                                  "device": "co_located_A6000", "seconds": costs[field],
                                  "measurement_status": "measured_round0_replay"})
    for arm in NEW_ARMS:
        for seed in SEEDS:
            cost_rows.append({"component": "full_downstream_run", "arm": arm + "_seed" + str(seed),
                              "device": "co_located_A6000", "seconds": runs[(arm, seed)]["wall_seconds"],
                              "measurement_status": "measured_wall"})
    write_csv(temp / "COMPONENT_COSTS.csv", cost_rows, list(cost_rows[0]))
    sampled_peak = None
    samples = S2 / "resource_samples.csv"
    if samples.exists():
        values = [int(row["gpu0_memory_used_mib"]) for row in read_csv(samples)
                  if row["gpu0_memory_used_mib"].isdigit()]
        sampled_peak = max(values) if values else None
    (temp / "DEPLOYMENT_AND_BYTES.md").write_text(
        "# 部署位置和字节\n\n"
        "公开 Qwen 候选在训练前生成并缓存；本机未重跑生成，T_generate 未测。"
        "客户端 MiniLM 对自身 staged 记录编码并形成整数投票，随后本地加离散几何噪声和选 ID；"
        "云端按公开 ID 恢复 64 条文本做 BP/GGD，下游 FedFwd 客户端仍用真实记录。"
        "此次仅在同一 A6000 主机上共址运行，非 Jetson 或实际客户端网络测量。\n\n"
        "冻结候选池 JSONL 文件为 %d B；若完整发送给两个未缓存客户端，文件内容共 %d B。"
        "对已有 D01 发布采用明确 JSON ID 参考编码，两个客户端 ID 负载共 %d B、"
        "独立头部共 %d B；已有完整文本发布文件为 %d B。"
        "这些是文件或参考序列化器的真实字节计数，**不是**现有 MPI 的线上流量；"
        "缓存命中后候选下行可为 0。原训练梯度消息含完整参数张量，"
        "仅按 float32 内容推导每次 267,252,880 B，不含 pickle/MPI 头、模型下发或 GGD。\n\n"
        "MiniLM 加载 %.3f s，编码投票 %.3f s，公开零直方图 20 次噪声加选样的中位时间 %.4f s；"
        "样本 GPU 显存峰值 %s MiB（5 s 采样）。T_serialize、真实链路、能耗未测。"
        "一次性准备成本须与每轮云 BP/GGD 和 FedFwd 临界路径分列，不把并行客户端时间相加。\n" %
        (profile["public_pool_jsonl_file_bytes"],
         2 * profile["public_pool_jsonl_file_bytes"],
         profile["reference_id_payload_bytes_total"],
         profile["reference_id_header_bytes_total"],
         profile["released_full_text_jsonl_file_bytes"],
         profile["model_load_seconds"], profile["T_encode_vote_seconds"],
         profile["T_noise_select_public_zero_hist_median_seconds"],
         sampled_peak if sampled_peak is not None else "NA"))
    (temp / "PRIVACY_RELEASE_LEDGER_UPDATE.md").write_text(
        "# 隐私发布账本更新\n\n"
        "**no new release**。Stage-2 A2 只对已有 D01–D03 指导发布的 64 条公开文本做固定标签循环置换；"
        "A1 不使用指导数据；B 的计时噪声输入是公开零直方图，选择输出未发布。"
        "原每份独立发布在固定 staged 记录域、replace-one 邻接下 ε≤ln49、δ=0；"
        "若三份都可见，顺序组合上界 3ln49。该声明仅覆盖合成指导选择发布通道，"
        "不覆盖真实指导、无噪声私有选择、FL 更新或开发集指标。\n")
    dp_no = pair_means[("PRIVATE_DP_E4", "NO_GUIDANCE")]
    dp_shuffle = pair_means[("PRIVATE_DP_E4", "SHUFFLED_DP_LABEL")]
    private_nli = statistics.mean(by_arm["PRIVATE_DP_E4"]) - statistics.mean(by_arm["PUB_NLI"])
    supported = all(p["final_accuracy_difference_pp"] > 0 for p in pairs
                    if p["left_arm"] == "PRIVATE_DP_E4" and
                    p["right_arm"] == "NO_GUIDANCE") and dp_shuffle > 0
    (temp / "STAGE2_REVIEW.md").write_text(
        "# Stage-2 研究审阅\n\n"
        "A 包 6/6 新的 50-round 终点通过核验；连同 Stage-1 15/15，共 21 条同协议曲线。"
        "DP 指导相对纯 ZOO 的平均最终差为 %s pp，相对同文本错标签为 %s pp；"
        "DP 相对强公开 NLI 的平均差为 %s pp。%s"
        "这些是两客户端、三 seed 的 AG News 开发集观察，不是显著性、全 transcript DP 或跨客户泛化证据。\n\n"
        "机器可读逐 seed、末十轮、AUC、macro-F1、loss、51 点曲线及完整性核验见同目录 CSV/JSON。"
        "B1 %s；B2/B3 的共址选择时间、集合交集与参考编码字节已测。"
        "C 的新来源客户端组验证本轮未启动，不能判断额外私有增益是否跨来源稳定。\n" %
        (fmt(dp_no), fmt(dp_shuffle), fmt(private_nli),
         "三 seed 的 DP−NO 均为正且 DP−错标签均值为正。" if supported else
         "配对证据不足以说指导稳定优于两个对照。",
         "完成共同 round-0 重建诊断" if diagnostic else "尚未完成"))
    with (temp / "STAGE2_REVIEW.md").open("a") as handle:
        handle.write("\nNO_GUIDANCE 的 raw-z 方向范数约 671，预检最大更新范数约 288，"
                     "首轮 loss 由 1.386 升至 3.309；有限数值检查通过不等于尺度公平性通过。"
                     "其与 guided 单位量级方向的查询半径/更新尺度差异，限制了 DP−NO 的因果解释。"
                     "相同文本的标签置换对照提供更直接的语义证据。独立尺度校准须另行预注册并经作者决定。\n")
    (temp / "NEXT_DECISION.md").write_text(
        "# 下一步决策\n\n"
        "1. 匹配无指导/错标签证据：DP−NO 平均 %s pp，DP−错标签平均 %s pp；"
        "逐 seed 正负与全曲线见 A 包，不能只挑最终点。\n"
        "2. 私有相对公开 NLI：当前同来源平均差 %s pp；只有三个 seed，"
        "尚不足以证明额外私有信息值得隐私和端侧成本。\n"
        "3. 实际 ZGR：%s。\n"
        "4. 发布边界和共址成本已量化，T_generate、真实端机时间/内存、真实网络字节仍缺。\n"
        "5. 新来源 C 尚未运行，因此不能作跨来源结论；预注册的 C 只在 A+B gate、"
        "资源与作者授权后比较 PUB_NLI/PRIVATE_ZERO/REAL_MATCHED × 57/58。"
        "若两 seed 私有相对 NLI 均正且均值约≥3 pp，再考虑新组 DP 选择；否则"
        "应认真考虑 public-only guidance 或重新设计选择，不通过挑种子寻找私有优势。"
        "只有跨来源增益与成本/隐私边界支持，才值得另行设计固定配置 ε={1,2,4,8} 扫描；本轮不启动。\n" %
        (fmt(dp_no), fmt(dp_shuffle), fmt(private_nli),
         "共同 round-0 指标见 ACTUAL_ZGR_DIAGNOSTICS.md" if diagnostic else
         "未完成共同 checkpoint，不能判定方向差异"))
    (temp / "FAILURES_AND_BLOCKERS.md").write_text(
        "# 失败与阻断\n\n"
        "CPU profile 首次因错误的 MiniLM 根目录在模型加载前失败；原日志保留在受限目录，"
        "定位到保存的数据盘模型后成功。未因精度结果重跑任何完整终点。"
        "B1 首次在参数量检查处退出：create_model 返回 1,040,932 个可训练参数，"
        "正式 trainer 会移除 pre_classifier 的 590,592 个参数，实际训练维度为 450,340。"
        "修复为先调用正式 trainer 初始化，再哈希和校验共同 checkpoint；"
        "原脚本及失败日志保留。只补跑离线诊断，不重跑任何训练或 DP 发布。"
        "C 尚未通过执行 gate，本轮未启动。"
        + ("\n" if diagnostic else "B1 共同 checkpoint 诊断仍待完成。\n"))
    validation = {"schema_version": 1, "stage1_reused_endpoints": 15,
                  "stage2_new_endpoints": 6, "all_21_complete": True,
                  "all_seed_preupdate_metrics_identical": True,
                  "rounds_per_run": 50, "evaluations_per_run": 51,
                  "objective_queries_per_run": 3000,
                  "official_test_used": False, "new_dp_release": False,
                  "round0_zgr_diagnostic_complete": diagnostic is not None,
                  "privacy_scope": "synthetic_guidance_selection_release_only"}
    (temp / "VALIDATION.json").write_text(json.dumps(validation, indent=2, sort_keys=True) + "\n")
    handoff = temp / "paper_handoff"
    handoff.mkdir()
    write_csv(handoff / "TABLES_FOR_PAPER.csv", metric_rows, list(metric_rows[0]))
    (handoff / "RESULTS_FOR_PAPER.md").write_text(
        "# 可供论文核验的 Stage-2 结果\n\n" + "\n".join(arm_lines) +
        "\n\n该表是 AG News 两客户端、512 开发集、50 轮、每臂三 seed 的结果；"
        "没有官方测试、显著性检验或跨来源验证。逐 seed 和轨迹见上级 CSV。"
        "论文正文只可写实际完成的配对差；NO_GUIDANCE 原始 z 与 guided 单位方向的查询半径不同。\n")
    (handoff / "METHOD_AND_PRIVACY_FOR_PAPER.md").write_text(
        "# 方法与隐私边界\n\n"
        "A1 独立纯 ZOO：z ∼ N(0,I_d)、中心差分 h=0.01、LR 0.01，无云 BP/指导数据。"
        "A2 对已有 D01–D03 发布做公开标签循环置换，文本和 FL seeds 配对。"
        "所有臂 2 客户端、3,000 查询/运行、50 轮，评估仅在固定 512 开发集。"
        "Stage-1 每份指导发布 ε≤ln49、δ=0，三份共同可见上界 3ln49；"
        "本轮没有新发布，完整 FL transcript 不受该保证。\n")
    (handoff / "CLAIM_EVIDENCE_MAP.md").write_text(
        "# Claim—evidence\n\n"
        "| 候选表述 | 证据 | 边界 |\n|---|---|---|\n"
        "| DP 指导相对纯 ZOO 的开发集表现 | PAIRED_DIFFERENCES.csv + 51 点 CURVES.csv | 3 seed、查询半径不同；非显著性证据 |\n"
        "| 文本标签语义贡献 | 同文本 SHUFFLED_DP_LABEL 与 DP 配对 | 仅当前候选/标签置换 |\n"
        "| 私有选择优于公开 NLI | MATCHED_METRICS.csv | 当前增益小，未跨来源验证 |\n"
        "| 选择发布隐私 | Stage-1 审计 + PRIVACY_RELEASE_LEDGER_UPDATE.md | 不覆盖 FL transcript |\n"
        "| 梯度/子空间机制 | ACTUAL_ZGR_DIAGNOSTICS.md | 仅共同 round-0 重建；无中期轨迹 |\n"
        "| 资源成本 | COMPONENT_COSTS.csv + DEPLOYMENT_AND_BYTES.md | 共址设备，非端机/真实链路 |\n")
    temp.rename(OUT)
    print(json.dumps({"report": str(OUT), "runs": len(runs),
                      "new_runs": 6, "diagnostic": diagnostic is not None}))


if __name__ == "__main__":
    main()
