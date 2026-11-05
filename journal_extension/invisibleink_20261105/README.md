# InvisibleInk synthetic guidance pilot, 2026-11-05

本轮已完成 **14个50轮终点**：PUBLIC、NONPRIVATE、ε=1/4/8/12/16，各FL seeds57/58。每条件一份64条生成工件，两个FL seed共用；没有独立生成重复。ε=20未生成、未训练。

- [中文结果分析与下一步计划](analysis/RESULTS_AND_NEXT_PLAN_CN.md)
- [逐seed指标](results/metrics.csv)、[配对差值](results/paired.csv)、[714点评估曲线](results/curves.csv)
- [隐私—精度图](results/privacy_accuracy.png)、[50轮训练曲线](results/training_curves.png)
- [机器结果](results/RESULTS.json)、[独立指标复算](analysis/FINAL_ANALYSIS.json)
- [分类别解释更正](analysis/LABEL_INTERPRETATION_CORRECTION.md)
- [下一步P1四个参考终点](plans/P1_CONTROLS.md)
- [实际发布预算](analysis/ACTUAL_RELEASE_ACCOUNTING.json)、[历史隐私审查](protocol/PRIVACY_AUDIT_HISTORY.md)

平均开发集accuracy：PUBLIC61.04%，NONPRIVATE69.63%，ε1/4/8/12/16分别64.26/69.53/58.20/68.65/71.19%。δ=1e-5。ε8负面结果保留；不存在单调性、最优预算或等效精度结论。PUBLIC/NONPRIVATE沿用ε8校准的公开解码支持集，其余DP支持集随C变化。NONPRIVATE不是原真实指导。

ε4/seed57具有完整51次评估、750更新、1500方向、3000查询和4个检查点，但历史退出码未观察到；保留审计例外。其余13个终点干净退出。本次ε16补训练复用原工件；旧部分生成及中断训练保留，不按精度挑选工件。

## 机制与证据范围

生成器为冻结Qwen2.5-3B-Instruct，使用InvisibleInk差分logits裁剪和公开top-k+采样。没有生成器LoRA训练或DP-SGD。固定槽位完整记录替换的理想会计采用rho=2TC²/(B²tau²)，T96/B3/tau1。隐私界限只针对合成发布通道；浮点采样是研究实现，下游FL、开发指标、检查点及研究档案不是该机制的DP输出。

本实验为AG News、两个逻辑客户端100/101、各120训练记录、DistilBERT适配器450340个可训练参数的受控研究；不代表原80客户端规模或全部LLM工作负载。官方测试集未访问。未把旧投票或LoRA结果拼入该曲线。

## 代码恢复

`code/`按实际运行阶段保留原脚本字节，包括过往计划中未执行的条件；**它们是审计源码，不是启动指令**。`wide/generate.py`仍包含历史ε20参数，不代表授权或已执行。下一次运行必须使用新的批准范围、输出目录及GPU预检，不可直接重启历史队列。

1. 在新的隔离克隆中检出源码基准`04cb83d5ebb5cfb05e1851629d2578855f2f3906`，不reset用户工作树。
2. 将本目录`downstream_overlay/`覆盖到这个新克隆。此补丁来自已核验的SOURCE_RECOVERY_FINAL.zip，原包SHA256为`933acede0340f623fd678a95270736b2e7ef34bd268fa85fb4e10dbd4110fbfc`。核对`protocol/DOWNSTREAM_SOURCE_GATE.json`的6个关键文件。
3. 下游Python3.9、torch1.13.1+cu116、adapter-transformers3.1.0（transformers4.21.3）；生成环境Python3.10、torch2.7.1+cu126、transformers4.54.1。模型、FedNLP训练数据、冻结划分和指导H5需从受限备份恢复，不能只凭本仓库重建相同历史工件。
4. 历史脚本含当时机器路径和GPU UUID；新部署先显式适配路径、核对输入哈希/标签映射/初始化，再执行共享GPU权限与空闲预检。运行中的其他用户进程只记录，不因其出现就终止健康训练；禁止处理他人进程。
5. 仅复算本次指标可在无模型环境运行`python3 verify_results.py`。

## 发布内容

本目录提供代码、固定协议、聚合指标、混淆矩阵、曲线和审计说明。不包含原始记录、生成文本、未加噪私有统计、原始梯度、检查点、凭据或DP秘密随机状态。完整本地备份分布于invink_pilot_20261103、invink_diagnostics_20261103、invink_tradeoff_dense_20261104、invink_tradeoff_wide_20261104及invink_e16_completion_20261105；新增终点备份的97个文件已核验。

本目录单独追加，不改main和原算法文件，也不纳入本地其他未提交工作。上游许可和署名见[第三方说明](THIRD_PARTY_NOTICES.md)。
