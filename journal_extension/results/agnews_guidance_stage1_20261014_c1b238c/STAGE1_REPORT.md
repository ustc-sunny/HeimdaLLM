# HeimdaLLM+ AG News Stage 1 实验报告：本地审计完成，服务器运行受阻

**后续状态更新：**2026-10-15 已成功登录 A40，原 SSH 阻断解除；四张卡空闲显存均低于既有 14000 MiB 门槛，故仍未启动新训练。当前状态见 `RESUME_CHECK_20261015.md`。以下正文保留 10 月 14 日审计时的事实。

2026-10-14；协议目录 `stage1_agnews_20261014_c1b238c/`。本轮没有启动新的服务器实验。目标 15 个下游终点中，3 个 seed-57 旧终点通过本地归档完整性与协议预检查，暂列**待现场确认复用**；其余 12 个没有开始。E0 仅完成静态代码和本地工件审计，E1 新训练 0/9，E2 DP 0/3，E3 共同 checkpoint 诊断 0。SSH 在密码交互后由远端断开，无法确认共享 GPU 使用条件；同时当前 DP 投票选择器尚无可通过的实现审查。故按任务包的阻断停止条件停止，不扩大实验范围。

## 已核验结果（仅固定池、seed 57、开发集；Non-DP）

| 指导 | Final acc | Macro-F1 | 最后 10 次 acc 均值 | 状态 |
|---|---:|---:|---:|---|
| PRIVATE_ZERO | 75.00% | 0.7454 | 74.80% | 历史归档本地核验、现场复用待定 |
| PUB_RANDOM | 70.31% | 0.6872 | 70.16% | 同上 |
| REAL_MATCHED | 72.07% | 0.7064 | 71.58% | 同上；真实指导是非隐私 oracle |
| PUB_NLI | NA | NA | NA | 只完成公开选择，未有下游训练 |
| PRIVATE_DP_E4 | NA | NA | NA | 机制未实现/审查，未发布 |

PRIVATE_ZERO−PUB_RANDOM 为 +4.6875 个百分点，PRIVATE_ZERO−REAL_MATCHED 为 +2.9296875 个百分点，仅各有一个 FL seed。PUB_NLI 选择上的 92.1875% 是辅助分类器标签同意率，不是下游 accuracy。PRIVATE_ZERO 相对 PUB_NLI、DP 相对任何臂的下游差都为 NA。逐轮轨迹、per-class 指标、AUC、达到阈值首次更新、配对差及全部 15 单元见 `curves.csv`, `per_class_metrics.csv`, `metrics.csv`, `paired_comparisons.csv`。这些是对本地历史归档的重算，不能作为重复 seed 显著性、收敛或 KDD 全规模性能证据。

## 匹配性与重要审计发现

三个归档的 SHA256 sidecar、50 次更新及 51 个评估点、退出和完成标志通过。本地重放公开随机选择逐字节一致；公开 NLI 的 top-16/类交错分配逐条重建一致。128 候选、64 cloud 槽位、64 唯一文本、每类 16、无 cloud 去重。历史命令均为 `alpha=.5`, LR=.01, DistilBERT adapter/FedFwd，两个逻辑客户端 1、21；历史日志对应每任务约 3000 客户端正负目标查询与 3200 cloud 样本访问，未实测新任务。源 AG News 与 partition 的历史哈希尚不能现场重核，实际更新的同 checkpoint 单步等价/参数 delta 也尚未核验，所以复用状态保留 provisional。

当前 `PRIVATE_ZERO` 不是相互独立的两客户端 top-8：每类客户端 21 从客户端 1 已选 ID 之外选择，并按私有票排序输出。缺失某类别时会异常退出。直接加入 Laplace 后独立选两个 top-8，会改变零噪声语义；现有代码没有满足同管线的 ε4 版本。理想直方图机制在一条记录一票时有全局 L1 敏感度≤2，ε4 的 Laplace scale=.5；这是**条件性数学设计**，不代表实现已审查或隐私保证已达成。三次独立 release 若都公开，条件性基本联合上界为 ε≤12；实际 release 数为 0。保护范围只能是合成指导发布通道，不能覆盖 FL 梯度通信 transcript。

实际 FedFwd 路径发送梯度张量到服务器；服务端聚合代码把每 worker 的梯度直接求和再除总样本数，虽然旁注说按样本量加权。云 BP 可见路径发送归一化 guidance 方向，没有看到直接写正式模型的路径，但未作现场 delta 检查。当前 `pool_size=1` 对应 rank-1 cloud 方向，不能把理论的任意 V 子空间视作已实现。这些会影响论文方法表述与 E0 正确性门槛；详见 `CORRECTNESS_AND_PROTOCOL_AUDIT.md` 和 `paper_handoff/THEORY_IMPLEMENTATION_MAP.md`。

## 成本、隐私与论文可用性

仅有历史 cloud H5 pack 耗时约 0.36/0.36/0.52 秒，见 `COMPONENT_COSTS.csv`。它不包含生成、传输、编码/投票、DP、cloud BP、GPU 峰值或真实边端设备性能，不支持速度或内存收益结论。GPU 的 14000 MiB 是历史预检门槛，不是实测峰值。

可在草稿中谨慎描述“固定小规模设置下，一个旧 seed 的 Non-DP 私有选择比公开随机高 4.69 pp”；投稿级主表尚不能声称私有优于强公开 NLI、DP 保持效用、接近 KDD 全规模精度或完整 FL 隐私。`paper_handoff/` 将三个维度明确分开：数学条件、实现审查、实际结果；所有未完成单元为 NA。

## 恢复与停止判定

阻断详情和恢复命令见 `FAILURES_AND_BLOCKERS.md`。首先恢复交互 SSH shell，并核验仓库、原始数据、GPU 与共享审批；完成同 checkpoint 单步 E0 检查，确认历史结果是否可正式复用。随后在**同一**冻结协议下补 PUB_NLI 57 和四臂 seeds 58/59；再单独实现并审查与 PRIVATE_ZERO 一致的 DP 选择，最后才运行 D01–D03。E3 需先取得共同 REAL_MATCHED checkpoint。没有为精度重试或选择有利噪声工件。当前**不进入**扩展 ε/Q 扫描、新数据集或论文规模实验。
