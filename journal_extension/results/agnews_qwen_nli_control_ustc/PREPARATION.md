# AG News Qwen 公开 NLI 选择对照：准备状态

**2026-10-14。状态：公开选择与质量审计已完成，下游 50 轮训练未启动。**

本对照沿用 2026-10-13 已验证为完全公开重放的 128 条 Qwen 候选池，SHA-256 为 `7e54c05ce2dc94d754b178218963dedfceb781c491b97ed1d9db5ab9e7d7c54f`。冻结的公开 `facebook/bart-large-mnli` 模型（revision `d7645e127eaf1aefc7862fd59a17a5aa8558b8ce`）按四类 AG News 语义假设对候选评分。每类选目标类别分数最高的 16 条，并交错分给两个客户端槽位，每槽位每类 8 条。选择器只读取公开候选，不读取私有训练记录、旧投票结果或开发集标签；没有 DP 机制。

选择输出共 64 条，每类 16 条、每客户端 32 条。输出 SHA-256 为 `52810c8290a47533048d82a33bc92279c50d4df77f3628a1539031ba2905194a`，本地备份重新计算后与清单一致。独立 AG News 分类器的目标类别同意率为 **59/64＝92.19%**：World 16/16、Sports 16/16、Business 11/16、Science/Technology 16/16。所有样本均不超过下游 64-token 限制。这些是质量代理指标，不是下游精度或隐私结果。

下游预定协议固定为 seed 57、50 FL rounds、DistilBERT adapter FedFwd、两个源客户端、相同的固定 512 条开发集，以及与私有投票和公开随机选择对照相同的 64 条指导预算。运行器将保留每轮评估曲线，并把原始归档留在服务器和本地备份；GitHub 只保存聚合报告与代码。

服务器四张 A40 当前均有其他用户任务。GPU1 在启动检查时约有 9.25 GiB 空闲，低于原实验 `check_gpu` 的 **14 GiB** 门槛，因此没有启动训练。运行器保留该门槛，并可通过 `--gpu 0..3` 选择符合门槛的空闲卡。现阶段不能从 92.19% 的类别同意率推断下游会超过公开随机选择的 70.31% 或私有投票的 75.00%。

公开选择器：`journal_extension/select_public_qwen_nli.py`；训练驱动：`journal_extension/run_ustc_qwen_nli_control.py`。原始选择 JSONL、清单与质量审计在本地 `local_backups/agnews_qwen_nli_control_20261014/selection/`，不进入 Git。
