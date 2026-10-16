# Stage-2 matched controls and common-checkpoint diagnostics

本轮已完成：6 个新增 50-round 训练终点、15 个复用终点、seed57 共同 round-0 的 6 臂 BP/GGD 诊断，以及可完成的共址成本/选择审计。未启动 C、新 DP 选择发布或 epsilon 扫描。

- [中文结果报告](report/STAGE2_REVIEW.md)
- [逐 seed 指标](report/MATCHED_METRICS.csv)、[配对差值](report/PAIRED_DIFFERENCES.csv)、[完整曲线](report/CURVES.csv)
- [实际 BP/GGD 诊断](report/ACTUAL_ZGR_DIAGNOSTICS.md)及[机器可读数据](report/ZGR_DIAGNOSTICS.json)
- [核验记录](report/VALIDATION.json)、[失败与修复](report/FAILURES_AND_BLOCKERS.md)
- [论文证据交接](report/paper_handoff/RESULTS_FOR_PAPER.md)、[下一步决策](report/NEXT_DECISION.md)

`STAGE2_RUN_SPEC.md` 保留执行前冻结规范与现场附录，最新完成状态以 `report/` 为准。没有修改 KDD 的主算法文件；A1 raw-z 对照由 `prepare_a1_source.py` 在隔离克隆中构造。其查询半径与 guided 不同，DP−NO 的因果解释受此限制。B1 是受控初始化重建，不是历史中期状态的比特级复现。

## 下次部署

先获取扩刊分支：

```bash
git clone --branch 'HeimdaLLM+' https://github.com/ustc-sunny/HeimdaLLM.git
```

本目录保存当次执行脚本，路径和源码门槛是冻结协议的一部分，不能在新机器上直接启动旧队列：

1. 挂载保留的数据盘，恢复 `/mnt/heimdallm-stage1-20261015` 和 `/mnt/heimdallm-stage2-20261016`。前者包含既有 D01–D03 工件、训练命令、分区与受限输入；后者包含本轮完整日志、源码快照和 A1 补丁。本仓库只提供代码和经过检查的汇总材料，不能替代这些输入。
2. 环境恢复包在 `/mnt/heimdallm-stage1-20261015/recovery/heimdallm-kdd-env-20261015.tar.gz`，同目录有 SHA256、Conda/Pip 清单与 `RESTORE.md`。先检查哈希；目标环境路径不存在且基础镜像兼容时，按恢复说明解包。模型、数据和 MPI 也须检查，不能仅依靠 pip 安装清单。
3. 当次运行使用 `/root/miniconda3/envs/heimdallm-kdd/bin/python`，模型和数据位于 `/root/heimdallm-work/`。源码基准为 `ad325aa320789c4f59f911898033844d5b19ccec`。新分支提交仅归档扩展脚本和证据，不能冒充当次未经修改的源码提交。
4. 如需审计性复现，在**新建的隔离克隆**中 checkout 上述基准提交，再复制本目录脚本。A2 使用干净基准；A1 在另一干净克隆中运行 `prepare_a1_source.py` 后核对两个文件的 SHA256。不要 reset 已有用户工作树。具体路径见各脚本常量，先审计再适配。
5. 先重新核验 GPU 所有者/权限、空闲显存、输入哈希和模型初始化。下一阶段使用新的输出命名空间和预注册协议；已完成的 21 个终点优先复用。旧脚本会拒绝覆盖已有输出，不应通过删除历史结果绕过检查。

## 保存与披露范围

GitHub 包含脚本、协议、聚合指标、曲线、诊断汇总和论文交接材料。私有记录、未加噪投票、私有选择 ID、原始梯度、checkpoint、原始日志、环境归档和凭据均不在本次提交中。训练 seed 57/58/59 是公开 FL seed，不是 DP 噪声秘密随机性。

Stage-2 复用既有 DP 选择输出；`no new release` 指没有新建合成选择发布。研究指标和诊断汇总本身不因此获得 DP 保证，完整 FL transcript 也不在选择发布的保证内。

完整本地备份位于工作区 `stage2_agnews_a6000_20261016/`，含校验记录；初始占位报告只保留本地，不作为当前 GitHub 结果发布。
