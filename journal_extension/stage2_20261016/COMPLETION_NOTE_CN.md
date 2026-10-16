# Stage-2 本轮完成说明

本轮 A 包新增 6/6 个 50-round 终点，复用 Stage-1 的 15 个终点，共 21 条完整曲线。B1 共同 round-0 的 6 臂 BP/GGD 诊断、B3 选择审计和可执行的共址组件测量均完成；没有启动 C、新 DP 发布或 epsilon 扫描。

当前完整报告在 `report/STAGE2_REVIEW.md`，论文交接材料在 `report/paper_handoff/`。原初始报告保留为 `report_initial_snapshot/`；服务器上首次诊断失败时的汇总保留为 `report_public/`，补齐诊断后的版本为 `report_public_v2/`。

诊断失败原因是初始化阶段尚未移除 DistilBERT pre_classifier；修复后使用正式 trainer 的结构再做参数量与共享 checkpoint 哈希校验。实际 GGD 输出的能量捕获也已与 BP rank-1 span 核对。6 个完整训练没有重跑。诊断仅为 seed57 的受控 round-0 重建，不能当作历史中期 checkpoint 或跨 seed 机制证据。

结果支持当前配置下语义正确指导优于同文本错标签；DP 相对公开 NLI 的平均最终准确率差仍仅约 0.20 个百分点，不能证明额外私有信息有稳定增益。NO_GUIDANCE raw-z 的查询半径和更新尺度与 guided 不同，数值预检通过不等于公平性通过，报告已明确这一限制。

同级工作区 `stage2_agnews_a6000_20261016/` 保存完整服务器归档、解包后的日志/指标/受限诊断、准确的源码快照及 A1 补丁；229 个文件已逐项通过 SHA256 和大小验证。受限备份未放入 Git 仓库。当前本地源码与服务器诊断/报告脚本哈希一致。GitHub 上传限于本目录经过审查的代码与汇总材料；未直接修改论文。

仍未完成：中期共同 checkpoint、真实端机/链路测量、生成耗时与完整序列化计时、C 新来源验证。这些边界详见 report/NEXT_DECISION.md 与 DEPLOYMENT_AND_BYTES.md；不是待后台自动运行的队列。
