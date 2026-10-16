# Claim—evidence

| 候选表述 | 证据 | 边界 |
|---|---|---|
| DP 指导相对纯 ZOO 的开发集表现 | PAIRED_DIFFERENCES.csv + 51 点 CURVES.csv | 3 seed、查询半径不同；非显著性证据 |
| 文本标签语义贡献 | 同文本 SHUFFLED_DP_LABEL 与 DP 配对 | 仅当前候选/标签置换 |
| 私有选择优于公开 NLI | MATCHED_METRICS.csv | 当前增益小，未跨来源验证 |
| 选择发布隐私 | Stage-1 审计 + PRIVACY_RELEASE_LEDGER_UPDATE.md | 不覆盖 FL transcript |
| 梯度/子空间机制 | ACTUAL_ZGR_DIAGNOSTICS.md | 仅共同 round-0 重建；无中期轨迹 |
| 资源成本 | COMPONENT_COSTS.csv + DEPLOYMENT_AND_BYTES.md | 共址设备，非端机/真实链路 |
