# 可供论文核验的 Stage-2 结果

| Arm | seeds 57/58/59 final dev accuracy (%) | mean (%) |
|---|---|---:|
| NO_GUIDANCE | 26.37/33.01/33.40 | 30.92 |
| PUB_RANDOM | 70.31/69.14/65.04 | 68.16 |
| PUB_NLI | 75.00/71.29/64.45 | 70.25 |
| PRIVATE_ZERO | 75.00/71.88/65.23 | 70.70 |
| PRIVATE_DP_E4 | 76.95/68.95/65.43 | 70.44 |
| SHUFFLED_DP_LABEL | 28.52/35.74/25.39 | 29.88 |
| REAL_MATCHED | 72.07/73.05/72.07 | 72.40 |

该表是 AG News 两客户端、512 开发集、50 轮、每臂三 seed 的结果；没有官方测试、显著性检验或跨来源验证。逐 seed 和轨迹见上级 CSV。论文正文只可写实际完成的配对差；NO_GUIDANCE 原始 z 与 guided 单位方向的查询半径不同。
