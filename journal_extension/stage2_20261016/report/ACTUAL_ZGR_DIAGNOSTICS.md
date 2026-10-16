# Stage-2 B1：共同 round-0 初始化的实际 BP/GGD 诊断

从 seed57、相同 adapter 初始化重建并逐臂核对 trainable 参数 SHA256；调用正式 `train_model_bp` 与 `FedSGDCloud.create_perturbation`。这是受控的 round-0 重建，不是历史 Stage-1 的比特级 GGD 复现。

| 指导源 | 相对 REAL_MATCHED cosine | 相对 L2 误差 | sign agreement | rank-1 能量捕获 |
|---|---:|---:|---:|---:|
| PRIVATE_DP_E4_D01 | 0.5337 | 1.1873 | 0.6064 | 0.2848 |
| PRIVATE_ZERO | 0.5322 | 1.2105 | 0.5964 | 0.2832 |
| PUB_NLI | 0.4774 | 1.2600 | 0.5957 | 0.2279 |
| PUB_RANDOM | 0.4863 | 1.2150 | 0.5896 | 0.2365 |
| REAL_MATCHED | 1.0000 | 0.0000 | 1.0000 | 1.0000 |
| SHUFFLED_DP_LABEL | -0.1272 | 1.7660 | 0.4764 | 0.0162 |

当前 production GGD 的 pool_size=1，因此 V 为该 BP 梯度归一化形成的 rank-1 方向；能量捕获由 actual create_perturbation 返回方向计算，并核验与 BP cos² 一致；不是未测的高秩子空间。没有中期共同 checkpoint，round10/25/49 不报告。原始梯度与私有记录只留受限目录。
