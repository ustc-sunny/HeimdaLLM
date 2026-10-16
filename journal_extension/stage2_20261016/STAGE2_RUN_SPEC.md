# Stage-2 预注册运行规范（2026-10-16；尚未启动训练）

基准仓库：`HeimdaLLM+`，预检 HEAD `ad325aa320789c4f59f911898033844d5b19ccec`，工作树原为 clean。服务器恢复后必须重新记录 HEAD、diff、GPU/进程所有者、显存、数据盘、模型/数据/候选池 SHA256；不得 reset、覆盖 Stage-1 或绕过共享 GPU 审批。当前原 Matpool 地址在 SSH 密钥交换前关闭连接，尚无新服务器现场审查，因此本文件冻结意图但**不授权无预检启动**。

| 包 | 预定运行 | 配对工件 / 目的 | 门槛 |
|---|---|---|---|
| A1 | `NO_GUIDANCE` × seeds 57/58/59，3×50 rounds | 同固定客户端 1/21、AG News 512 dev、初始模型与 3,000 objective queries/run；纯 ZOO，无云端指导 | 先核对原 KDD 非指导方向 `z` 与当前代码 `alpha=1` 的 `z/√d`、中心差分、估计器缩放和 `ΔW`。不能未经审计把 `alpha=1` 当同算法；数值 sanity 失败则停止。LR 预定 0.01，不事后单独调参。 |
| A2 | `SHUFFLED_DP_LABEL` × seeds 57/58/59，3×50 rounds | 对应 D01/D02/D03 的完全相同 64 条公开文本，固定标签循环 `1→2→3→4→1`，保持每类 16 / 每客户端原始类 8 槽；无新私有查询 | 校验源 release SHA、文本原顺序、配额、pack H5 `test=0`；沿用 Stage-1 训练 flags/数据/硬件预检。 |
| B1 | 首选 seed57 的共同 round-0 checkpoint；仅复现成功时最多一条 D01 seed57 50-round 诊断重放 | 相同模型状态比较五臂+shuffled 的 BP 梯度与 production ZGR `V`；round10/25/49 只在重放每轮与原数值一致时测 | 无共同 checkpoint/hash 不写“相同 checkpoint”；不发布私有梯度或记录。 |
| B2/B3 | 不新增完整训练 | 候选/ID 选择审计、实际序列化 bytes、组件计时与部署位置、共址资源曲线 | 原始票数/未噪声私有选择只在受限环境，公开件只含审查过的汇总；无端机不能写 Jetson 实测。 |
| C（暂不执行） | A+B gate 后才讨论另一预选客户端组，最多 3 arms × seeds 57/58 = 6 终点 | 只比较 PUB_NLI、PRIVATE_ZERO、REAL_MATCHED；原 128 候选池 | 先冻结新客户端元数据规则并检查数据不相交、每类真实指导预算；需有资源与作者授权，不按结果改组。 |

冻结不变量：`FROZEN_PROTOCOL.json` 的 protocol ID `agnews_pool128_guidance64_fedfwd_legacy50_20261015`；池 SHA256 `7e54c05ce2dc94d754b178218963dedfceb781c491b97ed1d9db5ab9e7d7c54f`；固定分区 SHA256 `e16e8551a87ba702cc23b4d56d57b9762b77c4fea8bf341b96d5759be7996a6c`；DistilBERT adapter + FedFwd、batch 8、`fd_step=.01`、`beta=1`、50 rounds、每轮评估、dev 512，官方 test=0。指导臂沿用 `alpha=.5`, `pool_size=1`, `zo_estimator=legacy`；A1 的纯 ZOO 精确缩放需通过门槛后单独冻结，不以历史不同协议 no-guidance 代替。A2 每 seed 与既有 DP 结果一对一。

成功判据：每 run exit 0、50 rounds/51 次开发集评估、750 内层更新、1,500 FD 方向、3,000 objective queries；无官方 test；源与 H5 哈希、文本配对和 seed 固定；失败/OOM 记入 manifest，不因数字差重试。汇总最终/末十轮/AUC/macro-F1/loss，配对 DP−NO、DP−SHUFFLED、NLI−NO、REAL−NO；不把 51 个时间点当独立样本。

资源预算：A 最多 6 个新终点，按 Stage-1 同机单 run 30.28–39.14 分钟估计训练墙钟约 3.0–3.9 小时（不含预检/准备/排队，且不同硬件不可套用该时间）；B 最多 1 个额外 50-round 重放；C 最多 6 个终点且当前未获 gate。任何新公开 DP 选择发布都不在本规范内；A/B 复用 D01–D03，不新增 ε。三份共同可见输出的 Stage-1 上界仍是 $3\ln49$，完整 FL transcript 不受此保证。

## 现场执行附录（2026-10-16；不改变以上预注册矩阵）

新租的单卡 RTX A6000 服务器 `qdnvA4` 已通过独占 GPU 预检，Stage-1 数据盘与冻结候选池、源数据、划分的 SHA256 均核对一致。根目录旧仓库未修改；从 Stage-1 Git bundle 在独立目录恢复 `ad325aa`，A2 使用原始受指导源码，A1 使用第二份隔离克隆。A2 的 D01/57、D02/58、D03/59 依次运行；一轮预检已通过，正式队列已启动。

A1 算子门槛的结论在看任何 A2 终点准确率之前固定：原论文 Eq. (ZOO) 的纯扰动为 $z\sim N(0,I_d)$，而现有 `alpha=1` 分支返回 $z/\sqrt d$。因此 A1 单独采用 `zo_estimator=isotropic_raw`、`alpha=1`、`beta=1`，只在隔离克隆中把该分支改为原始 $z$，保持中心有限差分 `h=0.01`、外部乘子 1、学习率 0.01、相同 3,000 客户端目标查询；云端 MPI rank 保留同步屏障，但跳过 BP 且不加载指导训练数据。CPU 二次型精确导数检查通过，实际一轮更新和开发集预更新一致性仍待 A2 结束后核验。这个基线忠于原论文但与 guided 分支的单位半径方向有不同查询半径，报告中将明示。

CPU 选择组件的第一次独立 profile 因 MiniLM 路径指向旧根目录而在模型加载前失败；失败日志留存，修正到保留的数据盘模型目录后成功。该 profile 不创建新的 DP 发布。所有完整训练的验证结果与资源采样必须以后续 manifest 为准，不能凭队列启动视为完成。
