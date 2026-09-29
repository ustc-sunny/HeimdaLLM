# AG News DP v1 实验完成报告

核验日期：2026-09-29。全部 **12/12** 次正式实验已完成，未继续启动新实验。

正式队列从 2026-09-28 17:39:16 运行到 2026-09-29 07:08:35（北京时间），总耗时 **13 小时 29 分 19 秒**。所有任务使用校内共享 A40 的物理 GPU1；耗时不作为方法速度指标。

## 协议与隐私范围

- ε targets：8、4、2、1；δ=1e-5；各使用 seeds 57、58、59。
- 源 clients 1 和 21，每个 client 120 条训练记录；独立训练 DistilGPT2 LoRA，5 epochs。
- 每类生成 32 条，共 128 条 synthetic records。
- 每次 50 FL rounds；训练前及每轮评估，使用固定的 512 条平衡 AG News dev records。
- 每条完整轨迹有 51 个评估点，合计 **612** 个评估点。
- 本轮保护的是 DP 生成器发布的合成数据；下游 FL 更新仍为 Non-DP，不能声称端到端 FL DP。

实际实验代码冻结在 `590a20e161c664d4d5ced292dac2ef7822433030`。后续 GitHub 提交保存报告与结果，不改变已经完成的运行代码。

## 最终准确率

下表是 round 49 的 dev accuracy。均值和标准差按三个 seeds 计算；标准差是 sample SD，不是标准误。

| ε target | seed 57 (%) | seed 58 (%) | seed 59 (%) | 均值 ± SD (%) | 平均 macro F1 |
|---:|---:|---:|---:|---:|---:|
| 8 | 46.8750 | 35.7422 | 36.9141 | **39.84 ± 6.12** | 0.3527 |
| 4 | 47.8516 | 35.3516 | 35.9375 | **39.71 ± 7.05** | 0.3597 |
| 2 | 47.2656 | 36.1328 | 45.8984 | **43.10 ± 6.07** | 0.3903 |
| 1 | 46.6797 | 34.3750 | 39.6484 | **40.23 ± 6.17** | 0.3642 |

## 隐私会计结果

每个 ε 条件的三个 seeds 得到相同的 accountant bound，因为公开训练步数、Poisson sampling rate 和 noise multiplier 相同；各次训练的实际采样与噪声使用独立的私有随机性。

| ε target | 实际 ε（max across clients） | Noise multiplier σ |
|---:|---:|---:|
| 8 | 7.9993157580 | 0.6866455078125 |
| 4 | 3.9935493851 | 0.9094238281250 |
| 2 | 1.9917902059 | 1.2597656250000 |
| 1 | 0.9929987894 | 1.9531250000000 |

全部实际 ε 均未超过 target。发布 manifests 声明的采样与噪声随机性均为 `fresh_private_entropy_not_derived_from_public_seed`。这是 PyTorch PRNG 的研究实现，不声称密码学部署。

## 与已有 v3 的描述性对照

已有 Non-DP v3 的最终平均准确率为：client-local LoRA synthetic **76.1719%**，public pretrained generator synthetic **39.0625%**，no guidance **28.7109%**。v3 的 no-guidance 是已报告的较弱、基本不学习的对照，不把与它的差距解释为相对充分优化 FL 基线的收益。

| 本轮条件 | 相对 v3 Non-DP client synthetic (pp) | 相对 v3 public synthetic (pp) |
|---|---:|---:|
| ε=8 | −36.33 | +0.78 |
| ε=4 | −36.46 | +0.65 |
| ε=2 | −33.07 | +4.04 |
| ε=1 | −35.94 | +1.17 |

这些结果显示：本轮 DP 生成方案未保留 v3 Non-DP client-local LoRA 的大幅效用收益，准确率整体接近 public-generator 水平。这个现象与额外 client information 的效用减弱相容，但仅凭这组效用结果不能确定原因。

ε=2 的均值最高，但结果没有呈现清晰单调的 privacy–accuracy 曲线；各条件的 seed 波动约 6–7 pp，仅有三个 seeds，不能据此认定 ε=2 更优或存在统计显著差异。

v3 对比包含 per-example clipping、Poisson sampling、DP noise、float32 生成器训练，以及部署环境差异。因此不能把准确率下降全部归因于 Gaussian noise。要定位原因，需要后续单独设计匹配这些设置的 Non-DP 对照；本次未启动这类额外实验。

## 完整性与备份

本地重新核验了：

- 四份原始归档的 SHA256；
- 12 次运行的完成状态、全部评估轮次、最终准确率；
- 原始 metrics 和 generator manifests 与公开汇总中的哈希一致；
- 12 份 DP release validation 全部通过，实际 ε 未超过目标；
- CSV 共 612 行评估，与原始 metrics 逐运行的轮次计数一致；
- 汇总中的准确率均值与 sample SD 与逐 seed 数据一致。

服务器结果根目录：`/home/zzkevin/heimdallm-journal-20260928/results/`。

本地原始归档根目录：`ToN26/local_backups/agnews_dp_v1_ustc_20260928/`。四个文件为 `ustc_agnews_dp_eps{8,4,2,1}_20260928.tar.gz`，各有对应 `.sha256`；完成队列日志也已保存。

VPN 断开使之前本地与 GitHub 的进度报告滞留在 0/12，服务器训练和归档正常完成。连接恢复后，完整结果于 2026-09-29 同步至本地及 GitHub `HeimdaLLM+` 分支，结果提交为 `057ccc7`；临时同步重试进程已停止。

同目录 `summary.json` 提供完整逐 seed 统计、实际 ε 和 artifact hashes；`curves.csv` 保存全程 accuracy、macro F1 和 loss；`REPORT.md` 是机器生成的整体汇总；`DEPLOYMENT.md` 保留启动时的部署记录。
