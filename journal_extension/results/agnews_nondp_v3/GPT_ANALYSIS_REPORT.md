# 给 GPT 分析的实验报告：AG News Non-DP Guidance Validation v3

请把这份报告当作一个独立的实验材料进行审慎分析。希望重点判断：实验是否回答了研究问题、结论强度是否合适、public-generator control 是否足以区分 prompt prior 与 client-local LoRA 信息、梯度诊断是否与端到端结果一致，以及下一步实验应该如何设计。

---

## 1. 研究背景

这是 HeimdaLLM+ journal extension 中的 AG News synthetic guidance 实验。

此前 v2 只运行了 5 个 FL rounds，在 seeds 57/58/59 上得到：

- No guidance：28.71%
- Client synthetic guidance：37.43%
- Same-source real guidance：41.34%
- Held-out real guidance：39.65%
- Shuffled-label synthetic：27.80%
- Client synthetic − no-guidance：平均 +8.72 percentage points

v2 表明 synthetic guidance 在训练早期可能有效，但不能回答长期效果，也不能区分收益来自 client-local LoRA，还是来自 pretrained DistilGPT2 加公开类别语义 prompt。

v3 不加入 DP，只回答以下问题：

### RQ1

Client synthetic guidance 的收益能否持续到更长的 50-round FL training？可能情况包括：

1. 只加速早期训练；
2. 加速训练并提高 50-round 最终 accuracy；
3. 早期有收益，但后期 no-guidance 追上；
4. 收益不稳定或消失。

### RQ2

Client-local LoRA 是否提供了 public pretrained LM 与 semantic category prompt 之外的额外 client/domain information？

### RQ3

Client synthetic、public synthetic、真实数据和 shuffled-label control 在初始 checkpoint 上的梯度方向与两 client 子空间有何区别？这些诊断是否支持端到端结果？

---

## 2. 实验配置

| 项目 | 配置 |
|---|---|
| 数据集 | AG News，4 类，FedNLP H5 数据与 partition |
| Source clients | 1、21 |
| Same-source real | 来自 source clients 1、21 |
| Held-out real clients | 800–807 |
| Fixed development clients | 900–908 |
| Development set | 每类 128，共 512，类别平衡 |
| Seeds | 57、58、59 |
| 下游模型 | DistilBERT + adapter |
| FL 算法 | FedFwd |
| FL rounds | 50，rounds 0–49 |
| Evaluation | 训练前评估一次，随后每轮评估；每个 arm 共 51 点 |
| Logical clients | 2；每轮两个 clients 全部参与 |
| Task batch size | train 8，eval 8 |
| Task max sequence length | 64 |
| Generator | DistilGPT2 |
| Client generator adaptation | Client-local LoRA，5 epochs |
| Generator prompt | `Category: {label}\nNews article:\n` |
| Generator max training length | 192 |
| Generator max new tokens | 80 |
| Guidance sample budget | 每类 32，共 128，类别平衡 |
| Guidance coefficient | Guidance arms `alpha=0.5`；no-guidance `alpha=1.0` |
| FedFwd directions | `v_num=1`，`pool_size=1`，无 adaptive retry |
| Client objective-query budget | 每个 arm 3000，经 cross-arm validator 核验 |
| GPU | 一张 A40 48 GB；三个 seed jobs 并发运行 |
| DP | 完全没有：noise multiplier=0，无 clipping，无 epsilon accountant |

本实验只比较准确率和优化行为，不比较 wall-clock speed。原 KDD 实验使用 4 张 A40，而 v3 使用 1 张 A40；本报告的结论来自 v3 内部配对比较，不声称和 KDD 版本具有相同吞吐性能。

---

## 3. 六个实验 arms

### 3.1 No guidance

- `alpha=1.0`。
- 每轮跳过 cloud backward pass。
- 保留相同通信与同步流程。
- cloud artifact 只作为 provenance 记录，不参与更新。

### 3.2 Client-LoRA synthetic guidance

- 每个 source client 使用自己的私有训练记录训练 local LoRA generator。
- 使用 LoRA-adapted DistilGPT2 生成 client synthetic samples。
- 下游使用正确类别标签。

### 3.3 Public-synthetic guidance

该 baseline 与 client synthetic 保持以下条件一致：

- 相同 pretrained DistilGPT2 base；
- 相同类别语义 prompt；
- 相同 decoding 参数、长度和 seed/quota 结构；
- 相同 sample budget、class balance 和 filtering rules；
- 相同下游训练预算和 guidance coefficient。

唯一核心差异：

- public generator 不构造 LoRA；
- 不运行 generator optimizer；
- optimizer steps=0；
- client private records 不用于 generator training。

为了执行与 client synthetic 相同的 exact-private-match release filter，public 路径会读取 source records 进行字符串匹配，但三个 seeds 的 public/client exact-match reject 都为 0，因此这个过滤步骤没有实际改变 public samples。

### 3.4 Same-source real guidance

- 使用 source clients 1、21 的真实私有记录。
- 总量、每类数量和标签序列与 synthetic guidance 匹配。
- 这是不可部署的 oracle，只用于同来源真实信号参照。

### 3.5 Held-out real guidance

- 使用保留 clients 800–807 的真实记录。
- 总量和 class balance 与 synthetic guidance 匹配。

### 3.6 Shuffled-label synthetic guidance

- 使用与 client synthetic 完全相同的文本序列。
- 保持相同标签直方图。
- 只对标签对应关系进行确定性置换。
- 用于判断收益是否依赖正确的文本—类别语义关系。

---

## 4. 最终结果

第 49 轮代表完成 50 次 FL 更新后的终点。

| Seed | No guidance | Client synthetic | Public synthetic | Same-source real | Held-out real | Shuffled-label |
|---:|---:|---:|---:|---:|---:|---:|
| 57 | 27.73% | **81.05%** | 42.58% | 81.25% | 76.95% | 25.39% |
| 58 | 36.52% | **75.98%** | 36.52% | 77.54% | 79.88% | 35.94% |
| 59 | 21.88% | **71.48%** | 38.09% | 79.69% | 79.69% | 21.09% |
| Mean | 28.71% | **76.17%** | 39.06% | 79.49% | 78.84% | 27.47% |
| Sample SD | 7.37 | 4.79 | 3.14 | 1.86 | 1.64 | 7.64 |

最终 macro-F1 均值：

| Arm | Final macro-F1 |
|---|---:|
| No guidance | 20.84% |
| Client synthetic | **75.41%** |
| Public synthetic | 34.07% |
| Same-source real | 78.98% |
| Held-out real | 78.49% |
| Shuffled-label | 16.24% |

accuracy 与 macro-F1 给出相同的总体排序。

---

## 5. 长程 trajectory

三个 seeds 的平均 accuracy checkpoints：

| Arm | Round 0 | Round 4 | Round 9 | Round 24 | Round 49 |
|---|---:|---:|---:|---:|---:|
| No guidance | 28.71% | 28.71% | 28.71% | 28.71% | 28.71% |
| Client synthetic | 29.49% | 43.36% | 55.73% | 70.96% | **76.17%** |
| Public synthetic | 28.26% | 29.36% | 31.97% | 36.20% | 39.06% |
| Same-source real | 30.40% | 51.63% | 66.34% | 76.63% | 79.49% |
| Held-out real | 30.53% | 46.29% | 63.74% | 75.39% | 78.84% |
| Shuffled-label | 28.32% | 27.67% | 27.67% | 27.41% | 27.47% |

窗口统计：

| Arm | Early rounds 0–9 | Late rounds 40–49 | Final | Rounds 0–49 mean |
|---|---:|---:|---:|---:|
| No guidance | 28.71% | 28.71% | 28.71% | 28.71% |
| Client synthetic | **43.46%** | **76.09%** | **76.17%** | **66.06%** |
| Public synthetic | 29.80% | 38.98% | 39.06% | 35.64% |
| Same-source real | 51.17% | 79.36% | 79.49% | 71.61% |
| Held-out real | 48.05% | 78.61% | 78.84% | 70.19% |
| Shuffled-label | 27.98% | 27.41% | 27.47% | 27.63% |

配对差值：

| Comparison | Early 0–9 | Late 40–49 | Final | Rounds 0–49 mean |
|---|---:|---:|---:|---:|
| Client − No guidance | +14.75 pp | **+47.38 pp** | **+47.46 pp** | +37.35 pp |
| Client − Public | +13.66 pp | **+37.11 pp** | **+37.11 pp** | +30.42 pp |
| Client − Shuffled | +15.48 pp | **+48.68 pp** | **+48.70 pp** | +38.43 pp |
| Public − No guidance | +1.09 pp | +10.27 pp | +10.35 pp | +6.93 pp |

Client − no-guidance 的最终逐 seed 差值：

- seed 57：+53.32 pp；
- seed 58：+39.45 pp；
- seed 59：+49.61 pp。

Client − public 的最终逐 seed 差值：

- seed 57：+38.48 pp；
- seed 58：+39.45 pp；
- seed 59：+33.40 pp。

Public − no-guidance 的最终逐 seed 差值：

- seed 57：+14.84 pp；
- seed 58：0.00 pp；
- seed 59：+16.21 pp。

因此 public prompt prior 有有限收益，但终点只在 2/3 seeds 上严格优于 no-guidance，且远小于 local-LoRA synthetic。

---

## 6. 学习速度

达到 50% accuracy 的第一轮：

| Arm | Seed 57 | Seed 58 | Seed 59 |
|---|---:|---:|---:|
| Client synthetic | 4 | 9 | 9 |
| Same-source real | 2 | 5 | 5 |
| Held-out real | 3 | 7 | 7 |
| Public synthetic | 未达到 | 未达到 | 未达到 |
| No guidance | 未达到 | 未达到 | 未达到 |
| Shuffled-label | 未达到 | 未达到 | 未达到 |

Public synthetic 只有 seed 57 达到 40% accuracy，发生在 round 29；seed 58/59 均未达到 40%。

---

## 7. 与真实 guidance 的差距

- Client synthetic 最终平均 76.17%。
- Same-source real 最终平均 79.49%，client synthetic 平均低 3.32 pp。
- Held-out real 最终平均 78.84%，client synthetic 平均低 2.67 pp。

逐 seed 比较并不完全一致：

- seed 57：client synthetic 比 held-out real 高 4.10 pp；
- seed 58：client synthetic 比 held-out real 低 3.91 pp；
- seed 59：client synthetic 比 held-out real 低 8.20 pp。

这说明 synthetic guidance 已接近相同预算真实 guidance 的量级，但 seed 稳定性和最弱 seed 的生成质量仍有改进空间。

---

## 8. Public-generator control 审计结果

三个 seeds 的 public validation manifests 均为 `complete`。以下 8 项全部通过：

1. client records 未用于 public model training；
2. filtering configuration 相同；
3. pretrained base model 相同；
4. prompt 与 decoding 相同；
5. public LoRA 未构造；
6. public optimizer steps=0；
7. sample budget 与 class balance 相同；
8. synthetic hashes 已验证。

三个 seeds 中，client synthetic 和 public synthetic 的 exact-private-match reject 都是 0。

---

## 9. Gradient diagnostics

这些诊断在相同的初始 DistilBERT checkpoint 上离线执行，不是多轮 FL 结果。

- 每种 guidance/source/held-out 使用 128 个平衡样本。
- diagnostic dev 使用 256 个平衡样本。
- trainable adapter parameters：450,340。
- 比较 aggregate gradient 与 source-real/dev gradient 的 signed cosine。
- 对每种 gradient 进行 equal-norm one-step update，观察 dev loss decrease。

三个 seeds 的均值：

| Gradient source | Cosine to source-real | Cosine to dev | Equal-norm one-step dev loss decrease |
|---|---:|---:|---:|
| Client synthetic | **0.8148** | **0.8030** | **0.006561** |
| Public synthetic | 0.6678 | 0.6352 | 0.005152 |
| Shuffled-label | 0.1682 | 0.1820 | 0.001430 |
| Held-out real | 0.8770 | 0.9129 | 0.007481 |
| Source real | 1.0000 | 0.9113 | 0.007435 |

Client synthetic 对 source-real 的 cosine 在每个 seed 上都高于 public synthetic：

- seed 57：0.7768 vs 0.6007；
- seed 58：0.8754 vs 0.7903；
- seed 59：0.7923 vs 0.6123。

值得注意：shuffled-label 在 seed 58 的 source cosine 为 0.5766，明显高于另外两个 seeds，但它的端到端训练仍失败。这表明单个 seed、单个 checkpoint 的 aggregate cosine 不能单独预测最终表现。

---

## 10. Two-client subspace diagnostics

对 source clients 1 和 21 的均值梯度构造 rank≤2 子空间，计算 source-real 与 dev gradient energy 的投影覆盖率。

| Two-client subspace | Source-real energy captured | Dev energy captured |
|---|---:|---:|
| Client synthetic | **0.6681** | **0.6478** |
| Public synthetic | 0.4539 | 0.4119 |
| Shuffled-label | 0.1318 | 0.1237 |
| Source real | 1.0000 | 0.8314 |

Client synthetic 与 public synthetic 的逐 seed source-real energy：

- seed 57：0.6100 vs 0.3619；
- seed 58：0.7664 vs 0.6246；
- seed 59：0.6279 vs 0.3752。

Client 与 public synthetic 子空间的 principal cosines：

- seed 57：`[0.7063, 0.1064]`；
- seed 58：`[0.8524, 0.1701]`；
- seed 59：`[0.6854, 0.1693]`。

可能的解释是：public prompt prior 和 local LoRA 共享一个较强方向，但第二方向接近正交；local LoRA 增加了 public generator 中缺失的 client/domain direction。不过该解释仍需更多 clients 和 checkpoints 验证。

---

## 11. 完整性与数值质量

- 18/18 arms 完成。
- 每个 arm 有 51 个评估点。
- 每个 arm 有 3000 次 client objective queries。
- 所有 cross-arm validation 状态为 `complete`，errors 为空。
- 三个 public-generator validation 状态为 `complete`。
- 三个 gradient/subspace diagnostics 状态为 `complete`。
- 本地重新汇总得到的 `summary.json` 和 `curves.csv` 与服务器输出逐字节一致。

长程 no-guidance 中出现少量 central finite-difference 零方向：

- seed 57：2/1500 directions 为零，非零率 99.8667%；
- seed 58：1/1500 为零，非零率 99.9333%；
- seed 59：2/1500 为零，非零率 99.8667%。

每次训练调用都至少有一个非零方向。其余五个 arms 的全部 seeds 都是 100% 非零。对应轮次没有删除，validator 显式记录了数量。

这几个零方向数量很少，似乎不足以解释 37–49 pp 的 guidance 差异；但 no-guidance accuracy 在整个 50 rounds 完全不变，而 loss 只轻微变化，因此仍应检查 local-only update 的数值尺度、finite-difference epsilon、学习率和 adapter 更新是否合理。

---

## 12. 当前结论

### 对 RQ1

在当前 50-round 窗口内，结果符合情形 2：

> Client synthetic guidance 同时加速训练，并提高 50-round 最终 accuracy；no-guidance 没有在后期追上。

理由：

- early client − no-guidance：+14.75 pp；
- late client − no-guidance：+47.38 pp；
- final：+47.46 pp；
- 三个 seeds 的 late 与 final 差值全部为正；
- 差距随训练扩大后平台化，没有收窄迹象。

这只能说明 50 rounds 内的持续性，不能证明无限长训练后的最终收敛点。

### 对 RQ2

结果支持 client-local LoRA 提供 public prompt prior 之外的额外信息：

- client synthetic 最终比 public synthetic 高 37.11 pp；
- 3/3 seeds 都为正；
- late-window 差同样为 37.11 pp；
- client gradient/source alignment 在 3/3 seeds 上更高；
- client two-client subspace 在 3/3 seeds 上捕获更多 source-real energy。

Public semantic prompt prior 并非完全无效：它比 no-guidance 平均高 10.35 pp，但只在 2/3 seeds 的最终点严格更高。因此 public prior 只能解释一部分收益。

### 对标签语义

Shuffled-label control 最终平均只有 27.47%，而正确标签 client synthetic 为 76.17%。相同文本本身不能解释收益，正确的文本—标签对应关系是必要条件。

---

## 13. 不能做的声明

当前结果不能支持：

- 统计显著性：只有 3 个 seeds，没有为假设检验设计样本量；
- 最终收敛：只训练了 50 rounds；
- 跨 partition、跨数据集、跨模型的普遍性；
- DP 可行性或任何隐私保证；
- local LoRA 不记忆私有文本；
- gradient/subspace diagnostic 单独证明多轮因果机制。

此外，development set 每轮重复评估，但没有基于 dev 选择 checkpoint 或调参；报告使用预先指定的 round 49、early 0–9 和 late 40–49。

---

## 14. 希望 GPT 重点分析的问题

请基于上述数据逐项回答：

1. RQ1 选择“加速并提高 50-round 最终 accuracy”是否充分？是否还有其他合理解释？
2. Public-synthetic baseline 是否足以隔离 client-local LoRA 的贡献？还存在哪些混杂因素？
3. No-guidance accuracy 50 rounds 完全不变是否可疑？需要增加哪些数值或实现检查？
4. 极少数 no-guidance finite-difference 零方向是否可能实质影响结论？
5. Gradient cosine、equal-norm one-step loss 和 rank-2 subspace coverage 应如何解释？哪些表述可能过强？
6. Client synthetic 接近真实 guidance，但 seed 59 仍落后 8.20 pp。最可能的生成质量问题是什么？
7. 只有 3 个 seeds 时，论文正文应如何表述结果，避免错误声称统计显著？
8. 下一轮最重要的 Non-DP 验证是什么：更多 seeds、更多 clients/partitions、更长 rounds、不同 synthetic budget，还是生成质量消融？请排序并说明理由。
9. 在 Non-DP 机制验证充分之后，如何设计最小但有说服力的 DP privacy–utility 实验？这里只需要方案，不要把当前结果解释成 DP 结果。
10. 请给出适合 journal paper 的 claims、limitations 和 ablation 段落草稿。

请明确区分：

- 数据直接支持的结论；
- 合理但尚未证明的机制推断；
- 需要新实验才能回答的问题。

---

## 15. 可选附件

如果需要进一步检查，可同时提供：

- `summary.json`：逐 seed、逐 arm、paired 指标和 diagnostics；
- `curves.csv`：18 个 runs、918 个 evaluation records；
- `v3_training_curves.png` 或 PDF；
- 原始日志与 manifests 的 `agnews_nondp_v3_results.tar.gz`。

本报告对应的是 **Non-DP v3**，没有运行任何 DP-SGD、noise 或 epsilon scan。
