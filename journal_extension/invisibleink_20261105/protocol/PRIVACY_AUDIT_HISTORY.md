# 隐私机制与实现核对

采用 [InvisibleInk 原论文](https://proceedings.nips.cc/paper_files/paper/2025/file/61432775ec681f7ac6e7d335fa933e4f-Paper-Conference.pdf) 的差分 logits 裁剪及公开支持采样；[官方代码](https://github.com/cerai-iitm/invisibleink) 固定在 `2141ee265094612e85202b9e2745962b548b11b5`。上游归档与 GPL-3.0 许可证随恢复包保留。这里是经审查的接口适配，不是原包未经修改的执行。

## 数学保证（精确算术及理想采样条件下）

邻接定义为固定客户端、固定记录槽位数下替换一个完整记录（含标签和文本）。B=3，固定公共前缀和已生成前缀下，令公开 logits 为 p，私有参考 logits 为 l_j；分数是 `s=p+mean_j clip(l_j-p,[-C,C])`。单个替换只能改变一个参考行，因此每个坐标变化最多 2C/B。

支持集仅由公开模型、公开目标类别、同一输出的已发布前缀及固定 C/B 决定。给定前缀，相邻输入的支持相同。softmax(s/tau) 的隐私损失区间长度最多 `4C/(B*tau)`。有界区间指数机制每 token 的 zCDP 参数为 `2*C^2/(B^2*tau^2)`。自适应组合最多 T=96 个 token 得到 `rho=2*T*C^2/(B^2*tau^2)`。设 L=log(1/delta)，采用保守转换 `epsilon=rho+2*sqrt(rho*L)`，固定 epsilon=8、delta=1e-5、tau=1 求 C。

这一替换邻接的 rho 是上游 replace-by-null 同 C 公式的四倍；没有直接套用上游 epsilon。32 个组不相交，两个客户端的记录槽位不相交，输出之间不传递私有上下文，故一个记录只影响一个输出组，可用并行组合。不声称抽样放大。未使用的后 24 个槽位不参与生成。不得私下追加/重试一份 DP 输出；如将来追加，需单独组合预算。

固定数据规模、公开类别、固定分组、公开预训练模型与固定超参数属于条件。公开 pretrained prior 是否已记忆 AG News 不是这里能排除的；不声称解决预训练泄漏。只保护对本次参考记录访问的增量生成通道。

## 实现对应与修正

- `mechanism.py::scores`：float64 上做差分裁剪；`support` 与上游纯函数比较，B 明确为私有参考数。
- `generate.py::prompt`：每个私有记录单独构造上下文；标签不会用于分组。公开输出类别由索引 i%4 决定。固定截断 160 个参考 token，不从私有长度估计输出预算。
- 每个参考单独前向，独立 KV cache；同一输出的生成前缀共享。公开行不和私有行一起 padding。model.eval，不微调权重。
- 官方包的自动长度、空串删除和公开 RNG seed 不启用。DP 使用新 OS 随机位，不写 seed/RNG state；PUBLIC/NONPRIVATE 的 5703 是普通可复现采样 seed。
- 固定 96-token 上限用于整个账本，EOS 提前停止不降低声明预算。解码、空文本的公开回退、序列化是输出后处理，不基于原始数据过滤。
- 环境通过 `PYTHONNOUSERSITE=1` 避免用户包覆盖；生成与旧 FL 分别复用隔离环境。全程只使用已有训练/开发备份。

## 实际审查状态和限制

CPU 测试检查邻接敏感度、共同支持、模拟 Rényi 上界、预算反解和上游纯函数一致性；这些测试不替代证明。审查后可执行受限可行性试验。

模型 BF16、logit 变换 float64、有限随机位和浮点 softmax 属于研究实现，尚无覆盖所有数值误差/下溢的机器级 DP 认证。理想数学 epsilon=8 不应被写成无条件的生产安全认证。GPU 时间、内存、异常、内部日志和 Non-DP 对照均保留在受限目录，不属于对外 DP 输出接口。失败时停下，不以更好结果为目标重采样。真实客户端运行的权限隔离和故障侧信道还需生产审查。

## 证据范围

这次小试回答固定条件下生成指导是否值得继续；不能只凭语言流畅、标签语义一致或 cosine 推导下游准确率。最终以相同开发集上的完整 50-round 轨迹比较。无新结果前不改论文。

## Multi-budget extension, 2026-11-03

Same full-record replacement adjacency, public fixed slots and unchanged pure mechanism. New epsilon=4/16/32, delta=1e-5; reuse epsilon=8. Clip thresholds approximately0.11812/0.40101/0.69355. Top-k stays100; the public support threshold changes with C/B by the unchanged mechanism. T=96, B=3 and tau=1 stay fixed.

A record reused across epsilon releases requires sequential composition. Parallel composition within each disjoint grouping does not apply across releases. The four DP artifacts together have ideal joint epsilon bound approximately41.36 at delta1e-5. This is not a user-approved deployment budget. It does not apply to research archives containing NONPRIVATE text, raw training states or internal statistics. Partial failed releases are recorded conservatively and never rerolled.

The two FL seeds reuse each artifact and cause no further synthetic generation release. FL still reads raw private records, so model/metric releases are not covered as postprocessing of generation. All artifacts remain restricted, no automatic publication or Git push.

## Superseding active scope, 2026-11-04

This scope replaces the uncompleted4/16/32 plan. The active curve is4/8/12; reuse immutable epsilon4 and8 artifacts, and generate epsilon12 exactly once. Original epsilon16 partial28, paused epsilon4 FL logs, and previous frozen protocols remain intact. No epsilon2/16/32 generation is authorized in this stage. epsilon12 clip is approximately0.3152208242462301 with the same B3, T96, tau1 and delta1e-5. All full-release and historical-attempt accounting must be distinguished; no guarantee applies to the mixed raw research archive.

## Active authorized extension,2026-11-04 after dense completion

The6-point exploratory curve uses1/4/8/12/16/20. Only1/16/20 are new; all prior complete artifacts and endpoints are immutable and reused. The new epsilon16 is one fresh complete attempt after the old28-record interruption; both attempts remain in the conservative ledger. No generator, prompt, sampler, dataset partition, logical-client count, guide budget, or downstream operator change is permitted. Epsilon changes only calibrated C and the original public support rule. The observed accuracy need not be monotone. Each budget has one generation artifact and two FL seeds; no independent-generation uncertainty estimate is claimed.
