# E0 正确性、协议与复用审计（受阻时的本地部分）

日期：2026-10-14。仓库 `HeimdaLLM-github` 当前分支 `HeimdaLLM+`，HEAD `c1b238c9f48eb73392e0e5d8e6af57cffb462830`；审计开始时工作树清洁。本目录为新建交付，不覆盖旧结果。审计脚本是 `audit_local.py`，重跑命令为 `python3 stage1_agnews_20261014_c1b238c/audit_local.py`（从工作区根目录）。

## 服务器与完成边界

VPN 后 SSH 可以到达原校内 A40，密码交互出现，但输入用户再次确认的认证信息后，服务端立即关闭会话，返回 255。批处理公钥登录也被拒绝。无法进入服务器核验当前仓库、源 AG News 文件、权限、作业、GPU 利用率或 ≥14000 MiB 空闲显存门槛。没有启动新训练、DP 发布或单步 PyTorch 测试；未降低预检门槛。服务端断开的具体原因未知，不能归因为 GPU、VPN 空闲或密码错误中的任一种。恢复时先确认原 A40 SSH 可进入 shell；再检查 `git status`, `git rev-parse HEAD`, `nvidia-smi`, 源数据 SHA 与共享 GPU 审批，然后继续 E0。后续 10 月 15 日状态见 `RESUME_CHECK_20261015.md`。

## 本地历史工件

三个原始 `.tar.gz` 均通过同目录 SHA256 sidecar；内部 `metrics.json` 状态为 complete、六项完成检查均为 true、退出码 0，分别有更新前及 50 次更新后共 51 个开发集评估点。`REUSE_MANIFEST.json` 记录三个归档和具体哈希。历史命令均使用同一 DistilBERT adapter + FedFwd 设置：2 个逻辑客户端（1、21），各 120 条（由历史协议报告，源文件待核）；`pilot_uniform_2`，`alpha=.5`, `beta=1`, `pool_size=1`, `zo_estimator=legacy`, `fd_step=.01`, `learning_rate=.01`, batch 8, seed 57, 50 rounds，每轮评估。三者只是云端指导源不同。固定分区归档文件字节哈希一致；源数据和源 partition 仅有历史记录的哈希，尚不能本地从原始文件重新计算。历史运行的完整服务器提交号未保存在三份 metrics 内，当前代码相容性是静态核对，不等于现场复现。

已冻结的公开候选池 SHA256 为 `7e54c05ce2dc94d754b178218963dedfceb781c491b97ed1d9db5ab9e7d7c54f`，128 条、4 类每类 32、归一化后 128 条唯一。按 `select_public_qwen_candidates.py` 与 seed 57 重新计算公开随机集合，和历史选择字节一致。公开 NLI 选择的 64 条（每类 16；每客户端每类 8）可从已存 `rank_summary` 逐条重建，SHA256 为 `52810c8290a47533048d82a33bc92279c50d4df77f3628a1539031ba2905194a`；它尚无下游训练结果。NLI 的辅助分类器 59/64=92.1875% 类别同意率只是代理指标。

原始候选生成器 `augpe_nondp_pilot.py` 有使用私有文本的 `forbidden` 精确重合过滤，因此该通用生成路径并非天然公开独立。已有 `replay_public_qwen_pool.py` 的无私有访问重放报告给出当前**这一池**同 SHA；本次直接核对了归档池及其选择，但服务器上的重放环境和原始数据无法再次查验。后续 DP 应显式使用冻结的公开重放池，且禁止私有依赖再生成。NLI 选择器读取公开池和公开模型，其冻结版本见选择 manifest；不能用开发集结果再挑 prompt。

三份 cloud pack 均为 64 槽、每类 16、64 个唯一文本、`duplicates_dropped=0`、索引唯一、无云端去重；私有及随机指导文本都来自同一公开池，真实指导则来自源客户端记录。历史训练日志可数得每 run 1500 个 finite-difference 方向、3000 次客户端正负目标函数查询，另有 50 次 cloud BP 和 400 个 cloud BP batch 日志（8 batch/round × batch 8，名义 3200 样本访问）；这是历史日志计数，不是新实测。开发集 512 条，归档报告未读官方 test。64 是槽位预算，不应与旧 v3 的 128 指导预算合并。

## 实际更新路径的静态追踪

- `forward_training/tc_transformer_trainer_distribute.py` 在当前代码构造 `z_h=β√(α/d)z+β√((1−α)/pool_size)p_cloud`；历史 `d=450340`, `α=.5`, `β=1`, `pool_size=1`。云方向归一化，全局 1 个 rank-1 方向；当前 wire format 对其他 pool size 抛错。它不是可直接声称的多列 V 子空间。
- `forward_training/utils/fwdgrad_utils.py:67` 用 FP32 关闭 autocast 的中心差分 `(L(w+hz_h)−L(w−hz_h))/(2h)`；`h=.01`。CrossEntropy 默认 mean，模型评估路径禁用 dropout。客户端累加 `jvp*z_h`，legacy 乘数为 1；旧无指导 `isotropic` 修正后的实验不能与这些历史 guided 臂说成仅指导源差异。
- `FedML/fedml_api/distributed/fedsgd/FedSgdClient.py:49` 返回完整梯度张量列表，`FedSgdClientManager.py:184` 通过模型参数消息字段发送。当前实验实现不能描述为“只上传标量损失”的 FL transcript。`FedSgdServer.py:120` 附近虽然注释写按样本数加权，实际循环 `averaged_grads[j] += g`，再除以 `training_num`；学习率还有 warmup/decay ratio。若把 per-client `g` 当平均梯度，这不是标准样本量加权平均。需要现场单步检验其实际尺度和历史语义，不能静默修正后沿用旧结果。
- `FedSgdCloudManager.py` 的可见路径发送 cloud perturbation，未见 cloud BP 对全局正式模型直接执行权重更新；但无同 checkpoint 的参数 delta 检查，尚不能标“实际更新算子已验证”。历史启动命令没有 `--audit_updates`，原始日志无可靠的逐步参数 delta norm。

缺失的 E0 动态核验：相同 checkpoint、batch、seed 的光滑目标方向导数、客户端/服务端一致性、实际 `h z_h` 和 loss 差、更新 delta/舍入、缓存/清零、cloud 权重混入；云端源数据索引、tokenizer revision 和前后固定 checkpoint 也待核。当前本地 Python 环境没有可运行该仓库训练的 PyTorch/模型/数据组合。仅有静态审计和归档重算，不能宣布 E0 通过。

## 复用判定

`PRIVATE_ZERO`, `PUB_RANDOM`, `REAL_MATCHED` 的 seed 57 在本地校验和、训练 flags、预算和开发集轨迹上匹配，可**暂列为历史待现场确认复用**；不是新完成任务，也没有正式论文级跨 seed 结论。其余 seed 58/59 与 PUB_NLI 57、DP 三次都没有可复用终点。复用前须确认服务器仓库代码/原始数据/分区、单步更新语义、模型和 tokenizer 工件以及当前共享 GPU 条件。`cross_arm_validation.json` 如实标记 partial。
