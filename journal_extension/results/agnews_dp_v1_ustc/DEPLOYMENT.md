# AG News 校内 A40 续跑记录

更新时间：2026-09-28 17:40（北京时间）。这是部署和启动记录；正式效用结果以同目录 `REPORT.md`、`summary.json` 和 `curves.csv` 为准。

## 当前状态

- 已从 GitHub `HeimdaLLM+` 分支部署，并完成独立环境配置。
- 小规模完整流程验证已经通过。
- 正式队列于 **2026-09-28 17:39:16 +08:00** 启动，首先运行 ε=8。
- 此时正式完成数为 **0/12**；不从部署验证的准确率推断 DP 效用。
- 原租用服务器已释放。之前的 Non-DP v3 报告、918 条评估记录和本地原始结果备份仍可用；此前未上传的租用服务器 DP 输出不纳入本轮分析。

## 代码与数据

本次实际实验代码冻结在提交：

`590a20e161c664d4d5ced292dac2ef7822433030`

这是最新 GitHub 实验代码加上已验证部署修复后的版本。后续报告提交可以更新分支 HEAD，正在运行的服务器代码保持上述提交。

AG News 两个源文件与原租用实验的 SHA256 一致：

| 文件 | SHA256 |
|---|---|
| `agnews_data.h5` | `1474dd264d535b50ad2a7b1cf714bf11e182c2fe5b07f5e982feb31a9b114616` |
| `agnews_partition.h5` | `fd6196bc2518457856365ec8f8207b1eb4b278d165a829320fc5b9fdd0a67f12` |

DistilGPT2 的 `config.json`、`generation_config.json`、`tokenizer.json` 和 `tokenizer_config.json` 哈希与 v3 原始生成 manifest 相同。DistilBERT 使用服务器已有的本地 pretrained 模型。

## 运行环境

| 项目 | 本轮配置 |
|---|---|
| 系统 | Ubuntu 22.04.5 LTS |
| Python | 3.9.23，两个独立 venv，禁用 user site-packages |
| GPU | 物理 GPU1，NVIDIA A40；与其他用户共享计算资源 |
| PyTorch / CUDA runtime | 1.13.1+cu116 / 11.6 |
| FL 环境 | adapter-transformers 3.1.0、functorch 1.13.1、mpi4py 3.1.6、protobuf 3.20.3 |
| 生成环境 | transformers 4.35.0、PEFT 0.6.1、accelerate 0.24.1、Opacus 1.4.0 |
| MPI | 系统 OpenMPI 4.1.2 |

两个完整依赖清单 `kdd-freeze.txt` 和 `ton-freeze.txt` 已保存到服务器和本地备份。相较原租用环境，操作系统与 Python 版本有所变化；与 v3 的比较按分析计划作描述性解释。

部署修复已经提交到 GitHub：固定旧版 FedML 的 protobuf，固定 mpi4py 构建工具，使用系统编译器/链接器编译 OpenMPI 扩展；PyTorch 镜像文件通过官方索引 SHA256 校验后安装。没有更改其他用户或现有 Conda 环境。

## 小规模验证

运行 ID：`ustc_agnews_dp_smoke_20260928`。

- seed=57；两个 client，每个 client 8 条生成器训练记录，1 epoch。
- 四个类别各生成 4 条，共 16 条 synthetic records。
- ε target=8，δ=1e-5；实际 accountant ε=**7.991205646002187**。
- 使用独立私有随机性进行 Poisson sampling 和 Gaussian noise。
- 1 FL round；128 条平衡 dev records；评估 round=-1 和 round=0。
- 生成发布校验、所有预期评估、最终评估、正常退出、唯一完成标记等检查全部通过。
- GPU mapping 为 `amax: [0, 3, 0, 0]`，三个 MPI ranks 都在 GPU1；实际总显存约 9.5 GiB。

该结果仅用于验证部署，不计入正式效用表。原始归档已经在服务器和本地保存，并验证 SHA256：

`96aef942863dc47ae2076304e0c2a2c7ff69657c41759046fd3efadeca3280a9`

## 正式实验

采用已有 `AGNEWS_DP_V1_PLAN.md` 中的固定协议：

- ε targets：8、4、2、1，顺序执行；δ=1e-5。
- 每个 ε 使用 seeds 57、58、59，共 12 次正式运行。
- clients 1 和 21，各 120 条训练记录；各自独立训练 DistilGPT2 LoRA，5 epochs。
- 四个类别各 32 条 synthetic records，总数 128；保持语义类别提示与既有解码参数。
- 每次 50 FL rounds；训练前评估一次，每轮评估一次；固定 512 条 dev records。
- 只使用物理 GPU1；共享 GPU 的耗时不作为方法性能指标。

隐私范围是生成器发布的合成数据。下游 FL 更新保持 Non-DP，不声称完整 FL 通信的端到端 DP。DP 采样与噪声随机性不再由公开 seed 派生；使用 PyTorch PRNG 的研究实现，不声称密码学部署。

队列启动命令：

```bash
cd /home/zzkevin/heimdallm-journal-20260928/HeimdaLLM
nohup env -u DISPLAY bash journal_extension/run_ustc_agnews_dp_grid.sh \
  > /home/zzkevin/heimdallm-journal-20260928/logs/ustc-agnews-dp-grid.log \
  2>&1 < /dev/null &
```

启动时队列 PID 为 `1547303`。每个完成的 ε 条件会在服务器归档并生成校验值，然后更新汇总报告。

## 保存位置与后续检查

服务器工作根目录：`/home/zzkevin/heimdallm-journal-20260928`。

| 内容 | 服务器相对路径 |
|---|---|
| 代码 | `HeimdaLLM/` |
| 每个 ε 的结果 | `results/ustc_agnews_dp_eps{epsilon}_20260928/` |
| 汇总报告 | `results/agnews_dp_v1_ustc_summary/` |
| 队列日志 | `logs/ustc-agnews-dp-grid.log` |
| 分条件日志 | `logs/ustc_agnews_dp_eps{epsilon}_20260928.log` |
| 压缩归档及 SHA256 | `backups/` |

本地备份目录：`ToN26/local_backups/agnews_dp_v1_ustc_20260928/`，已经包含代码快照、依赖清单、部署验证原始归档和进度报告。

本地 `sync_ustc_agnews_results.py --publish --watch` 已启动，每 120 秒检查一次。完成条件的原始归档校验后保存到本地；只有报告、汇总统计和训练曲线上传 GitHub。密码、模型 checkpoint、原始训练文本和 private staging 不上传 GitHub。

自动本地同步要求电脑保持唤醒、VPN 连通。断开期间服务器实验与服务器归档仍继续；恢复后可以重新运行同步脚本收集已完成结果。当前 source code archive 的 SHA256 为：

`034057f853955ced9a9124a7ebf1d01ddb82c2286e850541e0fcfeb02d392da4`
