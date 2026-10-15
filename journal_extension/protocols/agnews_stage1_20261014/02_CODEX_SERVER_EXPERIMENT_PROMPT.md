# 给 Codex：执行 HeimdaLLM+ 扩刊第一阶段服务器实验

## 你的任务

你是本项目的实验执行与验证负责人。请**读代码、审计、实际执行已授权实验、校验、归档、交接证据**，不是只写一份建议或生成未执行脚本。

本轮只完成“固定公开候选池的强公开对照＋单个正式DP预算”这一最小闭环。不要自主继续更换生成器或扩大研究范围。与本文件同包的 `03_EXPERIMENT_PLAN.md` 给出完整协议；请先读它。本文的关键约束也独立写在下面。

论文的新增贡献定位是：面向原HeimdaLLM云端BP/GGD的隐私合成指导供给及其分析。保留原ZGR和SPC/DTC；不研究全新的通用生成模型、不再新造系统控制器、不将下游任务改成LoRA。

## 1. 首先读取真实状态，不能把历史待办当作当前事实

发现起点：

- 仓库：`ustc-sunny/HeimdaLLM`；扩刊分支名：`HeimdaLLM+`。
- 最近用户报告的快照提交：`c1b238c`，**不是强制reset/checkout的目标**。
- 服务器历史根目录：`/home/zzkevin/heimdallm-journal-20260928/`。
- 本地可能路径：`HeimdaLLM-github/journal_extension/`。
- 用户最新报告：`HEIMDALLM_PLUS_RESEARCH_REVIEW_20261014.md`（任务包inputs内有副本）。

请读当前工作区的AGENTS.md/README、git状态、现有SSH配置和实验文档，使用已经获授权的连接。不要猜密码、读取无关凭据或新租服务器。不打印密钥/凭据。不覆盖未提交修改。

优先找这些既有报告和配置：

- `journal_extension/results/agnews_private_vote_pilot_ustc/REPORT.md`
- `journal_extension/results/agnews_qwen_public_control_ustc/REPORT.md`
- `journal_extension/results/agnews_qwen_nli_control_ustc/PREPARATION.md`
- `journal_extension/results/agnews_nondp_v3/REPORT.md`
- `journal_extension/results/agnews_matched_ablation_ustc/formal_REPORT.md`
- `journal_extension/results/agnews_dp_noise_bridge_ustc/REPORT.md`
- 相关原始protocol、run spec、manifest、curves、训练日志和源码。

若现状比用户报告更新，以可核验的新工件为准，并记录差别。禁止用同名报告就认定同版本；读commit、配置与hash。

### 禁止的操作

不reset仓库、不清理别人文件、不覆盖旧结果、不杀无关作业、不降低共享GPU预检门槛、不绕过审批、不购买资源、不自动push、不自动启动其他阶段。

若服务器认证/权限/资源被阻断，输出具体blocked原因和可恢复命令；不能说训练已完成。沿用当前批准的资源策略。历史14000MiB门槛不是已测内存，更不是你可以自行调整的参数。

## 2. 固定范围：最多15个目标终点，不开大网格

目标arm×FL seed：

| Arm | seeds | 说明 |
|---|---|---|
| PUB_RANDOM | 57/58/59 | 同一公开Qwen池，既有公开确定性随机选择 |
| PUB_NLI | 57/58/59 | 同一池，已准备的公开NLI选择 |
| PRIVATE_ZERO | 57/58/59 | 当前无噪声私有投票 |
| REAL_MATCHED | 57/58/59 | 当前同源真实指导 |
| PRIVATE_DP_E4 | 57/58/59 | 同一私有投票管线，机制审计后一次性ε=4加噪 |

共15个完整下游终点；如果seed57前三个已完成结果匹配且可复用，预计新增12个。已有更多完成结果则更少。旧结果不兼容时最多重跑此15个目标；最多额外2次技术故障重试，全局计数。不得为更高accuracy重试。

不自动执行：全epsilon网格、增加seeds、不同候选池、新客户端组、官方测试集、新模型、新生成器、DP-LoRA大扫参、聚类、概率采样、在线刷新或自适应alpha。

本輪保留现有Qwen池是为保持配对，不认定Qwen最佳；不把Qwen或NLI加载到客户端作为新要求。

## 3. E0：先完成正确性、协议与复用审计

保存当前HEAD、工作树差异、环境、所有相关脚本和配置。输出：

- `CORRECTNESS_AND_PROTOCOL_AUDIT.md`
- `REUSE_MANIFEST.json`
- `FROZEN_PROTOCOL.json`

### 3.1 实际更新算子

逐项追踪trainable parameters、finite-difference epsilon、方向norm/协方差、梯度恢复系数、optimizer、loss reduction、sync/aggregation、guidance interval。

明确旧无指导估计器修正影响哪些实验臂。不同更新版本不能在一张主表中被描述为只改变指导数据。

用相同checkpoint/batch/seed做单步检查：

1. 微型光滑目标上，有限差分是否近似BP方向导数；不要错误要求单次ZO梯度等于完整BP梯度。
2. 相同方向、loss和seed下，客户端/聚合器重建是否一致。
3. 记录实际扰动、loss差、估计梯度和参数delta的norm；检查极小更新、dtype舍入、缓存梯度、未清零梯度。
4. 检查cloud BP是否通过未声明路径直接更新正式任务模型；guided方向的合法影响与意外权重混入要区分。
5. 诊断副本不能修改正式轨迹、RNG、optimizer或checkpoint。

若发现改变语义的错误，先修复并单元测试，记录受影响历史结果，不选择性只重跑有利臂。若修复会超出此15任务矩阵或改变整个研究算法，停止执行并请求作者确认。

### 3.2 数据和计算预算

核对pool的公开性、类别映射、文本hash、token长度、source/dev/reference索引、数据处理与顺序。

分别报告64槽位、unique texts、跨客户端重复、重复权重、cloud去重、实际cloud BP样本访问数。不得用“都是64条”跳过这些区别。

本轮保留已定义选择器：PRIVATE_ZERO为每客户端每类top8；PUB_NLI为每类全局top16再交错分配。若它们自然产生不同唯一数，报告这一方法差异，不偷偷重采使其相等。

固定同seed任务初始化、client schedule、client/云端mini-batch规则、50次更新和目标函数查询预算。若新正确协议的query数不是旧报告3000，以真实冻结定义为准，不伪造3000；所有比较臂必须一致。

## 4. E1：完成四臂配对训练

默认沿用已经核验的协议：

- source clients1、21，各120条记录；当前AGNews数据和partition，以文件核验；
- 同一128候选池（每类32）；无私有记录参与生成或公开筛选；
- 64指导槽位，每类16、每客户端每类8；
- 当前DistilBERT adapter＋FedFwd，guidance alpha0.5、LR0.01（以真实冻结配置校验）；
- 50次更新，训练前一次评估，再每次更新后评估；
- 固定512条平衡开发集，不读取官方测试集；
- seeds57、58、59仅改变下游训练随机性，pool和各Non-DP指导集合保持固定。

先完成缺失PUB_NLI的seed57；然后补齐58/59四臂。不能根据NLI单次结果停止报告其余预定重复，也不能调NLI prompt直到胜出。

优先复用可证明相同的已完成结果，不按文件名判断。每个run独立输出目录，完整run spec和hash；使用唯一run_id，禁止覆盖。

## 5. E2：与PRIVATE_ZERO严格匹配的DP选择

### 5.1 隐私定义与数学机制

写出 `PRIVATE_GUIDANCE_RELEASE_SPEC.md` 和 `DP_IMPLEMENTATION_REVIEW.md`。

本轮邻接：固定客户端名册，替换某客户端一条完整记录，含文本与标签；不改变总记录数，不保护客户端是否参与，不是客户端级DP。

固定候选池、公开编码器和每条记录独立映射。若真实规则是一记录一张票：

- h_i是全部128个候选的整数直方图，跨4类；
- replace-one全局L1敏感度≤2；
- 一次性每坐标加独立Laplace噪声，scale=2/epsilon=0.5；
- 在同一个带噪h_i上逐类选top8；
- 发布候选ID，cloud恢复公开文本；噪声前分数和逐记录匹配不出客户端信任边界。

不做softmax/probability sampling、类别自适应配额、私有去重、数据依赖拒绝或新的reranking。负带噪分数保持参与排序；公开ID确定性tie-break。输出顺序按冻结公开规则规范化，不能用原始私有排名另行排序泄露。

**若当前规则是加权/多票或依赖整批数据的归一化，不得套用敏感度2。先证明真实贡献界并明确版本差异；需要改变PRIVATE_ZERO语义时停止DP训练，不能默默换方法。**

### 5.2 审查条件

候选和公开NLI独立于目标数据；Private-Zero仅为Non-DP控制，不得声称隐私。敏感度测试至少含跨标签替换、缺失本地类别、并列、重复记录。对整个机制核对发布对象和任何metadata、失败返回、过滤、日志路径。

用合适的经审查噪声采样实现，分别说明理想数学保证与浮点/随机数实现假设。不能通过几个neighbor Monte Carlo测试就宣布已证明DP；形式化贡献界和数据流分析必不可少。数值实现不能支撑声明时，设置 `privacy_status=reference_only`，报告不足，不伪造 verified。

不在公开run spec里存可恢复噪声的seed。使用独立安全随机源，和模型/selection公共RNG隔离。保留受限重现工件或密钥管理记录，不发布原始votes、私有文本、可逆日志。公开trial_id不是noise seed。

### 5.3 预算与重复

每个FL seed对应一个独立DP工件：D01/D02/D03；每个客户端一次ε4发布。两个客户端记录不交叉时，对“一处记录改变”的联合一次发布采用并行组合；相同客户端的3个独立工件若均发布，基本联合预算为12。准确记账，不把3次独立输出总称ε4。

现有下游FL更新、真实指导和Non-DP实验工件不包含在这条保证内。不要把公共AGNews分片的研究模拟说成真实用户全transcript已受保护。

只有机制审查通过才运行3个DP终点；不要先看噪声后效用再挑release。用同一非噪声步骤保证PRIVATE_DP_E4的noise=0模式与PRIVATE_ZERO字节级一致。

一旦3个DP任务结束，停止；不因结果差而自动提高epsilon或加样本。

## 6. 指标、诊断、资源测量

主终点updates_completed=50；报告accuracy、macro-F1、loss、最后10次更新均值、normalized trapezoidal AUC和达到50/60/70%的首次更新数及queries。未到标right_censored，不能按50替代。

给出每seed配对差；探索性均值/SD/区间的单位是run，不是51轮。3个DP trials混合了DP噪声与训练随机性，不能分开估计其方差。

保存per-class precision/recall/F1和混淆矩阵；特别检验Business↔Science/Technology。辅助AGNews分类器同意率只是proxy。

诊断使用每seed REAL_MATCHED轨迹的同一checkpoint（更新前/10/50），所有指导来源在相同w上计算：gradient norm/cosine/relative error及实际可提取的ZGR量。不能对各臂不同w直接解释“数据梯度差”；不能用相同z_g把不同basis的GGD cosine当basis-invariant度量；rank≤2诊断明确标代理。

组件测量包括：云端生成整个128池、加载/缓存、候选下发、端侧编码/投票/加噪/topk、索引上传、cloud BP。记录逻辑执行位置和实测设备，真实设备不可用如实标注。共享GPU耗时不用于公平速度结论，不用门槛当内存。避免重跑生成只为计时；历史成本不可获得可标未测。

## 7. 实验与论文的交接

创建 `paper_handoff/`，至少输出：

1. `PAPER_EVIDENCE_MANIFEST.json`：参照任务包 `06_PAPER_EVIDENCE_MANIFEST_TEMPLATE.json`；每个数字与claim能追到具体工件/字段/协议。
2. `RESULTS_FOR_PAPER.md`：英语可用段落，明确探索阶段、固定池、FL seeds、指导预算、更新数和隐私范围。DP结果未完成就写TBD，不编写乐观结论。
3. `PROTOCOL_FOR_PAPER.md`：已审计的实际选择机制、邻接、敏感度、发布对象、guidance构造、部署位置。
4. `TABLES_FOR_PAPER.csv` 和已验证图；未完成单元为NA/status，绝不填0或估计值。
5. `THEORY_IMPLEMENTATION_MAP.md`：V/Σ/α/d/m等理论符号与真实函数/张量路径对应；不相符的条目明示未映射。

不要直接改 `IEEE_ToN/`，不要让实验脚本自动覆盖论文。写作Codex需要单独验证后吸收上述材料。

## 8. 归档与输出

建议新root名 `agnews_guidance_stage1_20261014_<protocol_id>`，由当前仓库和结果根实际路径生成。原始材料放受限server results/private_staging/backups，聚合交接材料单独同步；不把GitHub当唯一备份。

最终生成 `STAGE1_REPORT.md`，至少回答：

- 15目标中哪些复用、哪些新完成、哪些失败/blocked，原因是什么；
- 协议/实现是否发现问题，哪些旧结果不能复用；
- PRIVATE_ZERO相对PUB_NLI、公随机和真实参考分别如何；
- PRIVATE_DP_E4相对零噪声、强公开和真实参考如何；
- 隐私保证覆盖什么，实际数值/RNG审查与联合发布预算如何；
- 成本测到了什么、未测到什么；
- 哪些claim可写，哪些仍待验证；
- 下一阶段应否扩展，依据是什么。

必须忠实报告负面结果。运行受阻时不冒充完成；不要无限轮询占用环境，也不要承诺未提交的后台任务。

完成本轮15目标矩阵或因明确机制/资源阻断停止后，**停止并报告**。不得自主执行03方案的后续投稿矩阵。
