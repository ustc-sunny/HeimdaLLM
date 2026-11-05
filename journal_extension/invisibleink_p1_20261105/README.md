# P1同协议参考条件：已启动

这是作者2026-11-05授权的下一步，于14:29在GPU0启动；**截至14:30，新增完整终点为0/4，第1组REAL_MATCHED/seed57已完成1/50轮**。现场预检和初始模型哈希核验通过，运行监测与定期本地备份已启用。状态是带时间戳的快照，不是实时看板。首次预检的分片数量断言已修正：固定文件含两个训练分片及一个纯开发评估分片，原数据哈希与训练设置均不变。

- [冻结范围](SCOPE.json)和[实验计划](PLAN_CN.md)
- [本地命令核对](COMMAND_AUDIT_LOCAL.json)、[实际状态](STATUS.json)
- [上轮完整结果](../invisibleink_20261105/README.md)

REAL_MATCHED和NO_GUIDANCE_TRACE_LEGACY各运行FL seeds57/58、50轮；保持当前客户端100/101及相同开发划分。真实指导复用64条既有H5；不生成新文本、不发布新DP工件，不自动执行P2或修改论文。

## 恢复与启动前条件

代码保留准备时的路径与GPU0 UUID，新机器必须先核对并显式适配。需要之前pilot/e16的受限备份、原始FedNLP训练文件与固定划分、已缓存模型，以及现有下游环境。原训练源码需匹配`references/DOWNSTREAM_SOURCE_GATE.json`，不能直接用任意最新工作树。

`guidance/REAL_MATCHED/cloud_data.h5`和`cloud_partition.h5`只从既有受限备份恢复，不在GitHub发布。预期SHA256分别为`2298c69a4791874cf5913c6e88c0c7bbfbb5da4818ecd946803a9f7541805888`和`b740b69996959c2ad6f4404dbb08b1bf6b5b661e45d4792c520b0f6bcd835dd8`；同时恢复原`pack_manifest.json`，不可重挑数据。内部标签映射1→0、2→2、3→3、4→1。

现场步骤：确认VPN和SSH可达；在新的P1输出根目录部署；恢复已授权的RESOURCE_AUTHORIZATION.json；重新执行两个CPU保护检查；用下游环境运行`code/preflight.py`完成源码、H5、环境和GPU0准入检查；仅在REVIEW_GATE通过后，使用系统Python运行`code/pipeline.py`。该队列最多运行4组，并保留日志、资源事件和实际检查点；每组结束核验51次评估及750/1500/3000预算。同步备份程序位于受限本地工作区，上传前不会保存凭据。

守护进程只处理本任务的、经UID和PID起始时间验证的进程。新终点要求GPU0无其他计算任务；已准入的健康训练不会仅因别人进程出现而被终止。未经用户授权不换GPU，也不操作他人进程。

历史无指导使用客户端1/21，不可直接复用；此处更换为当前100/101，但保留legacy筛选算子。此对照不能称为充分调优、无偏的纯ZOO算法。预计3–4小时，不含连接或GPU等待时间。
