# P1同协议参考条件：已完成

作者授权的4组P1参考实验已于2026-11-05 **17:30:54全部完成**，总耗时约181.5分钟；加入原14组后共18个终点。新增4组均通过exit0、51次评估、750更新、1500方向、3000查询及4个检查点核验；本地最终备份155个文件哈希一致。GPU0已释放。

真实指导seed57/58为77.73%/76.37%，平均77.05%；无指导legacy消融为27.73%/36.52%，平均32.13%。ε16合成指导仍比同协议真实指导低5.86个百分点，ε4低7.52个百分点。无指导两个seed的分类混淆矩阵在50轮内均未变化，因此它是当前legacy尺度的弱消融，不能冒充充分调优的纯ZOO基线。

- [完成与结果解读](analysis/P1_COMPLETION_AND_INTERPRETATION_CN.md)
- [全部18个终点](results/metrics.csv)、[同seed配对差值](results/paired.csv)、[918点完整训练曲线数据](results/curves.csv)
- [独立复核](analysis/P1_FINAL_ANALYSIS.json)、[备份核验](analysis/COMPLETION_VERIFIED.json)

本轮未生成数据、未新增DP发布、未执行P2、未访问官方测试集、未改论文。

- [冻结范围](SCOPE.json)和[实验计划](PLAN_CN.md)
- [本地命令核对](COMMAND_AUDIT_LOCAL.json)、[实际状态](STATUS.json)
- [上轮完整结果](../invisibleink_20261105/README.md)

REAL_MATCHED和NO_GUIDANCE_TRACE_LEGACY各运行FL seeds57/58、50轮；保持当前客户端100/101及相同开发划分。真实指导复用64条既有H5；不生成新文本、不发布新DP工件，不自动执行P2或修改论文。

## 历史执行与恢复条件

代码保留准备时的路径与GPU0 UUID，新机器必须先核对并显式适配。需要之前pilot/e16的受限备份、原始FedNLP训练文件与固定划分、已缓存模型，以及现有下游环境。原训练源码需匹配`references/DOWNSTREAM_SOURCE_GATE.json`，不能直接用任意最新工作树。

`guidance/REAL_MATCHED/cloud_data.h5`和`cloud_partition.h5`只从既有受限备份恢复，不在GitHub发布。预期SHA256分别为`2298c69a4791874cf5913c6e88c0c7bbfbb5da4818ecd946803a9f7541805888`和`b740b69996959c2ad6f4404dbb08b1bf6b5b661e45d4792c520b0f6bcd835dd8`；同时恢复原`pack_manifest.json`，不可重挑数据。内部标签映射1→0、2→2、3→3、4→1。

现场步骤：确认VPN和SSH可达；在新的P1输出根目录部署；恢复已授权的RESOURCE_AUTHORIZATION.json；重新执行两个CPU保护检查；用下游环境运行`code/preflight.py`完成源码、H5、环境和GPU0准入检查；仅在REVIEW_GATE通过后，使用系统Python运行`code/pipeline.py`。该队列最多运行4组，并保留日志、资源事件和实际检查点；每组结束核验51次评估及750/1500/3000预算。同步备份程序位于受限本地工作区，上传前不会保存凭据。

守护进程只处理本任务的、经UID和PID起始时间验证的进程。新终点要求GPU0无其他计算任务；已准入的健康训练不会仅因别人进程出现而被终止。未经用户授权不换GPU，也不操作他人进程。

历史无指导使用客户端1/21，不可直接复用；此处更换为当前100/101，但保留legacy筛选算子。此对照不能称为充分调优、无偏的纯ZOO算法。实际队列用时约181.5分钟。以上脚本保留用于审计；不要覆盖已完成目录或不经新范围审批重跑。

只复算已发布指标可运行 `python3 verify_results.py`，无需模型、原始数据或网络。
