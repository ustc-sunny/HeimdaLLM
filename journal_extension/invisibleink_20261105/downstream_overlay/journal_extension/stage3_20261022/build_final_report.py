"""Export only independently verified results; fail if any required evidence is missing."""
import csv
from statistics import mean,stdev
from common import *

def csv_out(path,rows):
 with path.open('w',newline='') as f:
  w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)

def main():
 out=ROOT/'report';audit=json.loads((out/'FINAL_EVIDENCE_AUDIT.json').read_text())
 assert audit['status']=='complete' and audit['complete_goal_evidence']
 metrics=json.loads((out/'metrics.json').read_text());assert metrics['complete']==12
 diag=json.loads((out/'COMMON_CHECKPOINT_DIAGNOSTICS.json').read_text());assert diag['status']=='complete'
 rows=metrics['endpoints'];assert len(rows)==12 and all(r['status']=='complete' for r in rows)
 csv_out(out/'FINAL_METRICS.csv',rows);csv_out(out/'PAIRED_DIFFERENCES.csv',metrics['paired'])
 curves=[]
 for row in rows:
  path=ROOT/'runs'/row['arm']/('seed_%d'%row['seed'])/'metrics.json'
  for e in json.loads(path.read_text())['evaluations']:
   curves.append({'group':row['group'],'arm':row['arm'],'seed':row['seed'],'completed_rounds':e['round']+1,'dev_accuracy':e['acc'],'macro_f1':e['macro_f1'],'eval_loss':e['eval_loss']})
 assert len(curves)==612;csv_out(out/'CURVES.csv',curves)
 diagnostic_rows=[]
 for entry in diag['reports']:
  for arm,s in entry['overall'].items():
   diagnostic_rows.append({'completed_rounds':entry['round'],'arm':arm,'bp_cosine_to_real64':s['cosine'],'relative_l2_error':s['relative_l2_error'],'sign_agreement':s['sign_agreement'],'rank1_energy_capture':s['zgr_rank1_normalized_energy_capture'],'gradient_l2':s['gradient_l2'],'checkpoint_sha256':entry['checkpoint_sha256']})
 csv_out(out/'COMMON_CHECKPOINT_DIAGNOSTICS.csv',diagnostic_rows)
 lines=['# HeimdaLLM Stage-3 实验完成报告','', '生成时间（UTC）：'+now(),'',
 '本批固定的 **12/12 个下游终点**及共同检查点 **0/10/25/50** 的实际 BP/GGD 诊断均已完成并通过独立验收。没有新增 DP 发布，没有读取官方测试集，没有按精度重试。','',
 '## 实验范围','',
 '- 原组客户端 1/21：NO_GUIDANCE_TRACE_LEGACY × seeds 57/58/59。',
 '- 新组固定客户端 100/101：PUB_NLI / PRIVATE_ZERO / REAL_MATCHED × seeds 57/58/59。',
 '- 每次 50 FL rounds、750 内层更新、1,500 FD 方向、3,000 函数查询；每客户端 120 条真实记录，评估使用 512 条固定开发数据。新组两种合成指导从同一 128 条冻结公开候选中选取，每个有指导条件使用 64 条云端指导；无指导条件不加载云端指导。',
 '- 单 RTX A6000 48 GB 共址运行。早期双任务并行，恢复后的三个未启动任务改为串行；运行时间仅用于进度/资源记录，不作为加速实验证据。','',
 '## 下游结果','',
 '下面为最终开发集准确率，单位为百分比；± 表示三个 FL seeds 的样本标准差。不同客户端组分开解释，不跨组直接归因。','',
 '| 客户端组 | 条件 | seed 57 | seed 58 | seed 59 | 平均 ± SD |','|---|---|---:|---:|---:|---:|']
 for arm in ['NO_GUIDANCE_TRACE_LEGACY','PUB_NLI','PRIVATE_ZERO','REAL_MATCHED']:
  selected=sorted([r for r in rows if r['arm']==arm],key=lambda r:r['seed']);vals=[r['final_accuracy_percent'] for r in selected]
  lines.append('| %s | %s | %.2f | %.2f | %.2f | %.2f ± %.2f |'%(selected[0]['group'],arm,*vals,mean(vals),stdev(vals)))
 lines+=['','新组内的配对差值（percentage points）：','','| 差值 | seed 57 | seed 58 | seed 59 | 均值 |','|---|---:|---:|---:|---:|']
 for contrast in ['PRIVATE_ZERO minus PUB_NLI','REAL_MATCHED minus PUB_NLI','PRIVATE_ZERO minus REAL_MATCHED']:
  vals=[r['final_accuracy_difference_pp'] for r in sorted(metrics['paired'],key=lambda x:x['seed']) if r['contrast']==contrast]
  assert len(vals)==3;lines.append('| %s | %+.2f | %+.2f | %+.2f | %+.2f |'%(contrast,*vals,mean(vals)))
 lines+=['','本次新组中，PRIVATE_ZERO 相对 PUB_NLI 的逐 seed 差值有正有负，均值接近零，未观察到稳定的下游精度优势。两种合成指导仍低于同预算 REAL_MATCHED。不能把原组无指导与新组有指导的跨客户端差值解释为指导的因果收益。','', '## 共同检查点诊断','',
 '锚点事先固定为新组 PUB_NLI seed 57 的实际聚合器状态。每个时点均在同一参数状态、配对诊断随机性下调用真实 cloud BP 和 rank-1 GGD；参考为 64 条 REAL_MATCHED 云端数据的梯度，不是客户端总体风险梯度。','',
 '| 已完成轮数 | 指导源 | BP cosine | 相对 L2 误差 | rank-1 能量捕获 |','|---:|---|---:|---:|---:|']
 for r in diagnostic_rows:lines.append('| %d | %s | %.4f | %.4f | %.4f |'%(r['completed_rounds'],r['arm'],r['bp_cosine_to_real64'],r['relative_l2_error'],r['rank1_energy_capture']))
 lines+=['','在这条固定锚点轨迹的四个时点，PRIVATE_ZERO 相对真实指导的梯度 cosine 和 rank-1 能量捕获均高于 PUB_NLI，但这一诊断优势没有对应为上述稳定的下游精度优势。REAL_MATCHED 与自身比较所得的 1.0 是参考一致性检查，不是新的性能收益。','', '## 验收及限制','',
 '- 独立验收检查了每个终点的退出码、51 次评估、原始日志与指标一致性、512 样本混淆矩阵、accuracy/macro-F1 重算、查询预算，以及四个实际检查点的参数数目、有限性和哈希。总计 36,000 次函数查询，48 个训练检查点。',
 '- 诊断核验了四个实际锚点文件和参数哈希、三臂指导数据哈希、unit GGD 范数和 rank-1 能量一致性。只覆盖一条轨迹；不能外推高秩子空间或多 seed 收敛保证。',
 '- 归一化无指导对照保留 legacy 历史候选策略；独立高斯探针的 g/d 推导不是完整训练轨迹的无偏性证明。半径/名义尺度匹配也不等于期望更新尺度匹配，不能将其差值直接当作干净的语义指导因果效应。',
 '- 新组是同一 AG News uniform 分区内的客户端重复验证，不是跨域实验。只有三个 FL seeds 和一个新客户端组，不声称统计显著、原 KDD 完整规模精度或普遍泛化。',
 '- 本批没有新增 DP 噪声工件或 epsilon 扫描，PRIVATE_ZERO 是无噪声私有选择，不是 ε=0。模型状态、训练日志、开发指标和真实指导均没有 DP 保证。','',
 '## 故障及恢复记录','',
 '首次历史桥接短测失败后，发现旧 HDF5 读取器按线程完成顺序追加文本/标签，导致缓存重建时样本顺序不由 seed 唯一确定；文本与标签仍成对，guid 可能错位。新组在任何正式运行前改为冻结分区顺序，原组对照重放经源数据多重集核验的相应历史 PUB_NLI 缓存。旧结果保留；不能声称所有旧臂仅凭 FL seed 就完全配对。另将历史桥接改为首轮检查，避免缩短总轮数改变第二轮学习率调度。',
 '',
 '18:42 队列在 PUB_NLI seed 59 启动前被 GPU 保护检查挡住；当时检测到利用率但未识别为可见自有进程，确切归属异常未完全证实。另一项已在运行的无指导实验正常完成，实际累计九项完成。20:43 经人工检查九项原始结果及三次空闲 GPU 样本后，仅恢复三个从未启动的 seed 59 任务，改为串行，保留原 GPU 预检。没有覆盖历史结果，也没有因精度差重跑。完整记录保存在 preflight_failure_01/ 和 recovery_20261022_serial/。','',
 '## 文件','',
 '- `FINAL_EVIDENCE_AUDIT.json`：最终独立验收；`FINAL_METRICS.csv`：逐终点指标。',
 '- `CURVES.csv`：612 个评估时点；`PAIRED_DIFFERENCES.csv`：新组内配对差值。',
 '- `COMMON_CHECKPOINT_DIAGNOSTICS.json/.csv`：四个时点的三臂实际诊断。',
 '- `runs/`：原始日志、命令、指标、预算校验和实际检查点，含受限训练工件。',
 '- 源码由基础 Git 提交 `04cb83d5ebb5cfb05e1851629d2578855f2f3906`、`source.patch` 和 `journal_extension/stage3_20261022/` 共同确定，不能只用基础提交号表示全部运行代码。',
 '', '本报告不宣称本地传输已完成；本地备份完成与否应另以本地哈希核验文件为准。没有自动 push，也没有修改 IEEE_ToN 论文。']
 (out/'STAGE3_FINAL_REPORT_ZH.md').write_text('\n'.join(lines)+'\n')
 print(json.dumps({'status':'report_written','endpoints':12,'curve_rows':len(curves),'diagnostic_rows':len(diagnostic_rows),'report':str(out/'STAGE3_FINAL_REPORT_ZH.md')}))
if __name__=='__main__':main()
