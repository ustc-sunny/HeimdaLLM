"""Summarize only completed, validator-approved endpoints (stdlib, no model)."""
import json,csv,time,shutil
from common import ROOT,write
def main():
    rows=[]
    for arm in ['PUBLIC','NONPRIVATE','DP_E8']:
        p=ROOT/'runs'/arm/'seed_57'
        if not (p/'run_spec.json').exists(): continue
        spec=json.loads((p/'run_spec.json').read_text())
        if spec['status']!='complete':continue
        cp=ROOT/'scratch'/arm/'model_output/diagnostic_checkpoints'
        if cp.exists() and not (p/'diagnostic_checkpoints').exists():
            shutil.copytree(cp,p/'diagnostic_checkpoints')
        m=json.loads((p/'metrics.json').read_text()); ev=m['evaluations']; last=ev[-1]
        assert len(ev)==51
        auc=sum((a['acc']+b['acc'])/2 for a,b in zip(ev,ev[1:]))/50
        rows.append({'arm':arm,'seed':57,'final_dev_acc':last['acc'],
            'macro_f1':last.get('macro_f1'),'dev_ce':last.get('eval_loss'),
            'accuracy_auc':auc,'rounds':50,'evaluation_points':len(ev),'wall_seconds':spec['wall_seconds']})
    reference=json.loads((ROOT/'historical_real_seed57/metrics.json').read_text())['evaluations'][-1]
    write(ROOT/'RESULTS.json',{'time':time.time(),'new_completed_endpoints':rows,
        'historical_REAL_MATCHED':{'final_dev_acc':reference['acc'],'caveat':'historical A6000; source/model/data matched, not a new A40 paired rerun'},
        'scope':'single seed, dev-only feasibility pilot; no inference of statistical significance or full-transcript DP'})
    if rows:
        with (ROOT/'metrics.csv').open('w') as f:
            w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    lines=['# InvisibleInk 小试执行记录','',f'已通过验证的新终点：{len(rows)}/3。','',
           '| 条件 | FL seed | 50轮 dev accuracy | macro-F1 | 轨迹 AUC |',
           '|---|---:|---:|---:|---:|']
    for row in rows:
        lines.append('| %s | 57 | %.2f%% | %.4f | %.4f |'%(row['arm'],100*row['final_dev_acc'],row['macro_f1'],row['accuracy_auc']))
    lines+=['', '旧 REAL_MATCHED seed57 开发集准确率 %.2f%%；作为历史参照，不伪装成本次 A40 新配对终点。'%(reference['acc']*100),
        '', '只有通过全部51点评估和更新预算校验的终点进入上表。DP_E8 的数学条件及数值实现限制见 PRIVACY_AUDIT.md。NONPRIVATE、真实指导和 FL transcript 不是该生成机制的 DP 发布。单seed不能形成正式隐私曲线。',
        '', '未自动 push、未改论文、未新增 epsilon 或数据集。失败见 QUEUE_STATUS.json 和 failures/；日志和私有中间工件仅在受限目录。']
    (ROOT/'PILOT_REPORT.md').write_text('\n'.join(lines)+'\n')
if __name__=='__main__':main()
