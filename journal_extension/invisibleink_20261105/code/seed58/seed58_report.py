"""Paired FL-seed comparison conditional on three fixed generation artifacts."""
import csv,json,time
from diag_common import *
def main():
    rows=[]
    for seed,base in [(57,PILOT),(58,ROOT)]:
        for arm in ['PUBLIC','NONPRIVATE','DP_E8']:
            p=base/'runs'/arm/('seed_%d'%seed)
            if not (p/'run_spec.json').exists():continue
            spec=json.loads((p/'run_spec.json').read_text())
            if spec['status']!='complete':continue
            m=json.loads((p/'metrics.json').read_text());ev=m['evaluations'];assert len(ev)==51
            assert all(m['checks'].values())
            rows.append({'arm':arm,'seed':seed,'dev_accuracy':ev[-1]['acc'],'macro_f1':ev[-1]['macro_f1'],
                         'dev_ce':ev[-1]['eval_loss'],'accuracy_auc':sum((a['acc']+b['acc'])/2 for a,b in zip(ev,ev[1:]))/50,
                         'rounds':50,'evaluation_points':51,'wall_seconds':spec['wall_seconds']})
    paired=[]
    for seed in [57,58]:
        r={x['arm']:x for x in rows if x['seed']==seed}
        if len(r)==3:paired.append({'seed':seed,'nonprivate_minus_public_pp':100*(r['NONPRIVATE']['dev_accuracy']-r['PUBLIC']['dev_accuracy']),
            'dp_minus_nonprivate_pp':100*(r['DP_E8']['dev_accuracy']-r['NONPRIVATE']['dev_accuracy']),
            'dp_minus_public_pp':100*(r['DP_E8']['dev_accuracy']-r['PUBLIC']['dev_accuracy'])})
    write(ROOT/'SEED_COMPARISON.json',{'rows':rows,'paired':paired,'time':time.time(),
           'scope':'FL-seed robustness conditional on one frozen generation artifact per arm; not independent generation replicates or full FL DP'})
    with (ROOT/'seed_comparison.csv').open('w') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    lines=['# 固定生成工件的 seed58 复核','', '| 条件 | FL seed | 50轮开发集准确率 | macro-F1 |','|---|---:|---:|---:|']
    for r in rows:lines.append('| %s | %d | %.2f%% | %.4f |'%(r['arm'],r['seed'],100*r['dev_accuracy'],r['macro_f1']))
    lines+=['','仅纳入通过51点评估和更新预算检查的完整终点。seed57为先前完成的GPU3实验，seed58使用同机GPU0；每个seed内三臂串行使用同一张卡。',
            '', '三份生成工件及训练协议保持不变。此复核只检查FL随机性，不能代替独立生成重复，也不代表整条FL transcript满足DP。固定前缀诊断是受限Non-DP内部统计，不可自动公开。']
    (ROOT/'SEED58_REPORT.md').write_text('\n'.join(lines)+'\n')
if __name__=='__main__':main()
