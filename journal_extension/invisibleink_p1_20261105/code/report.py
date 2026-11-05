"""P1 reference results, with all fourteen immutable synthetic endpoints retained."""
import csv,statistics
from common import *

PRIOR=Path('/data/zzkevin/heimdallm-invink-e16-completion-20261105')
ORDER=['NO_GUIDANCE_TRACE_LEGACY','PUBLIC','NONPRIVATE','DP_E1','DP_E4','DP_E8','DP_E12','DP_E16','REAL_MATCHED']

def main():
    expected=json.loads((ROOT/'references/PRIOR_RESULTS.json').read_text())
    assert digest(PRIOR/'RESULTS.json')==expected['sha256']
    old=json.loads((PRIOR/'RESULTS.json').read_text());assert old['complete_endpoints']==14
    rows=[dict(r,reused=True) for r in old['rows']];new=[]
    for arm in ARMS:
        for seed in SEEDS:
            d=ROOT/'runs'/arm/('seed_%d'%seed)
            if not endpoint_complete(d):continue
            m=json.loads((d/'metrics.json').read_text());sp=json.loads((d/'run_spec.json').read_text())
            ev=m['evaluations'];assert len(ev)==51
            assert json.loads((d/'budget_audit.json').read_text())=={'status':'pass','updates':750,'directions':1500,'queries':3000}
            new.append({'arm':arm,'epsilon':None,'seed':seed,'dev_accuracy':ev[-1]['acc'],
                'macro_f1':ev[-1]['macro_f1'],'dev_ce':ev[-1]['eval_loss'],
                'accuracy_auc':sum((a['acc']+b['acc'])/2 for a,b in zip(ev,ev[1:]))/50,
                'last10_accuracy':statistics.mean(e['acc'] for e in ev[-10:]),'rounds':50,
                'evaluation_points':51,'reused':False,'source_path':str(d),
                'wall_seconds':sp['wall_seconds'],'completion_status':'complete','exit_code_observed':0})
    rows+=new
    paired=[]
    for seed in SEEDS:
        by={r['arm']:r for r in rows if r['seed']==seed}
        for ref in ARMS:
            if ref not in by:continue
            for arm in ORDER:
                if arm in by and arm not in ARMS:
                    paired.append({'arm':arm,'seed':seed,'reference':ref,
                        'accuracy_difference_pp':100*(by[arm]['dev_accuracy']-by[ref]['dev_accuracy'])})
    write(ROOT/'RESULTS.json',{'time':time.time(),'complete_endpoints':len(rows),'new_complete_endpoints':len(new),
        'planned_new_endpoints':4,'rows':rows,'paired':paired,
        'scope':'Same current-cohort reference endpoints; no new synthetic generation; official test unused'})
    for name,data in [('metrics.csv',rows),('paired.csv',paired)]:
        if data:
            with (ROOT/name).open('w',newline='') as f:
                w=csv.DictWriter(f,fieldnames=list(data[0]),lineterminator='\n');w.writeheader();w.writerows(data)
    lines=['# P1 同协议参考条件','',f'新增完整终点{len(new)}/4；已有14个合成指导终点原样复用。',
        '', '| 条件 | seed57 | seed58 | 平均accuracy |','|---|---:|---:|---:|']
    for arm in ORDER:
        vals={r['seed']:r['dev_accuracy'] for r in rows if r['arm']==arm}
        lines.append('| %s | %s | %s | %s |'%(arm,
            ('%.2f%%'%(100*vals[57])) if 57 in vals else '待完成',
            ('%.2f%%'%(100*vals[58])) if 58 in vals else '待完成',
            ('%.2f%%'%(100*statistics.mean(vals.values()))) if len(vals)==2 else '—'))
    lines+=['','全部是固定512条开发集的结果。当前两客户端100/101，各120训练记录，50轮，750更新、1500方向、3000查询。',
        'REAL_MATCHED复用冻结64条真实指导；NO_GUIDANCE_TRACE_LEGACY保留原历史方向筛选，只移除云端指导分量，是legacy算子消融。',
        '旧ε4/seed57退出码未知但训练证据完整的审计例外继续保留；新增终点均要求exit0及全部检查通过。',
        '本阶段无新生成或DP发布，不执行P2或ε20；真实指导及真实数据FL不在合成发布DP保证内。',
        '每个合成条件仍只有一份生成工件；补参考条件不能代替独立生成重复。PUBLIC支持集匹配限制仍然存在。']
    (ROOT/'P1_REPORT_CN.md').write_text('\n'.join(lines)+'\n')

if __name__=='__main__':main()
