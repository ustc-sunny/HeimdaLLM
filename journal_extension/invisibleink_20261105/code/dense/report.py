"""Report complete endpoints only; no selected best seed or artificial curve."""
import csv, statistics
from common import *

def main():
    scope=json.loads((ROOT/'SCOPE.json').read_text()) if (ROOT/'SCOPE.json').exists() else {'active_new_arms':list(BUDGETS),'expected_total_endpoints':10}
    active_arms=scope['active_new_arms'];expected_total=scope['expected_total_endpoints']
    displayed_arms=['PUBLIC','NONPRIVATE']+['DP_E%d'%e for e in [4,8,12] if e==8 or 'DP_E%d'%e in active_arms]
    rows=[]; trajectories={}
    for seed in SEEDS:
        for arm in displayed_arms:
            base=(PILOT if seed==57 else FOLLOWUP) if arm in ['PUBLIC','NONPRIVATE','DP_E8'] else ROOT
            d=base/'runs'/arm/('seed_%d'%seed);sp=d/'run_spec.json'
            if not endpoint_complete(d):continue
            spec=json.loads(sp.read_text());m=json.loads((d/'metrics.json').read_text());ev=m['evaluations']
            assert len(ev)==51
            assert json.loads((d/'budget_audit.json').read_text())['status']=='pass'
            epsilon=int(arm[4:]) if arm.startswith('DP_E') else None
            rows.append({'arm':arm,'epsilon':epsilon,'seed':seed,'dev_accuracy':ev[-1]['acc'],
                'macro_f1':ev[-1]['macro_f1'],'dev_ce':ev[-1]['eval_loss'],
                'accuracy_auc':sum((a['acc']+b['acc'])/2 for a,b in zip(ev,ev[1:]))/50,
                'last10_accuracy':statistics.mean(e['acc'] for e in ev[-10:]),'rounds':50,
                'evaluation_points':51,'reused':base!=ROOT,'source_path':str(d),'wall_seconds':spec.get('wall_seconds'),
                'completion_status':spec['status'],'exit_code_observed':spec.get('exit_code')})
            trajectories[(arm,seed)]=[e['acc'] for e in ev]
    paired=[]
    for seed in SEEDS:
        ref={r['arm']:r for r in rows if r['seed']==seed}
        for arm in ['DP_E4','DP_E8','DP_E12']:
            if arm in ref:
                paired.append({'seed':seed,'epsilon':ref[arm]['epsilon'],
                    'dp_minus_public_pp':100*(ref[arm]['dev_accuracy']-ref['PUBLIC']['dev_accuracy']),
                    'dp_minus_nonprivate_pp':100*(ref[arm]['dev_accuracy']-ref['NONPRIVATE']['dev_accuracy'])})
    summaries=[]
    for arm in ['PUBLIC','NONPRIVATE','DP_E4','DP_E8','DP_E12']:
        r=[x for x in rows if x['arm']==arm]
        if len(r)==2:
            vals=[x['dev_accuracy'] for x in r]
            summaries.append({'arm':arm,'epsilon':r[0]['epsilon'],'n_fl_seeds':2,
                'mean_accuracy':statistics.mean(vals),'min_accuracy':min(vals),'max_accuracy':max(vals),
                'sample_std_accuracy':statistics.stdev(vals),'generation_artifacts':1})
    out={'time':time.time(),'complete_endpoints':len(rows),'planned_total_endpoints':expected_total,'rows':rows,
         'paired':paired,'summary':summaries,'scope':'One generation artifact per condition, two paired FL seeds; exploratory curve, no independent generation repeats, no whole-FL DP.'}
    write(ROOT/'RESULTS.json',out)
    for name,data in [('metrics.csv',rows),('paired.csv',paired),('summary.csv',summaries)]:
        if data:
            with (ROOT/name).open('w',newline='') as f:
                w=csv.DictWriter(f,fieldnames=list(data[0]));w.writeheader();w.writerows(data)
    # Update the accountant from attempted generation directories, including partial failures.
    ledger=json.loads((ROOT/'PRIVACY_LEDGER_PLANNED.json').read_text())
    ledger['actual_new_artifacts']={a:{'attempted':(ROOT/'generation'/a).exists(),
        'complete':(ROOT/'generation'/a/'manifest.json').exists(),
        'origin':'reused_byte_identical' if a=='DP_E4' else 'new_single_attempt'} for a in BUDGETS}
    ledger['status']='completed_selected_scope' if all(ledger['actual_new_artifacts'][a]['complete'] for a in active_arms) else 'in_progress_or_partial'
    ledger['active_new_arms']=active_arms
    ledger['skipped_by_user']=scope.get('skipped_arms',[])
    ledger['recovery_note']='DP_E4 exact artifact reuse; DP_E12 first and only planned artifact. Old interrupted DP_E16 is excluded from this curve. Paused DP_E4 training remains archived and is not a completed endpoint.'
    ledger['interrupted_attempt_source']='/data/zzkevin/heimdallm-invink-tradeoff-20261103/generation/DP_E16'
    write(ROOT/'PRIVACY_LEDGER_ACTUAL.json',ledger)
    lines=['# 隐私预算—准确率小规模实验','',
        '只纳入完成性与预算核验通过的终点。当前完成 %d/%d（含复用6个）。'%(len(rows),expected_total),'',
        '| 条件 | seed 57 | seed 58 | 平均 |','|---|---:|---:|---:|']
    for arm in displayed_arms:
        vals={r['seed']:r['dev_accuracy'] for r in rows if r['arm']==arm}
        lines.append('| %s | %s | %s | %s |'%(arm,
            ('%.2f%%'%(100*vals[57])) if 57 in vals else '待完成',
            ('%.2f%%'%(100*vals[58])) if 58 in vals else '待完成',
            ('%.2f%%'%(100*statistics.mean(vals.values()))) if len(vals)==2 else '—'))
    lines += ['', '本轮只改变预算 ε 及原机制据此校准的裁剪阈值/公开支持阈值；top-k=100、T=96、B=3、生成样本数64、客户端划分及50轮下游协议不变。',
        '', '每档只有一份生成数据，两个FL seed复用该工件；不能把FL误差线当作生成随机性的置信区间。旧seed57在同机GPU3，seed58及新增试验在GPU0，软件与参数核验不变。',
        '', 'δ固定1e-5。各档为单份完整合成发布的理想数学预算，不能将多档合集称为ε=8。原ε=16中断尝试的28条及内部状态另行保留，恢复须单独审批并记录所有尝试。研究归档包含Non-DP对照、原始训练状态、失败日志，不是DP发布包。',
        '', '没有读取官方测试集，没有重采样挑选更好工件，没有修改论文或自动push。']
    if any(r['completion_status']=='completed_training_teardown_unobserved' for r in rows):
        lines += ['', '收尾例外：ε=4、seed57已完成51次评估、750次更新、四个检查点及三进程正常结束日志，但监测退出竞态中断了包装程序，退出码未记录。结果按完整训练证据保留，明确标为completed_training_teardown_unobserved；exit_code_zero保持未知，不冒充全部严格检查通过。原日志和异常审计保留，未重跑或选择更好结果。']
    if len(rows)==expected_total:
        plot(rows,trajectories,summaries)
        lines += ['', '图：privacy_accuracy.png / privacy_accuracy.pdf。曲线显示各seed及平均值，误差棒为两个FL seed的最小—最大范围，不是95%置信区间。完整训练轨迹见training_curves.png / .pdf。']
    (ROOT/'TRADEOFF_REPORT.md').write_text('\n'.join(lines)+'\n')

def plot(rows,trajectories,summaries):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams.update({'font.size':10,'pdf.fonttype':42})
    fig,ax=plt.subplots(figsize=(7,4.6))
    for seed,color in [(57,'#4477AA'),(58,'#EE7733')]:
        rr=sorted([r for r in rows if r['seed']==seed and r['epsilon'] is not None],key=lambda r:r['epsilon'])
        ax.plot([r['epsilon'] for r in rr],[100*r['dev_accuracy'] for r in rr],'o--',color=color,alpha=.7,label='DP, FL seed %d'%seed)
    ss=sorted([s for s in summaries if s['epsilon'] is not None],key=lambda s:s['epsilon'])
    mean=[100*s['mean_accuracy'] for s in ss]
    ax.errorbar([s['epsilon'] for s in ss],mean,
        yerr=[[100*(s['mean_accuracy']-s['min_accuracy']) for s in ss],
              [100*(s['max_accuracy']-s['mean_accuracy']) for s in ss]],
        fmt='o-',color='black',capsize=4,label='DP mean; FL-seed range')
    for arm,color in [('PUBLIC','#228833'),('NONPRIVATE','#AA3377')]:
        val=next(s['mean_accuracy'] for s in summaries if s['arm']==arm)
        ax.axhline(100*val,color=color,linestyle=':',linewidth=2,label=arm+' mean')
    epsilons=[s['epsilon'] for s in ss]
    ax.set_xticks(epsilons,[str(e) for e in epsilons])
    ax.set_xlabel(r'Privacy budget $\epsilon$ (larger = weaker constraint)')
    ax.set_ylabel('Final development accuracy (%)');ax.grid(alpha=.2)
    ax.set_title(r'AG News synthetic guidance, $\delta=10^{-5}$, 50 FL rounds')
    ax.legend(fontsize=8);fig.text(.5,.015,'One generation artifact per condition; two FL seeds; exploratory results.',ha='center',fontsize=8)
    fig.tight_layout(rect=(0,.045,1,1))
    for ext in ['png','pdf']:fig.savefig(ROOT/('privacy_accuracy.'+ext),dpi=200)
    plt.close(fig)
    fig,axes=plt.subplots(1,2,figsize=(11,4.3),sharey=True)
    for ax,seed in zip(axes,SEEDS):
        for arm in ['PUBLIC','NONPRIVATE','DP_E4','DP_E8','DP_E12']:
            if (arm,seed) not in trajectories:continue
            ax.plot(range(51),[100*x for x in trajectories[(arm,seed)]],label=arm)
        ax.set_title('FL seed %d'%seed);ax.set_xlabel('Completed FL rounds');ax.grid(alpha=.2)
    axes[0].set_ylabel('Development accuracy (%)');axes[1].legend(fontsize=8)
    fig.tight_layout()
    for ext in ['png','pdf']:fig.savefig(ROOT/('training_curves.'+ext),dpi=200)
    plt.close(fig)

if __name__=='__main__':main()
