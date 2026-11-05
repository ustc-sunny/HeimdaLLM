"""Report every complete endpoint, including the explicit inherited teardown exception."""
import csv, statistics
from common import *

ARMS=['PUBLIC','NONPRIVATE']+['DP_E%d'%e for e in CURVE_EPSILONS]

def main():
    rows=[];trajectories={}
    for seed in SEEDS:
        for arm in ARMS:
            base=endpoint_source(arm,seed);d=base/'runs'/arm/('seed_%d'%seed)
            if not endpoint_complete(d):continue
            spec=json.loads((d/'run_spec.json').read_text());m=json.loads((d/'metrics.json').read_text())
            ev=m['evaluations'];assert len(ev)==51
            assert json.loads((d/'budget_audit.json').read_text())=={'status':'pass','updates':750,'directions':1500,'queries':3000}
            epsilon=int(arm[4:]) if arm.startswith('DP_E') else None
            rows.append({'arm':arm,'epsilon':epsilon,'seed':seed,'dev_accuracy':ev[-1]['acc'],
                'macro_f1':ev[-1]['macro_f1'],'dev_ce':ev[-1]['eval_loss'],
                'accuracy_auc':sum((a['acc']+b['acc'])/2 for a,b in zip(ev,ev[1:]))/50,
                'last10_accuracy':statistics.mean(e['acc'] for e in ev[-10:]),'rounds':50,
                'evaluation_points':51,'reused':base!=ROOT,'source_path':str(d),
                'wall_seconds':spec.get('wall_seconds'),'completion_status':spec['status'],
                'exit_code_observed':spec.get('exit_code')})
            trajectories[(arm,seed)]=[e['acc'] for e in ev]
    paired=[]
    for seed in SEEDS:
        ref={r['arm']:r for r in rows if r['seed']==seed}
        for arm in ARMS[2:]:
            if arm in ref:
                paired.append({'seed':seed,'epsilon':ref[arm]['epsilon'],
                    'dp_minus_public_pp':100*(ref[arm]['dev_accuracy']-ref['PUBLIC']['dev_accuracy']),
                    'dp_minus_nonprivate_pp':100*(ref[arm]['dev_accuracy']-ref['NONPRIVATE']['dev_accuracy'])})
    summaries=[]
    for arm in ARMS:
        rr=[r for r in rows if r['arm']==arm]
        if len(rr)==2:
            vals=[r['dev_accuracy'] for r in rr]
            summaries.append({'arm':arm,'epsilon':rr[0]['epsilon'],'n_fl_seeds':2,
                'mean_accuracy':statistics.mean(vals),'min_accuracy':min(vals),'max_accuracy':max(vals),
                'sample_std_accuracy':statistics.stdev(vals),'generation_artifacts':1})
    write(ROOT/'RESULTS.json',{'time':time.time(),'complete_endpoints':len(rows),'planned_total_endpoints':14,
        'rows':rows,'paired':paired,'summary':summaries,
        'scope':'One generation artifact per budget, two FL seeds; exploratory, no independent generation replication, no whole-FL DP'})
    for name,data in [('metrics.csv',rows),('paired.csv',paired),('summary.csv',summaries)]:
        if data:
            with (ROOT/name).open('w',newline='') as f:
                w=csv.DictWriter(f,fieldnames=list(data[0]));w.writeheader();w.writerows(data)
    lines=['# ε=16补齐后的五档隐私预算探索实验','',
        '完整终点 %d/14；其中原有12个复用，本次仅补齐ε=16的两个训练终点。'%(len(rows)),
        '', '| 条件 | seed57 | seed58 | 平均 |','|---|---:|---:|---:|']
    for arm in ARMS:
        vals={r['seed']:r['dev_accuracy'] for r in rows if r['arm']==arm}
        lines.append('| %s | %s | %s | %s |'%(arm,
            ('%.2f%%'%(100*vals[57])) if 57 in vals else '待完成',
            ('%.2f%%'%(100*vals[58])) if 58 in vals else '待完成',
            ('%.2f%%'%(100*statistics.mean(vals.values()))) if len(vals)==2 else '—'))
    lines += ['', '固定AG News、两个逻辑客户端、64条指导数据和50轮；沿用原ZGR训练命令，只替换指导路径和配对seed。各终点51次开发集评估、750次更新、1500方向、3000查询。没有访问官方测试集。',
        '', '五档ε为1/4/8/12/16；ε=20按用户要求取消，未生成或训练，δ=1e-5。ε不是噪声标准差；它通过原公式改变差分logits裁剪阈值和公开支持阈值。越大表示约束越宽松，不保证准确率越高。',
        '', '每档只有一份生成数据，两个FL seed复用；误差范围只能说明条件于该工件的训练seed差异。不能据此断言最优ε、生成随机性的置信区间或统计显著性。NONPRIVATE仍为合成数据对照，不代表原真实指导的精度。',
        '', 'ε=16复用11月4日wide实验中已完整生成的64条文本和原H5，哈希逐项核对。本轮不生成文本、不增加合成发布预算。早先28条部分生成、wide中615更新的中断训练均原样保留。seed57因缺失完整恢复状态从原初始化重跑，seed58首次训练；不以精度选取结果。累计生成账本沿用PRIOR_GENERATION_LEDGER.json。',
        '', '隐私界限仅指理想条件下的合成指导发布，数值实现及部署限制见PRIVACY_AUDIT.md。下游训练访问真实记录，模型、日志、检查点及研究备份不是该生成机制的DP输出。',
        '', '历史ε=4/seed57的51次评估、750次更新和4个检查点完整，但退出码未记录；沿用completed_training_teardown_unobserved标记及原审计，不声称全部严格退出检查通过。其他终点均需观察到exit=0。',
        '', '原seed57的PUBLIC/NONPRIVATE/ε8在同机GPU3；seed58与其他预算在GPU0。代码和协议核对一致，不分析硬件速度。未修改论文，未自动push。新训练仍在GPU0空闲时准入；运行中仅出现其他进程不会强制终止健康训练，但会记录PID、用户名、显存；实际报错或持续资源不足时停止己方任务。']
    if len(rows)==14:
        plot(rows,trajectories,summaries)
        lines += ['', '图：privacy_accuracy.png/.pdf；training_curves.png/.pdf。连线只辅助阅读离散点，误差棒为两个FL seed的最小—最大范围，不是95%置信区间。']
    (ROOT/'TRADEOFF_REPORT.md').write_text('\n'.join(lines)+'\n')

def plot(rows,trajectories,summaries):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams.update({'font.size':10,'pdf.fonttype':42})
    fig,ax=plt.subplots(figsize=(8,4.8))
    for seed,color in [(57,'#4477AA'),(58,'#EE7733')]:
        rr=sorted([r for r in rows if r['seed']==seed and r['epsilon'] is not None],key=lambda r:r['epsilon'])
        ax.plot([r['epsilon'] for r in rr],[100*r['dev_accuracy'] for r in rr],'o--',color=color,alpha=.7,label='DP, FL seed %d'%seed)
    ss=sorted([s for s in summaries if s['epsilon'] is not None],key=lambda s:s['epsilon'])
    ax.errorbar([s['epsilon'] for s in ss],[100*s['mean_accuracy'] for s in ss],
        yerr=[[100*(s['mean_accuracy']-s['min_accuracy']) for s in ss],
              [100*(s['max_accuracy']-s['mean_accuracy']) for s in ss]],
        fmt='o-',color='black',capsize=4,label='DP mean; FL-seed range')
    for arm,color in [('PUBLIC','#228833'),('NONPRIVATE','#AA3377')]:
        val=next(s['mean_accuracy'] for s in summaries if s['arm']==arm)
        ax.axhline(100*val,color=color,linestyle=':',linewidth=2,label=arm+' mean')
    ax.set_xticks(list(CURVE_EPSILONS),[str(e) for e in CURVE_EPSILONS])
    ax.set_xlabel(r'Privacy budget $\epsilon$ (larger = weaker constraint)')
    ax.set_ylabel('Final development accuracy (%)');ax.grid(alpha=.2)
    ax.set_title(r'AG News synthetic guidance, $\delta=10^{-5}$, 50 FL rounds')
    ax.legend(fontsize=8)
    fig.text(.5,.015,'One generation artifact per condition; two FL seeds; exploratory results.',ha='center',fontsize=8)
    fig.tight_layout(rect=(0,.045,1,1))
    for ext in ['png','pdf']:fig.savefig(ROOT/('privacy_accuracy.'+ext),dpi=200)
    plt.close(fig)
    fig,axes=plt.subplots(1,2,figsize=(12,4.6),sharey=True)
    for ax,seed in zip(axes,SEEDS):
        for arm in ARMS:ax.plot(range(51),[100*x for x in trajectories[(arm,seed)]],label=arm)
        ax.set_title('FL seed %d'%seed);ax.set_xlabel('Completed FL rounds');ax.grid(alpha=.2)
    axes[0].set_ylabel('Development accuracy (%)');axes[1].legend(fontsize=8)
    fig.tight_layout()
    for ext in ['png','pdf']:fig.savefig(ROOT/('training_curves.'+ext),dpi=200)
    plt.close(fig)

if __name__=='__main__':main()
