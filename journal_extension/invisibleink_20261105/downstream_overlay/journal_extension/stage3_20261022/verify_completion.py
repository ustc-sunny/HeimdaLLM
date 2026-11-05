"""Independently audit actual endpoint/checkpoint evidence, not completion markers alone."""
import argparse,math,re,shlex,sys
from common import *
sys.path.insert(0,str(REPO/'journal_extension'))
from summarize_fed_pilot import parse_metrics

ARMS=['NO_GUIDANCE_TRACE_LEGACY','PUB_NLI','PRIVATE_ZERO','REAL_MATCHED']

def checkpoint(path,seed,round_number):
 import torch
 state=torch.load(path,map_location='cpu');meta=state['metadata'];h=hashlib.sha256()
 assert meta['seed']==seed and meta['completed_round']==round_number
 assert meta['kind']=='actual_aggregator_model_state'
 assert sum(t.numel() for t in state['trainable_state'].values())==450340
 for name,tensor in sorted(state['trainable_state'].items()):
  assert torch.isfinite(tensor).all(),name
  h.update(name.encode());h.update(tensor.numpy().tobytes())
 assert h.hexdigest()==meta['trainable_sha256']
 assert json.loads(path.with_suffix('.json').read_text())==meta
 return {'file_sha256':digest(path),'trainable_sha256':h.hexdigest(),'round':round_number}

def main():
 ap=argparse.ArgumentParser();ap.add_argument('--partial',action='store_true');a=ap.parse_args()
 import torch
 torch.set_num_threads(1)
 output=ROOT/'report'/('PARTIAL_EVIDENCE_AUDIT.json' if a.partial else 'FINAL_EVIDENCE_AUDIT.json')
 rows=[];pending=[];diagnostics=[];cohort=json.loads((ROOT/'cohort/AUDIT.json').read_text())
 assert cohort['client_ids']==[100,101] and cohort['dev_records']==512 and not cohort['new_dp_release']
 for arm in ARMS:
  for seed in [57,58,59]:
   d=ROOT/'runs'/arm/('seed_%d'%seed);p=d/'run_spec.json'
   if not p.exists() or json.loads(p.read_text())['status']!='complete':
    pending.append({'arm':arm,'seed':seed});continue
   s=json.loads(p.read_text());m=json.loads((d/'metrics.json').read_text());manifest=json.loads((d/'manifest.json').read_text())
   assert s['exit_code']==0 and s['source_commit']==BASE and s['seed']==seed and s['arm']==arm and s['rounds']==50
   assert not s['new_dp_release'] and not s['official_test_used']
   assert (d/'train.exit_code.txt').read_text().strip()=='0'
   assert manifest['log_sha256']==digest(d/'train.log') and manifest['command_sha256']==digest(d/'train.command.txt')
   assert m['status']=='complete' and all(m['checks'].values()) and not m['rejected_metric_lines']
   es=m['evaluations'];assert len(es)==51 and [e['round'] for e in es]==list(range(-1,50))
   log=(d/'train.log').read_text(errors='replace');actual,rejected=parse_metrics(log)
   assert not rejected and len(actual)==51 and log.count('training is finished')==1
   for observed,reported in zip(actual,es):
    for key in ['acc','macro_f1','eval_loss','confusion_matrix']:assert observed[key]==reported[key]
    assert all(math.isfinite(reported[k]) for k in ['acc','macro_f1','eval_loss'])
    cm=reported['confusion_matrix'];assert len(cm)==4 and all(len(r)==4 and sum(r)==128 for r in cm)
    assert all(isinstance(v,int) and v>=0 for r in cm for v in r)
    assert abs(sum(cm[i][i] for i in range(4))/512-reported['acc'])<1e-12
    f1=sum(2*cm[i][i]/(sum(cm[i])+sum(r[i] for r in cm)) for i in range(4))/4
    assert abs(f1-reported['macro_f1'])<1e-12
   assert log.count('aggregate time cost')==750
   assert sum(map(int,re.findall(r'\[ZGR\] finite differences: nonzero=\d+/(\d+)',log)))==1500
   budget=json.loads((d/'budget_audit.json').read_text());assert budget['objective_queries']==3000 and budget['status']=='pass'
   command=shlex.split((d/'train.command.txt').read_text())
   for flag,value in [('--comm_round','50'),('--train_batch_size','8'),('--max_seq_length','64'),('--lr','0.01'),('--learning_rate','0.01'),('--fd_step','0.01'),('--beta','1'),('--pool_size','1'),('--zo_estimator','legacy'),('--max_var_retries','0'),('--manual_seed',str(seed))]:assert command[command.index(flag)+1]==value,(arm,seed,flag)
   fixed=S1/'shared/fixed_partition.h5' if arm==ARMS[0] else ROOT/'cohort/fixed_partition.h5'
   assert s['fixed_partition_sha256']==digest(fixed)
   if arm==ARMS[0]:
    assert '--cloud_data_file_path' not in command and float(command[command.index('--alpha')+1])==1
    cache=json.loads((d/'cache_replay_audit.json').read_text());assert cache and all(x['multiset_verified'] and x['rank'] in [1,2] for x in cache)
   else:
    assert command[command.index('--cloud_data_file_path')+1]==str(ROOT/'cohort'/arm/'cloud_data.h5')
    assert float(command[command.index('--alpha')+1])==.5
    assert s['cloud_data_sha256']==cohort['guidance'][arm]['cloud_data.h5']==digest(ROOT/'cohort'/arm/'cloud_data.h5')
   cps=[checkpoint(d/'diagnostic_checkpoints'/('round_%02d.pt'%r),seed,r) for r in [0,10,25,50]]
   rows.append({'arm':arm,'seed':seed,'final_accuracy_percent':100*es[-1]['acc'],'queries':3000,'checkpoints':cps,'log_sha256':digest(d/'train.log')})
 diag=ROOT/'report/COMMON_CHECKPOINT_DIAGNOSTICS.json'
 if diag.exists():
  bundle=json.loads(diag.read_text());assert bundle['status']=='complete' and bundle['rounds']==[0,10,25,50]
  for r,entry in zip([0,10,25,50],bundle['reports']):
   cp=ROOT/'runs/PUB_NLI/seed_57/diagnostic_checkpoints'/('round_%02d.pt'%r)
   meta=json.loads(cp.with_suffix('.json').read_text())
   assert entry['round']==r and entry['checkpoint_sha256']==meta['trainable_sha256'] and entry['actual_checkpoint_file_sha256']==digest(cp)
   assert entry['trainable_count']==450340 and set(entry['overall'])==set(ARMS[1:])
   for arm,stats in entry['overall'].items():
    assert all(math.isfinite(v) for v in stats.values() if isinstance(v,(int,float)))
    assert stats['gradient_l2']>0 and abs(stats['production_direction_l2']-1)<1e-5
    assert -1.000001<=stats['cosine']<=1.000001
    assert abs(stats['production_direction_reference_cosine']**2-stats['zgr_rank1_normalized_energy_capture'])<1e-5
    prov=entry['provenance'][arm]
    assert prov['records']==64 and prov['checkpoint_sha256']==meta['trainable_sha256']
    assert prov['cloud_h5_sha256']==cohort['guidance'][arm]['cloud_data.h5']
   diagnostics.append({'round':r,'actual_checkpoint_sha256':digest(cp),'arms':list(entry['overall'])})
 done=len(rows)==12 and len(diagnostics)==4 and not pending
 audit={'time':now(),'status':'complete' if done else 'partial','completed_endpoints_verified':len(rows),'target':12,'pending':pending,'endpoints':rows,'diagnostics_verified':diagnostics,'complete_goal_evidence':done,'new_dp_releases':False,'total_completed_objective_queries':sum(r['queries'] for r in rows),'scope':'development-set fixed protocol; no full FL DP, no statistical significance or system speedup claim'}
 write(output,audit)
 print(json.dumps({'status':audit['status'],'completed_verified':len(rows),'diagnostics_verified':len(diagnostics),'pending':pending,'path':str(output)}))
 if not a.partial:assert done,'completion not proven'
if __name__=='__main__':main()
