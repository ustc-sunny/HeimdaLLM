"""Bounded Stage-3 queue; gate on reproducibility/resources, never accuracy gain."""
import csv,time,traceback
from common import *
from run_endpoint import ARMS

def endpoint(arm,seed,smoke=False,tag='formal'):
 cmd=[str(PY),str(CODE/'run_endpoint.py'),'--arm',arm,'--seed',str(seed),'--tag',tag]
 if smoke:cmd+=['--smoke']
 log=ROOT/('launcher_%s_%s_%d.log'%(tag,arm,seed))
 if log.exists():raise FileExistsError(log)
 with log.open('w') as f:p=subprocess.Popen(cmd,stdout=f,stderr=subprocess.STDOUT,env=env(seed))
 return p

def result(arm,seed,tag=None):
 d=(ROOT/'preflight'/tag if tag else ROOT/'runs')/arm/('seed_%d'%seed)
 s=json.loads((d/'run_spec.json').read_text())
 assert s['status'] in ['complete','smoke_complete'],(d,s['status'])
 assert json.loads((d/'budget_audit.json').read_text())['status']=='pass'
 return s,json.loads((d/'metrics.json').read_text())

def compare(a,b):
 rows=[]
 assert len(a)==len(b)
 for x,y in zip(a,b):
  assert x['round']==y['round']
  diffs={k:abs(x[k]-y[k]) for k in ['acc','macro_f1','eval_loss']}
  assert diffs['acc']<1e-12 and diffs['macro_f1']<1e-12 and diffs['eval_loss']<=1e-6,diffs
  rows.append({'round':x['round'],'absolute_difference':diffs})
 return rows

def resource_window(start,end):
 from datetime import datetime
 rows=[]
 with (ROOT/'resources_orderfix.csv').open() as f:
  for r in csv.DictReader(f):
   t=datetime.fromisoformat(r['utc_time']).timestamp()
   if start<=t<=end and r['gpu0_memory_used_mib']!='NA':rows.append(r)
 assert rows,'missing resource measurements'
 return {'samples':len(rows),'peak_gpu_memory_mib':max(int(r['gpu0_memory_used_mib']) for r in rows),'peak_fednlp_rss_sum_bytes':max(int(r['fednlp_rss_sum_bytes']) for r in rows),'mean_gpu_utilization_percent':sum(int(r['gpu0_utilization_percent']) for r in rows)/len(rows)}

def gate():
 # An already launched trace smoke is awaited; never retried or overwritten.
 trace=ROOT/'preflight/trace_single/NO_GUIDANCE_TRACE_LEGACY/seed_57/run_spec.json'
 deadline=time.monotonic()+1200
 while True:
  s=json.loads(trace.read_text()) if trace.exists() else {}
  if s.get('status')=='smoke_complete':break
  if 'failed' in s.get('status',''):raise RuntimeError('trace smoke failed')
  if time.monotonic()>deadline:raise TimeoutError('trace smoke completion')
  time.sleep(5)
 ts,tm=result('NO_GUIDANCE_TRACE_LEGACY',57,'trace_single')
 import re
 content=(trace.parent/'train.log').read_text(errors='replace')
 update_norms=[float(x) for x in re.findall(r'\[UPDATE_AUDIT\].*?actual_l2=([^ ]+)',content)]
 assert len(update_norms)==30 and all(0<x<1e6 for x in update_norms)
 start=time.time();p=endpoint('BRIDGE_PUB_NLI',57,True,'bridge_single_orderfix');assert p.wait()==0
 single_elapsed=time.time()-start;single_resources=resource_window(start,time.time())
 ss,sm=result('BRIDGE_PUB_NLI',57,'bridge_single_orderfix')
 historical=json.loads((S1/'runs/PUB_NLI/seed_57/metrics.json').read_text())
 bridge=compare(sm['evaluations'],historical['evaluations'][:len(sm['evaluations'])])
 start=time.time();p1=endpoint('BRIDGE_PUB_NLI',57,True,'bridge_dual_orderfix');p2=endpoint('BRIDGE_PUB_NLI',58,True,'bridge_dual_orderfix')
 codes=[p1.wait(),p2.wait()];assert codes==[0,0],codes
 dual_elapsed=time.time()-start;dual_resources=resource_window(start,time.time())
 _,d57=result('BRIDGE_PUB_NLI',57,'bridge_dual_orderfix');_,d58=result('BRIDGE_PUB_NLI',58,'bridge_dual_orderfix')
 dual_repro=compare(sm['evaluations'],d57['evaluations'])
 historical58=json.loads((S1/'runs/PUB_NLI/seed_58/metrics.json').read_text())
 bridge58=compare(d58['evaluations'],historical58['evaluations'][:len(d58['evaluations'])])
 ratio=2*single_elapsed/dual_elapsed
 concurrency=2 if ratio>=1.10 and dual_resources['peak_gpu_memory_mib']<=49140-6144 and dual_resources['peak_fednlp_rss_sum_bytes']<70*1024**3 else 1
 g={'status':'pass','time':now(),'trace_updates':len(update_norms),'trace_min_actual_update_l2':min(update_norms),'historical_seed57_reproduction':bridge,'historical_seed58_reproduction':bridge58,'concurrent_seed57_reproduction':dual_repro,'single_wall_seconds':single_elapsed,'dual_pair_wall_seconds':dual_elapsed,'estimated_throughput_ratio':ratio,'single_resources':single_resources,'dual_resources':dual_resources,'selected_concurrency':concurrency,'selection_rule':'2 only if measured throughput >=1.10x, GPU headroom >=6GiB and aggregate FedNLP RSS <70GiB; otherwise 1','timing_limitation':'One-round bridge smoke including startup/cache. Not paper speedup evidence.','endpoint_code_sha256':digest(CODE/'run_endpoint.py')}
 write(ROOT/'TRAINING_GATE.json',g)
 return g

def summarize():
 from statistics import mean,stdev
 rows=[]
 for arm in ARMS:
  for seed in [57,58,59]:
   p=ROOT/'runs'/arm/('seed_%d'%seed)/'run_spec.json'
   if not p.exists():continue
   s=json.loads(p.read_text());r={'arm':arm,'seed':seed,'status':s['status']}
   if s['status']=='complete':
    _,m=result(arm,seed);es=m['evaluations'];acc=[e['acc'] for e in es]
    r.update(group=s['group'],final_accuracy_percent=100*acc[-1],final_macro_f1=es[-1]['macro_f1'],late10_accuracy_percent=10*sum(acc[-10:]),accuracy_auc=(.5*acc[0]+sum(acc[1:-1])+.5*acc[-1])/50,wall_seconds=s['wall_seconds'])
   rows.append(r)
 complete=[r for r in rows if r['status']=='complete']
 paired=[]
 for seed in [57,58,59]:
  by={r['arm']:r for r in complete if r['seed']==seed}
  for a,b in [('PRIVATE_ZERO','PUB_NLI'),('REAL_MATCHED','PUB_NLI'),('PRIVATE_ZERO','REAL_MATCHED')]:
   if a in by and b in by:paired.append({'seed':seed,'group':'new_100_101','contrast':a+' minus '+b,'final_accuracy_difference_pp':by[a]['final_accuracy_percent']-by[b]['final_accuracy_percent']})
 out=ROOT/'report';out.mkdir(exist_ok=True)
 write(out/'metrics.json',{'complete':len(complete),'target':12,'endpoints':rows,'paired':paired,'new_dp_release':False})
 lines=['# Stage-3 experiment status','','Completed: %d/12. Updated: %s.'%(len(complete),now()),'','Original cohort 1/21: normalized legacy no-guidance ablation. New cohort 100/101: PUB_NLI, PRIVATE_ZERO, REAL_MATCHED. No new DP release; no official test access.','','| Arm | Complete | Final dev accuracy (mean ± sample SD, %) |','|---|---:|---:|']
 for arm in ARMS:
  vals=[r['final_accuracy_percent'] for r in complete if r['arm']==arm]
  val=('%.4f ± %.4f'%(mean(vals),stdev(vals))) if len(vals)>1 else '%.4f'%vals[0] if vals else 'pending'
  lines.append('| %s | %d/3 | %s |'%(arm,len(vals),val))
 lines+=['','These are small-cohort development-set experiments, not full-scale KDD replication or an end-to-end FL DP claim. New clients are from the same uniform partition; this is cohort replication, not distribution shift.','Concurrency timings are operational observations, not system speedup results. Checkpoint diagnostics have a separate completion record.']
 (out/'STAGE3_REPORT.md').write_text('\n'.join(lines)+'\n')

def main():
 os.umask(0o077)
 lock=ROOT/'ORCHESTRATOR_STARTED.json'
 if lock.exists():raise FileExistsError(lock)
 write(lock,{'pid':os.getpid(),'started':now()})
 active=[]
 try:
  g=gate()
  queue=[(a,s) for s in [57,58,59] for a in ARMS]
  write(ROOT/'QUEUE.json',{'status':'running','concurrency':g['selected_concurrency'],'fixed_order':[{'arm':a,'seed':s} for a,s in queue],'target_endpoints':12,'start':now()})
  while queue or active:
   for item in active[:]:
    p,a,s=item
    code=p.poll()
    if code is not None:
     active.remove(item)
     if code:raise RuntimeError('endpoint failure %s seed %d code %d; no automatic retry'%(a,s,code))
     result(a,s);summarize()
   while queue and len(active)<g['selected_concurrency']:
    a,s=queue.pop(0);active.append((endpoint(a,s),a,s));summarize()
    time.sleep(3)
   time.sleep(10)
  summarize();write(ROOT/'QUEUE_COMPLETE.json',{'status':'complete','endpoints':12,'time':now(),'diagnostics':'pending_separate_execution'})
 except BaseException as e:
  write(ROOT/'BLOCKED.json',{'status':'blocked','time':now(),'error':str(e),'traceback':traceback.format_exc(),'active_jobs_allowed_to_finish':[{'arm':a,'seed':s,'pid':p.pid} for p,a,s in active],'automatic_retry':False})
  raise
 finally:
  if not active:(ROOT/'MONITOR_STOP').touch()
if __name__=='__main__':main()
