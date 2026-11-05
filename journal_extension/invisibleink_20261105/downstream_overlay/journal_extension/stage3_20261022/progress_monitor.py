"""Record live progress and durable copies; never alters or restarts training."""
import ast,re,time,shutil,csv
from common import *

def snapshot():
 rows=[];curves=[]
 for d in sorted((ROOT/'runs').glob('*/seed_*')):
  if not (d/'run_spec.json').exists():continue
  s=json.loads((d/'run_spec.json').read_text());scratch=Path(s['scratch'])
  p=d/'train.log' if (d/'train.log').exists() else scratch/'train.log'
  content=p.read_text(errors='replace') if p.exists() else ''
  evaluations=[]
  for line in content.splitlines():
   m=re.search(r'eval_model\(\):\s*(\{.*\})\s*$',line)
   if m:
    try:e=ast.literal_eval(m.group(1))
    except (ValueError,SyntaxError):continue
    if 'acc' in e and 'eval_loss' in e:evaluations.append(e)
  for i,e in enumerate(evaluations):curves.append({'arm':s['arm'],'seed':s['seed'],'completed_rounds':i,'dev_accuracy':e['acc'],'macro_f1':e.get('macro_f1'),'eval_loss':e['eval_loss'],'endpoint_complete':s['status']=='complete'})
  updates=content.count('aggregate time cost')
  rows.append({'arm':s['arm'],'seed':s['seed'],'status':s['status'],'completed_rounds':max(0,len(evaluations)-1),'inner_updates_observed':updates,'last_dev_accuracy':evaluations[-1]['acc'] if evaluations else None,'start':s['start']})
  if s['status']!='complete' and p.exists():
   live=ROOT/'durable_live'/s['arm']/('seed_%d'%s['seed']);live.mkdir(parents=True,exist_ok=True)
   shutil.copy2(p,live/'train.log')
   cp=scratch/'model_output/diagnostic_checkpoints'
   for meta in cp.glob('*.json'):
    target=live/'diagnostic_checkpoints';target.mkdir(exist_ok=True)
    if not (target/meta.name).exists():
     shutil.copy2(meta.with_suffix('.pt'),target/meta.with_suffix('.pt').name);shutil.copy2(meta,target/meta.name)
 report={'time':now(),'complete_endpoints':sum(r['status']=='complete' for r in rows),'target_endpoints':12,'observed_total_rounds':sum(r['completed_rounds'] for r in rows),'target_rounds':600,'endpoints':rows,'training_blocked':(ROOT/'BLOCKED.json').exists(),'diagnostics_complete':(ROOT/'DIAGNOSTICS_COMPLETE.json').exists(),'not_final_results':True}
 write(ROOT/'LIVE_PROGRESS.json',report)
 if curves:
  p=ROOT/'report/CURVES_PARTIAL.csv';p.parent.mkdir(exist_ok=True)
  with p.with_suffix('.tmp').open('w') as f:
   w=csv.DictWriter(f,fieldnames=list(curves[0]));w.writeheader();w.writerows(curves)
  p.with_suffix('.tmp').replace(p)
 return report

def main():
 os.umask(0o077)
 for _ in range(14*60):
  try:r=snapshot()
  except Exception as e:
   print(now(),repr(e),flush=True);time.sleep(60);continue
  if r['complete_endpoints']==12 or r['training_blocked']:break
  time.sleep(60)
if __name__=='__main__':main()
