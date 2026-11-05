"""Persistent health checks; does not stop, retry, or modify training."""
import time,traceback
from common import *

def alive(pid):
 try:os.kill(int(pid),0);return True
 except ProcessLookupError:return False

def check():
 alerts=[];active=[];complete=0
 for p in sorted((ROOT/'runs').glob('*/seed_*/run_spec.json')):
  s=json.loads(p.read_text())
  if s['status']=='complete':complete+=1;continue
  log=Path(s['scratch'])/'train.log';age=time.time()-log.stat().st_mtime if log.exists() else None
  record={'arm':s['arm'],'seed':s['seed'],'status':s['status'],'log_age_seconds':age}
  proc=p.parent/'process.json'
  if proc.exists():
   pid=json.loads(proc.read_text())['pid'];record['process_alive']=alive(pid)
   if not record['process_alive'] and age is not None and age>120:alerts.append('training process absent: '+s['arm'])
  if age is not None and age>300:alerts.append('training log stalled >5min: '+s['arm'])
  if 'failed' in s['status']:alerts.append('endpoint failed: '+s['arm'])
  active.append(record)
 for name in ['BLOCKED.json','DIAGNOSTICS_BLOCKED.json']:
  if (ROOT/name).exists():alerts.append(name+': '+json.loads((ROOT/name).read_text()).get('error','unknown'))
 rec=ROOT/'recovery_20261022_serial/processes.json'
 if rec.exists():
  processes=json.loads(rec.read_text())
  if complete<12 and not alive(processes['controller']):alerts.append('serial queue controller exited before all 12 endpoints')
  if complete==12 and not (ROOT/'DIAGNOSTICS_COMPLETE.json').exists():
   watcher=[x for x in processes['monitors'] if x['script'].endswith('/after_queue.py')][0]
   if not alive(watcher['pid']):alerts.append('diagnostic watcher exited without completion')
 done=complete==12 and (ROOT/'DIAGNOSTICS_COMPLETE.json').exists() and not alerts
 return {'time':now(),'status':'complete' if done else 'attention_required' if alerts else 'running','completed_endpoints':complete,'target_endpoints':12,'diagnostics_complete':(ROOT/'DIAGNOSTICS_COMPLETE.json').exists(),'active':active,'alerts':alerts,'training_automatically_restarted':False}

def main():
 os.umask(0o077);previous=None
 for _ in range(12*60):
  try:
   state=check();write(ROOT/'HEALTH.json',state)
   key=(state['status'],state['completed_endpoints'],str(state['alerts']))
   if key!=previous:
    with (ROOT/'HEALTH_EVENTS.jsonl').open('a') as f:f.write(json.dumps(state)+'\n')
    print(json.dumps(state),flush=True);previous=key
   if state['status']=='complete':return
  except Exception:
   print(traceback.format_exc(),flush=True)
  time.sleep(30)
 write(ROOT/'HEALTH_WATCH_TIMEOUT.json',{'time':now(),'status':'attention_required','reason':'six-hour monitoring window elapsed'})
if __name__=='__main__':main()
