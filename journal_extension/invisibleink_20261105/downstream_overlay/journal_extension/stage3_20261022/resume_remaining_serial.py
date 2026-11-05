"""Reviewed recovery: run only three never-started seed59 endpoints, serially."""
import shutil,time,traceback
from common import *
from orchestrate import result,summarize
from progress_monitor import snapshot
from run_endpoint import gpu_check

def main():
 os.umask(0o077)
 recovery=ROOT/'recovery_20261022_serial'
 if recovery.exists():raise FileExistsError(recovery)
 completed=[]
 for arm in ['NO_GUIDANCE_TRACE_LEGACY','PUB_NLI','PRIVATE_ZERO','REAL_MATCHED']:
  for seed in [57,58,59]:
   d=ROOT/'runs'/arm/('seed_%d'%seed)
   if d.exists():
    s,m=result(arm,seed)
    assert s['status']=='complete' and len(m['evaluations'])==51 and all(m['checks'].values())
    completed.append({'arm':arm,'seed':seed})
 assert len(completed)==9
 remaining=['PUB_NLI','PRIVATE_ZERO','REAL_MATCHED']
 for arm in remaining:assert not (ROOT/'runs'/arm/'seed_59').exists()
 preflight=gpu_check();assert not preflight['owned_gpu_pids']
 recovery.mkdir(mode=0o700)
 for name in ['BLOCKED.json','DIAGNOSTICS_BLOCKED.json','DIAGNOSTIC_WATCHER_STARTED.json','MONITOR_STOP']:
  p=ROOT/name
  if p.exists():p.rename(recovery/name)
 shutil.copy2(ROOT/'launcher_formal_PUB_NLI_59.log',recovery/'prelaunch_failure.log')
 write(recovery/'REVIEW.json',{'time':now(),'validated_completed':completed,'remaining':[{'arm':a,'seed':59} for a in remaining],'concurrency':1,'gpu_preflight':preflight,'cause':'GPU guard could not attribute sampled utilization to owned visible processes; blocked before PUB_NLI seed59 output directory/training existed. Exact attribution failure not proven.','review':'Three fresh idle GPU samples showed 3MiB/0% and no compute PIDs. Preserve guard unchanged; use serial execution.','prior_eta_invalidated':True,'not_accuracy_based_retry':True,'completed_outputs_untouched':True,'endpoint_code_sha256':digest(CODE/'run_endpoint.py')})
 monitors=[]
 try:
  for script,extra,logname in [
   (CODE/'progress_monitor.py',[],'progress.log'),
   (REPO/'journal_extension/stage2_20261016/monitor_resources.py',['--output',str(recovery/'resources.csv'),'--stop-file',str(recovery/'MONITOR_STOP')],'resources.log'),
   (CODE/'after_queue.py',[],'diagnostics_watcher.log')]:
   with (recovery/logname).open('x') as f:
    p=subprocess.Popen([str(PY),str(script)]+extra,env=env(),stdout=f,stderr=subprocess.STDOUT,start_new_session=True)
    monitors.append({'script':str(script),'pid':p.pid})
  write(recovery/'processes.json',{'controller':os.getpid(),'monitors':monitors})
  summarize();snapshot()
  for arm in remaining:
   # Cooldown is read-only. Keep the original endpoint GPU guard intact.
   for attempt in range(18):
    raw=subprocess.check_output(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader,nounits'],text=True).strip()
    mem,util=[int(x.strip()) for x in subprocess.check_output(['nvidia-smi','--query-gpu=memory.used,utilization.gpu','--format=csv,noheader,nounits'],text=True).splitlines()[0].split(',')]
    if not raw and mem<500 and util<=20:break
    time.sleep(5)
   else:raise RuntimeError('GPU did not become idle; no launch')
   with (recovery/('launch_'+arm+'_59.log')).open('x') as f:
    p=subprocess.run([str(PY),str(CODE/'run_endpoint.py'),'--arm',arm,'--seed','59'],env=env(59),stdout=f,stderr=subprocess.STDOUT)
   if p.returncode:raise RuntimeError('endpoint failed '+arm+'; no automatic retry')
   result(arm,59);summarize();snapshot()
  write(ROOT/'QUEUE_COMPLETE.json',{'status':'complete','endpoints':12,'time':now(),'recovery':'reviewed serial launch of three never-started endpoints','diagnostics':'pending_separate_execution'})
 except BaseException as e:
  write(ROOT/'BLOCKED.json',{'status':'blocked','time':now(),'error':str(e),'traceback':traceback.format_exc(),'automatic_retry':False,'recovery':str(recovery)})
  raise
 finally:
  (recovery/'MONITOR_STOP').touch()
if __name__=='__main__':main()
