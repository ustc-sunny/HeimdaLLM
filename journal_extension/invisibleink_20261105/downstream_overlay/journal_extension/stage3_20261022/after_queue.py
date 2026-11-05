"""Run four predeclared common-checkpoint diagnostics after all endpoints finish."""
import time,traceback
from common import *

def main():
 os.umask(0o077)
 marker=ROOT/'DIAGNOSTIC_WATCHER_STARTED.json'
 if marker.exists():raise FileExistsError(marker)
 write(marker,{'pid':os.getpid(),'time':now(),'anchor':'new cohort PUB_NLI seed57; rounds 0,10,25,50','reference':'64 REAL_MATCHED guidance records'})
 try:
  deadline=time.monotonic()+14*3600
  while not (ROOT/'QUEUE_COMPLETE.json').exists():
   if (ROOT/'BLOCKED.json').exists():raise RuntimeError('training queue blocked; diagnostics not started')
   if time.monotonic()>deadline:raise TimeoutError('queue not complete within 14h')
   time.sleep(30)
  assert json.loads((ROOT/'QUEUE_COMPLETE.json').read_text())['endpoints']==12
  reports=[]
  for r in [0,10,25,50]:
   e=env(57);e['HEIMDALLM_STAGE3_RUN_ID']='diagnostic_round_%02d'%r
   with (ROOT/('diagnostic_round_%02d.log'%r)).open('x') as f:
    p=subprocess.run([str(PY),str(CODE/'diagnose_checkpoints.py'),'--round',str(r)],cwd=REPO,env=e,stdout=f,stderr=subprocess.STDOUT,timeout=900)
   if p.returncode:raise RuntimeError('diagnostic failed round %d; no automatic retry'%r)
   result=ROOT/'restricted_diagnostics'/('round_%02d'%r)/'diagnostics.json'
   reports.append(json.loads(result.read_text()))
  write(ROOT/'report/COMMON_CHECKPOINT_DIAGNOSTICS.json',{'status':'complete','source_trajectory':'new_100_101 PUB_NLI seed57','rounds':[0,10,25,50],'reports':reports,'scope':'matched-real 64-record reference, single trajectory; not full-population gradient or multi-seed convergence proof'})
  write(ROOT/'DIAGNOSTICS_COMPLETE.json',{'status':'complete','time':now(),'rounds':[0,10,25,50]})
 except BaseException as e:
  write(ROOT/'DIAGNOSTICS_BLOCKED.json',{'status':'blocked','time':now(),'error':str(e),'traceback':traceback.format_exc(),'automatic_retry':False})
  raise
if __name__=='__main__':main()
