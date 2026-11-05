"""Bounded serial pipeline; no name collision with Python's queue module."""
import os,subprocess,time,fcntl,traceback,argparse
from common import *
def main():
    os.umask(0o077)
    lock=(ROOT/'queue.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    ap=argparse.ArgumentParser();ap.add_argument('--resume-infrastructure',action='store_true');a=ap.parse_args()
    completed=[]
    if (ROOT/'QUEUE_STATUS.json').exists():
        previous=json.loads((ROOT/'QUEUE_STATUS.json').read_text())
        if not a.resume_infrastructure or previous['status']!='blocked':
            raise FileExistsError('queue has already been launched')
        # Only pre-job guards may be resumed here. Never repeat a model run,
        # generation, DP output, or unfinished train directory automatically.
        if 'GPU3 unexplained utilization' not in previous.get('error',''):
            p=ROOT/'INFRA_RESUME_AUDIT.json'
            if not p.exists():raise RuntimeError('failure requires separate audit, not automatic resume')
            audit=json.loads(p.read_text())
            assert audit['previous_pipeline_end']==previous['end']
            assert audit['status']=='reviewed_infrastructure_only' and audit['completed_training_updates']==0
            assert audit['regenerate_any_text']==False
        write(ROOT/'failures'/('guard_%d.json'%int(time.time())),previous)
        completed=previous['completed_steps']
    status={'status':'running','start':time.time(),'pid':os.getpid(),'completed_steps':completed,'max_new_endpoints':3}
    write(ROOT/'QUEUE_STATUS.json',status)
    try:
        # One immutable artifact for each arm, then three sequential FL endpoints.
        steps=[]
        for arm in ['PUBLIC','NONPRIVATE','DP_E8']:
            steps += [('generate',arm,GENPY,'generate.py'),('pack',arm,FLPY,'pack.py')]
        for arm in ['PUBLIC','NONPRIVATE','DP_E8']:steps += [('train',arm,FLPY,'run_fl.py')]
        for stage,arm,py,script in steps:
            if [stage,arm] in status['completed_steps']:
                if stage=='generate':
                    p=ROOT/'generation'/arm
                    m=json.loads((p/'manifest.json').read_text())
                    assert m['status']=='complete' and digest(p/'synthetic.jsonl')==m['sha256']
                elif stage=='pack':
                    p=ROOT/'guidance'/arm;m=json.loads((p/'manifest.json').read_text())
                    assert digest(p/'cloud_data.h5')==m['cloud_data_sha256']
                else:assert json.loads((ROOT/'runs'/arm/'seed_57/run_spec.json').read_text())['status']=='complete'
                continue
            status.update(stage=stage,arm=arm,step_start=time.time());write(ROOT/'QUEUE_STATUS.json',status)
            if stage!='pack':guard()
            cmd=[str(py),str(ROOT/'code'/script)]+(['--arm',arm] if stage=='generate' else [arm])
            timeout=7200 if stage=='generate' else 15000 if stage=='train' else 300
            logpath=ROOT/'logs'/('%s_%s.log'%(stage,arm));logpath.parent.mkdir(exist_ok=True)
            with logpath.open('x') as log:
                proc=subprocess.Popen(['timeout','--signal=TERM','--kill-after=60s',str(timeout)]+cmd,
                    env=env(),stdout=log,stderr=subprocess.STDOUT)
                status['child_pid']=proc.pid;write(ROOT/'QUEUE_STATUS.json',status)
                code=proc.wait()
            if code:raise RuntimeError('%s %s exit %d'%(stage,arm,code))
            status['completed_steps'].append([stage,arm])
        status.update(status='complete',end=time.time(),stage='done');write(ROOT/'QUEUE_STATUS.json',status)
    except BaseException as ex:
        status.update(status='blocked',error=str(ex),end=time.time());write(ROOT/'QUEUE_STATUS.json',status)
        traceback.print_exc();raise
if __name__=='__main__':main()
