"""One bounded serial queue, owner-scoped GPU monitoring, no rerolls or retries."""
import fcntl, signal, subprocess, time, traceback
from common import *

EXPECTED_INIT = {
    57: '14b42621935db8b4fc75724b193b366d194bd8757541da410a75cf8cb7d5243b',
    58: 'c71983cc423bec2d2827603811257cd26829c4acdc6c51ceb226b8dd6fab1cd5',
}

def stop_own(proc, stage, arm, seed):
    groups = [proc.pid]
    if stage == 'train':
        p = ROOT/'runs'/arm/('seed_%d'%seed)/'process.json'
        if p.exists():
            ids = json.loads(p.read_text())
            if ours(ids['pid']):
                try:
                    if os.getpgid(ids['pid']) == ids['pgid']: groups.append(ids['pgid'])
                except ProcessLookupError: pass
    for sig in [signal.SIGTERM, signal.SIGKILL]:
        for pgid in groups:
            try:
                if ours(pgid) and os.getpgid(pgid) == pgid: os.killpg(pgid, sig)
            except ProcessLookupError: pass
        if sig == signal.SIGTERM: time.sleep(3)

def main():
    os.umask(0o077)
    lock=(ROOT/'queue.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    assert json.loads((ROOT/'RESOURCE_AUTHORIZATION.json').read_text())['authorized'] is True
    assert json.loads((ROOT/'REVIEW_GATE.json').read_text())['status']=='pass'
    status_path=ROOT/'QUEUE_STATUS.json'
    if status_path.exists(): raise FileExistsError('Queue already attempted; no automatic retry')
    steps=[('train','DP_E4',57),('train','DP_E4',58),
           ('generate','DP_E12',None),('pack','DP_E12',None),
           ('train','DP_E12',57),('train','DP_E12',58)]
    state={'status':'running','pid':os.getpid(),'start':time.time(),'completed_steps':[],
           'maximum_new_endpoints':4,'maximum_new_generation_artifacts':1,'reused_endpoints':6,
           'active_epsilon_curve':[4,8,12],'reused_generation_artifacts':['DP_E4']}
    write(status_path,state)
    try:
        for stage,arm,seed in steps:
            preflight=guard()
            state.update(stage=stage,arm=arm,seed=seed,step_start=time.time(),preflight=preflight)
            write(status_path,state)
            if stage=='generate':cmd=[str(GENPY),str(ROOT/'code/generate.py'),'--arm',arm];limit=1800
            elif stage=='pack':cmd=[str(FLPY),str(ROOT/'code/pack.py'),arm];limit=300
            else:cmd=[str(FLPY),str(ROOT/'code/run_fl.py'),arm,str(seed)];limit=9000
            logpath=ROOT/'logs'/('%s_%s%s.log'%(stage,arm,('_'+str(seed)) if seed else ''))
            logpath.parent.mkdir(exist_ok=True)
            with logpath.open('x') as log:
                proc=subprocess.Popen(cmd,cwd=ROOT,env=env(),stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
                state['child_pid']=proc.pid;write(status_path,state)
                try:
                    while proc.poll() is None:
                        s=gpu_state();s.update(stage=stage,arm=arm,seed=seed)
                        with (ROOT/'RESOURCE_TRACE.jsonl').open('a') as f:f.write(json.dumps(s)+'\n')
                        write(ROOT/'HEALTH.json',s)
                        if s['foreign_pids']:raise RuntimeError('Foreign process on approved GPU0; stop own task only')
                        if time.time()-state['step_start']>limit:raise TimeoutError('Bounded stage timeout')
                        if stage=='train':
                            cp=ROOT/'scratch'/arm/('seed_%d'%seed)/'model_output/diagnostic_checkpoints/round_00.json'
                            if cp.exists():
                                init=json.loads(cp.read_text())
                                if init['trainable_sha256']!=EXPECTED_INIT[seed]:raise RuntimeError('Initialization mismatch')
                            lp=ROOT/'runs'/arm/('seed_%d'%seed)/'train.log'
                            if lp.exists():
                                with lp.open('rb') as f:
                                    f.seek(max(0,lp.stat().st_size-20000));tail=f.read().decode(errors='replace')
                                if any(x in tail for x in ['Traceback (most recent call last):','CUDA out of memory','MPI_ABORT was invoked']):
                                    raise RuntimeError('Fatal training log detected')
                                if time.time()-lp.stat().st_mtime>300:raise TimeoutError('No training log activity for 300 seconds')
                        time.sleep(10)
                    if proc.returncode:raise RuntimeError('%s %s seed=%s exit=%s'%(stage,arm,seed,proc.returncode))
                except BaseException:
                    stop_own(proc,stage,arm,seed)
                    raise
            state['completed_steps'].append([stage,arm,seed]);write(status_path,state)
            import report;report.main()
        state.update(status='complete',stage='done',end=time.time());write(status_path,state)
    except BaseException as ex:
        state.update(status='blocked',error=str(ex),end=time.time());write(status_path,state)
        traceback.print_exc();raise
    finally:
        import report;report.main()
        if state['status']=='complete':
            manifest={}
            for p in sorted(ROOT.rglob('*')):
                rel=p.relative_to(ROOT)
                if p.is_file() and not any(x in rel.parts for x in ['scratch','__pycache__']) and p.name not in ['FINAL_BACKUP_MANIFEST.json','launcher.log']:
                    manifest[str(rel)]={'sha256':digest(p),'bytes':p.stat().st_size}
            write(ROOT/'FINAL_BACKUP_MANIFEST.json',{'status':'complete','time':time.time(),'files':manifest})

if __name__=='__main__':main()
