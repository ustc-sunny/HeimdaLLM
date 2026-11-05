"""Exactly two epsilon16 endpoints; no generation and no epsilon20 code path."""
import fcntl,signal,subprocess,time,traceback,shutil
from common import *
from runtime_policy import resource_stop_reason

EXPECTED_INIT={57:'14b42621935db8b4fc75724b193b366d194bd8757541da410a75cf8cb7d5243b',
               58:'c71983cc423bec2d2827603811257cd26829c4acdc6c51ceb226b8dd6fab1cd5'}

def stop_own(proc,seed,reason):
    groups=[proc.pid]
    p=ROOT/'runs/DP_E16'/('seed_%d'%seed)/'process.json'
    if p.exists():
        ids=json.loads(p.read_text())
        try:
            if ours(ids['pid']) and os.getpgid(ids['pid'])==ids['pgid']:groups.append(ids['pgid'])
        except ProcessLookupError:pass
    for sig in [signal.SIGTERM,signal.SIGKILL]:
        for pgid in groups:
            try:
                if ours(pgid) and os.getpgid(pgid)==pgid:
                    os.killpg(pgid,sig)
                    with (ROOT/'STOP_SIGNALS.jsonl').open('a') as f:
                        f.write(json.dumps({'time':time.time(),'pgid':pgid,'signal':int(sig),'reason':reason,'issued_by':'own_pipeline_monitor'})+'\n')
            except ProcessLookupError:pass
        if sig==signal.SIGTERM:time.sleep(3)

def admit(seed,state):
    start=time.time()
    while True:
        try:return guard()
        except RuntimeError as ex:
            state.update(status='waiting_for_gpu',stage='admission',seed=seed,
                         resource_wait_reason=str(ex),resource_wait_started=start)
            write(ROOT/'QUEUE_STATUS.json',state)
            if time.time()-start>=3600:raise TimeoutError('GPU0 did not pass admission within1hour') from ex
            time.sleep(30)

def mirror_checkpoints(seed):
    source=ROOT/'scratch/DP_E16'/('seed_%d'%seed)/'model_output/diagnostic_checkpoints'
    dest=ROOT/'live_checkpoints/DP_E16'/('seed_%d'%seed)
    for meta in source.glob('round_*.json'):
        try:data=json.loads(meta.read_text())
        except (ValueError,OSError):continue
        pt=meta.with_suffix('.pt')
        if not pt.is_file():continue
        if (dest/meta.name).exists():continue
        dest.mkdir(parents=True,exist_ok=True)
        shutil.copy2(pt,dest/pt.name);shutil.copy2(meta,dest/meta.name)
    initial=source/'round_00.json'
    if initial.exists():
        assert json.loads(initial.read_text())['trainable_sha256']==EXPECTED_INIT[seed],'Initialization mismatch'

def main():
    os.umask(0o077)
    lock=(ROOT/'queue.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    assert json.loads((ROOT/'SCOPE.json').read_text())['active_new_arms']==['DP_E16']
    assert json.loads((ROOT/'RESOURCE_AUTHORIZATION.json').read_text())['authorized']
    assert json.loads((ROOT/'REVIEW_GATE.json').read_text())['status']=='pass'
    for name in ['PROCESS_GUARD_TESTS.json','RUNTIME_POLICY_TESTS.json']:
        assert json.loads((ROOT/'code'/name).read_text())['status']=='pass'
    if (ROOT/'QUEUE_STATUS.json').exists():raise FileExistsError('This completion attempt already launched; no automatic retry')
    state={'status':'running','pid':os.getpid(),'start':time.time(),'completed_steps':[],
           'maximum_new_endpoints':2,'new_generation_artifacts':0,'reused_endpoints':12,
           'active_epsilon_curve':list(CURVE_EPSILONS),'excluded_budgets':[20],
           'historical_teardown_unobserved_endpoints':1}
    write(ROOT/'QUEUE_STATUS.json',state)
    try:
        for seed in SEEDS:
            preflight=admit(seed,state)
            state.update(status='running',stage='train',arm='DP_E16',seed=seed,
                         step_start=time.time(),preflight=preflight)
            state.pop('resource_wait_reason',None)
            write(ROOT/'QUEUE_STATUS.json',state)
            lp=ROOT/'logs'/('train_DP_E16_%d.log'%seed);lp.parent.mkdir(exist_ok=True)
            with lp.open('x') as log:
                proc=subprocess.Popen([str(FLPY),str(ROOT/'code/run_fl.py'),'DP_E16',str(seed)],
                    cwd=ROOT,env=env(),stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
                state['child_pid']=proc.pid;write(ROOT/'QUEUE_STATUS.json',state)
                low_since=None;last_foreign=[]
                try:
                    while proc.poll() is None:
                        s=gpu_state();s.update(stage='train',arm='DP_E16',seed=seed)
                        with (ROOT/'RESOURCE_TRACE.jsonl').open('a') as f:f.write(json.dumps(s)+'\n')
                        write(ROOT/'HEALTH.json',s)
                        if s['foreign_pids']!=last_foreign:
                            with (ROOT/'CO_TENANCY_EVENTS.jsonl').open('a') as f:
                                f.write(json.dumps(dict(s,action='log_only; healthy existing training continues'))+'\n')
                            last_foreign=s['foreign_pids']
                        reason,low_since=resource_stop_reason(s,low_since,time.time())
                        if reason:raise RuntimeError(reason)
                        if time.time()-state['step_start']>9000:raise TimeoutError('Training endpoint exceeded150minutes')
                        mirror_checkpoints(seed)
                        trainlog=ROOT/'runs/DP_E16'/('seed_%d'%seed)/'train.log'
                        if trainlog.exists():
                            with trainlog.open('rb') as f:
                                f.seek(max(0,trainlog.stat().st_size-20000));tail=f.read().decode(errors='replace')
                            if any(x in tail for x in ['Traceback (most recent call last):','CUDA out of memory','MPI_ABORT was invoked']):
                                raise RuntimeError('Fatal training log detected')
                            if time.time()-trainlog.stat().st_mtime>300:raise TimeoutError('No training log activity for300seconds')
                        time.sleep(10)
                    if proc.returncode:raise RuntimeError('DP_E16 seed%d exit%d'%(seed,proc.returncode))
                except BaseException as ex:
                    try:mirror_checkpoints(seed)
                    except Exception as mirror_error:
                        write(ROOT/'CHECKPOINT_MIRROR_ERROR.json',{'seed':seed,'error':str(mirror_error),'time':time.time()})
                    finally:stop_own(proc,seed,str(ex))
                    raise
            assert endpoint_complete(ROOT/'runs/DP_E16'/('seed_%d'%seed))
            state['completed_steps'].append(['train','DP_E16',seed]);write(ROOT/'QUEUE_STATUS.json',state)
            import report;report.main()
        state.update(status='complete',stage='done',end=time.time());write(ROOT/'QUEUE_STATUS.json',state)
    except BaseException as ex:
        state.update(status='blocked',error=str(ex),end=time.time());write(ROOT/'QUEUE_STATUS.json',state)
        traceback.print_exc();raise
    finally:
        import report;report.main()
        if state['status']=='complete':
            files={}
            for p in sorted(ROOT.rglob('*')):
                rel=p.relative_to(ROOT)
                if p.is_file() and not any(x in rel.parts for x in ['scratch','__pycache__']) and p.name not in ['FINAL_BACKUP_MANIFEST.json','launcher.log']:
                    files[str(rel)]={'sha256':digest(p),'bytes':p.stat().st_size}
            write(ROOT/'FINAL_BACKUP_MANIFEST.json',{'status':'complete','time':time.time(),'files':files})

if __name__=='__main__':main()
