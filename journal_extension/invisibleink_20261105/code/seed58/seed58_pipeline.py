"""Three serial endpoints with continuous GPU ownership monitoring; no retries."""
import os,subprocess,time,signal,fcntl,traceback
from diag_common import *
def main():
    os.umask(0o077)
    lock=(ROOT/'seed58.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    status_path=ROOT/'SEED58_STATUS.json'
    if status_path.exists():raise FileExistsError('seed58 attempted already')
    assert json.loads((ROOT/'SEED58_REVIEW_GATE.json').read_text())['status']=='pass'
    state={'status':'running','pid':os.getpid(),'start':time.time(),'completed_arms':[],'maximum_new_endpoints':3,'new_generation':False}
    write(status_path,state)
    try:
        for arm in ['PUBLIC','NONPRIVATE','DP_E8']:
            check=guard();state.update(arm=arm,arm_start=time.time(),preflight=check);write(status_path,state)
            logpath=ROOT/'logs'/('seed58_'+arm+'.log');logpath.parent.mkdir(exist_ok=True)
            with logpath.open('x') as log:
                proc=subprocess.Popen([str(FLPY),str(ROOT/'code/run_seed58.py'),arm],cwd=ROOT,env=env(),
                       stdout=log,stderr=subprocess.STDOUT)
                state['child_pid']=proc.pid;write(status_path,state)
                while proc.poll() is None:
                    s=gpu_state();s.update(arm=arm,seed=58)
                    with (ROOT/'FL_RESOURCE_TRACE.jsonl').open('a') as f:f.write(json.dumps(s)+'\n')
                    write(ROOT/'FL_HEALTH.json',s)
                    if s['foreign_pids']:
                        write(ROOT/'RESOURCE_STOP.json',{'time':time.time(),'reason':'foreign GPU0 process; stop own task only'})
                        p=ROOT/'runs'/arm/'seed_58/process.json'
                        if p.exists():
                            ids=json.loads(p.read_text());pid=ids['pid']
                            if ours(pid) and os.getpgid(pid)==ids['pgid']:os.killpg(ids['pgid'],signal.SIGTERM)
                        if ours(proc.pid):os.kill(proc.pid,signal.SIGTERM)
                    # Fatal MPI thread errors may not terminate mpirun. Fail closed.
                    lp=ROOT/'runs'/arm/'seed_58/train.log'
                    if lp.exists():
                        tail=lp.read_text(errors='replace')[-20000:]
                        if any(x in tail for x in ['Traceback (most recent call last):','CUDA out of memory','MPI_ABORT was invoked']):
                            ids=json.loads((lp.parent/'process.json').read_text());pid=ids['pid']
                            if ours(pid) and os.getpgid(pid)==ids['pgid']:os.killpg(ids['pgid'],signal.SIGTERM)
                    time.sleep(15)
                if proc.returncode:raise RuntimeError('seed58 '+arm+' failed, exit '+str(proc.returncode))
            state['completed_arms'].append(arm);write(status_path,state)
            import seed58_report;seed58_report.main()
        state.update(status='complete',end=time.time());write(status_path,state)
    except BaseException as ex:
        state.update(status='blocked',error=str(ex),end=time.time());write(status_path,state);traceback.print_exc();raise
    finally:
        import seed58_report;seed58_report.main()
if __name__=='__main__':main()
