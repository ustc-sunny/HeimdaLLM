"""Detached bounded diagnostic, preserve failures; no auto-training."""
import subprocess,os,time,signal
from diag_common import *
def main():
    os.umask(0o077)
    p=ROOT/'RUNNER_STATUS.json'
    if p.exists():raise FileExistsError('diagnostic already attempted')
    check=guard();start=time.time()
    state={'status':'running','start':start,'gpu':check}
    write(p,state)
    with (ROOT/'diagnostic.log').open('x') as log:
        proc=subprocess.Popen(['timeout','--signal=TERM','--kill-after=30s','1200',str(GENPY),str(ROOT/'code/diagnose_logits.py')],
               cwd=ROOT,env=env(),stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
        state['pid']=proc.pid;state['pgid']=proc.pid;write(p,state)
        while proc.poll() is None:
            s=gpu_state()
            with (ROOT/'resource_trace.jsonl').open('a') as f:f.write(json.dumps(s)+'\n')
            if s['foreign_pids']:
                state['resource_stop']='foreign GPU0 task appeared'
                if ours(proc.pid):os.killpg(proc.pid,signal.SIGTERM)
            time.sleep(5)
        code=proc.returncode
    state.update(exit_code=code,status='complete' if code==0 else 'blocked',end=time.time())
    write(p,state)
if __name__=='__main__':main()
