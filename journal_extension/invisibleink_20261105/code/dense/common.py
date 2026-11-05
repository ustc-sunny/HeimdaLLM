"""Owner-scoped GPU0 privacy-budget experiment; isolated immutable parent artifacts."""
import json,hashlib,os,subprocess,time
from pathlib import Path
ROOT=Path('/data/zzkevin/heimdallm-invink-tradeoff-dense-20261104')
PILOT=Path('/data/zzkevin/heimdallm-invink-pilot-20261103')
GPU='GPU-4a4ad872-cd00-9044-2444-64ec68c6ff55'
GENPY=Path('/home/zzkevin/miniconda3/envs/vllm/bin/python')
FLPY=Path('/home/zzkevin/heimdallm-journal-20260928/envs/kdd/bin/python')
MARKER='HEIMDALLM_INVINK_DENSE=20261104'
def digest(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda:f.read(1048576),b''):h.update(b)
    return h.hexdigest()
def write(p,v):
    p=Path(p);p.parent.mkdir(parents=True,exist_ok=True)
    tmp=p.with_suffix(p.suffix+'.tmp');tmp.write_text(json.dumps(v,indent=2,sort_keys=True)+'\n');os.chmod(tmp,0o600);tmp.replace(p)
_OWN_IDENTITIES={}
def process_identity(pid,proc_root=Path('/proc')):
    p=proc_root/str(pid)
    try:
        fields=(p/'stat').read_text().rsplit(')',1)[1].split()
        return {'uid':p.stat().st_uid,'state':fields[0],'start_ticks':int(fields[19])}
    except FileNotFoundError:return None

def classify_process(pid,proc_root=Path('/proc')):
    try:
        before=process_identity(pid,proc_root)
        if before is None or before['state'] in ['Z','X','x']:return 'gone'
        try:marked=MARKER.encode() in (proc_root/str(pid)/'environ').read_bytes()
        except (FileNotFoundError,PermissionError):marked=False
        after=process_identity(pid,proc_root)
        if after is None or after['state'] in ['Z','X','x']:return 'gone'
        identity=(after['uid'],after['start_ticks'])
        # Cache only identities positively verified earlier; PID reuse cannot inherit ownership.
        if after['uid']==os.getuid() and (marked or _OWN_IDENTITIES.get(pid)==identity):
            if before['start_ticks']!=after['start_ticks']:return 'foreign'
            _OWN_IDENTITIES[pid]=identity
            return 'owned'
        return 'foreign'
    except (PermissionError,ValueError,IndexError):return 'foreign'

def ours(pid):return classify_process(pid)=='owned'
def env():
    e=os.environ.copy()
    e.update(CUDA_VISIBLE_DEVICES=GPU,PYTHONNOUSERSITE='1',HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',
             TOKENIZERS_PARALLELISM='false',OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',
             HEIMDALLM_INVINK_DENSE='20261104',WANDB_MODE='disabled',WANDB_SILENT='true',
             PYTHONHASHSEED='58',OMPI_MCA_btl_vader_single_copy_mechanism='none')
    return e
def gpu_state():
    raw=subprocess.check_output(['nvidia-smi','--query-gpu=index,uuid,memory.free,memory.used,utilization.gpu','--format=csv,noheader,nounits'],text=True)
    row=next(s.split(',') for s in raw.splitlines() if GPU in s)
    apps=subprocess.check_output(['nvidia-smi','--query-compute-apps=gpu_uuid,pid,used_gpu_memory','--format=csv,noheader,nounits'],text=True)
    own=[];foreign=[];exited=[]
    for line in apps.splitlines():
        if GPU not in line:continue
        pid=int(line.split(',')[1])
        classification=classify_process(pid)
        {'owned':own,'foreign':foreign,'gone':exited}[classification].append(pid)
    return {'time':time.time(),'physical_index':int(row[0]),'uuid':GPU,'free_mib':int(row[2]),'used_mib':int(row[3]),'utilization':int(row[4]),'owned_pids':own,'foreign_pids':foreign,'exited_stale_gpu_pids':exited}
def guard(idle=True,min_free=16000):
    for k in range(11):
        s=gpu_state()
        if s['foreign_pids']:raise RuntimeError('GPU0 foreign process; stop '+str(s['foreign_pids']))
        if idle and s['owned_pids']:raise RuntimeError('Own GPU0 task already running')
        if s['free_mib']<min_free:raise RuntimeError('GPU0 insufficient free memory')
        if s['owned_pids'] or s['utilization']<=20:return s
        if k<10:time.sleep(2)
    raise RuntimeError('GPU0 unexplained utilization after settling')

FOLLOWUP=Path('/data/zzkevin/heimdallm-invink-diagnostics-20261103')
OLD=Path('/home/zzkevin/heimdallm-journal-20260928')
REPO=PILOT/'HeimdaLLM'
DATA=OLD/'data/fednlp_data/data_files/agnews_data.h5'
MODEL=OLD/'models/distilbert-base-uncased'
GENMODEL=OLD/'models/Qwen2.5-3B-Instruct'
BUDGETS={'DP_E4':4.,'DP_E12':12.}
SEEDS=(57,58)

def endpoint_complete(directory):
    d=Path(directory)
    if not (d/'metrics.json').exists() or not (d/'run_spec.json').exists():return False
    spec=json.loads((d/'run_spec.json').read_text());m=json.loads((d/'metrics.json').read_text())
    if len(m.get('evaluations',[]))!=51:return False
    if spec['status']=='complete':return spec.get('exit_code')==0 and all(m['checks'].values())
    if spec['status']!='completed_training_teardown_unobserved':return False
    audit=json.loads((d/'completion_recovery.json').read_text())
    return (audit['status']=='pass' and spec.get('exit_code') is None and m['checks'].get('exit_code_zero') is None
        and all(v for k,v in m['checks'].items() if k!='exit_code_zero')
        and digest(d/'train.log')==audit['log_sha256']
        and all(digest(d/'diagnostic_checkpoints'/n)==h for n,h in audit['checkpoint_file_sha256'].items()))
