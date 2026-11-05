"""Authorized GPU0 diagnostics/seed58 helpers, isolated from completed pilot."""
import json,hashlib,os,subprocess,time
from pathlib import Path
ROOT=Path('/data/zzkevin/heimdallm-invink-diagnostics-20261103')
PILOT=Path('/data/zzkevin/heimdallm-invink-pilot-20261103')
GPU='GPU-4a4ad872-cd00-9044-2444-64ec68c6ff55'
GENPY=Path('/home/zzkevin/miniconda3/envs/vllm/bin/python')
FLPY=Path('/home/zzkevin/heimdallm-journal-20260928/envs/kdd/bin/python')
MARKER='HEIMDALLM_INVINK_FOLLOWUP=20261103'
def digest(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda:f.read(1048576),b''):h.update(b)
    return h.hexdigest()
def write(p,v):
    p=Path(p);p.parent.mkdir(parents=True,exist_ok=True)
    tmp=p.with_suffix(p.suffix+'.tmp');tmp.write_text(json.dumps(v,indent=2,sort_keys=True)+'\n');os.chmod(tmp,0o600);tmp.replace(p)
def ours(pid):
    p=Path('/proc')/str(pid)
    try:return p.stat().st_uid==os.getuid() and MARKER.encode() in (p/'environ').read_bytes()
    except (FileNotFoundError,PermissionError):return False
def env():
    e=os.environ.copy()
    e.update(CUDA_VISIBLE_DEVICES=GPU,PYTHONNOUSERSITE='1',HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',
             TOKENIZERS_PARALLELISM='false',OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',
             HEIMDALLM_INVINK_FOLLOWUP='20261103',WANDB_MODE='disabled',WANDB_SILENT='true',
             PYTHONHASHSEED='58',OMPI_MCA_btl_vader_single_copy_mechanism='none')
    return e
def gpu_state():
    raw=subprocess.check_output(['nvidia-smi','--query-gpu=index,uuid,memory.free,memory.used,utilization.gpu','--format=csv,noheader,nounits'],text=True)
    row=next(s.split(',') for s in raw.splitlines() if GPU in s)
    apps=subprocess.check_output(['nvidia-smi','--query-compute-apps=gpu_uuid,pid,used_gpu_memory','--format=csv,noheader,nounits'],text=True)
    own=[];foreign=[]
    for line in apps.splitlines():
        if GPU not in line:continue
        pid=int(line.split(',')[1])
        if not (Path('/proc')/str(pid)).exists():continue
        (own if ours(pid) else foreign).append(pid)
    return {'time':time.time(),'physical_index':int(row[0]),'uuid':GPU,'free_mib':int(row[2]),'used_mib':int(row[3]),'utilization':int(row[4]),'owned_pids':own,'foreign_pids':foreign}
def guard(idle=True,min_free=16000):
    for k in range(11):
        s=gpu_state()
        if s['foreign_pids']:raise RuntimeError('GPU0 foreign process; stop '+str(s['foreign_pids']))
        if idle and s['owned_pids']:raise RuntimeError('Own GPU0 task already running')
        if s['free_mib']<min_free:raise RuntimeError('GPU0 insufficient free memory')
        if s['owned_pids'] or s['utilization']<=20:return s
        if k<10:time.sleep(2)
    raise RuntimeError('GPU0 unexplained utilization after settling')
