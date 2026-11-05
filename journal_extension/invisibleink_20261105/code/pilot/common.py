"""Isolated, owner-scoped A40 pilot helpers. No shared worktree mutations."""
import hashlib, json, os, subprocess, time
from pathlib import Path
ROOT = Path('/data/zzkevin/heimdallm-invink-pilot-20261103')
OLD = Path('/home/zzkevin/heimdallm-journal-20260928')
REPO = ROOT / 'HeimdaLLM'
GPU = 'GPU-3e744c28-39dd-82d2-288d-673c6d895b70'
GENPY = Path('/home/zzkevin/miniconda3/envs/vllm/bin/python')
FLPY = OLD / 'envs/kdd/bin/python'
DATA = OLD / 'data/fednlp_data/data_files/agnews_data.h5'
MODEL = OLD / 'models/distilbert-base-uncased'
GENMODEL = OLD / 'models/Qwen2.5-3B-Instruct'
def digest(p):
    h = hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda: f.read(1048576), b''): h.update(b)
    return h.hexdigest()
def write(p, obj):
    p = Path(p); p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_suffix(p.suffix + '.tmp')
    tmp.write_text(json.dumps(obj, indent=2, sort_keys=True) + '\n')
    os.chmod(tmp, 0o600); tmp.replace(p)
def env():
    e = os.environ.copy()
    e.update(CUDA_VISIBLE_DEVICES=GPU, PYTHONNOUSERSITE='1',
             HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1',
             TOKENIZERS_PARALLELISM='false', OMP_NUM_THREADS='1', MKL_NUM_THREADS='1',
             WANDB_MODE='disabled', WANDB_SILENT='true', PYTHONHASHSEED='57',
             HEIMDALLM_INVINK_PILOT='20261103',
             OMPI_MCA_btl_vader_single_copy_mechanism='none')
    return e
def guard(require_idle=True, minimum_free=16000, settle_attempt=0):
    s = subprocess.check_output(['nvidia-smi', '--query-gpu=index,uuid,memory.free,utilization.gpu', '--format=csv,noheader,nounits'], text=True)
    row = next(r.split(',') for r in s.splitlines() if GPU in r)
    apps = subprocess.check_output(['nvidia-smi', '--query-compute-apps=gpu_uuid,pid,used_gpu_memory', '--format=csv,noheader,nounits'], text=True)
    owned = []
    for line in apps.splitlines():
        if GPU not in line: continue
        pid = int(line.split(',')[1]); p = Path('/proc') / str(pid)
        try:
            if p.stat().st_uid != os.getuid() or b'HEIMDALLM_INVINK_PILOT=20261103' not in (p/'environ').read_bytes():
                raise RuntimeError('GPU3 has foreign process %d; no job launched' % pid)
        except FileNotFoundError: continue
        owned.append(pid)
    if require_idle and owned: raise RuntimeError('own pilot already running')
    if int(row[2]) < minimum_free: raise RuntimeError('GPU3 insufficient free memory')
    if not owned and int(row[3]) > 20:
        # nvidia-smi utilization is a trailing sample: our previous process may
        # just have exited. Wait and repeat ALL ownership/memory checks.
        if settle_attempt < 10:
            time.sleep(2)
            return guard(require_idle,minimum_free,settle_attempt+1)
        raise RuntimeError('GPU3 unexplained utilization after 20s settling')
    return {'time':time.time(), 'physical_gpu':3, 'uuid':GPU, 'free_mib':int(row[2]), 'utilization':int(row[3]), 'owned_pids':owned}
