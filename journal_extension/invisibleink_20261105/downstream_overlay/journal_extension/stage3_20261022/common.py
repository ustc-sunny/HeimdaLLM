import hashlib,json,os,subprocess
from pathlib import Path
from datetime import datetime,timezone
ROOT=Path('/mnt/heimdallm-stage3-20261022')
REPO=Path('/root/heimdallm-stage3-20261022/HeimdaLLM')
S1=Path('/mnt/heimdallm-stage1-20261015')
PY=Path('/root/miniconda3/envs/heimdallm-kdd/bin/python')
DATA=Path('/root/heimdallm-work/data/fednlp_data/data_files/agnews_data.h5')
PART=Path('/root/heimdallm-work/data/fednlp_data/partition_files/agnews_partition.h5')
MODEL=Path('/root/heimdallm-work/models/distilbert-base-uncased')
CODE=REPO/'journal_extension/stage3_20261022'
BASE='04cb83d5ebb5cfb05e1851629d2578855f2f3906'
EXPECTED={'data':'1474dd264d535b50ad2a7b1cf714bf11e182c2fe5b07f5e982feb31a9b114616','partition':'fd6196bc2518457856365ec8f8207b1eb4b278d165a829320fc5b9fdd0a67f12','fixed':'e16e8551a87ba702cc23b4d56d57b9762b77c4fea8bf341b96d5759be7996a6c','pool':'7e54c05ce2dc94d754b178218963dedfceb781c491b97ed1d9db5ab9e7d7c54f'}
def now():return datetime.now(timezone.utc).isoformat()
def digest(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(1048576),b''):h.update(b)
 return h.hexdigest()
def write(p,v):
 p=Path(p);p.parent.mkdir(parents=True,exist_ok=True);t=p.with_suffix(p.suffix+'.tmp');t.write_text(json.dumps(v,indent=2,sort_keys=True)+'\n');os.chmod(t,0o600);t.replace(p)
def env(seed=57):
 e=os.environ.copy();e.pop('CUDA_VISIBLE_DEVICES',None)
 e.update(PYTHONPATH=':'.join(map(str,[REPO/'experiments/distributed/transformer_exps/run_tc_exps',REPO,REPO/'FedML'])),PYTHONHASHSEED=str(seed),PYTHONNOUSERSITE='1',HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',TOKENIZERS_PARALLELISM='false',OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',WANDB_MODE='disabled',WANDB_SILENT='true',OMPI_MCA_btl_vader_single_copy_mechanism='none',OMPI_ALLOW_RUN_AS_ROOT='1',OMPI_ALLOW_RUN_AS_ROOT_CONFIRM='1')
 return e
