"""Record restricted trainable-state diagnostics without consuming training RNG."""
import hashlib,json,os,random,pickle
from pathlib import Path
import numpy as np
import torch

def capture(model,args,completed_round):
 if completed_round not in (0,10,25,50):return
 if not os.environ.get('HEIMDALLM_STAGE3_RUN_ID'):return
 dest=Path(args.output_dir)/'diagnostic_checkpoints';dest.mkdir(parents=True,exist_ok=True)
 target=dest/('round_%02d.pt'%completed_round)
 if target.exists():raise FileExistsError(str(target))
 before=(pickle.dumps(random.getstate()),pickle.dumps(np.random.get_state()),torch.get_rng_state().clone(),[x.clone() for x in torch.cuda.get_rng_state_all()] if torch.cuda.is_available() else [])
 named={n:p.detach().cpu().clone() for n,p in model.named_parameters() if p.requires_grad}
 buffers={n:b.detach().cpu().clone() for n,b in model.named_buffers()}
 assert sum(x.numel() for x in named.values())==450340
 h=hashlib.sha256()
 for n,x in sorted(named.items()):h.update(n.encode());h.update(x.numpy().tobytes())
 metadata={'completed_round':completed_round,'seed':args.manual_seed,'trainable_count':450340,'trainable_sha256':h.hexdigest(),'kind':'actual_aggregator_model_state','privacy':'restricted_training_state_not_dp','base_model':args.model_name}
 torch.save({'metadata':metadata,'trainable_state':named,'buffers':buffers},target);os.chmod(target,0o600)
 target.with_suffix('.json').write_text(json.dumps(metadata,indent=2)+'\n')
 assert before[0]==pickle.dumps(random.getstate()) and before[1]==pickle.dumps(np.random.get_state()) and torch.equal(before[2],torch.get_rng_state())
 after=torch.cuda.get_rng_state_all() if torch.cuda.is_available() else []
 assert len(before[3])==len(after) and all(torch.equal(a,b) for a,b in zip(before[3],after))
