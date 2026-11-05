"""Restricted, no-sampling fixed-prefix diagnostic. No new DP release or FL run."""
import sys,json,os,time
from pathlib import Path
from diag_common import *
sys.path.insert(0,str(PILOT/'code'))
from generate import prompt
from mechanism import calibration,scores,support,probabilities
import numpy as np

def distribution_stats(public,private,cfg):
    C=cfg['clip'];mask=support(public,100,C,3)
    ids,pp=probabilities(public,mask,1.)
    raw=scores(public,private,'NONPRIVATE',C);dp=scores(public,private,'DP_E8',C)
    _,pn=probabilities(raw,mask,1.);_,pd=probabilities(dp,mask,1.)
    delta=private-public;clipped=np.clip(delta,-C,C)
    # Given identical PUBLIC prefix/support, zero private contribution is exact.
    zero=scores(public,private,'DP_E8',0.)
    zero_error=float(np.max(np.abs(zero-public)));assert zero_error==0.
    assert np.isfinite(dp).all() and np.max(np.abs(dp-public))<=C+1e-10
    tv_np=float(abs(pn-pp).sum()/2);tv_dp=float(abs(pd-pp).sum()/2)
    return {'support_size':int(mask.sum()),
        'raw_reference_delta_abs_median_support':float(np.median(abs(delta[:,mask]))),
        'clipping_fraction_support':float((abs(delta[:,mask])>C).mean()),
        'clipping_fraction_public_probability_weighted':float(np.mean(((abs(delta[:,ids])>C)*pp).sum(axis=1))),
        'tv_nonprivate_vs_public':tv_np,'tv_dp_vs_public':tv_dp,
        'tv_dp_vs_nonprivate':float(abs(pd-pn).sum()/2),
        'relative_tv_deviation_from_public':tv_dp/tv_np if tv_np>1e-12 else None,
        'argmax_dp_matches_public':int(np.argmax(pd)==np.argmax(pp)),
        'argmax_dp_matches_nonprivate':int(np.argmax(pd)==np.argmax(pn)),
        'zero_contribution_max_abs_error':zero_error}

def describe(rows):
    out={}
    for k in rows[0]:
        if k in ['client','group','target_label','prefix_tokens','same_label_refs']:continue
        a=np.array([r[k] for r in rows if r[k] is not None],dtype=float)
        if len(a):out[k]={'mean':float(a.mean()),'median':float(np.median(a)),'p10':float(np.quantile(a,.1)),'p90':float(np.quantile(a,.9))}
    return out

def main():
    os.umask(0o077)
    dest=ROOT/'fixed_prefix'
    if dest.exists():raise FileExistsError('Diagnostic output exists; preserve it')
    check=guard();dest.mkdir(parents=True)
    cfg=calibration();start=time.time()
    for arm in ['PUBLIC','NONPRIVATE','DP_E8']:
        p=PILOT/'generation'/arm;m=json.loads((p/'manifest.json').read_text())
        assert digest(p/'synthetic.jsonl')==m['sha256']
    code=json.loads((PILOT/'DEPLOYMENT_MANIFEST_FINAL.json').read_text())['code_sha256']
    for name in ['generate.py','mechanism.py']:assert digest(PILOT/'code'/name)==code[name]
    import torch
    from transformers import AutoTokenizer,AutoModelForCausalLM
    from common import GENMODEL
    assert torch.cuda.device_count()==1
    assert 'GPU-'+str(torch.cuda.get_device_properties(0).uuid).removeprefix('GPU-')==GPU
    torch.set_num_threads(1)
    tok=AutoTokenizer.from_pretrained(str(GENMODEL),local_files_only=True,trust_remote_code=False)
    model=AutoModelForCausalLM.from_pretrained(str(GENMODEL),local_files_only=True,trust_remote_code=False,
            torch_dtype=torch.bfloat16,attn_implementation='eager').to('cuda:0').eval()
    public=[json.loads(l) for l in (PILOT/'generation/PUBLIC/synthetic.jsonl').read_text().splitlines()]
    allrows=[]
    write(dest/'STATUS.json',{'status':'running','completed_groups':0,'total_groups':64,'gpu_preflight':check})
    with torch.inference_mode(),(dest/'RESTRICTED_PREFIX_METRICS.jsonl').open('x') as fh:
        for client in (100,101):
            refs=[json.loads(l) for l in (PILOT/'cohort/restricted'/('client_%d.jsonl'%client)).read_text().splitlines()]
            assert len(refs)==120
            for i in range(32):
                guard(idle=False,min_free=2000)
                label=str(i%4+1);rr=refs[3*i:3*i+3]
                pubtext=public[(client-100)*32+i]['text']
                prefix_ids=tok.encode(pubtext,add_special_tokens=False)
                positions=[p for p in (0,4,12,20) if p<=len(prefix_ids)]
                prompts=[prompt(tok,label)]+[prompt(tok,label,r) for r in rr]
                by_row=[]
                for pr in prompts:
                    n=pr.shape[1]
                    suffix=torch.tensor([prefix_ids[:max(positions)]],dtype=torch.long)
                    inp=torch.cat([pr,suffix],dim=1).to('cuda:0')
                    output=model(input_ids=inp,use_cache=False)
                    logits=output.logits[0,[n+p-1 for p in positions]].float().cpu().numpy().astype('float64')
                    assert np.isfinite(logits).all();by_row.append(logits)
                    del output,inp
                for j,p in enumerate(positions):
                    values=distribution_stats(by_row[0][j],np.stack([row[j] for row in by_row[1:]]),cfg)
                    values.update(client=client,group=i,target_label=label,prefix_tokens=p,
                                  same_label_refs=sum(r['label']==label for r in rr))
                    allrows.append(values);fh.write(json.dumps(values)+'\n');fh.flush()
                del by_row,prompts
                done=(client-100)*32+i+1
                write(dest/'STATUS.json',{'status':'running','completed_groups':done,'total_groups':64,
                    'prefix_observations':len(allrows),'wall_seconds':time.time()-start,
                    'peak_allocated_mib':torch.cuda.max_memory_allocated()/2**20})
                print(json.dumps({'completed_groups':done,'seconds':round(time.time()-start,1)}),flush=True)
    result={'status':'complete','groups':64,'prefix_observations':len(allrows),'calibration':cfg,
        'overall':describe(allrows),'by_target_class':{c:describe([r for r in allrows if r['target_label']==c]) for c in ['1','2','3','4']},
        'by_prefix':{str(p):describe([r for r in allrows if r['prefix_tokens']==p]) for p in (0,4,12,20)},
        'by_reference_alignment':{k:describe([r for r in allrows if (r['same_label_refs']==0)==(k=='zero')]) for k in ['zero','at_least_one']},
        'wall_seconds':time.time()-start,'peak_allocated_mib':torch.cuda.max_memory_allocated()/2**20,
        'new_generated_texts':0,'private_logit_arrays_saved':False,'secret_rng_state_accessed':False,
        'privacy':'RESTRICTED NON-DP diagnostics; not part of public DP artifact',
        'limitations':'Teacher forcing on re-tokenized PUBLIC text prefixes; not exact replay of private trajectories; distribution differences do not establish accuracy causation.'}
    write(dest/'DIAGNOSTIC_RESULTS.json',result)
    write(dest/'STATUS.json',{'status':'complete','groups':64,'prefix_observations':len(allrows),'wall_seconds':result['wall_seconds']})
    print(json.dumps({'status':'complete','prefix_observations':len(allrows),'wall_seconds':result['wall_seconds']}),flush=True)
if __name__=='__main__':main()
