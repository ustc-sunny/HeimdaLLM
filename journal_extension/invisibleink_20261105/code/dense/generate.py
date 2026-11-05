"""Fixed one-artifact-per-arm generator; never rerolls or filters on private data."""
import argparse, json, os, random, secrets, time
from pathlib import Path
from common import ROOT, PILOT, GENMODEL, BUDGETS, guard, write, digest
from mechanism import calibration, scores, support, probabilities, sample

NAMES={'1':'World news and international affairs','2':'Sports',
       '3':'Business, finance and markets','4':'Science and technology'}
SYSTEM='You write short English news reports for a four-category news classification dataset.'

def prompt(tok, label, record=None):
    instruction=('Write one realistic English news report of about 50 words. '
        'The required output category is '+NAMES[label]+'. '
        'Output only the news report, with no category heading, explanation, or quotation marks.')
    if record is not None:
        # One independently bounded complete record per row; label is PRIVATE too.
        ref=tok.decode(tok.encode(record['text'],add_special_tokens=False)[:160],skip_special_tokens=True)
        instruction+=('\nUse the following private example only as a reference for news style and vocabulary. '
            'Its category may differ: follow the REQUIRED OUTPUT CATEGORY above. '
            'Write a new report rather than repeating the example.\nREFERENCE CATEGORY: '
            +NAMES[str(record['label'])]+'\nREFERENCE REPORT:\n'+ref)
    return tok.apply_chat_template([{'role':'system','content':SYSTEM},{'role':'user','content':instruction}],
                                  tokenize=True,add_generation_prompt=True,return_tensors='pt')

def main():
    os.umask(0o077)
    ap=argparse.ArgumentParser();ap.add_argument('--arm',required=True,choices=list(BUDGETS));a=ap.parse_args()
    assert a.arm=='DP_E12', 'Only one new epsilon12 generation is approved; epsilon4 is reused'
    root=ROOT/'generation'/a.arm
    if root.exists(): raise FileExistsError('Artifact directory exists; no retry or replacement: '+str(root))
    check=guard(); root.mkdir(parents=True)
    cfg=calibration(epsilon=BUDGETS[a.arm]); t0=time.monotonic()
    if a.arm in BUDGETS:
        assert json.loads((ROOT/'code/CPU_TESTS.json').read_text())['status']=='pass'
        assert json.loads((ROOT/'PRIVACY_GATE.json').read_text())['status']=='reviewed_research_pilot'
        assert json.loads((ROOT/'RESOURCE_AUTHORIZATION.json').read_text())['authorized'] is True
    import torch
    from transformers import AutoTokenizer,AutoModelForCausalLM
    assert torch.cuda.device_count()==1
    torch.set_num_threads(1)
    tok=AutoTokenizer.from_pretrained(str(GENMODEL),local_files_only=True,trust_remote_code=False)
    model=AutoModelForCausalLM.from_pretrained(str(GENMODEL),local_files_only=True,trust_remote_code=False,
             torch_dtype=torch.bfloat16,attn_implementation='eager').to('cuda:0').eval()
    assert not model.training
    eos=model.generation_config.eos_token_id
    eos=set(eos if isinstance(eos,list) else [eos])
    rng=random.Random(5703) if a.arm not in BUDGETS else None
    total_tokens=0; fallback=0; rows=[]
    write(root/'status.json',{'status':'generating','arm':a.arm,'completed':0,'total':64,'gpu_preflight':check})
    with torch.inference_mode(), (root/'synthetic.jsonl').open('x') as out:
        for client in (100,101):
            records=None
            if a.arm!='PUBLIC':
                records=[json.loads(s) for s in (PILOT/'cohort/restricted'/('client_%d.jsonl'%client)).read_text().splitlines()]
                assert len(records)==120
            for i in range(32):
                # Public slot order, no label-dependent grouping or lookup.
                label=str(i%4+1)
                prompts=[prompt(tok,label)]
                if records is not None: prompts += [prompt(tok,label,r) for r in records[3*i:3*i+3]]
                inputs=[p.to('cuda:0') for p in prompts]; caches=[None]*len(inputs); ids=[]
                for step in range(cfg['max_tokens']):
                    logits=[]
                    for j in range(len(inputs)):
                        output=model(input_ids=inputs[j],past_key_values=caches[j],use_cache=True)
                        caches[j]=output.past_key_values
                        logits.append(output.logits[0,-1].float().cpu().numpy().astype('float64'))
                        del output
                    pub=logits[0]
                    mask=support(pub,100,cfg['clip'],cfg['references'])
                    merged=scores(pub,logits[1:],'DP_E8',cfg['clip'])
                    eligible,p=probabilities(merged,mask,cfg['temperature'])
                    u=(secrets.randbits(53)+0.5)/(2**53) if a.arm in BUDGETS else rng.random()
                    token=sample(eligible,p,u); ids.append(token)
                    if token in eos: break
                    inputs=[torch.tensor([[token]],device='cuda:0') for _ in inputs]
                text=' '.join(tok.decode(ids,skip_special_tokens=True).split())
                if not text:
                    text='A news report discusses recent developments in '+NAMES[label]+'.'; fallback+=1
                row={'text':text,'label':label}
                out.write(json.dumps(row,ensure_ascii=True)+'\n');out.flush();os.fsync(out.fileno())
                rows.append(row); total_tokens+=len(ids)
                del caches,inputs,prompts
                elapsed=time.monotonic()-t0
                write(root/'status.json',{'status':'generating','arm':a.arm,'completed':len(rows),'total':64,
                      'generated_tokens':total_tokens,'elapsed_seconds':elapsed,
                      'peak_allocated_mib':torch.cuda.max_memory_allocated()/2**20})
                print(json.dumps({'arm':a.arm,'completed':len(rows),'elapsed_seconds':round(elapsed,2)}),flush=True)
    result={'status':'complete','arm':a.arm,'records':len(rows),'tokens':total_tokens,'empty_fallback':fallback,
            'duplicates_retained':len(rows)-len({r['text'] for r in rows}),
            'wall_seconds':time.monotonic()-t0,'peak_allocated_mib':torch.cuda.max_memory_allocated()/2**20,
            'sha256':digest(root/'synthetic.jsonl'),'privacy':cfg if a.arm in BUDGETS else None,
            'secret_state_saved':False,'public_sampling_seed':5703 if a.arm not in BUDGETS else None,
            'scope':'DP candidate text only; controlled simulation; no entire FL transcript guarantee'}
    write(root/'manifest.json',result); write(root/'status.json',result);print(json.dumps(result),flush=True)
if __name__=='__main__':main()
