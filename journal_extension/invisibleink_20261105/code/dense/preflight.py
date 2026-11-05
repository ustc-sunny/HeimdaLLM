"""CPU audits and bounded device-mapping checks before private generation."""
import ast, shlex, subprocess
from common import *

def normalized(command):
    cmd=shlex.split(command)
    for key in ['orte_tmpdir_base','--hostfile','--gpu_mapping_file','--output_dir',
                '--manual_seed','--run_id','--cloud_data_file_path','--cloud_partition_file_path']:
        assert cmd.count(key)==1
        cmd[cmd.index(key)+1]='<'+key+'>'
    return cmd

def main():
    os.umask(0o077)
    assert json.loads((ROOT/'RESOURCE_AUTHORIZATION.json').read_text())['authorized']
    tests=json.loads((ROOT/'code/CPU_TESTS.json').read_text());assert tests['status']=='pass'
    assert len(tests['per_budget'])==3 and tests['total_cases']==1500
    check=guard()
    for rel,h in json.loads((PILOT/'DOWNSTREAM_SOURCE_GATE.json').read_text())['sha256'].items():
        assert digest(REPO/rel)==h,rel
    assert digest(PILOT/'cohort/fixed_partition.h5')=='43d324efaaaf9f2e1e169a328e7b58522d004cbc37441d90c9bc9eb5163933d5'
    assert digest(DATA)=='1474dd264d535b50ad2a7b1cf714bf11e182c2fe5b07f5e982feb31a9b114616'
    assert digest(ROOT/'code/mechanism.py')==digest(PILOT/'code/mechanism.py')
    def prompt_tree(p):
        t=ast.parse(p.read_text())
        return ast.dump(next(x for x in t.body if isinstance(x,ast.FunctionDef) and x.name=='prompt'),include_attributes=False)
    assert prompt_tree(ROOT/'code/generate.py')==prompt_tree(PILOT/'code/generate.py')
    original=normalized((PILOT/'runs/DP_E8/seed_57/train.command.txt').read_text())
    reused=[]
    for seed,base in [(57,PILOT),(58,FOLLOWUP)]:
        for arm in ['PUBLIC','NONPRIVATE','DP_E8']:
            d=base/'runs'/arm/('seed_%d'%seed)
            spec=json.loads((d/'run_spec.json').read_text());m=json.loads((d/'metrics.json').read_text())
            assert spec['status']=='complete' and spec['exit_code']==0
            assert m['status']=='complete' and all(m['checks'].values()) and len(m['evaluations'])==51
            assert json.loads((d/'budget_audit.json').read_text())=={'status':'pass','updates':750,'directions':1500,'queries':3000}
            assert normalized((d/'train.command.txt').read_text())==original
            g=PILOT/'generation'/arm;gm=json.loads((g/'manifest.json').read_text())
            assert digest(g/'synthetic.jsonl')==gm['sha256']
            c=PILOT/'guidance'/arm;cm=json.loads((c/'manifest.json').read_text())
            assert digest(c/'cloud_data.h5')==cm['cloud_data_sha256']
            assert digest(c/'cloud_partition.h5')==cm['cloud_partition_sha256']
            reused.append({'arm':arm,'seed':seed,'source_path':str(d),'final_dev_accuracy':m['final']['acc'],
                'metrics_sha256':digest(d/'metrics.json'),'guide_sha256':cm['cloud_data_sha256']})
    # Modern torch resolves numeric visible index to UUID; old torch must see one device.
    e=env();e.update(CUDA_DEVICE_ORDER='PCI_BUS_ID',CUDA_VISIBLE_DEVICES='0')
    modern=subprocess.check_output([str(GENPY),'-c',
        "import torch,json;print(json.dumps({'count':torch.cuda.device_count(),'uuid':str(torch.cuda.get_device_properties(0).uuid)}))"],env=e,text=True,timeout=60)
    v=json.loads(modern.strip().splitlines()[-1])
    assert v['count']==1 and v['uuid'].removeprefix('GPU-')==GPU.removeprefix('GPU-'),v
    old=subprocess.check_output([str(FLPY),'-c','import torch;print(torch.cuda.device_count())'],env=e,text=True,timeout=60)
    assert old.strip()=='1',old
    cpu_env=env();cpu_env['CUDA_VISIBLE_DEVICES']=''
    subprocess.run([str(FLPY),'-c','import h5py,numpy'],env=cpu_env,check=True,timeout=60)
    subprocess.run(['/usr/bin/python3','-c',
        'import matplotlib;matplotlib.use("Agg");import matplotlib.pyplot as p;f,a=p.subplots();a.set_xscale("log",base=2);a.set_xticks([4,8],["4","8"]);p.close(f)'],
        env=cpu_env,check=True,timeout=60)
    write(ROOT/'GPU0_MAPPING_GATE.json',{'status':'pass','uuid':GPU,'numeric_index':0,'modern_device_count':1,'legacy_device_count':1})
    write(ROOT/'REUSE_AUDIT.json',{'status':'pass','reused_endpoints':reused,'downstream_normalized_command_equal':True})
    write(ROOT/'PRIVACY_GATE.json',{'status':'reviewed_research_pilot','mechanism_bytes_unchanged':True,
        'prompt_ast_unchanged':True,'cpu_cases':1500,'approved_new_budgets':[12],
        'scope':'Ideal theorem with numerical research-implementation limits; no full FL DP; composed budget separately recorded.'})
    write(ROOT/'REVIEW_GATE.json',{'status':'pass','time':time.time(),'preflight':check,
        'source_hashes_pass':True,'data_partition_hashes_pass':True,'reused_endpoints':6,'approved_new_endpoints':4})
    print(json.dumps({'status':'pass','reused_endpoints':6,'cpu_cases':1500,'gpu':GPU}),flush=True)

if __name__=='__main__':main()
