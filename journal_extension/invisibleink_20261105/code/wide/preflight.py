"""Read-only reuse/source audit and bounded GPU mapping verification."""
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
    auth=json.loads((ROOT/'RESOURCE_AUTHORIZATION.json').read_text())
    assert auth['authorized'] and auth['physical_gpu']==0 and auth['uuid']==GPU
    tests=json.loads((ROOT/'code/CPU_TESTS.json').read_text())
    assert tests['status']=='pass' and tests['total_cases']==1500
    assert {int(t['calibration']['epsilon']) for t in tests['per_budget']}=={1,16,20}
    assert json.loads((ROOT/'code/PROCESS_GUARD_TESTS.json').read_text())['status']=='pass'
    assert json.loads((DENSE/'QUEUE_STATUS.json').read_text())['status']=='complete'
    check=guard()
    for rel,h in json.loads((PILOT/'DOWNSTREAM_SOURCE_GATE.json').read_text())['sha256'].items():
        assert digest(REPO/rel)==h,rel
    assert digest(PILOT/'cohort/fixed_partition.h5')=='43d324efaaaf9f2e1e169a328e7b58522d004cbc37441d90c9bc9eb5163933d5'
    assert digest(DATA)=='1474dd264d535b50ad2a7b1cf714bf11e182c2fe5b07f5e982feb31a9b114616'
    for name in ['mechanism.py','pack.py','run_fl.py']:
        assert digest(ROOT/'code'/name)==digest(DENSE/'code'/name),name
    assert digest(ROOT/'code/mechanism.py')==digest(PILOT/'code/mechanism.py')
    gen=(ROOT/'code/generate.py').read_text().replace(
        "assert a.arm in ('DP_E1','DP_E16','DP_E20'), 'Only the three explicitly requested new budgets are authorized'",
        "assert a.arm=='DP_E12', 'Only one new epsilon12 generation is approved; epsilon4 is reused'")
    assert gen==(DENSE/'code/generate.py').read_text()
    old_partial=Path('/data/zzkevin/heimdallm-invink-tradeoff-20261103/generation/DP_E16/synthetic.jsonl')
    ledger=json.loads((ROOT/'PRIVACY_LEDGER_PLANNED.json').read_text())
    assert digest(old_partial)==ledger['historical_interrupted_epsilon16']['sha256']
    assert len(old_partial.read_text().splitlines())==28
    original=normalized((PILOT/'runs/DP_E8/seed_57/train.command.txt').read_text())
    reused=[]
    for seed in SEEDS:
        for arm in ['PUBLIC','NONPRIVATE','DP_E4','DP_E8','DP_E12']:
            base=endpoint_source(arm,seed);d=base/'runs'/arm/('seed_%d'%seed)
            assert endpoint_complete(d),(arm,seed)
            spec=json.loads((d/'run_spec.json').read_text());m=json.loads((d/'metrics.json').read_text())
            assert json.loads((d/'budget_audit.json').read_text())=={'status':'pass','updates':750,'directions':1500,'queries':3000}
            assert normalized((d/'train.command.txt').read_text())==original
            gb=guidance_source(arm);g=gb/'generation'/arm;gm=json.loads((g/'manifest.json').read_text())
            c=gb/'guidance'/arm;cm=json.loads((c/'manifest.json').read_text())
            assert digest(g/'synthetic.jsonl')==gm['sha256']==cm['source_sha256']
            assert digest(c/'cloud_data.h5')==cm['cloud_data_sha256']
            assert digest(c/'cloud_partition.h5')==cm['cloud_partition_sha256']
            assert len(list((d/'diagnostic_checkpoints').glob('*.pt')))==4
            reused.append({'arm':arm,'seed':seed,'source_path':str(d),'final_dev_accuracy':m['final']['acc'],
                'completion_status':spec['status'],'exit_code_observed':spec.get('exit_code'),
                'metrics_sha256':digest(d/'metrics.json'),'guide_sha256':cm['cloud_data_sha256']})
    e=env();e.update(CUDA_DEVICE_ORDER='PCI_BUS_ID',CUDA_VISIBLE_DEVICES='0')
    modern=subprocess.check_output([str(GENPY),'-c',
        "import torch,json;print(json.dumps({'count':torch.cuda.device_count(),'uuid':str(torch.cuda.get_device_properties(0).uuid)}))"],env=e,text=True,timeout=60)
    v=json.loads(modern.strip().splitlines()[-1])
    assert v['count']==1 and v['uuid'].removeprefix('GPU-')==GPU.removeprefix('GPU-'),v
    old=subprocess.check_output([str(FLPY),'-c','import torch;print(torch.cuda.device_count())'],env=e,text=True,timeout=60)
    assert old.strip()=='1',old
    cpu_env=env();cpu_env['CUDA_VISIBLE_DEVICES']=''
    subprocess.run([str(FLPY),'-c','import h5py,numpy'],env=cpu_env,check=True,timeout=60)
    subprocess.run(['/usr/bin/python3','-c','import matplotlib;matplotlib.use("Agg");import matplotlib.pyplot'],env=cpu_env,check=True,timeout=60)
    write(ROOT/'GPU0_MAPPING_GATE.json',{'status':'pass','uuid':GPU,'numeric_index':0,'modern_device_count':1,'legacy_device_count':1})
    write(ROOT/'REUSE_AUDIT.json',{'status':'pass','reused_endpoints':reused,'downstream_normalized_command_equal':True,
        'historical_teardown_exception_count':sum(r['completion_status']!='complete' for r in reused)})
    write(ROOT/'PRIVACY_GATE.json',{'status':'reviewed_research_pilot','mechanism_bytes_unchanged':True,
        'generator_numerical_lines_unchanged':True,'cpu_cases':1500,'approved_new_budgets':[1,16,20],
        'historical_epsilon16_partial_preserved':True,
        'scope':'Conditional ideal theorem; numerical implementation limits retained; no whole FL DP'})
    write(ROOT/'REVIEW_GATE.json',{'status':'pass','time':time.time(),'preflight':check,
        'source_hashes_pass':True,'data_partition_hashes_pass':True,'reused_endpoints':10,'approved_new_endpoints':6})
    import report
    report.main()
    print(json.dumps({'status':'pass','reused_endpoints':10,'cpu_cases':1500,'gpu':GPU}),flush=True)

if __name__=='__main__':main()
