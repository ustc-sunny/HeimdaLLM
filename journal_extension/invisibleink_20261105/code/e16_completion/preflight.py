"""Reuse audited data and twelve endpoints; no generator or private inference."""
import shlex,subprocess,shutil
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
    assert json.loads((ROOT/'SCOPE.json').read_text())['new_generation_artifacts']==0
    assert not (WIDE/'generation/DP_E20').exists()
    assert json.loads((WIDE/'QUEUE_STATUS.json').read_text())['status']=='blocked'
    assert json.loads((ROOT/'RESOURCE_AUTHORIZATION.json').read_text())['authorized']
    for n in ['PROCESS_GUARD_TESTS.json','RUNTIME_POLICY_TESTS.json']:
        assert json.loads((ROOT/'code'/n).read_text())['status']=='pass'
    check=guard()
    for rel,h in json.loads((PILOT/'DOWNSTREAM_SOURCE_GATE.json').read_text())['sha256'].items():
        assert digest(REPO/rel)==h,rel
    assert digest(ROOT/'code/run_fl.py')==digest(WIDE/'code/run_fl.py')
    assert digest(DATA)=='1474dd264d535b50ad2a7b1cf714bf11e182c2fe5b07f5e982feb31a9b114616'
    assert digest(PILOT/'cohort/fixed_partition.h5')=='43d324efaaaf9f2e1e169a328e7b58522d004cbc37441d90c9bc9eb5163933d5'
    # Copy complete epsilon16 byte for byte, preserving the source and old partial run.
    expected=json.loads((ROOT/'GUIDANCE_REUSE_EXPECTED.json').read_text())
    assert digest(WIDE/'generation/DP_E16/synthetic.jsonl')==expected['source_sha256']
    assert digest(WIDE/'guidance/DP_E16/cloud_data.h5')==expected['cloud_data_sha256']
    assert digest(WIDE/'guidance/DP_E16/cloud_partition.h5')==expected['cloud_partition_sha256']
    assert json.loads((WIDE/'generation/DP_E16/manifest.json').read_text())['records']==64
    for sub in ['generation','guidance']:
        source=WIDE/sub/'DP_E16';dest=ROOT/sub/'DP_E16'
        dest.parent.mkdir(exist_ok=True)
        if not dest.exists():shutil.copytree(source,dest)
        for f in source.rglob('*'):
            if f.is_file():assert digest(f)==digest(dest/f.relative_to(source))
    write(ROOT/'GUIDANCE_REUSE_AUDIT.json',{'status':'pass','origin':str(WIDE),'arm':'DP_E16',
        'records':64,'source_sha256':expected['source_sha256'],'new_generation':False,
        'cloud_data_sha256':expected['cloud_data_sha256'],'cloud_partition_sha256':expected['cloud_partition_sha256']})
    original=normalized((PILOT/'runs/DP_E8/seed_57/train.command.txt').read_text())
    reused=[]
    for seed in SEEDS:
        for arm in ['PUBLIC','NONPRIVATE','DP_E1','DP_E4','DP_E8','DP_E12']:
            d=endpoint_source(arm,seed)/'runs'/arm/('seed_%d'%seed)
            assert endpoint_complete(d),(arm,seed)
            assert normalized((d/'train.command.txt').read_text())==original
            assert json.loads((d/'budget_audit.json').read_text())=={'status':'pass','updates':750,'directions':1500,'queries':3000}
            assert len(list((d/'diagnostic_checkpoints').glob('*.pt')))==4
            m=json.loads((d/'metrics.json').read_text());sp=json.loads((d/'run_spec.json').read_text())
            reused.append({'arm':arm,'seed':seed,'source_path':str(d),'final_dev_accuracy':m['final']['acc'],
                'metrics_sha256':digest(d/'metrics.json'),'completion_status':sp['status'],'exit_code':sp.get('exit_code')})
    e=env();e.update(CUDA_DEVICE_ORDER='PCI_BUS_ID',CUDA_VISIBLE_DEVICES='0')
    modern=subprocess.check_output([str(GENPY),'-c',
        "import torch,json;print(json.dumps({'count':torch.cuda.device_count(),'uuid':str(torch.cuda.get_device_properties(0).uuid)}))"],env=e,text=True,timeout=60)
    v=json.loads(modern.strip().splitlines()[-1]);assert v['count']==1 and v['uuid'].removeprefix('GPU-')==GPU.removeprefix('GPU-')
    old=subprocess.check_output([str(FLPY),'-c','import torch;print(torch.cuda.device_count())'],env=e,text=True,timeout=60)
    assert old.strip()=='1'
    e['CUDA_VISIBLE_DEVICES']=''
    subprocess.run(['/usr/bin/python3','-c','import matplotlib;matplotlib.use("Agg");import matplotlib.pyplot'],env=e,check=True,timeout=60)
    write(ROOT/'GPU0_MAPPING_GATE.json',{'status':'pass','uuid':GPU,'numeric_index':0})
    write(ROOT/'REUSE_AUDIT.json',{'status':'pass','endpoints':reused,'count':12,'fixed_protocol_equal':True})
    write(ROOT/'PRIVACY_GATE.json',{'status':'reused_reviewed_artifact_only','new_generation':False,
        'no_additional_synthetic_release':True,'full_FL_not_DP':True})
    write(ROOT/'REVIEW_GATE.json',{'status':'pass','time':time.time(),'preflight':check,
        'reused_endpoints':12,'new_training_endpoints':2,'new_generation':0,'source_hashes_pass':True})
    import report;report.main()
    print(json.dumps({'status':'pass','reused_endpoints':12,'new_endpoints':2,'new_generation':0}),flush=True)

if __name__=='__main__':main()
