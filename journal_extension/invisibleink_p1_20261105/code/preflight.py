"""Input/source/environment/GPU review; no generator or FL execution."""
import shlex,subprocess
from common import *
from commands import audit_reference_commands

def main():
    os.umask(0o077)
    scope=json.loads((ROOT/'SCOPE.json').read_text())
    assert scope['active_new_arms']==list(ARMS) and scope['maximum_new_endpoints']==4
    assert scope['new_generation_artifacts']==0 and not scope['P2_authorized_in_this_queue']
    assert json.loads((ROOT/'RESOURCE_AUTHORIZATION.json').read_text())['authorized']
    check=guard()
    for name in ['PROCESS_GUARD_TESTS.json','RUNTIME_POLICY_TESTS.json']:
        assert json.loads((ROOT/'code'/name).read_text())['status']=='pass'
    gate=json.loads((ROOT/'references/DOWNSTREAM_SOURCE_GATE.json').read_text())
    assert subprocess.check_output(['git','rev-parse','HEAD'],cwd=REPO,text=True).strip()==gate['base_commit']
    for rel,h in gate['sha256'].items():assert digest(REPO/rel)==h,rel
    assert digest(DATA)==EXPECTED_DATA
    fixed=PILOT/'cohort/fixed_partition.h5';assert digest(fixed)==EXPECTED_PARTITION
    cloud=ROOT/'guidance/REAL_MATCHED';cm=json.loads((cloud/'pack_manifest.json').read_text())
    assert cm['records']==64 and cm['test_records']==0
    assert digest(cloud/'cloud_data.h5')==cm['data_sha256']=='2298c69a4791874cf5913c6e88c0c7bbfbb5da4818ecd946803a9f7541805888'
    assert digest(cloud/'cloud_partition.h5')==cm['partition_sha256']=='b740b69996959c2ad6f4404dbb08b1bf6b5b661e45d4792c520b0f6bcd835dd8'
    assert cm['label_vocab']=={'1':0,'2':2,'3':3,'4':1}
    # Both arms use exactly the fixed local/dev record indices; no official test loading.
    import h5py
    with h5py.File(fixed,'r') as f:
        method=f['pilot_uniform_2'];groups=method['partition_data'];counts={}
        # The frozen file retains three development shards; only 0/1 train.
        assert set(groups)=={'0','1','2'}
        assert list(map(int,json.loads(method.attrs['source_client_ids_json'])))==[100,101]
        train=[];dev=[]
        for cid,g in groups.items():
            tr=list(map(int,g['train'][()]));te=list(map(int,g['test'][()]))
            assert len(tr)==(120 if cid in ['0','1'] else 0)
            assert len(te)=={'0':171,'1':171,'2':170}[cid]
            counts[cid]={'train':len(tr),'dev':len(te)};train+=tr;dev+=te
        assert len(set(train))==240 and len(set(dev))==512 and not set(train)&set(dev)
    command_audit=audit_reference_commands(ROOT/'references')
    write(ROOT/'COMMAND_AUDIT.json',command_audit)
    e=env();e.update(CUDA_DEVICE_ORDER='PCI_BUS_ID',CUDA_VISIBLE_DEVICES='0')
    v=subprocess.check_output([str(GENPY),'-c',"import torch,json;print(json.dumps({'count':torch.cuda.device_count(),'uuid':str(torch.cuda.get_device_properties(0).uuid)}))"],env=e,text=True,timeout=60)
    v=json.loads(v.strip().splitlines()[-1]);assert v['count']==1 and v['uuid'].removeprefix('GPU-')==GPU.removeprefix('GPU-')
    assert subprocess.check_output([str(FLPY),'-c','import torch;print(torch.cuda.device_count())'],env=e,text=True,timeout=60).strip()=='1'
    e['CUDA_VISIBLE_DEVICES']=''
    subprocess.run(['/usr/bin/python3','-c','import matplotlib;matplotlib.use("Agg")'],env=e,check=True,timeout=60)
    write(ROOT/'GPU0_MAPPING_GATE.json',{'status':'pass','uuid':GPU,'numeric_index':0})
    write(ROOT/'REVIEW_GATE.json',{'status':'pass','time':time.time(),'gpu':check,
        'source_hashes_pass':True,'real_guidance_reused':True,'partition_counts':counts,
        'new_generation':0,'new_training_endpoints':4,'no_official_test_records_loaded':True})
    print(json.dumps({'status':'pass','new_endpoints':4,'new_generation':0}),flush=True)

if __name__=='__main__':main()
