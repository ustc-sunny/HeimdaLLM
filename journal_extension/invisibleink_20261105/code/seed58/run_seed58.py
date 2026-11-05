"""One unchanged-artifact FL endpoint; only FL seed, paths and assigned GPU differ."""
import sys,os,shlex,socket,subprocess,time,re,json,shutil
from diag_common import *
REPO=PILOT/'HeimdaLLM'
OLD=Path('/home/zzkevin/heimdallm-journal-20260928')
DATA=OLD/'data/fednlp_data/data_files/agnews_data.h5'
MODEL=OLD/'models/distilbert-base-uncased'
def opt(cmd,k,v):
    assert cmd.count(k)==1,k;cmd[cmd.index(k)+1]=str(v)
def main():
    os.umask(0o077);arm=sys.argv[1];assert arm in ['PUBLIC','NONPRIVATE','DP_E8']
    assert json.loads((ROOT/'SEED58_REVIEW_GATE.json').read_text())['status']=='pass'
    for rel,h in json.loads((PILOT/'DOWNSTREAM_SOURCE_GATE.json').read_text())['sha256'].items():
        assert digest(REPO/rel)==h,rel
    fixed=PILOT/'cohort/fixed_partition.h5'
    assert digest(fixed)=='43d324efaaaf9f2e1e169a328e7b58522d004cbc37441d90c9bc9eb5163933d5'
    cloud=PILOT/'guidance'/arm;cm=json.loads((cloud/'manifest.json').read_text())
    assert digest(cloud/'cloud_data.h5')==cm['cloud_data_sha256']
    assert digest(cloud/'cloud_partition.h5')==cm['cloud_partition_sha256']
    gp=PILOT/'generation'/arm;gm=json.loads((gp/'manifest.json').read_text())
    assert digest(gp/'synthetic.jsonl')==gm['sha256']
    mapping_gate=json.loads((ROOT/'GPU0_MAPPING_GATE.json').read_text())
    assert mapping_gate['status']=='pass' and mapping_gate['uuid']==GPU
    check=guard();dest=ROOT/'runs'/arm/'seed_58';dest.mkdir(parents=True,exist_ok=False)
    scratch=ROOT/'scratch'/arm;scratch.mkdir(parents=True,exist_ok=False)
    host=socket.gethostname();hostfile=dest/'mpi_host_file';mapping=dest/'gpu_mapping.yaml'
    hostfile.write_text(host+' slots=3\n');mapping.write_text('mapping_private_vote:\n  %s: [3]\n'%host)
    reference=shlex.split((PILOT/'runs'/arm/'seed_57/train.command.txt').read_text());cmd=list(reference)
    opt(cmd,'orte_tmpdir_base',scratch/'mpi')
    for k,v in {'--hostfile':hostfile,'--gpu_mapping_file':mapping,'--output_dir':scratch/'model_output',
                '--manual_seed':58,'--run_id':58}.items():opt(cmd,k,v)
    # No model, guide, data, optimizer, scheduler, estimator, FD or budget changes.
    allowed={reference.index(k)+1 for k in ['orte_tmpdir_base','--hostfile','--gpu_mapping_file','--output_dir','--manual_seed','--run_id']}
    assert len(reference)==len(cmd) and all(a==b or i in allowed for i,(a,b) in enumerate(zip(reference,cmd)))
    assert cmd[cmd.index('-m')-1]==str(FLPY)
    for k,v in {'--comm_round':'50','--manual_seed':'58','--alpha':'0.5','--lr':'0.01',
                '--train_batch_size':'8','--fd_step':'0.01','--zo_estimator':'legacy'}.items():assert cmd[cmd.index(k)+1]==v
    e=env();e.update(CUDA_DEVICE_ORDER='PCI_BUS_ID',CUDA_VISIBLE_DEVICES='0',
         PYTHONPATH=':'.join(map(str,[REPO/'experiments/distributed/transformer_exps/run_tc_exps',REPO,REPO/'FedML'])),
         HEIMDALLM_STAGE3_RUN_ID='invink_'+arm+'_58',HEIMDALLM_FAULTHANDLER='1')
    state={'status':'running','arm':arm,'seed':58,'rounds':50,'start_time':time.time(),'gpu_preflight':check,
          'cloud_data_sha256':cm['cloud_data_sha256'],'synthetic_jsonl_sha256':gm['sha256'],
          'official_test_used':False,'new_generation':False,'new_DP_generation_release':False,
          'privacy':'same frozen generation artifact; downstream uses private data, whole FL is not DP'}
    write(dest/'run_spec.json',state);(dest/'train.command.txt').write_text(shlex.join(cmd)+'\n')
    write(dest/'PROTOCOL_DIFF.json',{'reference':str(PILOT/'runs'/arm/'seed_57/train.command.txt'),
           'allowed_changes_only':True,'changes':[{ 'argument':reference[i-1], 'old':reference[i], 'new':cmd[i]} for i in sorted(allowed)]})
    with (dest/'train.log').open('x') as log:
        p=subprocess.Popen(cmd,cwd=REPO/'experiments/distributed/transformer_exps/run_tc_exps',env=e,
                 stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
        write(dest/'process.json',{'pid':p.pid,'pgid':p.pid,'owner_uid':os.getuid()})
        code=p.wait()
    state.update(exit_code=code,wall_seconds=time.time()-state['start_time'],status='train_failed' if code else 'validating');write(dest/'run_spec.json',state)
    if code:raise RuntimeError('training failed; no retry')
    cp=scratch/'model_output/diagnostic_checkpoints'
    if cp.exists():shutil.copytree(cp,dest/'diagnostic_checkpoints')
    text=(dest/'train.log').read_text(errors='replace')
    updates=text.count('aggregate time cost');directions=sum(int(n) for n in re.findall(r'\[ZGR\] finite differences: nonzero=\d+/(\d+)',text))
    if (updates,directions)!=(750,1500):raise RuntimeError('update/direction budget mismatch')
    write(dest/'budget_audit.json',{'status':'pass','updates':updates,'directions':directions,'queries':2*directions})
    val=[str(FLPY),str(REPO/'journal_extension/summarize_fed_pilot.py'),'--log',str(dest/'train.log'),
        '--metrics-out',str(dest/'metrics.json'),'--manifest-out',str(dest/'manifest.json'),
        '--command-file',str(dest/'train.command.txt'),'--condition','client_syn','--cloud-source',arm.lower(),
        '--evaluation-mode','dev','--stage','invisibleink_seed58_reuse','--alpha','0.5','--seed','58',
        '--rounds','50','--eval-every','1','--pre-update-eval','--logical-clients','2','--clients-per-round','2',
        '--mpi-workers','1','--exit-code','0','--artifact','fixed_partition='+str(fixed),'--require-complete']
    with (dest/'validation.log').open('w') as log:
        v=subprocess.run(val,cwd=REPO,env=e,stdout=log,stderr=subprocess.STDOUT)
    state['status']='complete' if v.returncode==0 else 'validation_failed';write(dest/'run_spec.json',state)
    if v.returncode:raise RuntimeError('endpoint validation failed')
    print(json.dumps(state),flush=True)
if __name__=='__main__':main()
