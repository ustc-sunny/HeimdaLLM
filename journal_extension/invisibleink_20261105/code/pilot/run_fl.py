"""Stage-3 exact update source and flags, path/GPU adapter only, three new endpoints."""
import os,shlex,socket,subprocess,sys,time,re,json
from common import *
def opt(cmd,k,v):
    assert cmd.count(k)==1,k;cmd[cmd.index(k)+1]=str(v)
def main():
    os.umask(0o077);arm=sys.argv[1]; assert arm in ['PUBLIC','NONPRIVATE','DP_E8']
    audit=json.loads((ROOT/'DOWNSTREAM_SOURCE_GATE.json').read_text())
    for rel,h in audit['sha256'].items(): assert digest(REPO/rel)==h,rel
    fixed=ROOT/'cohort/fixed_partition.h5'
    assert digest(fixed)=='43d324efaaaf9f2e1e169a328e7b58522d004cbc37441d90c9bc9eb5163933d5'
    cloud=ROOT/'guidance'/arm
    dest=ROOT/'runs'/arm/'seed_57'
    if dest.exists():raise FileExistsError(str(dest))
    check=guard();dest.mkdir(parents=True)
    hostfile=dest/'mpi_host_file';mapping=dest/'gpu_mapping.yaml';host=socket.gethostname()
    hostfile.write_text(host+' slots=3\n');mapping.write_text('mapping_private_vote:\n  %s: [3]\n'%host)
    scratch=ROOT/'scratch'/arm;scratch.mkdir(parents=True,exist_ok=False)
    cmd=shlex.split((ROOT/'recovery/train.command.txt').read_text())
    cmd[cmd.index('-m')-1]=str(FLPY)
    opt(cmd,'orte_tmpdir_base',scratch/'mpi')
    for k,v in {'--hostfile':hostfile,'--gpu_mapping_file':mapping,'--data_file_path':DATA,
          '--partition_file_path':fixed,'--model_name':MODEL,'--cloud_data_file_path':cloud/'cloud_data.h5',
          '--cloud_partition_file_path':cloud/'cloud_partition.h5','--output_dir':scratch/'model_output'}.items():opt(cmd,k,v)
    for k,v in {'--comm_round':'50','--manual_seed':'57','--alpha':'0.5','--lr':'0.01',
           '--train_batch_size':'8','--fd_step':'0.01','--zo_estimator':'legacy'}.items():assert cmd[cmd.index(k)+1]==v
    e=env();e.update(PYTHONPATH=':'.join(str(p) for p in [REPO/'experiments/distributed/transformer_exps/run_tc_exps',REPO,REPO/'FedML']),
                    HEIMDALLM_STAGE3_RUN_ID='invink_'+arm+'_57',HEIMDALLM_FAULTHANDLER='1')
    # torch 1.13.1's NVML visible-device parser only accepts integers. CUDA
    # runtime accepts UUID but pickle deserialization uses the broken count.
    mapping_gate=json.loads((ROOT/'GPU_MAPPING_GATE.json').read_text())
    assert mapping_gate['status']=='pass' and mapping_gate['uuid']==GPU
    e.update(CUDA_DEVICE_ORDER='PCI_BUS_ID',CUDA_VISIBLE_DEVICES='3')
    spec={'status':'running','arm':arm,'seed':57,'rounds':50,'start_time':time.time(),'gpu':check,
          'official_test_used':False,'private_training_not_DP':True,'cloud_data_sha256':digest(cloud/'cloud_data.h5'),
          'fixed_partition_sha256':digest(fixed)}
    write(dest/'run_spec.json',spec);(dest/'train.command.txt').write_text(shlex.join(cmd)+'\n')
    with (dest/'train.log').open('x') as log:
        proc=subprocess.Popen(cmd,cwd=REPO/'experiments/distributed/transformer_exps/run_tc_exps',env=e,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
        write(dest/'process.json',{'pid':proc.pid,'pgid':proc.pid,'owner':os.getuid()})
        code=proc.wait()
    spec.update(exit_code=code,wall_seconds=time.time()-spec['start_time'],status='train_failed' if code else 'validating')
    write(dest/'run_spec.json',spec)
    if code:raise RuntimeError('training failed; no accuracy-driven retry')
    text=(dest/'train.log').read_text(errors='replace')
    updates=text.count('aggregate time cost')
    directions=sum(int(n) for n in re.findall(r'\[ZGR\] finite differences: nonzero=\d+/(\d+)',text))
    assert (updates,directions)==(750,1500),(updates,directions)
    write(dest/'budget_audit.json',{'status':'pass','updates':updates,'directions':directions,'queries':2*directions})
    val=[str(FLPY),str(REPO/'journal_extension/summarize_fed_pilot.py'),'--log',str(dest/'train.log'),
         '--metrics-out',str(dest/'metrics.json'),'--manifest-out',str(dest/'manifest.json'),
         '--command-file',str(dest/'train.command.txt'),'--condition','client_syn','--cloud-source',arm.lower(),
         '--evaluation-mode','dev','--stage','invisibleink_pilot','--alpha','0.5','--seed','57','--rounds','50',
         '--eval-every','1','--pre-update-eval','--logical-clients','2','--clients-per-round','2',
         '--mpi-workers','1','--exit-code','0','--artifact','fixed_partition='+str(fixed),'--require-complete']
    with (dest/'validation.log').open('w') as log:
        v=subprocess.run(val,cwd=REPO,env=e,stdout=log,stderr=subprocess.STDOUT)
    spec['status']='complete' if v.returncode==0 else 'validation_failed';write(dest/'run_spec.json',spec)
    if v.returncode:raise RuntimeError('endpoint validation failed')
    print(json.dumps(spec),flush=True)
if __name__=='__main__':main()
