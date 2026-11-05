"""One authorized P1 reference endpoint, preserving the frozen update protocol."""
import sys,shlex,socket,shutil,re
from common import *
from commands import build_command

def main():
    os.umask(0o077)
    arm=sys.argv[1];seed=int(sys.argv[2]);assert arm in ARMS and seed in SEEDS
    assert json.loads((ROOT/'REVIEW_GATE.json').read_text())['status']=='pass'
    for rel,h in json.loads((ROOT/'references/DOWNSTREAM_SOURCE_GATE.json').read_text())['sha256'].items():
        assert digest(REPO/rel)==h,rel
    fixed=PILOT/'cohort/fixed_partition.h5';assert digest(fixed)==EXPECTED_PARTITION
    cloud=ROOT/'guidance/REAL_MATCHED';cm=json.loads((cloud/'pack_manifest.json').read_text())
    if arm=='REAL_MATCHED':
        assert digest(cloud/'cloud_data.h5')==cm['data_sha256']
        assert digest(cloud/'cloud_partition.h5')==cm['partition_sha256']
    mapping=json.loads((ROOT/'GPU0_MAPPING_GATE.json').read_text());assert mapping['status']=='pass' and mapping['uuid']==GPU
    gpu=guard();dest=ROOT/'runs'/arm/('seed_%d'%seed);dest.mkdir(parents=True,exist_ok=False)
    scratch=ROOT/'scratch'/arm/('seed_%d'%seed);scratch.mkdir(parents=True,exist_ok=False)
    host=socket.gethostname();(dest/'mpi_host_file').write_text(host+' slots=3\n')
    (dest/'gpu_mapping.yaml').write_text('mapping_private_vote:\n  %s: [3]\n'%host)
    ref=shlex.split((ROOT/'references/guided_seed57.command.txt').read_text());cmd=build_command(ref,arm,seed)
    e=env();e.update(CUDA_DEVICE_ORDER='PCI_BUS_ID',CUDA_VISIBLE_DEVICES='0',PYTHONHASHSEED=str(seed),
        PYTHONPATH=':'.join(map(str,[REPO/'experiments/distributed/transformer_exps/run_tc_exps',REPO,REPO/'FedML'])),
        HEIMDALLM_STAGE3_RUN_ID='invink_P1_'+arm+'_'+str(seed),HEIMDALLM_FAULTHANDLER='1')
    state={'status':'running','arm':arm,'seed':seed,'rounds':50,'start_time':time.time(),
        'gpu_preflight':gpu,'fixed_partition_sha256':digest(fixed),
        'cloud_data_sha256':cm['data_sha256'] if arm=='REAL_MATCHED' else None,
        'alpha':.5 if arm=='REAL_MATCHED' else 1.,'private_client_cohort':[100,101],
        'official_test_used':False,'new_generation':False,'new_DP_generation_release':False,
        'scope':'Reference condition; neither REAL nor raw-data FL has a synthetic-release DP guarantee'}
    write(dest/'run_spec.json',state);(dest/'train.command.txt').write_text(shlex.join(cmd)+'\n')
    with (dest/'train.log').open('x') as log:
        proc=subprocess.Popen(cmd,cwd=REPO/'experiments/distributed/transformer_exps/run_tc_exps',env=e,
            stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
        write(dest/'process.json',{'pid':proc.pid,'pgid':proc.pid,'owner_uid':os.getuid()})
        code=proc.wait()
    state.update(exit_code=code,wall_seconds=time.time()-state['start_time'],status='train_failed' if code else 'validating');write(dest/'run_spec.json',state)
    if code:raise RuntimeError('Training failed; no automatic retry')
    cp=scratch/'model_output/diagnostic_checkpoints';shutil.copytree(cp,dest/'diagnostic_checkpoints')
    assert len(list((dest/'diagnostic_checkpoints').glob('*.pt')))==4
    assert json.loads((dest/'diagnostic_checkpoints/round_00.json').read_text())['trainable_sha256']==EXPECTED_INIT[seed]
    text=(dest/'train.log').read_text(errors='replace');updates=text.count('aggregate time cost')
    directions=sum(int(n) for n in re.findall(r'\[ZGR\] finite differences: nonzero=\d+/(\d+)',text))
    assert (updates,directions)==(750,1500),(updates,directions)
    write(dest/'budget_audit.json',{'status':'pass','updates':updates,'directions':directions,'queries':2*directions})
    val=[str(FLPY),str(REPO/'journal_extension/summarize_fed_pilot.py'),'--log',str(dest/'train.log'),
        '--metrics-out',str(dest/'metrics.json'),'--manifest-out',str(dest/'manifest.json'),
        '--command-file',str(dest/'train.command.txt'),'--condition','client_syn' if arm=='REAL_MATCHED' else 'no_cloud',
        '--cloud-source','real_matched' if arm=='REAL_MATCHED' else 'none','--evaluation-mode','dev',
        '--stage','invisibleink_P1_current_cohort','--alpha','0.5' if arm=='REAL_MATCHED' else '1.0',
        '--seed',str(seed),'--rounds','50','--eval-every','1','--pre-update-eval',
        '--logical-clients','2','--clients-per-round','2','--mpi-workers','1','--exit-code','0',
        '--artifact','fixed_partition='+str(fixed),'--require-complete']
    with (dest/'validation.log').open('w') as log:
        v=subprocess.run(val,cwd=REPO,env=e,stdout=log,stderr=subprocess.STDOUT)
    state['status']='complete' if v.returncode==0 else 'validation_failed';write(dest/'run_spec.json',state)
    if v.returncode:raise RuntimeError('Endpoint validation failed')
    print(json.dumps(state),flush=True)

if __name__=='__main__':main()
