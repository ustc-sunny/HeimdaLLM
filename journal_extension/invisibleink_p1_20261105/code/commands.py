"""Only guide/no-guide and output/seed changes to the frozen downstream command."""
import shlex
from common import *

CLOUD_KEYS=['--cloud_dataset','--cloud_data_file_path','--cloud_partition_file_path',
            '--cloud_partition_method','--cloud_client_ids','--cloud_max_seq_length']
PATH_KEYS=['orte_tmpdir_base','--hostfile','--gpu_mapping_file','--output_dir',
           '--data_file_path','--partition_file_path','--model_name',
           '--cloud_data_file_path','--cloud_partition_file_path']

def opt(cmd,key,value):
    assert cmd.count(key)==1,key
    cmd[cmd.index(key)+1]=str(value)

def normalized(cmd):
    cmd=list(cmd)
    cmd[cmd.index('-m')-1]='<PYTHON>'
    for key in PATH_KEYS+['--manual_seed','--run_id']:
        if key in cmd:opt(cmd,key,'<'+key+'>')
    return cmd

def build_command(reference,arm,seed):
    assert arm in ARMS and seed in SEEDS
    original=list(reference);cmd=list(reference)
    dest=ROOT/'runs'/arm/('seed_%d'%seed)
    scratch=ROOT/'scratch'/arm/('seed_%d'%seed)
    for k,v in {'orte_tmpdir_base':scratch/'mpi','--hostfile':dest/'mpi_host_file',
        '--gpu_mapping_file':dest/'gpu_mapping.yaml','--output_dir':scratch/'model_output',
        '--manual_seed':seed,'--run_id':seed}.items():opt(cmd,k,v)
    if arm=='REAL_MATCHED':
        cloud=ROOT/'guidance/REAL_MATCHED'
        opt(cmd,'--cloud_data_file_path',cloud/'cloud_data.h5')
        opt(cmd,'--cloud_partition_file_path',cloud/'cloud_partition.h5')
        assert normalized(cmd)==normalized(original)
    else:
        opt(cmd,'--alpha','1.0')
        for key in CLOUD_KEYS:
            assert cmd.count(key)==1,key
            i=cmd.index(key);del cmd[i:i+2]
        expected=list(original);opt(expected,'--alpha','1.0')
        for key in CLOUD_KEYS:
            i=expected.index(key);del expected[i:i+2]
        assert normalized(cmd)==normalized(expected)
    # These inputs are deliberately NOT replaced with the old no-guidance cohort.
    for k,v in {'--data_file_path':DATA,'--partition_file_path':PILOT/'cohort/fixed_partition.h5',
        '--model_name':MODEL,'--comm_round':'50','--manual_seed':str(seed),
        '--client_num_in_total':'2','--client_num_per_round':'2','--worker_num':'1',
        '--lr':'0.01','--learning_rate':'0.01','--train_batch_size':'8','--epochs':'1',
        '--fd_step':'0.01','--zo_estimator':'legacy','--beta':'1','--pool_size':'1',
        '--max_var_retries':'0','--v_num':'1'}.items():
        assert cmd[cmd.index(k)+1]==str(v),(k,v)
    assert cmd[cmd.index('-m')-1]==str(FLPY)
    assert '--evaluate_before_training' in cmd
    assert '--perturbation_sampling' in cmd and '--var_control' in cmd
    return cmd

def audit_reference_commands(reference_dir):
    reference_dir=Path(reference_dir)
    ref=shlex.split((reference_dir/'guided_seed57.command.txt').read_text())
    checks=[]
    for arm in ARMS:
        for seed in SEEDS:
            cmd=build_command(ref,arm,seed)
            if arm=='NO_GUIDANCE_TRACE_LEGACY':
                old=shlex.split((reference_dir/('historical_no_seed%d.command.txt'%seed)).read_text())
                assert normalized(cmd)==normalized(old),'Unexpected nonpath change to legacy no-guidance'
            checks.append({'arm':arm,'seed':seed,'command_length':len(cmd),
                'alpha':cmd[cmd.index('--alpha')+1],'client_cohort':'100/101',
                'expected_partition_sha256':EXPECTED_PARTITION,
                'shared_training_parameters_preserved':True})
    return {'status':'pass','endpoints':checks,'new_generation':False,'model_execution':False}
