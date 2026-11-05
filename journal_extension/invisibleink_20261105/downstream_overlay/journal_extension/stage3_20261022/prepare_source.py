import subprocess,json,os,platform,socket
from common import *
os.umask(0o077)
ROOT.mkdir(exist_ok=True)
assert subprocess.check_output(['git','rev-parse','HEAD'],cwd=REPO,text=True).strip()==BASE
paths={'data':DATA,'partition':PART,'fixed':S1/'shared/fixed_partition.h5','pool':S1/'reused/public_pool/candidates.jsonl'}
checks={k:{'path':str(p),'sha256':digest(p)} for k,p in paths.items()}
assert all(checks[k]['sha256']==EXPECTED[k] for k in checks)
# Freeze proposed group prospectively. No outcome-dependent replacement.
spec={'created_at_utc':now(),'status':'source_prepared_pending_operator_and_smoke_gates','base_commit':BASE,'max_complete_endpoints':12,'official_test_used':False,'new_dp_releases':False,'new_cohort_proposed':[100,101],'cohort_rule':'Fixed IDs 100 and 101 selected before any new-group accuracy; validate disjointness and >=8 real examples per class, stop if infeasible; no performance-based selection','fl_seeds':[57,58,59],'rounds':50,'queries':3000,'guidance_samples':64,'public_candidates':128,'checkpoints':[0,10,25,50],'primary_metric':'final_dev_accuracy','secondary':['late10','macro_f1','accuracy_auc'],'hardware':'actual NVIDIA RTX A6000 48GB; prior discussion of A100 not the delivered device','new_cohort_is_replication':'independent clients within same AG News uniform partition, not a new domain','no_guidance_interpretation':'trace-matched legacy direction ablation, not tuned unbiased ZOO baseline','matrices':[{'group':'original_1_21','arms':['NO_GUIDANCE_TRACE_LEGACY'],'seeds':[57,58,59]},{'group':'new_100_101','arms':['PUB_NLI','PRIVATE_ZERO','REAL_MATCHED'],'seeds':[57,58,59]}]}
if not (ROOT/'PROTOCOL.json').exists():write(ROOT/'PROTOCOL.json',spec)
else:
 prior=json.loads((ROOT/'PROTOCOL.json').read_text());assert prior['base_commit']==BASE and prior['new_cohort_proposed']==[100,101]
 write(ROOT/'PREPARATION_RECOVERY.json',{'reason':'Linux case-sensitive FedSgdServer filename corrected after initial setup stopped; no training or private release occurred','time':now()})
def patch(rel,old,new):
 p=REPO/rel;s=p.read_text()
 if s.count(old)==0 and s.count(new)==1:return
 assert s.count(old)==1,rel
 (ROOT/'source_before').mkdir(exist_ok=True)
 q=ROOT/'source_before'/p.name;q.write_text(s)
 p.write_text(s.replace(old,new))
patch('experiments/distributed/transformer_exps/run_tc_exps/fedavg_main_tc.py',
'''    dm = TextClassificationDataManager(args, model_args, preprocessor, process_id, args.client_num_per_round)
    train_data_num, train_data_global, test_data_global, train_data_local_num_dict, \\
    train_data_local_dict, test_data_local_dict, num_clients = dm.load_cloud_server_client_data(process_id=process_id)
''',
'''    if process_id == 0 and float(args.alpha) == 1.0:
        if args.cloud_data_file_path or args.cloud_partition_file_path:
            raise ValueError("no-guidance run must not provide cloud data")
        logging.info("[ZGR] pure ZOO: cloud rank has no training data")
        train_data_num, train_data_global, test_data_global = 0, None, None
        train_data_local_num_dict = train_data_local_dict = test_data_local_dict = None
        num_clients = args.client_num_in_total
    else:
        dm = TextClassificationDataManager(args, model_args, preprocessor, process_id, args.client_num_per_round)
        train_data_num, train_data_global, test_data_global, train_data_local_num_dict, \\
        train_data_local_dict, test_data_local_dict, num_clients = dm.load_cloud_server_client_data(process_id=process_id)
''')
patch('FedML/fedml_api/distributed/fedsgd/FedSgdServer.py',
'    def test_on_server_for_all_clients(self, round_idx, force=False):\n',
'    def test_on_server_for_all_clients(self, round_idx, force=False):\n        from journal_extension.stage3_20261022.checkpoint_capture import capture\n        capture(self.trainer.server_trainer.model, self.args, round_idx + 1)\n')
modified=subprocess.check_output(['git','diff','--name-only'],cwd=REPO,text=True).splitlines()
write(ROOT/'SOURCE_AUDIT.json',{'base_commit':BASE,'modified_sha256':{r:digest(REPO/r) for r in modified},'unmodified_update_operator_sha256':digest(REPO/'forward_training/tc_transformer_trainer_distribute.py'),'input_checks':checks,'frozen_at':now()})
(ROOT/'source.patch').write_text(subprocess.check_output(['git','diff'],cwd=REPO,text=True))
import torch,transformers,numpy
write(ROOT/'PREFLIGHT.json',{'time':now(),'host':socket.gethostname(),'torch':torch.__version__,'cuda':torch.version.cuda,'transformers':transformers.__version__,'numpy':numpy.__version__,'tf32_matmul':torch.backends.cuda.matmul.allow_tf32,'tf32_cudnn':torch.backends.cudnn.allow_tf32,'gpu':subprocess.check_output(['nvidia-smi'],text=True),'cpu_count':len(os.sched_getaffinity(0)),'meminfo':Path('/proc/meminfo').read_text(),'processes':subprocess.check_output(['nvidia-smi','--query-compute-apps=pid,process_name,used_gpu_memory','--format=csv,noheader'],text=True),'github_fetch':'unavailable; imported verified local Git bundle','historical_worktrees':'preserved'})
print('Input hashes verified; source hooks installed; no training started.')
