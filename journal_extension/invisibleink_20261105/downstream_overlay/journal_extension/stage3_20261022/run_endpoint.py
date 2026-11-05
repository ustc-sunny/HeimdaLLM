"""Guarded Stage-3 endpoint. Exclusive host, allow only owned concurrent jobs."""
import argparse,shlex,socket,time,signal,re
from common import *
ARMS=['NO_GUIDANCE_TRACE_LEGACY','PUB_NLI','PRIVATE_ZERO','REAL_MATCHED']
def opt(cmd,k,v):
 assert cmd.count(k)==1,k;cmd[cmd.index(k)+1]=str(v)
def remove(cmd,k):
 assert cmd.count(k)==1,k;i=cmd.index(k);del cmd[i:i+2]
def gpu_check():
 s=subprocess.check_output(['nvidia-smi','--query-gpu=index,name,memory.free,memory.used,utilization.gpu','--format=csv,noheader,nounits'],text=True)
 row=next(r.split(',') for r in s.splitlines() if r.split(',')[0].strip()=='0');assert 'A6000' in row[1]
 pids=subprocess.check_output(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader,nounits'],text=True).splitlines()
 owned=[]
 for pid in pids:
  pid=pid.strip()
  if not pid.isdigit():continue
  try:environ=Path('/proc')/pid/'environ';b=environ.read_bytes()
  except FileNotFoundError:continue
  if b'HEIMDALLM_STAGE3_RUN_ID=' not in b:raise RuntimeError('foreign GPU process '+pid)
  owned.append(int(pid))
 if int(row[2])<14000:raise RuntimeError('insufficient guarded free memory')
 if not owned and int(row[4])>20:raise RuntimeError('unexplained GPU utilization')
 return {'gpu_raw':s.strip(),'owned_gpu_pids':owned,'minimum_free_mib':14000,'time':now()}
def main():
 os.umask(0o077)
 ap=argparse.ArgumentParser();ap.add_argument('--arm',choices=ARMS+['BRIDGE_PUB_NLI'],required=True);ap.add_argument('--seed',type=int,choices=[57,58,59],required=True);ap.add_argument('--smoke',action='store_true');ap.add_argument('--tag',default='formal');a=ap.parse_args()
 assert all(c.isalnum() or c in '_-' for c in a.tag)
 if a.arm=='BRIDGE_PUB_NLI' and not a.smoke:raise ValueError('bridge is smoke only')
 audit=json.loads((ROOT/'SOURCE_AUDIT.json').read_text())
 assert subprocess.check_output(['git','rev-parse','HEAD'],cwd=REPO,text=True).strip()==BASE
 changed=set(subprocess.check_output(['git','diff','--name-only'],cwd=REPO,text=True).splitlines());assert changed==set(audit['modified_sha256'])
 for rel,h in audit['modified_sha256'].items():assert digest(REPO/rel)==h
 assert digest(REPO/'forward_training/tc_transformer_trainer_distribute.py')==audit['unmodified_update_operator_sha256']
 assert json.loads((ROOT/'OPERATOR_AUDIT.json').read_text())['status']=='pass'
 if not a.smoke:assert json.loads((ROOT/'TRAINING_GATE.json').read_text())['status']=='pass'
 newgroup=a.arm in ['PUB_NLI','PRIVATE_ZERO','REAL_MATCHED']
 fixed=ROOT/'cohort/fixed_partition.h5' if newgroup else S1/'shared/fixed_partition.h5'
 if newgroup:
  c=json.loads((ROOT/'cohort/AUDIT.json').read_text());assert c['status']=='pass' and digest(fixed)==c['fixed_partition_sha256']
 else:assert digest(fixed)==EXPECTED['fixed']
 if a.arm=='NO_GUIDANCE_TRACE_LEGACY':cloud=None
 elif a.arm=='BRIDGE_PUB_NLI':cloud=S1/'guidance/public_nli_selection'
 else:
  cloud=ROOT/'cohort'/a.arm
  for n,h in c['guidance'][a.arm].items():assert digest(cloud/n)==h
 root=(ROOT/'preflight'/a.tag) if a.smoke else ROOT/'runs'
 dest=root/a.arm/('seed_%d'%a.seed)
 if dest.exists():raise FileExistsError(str(dest))
 gpu=gpu_check();dest.mkdir(parents=True,mode=0o700)
 runid='%s_%s_%d'%(a.tag,a.arm,a.seed)
 scratch=Path('/root/heimdallm-stage3-20261022/scratch')/runid;scratch.mkdir(parents=True)
 if not newgroup and (ROOT/'DATA_ORDER_AMENDMENT.json').exists():
  from cache_replay import historical_cache
  write(dest/'cache_replay_audit.json',historical_cache(a.seed,scratch/'model_output/cache_dir',cloud is not None))
 host=socket.gethostname();hostfile=dest/'mpi_host_file';mapping=dest/'gpu_mapping.yaml'
 hostfile.write_text(host+' slots=3\n');mapping.write_text('mapping_private_vote:\n  %s: [3]\n'%host)
 template=shlex.split((S1/'runs/PRIVATE_DP_E4'/('seed_%d'%a.seed)/'train.command.txt').read_text())
 template[template.index('-m')-1]=str(PY)
 smoke_rounds=1 if a.arm=='BRIDGE_PUB_NLI' else 2
 for k,v in {'--hostfile':hostfile,'--gpu_mapping_file':mapping,'--data_file_path':DATA,'--partition_file_path':fixed,'--model_name':MODEL,'--manual_seed':a.seed,'--run_id':a.seed,'--comm_round':smoke_rounds if a.smoke else 50,'--output_dir':scratch/'model_output'}.items():opt(template,k,v)
 mpi=template.index('mpirun');template[mpi+1:mpi+1]=['--mca','orte_tmpdir_base',str(scratch/'mpi')]
 if cloud is None:
  opt(template,'--alpha',1.0);opt(template,'--zo_estimator','legacy')
  for f in ['--cloud_dataset','--cloud_data_file_path','--cloud_partition_file_path','--cloud_partition_method','--cloud_client_ids','--cloud_max_seq_length']:remove(template,f)
 else:
  opt(template,'--cloud_data_file_path',cloud/'cloud_data.h5');opt(template,'--cloud_partition_file_path',cloud/'cloud_partition.h5')
 if a.smoke:template.append('--audit_updates')
 for k,v in [('--lr','0.01'),('--fd_step','0.01'),('--beta','1'),('--pool_size','1'),('--max_var_retries','0'),('--train_batch_size','8')]:assert template[template.index(k)+1]==v,(k,template)
 spec={'status':'started','arm':a.arm,'seed':a.seed,'group':'new_100_101' if newgroup else 'original_1_21','rounds':2 if a.smoke else 50,'new_dp_release':False,'official_test_used':False,'source_commit':BASE,'modified_source_sha256':audit['modified_sha256'],'fixed_partition_sha256':digest(fixed),'cloud_data_sha256':digest(cloud/'cloud_data.h5') if cloud else None,'gpu_preflight':gpu,'start':now(),'run_id':runid,'smoke':a.smoke,'scratch':str(scratch),'checkpoint_scope':'training_state_restricted_not_DP'}
 if a.smoke:spec['rounds']=smoke_rounds
 spec['data_order_policy']='DATA_ORDER_AMENDMENT.json' if (ROOT/'DATA_ORDER_AMENDMENT.json').exists() else 'legacy'
 write(dest/'run_spec.json',spec);(dest/'train.command.txt').write_text(shlex.join(template)+'\n')
 e=env(a.seed);e.update(HEIMDALLM_STAGE3_RUN_ID=runid,HEIMDALLM_FAULTHANDLER='1')
 entry=REPO/'experiments/distributed/transformer_exps/run_tc_exps';t=time.monotonic()
 with (scratch/'train.log').open('w') as log:
  proc=subprocess.Popen(template,cwd=entry,env=e,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
  write(dest/'process.json',{'pid':proc.pid,'pgid':proc.pid,'run_id':runid,'owner':'root','start':now()})
  code=proc.wait()
 wall=time.monotonic()-t
 import shutil
 shutil.copy2(scratch/'train.log',dest/'train.log')
 cp=scratch/'model_output/diagnostic_checkpoints'
 if cp.exists():shutil.copytree(cp,dest/'diagnostic_checkpoints')
 (dest/'train.exit_code.txt').write_text(str(code)+'\n')
 spec.update(wall_seconds=wall,exit_code=code,end=now(),status='train_failed' if code else 'trained_pending_validation');write(dest/'run_spec.json',spec)
 if code:raise RuntimeError('training failed, preserved logs '+str(dest))
 content=(dest/'train.log').read_text(errors='replace')
 updates=content.count('aggregate time cost')
 directions=sum(int(n) for n in re.findall(r'\[ZGR\] finite differences: nonzero=\d+/(\d+)',content))
 budget={'updates':updates,'directions':directions,'objective_queries':2*directions,'expected_updates':15*spec['rounds'],'expected_directions':30*spec['rounds']}
 budget['status']='pass' if updates==budget['expected_updates'] and directions==budget['expected_directions'] else 'fail'
 write(dest/'budget_audit.json',budget)
 if budget['status']!='pass':
  spec['status']='budget_validation_failed';write(dest/'run_spec.json',spec);raise RuntimeError('update/query budget mismatch '+str(dest))
 val=[str(PY),str(REPO/'journal_extension/summarize_fed_pilot.py'),'--log',str(dest/'train.log'),'--metrics-out',str(dest/'metrics.json'),'--manifest-out',str(dest/'manifest.json'),'--command-file',str(dest/'train.command.txt'),'--condition','no_cloud' if cloud is None else 'client_syn','--cloud-source','none' if cloud is None else a.arm.lower(),'--evaluation-mode','dev','--stage','stage3_trace_cohort','--alpha','1.0' if cloud is None else '0.5','--seed',str(a.seed),'--rounds',str(spec['rounds']),'--eval-every','1','--pre-update-eval','--logical-clients','2','--clients-per-round','2','--mpi-workers','1','--exit-code','0','--artifact','fixed_partition='+str(fixed),'--require-complete']
 with (dest/'validate.log').open('w') as log:v=subprocess.run(val,cwd=REPO,env=e,stdout=log,stderr=subprocess.STDOUT)
 spec['status']='smoke_complete' if a.smoke and v.returncode==0 else 'complete' if v.returncode==0 else 'validation_failed';write(dest/'run_spec.json',spec)
 if v.returncode:raise RuntimeError('endpoint validator failed '+str(dest))
 print(json.dumps({'status':spec['status'],'arm':a.arm,'seed':a.seed,'wall_seconds':wall,'path':str(dest)}),flush=True)
if __name__=='__main__':main()
