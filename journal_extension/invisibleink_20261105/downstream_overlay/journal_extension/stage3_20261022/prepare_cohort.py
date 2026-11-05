"""Prospectively fixed new clients; CPU-only, no generator or DP sampling."""
import sys,shutil,random,time
from collections import Counter
import h5py,numpy as np
from common import *
sys.path.insert(0,str(REPO/'journal_extension'));sys.path.insert(0,str(S1))
from non_dp_client_synthetic import decode_scalar,load_staged_clients
from private_vote_dp_e4 import private_histograms,public_pool,select_from_scores,encoded_rows,HISTORICAL_ZERO_SHA

def main():
 os.umask(0o077)
 assert json.loads((ROOT/'PROTOCOL.json').read_text())['new_cohort_proposed']==[100,101]
 out=ROOT/'cohort';out.mkdir(exist_ok=True)
 if (out/'AUDIT.json').exists():raise FileExistsError('cohort already finalized')
 fixed=S1/'shared/fixed_partition.h5';groups=[100,101];clients={};indices={}
 with h5py.File(DATA,'r') as data,h5py.File(PART,'r') as part,h5py.File(fixed,'r') as old:
  attrs=json.loads(decode_scalar(data['attributes'][()]));universe=set(attrs['train_index_list']);source=part['uniform_client_1000']['partition_data']
  # Keep original evaluation shards exactly; read no official-test examples.
  og=old['pilot_uniform_2']['partition_data'];dev={k:list(map(int,og[k]['test'][()])) for k in og}
  old_train=set().union(*(set(map(int,og[str(i)]['train'][()])) for i in (0,1)))
  dev_set=set().union(*(set(x) for x in dev.values()))
  reserved=set().union(*(set(map(int,source[str(i)]['train'][()])) for i in list(range(100))+list(range(800,1000))))
  for c in groups:
   ids=list(map(int,source[str(c)]['train'][()]))
   # Fixed dataset has 120/client; never substitute a different cohort based on outcomes.
   assert len(ids)==120 and len(set(ids))==120 and set(ids)<=universe
   assert not set(ids)&(old_train|dev_set|reserved)
   indices[c]=ids;clients[c]=[{'text':decode_scalar(data['X'][str(i)][()]),'label':decode_scalar(data['Y'][str(i)][()])} for i in ids]
   counts=Counter(r['label'] for r in clients[c]);assert set(counts)=={'1','2','3','4'} and min(counts.values())>=8
  assert not set(indices[100])&set(indices[101])
  assert len(set(r['text'] for cr in clients.values() for r in cr))==240
  assert not set(r['text'] for cr in clients.values() for r in cr)&set(decode_scalar(data['X'][str(i)][()]) for i in old_train|dev_set)
  new=out/'fixed_partition.h5'
  if new.exists():raise FileExistsError(str(new))
  shutil.copy2(fixed,new)
 with h5py.File(out/'fixed_partition.h5','r+') as f:
  method=f['pilot_uniform_2'];method.attrs['source_client_ids_json']=json.dumps([str(c) for c in groups]);g=method['partition_data']
  for logical,c in enumerate(groups):
   del g[str(logical)]['train'];g[str(logical)].create_dataset('train',data=np.array(indices[c],dtype=np.int64))
  for k in dev:assert list(map(int,g[k]['test'][()]))==dev[k]
 # Private rows never printed or copied into Git.
 private=out/'restricted';private.mkdir(mode=0o700,exist_ok=True)
 for c in groups:(private/('client_%d.jsonl'%c)).write_bytes(encoded_rows(clients[c]))
 import torch
 from transformers import AutoTokenizer,AutoModel
 torch.set_num_threads(1)
 mpath=S1/'models/all-MiniLM-L6-v2';t=time.perf_counter()
 tok=AutoTokenizer.from_pretrained(str(mpath),local_files_only=True);model=AutoModel.from_pretrained(str(mpath),local_files_only=True).eval()
 candidates=public_pool(S1/'reused/public_pool/candidates.jsonl')
 oldclients,oldids,_,_,_=load_staged_clients(S1/'private_staging/source',[1,21])
 oldzero=select_from_scores(private_histograms(oldclients,oldids,candidates,model,tok),candidates,oldids)
 assert hashlib.sha256(encoded_rows(oldzero)).hexdigest()==HISTORICAL_ZERO_SHA,'embedding/selection does not reproduce previous zero selection'
 start=time.perf_counter();selected=select_from_scores(private_histograms(clients,groups,candidates,model,tok),candidates,groups);vote_seconds=time.perf_counter()-start
 zero=private/'private_zero.jsonl';zero.write_bytes(encoded_rows(selected))
 # Same-source real reference matched to original real-control sampler and label sequence.
 from build_matched_real_control import choose_label_records
 buckets={c:{label:[] for label in ('1','2','3','4')} for c in groups}
 for c in groups:
  for idx,row in zip(indices[c],clients[c]):buckets[c][row['label']].append(dict(row,source_index=idx,client_id=c))
 from collections import deque
 bylabel={label:deque(choose_label_records(label,16,groups,buckets,57)) for label in ('1','2','3','4')}
 real=[]
 for row in selected:
  r=bylabel[row['label']].popleft();real.append({'text':r['text'],'label':r['label'],'client_id':r['client_id']})
 assert len({r['text'] for r in real})==64
 realpath=private/'real_matched.jsonl';realpath.write_bytes(encoded_rows(real))
 for arm,rows in [('PRIVATE_ZERO',selected),('REAL_MATCHED',real)]:
  assert len(rows)==64 and dict(Counter(r['label'] for r in rows))=={str(i):16 for i in range(1,5)}
  d=out/arm;d.mkdir(mode=0o700)
  cmd=[str(PY),str(REPO/'journal_extension/pack_synthetic_h5.py'),'--jsonl',str(zero if arm=='PRIVATE_ZERO' else realpath),'--source-data-file',str(DATA),'--data-out',str(d/'cloud_data.h5'),'--partition-out',str(d/'cloud_partition.h5'),'--partition-method','synthetic_cloud','--cloud-clients','1','--manifest-out',str(d/'pack_manifest.json'),'--require-equal-label-counts']
  with (d/'pack.log').open('w') as log:subprocess.run(cmd,env=env(),stdout=log,stderr=subprocess.STDOUT,check=True)
 d=out/'PUB_NLI';d.mkdir(mode=0o700)
 for name in ('cloud_data.h5','cloud_partition.h5'):shutil.copy2(S1/'guidance/public_nli_selection'/name,d/name)
 result={'status':'pass','client_ids':groups,'per_client_records':{str(c):len(clients[c]) for c in groups},'per_client_label_counts':{str(c):dict(Counter(r['label'] for r in clients[c])) for c in groups},'dev_records':512,'dev_indices_and_order_identical':True,'excluded_old_clients_and_reserved':{'ranges':['0-99','800-999'],'new_vs_old_train_or_dev_index_overlap':0,'new_vs_old_train_or_dev_exact_text_overlap':0},'privacy':'non-DP private selection and real reference; restricted','new_dp_release':False,'official_test_used':False,'same_source_partition_not_domain_shift':True,'old_zero_reproduced_byte_identically':True,'cpu_new_vote_seconds':vote_seconds,'all_guide_counts':64,'fixed_partition_sha256':digest(out/'fixed_partition.h5'),'guidance':{a:{n:digest(out/a/n) for n in ('cloud_data.h5','cloud_partition.h5')} for a in ['PUB_NLI','PRIVATE_ZERO','REAL_MATCHED']},'time':now()}
 write(out/'AUDIT.json',result);print(json.dumps({k:result[k] for k in ['status','client_ids','dev_records','old_zero_reproduced_byte_identically','new_dp_release']}))
if __name__=='__main__':main()
