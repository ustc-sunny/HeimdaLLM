"""Replay historical sample order, after auditing cached text/label multisets."""
import pickle,shutil,sys
from collections import Counter
from common import *
sys.path.insert(0,str(REPO))
import h5py

def historical_cache(seed,out,include_cloud):
 old=S1/'runs/PUB_NLI'/('seed_%d'%seed)/'model_output/cache_dir'
 audit=[]
 for rank in ([0,1,2] if include_cloud else [1,2]):
  data=S1/'guidance/public_nli_selection/cloud_data.h5' if rank==0 else DATA
  part=S1/'guidance/public_nli_selection/cloud_partition.h5' if rank==0 else S1/'shared/fixed_partition.h5'
  identity={}
  for key,p in [('data',data),('partition',part)]:
   resolved=os.path.realpath(p);st=os.stat(resolved)
   identity[key]={'path':resolved,'device':st.st_dev,'inode':st.st_ino,'size':st.st_size,'mtime_ns':st.st_mtime_ns,'ctime_ns':st.st_ctime_ns}
  if rank==0:identity['cloud_selection']={'client_ids':'0','client_count':None}
  fp=hashlib.sha256(json.dumps(identity,sort_keys=True).encode()).hexdigest()[:16]
  files=list((old/('rank_%d'%rank)).glob('*'));assert len(files)==(2 if rank==2 else 1)
  for p in files:
   value=pickle.load(p.open('rb'))
   cid=p.name.rsplit('_',2)[-2]
   with h5py.File(data,'r') as d,h5py.File(part,'r') as f:
    attrs=json.loads(d['attributes'][()].decode());vocab=attrs['label_vocab']
    method='synthetic_cloud' if rank==0 else 'pilot_uniform_2';g=f[method]['partition_data']
    ids=['0'] if rank==0 else list(g.keys()) if rank==1 else [cid]
    for slot,split in [(0,'train'),(3,'test')]:
     examples=value[slot]
     if examples is None:continue
     indices=[int(i) for c in ids for i in g[c][split][()]]
     expected=Counter((d['X'][str(i)][()].decode(),vocab[d['Y'][str(i)][()].decode()]) for i in indices)
     assert Counter((e.text_a,e.label) for e in examples)==expected,(rank,cid,split)
   dest=out/('rank_%d'%rank)/(p.name.rsplit('_',1)[0]+'_'+fp)
   dest.parent.mkdir(parents=True,exist_ok=True)
   if dest.exists():raise FileExistsError(dest)
   shutil.copy2(p,dest)
   audit.append({'rank':rank,'source':str(p),'dest':str(dest),'sha256':digest(p),'multiset_verified':True,'guid_warning':'old cache guids may not identify the corresponding row because old thread completion order differed; training uses aligned text/labels, not guid'})
 return audit
