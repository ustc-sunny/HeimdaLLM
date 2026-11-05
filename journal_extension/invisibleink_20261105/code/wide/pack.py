"""Pack exactly 64 generated records; no private-dependent filtering/dedup."""
import json,sys,os
from collections import Counter
from common import ROOT,REPO,DATA,BUDGETS,digest,write
def main():
    os.umask(0o077);arm=sys.argv[1]
    assert arm in BUDGETS
    sys.path.insert(0,str(REPO/'journal_extension'))
    import pack_synthetic_h5 as p
    import h5py,numpy as np
    source=ROOT/'generation'/arm/'synthetic.jsonl'
    assert json.loads(source.with_name('manifest.json').read_text())['status']=='complete'
    rows=[json.loads(l) for l in source.read_text().splitlines()]
    assert len(rows)==64 and Counter(r['label'] for r in rows)=={'1':16,'2':16,'3':16,'4':16}
    # Read public schema only; never read the source H5 official test examples.
    with h5py.File(str(DATA),'r') as f: meta=p.json_from_h5_scalar(f['attributes'][()])
    attributes={'index_list':list(range(64)),'train_index_list':list(range(64)),
        'test_index_list':[],'label_vocab':meta['label_vocab'],
        'num_labels':meta.get('num_labels',4),'task_type':meta.get('task_type','text_classification')}
    dest=ROOT/'guidance'/arm;dest.mkdir(parents=True,exist_ok=False)
    p.write_data_h5(h5py,dest/'cloud_data.h5',rows,attributes)
    p.write_partition_h5(h5py,np,dest/'cloud_partition.h5','synthetic_cloud',[list(range(64))])
    p.verify_outputs(h5py,dest/'cloud_data.h5',dest/'cloud_partition.h5','synthetic_cloud',64,meta['label_vocab'])
    write(dest/'manifest.json',{'records':64,'duplicates_dropped':0,'official_test_used':False,
          'source_sha256':digest(source),'cloud_data_sha256':digest(dest/'cloud_data.h5'),
          'cloud_partition_sha256':digest(dest/'cloud_partition.h5')})
if __name__=='__main__':main()
