"""Recompute published aggregate evidence without models, data, or network access."""
import csv
import hashlib
import json
import statistics
from pathlib import Path

ROOT=Path(__file__).resolve().parent

def main():
    result=json.loads((ROOT/'results/RESULTS.json').read_text())
    provenance=json.loads((ROOT/'protocol/ENDPOINT_PROVENANCE.json').read_text())
    assert result['complete_endpoints']==14 and len(result['rows'])==14
    keyed={(r['arm'],r['seed']):r for r in provenance}
    points=0
    for row in result['rows']:
        p=ROOT/'results/endpoint_metrics'/('%s_seed%d.json'%(row['arm'],row['seed']))
        m=json.loads(p.read_text());assert len(m['evaluations'])==51
        entry=keyed[(row['arm'],row['seed'])]
        assert entry['budget']=={'status':'pass','updates':750,'directions':1500,'queries':3000}
        for e in m['evaluations']:
            cm=e['confusion_matrix'];assert sum(map(sum,cm))==512
            acc=sum(cm[i][i] for i in range(4))/512
            f1=[]
            for i in range(4):
                d=sum(cm[i])+sum(r[i] for r in cm)
                f1.append(2*cm[i][i]/d if d else 0.)
            assert abs(acc-e['acc'])<1e-12
            assert abs(statistics.mean(f1)-e['macro_f1'])<1e-12
            points+=1
        assert abs(m['final']['acc']-row['dev_accuracy'])<1e-12
        assert abs(statistics.mean(e['acc'] for e in m['evaluations'][-10:])-row['last10_accuracy'])<1e-12
    assert points==714
    manifest=ROOT/'PUBLICATION_MANIFEST.json'
    if manifest.exists():
        for name,v in json.loads(manifest.read_text())['files'].items():
            assert hashlib.sha256((ROOT/name).read_bytes()).hexdigest()==v['sha256'],name
    print(json.dumps({'status':'pass','endpoints':14,'recomputed_evaluations':714,
        'official_test_access':False,'model_execution':False}))

if __name__=='__main__':main()
