"""CPU-only verification of all18 published endpoints, without private inputs."""
import hashlib,json,statistics
from pathlib import Path

ROOT=Path(__file__).resolve().parent

def main():
    result=json.loads((ROOT/'results/RESULTS.json').read_text())
    provenance=json.loads((ROOT/'results/PROVENANCE.json').read_text())
    assert result['complete_endpoints']==18 and result['new_complete_endpoints']==4
    rows={(r['arm'],r['seed']):r for r in result['rows']};points=0
    for item in provenance:
        p=ROOT/item['relative_metrics_path'];assert hashlib.sha256(p.read_bytes()).hexdigest()==item['metrics_sha256']
        m=json.loads(p.read_text());assert len(m['evaluations'])==51
        for e in m['evaluations']:
            cm=e['confusion_matrix'];assert sum(map(sum,cm))==512
            assert abs(sum(cm[i][i] for i in range(4))/512-e['acc'])<1e-12
            f1=[]
            for i in range(4):
                d=sum(cm[i])+sum(r[i] for r in cm)
                f1.append(2*cm[i][i]/d if d else 0.)
            assert abs(statistics.mean(f1)-e['macro_f1'])<1e-12
            points+=1
        row=rows[(item['arm'],item['seed'])]
        assert abs(m['final']['acc']-row['dev_accuracy'])<1e-12
        if not row['reused']:assert row['exit_code_observed']==0 and all(m['checks'].values())
    assert points==918
    for name,v in json.loads((ROOT/'PUBLICATION_MANIFEST.json').read_text())['files'].items():
        assert hashlib.sha256((ROOT/name).read_bytes()).hexdigest()==v['sha256'],name
    print(json.dumps({'status':'pass','endpoints':18,'new_endpoints':4,'evaluation_points':points,
        'official_test_access':False,'model_execution':False}))

if __name__=='__main__':main()
