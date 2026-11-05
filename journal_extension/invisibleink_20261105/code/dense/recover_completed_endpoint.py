"""Preserve completed numerical training with an explicitly unobserved teardown exit."""
import importlib.util,re,shutil,datetime
from common import *

def main():
    os.umask(0o077);d=ROOT/'runs/DP_E4/seed_57';incident=ROOT/'incident_20261104'
    incident.mkdir(exist_ok=True)
    for name,p in [('queue_before_recovery.json',ROOT/'QUEUE_STATUS.json'),('run_spec_before_recovery.json',d/'run_spec.json')]:
        if (incident/name).exists():raise FileExistsError('Recovery already attempted')
        shutil.copy2(p,incident/name)
    trace=[json.loads(x) for x in (ROOT/'RESOURCE_TRACE.jsonl').read_text().splitlines()]
    last=trace[-1];previous=trace[-2]
    assert sorted(last['foreign_pids'])==sorted(previous['owned_pids'])==[718726,718727,718728]
    source=REPO/'journal_extension/summarize_fed_pilot.py'
    spec=importlib.util.spec_from_file_location('baseline_validator',source);v=importlib.util.module_from_spec(spec);spec.loader.exec_module(v)
    text=(d/'train.log').read_text(errors='replace');ev,rejected=v.parse_metrics(text);expected=v.expected_eval_rounds(50,1,True)
    assert len(ev)==51 and not rejected
    for i,e in enumerate(ev):e.update(evaluation_index=i,round=expected[i])
    finished=[i for i,line in enumerate(text.splitlines(),1) if 'training is finished' in line]
    fatal=[p for p in v.FATAL_PATTERNS if p in text]
    checks={'exit_code_zero':None,'single_completion_marker_present':len(finished)==1,
        'no_fatal_pattern':not fatal,'all_expected_evaluations_present':len(ev)==51,
        'final_round_evaluation_present':ev[-1]['round']==49,
        'completion_follows_final_evaluation':len(finished)==1 and finished[0]>ev[-1]['log_line']}
    assert all(x for k,x in checks.items() if k!='exit_code_zero')
    assert text.count('aggregate time cost')==750
    directions=sum(int(n) for n in re.findall(r'\[ZGR\] finite differences: nonzero=\d+/(\d+)',text));assert directions==1500
    for marker in ['FedSGD server is shutting down gracefully','FedSGD cloud is shutting down gracefully','FedSGD client is shutting down gracefully','done running']:
        assert marker in text,marker
    for pid in [718726,718727,718728]:assert any(line.startswith(str(pid)+' ') and 'handle_receive_message stopped' in line for line in text.splitlines())
    cp=ROOT/'scratch/DP_E4/seed_57/model_output/diagnostic_checkpoints'
    assert {p.stem for p in cp.glob('*.pt')}=={'round_00','round_10','round_25','round_50'}
    assert json.loads((cp/'round_50.json').read_text())['completed_round']==50
    assert json.loads((cp/'round_00.json').read_text())['trainable_sha256']=='14b42621935db8b4fc75724b193b366d194bd8757541da410a75cf8cb7d5243b'
    # Final logged completion predates the false-positive watchdog event.
    times=[datetime.datetime.strptime(m,'%Y-%m-%d,%H:%M:%S.%f').timestamp() for m in re.findall(r'(\d{4}-\d\d-\d\d,\d\d:\d\d:\d\d\.\d+)',text)]
    assert max(times)<last['time']
    shutil.copytree(cp,d/'diagnostic_checkpoints')
    audit={'status':'pass','time':time.time(),'reason':'Same previously verified MPI PIDs misclassified at teardown; all numerical work and graceful-stop logs precede incident.',
        'exit_code_observed':None,'strict_exit_code_check_passed':False,'log_sha256':digest(d/'train.log'),
        'checkpoint_file_sha256':{p.name:digest(p) for p in (d/'diagnostic_checkpoints').iterdir()},
        'final_log_time':max(times),'watchdog_time':last['time'],'evaluations':51,'updates':750,'directions':1500,'queries':3000,
        'rerun_training':False,'data_regenerated':False,'limitation':'No recorded process exit status; explicitly retained as teardown-unobserved completion, not strict clean-exit completion.'}
    write(d/'completion_recovery.json',audit)
    write(d/'metrics.json',{'schema_version':1,'status':'completed_training_teardown_unobserved','condition':'client_syn','cloud_source':'dp_e4','evaluation_mode':'dev','seed':57,
        'evaluations':ev,'expected_eval_rounds':expected,'final':ev[-1],'pre_update':ev[0],'checks':checks,'rejected_metric_lines':[],'fatal_matches':fatal})
    write(d/'budget_audit.json',{'status':'pass','updates':750,'directions':1500,'queries':3000})
    run=json.loads((d/'run_spec.json').read_text());run.update(status='completed_training_teardown_unobserved',exit_code=None,wall_seconds=None,
        observed_log_duration_seconds=max(times)-run['start_time'],completion_recovery=str(d/'completion_recovery.json'))
    write(d/'run_spec.json',run)
    assert endpoint_complete(d)
    write(ROOT/'TEARDOWN_RECOVERY_AUDIT.json',audit)
    print(json.dumps({'status':run['status'],'accuracy':ev[-1]['acc'],'exit_code_observed':None,'all_non_exit_checks_pass':True}))

if __name__=='__main__':main()
