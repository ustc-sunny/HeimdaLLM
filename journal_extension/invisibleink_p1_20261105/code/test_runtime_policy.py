"""Regression against the actual incident; retain sustained memory-pressure stop."""
import json
from pathlib import Path
from runtime_policy import resource_stop_reason

cases=[]
incident={'foreign_pids':[1042953],'free_mib':19849,'owned_pids':[1022068,1022069,1022070]}
assert resource_stop_reason(incident,None,0)==(None,None)
cases.append('Actual November4 foreign-arrival event does not kill healthy existing training')
assert resource_stop_reason({'foreign_pids':[],'free_mib':34853},None,0)==(None,None)
cases.append('Normal exclusive operation continues')
reason,t=resource_stop_reason({'foreign_pids':[1],'free_mib':512},None,0)
assert reason is None and t==0
reason,t=resource_stop_reason({'foreign_pids':[1],'free_mib':512},t,59)
assert reason is None and t==0
cases.append('Brief memory pressure does not cause immediate termination')
reason,t=resource_stop_reason({'foreign_pids':[1],'free_mib':512},t,60)
assert reason is not None
cases.append('Sustained low free memory triggers own-task stop')
assert resource_stop_reason({'foreign_pids':[1],'free_mib':2000},0,61)==(None,None)
cases.append('Recovered memory resets the low-memory timer')
out={'status':'pass','cases':cases}
(Path(__file__).parent/'RUNTIME_POLICY_TESTS.json').write_text(json.dumps(out,indent=2)+'\n')
print(json.dumps(out))
