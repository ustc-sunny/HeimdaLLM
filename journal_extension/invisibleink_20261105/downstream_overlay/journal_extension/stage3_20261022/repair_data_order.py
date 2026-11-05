"""Correct a verified threaded read-order defect before any formal Stage-3 endpoint."""
from common import *
import shutil
assert not (ROOT/'runs').exists(),'formal runs already started'
archive=ROOT/'preflight_failure_01';archive.mkdir()
for n in ['BLOCKED.json','DIAGNOSTICS_BLOCKED.json','ORCHESTRATOR_STARTED.json','DIAGNOSTIC_WATCHER_STARTED.json','MONITOR_STOP','SOURCE_AUDIT.json','source.patch']:
 p=ROOT/n
 if p.exists():shutil.copy2(p,archive/n)
for n in ['BLOCKED.json','DIAGNOSTICS_BLOCKED.json','ORCHESTRATOR_STARTED.json','DIAGNOSTIC_WATCHER_STARTED.json','MONITOR_STOP']:
 p=ROOT/n
 if p.exists():p.unlink()
p=REPO/'data_manager/text_classification_data_manager.py'
s=p.read_text();start=s.index('    def read_instance_from_h5(')
new='''    def read_instance_from_h5(self, data_file, index_list, desc=""):
        # Preserve the frozen partition order. Thread completion order is not
        # reproducible and previously also detached guid from its text row.
        X, y = [], []
        for idx in index_list:
            X.append(data_file["X"][str(idx)][()].decode("utf-8"))
            y.append(data_file["Y"][str(idx)][()].decode("utf-8"))
        return {"X": X, "y": y}
'''
assert s[start:].rstrip().endswith('return {"X": X, "y": y}')
(archive/'text_classification_data_manager.py').write_text(s);p.write_text(s[:start]+new)
audit=json.loads((ROOT/'SOURCE_AUDIT.json').read_text())
audit['modified_sha256'][str(p.relative_to(REPO))]=digest(p)
audit['data_order_amendment']=now();write(ROOT/'SOURCE_AUDIT.json',audit)
(ROOT/'source.patch').write_text(subprocess.check_output(['git','diff'],cwd=REPO,text=True))
write(ROOT/'DATA_ORDER_AMENDMENT.json',{'time':now(),'reason':'Historical loader appended text/label pairs in thread completion order; identical file hashes and seed did not guarantee identical cached order. Found before formal endpoints.','formal_endpoints_before_fix':0,'new_cohort':'sequential HDF5 reads preserve frozen partition order for all arms and seeds','original_control':'replay corresponding historical PUB_NLI seed cached feature order after multiset audit; this anchors the comparison to PUB_NLI, not all historical arms','bridge':'use exact historical caches; one round to preserve the 50-round LR schedule prefix; two-round shortened schedule also changes second-round LR','history_limitation':'old arms may include uncontrolled data-order randomness; paired FL seeds alone do not ensure fully paired input order. No historical result replaced.','retry_reason':'implementation reproducibility repair, not favorable-accuracy selection','first_failed_bridge_preserved':'preflight/bridge_single','not_changed':['candidate membership','private selection','guidance count','FL seeds','formal 50-round schedule','gradient/update operator','learning rate','query budget']})
print('Data order correction applied before formal training; first failure preserved.')
