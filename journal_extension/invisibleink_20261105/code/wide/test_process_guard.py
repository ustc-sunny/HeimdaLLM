"""Reproduce exiting-process false positives, while retaining foreign/PID-reuse blocking."""
import tempfile,shutil,json,os
from pathlib import Path
import common

def main():
    results=[]
    with tempfile.TemporaryDirectory() as tmp:
        root=Path(tmp);pid=12345;p=root/str(pid);p.mkdir()
        def put(state,start,marked):
            fields=[state]+['0']*18+[str(start)]+['0']*5
            (p/'stat').write_text(str(pid)+' (test process) '+' '.join(fields))
            (p/'environ').write_bytes(common.MARKER.encode()+b'\0' if marked else b'')
        def check(expected,label):
            actual=common.classify_process(pid,root);assert actual==expected,(label,actual,expected);results.append(label)
        put('S',100,True);check('owned','live marked task')
        put('Z',100,False);check('gone','exiting zombie with empty environment')
        put('S',100,False);check('owned','verified same PID/start-time after environment vanishes')
        put('S',200,False);check('foreign','PID reused by unmarked same-user task')
        put('S',201,False);check('foreign','unrelated active task remains blocked')
        put('X',201,False);check('gone','dead process is not foreign')
        shutil.rmtree(p);check('gone','GPU list can retain an already removed PID')
    out={'status':'pass','cases':len(results),'checks':results}
    (Path(__file__).parent/'PROCESS_GUARD_TESTS.json').write_text(json.dumps(out,indent=2)+'\n')
    print(json.dumps(out))

if __name__=='__main__':main()
