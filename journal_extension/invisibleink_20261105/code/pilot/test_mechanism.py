"""CPU checks of sensitivity, accountant, public support, upstream equivalence."""
import ast, json, math, zipfile
from pathlib import Path
import numpy as np
from mechanism import calibration, scores, support, probabilities

def main():
    cfg=calibration(); C=cfg['clip']; B=3; T=96; tau=1.
    rho=2*T*C*C/(B*B*tau*tau)
    eps=rho+2*math.sqrt(rho*math.log(1e5))
    assert abs(eps-8) < 1e-10
    root=Path(__file__).resolve().parent
    z=zipfile.ZipFile(root/'upstream.zip')
    source=z.read('invisibleink-main/src/invink/utils.py').decode()
    # Parse the two pure numerical upstream functions only. Avoid optional imports.
    source=source.replace('print(f"Model loaded successfully on {"all devices" if device_map == "auto" else device_map}")', 'print(device_map)')
    tree=ast.parse(source)
    nodes=[n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name in ('difference_clip','get_topk')]
    for node in nodes:
        node.returns=None
        for arg in node.args.args: arg.annotation=None
    ns={'np':np, 'torch':type('Dummy', (), {'Tensor':type('Tensor',(),{})})}
    exec(compile(ast.Module(body=nodes, type_ignores=[]), '<upstream-pure-functions>', 'exec'), ns)
    rng=np.random.default_rng(921)
    max_range=0.; max_renyi_ratio=0.
    for _ in range(500):
        pub=rng.normal(size=27); a=rng.normal(size=(B,27))*4
        b=a.copy(); b[1]=rng.normal(size=27)*4
        x=scores(pub,a,'DP_E8',C); y=scores(pub,b,'DP_E8',C)
        assert np.max(np.abs(x-y)) <= 2*C/B + 1e-12
        assert np.allclose(x, ns['difference_clip'](a,pub,C).mean(axis=0))
        mask=support(pub,10,C,B)
        assert np.array_equal(mask, ns['get_topk'](pub,10,C,B)[0])
        ids,p=probabilities(x,mask,tau); _,q=probabilities(y,mask,tau)
        llr=np.log(p/q)
        assert np.ptp(llr) <= 4*C/(B*tau)+1e-12
        max_range=max(max_range,float(np.ptp(llr)))
        for alpha in (1.1,2.,4.,8.):
            div=math.log(float(np.sum(p**alpha*q**(1-alpha))))/(alpha-1)
            bound=alpha*2*C*C/(B*B*tau*tau)
            assert div <= bound+1e-12
            max_renyi_ratio=max(max_renyi_ratio,div/bound)
    # Partitioning cannot inspect text/label; fixed slot indices are disjoint.
    indices=[list(range(i*B,(i+1)*B)) for i in range(32)]
    assert len(set(sum(indices,[])))==96
    result={'status':'pass','cases':500,'calibration':cfg,
            'upstream_pure_functions_match':True,'max_observed_llr_range':max_range,
            'max_observed_renyi_bound_ratio':max_renyi_ratio,
            'note':'Tests supplement the analytical proof; no end-to-end floating point DP certification.'}
    (root/'CPU_TESTS.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result))
if __name__=='__main__':main()
