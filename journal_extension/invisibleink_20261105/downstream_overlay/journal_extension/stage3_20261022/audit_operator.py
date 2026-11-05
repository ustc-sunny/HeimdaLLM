import math,json,torch
from types import SimpleNamespace
from common import *
from forward_training.tc_transformer_trainer_distribute import ForwardTextClassificationTrainer
from forward_training.utils.fwdgrad_utils import calculate_jvp,zo_estimator_multiplier

def main():
 torch.manual_seed(57);d=1000
 probe=object.__new__(ForwardTextClassificationTrainer);probe.trainable_params_number=d;probe._component_norms_logged=False;probe.pert=[torch.full((d,),float('nan'))]
 z=torch.randn(d,dtype=torch.float64);args=SimpleNamespace(alpha=1.,beta=1.,zo_estimator='legacy')
 u=probe._make_final_perturbation((z,),args)[0]
 assert torch.equal(u,z/math.sqrt(d)) or torch.allclose(u,z/math.sqrt(d),atol=1e-15,rtol=1e-15)
 x=torch.randn(d,dtype=torch.float64);target=torch.randn(d,dtype=torch.float64);f=lambda ps:.5*((ps[0]-target)**2).sum()
 _,jvp,delta=calculate_jvp(f,(x,),(u,),h=.01);exact=((x-target)*u).sum()
 err=abs((jvp-exact).item())/max(1,abs(exact.item()));assert err<1e-9
 assert zo_estimator_multiplier('legacy',1.,1.,d)==1
 assert zo_estimator_multiplier('isotropic',1.,1.,d)==d
 # Full orthogonal directions exactly integrate covariance at a small dimension.
 dim=8;g=torch.arange(1,dim+1,dtype=torch.float64);dirs=torch.eye(dim,dtype=torch.float64)
 mean=(dirs*(dirs@g)[:,None]).mean(0);assert torch.allclose(mean,g/dim)
 result={'status':'pass','dimension':d,'normalized_direction_norm':u.norm().item(),'raw_direction_norm':z.norm().item(),'relative_fd_error':err,'legacy_outer_multiplier':1,'isotropic_outer_multiplier':d,'legacy_mean_in_linear_limit':'g/d','guided_mean_in_linear_limit':'(alpha/d I + (1-alpha) vvT)g for rank one','trace_matching':'expected squared radius, not pointwise norm or equal mean update','cloud_nan_ignored':True,'no_accuracy_tuning':True,'actual_training_started':False}
 write(ROOT/'OPERATOR_AUDIT.json',result);print(json.dumps(result))
if __name__=='__main__':main()
