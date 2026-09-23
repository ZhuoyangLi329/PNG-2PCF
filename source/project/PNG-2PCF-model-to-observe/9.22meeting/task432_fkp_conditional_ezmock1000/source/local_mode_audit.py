"""独立Numpy复核模式正交、局部参数消除及25phase的不确定性。"""
import json
import numpy as np
from pathlib import Path
p=Path('outputs/task432_fkp_conditional_ezmock1000')
with np.load(p/'conditional_modes.npz') as a:
    M,C,B,J=a['modes'],a['covariance'],a['contrast_response'],a['jacobian']
    h,e,mu=a['halo_stack'],a['ez_stack'],a['model_anchor']
    hc=(h-mu)@B.T
    checks={'mode_C_identity_maxabs':float(np.max(abs(M@C@M.T-np.eye(5)))),
            'contrast_common_parameter_derivative_maxabs':float(np.max(abs(B@J))),
            'EZ_covariance_relative_Fro':float(np.linalg.norm(np.cov(e,rowvar=False)-C)/np.linalg.norm(C)),
            'halo_mean_matches_frozen':bool(np.allclose(h.mean(0),a['data'],rtol=1e-9,atol=1e-10)),
            'mode_all_finite':bool(np.all(np.isfinite(M)))}
    S=np.cov(hc,rowvar=False);mean=hc.mean(0)
    checks['mean_contrast_T2_using_2D_halo_covariance']=float(25*mean@np.linalg.solve(S,mean))
    checks['two_contrast_mean']=mean.tolist()
    checks['no_74D_halo_covariance_inversion']=True
assert checks['mode_C_identity_maxabs']<1e-8 and checks['contrast_common_parameter_derivative_maxabs']<1e-9
assert checks['EZ_covariance_relative_Fro']<1e-12 and checks['halo_mean_matches_frozen'] and checks['mode_all_finite']
checks['status']='pass'
(p/'independent_mode_audit.json').write_text(json.dumps(checks,indent=2)+'\n')
print(json.dumps(checks))
