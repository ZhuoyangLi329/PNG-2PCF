"""独立检验整个预定义低维协方差，包括模式间相关，不只检查方差。"""
from pathlib import Path
import numpy as np,json
p=Path('outputs/task432_fkp_conditional_ezmock1000')
with np.load(p/'conditional_modes.npz') as a:
    hm=a['halo_modes'];ez=a['ez_modes'];B=a['contrast_response'];C=a['covariance'];J=a['jacobian'];M=a['modes'];d=a['data'];mu=a['model_anchor']
rng=np.random.default_rng(4322411)
report={'seed':4322411,'n_draws':10000,'n_per_draw':25,'scope':'Finite25sample calibration of covariance shape in preselected mode subspaces. Mock means removed; no formal covariance replaced.','subspaces':{}}
for n in (2,5):
    H=np.cov(hm[:,:n],rowvar=False)
    obs=24*(np.trace(H)-np.linalg.slogdet(H)[1]-n)
    vals=[];eigen=[]
    for i in range(20):
        s=ez[rng.integers(0,len(ez),(500,25)),:n];s-=s.mean(1,keepdims=True)
        v=np.einsum('bni,bnj->bij',s,s)/24
        vals.extend(24*(np.trace(v,axis1=1,axis2=2)-np.linalg.slogdet(v)[1]-n))
        eigen.extend(np.linalg.eigvalsh(v))
    ratio=(1+np.count_nonzero(np.array(vals)>=obs))/10001
    report['subspaces'][str(n)]={'halo_covariance':H.tolist(),'halo_eigenvalues':np.linalg.eigvalsh(H).tolist(),'loss_statistic':float(obs),'EZ_resampling_upper_tail':float(ratio),'eigenvalue_95_intervals':np.percentile(eigen,[2.5,97.5],axis=0).tolist()}
(p/'lowdim_covariance_calibration.json').write_text(json.dumps(report,indent=2)+'\n')
print(json.dumps(report))
