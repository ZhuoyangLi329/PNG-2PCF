#!/usr/bin/env python3
from __future__ import annotations
import os,sys,json,time
from pathlib import Path
for key in ("OMP_NUM_THREADS","MKL_NUM_THREADS","OPENBLAS_NUM_THREADS"):
    os.environ.setdefault(key,"1")
import numpy as np
from scipy.optimize import least_squares
ROOT=Path("/pscratch/sd/l/lzy/PNG-2PCF-model-to-observe")
for d in (ROOT/"codes/task432",ROOT/"codes/task43",ROOT/"codes/task44"):
    if str(d) not in sys.path: sys.path.insert(0,str(d))
from task432_velocileptors_fullric_model import VeloPoles
from task432_full_ric_compile import Engine
from task432_hybrid_jaxpower_0918_contract import Metric
BASE=ROOT/"outputs/task43_outputs/rsd_validation/task432_model_repair/full_ric_p02xi02_v1"; COMPILED=BASE/"pilot/mean4xi_mean2P_prodrr_stochastic/engine.npz"; KERNEL=BASE/"geometry/validated_mean4_midpoint.npz"; COV=ROOT/"outputs/task43_outputs/rsd_validation/task432_model_repair/hybrid_ezmock1000_0918_contract_v1/frozen_inputs.npz"; OUT=ROOT/"9.30meeting/task432_velocileptors_fullric/diagnostics/direct_map.json"

def main():
    with np.load(KERNEL) as a: edges=np.asarray(a['separation_edges'],dtype='f8')
    with np.load(COV) as a: fitcenters=np.asarray(a['xi_centers'],dtype='f8'); data_p=np.asarray(a['data_p'],dtype='f8'); data_x=np.asarray(a['data_x'],dtype='f8'); C=np.asarray(a['covariance'],dtype='f8')
    with np.load(COMPILED) as a: xi_resp=np.asarray(a['xi_response'],dtype='f8'); shot=np.asarray(a['fixed_shot_vector'][22:],dtype='f8'); snresp=np.asarray(a['sn0_ric_response'][22:],dtype='f8')
    nodes,w=np.polynomial.legendre.leggauss(2); lo,hi=edges[:-1],edges[1:]; shell_r=0.5*(hi-lo)[:,None]*nodes[None,:]+0.5*(hi+lo)[:,None]; shell_w=0.5*(hi-lo)[:,None]*w[None,:]*shell_r**2; shell_den=(hi**3-lo**3)/3.; all_s=np.r_[fitcenters,shell_r.ravel()]; nfit=len(fitcenters); ns=len(lo)
    velo=VeloPoles(all_s); p_engine=Engine('p02',COMPILED); metric_x=Metric(C[22:,22:]);
    def xi_pred(t):
        t=np.asarray(t,dtype='f8'); poles=velo.evaluate(float(t[0]),float(t[1]),nint=300); shell=poles[:,nfit:].reshape(3,ns,2); shell=np.sum(shell*shell_w[None,:,:],axis=2)/shell_den[None,:]; raw=np.r_[poles[0,:nfit],poles[1,:nfit]]; ric=np.einsum('ild,ld->i',xi_resp,shell); return raw+ric+shot+float(t[2])*snresp
    def fit(kind,lam=1.):
        cov=C.copy(); cov[:22,22:]*=lam; cov[22:,:22]*=lam
        if kind=='xi': metric=Metric(cov[22:,22:]); start=np.array([0.,2.42,0.]); lo=np.array([-500.,.5,-1.]); hi=np.array([500.,5.,1.]); fun=lambda t:metric.white(data_x-xi_pred(t))
        else:
            metric=Metric(cov); start=np.array([0.,2.4,2.2,.08]); lo=np.array([-500.,.5,0.,-1.]); hi=np.array([500.,5.,30.,1.]); fun=lambda t:metric.white(np.r_[data_p,data_x]-np.r_[p_engine.predict(np.asarray(t)[None,:])[0],xi_pred([t[0],t[1],t[3]])])
        sols=[]
        for st in (start, start+np.array([30,.05,0,.02])[:len(start)], start+np.array([-30,-.05,0,-.02])[:len(start)]):
            sol=least_squares(fun,np.clip(st,lo+1e-8,hi-1e-8),bounds=(lo,hi),x_scale='jac',max_nfev=500,ftol=1e-9,xtol=1e-9,gtol=1e-8); sols.append(sol)
        sol=min(sols,key=lambda s:s.fun@s.fun); return {'theta':sol.x.tolist(),'chi2':float(sol.fun@sol.fun),'success':bool(sol.success),'nfev':int(sol.nfev)}
    report={'status':'diagnostic_only','model':'direct velocileptors xi with shell-averaged ell0/2/4 full-RIC response','nint':300,'shell_order':2,'results':{'xi_only':fit('xi')}}
    for lam in (0.,.5,1.): report['results'][f'joint_lambda{lam:g}']=fit('joint',lam)
    OUT.parent.mkdir(parents=True,exist_ok=True); OUT.write_text(json.dumps(report,indent=2)+"\n"); print(json.dumps(report,sort_keys=True))
if __name__=='__main__': main()
