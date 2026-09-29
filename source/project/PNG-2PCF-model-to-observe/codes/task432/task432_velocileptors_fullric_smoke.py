#!/usr/bin/env python3
from __future__ import annotations
import json, os, sys
from pathlib import Path
for key in ("OMP_NUM_THREADS","MKL_NUM_THREADS","OPENBLAS_NUM_THREADS"):
    os.environ.setdefault(key,"1")
import numpy as np
from scipy.interpolate import RegularGridInterpolator
from scipy.optimize import least_squares
ROOT=Path("/pscratch/sd/l/lzy/PNG-2PCF-model-to-observe")
for folder in (ROOT/"codes/task432",ROOT/"codes/task43",ROOT/"codes/task44"):
    if str(folder) not in sys.path: sys.path.insert(0,str(folder))
from task432_full_ric_compile import Engine
from task432_hybrid_jaxpower_0918_contract import Metric
GRID=ROOT/"outputs/task43_outputs/rsd_validation/task432_model_repair/velocileptors_fullric_v1/operator/velo_fullric_grid.npz"
COV=ROOT/"outputs/task43_outputs/rsd_validation/task432_model_repair/hybrid_ezmock1000_0918_contract_v1/frozen_inputs.npz"
COMPILED=ROOT/"outputs/task43_outputs/rsd_validation/task432_model_repair/full_ric_p02xi02_v1/pilot/mean4xi_mean2P_prodrr_stochastic/engine.npz"
OUT=ROOT/"9.30meeting/task432_velocileptors_fullric/diagnostics/smoke.json"
def main():
    with np.load(GRID,allow_pickle=False) as g:
        f,b=g["f_grid"],g["b_grid"]; raw= g["raw_xi"]; ric=g["ric_clustering"]; shot=g["fixed_shot"]; snresp=g["sn0_ric_response"]
    ir=RegularGridInterpolator((f,b),raw,bounds_error=True); ii=RegularGridInterpolator((f,b),ric,bounds_error=True)
    with np.load(COV,allow_pickle=False) as z:
        data_p,data_x,cov=z["data_p"],z["data_x"],z["covariance"]
    p_engine=Engine("p02",COMPILED); xi_engine=Engine("xi02",COMPILED); joint_engine=Engine("joint",COMPILED)
    def xi_pred(t):
        t=np.asarray(t); return ir([float(t[0]),float(t[1])])[0]+ii([float(t[0]),float(t[1])])[0]+shot+float(t[2])*snresp
    def p_pred(t): return p_engine.predict(np.asarray(t)[None,:])[0]
    def fit(kind,lam=1.):
        C=cov.copy(); C[:22,22:]*=lam; C[22:,:22]*=lam
        if kind=="xi": data=data_x; cc=C[22:,22:]; metric=Metric(cc); start=np.array([0.,2.42,0.]); lo=np.array([-500.,.5,-1.]); hi=np.array([500.,4.9,1.]); fun=lambda t:metric.white(data-xi_pred(t))
        else: data=np.r_[data_p,data_x]; cc=C; metric=Metric(cc); start=np.array([0.,2.4,2.2,.08]); lo=np.array([-500.,.5,0.,-1.]); hi=np.array([500.,4.9,30.,1.]); fun=lambda t:metric.white(data-np.r_[p_pred(t),xi_pred([t[0],t[1],t[3]])])
        sol=least_squares(fun,start,bounds=(lo,hi),x_scale="jac",max_nfev=1200,ftol=1e-10,xtol=1e-10,gtol=1e-9)
        return {"theta":sol.x.tolist(),"chi2":float(sol.fun@sol.fun),"success":bool(sol.success)}
    probes=[]
    for theta in ([0.,2.4,0.],[50.,2.5,.2],[-50.,2.3,-.2]):
        x=xi_pred(theta); recomposed=ir(theta[:2])[0]+ii(theta[:2])[0]+shot+theta[2]*snresp
        probes.append({"theta":list(theta),"recomposition_max_abs":float(np.max(np.abs(x-recomposed))),"finite":bool(np.all(np.isfinite(x)))})
    results={"xi_only":fit("xi"),"joint_lambda0":fit("joint",0.),"joint_lambda0p5":fit("joint",.5),"joint_lambda1":fit("joint",1.)}
    report={"status":"smoke_pass","operator":"velocileptors xi raw + full-RIC clustering response + fixed shot + sn0 response","grid_shape":list(raw.shape),"probes":probes,"results":results,"formal_status":"diagnostic_only","notes":["Coarse pilot grid f step 100, b step 0.4, nint=300; not a formal posterior.","Full-RIC direct velocileptors PNG operator is still minimal hybrid response, not complete CLPT-PNG."]}
    OUT.parent.mkdir(parents=True,exist_ok=True); OUT.write_text(json.dumps(report,indent=2)+"\n"); print(json.dumps(report,sort_keys=True))
if __name__=="__main__": main()
