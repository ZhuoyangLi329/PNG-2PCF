#!/usr/bin/env python3
from __future__ import annotations
import json, os, sys
from pathlib import Path
for key in ("OMP_NUM_THREADS","MKL_NUM_THREADS","OPENBLAS_NUM_THREADS"):
    os.environ.setdefault(key,"1")
import emcee
import numpy as np
from scipy.interpolate import RegularGridInterpolator
from scipy.optimize import least_squares
ROOT=Path("/pscratch/sd/l/lzy/PNG-2PCF-model-to-observe")
for d in (ROOT/"codes/task432",ROOT/"codes/task43",ROOT/"codes/task44"):
    if str(d) not in sys.path: sys.path.insert(0,str(d))
from task432_full_ric_compile import Engine
from task432_hybrid_jaxpower_0918_contract import Metric
GRID=ROOT/"outputs/task43_outputs/rsd_validation/task432_model_repair/velocileptors_fullric_v1/operator/local/velo_fullric_grid.npz"
COV=ROOT/"outputs/task43_outputs/rsd_validation/task432_model_repair/hybrid_ezmock1000_0918_contract_v1/frozen_inputs.npz"
COMPILED=ROOT/"outputs/task43_outputs/rsd_validation/task432_model_repair/full_ric_p02xi02_v1/pilot/mean4xi_mean2P_prodrr_stochastic/engine.npz"
OUT=ROOT/"9.30meeting/task432_velocileptors_fullric"

def main():
    if not GRID.is_file(): raise FileNotFoundError(GRID)
    with np.load(GRID,allow_pickle=False) as g:
        f,b=g["f_grid"],g["b_grid"]; raw=g["raw_xi"]; ric=g["ric_clustering"]; shot=g["fixed_shot"]; snresp=g["sn0_ric_response"]
    raw_i=RegularGridInterpolator((f,b),raw,bounds_error=True); ric_i=RegularGridInterpolator((f,b),ric,bounds_error=True)
    with np.load(COV,allow_pickle=False) as z: dp,dx,C=z["data_p"],z["data_x"],z["covariance"]
    p=Engine("p02",COMPILED); data=np.r_[dp,dx]
    def xi(t):
        t=np.asarray(t); return raw_i([t[0],t[1]])[0]+ric_i([t[0],t[1]])[0]+shot+t[2]*snresp
    def pp(t): return p.predict(np.asarray(t)[None,:])[0]
    def logp_x(t):
        if np.any(t<[-100,2.3,-1]) or np.any(t>[100,2.55,1]): return -np.inf
        return -.5*Metric(C[22:,22:]).chi2(dx-xi(t))
    def logp_j(t):
        if np.any(t<[-100,2.3,0,-1]) or np.any(t>[100,2.55,30,1]): return -np.inf
        return -.5*Metric(C).chi2(data-np.r_[pp(t),xi([t[0],t[1],t[3]])])
    def run(fn,start,names,seed):
        rng=np.random.default_rng(seed); ndim=len(start); nw=16; ns=1200; burn=200
        scales=np.array([8.,.02,.15]) if ndim==3 else np.array([8.,.02,.4,.05])
        init=np.asarray(start)[None,:]+rng.normal(size=(nw,ndim))*scales
        lo=np.array([-100.,2.3,-1.]) if ndim==3 else np.array([-100.,2.3,0.,-1.]); hi=np.array([100.,2.55,1.]) if ndim==3 else np.array([100.,2.55,30.,1.])
        init=np.clip(init,lo+1e-6,hi-1e-6); np.random.seed(seed); sam=emcee.EnsembleSampler(nw,ndim,fn); sam.run_mcmc(init,ns,progress=False); chain=sam.get_chain(discard=burn,flat=True); lp=sam.get_log_prob(discard=burn,flat=True); im=int(np.argmax(lp)); return {"chain":chain,"summary":{"map_theta":chain[im].tolist(),"map_chi2":float(-2*lp[im]),"q16_q50_q84":np.percentile(chain,[16,50,84],axis=0).tolist()},"parameter_names":names,"nwalkers":nw,"nsteps":ns,"burnin":burn}
    # Use existing formal P02 samples to align the 9.18 page; only xi/joint are rerun here.
    pp_path=ROOT/"outputs/task43_outputs/rsd_validation/task432_model_repair/full_ric_p02xi02_v1/formal_mean4/p02/samples.npz"
    with np.load(pp_path,allow_pickle=False) as z: pchain=np.asarray(z["chain"])[5000:].reshape(-1,4); pmap=np.asarray(z["map_theta"])
    chains={"diag_p02":{"chain":pchain[:200000],"summary":{"map_theta":pmap.tolist()},"parameter_names":["fNL","b1","sigma_s_P","sn0"]},"diag_xi02":run(logp_x,[0.,2.42,0.], ["fNL","b1","sn0"], 93002),"diag_joint":run(logp_j,[0.,2.41,2.2,.08],["fNL","b1","sigma_s_P","sn0"],93003)}
    np.savez_compressed(OUT/"diagnostic_chains.npz",xi_chain=chains["diag_xi02"]["chain"],joint_chain=chains["diag_joint"]["chain"],p_chain=chains["diag_p02"]["chain"])
    summary={k:{"summary":v["summary"],"parameter_names":v["parameter_names"],"nwalkers":v.get("nwalkers"),"nsteps":v.get("nsteps"),"burnin":v.get("burnin")} for k,v in chains.items()}
    (OUT/"diagnostic_chain_summary.json").write_text(json.dumps(summary,indent=2)+"\n")
    # Reuse the established 9.18 triangle renderer.
    import matplotlib; matplotlib.use("Agg")
    from matplotlib.backends.backend_pdf import PdfPages
    from task43_plot_ezmock506_covariance_mcmc_vs_jaxpower import triangle_page
    pdf=OUT/"figures"/"task432_velocileptors_fullric_9.18style_triangle.pdf"; pdf.parent.mkdir(parents=True,exist_ok=True)
    with PdfPages(pdf) as handle:
        triangle_page(handle,cov="diag",cov_display="velocileptors xi + full-RIC diagnostic",chains=chains,ranges={"fNL":(-100,100),"b1":(2.25,2.55)},truth_fnl=0.0)
    print(json.dumps({"status":"pass","pdf":str(pdf),"summary":str(OUT/"diagnostic_chain_summary.json")},sort_keys=True))
if __name__=="__main__": main()

