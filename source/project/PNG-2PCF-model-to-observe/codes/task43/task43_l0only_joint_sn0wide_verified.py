#!/usr/bin/env python3
"""EZmock900 P0+xi0 joint fit with a widened free sn0 prior."""
from __future__ import annotations
import json,time
from pathlib import Path
import emcee
import numpy as np
from scipy.linalg import solve_triangular
from scipy.optimize import least_squares
import task43_rsd_ezmock281_mcmc_compare as base
from task43_joint_rsd_pkxi_fit import summarize_chain

ROOT=base.PROJECT_ROOT; PARENT=ROOT/'outputs/task43_outputs/ezmock_rsd_lightcone_covariance_z0p4_0p8_x1000_common50_nzmatch_b0.30_z0.60'; SHARED=PARENT/'mcmc_preliminary_900'; OUT=PARENT/'mcmc_preliminary_900_l0only_sn0wide_verified'
NAMES=('fNL','b1','sigma_s','sn0'); LO=np.array([-500.,.5,0.,-3.]); HI=np.array([500.,5.,30.,3.]); INIT=np.array([4.,.015,.12,.08])
def get():
    pm,xm,dp,dx,_,_=base.build_models_and_data(); data=np.r_[dp[:13],dx[:base.N_XI]]
    def p(t): return pm.evaluate(np.asarray(t,dtype='f8'))[:13]
    def x(t): return xm.evaluate(np.asarray(t,dtype='f8')[:3],model='formal_gic',window_key='mean')[0][base.XI_MASK]
    def ev(t): return np.r_[p(t),x(t)]
    return ev,data
def main():
    ev,data=get()
    with np.load(SHARED/'ezmock900_covariance_and_stack.npz') as d: c=d['covariance']; idx=np.r_[np.arange(13),np.arange(22,48)]; cov=(c[np.ix_(idx,idx)]+c[np.ix_(idx,idx)].T)/2.
    meta=base.correction_factors(900,len(data),4); scale=np.sqrt(np.diag(cov)); corr=cov/np.outer(scale,scale); chol=np.linalg.cholesky(corr); precision=meta['hartlap']*np.linalg.inv(corr)/np.outer(scale,scale)
    starts=[np.array([0.,2.5,8.,0.]),np.array([-20.,2.4,4.,.5]),np.array([20.,2.6,12.,-1.])]
    sols=[least_squares(lambda t:solve_triangular(chol,(data-ev(t))/scale,lower=True),np.clip(s,LO+1e-7,HI-1e-7),bounds=(LO,HI),x_scale='jac',max_nfev=3000,xtol=1e-11,ftol=1e-11,gtol=1e-11) for s in starts]; best=min(sols,key=lambda s:float(s.fun@s.fun)); theta=best.x; chi2=float(best.fun@best.fun); print('MAP',theta.tolist(),chi2,flush=True)
    def logp(t):
        if np.any(t<LO) or np.any(t>HI): return -np.inf
        d=data-ev(t); return -.5*float(d@precision@d)
    rng=np.random.default_rng(20260930); init=theta[None,:]+rng.normal(size=(64,4))*INIT[None,:]; init=np.clip(init,LO+1e-7,HI-1e-7); sampler=emcee.EnsembleSampler(64,4,logp); tic=time.time()
    for i,_ in enumerate(sampler.sample(init,iterations=30000),1):
        if i%5000==0: print('PROGRESS',i,round(time.time()-tic),flush=True)
    chain=np.asarray(sampler.get_chain(discard=5000),dtype='f8'); lp=np.asarray(sampler.get_log_prob(discard=5000),dtype='f8'); raw=summarize_chain(chain,lp,NAMES); raw['posterior_raw']=json.loads(json.dumps(raw['posterior'])); result=base.corrected_summary(raw,meta['percival_m1']); result.update({'covariance':'ezmock900','variant':'joint_p0_xi0_sn0wide','multipoles':['ell0'],'excluded_multipoles':['ell2'],'sn0_prior':[-3.,3.],'map_theta':theta.tolist(),'map_chi2_single':chi2,'nwalkers':64,'nsteps':30000,'burnin':5000,'nsteps_postburn':len(chain),'acceptance_fraction_mean':float(sampler.acceptance_fraction.mean()),'hartlap_percival':meta,'data_contract':'P0 13 + xi0 26; full P0-xi0 covariance; sn0 prior [-3,3]'})
    OUT.mkdir(parents=True,exist_ok=True); np.savez_compressed(OUT/'chain_ezmock900_joint_p0_xi0_sn0wide.npz',chain=chain,logp=lp,data=data); (OUT/'summary_ezmock900_joint_p0_xi0_sn0wide.json').write_text(json.dumps(result,indent=2)+'\n'); print(json.dumps({'posterior':result['posterior'],'gates':result['gates'],'map':theta.tolist(),'chi2':chi2},sort_keys=True),flush=True)
if __name__=='__main__': main()
