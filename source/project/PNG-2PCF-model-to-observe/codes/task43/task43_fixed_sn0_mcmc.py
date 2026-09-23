#!/usr/bin/env python3
"""Formal joint MCMC with the P-side stochastic amplitude fixed to sn0=0."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import emcee
import numpy as np
import task43_rsd_ezmock281_mcmc_compare as base
from scipy.optimize import least_squares

ROOT = base.PROJECT_ROOT
SHARED = ROOT / 'outputs/task43_outputs/ezmock_rsd_lightcone_covariance_z0p4_0p8_x1000_common50_nzmatch_b0.30_z0.60/mcmc_preliminary_900'
OUT = ROOT / 'outputs/task43_outputs/ezmock_rsd_lightcone_covariance_z0p4_0p8_x1000_common50_nzmatch_b0.30_z0.60/mcmc_preliminary_900_fixed_sn0'
NAMES = ('fNL','b1','sigma_s')
LOWER = np.asarray([-500., .5, 0.], dtype='f8')
UPPER = np.asarray([500., 5., 30.], dtype='f8')
INIT_SCALE = np.asarray([4., .015, .12], dtype='f8')

def correction(ns, nb, npfit):
    hartlap=(ns-nb-2.)/(ns-1.)
    A=2./((ns-nb-1.)*(ns-nb-4.)); B=(ns-nb-2.)/((ns-nb-1.)*(ns-nb-4.))
    m1=(1.+B*(nb-npfit))/(1.+A+B*(npfit+1.))
    return {'hartlap':hartlap,'percival_m1':m1,'percival_sigma_factor':float(np.sqrt(m1))}

def make_models(pm, xm):
    def evaluate(theta):
        theta=np.asarray(theta, dtype='f8'); f,b,s=theta
        a=pm.evaluate(np.asarray([f,b,s,0.]))
        p=np.r_[a[:13],a[13:26][base.P2_KEEP]]
        z=xm.evaluate(np.asarray([f,b,s]), model='formal_gic', window_key='mean')
        x=np.r_[z[0][base.XI_MASK], z[2][base.XI_MASK]]
        return np.r_[p,x]
    return evaluate

def map_fit(evaluate, data, cov):
    chol=np.linalg.cholesky(.5*(cov+cov.T))
    def resid(t): return np.linalg.solve(chol, data-evaluate(t))
    starts=[np.array([0.,2.5,8.]),np.array([-20.,2.4,3.]),np.array([20.,2.6,12.])]
    sols=[least_squares(resid,np.clip(s,LOWER+1e-7,UPPER-1e-7),bounds=(LOWER,UPPER),max_nfev=3000,xtol=1e-11,ftol=1e-11,gtol=1e-11) for s in starts]
    best=min(sols,key=lambda z:float(z.fun@z.fun))
    return np.asarray(best.x),float(best.fun@best.fun)

def run_one(name,cov,data,evaluate,out,nwalkers,nsteps,burnin,seed):
    corr=correction(900,data.size,3); precision,_=base.precision_from_cov(cov,corr['hartlap'])
    start,chi2=map_fit(evaluate,data,cov); print('MAP',name,start.tolist(),chi2,flush=True)
    def logp(t):
        t=np.asarray(t)
        if np.any(t<LOWER) or np.any(t>UPPER): return -np.inf
        d=data-evaluate(t); return -.5*float(d@precision@d)
    rng=np.random.default_rng(seed); init=start[None,:]+rng.normal(size=(nwalkers,3))*INIT_SCALE[None,:]
    init=np.clip(init,LOWER+1e-7,UPPER-1e-7)
    sampler=emcee.EnsembleSampler(nwalkers,3,logp); print('CHAIN_START',name,flush=True)
    sampler.run_mcmc(init,nsteps,progress=False)
    chain=np.asarray(sampler.get_chain(discard=burnin),dtype='f8'); lp=np.asarray(sampler.get_log_prob(discard=burnin),dtype='f8')
    np.savez_compressed(out/f'chain_{name}_joint_fixed_sn0.npz',chain=chain,logp=lp,data=data)
    flat=chain.reshape(-1,3); q=np.percentile(flat,[16,50,84],axis=0); fac=corr['percival_sigma_factor']
    post={}
    for i,n in enumerate(NAMES):
        post[n]={'q16_raw':float(q[0,i]),'q50':float(q[1,i]),'q84_raw':float(q[2,i]),'q16':float(q[1,i]+fac*(q[0,i]-q[1,i])),'q84':float(q[1,i]+fac*(q[2,i]-q[1,i])),'sigma68':float(.5*fac*(q[2,i]-q[0,i]))}
    result={'covariance':name,'names':list(NAMES),'posterior':post,'map_theta':start.tolist(),'map_chi2':chi2,'nwalkers':nwalkers,'nsteps_postburn':chain.shape[0],'burnin':burnin,'hartlap_percival':corr,'acceptance_fraction_mean':float(np.mean(sampler.acceptance_fraction))}
    (out/f'summary_{name}_joint_fixed_sn0.json').write_text(json.dumps(result,indent=2,sort_keys=True)); print('CHAIN_DONE',name,json.dumps(post,sort_keys=True),flush=True)
    return result

def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--nwalkers',type=int,default=64); ap.add_argument('--nsteps',type=int,default=30000); ap.add_argument('--burnin',type=int,default=5000); ap.add_argument('--out',type=Path,default=OUT); a=ap.parse_args(); a.out.mkdir(parents=True,exist_ok=True)
    pm,xm,dp,dx,_,_=base.build_models_and_data(); data=np.r_[dp,dx]; evaluate=make_models(pm,xm)
    with np.load(SHARED/'ezmock900_covariance_and_stack.npz') as d: ce=np.asarray(d['covariance'],dtype='f8')
    with np.load(base.ANALYTIC_COV) as d: cj=np.asarray(d['rsd_joint'],dtype='f8')
    results={'ezmock900':run_one('ezmock900',ce,data,evaluate,a.out,a.nwalkers,a.nsteps,a.burnin,20260921),'jaxpower':run_one('jaxpower',cj,data,evaluate,a.out,a.nwalkers,a.nsteps,a.burnin,20260922)}
    (a.out/'comparison_summary_fixed_sn0.json').write_text(json.dumps(results,indent=2,sort_keys=True))

if __name__=='__main__': main()
