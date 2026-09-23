#!/usr/bin/env python3
"""Audited joint MCMC with separate sigma_s(P) and sigma_s(xi), sn0=0."""
from __future__ import annotations
import json, time
from pathlib import Path
import emcee
import numpy as np
from scipy.linalg import solve_triangular
from scipy.optimize import least_squares
import task43_rsd_ezmock281_mcmc_compare as base
from task43_joint_rsd_pkxi_fit import summarize_chain

ROOT=base.PROJECT_ROOT
PARENT=ROOT/'outputs/task43_outputs/ezmock_rsd_lightcone_covariance_z0p4_0p8_x1000_common50_nzmatch_b0.30_z0.60'
SHARED=PARENT/'mcmc_preliminary_900'
OUT=PARENT/'mcmc_preliminary_900_split_sigma_fixed_sn0_verified'
NAMES=('fNL','b1','sigma_s_P','sigma_s_xi')
LOWER=np.array([-500.,.5,0.,0.]); UPPER=np.array([500.,5.,30.,30.]); INIT=np.array([4.,.015,.12,.12])

def corr(ns,nb,npfit):
    if ns==0: return {'nmock':0.,'ndata':float(nb),'nparams':float(npfit),'hartlap':1.,'percival_m1':1.,'percival_sigma_factor':1.}
    h=(ns-nb-2.)/(ns-1.); A=2./((ns-nb-1.)*(ns-nb-4.)); B=(ns-nb-2.)/((ns-nb-1.)*(ns-nb-4.)); m=(1.+B*(nb-npfit))/(1.+A+B*(npfit+1.)); return {'nmock':float(ns),'ndata':float(nb),'nparams':float(npfit),'hartlap':h,'percival_m1':m,'percival_sigma_factor':float(np.sqrt(m))}

def models():
    pm,xm,dp,dx,_,_=base.build_models_and_data(); data=np.r_[dp,dx]
    def ev(t):
        f,b,sp,sx=t; p=pm.evaluate(np.array([f,b,sp,0.])); x=xm.evaluate(np.array([f,b,sx]),model='formal_gic',window_key='mean'); return np.r_[p[:13],p[13:26][base.P2_KEEP],x[0][base.XI_MASK],x[2][base.XI_MASK]]
    return ev,data

def run(name,cov,data,ev,seed,nsteps=30000,burnin=5000,nwalkers=64):
    meta=corr(900 if name=='ezmock900' else 0,data.size,4); cov=(cov+cov.T)/2.; scale=np.sqrt(np.diag(cov)); cor=cov/np.outer(scale,scale); chol=np.linalg.cholesky(cor); min_e=float(np.linalg.eigvalsh(cor).min())
    metric=lambda d: solve_triangular(chol,d/scale,lower=True,check_finite=False)
    starts=[np.array([0.,2.5,2.5,5.5]),np.array([-20.,2.4,2.5,5.5]),np.array([20.,2.6,8.,8.])]
    sols=[least_squares(lambda t:metric(data-ev(t)),np.clip(s,LOWER+1e-7,UPPER-1e-7),bounds=(LOWER,UPPER),x_scale='jac',max_nfev=3000,xtol=1e-11,ftol=1e-11,gtol=1e-11) for s in starts]; best=min(sols,key=lambda s:float(s.fun@s.fun)); theta=best.x; chi2=float(best.fun@best.fun); precision=meta['hartlap']*np.linalg.inv(cor)/np.outer(scale,scale)
    def logp(t):
        if np.any(t<LOWER) or np.any(t>UPPER): return -np.inf
        d=data-ev(t); return -.5*float(d@precision@d)
    rng=np.random.default_rng(seed); initial=theta[None,:]+rng.normal(size=(nwalkers,4))*INIT[None,:]; initial=np.clip(initial,LOWER+1e-7,UPPER-1e-7); sampler=emcee.EnsembleSampler(nwalkers,4,logp); tic=time.time()
    for i,_ in enumerate(sampler.sample(initial,iterations=nsteps),1):
        if i%5000==0: print('PROGRESS',name,i,round(time.time()-tic),flush=True)
    chain=np.asarray(sampler.get_chain(discard=burnin),dtype='f8'); lp=np.asarray(sampler.get_log_prob(discard=burnin),dtype='f8'); flat=chain.reshape(-1,4); raw=summarize_chain(chain,lp,NAMES); raw['posterior_raw']=json.loads(json.dumps(raw['posterior'])); result=base.corrected_summary(raw,meta['percival_m1']); result.update({'covariance':name,'variant':'joint_split_sigma_fixed_sn0','names':NAMES,'fixed_parameters':{'sn0':0.},'shared_parameters':['fNL','b1'],'separate_parameters':['sigma_s_P','sigma_s_xi'],'map_theta':theta.tolist(),'map_fit':{'chi2_single':chi2,'all_start_chi2':[float(s.fun@s.fun) for s in sols]},'nwalkers':nwalkers,'nsteps':nsteps,'burnin':burnin,'nsteps_postburn':len(chain),'acceptance_fraction_mean':float(sampler.acceptance_fraction.mean()),'hartlap_percival':meta,'covariance_min_correlation_eigenvalue':min_e,'likelihood_metric':'normalized covariance Cholesky, no eigen clipping','data_contract':'Abacus x25 mean; C_single; P0 13 P2 9 xi0 26 xi2 26; full P-xi cross'})
    OUT.mkdir(parents=True,exist_ok=True); np.savez_compressed(OUT/f'chain_{name}_joint_split_sigma_fixed_sn0.npz',chain=chain,logp=lp,data=data); (OUT/f'summary_{name}_joint_split_sigma_fixed_sn0.json').write_text(json.dumps(result,indent=2)+'\n'); print('RESULT',name,json.dumps({'map':theta.tolist(),'chi2':chi2,'posterior':result['posterior'],'gates':result['gates'],'split_rhat':result['split_rhat']},sort_keys=True),flush=True); return result

def main():
    ev,data=models()
    with np.load(SHARED/'ezmock900_covariance_and_stack.npz') as d: ce=d['covariance']
    with np.load(base.ANALYTIC_COV) as d: cj=d['rsd_joint']
    results={'ezmock900':run('ezmock900',ce,data,ev,20260925),'jaxpower':run('jaxpower',cj,data,ev,20260926)}; (OUT/'comparison_summary_split_sigma_fixed_sn0.json').write_text(json.dumps(results,indent=2)+'\n')
if __name__=='__main__': main()
