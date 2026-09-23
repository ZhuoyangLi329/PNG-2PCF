#!/usr/bin/env python3
"""EZmock900 l=0-only P0+xi0 MCMC with free sn0 and shared sigma_s."""
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
SHARED=PARENT/'mcmc_preliminary_900'; OUT=PARENT/'mcmc_preliminary_900_l0only_verified'
LOWER=np.array([-500.,.5,0.,-1.]); UPPER=np.array([500.,5.,30.,1.]); INIT=np.array([4.,.015,.12,.025])

def corrections(ns,nb,npfit):
    h=(ns-nb-2.)/(ns-1.); A=2./((ns-nb-1.)*(ns-nb-4.)); B=(ns-nb-2.)/((ns-nb-1.)*(ns-nb-4.)); m=(1.+B*(nb-npfit))/(1.+A+B*(npfit+1.)); return {'nmock':float(ns),'ndata':float(nb),'nparams':float(npfit),'hartlap':h,'percival_m1':m,'percival_sigma_factor':float(np.sqrt(m))}

def get_models():
    pm,xm,dp,dx,_,_=base.build_models_and_data(); p0full=dp[:13]; x0full=dx[:base.N_XI]; data=np.r_[p0full,x0full]
    def p0(t):
        a=pm.evaluate(np.asarray(t,dtype='f8')); return a[:13]
    def x0(t):
        a=xm.evaluate(np.asarray(t,dtype='f8')[:3],model='formal_gic',window_key='mean'); return a[0][base.XI_MASK]
    def joint(t): return np.r_[p0(t),x0(t)]
    return p0,x0,joint,data

def run(name,ev,data,cov,npar,seed):
    meta=corrections(900,len(data),npar); cov=(cov+cov.T)/2.; scale=np.sqrt(np.diag(cov)); cor=cov/np.outer(scale,scale); chol=np.linalg.cholesky(cor); metric=lambda d: solve_triangular(chol,d/scale,lower=True,check_finite=False)
    if npar==3: starts=[s[:3] for s in base.OPTIMIZER_STARTS]; lo,hi=LOWER[:3],UPPER[:3]; names=('fNL','b1','sigma_s')
    else: starts=[s for s in base.OPTIMIZER_STARTS]; lo,hi=LOWER,UPPER; names=('fNL','b1','sigma_s','sn0')
    sols=[least_squares(lambda t:metric(data-ev(t)),np.clip(s,lo+1e-7,hi-1e-7),bounds=(lo,hi),x_scale='jac',max_nfev=3000,xtol=1e-11,ftol=1e-11,gtol=1e-11) for s in starts]; best=min(sols,key=lambda s:float(s.fun@s.fun)); theta=best.x; chi2=float(best.fun@best.fun); precision=meta['hartlap']*np.linalg.inv(cor)/np.outer(scale,scale)
    def logp(t):
        if np.any(t<lo) or np.any(t>hi): return -np.inf
        d=data-ev(t); return -.5*float(d@precision@d)
    nwalkers=64; nsteps=30000; burnin=5000; rng=np.random.default_rng(seed); initial=theta[None,:]+rng.normal(size=(nwalkers,npar))*INIT[:npar][None,:]; initial=np.clip(initial,lo+1e-7,hi-1e-7); sampler=emcee.EnsembleSampler(nwalkers,npar,logp); tic=time.time()
    for i,_ in enumerate(sampler.sample(initial,iterations=nsteps),1):
        if i%5000==0: print('PROGRESS',name,i,round(time.time()-tic),flush=True)
    chain=np.asarray(sampler.get_chain(discard=burnin),dtype='f8'); lp=np.asarray(sampler.get_log_prob(discard=burnin),dtype='f8'); raw=summarize_chain(chain,lp,names); raw['posterior_raw']=json.loads(json.dumps(raw['posterior'])); result=base.corrected_summary(raw,meta['percival_m1']); result.update({'covariance':'ezmock900','variant':name,'multipoles':['ell0'],'excluded_multipoles':['ell2'],'names':names,'map_theta':theta.tolist(),'map_fit':{'chi2_single':chi2,'all_start_chi2':[float(s.fun@s.fun) for s in sols]},'nwalkers':nwalkers,'nsteps':nsteps,'burnin':burnin,'nsteps_postburn':len(chain),'acceptance_fraction_mean':float(sampler.acceptance_fraction.mean()),'hartlap_percival':meta,'data_contract':'P0 13 bins + xi0 26 BAO-masked bins; full P0-xi0 cross covariance; sn0 free'})
    OUT.mkdir(parents=True,exist_ok=True); np.savez_compressed(OUT/f'chain_ezmock900_{name}.npz',chain=chain,logp=lp,data=data); (OUT/f'summary_ezmock900_{name}.json').write_text(json.dumps(result,indent=2)+'\n'); print('RESULT',name,json.dumps({'map':theta.tolist(),'chi2':chi2,'posterior':result['posterior'],'gates':result['gates']},sort_keys=True),flush=True); return result

def main():
    p0,x0,joint,data=get_models()
    with np.load(SHARED/'ezmock900_covariance_and_stack.npz') as d: c=d['covariance']; idx=np.r_[np.arange(13),np.arange(22,48)]; cov=c[np.ix_(idx,idx)]
    results={'p0_only':run('p0_only',p0,data[:13],cov[:13,:13],4,20260927),'xi0_only':run('xi0_only',x0,data[13:],cov[13:,13:],3,20260928),'joint_p0_xi0':run('joint_p0_xi0',joint,data,cov,4,20260929)}
    (OUT/'comparison_summary_ezmock900_l0only.json').write_text(json.dumps(results,indent=2)+'\n')
if __name__=='__main__': main()
