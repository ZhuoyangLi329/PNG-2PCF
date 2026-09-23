#!/usr/bin/env python3
"""P02-only MCMC with the P-side stochastic parameter fixed to sn0=0."""
from __future__ import annotations
import json
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
OUT=PARENT/'mcmc_preliminary_900_fixed_sn0_verified'
NAMES=('fNL','b1','sigma_s')
LOWER,UPPER=base.BOUNDS_LO[:3],base.BOUNDS_HI[:3]

def correction(ns,nb,npfit):
    hartlap=(ns-nb-2.)/(ns-1.); A=2./((ns-nb-1.)*(ns-nb-4.)); B=(ns-nb-2.)/((ns-nb-1.)*(ns-nb-4.)); m1=(1.+B*(nb-npfit))/(1.+A+B*(npfit+1.))
    return {'nmock':float(ns),'ndata':float(nb),'nparams':float(npfit),'hartlap':hartlap,'percival_m1':m1,'percival_sigma_factor':float(np.sqrt(m1))}

def main():
    pmodel,_,data_p,_,_,_=base.build_models_and_data()
    def evaluate(t):
        a=pmodel.evaluate(np.r_[np.asarray(t,dtype='f8'),0.]); return np.r_[a[:13],a[13:26][base.P2_KEEP]]
    with np.load(SHARED/'ezmock900_covariance_and_stack.npz') as d: cov_e=np.asarray(d['covariance'][:22,:22],dtype='f8')
    with np.load(base.ANALYTIC_COV) as d: cov_j=np.asarray(d['rsd_joint'][:22,:22],dtype='f8')
    starts=[s[:3] for s in base.OPTIMIZER_STARTS]+[np.array([0.,2.4,1.])]
    def run(name,cov,seed):
        corr=correction(900,22,3) if name=='ezmock900' else correction(0,22,3); corr['hartlap']=corr['hartlap'] if name=='ezmock900' else 1.; corr['percival_m1']=corr['percival_m1'] if name=='ezmock900' else 1.; corr['percival_sigma_factor']=float(np.sqrt(corr['percival_m1']))
        cov=(cov+cov.T)/2.; scale=np.sqrt(np.diag(cov)); corrmat=cov/np.outer(scale,scale); chol=np.linalg.cholesky(corrmat)
        def residual(t): return solve_triangular(chol,(data_p-evaluate(t))/scale,lower=True)
        sols=[least_squares(residual,np.clip(s,LOWER+1e-7,UPPER-1e-7),bounds=(LOWER,UPPER),x_scale='jac',max_nfev=3000,xtol=1e-11,ftol=1e-11,gtol=1e-11) for s in starts]
        best=min(sols,key=lambda s:float(s.fun@s.fun)); theta=best.x; chi2=float(best.fun@best.fun); invcorr=np.linalg.inv(corrmat); precision=corr['hartlap']*invcorr/np.outer(scale,scale)
        def logp(t):
            if np.any(t<LOWER) or np.any(t>UPPER): return -np.inf
            d=data_p-evaluate(t); return -.5*float(d@precision@d)
        rng=np.random.default_rng(seed); init=theta[None,:]+rng.normal(size=(64,3))*np.array([4.,.015,.12])[None,:]; init=np.clip(init,LOWER+1e-7,UPPER-1e-7)
        sampler=emcee.EnsembleSampler(64,3,logp); sampler.run_mcmc(init,30000,progress=False)
        chain=np.asarray(sampler.get_chain(discard=5000),dtype='f8'); lp=np.asarray(sampler.get_log_prob(discard=5000),dtype='f8'); flat=chain.reshape(-1,3)
        raw=summarize_chain(chain,lp,NAMES); raw['posterior_raw']=json.loads(json.dumps(raw['posterior'])); result=base.corrected_summary(raw,corr['percival_m1']); result.update({'covariance':name,'variant':'p02_fixed_sn0','names':NAMES,'fixed_parameters':{'sn0':0.},'map_theta':theta.tolist(),'map_chi2':chi2,'hartlap_percival':corr,'nwalkers':64,'nsteps':30000,'burnin':5000,'nsteps_postburn':len(chain),'acceptance_fraction_mean':float(sampler.acceptance_fraction.mean()),'data_contract':'P0 13 bins + P2 9 bins; sn0 fixed to zero'})
        np.savez_compressed(OUT/f'chain_{name}_p02_fixed_sn0.npz',chain=chain,logp=lp,data=data_p); (OUT/f'summary_{name}_p02_fixed_sn0.json').write_text(json.dumps(result,indent=2)+'\n'); print(name,json.dumps({'map':theta.tolist(),'chi2':chi2,'posterior':result['posterior'],'gates':result['gates']}),flush=True)
    run('ezmock900',cov_e,20260923); run('jaxpower',cov_j,20260924)

if __name__=='__main__': main()
