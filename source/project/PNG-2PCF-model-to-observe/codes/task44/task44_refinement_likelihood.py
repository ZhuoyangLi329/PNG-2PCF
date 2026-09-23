#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""25×重拟合公共数值工具：P0快速模型、多起点MAP及独立ensemble诊断。

大纲：1. 精确保留P0对b/PNG幅度/sn0的代数关系，只在sigma轴插值并验证；
2. MAP用sigma²坐标避免零点导数退化，报告参数仍为sigma且保持原先验；
3. emcee在原参数坐标采样，逐ensemble给出时间批次MCSE；
4. Rhat只比较不同ensemble中的指定walker，绝不把同ensemble walkers
   充作独立链；未通过跨ensemble门时禁止混合成一个最终posterior。
"""
from __future__ import annotations
import os,sys,time
from pathlib import Path
for key in ['OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS','NUMEXPR_NUM_THREADS']:os.environ[key]='1'
import numpy as np
import emcee
from scipy.interpolate import CubicSpline
from scipy.optimize import least_squares
from scipy.stats import rankdata,norm
R=Path('/pscratch/sd/l/lzy/PNG-2PCF-model-to-observe');sys.path.insert(0,str(R/'codes/task44'))
import task44_fit_pk_lrg2 as pk
from task43_theory_template import build_template_arrays,load_task41
from task44_scalar_forward_model import coefficients,sigma_grid
from task44_fit_validation import file_digest
from task44_diagnose_pk_xi_transfer import read,save,whiten_matrix
from task44_measure_random75x_xi import atomic_savez
O=R/'outputs/task44_outputs/method_refinement_20260905'


class FastPk:
    """已验证的P0批量模型；sn0窗口响应精确线性，sigma不外推。"""
    def __init__(self,path):
        self.path=Path(path)
        with np.load(path) as z:
            self.sigma=z['sigma_grid'];self.p=float(z['p_fixed']);self.f=float(z['f_growth']);self.sn=z['sn0_template']
            self.spline=CubicSpline(self.sigma,z['table'],axis=0,extrapolate=False)
    def __call__(self,theta):
        scalar=np.asarray(theta).ndim==1;t=np.atleast_2d(theta);c=coefficients(t,self.p,self.f)
        out=np.einsum('nbs,nb->ns',self.spline(t[:,2]),c)+t[:,3,None]*self.sn[None,:]
        if not np.all(np.isfinite(out)):raise ValueError('sigma outside validated P0 cache')
        return out[0] if scalar else out


def build_pk_cache(sample):
    """使用实际25×窗口和原P0角向求积，验证插值模型后写入缓存。"""
    root=O/'model_cache'/sample;root.mkdir(parents=True,exist_ok=True);out=root/'pk_forward_cache.npz';js=out.with_suffix('.json')
    payload=O/'fit_inputs'/sample/'pk_fit_payload.npz';entry=read(O/'inventory.json')['samples'][sample]
    original=read(entry['historical_results']['pk']['provenance']['summary']);cfg=original['config']
    if js.exists():
        old=read(js);assert old['payload_sha256']==file_digest(payload) and old['script_sha256']==file_digest(__file__) and old['output_sha256']==file_digest(out)
        return out
    data=pk.FitData(payload);task41=load_task41()
    k=np.logspace(np.log10(min(1e-5,np.min(data.theory_k[data.theory_k>0])*.5)),np.log10(20.),20000)
    template,_=build_template_arrays(task41,k,z=data.zeff,cosmology=cfg['cosmology'])
    cache=pk.RSDTheoryCache(data,template,nmu=cfg['nmu'],fog_model=cfg['fog_model'])
    good=cache.positive.copy()
    if cfg['window_theory_kmin'] is not None:good &= cache.k>=cfg['window_theory_kmin']
    ids=np.flatnonzero(good);ells=cache.ell[good];alpha=cache.alpha[good];matter=cache.pk_dd[good]
    def basis(sigma):
        a=(cache.k[good,None]*cache.mu[None,:]*sigma)**2
        damp=(1+.5*a)**-2 if cfg['fog_model']=='lorentzian' else np.exp(-a)
        d=np.zeros((3,len(ids)))
        for ell in cache.ells:
            select=ells==ell
            for power in range(3):
                integral=.5*(2*ell+1)*np.sum(damp*cache.wmu[None,:]*cache.legendre_by_ell[ell][None,:]*cache.mu[None,:]**(2*power),axis=1)
                d[power,select]=integral[select]
        values=np.zeros((6,len(cache.k)));values[:,ids]=matter[None,:]*np.array([d[0],alpha*d[0],alpha**2*d[0],d[1],alpha*d[1],d[2]])
        return values@data.window_matrix.T
    snvec=np.zeros(len(cache.k));snvec[good&(cache.ell==0)]=cfg['sn0_scale'];snvec=data.window_matrix@snvec
    def reference(theta):
        return pk.model_pk(data,cache,dict(zip(pk.PARAM_NAMES,theta)),p_fixed=cfg['p_fixed'],sn0_scale=cfg['sn0_scale'],fog_model=cfg['fog_model'],window_theory_kmin=cfg['window_theory_kmin'])
    rng=np.random.default_rng(20260912);bounds=np.array([cfg['priors'][key] for key in pk.PARAM_NAMES]);tests=rng.uniform(bounds[:,0],bounds[:,1],size=(100,4))
    map_point=np.array([original['maximum_posterior_sample']['point'][key] for key in pk.PARAM_NAMES]);tests=np.vstack([map_point,tests])
    truth=np.array([reference(x) for x in tests]);W=whiten_matrix(data.covariance);attempts=[]
    for refinement in [1,2]:
        grid=sigma_grid(refinement);table=np.array([basis(sigma) for sigma in grid])
        atomic_savez(out,sigma_grid=grid,table=table,sn0_template=snvec,p_fixed=np.array(cfg['p_fixed']),f_growth=np.array(cache.f_growth))
        diff=FastPk(out)(tests)-truth
        errors={'sigma_nodes':len(grid),'max_abs':float(np.max(abs(diff))),'max_covariance_norm':float(np.max(np.linalg.norm(diff@W.T,axis=1)))}
        attempts.append(errors)
        if errors['max_covariance_norm']<.05:break
    else:raise RuntimeError(f'P0 interpolation budget failed {attempts}')
    save(js,{'status':'numerically_validated','sample':sample,'payload_sha256':file_digest(payload),'script_sha256':file_digest(__file__),
             'output_sha256':file_digest(out),'validation':attempts,'template_cosmology':cfg['cosmology'],'model_config':cfg})
    print('[P0 cache]',sample,errors,flush=True);return out


def map_fit(model,data,covariance,bounds,baseline):
    """多起点有界MAP，内部sigma²数值坐标不改变实际sigma先验。"""
    bounds=np.asarray(bounds);lo=bounds[:,0].copy();hi=bounds[:,1].copy();lo[2]**=2;hi[2]**=2
    W=whiten_matrix(covariance);baseline=np.asarray(baseline);starts=[baseline.copy()]
    for fnl in [-450.,-150.,0.,150.,450.]:
        for sigma in [0.,8.,25.]:
            x=baseline.copy();x[0]=fnl;x[2]=sigma;starts.append(x)
    for b in [model.p-.08,model.p+.08]:
        for fnl in [-350.,350.]:
            x=baseline.copy();x[0]=fnl;x[1]=b;starts.append(x)
    def physical(u):
        theta=np.array(u,copy=True);theta[2]=np.sqrt(max(theta[2],0));return theta
    def residual(u):return W@(data-model(physical(u)))
    candidates=[]
    for initial in starts:
        u=initial.copy();u[2]**=2;u=np.clip(u,lo+1e-10,hi-1e-10)
        result=least_squares(residual,u,bounds=(lo,hi),x_scale='jac',max_nfev=1500,ftol=1e-11,xtol=1e-11,gtol=1e-10)
        candidates.append({'theta':physical(result.x),'chi2':float(result.fun@result.fun),'success':bool(result.success),'nfev':int(result.nfev)})
    good=[x for x in candidates if x['success']]
    if not good:raise RuntimeError('no multi-start optimizer converged')
    best=min(good,key=lambda x:x['chi2']);theta=best['theta'];prediction=model(theta)
    return {**best,'prediction':prediction,'candidates':candidates,'dof':int(len(data)-len(theta)),
            'at_prior_bound':[i for i in range(len(theta)) if min(theta[i]-bounds[i,0],bounds[i,1]-theta[i])<1e-4],
            'optimizer_coordinates':'fnl,b1,sigma_s_squared[,sn0]; exported/prior coordinates retain sigma_s'}


def chain_diagnostics(chain,names):
    """用时间批次保留同ensemble内walker相关性，估计参数分位点的MCSE。"""
    nstep,nwalker,ndim=chain.shape;flat=chain.reshape(-1,ndim);q=np.quantile(flat,[.16,.5,.84],axis=0);sigma=(q[2]-q[0])/2
    tau=np.asarray(emcee.autocorr.integrated_time(chain,quiet=True,tol=0))
    mean_series=chain.mean(axis=1)
    tau_mean=np.array([float(np.asarray(emcee.autocorr.integrated_time(mean_series[:,i],quiet=True,tol=0)).ravel()[0]) for i in range(ndim)])
    safe_tau=max(1.,float(np.max(tau)),float(np.max(tau_mean)))
    nblocks=min(30,int(nstep//max(1,int(np.ceil(10*safe_tau)))))
    mcse=np.full_like(q,np.inf)
    if nblocks>=4:
        length=nstep//nblocks
        block_q=np.array([np.quantile(chain[i*length:(i+1)*length].reshape(-1,ndim),[.16,.5,.84],axis=0) for i in range(nblocks)])
        mcse=np.std(block_q,axis=0,ddof=1)/np.sqrt(nblocks)
    half=nstep//2;shift=abs(np.median(chain[:half].reshape(-1,ndim),axis=0)-np.median(chain[-half:].reshape(-1,ndim),axis=0))/sigma
    posterior={name:{'q16':q[0,i],'q50':q[1,i],'q84':q[2,i],'sigma68':sigma[i]} for i,name in enumerate(names)}
    gates={'finite_chain':bool(np.isfinite(chain).all()),'positive_finite_tau':bool(np.all(np.isfinite(tau)) and np.all(tau>0)),
           'postburn_length_above_50tau':bool(np.min(nstep/tau)>50),'at_least_10_time_batches':nblocks>=10,
           'quantile_mcse_below_0p03sigma':bool(np.max(mcse/sigma[None,:])<.03),'half_median_shift_below_0p1sigma':bool(np.max(shift)<.1)}
    return {'posterior':posterior,'quantiles':q,'quantile_mcse':mcse,'tau':tau,'tau_ensemble_mean':tau_mean,
            'time_batches':nblocks,'half_shift_sigma':shift,'gates':gates,'all_internal_gates':bool(all(gates.values()))}


def run_ensemble(model,data,covariance,bounds,start,*,nwalkers=48,nsteps=12000,burnin=2000,seed=1):
    """独立随机种子运行一个ensemble；不会把另一个ensemble状态作为输入。"""
    bounds=np.asarray(bounds);W=whiten_matrix(covariance);rng=np.random.default_rng(seed);ndim=len(bounds)
    def log_probability(theta):
        t=np.atleast_2d(theta);good=np.all((t>=bounds[:,0])&(t<=bounds[:,1]),axis=1)&np.all(np.isfinite(t),axis=1)
        result=np.full(len(t),-np.inf)
        if np.any(good):
            r=(data[None,:]-model(t[good]))@W.T;result[good]=-.5*np.sum(r*r,axis=1)
        return result
    scale=np.array([5.,.015,.2]+([.05] if ndim==4 else []))
    initial=np.array(start)[None,:]+rng.normal(size=(nwalkers,ndim))*scale[None,:]
    initial=np.clip(initial,bounds[:,0]+1e-8,bounds[:,1]-1e-8)
    np.random.seed(seed);sampler=emcee.EnsembleSampler(nwalkers,ndim,log_probability,vectorize=True)
    t=time.time();sampler.run_mcmc(initial,nsteps,progress=False)
    chain=sampler.get_chain(discard=burnin);logp=sampler.get_log_prob(discard=burnin)
    names=['fnl_loc','b1','sigma_s']+(['sn0'] if ndim==4 else [])
    summary=chain_diagnostics(chain,names);acceptance=float(np.mean(sampler.acceptance_fraction))
    summary.update(seed=seed,nwalkers=nwalkers,nsteps=nsteps,burnin=burnin,acceptance_fraction=acceptance,elapsed_seconds=time.time()-t)
    summary['gates']['acceptance_between_0p1_0p8']=.1<acceptance<.8
    summary['all_internal_gates']=bool(all(summary['gates'].values()))
    return summary,chain,logp


def _rhat(values):
    """对真正独立链的已分裂/变换数组(chain,step)计算普通PSR。"""
    n=values.shape[1];W=np.mean(np.var(values,axis=1,ddof=1));B=n*np.var(np.mean(values,axis=1),ddof=1)
    return float(np.sqrt(((n-1)/n*W+B/n)/W)) if W>0 else float('inf')


def independent_ensemble_check(chains,summaries):
    """每次仅取不同ensemble中的同一个walker作独立链，计算rank/folded Rhat。

    对4个walker位置分别检查后取最差值；这些位置不会合并为额外独立链。
    同时用保留ensemble相关性的批次MCSE比较分位点。
    """
    assert len(chains)>=2
    length=min(len(c) for c in chains);nw=chains[0].shape[1];ndim=chains[0].shape[2];h=length//2
    checks=[]
    for walker in sorted(set([0,nw//3,2*nw//3,nw-1])):
        selected=np.array([c[-2*h:,walker,:] for c in chains])
        split=np.concatenate([selected[:,:h],selected[:,h:]],axis=0)
        vals=[]
        for dimension in range(ndim):
            x=split[:,:,dimension];total=x.size
            transformed=norm.ppf((rankdata(x.ravel()).reshape(x.shape)-.375)/(total+.25))
            folded=abs(x-np.median(x));folded=norm.ppf((rankdata(folded.ravel()).reshape(x.shape)-.375)/(total+.25))
            vals.append(max(_rhat(transformed),_rhat(folded)))
        checks.append({'walker_index':walker,'rank_folded_split_rhat':vals})
    maxr=float(np.max([x['rank_folded_split_rhat'] for x in checks]));qdiff=abs(summaries[0]['quantiles']-summaries[1]['quantiles'])
    error=np.sqrt(summaries[0]['quantile_mcse']**2+summaries[1]['quantile_mcse']**2)
    sig=.25*((summaries[0]['quantiles'][2]-summaries[0]['quantiles'][0])+(summaries[1]['quantiles'][2]-summaries[1]['quantiles'][0]))
    gates={'independent_walker_rhat_below_1p01':maxr<1.01,'quantile_difference_below_4p5_combined_mcse':bool(np.max(qdiff/error)<4.5),
           'quantile_difference_below_0p1sigma':bool(np.max(qdiff/sig[None,:])<.1),'each_ensemble_internal_gates':all(s['all_internal_gates'] for s in summaries)}
    return {'method':'designated walker from each independently seeded ensemble; separate walker checks, never pooled as independent chains',
            'walker_checks':checks,'maximum_rhat':maxr,'quantile_difference':qdiff,'quantile_difference_in_combined_mcse':qdiff/error,
            'quantile_difference_in_sigma':qdiff/sig[None,:],'gates':gates,'pass':bool(all(gates.values()))}
