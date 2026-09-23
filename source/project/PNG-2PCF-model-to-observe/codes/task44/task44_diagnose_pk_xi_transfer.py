#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Task44 P0/xi0 参数交叉预测诊断（不更新正式约束）。

大纲：
1. 从 active 审计精确定位 75x 全样本与 20x 六个子样本，校验输入摘要哈希。
2. 复用正式 P0/xi0 模板、核、窗口、尺度和 covariance，重现已保存 MAP。
3. 分别交换共同参数、仅固定对方 fNL 并 profile 其余参数。
4. 用保留单-bin 方差的 diagonal covariance 作归因诊断；不替换正式 covariance。
5. 输出逐样本结果及一个总 manifest；不跑 MCMC、不把 Δchi2 当独立实验张力。
资源：串行单 CPU，禁止安装包与重新生成 catalog/window/covariance。
"""
from __future__ import annotations
import os
for _key in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS','NUMEXPR_NUM_THREADS'):
    os.environ[_key] = '1'
os.environ['JAX_PLATFORMS'] = 'cpu'
os.environ['CUDA_VISIBLE_DEVICES'] = ''
os.environ['PYTHONDONTWRITEBYTECODE'] = '1'
import argparse
import hashlib
import json
import sys
import time
from pathlib import Path
import numpy as np
from scipy.optimize import least_squares

ROOT = Path('/pscratch/sd/l/lzy/PNG-2PCF-model-to-observe')
sys.path.insert(0, str(ROOT/'codes/task44'))
import task44_fit_pk_lrg2 as pkmod
import task44_fit_lrg2_rsd as ximod
from task44_rsd_theory import build_theory_context, mu_moments
from task43_theory_template import build_template_arrays, load_task41

NAMES = ['fnl_loc','b1','sigma_s']
OUT = ROOT/'outputs/task44_outputs/pk_xi_transfer_diagnostic_20260905'

def read(path):
    """输入 JSON 路径，返回字典。"""
    return json.loads(Path(path).read_text())

def digest(path):
    """流式计算 SHA256；不会将大 chain 全部加载到内存。"""
    h=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024*1024), b''): h.update(chunk)
    return h.hexdigest()

def jsonable(x):
    """把 NumPy 数组/标量递归转为 JSON，保留可复查的数值向量。"""
    if isinstance(x,dict): return {k:jsonable(v) for k,v in x.items()}
    if isinstance(x,(tuple,list)):return [jsonable(v) for v in x]
    if isinstance(x,np.ndarray):return x.tolist()
    if isinstance(x,np.generic):return x.item()
    return x

def save(path,payload):
    """原子写入本诊断目录内 JSON；不覆盖任何正式输入。"""
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    tmp=path.with_suffix('.tmp');tmp.write_text(json.dumps(jsonable(payload),indent=2,allow_nan=False)+'\n');tmp.replace(path)

def get_cases():
    """读取两个 active 总审计，返回八套观测对及其源摘要。"""
    paths=[ROOT/'outputs/task44_outputs/random75x_lightcone/summary/main/task44_lrg_qso_lightcone_random75x_M75_replot_audit.json',ROOT/'outputs/task44_outputs/random20x_subsamples/summary/task44_random20x_subsamples_pk_xi_final_audit.json']
    a,b=map(read,paths)
    cases={name+'all':{'pk':v['P0'],'xi':v['xi0'],'campaign':'random75x','audit_path':str(paths[0])} for name,v in a['tracers'].items()}
    cases.update({name:{'pk':v['pk'],'xi':v['xi'],'campaign':'random20x','audit_path':str(paths[1])} for name,v in b['results'].items()})
    return cases

def whiten_matrix(cov):
    """返回对称 covariance 的逆平方根行算子，使 ||Wr||²=rᵀC⁻¹r。"""
    eig,vec=np.linalg.eigh((cov+cov.T)/2)
    if eig[0]<=0:raise ValueError('non-SPD covariance')
    return vec.T/np.sqrt(eig)[:,None]

def fit(model,data,cov,bounds,starts,fixed=None):
    """多起点有界最小二乘；fixed 用参数索引固定值，返回 profile/MAP 诊断。"""
    fixed={} if fixed is None else fixed
    lo,hi=map(np.asarray,bounds);ids=[i for i in range(lo.size) if i not in fixed]
    W=whiten_matrix(cov)
    def expand(x):
        theta=np.zeros(lo.size);theta[ids]=x
        for i,v in fixed.items():theta[i]=v
        return theta
    def res(x):return W@(data-model(expand(x)))
    candidates=[]
    for initial in starts:
        x=np.clip(np.asarray(initial)[ids],lo[ids]+1e-9,hi[ids]-1e-9)
        result=least_squares(res,x,bounds=(lo[ids],hi[ids]),x_scale='jac',ftol=1e-10,xtol=1e-10,gtol=1e-9,max_nfev=500)
        candidates.append((float(result.fun@result.fun),result))
    chi,result=min(candidates,key=lambda x:x[0]);theta=expand(result.x)
    return {'theta':theta,'chi2':chi,'success':bool(result.success),'nfev':int(result.nfev),'nfree':len(ids),'fixed':{str(k):v for k,v in fixed.items()},'at_bounds':[int(i) for i in ids if min(theta[i]-lo[i],hi[i]-theta[i])<1e-4],'starts_chi2':[x[0] for x in candidates]}

def metrics(model,theta,data,cov,coords):
    """评估预测残差；报告完整/对角 χ² 和逐 bin 标准化残差，不作联合张力解释。"""
    pred=model(theta);res=data-pred;W=whiten_matrix(cov);z=res/np.sqrt(np.diag(cov))
    corr=cov/np.sqrt(np.outer(np.diag(cov),np.diag(cov)))
    eig,U=np.linalg.eigh(corr); scores=(U.T@z)/np.sqrt(eig)
    order=np.argsort(scores**2)[::-1];top=order[:5]
    return {'theta':theta,'prediction':pred,'residual':res,'z_diag':z,'chi2':float(np.sum((W@res)**2)),'chi2_diag':float(z@z),'correlation_eigenvalues':eig,'top5_correlation_modes':top,'top5_chi2':scores[top]**2,'coords':coords}

def main():
    """顺序处理指定样本，保存自重现门及参数转移诊断，出错则停止以免误解结果。"""
    parser=argparse.ArgumentParser();parser.add_argument('--samples',default='lrgall,qsoall,lrg1,lrg2,lrg3,qso1,qso2,qso3');parser.add_argument('--force',action='store_true');args=parser.parse_args()
    t0=time.time();cases=get_cases();manifest={'task':'task44_pk_xi_parameter_transfer','status':'running','role':'diagnostic_only_not_joint_likelihood','source_script_sha256':digest(__file__),'cpu_affinity':sorted(os.sched_getaffinity(0)),'campaigns':{'random75x':'physical_single_only','random20x':'physical_single_plus_finite_random_20'},'samples':{}}
    if len(os.sched_getaffinity(0))>8:raise RuntimeError('CPU affinity must be <=8')
    task41=load_task41();context_base=None
    for name in args.samples.split(','):
        path=OUT/'samples'/f'{name}.json'
        if path.exists() and not args.force:
            existing=read(path)
            if existing.get('status')=='complete' and existing['script_sha256']==manifest['source_script_sha256']:
                manifest['samples'][name]={'path':str(path),'sha256':digest(path)};print('[reuse]',name,flush=True);continue
            raise FileExistsError(f'existing unvalidated diagnostic: {path}')
        print('[start]',name,flush=True);case=cases[name];ppr=case['pk']['provenance'];xpr=case['xi']['provenance']
        for prov in (ppr,xpr):
            if digest(prov['summary'])!=prov['summary_sha256']:raise RuntimeError('source summary hash changed')
        ps,xs=read(ppr['summary']),read(xpr['summary'])
        cfg=ps['config'];xcfg=xs['theory'];pfix=cfg['p_fixed']
        if pfix!=xs['p_fixed'] or cfg['cosmology']!=xcfg['cosmology']:raise ValueError('p/cosmology mismatch')
        data=pkmod.FitData(Path(ppr['payload']))
        if abs(data.zeff-xs['zeff'])>1e-10:raise ValueError('zeff mismatch')
        ktmp=np.logspace(np.log10(min(1e-5,np.min(data.theory_k[data.theory_k>0])*.5)),np.log10(20.0),20000)
        template,_=build_template_arrays(task41,ktmp,z=data.zeff,cosmology=cfg['cosmology'])
        cache=pkmod.RSDTheoryCache(data,template,nmu=cfg['nmu'],fog_model=cfg['fog_model'])
        def pk(theta):
            point=dict(zip(pkmod.PARAM_NAMES,theta))
            return pkmod.model_pk(data,cache,point,p_fixed=pfix,sn0_scale=cfg['sn0_scale'],fog_model=cfg['fog_model'],window_theory_kmin=cfg['window_theory_kmin'])
        chainpath=Path(xpr['chains'])
        with np.load(chainpath,allow_pickle=False) as f:
            s=f['s'];xdata=f['xi_vector'];xcov=f['covariance'];xpred=f['formal_gic_model_map']
        # 重用 gq/rebin 几何，避免为每个红移重复构建相同母盒网格。
        if context_base is None:
            context_base=build_theory_context(xs['zeff'],kmax=xcfg['kmax'],ndense=xcfg['ndense'],boxsize=xcfg['boxsize'],cosmology=xcfg['cosmology'])
        theory=dict(context_base)
        if xcfg['kmax']!=5 or xcfg['ndense']!=8000 or xcfg['boxsize']!=2000:raise ValueError('unexpected xi theory grid')
        ktemplate=np.geomspace(min(1e-4,theory['kfund']/2),max(1.,xcfg['kmax']*1.2),4000)
        theory['template'],_=build_template_arrays(task41,ktemplate,z=xs['zeff'],cosmology=xcfg['cosmology'])
        kd=theory['k_dense'];ke=theory['k_eff'];g=theory['g_nz'];vol=theory['volume']
        pdd=task41.interp_logk(kd,theory['template']['k'],theory['template']['pk_dd']);alpha=task41.interp_logk(kd,theory['template']['k'],theory['template']['alpha']);fg=xcfg['f_growth']
        edges=np.column_stack([s-5,s+5]);kernel,_=ximod.build_xi_kernel(ke,s,edges,xcfg['xi_kernel'],ell=0)
        wp=xpr.get('formal_gic_cache',xpr.get('formal_gic_window'))
        with np.load(wp,allow_pickle=False) as w:
            if not np.array_equal(w['k_eff'],ke):raise ValueError('GIC k grid mismatch')
            w2=w['w2']
        # 使用正式 interp_logk，先产生离散求和权重；不改变理论积分支持。
        def xi(theta):
            fnl,b1,sigma=theta[:3];amp=b1+fnl*2*1.686*(b1-pfix)*alpha
            i0,i2,i4=mu_moments(kd,sigma,fog_model=xcfg['fog_model'])
            pd=pdd*(amp*amp*i0+2*amp*fg*i2+fg*fg*i4)
            pe=task41.interp_logk(ke,kd,pd);weight=g*pe/vol
            return weight@kernel-np.sum(weight*w2)
        pmap=np.array([ps['maximum_posterior_sample']['point'][k] for k in pkmod.PARAM_NAMES]);xmap=np.array([xs['models']['formal_gic']['map'][k] for k in NAMES])
        pcheck=metrics(pk,pmap,data.pk_data,data.covariance,data.k_obs);xcheck=metrics(xi,xmap,xdata,xcov,s)
        validation={'pk_prediction_max_abs':float(np.max(np.abs(pcheck['prediction']-ps['maximum_posterior_sample']['pk_model_best']))),'xi_prediction_max_abs':float(np.max(np.abs(xcheck['prediction']-xpred))),'pk_chi2_delta':pcheck['chi2']-ps['maximum_posterior_sample']['chi2'],'xi_chi2_delta':xcheck['chi2']-xs['models']['formal_gic']['data']['chi2_map']}
        validation['pass']=validation['pk_prediction_max_abs']<1e-5 and validation['xi_prediction_max_abs']<1e-10 and abs(validation['pk_chi2_delta'])<1e-5 and abs(validation['xi_chi2_delta'])<1e-5
        if not validation['pass']:raise RuntimeError(f'self reproduction failed {name}: {validation}')
        print('[reproduced]',name,validation,flush=True)
        pbounds=np.array([cfg['priors'][k] for k in pkmod.PARAM_NAMES]).T
        # 正式 xi prior 与 P0 prior 不同，按已核对的原函数 log_prior 保留。
        pri=xs['models']['formal_gic'].get('priors',{'fnl_loc':[-500.,500.],'b1':[0.2,10.],'sigma_s':[0.,30.]})
        if xs['sampling'].get('b1_gaussian_prior') is not None:raise ValueError('unexpected conditional b1 prior')
        xbounds=np.array([pri[k] for k in NAMES]).T
        pstart=[pmap,np.array([pmap[0],pmap[1],8.,pmap[3]]),np.array([*xmap,pmap[3]])]
        xstart=[xmap,pmap[:3],np.array([xmap[0],xmap[1],15.])]
        pf=fit(pk,data.pk_data,data.covariance,pbounds,pstart);xf=fit(xi,xdata,xcov,xbounds,xstart)
        ptransfer=np.array([*xf['theta'],pf['theta'][3]])
        pprof=fit(pk,data.pk_data,data.covariance,pbounds,[ptransfer],fixed={i:xf['theta'][i] for i in range(3)})
        # 仅交换 fNL 并重新优化各自 nuisance，避免把全参数差归咎于 PNG。
        pfnl=fit(pk,data.pk_data,data.covariance,pbounds,pstart,fixed={0:xf['theta'][0]})
        xfnl=fit(xi,xdata,xcov,xbounds,xstart,fixed={0:pf['theta'][0]})
        diagp=fit(pk,data.pk_data,np.diag(np.diag(data.covariance)),pbounds,pstart)
        diagx=fit(xi,xdata,np.diag(np.diag(xcov)),xbounds,xstart)
        result={'sample':name,'campaign':case['campaign'],'status':'complete','script_sha256':manifest['source_script_sha256'],'audit_path':case['audit_path'],'audit_sha256':digest(case['audit_path']),'sources':{'pk':ppr,'xi':xpr},'zeff':data.zeff,'p_fixed':pfix,'parameter_names':{'pk':list(pkmod.PARAM_NAMES),'xi':NAMES},'bounds':{'pk':pbounds,'xi':xbounds},'self_reproduction':validation,'baseline_saved_map':{'pk':pcheck,'xi':xcheck},'reoptimized':{'pk':pf,'xi':xf},'cross_prediction':{'pk_at_xi_shared_profile_sn0':pprof,'xi_at_pk_shared':metrics(xi,pf['theta'][:3],xdata,xcov,s)},'fix_other_fnl_profile_nuisance':{'pk':pfnl,'xi':xfnl},'diagonal_covariance_diagnostic':{'pk':diagp,'xi':diagx},'interpretation':'All profile differences are conditional within one observable, not a joint tension significance. Diagonal covariance is not a replacement primary.'}
        save(path,result);manifest['samples'][name]={'path':str(path),'sha256':digest(path)};save(OUT/'summary.json',manifest)
        print('[done]',name,'MAP',pf['theta'],xf['theta'],'fnl-profile delta',pfnl['chi2']-pf['chi2'],xfnl['chi2']-xf['chi2'],'diag',diagp['theta'],diagx['theta'],flush=True)
    manifest['status']='complete';manifest['elapsed_seconds']=time.time()-t0;save(OUT/'summary.json',manifest)

if __name__=='__main__':main()
