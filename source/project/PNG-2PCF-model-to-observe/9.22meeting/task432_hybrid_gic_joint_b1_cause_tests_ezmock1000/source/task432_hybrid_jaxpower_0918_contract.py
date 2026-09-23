#!/usr/bin/env python3
"""Reproducible login-node hybrid comparison under the exact 9.18 contract.

Outline: freeze original arrays; validate direct GSM against interpolation;
sample three likelihoods with shared input arrays; reproduce the original page.
All scientific outputs are new. No legacy file is overwritten.
"""
from __future__ import annotations

# 执行大纲：冻结 9.18 输入 → 检查 hybrid 数值精度 → 三组独立长链 → 直接模型复核 → 原风格 PDF。
# 所有 CPU 数值库单线程，进程限制在同一组最多 8 个 CPU；不提交 Slurm，不覆盖既有主图。
import os
for name in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS'):
    os.environ[name] = '1'
os.environ['JAX_PLATFORMS'] = 'cpu'
os.environ['XLA_FLAGS'] = '--xla_cpu_multi_thread_eigen=false intra_op_parallelism_threads=1'
if hasattr(os, 'sched_getaffinity'):
    os.sched_setaffinity(0, sorted(os.sched_getaffinity(0))[:8])
import argparse
import hashlib
import json
import socket
import sys
import time
from pathlib import Path
import numpy as np
from scipy.interpolate import PPoly, RegularGridInterpolator, RectBivariateSpline
from scipy.linalg import solve_triangular
from scipy.optimize import least_squares

ROOT = Path('/pscratch/sd/l/lzy/PNG-2PCF-model-to-observe')
for directory in ('codes/task43','codes/task432','codes/task44'):
    sys.path.insert(0,str(ROOT/directory))
OUT = ROOT/'outputs/task43_outputs/rsd_validation/task432_model_repair/hybrid_jaxpower_0918_contract_v1'
MEETING = ROOT/'9.22meeting/task432_hybrid_jaxpower_9.18contract'
REFERENCE = ROOT/'9.18meeting/task43_ezmock_covariance_mcmc_vs_jaxpower/task43_ezmock604_vs_jaxpower_rsd_P02xi02_joint_kmax0p08_smin50_baomask80_120_v1.json'
# 默认仍为既有 jaxpower；协方差驱动脚本可指定基准标签和独立随机种子。
REFERENCE_COV = 'jaxpower'
SEED_BASE = 932918
OLD_EMULATOR = ROOT/'outputs/task43_outputs/rsd_validation/task432_model_repair/lightcone_png_velocileptors_gsm_longchain/lightcone_xi_emulator.npz'
LOWER=np.array([-500.,.5,0.,-1.]); UPPER=np.array([500.,5.,30.,1.])

def clean(obj):
    """把 NumPy 数组、标量和 Path 递归转换为可写入 JSON 的对象。"""
    if isinstance(obj,dict):return {str(k):clean(v) for k,v in obj.items()}
    if isinstance(obj,(list,tuple)):return [clean(x) for x in obj]
    if isinstance(obj,np.ndarray):return obj.tolist()
    if isinstance(obj,np.generic):return obj.item()
    if isinstance(obj,Path):return str(obj)
    return obj

def save(path,obj):
    """将对象写入临时 JSON，再原子替换目标；path 为输出路径，obj 为审计内容。"""
    path.parent.mkdir(parents=True,exist_ok=True)
    temp=path.with_suffix(path.suffix+'.tmp')
    temp.write_text(json.dumps(clean(obj),indent=2,sort_keys=True)+'\n');temp.replace(path)

def sha(path):
    """分块计算输入文件的 SHA256，用于冻结数据、代码和图的来源。"""
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda:f.read(8*1024*1024),b''):h.update(chunk)
    return h.hexdigest()

def log(event,**kw):
    """输出带 UTC 时间的单行 JSON 进度；event 为事件名，其余字段记录状态。"""
    print(json.dumps(clean({'event':event,'utc':time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()),**kw})),flush=True)

class Metric:
    def __init__(self,cov):
        """输入单 realization 协方差，先标准化为相关矩阵，再构造 Cholesky 白化算子。"""
        self.scale=np.sqrt(np.diag(cov))
        corr=cov/np.outer(self.scale,self.scale)
        self.chol=np.linalg.cholesky((corr+corr.T)/2)
        # Cholesky inverse in normalized coordinates, not raw-unit eigendecomposition.
        self.whitener=solve_triangular(self.chol,np.eye(len(cov)),lower=True)/self.scale[None,:]
    def white(self,r):
        """对白化前残差向量或向量批次进行转换，返回单位协方差空间的残差。"""
        return np.asarray(r)@self.whitener.T
    def chi2(self,r):
        """输入一个或一批残差，返回各自白化残差平方和；不改变协方差归一化。"""
        return np.sum(self.white(r)**2,axis=-1)

class Emulator:
    def __init__(self,path,method='linear'):
        """读取既有 hybrid 网格；method 控制数值插值方式，不增加物理自由参数。"""
        with np.load(path,allow_pickle=False) as a:
            self.f=a['f_grid'];self.b=a['b_grid'];self.values=a['values']
        self.method=method
        if method=='linear':self.interp=RegularGridInterpolator((self.f,self.b),self.values,bounds_error=True)
        else:self.splines=[RectBivariateSpline(self.f,self.b,self.values[:,:,i],kx=3,ky=3,s=0) for i in range(self.values.shape[-1])]
    def __call__(self,theta):
        """输入参数批次，使用前两列 fNL、b1 返回 52 维 xi 数据向量的预测。"""
        t=np.atleast_2d(theta)
        if self.method=='linear':return self.interp(t[:,:2])
        return np.column_stack([s.ev(t[:,0],t[:,1]) for s in self.splines])

def prepare():
    """复用 9.18 数据构造器并冻结解析协方差；核对直接 GSM、积分精度和批量 P 计算。"""
    if (OUT/'input_audit.json').exists():raise FileExistsError('Frozen inputs already exist')
    OUT.mkdir(parents=True,exist_ok=True)
    import task43_rsd_ezmock281_mcmc_compare as old
    from task432_lightcone_png_velocileptors import LightconePNGVelocileptors,build_cache
    log('prepare_start',host=socket.gethostname(),cpu_affinity=sorted(os.sched_getaffinity(0)))
    pmodel,_,dp,dx,payload,_=old.build_models_and_data()
    with np.load(old.ANALYTIC_COV,allow_pickle=False) as a:cov=np.asarray(a['rsd_joint'])
    assert dp.shape==(22,) and dx.shape==(52,) and cov.shape==(74,74)
    metric_x=Metric(cov[22:,22:]);metric_j=Metric(cov)
    rows=np.r_[np.arange(13),13+old.P2_KEEP]
    # Exact batched evaluation of the existing P spline and polynomial coefficients.
    np.savez_compressed(OUT/'frozen_inputs.npz',data_p=dp,data_x=dx,covariance=cov,
        p_spline_c=pmodel.spline.c[:, :, rows, :],p_spline_x=pmodel.spline.x,
        shot=pmodel.shot_response[rows],f_growth=pmodel.f_growth,
        p_edges=old.P_EDGES,xi_centers=old.XI_CENTERS[old.XI_MASK])
    emu=Emulator(OLD_EMULATOR)
    assert emu.values.shape[-1]==52 and np.all(np.isfinite(emu.values))
    np.savez_compressed(OUT/'xi_emulator.npz',f_grid=emu.f,b_grid=emu.b,values=emu.values)
    zeff=float(np.asarray(payload['zeff']).item())
    cache=build_cache(zeff=zeff,boxsize=2000.,kmax=3.,ells=(0,2),cosmology='abacus_c000')
    with np.load(cache,allow_pickle=False) as a:
        with np.load(old.ELL2_SUMMARY,allow_pickle=False) as x:s=np.asarray(x['s'])
        direct=LightconePNGVelocileptors(a['k_eff'],a['pk_dd'],a['alpha'],float(a['f_growth']),s)
    mask=(s>=50)&(s<350)&~((s>=80)&(s<120))
    def eval_direct(t,nint):
        """输入 fNL、b1 和积分点数，返回同一尺度 mask 下的直接 GSM xi0、xi2 向量。"""
        y=direct.evaluate(fnl=float(t[0]),b1=float(t[1]),nint=nint)
        return np.r_[y[0][mask],y[2][mask]]
    probes=[[-7.73,2.3904],[-5.60,2.4368],[-63.7,2.273],[-31.3,2.517],[18.7,2.347],[57.3,2.643],[96.1,2.177],[-102.3,2.713],[0.,2.4],[25.,2.5]]
    records=[];truth=[]
    for t in probes:
        y600=eval_direct(t,600); y1200=eval_direct(t,1200);truth.append(y1200)
        diff=emu(t)[0]-y1200
        record={'theta':t,'integration_error_chi2_joint':metric_j.chi2(np.r_[np.zeros(22),y600-y1200]),
                'emulator_error_chi2_joint':metric_j.chi2(np.r_[np.zeros(22),diff]),
                'emulator_error_chi2_xi':metric_x.chi2(diff),
                'xi_likelihood_chi2_change':metric_x.chi2(dx-emu(t)[0])-metric_x.chi2(dx-y1200)}
        records.append(record);log('validation_point',**record)
    integration_pass=max(r['integration_error_chi2_joint'] for r in records)<.001
    def errors(model):
        """在固定验证点计算候选插值模型相对直接预测的联合协方差加权误差。"""
        return [float(metric_j.chi2(np.r_[np.zeros(22),model(t)[0]-y])) for t,y in zip(probes,truth)]
    methods={'linear':errors(emu)};method='linear'
    if max(methods['linear'])>=.01:
        cubic=Emulator(OLD_EMULATOR,'cubic');methods['cubic']=errors(cubic);method='cubic'
    interpolation_pass=max(methods[method])<.01
    np.savez_compressed(OUT/'validation_predictions.npz',theta=probes,direct_nint1200=truth)
    audit={'reference':str(REFERENCE),'reference_sha256':sha(REFERENCE),
        'analytic_covariance':str(old.ANALYTIC_COV),'analytic_covariance_sha256':sha(old.ANALYTIC_COV),
        'p_payload':str(old.PAYLOAD_NPZ),'p_payload_sha256':sha(old.PAYLOAD_NPZ),
        'xi_summary':str(old.ELL2_SUMMARY),'xi_summary_sha256':sha(old.ELL2_SUMMARY),
        'old_emulator':str(OLD_EMULATOR),'old_emulator_sha256':sha(OLD_EMULATOR),
        'hybrid_source_sha256':sha(ROOT/'codes/task432/task432_lightcone_png_velocileptors.py'),
        'runner_sha256':sha(__file__),'host':socket.gethostname(),'priors':{'lower':LOWER,'upper':UPPER},
        'data_contract':'9.18 page 2: P0 13 + P2 9 + xi0 26 + xi2 26; x25 mean; C_single; full cross covariance',
        'xi_model_contract':'Existing hybrid at bin centers, no formal GIC, no xi FoG parameter; unchanged physical implementation',
        'emulator_method':method,'validation':records,'method_error_chi2_joint':methods,
        'gates':{'integration_pass':integration_pass,'interpolation_pass':interpolation_pass}}
    # Validate batched P against the original scalar implementation.
    engine=Engine('p02',method)
    rng=np.random.default_rng(91822)
    ts=np.column_stack([rng.uniform(-100,100,10),rng.uniform(2.,3.,10),rng.uniform(0.,20.,10),rng.uniform(-.5,.5,10)])
    target=np.stack([pmodel.evaluate(t)[rows] for t in ts])
    diff=engine.predict(ts)-target
    audit['p_batch_max_relative_error']=np.max(np.abs(diff)/np.maximum(np.abs(target),1.))
    audit['gates']['p_batch_pass']=bool(audit['p_batch_max_relative_error']<1e-11)
    audit['status']='pass' if all(audit['gates'].values()) else 'failed'
    save(OUT/'input_audit.json',audit);log('prepare_done',status=audit['status'],method=method,gates=audit['gates'])
    if audit['status']!='pass':raise RuntimeError('Preflight gate failed; do not sample')

class Engine:
    def __init__(self,variant,method):
        """按 variant 选择 P、xi 或 joint 数据及协方差块，保留完整交叉块和原先验。"""
        self.variant=variant
        with np.load(OUT/'frozen_inputs.npz',allow_pickle=False) as a:
            self.dp=a['data_p'];self.dx=a['data_x'];cov=a['covariance'];self.f=float(a['f_growth'])
            self.ps=PPoly.construct_fast(a['p_spline_c'],a['p_spline_x']);self.shot=a['shot']
        self.x=Emulator(OUT/'xi_emulator.npz',method)
        self.data=self.dp if variant=='p02' else self.dx if variant=='xi02' else np.r_[self.dp,self.dx]
        self.cov=cov[:22,:22] if variant=='p02' else cov[22:,22:] if variant=='xi02' else cov
        self.metric=Metric(self.cov)
        self.names=['fNL','b1'] if variant=='xi02' else ['fNL','b1','sigma_s_P','sn0']
        self.lower=LOWER[:len(self.names)];self.upper=UPPER[:len(self.names)]
    def predict(self,theta):
        """批量计算指定探针预测；P 使用原 spline，xi 使用 hybrid 插值，共享 fNL 和 b1。"""
        t=np.atleast_2d(theta)
        if self.variant=='xi02':return self.x(t)
        fnl,b,sig,sn=t.T;q=fnl*2*1.686*(b-1);f=self.f
        coeff=np.column_stack([b*b,2*b*q,q*q,2*b*f,2*q*f,np.full(len(t),f*f)])
        p=np.einsum('nij,nj->ni',self.ps(sig),coeff)+sn[:,None]*self.shot
        return p if self.variant=='p02' else np.column_stack([p,self.x(t)])
    def residual(self,t):
        """返回给定参数下的白化残差，供多起点最小二乘求最大似然点。"""
        return self.metric.white(self.data-self.predict(t)[0])
    def logp(self,t):
        """先检查原均匀先验，再批量计算 Gaussian 对数似然；越界点返回负无穷。"""
        t=np.atleast_2d(t);valid=np.all((t>self.lower)&(t<self.upper),axis=1)
        out=np.full(len(t),-np.inf)
        if np.any(valid):out[valid]=-.5*self.metric.chi2(self.data-self.predict(t[valid]))
        return out

def summarize(chain,logp,names,burn):
    """输入按步存储的链和对数似然，扣除 burn-in 后计算分位数及三项收敛门限。"""
    import emcee
    post=chain[burn:];flat=post.reshape(-1,len(names));q=np.percentile(flat,[16,50,84],axis=0)
    tau=emcee.autocorr.integrated_time(post,quiet=True,tol=0)
    n=post.shape[0]//2;split=np.concatenate([post[:n],post[-n:]],axis=1)
    within=np.mean(np.var(split,axis=0,ddof=1),axis=0);between=n*np.var(np.mean(split,axis=0),axis=0,ddof=1)
    rhat=np.sqrt(((n-1)*within/n+between/n)/within)
    drift=np.abs(np.median(post[:n].reshape(-1,len(names)),axis=0)-np.median(post[-n:].reshape(-1,len(names)),axis=0))/(.5*(q[2]-q[0]))
    best=np.argmax(logp[burn:].reshape(-1))
    return {'parameter_names':names,'posterior':{name:{'q16':q[0,i],'q50':q[1,i],'q84':q[2,i],'sigma68':.5*(q[2,i]-q[0,i])} for i,name in enumerate(names)},
        'tau':dict(zip(names,tau)),'split_rhat':dict(zip(names,rhat)),
        'postburn_length_over_tau':dict(zip(names,len(post)/tau)),'half_chain_shift_sigma':dict(zip(names,drift)),
        'best_chain_theta':flat[best],'best_chain_chi2':-2*np.max(logp[burn:]),
        'gates':{'split_rhat_max_below_1p01':np.max(rhat)<1.01,'postburn_length_min_above_50tau':np.min(len(post)/tau)>50,'half_chain_shift_max_below_0p1sigma':np.max(drift)<.1}}

def fit(variant):
    """重新拟合指定探针：多起点优化、64 walkers 采样、必要时延长链并保存原始状态和审计。"""
    import emcee
    audit=json.loads((OUT/'input_audit.json').read_text());assert audit['status']=='pass'
    engine=Engine(variant,audit['emulator_method']);d=len(engine.names)
    destination=OUT/'fits'/variant;destination.mkdir(parents=True,exist_ok=True)
    if (destination/'summary.json').exists():raise FileExistsError('Completed immutable fit already exists')
    starts=[[-12,2.42,2.2,.14],[0,2.4,1.,.1],[-8,2.39,2.17,.07],[30,2.3,5.,0.],[-50,2.55,8.,-.1]]
    solutions=[least_squares(engine.residual,np.array(t[:d]),bounds=(engine.lower,engine.upper),x_scale='jac',max_nfev=1500,ftol=1e-11,xtol=1e-11,gtol=1e-11) for t in starts]
    best=min(solutions,key=lambda r:sum(r.fun**2));theta=best.x
    log('map',variant=variant,theta=theta,chi2=np.sum(best.fun**2))
    seed=SEED_BASE+['p02','xi02','joint'].index(variant)
    rng=np.random.default_rng(seed);scale=np.array([4,.015,.12,.025])[:d]
    initial=np.clip(theta+rng.normal(size=(64,d))*scale,engine.lower+1e-7,engine.upper-1e-7)
    np.random.seed(seed)
    backend=emcee.backends.HDFBackend(str(destination/'chain.h5'));backend.reset(64,d)
    sampler=emcee.EnsembleSampler(64,d,engine.logp,vectorize=True,backend=backend)
    state=initial;done=0;started=time.monotonic();summary=None
    for target in [30000,45000,60000,90000]:
        while done<target:
            step=min(1000,target-done);state=sampler.run_mcmc(state,step,progress=False);done+=step
            log('sampling',variant=variant,steps=done,elapsed_seconds=time.monotonic()-started)
            save(destination/'progress.json',{'variant':variant,'steps':done,'elapsed_seconds':time.monotonic()-started,'host':socket.gethostname()})
        summary=summarize(sampler.get_chain(),sampler.get_log_prob(),engine.names,5000)
        save(destination/'convergence_latest.json',summary);log('convergence',variant=variant,steps=done,gates=summary['gates'])
        if all(summary['gates'].values()):break
    # Recover a better optimum if any posterior sample improves on multistart.
    extra=least_squares(engine.residual,np.asarray(summary['best_chain_theta']),bounds=(engine.lower,engine.upper),x_scale='jac',max_nfev=1500,ftol=1e-11,xtol=1e-11,gtol=1e-11)
    if sum(extra.fun**2)<sum(best.fun**2):best=extra
    summary.update({'map_theta':best.x,'map_chi2':sum(best.fun**2),'map_success':best.success,'multistart_chi2':[sum(s.fun**2) for s in solutions],
        'nwalkers':64,'nsteps':done,'burnin':5000,'seed':seed,'elapsed_seconds':time.monotonic()-started,'acceptance_fraction_mean':np.mean(sampler.acceptance_fraction),
        'data_dimension':len(engine.data),'priors':{'lower':engine.lower,'upper':engine.upper},'input_audit_sha256':sha(OUT/'input_audit.json')})
    if variant=='p02' and REFERENCE_COV is not None:
        ref=json.loads(REFERENCE.read_text())['constraints'][REFERENCE_COV+'_p02']
        compare={k:abs(summary['posterior'][k]['q50']-ref[k]['q50'])/ref[k]['sigma68'] for k in ('fNL','b1')}
        summary['reference_posterior_shift_sigma']=compare
        summary['gates']['p_baseline_reproduced']=max(compare.values())<.1
    summary['status']='pass' if all(summary['gates'].values()) else 'failed'
    save(destination/'summary.json',summary)
    # Keep full step-indexed chain in HDF5; standard NPZ contains post-burn samples.
    np.savez_compressed(destination/'samples.npz',chain=sampler.get_chain(discard=5000),parameter_names=engine.names,map_theta=best.x)
    log('fit_done',variant=variant,status=summary['status'],posterior=summary['posterior'])
    if summary['status']!='pass':raise RuntimeError('Fit gates failed')

def plot():
    """调用原 9.18 绘图函数生成单页 PDF，并分别检查 joint 的 b1 中位数和最大似然是否居中。"""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.backends.backend_pdf import PdfPages
    from task43_plot_ezmock506_covariance_mcmc_vs_jaxpower import triangle_page
    ref=json.loads(REFERENCE.read_text());chains={};results={}
    for variant in ('p02','xi02','joint'):
        root=OUT/'fits'/variant;summary=json.loads((root/'summary.json').read_text());assert summary['status']=='pass'
        with np.load(root/'samples.npz',allow_pickle=False) as a:chain=a['chain'].reshape(-1,len(summary['parameter_names']))
        chains['jaxpower_'+variant]={'chain':chain,'summary':summary};results[variant]=summary
    plt.rcParams.update({'font.family':'sans-serif','font.sans-serif':['Arial','DejaVu Sans','Liberation Sans'],'pdf.fonttype':42,'ps.fonttype':42,'font.size':10.,'axes.linewidth':1.,'xtick.direction':'in','ytick.direction':'in','xtick.top':True,'ytick.right':True})
    MEETING.mkdir(parents=True,exist_ok=True)
    path=MEETING/'task432_hybrid_jaxpower_P02xi02_joint_kmax0p08_smin50_baomask80_120_9.18style_v1.pdf'
    if path.exists():raise FileExistsError(path)
    with PdfPages(path) as pdf:
        triangle_page(pdf,cov='jaxpower',cov_display='jaxpower analytic (hybrid xi)',chains=chains,ranges=ref['shared_axis_ranges'])
    b={v:{'q50':r['posterior']['b1']['q50'],'maximum_likelihood':r['map_theta'][1],'q16':r['posterior']['b1']['q16'],'q84':r['posterior']['b1']['q84']} for v,r in results.items()}
    between={key:min(b['p02'][key],b['xi02'][key])<=b['joint'][key]<=max(b['p02'][key],b['xi02'][key]) for key in ('q50','maximum_likelihood')}
    result={'status':'pass','output_pdf':str(path),'output_pdf_sha256':sha(path),'reference':str(REFERENCE.with_suffix('.pdf')),'reference_page':2,'axis_ranges':ref['shared_axis_ranges'],'input_audit':str(OUT/'input_audit.json'),'results':results,'b1_comparison':b,'joint_b1_between_marginals':between,'model_note':'Existing 9.22 hybrid: center-valued xi, no formal GIC; P nuisance only. Jaxpower C_single, same 9.18 data points.'}
    save(path.with_suffix('.json'),result);log('plot_done',path=path,b1=b,between=between)

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('action',choices=['prepare','p02','xi02','joint','plot']);args=parser.parse_args()
    if args.action=='prepare':prepare()
    elif args.action=='plot':plot()
    else:fit(args.action)
