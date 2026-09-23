#!/usr/bin/env python3
"""六种预选尺度口径的完整RIC诊断与可恢复MCMC。

大纲：只读冻结74向量 -> 按实际bin选取covariance后重新求逆 ->
多起点MAP/条件残差 -> 两种代表配置的严格MCMC。BAO的82维输入由独立
扩展脚本构建；原模型、权重、物理参数与先验保持不变。
"""
import os
for key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):
    os.environ[key]='1'
import argparse,copy,json,sys,time,socket
from pathlib import Path
import numpy as np
from scipy.optimize import least_squares
from scipy.interpolate import RectBivariateSpline
ROOT=Path('/pscratch/sd/l/lzy/PNG-2PCF-model-to-observe')
for d in ('codes/task432','codes/task43','codes/task44'):sys.path.insert(0,str(ROOT/d))
from task432_full_ric_compile import Engine as OriginalEngine,common
from task432_full_ric_geometry import BASE,sha
from task432_full_ric_backend import OUT as RIC
from task432_hybrid_gic_0918_contract import finite_factors
OUT=BASE.parent/'scale_selection_ezmock1000'
COMPILED=RIC/'pilot/mean4xi_mean2P_prodrr_stochastic/engine.npz'
VARIANTS=('p02','xi02','joint')
CASES={'baseline':{},'kmin0p013':{'kmin':.013},'kmax0p06':{'kmax':.06},
       'smin80':{'smin':80},'smax250':{'smax':250},'bao_unmasked':{'bao':False}}
STARTS=np.array([[-12.,2.42,2.2,.14],[0.,2.4,1.,.1],[30.,2.3,5.,0.],[-50.,2.55,8.,-.1]])

def save(name,obj):
    """将可含NumPy值的对象原子写入本任务目录；不修改原正式文件。"""
    common.save(OUT/name,obj)

def load_master(extended=False):
    """返回明确排序的data/cov/halo/EZ；extended仅用于恢复BAO八个点。"""
    if extended:
        with np.load(OUT/'expanded_inputs.npz') as a:return {k:a[k] for k in a.files}
    with np.load(BASE/'frozen_inputs.npz') as a:
        d={'data':np.r_[a['data_p'],a['data_x']],'covariance':a['covariance'],
           'p_edges':a['p_edges'],'xi_centers':a['xi_centers']}
    with np.load(BASE.parent/'fkp_conditional_diagnostics/conditional_modes.npz') as a:
        d.update(halo_stack=a['halo_stack'],ez_stack=a['ez_stack'])
    return d

def selected(case,master):
    """k按完整bin边界，s按中心；始终先保留原P2低k规则，返回joint索引。"""
    c=CASES[case];pe=master['p_edges'];s=master['xi_centers']
    pk=(pe[:,0]>=c.get('kmin',.005)-1e-10)&(pe[:,1]<=c.get('kmax',.08)+1e-10)
    x=(s>=c.get('smin',50))&(s<c.get('smax',350))
    if c.get('bao',True):x&=~((s>=80)&(s<120))
    return np.flatnonzero(np.r_[pk,pk[4:],x,x])

class ExpandedXi:
    def __init__(self):
        """加载经直接模型检查的60维xi；原52维网格逐元素保留。"""
        with np.load(OUT/'expanded_model.npz') as a:
            self.splines=[RectBivariateSpline(a['f_grid'],a['b_grid'],a['values'][:,:,i],s=0) for i in range(60)]
            self.fixed=a['fixed'];self.stochastic=a['stochastic']
    def __call__(self,theta):
        """输入三/四参数数组，输出完整聚类+确定性Poisson+既有sn0响应。"""
        t=np.atleast_2d(theta);sn=t[:,-1] if t.shape[1] in (3,4) else np.zeros(len(t))
        return np.column_stack([s.ev(t[:,0],t[:,1]) for s in self.splines])+self.fixed+sn[:,None]*self.stochastic

class Engine(OriginalEngine):
    def __init__(self,case,variant):
        """复用原物理预测，只切最终向量；finite-mock修正按实际维数重算。"""
        super().__init__(variant,COMPILED)
        self.case=case;self.master=load_master(case=='bao_unmasked')
        if case=='bao_unmasked':self.x=ExpandedXi()
        ids=selected(case,self.master)
        ids=ids[ids<22] if variant=='p02' else ids[ids>=22] if variant=='xi02' else ids
        self.ids=ids;self.predict_ids=ids-22 if variant=='xi02' else ids
        self.data=self.master['data'][ids];self.cov=self.master['covariance'][np.ix_(ids,ids)]
        self.raw_metric=common.Metric(self.cov)
        self.corrections=finite_factors(len(ids),len(self.names))
        self.metric=common.Metric(self.cov/self.corrections['hartlap'])
    def predict(self,theta):
        """各case维数由self.predict_ids指定；没有截断内部窗口/IC积分。"""
        return OriginalEngine.predict(self,theta)[:,self.predict_ids]
    def full_prediction(self,theta):
        """条件检验需要相同theta的完整joint预测，不重新拟合被删点。"""
        if self.variant!='joint':raise ValueError('Only joint prediction requested')
        return OriginalEngine.predict(self,theta)

def optimize(e,data=None,starts=None):
    """在同一有界先验多起点优化；可指定配对噪声data，返回原始结果字典。"""
    d=e.data if data is None else data
    cols=[0,1,3] if e.variant=='xi02' else [0,1,2,3]
    starts=STARTS[:,cols] if starts is None else np.atleast_2d(starts)
    fits=[least_squares(lambda t:e.metric.white(d-e.predict(t)[0]),x,bounds=(e.lower,e.upper),
         x_scale='jac',max_nfev=1500,ftol=1e-11,xtol=1e-11,gtol=1e-11) for x in starts]
    best=min(fits,key=lambda x:x.fun@x.fun);pred=e.predict(best.x)[0]
    # SVD检查退化；边界时仅报告局部curvature，不声称是可信区间。
    h=best.jac.T@best.jac;local=np.linalg.pinv(h)
    return {'theta':best.x,'parameter_names':e.names,'raw_chi2':float(e.raw_metric.chi2(d-pred)),
        'hartlap_chi2':float(best.fun@best.fun),'success':bool(best.success),'prediction':pred,
        'multistart_chi2':[float(x.fun@x.fun) for x in fits],'ndata':len(d),'indices':e.ids,
        'finite_mock_corrections':e.corrections,'local_covariance':local,
        'prior_boundary':[e.names[i] for i in range(len(e.names)) if min(best.x[i]-e.lower[i],e.upper[i]-best.x[i])<1e-5],
        'local_sigma_not_posterior':dict(zip(e.names,np.sqrt(np.diag(local))*e.corrections['percival_sigma_factor']))}

def conditional(e,theta):
    """Schur分解raw C的完整χ²；deleted残差对kept数据条件化，禁止切原precision。"""
    C=e.master['covariance'];r=e.master['data']-e.full_prediction(theta)[0]
    K=e.ids;D=np.setdiff1d(np.arange(len(r)),K)
    if not len(D):return None
    A=np.linalg.solve(C[np.ix_(K,K)],C[np.ix_(K,D)]).T
    V=C[np.ix_(D,D)]-A@C[np.ix_(K,D)];rd=r[D]-A@r[K]
    ck=float(e.raw_metric.chi2(r[K]));cd=float(common.Metric(V).chi2(rd));cf=float(common.Metric(C).chi2(r))
    return {'deleted_indices':D,'conditional_residual':rd,'conditional_covariance':V,
            'chi2_kept':ck,'chi2_deleted_given_kept':cd,'chi2_full_same_theta':cf,
            'identity_error':abs(cf-ck-cd),'scope':'Raw C_single identity at same theta; no cross-dimension fit-quality claim'}

def map_scan(cases):
    """运行给定预选配置，保存全部结果；baseline需要精确回归原MAP。"""
    OUT.mkdir(parents=True,exist_ok=True);cache={}
    previous=json.loads((OUT/'map_scan.json').read_text()) if (OUT/'map_scan.json').exists() else {'cases':{}}
    for case in cases:
        result={}
        for variant in VARIANTS:
            e=Engine(case,variant);key=(tuple(e.ids),variant,case=='bao_unmasked')
            if key not in cache:cache[key]=optimize(e)
            result[variant]=cache[key]
        b={v:result[v]['theta'][1] for v in VARIANTS};ej=Engine(case,'joint')
        row={'fits':result,'b1_MAP':b,'contrasts':np.array([b['joint']-b['p02'],b['joint']-b['xi02']]),
             'outside_distance':max(0,min(b['p02'],b['xi02'])-b['joint'],b['joint']-max(b['p02'],b['xi02'])),
             'joint_between_MAP':min(b['p02'],b['xi02'])<=b['joint']<=max(b['p02'],b['xi02']),
             'conditional':conditional(ej,result['joint']['theta'])}
        if case=='baseline':
            for v in VARIANTS:
                old=json.loads((RIC/'formal_mean4'/v/'summary.json').read_text())
                assert abs(result[v]['theta'][1]-old['map_theta'][1])<2e-5
                assert abs(result[v]['raw_chi2']-old['map_raw_chi2'])<1e-7
        if case=='bao_unmasked':
            # 新增BAO也从相同82维C中定义条件增量，在旧74点拟合和新拟合处分别评价。
            reduced=Engine('bao_unmasked','joint')
            sm=reduced.master['xi_centers'];mask=~((sm>=80)&(sm<120))
            reduced.ids=np.flatnonzero(np.r_[np.ones(22,dtype=bool),mask,mask])
            reduced.raw_metric=common.Metric(reduced.master['covariance'][np.ix_(reduced.ids,reduced.ids)])
            oldtheta=json.loads((RIC/'formal_mean4/joint/summary.json').read_text())['map_theta']
            row['BAO_conditional_at_old_MAP']=conditional(reduced,oldtheta)
            row['BAO_conditional_at_new_MAP']=conditional(reduced,result['joint']['theta'])
        previous['cases'][case]=row;previous.update(status='MAP_diagnostics',source_sha256=sha(__file__),prespecified_cases=CASES)
        save('map_scan.json',previous)
        print(json.dumps({'event':'scale_map','case':case,'b1':b,'Dout':row['outside_distance']}),flush=True)

def sample(case,variant):
    """64 walkers HDF续跑；严格收敛通过才写完成summary，原链不覆盖。"""
    import emcee
    if case=='bao_unmasked':assert json.loads((OUT/'expanded_validation.json').read_text())['status']=='pass'
    e=Engine(case,variant);d=len(e.names);dest=OUT/'chains'/case/variant;dest.mkdir(parents=True,exist_ok=True)
    if (dest/'summary.json').exists() and json.loads((dest/'summary.json').read_text())['status']=='pass':return
    seed=4322500+10*list(CASES).index(case)+VARIANTS.index(variant)
    manifest={'case':case,'variant':variant,'seed':seed,'nwalkers':64,'burnin':5000,'indices':e.ids,
              'source_sha256':sha(__file__),'compiled_sha256':sha(COMPILED),'baseline_sha256':sha(BASE/'frozen_inputs.npz'),
              'expanded_model_sha256':sha(OUT/'expanded_model.npz') if case=='bao_unmasked' else None}
    if (dest/'input_manifest.json').exists():assert json.loads((dest/'input_manifest.json').read_text())==common.clean(manifest)
    else:common.save(dest/'input_manifest.json',manifest)
    best=optimize(e);backend=emcee.backends.HDFBackend(str(dest/'chain.h5'))
    if backend.initialized:state=None;done=backend.iteration
    else:
        backend.reset(64,d);rng=np.random.default_rng(seed);cols=[0,1,3] if d==3 else [0,1,2,3]
        state=np.clip(best['theta']+rng.normal(size=(64,d))*np.array([4.,.015,.12,.025])[cols],e.lower+1e-7,e.upper-1e-7);done=0
    np.random.seed(seed);sampler=emcee.EnsembleSampler(64,d,e.logp,vectorize=True,backend=backend);started=time.monotonic()
    for target in (30000,45000,60000,90000,120000):
        while done<target:
            n=min(1000,target-done);state=sampler.run_mcmc(state,n,progress=False);done+=n
            common.save(dest/'progress.json',{'case':case,'variant':variant,'steps':done,'elapsed_s':time.monotonic()-started,'host':socket.gethostname(),'affinity':sorted(os.sched_getaffinity(0))})
        s=common.summarize(sampler.get_chain(),sampler.get_log_prob(),e.names,5000)
        common.save(dest/'convergence_latest.json',s)
        print(json.dumps({'event':'scale_convergence','case':case,'variant':variant,'steps':done,'gates':common.clean(s['gates'])}),flush=True)
        if all(s['gates'].values()):break
    extra=optimize(e,starts=s['best_chain_theta'])
    if extra['raw_chi2']<best['raw_chi2']:best=extra
    s['posterior_raw']=copy.deepcopy(s['posterior'])
    for item in s['posterior'].values():
        fac=e.corrections['percival_sigma_factor'];item['q16']=item['q50']+fac*(item['q16']-item['q50']);item['q84']=item['q50']+fac*(item['q84']-item['q50']);item['sigma68']*=fac
    s.update(status='pass' if all(s['gates'].values()) else 'failed',case=case,variant=variant,map_theta=best['theta'],map_raw_chi2=best['raw_chi2'],nsteps=done,burnin=5000,nwalkers=64,
        finite_mock_corrections=e.corrections,data_dimension=len(e.data),priors={'lower':e.lower,'upper':e.upper},map_prediction=best['prediction'],source_sha256=sha(__file__))
    common.save(dest/'summary.json',s)
    np.savez_compressed(dest/'samples.npz',chain=sampler.get_chain(discard=5000),parameter_names=e.names,map_theta=best['theta'])
    assert s['status']=='pass','Chain did not pass convergence'

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('action',choices=('map','sample'));p.add_argument('--cases',nargs='+',default=list(CASES)[:-1]);p.add_argument('--case',choices=list(CASES));p.add_argument('--variant',choices=VARIANTS)
    a=p.parse_args();map_scan(a.cases) if a.action=='map' else sample(a.case,a.variant)
