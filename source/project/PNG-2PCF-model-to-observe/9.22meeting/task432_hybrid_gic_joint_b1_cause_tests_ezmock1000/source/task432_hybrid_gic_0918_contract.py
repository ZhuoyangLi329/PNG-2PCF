#!/usr/bin/env python3
"""以 hybrid+GIC 更新两套 9.18 口径结果：窗口验证→网格→冻结输入→拟合→审计→PDF。

所有物理参数、数据点、协方差、P 模型沿用已冻结结果；GIC 由 hybrid 自身确定。
CPU 库单线程、全部进程共享最多 8 个 CPU，登录节点运行。旧结果只保留为溯源。
"""
from __future__ import annotations
import task432_hybrid_jaxpower_0918_contract as common
from task432_hybrid_jaxpower_0918_contract import np,Path,json,os,sha,save,log,Metric
from scipy.interpolate import RegularGridInterpolator,RectBivariateSpline
from scipy.optimize import least_squares
import argparse,copy,shutil,time,socket
from concurrent.futures import ProcessPoolExecutor,as_completed
import multiprocessing as mp
from task432_hybrid_gic import HybridGIC,PairWindow,SHARED,ROOT,MEAN_WINDOW

BASES={c:ROOT/f'outputs/task43_outputs/rsd_validation/task432_model_repair/hybrid_{c}_0918_contract_v1' for c in ('jaxpower','ezmock1000')}
OUTS={c:ROOT/f'outputs/task43_outputs/rsd_validation/task432_model_repair/hybrid_{c}_gic_0918_contract' for c in BASES}
MEETINGS={c:ROOT/f'9.22meeting/task432_hybrid_{c}_gic_9.18contract' for c in BASES}
BASE_ENGINE=common.Engine
GRID=SHARED/'hybrid_gic_grid.npz'


def finite_factors(nb,npar):
    """1000 mocks 的 Hartlap 似然与 Percival 区间因子，按实际数据和参数维数计算。"""
    ns=1000; h=(ns-nb-2)/(ns-1);a=2/((ns-nb-1)*(ns-nb-4));b=(ns-nb-2)/((ns-nb-1)*(ns-nb-4))
    m=(1+b*(nb-npar))/(1+a+b*(npar+1))
    return {'nmock':ns,'ndata':nb,'nparams':npar,'hartlap':h,'percival_m1':m,'percival_sigma_factor':np.sqrt(m)}


class GICEmulator:
    def __init__(self,path,method='linear'):
        """原 xi 采用已验证插值，GIC 用双三次插值；文件 values 同时保存最终修正后网格。"""
        with np.load(path,allow_pickle=False) as z:
            self.f=z['f_grid'];self.b=z['b_grid'];self.values=z['values'];self.raw=z['values_no_gic'];self.cg=z['gic_grid']
        self.method=method
        self.c=RectBivariateSpline(self.f,self.b,self.cg,kx=3,ky=3,s=0)
        if method=='linear':self.x=RegularGridInterpolator((self.f,self.b),self.raw,bounds_error=True)
        else:self.xs=[RectBivariateSpline(self.f,self.b,self.raw[:,:,i],kx=3,ky=3,s=0) for i in range(52)]

    def correction(self,theta):
        """参数批次→确定性 GIC 常数；没有额外采样参数。"""
        t=np.atleast_2d(theta)
        if np.any((t[:,0]<self.f[0])|(t[:,0]>self.f[-1])|(t[:,1]<self.b[0])|(t[:,1]>self.b[-1])):raise ValueError('Outside emulator grid')
        return self.c.ev(t[:,0],t[:,1])

    def __call__(self,theta):
        """输出 xi0-C_W 与 xi2 共 52 维，严格保留此前 26+26 bin 顺序。"""
        t=np.atleast_2d(theta)
        y=self.x(t[:,:2]) if self.method=='linear' else np.column_stack([s.ev(t[:,0],t[:,1]) for s in self.xs])
        y[:,:26]-=self.correction(t)[:,None]
        return y


def configure(cov):
    """给公共采样器注入当前输出目录、GIC 模型和协方差修正，复用稳定 MCMC 代码。"""
    common.OUT=OUTS[cov];common.MEETING=MEETINGS[cov];common.REFERENCE_COV=None
    common.SEED_BASE=4321900+(0 if cov=='jaxpower' else 1000)
    common.Emulator=GICEmulator
    class Engine(BASE_ENGINE):
        def __init__(self,variant,method):
            """使用原 covariance；EZmock 仅在似然中乘 Hartlap，不改变原始协方差文件。"""
            super().__init__(variant,method)
            if cov=='ezmock1000':
                self.corrections=finite_factors(len(self.data),len(self.names))
                self.metric=Metric(self.cov/self.corrections['hartlap'])
    common.Engine=Engine
    return Engine


def validate_window():
    """验证窗口核、扩展 r 表、批量实现和 GIC 积分；量化数值差异对正式似然的影响。"""
    from task43_rsd_model import shell_jell_kernel
    SHARED.mkdir(parents=True,exist_ok=True)
    w=PairWindow();m=HybridGIC(w);refined=HybridGIC(w,radial_refinement=2)
    active=(w.k>0)&(w.k<=.5)
    window_error=float(np.max(abs(shell_jell_kernel(w.k[active],w.edges,0)@w.prob-w.w2[active])))
    with np.load(BASES['jaxpower']/'frozen_inputs.npz',allow_pickle=False) as a:metric=Metric(a['covariance'])
    gain=float(metric.chi2(np.r_[np.zeros(22),np.ones(26),np.zeros(26)]))
    report={'status':'running','window':w.meta,'radial':m.radial_meta,'window_kernel_max_abs_error':window_error,
            'constant_unit_chi2_joint':gain,'probes':[],'source_hashes':{str(p):sha(p) for p in [Path(__file__),ROOT/'codes/task432/task432_hybrid_gic.py',ROOT/'codes/task432/task432_lightcone_png_velocileptors.py']},
            'scope':'Same isotropized formal-GIC scalar prescription as 9.18; RR integration of the current hybrid monopole. This is not the general anisotropic cross-window operator.',
            'low_k_contract':'Inherited Task432 kint, plin log-extrapolation, alpha log-k interpolation including constant boundary extrapolation, and Tukey transform taper. No old Kaiser/FoG GIC value is added.',
            'equivalence_note':'Old W2 reconstructed with exact shell-averaged j0 agrees. Old discrete theory and its numerical k<=0.5 window cut are reference inputs only; current correction uses the same continuous hybrid as xi.'}
    probes=[[0.,2.4],[-10.,2.4],[80.,2.7],[-100.,2.2],[500.,5.],[-500.,.5]]
    for t in probes:
        f,b=t
        settings=[(4,600,4),(8,1200,4),(16,2400,4),(8,1200,8)]
        cc={f'q{q}_n{n}_a{a}':m.correction(fnl=f,b1=b,nquad=q,nint=n,ngauss=a) for q,n,a in settings}
        cr=refined.correction(fnl=f,b1=b,nquad=8,nint=1200)
        m._set_cumulants(f,b)
        native=np.asarray([m.gsm.compute_xi_ell(float(s),m.f_growth,b-1,*([0.]*8),rwidth=100.,Nint=1200,ngauss=4,update_cumulants=False)[:2] for s in m.s]).T
        batch=m.poles(m.s,nint=1200)[:2]
        # 大尺度检验：同一连续谱、PNG alpha 延拓和 FFT 窗口下的 Kaiser 极限。
        rr=np.array([600.,800.,1000.,1500.,2000.,3000.,3400.])
        aa=np.interp(np.log(m.gsm.kint),np.log(m.k),m.alpha);q=f*2*1.686*(b-1);fg=m.f_growth
        pk0=((b+q*aa)**2+2/3*fg*(b+q*aa)+fg*fg/5)*m.gsm.plin
        kr,xx=m.gsm.sph_gsm.sph(0,pk0*m.gsm.window)
        linear=np.interp(rr,kr,xx);nonlinear=m.poles(rr,nint=1200)[0]
        linear_rel=float(np.linalg.norm(nonlinear-linear)/max(np.linalg.norm(linear),1e-10))
        error={k:gain*(v-cc['q16_n2400_a4'])**2 for k,v in cc.items()}
        error['radial_refinement']=gain*(cr-cc['q8_n1200_a4'])**2
        item={'theta':t,'corrections':cc,'refined_correction':cr,'error_chi2_joint':error,'batch_native_max_abs':float(np.max(abs(native-batch))),
              'large_s':rr,'large_s_hybrid_xi0':nonlinear,'large_s_linear_xi0':linear,'large_s_linear_relative_l2':linear_rel}
        report['probes'].append(item);log('window_validation',**item)
    report['gates']={'window_reproduction':window_error<1e-12,'batch_matches_native':max(x['batch_native_max_abs'] for x in report['probes'])<1e-12,
                     'gic_quadrature_and_radial_error':max(v for x in report['probes'] for v in x['error_chi2_joint'].values())<.001,
                     'original_r_nodes_preserved':m.radial_meta['original_nodes_preserved']}
    report['status']='pass' if all(report['gates'].values()) else 'failed'
    save(SHARED/'window_audit.json',report)
    np.savez_compressed(SHARED/'frozen_pair_window.npz',s_edges=w.edges,pair_probability=w.prob,pair_probability_by_phase=w.phase_prob,k=w.k,w2=w.w2)
    if report['status']!='pass':raise RuntimeError('Window validation failed')


def init_grid_worker():
    """每个单线程进程初始化一份 GSM，所有进程共享 taskset 的最多 8 个 CPU。"""
    global WORKER_MODEL
    WORKER_MODEL=HybridGIC()


def grid_row(task):
    """输入 (行号,fNL,b1 网格)，输出该 fNL 下的 C_W 数组。"""
    i,f,bb=task
    return i,np.array([WORKER_MODEL.correction(fnl=f,b1=b,nquad=4,nint=600) for b in bb])


def build_grid():
    """在原参数网格上预计算 GIC；6 个单线程进程，逐行原子保存可恢复检查点。"""
    audit=json.loads((SHARED/'window_audit.json').read_text());assert audit['status']=='pass'
    if GRID.exists():raise FileExistsError('Completed GIC grid exists')
    with np.load(BASES['jaxpower']/'xi_emulator.npz',allow_pickle=False) as a:f=a['f_grid'];b=a['b_grid'];raw=a['values']
    checkpoint=SHARED/'logs/grid_checkpoint.npz';checkpoint.parent.mkdir(parents=True,exist_ok=True)
    values=np.full((len(f),len(b)),np.nan)
    if checkpoint.exists():
        with np.load(checkpoint,allow_pickle=False) as a:
            if not np.array_equal(a['f_grid'],f) or not np.array_equal(a['b_grid'],b):raise RuntimeError('Grid mismatch')
            values=a['gic_grid']
    todo=[(i,float(ff),b) for i,ff in enumerate(f) if not np.all(np.isfinite(values[i]))]
    started=time.monotonic()
    with ProcessPoolExecutor(max_workers=6,mp_context=mp.get_context('spawn'),initializer=init_grid_worker) as pool:
        for result in as_completed([pool.submit(grid_row,t) for t in todo]):
            i,row=result.result();values[i]=row
            temporary=checkpoint.with_suffix('.tmp.npz')
            np.savez_compressed(temporary,f_grid=f,b_grid=b,gic_grid=values);temporary.replace(checkpoint)
            log('grid_row',index=i,completed=int(np.all(np.isfinite(values),axis=1).sum()),total=len(f),elapsed=time.monotonic()-started)
    corrected=raw.copy();corrected[:,:,:26]-=values[:,:,None]
    np.savez_compressed(GRID,f_grid=f,b_grid=b,gic_grid=values,values_no_gic=raw,values=corrected)
    save(SHARED/'grid_manifest.json',{'status':'pass','output':str(GRID),'sha256':sha(GRID),'window_audit_sha256':sha(SHARED/'window_audit.json'),
         'parent_emulator':str(BASES['jaxpower']/'xi_emulator.npz'),'parent_emulator_sha256':sha(BASES['jaxpower']/'xi_emulator.npz'),
         'shape':values.shape,'quadrature':{'shell_order':4,'Nint':600,'ngauss':4,'rmax':4000},'elapsed_seconds':time.monotonic()-started,
         'C_min':values.min(),'C_max':values.max(),'cpu_affinity':sorted(os.sched_getaffinity(0))})


def prepare(cov):
    """冻结原数据/协方差，复用 P-only，使用直接 hybrid+GIC 检验正式预测与插值。"""
    out=OUTS[cov];parent=BASES[cov]
    if (out/'input_audit.json').exists():raise FileExistsError('Frozen run already exists')
    out.mkdir(parents=True,exist_ok=True)
    for name in ('window_audit.json','grid_manifest.json'):
        assert json.loads((SHARED/name).read_text())['status']=='pass'
    assert json.loads((parent/'input_audit.json').read_text())['status']=='pass'
    shutil.copy2(parent/'frozen_inputs.npz',out/'frozen_inputs.npz');shutil.copy2(GRID,out/'xi_emulator.npz')
    with np.load(parent/'frozen_inputs.npz',allow_pickle=False) as a, np.load(out/'frozen_inputs.npz',allow_pickle=False) as b:
        unchanged={k:bool(np.array_equal(a[k],b[k])) for k in a.files};covmat=b['covariance']
    assert all(unchanged.values())
    covcheck={}
    if cov=='ezmock1000':
        with np.load(parent/'ezmock1000_covariance_and_stack.npz',allow_pickle=False) as a:
            rebuilt=np.cov(a['stack'],rowvar=False,ddof=1)
            assert np.array_equal(a['production_indices'],np.arange(1000))
            covcheck={'nmock':1000,'stack_shape':list(a['stack'].shape),'rebuild_relative_frobenius':float(np.linalg.norm(rebuilt-covmat)/np.linalg.norm(covmat))}
            assert covcheck['rebuild_relative_frobenius']<1e-12
    metric=Metric(covmat);direct=HybridGIC()
    probes=[[-7.73,2.3904],[-5.60,2.4368],[-63.7,2.273],[-31.3,2.517],[18.7,2.347],[57.3,2.643],[96.1,2.177],[-102.3,2.713],[0.,2.4],[25.,2.5]]
    truth=[];cs=[]
    for t in probes:
        y,c=direct.prediction(t);truth.append(y);cs.append(c)
    methods={}
    for method in ('linear','cubic'):
        emu=GICEmulator(out/'xi_emulator.npz',method);delta=emu(probes)-truth
        errors=metric.chi2(np.column_stack([np.zeros((len(probes),22)),delta]));methods[method]=errors
    method=min(methods,key=lambda k:float(max(methods[k])))
    emu=GICEmulator(out/'xi_emulator.npz',method)
    cerror=emu.correction(probes)-cs
    np.savez_compressed(out/'validation_predictions.npz',theta=probes,direct=np.asarray(truth),gic=np.asarray(cs))
    pdest=out/'fits/p02';pdest.mkdir(parents=True,exist_ok=True)
    s=json.loads((parent/'fits/p02/summary.json').read_text());assert s['status']=='pass'
    s['reused_from']=str(parent/'fits/p02');s['reuse_reason']='P data, covariance, prior and prediction unchanged by xi GIC fix'
    save(pdest/'summary.json',s)
    for name in ('samples.npz','chain.h5'):
        dest=pdest/name
        if not dest.exists():dest.symlink_to(parent/'fits/p02'/name)
    audit={'status':'pass' if max(methods[method])<.01 else 'failed','covariance':cov,'parent_run':str(parent),'parent_audit_sha256':sha(parent/'input_audit.json'),
        'unchanged_input_arrays':unchanged,'frozen_inputs_sha256':sha(out/'frozen_inputs.npz'),'emulator_method':method,'emulator_sha256':sha(out/'xi_emulator.npz'),
        'numerical_error_chi2_by_method':methods,'gic_interpolation_error':cerror,'window_audit':str(SHARED/'window_audit.json'),'window_audit_sha256':sha(SHARED/'window_audit.json'),
        'data_contract':'9.18 22 P + 52 xi = 74 joint; x25 mean; C_single; full cross; kmax0.08 smin50 BAO mask80-120',
        'model_contract':'Task432 hybrid at original bin centers + mandatory deterministic isotropized formal-GIC. xi0 -= C_W(hybrid); xi2 unchanged. No new free parameter.',
        'mock_crosscheck':covcheck,'priors':{'lower':common.LOWER,'upper':common.UPPER},'host':socket.gethostname(),'cpu_affinity':sorted(os.sched_getaffinity(0)),
        'source_hashes':{str(p):sha(p) for p in [Path(__file__),ROOT/'codes/task432/task432_hybrid_gic.py',Path(common.__file__)]}}
    save(out/'input_audit.json',audit);log('prepare_done',covariance=cov,status=audit['status'],max_error_chi2=max(methods[method]))
    if audit['status']!='pass':raise RuntimeError('Prediction validation failed')


def fit(cov,variant):
    """运行原采样器，EZmock 在采样后只修正汇总区间，原始链保持原样。"""
    configure(cov);common.fit(variant)
    path=OUTS[cov]/'fits'/variant/'summary.json';s=json.loads(path.read_text())
    s['model']='hybrid+GIC';s['covariance']=cov
    if cov=='ezmock1000':
        s['posterior_raw']=copy.deepcopy(s['posterior']);s['finite_mock_corrections']=finite_factors(s['data_dimension'],len(s['parameter_names']))
        fac=s['finite_mock_corrections']['percival_sigma_factor']
        for name,p in s['posterior'].items():
            for key in ('q16','q84'):p[key]=p['q50']+fac*(p[key]-p['q50'])
            p['sigma68']*=fac
    save(path,s)


def postflight(cov):
    """在真实后验点复核模型精度，用直接 GSM+GIC 求 ML，并报告 GIC 的后验范围。"""
    Engine=configure(cov);out=OUTS[cov];audit=json.loads((out/'input_audit.json').read_text());method=audit['emulator_method']
    direct=HybridGIC();p=Engine('p02',method);j=Engine('joint',method)
    rng=np.random.default_rng(4321909);report={'covariance':cov,'fits':{}}
    for variant in ('xi02','joint'):
        engine=Engine(variant,method);dest=out/'fits'/variant;s=json.loads((dest/'summary.json').read_text());assert s['status']=='pass'
        with np.load(dest/'samples.npz',allow_pickle=False) as a:flat=a['chain'].reshape(-1,len(engine.names))
        # 随机后验点 + 参数尾部 + MAP，避免只验证后验中心。
        ids=rng.choice(len(flat),24,replace=False)
        tailids=[int(np.argmin(flat[:,0])),int(np.argmax(flat[:,0])),int(np.argmin(flat[:,1])),int(np.argmax(flat[:,1]))]
        points=np.vstack([flat[ids],flat[tailids],s['map_theta']]);checks=[]
        for t in points:
            truth,c=direct.prediction(t,nquad=8,nint=1200)
            delta=engine.x(t)[0]-truth
            checks.append({'theta':t,'model_error_chi2_joint':j.metric.chi2(np.r_[np.zeros(22),delta]),'direct_gic':c,'emulator_gic':float(engine.x.correction(t)[0])})
        def prediction(t):
            """只替换 xi 为直接预测，P 继续使用已冻结 spline。"""
            x,_=direct.prediction(t,nquad=8,nint=1200)
            return x if variant=='xi02' else np.r_[p.predict(t)[0],x]
        def residual(t):
            """返回直接模型白化残差供有界最小二乘优化。"""
            return engine.metric.white(engine.data-prediction(t))
        sol=least_squares(residual,np.asarray(s['map_theta']),bounds=(engine.lower,engine.upper),x_scale='jac',max_nfev=500,ftol=1e-9,xtol=1e-9,gtol=1e-9)
        maximum=max(float(x['model_error_chi2_joint']) for x in checks)
        cpost=engine.x.correction(flat[::20]);gq=np.percentile(cpost,[16,50,84])
        # 与更高精度积分的独立终点比较。
        y,c=direct.prediction(sol.x,nquad=16,nint=2400)
        yh,ch=direct.prediction(sol.x,nquad=8,nint=1200)
        int_error=float(j.metric.chi2(np.r_[np.zeros(22),y-yh]))
        item={'posterior_checks':checks,'max_model_error_chi2_joint':maximum,'interpolation_gate':maximum<.01,
              'emulated_map_theta':s['map_theta'],'emulated_map_chi2':s['map_chi2'],'direct_map_theta':sol.x,'direct_map_chi2':sum(sol.fun**2),
              'direct_optimizer_success':sol.success,'higher_integration_error_chi2':int_error,'integration_gate':int_error<.001,
              'gic_at_direct_map':ch,'gic_posterior_q16_q50_q84':gq}
        report['fits'][variant]=item
        if item['interpolation_gate'] and sol.success and item['integration_gate']:
            s['emulated_map_theta']=s['map_theta'];s['emulated_map_chi2']=s['map_chi2'];s['map_theta']=sol.x;s['map_chi2']=sum(sol.fun**2)
            s['map_evaluation']='direct hybrid+GIC, Nint1200 shell quadrature8; posterior uses validated emulator'
            s['gic_at_direct_map']=ch;s['gic_posterior_q16_q50_q84']=gq;save(dest/'summary.json',s)
        log('postflight',covariance=cov,variant=variant,map_theta=sol.x,chi2=sum(sol.fun**2),max_model_error_chi2=maximum,integration_error_chi2=int_error,gic=ch)
    report['status']='pass' if all(x['interpolation_gate'] and x['direct_optimizer_success'] and x['integration_gate'] for x in report['fits'].values()) else 'failed'
    save(out/'postflight_audit.json',report)
    if report['status']!='pass':raise RuntimeError('Postflight failed')


def plot(cov):
    """生成与原页同风格的正式 PDF；展示有限 mock 修正，并单列 joint b1 位置判断。"""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.backends.backend_pdf import PdfPages
    from scipy.stats import chi2 as chi2dist
    from task43_plot_ezmock506_covariance_mcmc_vs_jaxpower import triangle_page
    out=OUTS[cov];meeting=MEETINGS[cov];assert json.loads((out/'postflight_audit.json').read_text())['status']=='pass'
    ref=json.loads(common.REFERENCE.read_text());chains={};results={};quantile_checks={}
    for variant in ('p02','xi02','joint'):
        dest=out/'fits'/variant;s=json.loads((dest/'summary.json').read_text());assert s['status']=='pass'
        with np.load(dest/'samples.npz',allow_pickle=False) as a:raw=a['chain'].reshape(-1,len(s['parameter_names']))
        if cov=='ezmock1000':
            factor=finite_factors(s['data_dimension'],len(s['parameter_names']))['percival_sigma_factor'];center=np.median(raw,axis=0)
            display=center+(raw-center)*factor
        else:display=raw
        q=np.percentile(display,[16,50,84],axis=0)
        target=np.array([[s['posterior'][p][key] for p in s['parameter_names']] for key in ('q16','q50','q84')])
        quantile_checks[variant]=float(np.max(abs(q-target)));assert np.allclose(q,target,atol=1e-12,rtol=1e-12)
        s['dof']=s['data_dimension']-len(s['parameter_names']);s['pte_gaussian_approx']=float(chi2dist.sf(s['map_chi2'],s['dof']))
        chains[cov+'_'+variant]={'chain':display,'summary':s};results[variant]=s
    plt.rcParams.update({'font.family':'sans-serif','font.sans-serif':['Arial','DejaVu Sans','Liberation Sans'],'pdf.fonttype':42,'ps.fonttype':42,'font.size':10.,'axes.linewidth':1.,'xtick.direction':'in','ytick.direction':'in','xtick.top':True,'ytick.right':True})
    meeting.mkdir(parents=True,exist_ok=True)
    path=meeting/f'task432_hybrid_gic_{cov}_P02xi02_joint_kmax0p08_smin50_baomask80_120.pdf'
    if path.exists():raise FileExistsError('Completed PDF exists')
    with PdfPages(path) as pdf:
        class AnnotatedPDF:
            def savefig(self,figure,**kwargs):
                """固定页脚标注 GIC 与 finite-mock 口径，保留原来的布局与线条颜色。"""
                foot='Hybrid xi + fixed-window GIC; no extra free parameter'
                if cov=='ezmock1000':foot+='; Hartlap likelihood, Percival-rescaled contours'
                figure.text(.5,.018,foot,ha='center',fontsize=7,color='0.35');pdf.savefig(figure,**kwargs)
        title='jaxpower analytic (hybrid xi + GIC)' if cov=='jaxpower' else 'EZmock-1000 (hybrid xi + GIC)'
        triangle_page(AnnotatedPDF(),cov=cov,cov_display=title,chains=chains,ranges=ref['shared_axis_ranges'])
    b={v:{'q50':s['posterior']['b1']['q50'],'maximum_likelihood':s['map_theta'][1],'q16':s['posterior']['b1']['q16'],'q84':s['posterior']['b1']['q84']} for v,s in results.items()}
    between={k:min(b['p02'][k],b['xi02'][k])<=b['joint'][k]<=max(b['p02'][k],b['xi02'][k]) for k in ('q50','maximum_likelihood')}
    old={v:json.loads((BASES[cov]/'fits'/v/'summary.json').read_text()) for v in results}
    changes={v:{p:results[v]['posterior'][p]['q50']-old[v]['posterior'][p]['q50'] for p in ('fNL','b1')} for v in results}
    result={'status':'pass','covariance':cov,'output_pdf':str(path),'output_pdf_sha256':sha(path),'results':results,'b1_comparison':b,'joint_b1_between_marginals':between,
            'median_change_from_missing_gic':changes,'axis_ranges':ref['shared_axis_ranges'],'display_summary_quantile_max_error':quantile_checks,
            'model_contract':'Mandatory deterministic isotropized formal-GIC computed from current hybrid over full fixed RR window; no new free parameter',
            'input_audit':str(out/'input_audit.json'),'postflight_audit':str(out/'postflight_audit.json'),
            'finite_mock_note':'EZmock1000 only: Hartlap likelihood and Percival-rescaled display samples/intervals; raw chains preserved.'}
    save(path.with_suffix('.json'),result);log('plot_done',covariance=cov,path=path,b1=b,between=between)


def main():
    """命令行入口：共享窗口和网格各运行一次，之后每种 covariance 独立更新拟合。"""
    p=argparse.ArgumentParser();p.add_argument('action',choices=['window','grid','prepare','xi02','joint','postflight','plot']);p.add_argument('--cov',choices=list(BASES),default='jaxpower');args=p.parse_args()
    if args.action=='window':validate_window()
    elif args.action=='grid':build_grid()
    elif args.action=='prepare':prepare(args.cov)
    elif args.action in ('xi02','joint'):fit(args.cov,args.action)
    elif args.action=='postflight':postflight(args.cov)
    elif args.action=='plot':plot(args.cov)

if __name__=='__main__':main()
