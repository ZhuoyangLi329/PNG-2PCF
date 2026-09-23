#!/usr/bin/env python3
"""固定 hybrid+GIC 的 joint b1 诊断，不改写主结果。

执行大纲：冻结/复现 → 交叉块扫描 → 参数共享/尺度/多极矩 → Schur 归因 →
1000 mock bootstrap → 同模型噪声闭合 → 25 phase/leave-one → 观测算子检查。
所有拟合使用 C_single；只在明确标注的模拟中以 C_single/25 生成均值噪声。
"""
from __future__ import annotations
import task432_hybrid_jaxpower_0918_contract as common
from task432_hybrid_jaxpower_0918_contract import np, Path, json, os, save, sha, log, Metric
from task432_hybrid_gic_0918_contract import configure, OUTS, BASES, finite_factors
from scipy.optimize import least_squares
from scipy.linalg import solve_triangular
import argparse, time, socket, sys

ROOT=common.ROOT
OUT=ROOT/'outputs/task43_outputs/rsd_validation/task432_model_repair/hybrid_gic_joint_b1_diagnostics'
GROUPS={'P0':np.arange(13),'P2':np.arange(13,22),'xi0':np.arange(22,48),'xi2':np.arange(48,74)}
IDS={'p02':np.arange(22),'xi02':np.arange(22,74),'joint':np.arange(74)}
NAMES=['fNL','b1','sigma_s_P','sn0']
START=np.array([0.,2.42,2.2,.08])
MODEL=None; DATA=None; COVS={}; FROZEN={}

def initialize():
    """按用户最新要求，只读加载 EZmock1000 已冻结输入，后续仅分析该协方差。"""
    global MODEL, DATA, COVS, FROZEN
    OUT.mkdir(parents=True,exist_ok=True)
    MODEL=configure('ezmock1000')('joint','cubic')
    DATA=MODEL.data.copy()
    for c,p in OUTS.items():
        if c!='ezmock1000':continue
        with np.load(p/'frozen_inputs.npz',allow_pickle=False) as z:
            COVS[c]=z['covariance'];FROZEN[c]={k:z[k] for k in z.files}
            assert np.array_equal(np.r_[z['data_p'],z['data_x']],DATA)
    assert DATA.shape==(74,) and COVS['ezmock1000'].shape==(74,74)

def prediction(theta,mode='shared'):
    """参数共享或拆分时仅改变参数映射，保持 hybrid+GIC 动力学和观测向量不变。"""
    t=np.atleast_2d(theta)
    if mode=='shared':return MODEL.predict(t)
    if mode=='split_b':
        p=t[:,[0,1,3,4]];x=t[:,[0,2]]
    elif mode=='split_f':
        p=t[:,[0,2,3,4]];x=t[:,[1,2]]
    else:raise ValueError(mode)
    return np.column_stack([MODEL.predict(p)[:,:22],MODEL.x(x)])

def fit(cov,ids=None,data=None,mode='shared',fixed=None,start=None,multi=False,predict=None):
    """协方差先取子块再求逆；固定参数从优化空间移除，返回 ML 与局部 Fisher 误差。"""
    ids=IDS['joint'] if ids is None else np.asarray(ids)
    d=DATA if data is None else np.asarray(data)
    fixed={} if fixed is None else dict(fixed)
    func=(lambda t:prediction(t,mode)) if predict is None else predict
    if mode=='shared':
        names=NAMES.copy();lo=common.LOWER.copy();hi=common.UPPER.copy();x0=START.copy();scale=np.array([30,.1,3,.1])
        if not np.any(ids<22):fixed.update({2:START[2],3:START[3]})
    elif mode=='split_b':
        names=['fNL','b1_P','b1_xi','sigma_s_P','sn0'];lo=np.array([-500,.5,.5,0,-1.]);hi=np.array([500,5,5,30,1.]);x0=np.array([0,2.42,2.42,2.2,.08]);scale=np.array([30,.1,.1,3,.1])
    else:
        names=['fNL_P','fNL_xi','b1','sigma_s_P','sn0'];lo=np.array([-500,-500,.5,0,-1.]);hi=np.array([500,500,5,30,1.]);x0=np.array([0,0,2.42,2.2,.08]);scale=np.array([30,30,.1,3,.1])
    if start is not None:x0=np.asarray(start).copy()
    for k,v in fixed.items():x0[k]=v
    free=np.array([i for i in range(len(x0)) if i not in fixed]);metric=Metric(cov[np.ix_(ids,ids)])
    def unpack(u):
        """将自由参数重新放回完整参数向量。"""
        t=x0.copy();t[free]=u;return t
    def residual(u):
        """返回同一 C_single 度量下的残差。"""
        return metric.white(d[ids]-func(unpack(u))[0,ids])
    starts=[x0.copy()]
    if multi:
        for delta in (-1,1):
            s=x0.copy();s[0]+=delta*25;s[1]+=delta*.08
            if len(x0)==4:s[2]=5.
            starts.append(s)
    solutions=[]
    for s in starts:
        r=least_squares(residual,np.clip(s[free],lo[free]+1e-8,hi[free]-1e-8),bounds=(lo[free],hi[free]),x_scale=scale[free],max_nfev=450,ftol=3e-10,xtol=3e-10,gtol=3e-9)
        solutions.append(r)
    best=min(solutions,key=lambda r:r.fun@r.fun);t=unpack(best.x)
    fisher=best.jac.T@best.jac;fcov=np.linalg.pinv(fisher,rcond=1e-12)
    errs=np.zeros(len(x0));errs[free]=np.sqrt(np.maximum(0,np.diag(fcov)))
    fullfcov=np.zeros((len(x0),len(x0)));fullfcov[np.ix_(free,free)]=fcov
    return {'names':names,'theta':t,'chi2':best.fun@best.fun,'ndata':len(ids),'nfree':len(free),'ids':ids,
            'success':bool(best.success),'optimality':best.optimality,'nfev':best.nfev,'local_sigma_Csingle':errs,
            'local_parameter_covariance':fullfcov,'multistart_chi2':[r.fun@r.fun for r in solutions],
            'active_bounds':{names[free[i]]:int(v) for i,v in enumerate(best.active_mask) if v},'fixed':{names[k]:v for k,v in fixed.items()}}

def jacobian(t):
    """批量中心差分导数，用于局部响应归因；步长远小于后验尺度。"""
    steps=np.array([.02,.0001,.003,.0001]);ts=np.tile(t,(8,1))
    for i in range(4):ts[2*i,i]+=steps[i];ts[2*i+1,i]-=steps[i]
    y=prediction(ts)
    return np.column_stack([(y[2*i]-y[2*i+1])/(2*steps[i]) for i in range(4)])

def conditional(cov,t):
    """Schur 条件残差、典型相关模态及线性 b1 响应；归因依赖于选择的线性化点。"""
    metric=Metric(cov);r=DATA-prediction(t)[0];J=jacobian(t)
    JW=metric.white(J.T).T;rw=metric.white(r)
    fcov=np.linalg.inv(JW.T@JW);response=fcov@JW.T@metric.whitener
    delta=response@r;contribution=response[1]*r
    scale=np.sqrt(np.diag(cov));R=cov/np.outer(scale,scale)
    Lp=np.linalg.cholesky(R[:22,:22]);Lx=np.linalg.cholesky(R[22:,22:])
    B=solve_triangular(Lp,R[:22,22:],lower=True)
    B=solve_triangular(Lx,B.T,lower=True).T
    u,s,vt=np.linalg.svd(B,full_matrices=True)
    yp=solve_triangular(Lp,r[:22]/scale[:22],lower=True)
    yx=solve_triangular(Lx,r[22:]/scale[22:],lower=True)
    S=np.eye(52)-B.T@B;vals,vec=np.linalg.eigh(S)
    cond=yx-B.T@yp;z=vec.T@cond/np.sqrt(vals)
    schur=yp@yp+z@z
    assert abs(schur-metric.chi2(r))<1e-8
    jp=solve_triangular(Lp,J[:22]/scale[:22,None],lower=True)
    jx=solve_triangular(Lx,J[22:]/scale[22:,None],lower=True)
    jc=(vec.T@(jx-B.T@jp))/np.sqrt(vals)[:,None]
    # 参数响应分为边际 P 模态和给定 P 后的 xi 模态，二者与整体白化等价。
    conditional_b=(fcov[1]@jc.T)*z
    marginal_b=(fcov[1]@jp.T)*yp
    assert abs(conditional_b.sum()+marginal_b.sum()-delta[1])<1e-8
    return {'anchor_theta':t,'chi2_full':metric.chi2(r),'chi2_P':yp@yp,'chi2_xi_given_P':z@z,
            'schur_identity_error':schur-metric.chi2(r),'canonical_correlations':s,'conditional_eigenvalues':vals,
            'conditional_residual_modes':z,'conditional_mode_b1_contribution':conditional_b,
            'marginal_P_b1_contribution':marginal_b.sum(),'conditional_xi_b1_contribution':conditional_b.sum(),
            'linear_delta_theta':delta,'linear_predicted_theta':t+delta,'b1_contribution_per_original_bin':contribution,
            'b1_contribution_by_group':{k:contribution[ii].sum() for k,ii in GROUPS.items()},
            'whitened_residual':rw,'residual':r,'jacobian':J,'response':response}

def primary():
    """第一轮扫描和多起点复现，所有变体仅作为诊断，不替换基准模型。"""
    report={'contract':'Frozen hybrid+GIC; x25 mean; raw C_single; all estimates ML, not MCMC medians','covariances':{}}
    for c,cov in COVS.items():
        rr={'baseline':{},'cross_scan':[],'fixed_fNL0':{},'fixed_P_nuisance':{},'multipoles':{},'scales':{}}
        for v,ids in IDS.items():
            rr['baseline'][v]=fit(cov,ids,multi=True)
            old=json.loads((OUTS[c]/'fits'/v/'summary.json').read_text())
            expected=np.asarray(old['map_theta'])
            diff=rr['baseline'][v]['theta'][:len(expected)]-expected
            assert abs(diff[1])<5e-4,(c,v,diff)
        for lam in np.linspace(0,1,5):
            cc=cov.copy();cc[:22,22:]*=lam;cc[22:,:22]*=lam
            res=fit(cc,multi=True);res['lambda']=lam
            res['min_correlation_eigenvalue']=np.linalg.eigvalsh(cc/np.sqrt(np.outer(np.diag(cc),np.diag(cc)))).min()
            rr['cross_scan'].append(res)
        for v,ids in IDS.items():rr['fixed_fNL0'][v]=fit(cov,ids,fixed={0:0},multi=True)
        pt=rr['baseline']['p02']['theta'];jt=rr['baseline']['joint']['theta']
        for label,fixed in [('sigma_at_P',{2:pt[2]}),('sn_at_P',{3:pt[3]}),('both_at_P',{2:pt[2],3:pt[3]})]:rr['fixed_P_nuisance'][label]=fit(cov,fixed=fixed,multi=True)
        for mode in ('split_b','split_f'):
            ss=fit(cov,mode=mode,multi=True);V=ss['local_parameter_covariance']
            i,j=(1,2) if mode=='split_b' else (0,1)
            ss['difference']=ss['theta'][i]-ss['theta'][j];ss['difference_local_sigma_Csingle']=np.sqrt(V[i,i]+V[j,j]-2*V[i,j])
            ss['delta_chi2_vs_shared']=rr['baseline']['joint']['chi2']-ss['chi2'];rr[mode]=ss
        for label,groups in [('P0_xi0',['P0','xi0']),('P02_xi0',['P0','P2','xi0']),('P0_xi02',['P0','xi0','xi2']),('P02_xi02',list(GROUPS))]:
            ids=np.sort(np.concatenate([GROUPS[g] for g in groups]));rr['multipoles'][label]=fit(cov,ids,multi=True)
        for label,groups in [('P0',['P0']),('xi0',['xi0']),('xi2',['xi2'])]:rr['multipoles'][label]=fit(cov,np.concatenate([GROUPS[g] for g in groups]),multi=True)
        edges=FROZEN[c]['p_edges'];s=FROZEN[c]['xi_centers']
        for kmax in (.05,.06,.08):
            # P0 的边界是 13x2；P2 取原索引 4..12。
            kp=np.r_[edges[:,1]<=kmax+1e-10,edges[4:,1]<=kmax+1e-10]
            for smin in (50,80,100,150):
                ids=np.flatnonzero(np.r_[kp,s>=smin,s>=smin]);label=f'k{kmax:.2f}_s{smin}'
                rr['scales'][label]=fit(cov,ids,multi=False,start=jt)
        rr['conditional_at_P_ML']=conditional(cov,pt)
        rr['conditional_at_joint_ML']=conditional(cov,jt)
        rr['finite_mock_note']='Raw C_single chi2 and local Fisher errors; multiply chi2 by variant-specific Hartlap for EZ1000. ML unchanged. No Percival intervals claimed.'
        report['covariances'][c]=rr;save(OUT/'primary.json',report)
        log('primary_cov_done',covariance=c,b1={v:r['theta'][1] for v,r in rr['baseline'].items()},scan=[r['theta'][1] for r in rr['cross_scan']])
    save(OUT/'primary.json',report)

def covariance_bootstrap(n=200):
    """从完整 1000 配对 mock 有放回重采样，量化估计噪声而非调整协方差到期望结果。"""
    path=BASES['ezmock1000']/'ezmock1000_covariance_and_stack.npz'
    with np.load(path,allow_pickle=False) as z:stack=z['stack'];indices=z['production_indices']
    assert np.array_equal(indices,np.arange(1000)) and stack.shape==(1000,74)
    assert np.allclose(np.cov(stack,rowvar=False),COVS['ezmock1000'],rtol=1e-10,atol=1e-14)
    rng=np.random.default_rng(4322301);rows=[]
    for i in range(n):
        cc=np.cov(stack[rng.integers(0,1000,1000)],rowvar=False)
        fits={v:fit(cc,ids) for v,ids in IDS.items()}
        bb=[fits[v]['theta'][1] for v in IDS]
        rows.append({'b1':bb,'theta_joint':fits['joint']['theta'],'chi2':fits['joint']['chi2'],'outside_low':bb[2]<min(bb[:2]),'success':all(f['success'] for f in fits.values())})
        if (i+1)%25==0:log('bootstrap',completed=i+1,total=n)
    save(OUT/'covariance_bootstrap.json',{'seed':4322301,'n':n,'stack_path':str(path),'stack_sha256':sha(path),'rows':rows,'scope':'Nonparametric covariance-estimation sensitivity conditional on these 1000 paired mocks; not cosmological truth or independent realizations.'})

def calibration(n=300):
    """相关 Gaussian 注入在同一模型下检验 ML 外移概率；mean25 噪声需要独立 phase 假设。"""
    pri=json.loads((OUT/'primary.json').read_text());rng=np.random.default_rng(4322302)
    report={'seed':4322302,'n_each':n,'cases':{},'scope':'Fitting always uses C_single. mean25 generation assumes 25 independent phases with the chosen covariance. Same-model closure cannot establish physical model correctness.'}
    for c,cov in COVS.items():
        anchor=np.array([0.,2.42,2.15,.08]);truth=prediction(anchor)[0];L=np.linalg.cholesky(cov)
        closure={v:fit(cov,ids,data=truth,start=anchor,multi=True) for v,ids in IDS.items()}
        for v,r in closure.items():assert np.max(abs(r['theta'][:2]-anchor[:2]))<1e-4
        for denom in (25,1):
            rows=[]
            for i in range(n):
                d=truth+L@rng.normal(size=74)/np.sqrt(denom)
                fits={v:fit(cov,ids,data=d,start=anchor) for v,ids in IDS.items()}
                bb=np.array([fits[v]['theta'][1] for v in IDS]);rows.append({'b1':bb,'shift':bb[2]-bb[:2],'chi2_joint':fits['joint']['chi2'],'active_bounds':{v:r['active_bounds'] for v,r in fits.items() if r['active_bounds']},'success':all(r['success'] for r in fits.values())})
                if (i+1)%50==0:log('calibration',covariance=c,noise_divisor=denom,completed=i+1,total=n)
            obs=np.array([pri['covariances'][c]['baseline'][v]['theta'][1] for v in IDS]);shift=obs[2]-obs[:2]
            shifts=np.array([r['shift'] for r in rows]);mu=shifts.mean(axis=0);V=np.cov(shifts,rowvar=False);iv=np.linalg.inv(V)
            stat=(shift-mu)@iv@(shift-mu);sim=np.einsum('ni,ij,nj->n',shifts-mu,iv,shifts-mu)
            same=np.all(shifts<0,axis=1);aslow=np.all(shifts<=shift,axis=1)
            report['cases'][f'{c}_noise_div{denom}']={'covariance':c,'noise_divisor':denom,'anchor':anchor,'noisefree':closure,'rows':rows,'observed_shift':shift,'mean_shift':mu,'shift_covariance':V,'observed_2d_statistic':stat,'empirical_2d_tail_fraction':(1+np.sum(sim>=stat))/(n+1),'outside_low_fraction':same.mean(),'at_least_as_low_both_fraction':(1+aslow.sum())/(n+1),'observed_joint_chi2':pri['covariances'][c]['baseline']['joint']['chi2'],'sim_joint_chi2_quantiles':np.percentile([r['chi2_joint'] for r in rows],[5,50,95]),'chi2_upper_tail_fraction':(1+sum(r['chi2_joint']>=pri['covariances'][c]['baseline']['joint']['chi2'] for r in rows))/(n+1)}
            save(OUT/'calibration.json',report)

def load_phase_data():
    """对照冻结口径重建 25 个配对 phase 数据，均值必须逐元素吻合。"""
    import task43_rsd_ezmock281_mcmc_compare as old
    with np.load(old.PAYLOAD_NPZ,allow_pickle=False) as z:idx=z['fit_bin_indices'][:13]
    ps=[];paths=[]
    for i in range(25):
        p=old.MEASURE_DIR/f'task43_rsd_lightcone_p02_ph{i:03d}_mesh256_kmax0p300_dk0p002.npz'
        with np.load(p,allow_pickle=False) as z:ps.append(np.r_[z['pk0'][idx],z['pk2'][idx[4:]]])
        paths.append(p)
    with np.load(old.ELL2_SUMMARY,allow_pickle=False) as z:
        xx=z['xi_multipoles_by_phase'];s=z['s'];mask=(s>=50)&~((s>=80)&(s<120));xs=xx[:,:,mask].reshape(25,52)
        zeff=z['zeff_by_phase'] if 'zeff_by_phase' in z else None
    data=np.column_stack([ps,xs]);err=np.max(abs(data.mean(axis=0)-DATA)/np.maximum(abs(DATA),1e-8))
    assert np.allclose(data.mean(axis=0),DATA,rtol=1e-9,atol=1e-10),err
    return data,{'p_files':[{'path':str(p),'sha256':sha(p)} for p in paths],'xi_file':str(old.ELL2_SUMMARY),'xi_sha256':sha(old.ELL2_SUMMARY),'mean_max_relative_error':err,'zeff_by_phase':zeff}

def phases():
    """单 phase 和 leave-one 均值拟合；25 样本不用于反演 74 维经验协方差。"""
    stack,meta=load_phase_data();report={'metadata':meta,'covariances':{},'scope':'Same mean forward operator applied to each phase; no phase-specific geometry refit. Leave-one fits are strongly correlated. Empirical 25-phase covariance is never inverted.'}
    np.savez_compressed(OUT/'paired_phase_data.npz',data=stack)
    for c,cov in COVS.items():
        rows=[];leave=[]
        for i,d in enumerate(stack):
            rr={v:fit(cov,ids,data=d,multi=True) for v,ids in IDS.items()};rows.append(rr)
            dd=(25*DATA-d)/24;leave.append({v:fit(cov,ids,data=dd) for v,ids in IDS.items()})
        t=np.array(json.loads((OUT/'primary.json').read_text())['covariances'][c]['baseline']['joint']['theta'])
        J=jacobian(t);metric=Metric(cov);WJ=metric.white(J.T).T;response=np.linalg.solve(WJ.T@WJ,WJ.T)@metric.whitener
        emp=np.cov(stack,rowvar=False);R=metric.white(stack-DATA)
        report['covariances'][c]={'phase_fits':rows,'leave_one_fits':leave,'empirical_projected_parameter_covariance':response@emp@response.T,'assumed_projected_parameter_covariance':response@cov@response.T,'whitened_scatter_trace_per_dimension':np.trace(np.cov(R,rowvar=False))/74,'phase_b1':[[r[v]['theta'][1] for v in IDS] for r in rows],'leave_one_b1':[[r[v]['theta'][1] for v in IDS] for r in leave]}
        save(OUT/'phases.json',report);log('phases_done',covariance=c)

def manifest():
    """保存输入代码指纹和执行口径，保证原主结果未被更改。"""
    save(OUT/'run_manifest.json',{'host':socket.gethostname(),'cpu_affinity':sorted(os.sched_getaffinity(0)),'source':str(Path(__file__).resolve()),'source_sha256':sha(__file__),'current_scope':'EZmock1000 only; earlier shared-scope computations preserved in provenance','inputs':{c:{str(p/name):sha(p/name) for name in ('frozen_inputs.npz','xi_emulator.npz','input_audit.json')} for c,p in OUTS.items() if c=='ezmock1000'},'formal_contract':'9.18 P0 13 + P2 9 + xi0 26 + xi2 26; mandatory scalar GIC; C_single; diagnostics retain originals','outputs':[str(p) for p in OUT.glob('*.json')]})

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('action',choices=['primary','bootstrap','calibration','phases','manifest']);p.add_argument('--n',type=int);a=p.parse_args();initialize();started=time.monotonic()
    if a.action=='primary':primary()
    elif a.action=='bootstrap':covariance_bootstrap(a.n or 200)
    elif a.action=='calibration':calibration(a.n or 300)
    elif a.action=='phases':phases()
    manifest();log('action_done',action=a.action,seconds=time.monotonic()-started)
