#!/usr/bin/env python3
"""full-RIC之后的低维条件模式诊断，仅复用已有向量，不改正式拟合。

大纲：锁定输入 -> 固定fiducial导数 -> nuisance边缘化的b1响应 -> 预定义
5个C正交模式 -> halo/EZ散布和均值校准 -> 输出响应算子供FKP配对使用。
模式定义不使用halo残差；只有25个halo，绝不反演其74维样本协方差。
"""
import os
for key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):
    os.environ[key]='1'
import json, sys
from pathlib import Path
import numpy as np
from scipy.optimize import least_squares
from scipy.stats import chi2
ROOT=Path('/pscratch/sd/l/lzy/PNG-2PCF-model-to-observe')
sys.path.insert(0,str(ROOT/'codes/task432'))
from task432_full_ric_compile import Engine, common
from task432_full_ric_geometry import BASE, sha
from task432_full_ric_backend import OUT as RIC
OUT=BASE.parent/'fkp_conditional_diagnostics'
COMPILED=RIC/'pilot/mean4xi_mean2P_prodrr_stochastic/engine.npz'
THETA=np.array([0.,2.42,2.2,.08])
STEPS=np.array([.02,.0001,.003,.0001])

def jacobian(engine, theta, steps=STEPS):
    """中心差分模型导数，输入4参数fiducial，输出74×4；可用半步复核。"""
    points=np.tile(theta,(8,1))
    for i,h in enumerate(steps):points[2*i,i]+=h;points[2*i+1,i]-=h
    y=engine.predict(points)
    return np.column_stack([(y[2*i]-y[2*i+1])/(2*h) for i,h in enumerate(steps)])

def response(J,C,ids,parameters):
    """对指定边缘协方差/参数作局部GLS，返回嵌入74维的b1响应。"""
    W=common.Metric(C[np.ix_(ids,ids)]).whitener
    jw=W@J[np.ix_(ids,parameters)]
    # 按列规范化再SVD，避免fNL与sn0单位差及弱xi-shot响应造成数值病态。
    scales=np.linalg.norm(jw,axis=0)
    rr=(np.linalg.pinv(jw/scales,rcond=1e-12)/scales[:,None])@W
    full=np.zeros((len(parameters),74));full[:,ids]=rr
    return full[list(parameters).index(1)],full

def main():
    """执行可复现的5维检验，写小型json/npz；bootstrap仅作有限样本校准。"""
    OUT.mkdir(parents=True,exist_ok=True)
    e=Engine('joint',COMPILED);C=e.cov;J=jacobian(e,THETA);mu=e.predict(THETA)[0]
    mockpath=BASE.parent/'hybrid_ezmock1000_0918_contract_v1/ezmock1000_covariance_and_stack.npz'
    phasepath=BASE.parent/'hybrid_gic_joint_b1_diagnostics/paired_phase_data.npz'
    with np.load(mockpath) as a:ez=a['stack'];indices=a['production_indices']
    with np.load(phasepath) as a:halo=a['data']
    assert ez.shape==(1000,74) and halo.shape==(25,74)
    assert np.array_equal(indices,np.arange(1000))
    assert np.allclose(halo.mean(0),e.data,rtol=1e-9,atol=1e-10)
    assert np.allclose(np.cov(ez,rowvar=False),C,rtol=1e-10,atol=1e-14)
    gj,rj=response(J,C,np.arange(74),[0,1,2,3])
    gp,rp=response(J,C,np.arange(22),[0,1,2,3])
    gx,rx=response(J,C,np.arange(22,74),[0,1,3])
    B=np.array([gj-gp,gj-gx]);BC=B@C@B.T
    wp=common.Metric(C[:22,:22]).whitener;wx=common.Metric(C[22:,22:]).whitener
    cross=wp@C[:22,22:]@wx.T;u,s,vt=np.linalg.svd(cross,full_matrices=False)
    A=np.linalg.solve(C[:22,:22],C[:22,22:]).T
    cond=np.column_stack([-A,np.eye(52)])
    candidates=[*B,gj,*[vt[i]@wx@cond/np.sqrt(1-s[i]**2) for i in range(2)]]
    labels=['joint_minus_P_b1','joint_minus_xi_b1_orthogonal','joint_b1_orthogonal',
            'conditional_canonical1_orthogonal','conditional_canonical2_orthogonal']
    modes=[]
    for v in candidates:
        v=v.copy()
        for q in modes:v-=float(v@C@q)*q
        norm=np.sqrt(v@C@v);assert norm>1e-10
        modes.append(v/norm)
    M=np.array(modes);assert np.max(abs(M@C@M.T-np.eye(5)))<1e-8
    h=(halo-mu)@M.T;z=(ez-ez.mean(0))@M.T;hc=np.cov(h,rowvar=False)
    rng=np.random.default_rng(4322401);nboot=10000
    # 分批抽样25个真实EZ，校准小样本方差；不会把各phase当独立拟合显著性。
    variance_draws=[];mean_draws=[]
    for _ in range(20):
        samples=z[rng.integers(0,1000,(nboot//20,25))]
        variance_draws.append(samples.var(1,ddof=1));mean_draws.append(samples.mean(1))
    variance_draws=np.concatenate(variance_draws);mean_draws=np.concatenate(mean_draws)
    hb=(halo-mu)@B.T;mean_b=hb.mean(0);emp_b=np.cov(hb,rowvar=False)
    statistic=float(25*mean_b@np.linalg.solve(BC,mean_b))
    # 旧模型结论不直接转移；这里只报告当前full-RIC的局部线性均值偏移检验。
    be=(ez-ez.mean(0))@B.T;null=[]
    for _ in range(20):
        m=be[rng.integers(0,1000,(500,25))].mean(1)
        null.extend(25*np.einsum('ni,ij,nj->n',m,np.linalg.inv(BC),m))
    current={}
    for v in ('p02','xi02','joint'):
        a=json.loads((RIC/'formal_mean4'/v/'summary.json').read_text())
        current[v]={'theta':a['map_theta'],'b1':a['map_theta'][1],'chi2':a['map_raw_chi2']}
    report={'status':'pass','theta_anchor':THETA,'mode_selection':'Fixed fiducial derivatives and EZ covariance only; no halo residual ranking',
        'labels':labels,'n_halo':25,'n_ez':1000,'seed':4322401,'canonical_correlations':s,
        'mean_modes':h.mean(0),'mean_modes_in_EZ_mean_sigma':5*h.mean(0),
        'halo_mode_covariance':hc,'halo_variance_ratios':np.diag(hc),
        'EZ_25sample_variance_95_interval':np.percentile(variance_draws,[2.5,97.5],axis=0),
        'variance_lower_tail_frequency':(1+(variance_draws<=np.diag(hc)).sum(0))/(nboot+1),
        'mean_b1_contrasts_linear':mean_b,'contrast_covariance_Csingle':BC,'contrast_empirical_covariance':emp_b,
        'contrast_variance_ratios':np.diag(emp_b)/np.diag(BC),
        'contrast_mean_mahalanobis_EZ':statistic,'contrast_mean_chi2_2df_reference':chi2.sf(statistic,2),
        'EZ_bootstrap_mean_contrast_tail':(1+np.count_nonzero(np.asarray(null)>=statistic))/(nboot+1),
        'local_common_model_null_error':float(np.max(abs(B@J))),
        'derivative_halfstep_relative_error':float(np.max(np.linalg.norm(jacobian(e,THETA,STEPS/2)-J,axis=0)/np.linalg.norm(J,axis=0))),
        'mode_covariance_identity_error':float(np.max(abs(M@C@M.T-np.eye(5)))),
        'current_formal_MAP':current,'fit_weight':'C_single unchanged; divide by25 only for mean-noise diagnostics',
        'caveat':'Linear contrasts assume fiducial tangent model. Modes3-5 means are not postfit p values. EZ bootstrap conditions on estimated covariance; 25 halo sample covariance is not inverted.',
        'inputs':{str(p):sha(p) for p in (COMPILED,mockpath,phasepath,BASE/'frozen_inputs.npz')},
        'source_sha256':sha(__file__),'cpu_affinity':sorted(os.sched_getaffinity(0))}
    common.save(OUT/'conditional_modes.json',report)
    np.savez_compressed(OUT/'conditional_modes.npz',modes=M,contrast_response=B,joint_response=gj,p_response=gp,xi_response=gx,
        covariance=C,data=e.data,model_anchor=mu,jacobian=J,theta_anchor=THETA,halo_stack=halo,ez_stack=ez,
        halo_modes=h,ez_modes=z,mode_labels=labels,canonical_correlations=s)
    print(json.dumps({'status':'pass','mode_variance_ratios':np.diag(hc).tolist(),'b1_contrast_variance_ratios':report['contrast_variance_ratios'].tolist(),
        'b1_contrasts':mean_b.tolist(),'mean_contrast_T':statistic,'EZ_mean_tail':report['EZ_bootstrap_mean_contrast_tail']}),flush=True)

if __name__=='__main__':main()
