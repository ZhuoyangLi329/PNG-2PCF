#!/usr/bin/env python3
"""EZmock1000 定向归因：冻结输入→shot 响应审计→候选模板反事实→RIC 单项边界。

本文件不改 GSM、正式 covariance 或已有主结果；所有新结果保存在独立目录。
拟合与数值库在同一组最多 8 CPU 上运行；把近似探针与已验证修正明确区分。
"""
from __future__ import annotations
import task432_joint_b1_diagnostics as d
from task432_joint_b1_diagnostics import np,Path,json,os,save,sha,log,Metric
from scipy.optimize import least_squares
import argparse,inspect,time,socket

ROOT=d.ROOT;OUT=ROOT/'outputs/task43_outputs/rsd_validation/task432_model_repair/hybrid_gic_joint_b1_cause_tests_v1'
MEETING=ROOT/'9.22meeting/task432_hybrid_gic_joint_b1_cause_tests_ezmock1000'

def initialize():
    """加载 EZmock1000 已冻结模型，审计每次执行的源文件和输入。"""
    d.initialize();OUT.mkdir(parents=True,exist_ok=True)
    global COV,PRIMARY,ANCHORS
    COV=d.COVS['ezmock1000'];PRIMARY=json.loads((d.OUT/'primary.json').read_text())['covariances']['ezmock1000']
    ANCHORS={v:np.array(PRIMARY['baseline'][v]['theta']) for v in d.IDS}
    save(OUT/'input_manifest.json',{'covariance':'ezmock1000','contract':'Frozen 74-vector, x25 mean, C_single, full cross, hybrid+GIC. Diagnostic alternatives are not formal replacements.','inputs':{str(p):sha(p) for p in [d.OUTS['ezmock1000']/'frozen_inputs.npz',d.OUTS['ezmock1000']/'xi_emulator.npz',d.OUT/'primary.json']},'runner_sha256':sha(__file__),'host':socket.gethostname(),'cpu_affinity':sorted(os.sched_getaffinity(0))})

def solve(predict,ids=None,data=None,start=None,xi_sigma=False,fixed_f=False,multi=True):
    """通用 ML：一致 Kaiser 对照允许 xi 使用已有的共享 sigma，sn0 仍仅在 P 侧。"""
    ids=d.IDS['joint'] if ids is None else np.array(ids);data=d.DATA if data is None else np.array(data)
    t0=np.array(d.START if start is None else start);free=[0,1,2,3]
    if not np.any(ids<22):
        free.remove(3)
        if not xi_sigma:free.remove(2)
    if fixed_f:t0[0]=0;free.remove(0)
    free=np.array(free);metric=Metric(COV[np.ix_(ids,ids)])
    def expand(u):
        """自由参数填回完整的四列向量，未使用列不会出现在参数结果中。"""
        t=t0.copy();t[free]=u;return t
    def residual(u):
        """固定 raw C_single 下计算残差；Hartlap 为标量，不改变 ML。"""
        return metric.white(data[ids]-predict(expand(u))[0,ids])
    starts=[t0]
    if multi:
        for delta in (-1,1):
            t=t0.copy();t[0]+=20*delta;t[1]+=.07*delta;t[2]=5.;starts.append(t)
    results=[]
    for s in starts:
        r=least_squares(residual,np.clip(s[free],d.common.LOWER[free]+1e-8,d.common.UPPER[free]-1e-8),bounds=(d.common.LOWER[free],d.common.UPPER[free]),x_scale=np.array([30,.1,3,.1])[free],max_nfev=800,ftol=1e-10,xtol=1e-10,gtol=1e-9);results.append(r)
    best=min(results,key=lambda r:r.fun@r.fun)
    return {'theta':expand(best.x),'parameters':{d.NAMES[i]:expand(best.x)[i] for i in free},'fixed_fNL0':fixed_f,'chi2_Csingle':best.fun@best.fun,'success':bool(best.success),'active_bounds':{d.NAMES[free[i]]:int(v) for i,v in enumerate(best.active_mask) if v},'multistart_chi2':[r.fun@r.fun for r in results],'nfree':len(free),'ndata':len(ids)}

def model_change(predict,label,xi_sigma=False):
    """同参数点的模型差异、线性位移及重优化结果一起保存，避免只比较最佳拟合曲线。"""
    metric=Metric(COV);changes={};response=np.array(PRIMARY['conditional_at_joint_ML']['response'])
    for name,t in ANCHORS.items():
        dm=predict(t)[0]-d.prediction(t)[0]
        changes[name]={'delta_model':dm,'delta_chi2_Csingle':metric.chi2(dm),'linear_parameter_shift_at_joint_J':-response@dm}
    fits={v:solve(predict,ids,start=ANCHORS[v],xi_sigma=xi_sigma) for v,ids in d.IDS.items()}
    return {'label':label,'changes_at_same_theta':changes,'fits':fits,'joint_minus_marginals':[fits['joint']['theta'][1]-fits[v]['theta'][1] for v in ('p02','xi02')]}

def load_p():
    """读取真实 P 窗口的 22 个观测行并复现冻结的 P 理论。"""
    import task43_rsd_ezmock281_mcmc_compare as old
    with np.load(old.PAYLOAD_NPZ,allow_pickle=False) as z:idx=z['fit_bin_indices'][:13];zeff=float(z['zeff'])
    p=old.MEASURE_DIR/'task43_rsd_lightcone_p02_ph000_mesh256_kmax0p300_dk0p002.npz'
    with np.load(p,allow_pickle=False) as z:
        rows=np.r_[idx,150+idx[4:]];win=z['window_matrix'][rows];k=z['theory_k'];ell=z['theory_ell']
    pm=old.WindowConvolvedP02Model(win,k,ell,zeff=zeff)
    assert max(np.max(abs(pm.evaluate(t)-d.prediction(t)[0,:22])) for t in ANCHORS.values())<1e-7
    return old,pm,p

def shot():
    """独立核对权重和归一化，比较当前、全理论支持及观测白噪声接触模板。"""
    old,pm,source=load_p();rows=[]
    for i in range(25):
        p=old.MEASURE_DIR/f'task43_rsd_lightcone_p02_ph{i:03d}_mesh256_kmax0p300_dk0p002.npz';meta=json.loads(p.with_suffix('.json').read_text())
        D=meta['data'];R=meta['random'];alpha=D['weight_sum']/R['weight_sum'];shotnum=D['weight2_sum']+alpha**2*R['weight2_sum']
        with np.load(p,allow_pickle=False) as z:
            norm=float(np.ravel(z['norm_ell0'])[0]);num=np.array(z['num_shotnoise_ell0']);sn=np.array(z['shotnoise_ell0']);sn2=np.array(z['shotnoise_ell2'])
            nerr=float(np.max(abs(num-shotnum))/max(abs(shotnum),1));serr=float(np.max(abs(sn-shotnum/norm))/max(abs(shotnum/norm),1))
        rows.append({'phase':i,'alpha':alpha,'norm':norm,'analytic_shot_numerator':shotnum,'stored_numerator_minmax':[num.min(),num.max()],'shotnoise_minmax':[sn.min(),sn.max()],'ell2_shot_maxabs':np.max(abs(sn2)),'numerator_relative_error':nerr,'shot_relative_error':serr,'metadata_path':str(p.with_suffix('.json')),'metadata_sha256':sha(p.with_suffix('.json'))})
    current=d.MODEL.shot;full=pm.window@np.where(pm.theory_ell==0,1e4,0.);contact=np.r_[np.full(13,1e4),np.zeros(9)]
    # contact 是各向同性白噪声在理想连续估计器上的自配对响应；非 Poisson halo 随机性不据此被判定为白噪声。
    templates={'current_window_cut':current,'window_no_kcut':full,'observed_contact':contact}
    report={'metadata_audit':rows,'model_sn0_convention':'sn0*1e4 in P units; fitted residual after the estimator Poisson subtraction','baseline_sn0':{v:float(ANCHORS[v][3]) for v in ('p02','joint')},'templates':templates,'fits':{},'scope':'Template probes do not establish the shape of non-Poisson halo stochasticity. No constant is inserted into finite-separation xi.'}
    for name,S in templates.items():
        def predict(t):
            """只替换已有 sn0 参数的 P 响应，不增加任何参数。"""
            t=np.atleast_2d(t);y=d.prediction(t);y[:,:22]+=t[:,3,None]*(S-current);return y
        report['fits'][name]=model_change(predict,name)
    report['gates']={'weight_numerator_matches':max(r['numerator_relative_error'] for r in rows)<1e-6,'normalized_shot_matches':max(r['shot_relative_error'] for r in rows)<1e-6}
    save(OUT/'shot_response.json',report);log('shot_done',gates=report['gates'],b1={k:v['fits']['joint']['theta'][1] for k,v in report['fits'].items()})

def ric_probe():
    """量化旧 boxsafe 单自项 RIC 探针的能力边界，不把它称为完整径向 IC 修正。"""
    old,pm,source=load_p();paths=list((ROOT/'outputs/task43_outputs/rsd_validation/lightcone_boxsafe_zobs0p4_0p8/ric_singleterm/operators').glob('*.npz'));assert len(paths)==1
    with np.load(paths[0],allow_pickle=False) as z:
        keys=z.files;rk=z['pk_theory_k'];re=z['pk_theory_ell'];ric=z['pk_ric_matrix'];meta=json.loads(str(z['meta_json'].item())) if 'meta_json' in z else {}
        arrays={k:z[k] for k in keys if 'xi' in k or 'basis' in k}
    assert np.array_equal(rk,pm.theory_k) and np.array_equal(re,pm.theory_ell)
    R=np.zeros_like(pm.window);R[:13]=ric[:13]
    from scipy.interpolate import CubicSpline
    basis=np.array([R@pm._theory_basis(float(s)) for s in pm.sigma_grid]);sp=CubicSpline(pm.sigma_grid,basis,axis=0)
    def delta(t):
        """原单自项算子仅作用于物理 clustering，不把 sn0 也按 RIC 扣除。"""
        t=np.atleast_2d(t);f,b,s,n=t.T;q=f*2*1.686*(b-1);fg=pm.f_growth
        co=np.column_stack([b*b,2*b*q,q*q,2*b*fg,2*q*fg,np.full(len(t),fg*fg)])
        return -np.einsum('nij,nj->ni',sp(s),co)
    def predict(t):
        """保持原 hybrid+GIC，仅添加已存旧 P0 单项探针。"""
        t=np.atleast_2d(t);y=d.prediction(t);y[:,:22]+=delta(t);return y
    report=model_change(predict,'legacy_boxsafe_single_auto_P0_RIC_only')
    report.update({'operator':str(paths[0]),'operator_sha256':sha(paths[0]),'operator_keys':keys,'metadata':meta,'additional_array_shapes':{k:v.shape for k,v in arrays.items()},'scope':'Incomplete single radial-auto term, theory ell0 to observed P0 only; no cross terms or P2/xi completion. Diagnostic bound, never a formal fix.','no_new_parameters':True})
    save(OUT/'ric_existing_probe.json',report);log('ric_probe_done',theta=report['fits']['joint']['theta'])

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('action',choices=['shot','ric']);a=p.parse_args();initialize()
    if a.action=='shot':shot()
    else:ric_probe()
