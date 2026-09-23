#!/usr/bin/env python3
"""Check posterior-domain interpolation and compute direct-GSM ML points."""

# 执行大纲：冻结 9.18 输入 → 检查 hybrid 数值精度 → 三组独立长链 → 直接模型复核 → 原风格 PDF。
# 所有 CPU 数值库单线程，进程限制在同一组最多 8 个 CPU；不提交 Slurm，不覆盖既有主图。
from task432_hybrid_jaxpower_0918_contract import *

def main():
    """对实际后验抽样点核对插值误差，并用直接 GSM 重新优化 xi 与 joint 的最大似然点。"""
    from task432_lightcone_png_velocileptors import LightconePNGVelocileptors,build_cache,RSD_X_SUMMARY,RSD_P_PAYLOAD
    audit=json.loads((OUT/'input_audit.json').read_text())
    with np.load(RSD_P_PAYLOAD,allow_pickle=False) as a:zeff=float(a['zeff'])
    cache=build_cache(zeff=zeff,boxsize=2000.,kmax=3.,ells=(0,2),cosmology='abacus_c000')
    with np.load(RSD_X_SUMMARY,allow_pickle=False) as a:s=a['s']
    with np.load(cache,allow_pickle=False) as a:direct=LightconePNGVelocileptors(a['k_eff'],a['pk_dd'],a['alpha'],float(a['f_growth']),s)
    mask=(s>=50)&(s<350)&~((s>=80)&(s<120))
    p=Engine('p02',audit['emulator_method']);j=Engine('joint',audit['emulator_method'])
    def xv(t,nint=1200):
        """在固定 9.22 hybrid 物理实现下直接求 xi，默认使用 1200 个 streaming 积分点。"""
        poles=direct.evaluate(fnl=float(t[0]),b1=float(t[1]),nint=nint)
        return np.r_[poles[0][mask],poles[2][mask]]
    report={'purpose':'Numerical validation only; unchanged physical hybrid','fits':{}}
    rng=np.random.default_rng(918932)
    for variant in ('xi02','joint'):
        engine=Engine(variant,audit['emulator_method']);root=OUT/'fits'/variant
        summary=json.loads((root/'summary.json').read_text());assert summary['status']=='pass'
        with np.load(root/'samples.npz',allow_pickle=False) as a:flat=a['chain'].reshape(-1,len(engine.names))
        # Evaluate actual posterior points, including the low/high b1 samples.
        ids=rng.choice(len(flat),size=24,replace=False)
        points=flat[ids];tests=[]
        for t in points:
            truth=xv(t);delta=engine.x(t)[0]-truth
            tests.append({'theta':t,'joint_weighted_model_error_chi2':j.metric.chi2(np.r_[np.zeros(22),delta])})
        def prediction(t):
            """根据探针拼接原 P 预测与直接 GSM xi 预测，供精确最大似然复核。"""
            x=xv(t)
            return x if variant=='xi02' else np.r_[p.predict(t)[0],x]
        def residual(t):
            """把直接模型与观测之差白化，返回用于最小二乘优化的残差向量。"""
            return engine.metric.white(engine.data-prediction(t))
        sol=least_squares(residual,np.asarray(summary['map_theta']),bounds=(engine.lower,engine.upper),x_scale='jac',max_nfev=500,ftol=1e-9,xtol=1e-9,gtol=1e-9)
        maximum=max(float(t['joint_weighted_model_error_chi2']) for t in tests)
        item={'posterior_point_checks':tests,'max_model_error_chi2_joint':maximum,'interpolation_gate':maximum<.01,
              'emulated_map_theta':summary['map_theta'],'emulated_map_chi2':summary['map_chi2'],
              'direct_map_theta':sol.x,'direct_map_chi2':sum(sol.fun**2),'direct_optimizer_success':sol.success,'nint':1200}
        report['fits'][variant]=item;log('direct_map',variant=variant,theta=sol.x,chi2=sum(sol.fun**2),max_model_error_chi2=maximum)
        # Preserve both numerical results; use direct ML for figure annotations.
        if maximum<.01 and sol.success:
            summary['emulated_map_theta']=summary['map_theta'];summary['emulated_map_chi2']=summary['map_chi2']
            summary['map_theta']=sol.x;summary['map_chi2']=sum(sol.fun**2)
            summary['map_evaluation']='direct hybrid GSM, Nint=1200; posterior sampled validated emulator'
            save(root/'summary.json',summary)
    report['status']='pass' if all(x['interpolation_gate'] and x['direct_optimizer_success'] for x in report['fits'].values()) else 'failed'
    save(OUT/'postflight_audit.json',report)
    if report['status']!='pass':raise RuntimeError('Posterior interpolation validation failed')

if __name__=='__main__':main()
