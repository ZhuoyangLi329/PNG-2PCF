#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""900 EZmock 联合拟合 b1 下移的受控诊断，不修改现有链或主图。

大纲：读取已冻结的数据/模型与 900 covariance -> 标准化 Cholesky 拟合
-> 审计旧 precision 平方根 -> 关闭 P-xi cross、分离两侧 FoG 参数
-> 固定 FoG 扫描、条件似然分解 -> 将结果写到独立 diagnostics JSON。
所有拟合使用单个 survey covariance；Hartlap 为常数，不改变 ML 点。
这些小实验只做 ML/profile 诊断，不替代正式 MCMC 或 science closure。
"""
import json
from pathlib import Path
import numpy as np
from scipy.linalg import solve_triangular
from scipy.optimize import least_squares
import task43_rsd_ezmock281_mcmc_compare as r

ROOT = r.PROJECT_ROOT
RUN = ROOT / 'outputs/task43_outputs/ezmock_rsd_lightcone_covariance_z0p4_0p8_x1000_common50_nzmatch_b0.30_z0.60/mcmc_preliminary_900'
OUT = RUN / 'diagnostics/joint_b1/task43_joint_b1_diagnostic.json'
NAMES = ['fNL', 'b1', 'sigma_s', 'sn0']

class Metric:
    """输入 Csingle，输出无量纲白化残差，避免 P/xi 的单位跨度。"""
    def __init__(self, cov):
        self.scale = np.sqrt(np.diag(cov))
        self.corr = cov / np.outer(self.scale, self.scale)
        self.chol = np.linalg.cholesky((self.corr+self.corr.T)/2)
    def residual(self, diff):
        return solve_triangular(self.chol, diff/self.scale, lower=True, check_finite=False)
    def chi2(self, diff):
        a = self.residual(diff)
        return float(a@a)

def fit(model, data, cov, starts, names, lower, upper):
    """多起点最小二乘：输入模型/数据/协方差/边界，输出最优参数和 chi2。"""
    met=Metric(cov)
    solutions=[]
    for start in starts:
        sol=least_squares(lambda t:met.residual(data-model(t)), np.clip(start,np.asarray(lower)+1e-7,np.asarray(upper)-1e-7), bounds=(lower,upper), x_scale='jac', xtol=1e-11, ftol=1e-11, gtol=1e-11, max_nfev=2500)
        solutions.append(sol)
    best=min(solutions,key=lambda s:float(s.fun@s.fun))
    result={'theta':dict(zip(names,map(float,best.x))),'chi2_single':float(best.fun@best.fun),'success':bool(best.success),'optimality':float(best.optimality),'all_start_chi2':[float(s.fun@s.fun) for s in solutions]}
    return result,best.x

def main():
    """复用现有模型并运行有限个参数/协方差对照，保存可复核的诊断。"""
    pm,xm,dp,dx,_,_=r.build_models_and_data()
    def p(t):
        a=pm.evaluate(t)
        return np.r_[a[:13],a[13:26][r.P2_KEEP]]
    def x(t):
        a=xm.evaluate(np.asarray(t)[:3],model='formal_gic',window_key='mean')
        return np.r_[a[0][r.XI_MASK],a[2][r.XI_MASK]]
    def joint(t): return np.r_[p(t),x(t)]
    def split(t): return np.r_[p(t[:4]),x([t[0],t[1],t[4]])]
    data=np.r_[dp,dx]
    with np.load(RUN/'ezmock900_covariance_and_stack.npz') as d:
        ce=d['covariance'].copy(); ids=d['production_indices'].copy()
    assert np.array_equal(ids,np.arange(900))
    with np.load(r.ANALYTIC_COV) as d: ca=d['rsd_joint'].copy()
    output={'scope':'ML attribution diagnostic; no changes to published chains; C_single, never /25','production_indices':[int(ids[0]),int(ids[-1])],'covariances':{}}
    for covname,cov in [('ezmock900',ce),('jaxpower',ca)]:
        met=Metric(cov); old={}; corrchain={}; audits={}; starts=[a.copy() for a in r.OPTIMIZER_STARTS]
        for variant,evalfn,obs,cc in [('p02',p,dp,cov[:22,:22]),('xi02',x,dx,cov[22:,22:]),('joint',joint,data,cov)]:
            summary=json.loads((RUN/f'summary_{covname}_{variant}.json').read_text())
            n=3 if variant=='xi02' else 4
            with np.load(RUN/f'chain_{covname}_{variant}.npz') as d:
                a=d['chain']; logp=d['logp']; flat=a.reshape(-1,n); tbest=flat[np.argmax(logp)].copy()
                corrchain[variant]=np.corrcoef(flat,rowvar=False).tolist()
                data_equal=bool(np.array_equal(d['data'],obs))
            local=Metric(cc); hp=float(summary['hartlap_percival']['hartlap'])
            precision,_=r.precision_from_cov(cc,hp)
            vals,vecs=np.linalg.eigh(precision)
            root=(vecs*np.sqrt(np.maximum(vals,0))[None,:])@vecs.T
            t_old=np.array(summary['map_theta'])
            error=obs-evalfn(t_old)
            audits[variant]={'saved_map':t_old.tolist(),'saved_map_chi2_single':local.chi2(error),'chain_best':tbest.tolist(),'chain_best_chi2_single':local.chi2(obs-evalfn(tbest)),'old_root_chi2_div_hartlap':float(np.sum((root@error)**2)/hp),'precision_quadratic_chi2_div_hartlap':float(error@precision@error/hp),'unscaled_precision_min_eigen':float(vals[0]),'unscaled_precision_max_eigen':float(vals[-1]),'saved_chain_data_exact_match':data_equal,'chain_logp_best':float(np.max(logp))}
            localstarts=[s[:n] for s in starts]+[t_old,tbest]
            rr,tt=fit(evalfn,obs,cc,localstarts,NAMES[:n],r.BOUNDS_LO[:n],r.BOUNDS_HI[:n])
            old[variant]=(rr,tt)
            if n==4: starts.append(tt)
            print(covname,variant,json.dumps(rr),flush=True)
        c0=cov.copy(); c0[:22,22:]=0.; c0[22:,:22]=0.
        nocross,tn=fit(joint,data,c0,starts,NAMES,r.BOUNDS_LO,r.BOUNDS_HI)
        splitstarts=[np.r_[s,old['xi02'][1][2]] for s in starts]
        separate,ts=fit(split,data,cov,splitstarts,NAMES+['sigma_s_xi'],np.r_[r.BOUNDS_LO,0],np.r_[r.BOUNDS_HI,30])
        separate0,ts0=fit(split,data,c0,splitstarts,NAMES+['sigma_s_xi'],np.r_[r.BOUNDS_LO,0],np.r_[r.BOUNDS_HI,30])
        fixed=[]
        for sig in [0.,2.5,5.7]:
            for variant,fn,obs,cc,nd in [('p02',p,dp,cov[:22,:22],3),('xi02',x,dx,cov[22:,22:],2),('joint',joint,data,cov,3)]:
                def unpack(t): return np.array([t[0],t[1],sig,t[2] if nd==3 else 0.])
                rr,tt=fit(lambda t:fn(unpack(t)),obs,cc,[np.array([s[0],s[1],s[3]])[:nd] for s in starts],['fNL','b1','sn0'][:nd],np.array([-500,.5,-1])[:nd],np.array([500,5,1])[:nd])
                fixed.append({'sigma_s_fixed':sig,'variant':variant,**rr})
        # 在无量纲单位实施精确 Schur 分解，以避免单位差造成求逆问题。
        cor=met.corr; cp=cor[:22,:22]; cx=cor[22:,22:]; k=cor[22:,:22]@np.linalg.inv(cp)
        schur=cx-k@cor[:22,22:]; conditional=Metric(schur)
        pieces={}
        for label,t in [('p_best',old['p02'][1]),('xi_best',np.r_[old['xi02'][1],old['p02'][1][3]]),('joint',old['joint'][1]),('split_sigma',ts)]:
            pred=split(t) if label=='split_sigma' else joint(t)
            d=(data-pred)/met.scale
            chp=Metric(cp).chi2(d[:22]); chcond=conditional.chi2(d[22:]-k@d[:22])
            chjoint=met.chi2(data-pred)
            pieces[label]={'chi2_P':chp,'chi2_xi_given_P':chcond,'chi2_joint':chjoint,'sum_bridge':chp+chcond-chjoint,'chi2_xi_marginal':Metric(cx).chi2(d[22:])}
        output['covariances'][covname]={'baseline':{k:v[0] for k,v in old.items()},'joint_zero_cross':nocross,'joint_split_sigma':separate,'joint_split_sigma_zero_cross':separate0,'fixed_sigma':fixed,'schur_decomposition':pieces,'old_optimizer_audit':audits,'posterior_correlations':corrchain,'max_abs_P_xi_correlation':float(abs(cor[:22,22:]).max()),'correlation_eigen_min':float(np.linalg.eigvalsh(cor).min()),'zeff_P':float(pm.zeff),'zeff_xi':float(xm.exact.zeff)}
        OUT.parent.mkdir(parents=True,exist_ok=True)
        OUT.write_text(json.dumps(output,indent=2)+'\n')
        print('FINISHED',covname,'no_cross',nocross,'split_sigma',separate,'split_no_cross',separate0,flush=True)
    print('OUTPUT',OUT,flush=True)

if __name__=='__main__': main()
