#!/usr/bin/env python3
"""针对首轮证据补充 profile、xi0 最小尺度删点和 25-phase 经验噪声闭合。

profile 自由化其余所有原参数；删点重新提取协方差子块；phase bootstrap
先减均值再构造同模型模拟，避免将待检验系统偏差注入零假设。
"""
import task432_joint_b1_diagnostics as d
from task432_joint_b1_diagnostics import np,json,save,log

def run():
    """三个针对性检查在同一冻结模型上执行，不额外引入正式自由参数。"""
    d.initialize();primary=json.loads((d.OUT/'primary.json').read_text());report={'covariances':{}}
    with np.load(d.OUT/'paired_phase_data.npz',allow_pickle=False) as z:stack=z['data']
    centered=stack-stack.mean(axis=0);rng=np.random.default_rng(4322303)
    # 两套拟合使用相同的 1000 个经验噪声抽样，便于直接比较权重差异。
    noises=np.array([centered[rng.integers(0,25,25)].mean(axis=0) for i in range(1000)])
    anchor=np.array([0.,2.42,2.15,.08]);truth=d.prediction(anchor)[0]
    for c,cov in d.COVS.items():
        rr={'b1_profiles':{},'split_b_profile':[],'xi0_deletions':{}};base=primary['covariances'][c]['baseline']
        for v,ids in d.IDS.items():
            theta=np.array(base[v]['theta']);grid=np.linspace(2.20,2.60,81);rows=[]
            for b in grid:
                r=d.fit(cov,ids,fixed={1:b},start=theta);rows.append({'b1':b,'chi2':r['chi2'],'success':r['success']})
            rr['b1_profiles'][v]=rows
        for diff in np.linspace(-.18,.12,61):
            def predict(t):
                """使用 b_xi 作共同坐标，并把 b_P 设为 b_xi+Delta。"""
                tt=np.atleast_2d(t);split=np.column_stack([tt[:,0],tt[:,1]+diff,tt[:,1],tt[:,2],tt[:,3]])
                return d.prediction(split,'split_b')
            fit=d.fit(cov,predict=predict,start=np.array(base['joint']['theta']))
            rr['split_b_profile'].append({'delta_b_P_minus_xi':diff,'chi2':fit['chi2'],'success':fit['success']})
        for label,remove in [('xi0_55',[22]),('xi0_65',[23]),('xi0_75',[24]),('xi0_55to75',[22,23,24]),('xi2_55to75',[48,49,50]),('P0_highk',[11,12]),('P2_highk',[20,21])]:
            rr['xi0_deletions'][label]=d.fit(cov,np.setdiff1d(d.IDS['joint'],remove),multi=True)
        rows=[]
        for i,noise in enumerate(noises):
            fits={v:d.fit(cov,ids,data=truth+noise,start=anchor) for v,ids in d.IDS.items()};bb=np.array([fits[v]['theta'][1] for v in d.IDS])
            rows.append({'b1':bb,'shift':bb[2]-bb[:2],'chi2_joint':fits['joint']['chi2'],'success':all(r['success'] for r in fits.values())})
        shifts=np.array([r['shift'] for r in rows]);mu=shifts.mean(axis=0);V=np.cov(shifts,rowvar=False);iv=np.linalg.inv(V)
        obs=np.array([base[v]['theta'][1] for v in d.IDS]);shift=obs[2]-obs[:2];stat=(shift-mu)@iv@(shift-mu);sim=np.einsum('ni,ij,nj->n',shifts-mu,iv,shifts-mu)
        rr['empirical_phase_noise_calibration']={'n':1000,'seed':4322303,'anchor':anchor,'rows':rows,'observed_shift':shift,'mean_shift':mu,'shift_covariance':V,'observed_2d_statistic':stat,'empirical_2d_tail_fraction':(1+sum(sim>=stat))/1001,'outside_low_fraction':np.all(shifts<0,axis=1).mean(),'at_least_as_low_both_fraction':(1+np.all(shifts<=shift,axis=1).sum())/1001,'chi2_upper_tail_fraction':(1+sum(r['chi2_joint']>=base['joint']['chi2'] for r in rows))/1001,'scope':'Resample centered paired residuals from only 25 observed phases; conditional finite-sample sensitivity, not independent validation. Same mean operator throughout.'}
        report['covariances'][c]=rr;save(d.OUT/'refinement.json',report);log('refinement_cov_done',covariance=c)

if __name__=='__main__':run()
