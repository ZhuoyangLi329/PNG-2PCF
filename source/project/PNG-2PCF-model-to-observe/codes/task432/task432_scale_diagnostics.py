#!/usr/bin/env python3
"""六种固定cut的配对稳定性和条件残差归因，不增加拟合配置。

大纲：同一82向量定义局部响应 -> phase配对bootstrap -> 去均值EZ噪声
重复六配置选择 -> 对关键cut在旧/新MAP处作Schur分解。局部检验与有界
非线性后验分开记录，不能将bootstrap尾率冒充精确后验显著性。
"""
import os
for k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):os.environ[k]='1'
import json
import numpy as np
from task432_scale_selection import OUT,RIC,CASES,Engine,load_master,save,sha,conditional

def response(J,C,ids,pars):
    """用列规范化SVD求GLS参数响应，并嵌回同一82维，保留全部交叉相关。"""
    from task432_scale_selection import common
    w=common.Metric(C[np.ix_(ids,ids)]).whitener;j=w@J[np.ix_(ids,pars)];scale=np.linalg.norm(j,axis=0)
    operator=(np.linalg.pinv(j/scale,rcond=1e-12)/scale[:,None])@w
    full=np.zeros((len(pars),len(C)));full[:,ids]=operator
    return full[pars.index(1)]

def outside(c):
    """两个joint-minus-single对比对应的区间外距离，可处理任意前置轴。"""
    return np.maximum(0,np.maximum(np.min(c,axis=-1),np.min(-c,axis=-1)))

def run():
    """固定fiducial与seed，所有cut共用相同抽样索引，输出有明确适用范围的检验。"""
    master=load_master(True);C=master['covariance'];e=Engine('bao_unmasked','joint');theta=np.array([0.,2.42,2.2,.08]);steps=np.array([.02,.0001,.003,.0001])
    t=np.tile(theta,(8,1))
    for i,h in enumerate(steps):t[2*i,i]+=h;t[2*i+1,i]-=h
    yp=e.predict(t);J=np.column_stack([(yp[2*i]-yp[2*i+1])/(2*h) for i,h in enumerate(steps)]);anchor=e.predict(theta)[0]
    contrasts=[];joint=[];case_ids={}
    for case in CASES:
        ce=Engine(case,'joint');ids=ce.ids if case=='bao_unmasked' else master['old_joint_indices'][ce.ids]
        p=ids[ids<22];x=ids[ids>=22]
        gj=response(J,C,ids,[0,1,2,3]);gp=response(J,C,p,[0,1,2,3]);gx=response(J,C,x,[0,1,3])
        contrasts.append([gj-gp,gj-gx]);joint.append(gj);case_ids[case]=ids
    B=np.array(contrasts);G=np.array(joint);null_error=float(np.max(abs(B@J)));assert null_error<1e-8
    h=master['halo_stack']-anchor;z=master['ez_stack']-master['ez_stack'].mean(0);rng=np.random.default_rng(4322561)
    obs=np.einsum('cij,j->ci',B,h.mean(0));obs_d=outside(obs);phase=[];noise=[];phase_means=[]
    for _ in range(10):
        hm=h[rng.integers(0,25,(200,25))].mean(1);zm=z[rng.integers(0,1000,(200,25))].mean(1)
        phase_means.append(hm);phase.append(np.einsum('cij,nj->nci',B,hm));noise.append(np.einsum('cij,nj->nci',B,zm))
    phase=np.concatenate(phase);noise=np.concatenate(noise);pd=outside(phase);nd=outside(noise)
    scan=json.loads((OUT/'map_scan.json').read_text());rows={};oldtheta=scan['cases']['baseline']['fits']['joint']['theta']
    for i,case in enumerate(CASES):
        cc=scan['cases'][case];change=phase[:,i]-phase[:,0];cov=B[i]@C@B[i].T
        row={'linear_mean_contrasts':obs[i],'linear_outside_distance':obs_d[i],'contrast_noise_covariance_Csingle':cov,
             'phase_bootstrap_paired_contrast_change_16_50_84':np.percentile(change,[16,50,84],axis=0),
             'phase_bootstrap_outside_reduction_fraction':float(np.mean(pd[:,i]<pd[:,0])),
             'phase_bootstrap_joint_b1_shift_16_50_84':None}
        # 同一重采样索引的精确joint响应另由原phase权重bootstrap生成。
        bs=np.concatenate(phase_means)@(G[i]-G[0])
        row['phase_bootstrap_joint_b1_shift_16_50_84']=np.percentile(bs,[16,50,84])
        ce=Engine(case,'joint')
        if case not in ('baseline','bao_unmasked'):
            row['conditional_at_baseline_MAP']=conditional(ce,oldtheta)
            row['conditional_at_cut_MAP']=cc['conditional']
        rows[case]=row
    obs_best=float(obs_d[0]-obs_d.min());null_best=nd[:,0]-nd.min(axis=1)
    result={'status':'pass','cases':rows,'theta_anchor':theta,'steps':steps,'n_bootstrap':2000,'seed':4322561,
        'common_parameter_null_error':null_error,'prespecified_cases':list(CASES),'observed_best_linear_reduction':obs_best,
        'EZ_noise_scan_best_reduction_upper_tail':float((1+np.sum(null_best>=obs_best))/2001),
        'noise_mean_divisor':25,'fit_covariance_divisor':1,
        'scope':'Approximate tangent-model paired stability, conditional on estimated EZ C. Halo phases reused, not an independent blind sample. Priors/boundaries require actual MCMC interpretation.',
        'source_sha256':sha(__file__)}
    save('paired_stability.json',result)
    np.savez_compressed(OUT/'paired_stability.npz',case_names=list(CASES),contrast_response=B,joint_response=G,jacobian=J,
                        phase_bootstrap_contrasts=phase,EZ_noise_contrasts=noise,theta_anchor=theta)
    print(json.dumps({'status':'pass','common_parameter_null_error':null_error,'linear_Dout':dict(zip(CASES,obs_d.tolist())),'approx_scan_tail':result['EZ_noise_scan_best_reduction_upper_tail']}),flush=True)

if __name__=='__main__':run()
