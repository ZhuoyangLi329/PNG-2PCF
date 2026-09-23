#!/usr/bin/env python3
"""少数cut的实际后验数值验收，不重新选择模型或阈值。

流程：读取四条新链的随机/尾部/MAP -> 同一底层hybrid直接计算 ->
按各cut实际covariance验证插值、独立IC核、P spline和积分精度；并检查
旧74向量回切、参数映射、数据子集与有限mock修正的维数。
"""
import os
for k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):os.environ[k]='1'
import json
import numpy as np
from task432_scale_selection import OUT,RIC,BASE,COMPILED,Engine,OriginalEngine,load_master,save,sha,common,CASES
from task432_full_ric_validation import shell_poles,HybridGIC,paths,WindowConvolvedP02Model,PAYLOAD_NPZ
from task432_full_ric_geometry import PK_DIR
from task432_full_ric_response import Response

def run():
    """全部误差使用各cut实际C_single；原门限不改变，失败时不交付正式约束。"""
    d=load_master(True);g=RIC/'geometry';model=HybridGIC();expanded=Engine('bao_unmasked','joint');old=OriginalEngine('joint',COMPILED)
    with np.load(OUT/'expanded_model.npz') as a:matrix=a['matrix'];fixed=a['fixed'];stoch=a['stochastic'];edges=a['separation_edges']
    delta_x=Response(g/'refinement_independent_mean2_midpoint.npz').xi_matrix(d['xi_centers'],rr_smu=d['rr_mean'])-Response(g/'validated_mean2_midpoint.npz').xi_matrix(d['xi_centers'],rr_smu=d['rr_mean'])
    with np.load(PK_DIR/'task43_rsd_lightcone_p02_ph000_mesh256_kmax0p300_dk0p002.npz') as a:tk,te,tedges=a['theory_k'],a['theory_ell'],a['theory_edges']
    with np.load(PAYLOAD_NPZ) as a:zeff=float(a['zeff'])
    pm=WindowConvolvedP02Model(np.zeros((1,len(tk))),tk,te,zeff=zeff,sigma_step=30.)
    with np.load(COMPILED) as a:pmatrix=a['p_response']
    prows=np.r_[np.arange(13),np.arange(17,26)];pp=paths()
    delta_p=Response(pp['p2']).pk_theory_matrix(d['p_edges'],tedges,te,nquad=16,inner_nquad=16).reshape(26,-1)[prows]-Response(pp['p1']).pk_theory_matrix(d['p_edges'],tedges,te,nquad=16,inner_nquad=16).reshape(26,-1)[prows]
    rng=np.random.default_rng(4322551);records=[]
    for case,variant in [('kmax0p06','p02'),('kmax0p06','joint'),('bao_unmasked','xi02'),('bao_unmasked','joint')]:
        dest=OUT/'chains'/case/variant;r=json.loads((dest/'summary.json').read_text());assert r['status']=='pass'
        ce=Engine(case,'joint');ids=d['old_joint_indices'][ce.ids] if case!='bao_unmasked' else ce.ids
        metric=common.Metric(d['covariance'][np.ix_(ids,ids)])
        with np.load(dest/'samples.npz') as a:flat=a['chain'].reshape(-1,len(r['parameter_names']))
        points=[('random',flat[i]) for i in rng.choice(len(flat),12,replace=False)]
        for col in (0,1):
            for quantile in (1,99):points.append((f'tail{col}_{quantile}',flat[np.argmin(abs(flat[:,col]-np.percentile(flat[:,col],quantile)))]))
        points.append(('MAP',np.array(r['map_theta'])))
        for label,t in points:
            f,b,sn=t[0],t[1],t[-1];inner=shell_poles(model,edges,f,b);raw=model.poles(d['xi_centers'],nint=1200)[:2].reshape(-1)
            truth=raw+np.einsum('ild,ld->i',matrix,inner)+fixed+sn*stoch
            err=np.r_[np.zeros(22),expanded.x(t)[0]-truth];geom=np.r_[np.zeros(22),np.einsum('ild,ld->i',delta_x,inner)]
            row={'case':case,'variant':variant,'point':label,'theta':t,'emulator_q':float(metric.chi2(err[ids])),
                 'geometry_q':float(metric.chi2(geom[ids]))}
            if len(t)==4:
                q=f*2*1.686*(b-1);fg=old.f;coeff=np.array([b*b,2*b*q,q*q,2*b*fg,2*q*fg,fg*fg]);theory=pm._theory_basis(t[2])@coeff
                pe=np.r_[old.delta_p(t[2])@coeff-pmatrix@theory,np.zeros(60)]
                pg=np.r_[delta_p@theory,np.zeros(60)]
                row['P_spline_q']=float(metric.chi2(pe[ids]));row['P_geometry_q']=float(metric.chi2(pg[ids]))
            if label=='MAP':
                inner2=shell_poles(model,edges,f,b,nquad=8,nint=2400);raw2=model.poles(d['xi_centers'],nint=2400)[:2].reshape(-1)
                dif=np.r_[np.zeros(22),raw2-raw+np.einsum('ild,ld->i',matrix,inner2-inner)]
                row['integration_q']=float(metric.chi2(dif[ids]))
            records.append(row)
        print(json.dumps({'event':'posterior_numeric_probe','case':case,'variant':variant,'points':len(points)}),flush=True)
    maxima={k:max(x.get(k,0) for x in records) for k in ('emulator_q','geometry_q','P_spline_q','P_geometry_q','integration_q')}
    limits={'emulator_q':.01,'geometry_q':.01,'P_spline_q':.001,'P_geometry_q':.01,'integration_q':.001}
    gates={k:maxima[k]<limits[k] for k in maxima}
    ts=np.array([[0.,2.4,2.2,.1],[-80.,2.65,8.,-.5],[60.,2.3,12.,.8]])
    for case in CASES:
        j,p,x=[Engine(case,v) for v in ('joint','p02','xi02')]
        err=float(np.max(abs(j.predict(ts)-np.column_stack([p.predict(ts),x.predict(ts[:,[0,1,3]])]))))
        gates[case+'_parameter_mapping']=err<1e-10
        gates[case+'_mock_dimension']=all(e.corrections['ndata']==len(e.data) and e.corrections['nparams']==len(e.names) and e.corrections['nmock']==1000 for e in (j,p,x))
    scan=json.loads((OUT/'map_scan.json').read_text());gates['six_prespecified_cases']=set(scan['cases'])==set(CASES)
    errors=[]
    for row in scan['cases'].values():
        for k in ('conditional','BAO_conditional_at_old_MAP','BAO_conditional_at_new_MAP'):
            if row.get(k):errors.append(row[k]['identity_error'])
    gates['Schur_identities']=max(errors)<1e-9
    for case,variant in [('kmax0p06','xi02'),('bao_unmasked','p02')]:
        e=Engine(case,variant);o=OriginalEngine(variant,COMPILED)
        t=ts[:,[0,1,3]] if variant=='xi02' else ts
        gates[case+'_'+variant+'_reuse_original_chain']=np.array_equal(e.data,o.data) and np.array_equal(e.cov,o.cov) and np.max(abs(e.predict(t)-o.predict(t)))<1e-10
    result={'status':'pass' if all(gates.values()) else 'failed','gates':gates,'maxima':maxima,'thresholds':limits,'records':records,'max_Schur_identity_error':max(errors),
        'scope':'Actual posterior random/tail/MAP numerical checks in each cut joint C_single. No model or covariance retuning.',
        'source_sha256':sha(__file__),'input_hashes':{str(p):sha(p) for p in [COMPILED,BASE/'frozen_inputs.npz',OUT/'expanded_inputs.npz',OUT/'expanded_model.npz']}}
    save('postflight.json',result);print(json.dumps(common.clean({k:v for k,v in result.items() if k!='records'})),flush=True)
    assert result['status']=='pass'

if __name__=='__main__':run()
