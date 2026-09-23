#!/usr/bin/env python3
"""为唯一新增点配置恢复BAO八点，不改变已有74维模型。

执行大纲：同1000EZ/25halo组装82向量和RR -> 仅计算缺失85/95/105/115
中心的hybrid网格 -> 用原四核平均投影完整IC、Poisson和sn0 -> 核对回切
及独立直接模型/双核均值误差。新文件全部写独立scale-selection目录。
"""
import os
for key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):os.environ[key]='1'
import argparse,json,time,multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor,as_completed
from pathlib import Path
import numpy as np
from task432_scale_selection import OUT,BASE,RIC,ROOT,COMPILED,save,sha,load_master,common
from task432_full_ric_geometry import HALO_MANIFEST
from task432_full_ric_response import Response
from task432_full_ric_validation import HybridGIC,shell_poles

def prepare_data():
    """保持EZ配对和phase均值；扩展后回切必须复现原数据、C、RR。"""
    if (OUT/'expanded_inputs.npz').exists():return
    import task43_rsd_ezmock281_mcmc_compare as old
    original=load_master();s=np.arange(55.,350.,10.);keep=~((s>=80)&(s<120))
    old_ids=np.flatnonzero(np.r_[np.ones(22,dtype=bool),keep,keep]);xi_ids=np.flatnonzero(np.r_[keep,keep])
    source=BASE.parent/'hybrid_ezmock1000_0918_contract_v1/mock_selection_manifest.json'
    rows=json.loads(source.read_text())['rows'];assert [x['production_index'] for x in rows]==list(range(1000))
    vectors=[];meta=[]
    for i,row in enumerate(rows):
        path=Path(row['xi_path']);jm=json.loads(path.with_suffix('.json').read_text())
        assert jm['status']=='done' and jm['fix_amplitude'] is False and jm['ells']==[0,2]
        with np.load(path) as a:
            ids=[np.flatnonzero(np.isclose(a['s'],c,atol=1e-8,rtol=0))[0] for c in s]
            assert int(a['production_index'])==i if 'production_index' in a else True
            v=np.r_[original['ez_stack'][i,:22],a['xi0'][ids],a['xi2'][ids]]
            assert np.all(np.isfinite(v));vectors.append(v)
        meta.append({'path':str(path),'sha256':sha(path)})
    vectors=np.array(vectors);cov=np.cov(vectors,rowvar=False)
    with np.load(old.ELL2_SUMMARY) as a:
        ids=[np.flatnonzero(np.isclose(a['s'],c,atol=1e-8,rtol=0))[0] for c in s]
        xx=a['xi_multipoles_by_phase'][:,:,ids]
    assert xx.shape==(25,2,30)
    halo=np.column_stack([original['halo_stack'][:,:22],xx.reshape(25,60)]);data=halo.mean(0)
    sources=[json.loads(l) for l in HALO_MANIFEST.read_text().splitlines() if l.strip()]
    rr=[];rmeta=[]
    for row in sources:
        path=Path(row['lightcone_xi_path'])
        with np.load(path) as a:
            ids=[np.flatnonzero(np.isclose(a['s'],c,atol=1e-8,rtol=0))[0] for c in s]
            rr.append(a['RR_smu'][ids]);assert np.array_equal(a['mu_edges'],np.linspace(-1,1,41))
        rmeta.append({'path':str(path),'sha256':sha(path)})
    rr=np.array(rr);assert rr.shape==(25,30,40) and np.all(rr>0)
    with np.load(RIC/'geometry/production_rr_mu.npz') as a:assert np.array_equal(rr[:,keep],a['rr_by_phase'])
    checks={'stack_restrict':bool(np.allclose(vectors[:,old_ids],original['ez_stack'],rtol=1e-12,atol=1e-14)),
        'halo_restrict':bool(np.allclose(halo[:,old_ids],original['halo_stack'],rtol=1e-12,atol=1e-14)),
        'data_restrict':bool(np.allclose(data[old_ids],original['data'],rtol=1e-10,atol=1e-10)),
        'covariance_restrict':bool(np.allclose(cov[np.ix_(old_ids,old_ids)],original['covariance'],rtol=1e-10,atol=1e-14)),
        'positive_definite':bool(np.linalg.eigvalsh(cov/np.sqrt(np.outer(np.diag(cov),np.diag(cov)))).min()>0)}
    assert all(checks.values()),checks
    np.savez_compressed(OUT/'expanded_inputs.npz',data=data,covariance=cov,halo_stack=halo,ez_stack=vectors,p_edges=original['p_edges'],xi_centers=s,
                        old_joint_indices=old_ids,old_xi_indices=xi_ids,rr_mean=rr.mean(0))
    save('expanded_input_audit.json',{'status':'pass','checks':checks,'mock_count':1000,'halo_count':25,'dimension':82,
        'mock_manifest':str(source),'mock_manifest_sha256':sha(source),'xi_sources':meta,'halo_xi_summary':str(old.ELL2_SUMMARY),
        'halo_xi_summary_sha256':sha(old.ELL2_SUMMARY),'RR_sources':rmeta,'source_sha256':sha(__file__)})
    print(json.dumps({'event':'expanded_inputs_pass','checks':checks}),flush=True)

def init_worker():
    """单线程worker初始化同一已验证hybrid；只计算四个新中心。"""
    global MODEL
    MODEL=HybridGIC()

def raw_row(task):
    """输入fNL行及原b网格，输出两多极四个BAO中心的未加IC值。"""
    i,f,bb=task;values=[]
    for b in bb:
        MODEL._set_cumulants(f,float(b));values.append(MODEL.poles([85.,95.,105.,115.],nint=600)[:2].reshape(-1))
    return i,np.array(values)

def build_model():
    """原52输出逐元素复用；新8输出沿相同网格，不外推且无新参数。"""
    if (OUT/'expanded_model.npz').exists():return
    with np.load(BASE/'xi_emulator.npz') as a:ff=a['f_grid'];bb=a['b_grid'];oldraw=a['values_no_gic']
    checkpoint=OUT/'logs/bao_raw_grid';checkpoint.mkdir(parents=True,exist_ok=True)
    raw=np.full((len(ff),len(bb),8),np.nan);tasks=[]
    for i,f in enumerate(ff):
        p=checkpoint/f'row_{i:03d}.npz'
        if p.exists():
            with np.load(p) as a:assert float(a['fnl'])==f;assert np.array_equal(a['b_grid'],bb);raw[i]=a['values']
        else:tasks.append((i,float(f),bb))
    with ProcessPoolExecutor(max_workers=6,mp_context=mp.get_context('spawn'),initializer=init_worker) as pool:
        for fut in as_completed([pool.submit(raw_row,t) for t in tasks]):
            i,value=fut.result();assert np.all(np.isfinite(value));raw[i]=value
            np.savez_compressed(checkpoint/f'row_{i:03d}.npz',fnl=ff[i],b_grid=bb,values=value)
            print(json.dumps({'event':'bao_grid','row':i,'completed':int(np.all(np.isfinite(raw),axis=(1,2)).sum())}),flush=True)
    with np.load(OUT/'expanded_inputs.npz') as a:s=a['xi_centers'];rr=a['rr_mean'];oldidx=a['old_xi_indices']
    newidx=np.setdiff1d(np.arange(60),oldidx);fullraw=np.empty((len(ff),len(bb),60));fullraw[:,:,oldidx]=oldraw;fullraw[:,:,newidx]=raw
    g=RIC/'geometry';response=Response(g/'validated_mean4_midpoint.npz');matrix=response.xi_matrix(s,rr_smu=rr)
    with np.load(g/'hybrid_inner_refine1_q2_n600.npz') as a:
        assert np.array_equal(a['separation_edges'],response.edges);ric=np.einsum('ild,abld->abi',matrix,a['poles'])
    shot=json.loads((RIC/'smoke/full_response_pilot.json').read_text())
    fixed=np.zeros(60)
    for kind,n,width,amplitude in [('radial',65536,2,shot['data_shot_amplitude']),('global',16384,0,shot['random_xi_shot_equivalent_P_amplitude'])]:
        with np.load(g/f'ph000_shot_n{n}_dchi{width}_refine1_midpoint_ell8.npz') as a:moment=(a['auto']-a['cross'])/response.meta['pair_volume']
        fixed+=amplitude*response.xi_project(moment,s,rr_smu=rr)
    with np.load(COMPILED) as a:
        meta=json.loads(str(a['meta_json'].item()));ratio=meta['stochastic_I2_ratio']
        assert np.allclose(ric[:,:,oldidx],a['xi_ric_grid'],rtol=1e-11,atol=1e-14)
        assert np.allclose(fixed[oldidx],a['fixed_shot_vector'][22:],rtol=1e-11,atol=1e-14)
        # 使旧输出保持原始浮点值；不能把数值重编译差误解为选点效应。
        ric[:,:,oldidx]=a['xi_ric_grid'];fixed[oldidx]=a['fixed_shot_vector'][22:]
    with np.load(g/'ph000_shot_white_n65536_dchi2_refine1_midpoint_ell8.npz') as a:moment=(a['auto']-a['cross'])/response.meta['pair_volume']
    stochastic=1e4*ratio*response.xi_project(moment,s,rr_smu=rr)
    with np.load(COMPILED) as a:
        assert np.allclose(stochastic[oldidx],a['sn0_ric_response'][22:],rtol=1e-11,atol=1e-14);stochastic[oldidx]=a['sn0_ric_response'][22:]
    np.savez_compressed(OUT/'expanded_model.npz',f_grid=ff,b_grid=bb,values=fullraw+ric,raw_values=fullraw,ric_values=ric,
                        fixed=fixed,stochastic=stochastic,matrix=matrix,separation_edges=response.edges)
    save('expanded_model_manifest.json',{'status':'compiled_pending_validation','source_sha256':sha(__file__),
        'input_sha256':sha(OUT/'expanded_inputs.npz'),'original_compiled_sha256':sha(COMPILED),
        'geometry_sha256':sha(response.path),'physical_grid_sha256':sha(g/'hybrid_inner_refine1_q2_n600.npz'),
        'new_centers':[85,95,105,115],'free_parameters_changed':False,'raw_grid_nint':600})

def validate(posterior_cases=None):
    """检查完整82度量；后验模式补查新cut的MAP/尾部，门限与原模型相同。"""
    from task432_scale_selection import Engine,OriginalEngine
    e=Engine('bao_unmasked','joint');g=RIC/'geometry';d=load_master(True)
    with np.load(OUT/'expanded_model.npz') as a:matrix=a['matrix'];fixed=a['fixed'];stochastic=a['stochastic'];edges=a['separation_edges']
    delta=Response(g/'refinement_independent_mean2_midpoint.npz').xi_matrix(d['xi_centers'],rr_smu=d['rr_mean'])-Response(g/'validated_mean2_midpoint.npz').xi_matrix(d['xi_centers'],rr_smu=d['rr_mean'])
    model=HybridGIC();metric=common.Metric(d['covariance']);rng=np.random.default_rng(4322527)
    probes=[('anchor',[f,b,.1]) for f,b in [(0,2.4),(-63.7,2.273),(-31.3,2.517),(18.7,2.347),(57.3,2.643),(96.1,2.177),(-102.3,2.713),(-150,2.85),(100,2.85)]]
    if posterior_cases:
        probes=[]
        for case in posterior_cases:
            for variant in ('xi02','joint'):
                dest=OUT/'chains'/case/variant
                if not (dest/'summary.json').exists():continue
                summary=json.loads((dest/'summary.json').read_text());assert summary['status']=='pass'
                with np.load(dest/'samples.npz') as a:flat=a['chain'].reshape(-1,len(summary['parameter_names']))
                pts=[('random',flat[i]) for i in rng.choice(len(flat),12,replace=False)]
                for col in (0,1):
                    for q in (1,99):pts.append((f'tail{col}_{q}',flat[np.argmin(abs(flat[:,col]-np.percentile(flat[:,col],q))) ]))
                pts.append(('MAP',np.asarray(summary['map_theta'])))
                probes += [(case+':'+variant+':'+label,[t[0],t[1],t[-1]]) for label,t in pts]
    records=[]
    for label,t in probes:
        f,b,sn=t;inner=shell_poles(model,edges,f,b);raw=model.poles(d['xi_centers'],nint=1200)[:2].reshape(-1)
        truth=raw+np.einsum('ild,ld->i',matrix,inner)+fixed+sn*stochastic
        err=e.x(t)[0]-truth;geom=np.einsum('ild,ld->i',delta,inner)
        row={'label':label,'theta':t,'emulator_q':float(metric.chi2(np.r_[np.zeros(22),err])),
             'geometry_q':float(metric.chi2(np.r_[np.zeros(22),geom]))}
        if label=='anchor' and f==0 or label.endswith('MAP'):
            inner2=shell_poles(model,edges,f,b,nquad=8,nint=2400);raw2=model.poles(d['xi_centers'],nint=2400)[:2].reshape(-1)
            row['integration_q']=float(metric.chi2(np.r_[np.zeros(22),raw2-raw+np.einsum('ild,ld->i',matrix,inner2-inner)]))
        records.append(row)
    old=OriginalEngine('joint',COMPILED);ts=np.array([[0,2.4,2.2,.1],[-80,2.65,8.,-.5],[60,2.3,12.,.8]])
    restriction=float(np.max(abs(e.predict(ts)[:,d['old_joint_indices']]-old.predict(ts))))
    maxima={k:max(x.get(k,0) for x in records) for k in ('emulator_q','geometry_q','integration_q')}
    gates={'emulator':maxima['emulator_q']<.01,'geometry':maxima['geometry_q']<.01,'integration':maxima['integration_q']<.001,'old_predictions_preserved':restriction<1e-8}
    report={'status':'pass' if all(gates.values()) else 'failed','gates':gates,'maxima':maxima,'old_prediction_maxabs':restriction,'records':records,
        'model_sha256':sha(OUT/'expanded_model.npz'),'scope':'Full82 C_single. Independent direct hybrid and independent two-kernel means; no threshold relaxation.'}
    save('expanded_posterior_validation.json' if posterior_cases else 'expanded_validation.json',report)
    print(json.dumps({k:v for k,v in report.items() if k!='records'}),flush=True);assert report['status']=='pass'

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('action',choices=('build','validate'));p.add_argument('--posterior-cases',nargs='+');a=p.parse_args()
    if a.action=='build':OUT.mkdir(parents=True,exist_ok=True);prepare_data();build_model();validate()
    else:validate(a.posterior_cases)
