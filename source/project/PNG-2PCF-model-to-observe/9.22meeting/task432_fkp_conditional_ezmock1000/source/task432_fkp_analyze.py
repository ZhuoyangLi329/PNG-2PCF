#!/usr/bin/env python3
"""汇总FKP配对、重算普通窗口与IC模型增量，量化条件b1响应。

执行大纲：精确匹配74bins -> 每phase数据/模型差分 -> 独立scramble精度 ->
给出局部响应和恒定增量诊断优化；不生成新的正式posterior或covariance。
"""
import os
for k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):os.environ[k]='1'
import argparse,json,sys
from pathlib import Path
import numpy as np
from scipy.optimize import least_squares
ROOT=Path('/pscratch/sd/l/lzy/PNG-2PCF-model-to-observe')
for f in ('codes/task43','codes/task432','codes/task44'):sys.path.insert(0,str(ROOT/f))
from task432_fkp_conditional_modes import OUT,THETA,COMPILED
from task432_full_ric_compile import Engine,common
from task432_full_ric_geometry import BASE,sha
from task43_rsd_boxsafe_p02_increment import WindowConvolvedP02Model,PAYLOAD_NPZ

def select_p(a,edges):
    """匹配真实测量的k边界，返回数据22vector和相应窗口22行。"""
    pe=a['k_edges'];pe=np.column_stack([pe[:-1],pe[1:]]) if pe.ndim==1 else pe
    ids=[]
    for row in edges:
        found=np.flatnonzero(np.all(np.isclose(pe,row,rtol=0,atol=1e-10),axis=1));assert len(found)==1;ids.append(found[0])
    ids=np.asarray(ids);rows=np.r_[ids,len(pe)+ids[4:]]
    return np.r_[a['multipoles'][0,ids],a['multipoles'][1,ids[4:]]],a['window_matrix'][rows]

def analyze(phases,require_ric=False):
    """以原C_single作诊断标尺；所有均值来自明确列出的pilot phases。"""
    with np.load(OUT/'conditional_modes.npz') as a:
        C=a['covariance'];B=a['contrast_response'];G=np.array([a['p_response'],a['xi_response'],a['joint_response']]);M=a['modes']
    with np.load(BASE/'frozen_inputs.npz') as a:pe=a['p_edges'];xc=a['xi_centers']
    with np.load(PAYLOAD_NPZ) as a:zeff=float(a['zeff'])
    metric=common.Metric(C);records=[];delta_data=[];delta_model=[];delta_corrected=[]
    arrays={}
    for phase in phases:
        dest=OUT/'paired'/phase
        if not all((dest/f'{b}_pk.npz').exists() and list(dest.glob(f'{b}_xi_block*.npz')) for b in ('own','independent')):continue
        pk={};xi={};models={};norms={};shots={};blocks={}
        for branch in ('own','independent'):
            with np.load(dest/f'{branch}_pk.npz') as a:
                pk[branch],W=select_p(a,pe)
                pmodel=WindowConvolvedP02Model(W,a['theory_k'],a['theory_ell'],zeff=zeff,sigma_step=30.)
                models[branch]=np.r_[pmodel.direct(THETA),np.zeros(52)]
                norms[branch]=float(a['norms'][0].ravel()[0]);shots[branch]=float(a['shotnoise'][0].ravel()[0])
            blocks[branch]=[]
            for path in sorted(dest.glob(f'{branch}_xi_block*.npz')):
                with np.load(path) as a:
                    ids=[np.flatnonzero(np.isclose(a['s'],s,rtol=0,atol=1e-8))[0] for s in xc]
                    blocks[branch].append(a['xi_multipoles'][:,ids].reshape(-1))
            xi[branch]=np.mean(blocks[branch],axis=0)
        assert len(blocks['own'])==len(blocks['independent'])
        dd=np.r_[pk['independent']-pk['own'],xi['independent']-xi['own']]
        dm=models['independent']-models['own'];ordinary=dm.copy()
        rics=[]
        for path in sorted(dest.glob('ric_response_n*_seed*.npz')):
            with np.load(path) as a:rics.append(a['difference'])
        if require_ric and len(rics)<2:raise RuntimeError(f'Need2 independent differentialIC kernels for {phase}')
        if rics:dm+=np.mean(rics,axis=0)
        net=dd-dm
        xb=np.array(blocks['independent'])-np.array(blocks['own'])
        br=np.column_stack([np.tile(dd[:22],(len(xb),1)),xb])-dm
        block_responses=br@B.T
        if len(rics)>1:kernel_error=float(metric.chi2(rics[0]-rics[1]))
        else:kernel_error=None
        record={'phase':phase,'n_xi_blocks':len(xb),'normalizations':norms,'shotnoise':shots,
            'data_change_norm_Csingle':float(np.sqrt(metric.chi2(dd))),
            'ordinary_window_change_norm_Csingle':float(np.sqrt(metric.chi2(ordinary))),
            'IC_change_norm_Csingle':float(np.sqrt(metric.chi2(dm-ordinary))),
            'residual_change_norm_Csingle':float(np.sqrt(metric.chi2(net))),
            'b1_data_response_P_xi_joint':G@dd,'b1_net_response_P_xi_joint':G@net,
            'b1_contrast_net_response':B@net,'conditional_mode_net_response':M@net,
            'n_independent_IC_differences':len(rics),'IC_difference_error_chi2_Csingle':kernel_error,
            'xi_block_contrast_responses':block_responses,
            'model_scope':'ordinary_window_plus_fullIC' if len(rics)>=2 else 'ordinary_window_only; IC weight-response pending'}
        records.append(record);delta_data.append(dd);delta_model.append(dm);delta_corrected.append(net)
        arrays[phase+'_xi_block_net_response']=block_responses
    if not records:raise RuntimeError('No completed paired measurements')
    d=np.mean(delta_data,axis=0);m=np.mean(delta_model,axis=0);net=d-m
    response=np.array(delta_corrected)@B.T
    out={'status':'complete_diagnostic' if require_ric else 'pilot_partial_model','phases':[r['phase'] for r in records],
        'nphase':len(records),'theta_anchor':THETA,'records':records,
        'mean_b1_response_P_xi_joint':G@net,'mean_b1_contrast_response':B@net,
        'mean_b1_contrast_data_only':B@d,'mean_b1_contrast_model_only':B@m,
        'phase_SEM_b1_contrast_response':response.std(0,ddof=1)/np.sqrt(len(response)) if len(response)>1 else None,
        'mean_residual_change_norm_Csingle':float(np.sqrt(metric.chi2(net))),
        'source_sha256':sha(__file__),
        'scope':'Same catalog/random positions; independent-minus-own FKP. Mean across listed pilot phases, not25phase remeasurement. Original EZ1000 C_single used only as diagnostic metric. No posterior promoted.',
        'limitations':['Finite24phase reference nbar; cross-phase responses correlated through reference','Small number of pilot phases; not ensemble calibration','Limited common LS blocks; block spread reports differential random sensitivity','Model delta evaluated at fixed fiducial; constant-delta optimization is a sensitivity check, not final fit']}
    if require_ric:
        out['diagnostic_constant_delta_MAP']={}
        for variant,ids in (('p02',slice(0,22)),('xi02',slice(22,74)),('joint',slice(None))):
            e=Engine(variant,COMPILED);data=e.data+net[ids]
            starts=[THETA[[0,1,3]] if variant=='xi02' else THETA]
            starts.append(np.array([10.,2.38,.1]) if variant=='xi02' else np.array([10.,2.38,3.,.1]))
            solutions=[least_squares(lambda t:e.raw_metric.white(data-e.predict(t)[0]),x,bounds=(e.lower,e.upper),x_scale='jac',max_nfev=1500,ftol=1e-11,xtol=1e-11,gtol=1e-11) for x in starts]
            best=min(solutions,key=lambda r:r.fun@r.fun)
            out['diagnostic_constant_delta_MAP'][variant]={'theta':best.x,'raw_chi2':float(best.fun@best.fun),'success':bool(best.success)}
    suffix='paired_fkp_results' if require_ric else 'paired_fkp_preview'
    common.save(OUT/f'{suffix}.json',out)
    np.savez_compressed(OUT/f'{suffix}.npz',phases=out['phases'],delta_data=delta_data,delta_model=delta_model,delta_residual=delta_corrected,mean_delta_data=d,mean_delta_model=m,**arrays)
    print(json.dumps({'status':out['status'],'phases':out['phases'],'b1_contrast_response':out['mean_b1_contrast_response'].tolist(),'b1_P_xi_joint_response':out['mean_b1_response_P_xi_joint'].tolist(),'phase_SEM':None if out['phase_SEM_b1_contrast_response'] is None else out['phase_SEM_b1_contrast_response'].tolist()}),flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--phases',nargs='+',default=['ph000','ph012','ph024']);p.add_argument('--require-ric',action='store_true');a=p.parse_args();analyze(a.phases,a.require_ric)
