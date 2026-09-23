#!/usr/bin/env python3
"""本轮结束审计：差分精度、CPU桥接、冻结输入、文件归属及小型交付清单。

大纲：重新hash正式输入 -> 检查两个独立IC差分 -> paired向量/shot/恒等式
-> 对照原cucount相同blocks -> 打包小结果与源码，较大目录/几何核留远端复现。
"""
import os
for k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):os.environ[k]='1'
import json,sys,hashlib,tarfile
from pathlib import Path
import numpy as np
ROOT=Path('/pscratch/sd/l/lzy/PNG-2PCF-model-to-observe')
for f in ('codes/task43','codes/task432'):sys.path.insert(0,str(ROOT/f))
from task432_fkp_conditional_modes import OUT
from task432_full_ric_geometry import BASE,sha
from task432_hybrid_jaxpower_0918_contract import Metric,save
from task432_full_ric_response import angular_bin_integrals

def main():
    """所有精度门面向FKP增量，不冒充绝对核的正式后验验证。"""
    modes=json.loads((OUT/'conditional_modes.json').read_text());res=json.loads((OUT/'paired_fkp_results.json').read_text())
    with np.load(OUT/'conditional_modes.npz') as a:
        C=a['covariance'];B=a['contrast_response'];G=a['joint_response'];M=a['modes'];J=a['jacobian']
    metric=Metric(C);checks={'frozen_inputs_unchanged':all(sha(p)==h for p,h in modes['inputs'].items()),
        'mode_identity':bool(np.max(abs(M@C@M.T-np.eye(5)))<1e-8),'common_parameter_null':bool(np.max(abs(B@J))<1e-9),
        'three_pilot_phases':res['phases']==['ph000','ph012','ph024']}
    kernel_reports=[];bridges=[];pk_reports=[]
    for phase in res['phases']:
        dest=OUT/'paired'/phase
        paths=sorted(dest.glob('ric_response_n*_seed*.npz'));assert len(paths)==2
        differences=[]
        for p in paths:
            with np.load(p) as a:differences.append(a['difference'])
        error=differences[0]-differences[1];b_error=B@error;j_error=G@error
        # 要分辨约0.04的位移，投影积分误差门设0.001；Csingle全向量门0.001。
        passed=bool(np.max(abs(b_error))<.001 and abs(j_error)<.001 and metric.chi2(error)<.001)
        checks[phase+'_IC_difference_accuracy']=passed
        kernel_reports.append({'phase':phase,'paths':[str(p) for p in paths],'error_chi2_Csingle':float(metric.chi2(error)),
            'b1_contrast_discrepancy':b_error,'joint_b1_discrepancy':float(j_error),'pass':passed})
        for p in dest.glob('kernel_*.npz'):
            with np.load(p) as a:
                closure=max(np.max(abs(a['cross'][:,0].sum(-1)-2*a['rr'])),np.max(abs(a['auto'][:,0].sum(-1)-a['rr'])))/np.max(abs(a['rr']))
            checks[phase+'_'+p.stem+'_constant_mode']=bool(closure<1e-10)
        for branch in ('own','independent'):
            a=json.loads((dest/f'{branch}_pk.json').read_text());checks[phase+'_'+branch+'_analytic_shot']=a['shot_relative_error']<1e-10
            checks[phase+'_'+branch+'_CPU_limit']=len(a['cpu_affinity'])<=8
            pk_reports.append(a)
        # 与生产完全相同的两块对照，仅复核CPU/FFT桥接，配对差分仍来自同一CPU后端。
        meta=json.loads((dest/'catalogs.json').read_text());blockpaths=sorted(dest.glob('own_xi_block*.npz'))
        with np.load(meta['xi_production_path']) as a:smu=a['xi_smu_by_random'][:len(blockpaths)];s=a['s']
        with np.load(BASE/'frozen_inputs.npz') as a:xc=a['xi_centers']
        ids=[np.flatnonzero(np.isclose(s,x,rtol=0,atol=1e-8))[0] for x in xc]
        weights=.5*np.array([1.,5.])[:,None]*angular_bin_integrals([0,2],np.linspace(-1,1,41))
        original=np.mean(smu,axis=0)@weights.T;old=original[ids].T.reshape(-1)
        measured=[]
        for p in blockpaths:
            with np.load(p) as a:measured.append(a['xi_multipoles'][:,ids].reshape(-1))
        delta=np.r_[np.zeros(22),np.mean(measured,axis=0)-old]
        bridges.append({'phase':phase,'nblocks':len(blockpaths),'error_chi2_Csingle':float(metric.chi2(delta)),
                        'b1_contrast_projection':B@delta,'scope':'Absolute CPU vs original FFT estimate; paired weight increments use same CPU positions and backend.'})
        checks[phase+'_production_bridge']=bool(metric.chi2(delta)<.001)
    with np.load(OUT/'paired_fkp_results.npz') as a:
        checks['paired_arrays_finite']=all(np.all(np.isfinite(a[k])) for k in ('delta_data','delta_model','delta_residual'))
        checks['paired_residual_identity']=bool(np.allclose(a['delta_data']-a['delta_model'],a['delta_residual'],atol=1e-12,rtol=1e-10))
    checks['diagnostic_optimizers_success']=all(x['success'] for x in res['diagnostic_constant_delta_MAP'].values())
    report={'status':'pass' if all(checks.values()) else 'failed','checks':checks,'kernel_precision':kernel_reports,'production_bridges':bridges,
        'P_measurement_audits':pk_reports,'thresholds':{'IC_delta_q':.001,'IC_delta_b1':.001,'CPU_bridge_q':.001},
        'scope':'Diagnostic numerical gates only; no statement that frozen EZ covariance matches shuffled halo or that3phases determine an ensemble correction.',
        'source_sha256':sha(__file__)}
    save(OUT/'postflight.json',report)
    if not all(checks.values()):
        print(json.dumps({'status':'failed','failed_checks':[k for k,v in checks.items() if not v]}),flush=True);raise SystemExit(2)
    # 小交付仅包含结果/审计/源码；大目录、窗口和几何核在OUT有明确路径，不重复搬运。
    files=[OUT/f for f in ('conditional_modes.json','conditional_modes.npz','paired_fkp_results.json','paired_fkp_results.npz','external_bias_inventory.json','postflight.json')]
    files+=[ROOT/'codes/task432'/name for name in ('task432_fkp_conditional_modes.py','task432_fkp_paired.py','task432_fkp_ric_response.py','task432_fkp_analyze.py','task432_fkp_postflight.py','task432_fkp_inventory.py','run_task432_fkp_ric.sh')]
    manifest=[]
    for p in files:
        name=('source/'+p.name) if 'codes/task432' in str(p) else p.name
        manifest.append({'path':str(p),'delivery_path':name,'sha256':sha(p),'bytes':p.stat().st_size})
    save(OUT/'delivery_manifest.json',{'status':'pass','files':manifest})
    with tarfile.open(OUT/'diagnostic_delivery.tar.gz','w:gz') as archive:
        for p,row in zip(files,manifest):archive.add(p,arcname=row['delivery_path'])
        archive.add(OUT/'delivery_manifest.json',arcname='delivery_manifest.json')
    inventory=[{'path':str(p.relative_to(OUT)),'bytes':p.stat().st_size} for p in OUT.rglob('*') if p.is_file()]
    save(OUT/'repository_audit.json',{'scope':str(OUT),'file_count':len(inventory),'total_bytes':sum(r['bytes'] for r in inventory),
        'files':inventory,'science_inputs_removed':False,'recommended_entry':'paired_fkp_results.json and postflight.json',
        'large_files_retained':'paired/catalogs.npz and paired/kernel*.npz are reproducibility inputs; not default reading.'})
    print(json.dumps({'status':'pass','checks':len(checks),'kernel_precision':[{k:v for k,v in r.items() if k!='paths'} for r in kernel_reports]},default=lambda x:x.tolist()),flush=True)

if __name__=='__main__':main()
