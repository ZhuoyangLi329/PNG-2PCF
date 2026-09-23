#!/usr/bin/env python3
"""最终数值审计：既有主输入未变、优化收敛、GIC 直接预测、EZ-only 默认和源码溯源。"""
import task432_joint_b1_diagnostics as d
from task432_joint_b1_diagnostics import np,json,save,sha,Path,Metric
from task432_hybrid_gic import HybridGIC

def inspect(obj,path=''):
    """递归统计带 success 字段的结果，保留失败位置而不静默跳过。"""
    found=[]
    if isinstance(obj,dict):
        if 'success' in obj:found.append((path,bool(obj['success'])))
        for k,v in obj.items():found+=inspect(v,path+'/'+str(k))
    elif isinstance(obj,list):
        for i,v in enumerate(obj):found+=inspect(v,path+'/'+str(i))
    return found

def run():
    """所有通过门限写入 postflight_audit；任何门限失败都终止交付。"""
    d.initialize();assert list(d.COVS)==['ezmock1000']
    old=json.loads((d.OUT/'run_manifest.json').read_text());frozen={p:sha(p)==expected for p,expected in old['inputs']['ezmock1000'].items()};assert all(frozen.values())
    results={name:json.loads((d.OUT/name).read_text()) for name in ['primary.json','refinement.json','covariance_bootstrap.json','calibration.json','phases.json','operators.json','contract_audit.json']}
    checked={}
    for name,r in results.items():
        scope=r['covariances']['ezmock1000'] if 'covariances' in r else {k:v for k,v in r['cases'].items() if k.startswith('ezmock1000')} if 'cases' in r else r
        items=inspect(scope);checked[name]={'success_items':len(items),'failures':[k for k,v in items if not v]}
        assert not checked[name]['failures'],checked[name]
    model=HybridGIC();metric=Metric(d.COVS['ezmock1000']);checks=[]
    for v,r in results['primary.json']['covariances']['ezmock1000']['baseline'].items():
        t=np.array(r['theta']);direct,c=model.prediction(t);err=np.r_[np.zeros(22),direct-d.MODEL.x(t)[0]]
        checks.append({'variant':v,'theta':t,'GIC':c,'direct_emulator_error_chi2_Cmean25':metric.chi2(err)*25})
    assert max(r['direct_emulator_error_chi2_Cmean25'] for r in checks)<.001
    with np.load(d.OUT/'operator_delta_grid.npz',allow_pickle=False) as z:assert len(z['missing_window_phases'])==24 and 'p_window_mean' not in z
    scan=results['primary.json']['covariances']['ezmock1000']['cross_scan'];assert min(r['min_correlation_eigenvalue'] for r in scan)>0
    # 均值注入闭合的优化结果已单独保存，此处再检查中心和误差口径。
    closure=results['calibration.json']['cases']['ezmock1000_noise_div25']['noisefree'];anchor=np.array([0,2.42])
    assert max(np.max(abs(np.array(v['theta'])[:2]-anchor)) for v in closure.values())<1e-4
    report={'status':'pass','default_covariance':'ezmock1000','formal_inputs_unchanged':frozen,'optimizer_checks':checked,'direct_model_checks':checks,'cpu_affinity':sorted(d.os.sched_getaffinity(0)),'source_hashes':{str(p):sha(p) for p in [Path(d.__file__),d.ROOT/'codes/task432/task432_joint_b1_operators.py',d.ROOT/'codes/task432/task432_joint_b1_refine.py',d.ROOT/'codes/task432/task432_joint_b1_contract_audit.py',d.ROOT/'codes/task432/task432_plot_joint_b1_diagnostics.py']},'scope_changes':'Initial shared-covariance numerical results remain archived. All final interpretation/figures and subsequent runner defaults use EZmock1000 only.','limitations':results['contract_audit.json']['limitations'],'gates':{'all_optimizers_converged':True,'frozen_inputs_unchanged':True,'direct_model_accuracy':True,'positive_definite_cross_scan':True,'noisefree_recovery':True,'missing_mean_window_explicit':True}}
    save(d.OUT/'postflight_audit.json',report);d.manifest();print(json.dumps({'status':'pass','checks':checked}))

if __name__=='__main__':run()
