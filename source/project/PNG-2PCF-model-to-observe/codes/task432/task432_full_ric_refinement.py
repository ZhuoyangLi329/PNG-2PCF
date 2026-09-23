#!/usr/bin/env python3
"""针对后验尾部触发的积分误差，增加两个独立scramble并保留原门限。

执行大纲：核对预先记录的新增seeds -> 比较两个互相独立的双核均值 ->
在原87个posterior检查点及扩展边缘点上检查0.01门限 -> 输出四核均值。
只降低几何积分噪声，不按b1结果选择kernel，不修改观测或covariance。
"""
import os
for key in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ[key] = '1'
import json
import numpy as np
from task432_full_ric_validation import merge, shell_poles, HybridGIC, Metric
from task432_full_ric_response import Response
from task432_full_ric_geometry import OUT, BASE, sha, save_json


def run():
    """以独立双核均值之间的差异而非拟合改善作为是否接收加密的标准。"""
    g = OUT/'geometry'
    plan = json.loads((g/'refinement_plan.json').read_text())
    new = [g/f'ph000_qmc{seed}_n32768_dchi2_refine1_midpoint_inmidpoint_ell8.npz'
           for seed in plan['additional_seeds']]
    if any(not p.exists() for p in new):
        raise RuntimeError('Additional independent kernels are incomplete')
    newmean = g/'refinement_independent_mean2_midpoint.npz'
    merge(new, newmean)
    with np.load(BASE/'frozen_inputs.npz') as a:
        centers, cov = a['xi_centers'], a['covariance']
    with np.load(g/'production_rr_mu.npz') as a:
        rr = a['rr_mean']
    original = Response(g/'validated_mean2_midpoint.npz'); refined = Response(newmean)
    dx = refined.xi_matrix(centers, rr_smu=rr)-original.xi_matrix(centers, rr_smu=rr)
    parent = OUT/'formal/postflight.json'
    old = json.loads(parent.read_text())
    probes = [(r['variant']+':'+r['point'], r['theta'][:2]) for r in old['records']]
    probes += [('extended', [f, b]) for f, b in ((-150., 2.85), (100., 2.85), (-150., 2.1), (100., 2.1))]
    model = HybridGIC(); metric = Metric(cov); records = []
    for label, (f, b) in probes:
        poles = shell_poles(model, original.edges, f, b)
        error = np.einsum('ild,ld->i', dx, poles)
        records.append({'point': label, 'theta': [f, b], 'error_chi2_Csingle': float(metric.chi2(np.r_[np.zeros(22), error]))})
    maximum = max(r['error_chi2_Csingle'] for r in records)
    report = {'status': 'pass' if maximum < .01 else 'failed', 'threshold_chi2_Csingle': .01,
              'max_error_chi2_Csingle': maximum, 'records': records,
              'comparison': [str(original.path), str(newmean)], 'reason': plan['reason'],
              'original_geometry_audit_sha256': sha(OUT/'smoke/geometry_validation.json'),
              'original_postflight_sha256': sha(parent),
              'method': 'Compare independent means of two scrambles, then average all four. Same error threshold as before; physical binning, data, covariance and priors unchanged.',
              'kernels': {str(p): sha(p) for p in [original.path, *new]}}
    save_json(OUT/'smoke/geometry_validation_mean4.json', report)
    if report['status'] == 'pass':
        merge([original.path, newmean], g/'validated_mean4_midpoint.npz')
    print(json.dumps({k: v for k, v in report.items() if k != 'records'}), flush=True)


if __name__ == '__main__':
    run()
