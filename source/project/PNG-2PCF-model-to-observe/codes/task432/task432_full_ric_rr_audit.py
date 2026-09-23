#!/usr/bin/env python3
"""冻结生产 25-phase RR(mu)，审计积分目录 RR 和有限多极截断的影响。

执行大纲：从实际生产 xi 文件读取已归一化 RR_smu -> 完全匹配 9.18 bin
-> 比较 phase0、phase mean 与积分 RR -> qmax6/8 比较 -> 保存固定输入。
生产文件只读；有限 block 分母另由 bootstrap shot 的实际 LS 实验约束。
"""
import os
for key in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ[key] = '1'
import sys
import json
import argparse
import copy
import numpy as np
from pathlib import Path
from task432_full_ric_backend import ROOT, OUT
from task432_full_ric_geometry import BASE, HALO_MANIFEST, save_json, sha
from task432_full_ric_response import Response
for folder in ('codes/task432', 'codes/task43', 'codes/task44'):
    sys.path.insert(0, str(ROOT/folder))
from task432_hybrid_gic import HybridGIC
from task432_hybrid_jaxpower_0918_contract import Metric


def run(path):
    """用共同物理输入量化 RR 分母与 qmax 截断，不比较不同动力学。"""
    with np.load(BASE/'frozen_inputs.npz') as a:
        centers, covariance = a['xi_centers'], a['covariance']
    rows = [json.loads(line) for line in HALO_MANIFEST.read_text().splitlines() if line.strip()]
    allrr, metadata = [], []
    for row in rows:
        source = Path(row['lightcone_xi_path'])
        with np.load(source) as a:
            indices = [np.flatnonzero(np.isclose(a['s'], s, atol=1e-8, rtol=0))[0] for s in centers]
            rr = a['RR_smu'][indices]
            if rr.shape != (26, 40) or np.any(rr <= 0):
                raise RuntimeError('Production RR bins do not match frozen selection')
            if not np.array_equal(a['mu_edges'], np.linspace(-1, 1, 41)):
                raise RuntimeError('Production mu edges differ')
        allrr.append(rr)
        metadata.append({'phase': row['phase'], 'path': str(source), 'sha256': sha(source)})
    allrr = np.asarray(allrr)
    np.savez_compressed(OUT/'geometry/production_rr_mu.npz', rr_by_phase=allrr, rr_mean=allrr.mean(axis=0),
                        xi_centers=centers, meta_json=np.array(json.dumps(metadata)))
    response = Response(path)
    matrices = {'integration_rr': response.xi_matrix(centers),
                'production_ph000_rr': response.xi_matrix(centers, rr_smu=allrr[0]),
                'production_mean_rr': response.xi_matrix(centers, rr_smu=allrr.mean(axis=0))}
    truncated = copy.copy(response)
    truncated.ells = response.ells[:4]
    truncated.rr = response.rr[:4]
    truncated.kernel = response.kernel[:4]
    matrices['qmax6_mean_rr'] = truncated.xi_matrix(centers, rr_smu=allrr.mean(axis=0))
    q, w = np.polynomial.legendre.leggauss(4)
    lo, hi = response.edges[:-1, None], response.edges[1:, None]
    radii = (hi-lo)*q/2+(hi+lo)/2
    sw = (hi-lo)/2*w*radii**2/((hi**3-lo**3)/3)
    model = HybridGIC(); metric = Metric(covariance)
    records = []
    for f, b in ((0., 2.4), (30., 2.45), (-30., 2.35)):
        model._set_cumulants(f, b)
        poles = np.einsum('ldq,dq->ld', model.poles(radii.ravel(), nint=1200).reshape(3, *radii.shape), sw)
        vectors = {name: np.einsum('ild,ld->i', matrix, poles) for name, matrix in matrices.items()}
        comparisons = {}
        for first, second in (('integration_rr', 'production_ph000_rr'), ('production_ph000_rr', 'production_mean_rr'),
                              ('qmax6_mean_rr', 'production_mean_rr')):
            delta = vectors[first]-vectors[second]
            comparisons[first+'__'+second] = {'chi2_Csingle': float(metric.chi2(np.r_[np.zeros(22), delta])),
                                             'delta_xi': delta.tolist()}
        records.append({'theta': [f, b], 'comparisons': comparisons})
        print(json.dumps({'theta': [f, b], 'chi2': {k: v['chi2_Csingle'] for k, v in comparisons.items()}}), flush=True)
    save_json(OUT/'smoke/production_rr_audit.json', {'status': 'RR_numerical_audit', 'kernel': str(path),
                                                  'production_sources': metadata, 'records': records})


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('kernel', type=Path)
    run(parser.parse_args().kernel)
