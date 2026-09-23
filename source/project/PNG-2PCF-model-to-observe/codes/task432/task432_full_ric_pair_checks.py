#!/usr/bin/env python3
"""以现有 Corrfunc CPU 后端桥接生产 midpoint/40mu/W² LS 估计量。

执行大纲：小目录有序 pair 的独立 dense sum -> phase000 第一个真实 random
block 的 DD/DR/RR -> 与原 cucount xi_smu_by_random 的偶多极比较并缓存计数。
只写独立验证目录，不改正式 data/random/covariance；最多8 CPU。
"""
import os
for key in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ[key] = '8' if key == 'OMP_NUM_THREADS' else '1'
import argparse
import json
import time
import numpy as np
from pathlib import Path
from Corrfunc.mocks import DDsmu_mocks
from task432_full_ric_backend import ROOT, OUT
from task432_full_ric_geometry import BASE, HALO_MANIFEST, save_json, sha
from task432_full_ric_response import angular_bin_integrals
from task432_full_ric_reference import pair_geometry

EDGES = np.arange(30., 351., 10.)
MU_EDGES = np.linspace(-1., 1., 41)


def count(A, B=None, edges=EDGES, nmu=40, threads=8):
    """返回 ordered weighted counts，归一化在调用方显式使用生产 W²。"""
    position, weight = A
    kwargs = {} if B is None else dict(X2=B[0][:, 0], Y2=B[0][:, 1], Z2=B[0][:, 2], weights2=B[1])
    result = DDsmu_mocks(B is None, threads, edges, 1., nmu, *position.T,
                         weights1=weight, weight_type='pair_product', los_type='midpoint',
                         gpu=False, bin_type='custom', **kwargs)
    return (result['npairs']*result['weightavg']).reshape(len(edges)-1, nmu)


def smoke():
    """用小 Cartesian 目录验证包版本的坐标、mu方向与有序 pair 计数语义。"""
    rng = np.random.default_rng(4322229)
    a = rng.uniform(100, 1000, (181, 3)); b = rng.uniform(100, 1000, (137, 3))
    wa, wb = rng.uniform(.5, 1.5, len(a)), rng.uniform(.5, 1.5, len(b))
    edges, mu_edges = np.arange(30., 931., 100.), np.linspace(-1, 1, 9)
    reports = []
    for autocorr in (True, False):
        pa, pb = a, a if autocorr else b
        w1, w2 = wa, wa if autocorr else wb
        g = pair_geometry(np.concatenate([pa, pb]))
        n = len(pa); s, mu = g['s'][:n, n:], g['mu_mid'][:n, n:]
        direct = np.histogram2d(s.ravel(), mu.ravel(), bins=(edges, mu_edges), weights=np.outer(w1, w2).ravel())[0]
        measured = count((pa, w1), None if autocorr else (pb, w2), edges, 8, 2)
        errors = [float(np.max(abs(measured-direct))), float(np.max(abs(measured-direct[:, ::-1])))]
        relative = min(errors)/max(float(direct.max()), 1.)
        reports.append({'autocorr': autocorr, 'mu_sign_error': errors, 'relative_error': relative})
    passed = max(r['relative_error'] for r in reports) < 1e-10
    save_json(OUT/'smoke/corrfunc_dense_bridge.json', {'status': 'pass' if passed else 'failed', 'reports': reports})
    if not passed:
        raise RuntimeError('Corrfunc pair semantics failed independent dense sum')
    print(json.dumps({'event': 'Corrfunc_dense_bridge', 'status': 'pass'}), flush=True)


def read_catalog(path, z_edges, fkp, limit=None):
    """完全复用生产的 float32 Cartesian 和分段 FKP；可读取首个连续 random block。"""
    with np.load(path, allow_pickle=False) as a:
        sl = slice(None, limit)
        p = np.column_stack([a[k][sl] for k in ('X', 'Y', 'Zcart')]).astype('f4')
        z = a['Z'][sl].astype('f8')
        base = a['WEIGHT'][sl].astype('f8') if 'WEIGHT' in a.files else np.ones(len(z))
    index = np.clip(np.searchsorted(z_edges, z, side='right')-1, 0, len(fkp)-1)
    return p, (base*fkp[index]).astype('f4')


def project(xi):
    """保持生产 Legendre 精确 mu-bin 积分而非 bin-center 近似。"""
    weights = .5*np.array([1., 5.])[:, None]*angular_bin_integrals([0, 2], MU_EDGES)
    return (xi@weights.T).T


def bridge():
    """对完整 ph000+block0 桥接，保存后续 paired 测量可复用的 DD 与基准。"""
    smoke()
    rows = [json.loads(line) for line in HALO_MANIFEST.read_text().splitlines() if line.strip()]
    row = next(r for r in rows if r['phase'] == 'ph000')
    with np.load(row['lightcone_fkp_path']) as a:
        ze, fw = a['z_edges'], a['fkp_weights']
    data = read_catalog(row['lightcone_catalog_path'], ze, fw)
    n = len(data[1])
    randoms = read_catalog(row['lightcone_random_path'], ze, fw, n)
    with np.load(row['lightcone_random_path']) as a:
        if not np.all(a['RANDOM_INDEX'][:n] == 0):
            raise RuntimeError('First random block is not contiguous block 0')
    dest = OUT/'paired/halo_ph000'; dest.mkdir(parents=True, exist_ok=True)
    totals = [np.sum(c[1], dtype='f8') for c in (data, randoms)]
    measured = {}
    for name, A, B, norm in (('DD', data, None, totals[0]**2),
                              ('DR', data, randoms, totals[0]*totals[1]),
                              ('RR', randoms, None, totals[1]**2)):
        target = dest/f'bridge_block0_{name}.npz'
        if target.exists():
            with np.load(target) as a:
                measured[name] = a['normalized_counts']
            continue
        started = time.monotonic()
        raw = count(A, B)
        measured[name] = raw/norm
        np.savez_compressed(target, weighted_counts=raw, norm=norm, normalized_counts=measured[name])
        print(json.dumps({'event': 'production_pair_bridge', 'pair': name, 'ndata': n,
                          'elapsed_s': time.monotonic()-started}), flush=True)
    xi = (measured['DD']-2*measured['DR']+measured['RR'])/measured['RR']
    with np.load(row['lightcone_xi_path']) as a:
        truth = a['xi_smu_by_random'][0]; centers = a['s']
    prediction, reference = project(xi), project(truth)
    with np.load(BASE/'frozen_inputs.npz') as a:
        covariance = a['covariance']; fitcenters = a['xi_centers']
    ids = [np.flatnonzero(np.isclose(centers, s, atol=1e-8, rtol=0))[0] for s in fitcenters]
    delta = (prediction-reference)[:, ids].reshape(-1)
    vector = np.r_[np.zeros(22), delta]
    metric = float(vector@np.linalg.solve(covariance, vector))
    report = {'status': 'pass' if metric < .001 else 'failed', 'error_chi2_Csingle': metric,
              'max_absolute_multipole_error': float(np.max(abs(prediction-reference))),
              'production_path': row['lightcone_xi_path'], 'production_sha256': sha(row['lightcone_xi_path']),
              'estimator': 'Corrfunc midpoint signed40mu, explicit W² norms, first production Nrandom=Ndata block',
              'cpu_affinity': sorted(os.sched_getaffinity(0))}
    save_json(OUT/'smoke/corrfunc_production_bridge.json', report)
    np.savez_compressed(dest/'production_bridge.npz', xi_smu=xi, xi_multipoles=prediction,
                        production_xi_smu=truth, production_multipoles=reference, s=centers)
    print(json.dumps(report), flush=True)
    if report['status'] != 'pass':
        raise RuntimeError('Production Corrfunc bridge failed')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=('smoke', 'bridge'))
    args = parser.parse_args()
    (smoke if args.action == 'smoke' else bridge)()
