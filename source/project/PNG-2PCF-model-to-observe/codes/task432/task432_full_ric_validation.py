#!/usr/bin/env python3
"""在正式 MCMC 前审计几何收敛、phase、径向分辨率和 hybrid 插值。

执行大纲：共同理论输入比较独立几何核 -> 全局与径向 IC 归因 ->
仅在核通过后取两个独立 scramble 的预定平均 -> 编译候选 -> 直接模型校验。
核选择不使用拟合 b1 是否居中作为标准。
"""
import os
for key in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ[key] = '1'
import sys
import argparse
import json
from pathlib import Path
import numpy as np
from task432_full_ric_backend import ROOT, OUT
from task432_full_ric_geometry import BASE, PK_DIR, save_json, sha
from task432_full_ric_response import Response
for folder in ('codes/task432', 'codes/task43', 'codes/task44'):
    sys.path.insert(0, str(ROOT/folder))
from task432_hybrid_gic import HybridGIC
from task432_hybrid_jaxpower_0918_contract import Metric
from task43_rsd_boxsafe_p02_increment import WindowConvolvedP02Model, PAYLOAD_NPZ


def shell_poles(model, edges, f, b, nquad=4, nint=1200):
    """高精度、独立于候选参数插值的 hybrid 完整内层多极。"""
    q, w = np.polynomial.legendre.leggauss(nquad)
    lo, hi = edges[:-1, None], edges[1:, None]
    nodes = (hi-lo)*q/2+(hi+lo)/2
    weight = (hi-lo)/2*w*nodes**2/((hi**3-lo**3)/3)
    model._set_cumulants(f, b)
    return np.einsum('ldq,dq->ld', model.poles(nodes.ravel(), nint=nint).reshape(3, *nodes.shape), weight)


def paths():
    """预先指定独立取样和分辨率检查，不按拟合结果挑核。"""
    g = OUT/'geometry'
    return {name: g/file for name, file in {
        'x1': 'ph000_qmc4322217_n32768_dchi2_refine1_midpoint_inmidpoint_ell8.npz',
        'x2': 'ph000_qmc4322257_n32768_dchi2_refine1_midpoint_inmidpoint_ell8.npz',
        'xphase': 'ph001_qmc4322217_n32768_dchi2_refine1_midpoint_inmidpoint_ell8.npz',
        'xwidth1': 'ph000_qmc4322217_n32768_dchi1_refine1_midpoint_inmidpoint_ell8.npz',
        'xglobal': 'ph000_qmc4322217_n4096_dchi0_refine1_midpoint_inmidpoint_ell8.npz',
        'p1': 'ph000_qmc4322217_n32768_dchi2_refine1_endpoint_inmidpoint_ell2.npz',
        'p2': 'ph000_qmc4322257_n32768_dchi2_refine1_endpoint_inmidpoint_ell2.npz',
        'pglobal': 'ph000_qmc4322217_n4096_dchi0_refine1_endpoint_inmidpoint_ell2.npz'}.items()}


def merge(input_paths, output):
    """平均已经归一化的 cross/auto 概率核，避免两个积分目录总权重差混入结果。"""
    arrays = []
    for p in input_paths:
        with np.load(p) as a:
            arrays.append({k: a[k] for k in a.files})
    if not all(np.array_equal(arrays[0]['separation_edges'], a['separation_edges']) for a in arrays):
        raise RuntimeError('Cannot average different separation grids')
    first = arrays[0]
    meta = json.loads(str(first['meta_json'].item()))
    meta.update({'status': 'validated_geometry_candidate', 'sampling': 'Predetermined mean of two independent scrambled Sobol kernels',
                 'constituents': [{'path': str(p), 'sha256': sha(p)} for p in input_paths]})
    combined = {key: np.mean([a[key]/float(a['normalization']) for a in arrays], axis=0) for key in ('rr', 'cross', 'auto')}
    np.savez_compressed(output, separation_edges=first['separation_edges'], radial_labels=first['radial_labels'],
                        normalization=np.array(1.), meta_json=np.array(json.dumps(meta)), **combined)
    save_json(output.with_suffix('.json'), meta)


def geometry_audit():
    """检查独立 scramble、phase 和径向分辨率；误差门限对应完整 C_single。"""
    source = paths(); missing = [str(p) for p in source.values() if not p.exists()]
    if missing:
        save_json(OUT/'smoke/geometry_validation.json', {'status': 'waiting_for_kernels', 'missing': missing})
        print(json.dumps({'event': 'waiting_for_kernels', 'missing': missing}), flush=True)
        return
    with np.load(BASE/'frozen_inputs.npz') as a:
        pedges, centers, covariance = a['p_edges'], a['xi_centers'], a['covariance']
    with np.load(OUT/'geometry/production_rr_mu.npz') as a:
        rr = a['rr_mean']
    with np.load(PK_DIR/'task43_rsd_lightcone_p02_ph000_mesh256_kmax0p300_dk0p002.npz') as a:
        tk, te, edges = a['theory_k'], a['theory_ell'], a['theory_edges']
    with np.load(PAYLOAD_NPZ) as a:
        zeff = float(a['zeff'])
    responses = {name: Response(p) for name, p in source.items()}
    rows = np.r_[np.arange(13), np.arange(17, 26)]
    matrices = {name: response.xi_matrix(centers, rr_smu=rr) if name.startswith('x') else
                response.pk_theory_matrix(pedges, edges, te, nquad=16, inner_nquad=16).reshape(26, -1)[rows]
                for name, response in responses.items()}
    metric = Metric(covariance); hybrid = HybridGIC()
    pmodel = WindowConvolvedP02Model(np.zeros((1, len(tk))), tk, te, zeff=zeff, sigma_step=30.)
    records = []
    for f, b, sig in ((0., 2.4, 7.5), (30., 2.45, 3.), (-30., 2.35, 15.), (4.5, 2.38, 2.2)):
        poles = shell_poles(hybrid, responses['x1'].edges, f, b)
        x = {name: np.einsum('ild,ld->i', matrix, poles) for name, matrix in matrices.items() if name.startswith('x')}
        q = f*2*1.686*(b-1); fg = pmodel.f_growth
        coeff = np.array([b*b, 2*b*q, q*q, 2*b*fg, 2*q*fg, fg*fg])
        theory = pmodel._theory_basis(sig)@coeff
        p = {name: matrix@theory for name, matrix in matrices.items() if name.startswith('p')}
        tests = {'xi_independent_scramble': np.r_[np.zeros(22), x['x2']-x['x1']],
                 'xi_phase': np.r_[np.zeros(22), x['xphase']-x['x1']],
                 'xi_radial_1_vs_2': np.r_[np.zeros(22), x['xwidth1']-x['x1']],
                 'P_independent_scramble': np.r_[p['p2']-p['p1'], np.zeros(52)]}
        scores = {key: float(metric.chi2(v)) for key, v in tests.items()}
        oldgic = hybrid.correction(fnl=f, b1=b, nquad=8, nint=1200)
        global_vs_old = x['xglobal'].copy(); global_vs_old[:26] += oldgic
        record = {'theta': [f, b, sig], 'numerical_error_chi2': scores,
                  'global_vs_scalar_GIC_chi2': float(metric.chi2(np.r_[p['pglobal'], global_vs_old])),
                  'radial_minus_global': np.r_[.5*(p['p1']+p['p2'])-p['pglobal'],
                                              .5*(x['x1']+x['x2'])-x['xglobal']].tolist()}
        records.append(record)
        print(json.dumps({'event': 'geometry_validation_point', 'theta': [f, b, sig], 'scores': scores}), flush=True)
    gates = {key: max(r['numerical_error_chi2'][key] for r in records) < .01 for key in records[0]['numerical_error_chi2']}
    report = {'status': 'pass' if all(gates.values()) else 'failed', 'gates': gates, 'threshold_chi2_Csingle': .01,
              'records': records, 'source_sha256': sha(__file__), 'kernels': {k: str(v) for k, v in source.items()}}
    save_json(OUT/'smoke/geometry_validation.json', report)
    if report['status'] != 'pass':
        return
    merge([source['x1'], source['x2']], OUT/'geometry/validated_mean2_midpoint.npz')
    merge([source['p1'], source['p2']], OUT/'geometry/validated_mean2_endpoint.npz')


def model_audit(compiled):
    """独立直接 hybrid 与候选 emulator 比较，包括内层积分和原 raw-xi 插值。"""
    from task432_full_ric_compile import Engine
    engine = Engine('joint', compiled)
    with np.load(compiled) as a:
        matrix, fixed_shot = a['xi_response'], a['fixed_shot_vector'][22:]
        meta = json.loads(str(a['meta_json'].item()))
    with np.load(BASE/'frozen_inputs.npz') as a:
        centers = a['xi_centers']; covariance = a['covariance']
    edges = Response(Path(meta['xi_kernel'])).edges
    model = HybridGIC(); metric = Metric(covariance)
    points = [(0., 2.4), (4.5, 2.38), (-7.73, 2.3904), (-63.7, 2.273), (-31.3, 2.517),
              (18.7, 2.347), (57.3, 2.643), (96.1, 2.177), (-102.3, 2.713)]
    records = []
    for f, b in points:
        inner = shell_poles(model, edges, f, b)
        raw = model.poles(centers, nint=1200)[:2].reshape(-1)
        correction = np.einsum('ild,ld->i', matrix, inner)
        truth = raw+correction+fixed_shot
        emulated = engine.x([f, b])[0]
        records.append({'theta': [f, b], 'error_chi2_Csingle': float(metric.chi2(np.r_[np.zeros(22), emulated-truth]))})
    maximum = max(r['error_chi2_Csingle'] for r in records)
    report = {'status': 'pass' if maximum < .01 else 'failed', 'records': records,
              'max_error_chi2_Csingle': maximum, 'compiled_sha256': sha(compiled),
              'scope': 'Independent Nint1200/inner-quadrature4 truth vs original raw emulator + new RIC grid; posterior-tail postflight still required.'}
    save_json(compiled.parent/'model_validation.json', report)
    print(json.dumps(report), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=('geometry', 'model'))
    parser.add_argument('--compiled', type=Path)
    args = parser.parse_args()
    geometry_audit() if args.action == 'geometry' else model_audit(args.compiled)
