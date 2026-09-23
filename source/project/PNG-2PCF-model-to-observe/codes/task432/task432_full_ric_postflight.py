#!/usr/bin/env python3
"""在实际 posterior 随机点、尾部和 MAP 检查 hybrid 数值与核误差。

执行大纲：读取已收敛原始链 -> 每组24点及两端尾部/MAP ->
直接计算替代插值 -> 检查独立几何核 -> MAP 高阶积分复查。
不根据结果移动后验或改变 covariance。
"""
import os
for key in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ[key] = '1'
import argparse
import json
from pathlib import Path
import numpy as np
from task432_full_ric_validation import shell_poles, paths, HybridGIC, Metric, WindowConvolvedP02Model, PAYLOAD_NPZ
from task432_full_ric_compile import Engine
from task432_full_ric_geometry import BASE, OUT, PK_DIR, sha, save_json
from task432_full_ric_response import Response


def run(compiled, run_name='formal', comparison=None):
    """全部检查用冻结完整 C_single 衡量，原 sn0 响应两侧一致加入。"""
    model = HybridGIC(); engine = Engine('joint', compiled)
    with np.load(BASE/'frozen_inputs.npz') as a:
        centers, cov, pedges = a['xi_centers'], a['covariance'], a['p_edges']
    with np.load(OUT/'geometry/production_rr_mu.npz') as a:
        rr = a['rr_mean']
    metric = Metric(cov)
    with np.load(compiled) as a:
        matrix = a['xi_response']; fixed = a['fixed_shot_vector'][22:]
        pmatrix = a['p_response']
        stochastic = a['sn0_ric_response'][22:]
        meta = json.loads(str(a['meta_json'].item()))
    edges = Response(Path(meta['xi_kernel'])).edges
    with np.load(PK_DIR/'task43_rsd_lightcone_p02_ph000_mesh256_kmax0p300_dk0p002.npz') as a:
        tk, te, tedges = a['theory_k'], a['theory_ell'], a['theory_edges']
    with np.load(PAYLOAD_NPZ) as a:
        zeff = float(a['zeff'])
    pm = WindowConvolvedP02Model(np.zeros((1, len(tk))), tk, te, zeff=zeff, sigma_step=30.)
    px = paths()
    if comparison:
        px['x1'], px['x2'] = comparison
    delta_matrix = Response(px['x2']).xi_matrix(centers, rr_smu=rr)-Response(px['x1']).xi_matrix(centers, rr_smu=rr)
    prows = np.r_[np.arange(13), np.arange(17, 26)]
    independent_p = [Response(px[key]).pk_theory_matrix(pedges, tedges, te, nquad=16, inner_nquad=16).reshape(26, -1)[prows]
                     for key in ('p1', 'p2')]
    delta_pmatrix = independent_p[1]-independent_p[0]
    rng = np.random.default_rng(4322327)
    records = []
    for variant in ('p02', 'xi02', 'joint'):
        dest = OUT/run_name/variant
        summary = json.loads((dest/'summary.json').read_text()); assert summary['status'] == 'pass'
        with np.load(dest/'samples.npz') as a:
            flat = a['chain'].reshape(-1, len(summary['parameter_names']))
        probes = [('random', flat[i]) for i in rng.choice(len(flat), 24, replace=False)]
        for col in (0, 1):
            for quantile in (1., 99.):
                i = np.argmin(abs(flat[:, col]-np.percentile(flat[:, col], quantile)))
                probes.append((f'tail_{col}_{quantile:g}', flat[i]))
        probes.append(('MAP', np.asarray(summary['map_theta'])))
        for label, t in probes:
            f, b, sn = float(t[0]), float(t[1]), float(t[-1])
            inner = shell_poles(model, edges, f, b)
            raw = model.poles(centers, nint=1200)[:2].reshape(-1)
            direct = raw+np.einsum('ild,ld->i', matrix, inner)+fixed+sn*stochastic
            error = engine.x([f, b, sn])[0]-direct
            geometry_error = np.einsum('ild,ld->i', delta_matrix, inner)
            rec = {'variant': variant, 'point': label, 'theta': t.tolist(),
                   'emulator_chi2_Csingle': float(metric.chi2(np.r_[np.zeros(22), error])),
                   'independent_kernel_chi2_Csingle': float(metric.chi2(np.r_[np.zeros(22), geometry_error]))}
            if len(t) == 4:
                q = f*2*1.686*(b-1); fg = engine.f
                coeff = np.array([b*b, 2*b*q, q*q, 2*b*fg, 2*q*fg, fg*fg])
                pd = engine.delta_p(t[2])@coeff-pmatrix@(pm._theory_basis(t[2])@coeff)
                rec['P_spline_chi2_Csingle'] = float(metric.chi2(np.r_[pd, np.zeros(52)]))
                gp = delta_pmatrix@(pm._theory_basis(t[2])@coeff)
                rec['P_independent_kernel_chi2_Csingle'] = float(metric.chi2(np.r_[gp, np.zeros(52)]))
            if label == 'MAP':
                refined = shell_poles(model, edges, f, b, nquad=8, nint=2400)
                refined_raw = model.poles(centers, nint=2400)[:2].reshape(-1)
                dd = refined_raw-raw+np.einsum('ild,ld->i', matrix, refined-inner)
                rec['integration_chi2_Csingle'] = float(metric.chi2(np.r_[np.zeros(22), dd]))
            records.append(rec)
        print(json.dumps({'event': 'postflight_variant', 'variant': variant, 'points': len(probes)}), flush=True)
    maxima = {key: max(r.get(key, 0.) for r in records) for key in
              ('emulator_chi2_Csingle', 'independent_kernel_chi2_Csingle', 'integration_chi2_Csingle', 'P_spline_chi2_Csingle', 'P_independent_kernel_chi2_Csingle')}
    gates = {'emulator_below_0p01': maxima['emulator_chi2_Csingle'] < .01,
             'geometry_below_0p01': maxima['independent_kernel_chi2_Csingle'] < .01,
             'integration_below_0p001': maxima['integration_chi2_Csingle'] < .001,
             'P_spline_below_0p001': maxima['P_spline_chi2_Csingle'] < .001,
             'P_geometry_below_0p01': maxima['P_independent_kernel_chi2_Csingle'] < .01}
    # 独立从1000条冻结mock向量重建C，并核验三种参数接口映射到同一预测。
    covariance_source = BASE.parent/'hybrid_ezmock1000_0918_contract_v1/ezmock1000_covariance_and_stack.npz'
    with np.load(covariance_source) as a:
        rebuilt = np.cov(a['stack'], rowvar=False, ddof=1)
        indices_valid = bool(np.array_equal(a['production_indices'], np.arange(1000)))
        stack_shape = list(a['stack'].shape)
    pe, xe = Engine('p02', compiled), Engine('xi02', compiled)
    probes = np.asarray([[0., 2.4, 7.5, .2], [-30., 2.35, 3., -.4], [40., 2.5, 12., .7]])
    joint_predictions = engine.predict(probes)
    split_predictions = np.column_stack([pe.predict(probes), xe.predict(probes[:, [0, 1, 3]])])
    contract = {'nmock': 1000, 'stack_shape': stack_shape, 'production_indices_complete': indices_valid,
                'covariance_rebuild_relative_frobenius': float(np.linalg.norm(rebuilt-cov)/np.linalg.norm(cov)),
                'split_joint_max_abs_prediction_difference': float(np.max(abs(joint_predictions-split_predictions))),
                'xi_parameter_names': xe.names, 'xi_bounds': [xe.lower.tolist(), xe.upper.tolist()],
                'covariance_source_sha256': sha(covariance_source), 'frozen_input_sha256': sha(BASE/'frozen_inputs.npz')}
    gates['frozen_EZ1000_contract'] = indices_valid and stack_shape == [1000, 74] and contract['covariance_rebuild_relative_frobenius'] < 1e-12
    gates['parameter_mapping_consistent'] = contract['split_joint_max_abs_prediction_difference'] < 1e-10 and xe.names == ['fNL', 'b1', 'sn0']
    report = {'status': 'pass' if all(gates.values()) else 'failed', 'gates': gates, 'maxima': maxima,
              'records': records, 'compiled_sha256': sha(compiled), 'contract': contract,
              'scope': '24 random actual posterior samples + fNL/b1 1/99 percentile nearest samples + MAP for each fit; refinement at each MAP. Independent kernels tested with a common physical input.'}
    save_json(OUT/run_name/'postflight.json', report)
    print(json.dumps({k: v for k, v in report.items() if k != 'records'}), flush=True)


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__); p.add_argument('--compiled', type=Path, required=True)
    p.add_argument('--run-name', default='formal')
    p.add_argument('--xi-comparison', type=Path, nargs=2)
    a = p.parse_args(); run(a.compiled, a.run_name, a.xi_comparison)
