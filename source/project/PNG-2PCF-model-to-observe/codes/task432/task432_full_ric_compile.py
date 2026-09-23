#!/usr/bin/env python3
"""编译完整 RIC 的快速模型并执行明确标为 pilot 的多起点优化。

执行大纲：冻结原 74 维 data/covariance -> P 响应编译 sigma spline ->
原 hybrid raw xi 加确定性 RIC 网格 -> 保留原参数与 Hartlap -> 多起点优化。
该入口不采样 MCMC，不能把未通过全部几何 gate 的结果作为正式 posterior。
"""
import os
for key in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ[key] = '1'
import sys
import argparse
import json
import numpy as np
from pathlib import Path
from scipy.interpolate import CubicSpline, PPoly, RectBivariateSpline, RegularGridInterpolator
from scipy.optimize import least_squares
from task432_full_ric_backend import ROOT, OUT
from task432_full_ric_geometry import BASE, PK_DIR, save_json, sha
from task432_full_ric_response import Response
for folder in ('codes/task432', 'codes/task43', 'codes/task44'):
    sys.path.insert(0, str(ROOT/folder))
import task432_hybrid_jaxpower_0918_contract as common
from task432_hybrid_gic_0918_contract import finite_factors
from task43_rsd_boxsafe_p02_increment import WindowConvolvedP02Model, PAYLOAD_NPZ


class Emulator:
    def __init__(self, compiled, method):
        """保持 raw hybrid 原插值法，RIC 作为光滑确定性函数用双三次插值。"""
        with np.load(BASE/'xi_emulator.npz') as a:
            self.f, self.b, self.raw = a['f_grid'], a['b_grid'], a['values_no_gic']
        with np.load(compiled) as a:
            self.ric = a['xi_ric_grid']; self.shot = a['fixed_shot_vector'][22:]
            self.stochastic = a['sn0_ric_response'][22:] if 'sn0_ric_response' in a else np.zeros(52)
        self.method = method
        if method == 'linear':
            self.raw_interpolator = RegularGridInterpolator((self.f, self.b), self.raw, bounds_error=True)
        else:
            self.raw_splines = [RectBivariateSpline(self.f, self.b, self.raw[:, :, i], s=0) for i in range(52)]
        self.ric_splines = [RectBivariateSpline(self.f, self.b, self.ric[:, :, i], s=0) for i in range(52)]

    def correction(self, theta):
        """返回聚类 RIC；固定 Poisson shot 单独存储用于审计。"""
        t = np.atleast_2d(theta)
        return np.column_stack([s.ev(t[:, 0], t[:, 1]) for s in self.ric_splines])

    def __call__(self, theta):
        """移除旧 scalar GIC，以完整径向 cross+auto 和固定 shot 替代。"""
        t = np.atleast_2d(theta)
        raw = self.raw_interpolator(t[:, :2]) if self.method == 'linear' else np.column_stack([s.ev(t[:, 0], t[:, 1]) for s in self.raw_splines])
        sn = t[:, -1] if t.shape[1] in (3, 4) else np.zeros(len(t))
        return raw+self.correction(t)+self.shot+sn[:, None]*self.stochastic


class Engine(common.Engine):
    def __init__(self, variant, compiled):
        """复用原数据、先验、物理参数和 covariance；只注入确定性模型响应。"""
        audit = json.loads((BASE/'input_audit.json').read_text())
        previous_output = common.OUT
        try:
            common.OUT = BASE
            super().__init__(variant, audit['emulator_method'])
        finally:
            common.OUT = previous_output
        self.x = Emulator(compiled, audit['emulator_method'])
        with np.load(compiled) as a:
            self.delta_p = PPoly.construct_fast(a['p_ric_spline_c'], a['p_ric_spline_x'])
            self.fixed_p_shot = a['fixed_shot_vector'][:22]
            self.stochastic_p = a['sn0_ric_response'][:22] if 'sn0_ric_response' in a else np.zeros(22)
            stochastic_complete = 'sn0_ric_response' in a
        if variant == 'xi02' and stochastic_complete:
            self.names = ['fNL', 'b1', 'sn0']
            self.lower = common.LOWER[[0, 1, 3]]
            self.upper = common.UPPER[[0, 1, 3]]
        self.corrections = finite_factors(len(self.data), len(self.names))
        self.raw_metric = common.Metric(self.cov)
        self.metric = common.Metric(self.cov/self.corrections['hartlap'])

    def predict(self, theta):
        """批量 P、xi、joint 预测，无额外 RIC 振幅参数。"""
        t = np.atleast_2d(theta)
        if self.variant == 'xi02':
            return self.x(t)
        fnl, b, sig, sn = t.T
        q = fnl*2*1.686*(b-1); f = self.f
        coefficients = np.column_stack([b*b, 2*b*q, q*q, 2*b*f, 2*q*f, np.full(len(t), f*f)])
        p = np.einsum('nij,nj->ni', self.ps(sig)+self.delta_p(sig), coefficients)+sn[:, None]*(self.shot+self.stochastic_p)+self.fixed_p_shot
        return p if self.variant == 'p02' else np.column_stack([p, self.x(t)])


def compile_engine(tag, xi_path, p_path, production_rr=False):
    """生成有明确几何来源和源 hash 的独立 engine，不覆盖已存在文件。"""
    destination = OUT/'pilot'/tag
    destination.mkdir(parents=True, exist_ok=True)
    target = destination/'engine.npz'
    if target.exists():
        return target
    with np.load(BASE/'frozen_inputs.npz') as a:
        pedges, centers = a['p_edges'], a['xi_centers']
    with np.load(PK_DIR/'task43_rsd_lightcone_p02_ph000_mesh256_kmax0p300_dk0p002.npz') as a:
        tk, te, edges = a['theory_k'], a['theory_ell'], a['theory_edges']
    with np.load(PAYLOAD_NPZ) as a:
        zeff = float(a['zeff'])
    pr, xr = Response(p_path), Response(xi_path)
    rows = np.r_[np.arange(13), np.arange(17, 26)]
    p_matrix = pr.pk_theory_matrix(pedges, edges, te, nquad=16, inner_nquad=16).reshape(26, -1)[rows]
    pmodel = WindowConvolvedP02Model(p_matrix, tk, te, zeff=zeff)
    rr = None
    if production_rr:
        with np.load(OUT/'geometry/production_rr_mu.npz') as a:
            rr = a['rr_mean']
    x_matrix = xr.xi_matrix(centers, rr_smu=rr)
    physical_cache = OUT/'geometry/hybrid_inner_refine1_q2_n600.npz'
    with np.load(physical_cache) as a:
        if not np.array_equal(a['separation_edges'], xr.edges):
            raise RuntimeError('Physical cache and geometry distance bins disagree')
        ff, bb, poles = a['f_grid'], a['b_grid'], a['poles']
    ric_grid = np.einsum('ild,abld->abi', x_matrix, poles)
    # 按本次选定的几何、真实 RR 和正式 measured shot 幅度重算固定项。
    shot_audit = json.loads((OUT/'smoke/full_response_pilot.json').read_text())
    da = shot_audit['data_shot_amplitude']
    rp = shot_audit['random_P_shot_amplitude']
    rx = shot_audit['random_xi_shot_equivalent_P_amplitude']
    shot_templates = {}
    for los, response, ell in (('midpoint', xr, 8), ('endpoint', pr, 2)):
        for kind, n, width in (('radial', 65536, 2), ('global', 16384, 0)):
            path = OUT/f'geometry/ph000_shot_n{n}_dchi{width}_refine1_{los}_ell{ell}.npz'
            with np.load(path) as a:
                moment = (a['auto']-a['cross'])/response.meta['pair_volume']
            shot_templates[los, kind] = response.xi_project(moment, centers, rr_smu=rr) if los == 'midpoint' else response.p_project(moment, pedges, nquad=16).reshape(26)[rows]
    fixed_shot = np.r_[da*shot_templates['endpoint', 'radial']+rp*shot_templates['endpoint', 'global'],
                      da*shot_templates['midpoint', 'radial']+rx*shot_templates['midpoint', 'global']]
    from task432_full_ric_stochastic_audit import stochastic_response
    stochastic, stochastic_ratio, _, _ = stochastic_response(xi_path, p_path, rr)
    provenance = {'status': 'pilot_unvalidated', 'xi_kernel': str(xi_path), 'p_kernel': str(p_path),
                  'xi_kernel_sha256': sha(xi_path), 'p_kernel_sha256': sha(p_path),
                  'physical_grid': str(physical_cache), 'physical_grid_sha256': sha(physical_cache),
                  'baseline_data_covariance_sha256': sha(BASE/'frozen_inputs.npz'),
                  'source_sha256': sha(__file__), 'joint_free_parameters_changed': False,
                  'parameter_names': {'p02': ['fNL', 'b1', 'sigma_s_P', 'sn0'], 'xi02': ['fNL', 'b1', 'sn0'], 'joint': ['fNL', 'b1', 'sigma_s_P', 'sn0']},
                  'stochastic_note': 'Existing sn0 has its deterministic RIC response in all four observables; xi-only now marginalizes the same sn0 prior [-1,1]. No new RIC amplitude.',
                  'stochastic_I2_ratio': stochastic_ratio,
                  'xi_RR': 'production_25phase_mean' if production_rr else 'integration_catalogue',
                  'warning': 'Diagnostic optimization only; formal MCMC requires remaining geometry/LOS/interpolation gates.'}
    np.savez_compressed(target, p_ric_spline_c=pmodel.spline.c, p_ric_spline_x=pmodel.spline.x,
                        xi_ric_grid=ric_grid, f_grid=ff, b_grid=bb, fixed_shot_vector=fixed_shot,
                        sn0_ric_response=stochastic,
                        xi_response=x_matrix, p_response=p_matrix, meta_json=np.array(json.dumps(provenance)))
    save_json(destination/'provenance.json', provenance)
    return target


def optimize(compiled):
    """多起点 pilot 优化，保存 raw 与 Hartlap chi²，明确区分于最终 posterior。"""
    starts = [[-12., 2.42, 2.2, .14], [0., 2.4, 1., .1], [30., 2.3, 5., 0.], [-50., 2.55, 8., -.1]]
    result = {}
    for variant in ('p02', 'xi02', 'joint'):
        engine = Engine(variant, compiled)
        npar = len(engine.names)
        selected = [[x[0], x[1], x[3]] if variant == 'xi02' and npar == 3 else x[:npar] for x in starts]
        solutions = [least_squares(engine.residual, x, bounds=(engine.lower, engine.upper),
                                  x_scale='jac', max_nfev=1200, ftol=1e-11, xtol=1e-11, gtol=1e-11) for x in selected]
        best = min(solutions, key=lambda s: sum(s.fun*s.fun))
        prediction = engine.predict(best.x)[0]
        result[variant] = {'parameter_names': engine.names, 'theta': best.x.tolist(),
                           'raw_chi2': float(engine.raw_metric.chi2(engine.data-prediction)),
                           'hartlap_chi2': float(sum(best.fun*best.fun)),
                           'success': bool(best.success), 'multistart_chi2': [float(sum(s.fun*s.fun)) for s in solutions],
                           'prediction': prediction.tolist(), 'jacobian': best.jac.tolist()}
        print(json.dumps({'event': 'pilot_map', 'variant': variant, 'theta': best.x.tolist(),
                          'raw_chi2': result[variant]['raw_chi2']}), flush=True)
    biases = {v: r['theta'][1] for v, r in result.items()}
    report = {'status': 'pilot_not_final', 'compiled_engine': str(compiled), 'results': result, 'b1': biases,
              'joint_b1_between_MAP': min(biases['p02'], biases['xi02']) <= biases['joint'] <= max(biases['p02'], biases['xi02'])}
    save_json(compiled.parent/'pilot_map.json', report)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--tag', required=True)
    parser.add_argument('--xi-kernel', type=Path, required=True)
    parser.add_argument('--p-kernel', type=Path, required=True)
    parser.add_argument('--production-rr', action='store_true')
    args = parser.parse_args()
    optimize(compile_engine(args.tag, args.xi_kernel, args.p_kernel, args.production_rr))
