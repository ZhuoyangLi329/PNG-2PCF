#!/usr/bin/env python3
"""在冻结数据合同上检查 P 与 xi 完整 RIC 响应，输出可审计的数值误差。

执行大纲：复用 P 理论原参数化 -> 编译 band 响应并验证积分精度 ->
以实测 Poisson shot 幅度接入两套估计量 -> 保存对原 GIC 模型的定点变化。
本脚本不运行正式拟合，不用模型改变大小替代拟合优劣结论。
"""
import os
for key in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ[key] = '1'
import json
import sys
import numpy as np
from task432_full_ric_backend import ROOT, OUT
from task432_full_ric_geometry import BASE, PK_DIR, save_json
from task432_full_ric_response import Response
for folder in ('codes/task432', 'codes/task43', 'codes/task44'):
    sys.path.insert(0, str(ROOT/folder))
from task432_hybrid_jaxpower_0918_contract import Metric
from task43_rsd_boxsafe_p02_increment import WindowConvolvedP02Model, PAYLOAD_NPZ
from task432_hybrid_gic import HybridGIC


def run():
    """使用真正的理论 band、观测 bin 和 P normalization，检查全部 74 维。"""
    with np.load(BASE/'frozen_inputs.npz') as a:
        pedges, centers, covariance = a['p_edges'], a['xi_centers'], a['covariance']
    metric = Metric(covariance)
    pfile = PK_DIR/'task43_rsd_lightcone_p02_ph000_mesh256_kmax0p300_dk0p002.npz'
    with np.load(pfile) as a:
        theory_edges, tk, te = a['theory_edges'], a['theory_k'], a['theory_ell']
        measured_shot = float(np.ravel(a['shotnoise_ell0'])[0])
        norm = float(np.ravel(a['norm_ell0'])[0])
        if not np.allclose(a['shotnoise_ell0'], measured_shot):
            raise RuntimeError('Expected constant measured Poisson amplitude')
    with np.load(PAYLOAD_NPZ) as a:
        zeff = float(a['zeff'])
    responses = {los: Response(OUT/f'geometry/ph000_independent_n32768_dchi2_refine1_{los}_ell{ell}.npz')
                 for los, ell in (('midpoint', 8), ('endpoint', 2))}
    pr, xr = responses['endpoint'], responses['midpoint']
    metadata = json.loads(pfile.with_suffix('.json').read_text())
    dw, rw = metadata['data'], metadata['random']
    alpha = dw['weight_sum']/rw['weight_sum']
    data_amplitude = dw['weight2_sum']/norm
    random_p_amplitude = alpha**2*rw['weight2_sum']/norm
    if not np.isclose(data_amplitude+random_p_amplitude, measured_shot, rtol=1e-6):
        raise RuntimeError('P measured shot does not equal recorded data+random amplitudes')
    # xi 的生产调用为 count2(D,D)，norm=W_D^2；25 个 N_R=N_D blocks 各自除 RR。
    # 随机 bootstrap 的条件噪声只受全局归一化约束，不能套用数据的径向投影。
    with np.load(rw['path'], allow_pickle=False) as a:
        weights, blocks = a['WEIGHT_TOTAL'].astype('f8'), a['RANDOM_INDEX']
    bw = np.bincount(blocks, weights=weights)
    bt = np.bincount(blocks, weights=weights**2)
    random_x_amplitude = xr.meta['pair_volume']*float(np.mean(bt/bw**2))
    kept = np.r_[np.arange(13), np.arange(17, 26)]
    matrices = {q: pr.pk_theory_matrix(pedges, theory_edges, te, nquad=q, inner_nquad=q).reshape(26, -1)[kept]
                for q in (8, 16)}
    # 单行虚拟窗口仅用于复用冻结 P 物理基函数，正式窗口不变。
    pmodel = WindowConvolvedP02Model(np.zeros((1, len(tk))), tk, te, zeff=zeff, sigma_step=30.)
    xm = HybridGIC()
    points = [(0., 2.4, 7.5, 0.), (30., 2.45, 3., .1), (-30., 2.35, 15., -.1)]
    shot_templates = {}; global_templates = {}
    for los, response in responses.items():
        ell = 8 if los == 'midpoint' else 2
        path = OUT/f'geometry/ph000_shot_n65536_dchi2_refine1_{los}_ell{ell}.npz'
        with np.load(path) as a:
            if not np.array_equal(a['separation_edges'], response.edges):
                raise RuntimeError('Shot and clustering grids differ')
            moment = (a['auto']-a['cross'])/response.meta['pair_volume']
        if los == 'midpoint':
            shot_templates[los] = response.xi_project(moment, centers)
        else:
            shot_templates[los] = response.p_project(moment, pedges, nquad=16).reshape(26)[kept]
        path = OUT/f'geometry/ph000_shot_n16384_dchi0_refine1_{los}_ell{ell}.npz'
        with np.load(path) as a:
            moment = (a['auto']-a['cross'])/response.meta['pair_volume']
        global_templates[los] = response.xi_project(moment, centers) if los == 'midpoint' else response.p_project(moment, pedges, nquad=16).reshape(26)[kept]
    shot = np.r_[data_amplitude*shot_templates['endpoint']+random_p_amplitude*global_templates['endpoint'],
                 data_amplitude*shot_templates['midpoint']+random_x_amplitude*global_templates['midpoint']]
    high_density_approximation = measured_shot*np.r_[shot_templates['endpoint'], shot_templates['midpoint']]
    records = []
    q, w = np.polynomial.legendre.leggauss(4)
    lo, hi = xr.edges[:-1, None], xr.edges[1:, None]
    radii = (hi-lo)*q/2+(hi+lo)/2
    sw = (hi-lo)/2*w*radii**2/((hi**3-lo**3)/3)
    xmatrix = xr.xi_matrix(centers)
    for f, b, sig, sn in points:
        fgr = pmodel.f_growth; png = f*2*1.686*(b-1)
        coeff = np.array([b*b, 2*b*png, png*png, 2*b*fgr, 2*png*fgr, fgr*fgr])
        theory = pmodel._theory_basis(sig)@coeff
        dp = matrices[16]@theory
        dp_diff = (matrices[8]-matrices[16])@theory
        xm._set_cumulants(f, b)
        poles = xm.poles(radii.ravel(), nint=1200).reshape(3, *radii.shape)
        averaged = np.einsum('ldq,dq->ld', poles, sw)
        dx = np.einsum('ild,ld->i', xmatrix, averaged)
        center_dx = np.einsum('ild,ld->i', xmatrix, xm.poles(xr.center, nint=1200))
        gic = xm.correction(fnl=f, b1=b, nquad=8, nint=1200)
        relative = np.r_[dp, dx]+shot
        relative[22:48] += gic
        records.append({'theta': [f, b, sig, sn], 'delta_p_clustering': dp.tolist(), 'delta_x_clustering': dx.tolist(),
                        'old_gic': float(gic), 'delta_from_baseline': relative.tolist(),
                        'model_shift_chi2_Csingle': float(metric.chi2(relative)),
                        'P_quadrature_error_chi2': float(metric.chi2(np.r_[dp_diff, np.zeros(52)])),
                        'xi_center_vs_shell_error_chi2': float(metric.chi2(np.r_[np.zeros(22), center_dx-dx]))})
        print(json.dumps({k: v for k, v in records[-1].items() if not isinstance(v, list)}), flush=True)
    report = {'status': 'pilot_unvalidated', 'measured_Poisson_shot': measured_shot,
              'data_shot_amplitude': data_amplitude, 'random_P_shot_amplitude': random_p_amplitude,
              'random_xi_shot_equivalent_P_amplitude': random_x_amplitude,
              'shot_rule': 'Data radial projection + random global projection, with production W^2 pair normalization; existing sn0 nuisance unchanged.',
              'shot_delta_vector': shot.tolist(), 'shot_chi2_Csingle': float(metric.chi2(shot)),
              'finite_random_vs_high_density_approx_chi2': float(metric.chi2(shot-high_density_approximation)), 'records': records,
              'pending': ['finite-random LS/shot check', 'geometry accuracy', 'P input LOS', 'phase/window bridge']}
    save_json(OUT/'smoke/full_response_pilot.json', report)
    np.savez_compressed(OUT/'geometry/pilot_compiled_response.npz', p_matrix=matrices[16],
                        xi_matrix=xmatrix, xi_inner_nodes=radii, xi_inner_weights=sw,
                        shot_vector=shot, theory_edges=theory_edges, theory_ell=te, theory_k=tk,
                        p_edges=pedges, xi_centers=centers)


if __name__ == '__main__':
    run()
