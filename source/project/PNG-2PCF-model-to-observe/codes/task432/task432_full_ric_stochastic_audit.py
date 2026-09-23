#!/usr/bin/env python3
"""量化既有 sn0 常数 stochastic 参数经完整 RIC 后的 P 与 xi 响应。

执行大纲：使用 nbar 加权白噪声接触核 -> 固定原 sn0*1e4 单位 ->
与完整 covariance 比较 -> 判断 xi-only 是否必须纳入这个既有 nuisance。
不引入独立 RIC 振幅；计算只使用冻结选择和已核对的 RR。
"""
import json
import numpy as np
from task432_full_ric_backend import OUT
from task432_full_ric_geometry import BASE, save_json
from task432_full_ric_response import Response


def stochastic_response(xi_path, p_path, rr):
    """在指定几何/RR 上投影每单位原 sn0 的确定性白噪声修正。"""
    with np.load(BASE/'frozen_inputs.npz') as a:
        pedges, centers = a['p_edges'], a['xi_centers']
    pool = json.loads((OUT/'geometry/random_pool_ph000.json').read_text())
    with np.load(pool['fkp_summary']) as a:
        continuum_I2 = float(np.sum(a['nbar']**2*a['fkp_weights']**2*a['volume_shell']))
    ratio = continuum_I2/pool['P_I2']
    pieces = []
    for los, ell, geometry in (('endpoint', 2, p_path), ('midpoint', 8, xi_path)):
        response = Response(geometry)
        with np.load(OUT/f'geometry/ph000_shot_white_n65536_dchi2_refine1_{los}_ell{ell}.npz') as a:
            probability = (a['auto']-a['cross'])/response.meta['pair_volume']
        if los == 'midpoint':
            vector = response.xi_project(probability, centers, rr_smu=rr)
        else:
            vector = response.p_project(probability, pedges, nquad=16).reshape(26)[np.r_[np.arange(13), np.arange(17, 26)]]
        pieces.append(1e4*ratio*vector)
    full = np.r_[pieces[0], pieces[1]]
    return full, ratio, continuum_I2, pool['P_I2']


def run():
    """报告原 sn0 先验内忽略 xi 白噪声投影造成的模型误差。"""
    with np.load(BASE/'frozen_inputs.npz') as a:
        cov = a['covariance']
    with np.load(OUT/'geometry/production_rr_mu.npz') as a:
        rr = a['rr_mean']
    x = OUT/'geometry/ph000_qmc4322217_n32768_dchi2_refine1_midpoint_inmidpoint_ell8.npz'
    p = OUT/'geometry/ph000_qmc4322217_n32768_dchi2_refine1_endpoint_inmidpoint_ell2.npz'
    full, ratio, continuum_I2, measured_I2 = stochastic_response(x, p, rr)
    def chi(vector):
        return float(vector@np.linalg.solve(cov, vector))
    xi_vector = np.r_[np.zeros(22), full[22:]]
    values = {str(sn): {'full_chi2_Csingle': chi(sn*full), 'xi_only_error_in_joint_metric': chi(sn*xi_vector)}
              for sn in (.1, .3, .5, 1.)}
    report = {'status': 'stochastic_response_audit', 'continuum_I2': continuum_I2, 'production_I2': measured_I2,
              'I2_ratio': ratio, 'existing_sn0_prior': [-1., 1.], 'per_unit_sn0_response': full.tolist(),
              'amplitude_probes': values, 'xi_response_negligible_over_original_prior': chi(xi_vector) < .01,
              'rule': 'Constant underlying P stochastic term has a nbar-weighted IC contact kernel; geometry has no free amplitude beyond existing sn0.'}
    np.savez_compressed(OUT/'geometry/stochastic_sn0_response.npz', response=full, I2_ratio=ratio)
    save_json(OUT/'smoke/stochastic_sn0_audit.json', report)
    print(json.dumps({k: v for k, v in report.items() if k != 'per_unit_sn0_response'}), flush=True)


if __name__ == '__main__':
    run()
