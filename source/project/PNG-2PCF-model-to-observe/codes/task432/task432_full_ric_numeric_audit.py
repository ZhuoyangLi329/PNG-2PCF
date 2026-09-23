#!/usr/bin/env python3
"""独立积分校验解析 Hankel，并汇总已经完成的几何核收敛试算。

执行大纲：解析原函数逐点对比 quad -> band 积分对比直接积分 ->
对已存在的 midpoint 核运行共同 hybrid 输入；不启动拟合。
"""
import os
for key in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ[key] = '1'
import argparse
import numpy as np
from scipy.integrate import quad
from scipy.special import spherical_jn
from task432_full_ric_backend import OUT
from task432_full_ric_geometry import save_json
from task432_full_ric_response import bessel_volume_primitive, pk_band_to_xi, hybrid_pilot


def audit():
    """用自适应积分校验 ell0/2/4，返回明确的通过状态。"""
    rows = []
    for ell in (0, 2, 4):
        for x in (.0001, .05, .3, .99, 1., 4., 30., 150.):
            truth = quad(lambda t: t*t*spherical_jn(ell, t), 0., x,
                         epsabs=1e-12, epsrel=1e-12, limit=400)[0]
            prediction = bessel_volume_primitive(ell, np.array([x]))[0]
            rows.append({'ell': ell, 'x': x, 'relative_error': float(abs(prediction-truth)/max(abs(truth), 1e-10))})
    band_rows = []
    for ell in (0, 2, 4):
        for r in (1., 55., 500., 3300.):
            for edges in ((0., .0011), (.0077, .0088), (.27, .28)):
                truth = (-1)**(ell//2)*quad(lambda k: k*k*spherical_jn(ell, r*k), *edges,
                                          epsabs=1e-15, epsrel=1e-11, limit=400)[0]/(2*np.pi**2)
                prediction = pk_band_to_xi([r], [edges], ell)[0, 0]
                band_rows.append({'ell': ell, 'r': r, 'k_edges': edges,
                                  'scaled_error': float(abs(prediction-truth)/max(abs(truth), 1e-14))})
    passed = max(r['relative_error'] for r in rows) < 1e-7 and max(r['scaled_error'] for r in band_rows) < 1e-7
    save_json(OUT/'smoke/hankel_primitive.json', {'status': 'pass' if passed else 'failed',
                                                'primitive': rows, 'band_integral': band_rows})
    if not passed:
        raise RuntimeError('Hankel integration audit failed')
    print('Hankel primitive and band quadrature: pass', flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--hybrid', action='store_true')
    args = parser.parse_args()
    audit()
    if args.hybrid:
        paths = [OUT/'geometry'/name for name in (
            'ph000_independent_n8192_dchi10_refine1_midpoint_ell8.npz',
            'ph000_independent_n16384_dchi4_refine1_midpoint_ell8.npz',
            'ph000_independent_n32768_dchi2_refine1_midpoint_ell8.npz')]
        hybrid_pilot([p for p in paths if p.exists()], OUT/'smoke/hybrid_geometry_convergence.json')
