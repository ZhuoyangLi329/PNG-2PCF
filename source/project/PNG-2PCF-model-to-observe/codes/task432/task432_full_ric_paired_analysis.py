#!/usr/bin/env python3
"""整理真实四分量 shuffled/reference 配对，不把少量 realization 当精密标定。

执行大纲：精确匹配冻结 k/s bins -> 差分对比径向减全局的模型 ->
加入对应 data-Poisson 差分 -> 输出逐分量符号、幅度及统计噪声尺度。
这里的 C_single 仅供标尺，不是配对差分的 covariance。
"""
import os
for k in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ[k] = '1'
import json
import numpy as np
from task432_full_ric_geometry import BASE, OUT, save_json
from task432_full_ric_response import Response


def run():
    """使用同一固定理论点，不据少量目录拟合经验 RIC 振幅。"""
    with np.load(BASE/'frozen_inputs.npz') as a:
        pe, xc, cov = a['p_edges'], a['xi_centers'], a['covariance']
    with np.load(OUT/'geometry/production_rr_mu.npz') as a:
        rr = a['rr_mean']
    geometry = json.loads((OUT/'smoke/geometry_validation.json').read_text())
    assert geometry['status'] == 'pass'
    theory = np.asarray(geometry['records'][0]['radial_minus_global'])
    da = json.loads((OUT/'smoke/full_response_pilot.json').read_text())['data_shot_amplitude']
    correction = []
    for los, ell in (('endpoint', 2), ('midpoint', 8)):
        r = Response(OUT/f'geometry/validated_mean2_{los}.npz')
        terms = []
        for n, width in ((65536, 2), (16384, 0)):
            with np.load(OUT/f'geometry/ph000_shot_n{n}_dchi{width}_refine1_{los}_ell{ell}.npz') as a:
                mom = (a['auto']-a['cross'])/r.meta['pair_volume']
            terms.append(r.xi_project(mom, xc, rr_smu=rr) if los == 'midpoint' else
                         r.p_project(mom, pe, nquad=16).reshape(26)[np.r_[np.arange(13), np.arange(17, 26)]])
        correction.append(da*(terms[0]-terms[1]))
    theory += np.concatenate(correction)
    pieces = {'P0': slice(0, 13), 'P2': slice(13, 22), 'xi0': slice(22, 48), 'xi2': slice(48, 74)}
    records = []; vectors = []
    for name in ('halo_ph000', 'ezmock_0', 'ezmock_1'):
        dest = OUT/'paired'/name
        with np.load(dest/'paired_pk.npz') as a:
            edges = a['k_edges']; p = a['difference']
            if edges.ndim == 1:
                edges = np.column_stack([edges[:-1], edges[1:]])
            ids = []
            for edge in pe:
                found = np.flatnonzero(np.all(np.isclose(edges, edge, rtol=0, atol=1e-10), axis=1))
                if len(found) != 1:
                    raise RuntimeError(f'P bins do not match: {name}, {edge}')
                ids.append(found[0])
            pv = np.r_[p[0, ids], p[1, np.asarray(ids)[4:]]]
        with np.load(dest/'paired_xi.npz') as a:
            ids = [np.flatnonzero(np.isclose(a['s'], s, atol=1e-8))[0] for s in xc]
            xv = a['difference'][:, ids].reshape(-1)
        v = np.r_[pv, xv]
        if len(v) != 74 or not np.all(np.isfinite(v)):
            raise RuntimeError('Invalid paired vector')
        vectors.append(v)
        metrics = {}
        for label, sl in pieces.items():
            c = cov[sl, sl]; t, y = theory[sl], v[sl]
            ip = lambda x, z: float(x@np.linalg.solve(c, z))
            metrics[label] = {'model_norm_Csingle': ip(t, t)**.5,
                              'measured_norm_Csingle': ip(y, y)**.5,
                              'difference_norm_Csingle': ip(y-t, y-t)**.5,
                              'whitened_direction_cosine': ip(t, y)/max((ip(t,t)*ip(y,y))**.5, 1e-100),
                              'component_sign_agreement': float(np.mean(t*y > 0))}
        records.append({'name': name, 'components': metrics,
                        'measurement_metadata': json.loads((dest/'paired_catalogs.json').read_text())})
    np.savez_compressed(OUT/'paired/paired_74vectors.npz', names=np.array([r['name'] for r in records]),
                        differences=np.asarray(vectors), model_reference=theory)
    report = {'status': 'awaiting_scientific_review', 'theory_theta': geometry['records'][0]['theta'],
              'records': records, 'model_reference': theory.tolist(),
              'scope': 'One halo and two actual covariance EZ mocks, P with 4N randoms and xi with two N-sized blocks. Finite-realization direction/scale check, not an ensemble mean calibration.',
              'limitations': 'C_single is only a noise ruler, not paired covariance. The common halo fiducial model is approximate for EZ mocks. Halo reference selection pools 24 independent phases. No covariance update or empirical RIC amplitude fitted.'}
    save_json(OUT/'smoke/paired_validation.json', report)
    print(json.dumps({'records': [{k: v for k, v in r.items() if k != 'measurement_metadata'} for r in records]}), flush=True)


if __name__ == '__main__':
    run()
