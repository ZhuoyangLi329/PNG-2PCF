#!/usr/bin/env python3
"""检验有限 random/data 比例下，实际 shuffled-z 的 shot 投影分工。

执行大纲：固定总数 data multinomial -> 每次从 data 径向分布重新抽 random
-> 分别以实际权重和总权重归一 -> 检查四分量二次型的 Monte Carlo 均值。
预期 leading-order covariance 为 Q_rad D Q_rad^T / Nd 加
Q_global D Q_global^T / Nr；不能将有限 random 的噪声全部径向投影。
本实验验证实际 bootstrap 的噪声逻辑，不替代真实目录的全部几何验证。
"""
import os
for key in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ[key] = '1'
import json
import numpy as np
from task432_full_ric_backend import OUT
from task432_full_ric_geometry import save_json
from task432_full_ric_reference import estimator_matrices, pair_geometry
from task432_full_ric_response import angular_bin_integrals


def run():
    """两种 NR/ND 比例使用相同离散 survey，统计实际均值与预言误差。"""
    rng = np.random.default_rng(4322221)
    nr, na = 5, 16
    labels = np.repeat(np.arange(nr), na)
    directions = abs(rng.normal(size=(na, 3)))
    directions /= np.linalg.norm(directions, axis=1)[:, None]
    position = (np.linspace(900., 1700., nr)[:, None, None]*directions[None]).reshape(-1, 3)
    pr = np.array([.13, .17, .24, .26, .20])
    p = np.repeat(pr/na, na)
    weight = np.repeat([.30, .38, .35, .31, .28], na)
    u = p*weight/(p@weight)
    diagonal = np.diag(p*weight**2/(p@weight)**2)
    same = labels[:, None] == labels[None, :]
    Br = u[:, None]*same/(same@u)[None, :]
    Bg = np.outer(u, np.ones(len(u)))
    Qr, Qg = np.eye(len(u))-Br, np.eye(len(u))-Bg
    # 使用相同真实 LOS 的四分量二次型；每类取一个有限尺度 bin。
    E, names, _ = estimator_matrices(position, u, np.array([300., 900.]), np.linspace(-1, 1, 5), [.007])
    # estimator_matrices 原本作用于 overdensity，转为作用于归一化 weighted counts。
    matrices = E/np.outer(u, u)[None]
    geometry = pair_geometry(position)
    mu_edges = np.linspace(-1, 1, 5)
    selectors = np.asarray([(geometry['s'] >= 300.) & (geometry['s'] < 900.) &
                            (geometry['mu_mid'] >= a) & (geometry['mu_mid'] < b)
                            for a, b in zip(mu_edges[:-1], mu_edges[1:])], dtype='f8')
    projection = .5*np.array([1., 5.])[:, None]*angular_bin_integrals([0, 2], mu_edges)
    ndata, nreal, batch = 20000, 32768, 256
    records = []
    for ratio in (1, 25):
        nrandom = ndata*ratio
        full_cov = Qr@diagonal@Qr.T/ndata + Qg@diagonal@Qg.T/nrandom
        expected_cov = full_cov-diagonal*(1/ndata+1/nrandom)
        naive_cov = Qr@diagonal@Qr.T*(1/ndata+1/nrandom)-diagonal*(1/ndata+1/nrandom)
        expected = np.einsum('aij,ij->a', matrices, expected_cov)
        naive = np.einsum('aij,ij->a', matrices, naive_cov)
        total = np.zeros(len(matrices)); total2 = total.copy(); ratio_delta = np.zeros(2)
        for first in range(0, nreal, batch):
            data = rng.multinomial(ndata, p, size=batch)
            group_fraction = data.reshape(batch, nr, na).sum(axis=2)/ndata
            probabilities = np.repeat(group_fraction/na, na, axis=1)
            randoms = np.asarray([rng.multinomial(nrandom, row) for row in probabilities])
            dw, rw = data*weight, randoms*weight
            dn, rn = dw/dw.sum(axis=1)[:, None], rw/rw.sum(axis=1)[:, None]
            field = dn-rn
            value = np.einsum('bi,aij,bj->ba', field, matrices, field, optimize=True)
            # 当前四个 E 在 s=0 处为零；保留一般接触项扣除以明确 shot 记账。
            contact = (data*weight**2)/dw.sum(axis=1)[:, None]**2 + (randoms*weight**2)/rw.sum(axis=1)[:, None]**2
            value -= contact@np.diagonal(matrices, axis1=1, axis2=2).T
            numerator_mu = np.einsum('bi,mij,bj->bm', field, selectors, field, optimize=True)
            denominator_mu = np.einsum('bi,mij,bj->bm', rn, selectors, rn, optimize=True)
            actual_ls = (numerator_mu/denominator_mu)@projection.T
            ratio_delta += (actual_ls-value[:, -2:]).sum(axis=0)
            value[:, -2:] = actual_ls
            total += value.sum(axis=0); total2 += (value*value).sum(axis=0)
        mean = total/nreal
        stderr = np.sqrt(np.maximum(total2/nreal-mean*mean, 0)/(nreal-1))
        z = (mean-expected)/stderr
        records.append({'random_multiplier': ratio, 'ndata': ndata, 'nreal': nreal, 'observables': names,
                        'mean': mean.tolist(), 'standard_error': stderr.tolist(), 'expected': expected.tolist(),
                        'z_scores': z.tolist(), 'naive_radial_all_noise': naive.tolist(),
                        'actual_LS_minus_fixed_RR_mean': (ratio_delta/nreal).tolist(),
                        'naive_minus_expected_over_stderr': ((naive-expected)/stderr).tolist()})
        print(json.dumps({'ratio': ratio, 'z_scores': z.tolist()}), flush=True)
    passed = max(abs(v) for r in records for v in r['z_scores']) < 5.
    save_json(OUT/'smoke/shuffled_finite_random_shot.json', {'status': 'pass' if passed else 'failed',
              'records': records, 'interpretation': 'Finite random noise has global normalization projection; data noise has radial projection. Leading 1/N expansion checked with actual multinomial radial bootstrap.'})
    if not passed:
        raise RuntimeError('Finite random shot bootstrap test failed')


if __name__ == '__main__':
    run()
