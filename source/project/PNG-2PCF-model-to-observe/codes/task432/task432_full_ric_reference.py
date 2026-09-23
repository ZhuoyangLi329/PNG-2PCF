#!/usr/bin/env python3
"""完整径向 IC 的小目录参考实现与独立代数/Poisson 检查。

执行大纲
--------
1. 用分组加权平均构造径向投影 Pi，显式检查 Pi^2=Pi 和 Q 的零模。
2. 分别计算两个 cross、auto，并与 Q Xi Q^T 的直接矩阵结果比较。
3. 将相同场二点函数送入 endpoint Fourier 和 midpoint LS 两个估计器。
4. 用独立 Poisson 抽样检查投影后的固定噪声项；不把它当正式 survey 校准。

本文件是可扩展几何核的可信小样本参考，不产生正式 RIC 拟合结果。
Xi 指密度场二点函数，与 EZmock 的 74 维数据协方差完全不同。
"""
from __future__ import annotations

import os
for _name in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):
    os.environ[_name] = '1'
import argparse
import json
from pathlib import Path
import numpy as np
from scipy.special import eval_legendre, spherical_jn


def radial_projector(labels, weights):
    """输入每点径向分组和正投影权重，输出 Pi；组内归一化与外层 FKP 权重区分。"""
    labels = np.asarray(labels)
    weights = np.asarray(weights, dtype=float)
    if np.any(weights <= 0) or labels.shape != weights.shape:
        raise ValueError('Labels and positive weights must have equal shape')
    same = labels[:, None] == labels[None, :]
    return same * weights[None, :] / (same @ weights)[:, None]


def projected_terms(xi, labels, weights):
    """用逐组求和返回两个 cross 和 auto；独立于 dense Q@Xi@Q.T 的实现。"""
    xi = np.asarray(xi)
    left = np.zeros_like(xi)
    right = np.zeros_like(xi)
    auto = np.zeros_like(xi)
    groups = [np.flatnonzero(labels == label) for label in np.unique(labels)]
    for ids in groups:
        prob = weights[ids] / weights[ids].sum()
        left[ids, :] = prob @ xi[ids, :]
        right[:, ids] = (xi[:, ids] @ prob)[:, None]
    for ids in groups:
        prob = weights[ids] / weights[ids].sum()
        auto[ids, :] = prob @ right[ids, :]
    return {'cross_left': left, 'cross_right': right, 'auto': auto,
            'corrected': xi - left - right + auto}


def pair_geometry(position):
    """返回有序点对的距离、midpoint mu 和两个端点 mu；仅在自对处将 mu 置零。"""
    d = position[:, None, :] - position[None, :, :]
    s = np.linalg.norm(d, axis=-1)
    shat = np.divide(d, s[..., None], out=np.zeros_like(d), where=s[..., None] > 0)
    direction = position / np.linalg.norm(position, axis=1)[:, None]
    midpoint = position[:, None, :] + position[None, :, :]
    midpoint /= np.linalg.norm(midpoint, axis=-1)[..., None]
    return {'s': s, 'mu_mid': np.sum(shat * midpoint, axis=-1),
            'mu_first': np.einsum('ijk,ik->ij', shat, direction),
            'mu_second': np.einsum('ijk,jk->ij', shat, direction)}


def estimator_matrices(position, outer_weight, s_edges, mu_edges, k_values, *, require_full_rr=True):
    """构造小目录的线性估计器矩阵 E，使预测为 sum(E*Xi)。

    P 使用端点对称化的局部 LOS 与 (-i)^ell j_ell，归一化单位 pair volume。
    xi 使用逐 (s,mu) 单元 RR 归一化及 Legendre 的精确 bin 积分。
    这里只验证估计器代数；真实 survey 的 I2、随机目录和离散分箱另行接入。
    """
    g = pair_geometry(position)
    s = g['s']
    pair_weight = np.outer(outer_weight, outer_weight)
    # 不计物理自对；被投影到不同点的 shot 项依然必须保留。
    pair_weight[np.diag_indices_from(pair_weight)] = 0
    result, names = [], []
    for ell in (0, 2):
        angular = .5 * (eval_legendre(ell, g['mu_first']) + eval_legendre(ell, g['mu_second']))
        for k in k_values:
            result.append((2 * ell + 1) * (-1)**(ell // 2) * spherical_jn(ell, k*s)
                          * angular * pair_weight / pair_weight.sum())
            names.append(f'P{ell}_k{k:g}')
    rr = np.zeros((len(s_edges)-1, len(mu_edges)-1))
    for ell in (0, 2):
        coeff = np.zeros(ell+1); coeff[ell] = 1
        primitive = np.polynomial.legendre.legint(coeff)
        angular_integral = np.diff(np.polynomial.legendre.legval(mu_edges, primitive))
        for i, (lo, hi) in enumerate(zip(s_edges[:-1], s_edges[1:])):
            E = np.zeros_like(s)
            for j, (a, b) in enumerate(zip(mu_edges[:-1], mu_edges[1:])):
                selected = (s >= lo) & (s < hi) & (g['mu_mid'] >= a) & (g['mu_mid'] < b)
                weight = pair_weight * selected
                rr[i, j] = weight.sum()
                if rr[i, j] <= 0:
                    if require_full_rr:
                        raise ValueError(f'Empty RR bin {(i, j)}')
                    continue
                E += (2*ell+1)/2 * angular_integral[j] * weight / rr[i, j]
            result.append(E); names.append(f'xi{ell}_s{lo:g}_{hi:g}')
    return np.asarray(result), names, rr


def smoke(output):
    """小目录确定性代数与独立 Poisson 检验，输出明确 scope 的 JSON。"""
    rng = np.random.default_rng(4322201)
    nr, na = 8, 24
    angular = np.abs(rng.normal(size=(na, 3)))
    angular /= np.linalg.norm(angular, axis=1)[:, None]
    radii = np.linspace(850., 1650., nr)
    labels = np.repeat(np.arange(nr), na)
    position = (radii[:, None, None] * angular[None, :, :]).reshape(-1, 3)
    # 刻意使用不均匀投影权重，检查 Pi 非对称时是否错误省略了 transpose。
    projection_weight = rng.uniform(.7, 1.3, len(position))
    outer_weight = rng.uniform(.5, 1.4, len(position))
    pi = radial_projector(labels, projection_weight)
    q = np.eye(len(pi)) - pi
    g = pair_geometry(position)
    s = g['s']
    xi = np.exp(-(s/280.)**2) + .2 * (s/220.)**2*np.exp(-(s/220.)**2)*eval_legendre(2, g['mu_mid'])
    terms = projected_terms(xi, labels, projection_weight)
    direct = q @ xi @ q.T
    matrices, names, rr = estimator_matrices(position, outer_weight,
                                             np.array([300., 500., 700.]), np.linspace(-1., 1., 9), [.005, .015])
    observable = lambda x: np.einsum('aij,ij->a', matrices, x)
    radial = np.sin(radii/500)[labels]
    errors = {'idempotence': float(np.max(abs(pi@pi-pi))),
              'constant_annihilation': float(np.max(abs(q@np.ones(len(q))))),
              'radial_annihilation': float(np.max(abs(q@radial))),
              'group_sum_vs_dense': float(np.max(abs(terms['corrected']-direct))),
              'four_component_estimator': float(np.max(abs(observable(terms['corrected'])-observable(direct))))}
    # Poisson fractional density -> 同一 Q；检验 shot 项在有限分离的响应。
    rate = rng.uniform(25., 60., len(position))
    noise = np.diag(1/rate)
    nprojected = q@noise@q.T
    check = projected_terms(noise, labels, projection_weight)
    errors['shot_terms_vs_dense'] = float(np.max(abs(check['corrected']-nprojected)))
    nreal = 8192
    delta = (rng.poisson(rate, size=(nreal, len(rate))) - rate)/rate
    transformed = delta@q.T
    selected = [0, 2, 4, 6]  # 一个 P0、P2、xi0、xi2；全部由独立 Poisson 抽样验证。
    samples = np.column_stack([np.sum((transformed@matrices[i])*transformed, axis=1) for i in selected])
    truth = observable(nprojected)[selected]
    uncertainty = samples.std(axis=0, ddof=1)/np.sqrt(nreal)
    zscore = (samples.mean(axis=0)-truth)/uncertainty
    gates = {'algebra': max(errors.values()) < 2e-12,
             'poisson_four_components': bool(np.max(abs(zscore)) < 4.5),
             'rr_positive': bool(np.all(rr > 0))}
    report = {'status': 'pass' if all(gates.values()) else 'failed', 'gates': gates,
              'scope': 'Small discrete catalogue reference only; not survey validation or exact finite-count LS calibration.',
              'errors': errors, 'npoints': len(position), 'names': names,
              'raw_prediction': observable(xi).tolist(), 'projected_prediction': observable(direct).tolist(),
              'cross_left': observable(terms['cross_left']).tolist(),
              'cross_right': observable(terms['cross_right']).tolist(), 'auto': observable(terms['auto']).tolist(),
              'poisson': {'nreal': nreal, 'names': [names[i] for i in selected],
                          'expected': truth.tolist(), 'measured': samples.mean(axis=0).tolist(),
                          'standard_error': uncertainty.tolist(), 'zscore': zscore.tolist()},
              'cpu_affinity': sorted(os.sched_getaffinity(0)) if hasattr(os, 'sched_getaffinity') else None}
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2)+'\n')
    print(json.dumps({'status': report['status'], 'errors': errors, 'poisson_z': zscore.tolist()}), flush=True)
    if report['status'] != 'pass':
        raise RuntimeError('Reference smoke failed')


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output', type=Path, required=True)
    smoke(p.parse_args().output)
