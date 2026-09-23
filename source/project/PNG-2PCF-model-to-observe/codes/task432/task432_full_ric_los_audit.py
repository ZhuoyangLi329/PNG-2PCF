#!/usr/bin/env python3
"""用独立小目录积分量化 P 正式 local-two-endpoint 理论与 midpoint RIC 的差。

执行大纲：解析积分 L2(k.a)L2(k.b) -> 冻结 P-band Hankel 多极 ->
独立 A/B 目录投影差值 -> endpoint Fourier 估计量 -> 完整 covariance 度量。
local 表达式直接来自冻结 jaxpower mesh2.py 的 coeff2 映射，而非猜测 wide-angle 阶数。
本文件只做误差审计；不会默默把任何新动力学项加入正式模型。
"""
import os
for key in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ[key] = '1'
import sys
import argparse
import json
import numpy as np
from scipy.special import eval_legendre, spherical_jn
from task432_full_ric_backend import ROOT, OUT
from task432_full_ric_geometry import BASE, PK_DIR, save_json
from task432_full_ric_qmc import samples
from task432_full_ric_response import pk_band_to_xi, bessel_volume_primitive
for folder in ('codes/task432', 'codes/task43', 'codes/task44'):
    sys.path.insert(0, str(ROOT/folder))
from task432_hybrid_jaxpower_0918_contract import Metric
from task43_rsd_boxsafe_p02_increment import WindowConvolvedP02Model, PAYLOAD_NPZ


def geometry(a, b):
    """返回矩形点对的距离、两个端点 mu、midpoint mu 和端点夹角。"""
    delta = a[:, None]-b[None]
    s = np.linalg.norm(delta, axis=-1)
    direction = delta/np.maximum(s[..., None], 1e-30)
    ua = a/np.linalg.norm(a, axis=1)[:, None]
    ub = b/np.linalg.norm(b, axis=1)[:, None]
    mu1 = np.einsum('ijk,ik->ij', direction, ua)
    mu2 = np.einsum('ijk,jk->ij', direction, ub)
    mid = a[:, None]+b[None]
    mid /= np.linalg.norm(mid, axis=-1)[..., None]
    mum = np.einsum('ijk,ijk->ij', direction, mid)
    return s, mu1, mu2, mum, ua@ub.T


def coefficients(mu1, mu2, c):
    """local P4 对 H0(P4)、H2(P4)、H4(P4) 的精确角系数。"""
    # H_l = i^l integral k²P j_l /(2pi²)；因此 j2 tensor 的系数须反号。
    j2 = (-2+3*c*c+3*mu1*mu1+3*mu2*mu2-9*c*mu1*mu2)/7
    j4 = 9/4*((1+2*c*c)/35-(mu1*mu1+mu2*mu2+4*c*mu1*mu2)/7+mu1*mu1*mu2*mu2)
    return (7/18*(eval_legendre(2, c)-1),
            -5/18*(eval_legendre(2, mu1)+eval_legendre(2, mu2))-35/18*j2,
            35/18*j4)


def coefficient_smoke():
    """平行 LOS 极限必须回到 xi_l L_l，避免解析 tensor 推导中的符号错误。"""
    mu = np.linspace(-1, 1, 101)
    a, b, c = coefficients(mu, mu, np.ones_like(mu))
    error = max(np.max(abs(a)), np.max(abs(b)), np.max(abs(c-eval_legendre(4, mu))))
    # 对非平行方向独立积分 Fourier 球面，不能只测试退化极限。
    z, wz = np.polynomial.legendre.leggauss(48)
    phi = np.arange(64)*2*np.pi/64
    khat = np.stack(np.broadcast_arrays(np.sqrt(1-z[:, None]**2)*np.cos(phi),
                                       np.sqrt(1-z[:, None]**2)*np.sin(phi), z[:, None]), axis=-1)
    aa = np.array([.6, 0., .8]); bb = np.array([0., .8, .6])
    l1, l2 = eval_legendre(2, khat@aa), eval_legendre(2, khat@bb)
    integrand = -7/18-5/18*(l1+l2)+35/18*l1*l2
    c0, c2, c4 = coefficients(aa[2], bb[2], aa@bb)
    for x in (.1, 1.3, 3.7, 15.):
        direct = np.sum(wz[:, None]/128*integrand*np.cos(x*z[:, None]))
        analytic = c0*spherical_jn(0, x)-c2*spherical_jn(2, x)+c4*spherical_jn(4, x)
        error = max(error, abs(direct-analytic))
    if error > 1e-12:
        raise RuntimeError('Local tensor plane-parallel limit failed')
    return float(error)


def difference(a, b, rgrid, hankels):
    """分块计算 Xi_local-Xi_midpoint；两个输入目录彼此独立，避免积分自对。"""
    result = np.empty((len(a), len(b)))
    for start in range(0, len(a), 128):
        s, u, v, m, c = geometry(a[start:start+128], b)
        h2, h40, h42, h44 = [np.interp(s, rgrid, h) for h in hankels]
        c0, c2, c4 = coefficients(u, v, c)
        result[start:start+128] = (h2*(.5*(eval_legendre(2, u)+eval_legendre(2, v))-eval_legendre(2, m))
                                  +h40*c0+h42*c2+h44*(c4-eval_legendre(4, m)))
    return result


def group_reduce(matrix, labels, weights, nlabel):
    """对末轴目录在径向组内加权平均；不建立庞大的 dense Pi。"""
    order = np.argsort(labels)
    boundaries = np.r_[0, np.flatnonzero(np.diff(labels[order]))+1]
    norm = np.bincount(labels, weights=weights, minlength=nlabel)
    if len(boundaries) != nlabel or np.any(norm == 0):
        raise RuntimeError('Missing radial group in LOS audit')
    return np.add.reduceat(matrix[:, order]*weights[order][None], boundaries, axis=1)/norm[None]


def project_difference(catalogs, rgrid, hankels):
    """独立 A/B 积分 cross+auto 的 local-minus-midpoint 差。"""
    D, A, B = catalogs
    nlabel = len(np.unique(D[2]))
    da = group_reduce(difference(D[0], A[0], rgrid, hankels), A[2], A[1], nlabel)
    db = group_reduce(difference(D[0], B[0], rgrid, hankels), B[2], B[1], nlabel)
    cross = .5*(da[:, D[2]]+db[:, D[2]])
    cross = cross+cross.T
    ab = difference(A[0], B[0], rgrid, hankels)
    grouped = group_reduce(ab, B[2], B[1], nlabel)
    grouped = group_reduce(grouped.T, A[2], A[1], nlabel).T
    grouped = .5*(grouped+grouped.T)
    return grouped[D[2][:, None], D[2][None, :]]-cross


def fourier(catalog, correction, pedges, volume):
    """对 outer 独立目录直接做 endpoint P estimator，k band 用精确解析积分。"""
    p, w, _ = catalog
    result = np.zeros((2, len(pedges)))
    norm = w.sum()**2
    for start in range(0, len(p), 128):
        s, u, v, _, _ = geometry(p[start:start+128], p)
        pair = w[start:start+128, None]*w[None]*correction[start:start+128]/norm
        pair[s < 1e-8] = 0.
        safe = np.maximum(s, 1e-20)
        for il, ell in enumerate((0, 2)):
            angular = .5*(eval_legendre(ell, u)+eval_legendre(ell, v))
            for j, (lo, hi) in enumerate(pedges):
                bessel = (bessel_volume_primitive(ell, safe*hi)-bessel_volume_primitive(ell, safe*lo))/(safe**3*(hi**3-lo**3)/3)
                result[il, j] += volume*(2*ell+1)*(-1)**(ell//2)*np.sum(pair*angular*bessel)
    return result.reshape(-1)[np.r_[np.arange(13), np.arange(17, 26)]]


def run(args):
    """固定 fiducial P 物理输入，量化 LOS 近似误差而不改变任何正式输入。"""
    limit_error = coefficient_smoke()
    catalogs, occupied, volume, _ = samples(args.nsub, args.seed, args.radial_width, 'ph000')
    with np.load(BASE/'frozen_inputs.npz') as a:
        pedges, covariance = a['p_edges'], a['covariance']
    with np.load(PK_DIR/'task43_rsd_lightcone_p02_ph000_mesh256_kmax0p300_dk0p002.npz') as a:
        tk, te, edges = a['theory_k'], a['theory_ell'], a['theory_edges']
    with np.load(PAYLOAD_NPZ) as a:
        zeff = float(a['zeff'])
    model = WindowConvolvedP02Model(np.zeros((1, len(tk))), tk, te, zeff=zeff, sigma_step=30.)
    metric = Metric(covariance)
    rgrid = np.unique(np.r_[np.geomspace(.001, 5., 80), np.arange(5., 3400.1, args.rstep)])
    transforms = [pk_band_to_xi(rgrid, edges[te == pell], hell) for pell, hell in ((2, 2), (4, 0), (4, 2), (4, 4))]
    records = []
    points = ((0., 2.4, 7.5),) if args.fiducial_only else ((0., 2.4, 7.5), (30., 2.45, 3.), (-30., 2.35, 15.))
    for f, b, sig in points:
        q = f*2*1.686*(b-1); fg = model.f_growth
        coeff = np.array([b*b, 2*b*q, q*q, 2*b*fg, 2*q*fg, fg*fg])
        theory = model._theory_basis(sig)@coeff
        hankels = [matrix@theory[te == pell] for matrix, pell in zip(transforms, (2, 4, 4, 4))]
        correction = project_difference(catalogs, rgrid, hankels)
        delta = fourier(catalogs[0], correction, pedges, volume)
        record = {'theta': [f, b, sig], 'delta_p': delta.tolist(),
                  'difference_chi2_Csingle': float(metric.chi2(np.r_[delta, np.zeros(52)]))}
        records.append(record)
        print(json.dumps(record), flush=True)
    report = {'status': 'LOS_approximation_audit', 'nsub': args.nsub, 'seed': args.seed,
              'radial_width': args.radial_width, 'rstep': args.rstep, 'parallel_limit_error': limit_error,
              'records': records, 'scope': 'Exact frozen jaxpower two-endpoint coeff2 theory minus midpoint, propagated through full cross+auto by independent dense-group reference; integration-sample convergence must be checked.'}
    save_json(OUT/f'smoke/los_local_vs_midpoint_n{args.nsub}_seed{args.seed}.json', report)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--nsub', type=int, default=2048)
    parser.add_argument('--seed', type=int, default=4322233)
    parser.add_argument('--radial-width', type=float, default=4.)
    parser.add_argument('--rstep', type=float, default=.5)
    parser.add_argument('--fiducial-only', action='store_true')
    run(parser.parse_args())
