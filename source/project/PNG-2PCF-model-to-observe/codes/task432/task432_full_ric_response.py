#!/usr/bin/env python3
"""将 eBOSS 原始 cross/auto 核编译成 Fourier 与 LS 多极响应。

执行大纲：读取已计数核 -> 保持内层完整距离范围 -> P 使用球 Bessel 响应
-> xi 在 mu bin 先除 RR 再 Legendre 投影 -> 用当前 hybrid 给出数值 pilot。
本阶段只比较修正和核收敛，不拟合、不改变 frozen data/covariance。
"""
from __future__ import annotations
import os
for _key in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):
    os.environ[_key] = '1'
import argparse
import json
from pathlib import Path
import sys
import numpy as np
from scipy.special import spherical_jn, sici, factorial2
from task432_full_ric_backend import ROOT, OUT
from task432_full_ric_geometry import BASE, save_json, sha


def angular_bin_integrals(ells, edges):
    """返回各 Legendre 多项式在每个 mu bin 的精确积分，轴序 ell,mu。"""
    rows = []
    for ell in ells:
        coeff = np.zeros(ell+1); coeff[ell] = 1.
        primitive = np.polynomial.legendre.legint(coeff)
        rows.append(np.diff(np.polynomial.legendre.legval(edges, primitive)))
    return np.asarray(rows)


def bessel_volume_primitive(ell, x):
    """返回 integral_0^x t^2 j_ell(t)dt；大 x 用解析式，小 x 用级数防消减误差。"""
    x = np.asarray(x, dtype='f8')
    safe = np.maximum(abs(x), 1e-30)
    if ell == 0:
        value = np.sin(x)-x*np.cos(x)
    elif ell == 2:
        value = 3*sici(x)[0]+x*np.cos(x)-4*np.sin(x)
    elif ell == 4:
        value = 52.5*(np.cos(x)/safe-np.sin(x)/safe**2)+7.5*sici(x)[0]-x*np.cos(x)+11*np.sin(x)
    else:
        raise ValueError('Only the frozen theory multipoles 0,2,4 are supported')
    small = abs(x) < 1.
    if np.any(small):
        z = x[small]
        term = z**(ell+3)/float(factorial2(2*ell+1))/(ell+3)
        series = term.copy()
        for m in range(1, 8):
            term *= -z*z/(2*m*(2*ell+2*m+1))*(ell+2*m+1)/(ell+2*m+3)
            series += term
        value[small] = series
    return value


def pk_band_to_xi(radii, k_edges, ell):
    """分片常数 P(k) -> xi_ell(r) 的精确 band 积分，避免高 kr 振荡下固定节点欠采样。"""
    r = np.asarray(radii)[:, None]
    edges = np.asarray(k_edges)
    lower = bessel_volume_primitive(ell, r*edges[None, :, 0])
    upper = bessel_volume_primitive(ell, r*edges[None, :, 1])
    return (-1)**(ell//2)*(upper-lower)/(2*np.pi**2*r**3)


class Response:
    def __init__(self, path):
        """加载已冻结原始核；内部距离支持不受外层 fit mask 截断。"""
        self.path = Path(path)
        with np.load(path, allow_pickle=False) as a:
            self.edges = a['separation_edges']
            self.meta = json.loads(str(a['meta_json'].item()))
            self.norm = float(a['normalization'])
            self.rr = a['rr']/self.norm
            self.kernel = (a['auto']-a['cross'])/self.norm
        self.ells = np.asarray(self.meta['ells_out'])
        self.ells_in = np.asarray(self.meta['ells_in'])
        lo, hi = self.edges[:-1], self.edges[1:]
        self.center = .75*(hi**4-lo**4)/(hi**3-lo**3)

    def xi_project(self, moments, centers, nmu=40, rr_smu=None):
        """将任意 (ell_out,s,...) 原始概率矩在 mu 域除 RR 后投影到 xi0/2。"""
        if self.meta['los_out'] != 'midpoint':
            raise ValueError('LS xi requires midpoint geometry')
        mu_edges = np.linspace(-1., 1., nmu+1)
        integral = angular_bin_integrals(self.ells, mu_edges)
        # 原始 counts 保存 sum L_l，重构 mu-bin 概率的系数为 (2l+1)/2。
        reconstruction = (.5*(2*self.ells+1))[:, None]*integral
        projection = (.5*np.array([1., 5.]))[:, None]*angular_bin_integrals([0, 2], mu_edges)
        moments = np.asarray(moments)
        tail = moments.shape[2:]
        flattened = moments.reshape(len(self.ells), len(self.center), -1)
        result = np.zeros((2, len(centers), flattened.shape[-1]))
        minima = []
        for index, center in enumerate(centers):
            lo, hi = center-5., center+5.
            mask = (self.edges[:-1] >= lo-1e-8) & (self.edges[1:] <= hi+1e-8)
            if not np.any(mask) or not np.isclose(np.sum(np.diff(self.edges)[mask]), hi-lo):
                raise ValueError('Kernel edges must cover each target shell exactly')
            rrmu = np.sum(self.rr[:, mask], axis=-1)@reconstruction if rr_smu is None else np.asarray(rr_smu)[index]
            minima.append(float(rrmu.min()))
            if np.any(rrmu <= 0):
                raise RuntimeError(f'Nonpositive truncated-multipole RR reconstruction at s={center}')
            numerator = np.einsum('lq,lm->mq', flattened[:, mask].sum(axis=1), reconstruction)
            result[:, index] = np.einsum('om,mq->oq', projection/rrmu[None, :], numerator)
        self.rr_min = min(minima)
        return result.reshape((2*len(centers),)+tail)

    def xi_matrix(self, centers, nmu=40, rr_smu=None):
        """返回 (52,ell_in,d) LS 修正矩阵，逐目标 s bin 聚合后在 mu 域除 RR。"""
        return self.xi_project(self.kernel.transpose(0, 2, 1, 3), centers, nmu, rr_smu)

    def p_matrix(self, k_edges, nquad=8):
        """返回 (2,nk,ell_in,d) Fourier 修正；k 和 s 均作体积加权壳内积分。"""
        return self.p_project(self.kernel.transpose(0, 2, 1, 3), k_edges, nquad)

    def p_project(self, moments, k_edges, nquad=8):
        """将 (ell_out,s,...) 概率矩投影到当前 P 多极，包含正式 I2 归一化。"""
        if self.meta['los_out'] != 'endpoint':
            raise ValueError('Current P output uses endpoint/local geometry')
        k_edges = np.asarray(k_edges)
        q, w = np.polynomial.legendre.leggauss(nquad)
        klo, khi = k_edges[:, 0, None], k_edges[:, 1, None]
        knodes = (khi-klo)*q/2+(khi+klo)/2
        kw = w[None, :]*(khi-klo)/2*knodes**2/((khi**3-klo**3)/3)
        slo, shi = self.edges[:-1, None], self.edges[1:, None]
        snodes = (shi-slo)*q/2+(shi+slo)/2
        sw = w[None, :]*(shi-slo)/2*snodes**2/((shi**3-slo**3)/3)
        result = []
        for ell in (0, 2):
            kernel = np.asarray(moments)[np.flatnonzero(self.ells == ell)[0]]
            j = spherical_jn(ell, knodes[:, :, None, None]*snodes[None, None, :, :])
            transform = np.einsum('kasu,ka,su->ks', j, kw, sw)
            result.append(self.meta['pair_volume']*(2*ell+1)*(-1)**(ell//2)
                          * np.einsum('ks,s...->k...', transform, kernel))
        return np.asarray(result)

    def pk_theory_matrix(self, observed_edges, theory_edges, theory_ells, nquad=8, inner_nquad=8):
        """将当前 P 理论 band 向量直接映射为两个观测多极的完整 clustering RIC。"""
        response = self.p_matrix(observed_edges, nquad=nquad)
        result = np.zeros(response.shape[:2]+(len(theory_ells),))
        q, w = np.polynomial.legendre.leggauss(inner_nquad)
        lo, hi = self.edges[:-1, None], self.edges[1:, None]
        radii = (hi-lo)*q/2+(hi+lo)/2
        weights = (hi-lo)/2*w*radii**2/((hi**3-lo**3)/3)
        for index, ell in enumerate(self.ells_in):
            selected = np.asarray(theory_ells) == ell
            hankel = pk_band_to_xi(radii.ravel(), np.asarray(theory_edges)[selected], ell)
            hankel = np.einsum('dq,dqj->dj', weights, hankel.reshape(*radii.shape, -1))
            result[:, :, selected] = np.einsum('akd,dj->akj', response[:, :, index], hankel)
        return result


def hybrid_pilot(paths, output):
    """同一 fiducial hybrid 输入比较 xi 核；使用正式 covariance 度量但不拟合。"""
    for folder in ('codes/task432', 'codes/task43', 'codes/task44'):
        sys.path.insert(0, str(ROOT/folder))
    import task432_hybrid_jaxpower_0918_contract as common
    from task432_hybrid_gic import HybridGIC
    common.OUT = BASE
    with np.load(BASE/'frozen_inputs.npz', allow_pickle=False) as a:
        centers = a['xi_centers']; covariance = a['covariance']
    metric = common.Metric(covariance)
    model = HybridGIC()
    points = [(0., 2.4), (30., 2.45), (-30., 2.35)]
    direct_cache = {}
    rows = []
    for path in paths:
        response = Response(path)
        matrix = response.xi_matrix(centers)
        key = tuple(response.center)
        if key not in direct_cache:
            allpoles = []
            for f, b in points:
                model._set_cumulants(f, b)
                allpoles.append(model.poles(response.center, nint=1200))
            direct_cache[key] = np.asarray(allpoles)
        spectra = direct_cache[key]
        correction = np.einsum('ild,tld->ti', matrix, spectra)
        # 相对于旧 scalar GIC 的变化：先恢复其 C_W，再加入完整径向修正。
        oldgic = np.asarray([model.correction(fnl=f, b1=b, nquad=8, nint=1200) for f, b in points])
        relative = correction.copy(); relative[:, :len(centers)] += oldgic[:, None]
        row = {'path': str(path), 'meta': response.meta, 'theta': points,
               'delta_xi_full_ric': correction.tolist(), 'delta_from_old_gic': relative.tolist(),
               'RR_reconstruction_min': response.rr_min,
               'shift_metric_Csingle': metric.chi2(np.column_stack([np.zeros((len(points), 22)), relative])).tolist(),
               'theory_range': [float(spectra.min()), float(spectra.max())]}
        rows.append(row)
        print(json.dumps({'event': 'response_pilot', 'path': str(path), 'metric': row['shift_metric_Csingle']}), flush=True)
    differences = []
    for first, second in zip(rows[:-1], rows[1:]):
        diff = np.asarray(second['delta_xi_full_ric'])-first['delta_xi_full_ric']
        differences.append({'first': first['path'], 'second': second['path'],
                            'difference_chi2_Csingle': metric.chi2(np.column_stack([np.zeros((len(points), 22)), diff])).tolist()})
    save_json(output, {'status': 'pilot_only', 'rows': rows, 'differences': differences,
                       'scope': 'Clustering RIC only; shot, production RR/block averaging and P input LOS audit pending.',
                       'source_sha256': sha(__file__)})


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('kernels', nargs='+', type=Path)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    hybrid_pilot(args.kernels, args.output)
