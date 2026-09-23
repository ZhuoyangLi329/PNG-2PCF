#!/usr/bin/env python3
"""复用 eBOSS 作者 pycute/pywindow 的 2/3/4 点核，提供项目隔离兼容接口。

执行大纲
--------
1. 仅在当前进程补充旧 SciPy 的 NumPy aliases，不改共享环境或上游文件。
2. 从固定 commit 源码加载所需三个计数模块，跳过无关的旧 nbodykit FFT 入口。
3. 按作者径向分组与权重定义累加 RR、合并 cross、auto 的原始计数。
4. 与独立 dense Q Xi Q^T 参考比较多极响应，避免漏项、乘二与轴序错误。

核计数未包含球壳/多极归一化；后续响应接口显式完成这些步骤。
"""
from __future__ import annotations
import os
for _key in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):
    os.environ[_key] = '1'
import sys
import types
import importlib
import argparse
import hashlib
import json
import time
from pathlib import Path
import numpy as np
from scipy.special import eval_legendre
from task432_full_ric_reference import pair_geometry, projected_terms

ROOT = Path('/pscratch/sd/l/lzy/PNG-2PCF-model-to-observe')
OUT = ROOT/'outputs/task43_outputs/rsd_validation/task432_model_repair/full_ric_p02xi02_v1'
REF = OUT/'reference'
COMMITS = {'pycute': 'e29ea2ee95ba9127d00187e9e341a63a9431c437',
           'pywindow': '9857ebef68153660454fee34c4fda60e90f7289e'}


def load_backend(reference=REF):
    """输入保存固定 commit 源码的目录，返回作者三个计数类与版本审计。"""
    import scipy
    for name in dir(np):
        if not name.startswith('_') and name not in scipy.__dict__:
            setattr(scipy, name, getattr(np, name))
    scipy.random = np.random
    cute_path = reference/f'pycute-{COMMITS["pycute"]}'
    window_path = reference/f'pywindow-{COMMITS["pywindow"]}'
    sys.path.insert(0, str(cute_path))
    # 使用独立 namespace 加载纯计数模块；无需触发上游 __init__ 的 FFT 依赖。
    namespace = '_task432_eboss_window'
    if namespace not in sys.modules:
        module = types.ModuleType(namespace)
        module.__path__ = [str(window_path/'pywindow')]
        sys.modules[namespace] = module
    classes = [getattr(importlib.import_module(f'{namespace}.{m}'), c) for m, c in
               [('pyreal2pcf', 'PyReal2PCF'), ('pyreal3pcf', 'PyReal3PCF'), ('pyreal4pcf', 'PyReal4PCFBinned')]]
    files = [cute_path/'pycute/lib/cute.so', cute_path/'src/correlator.c',
             window_path/'pywindow/pyreal2pcf.py', window_path/'pywindow/pyreal3pcf.py',
             window_path/'pywindow/pyreal4pcf.py']
    provenance = {'commits': COMMITS, 'files': {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in files},
                  'compatibility': 'Process-local NumPy aliases for old scipy names; pure count modules loaded without FFT/nbodykit imports.'}
    return (*classes, provenance)


def catalog(position, weight, labels):
    """将坐标、权重和零起点连续径向标签转换为作者接口需要的字典。"""
    return {'Position': np.asarray(position, dtype='f8'), 'Weight': np.asarray(weight, dtype='f8'),
            'ibin': np.asarray(labels, dtype='i8')}


def counts(position, weight, labels, edges, *, ells_out=(0, 2), ells_in=(0, 2, 4),
           los_out='midpoint', los_in='midpoint', nthreads=8, reference=REF, progress=False,
           independent_projection=None):
    """返回 RR、cross、auto 原始加权计数；cross 已包含两个交叉项。

    输入标签须为连续整数；投影目录在每个径向 bin 内以 weight 归一化。
    输出 cross/auto 轴序是 (ell_out,ell_in,s_outer,s_inner)，RR 为 (ell_out,s)。
    independent_projection 可提供两组 (position,weight,labels)；它们与 outer
    以及彼此独立，避免小规模积分目录的重复点被误当作物理 clustering。
    本接口匹配作者有限径向 bin 定义；实际 shuffle 极限须另做分辨率收敛。
    """
    if nthreads > 8:
        raise ValueError('At most 8 login-node CPUs')
    C2, C3, C4, provenance = load_backend(reference)
    labels = np.asarray(labels, dtype=int)
    unique = np.unique(labels)
    if not np.array_equal(unique, np.arange(len(unique))):
        raise ValueError('Radial labels must be contiguous and occupied')
    nb = len(unique)
    sums = np.bincount(labels, weights=weight, minlength=nb)
    normalized = weight/sums[labels]
    full = catalog(position, weight, labels)
    projected = catalog(position, normalized, labels)
    projections = [projected]
    if independent_projection is not None:
        if len(independent_projection) != 2:
            raise ValueError('Exactly two independent projection catalogues required')
        projections = []
        for pp, ww, ll in independent_projection:
            if not np.array_equal(np.unique(ll), unique):
                raise ValueError('Projection catalogue must cover every radial bin')
            norm = np.bincount(ll, weights=ww, minlength=nb)
            projections.append(catalog(pp, ww/norm[ll], ll))
    eo, ei = list(ells_out), list(ells_in)
    grid = {'sedges': np.asarray(edges), 'sbinning': 'custom', 'verbose': 'quiet'}
    t0 = time.monotonic()
    obj = C2(**grid, ells=eo, los=los_out, nthreads=nthreads)
    obj.set_grid(); obj.run(full, full)
    rr = obj.counts.copy()
    shape = (len(eo), len(ei), len(edges)-1, len(edges)-1)
    cross = np.zeros(shape); auto = np.zeros(shape)
    for label in unique:
        selected = labels == label
        group = catalog(position[selected], weight[selected], labels[selected])
        for index, proj in enumerate(projections):
            selected_p = proj['ibin'] == label
            pg = {k: v[selected_p] for k, v in proj.items()}
            obj = C3(**grid, ells=[eo, ei], los=[los_out, los_in], nthreads=nthreads)
            obj.set_grid(); obj.run(full, group, pg)
            cross += obj.counts/len(projections)
            other = projections[-1-index]
            obj = C4(**grid, ells=[eo, ei], los=[los_out, los_in], binsize=nb, nthreads=nthreads)
            obj.set_grid(); obj.run(group, full, pg, other, tobin=[2, 4])
            auto += obj.counts/len(projections)
        if progress and (label % max(1, nb//10) == 0 or label == unique[-1]):
            print(json.dumps({'event': 'kernel_radial_bin', 'completed': int(label+1), 'total': nb,
                              'elapsed_s': time.monotonic()-t0}), flush=True)
    return {'rr': rr, 'cross': cross, 'auto': auto, 'normalization': float(np.sum(weight)**2),
            'elapsed_s': time.monotonic()-t0, 'backend': provenance,
            'ells_out': eo, 'ells_in': ei, 'los_out': los_out, 'los_in': los_in,
            'independent_projection': independent_projection is not None}


def backend_smoke(output):
    """在小目录上用分片常数多极作为输入，独立逐对求和验证作者核的轴序与系数。"""
    rng = np.random.default_rng(4322202)
    nr, na = 3, 7
    directions = np.abs(rng.normal(size=(nr*na, 3)))
    directions /= np.linalg.norm(directions, axis=1)[:, None]
    labels = np.repeat(np.arange(nr), na)
    position = directions*np.repeat([950., 1250., 1550.], na)[:, None]
    weight = rng.uniform(.7, 1.3, len(labels))
    edges = np.linspace(.001, 2800., 17)
    center = (edges[:-1]+edges[1:])/2
    theory = np.array([np.exp(-(center/500)**2), .2*np.exp(-(center/700)**2), -.03*np.exp(-(center/400)**2)])
    g = pair_geometry(position)
    ids = np.searchsorted(edges, g['s'], side='right')-1
    valid = (ids >= 0) & (ids < len(center))
    xi = np.zeros_like(g['s'])
    for ell, row in zip((0, 2, 4), theory):
        xi[valid] += row[ids[valid]]*eval_legendre(ell, g['mu_mid'][valid])
    target = projected_terms(xi, labels, weight)
    reports = []
    for los in ('midpoint', 'endpoint'):
        result = counts(position, weight, labels, edges, los_out=los, nthreads=2)
        pred = {k: np.einsum('abst,bt->as', result[k], theory) for k in ('cross', 'auto')}
        obsleg = [eval_legendre(ell, g['mu_mid']) if los == 'midpoint' else
                  .5*(eval_legendre(ell, g['mu_first'])+eval_legendre(ell, g['mu_second'])) for ell in (0, 2)]
        wanted = {}
        for name, matrix in [('cross', target['cross_left']+target['cross_right']), ('auto', target['auto'])]:
            wanted[name] = np.array([np.bincount(ids[valid], weights=(np.outer(weight, weight)*leg*matrix)[valid],
                                                   minlength=len(center)) for leg in obsleg])
        errors = {k: float(np.max(abs(pred[k]-wanted[k]))) for k in wanted}
        scale = max(float(np.max(abs(v))) for v in wanted.values())
        reports.append({'sampling': 'shared_reference', 'los_out': los, 'max_absolute_errors': errors, 'scale': scale,
                        'relative_error': max(errors.values())/scale,
                        'elapsed_s': result['elapsed_s'], 'backend': result['backend']})
    # 独立 A/B 投影目录：直接在三套坐标的 Xi 子矩阵上做矩阵乘法。
    references = []
    for _ in range(2):
        angular = np.abs(rng.normal(size=position.shape))
        angular /= np.linalg.norm(angular, axis=1)[:, None]
        references.append((angular*np.linalg.norm(position, axis=1)[:, None], rng.uniform(.7, 1.3, len(labels)), labels))
    allposition = np.concatenate([position]+[r[0] for r in references])
    allg = pair_geometry(allposition)
    allid = np.searchsorted(edges, allg['s'], side='right')-1
    mask = (allid >= 0) & (allid < len(center))
    allxi = np.zeros_like(allg['s'])
    for ell, row in zip((0, 2, 4), theory):
        allxi[mask] += row[allid[mask]]*eval_legendre(ell, allg['mu_mid'][mask])
    projectors = []
    for _, ww, ll in references:
        same = labels[:, None] == ll[None, :]
        projectors.append(same*ww[None, :]/(same@ww)[:, None])
    n = len(position); A, B = projectors
    right = .5*(allxi[:n, n:2*n]@A.T + allxi[:n, 2*n:]@B.T)
    cross_target = right+right.T
    auto_target = A@allxi[n:2*n, 2*n:]@B.T
    auto_target = .5*(auto_target+auto_target.T)
    for los in ('midpoint', 'endpoint'):
        result = counts(position, weight, labels, edges, los_out=los, nthreads=2, independent_projection=references)
        prediction = {k: np.einsum('abst,bt->as', result[k], theory) for k in ('cross', 'auto')}
        wanted = {}
        for name, matrix in [('cross', cross_target), ('auto', auto_target)]:
            legs = [eval_legendre(ell, g['mu_mid']) if los == 'midpoint' else
                    .5*(eval_legendre(ell, g['mu_first'])+eval_legendre(ell, g['mu_second'])) for ell in (0, 2)]
            wanted[name] = np.array([np.bincount(ids[valid], weights=(np.outer(weight, weight)*leg*matrix)[valid],
                                                minlength=len(center)) for leg in legs])
        errors = {k: float(np.max(abs(prediction[k]-wanted[k]))) for k in wanted}
        scale = max(float(np.max(abs(v))) for v in wanted.values())
        reports.append({'sampling': 'independent_A_B_reference', 'los_out': los, 'max_absolute_errors': errors,
                        'scale': scale, 'relative_error': max(errors.values())/scale, 'elapsed_s': result['elapsed_s']})
    report = {'status': 'pass' if max(r['relative_error'] for r in reports) < 1e-10 else 'failed',
              'scope': 'Small catalogue direct-sum cross/auto validation; input LOS midpoint; no survey accuracy claim.',
              'reports': reports}
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2)+'\n')
    print(json.dumps(report), flush=True)
    if report['status'] != 'pass':
        raise RuntimeError('eBOSS backend failed direct-sum bridge')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    backend_smoke(parser.parse_args().output)
