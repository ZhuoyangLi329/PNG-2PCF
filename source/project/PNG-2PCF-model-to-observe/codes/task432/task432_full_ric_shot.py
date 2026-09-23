#!/usr/bin/env python3
"""按 eBOSS 作者的两点计数构造完整 RIC Poisson shot 模板。

执行大纲：冻结 random 的独立 A/B 子集 -> 径向组内 cross/auto 两点计数
-> 各自以作者的相关 pair 总权重归一化 -> 零模及显式矩阵校验 -> 保存模板。
输入为目录几何、生产权重和径向宽度，不拟合新的 RIC 振幅；使用 measured
Poisson amplitude，原 sn0 保留论文中的独立常数 stochastic nuisance 口径。
有限随机目录与真实 LS block 的影响必须另行检验。
"""
import os
for key in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ[key] = '1'
import argparse
import json
import numpy as np
from task432_full_ric_backend import OUT, load_backend, catalog
from task432_full_ric_geometry import prepare_pool, separation_edges, save_json, sha
from task432_full_ric_reference import pair_geometry
from scipy.special import eval_legendre


def shot_counts(A, B, edges, ells, los='midpoint', nthreads=8, noise_density=None):
    """返回各自按总 pair 权重归一化的 cross(已乘二)、auto 和版本信息。"""
    C2, _, _, provenance = load_backend()
    pa, wa, la = A; pb, wb, lb = B
    fa, fb = (np.ones_like(wa), np.ones_like(wb)) if noise_density is None else noise_density
    if not np.array_equal(np.unique(la), np.unique(lb)):
        raise ValueError('Shot integration requires matched radial support')
    cross = np.zeros((len(ells), len(edges)-1)); auto = cross.copy()
    cross_norm = 0.; auto_norm = 0.
    for label in np.unique(la):
        a, b = la == label, lb == label
        sa, sb = wa[a].sum(), wb[b].sum()
        ta, tb = (wa[a]**2*fa[a]).sum(), (wb[b]**2*fb[b]).sum()
        ga, gb = catalog(pa[a], wa[a], la[a]), catalog(pb[b], wb[b], lb[b])
        for ca, cb in ((catalog(pa[a], wa[a]**2*fa[a]/sa, la[a]), gb),
                       (ga, catalog(pb[b], wb[b]**2*fb[b]/sb, lb[b]))):
            obj = C2(sedges=edges, sbinning='custom', ells=list(ells), los=los,
                     nthreads=nthreads, verbose='quiet')
            obj.set_grid(); obj.run(ca, cb)
            cross += obj.counts
            cross_norm += obj.weight_tot
        obj = C2(sedges=edges, sbinning='custom', ells=list(ells), los=los,
                 nthreads=nthreads, verbose='quiet')
        obj.set_grid(); obj.run(ga, gb)
        factor = .5*(ta/sa**2+tb/sb**2)
        auto += obj.counts*factor
        auto_norm += obj.weight_tot*factor
    return {'cross': cross*2/cross_norm, 'auto': auto/auto_norm,
            'cross_norm': cross_norm/2, 'auto_norm': auto_norm, 'backend': provenance}


def smoke():
    """用不同坐标目录逐对显式求和，验证系数、LOS 和归一化。"""
    rng = np.random.default_rng(4322211)
    labels = np.repeat(np.arange(3), 12)
    catalogs = []
    for _ in range(2):
        unit = abs(rng.normal(size=(len(labels), 3)))
        pos = unit/np.linalg.norm(unit, axis=1)[:, None]*(900.+labels*250.)[:, None]
        catalogs.append((pos, rng.uniform(.5, 1.5, len(labels)), labels))
    A, B = catalogs; n = len(labels)
    geometry = pair_geometry(np.concatenate([A[0], B[0]]))
    edges = np.linspace(.001, 3000., 21)
    index = np.searchsorted(edges, geometry['s'][:n, n:], side='right')-1
    same = labels[:, None] == labels[None, :]
    cross = np.zeros((n, n)); auto = cross.copy()
    for label in np.unique(labels):
        a = labels == label; m = a[:, None] & a[None, :]
        wa, wb = A[1], B[1]
        sa, sb = wa[a].sum(), wb[a].sum()
        ta, tb = sum(wa[a]**2), sum(wb[a]**2)
        cross[m] = (wa[:, None]*wb[None, :]*(wa[:, None]/sa+wb[None, :]/sb))[m]
        auto[m] = (wa[:, None]*wb[None, :]*.5*(ta/sa**2+tb/sb**2))[m]
    cross *= 2/cross.sum(); auto /= auto.sum()
    reports = []
    for los in ('midpoint', 'endpoint'):
        actual = shot_counts(A, B, edges, (0, 2, 4), los, 2)
        errors = {}
        for name, weights in (('cross', cross), ('auto', auto)):
            legs = [eval_legendre(ell, geometry['mu_mid'][:n, n:]) if los == 'midpoint' else
                    .5*(eval_legendre(ell, geometry['mu_first'][:n, n:])+
                        eval_legendre(ell, geometry['mu_second'][:n, n:])) for ell in (0, 2, 4)]
            direct = np.asarray([np.bincount(index[same], weights=(weights*leg)[same],
                                             minlength=len(edges)-1) for leg in legs])
            errors[name] = float(np.max(abs(actual[name]-direct)))
        reports.append({'los': los, 'absolute_errors': errors})
    passed = max(max(row['absolute_errors'].values()) for row in reports) < 1e-12
    save_json(OUT/'smoke/shot_backend_bridge.json', {'status': 'pass' if passed else 'failed', 'reports': reports})
    if not passed:
        raise RuntimeError('Shot dense-sum bridge failed')
    print('Shot direct pair bridge: pass', flush=True)


def build(nsub, width, refinement, los, maxell, white=False):
    """冻结独立 A/B 的 shot 核；其 mono 积分必须恰好抵消未投影的单位 shot。"""
    pool = prepare_pool()
    if 2*nsub > len(pool['weight']):
        raise ValueError('Requested shot sample exceeds pool')
    samples = []
    for i in (0, 1):
        p, w = pool['position'][i*nsub:(i+1)*nsub], pool['weight'][i*nsub:(i+1)*nsub]
        label = np.floor(np.linalg.norm(p, axis=1)/width).astype(int) if width > 0 else np.zeros(nsub, dtype=int)
        samples.append((p, w, label))
    edges = separation_edges(refinement)
    density = None
    if white:
        pool_meta = json.loads(str(pool['meta_json'].item()))
        with np.load(pool_meta['fkp_summary']) as a:
            zedges, nbar = a['z_edges'], a['nbar']
        z = pool['redshift'][:2*nsub]
        nb = nbar[np.clip(np.searchsorted(zedges, z, side='right')-1, 0, len(nbar)-1)]
        density = (nb[:nsub], nb[nsub:])
    result = shot_counts(*samples, edges, tuple(range(0, maxell+1, 2)), los, noise_density=density)
    closure = float(abs(1+np.sum(result['auto'][0]-result['cross'][0])))
    if closure > 1e-10:
        raise RuntimeError(f'Shot k=0 closure failed: {closure}')
    meta = {k: v for k, v in result.items() if k not in ('cross', 'auto')}
    meta.update({'nsub': nsub, 'radial_width': width, 'refinement': refinement, 'los_out': los,
                 'ells_out': list(range(0, maxell+1, 2)), 'zero_mode_error': closure,
                 'source_sha256': sha(__file__), 'noise_density': 'nbar: constant P stochastic field' if white else '1: Poisson sampling',
                 'scope': 'Normalized eBOSS IC shot shape; amplitude and finite-random validation applied separately.'})
    token = 'shot_white' if white else 'shot'
    stem = f'ph000_{token}_n{nsub}_dchi{width:g}_refine{refinement}_{los}_ell{maxell}'
    target = OUT/f'geometry/{stem}.npz'
    np.savez_compressed(target, separation_edges=edges, cross=result['cross'], auto=result['auto'],
                        meta_json=np.array(json.dumps(meta)))
    save_json(target.with_suffix('.json'), meta)
    print(json.dumps({'event': 'shot_done', 'path': str(target), 'zero_mode_error': closure}), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--smoke', action='store_true')
    parser.add_argument('--nsub', type=int, default=32768)
    parser.add_argument('--radial-width', type=float, default=2.)
    parser.add_argument('--refinement', type=int, default=1)
    parser.add_argument('--los', choices=('midpoint', 'endpoint'), default='midpoint')
    parser.add_argument('--maxell', type=int, default=8)
    parser.add_argument('--white', action='store_true')
    args = parser.parse_args()
    if args.smoke:
        smoke()
    else:
        build(args.nsub, args.radial_width, args.refinement, args.los, args.maxell, args.white)
