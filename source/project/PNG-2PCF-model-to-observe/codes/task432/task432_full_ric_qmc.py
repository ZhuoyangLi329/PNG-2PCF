#!/usr/bin/env python3
"""在生产 random 的同一分布上用独立 scrambled Sobol 点降低积分噪声。

执行大纲：读取生产的 observed-z 源目录与分段 FKP -> 以 empirical CDF
生成三套独立的正八分体积分目录 -> 原样调用 eBOSS cross/auto 后端。
只改变几何积分取样；正式 data、random、covariance 和动力学模型均不写入。
必须比较独立 scramble 和现有实际 random 子集结果才可用于正式拟合。
"""
import os
for key in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ[key] = '1'
import argparse
import json
import time
import numpy as np
from scipy.stats import qmc
from task432_full_ric_backend import ROOT, OUT, counts
from task432_full_ric_geometry import prepare_pool, separation_edges, save_json, sha


def samples(n, seed, width, phase):
    """与生产生成器同分布：角度均匀、observed-z 等概率有放回、分段 FKP。"""
    if n & (n-1):
        raise ValueError('Sobol sample count must be a power of two')
    pool = prepare_pool(phase)
    meta = json.loads(str(pool['meta_json'].item()))
    path = meta['row']['lightcone_catalog_path']
    with np.load(path, allow_pickle=False) as a:
        z = np.asarray(a['Z'], dtype='f8')
    with np.load(meta['fkp_summary'], allow_pickle=False) as a:
        z_edges, fkp = a['z_edges'], a['fkp_weights']
    z[z <= z_edges[0]] = float(np.nextafter(np.float32(z_edges[0]), np.float32(np.inf)))
    z[z >= z_edges[-1]] = float(np.nextafter(np.float32(z_edges[-1]), np.float32(-np.inf)))
    z.sort()
    from cosmoprimo.fiducial import AbacusSummit
    cosmo = AbacusSummit(0)
    zgrid = np.linspace(z_edges[0], z_edges[-1], 200001)
    chigrid = cosmo.comoving_radial_distance(zgrid)
    result = []
    for i in range(3):
        unit = qmc.Sobol(3, scramble=True, seed=seed+i*104729).random_base2(int(np.log2(n)))
        redshift = z[np.minimum((unit[:, 0]*len(z)).astype(int), len(z)-1)]
        radius = np.interp(redshift, zgrid, chigrid)
        phi, mu = unit[:, 1]*np.pi/2, unit[:, 2]
        transverse = np.sqrt(1-mu*mu)
        xyz = radius[:, None]*np.column_stack([transverse*np.cos(phi), transverse*np.sin(phi), mu])
        bins = np.clip(np.searchsorted(z_edges, redshift, side='right')-1, 0, len(fkp)-1)
        weight = fkp[bins]
        labels = np.floor(radius/width).astype(int) if width > 0 else np.zeros(n, dtype=int)
        result.append((xyz, weight, labels))
    occupied = np.unique(result[0][2])
    for i, (p, w, labels) in enumerate(result):
        if not np.array_equal(np.unique(labels), occupied):
            raise RuntimeError('Incomplete QMC radial support')
        result[i] = p, w, np.searchsorted(occupied, labels)
    return result, occupied, float(pool['pair_volume']), meta


def build(args):
    """运行一套可恢复、由参数唯一命名的 QMC 核，强制常数场闭合。"""
    stem = f'{args.phase}_qmc{args.seed}_n{args.nsub}_dchi{args.radial_width:g}_refine{args.refinement}_{args.los}_in{args.input_los}_ell{args.maxell}'
    target = OUT/f'geometry/{stem}.npz'
    if target.exists():
        print(json.dumps({'event': 'kernel_exists', 'path': str(target)}), flush=True)
        return
    catalogs, occupied, volume, source = samples(args.nsub, args.seed, args.radial_width, args.phase)
    edges = separation_edges(args.refinement)
    result = counts(*catalogs[0], edges, ells_out=tuple(range(0, args.maxell+1, 2)),
                    los_out=args.los, los_in=args.input_los, nthreads=8, progress=True,
                    independent_projection=catalogs[1:])
    scale = np.max(abs(result['rr']))
    error = max(np.max(abs(result['cross'][:, 0].sum(axis=-1)-2*result['rr'])),
                np.max(abs(result['auto'][:, 0].sum(axis=-1)-result['rr'])))/scale
    if error > 1e-10:
        raise RuntimeError(f'QMC constant-field closure failed: {error}')
    meta = {k: v for k, v in result.items() if k not in ('rr', 'cross', 'auto')}
    meta.update({'status': 'pilot_unvalidated', 'sampling': 'Three independently scrambled Sobol catalogs from production selection distribution',
                 'seed': args.seed, 'nsub': args.nsub, 'radial_width': args.radial_width,
                 'refinement': args.refinement, 'phase': args.phase, 'pair_volume': volume,
                 'source_sha256': sha(__file__), 'source_catalog': source['row']['lightcone_catalog_path'],
                 'fkp_sha256': source['fkp_sha256'], 'constant_field_relative_error': float(error)})
    temporary = target.with_suffix('.tmp.npz')
    np.savez_compressed(temporary, separation_edges=edges, radial_labels=occupied,
                        rr=result['rr'], cross=result['cross'], auto=result['auto'],
                        normalization=result['normalization'], meta_json=np.array(json.dumps(meta)))
    temporary.replace(target)
    save_json(target.with_suffix('.json'), meta)
    print(json.dumps({'event': 'qmc_kernel_done', 'path': str(target), 'elapsed_s': result['elapsed_s']}), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--phase', default='ph000')
    parser.add_argument('--seed', type=int, default=4322217)
    parser.add_argument('--nsub', type=int, default=32768)
    parser.add_argument('--radial-width', type=float, default=2.)
    parser.add_argument('--refinement', type=int, default=1)
    parser.add_argument('--los', choices=('midpoint', 'endpoint'), default='midpoint')
    parser.add_argument('--input-los', choices=('midpoint', 'endpoint'), default='midpoint')
    parser.add_argument('--maxell', type=int, default=8)
    build(parser.parse_args())
