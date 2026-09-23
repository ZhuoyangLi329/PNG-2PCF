#!/usr/bin/env python3
"""缓存 hybrid 在完整 RIC 内层距离范围的多极，不依赖某一套待验证几何核。

执行大纲：复用原 fNL/b1 网格 -> 6 个单线程 worker 计算 shell-averaged
xi0/2/4 -> 每行独立原子检查点 -> 汇总物理网格供各核共用。
此物理缓存可在几何收敛期间构建；正式拟合仍需要独立响应/插值验证通过。
"""
import os
for key in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ[key] = '1'
import argparse
import json
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
import multiprocessing as mp
import numpy as np
from task432_full_ric_backend import ROOT, OUT
from task432_full_ric_geometry import BASE, separation_edges, save_json, sha
for folder in ('codes/task432', 'codes/task43', 'codes/task44'):
    sys.path.insert(0, str(ROOT/folder))
from task432_hybrid_gic import HybridGIC


def initialize(edges, nquad, nint):
    """每个单线程进程只初始化一次 GSM，并冻结相同积分设置。"""
    global MODEL, RADII, WEIGHTS, NINT
    MODEL = HybridGIC()
    q, w = np.polynomial.legendre.leggauss(nquad)
    lo, hi = edges[:-1, None], edges[1:, None]
    RADII = (hi-lo)*q/2+(hi+lo)/2
    WEIGHTS = (hi-lo)/2*w*RADII**2/((hi**3-lo**3)/3)
    NINT = nint


def row(task):
    """返回某个 fNL 对应的完整 b1 多极网格，接入前不作任何 GIC/RIC。"""
    i, f, bb = task
    spectra = []
    for b in bb:
        MODEL._set_cumulants(f, float(b))
        value = MODEL.poles(RADII.ravel(), nint=NINT).reshape(3, *RADII.shape)
        value = np.einsum('ldq,dq->ld', value, WEIGHTS)
        if not np.all(np.isfinite(value)):
            raise RuntimeError(f'Nonfinite hybrid poles at {(f, b)}')
        spectra.append(value)
    return i, np.asarray(spectra)


def build(args):
    """按参数化路径恢复未完成的物理网格；不覆写已完整检查点。"""
    name = f'hybrid_inner_refine{args.refinement}_q{args.nquad}_n{args.nint}'
    target = OUT/f'geometry/{name}.npz'
    if target.exists():
        print(json.dumps({'event': 'grid_exists', 'path': str(target)}), flush=True)
        return
    checkpoints = OUT/f'logs/{name}'
    checkpoints.mkdir(parents=True, exist_ok=True)
    with np.load(BASE/'xi_emulator.npz', allow_pickle=False) as a:
        ff, bb = a['f_grid'], a['b_grid']
    edges = separation_edges(args.refinement)
    values = np.full((len(ff), len(bb), 3, len(edges)-1), np.nan)
    for i, f in enumerate(ff):
        path = checkpoints/f'row_{i:03d}.npz'
        if path.exists():
            with np.load(path, allow_pickle=False) as a:
                if float(a['f']) != f or not np.array_equal(a['b_grid'], bb):
                    raise RuntimeError('Hybrid grid checkpoint parameter mismatch')
                values[i] = a['poles']
    tasks = [(i, float(f), bb) for i, f in enumerate(ff) if not np.all(np.isfinite(values[i]))]
    started = time.monotonic()
    print(json.dumps({'event': 'grid_start', 'shape': values.shape, 'remaining_rows': len(tasks)}), flush=True)
    with ProcessPoolExecutor(max_workers=6, mp_context=mp.get_context('spawn'),
                             initializer=initialize, initargs=(edges, args.nquad, args.nint)) as pool:
        pending = [pool.submit(row, task) for task in tasks]
        for future in as_completed(pending):
            i, value = future.result()
            values[i] = value
            path = checkpoints/f'row_{i:03d}.npz'
            temporary = path.with_suffix('.tmp.npz')
            np.savez_compressed(temporary, f=ff[i], b_grid=bb, poles=value)
            temporary.replace(path)
            print(json.dumps({'event': 'grid_row', 'index': int(i), 'fnl': float(ff[i]),
                              'completed_rows': int(np.all(np.isfinite(values), axis=(1, 2, 3)).sum()),
                              'elapsed_s': time.monotonic()-started}), flush=True)
    meta = {'status': 'physical_cache_pending_response_validation', 'source_sha256': sha(__file__),
            'hybrid_source_sha256': sha(ROOT/'codes/task432/task432_hybrid_gic.py'),
            'nquad': args.nquad, 'nint': args.nint, 'refinement': args.refinement,
            'model': 'Unchanged hybrid; full ell0/2/4 shell averages without GIC/RIC',
            'elapsed_s': time.monotonic()-started, 'cpu_affinity': sorted(os.sched_getaffinity(0))}
    temporary = target.with_suffix('.tmp.npz')
    np.savez_compressed(temporary, f_grid=ff, b_grid=bb, separation_edges=edges,
                        poles=values, meta_json=np.array(json.dumps(meta)))
    temporary.replace(target)
    save_json(target.with_suffix('.json'), meta)
    print(json.dumps({'event': 'grid_done', 'path': str(target)}), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--refinement', type=int, default=1)
    parser.add_argument('--nquad', type=int, default=2)
    parser.add_argument('--nint', type=int, default=600)
    build(parser.parse_args())
