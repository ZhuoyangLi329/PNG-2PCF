#!/usr/bin/env python3
"""冻结通过检查的 full-RIC 模型，恢复式 MCMC，并保留原有限 mock 口径。

执行大纲：核验审计/hash -> 正确映射 xi 的既有 sn0 -> 多起点 MAP ->
64 walkers HDF5 可恢复采样 -> 三项收敛 gate -> Percival 区间修正。
所有产物写在新 formal 目录，不重置或覆盖任何旧链。
"""
import os
for key in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ[key] = '1'
import argparse
import copy
import json
import socket
import time
from pathlib import Path
import numpy as np
from scipy.optimize import least_squares
from task432_full_ric_compile import Engine, common
from task432_full_ric_backend import OUT
from task432_full_ric_geometry import BASE, sha


def fit(variant, compiled, run_name='formal', geometry_audit=None):
    """从原始 HDF 后端续跑；仅在全部门限通过后生成正式 summary。"""
    import emcee
    checks = [geometry_audit or OUT/'smoke/geometry_validation.json', compiled.parent/'model_validation.json',
              OUT/'smoke/paired_validation.json']
    for path in checks:
        check = json.loads(path.read_text())
        if check['status'] != 'pass':
            raise RuntimeError(f'Preflight not passed: {path}')
    if json.loads(checks[1].read_text())['compiled_sha256'] != sha(compiled):
        raise RuntimeError('Model was changed after validation')
    engine = Engine(variant, compiled)
    d = len(engine.names)
    assert d == (3 if variant == 'xi02' else 4)
    if '/' in run_name or run_name in ('.', '..'):
        raise ValueError('run_name must be one directory name')
    dest = OUT/run_name/variant; dest.mkdir(parents=True, exist_ok=True)
    manifest = {'variant': variant, 'compiled': str(compiled), 'compiled_sha256': sha(compiled),
                'baseline_frozen_sha256': sha(BASE/'frozen_inputs.npz'),
                'parameter_names': engine.names, 'seed': 4322300+['p02', 'xi02', 'joint'].index(variant),
                'nwalkers': 64, 'burnin': 5000,
                'source_sha256': sha(__file__), 'engine_source_sha256': sha(Path(__file__).with_name('task432_full_ric_compile.py')),
                'audit_sha256': {str(p): sha(p) for p in checks}}
    if (dest/'input_manifest.json').exists():
        if json.loads((dest/'input_manifest.json').read_text()) != manifest:
            raise RuntimeError('Refusing to resume a chain with different inputs/code')
    else:
        common.save(dest/'input_manifest.json', manifest)
    if (dest/'summary.json').exists() and json.loads((dest/'summary.json').read_text())['status'] == 'pass':
        common.log('fit_already_complete', variant=variant)
        return
    starts = [[-12., 2.42, 2.2, .14], [0., 2.4, 1., .1], [30., 2.3, 5., 0.], [-50., 2.55, 8., -.1]]
    indices = [0, 1, 3] if variant == 'xi02' else [0, 1, 2, 3]
    solutions = [least_squares(engine.residual, np.asarray(t)[indices], bounds=(engine.lower, engine.upper),
                              x_scale='jac', max_nfev=1500, ftol=1e-11, xtol=1e-11, gtol=1e-11) for t in starts]
    best = min(solutions, key=lambda s: sum(s.fun**2))
    common.log('formal_map', variant=variant, theta=best.x, raw_chi2=engine.raw_metric.chi2(engine.data-engine.predict(best.x)[0]))
    seed = manifest['seed']; rng = np.random.default_rng(seed)
    backend = emcee.backends.HDFBackend(str(dest/'chain.h5'))
    if backend.initialized:
        assert backend.shape == (64, d)
        state = None; done = backend.iteration
    else:
        backend.reset(64, d)
        scale = np.asarray([4., .015, .12, .025])[indices]
        state = np.clip(best.x+rng.normal(size=(64, d))*scale, engine.lower+1e-7, engine.upper-1e-7)
        done = 0
    np.random.seed(seed)
    sampler = emcee.EnsembleSampler(64, d, engine.logp, vectorize=True, backend=backend)
    started = time.monotonic(); summary = None
    for target in (30000, 45000, 60000, 90000, 120000):
        while done < target:
            step = min(1000, target-done)
            state = sampler.run_mcmc(state, step, progress=False); done += step
            progress = {'variant': variant, 'steps': done, 'elapsed_this_invocation_s': time.monotonic()-started,
                        'host': socket.gethostname(), 'cpu_affinity': sorted(os.sched_getaffinity(0))}
            common.save(dest/'progress.json', progress); common.log('sampling', **progress)
        summary = common.summarize(sampler.get_chain(), sampler.get_log_prob(), engine.names, 5000)
        common.save(dest/'convergence_latest.json', summary)
        common.log('convergence', variant=variant, steps=done, gates=summary['gates'])
        if all(summary['gates'].values()):
            break
    extra = least_squares(engine.residual, summary['best_chain_theta'], bounds=(engine.lower, engine.upper),
                          x_scale='jac', max_nfev=1500, ftol=1e-11, xtol=1e-11, gtol=1e-11)
    if sum(extra.fun**2) < sum(best.fun**2):
        best = extra
    summary['posterior_raw'] = copy.deepcopy(summary['posterior'])
    factor = engine.corrections['percival_sigma_factor']
    for p in summary['posterior'].values():
        p['q16'] = p['q50']+factor*(p['q16']-p['q50'])
        p['q84'] = p['q50']+factor*(p['q84']-p['q50'])
        p['sigma68'] *= factor
    prediction = engine.predict(best.x)[0]
    summary.update({'status': 'pass' if all(summary['gates'].values()) else 'failed',
                    'map_theta': best.x, 'map_chi2': sum(best.fun**2),
                    'map_raw_chi2': engine.raw_metric.chi2(engine.data-prediction),
                    'map_success': best.success, 'map_prediction': prediction,
                    'multistart_chi2': [sum(s.fun**2) for s in solutions],
                    'nwalkers': 64, 'nsteps': done, 'burnin': 5000, 'seed': seed,
                    'acceptance_fraction_mean': np.mean(sampler.acceptance_fraction),
                    'data_dimension': len(engine.data), 'priors': {'lower': engine.lower, 'upper': engine.upper},
                    'finite_mock_corrections': engine.corrections,
                    'interval_convention': 'Hartlap in likelihood; Percival rescales q16/q84 around q50. Raw chains unchanged.',
                    'input_manifest_sha256': sha(dest/'input_manifest.json')})
    common.save(dest/'summary.json', summary)
    np.savez_compressed(dest/'samples.npz', chain=sampler.get_chain(discard=5000),
                        parameter_names=engine.names, map_theta=best.x)
    common.log('formal_fit_done', variant=variant, status=summary['status'], posterior=summary['posterior'])
    if summary['status'] != 'pass':
        raise RuntimeError('Convergence gates not yet passed')


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('variant', choices=('p02', 'xi02', 'joint'))
    p.add_argument('--compiled', required=True, type=Path)
    p.add_argument('--run-name', default='formal')
    p.add_argument('--geometry-audit', type=Path)
    a = p.parse_args()
    fit(a.variant, a.compiled, a.run_name, a.geometry_audit)
