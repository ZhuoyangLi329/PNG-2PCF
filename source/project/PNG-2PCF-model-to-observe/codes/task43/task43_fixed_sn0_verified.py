#!/usr/bin/env python3
"""Audited sn0=0 joint run: shared fNL,b1,sigma_s and the full covariance.

Reuse the already sampled empirical chain only after checking its data and
likelihood. Analytic covariance receives no finite-mock corrections. Report
MAP and posterior quantiles separately, and require actual convergence gates.
"""
import argparse
import json
from pathlib import Path
import shutil
import time

import emcee
import numpy as np
from scipy.linalg import solve_triangular
from scipy.optimize import least_squares

import task43_rsd_ezmock281_mcmc_compare as base
from task43_joint_rsd_pkxi_fit import summarize_chain

ROOT = base.PROJECT_ROOT
PARENT = ROOT / 'outputs/task43_outputs/ezmock_rsd_lightcone_covariance_z0p4_0p8_x1000_common50_nzmatch_b0.30_z0.60'
SHARED = PARENT / 'mcmc_preliminary_900'
OUT = PARENT / 'mcmc_preliminary_900_fixed_sn0_verified'
OLD = PARENT / 'mcmc_preliminary_900_fixed_sn0'
NAMES = ('fNL', 'b1', 'sigma_s')
LOWER, UPPER = base.BOUNDS_LO[:3], base.BOUNDS_HI[:3]


class Metric:
    def __init__(self, covariance, hartlap):
        cov = (covariance + covariance.T) / 2
        self.scale = np.sqrt(np.diag(cov))
        corr = cov / np.outer(self.scale, self.scale)
        self.chol = np.linalg.cholesky(corr)
        self.hartlap = float(hartlap)
        self.min_eigen = float(np.linalg.eigvalsh(corr).min())

    def residual(self, diff):
        return solve_triangular(self.chol, diff / self.scale, lower=True, check_finite=False)

    def chi2(self, diff):
        z = self.residual(diff)
        return float(z @ z)


def evaluate_models():
    pm, xm, dp, dx, _, _ = base.build_models_and_data()
    def evaluate(t):
        f, b, s = t
        p = pm.evaluate(np.array([f, b, s, 0.]))
        x = xm.evaluate(np.array([f, b, s]), model='formal_gic', window_key='mean')
        return np.r_[p[:13], p[13:26][base.P2_KEEP], x[0][base.XI_MASK], x[2][base.XI_MASK]]
    return evaluate, np.r_[dp, dx]


def fit_map(evaluate, data, metric):
    starts = [s[:3] for s in base.OPTIMIZER_STARTS] + [np.array([0., 2.405, 1.])]
    sols = [least_squares(lambda t: metric.residual(data-evaluate(t)), start,
                          bounds=(LOWER, UPPER), x_scale='jac', max_nfev=3000,
                          xtol=1e-11, ftol=1e-11, gtol=1e-11) for start in starts]
    best = min(sols, key=lambda s: float(s.fun @ s.fun))
    if not best.success:
        raise RuntimeError(best.message)
    return best.x, {'chi2_single': float(best.fun @ best.fun),
                    'chi2_hartlap': metric.hartlap*float(best.fun @ best.fun),
                    'all_start_chi2': [float(s.fun @ s.fun) for s in sols]}


def run_one(name, covariance, data, evaluate, reuse, nsteps, burnin, nwalkers):
    corr = (base.correction_factors(900, data.size, 3) if name == 'ezmock900' else
            {'hartlap': 1., 'percival_m1': 1., 'percival_sigma_factor': 1.,
             'nmock': 0, 'ndata': int(data.size), 'nparams': 3})
    metric = Metric(covariance, corr['hartlap'])
    theta, ml = fit_map(evaluate, data, metric)
    print('MAP', name, theta.tolist(), ml, flush=True)
    def logp(t):
        if not np.all(np.isfinite(t)) or np.any(t < LOWER) or np.any(t > UPPER):
            return -np.inf
        return -0.5*metric.hartlap*metric.chi2(data-evaluate(t))

    target = OUT / f'chain_{name}_joint_fixed_sn0.npz'
    if reuse:
        source = OLD / f'chain_{name}_joint_fixed_sn0.npz'
        old_summary = json.loads((OLD/f'summary_{name}_joint_fixed_sn0.json').read_text())
        if not np.isclose(old_summary['hartlap_percival']['hartlap'], corr['hartlap'], rtol=0, atol=1e-14):
            raise ValueError('wrong finite-mock correction in source; resample instead')
        with np.load(source) as d:
            chain, lp = d['chain'], d['logp']
            if not np.array_equal(data, d['data']):
                raise ValueError('reused chain data mismatch')
        acceptance = old_summary['acceptance_fraction_mean']
        burnin = old_summary['burnin']
        nsteps, nwalkers = len(chain)+burnin, chain.shape[1]
        provenance = str(source)
    else:
        if target.exists():
            raise FileExistsError(target)
        rng = np.random.default_rng(2026092201)
        initial = theta + rng.normal(size=(nwalkers, 3))*np.array([4., .015, .12])
        initial = np.clip(initial, LOWER+1e-7, UPPER-1e-7)
        sampler = emcee.EnsembleSampler(nwalkers, 3, logp)
        sampler.random_state = np.random.RandomState(2026092202).get_state()
        tic = time.time()
        for i, _ in enumerate(sampler.sample(initial, iterations=nsteps), 1):
            if i % 5000 == 0:
                print('PROGRESS', name, i, 'elapsed_s', round(time.time()-tic), flush=True)
        chain = sampler.get_chain(discard=burnin)
        lp = sampler.get_log_prob(discard=burnin)
        acceptance = float(sampler.acceptance_fraction.mean())
        provenance = 'new chain with normalized-covariance Cholesky likelihood'

    flat = chain.reshape(-1, 3)
    flatlp = lp.reshape(-1)
    check_indices = np.linspace(0, len(flat)-1, 128, dtype=int)
    maxdiff = max(abs(logp(flat[i])-flatlp[i]) for i in check_indices)
    if maxdiff > 1e-8:
        raise ValueError(f'log likelihood mismatch: {maxdiff}')
    raw = summarize_chain(chain, lp, NAMES)
    raw['posterior_raw'] = json.loads(json.dumps(raw['posterior']))
    result = base.corrected_summary(raw, corr['percival_m1'])
    result.update({'covariance': name, 'variant': 'joint_fixed_sn0', 'names': NAMES,
                   'fixed_parameters': {'sn0': 0.}, 'shared_parameters': NAMES,
                   'hartlap_percival': corr, 'map_theta': theta.tolist(), 'map_fit': ml,
                   'nsteps': nsteps, 'burnin': burnin, 'nsteps_postburn': len(chain),
                   'nwalkers': nwalkers, 'acceptance_fraction_mean': acceptance,
                   'max_logp_check_difference': maxdiff, 'source_chain': provenance,
                   'likelihood_metric': 'normalized covariance Cholesky, no eigen clipping',
                   'covariance_min_correlation_eigenvalue': metric.min_eigen,
                   'data_contract': 'Abacus x25 mean; C_single, never /25; P0 13 P2 9 xi0 26 xi2 26; full cross'})
    if reuse:
        shutil.copy2(source, target)
    else:
        np.savez_compressed(target, chain=chain, logp=lp, data=data)
    (OUT/f'summary_{name}_joint_fixed_sn0.json').write_text(json.dumps(result, indent=2)+'\n')
    print('RESULT', name, json.dumps({'posterior': result['posterior'], 'gates': result['gates'],
                                     'split_rhat': result['split_rhat'], 'postburn_length_over_tau': result['postburn_length_over_tau']}), flush=True)
    if not all(result['gates'].values()):
        raise RuntimeError(f'Convergence gates failed for {name}')
    return result


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--nsteps', type=int, default=30000)
    ap.add_argument('--burnin', type=int, default=5000)
    ap.add_argument('--nwalkers', type=int, default=64)
    ap.add_argument('--covariances', nargs='+', default=['ezmock900', 'jaxpower'])
    args = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    evaluate, data = evaluate_models()
    with np.load(SHARED/'ezmock900_covariance_and_stack.npz') as d:
        assert np.array_equal(d['production_indices'], np.arange(900))
        ce = d['covariance']
    with np.load(base.ANALYTIC_COV) as d:
        cj = d['rsd_joint']
    result = {}
    for name in args.covariances:
        result[name] = run_one(name, ce if name == 'ezmock900' else cj, data, evaluate,
                               reuse=name == 'ezmock900', nsteps=args.nsteps,
                               burnin=args.burnin, nwalkers=args.nwalkers)
    (OUT/'comparison_summary_fixed_sn0.json').write_text(json.dumps(result, indent=2)+'\n')


if __name__ == '__main__':
    main()
