#!/usr/bin/env python3
"""Formal 900-EZmock P02+xi02 fit with separate FoG parameters.

The shared chain remains untouched.  This diagnostic runs the same likelihood
for the EZmock and jaxpower covariances with theta = (fNL, b1, sigma_s_P,
sigma_s_xi, sn0), then writes a PDF comparing shared and split-sigma contours.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import emcee
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import numpy as np
from scipy.stats import gaussian_kde

import task43_rsd_ezmock281_mcmc_compare as base


ROOT = base.PROJECT_ROOT
RUN_SHARED = ROOT / "outputs/task43_outputs/ezmock_rsd_lightcone_covariance_z0p4_0p8_x1000_common50_nzmatch_b0.30_z0.60/mcmc_preliminary_900"
RUN_SPLIT = ROOT / "outputs/task43_outputs/ezmock_rsd_lightcone_covariance_z0p4_0p8_x1000_common50_nzmatch_b0.30_z0.60/mcmc_preliminary_900_split_sigma"

NAMES = ("fNL", "b1", "sigma_s_P", "sigma_s_xi", "sn0")
LOWER = np.asarray([-500.0, 0.5, 0.0, 0.0, -1.0], dtype="f8")
UPPER = np.asarray([500.0, 5.0, 30.0, 30.0, 1.0], dtype="f8")
INIT_SCALE = np.asarray([4.0, 0.015, 0.12, 0.12, 0.025], dtype="f8")


def correction_factors(ns: int, nb: int, npfit: int) -> dict[str, float]:
    hartlap = (ns - nb - 2.0) / (ns - 1.0)
    A = 2.0 / ((ns - nb - 1.0) * (ns - nb - 4.0))
    B = (ns - nb - 2.0) / ((ns - nb - 1.0) * (ns - nb - 4.0))
    m1 = (1.0 + B * (nb - npfit)) / (1.0 + A + B * (npfit + 1.0))
    return {"hartlap": hartlap, "percival_m1": m1, "percival_sigma_factor": float(np.sqrt(m1))}


def evaluate_models(pmodel, xmodel):
    def p(theta):
        a = pmodel.evaluate(np.asarray(theta, dtype="f8"))
        return np.r_[a[:13], a[13:26][base.P2_KEEP]]

    def x(theta):
        a = xmodel.evaluate(np.asarray(theta, dtype="f8")[:3], model="formal_gic", window_key="mean")
        return np.r_[a[0][base.XI_MASK], a[2][base.XI_MASK]]

    def split(theta):
        theta = np.asarray(theta, dtype="f8")
        fNL, b1, sigma_p, sigma_x, sn0 = theta
        return np.r_[p([fNL, b1, sigma_p, sn0]), x([fNL, b1, sigma_x])]

    return p, x, split


def map_fit(evaluate, data, covariance, start):
    from scipy.optimize import least_squares

    cov = 0.5 * (np.asarray(covariance, dtype="f8") + np.asarray(covariance, dtype="f8").T)
    chol = np.linalg.cholesky(cov)

    def residual(theta):
        return np.linalg.solve(chol, np.asarray(data) - evaluate(theta))

    starts = [np.asarray(start, dtype="f8"), np.asarray([-10., 2.4, 2.5, 5.7, 0.05]), np.asarray([20., 2.35, 4., 6., 0.15])]
    sols = [least_squares(residual, np.clip(s, LOWER + 1e-7, UPPER - 1e-7), bounds=(LOWER, UPPER), max_nfev=3000, xtol=1e-11, ftol=1e-11, gtol=1e-11) for s in starts]
    best = min(sols, key=lambda s: float(s.fun @ s.fun))
    return np.asarray(best.x, dtype="f8"), float(best.fun @ best.fun)


def summary(chain, logp, correction, covname):
    flat = chain.reshape(-1, chain.shape[-1])
    q = np.percentile(flat, [16., 50., 84.], axis=0)
    fac = float(np.sqrt(correction["percival_m1"]))
    post = {}
    for i, name in enumerate(NAMES):
        raw16, raw50, raw84 = map(float, q[:, i])
        post[name] = {
            "q16_raw": raw16, "q50": raw50, "q84_raw": raw84,
            "q16": raw50 + fac * (raw16 - raw50),
            "q84": raw50 + fac * (raw84 - raw50),
            "sigma68_raw": 0.5 * (raw84 - raw16),
            "sigma68": 0.5 * fac * (raw84 - raw16),
            "mean": float(np.mean(flat[:, i])), "std": float(np.std(flat[:, i], ddof=1)),
        }
    imax = int(np.argmax(logp.reshape(-1)))
    return {"covariance": covname, "names": list(NAMES), "posterior": post,
            "map_chain": dict(zip(NAMES, map(float, flat[imax]))),
            "map_chain_log_probability": float(np.max(logp)),
            "nwalkers": int(chain.shape[1]), "nsteps": int(chain.shape[0]),
            "percival_m1": float(correction["percival_m1"]),
            "percival_sigma_factor": fac}


def run_one(covname, cov, data, evaluate, out, nwalkers, nsteps, burnin, seed):
    nb, nparams = data.size, 5
    corr = correction_factors(900, nb, nparams)
    precision, pmeta = base.precision_from_cov(cov, corr["hartlap"])
    start, chi2 = map_fit(evaluate, data, cov, np.asarray([-4.7, 2.39, 2.60, 5.73, 0.08]))
    print("MAP", covname, start.tolist(), chi2, flush=True)

    def log_probability(theta):
        theta = np.asarray(theta, dtype="f8")
        if np.any(theta < LOWER) or np.any(theta > UPPER):
            return -np.inf
        diff = np.asarray(data) - evaluate(theta)
        return -0.5 * float(diff @ precision @ diff)

    rng = np.random.default_rng(int(seed))
    initial = start[None, :] + rng.normal(size=(nwalkers, nparams)) * INIT_SCALE[None, :]
    initial = np.clip(initial, LOWER + 1e-7, UPPER - 1e-7)
    sampler = emcee.EnsembleSampler(nwalkers, nparams, log_probability)
    print("CHAIN_START", covname, nwalkers, nsteps, flush=True)
    sampler.run_mcmc(initial, nsteps, progress=False)
    chain = np.asarray(sampler.get_chain(discard=burnin), dtype="f8")
    logp = np.asarray(sampler.get_log_prob(discard=burnin), dtype="f8")
    np.savez_compressed(out / f"chain_{covname}_joint_split_sigma.npz", chain=chain, logp=logp, data=data)
    result = summary(chain, logp, corr, covname)
    result.update({"map_theta": start.tolist(), "map_chi2": chi2, "precision_meta": pmeta,
                   "hartlap_percival": corr, "acceptance_fraction_mean": float(np.mean(sampler.acceptance_fraction))})
    (out / f"summary_{covname}_joint_split_sigma.json").write_text(json.dumps(result, indent=2, sort_keys=True))
    print("CHAIN_DONE", covname, json.dumps(result["posterior"], sort_keys=True), flush=True)
    return result


def contour(ax, chain, ix, iy, color, label):
    flat = chain.reshape(-1, chain.shape[-1])
    x, y = flat[:, ix], flat[:, iy]
    if x.size > 60000:
        rng = np.random.default_rng(20260919)
        sel = rng.choice(x.size, 60000, replace=False)
        x, y = x[sel], y[sel]
    kde = gaussian_kde(np.vstack([x, y]))
    xx = np.linspace(np.percentile(x, .2), np.percentile(x, 99.8), 160)
    yy = np.linspace(np.percentile(y, .2), np.percentile(y, 99.8), 160)
    X, Y = np.meshgrid(xx, yy)
    Z = kde(np.vstack([X.ravel(), Y.ravel()])).reshape(X.shape)
    levels = np.sort([np.quantile(Z, .50), np.quantile(Z, .10)])
    ax.contour(X, Y, Z, levels=levels, colors=[color], linewidths=[1.1, 2.0])
    ax.plot(np.median(x), np.median(y), "o", ms=4, color=color, label=label)


def make_plot(out, results):
    with np.load(RUN_SHARED / "chain_ezmock900_joint.npz") as d:
        shared = np.asarray(d["chain"], dtype="f8")
    with np.load(out / "chain_ezmock900_joint_split_sigma.npz") as d:
        split = np.asarray(d["chain"], dtype="f8")
    with np.load(out / "chain_jaxpower_joint_split_sigma.npz") as d:
        split_j = np.asarray(d["chain"], dtype="f8")
    with np.load(RUN_SHARED / "chain_jaxpower_joint.npz") as d:
        shared_j = np.asarray(d["chain"], dtype="f8")

    pdf = out / "task43_ezmock900_joint_shared_vs_split_sigma_contours.pdf"
    with PdfPages(pdf) as pages:
        fig, ax = plt.subplots(figsize=(7.6, 6.4))
        contour(ax, shared, 0, 1, "#d62728", "shared σs · EZmock900")
        contour(ax, split, 0, 1, "#1f77b4", "split σP/σξ · EZmock900")
        contour(ax, shared_j, 0, 1, "#fb6a4a", "shared σs · jaxpower")
        contour(ax, split_j, 0, 1, "#6baed6", "split σP/σξ · jaxpower")
        ax.set(xlabel=r"$f_{\rm NL}$", ylabel=r"$b_1$", title="Task43 joint: shared vs separate FoG parameters")
        ax.grid(alpha=.18); ax.legend(fontsize=8, frameon=False); fig.tight_layout(); pages.savefig(fig); plt.close(fig)

        fig, axes = plt.subplots(1, 2, figsize=(10.8, 4.6))
        contour(axes[0], split, 2, 3, "#1f77b4", "EZmock900 split")
        contour(axes[0], split_j, 2, 3, "#6baed6", "jaxpower split")
        axes[0].axline((0, 0), slope=1, color="0.45", ls="--", lw=.9)
        axes[0].set(xlabel=r"$\sigma_{s,P}$", ylabel=r"$\sigma_{s,\xi}$", title="Independent FoG posterior")
        axes[0].grid(alpha=.18); axes[0].legend(fontsize=8, frameon=False)
        shared_flat = shared.reshape(-1, shared.shape[-1])
        split_flat = split.reshape(-1, split.shape[-1])
        for arr, color, label in [(shared_flat[:, 2], "#d62728", "shared σs · EZmock900"), (split_flat[:, 2], "#1f77b4", "σP · EZmock900"), (split_flat[:, 3], "#2ca02c", "σξ · EZmock900")]:
            axes[1].hist(arr, bins=70, density=True, histtype="step", lw=1.5, color=color, label=label)
        axes[1].set(xlabel=r"$\sigma_s$", ylabel="density", title="FoG marginal posteriors")
        axes[1].grid(alpha=.18); axes[1].legend(fontsize=8, frameon=False)
        fig.tight_layout(); pages.savefig(fig); plt.close(fig)
    print("PLOT", pdf, flush=True)
    return pdf


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--nwalkers", type=int, default=64)
    ap.add_argument("--nsteps", type=int, default=30000)
    ap.add_argument("--burnin", type=int, default=5000)
    ap.add_argument("--out", type=Path, default=RUN_SPLIT)
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    pmodel, xmodel, data_p, data_x, _, _ = base.build_models_and_data()
    data = np.r_[data_p, data_x]
    _, _, evaluate = evaluate_models(pmodel, xmodel)
    with np.load(RUN_SHARED / "ezmock900_covariance_and_stack.npz") as d:
        cov_e = np.asarray(d["covariance"], dtype="f8")
    with np.load(base.ANALYTIC_COV) as d:
        cov_j = np.asarray(d["rsd_joint"], dtype="f8")
    results = {}
    results["ezmock900"] = run_one("ezmock900", cov_e, data, evaluate, args.out, args.nwalkers, args.nsteps, args.burnin, 20260919)
    results["jaxpower"] = run_one("jaxpower", cov_j, data, evaluate, args.out, args.nwalkers, args.nsteps, args.burnin, 20260920)
    (args.out / "comparison_summary_split_sigma.json").write_text(json.dumps(results, indent=2, sort_keys=True))
    make_plot(args.out, results)


if __name__ == "__main__":
    main()
