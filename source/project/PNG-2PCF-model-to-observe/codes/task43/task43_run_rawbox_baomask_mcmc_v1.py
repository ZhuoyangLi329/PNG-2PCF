#!/usr/bin/env python3
"""Rawbox real/RSD xi0 MCMC with the canonical 80--120 Mpc/h BAO mask.

The P0 chains are frozen inputs.  This script refits only xi0, using the same
large-scale PNG response and finite-box operator as the historical pipeline,
then compares the new posteriors with the validated real- and redshift-space
P0 chains.
"""

from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import os
from pathlib import Path
import time
from typing import Any, Callable

for _name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_name, "1")

import emcee
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import numpy as np
from scipy.optimize import least_squares
from scipy.stats import chi2 as chi2_distribution

from task43_fit_rsd_rawbox_x25 import FastRSDModel, load_x25, set_affinity, split_rhat
from task43_plot_rsd_rawbox_pk0_vs_xi0_contours import draw_contour, plot_range
from task43_rawbox_numerics import GaussianMetric
from task43_rsd_common import (
    OUTPUT_ROOT,
    PHASES,
    PLOT_ROOT,
    S_EDGES,
    atomic_savez,
    atomic_write_json,
    rawbox_xi_primary_mask,
    sha256_file,
)
from task43_rsd_model import DELTA_C, FullDiscreteRSDModel, build_cache


OUT_DIR = OUTPUT_ROOT / "rawbox" / "standard_baomask80_120_v1"
OUT_NPZ = OUT_DIR / "task43_rawbox_real_rsd_xi0_baomask80_120_mcmc_v1.npz"
OUT_JSON = OUT_NPZ.with_suffix(".json")
OUT_PDF = PLOT_ROOT / "task43_rawbox_real_rsd_pk0_vs_xi0_baomask80_120_v1.pdf"

REAL_P_NPZ = OUTPUT_ROOT / "rawbox" / "realspace_check" / "fits" / "pk0_s16" / "samples.npz"
REAL_P_JSON = OUTPUT_ROOT / "rawbox" / "realspace_check" / "audits" / "task43_rsd_rawbox_realspace_pk_check.json"
RSD_P_NPZ = OUTPUT_ROOT / "rawbox" / "comparison" / (
    "task43_rsd_rawbox_x25_pk0_kmin0p003_vs_xi0_smin50_l0only_longchain.npz"
)
RSD_P_JSON = RSD_P_NPZ.with_suffix(".json")

REAL_NAMES = ("fNL", "b1")
RSD_NAMES = ("fNL", "b1", "sigma_s")
REAL_LO = np.asarray([-500.0, 0.2], dtype="f8")
REAL_HI = np.asarray([500.0, 10.0], dtype="f8")
RSD_LO = np.asarray([-500.0, 0.5, 0.0], dtype="f8")
RSD_HI = np.asarray([500.0, 5.0, 30.0], dtype="f8")
REAL_STARTS = (
    np.asarray([0.0, 2.65]),
    np.asarray([-100.0, 2.3]),
    np.asarray([100.0, 2.8]),
)
RSD_STARTS = (
    np.asarray([0.0, 2.55, 8.0]),
    np.asarray([-100.0, 2.3, 4.0]),
    np.asarray([100.0, 2.8, 12.0]),
)
NPHASE = len(PHASES)
_POOL_LOG_PROBABILITY: Callable[[np.ndarray], float] | None = None


def _evaluate_pool_log_probability(theta: np.ndarray) -> float:
    if _POOL_LOG_PROBABILITY is None:
        raise RuntimeError("MCMC worker likelihood was not initialized")
    return _POOL_LOG_PROBABILITY(theta)


def realspace_basis(exact: FullDiscreteRSDModel) -> np.ndarray:
    """Return the three finite-box xi bases for 1, alpha and alpha^2."""

    mode_weight = np.asarray(exact.g_nz * exact.pk_dd, dtype="f8")
    alpha = np.asarray(exact.alpha, dtype="f8")
    kernel0 = np.asarray(exact.kernels[0], dtype="f8")
    return np.stack(
        [
            (mode_weight * alpha**power) @ kernel0 / float(exact.volume)
            for power in range(3)
        ]
    )


def evaluate_realspace(theta: np.ndarray, basis: np.ndarray) -> np.ndarray:
    fnl, b1 = map(float, np.asarray(theta, dtype="f8")[:2])
    q = fnl * 2.0 * DELTA_C * (b1 - 1.0)
    return b1**2 * basis[0] + 2.0 * b1 * q * basis[1] + q**2 * basis[2]


def realspace_covariance(exact: FullDiscreteRSDModel, *, b1: float, nbar: float) -> np.ndarray:
    total2 = (float(b1) ** 2 * np.asarray(exact.pk_dd, dtype="f8") + 1.0 / float(nbar)) ** 2
    weight = 2.0 * np.asarray(exact.g_nz, dtype="f8") * total2 / float(exact.volume) ** 2
    kernel0 = np.asarray(exact.kernels[0], dtype="f8")
    return kernel0.T @ (weight[:, None] * kernel0)


def rsd_monopole_covariance(
    exact: FullDiscreteRSDModel,
    *,
    b1: float,
    sigma_s: float,
    nbar: float,
) -> np.ndarray:
    """Return the unfloored periodic Gaussian xi0 covariance."""

    x = (exact.k_eff[:, None] * exact.mu[None, :] * float(sigma_s)) ** 2
    damping = 1.0 / (1.0 + 0.5 * x) ** 2
    signal = exact.pk_dd[:, None] * (
        float(b1) + float(exact.f_growth) * exact.mu2[None, :]
    ) ** 2 * damping
    total2 = (signal + 1.0 / float(nbar)) ** 2
    angular = np.sum(exact.wmu[None, :] * total2, axis=1)
    weight = np.asarray(exact.g_nz, dtype="f8") * angular / float(exact.volume) ** 2
    kernel0 = np.asarray(exact.kernels[0], dtype="f8")
    return kernel0.T @ (weight[:, None] * kernel0)


def fit_map(
    data_full: np.ndarray,
    evaluate: Callable[[np.ndarray], np.ndarray],
    covariance_full: np.ndarray,
    mask: np.ndarray,
    *,
    names: tuple[str, ...],
    bounds: tuple[np.ndarray, np.ndarray],
    starts: tuple[np.ndarray, ...],
) -> tuple[dict[str, Any], GaussianMetric]:
    ids = np.flatnonzero(mask)
    data = np.asarray(data_full, dtype="f8")[ids]
    metric = GaussianMetric(np.asarray(covariance_full, dtype="f8")[np.ix_(ids, ids)])

    def residual(theta: np.ndarray) -> np.ndarray:
        return metric.residual(data - np.asarray(evaluate(theta), dtype="f8")[ids])

    solutions = [
        least_squares(
            residual,
            start,
            bounds=bounds,
            max_nfev=3000,
            xtol=1.0e-12,
            ftol=1.0e-12,
            gtol=1.0e-12,
        )
        for start in starts
    ]
    best = min(solutions, key=lambda result: float(result.fun @ result.fun))
    theta = np.asarray(best.x, dtype="f8")
    fisher_covariance = np.linalg.pinv(best.jac.T @ best.jac, rcond=1.0e-12)
    errors = np.sqrt(np.maximum(np.diag(fisher_covariance), 0.0))
    chi2_single = float(best.fun @ best.fun)
    chi2_mean = float(NPHASE * chi2_single)
    dof = int(ids.size - len(names))
    summary = {
        "parameter_names": list(names),
        "theta": {name: float(value) for name, value in zip(names, theta, strict=True)},
        "sigma_laplace_single": {name: float(value) for name, value in zip(names, errors, strict=True)},
        "covariance_laplace_single": fisher_covariance.tolist(),
        "chi2_single": chi2_single,
        "chi2_mean": chi2_mean,
        "dof": dof,
        "pte_mean": float(chi2_distribution.sf(chi2_mean, dof)),
        "success": bool(best.success),
        "message": str(best.message),
        "prediction_full": np.asarray(evaluate(theta), dtype="f8").tolist(),
    }
    return summary, metric


def run_chain(
    data_full: np.ndarray,
    evaluate: Callable[[np.ndarray], np.ndarray],
    mask: np.ndarray,
    metric: GaussianMetric,
    map_summary: dict[str, Any],
    *,
    names: tuple[str, ...],
    bounds: tuple[np.ndarray, np.ndarray],
    initial_scale: np.ndarray,
    nwalkers: int,
    nsteps: int,
    burnin: int,
    seed: int,
    nworkers: int,
) -> tuple[dict[str, Any], np.ndarray, np.ndarray]:
    ids = np.flatnonzero(mask)
    data = np.asarray(data_full, dtype="f8")[ids]
    lower, upper = bounds

    def log_probability(theta: np.ndarray) -> float:
        value = np.asarray(theta, dtype="f8")
        if np.any(value < lower) or np.any(value > upper):
            return -np.inf
        residual = metric.residual(data - np.asarray(evaluate(value), dtype="f8")[ids])
        return -0.5 * float(residual @ residual)

    center = np.asarray([map_summary["theta"][name] for name in names], dtype="f8")
    rng = np.random.default_rng(int(seed))
    initial = center[None, :] + rng.normal(size=(int(nwalkers), len(names))) * initial_scale[None, :]
    initial = np.clip(initial, lower + 1.0e-7, upper - 1.0e-7)
    np.random.seed(int(seed))
    workers = int(nworkers)
    if not 1 <= workers <= 8:
        raise ValueError("nworkers must be in [1, 8]")
    global _POOL_LOG_PROBABILITY
    _POOL_LOG_PROBABILITY = log_probability
    try:
        if workers == 1:
            sampler = emcee.EnsembleSampler(int(nwalkers), len(names), log_probability)
            sampler.run_mcmc(initial, int(nsteps), progress=False)
        else:
            # The parent affinity is already restricted to the requested CPU
            # set. Forked workers inherit both that set and the read-only model.
            with mp.get_context("fork").Pool(processes=workers) as pool:
                sampler = emcee.EnsembleSampler(
                    int(nwalkers), len(names), _evaluate_pool_log_probability, pool=pool
                )
                sampler.run_mcmc(initial, int(nsteps), progress=False)
    finally:
        _POOL_LOG_PROBABILITY = None
    chain = np.asarray(sampler.get_chain(discard=int(burnin)), dtype="f8")
    logp = np.asarray(sampler.get_log_prob(discard=int(burnin)), dtype="f8")
    flat = chain.reshape(-1, len(names))
    flat_logp = logp.reshape(-1)
    quantiles = np.percentile(flat, [16.0, 50.0, 84.0], axis=0)
    sigma68 = 0.5 * (quantiles[2] - quantiles[0])
    try:
        tau = np.asarray(emcee.autocorr.integrated_time(chain, quiet=True, tol=0), dtype="f8")
    except Exception:
        tau = np.full(len(names), np.inf)
    rhat = split_rhat(chain)
    half = chain.shape[0] // 2
    first = chain[:half].reshape(-1, len(names))
    second = chain[-half:].reshape(-1, len(names))
    half_shift = np.abs(np.median(first, axis=0) - np.median(second, axis=0)) / sigma68
    imax = int(np.argmax(flat_logp))
    summary = {
        "posterior": {
            name: {
                "q16": float(quantiles[0, index]),
                "q50": float(quantiles[1, index]),
                "q84": float(quantiles[2, index]),
                "sigma68": float(sigma68[index]),
                "mean": float(np.mean(flat[:, index])),
                "std": float(np.std(flat[:, index], ddof=1)),
            }
            for index, name in enumerate(names)
        },
        "map_chain": {name: float(flat[imax, index]) for index, name in enumerate(names)},
        "map_chain_log_probability": float(flat_logp[imax]),
        "nwalkers": int(nwalkers),
        "nsteps": int(nsteps),
        "burnin": int(burnin),
        "postburn_steps_per_walker": int(chain.shape[0]),
        "acceptance_fraction_mean": float(np.mean(sampler.acceptance_fraction)),
        "parallel_workers": workers,
        "tau": {name: float(tau[index]) for index, name in enumerate(names)},
        "split_rhat": {name: float(rhat[index]) for index, name in enumerate(names)},
        "postburn_length_over_tau": {
            name: float(chain.shape[0] / tau[index]) for index, name in enumerate(names)
        },
        "half_chain_shift_sigma": {
            name: float(half_shift[index]) for index, name in enumerate(names)
        },
    }
    summary["gates"] = {
        "split_rhat_max_below_1p01": bool(np.max(rhat) < 1.01),
        "postburn_length_min_above_50tau": bool(np.min(chain.shape[0] / tau) > 50.0),
        "half_chain_shift_max_below_0p1sigma": bool(np.max(half_shift) < 0.1),
    }
    return summary, chain, logp


def validated_frozen_p_inputs() -> dict[str, Any]:
    for path in (REAL_P_NPZ, REAL_P_JSON, RSD_P_NPZ, RSD_P_JSON):
        if not path.is_file():
            raise FileNotFoundError(f"missing frozen P0 input: {path}")
    real_json = json.loads(REAL_P_JSON.read_text(encoding="utf-8"))
    rsd_json = json.loads(RSD_P_JSON.read_text(encoding="utf-8"))
    if real_json.get("status") != "complete" or rsd_json.get("status") != "complete":
        raise RuntimeError("a frozen P0 input is not complete")
    if not all(rsd_json["pk0"]["mcmc"]["gates"].values()):
        raise RuntimeError("frozen RSD P0 chain did not pass convergence gates")
    with np.load(REAL_P_NPZ, allow_pickle=False) as payload:
        real_chain = np.asarray(payload["chain_by_step"], dtype="f8")
    with np.load(RSD_P_NPZ, allow_pickle=False) as payload:
        rsd_chain = np.asarray(payload["pk0_chain_by_step"], dtype="f8")
    if real_chain.ndim != 3 or real_chain.shape[2] != 2:
        raise RuntimeError(f"unexpected real P0 chain shape {real_chain.shape}")
    if rsd_chain.ndim != 3 or rsd_chain.shape[2] != 4:
        raise RuntimeError(f"unexpected RSD P0 chain shape {rsd_chain.shape}")
    return {
        "real_json": real_json,
        "rsd_json": rsd_json,
        "real_chain": real_chain,
        "rsd_chain": rsd_chain,
        "hashes": {str(path): sha256_file(path) for path in (REAL_P_NPZ, REAL_P_JSON, RSD_P_NPZ, RSD_P_JSON)},
    }


def posterior_comparison(
    left: dict[str, Any], right: dict[str, Any], names: tuple[str, ...]
) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for name in names:
        prow = left[name]
        xrow = right[name]
        p_sigma = 0.5 * (float(prow["q84"]) - float(prow["q16"]))
        x_sigma = float(xrow.get("sigma68", 0.5 * (float(xrow["q84"]) - float(xrow["q16"]))))
        delta = float(prow["q50"]) - float(xrow["q50"])
        result[name] = {
            "P0_q50": float(prow["q50"]),
            "P0_sigma68": p_sigma,
            "xi0_q50": float(xrow["q50"]),
            "xi0_sigma68": x_sigma,
            "P0_minus_xi0": delta,
            "difference_over_independent_quadrature_sigma_diagnostic": float(
                delta / np.hypot(p_sigma, x_sigma)
            ),
        }
    return result


def _posterior_from_samples(samples: np.ndarray, names: tuple[str, ...]) -> dict[str, Any]:
    quantiles = np.percentile(np.asarray(samples, dtype="f8"), [16.0, 50.0, 84.0], axis=0)
    return {
        name: {
            "q16": float(quantiles[0, index]),
            "q50": float(quantiles[1, index]),
            "q84": float(quantiles[2, index]),
        }
        for index, name in enumerate(names)
    }


def draw_triangle_page(
    pdf: PdfPages,
    p_samples: np.ndarray,
    xi_samples: np.ndarray,
    names: tuple[str, ...],
    *,
    title: str,
    legacy_xi_samples: np.ndarray | None = None,
    fnl_maximum_likelihood: dict[str, float] | None = None,
) -> None:
    labels = {
        "fNL": r"$f_{\rm NL}$",
        "b1": r"$b_1$",
        "sigma_s": r"$\sigma_s\ [h^{-1}{\rm Mpc}]$",
    }
    p_full = np.asarray(p_samples, dtype="f8")
    x_full = np.asarray(xi_samples, dtype="f8")
    legacy_full = None if legacy_xi_samples is None else np.asarray(legacy_xi_samples, dtype="f8")
    p = p_full[::8]
    x = x_full[::8]
    legacy = None if legacy_full is None else legacy_full[::8]
    ranges = {}
    for index, name in enumerate(names):
        comparison = x[:, index] if legacy is None else np.concatenate((x[:, index], legacy[:, index]))
        ranges[name] = plot_range(p[:, index], comparison, parameter=name)
    ndim = len(names)
    figure, axes = plt.subplots(ndim, ndim, figsize=(8.2, 7.8) if ndim == 3 else (6.6, 5.9))
    axes = np.atleast_2d(axes)
    for row, yname in enumerate(names):
        for column, xname in enumerate(names):
            axis = axes[row, column]
            if column > row:
                axis.set_axis_off()
                continue
            if row == column:
                bins = np.linspace(*ranges[xname], 90)
                axis.hist(p[:, column], bins=bins, density=True, histtype="step", lw=1.8, color="#252525")
                if legacy is not None:
                    axis.hist(
                        legacy[:, column], bins=bins, density=True, histtype="step", lw=1.8,
                        color="#4C72B0",
                    )
                axis.hist(x[:, column], bins=bins, density=True, histtype="step", lw=1.8, color="#C44E52")
                axis.set_xlim(*ranges[xname])
                axis.set_yticks([])
            else:
                if legacy is not None:
                    draw_contour(
                        axis,
                        legacy[:, column],
                        legacy[:, row],
                        color="#4C72B0",
                        xlim=ranges[xname],
                        ylim=ranges[yname],
                        zorder=1,
                    )
                draw_contour(
                    axis,
                    x[:, column],
                    x[:, row],
                    color="#C44E52",
                    xlim=ranges[xname],
                    ylim=ranges[yname],
                    zorder=2,
                )
                draw_contour(
                    axis,
                    p[:, column],
                    p[:, row],
                    color="#252525",
                    xlim=ranges[xname],
                    ylim=ranges[yname],
                    zorder=4,
                )
                axis.set(xlim=ranges[xname], ylim=ranges[yname])
            if xname == "fNL":
                axis.axvline(0.0, color="0.55", lw=0.8, ls="--", zorder=0)
            if row < ndim - 1:
                axis.tick_params(labelbottom=False)
            else:
                axis.set_xlabel(labels[xname])
            if column == 0 and row > 0:
                axis.set_ylabel(labels[yname])
            elif column > 0 and row != column:
                axis.tick_params(labelleft=False)
    def fnl_label(samples: np.ndarray, series: str) -> str:
        q16, q50, q84 = np.percentile(samples[:, names.index("fNL")], [16.0, 50.0, 84.0])
        maximum_likelihood = (
            q50
            if fnl_maximum_likelihood is None
            else float(fnl_maximum_likelihood[series])
        )
        return (
            rf"{maximum_likelihood:.2f}"
            rf"_{{-{maximum_likelihood - q16:.2f}}}"
            rf"^{{+{q84 - maximum_likelihood:.2f}}}"
        )

    handles = [
        plt.Line2D([], [], color="#252525", lw=2.0),
        *([] if legacy_full is None else [plt.Line2D([], [], color="#4C72B0", lw=2.0)]),
        plt.Line2D([], [], color="#C44E52", lw=2.0),
    ]
    legend_labels = [rf"frozen $P_0(k)$: $f_{{\rm NL}}^{{\rm ML}}={fnl_label(p_full, 'p0')}$"]
    if legacy_full is not None:
        legend_labels.append(
            rf"$\xi_0(s)$, no BAO mask: $f_{{\rm NL}}^{{\rm ML}}={fnl_label(legacy_full, 'xi0_unmasked')}$"
        )
    legend_labels.append(
        rf"$\xi_0(s)$, $80\!\leq\!s\!<\!120$ removed: "
        rf"$f_{{\rm NL}}^{{\rm ML}}={fnl_label(x_full, 'xi0_masked')}$"
    )
    figure.legend(
        handles,
        legend_labels,
        loc="upper right",
        bbox_to_anchor=(0.965, 0.875),
        frameon=False,
        fontsize=9.5,
    )
    figure.suptitle(title, fontsize=12.0, y=0.985)
    figure.subplots_adjust(left=0.13, right=0.97, bottom=0.10, top=0.93, wspace=0.08, hspace=0.08)
    pdf.savefig(figure, bbox_inches="tight", pad_inches=0.08)
    plt.close(figure)


def draw_residual_page(
    pdf: PdfPages,
    centers: np.ndarray,
    mask: np.ndarray,
    real_data: np.ndarray,
    rsd_data: np.ndarray,
    real_prediction: np.ndarray,
    rsd_prediction: np.ndarray,
    real_covariance: np.ndarray,
    rsd_covariance: np.ndarray,
    summaries: dict[str, Any],
) -> None:
    figure, axes = plt.subplots(2, 2, figsize=(11.0, 7.0), sharex="col", gridspec_kw={"height_ratios": [2.0, 1.0]})
    for column, (label, data, prediction, covariance, summary) in enumerate(
        (
            ("real space", real_data, real_prediction, real_covariance, summaries["real"]),
            ("redshift space", rsd_data, rsd_prediction, rsd_covariance, summaries["rsd"]),
        )
    ):
        sigma_mean = np.sqrt(np.diag(covariance) / NPHASE)
        excluded = (centers >= 50.0) & (centers < 350.0) & ~mask
        below = centers < 50.0
        axes[0, column].errorbar(
            centers[mask],
            centers[mask] ** 2 * data[mask],
            yerr=centers[mask] ** 2 * sigma_mean[mask],
            fmt="o",
            ms=4,
            color="#252525",
            label="fitted bins",
        )
        axes[0, column].errorbar(
            centers[excluded],
            centers[excluded] ** 2 * data[excluded],
            yerr=centers[excluded] ** 2 * sigma_mean[excluded],
            fmt="x",
            ms=6,
            color="#D68A22",
            label="excluded BAO bins",
        )
        axes[0, column].plot(centers, centers**2 * prediction, color="#C44E52", lw=1.6, label="MAP model")
        axes[0, column].scatter(centers[below], centers[below] ** 2 * data[below], s=15, facecolors="none", edgecolors="0.65")
        axes[0, column].axvspan(80.0, 120.0, color="#D68A22", alpha=0.10, lw=0.0)
        axes[0, column].set_title(
            f"{label}: $\\chi^2_{{\\rm mean}}={summary['chi2_mean']:.1f}/{summary['dof']}$, "
            f"PTE={summary['pte_mean']:.3f}"
        )
        axes[0, column].set_ylabel(r"$s^2\xi_0(s)$")
        axes[0, column].legend(frameon=False, fontsize=8.5)
        residual = (data - prediction) / sigma_mean
        axes[1, column].axhline(0.0, color="0.55", lw=0.8)
        axes[1, column].plot(centers[mask], residual[mask], "o-", ms=3.5, lw=0.8, color="#252525")
        axes[1, column].plot(centers[excluded], residual[excluded], "x", ms=6, color="#D68A22")
        axes[1, column].axvspan(80.0, 120.0, color="#D68A22", alpha=0.10, lw=0.0)
        axes[1, column].set(xlabel=r"$s\ [h^{-1}{\rm Mpc}]$", ylabel=r"residual / $\sigma_{\rm mean}$", xlim=(45.0, 355.0))
    figure.suptitle(r"Canonical rawbox mask: $50\leq s<350$, excluding $80\leq s<120\ h^{-1}{\rm Mpc}$", fontsize=12.0)
    figure.tight_layout()
    pdf.savefig(figure, bbox_inches="tight", pad_inches=0.08)
    plt.close(figure)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--threads", type=int, default=1)
    parser.add_argument("--nwalkers", type=int, default=64)
    parser.add_argument("--nsteps", type=int, default=30000)
    parser.add_argument("--burnin", type=int, default=5000)
    parser.add_argument("--seed", type=int, default=20260920)
    args = parser.parse_args()
    if int(args.burnin) >= int(args.nsteps):
        raise ValueError("--burnin must be smaller than --nsteps")
    if any(path.exists() for path in (OUT_NPZ, OUT_JSON, OUT_PDF, OUT_PDF.with_suffix(".json"))):
        raise FileExistsError("immutable BAO-mask output already exists")

    started = time.perf_counter()
    cpus = set_affinity(int(args.threads))
    frozen_p = validated_frozen_p_inputs()
    stacks, measurement_metadata, measurement_hashes = load_x25()
    nbar = float(np.mean([row["nbar_h3_mpc3_from_npz"] for row in measurement_metadata]))
    centers = 0.5 * (S_EDGES[:-1] + S_EDGES[1:])
    mask = rawbox_xi_primary_mask(centers)
    kept = centers[mask]
    excluded = centers[(centers >= 50.0) & (centers < 350.0) & ~mask]
    if kept.size != 26 or not np.array_equal(excluded, [85.0, 95.0, 105.0, 115.0]):
        raise RuntimeError("canonical rawbox BAO mask changed")

    cache = build_cache(zeff=0.725, boxsize=2000.0, kmax=3.0, ells=(0, 2), cosmology="abacus_c000")
    exact = FullDiscreteRSDModel(cache, nmu=64)
    basis = realspace_basis(exact)
    evaluate_real = lambda theta: evaluate_realspace(theta, basis)
    mean_real = np.mean(np.asarray(stacks["xi0_real"], dtype="f8"), axis=0)
    mean_rsd = np.mean(np.asarray(stacks["xi0_rsd"], dtype="f8"), axis=0)

    real_covariance_initial = realspace_covariance(exact, b1=2.5, nbar=nbar)
    real_prefit, _ = fit_map(
        mean_real,
        evaluate_real,
        real_covariance_initial,
        mask,
        names=REAL_NAMES,
        bounds=(REAL_LO, REAL_HI),
        starts=REAL_STARTS,
    )
    real_covariance = realspace_covariance(exact, b1=real_prefit["theta"]["b1"], nbar=nbar)
    real_map, real_metric = fit_map(
        mean_real,
        evaluate_real,
        real_covariance,
        mask,
        names=REAL_NAMES,
        bounds=(REAL_LO, REAL_HI),
        starts=REAL_STARTS,
    )

    rsd_fast = FastRSDModel(exact, sigma_step=0.05)
    surrogate_validation = rsd_fast.validate()
    if surrogate_validation["status"] != "pass":
        raise RuntimeError(f"RSD surrogate validation failed: {surrogate_validation}")
    evaluate_rsd = lambda theta: np.asarray(rsd_fast.evaluate(theta)[0], dtype="f8")
    rsd_covariance = rsd_monopole_covariance(exact, b1=2.55, sigma_s=8.0, nbar=nbar)
    rsd_map, rsd_metric = fit_map(
        mean_rsd,
        evaluate_rsd,
        rsd_covariance,
        mask,
        names=RSD_NAMES,
        bounds=(RSD_LO, RSD_HI),
        starts=RSD_STARTS,
    )

    real_mcmc, real_chain, real_logp = run_chain(
        mean_real,
        evaluate_real,
        mask,
        real_metric,
        real_map,
        names=REAL_NAMES,
        bounds=(REAL_LO, REAL_HI),
        initial_scale=np.asarray([4.0, 0.015]),
        nwalkers=int(args.nwalkers),
        nsteps=int(args.nsteps),
        burnin=int(args.burnin),
        seed=int(args.seed),
        nworkers=int(args.threads),
    )
    rsd_mcmc, rsd_chain, rsd_logp = run_chain(
        mean_rsd,
        evaluate_rsd,
        mask,
        rsd_metric,
        rsd_map,
        names=RSD_NAMES,
        bounds=(RSD_LO, RSD_HI),
        initial_scale=np.asarray([4.0, 0.015, 0.08]),
        nwalkers=int(args.nwalkers),
        nsteps=int(args.nsteps),
        burnin=int(args.burnin),
        seed=int(args.seed) + 1,
        nworkers=int(args.threads),
    )

    real_p_flat = np.asarray(frozen_p["real_chain"], dtype="f8").reshape(-1, 2)
    rsd_p_flat = np.asarray(frozen_p["rsd_chain"], dtype="f8").reshape(-1, 4)[:, :3]
    real_p_posterior = _posterior_from_samples(real_p_flat, REAL_NAMES)
    rsd_p_posterior = frozen_p["rsd_json"]["pk0"]["mcmc"]["posterior"]
    comparisons = {
        "realspace": posterior_comparison(real_p_posterior, real_mcmc["posterior"], REAL_NAMES),
        "redshift_space": posterior_comparison(rsd_p_posterior, rsd_mcmc["posterior"], RSD_NAMES),
    }

    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "DejaVu Sans", "Liberation Sans"],
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "font.size": 11.0,
            "axes.linewidth": 1.0,
            "xtick.direction": "in",
            "ytick.direction": "in",
            "xtick.top": True,
            "ytick.right": True,
        }
    )
    OUT_PDF.parent.mkdir(parents=True, exist_ok=True)
    temporary_pdf = OUT_PDF.with_name(f".{OUT_PDF.name}.{os.getpid()}.tmp.pdf")
    with PdfPages(temporary_pdf) as pdf:
        draw_triangle_page(
            pdf,
            real_p_flat,
            real_chain.reshape(-1, 2),
            REAL_NAMES,
            title="Rawbox real space: frozen P0 vs BAO-masked xi0",
            fnl_maximum_likelihood={
                "p0": float(frozen_p["real_json"]["map"]["fNL"]),
                "xi0_masked": float(real_map["theta"]["fNL"]),
            },
        )
        draw_triangle_page(
            pdf,
            rsd_p_flat,
            rsd_chain.reshape(-1, 3),
            RSD_NAMES,
            title="Rawbox redshift space: frozen P0 vs BAO-masked xi0",
            fnl_maximum_likelihood={
                "p0": float(frozen_p["rsd_json"]["pk0"]["nominal"]["theta"]["fNL"]),
                "xi0_masked": float(rsd_map["theta"]["fNL"]),
            },
        )
        draw_residual_page(
            pdf,
            centers,
            mask,
            mean_real,
            mean_rsd,
            np.asarray(real_map["prediction_full"], dtype="f8"),
            np.asarray(rsd_map["prediction_full"], dtype="f8"),
            real_covariance,
            rsd_covariance,
            {"real": real_map, "rsd": rsd_map},
        )
    temporary_pdf.replace(OUT_PDF)
    if OUT_PDF.read_bytes()[:5] != b"%PDF-":
        raise RuntimeError("output plot is not a PDF")

    atomic_savez(
        OUT_NPZ,
        s_centers=centers,
        s_edges=S_EDGES,
        primary_mask=mask,
        selected_indices=np.flatnonzero(mask),
        excluded_bao_centers=excluded,
        xi0_real=np.asarray(stacks["xi0_real"], dtype="f8"),
        xi0_rsd=np.asarray(stacks["xi0_rsd"], dtype="f8"),
        xi0_real_mean=mean_real,
        xi0_rsd_mean=mean_rsd,
        real_covariance_single=real_covariance,
        rsd_covariance_single=rsd_covariance,
        real_prediction_map=np.asarray(real_map["prediction_full"], dtype="f8"),
        rsd_prediction_map=np.asarray(rsd_map["prediction_full"], dtype="f8"),
        real_chain_by_step=real_chain,
        real_log_probability_by_step=real_logp,
        rsd_chain_by_step=rsd_chain,
        rsd_log_probability_by_step=rsd_logp,
    )

    chain_gates = {**{f"real_{key}": value for key, value in real_mcmc["gates"].items()}, **{f"rsd_{key}": value for key, value in rsd_mcmc["gates"].items()}}
    scientific_gates = {
        "canonical_mask_has_26_bins": bool(np.count_nonzero(mask) == 26),
        "canonical_mask_excludes_85_to_115": bool(np.array_equal(excluded, [85.0, 95.0, 105.0, 115.0])),
        "realspace_mean_pte_above_0p05": bool(real_map["pte_mean"] > 0.05),
        "redshift_space_mean_pte_above_0p05": bool(rsd_map["pte_mean"] > 0.05),
        **chain_gates,
    }
    status = "pass" if all(scientific_gates.values()) else "validation_failed"
    payload = {
        "task": "task43_run_rawbox_baomask_mcmc_v1",
        "status": status,
        "classification": "standard rawbox xi0 fit with BAO-region exclusion; frozen P0 comparison",
        "mask_policy": {
            "full_range_mpc_h": [50.0, 350.0],
            "excluded_half_open_range_mpc_h": [80.0, 120.0],
            "selected_centers_mpc_h": kept.tolist(),
            "excluded_centers_mpc_h": excluded.tolist(),
            "ndata": int(np.count_nonzero(mask)),
            "same_mask_for_real_and_rsd": True,
            "covariance_selection": "same indices on the data, model and full covariance matrix",
        },
        "frozen_P_contract": {
            "statement": "P0 data, models, chains and fit bins are unchanged and are hash-validated inputs",
            "inputs_sha256": frozen_p["hashes"],
        },
        "inputs": {
            "xi_measurement_sha256": measurement_hashes,
            "theory_cache": str(cache),
            "theory_cache_sha256": sha256_file(cache),
            "nphase": NPHASE,
            "nbar_mean_h3_mpc3": nbar,
        },
        "model": {
            "realspace": "frozen finite-box linear PNG density model; no FoG",
            "redshift_space": "FullDiscrete shell-averaged Kaiser x squared-Lorentzian FoG",
            "rsd_surrogate_validation": surrogate_validation,
        },
        "covariance": {
            "realspace": "analytic periodic Gaussian C_single, frozen at the preliminary BAO-mask b1 and then refit",
            "realspace_prefit": real_prefit,
            "redshift_space": "strict unfloored analytic periodic Gaussian xi0 C_single at b1=2.55, sigma_s=8",
            "validation": "GaussianMetric checks symmetry and positive definiteness in correlation units without eigenvalue flooring",
            "posterior": "C_single",
            "mean_goodness_of_fit": "C_mean=C_single/25",
        },
        "mcmc": {
            "nwalkers": int(args.nwalkers),
            "nsteps": int(args.nsteps),
            "burnin": int(args.burnin),
            "seed_real": int(args.seed),
            "seed_rsd": int(args.seed) + 1,
        },
        "realspace": {"map": real_map, "mcmc": real_mcmc},
        "redshift_space": {"map": rsd_map, "mcmc": rsd_mcmc},
        "comparisons": comparisons,
        "gates": scientific_gates,
        "output_npz": str(OUT_NPZ),
        "output_npz_sha256": sha256_file(OUT_NPZ),
        "output_pdf": str(OUT_PDF),
        "output_pdf_sha256": sha256_file(OUT_PDF),
        "code_sha256": sha256_file(Path(__file__)),
        "cpu_affinity": cpus,
        "elapsed_sec": float(time.perf_counter() - started),
    }
    atomic_write_json(OUT_JSON, payload)
    atomic_write_json(
        OUT_PDF.with_suffix(".json"),
        {
            "task": "task43_plot_rawbox_baomask_mcmc_v1",
            "status": "pass",
            "pages": ["real-space contours", "redshift-space contours", "xi0 residuals and excluded bins"],
            "input_npz": str(OUT_NPZ),
            "input_npz_sha256": sha256_file(OUT_NPZ),
            "output_pdf": str(OUT_PDF),
            "output_pdf_sha256": sha256_file(OUT_PDF),
        },
    )
    print(
        json.dumps(
            {
                "status": status,
                "realspace": {"map": real_map["theta"], "posterior": real_mcmc["posterior"]},
                "redshift_space": {"map": rsd_map["theta"], "posterior": rsd_mcmc["posterior"]},
                "comparisons": comparisons,
                "gates": scientific_gates,
                "output": str(OUT_JSON),
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
