#!/usr/bin/env python3
"""Fit and compare Task 4.3.2 rawbox P0(k) and xi0(s>=50).

This is intentionally a narrow l=0-only product:

* P0 uses the frozen Task43 15-bin DESI-PNG selection through kmax=0.10;
* xi0 uses only the frozen smin=50, smax=350 selection;
* no quadrupole and no scale scan are loaded into the comparison product.
"""

from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path
from typing import Any

for _name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_name, "1")

import emcee
import numpy as np
from scipy.interpolate import CubicSpline
from scipy.optimize import least_squares
from scipy.stats import chi2 as chi2_distribution

from task43_rsd_common import (
    OUTPUT_ROOT,
    PHASES,
    PLOT_ROOT,
    P_FIXED,
    S_EDGES,
    atomic_savez,
    atomic_write_json,
    sha256_file,
)
from task43_rsd_model import DELTA_C, FullDiscreteRSDModel, build_cache


PARAMETERS = ("fNL", "b1", "sigma_s", "sn0")
BOUNDS_LO = np.asarray([-500.0, 0.5, 0.0, -1.0], dtype="f8")
BOUNDS_HI = np.asarray([500.0, 5.0, 30.0, 1.0], dtype="f8")
SN0_SCALE = 1.0e4
LIGHTCONE_FIT_EDGES = np.asarray(
    [
        [0.005, 0.007], [0.007, 0.009], [0.009, 0.011],
        [0.013, 0.015], [0.017, 0.019], [0.021, 0.023],
        [0.029, 0.031], [0.037, 0.039], [0.045, 0.047],
        [0.053, 0.055], [0.061, 0.063], [0.069, 0.071],
        [0.077, 0.079], [0.085, 0.087], [0.093, 0.095],
    ],
    dtype="f8",
)
RAWBOX_FIT_EDGES = np.vstack([np.asarray([[0.003, 0.005]], dtype="f8"), LIGHTCONE_FIT_EDGES])
CANONICAL_PK_PAYLOAD = (
    Path("/pscratch/sd/l/lzy/PNG-2PCF-model-to-observe")
    / "plots/outputs/task43_outputs/pk_lightcone/summary"
    / "task43_pk_lightcone_mmin1p4e13_x25_fkpP010000_desi_rebin_kmax0p10_payload.npz"
)
XI_PREFIX = OUTPUT_ROOT / "rawbox/closure/task43_rsd_rawbox_x25_fulldiscrete_lorentzian"


def set_affinity(threads: int) -> list[int]:
    if not 1 <= int(threads) <= 8:
        raise ValueError("--threads must be in [1, 8]")
    available = sorted(os.sched_getaffinity(0))
    selected = available[: int(threads)]
    if len(selected) != int(threads):
        raise RuntimeError(f"requested {threads} CPUs, only {len(available)} available")
    os.sched_setaffinity(0, selected)
    return selected


def measurement_path(phase: str) -> Path:
    return OUTPUT_ROOT / "rawbox/pk" / (
        f"task43_rsd_rawbox_pk0_AbacusSummit_base_c000_{phase}_mmin1p4e13_mesh400.npz"
    )


def matching_indices(source_edges: np.ndarray, target_edges: np.ndarray) -> np.ndarray:
    source = np.asarray(source_edges, dtype="f8")
    out: list[int] = []
    for edge in np.asarray(target_edges, dtype="f8"):
        found = np.flatnonzero(np.all(np.isclose(source, edge[None, :], rtol=0.0, atol=1.0e-12), axis=1))
        if found.size != 1:
            raise ValueError(f"target edge {edge.tolist()} has {found.size} matches")
        out.append(int(found[0]))
    return np.asarray(out, dtype="i8")


def fit_edges_for_kmin(kmin_edge: float) -> np.ndarray:
    if np.isclose(float(kmin_edge), 0.003, rtol=0.0, atol=1.0e-12):
        return RAWBOX_FIT_EDGES.copy()
    if np.isclose(float(kmin_edge), 0.005, rtol=0.0, atol=1.0e-12):
        return LIGHTCONE_FIT_EDGES.copy()
    raise ValueError("--pk-kmin-edge must be 0.003 (rawbox primary) or 0.005 (lightcone-matched A/B)")


def validate_canonical_edges(fit_edges: np.ndarray) -> dict[str, Any]:
    if not CANONICAL_PK_PAYLOAD.is_file():
        raise FileNotFoundError(CANONICAL_PK_PAYLOAD)
    with np.load(CANONICAL_PK_PAYLOAD, allow_pickle=False) as data:
        edges = np.asarray(data["k_edges"], dtype="f8")
        kmax = float(np.asarray(data["kmax_fit"]).item())
    if not np.allclose(edges, LIGHTCONE_FIT_EDGES, rtol=0.0, atol=1.0e-12) or kmax != 0.1:
        raise RuntimeError("frozen Task43 15-bin P0 contract changed")
    requested = np.asarray(fit_edges, dtype="f8")
    if requested.shape == LIGHTCONE_FIT_EDGES.shape:
        if not np.allclose(requested, LIGHTCONE_FIT_EDGES, rtol=0.0, atol=1.0e-12):
            raise RuntimeError("requested 15-bin edges do not match the lightcone contract")
        policy = "legacy lightcone-matched 15-bin A/B"
    elif requested.shape == RAWBOX_FIT_EDGES.shape:
        if not np.allclose(requested[1:], LIGHTCONE_FIT_EDGES, rtol=0.0, atol=1.0e-12):
            raise RuntimeError("rawbox fit does not preserve the frozen 15-bin tail")
        if not np.allclose(requested[0], [0.003, 0.005], rtol=0.0, atol=1.0e-12):
            raise RuntimeError("rawbox fit does not prepend the fundamental-mode bin")
        policy = "rawbox primary: prepend [0.003,0.005] to the frozen 15-bin tail"
    else:
        raise RuntimeError(f"unsupported fit edge shape {requested.shape}")
    return {"sha256": sha256_file(CANONICAL_PK_PAYLOAD), "policy": policy}


def load_pk_x25(fit_edges: np.ndarray) -> dict[str, Any]:
    pk_real, pk_rsd, nbar, ndata, hashes, metadata_rows = [], [], [], [], [], []
    k = k_edges = nmodes = selected = None
    for phase in PHASES:
        path = measurement_path(phase)
        metadata_path = path.with_suffix(".json")
        if not path.is_file() or not metadata_path.is_file():
            raise FileNotFoundError(f"missing P0 measurement for {phase}: {path}")
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        digest = sha256_file(path)
        if metadata.get("status") != "pass" or metadata.get("output_sha256") != digest:
            raise RuntimeError(f"unvalidated P0 measurement for {phase}: {path}")
        if metadata.get("ells") != [0] or metadata.get("position_bridge", {}).get("array_equal") is not True:
            raise RuntimeError(f"P0 l=0/real-space bridge gate failed for {phase}")
        with np.load(path, allow_pickle=False) as data:
            this_edges = np.asarray(data["k_edges"], dtype="f8")
            this_selected = matching_indices(this_edges, fit_edges)
            this_k = np.asarray(data["k"], dtype="f8")[this_selected]
            this_nmodes = np.asarray(data["nmodes"], dtype="f8")[this_selected]
            if k_edges is None:
                k_edges, selected, k, nmodes = np.asarray(fit_edges).copy(), this_selected, this_k, this_nmodes
            else:
                if not np.array_equal(this_selected, selected):
                    raise RuntimeError(f"selected fine-bin indices changed for {phase}")
                if not np.allclose(this_k, k, rtol=0.0, atol=1.0e-14):
                    raise RuntimeError(f"mode-averaged k changed for {phase}")
                if not np.array_equal(this_nmodes, nmodes):
                    raise RuntimeError(f"mode counts changed for {phase}")
            pk_real.append(np.asarray(data["pk0_real"], dtype="f8")[this_selected])
            pk_rsd.append(np.asarray(data["pk0_rsd"], dtype="f8")[this_selected])
            nbar.append(float(np.asarray(data["nbar"]).item()))
            ndata.append(int(np.asarray(data["ndata"]).item()))
        hashes.append(digest)
        metadata_rows.append(metadata)
    result = {
        "k": np.asarray(k),
        "k_edges": np.asarray(k_edges),
        "fine_indices": np.asarray(selected),
        "nmodes": np.asarray(nmodes),
        "pk0_real": np.stack(pk_real),
        "pk0_rsd": np.stack(pk_rsd),
        "nbar": np.asarray(nbar),
        "ndata": np.asarray(ndata),
        "hashes": hashes,
        "metadata": metadata_rows,
    }
    if np.any(result["nmodes"] <= 0.0) or np.any(~np.isfinite(result["pk0_rsd"])):
        raise RuntimeError("selected P0 vector is empty or non-finite")
    return result


class ExactPeriodicPk0Model:
    """Exact parent-mode P0 bin average with a sigma_s spline surrogate."""

    def __init__(self, exact_xi: FullDiscreteRSDModel, edges: np.ndarray, *, sigma_step: float = 0.05) -> None:
        self.edges = np.asarray(edges, dtype="f8")
        self.boxsize = float(exact_xi.boxsize)
        self.kfund = 2.0 * np.pi / self.boxsize
        nmax = int(np.ceil(float(np.max(self.edges)) / self.kfund))
        integers = np.arange(-nmax, nmax + 1, dtype="i4")
        nx, ny, nz = np.meshgrid(integers, integers, integers, indexing="ij")
        n2 = (nx.astype("f8") ** 2 + ny.astype("f8") ** 2 + nz.astype("f8") ** 2).ravel()
        nz = nz.ravel().astype("f8")
        kval = self.kfund * np.sqrt(n2)
        bin_id = np.full(kval.size, -1, dtype="i4")
        for ibin, (lo, hi) in enumerate(self.edges):
            bin_id[(kval >= lo) & (kval < hi)] = ibin
        keep = bin_id >= 0
        self.k = kval[keep]
        self.mu2 = np.divide(nz[keep] ** 2, n2[keep], out=np.zeros(np.count_nonzero(keep)), where=n2[keep] > 0.0)
        self.bin_id = bin_id[keep]
        self.counts = np.bincount(self.bin_id, minlength=self.edges.shape[0]).astype("f8")
        self.k_mean = np.bincount(self.bin_id, weights=self.k, minlength=self.edges.shape[0]) / self.counts
        self.pk_dd = np.interp(np.log(self.k), np.log(exact_xi.k_eff), exact_xi.pk_dd)
        self.alpha = np.interp(np.log(self.k), np.log(exact_xi.k_eff), exact_xi.alpha)
        self.f_growth = float(exact_xi.f_growth)
        self.sigma_grid = np.arange(0.0, 30.0 + 0.5 * float(sigma_step), float(sigma_step), dtype="f8")
        self.basis = np.empty((self.sigma_grid.size, self.edges.shape[0], 6), dtype="f8")
        for isig, sigma_s in enumerate(self.sigma_grid):
            damping = 1.0 / (1.0 + 0.5 * (self.k**2 * self.mu2 * sigma_s**2)) ** 2
            columns = (
                self.pk_dd * damping,
                self.pk_dd * self.alpha * damping,
                self.pk_dd * self.alpha**2 * damping,
                self.pk_dd * self.mu2 * damping,
                self.pk_dd * self.alpha * self.mu2 * damping,
                self.pk_dd * self.mu2**2 * damping,
            )
            for icolumn, values in enumerate(columns):
                self.basis[isig, :, icolumn] = (
                    np.bincount(self.bin_id, weights=values, minlength=self.edges.shape[0]) / self.counts
                )
        self.spline = CubicSpline(self.sigma_grid, self.basis, axis=0)

    def evaluate(self, theta: np.ndarray) -> np.ndarray:
        fnl, b1, sigma_s, sn0 = map(float, np.asarray(theta, dtype="f8")[:4])
        q = fnl * 2.0 * DELTA_C * (b1 - P_FIXED)
        f = self.f_growth
        coefficients = np.asarray([b1 * b1, 2.0 * b1 * q, q * q, 2.0 * b1 * f, 2.0 * q * f, f * f])
        return np.asarray(self.spline(sigma_s), dtype="f8") @ coefficients + sn0 * SN0_SCALE

    def direct(self, theta: np.ndarray) -> np.ndarray:
        fnl, b1, sigma_s, sn0 = map(float, np.asarray(theta, dtype="f8")[:4])
        q = fnl * 2.0 * DELTA_C * (b1 - P_FIXED)
        amplitude = b1 + q * self.alpha
        damping = 1.0 / (1.0 + 0.5 * self.k**2 * self.mu2 * sigma_s**2) ** 2
        signal = self.pk_dd * (amplitude + self.f_growth * self.mu2) ** 2 * damping + sn0 * SN0_SCALE
        return np.bincount(self.bin_id, weights=signal, minlength=self.edges.shape[0]) / self.counts

    def validate(self) -> dict[str, Any]:
        trials = (
            np.asarray([0.0, 2.55, 8.0, 0.0]),
            np.asarray([-75.0, 2.2, 3.37, 0.2]),
            np.asarray([80.0, 2.8, 12.43, -0.3]),
            np.asarray([15.0, 2.5, 0.07, 0.1]),
            np.asarray([-20.0, 2.6, 29.93, -0.1]),
        )
        rows = []
        for theta in trials:
            fast, direct = self.evaluate(theta), self.direct(theta)
            rows.append(
                {
                    "theta": theta.tolist(),
                    "max_abs": float(np.max(np.abs(fast - direct))),
                    "relative_l2": float(np.linalg.norm(fast - direct) / np.linalg.norm(direct)),
                }
            )
        maximum = max(row["relative_l2"] for row in rows)
        return {"status": "pass" if maximum < 1.0e-8 else "fail", "max_relative_l2": maximum, "trials": rows}

    def gaussian_covariance(self, *, b1: float, sigma_s: float, nbar: float) -> np.ndarray:
        damping = 1.0 / (1.0 + 0.5 * self.k**2 * self.mu2 * float(sigma_s) ** 2) ** 2
        signal = self.pk_dd * (float(b1) + self.f_growth * self.mu2) ** 2 * damping
        total2 = (signal + 1.0 / float(nbar)) ** 2
        summed = np.bincount(self.bin_id, weights=total2, minlength=self.edges.shape[0])
        variance = 2.0 * summed / self.counts**2
        return np.diag(variance)


def split_rhat(chain: np.ndarray) -> np.ndarray:
    values = np.asarray(chain, dtype="f8")
    nstep, _, ndim = values.shape
    half = nstep // 2
    if half < 2:
        return np.full(ndim, np.inf)
    split = np.concatenate([values[:half].transpose(1, 0, 2), values[-half:].transpose(1, 0, 2)], axis=0)
    means = np.mean(split, axis=1)
    within = np.mean(np.var(split, axis=1, ddof=1), axis=0)
    between = half * np.var(means, axis=0, ddof=1)
    var_hat = (half - 1.0) / half * within + between / half
    return np.sqrt(var_hat / within)


def summarize_chain(chain: np.ndarray, logp: np.ndarray) -> dict[str, Any]:
    flat = np.asarray(chain, dtype="f8").reshape(-1, chain.shape[-1])
    flat_logp = np.asarray(logp, dtype="f8").reshape(-1)
    quantiles = np.percentile(flat, [16.0, 50.0, 84.0], axis=0)
    sigma68 = 0.5 * (quantiles[2] - quantiles[0])
    try:
        tau = np.asarray(emcee.autocorr.integrated_time(chain, quiet=True, tol=0), dtype="f8")
    except Exception:
        tau = np.full(chain.shape[-1], np.inf)
    rhat = split_rhat(chain)
    half = chain.shape[0] // 2
    first, second = chain[:half].reshape(-1, chain.shape[-1]), chain[-half:].reshape(-1, chain.shape[-1])
    half_shift = np.abs(np.median(first, axis=0) - np.median(second, axis=0)) / sigma68
    imax = int(np.argmax(flat_logp))
    return {
        "posterior": {
            name: {
                "q16": float(quantiles[0, index]),
                "q50": float(quantiles[1, index]),
                "q84": float(quantiles[2, index]),
                "sigma68": float(sigma68[index]),
                "mean": float(np.mean(flat[:, index])),
                "std": float(np.std(flat[:, index], ddof=1)),
            }
            for index, name in enumerate(PARAMETERS)
        },
        "map_chain": {name: float(flat[imax, index]) for index, name in enumerate(PARAMETERS)},
        "map_chain_log_probability": float(flat_logp[imax]),
        "tau": {name: float(tau[index]) for index, name in enumerate(PARAMETERS)},
        "postburn_length_over_tau": {name: float(chain.shape[0] / tau[index]) for index, name in enumerate(PARAMETERS)},
        "split_rhat": {name: float(rhat[index]) for index, name in enumerate(PARAMETERS)},
        "half_chain_shift_sigma": {name: float(half_shift[index]) for index, name in enumerate(PARAMETERS)},
        "gates": {
            "split_rhat_max_below_1p01": bool(np.max(rhat) < 1.01),
            "postburn_length_min_above_50tau": bool(np.min(chain.shape[0] / tau) > 50.0),
            "half_chain_shift_max_below_0p1sigma": bool(np.max(half_shift) < 0.1),
        },
    }


def fit_map(model: ExactPeriodicPk0Model, data: np.ndarray, covariance: np.ndarray) -> dict[str, Any]:
    chol = np.linalg.cholesky(covariance)

    def residual(theta: np.ndarray) -> np.ndarray:
        return np.linalg.solve(chol, np.asarray(data) - model.evaluate(theta))

    starts = (
        np.asarray([0.0, 2.55, 8.0, 0.0]),
        np.asarray([-80.0, 2.4, 4.0, 0.2]),
        np.asarray([80.0, 2.7, 12.0, -0.2]),
        np.asarray([0.0, 2.5, 15.0, 0.5]),
    )
    solutions = [
        least_squares(
            residual,
            start,
            bounds=(BOUNDS_LO, BOUNDS_HI),
            max_nfev=3000,
            xtol=1.0e-12,
            ftol=1.0e-12,
            gtol=1.0e-12,
        )
        for start in starts
    ]
    best = min(solutions, key=lambda result: float(result.fun @ result.fun))
    chi2_single = float(best.fun @ best.fun)
    dof = int(np.asarray(data).size - len(PARAMETERS))
    return {
        "theta": {name: float(value) for name, value in zip(PARAMETERS, best.x, strict=True)},
        "chi2_single_covariance": chi2_single,
        "chi2_mean_covariance": float(len(PHASES) * chi2_single),
        "dof": dof,
        "pte_mean_covariance": float(chi2_distribution.sf(len(PHASES) * chi2_single, dof)),
        "success": bool(best.success),
        "message": str(best.message),
        "at_parameter_boundary": bool(
            np.any(np.isclose(best.x, BOUNDS_LO, atol=[1.0, 0.01, 0.01, 0.01]))
            or np.any(np.isclose(best.x, BOUNDS_HI, atol=[1.0, 0.01, 0.01, 0.01]))
        ),
        "prediction": model.evaluate(best.x).tolist(),
    }


def run_mcmc(
    model: ExactPeriodicPk0Model,
    data: np.ndarray,
    covariance: np.ndarray,
    nominal: dict[str, Any],
    *,
    nwalkers: int,
    nsteps: int,
    burnin: int,
    seed: int,
) -> tuple[dict[str, Any], np.ndarray, np.ndarray]:
    precision = np.linalg.inv(covariance)

    def log_probability(theta: np.ndarray) -> float:
        values = np.asarray(theta, dtype="f8")
        if np.any(values < BOUNDS_LO) or np.any(values > BOUNDS_HI):
            return -np.inf
        diff = np.asarray(data) - model.evaluate(values)
        return -0.5 * float(diff @ precision @ diff)

    center = np.asarray([nominal["theta"][name] for name in PARAMETERS], dtype="f8")
    rng = np.random.default_rng(int(seed))
    scale = np.asarray([4.0, 0.015, 0.12, 0.025], dtype="f8")
    initial = center[None, :] + rng.normal(size=(int(nwalkers), len(PARAMETERS))) * scale[None, :]
    initial = np.clip(initial, BOUNDS_LO + 1.0e-7, BOUNDS_HI - 1.0e-7)
    sampler = emcee.EnsembleSampler(int(nwalkers), len(PARAMETERS), log_probability)
    sampler.run_mcmc(initial, int(nsteps), progress=False)
    chain = np.asarray(sampler.get_chain(discard=int(burnin)), dtype="f8")
    logp = np.asarray(sampler.get_log_prob(discard=int(burnin)), dtype="f8")
    summary = summarize_chain(chain, logp)
    summary.update(
        {
            "nwalkers": int(nwalkers),
            "nsteps": int(nsteps),
            "burnin": int(burnin),
            "postburn_steps_per_walker": int(chain.shape[0]),
            "acceptance_fraction_mean": float(np.mean(sampler.acceptance_fraction)),
        }
    )
    return summary, chain, logp


def load_xi0_only() -> dict[str, Any]:
    json_path, npz_path = XI_PREFIX.with_suffix(".json"), XI_PREFIX.with_suffix(".npz")
    if not json_path.is_file() or not npz_path.is_file():
        raise FileNotFoundError(f"missing xi0 closure: {json_path} / {npz_path}")
    summary = json.loads(json_path.read_text(encoding="utf-8"))
    nominal = summary["nominal"]
    if nominal["ells"] != [0] or float(nominal["smin_mpc_h"]) != 50.0:
        raise RuntimeError("xi0 nominal is not the frozen l=0, smin=50 result")
    centers = 0.5 * (S_EDGES[:-1] + S_EDGES[1:])
    mask = centers >= 50.0
    with np.load(npz_path, allow_pickle=False) as data:
        chain = np.asarray(data["chain_by_step"], dtype="f8")
        covariance_full = np.asarray(data["covariance_single_realization"], dtype="f8")
        mean_full = np.asarray(data["xi0_rsd_mean"], dtype="f8")
        prediction_full = np.asarray(data["nominal_prediction_xi0"], dtype="f8")
    nbin = centers.size
    covariance = covariance_full[:nbin, :nbin][np.ix_(mask, mask)]
    return {
        "json_path": json_path,
        "npz_path": npz_path,
        "json_sha256": sha256_file(json_path),
        "npz_sha256": sha256_file(npz_path),
        "s": centers[mask],
        "mean": mean_full[mask],
        "prediction": prediction_full[mask],
        "covariance": covariance,
        "chain": chain,
        "posterior": summary["mcmc_nominal"]["posterior"],
        "nominal": nominal,
        "convergence": summary["mcmc_nominal"]["gates"],
    }


def make_plot(
    path: Path,
    pk: dict[str, Any],
    pk_mean: np.ndarray,
    pk_prediction: np.ndarray,
    pk_covariance: np.ndarray,
    pk_chain: np.ndarray,
    xi: dict[str, Any],
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.backends.backend_pdf import PdfPages

    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp.pdf")
    with PdfPages(temporary) as pdf:
        figure, axes = plt.subplots(2, 2, figsize=(11.2, 7.6), sharex="col", gridspec_kw={"height_ratios": [2.2, 1.0]})
        pk_sigma_mean = np.sqrt(np.diag(pk_covariance) / len(PHASES))
        axes[0, 0].errorbar(pk["k"], pk_mean, yerr=pk_sigma_mean, fmt="o", ms=4, color="#2F2F2F", label=r"RSD rawbox $P_0$, x25 mean")
        axes[0, 0].plot(pk["k"], pk_prediction, color="#2F2F2F", lw=1.6, ls="--", label="best fit")
        axes[0, 0].set(xscale="log", yscale="log", ylabel=r"$P_0(k)\ [(h^{-1}{\rm Mpc})^3]$")
        axes[0, 0].legend(frameon=False, fontsize=9)
        axes[1, 0].axhline(0.0, color="0.5", lw=0.8)
        axes[1, 0].plot(pk["k"], (pk_mean - pk_prediction) / pk_sigma_mean, "o-", color="#2F2F2F", ms=4, lw=0.8)
        axes[1, 0].set(xscale="log", xlabel=r"$k\ [h\,{\rm Mpc}^{-1}]$", ylabel=r"residual / $\sigma_{\rm mean}$")

        s = np.asarray(xi["s"])
        xi_sigma_mean = np.sqrt(np.diag(xi["covariance"]) / len(PHASES))
        axes[0, 1].errorbar(s, s**2 * xi["mean"], yerr=s**2 * xi_sigma_mean, fmt="o", ms=4, color="#C44E52", label=r"RSD rawbox $\xi_0$, x25 mean")
        axes[0, 1].plot(s, s**2 * xi["prediction"], color="#C44E52", lw=1.6, ls="--", label="best fit")
        axes[0, 1].set(ylabel=r"$s^2\xi_0(s)\ [(h^{-1}{\rm Mpc})^2]$")
        axes[0, 1].legend(frameon=False, fontsize=9)
        axes[1, 1].axhline(0.0, color="0.5", lw=0.8)
        axes[1, 1].plot(s, (xi["mean"] - xi["prediction"]) / xi_sigma_mean, "o-", color="#C44E52", ms=4, lw=0.8)
        axes[1, 1].set(xlabel=r"$s\ [h^{-1}{\rm Mpc}]$", ylabel=r"residual / $\sigma_{\rm mean}$")
        figure.tight_layout()
        pdf.savefig(figure)
        plt.close(figure)

        figure, axes = plt.subplots(1, 3, figsize=(11.4, 3.5))
        xi_flat = np.asarray(xi["chain"], dtype="f8").reshape(-1, 3)
        pk_flat = np.asarray(pk_chain, dtype="f8").reshape(-1, 4)
        ranges = {"fNL": (-80.0, 80.0), "b1": (2.3, 2.8), "sigma_s": (0.0, 15.0)}
        labels = {"fNL": r"$f_{\rm NL}$", "b1": r"$b_1$", "sigma_s": r"$\sigma_s\ [h^{-1}{\rm Mpc}]$"}
        for index, name in enumerate(("fNL", "b1", "sigma_s")):
            lo, hi = ranges[name]
            bins = np.linspace(lo, hi, 70)
            axes[index].hist(pk_flat[:, index], bins=bins, density=True, histtype="step", lw=1.8, color="#2F2F2F", label=r"$P_0(k)$")
            axes[index].hist(xi_flat[:, index], bins=bins, density=True, histtype="step", lw=1.8, color="#C44E52", label=r"$\xi_0(s)$")
            if name == "fNL":
                axes[index].axvline(0.0, color="0.5", lw=0.8, ls="--")
            axes[index].set(xlabel=labels[name], ylabel="posterior density")
        axes[0].legend(frameon=False)
        figure.tight_layout()
        pdf.savefig(figure)
        plt.close(figure)
    temporary.replace(path)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--threads", type=int, default=8)
    parser.add_argument("--b1-cov", type=float, default=2.55)
    parser.add_argument("--sigma-s-cov", type=float, default=8.0)
    parser.add_argument("--sigma-grid-step", type=float, default=0.05)
    parser.add_argument("--nwalkers", type=int, default=64)
    parser.add_argument("--nsteps", type=int, default=12000)
    parser.add_argument("--burnin", type=int, default=3000)
    parser.add_argument("--seed", type=int, default=430350)
    parser.add_argument(
        "--pk-kmin-edge",
        type=float,
        default=0.003,
        help="P0 first bin lower edge; 0.003 is the rawbox primary and 0.005 reproduces the lightcone-matched A/B.",
    )
    parser.add_argument(
        "--output-prefix",
        type=Path,
        default=OUTPUT_ROOT / "rawbox/comparison/task43_rsd_rawbox_x25_pk0_vs_xi0_smin50_l0only",
    )
    parser.add_argument(
        "--plot",
        type=Path,
        default=PLOT_ROOT / "task43_rsd_rawbox_x25_pk0_vs_xi0_smin50_l0only.pdf",
    )
    args = parser.parse_args()
    if int(args.burnin) >= int(args.nsteps):
        raise ValueError("--burnin must be less than --nsteps")
    output_npz, output_json = args.output_prefix.with_suffix(".npz"), args.output_prefix.with_suffix(".json")
    if any(path.exists() for path in (output_npz, output_json, args.plot)):
        raise FileExistsError("immutable l=0 comparison output already exists")
    started = time.perf_counter()
    cpus = set_affinity(int(args.threads))
    fit_edges = fit_edges_for_kmin(float(args.pk_kmin_edge))
    canonical_validation = validate_canonical_edges(fit_edges)
    pk = load_pk_x25(fit_edges)
    xi = load_xi0_only()
    pk_mean = np.mean(pk["pk0_rsd"], axis=0)
    empirical_covariance = np.cov(pk["pk0_rsd"], rowvar=False, ddof=1)

    theory_cache = build_cache(zeff=0.725, boxsize=2000.0, kmax=3.0, ells=(0, 2), cosmology="abacus_c000")
    exact_xi = FullDiscreteRSDModel(theory_cache, nmu=64)
    model = ExactPeriodicPk0Model(exact_xi, fit_edges, sigma_step=float(args.sigma_grid_step))
    surrogate_validation = model.validate()
    if surrogate_validation["status"] != "pass":
        raise RuntimeError(f"P0 surrogate failed: {surrogate_validation}")
    if not np.array_equal(model.counts, pk["nmodes"]):
        raise RuntimeError(f"exact lattice counts {model.counts} != estimator nmodes {pk['nmodes']}")
    if not np.allclose(model.k_mean, pk["k"], rtol=0.0, atol=1.0e-14):
        raise RuntimeError("exact lattice k means do not match estimator coordinates")
    covariance = model.gaussian_covariance(
        b1=float(args.b1_cov), sigma_s=float(args.sigma_s_cov), nbar=float(np.mean(pk["nbar"]))
    )
    nominal = fit_map(model, pk_mean, covariance)
    mcmc, chain, logp = run_mcmc(
        model,
        pk_mean,
        covariance,
        nominal,
        nwalkers=int(args.nwalkers),
        nsteps=int(args.nsteps),
        burnin=int(args.burnin),
        seed=int(args.seed),
    )
    pk_prediction = np.asarray(nominal["prediction"], dtype="f8")
    analytic_sigma = np.sqrt(np.diag(covariance))
    empirical_sigma = np.sqrt(np.diag(empirical_covariance))
    correlation = empirical_covariance / np.outer(empirical_sigma, empirical_sigma)
    covariance_diagnostic = {
        "sample_std_over_gaussian_sigma": (empirical_sigma / analytic_sigma).tolist(),
        "sample_std_over_gaussian_sigma_median": float(np.median(empirical_sigma / analytic_sigma)),
        "sample_correlation_offdiagonal_absmax": float(np.max(np.abs(correlation - np.eye(correlation.shape[0])))),
    }
    comparison = {}
    for name in ("fNL", "b1", "sigma_s"):
        p0_row, xi_row = mcmc["posterior"][name], xi["posterior"][name]
        denominator = np.hypot(float(p0_row["sigma68"]), float(xi_row["sigma68"]))
        comparison[name] = {
            "p0_median": float(p0_row["q50"]),
            "p0_sigma68": float(p0_row["sigma68"]),
            "xi0_median": float(xi_row["q50"]),
            "xi0_sigma68": float(xi_row["sigma68"]),
            "center_difference_p0_minus_xi0": float(p0_row["q50"] - xi_row["q50"]),
            "difference_over_independent_quadrature_sigma_diagnostic": float((p0_row["q50"] - xi_row["q50"]) / denominator),
        }

    make_plot(args.plot, pk, pk_mean, pk_prediction, covariance, chain, xi)
    atomic_savez(
        output_npz,
        phases=np.asarray(PHASES),
        k=pk["k"],
        k_edges=pk["k_edges"],
        fine_bin_indices=pk["fine_indices"],
        nmodes=pk["nmodes"],
        pk0_real=pk["pk0_real"],
        pk0_rsd=pk["pk0_rsd"],
        pk0_rsd_mean=pk_mean,
        pk0_model_map=pk_prediction,
        pk0_covariance_single=covariance,
        pk0_covariance_mean=covariance / len(PHASES),
        pk0_empirical_covariance=empirical_covariance,
        pk0_chain_by_step=chain,
        pk0_log_probability_by_step=logp,
        s=xi["s"],
        xi0_rsd_mean=xi["mean"],
        xi0_model_map=xi["prediction"],
        xi0_covariance_single=xi["covariance"],
        xi0_chain_by_step=xi["chain"],
    )
    payload = {
        "task": "task43_fit_rsd_rawbox_pk0_vs_xi0_smin50",
        "status": "complete" if all(mcmc["gates"].values()) else "mcmc_diagnostic_failed",
        "scope": {
            "geometry": "periodic rawbox, z=0.725, plane-parallel LOS=z",
            "multipoles": [0],
            "explicitly_excluded": ["ell=2", "smin scan", "lightcone"],
            "pk0": (
                f"{fit_edges.shape[0]} bins; first edge=[{fit_edges[0, 0]:.3f},{fit_edges[0, 1]:.3f}], "
                "last edge=[0.093,0.095], kmax_fit=0.10 h/Mpc"
            ),
            "pk0_kmin_edge_h_mpc": float(fit_edges[0, 0]),
            "pk0_first_mode_averaged_k_h_mpc": float(pk["k"][0]),
            "pk0_first_bin_nmodes": int(pk["nmodes"][0]),
            "xi0": "s bin edges 50..350 Mpc/h, 30 bins; smin=50 only",
        },
        "cpu_affinity": cpus,
        "phases": list(PHASES),
        "nphase": len(PHASES),
        "measurement_paths": [str(measurement_path(phase)) for phase in PHASES],
        "measurement_sha256": pk["hashes"],
        "canonical_pk_bin_payload": str(CANONICAL_PK_PAYLOAD),
        "canonical_pk_bin_payload_sha256": canonical_validation["sha256"],
        "pk_fit_bin_policy": canonical_validation["policy"],
        "model": {
            "name": "exact periodic-mode-bin average Kaiser x squared-Lorentzian FoG PNG P0",
            "theory_cache": str(theory_cache),
            "p_fixed": P_FIXED,
            "free_parameters": list(PARAMETERS),
            "sn0_scale": SN0_SCALE,
            "surrogate_validation": surrogate_validation,
            "mode_count_gate": "exact equality with jaxpower nmodes",
            "mode_mean_k_gate_max_abs": float(np.max(np.abs(model.k_mean - pk["k"]))),
        },
        "covariance": {
            "definition": "exact periodic-mode diagonal Gaussian P0 covariance including Poisson 1/nbar",
            "fiducial": {"fNL": 0.0, "b1": float(args.b1_cov), "sigma_s": float(args.sigma_s_cov), "sn0": 0.0},
            "quoted_posterior": "single-realization covariance; never divided by 25",
            "mean_fit_quality": "Cmean=Csingle/25",
            "x25_scatter_diagnostic": covariance_diagnostic,
        },
        "pk0": {"nominal": nominal, "mcmc": mcmc},
        "xi0": {
            "source_json": str(xi["json_path"]),
            "source_json_sha256": xi["json_sha256"],
            "source_npz": str(xi["npz_path"]),
            "source_npz_sha256": xi["npz_sha256"],
            "nominal": xi["nominal"],
            "posterior": xi["posterior"],
            "convergence": xi["convergence"],
            "extraction_policy": "only the already-frozen ell=0, smin=50 branch is loaded",
        },
        "comparison": comparison,
        "output_npz": str(output_npz),
        "output_plot_pdf": str(args.plot),
        "elapsed_sec": time.perf_counter() - started,
    }
    atomic_write_json(output_json, payload)
    print(json.dumps({"status": payload["status"], "pk0_posterior": mcmc["posterior"], "output": str(output_json)}, sort_keys=True))
    if payload["status"] != "complete":
        raise SystemExit(2)


if __name__ == "__main__":
    main()
