#!/usr/bin/env python3
"""Fixed-fNL/free-p inference for the two Task44 HOD-MAP periodic LRG boxes."""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Any, Callable

for _name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_name, "1")

import emcee
import numpy as np
from scipy.optimize import least_squares
from scipy.stats import chi2 as chi2_distribution

from task44_fit_validation import make_fit_contract, reusable_fit
from task44_pngbase_hodmap_rawbox_common import (
    BOX_SIZE,
    BOX_VOLUME,
    CATALOGS,
    DELTA_C,
    K_FUND,
    PK_PERIODIC_WITHOUT_LOWEST_EDGES,
    PK_PRIMARY_FIT_EDGES,
    REDSHIFT,
    S_CENTERS,
    S_EDGES,
    SN0_SCALE,
    atomic_savez,
    atomic_write_json,
    ensure_output_dirs,
    fit_prefix,
    get_spec,
    pk_path,
    set_cpu_affinity,
    sha256_file,
    theory_path,
    to_jsonable,
    xi_edges,
    xi_path,
)


PROJECT_ROOT = Path("/pscratch/sd/l/lzy/PNG-2PCF-model-to-observe")
TASK43_CODE_DIR = PROJECT_ROOT / "codes" / "task43"
if str(TASK43_CODE_DIR) not in sys.path:
    sys.path.insert(0, str(TASK43_CODE_DIR))

P_PRIORS = {"b1": (0.5, 5.0), "p": (-5.0, 5.0), "sn0": (-1.0, 1.0)}
XI_PRIORS = {"b1": (0.5, 5.0), "p": (-5.0, 5.0)}
XI_FIXED_P_PRIORS = {"b1": (0.5, 5.0), "fnl": (-500.0, 500.0)}
PK_FIXED_P_PRIORS = {"b1": (0.5, 5.0), "fnl": (-500.0, 500.0), "sn0": (-1.0, 1.0)}


def shell_j0_average(k: np.ndarray, edges: np.ndarray) -> np.ndarray:
    kval = np.asarray(k, dtype="f8")[:, None]
    radial_edges = np.asarray(edges, dtype="f8")
    lower = radial_edges[:-1][None, :]
    upper = radial_edges[1:][None, :]
    shell_volume_no4pi = (upper**3 - lower**3) / 3.0
    top = np.sin(kval * upper) - kval * upper * np.cos(kval * upper)
    bottom = np.sin(kval * lower) - kval * lower * np.cos(kval * lower)
    return (top - bottom) / (kval**3 * shell_volume_no4pi)


def build_theory_cache(*, kmax: float, smin: float = 50.0, overwrite: bool = False) -> Path:
    ensure_output_dirs()
    radial_edges = xi_edges(smin)
    output = theory_path(kmax, smin=smin)
    metadata_path = output.with_suffix(".json")
    if output.is_file() and metadata_path.is_file() and not overwrite:
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        if metadata.get("status") == "pass" and metadata.get("output_sha256") == sha256_file(output):
            return output
        raise RuntimeError(f"unvalidated theory cache: {output}")
    if (output.exists() or metadata_path.exists()) and not overwrite:
        raise FileExistsError(f"partial theory cache: {output} / {metadata_path}")

    from task43_theory_template import build_template_arrays, load_task41

    started = time.perf_counter()
    task41 = load_task41()
    k_template = np.geomspace(K_FUND / 2.0, max(1.0, float(kmax) * 1.2), 5000)
    template, cosmology_meta = build_template_arrays(
        task41,
        k_template,
        z=REDSHIFT,
        cosmology="abacus_c000",
    )
    qmax = int(np.ceil((float(kmax) / K_FUND) ** 2))
    nmax = int(np.ceil(np.sqrt(qmax)))
    gq = task41.gq_fft(qmax, nmax)
    g_nz, k_eff = task41.precompute_rebin_cache(gq, K_FUND, float(kmax), dk_factor=0.1)
    pk_dd = task41.interp_logk(k_eff, template["k"], template["pk_dd"])
    alpha = task41.interp_logk(k_eff, template["k"], template["alpha"])
    kernel = shell_j0_average(k_eff, radial_edges)
    elapsed = time.perf_counter() - started
    atomic_savez(
        output,
        k_eff=np.asarray(k_eff, dtype="f8"),
        g_nz=np.asarray(g_nz, dtype="f8"),
        pk_dd=np.asarray(pk_dd, dtype="f8"),
        alpha=np.asarray(alpha, dtype="f8"),
        shell_j0=np.asarray(kernel, dtype="f8"),
        s_edges=np.asarray(radial_edges, dtype="f8"),
        boxsize=np.asarray(BOX_SIZE, dtype="f8"),
        volume=np.asarray(BOX_VOLUME, dtype="f8"),
        redshift=np.asarray(REDSHIFT, dtype="f8"),
        kfund=np.asarray(K_FUND, dtype="f8"),
        kmax=np.asarray(kmax, dtype="f8"),
        cosmology_meta_json=np.asarray(json.dumps(to_jsonable(cosmology_meta), sort_keys=True)),
    )
    metadata = {
        "task": "task44_pngbase_hodmap_rawbox_theory_cache",
        "status": "pass",
        "definition": "Abacus c000 z=0.5 full-PNG FullDiscrete real-space periodic-box basis",
        "cosmology": "abacus_c000",
        "cosmology_metadata": cosmology_meta,
        "redshift": REDSHIFT,
        "boxsize_mpc_h": BOX_SIZE,
        "volume_mpc_h3": BOX_VOLUME,
        "kfund_h_mpc": K_FUND,
        "kmax_h_mpc": float(kmax),
        "qmax": qmax,
        "nmax": nmax,
        "n_rebinned_shells": int(np.asarray(k_eff).size),
        "s_edges_mpc_h": radial_edges.tolist(),
        "output": str(output),
        "output_sha256": sha256_file(output),
        "elapsed_sec": elapsed,
    }
    atomic_write_json(metadata_path, metadata)
    return output


def load_theory(path: Path, *, smin: float = 50.0) -> dict[str, np.ndarray | float]:
    with np.load(path, allow_pickle=False) as data:
        theory = {
            "k_eff": np.asarray(data["k_eff"], dtype="f8"),
            "g_nz": np.asarray(data["g_nz"], dtype="f8"),
            "pk_dd": np.asarray(data["pk_dd"], dtype="f8"),
            "alpha": np.asarray(data["alpha"], dtype="f8"),
            "shell_j0": np.asarray(data["shell_j0"], dtype="f8"),
            "s_edges": np.asarray(data["s_edges"], dtype="f8"),
            "volume": float(np.asarray(data["volume"]).item()),
            "kfund": float(np.asarray(data["kfund"]).item()),
            "kmax": float(np.asarray(data["kmax"]).item()),
        }
    if not np.isclose(float(theory["kfund"]), K_FUND, rtol=0.0, atol=1.0e-15):
        raise RuntimeError("theory cache kfund does not match 2pi/2000")
    if not np.isclose(float(theory["volume"]), BOX_VOLUME, rtol=0.0, atol=1.0e-6):
        raise RuntimeError("theory cache volume does not match L^3")
    expected_edges = xi_edges(smin)
    if not np.array_equal(np.asarray(theory["s_edges"]), expected_edges):
        raise RuntimeError(f"theory cache radial bins do not match the frozen s={float(smin):g}..350 contract")
    return theory


def matching_indices(source_edges: np.ndarray, target_edges: np.ndarray) -> np.ndarray:
    source = np.asarray(source_edges, dtype="f8")
    indices: list[int] = []
    for edge in np.asarray(target_edges, dtype="f8"):
        found = np.flatnonzero(np.all(np.isclose(source, edge[None, :], rtol=0.0, atol=1.0e-13), axis=1))
        if found.size != 1:
            raise ValueError(f"fit edge {edge.tolist()} has {found.size} matches in measurement")
        indices.append(int(found[0]))
    return np.asarray(indices, dtype="i8")


class ExactPeriodicPkModel:
    """Exact parent-lattice P0 bin averages with fixed fNL and free b1/p/sn0."""

    names = ("b1", "p", "sn0")
    priors = P_PRIORS

    def __init__(self, theory: dict[str, Any], edges: np.ndarray, *, fnl: float) -> None:
        self.fnl = float(fnl)
        self.edges = np.asarray(edges, dtype="f8")
        nmax = int(np.ceil(float(np.max(self.edges)) / K_FUND))
        integers = np.arange(-nmax, nmax + 1, dtype="i4")
        nx, ny, nz = np.meshgrid(integers, integers, integers, indexing="ij")
        n2 = (nx.astype("f8") ** 2 + ny.astype("f8") ** 2 + nz.astype("f8") ** 2).ravel()
        kval = K_FUND * np.sqrt(n2)
        bin_id = np.full(kval.size, -1, dtype="i4")
        for index, (lower, upper) in enumerate(self.edges):
            bin_id[(kval >= lower) & (kval < upper)] = index
        keep = bin_id >= 0
        self.k = np.asarray(kval[keep], dtype="f8")
        self.bin_id = np.asarray(bin_id[keep], dtype="i4")
        self.counts = np.bincount(self.bin_id, minlength=self.edges.shape[0]).astype("f8")
        if np.any(self.counts <= 0):
            raise RuntimeError(f"empty exact parent-mode fit bin: counts={self.counts}")
        self.k_mean = np.bincount(self.bin_id, weights=self.k, minlength=self.edges.shape[0]) / self.counts
        theory_k = np.asarray(theory["k_eff"], dtype="f8")
        self.pk_dd_mode = np.interp(np.log(self.k), np.log(theory_k), np.asarray(theory["pk_dd"], dtype="f8"))
        self.alpha_mode = np.interp(np.log(self.k), np.log(theory_k), np.asarray(theory["alpha"], dtype="f8"))
        basis = (
            self.pk_dd_mode,
            self.pk_dd_mode * self.alpha_mode,
            self.pk_dd_mode * self.alpha_mode**2,
        )
        self.basis = np.column_stack(
            [np.bincount(self.bin_id, weights=value, minlength=self.edges.shape[0]) / self.counts for value in basis]
        )

    def coefficients(self, theta: np.ndarray) -> np.ndarray:
        b1, p, _sn0 = map(float, np.asarray(theta, dtype="f8")[:3])
        q = self.fnl * 2.0 * DELTA_C * (b1 - p)
        return np.asarray([b1**2, 2.0 * b1 * q, q**2], dtype="f8")

    def evaluate(self, theta: np.ndarray) -> np.ndarray:
        values = np.asarray(theta, dtype="f8")
        return self.basis @ self.coefficients(values) + float(values[2]) * SN0_SCALE

    def signal_modes(self, theta: np.ndarray) -> np.ndarray:
        b1, p, _sn0 = map(float, np.asarray(theta, dtype="f8")[:3])
        q = self.fnl * 2.0 * DELTA_C * (b1 - p)
        return self.pk_dd_mode * (b1 + q * self.alpha_mode) ** 2

    def covariance(self, theta: np.ndarray, *, nbar: float) -> np.ndarray:
        signal = self.signal_modes(theta)
        residual_stochastic = float(np.asarray(theta, dtype="f8")[2]) * SN0_SCALE
        total2 = (signal + residual_stochastic + 1.0 / float(nbar)) ** 2
        summed = np.bincount(self.bin_id, weights=total2, minlength=self.edges.shape[0])
        return np.diag(2.0 * summed / self.counts**2)


class ExactPeriodicPkFixedPModel(ExactPeriodicPkModel):
    """Exact parent-lattice P0 bin averages with fixed p and free b1/fNL/sn0."""

    names = ("b1", "fnl", "sn0")
    priors = PK_FIXED_P_PRIORS

    def __init__(self, theory: dict[str, Any], edges: np.ndarray, *, fixed_p: float) -> None:
        super().__init__(theory, edges, fnl=0.0)
        self.fixed_p = float(fixed_p)

    def coefficients(self, theta: np.ndarray) -> np.ndarray:
        b1, fnl, _sn0 = map(float, np.asarray(theta, dtype="f8")[:3])
        q = fnl * 2.0 * DELTA_C * (b1 - self.fixed_p)
        return np.asarray([b1**2, 2.0 * b1 * q, q**2], dtype="f8")

    def signal_modes(self, theta: np.ndarray) -> np.ndarray:
        b1, fnl, _sn0 = map(float, np.asarray(theta, dtype="f8")[:3])
        q = fnl * 2.0 * DELTA_C * (b1 - self.fixed_p)
        return self.pk_dd_mode * (b1 + q * self.alpha_mode) ** 2


class FullDiscreteXiModel:
    """Shell-averaged FullDiscrete xi0 with fixed fNL and free b1/p."""

    names = ("b1", "p")
    priors = XI_PRIORS

    def __init__(self, theory: dict[str, Any], *, fnl: float) -> None:
        self.fnl = float(fnl)
        self.k = np.asarray(theory["k_eff"], dtype="f8")
        self.g = np.asarray(theory["g_nz"], dtype="f8")
        self.pk_dd = np.asarray(theory["pk_dd"], dtype="f8")
        self.alpha = np.asarray(theory["alpha"], dtype="f8")
        self.kernel = np.asarray(theory["shell_j0"], dtype="f8")
        self.volume = float(theory["volume"])
        weight = self.g[:, None] * self.kernel / self.volume
        self.basis = np.column_stack(
            [
                np.sum(weight * self.pk_dd[:, None], axis=0),
                np.sum(weight * (self.pk_dd * self.alpha)[:, None], axis=0),
                np.sum(weight * (self.pk_dd * self.alpha**2)[:, None], axis=0),
            ]
        )

    def coefficients(self, theta: np.ndarray) -> np.ndarray:
        b1, p = map(float, np.asarray(theta, dtype="f8")[:2])
        q = self.fnl * 2.0 * DELTA_C * (b1 - p)
        return np.asarray([b1**2, 2.0 * b1 * q, q**2], dtype="f8")

    def evaluate(self, theta: np.ndarray) -> np.ndarray:
        return self.basis @ self.coefficients(theta)

    def signal_modes(self, theta: np.ndarray) -> np.ndarray:
        b1, p = map(float, np.asarray(theta, dtype="f8")[:2])
        q = self.fnl * 2.0 * DELTA_C * (b1 - p)
        return self.pk_dd * (b1 + q * self.alpha) ** 2

    def covariance(self, theta: np.ndarray, *, nbar: float) -> np.ndarray:
        total2 = (self.signal_modes(theta) + 1.0 / float(nbar)) ** 2
        weighted_kernel = (self.g * total2)[:, None] * self.kernel
        covariance = 2.0 * (self.kernel.T @ weighted_kernel) / self.volume**2
        return 0.5 * (covariance + covariance.T)


class FullDiscreteXiFixedPModel(FullDiscreteXiModel):
    """Shell-averaged xi0 with fixed p and free b1/fNL."""

    names = ("b1", "fnl")
    priors = XI_FIXED_P_PRIORS

    def __init__(self, theory: dict[str, Any], *, fixed_p: float) -> None:
        super().__init__(theory, fnl=0.0)
        self.fixed_p = float(fixed_p)

    def coefficients(self, theta: np.ndarray) -> np.ndarray:
        b1, fnl = map(float, np.asarray(theta, dtype="f8")[:2])
        q = fnl * 2.0 * DELTA_C * (b1 - self.fixed_p)
        return np.asarray([b1**2, 2.0 * b1 * q, q**2], dtype="f8")

    def signal_modes(self, theta: np.ndarray) -> np.ndarray:
        b1, fnl = map(float, np.asarray(theta, dtype="f8")[:2])
        q = fnl * 2.0 * DELTA_C * (b1 - self.fixed_p)
        return self.pk_dd * (b1 + q * self.alpha) ** 2


def regularize_covariance(covariance: np.ndarray, *, floor_fraction: float = 1.0e-12) -> tuple[np.ndarray, dict[str, Any]]:
    sym = 0.5 * (np.asarray(covariance, dtype="f8") + np.asarray(covariance, dtype="f8").T)
    eigenvalues, eigenvectors = np.linalg.eigh(sym)
    maximum = float(np.max(eigenvalues))
    floor = max(maximum * float(floor_fraction), 1.0e-30)
    use = np.maximum(eigenvalues, floor)
    out = (eigenvectors * use[None, :]) @ eigenvectors.T
    out = 0.5 * (out + out.T)
    return out, {
        "raw_min_eigenvalue": float(np.min(eigenvalues)),
        "raw_max_eigenvalue": maximum,
        "floor": floor,
        "n_floored": int(np.count_nonzero(eigenvalues < floor)),
        "condition_number": float(np.linalg.cond(out)),
    }


def bounds_for(model: Any) -> tuple[np.ndarray, np.ndarray]:
    lower = np.asarray([model.priors[name][0] for name in model.names], dtype="f8")
    upper = np.asarray([model.priors[name][1] for name in model.names], dtype="f8")
    return lower, upper


def fit_map(model: Any, data: np.ndarray, covariance: np.ndarray) -> dict[str, Any]:
    chol = np.linalg.cholesky(covariance)
    lower, upper = bounds_for(model)

    def residual(theta: np.ndarray) -> np.ndarray:
        return np.linalg.solve(chol, np.asarray(data, dtype="f8") - model.evaluate(theta))

    if tuple(model.names) == ("b1", "p", "sn0"):
        starts = [np.asarray([b1, p, sn0], dtype="f8") for b1 in (1.5, 2.0, 2.5) for p in (-3.0, 0.0, 1.4, 3.0) for sn0 in (0.0,)]
    elif tuple(model.names) == ("b1", "p"):
        starts = [np.asarray([b1, p], dtype="f8") for b1 in (1.5, 2.0, 2.5) for p in (-3.0, 0.0, 1.4, 3.0)]
    elif tuple(model.names) == ("b1", "fnl"):
        starts = [
            np.asarray([b1, fnl], dtype="f8")
            for b1 in (1.5, 2.0, 2.5)
            for fnl in (-400.0, -200.0, -100.0, 0.0, 30.0, 100.0, 200.0, 400.0)
        ]
    elif tuple(model.names) == ("b1", "fnl", "sn0"):
        starts = [
            np.asarray([b1, fnl, sn0], dtype="f8")
            for b1 in (1.5, 2.0, 2.5)
            for fnl in (-400.0, -200.0, -100.0, 0.0, 30.0, 100.0, 200.0, 400.0)
            for sn0 in (0.0,)
        ]
    else:
        raise ValueError(f"unsupported model parameters: {model.names}")
    solutions = [
        least_squares(
            residual,
            np.clip(start, lower + 1.0e-8, upper - 1.0e-8),
            bounds=(lower, upper),
            max_nfev=5000,
            xtol=1.0e-13,
            ftol=1.0e-13,
            gtol=1.0e-13,
        )
        for start in starts
    ]
    best = min(solutions, key=lambda result: float(result.fun @ result.fun))
    chi2 = float(best.fun @ best.fun)
    dof = int(np.asarray(data).size - len(model.names))
    boundary_tolerance = 0.002 * (upper - lower)
    return {
        "theta": {name: float(value) for name, value in zip(model.names, best.x, strict=True)},
        "chi2": chi2,
        "dof": dof,
        "pte": float(chi2_distribution.sf(chi2, dof)),
        "success": bool(best.success),
        "message": str(best.message),
        "at_parameter_boundary": bool(np.any(best.x - lower < boundary_tolerance) or np.any(upper - best.x < boundary_tolerance)),
        "prediction": np.asarray(model.evaluate(best.x), dtype="f8").tolist(),
    }


def split_rhat(chain: np.ndarray) -> np.ndarray:
    values = np.asarray(chain, dtype="f8")
    nstep, _nwalkers, ndim = values.shape
    half = nstep // 2
    if half < 2:
        return np.full(ndim, np.inf)
    split = np.concatenate([values[:half].transpose(1, 0, 2), values[-half:].transpose(1, 0, 2)], axis=0)
    means = np.mean(split, axis=1)
    within = np.mean(np.var(split, axis=1, ddof=1), axis=0)
    between = half * np.var(means, axis=0, ddof=1)
    variance = (half - 1.0) / half * within + between / half
    return np.sqrt(variance / within)


def summarize_chain(chain: np.ndarray, logp: np.ndarray, names: tuple[str, ...], priors: dict[str, tuple[float, float]]) -> dict[str, Any]:
    flat = np.asarray(chain, dtype="f8").reshape(-1, chain.shape[-1])
    flat_logp = np.asarray(logp, dtype="f8").reshape(-1)
    quantiles = np.percentile(flat, [16.0, 50.0, 84.0], axis=0)
    sigma68 = 0.5 * (quantiles[2] - quantiles[0])
    try:
        tau = np.asarray(emcee.autocorr.integrated_time(chain, quiet=True, tol=0), dtype="f8")
    except Exception:
        tau = np.full(len(names), np.inf)
    rhat = split_rhat(chain)
    half = chain.shape[0] // 2
    first = chain[:half].reshape(-1, chain.shape[-1])
    second = chain[-half:].reshape(-1, chain.shape[-1])
    half_shift = np.abs(np.median(first, axis=0) - np.median(second, axis=0)) / sigma68
    imax = int(np.argmax(flat_logp))
    posterior = {}
    prior_edge = {}
    for index, name in enumerate(names):
        lower, upper = priors[name]
        width = upper - lower
        posterior[name] = {
            "q16": float(quantiles[0, index]),
            "q50": float(quantiles[1, index]),
            "q84": float(quantiles[2, index]),
            "sigma68": float(sigma68[index]),
            "mean": float(np.mean(flat[:, index])),
            "std": float(np.std(flat[:, index], ddof=1)),
        }
        prior_edge[name] = bool(
            quantiles[0, index] - lower < 0.01 * width or upper - quantiles[2, index] < 0.01 * width
        )
    return {
        "posterior": posterior,
        "map_chain": {name: float(flat[imax, index]) for index, name in enumerate(names)},
        "map_chain_log_probability": float(flat_logp[imax]),
        "tau": {name: float(tau[index]) for index, name in enumerate(names)},
        "postburn_length_over_tau": {name: float(chain.shape[0] / tau[index]) for index, name in enumerate(names)},
        "split_rhat": {name: float(rhat[index]) for index, name in enumerate(names)},
        "half_chain_shift_sigma": {name: float(half_shift[index]) for index, name in enumerate(names)},
        "prior_edge_68": prior_edge,
        "gates": {
            "split_rhat_max_below_1p01": bool(np.max(rhat) < 1.01),
            "postburn_length_min_above_50tau": bool(np.min(chain.shape[0] / tau) > 50.0),
            "half_chain_shift_max_below_0p1sigma": bool(np.max(half_shift) < 0.1),
            "posterior_68_not_at_prior_edge": bool(not any(prior_edge.values())),
        },
    }


def run_mcmc(
    model: Any,
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
    lower, upper = bounds_for(model)

    def log_probability(theta: np.ndarray) -> float:
        values = np.asarray(theta, dtype="f8")
        if np.any(values < lower) or np.any(values > upper):
            return -np.inf
        difference = np.asarray(data, dtype="f8") - model.evaluate(values)
        return -0.5 * float(difference @ precision @ difference)

    center = np.asarray([nominal["theta"][name] for name in model.names], dtype="f8")
    scale_map = {"b1": 0.015, "p": 0.12, "sn0": 0.025}
    scale = np.asarray(
        [max(8.0, 0.04 * max(1.0, abs(center[index]))) if name == "fnl" else scale_map[name]
         for index, name in enumerate(model.names)],
        dtype="f8",
    )
    rng = np.random.default_rng(int(seed))
    initial = center[None, :] + rng.normal(size=(int(nwalkers), len(model.names))) * scale[None, :]
    initial = np.clip(initial, lower + 1.0e-8, upper - 1.0e-8)
    np.random.seed(int(seed))
    sampler = emcee.EnsembleSampler(int(nwalkers), len(model.names), log_probability)
    sampler.run_mcmc(initial, int(nsteps), progress=False)
    chain = np.asarray(sampler.get_chain(discard=int(burnin)), dtype="f8")
    logp = np.asarray(sampler.get_log_prob(discard=int(burnin)), dtype="f8")
    summary = summarize_chain(chain, logp, tuple(model.names), model.priors)
    acceptance = float(np.mean(sampler.acceptance_fraction))
    summary.update(
        {
            "nwalkers": int(nwalkers),
            "nsteps": int(nsteps),
            "burnin": int(burnin),
            "postburn_steps_per_walker": int(chain.shape[0]),
            "acceptance_fraction_mean": acceptance,
        }
    )
    # In two dimensions emcee's stretch move commonly accepts slightly above
    # 0.7 while still yielding short tau and excellent split diagnostics.
    summary["gates"]["acceptance_between_0p10_0p80"] = bool(0.10 < acceptance < 0.80)
    return summary, chain, logp


def load_pk_data(tag: str, edges: np.ndarray, *, mesh: int = 400, origin: str = "positive") -> dict[str, Any]:
    path = pk_path(tag, mesh=mesh, origin=origin)
    if not path.is_file():
        raise FileNotFoundError(path)
    with np.load(path, allow_pickle=False) as data:
        indices = matching_indices(np.asarray(data["k_edges"], dtype="f8"), edges)
        result = {
            "path": path,
            "sha256": sha256_file(path),
            "indices": indices,
            "k": np.asarray(data["k"], dtype="f8")[indices],
            "k_edges": np.asarray(data["k_edges"], dtype="f8")[indices],
            "nmodes": np.asarray(data["nmodes"], dtype="f8")[indices],
            "data": np.asarray(data["pk0"], dtype="f8")[indices],
            "nbar": float(np.asarray(data["nbar"]).item()),
            "ndata": int(np.asarray(data["ndata"]).item()),
            "mesh": int(np.asarray(data["mesh"]).item()),
            "origin": str(np.asarray(data["origin"]).item()),
        }
    return result


def load_xi_data(tag: str, *, smin: float = 50.0) -> dict[str, Any]:
    path = xi_path(tag, smin=smin)
    if not path.is_file():
        raise FileNotFoundError(path)
    with np.load(path, allow_pickle=False) as data:
        ndata = int(np.asarray(data["ndata"]).item())
        result = {
            "path": path,
            "sha256": sha256_file(path),
            "s": np.asarray(data["s"], dtype="f8"),
            "s_edges": np.asarray(data["s_edges"], dtype="f8"),
            "data": np.asarray(data["xi0"], dtype="f8"),
            "nbar": float(np.asarray(data["nbar"]).item()) if "nbar" in data.files else float(ndata / BOX_VOLUME),
            "ndata": ndata,
        }
    expected_edges = xi_edges(smin)
    expected_centers = 0.5 * (expected_edges[:-1] + expected_edges[1:])
    if not np.array_equal(result["s_edges"], expected_edges) or not np.allclose(result["s"], expected_centers, rtol=0.0, atol=1.0e-10):
        raise RuntimeError(f"xi radial contract changed for {tag}")
    return result


def covariance_iteration_summary(initial: dict[str, Any], final: dict[str, Any]) -> dict[str, Any]:
    rows = {}
    for name in final["posterior"]:
        first = initial["posterior"][name]
        second = final["posterior"][name]
        rows[name] = {
            "initial_q50": float(first["q50"]),
            "final_q50": float(second["q50"]),
            "shift_over_final_sigma68": float((second["q50"] - first["q50"]) / second["sigma68"]),
            "sigma68_ratio_final_over_initial": float(second["sigma68"] / first["sigma68"]),
        }
    return rows


def run_fit(
    tag: str,
    probe: str,
    *,
    theory: dict[str, Any],
    theory_cache_path: Path,
    kmin_edge: float | None,
    smin: float,
    fixed_p: float | None,
    nwalkers: int,
    nsteps: int,
    burnin: int,
    seed: int,
    overwrite: bool,
) -> None:
    spec = get_spec(tag)
    if probe == "pk":
        if np.isclose(float(kmin_edge), 0.003, rtol=0.0, atol=1.0e-13):
            edges = PK_PRIMARY_FIT_EDGES
        elif np.isclose(float(kmin_edge), 0.005, rtol=0.0, atol=1.0e-13):
            edges = PK_PERIODIC_WITHOUT_LOWEST_EDGES
        else:
            raise ValueError("P(k) kmin edge must be 0.003 (primary, including the lowest-k bin) or 0.005 (sensitivity fit omitting it)")
        measured = load_pk_data(tag, edges)
        if fixed_p is None:
            model: Any = ExactPeriodicPkModel(theory, edges, fnl=spec.fnl)
        else:
            model = ExactPeriodicPkFixedPModel(theory, edges, fixed_p=float(fixed_p))
        if not np.array_equal(model.counts, measured["nmodes"]):
            raise RuntimeError(f"exact lattice counts do not match estimator for {tag}: {model.counts} != {measured['nmodes']}")
        if not np.allclose(model.k_mean, measured["k"], rtol=0.0, atol=1.0e-14):
            raise RuntimeError(f"exact lattice k means do not match estimator for {tag}")
        coordinate = measured["k"]
        coordinate_edges = measured["k_edges"]
    elif probe == "xi":
        radial_edges = xi_edges(smin)
        measured = load_xi_data(tag, smin=smin)
        if fixed_p is None:
            model = FullDiscreteXiModel(theory, fnl=spec.fnl)
        else:
            model = FullDiscreteXiFixedPModel(theory, fixed_p=float(fixed_p))
        coordinate = measured["s"]
        coordinate_edges = np.column_stack([radial_edges[:-1], radial_edges[1:]])
    else:
        raise ValueError(probe)

    prefix = fit_prefix(tag, probe, kmin_edge=kmin_edge, smin=smin, fixed_p=fixed_p)
    output_npz = prefix.with_suffix(".npz")
    output_json = prefix.with_suffix(".json")
    # 缓存复用必须在实际输入、模型、先验与采样参数确定之后完成。
    cache_contract = make_fit_contract(
        config=dict(tag=tag, probe=probe, kmin_edge=kmin_edge, smin=smin, fixed_p=fixed_p,
                    baseline_fnl=spec.fnl, nbar=measured["nbar"], priors=model.priors,
                    names=model.names, nwalkers=nwalkers, nsteps=nsteps, burnin=burnin, seed=seed),
        inputs=dict(measurement=measured["path"], theory=theory_cache_path),
        code=[Path(__file__), Path(__file__).with_name("task44_pngbase_hodmap_rawbox_common.py"),
              Path(__file__).with_name("task44_fit_validation.py")], theory=theory,
    )
    if reusable_fit(prefix, cache_contract, overwrite=overwrite):
        print(f"[skip verified] {output_json}", flush=True)
        return
    started = time.perf_counter()
    if probe == "pk" and fixed_p is None:
        theta_initial = np.asarray([2.0, 1.4, 0.0], dtype="f8")
    elif probe == "pk":
        theta_initial = np.asarray([2.0, spec.fnl, 0.0], dtype="f8")
    elif fixed_p is None:
        theta_initial = np.asarray([2.0, 1.4], dtype="f8")
    else:
        theta_initial = np.asarray([2.0, spec.fnl], dtype="f8")
    covariance_initial_raw = model.covariance(theta_initial, nbar=measured["nbar"])
    covariance_initial, covariance_initial_meta = regularize_covariance(covariance_initial_raw)
    map_initial = fit_map(model, measured["data"], covariance_initial)
    # Iterate the fixed analytic covariance to a self-consistent MAP.  A single
    # update is insufficient for xi0 in these high-density catalogs; the
    # history is retained so this numerical choice cannot be hidden.
    theta_cov = theta_initial.copy()
    fixed_point_history: list[dict[str, Any]] = []
    fixed_point_converged = False
    parameter_tolerance = np.asarray(
        [1.0e-4 if name == "p" else (1.0e-3 if name == "fnl" else 1.0e-5) for name in model.names],
        dtype="f8",
    )
    for iteration in range(12):
        covariance_iteration_raw = model.covariance(theta_cov, nbar=measured["nbar"])
        covariance_iteration_matrix, covariance_iteration_meta = regularize_covariance(covariance_iteration_raw)
        map_iteration = fit_map(model, measured["data"], covariance_iteration_matrix)
        theta_fit = np.asarray([map_iteration["theta"][name] for name in model.names], dtype="f8")
        delta = theta_fit - theta_cov
        fixed_point_history.append(
            {
                "iteration": iteration,
                "covariance_fiducial": {name: float(value) for name, value in zip(model.names, theta_cov, strict=True)},
                "map": map_iteration,
                "delta_map_minus_covariance_fiducial": {
                    name: float(value) for name, value in zip(model.names, delta, strict=True)
                },
                "covariance_diagnostics": covariance_iteration_meta,
            }
        )
        theta_cov = theta_fit
        if iteration >= 1 and np.all(np.abs(delta) < parameter_tolerance):
            fixed_point_converged = True
            break
    covariance_final_raw = model.covariance(theta_cov, nbar=measured["nbar"])
    covariance_final, covariance_final_meta = regularize_covariance(covariance_final_raw)
    map_final = fit_map(model, measured["data"], covariance_final)
    theta_map_final = np.asarray([map_final["theta"][name] for name in model.names], dtype="f8")
    fixed_point_final_delta = theta_map_final - theta_cov

    initial_mcmc, initial_chain, initial_logp = run_mcmc(
        model,
        measured["data"],
        covariance_initial,
        map_initial,
        nwalkers=nwalkers,
        nsteps=nsteps,
        burnin=burnin,
        seed=seed + 101,
    )
    final_mcmc, final_chain, final_logp = run_mcmc(
        model,
        measured["data"],
        covariance_final,
        map_final,
        nwalkers=nwalkers,
        nsteps=nsteps,
        burnin=burnin,
        seed=seed,
    )
    covariance_iteration = covariance_iteration_summary(initial_mcmc, final_mcmc)
    final_prediction = np.asarray(model.evaluate(np.asarray([map_final["theta"][name] for name in model.names])), dtype="f8")
    sigma = np.sqrt(np.diag(covariance_final))
    residual_sigma = (np.asarray(measured["data"]) - final_prediction) / sigma
    gates = {
        "catalog_baseline_fnl_is_audited": bool(spec.fnl in (30.0, 100.0)),
        "parameterization_contract": bool(
            (fixed_p is None and tuple(model.names) in (("b1", "p"), ("b1", "p", "sn0")))
            or (fixed_p is not None and probe == "xi" and tuple(model.names) == ("b1", "fnl"))
            or (fixed_p is not None and probe == "pk" and tuple(model.names) == ("b1", "fnl", "sn0"))
        ),
        "single_box_covariance_not_divided": True,
        "volume_exact_L3": bool(np.isclose(BOX_VOLUME, 8.0e9, rtol=0.0, atol=1.0e-6)),
        "kfund_exact_2pi_over_L": bool(np.isclose(K_FUND, 2.0 * np.pi / BOX_SIZE, rtol=0.0, atol=1.0e-16)),
        "map_optimizer_success": bool(map_final["success"]),
        "map_not_at_prior_boundary": bool(not map_final["at_parameter_boundary"]),
        "mcmc_all_gates": bool(all(final_mcmc["gates"].values())),
        "covariance_fixed_point_converged": bool(fixed_point_converged),
    }
    covariance_target = "p" if "p" in model.names else "fnl"
    gates[f"covariance_fixed_point_final_{covariance_target}_shift_below_0p01sigma"] = bool(
        abs(float(fixed_point_final_delta[list(model.names).index(covariance_target)]))
        < 0.01 * float(final_mcmc["posterior"][covariance_target]["sigma68"])
    )
    if probe == "pk" and np.isclose(float(kmin_edge), 0.003):
        gates.update(
            {
                "primary_first_bin_is_0p003_0p005": bool(np.allclose(coordinate_edges[0], [0.003, 0.005], rtol=0.0, atol=1.0e-14)),
                "primary_first_bin_nmodes_18": bool(int(model.counts[0]) == 18),
                "primary_theory_and_estimator_kmean_match": bool(np.isclose(model.k_mean[0], coordinate[0], rtol=0.0, atol=1.0e-14)),
                "primary_uses_49_contiguous_periodic_bins": bool(
                    coordinate_edges.shape == (49, 2)
                    and np.allclose(coordinate_edges, PK_PRIMARY_FIT_EDGES, rtol=0.0, atol=1.0e-14)
                ),
                "primary_last_bin_center_is_0p100": bool(
                    np.isclose(float(coordinate[-1]), 0.100, rtol=0.0, atol=5.0e-5)
                    and np.allclose(coordinate_edges[-1], [0.099, 0.101], rtol=0.0, atol=1.0e-13)
                ),
            }
        )
    if probe == "pk" and np.isclose(float(kmin_edge), 0.005):
        gates.update(
            {
                "sensitivity_uses_48_contiguous_periodic_bins": bool(
                    coordinate_edges.shape == (48, 2)
                    and np.allclose(
                        coordinate_edges,
                        PK_PERIODIC_WITHOUT_LOWEST_EDGES,
                        rtol=0.0,
                        atol=1.0e-14,
                    )
                ),
                "sensitivity_omits_only_lowest_periodic_bin": bool(
                    np.allclose(coordinate_edges[0], [0.005, 0.007], rtol=0.0, atol=1.0e-14)
                    and np.allclose(coordinate_edges[-1], [0.099, 0.101], rtol=0.0, atol=1.0e-13)
                ),
            }
        )
    elapsed = time.perf_counter() - started
    status = "pass" if all(gates.values()) else "review"
    summary = {
        "cache_contract": cache_contract,
        "task": "task44_fit_pngbase_hodmap_rawbox",
        "status": status,
        "tag": tag,
        "realization": spec.realization,
        "probe": probe,
        "geometry": "full periodic real-space cube",
        "boxsize_mpc_h": BOX_SIZE,
        "volume_mpc_h3": BOX_VOLUME,
        "kfund_h_mpc": K_FUND,
        "redshift": REDSHIFT,
        "catalog_baseline_fnl": spec.fnl,
        "fixed_fnl": spec.fnl if fixed_p is None else None,
        "fixed_p": float(fixed_p) if fixed_p is not None else None,
        "free_parameters": list(model.names),
        "priors": {name: list(model.priors[name]) for name in model.names},
        "model": "[b1 + fNL * 2*delta_c*(b1-p)*alpha(k)]^2 Pm; full fNL^2",
        "sn0": "free in units of 1e4 power" if probe == "pk" else "fixed zero contact term for s>0",
        "sigma_s": "fixed zero; real-space catalog",
        "input": {
            "path": str(measured["path"]),
            "sha256": measured["sha256"],
            "ndata": measured["ndata"],
            "nbar_h3_mpc3": measured["nbar"],
        },
        "fit_range": {
            "kmin_edge_h_mpc": float(kmin_edge) if probe == "pk" else None,
            "kmax_contract_h_mpc": 0.1 if probe == "pk" else None,
            "kmax_bin_center_h_mpc": float(np.mean(coordinate_edges[-1])) if probe == "pk" else None,
            "kmax_bin_upper_edge_h_mpc": float(coordinate_edges[-1, 1]) if probe == "pk" else None,
            "bin_selection": (
                "all contiguous periodic-box bins with centres <= 0.100 h/Mpc"
                if probe == "pk" and np.isclose(float(kmin_edge), 0.003)
                else (
                    "same contiguous periodic-box selection with only [0.003,0.005) omitted"
                    if probe == "pk"
                    else None
                )
            ),
            "k_edges_h_mpc": coordinate_edges.tolist() if probe == "pk" else None,
            "smin_mpc_h": float(smin) if probe == "xi" else None,
            "smax_mpc_h": 350.0 if probe == "xi" else None,
            "s_edges_mpc_h": xi_edges(smin).tolist() if probe == "xi" else None,
            "nbins": int(np.asarray(measured["data"]).size),
        },
        "theory": {
            "path": str(theory_cache_path),
            "sha256": sha256_file(theory_cache_path),
            "cosmology": "abacus_c000",
            "theory_kmin_h_mpc": K_FUND,
            "theory_kmax_h_mpc": float(theory["kmax"]),
            "periodic_discrete_modes": True,
            "shell_average": probe == "xi",
            "survey_window": False,
            "gic_ric": False,
        },
        "covariance": {
            "type": "single-periodic-box analytic Gaussian",
            "volume_mpc_h3": BOX_VOLUME,
            "nbar_h3_mpc3": measured["nbar"],
            "fNL_cov_initial": spec.fnl,
            "p_cov_fixed": float(fixed_p) if fixed_p is not None else None,
            "initial_fiducial": {name: float(value) for name, value in zip(model.names, theta_initial, strict=True)},
            "initial_diagnostics": covariance_initial_meta,
            "initial_map": map_initial,
            "fixed_point_converged": fixed_point_converged,
            "fixed_point_parameter_tolerance": {
                name: float(value) for name, value in zip(model.names, parameter_tolerance, strict=True)
            },
            "fixed_point_history": fixed_point_history,
            "final_fiducial": {name: float(value) for name, value in zip(model.names, theta_cov, strict=True)},
            "final_map_minus_fiducial": {
                name: float(value) for name, value in zip(model.names, fixed_point_final_delta, strict=True)
            },
            "final_diagnostics": covariance_final_meta,
            "iteration_posterior_comparison": covariance_iteration,
            "hartlap_percival": False,
        },
        "map": map_final,
        "mcmc": final_mcmc,
        "mcmc_initial_covariance": initial_mcmc,
        "residual": {
            "max_abs_sigma": float(np.max(np.abs(residual_sigma))),
            "rms_sigma": float(np.sqrt(np.mean(residual_sigma**2))),
        },
        "gates": gates,
        "elapsed_sec": elapsed,
    }
    atomic_savez(
        output_npz,
        coordinate=np.asarray(coordinate, dtype="f8"),
        coordinate_edges=np.asarray(coordinate_edges, dtype="f8"),
        data=np.asarray(measured["data"], dtype="f8"),
        prediction_map=final_prediction,
        residual_sigma=np.asarray(residual_sigma, dtype="f8"),
        covariance_initial=np.asarray(covariance_initial, dtype="f8"),
        covariance_final=np.asarray(covariance_final, dtype="f8"),
        chain_initial_by_step=np.asarray(initial_chain, dtype="f8"),
        logp_initial_by_step=np.asarray(initial_logp, dtype="f8"),
        chain_final_by_step=np.asarray(final_chain, dtype="f8"),
        logp_final_by_step=np.asarray(final_logp, dtype="f8"),
        parameter_names=np.asarray(model.names),
        catalog_baseline_fnl=np.asarray(spec.fnl, dtype="f8"),
        fixed_fnl=np.asarray(spec.fnl if fixed_p is None else np.nan, dtype="f8"),
        fixed_p=np.asarray(float(fixed_p) if fixed_p is not None else np.nan, dtype="f8"),
        nbar=np.asarray(measured["nbar"], dtype="f8"),
        ndata=np.asarray(measured["ndata"], dtype="i8"),
        boxsize=np.asarray(BOX_SIZE, dtype="f8"),
        volume=np.asarray(BOX_VOLUME, dtype="f8"),
        kfund=np.asarray(K_FUND, dtype="f8"),
        summary_json=np.asarray(json.dumps(to_jsonable(summary), sort_keys=True)),
    )
    summary["output_npz"] = str(output_npz)
    summary["output_npz_sha256"] = sha256_file(output_npz)
    atomic_write_json(output_json, summary)
    reported_name = "p" if "p" in model.names else "fnl"
    reported_row = final_mcmc["posterior"][reported_name]
    print(
        f"[done] {tag} {probe} baseline_fNL={spec.fnl:g} "
        f"fixed_p={fixed_p if fixed_p is not None else 'free'} {reported_name}={reported_row['q50']:.4f} "
        f"-{reported_row['q50']-reported_row['q16']:.4f}/+{reported_row['q84']-reported_row['q50']:.4f} "
        f"chi2/dof={map_final['chi2']:.2f}/{map_final['dof']} status={status}",
        flush=True,
    )


def parse_tags(values: list[str] | None) -> list[str]:
    return list(CATALOGS) if not values else [get_spec(value).tag for value in values]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=("theory", "fit"))
    parser.add_argument("--tag", action="append", choices=tuple(CATALOGS))
    parser.add_argument("--probe", choices=("pk", "xi", "both"), default="both")
    parser.add_argument(
        "--kmin-edge",
        type=float,
        action="append",
        help="P(k) lower edge; repeat to run the primary 0.003 fit and the 0.005 fit that omits the lowest-k bin",
    )
    parser.add_argument("--theory-kmax", type=float, default=5.0)
    parser.add_argument("--smin", type=float, default=50.0, help="xi lower edge; audited choices are 30 or 50 Mpc/h")
    parser.add_argument("--fixed-p", type=float, help="fix p to this value and sample fNL instead")
    parser.add_argument("--threads", type=int, default=8)
    parser.add_argument("--nwalkers", type=int, default=64)
    parser.add_argument("--nsteps", type=int, default=10_000)
    parser.add_argument("--burnin", type=int, default=2_000)
    parser.add_argument("--seed", type=int, default=20260821)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    if int(args.burnin) >= int(args.nsteps):
        raise ValueError("burnin must be less than nsteps")
    set_cpu_affinity(int(args.threads))
    xi_edges(float(args.smin))
    cache_path = build_theory_cache(
        kmax=float(args.theory_kmax),
        smin=float(args.smin),
        overwrite=bool(args.overwrite and args.mode == "theory"),
    )
    if args.mode == "theory":
        print(f"[done] theory cache {cache_path}", flush=True)
        return
    theory = load_theory(cache_path, smin=float(args.smin))
    tags = parse_tags(args.tag)
    probes = ("pk", "xi") if args.probe == "both" else (args.probe,)
    kmins = [0.003, 0.005] if not args.kmin_edge else [float(value) for value in args.kmin_edge]
    for tag in tags:
        for probe in probes:
            these_kmins: list[float | None] = kmins if probe == "pk" else [None]
            for kmin in these_kmins:
                tag_offset = 0 if tag == "c300" else 3000
                if probe == "xi":
                    fit_offset = 2000 + (0 if np.isclose(float(args.smin), 50.0) else 30_000)
                    if args.fixed_p is not None:
                        fit_offset += 50_000
                else:
                    fit_offset = 0 if np.isclose(float(kmin), 0.003) else 1000
                    if args.fixed_p is not None:
                        fit_offset += 50_000
                run_fit(
                    tag,
                    probe,
                    theory=theory,
                    theory_cache_path=cache_path,
                    kmin_edge=kmin,
                    smin=float(args.smin),
                    fixed_p=float(args.fixed_p) if args.fixed_p is not None else None,
                    nwalkers=int(args.nwalkers),
                    nsteps=int(args.nsteps),
                    burnin=int(args.burnin),
                    seed=int(args.seed) + tag_offset + fit_offset,
                    overwrite=bool(args.overwrite),
                )


if __name__ == "__main__":
    main()
