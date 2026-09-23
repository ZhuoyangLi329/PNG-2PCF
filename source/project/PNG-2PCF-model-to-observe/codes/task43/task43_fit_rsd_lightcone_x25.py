#!/usr/bin/env python3
"""Production x25 radial-LOS closure for the Task 4.3.2 RSD model.

The quoted posterior uses the single-lightcone covariance.  The fit of the
25-phase mean is tested separately with C_mean = C_single / 25, while phase
profiles use C_single.  These covariance roles must never be interchanged.
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

from task43_build_rsd_formal_gic_window import output_path as phase_window_path
from task43_fit_rsd_rawbox_x25 import FastRSDModel
from task43_rsd_model import DELTA_C, FullDiscreteRSDModel, build_cache
from task43_rsd_common import OUTPUT_ROOT, PHASES, PLOT_ROOT, SMIN_SCAN, atomic_savez, atomic_write_json, sha256_file
from task43_summarize_rsd_lightcone_x25 import measurement_path, read_jsonl


PRIMARY_MODEL = "formal_gic"
MODELS = ("no_gic", "formal_gic")
ELL_SETS = ((0,), (0, 2))
PARAMETER_NAMES = ("fnl_loc", "b1", "sigma_s")
BOUNDS_LO = np.asarray([-500.0, 0.2, 0.0], dtype="f8")
BOUNDS_HI = np.asarray([500.0, 10.0, 30.0], dtype="f8")


def jsonable(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [jsonable(item) for item in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, Path):
        return str(value)
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    return str(value)


def set_affinity(nthreads: int) -> list[int]:
    if not 1 <= int(nthreads) <= 8:
        raise ValueError("--threads must be in 1..8")
    available = sorted(os.sched_getaffinity(0))
    selected = available[: int(nthreads)]
    if len(selected) != int(nthreads):
        raise RuntimeError(f"requested {nthreads} CPUs but only {len(available)} are available")
    os.sched_setaffinity(0, selected)
    return selected


def edge_pairs(edges: np.ndarray) -> np.ndarray:
    values = np.asarray(edges, dtype="f8")
    return values if values.ndim == 2 else np.column_stack([values[:-1], values[1:]])


def load_window(path: Path, *, expected_k: np.ndarray | None = None) -> dict[str, Any]:
    sidecar = path.with_suffix(".json")
    if not path.is_file():
        raise FileNotFoundError(path)
    with np.load(path, allow_pickle=False) as payload:
        k_eff = np.asarray(payload["k_eff"], dtype="f8")
        w2 = np.asarray(payload["w2"], dtype="f8")
        meta = json.loads(str(np.asarray(payload["meta_json"]).item()))
    if expected_k is not None and not np.array_equal(k_eff, np.asarray(expected_k, dtype="f8")):
        raise RuntimeError(f"formal-GIC k grid differs from theory: {path}")
    if w2.shape != k_eff.shape or not np.all(np.isfinite(w2)):
        raise RuntimeError(f"invalid formal-GIC window: {path}")
    if sidecar.is_file():
        side_meta = json.loads(sidecar.read_text(encoding="utf-8"))
        if side_meta.get("status") != "pass" or side_meta.get("output_sha256") != sha256_file(path):
            raise RuntimeError(f"unvalidated formal-GIC window sidecar: {path}")
        meta["task43_sidecar"] = side_meta
    return {"k_eff": k_eff, "w2": w2, "meta": meta}


def load_inputs(summary_path: Path, covariance_path: Path) -> dict[str, Any]:
    with np.load(summary_path, allow_pickle=False) as payload:
        phases = tuple(str(value) for value in np.asarray(payload["phases"]).tolist())
        if phases != PHASES:
            raise RuntimeError(f"unexpected phase order in {summary_path}: {phases}")
        data = {
            "s": np.asarray(payload["s"], dtype="f8"),
            "s_edges": edge_pairs(payload["s_edges"]),
            "xi": np.asarray(payload["xi_multipoles_by_phase"], dtype="f8"),
            "rr": np.asarray(payload["RR_by_phase"], dtype="f8"),
            "zeff": np.asarray(payload["zeff_by_phase"], dtype="f8"),
        }
    with np.load(covariance_path, allow_pickle=False) as payload:
        cov_s = np.asarray(payload["s"], dtype="f8")
        covariance = np.asarray(payload["covariance_single_realization"], dtype="f8")
        cov_ells = tuple(int(value) for value in np.asarray(payload["ells"]).ravel())
        covariance_meta = (
            json.loads(str(np.asarray(payload["meta_json"]).item()))
            if "meta_json" in payload.files and str(np.asarray(payload["meta_json"]).item())
            else {}
        )
    if cov_ells != (0, 2):
        raise RuntimeError(f"closure covariance must contain ell=(0,2), got {cov_ells}")
    if not np.array_equal(data["s"], cov_s):
        raise RuntimeError("measurement and covariance radial grids differ")
    expected = data["s"].size * len(cov_ells)
    if covariance.shape != (expected, expected):
        raise RuntimeError(f"unexpected covariance shape {covariance.shape}, expected {(expected, expected)}")
    covariance = 0.5 * (covariance + covariance.T)
    eigenvalues = np.linalg.eigvalsh(covariance)
    if not np.all(np.isfinite(covariance)) or float(eigenvalues[0]) <= 0.0:
        raise RuntimeError(f"closure covariance is not finite SPD: eigmin={eigenvalues[0]}")
    data.update(
        {
            "covariance": covariance,
            "covariance_ells": cov_ells,
            "covariance_meta": covariance_meta,
            "covariance_eigenvalues": eigenvalues,
        }
    )
    return data


class FastWindowRSDModel:
    """Exact-in-amplitudes cubic FoG surrogate plus phase-specific formal GIC."""

    def __init__(
        self,
        exact: FullDiscreteRSDModel,
        windows: dict[str, dict[str, Any]],
        *,
        sigma_step: float,
    ) -> None:
        self.exact = exact
        self.base = FastRSDModel(exact, sigma_step=float(sigma_step))
        self.window_keys = tuple(windows)
        self.windows = windows
        self.window_index = {key: index for index, key in enumerate(self.window_keys)}
        for key, window in windows.items():
            if not np.array_equal(np.asarray(window["k_eff"], dtype="f8"), exact.k_eff):
                raise RuntimeError(f"formal-GIC k grid differs from FullDiscrete grid for {key}")

        weighted_windows = np.column_stack(
            [
                np.asarray(exact.g_nz, dtype="f8")
                * np.asarray(windows[key]["w2"], dtype="f8")
                / float(exact.volume)
                for key in self.window_keys
            ]
        )
        gic_basis = np.empty(
            (self.base.sigma_grid.size, len(self.window_keys), 6), dtype="f8"
        )
        powers = (np.ones_like(exact.mu2), exact.mu2, exact.mu2 * exact.mu2)
        for isig, sigma_s in enumerate(self.base.sigma_grid):
            x = (exact.k_eff[:, None] * exact.mu[None, :] * float(sigma_s)) ** 2
            damping = 1.0 / (1.0 + 0.5 * x) ** 2
            moments = [
                0.5 * np.sum(exact.wmu[None, :] * damping * power[None, :], axis=1)
                for power in powers
            ]
            vectors = np.stack(
                [
                    exact.pk_dd * moments[0],
                    exact.pk_dd * exact.alpha * moments[0],
                    exact.pk_dd * exact.alpha**2 * moments[0],
                    exact.pk_dd * moments[1],
                    exact.pk_dd * exact.alpha * moments[1],
                    exact.pk_dd * moments[2],
                ]
            )
            gic_basis[isig] = (vectors @ weighted_windows).T
        self.gic_spline = CubicSpline(self.base.sigma_grid, gic_basis, axis=0)

    def coefficients(self, theta: np.ndarray) -> np.ndarray:
        fnl, b1, _ = map(float, np.asarray(theta, dtype="f8")[:3])
        q = fnl * 2.0 * DELTA_C * (b1 - 1.0)
        growth = float(self.exact.f_growth)
        return np.asarray(
            [b1 * b1, 2.0 * b1 * q, q * q, 2.0 * b1 * growth, 2.0 * q * growth, growth * growth],
            dtype="f8",
        )

    def gic_value(self, theta: np.ndarray, window_key: str) -> float:
        index = self.window_index[str(window_key)]
        sigma_s = float(np.asarray(theta, dtype="f8")[2])
        basis = np.asarray(self.gic_spline(sigma_s), dtype="f8")[index]
        return float(self.coefficients(theta) @ basis)

    def evaluate(
        self,
        theta: np.ndarray,
        *,
        model: str,
        window_key: str,
    ) -> dict[int, np.ndarray]:
        values = {ell: np.asarray(value, dtype="f8").copy() for ell, value in self.base.evaluate(theta).items()}
        if str(model) == "no_gic":
            return values
        if str(model) != "formal_gic":
            raise ValueError(f"unknown model {model!r}")
        values[0] -= self.gic_value(theta, str(window_key))
        return values

    def exact_pk0(self, theta: np.ndarray) -> np.ndarray:
        fnl, b1, sigma_s = map(float, np.asarray(theta, dtype="f8")[:3])
        q = fnl * 2.0 * DELTA_C * (b1 - 1.0)
        amplitude = b1 + q * self.exact.alpha
        x = (self.exact.k_eff[:, None] * self.exact.mu[None, :] * sigma_s) ** 2
        damping = 1.0 / (1.0 + 0.5 * x) ** 2
        pk_mu = self.exact.pk_dd[:, None] * (
            amplitude[:, None] + float(self.exact.f_growth) * self.exact.mu2[None, :]
        ) ** 2 * damping
        return 0.5 * np.sum(self.exact.wmu[None, :] * pk_mu, axis=1)

    def validate(self) -> dict[str, Any]:
        base_validation = self.base.validate()
        trials = (
            np.asarray([0.0, 2.55, 8.0]),
            np.asarray([-75.0, 2.2, 3.37]),
            np.asarray([80.0, 2.8, 12.43]),
            np.asarray([15.0, 2.5, 0.07]),
            np.asarray([-20.0, 2.6, 29.93]),
        )
        rows: list[dict[str, Any]] = []
        for theta in trials:
            exact_base = self.exact.evaluate(
                fnl=float(theta[0]), b1=float(theta[1]), sigma_s=float(theta[2]), p_fixed=1.0
            )
            pk0 = self.exact_pk0(theta)
            for key in self.window_keys:
                exact = {ell: np.asarray(value, dtype="f8").copy() for ell, value in exact_base.items()}
                window = np.asarray(self.windows[key]["w2"], dtype="f8")
                exact_gic = float(np.sum(self.exact.g_nz * pk0 * window) / float(self.exact.volume))
                exact[0] -= exact_gic
                fast = self.evaluate(theta, model="formal_gic", window_key=key)
                exact_vector = np.concatenate([exact[ell] for ell in self.exact.ell_values])
                fast_vector = np.concatenate([fast[ell] for ell in self.exact.ell_values])
                rows.append(
                    {
                        "theta": theta.tolist(),
                        "window_key": key,
                        "gic_exact": exact_gic,
                        "gic_fast": self.gic_value(theta, key),
                        "max_abs": float(np.max(np.abs(fast_vector - exact_vector))),
                        "relative_l2": float(
                            np.linalg.norm(fast_vector - exact_vector)
                            / max(np.linalg.norm(exact_vector), 1.0e-30)
                        ),
                    }
                )
        maximum = max(row["relative_l2"] for row in rows)
        status = "pass" if base_validation["status"] == "pass" and maximum < 1.0e-7 else "fail"
        return {
            "status": status,
            "sigma_grid_step_mpc_h": float(self.base.sigma_grid[1] - self.base.sigma_grid[0]),
            "base": base_validation,
            "formal_gic_max_relative_l2": float(maximum),
            "formal_gic_trials": rows,
        }


def select_vector_and_covariance(
    inputs: dict[str, Any],
    xi_by_ell: np.ndarray,
    rr: np.ndarray,
    *,
    smin: float,
    ells: tuple[int, ...],
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    s = np.asarray(inputs["s"], dtype="f8")
    mask = s >= float(smin)
    if not np.any(mask):
        raise ValueError(f"no radial bins for smin={smin}")
    indices: list[int] = []
    vectors: list[np.ndarray] = []
    for ell in ells:
        iell = inputs["covariance_ells"].index(int(ell))
        indices.extend((iell * s.size + np.flatnonzero(mask)).tolist())
        vectors.append(np.asarray(xi_by_ell[iell], dtype="f8")[mask])
    ids = np.asarray(indices, dtype="i8")
    covariance = np.asarray(inputs["covariance"], dtype="f8")[np.ix_(ids, ids)]
    return s[mask], np.concatenate(vectors), np.asarray(rr, dtype="f8")[mask], inputs["s_edges"][mask], covariance


def fit_one(
    *,
    inputs: dict[str, Any],
    xi_by_ell: np.ndarray,
    rr: np.ndarray,
    fast_model: FastWindowRSDModel,
    window_key: str,
    model: str,
    ells: tuple[int, ...],
    smin: float,
    nwalkers: int,
    nsteps: int,
    burnin: int,
    seed: int,
    optimizer_only: bool,
) -> tuple[dict[str, Any], np.ndarray, np.ndarray, np.ndarray]:
    s, xi, rr_selected, s_edges, covariance = select_vector_and_covariance(
        inputs, xi_by_ell, rr, smin=float(smin), ells=ells
    )
    del rr_selected, s_edges
    mask = np.asarray(inputs["s"], dtype="f8") >= float(smin)
    chol = np.linalg.cholesky(covariance)
    precision = np.linalg.inv(covariance)

    def prediction(theta: np.ndarray) -> np.ndarray:
        values = fast_model.evaluate(theta, model=str(model), window_key=str(window_key))
        return np.concatenate([np.asarray(values[int(ell)], dtype="f8")[mask] for ell in ells])

    def residual(theta: np.ndarray) -> np.ndarray:
        return np.linalg.solve(chol, xi - prediction(theta))

    starts = (
        np.asarray([0.0, 2.55, 8.0], dtype="f8"),
        np.asarray([-100.0, 2.3, 4.0], dtype="f8"),
        np.asarray([100.0, 2.8, 12.0], dtype="f8"),
    )
    solutions = [
        least_squares(
            residual,
            start,
            bounds=(BOUNDS_LO, BOUNDS_HI),
            max_nfev=1500,
            xtol=1.0e-11,
            ftol=1.0e-11,
            gtol=1.0e-11,
        )
        for start in starts
    ]
    optimum = min(solutions, key=lambda item: float(item.fun @ item.fun))
    theta_map = np.asarray(optimum.x, dtype="f8")
    chi2_map = float(optimum.fun @ optimum.fun)

    def log_probability(theta: np.ndarray) -> float:
        values = np.asarray(theta, dtype="f8")
        if np.any(values < BOUNDS_LO) or np.any(values > BOUNDS_HI):
            return -np.inf
        difference = xi - prediction(values)
        return -0.5 * float(difference @ precision @ difference)

    if optimizer_only:
        chain = theta_map[None, :]
        logp = np.asarray([log_probability(theta_map)], dtype="f8")
        acceptance: float | None = None
    else:
        rng = np.random.default_rng(int(seed))
        scale = np.asarray([4.0, 0.015, 0.10], dtype="f8")
        initial = theta_map[None, :] + rng.normal(size=(int(nwalkers), 3)) * scale[None, :]
        initial = np.clip(initial, BOUNDS_LO + 1.0e-7, BOUNDS_HI - 1.0e-7)
        sampler = emcee.EnsembleSampler(int(nwalkers), 3, log_probability)
        sampler.run_mcmc(initial, int(nsteps), progress=False)
        chain = np.asarray(sampler.get_chain(discard=int(burnin), flat=True), dtype="f8")
        logp = np.asarray(sampler.get_log_prob(discard=int(burnin), flat=True), dtype="f8")
        acceptance = float(np.mean(sampler.acceptance_fraction))

    quantiles = np.percentile(chain, [16.0, 50.0, 84.0], axis=0)
    summary: dict[str, Any] = {
        "model": str(model),
        "parameter_names": list(PARAMETER_NAMES),
        "priors": {
            name: [float(BOUNDS_LO[index]), float(BOUNDS_HI[index])]
            for index, name in enumerate(PARAMETER_NAMES)
        },
        "optimizer": {
            "success": bool(optimum.success),
            "message": str(optimum.message),
            "x": theta_map.tolist(),
            "chi2": chi2_map,
            "nfev": int(optimum.nfev),
        },
        "mcmc": {
            "nwalkers": int(nwalkers),
            "nsteps": int(nsteps),
            "burnin": int(burnin),
            "seed": int(seed),
            "nsamples": int(chain.shape[0]),
            "optimizer_only": bool(optimizer_only),
            "mean_acceptance_fraction": acceptance,
        },
        "theory": {
            "space": "redshift",
            "multipoles": [int(value) for value in ells],
            "p_fixed": 1.0,
            "f_growth": float(fast_model.exact.f_growth),
            "fog_model": "lorentzian",
            "xi_kernel": "shell-averaged",
            "xi_projector": "discrete-box",
            "likelihood_acceleration": "cubic-in-sigma exact-amplitude FullDiscrete surrogate",
            "common_x25_zeff": float(np.asarray(fast_model.exact.zeff).item()),
        },
        "precision_used": precision,
        "precision": {
            "source": "numpy.linalg.inv(selected covariance_single_realization)",
            "eigenvalue_min": float(np.linalg.eigvalsh(covariance)[0]),
            "condition_number": float(np.linalg.cond(covariance)),
        },
    }
    for index, name in enumerate(PARAMETER_NAMES):
        summary[name] = {
            "q16": float(quantiles[0, index]),
            "q50": float(quantiles[1, index]),
            "q84": float(quantiles[2, index]),
            "mean": float(np.mean(chain[:, index])),
            "std": float(np.std(chain[:, index], ddof=1)) if chain.shape[0] > 1 else 0.0,
        }
    if str(model) == "formal_gic":
        summary["formal_gic"] = {
            "sigma_w2_map": fast_model.gic_value(theta_map, str(window_key)),
            "window_key": str(window_key),
            "window": fast_model.windows[str(window_key)]["meta"],
        }
    dof = int(xi.size - 3)
    summary["task43_fit_context"] = {
        "smin_mpc_h": float(smin),
        "ells": [int(value) for value in ells],
        "ndata": int(xi.size),
        "dof_nominal": dof,
        "chi2_map_single_covariance": chi2_map,
        "pte_map_single_covariance": float(chi2_distribution.sf(chi2_map, dof)),
    }
    return summary, chain, logp, prediction(theta_map)


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
    variance = (half - 1.0) / half * within + between / half
    return np.sqrt(variance / within)


def convergence_diagnostics(
    chain_flat: np.ndarray,
    *,
    names: list[str],
    nwalkers: int,
    nsteps: int,
    burnin: int,
) -> dict[str, Any]:
    post_steps = int(nsteps) - int(burnin)
    expected = post_steps * int(nwalkers)
    if chain_flat.shape != (expected, len(names)):
        raise RuntimeError(f"unexpected flattened chain shape {chain_flat.shape}, expected {(expected, len(names))}")
    chain = np.asarray(chain_flat, dtype="f8").reshape(post_steps, int(nwalkers), len(names))
    try:
        tau = np.asarray(emcee.autocorr.integrated_time(chain, quiet=True, tol=0), dtype="f8")
    except Exception:
        tau = np.full(len(names), np.inf)
    rhat = split_rhat(chain)
    quantiles = np.percentile(chain.reshape(-1, len(names)), [16.0, 50.0, 84.0], axis=0)
    sigma68 = 0.5 * (quantiles[2] - quantiles[0])
    half = post_steps // 2
    first = chain[:half].reshape(-1, len(names))
    second = chain[-half:].reshape(-1, len(names))
    half_shift = np.abs(np.median(first, axis=0) - np.median(second, axis=0)) / sigma68
    length_over_tau = post_steps / tau
    gates = {
        "split_rhat_max_below_1p01": bool(np.all(np.isfinite(rhat)) and np.max(rhat) < 1.01),
        "postburn_length_min_above_50tau": bool(np.all(np.isfinite(length_over_tau)) and np.min(length_over_tau) > 50.0),
        "half_chain_shift_max_below_0p1sigma": bool(np.all(np.isfinite(half_shift)) and np.max(half_shift) < 0.1),
    }
    return {
        "parameter_names": names,
        "tau": {name: float(tau[index]) for index, name in enumerate(names)},
        "postburn_length_over_tau": {name: float(length_over_tau[index]) for index, name in enumerate(names)},
        "split_rhat": {name: float(rhat[index]) for index, name in enumerate(names)},
        "half_chain_shift_sigma": {name: float(half_shift[index]) for index, name in enumerate(names)},
        "gates": gates,
    }


def compact_fit(summary: dict[str, Any]) -> dict[str, Any]:
    out = dict(summary)
    out.pop("precision_used", None)
    if "formal_gic" in out and "window" in out["formal_gic"]:
        window = dict(out["formal_gic"]["window"])
        sidecar = window.get("task43_sidecar")
        if isinstance(sidecar, dict):
            window["task43_sidecar"] = {
                key: sidecar.get(key)
                for key in ("task", "status", "phase", "output", "output_sha256", "nsub", "seed")
            }
        out["formal_gic"] = {**out["formal_gic"], "window": window}
    return out


def theta_from_fit(summary: dict[str, Any]) -> dict[str, float]:
    names = list(summary["parameter_names"])
    values = list(summary["optimizer"]["x"])
    return {name: float(values[index]) for index, name in enumerate(names)}


def profile_record(phase: str, model: str, summary: dict[str, Any]) -> dict[str, Any]:
    context = summary["task43_fit_context"]
    theta = theta_from_fit(summary)
    boundary = bool(
        abs(theta["fnl_loc"]) > 499.0
        or theta["b1"] < 0.21
        or theta["b1"] > 9.99
        or theta["sigma_s"] < 0.01
        or theta["sigma_s"] > 29.99
    )
    return {
        "phase": phase,
        "model": model,
        "theta": theta,
        "optimizer_success": bool(summary["optimizer"]["success"]),
        "chi2": float(context["chi2_map_single_covariance"]),
        "dof": int(context["dof_nominal"]),
        "pte": float(context["pte_map_single_covariance"]),
        "at_parameter_boundary": boundary,
    }


def covariance_scatter_diagnostics(inputs: dict[str, Any]) -> dict[str, Any]:
    xi = np.asarray(inputs["xi"], dtype="f8")
    vector = np.concatenate([xi[:, 0], xi[:, 1]], axis=1)
    empirical = np.cov(vector, rowvar=False, ddof=1)
    analytic = np.asarray(inputs["covariance"], dtype="f8")
    empirical_sigma = np.sqrt(np.diag(empirical))
    analytic_sigma = np.sqrt(np.diag(analytic))
    empirical_corr = empirical / np.outer(empirical_sigma, empirical_sigma)
    analytic_corr = analytic / np.outer(analytic_sigma, analytic_sigma)
    centered = vector - np.mean(vector, axis=0)
    precision = np.linalg.inv(analytic)
    phase_chi2 = np.einsum("ij,jk,ik->i", centered, precision, centered)
    nbin = inputs["s"].size
    return {
        "sample_std_over_cov_sigma": (empirical_sigma / analytic_sigma).tolist(),
        "sample_std_over_cov_sigma_median": float(np.median(empirical_sigma / analytic_sigma)),
        "sample_std_over_cov_sigma_xi0_median": float(np.median(empirical_sigma[:nbin] / analytic_sigma[:nbin])),
        "sample_std_over_cov_sigma_xi2_median": float(np.median(empirical_sigma[nbin:] / analytic_sigma[nbin:])),
        "empirical_vs_cov_correlation_absmax": float(np.max(np.abs(empirical_corr - analytic_corr))),
        "empirical_vs_cov_correlation_frobenius": float(np.linalg.norm(empirical_corr - analytic_corr)),
        "phase_chi2_about_empirical_mean_full_xi02": phase_chi2.tolist(),
        "phase_chi2_mean_about_empirical_mean_full_xi02": float(np.mean(phase_chi2)),
        "expected_phase_chi2_mean_about_empirical_mean": float((len(PHASES) - 1) / len(PHASES) * vector.shape[1]),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--threads", type=int, default=8)
    parser.add_argument("--nsub-window", type=int, default=200000)
    parser.add_argument("--window-seed-base", type=int, default=430340)
    parser.add_argument("--nwalkers", type=int, default=48)
    parser.add_argument("--nsteps", type=int, default=8000)
    parser.add_argument("--burnin", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=430350)
    parser.add_argument("--nmu", type=int, default=64)
    parser.add_argument("--sigma-grid-step", type=float, default=0.05)
    parser.add_argument("--manifest", type=Path, default=None)
    parser.add_argument(
        "--phase-window-root",
        type=Path,
        default=OUTPUT_ROOT / "lightcone" / "formal_gic_windows",
    )
    parser.add_argument(
        "--summary-path", "--summary",
        dest="summary_path",
        type=Path,
        default=OUTPUT_ROOT / "lightcone" / "summary" /
        "task43_rsd_lightcone_x25_mean_xi02_s30_350_ds10.npz",
    )
    parser.add_argument(
        "--covariance-path", "--covariance",
        dest="covariance_path",
        type=Path,
        required=True,
    )
    parser.add_argument(
        "--mean-window-path",
        type=Path,
        default=OUTPUT_ROOT / "lightcone" / "formal_gic_windows" /
        "task43_rsd_formal_gic_x25_equal_phase_mean_nsub200000_seedbase430340.npz",
    )
    parser.add_argument(
        "--output-prefix",
        type=Path,
        default=OUTPUT_ROOT / "lightcone" / "closure" /
        "task43_rsd_lightcone_x25_jaxpower_rrdeconv_rrnran300k_fulldiscrete_lorentzian",
    )
    parser.add_argument(
        "--plot",
        type=Path,
        default=PLOT_ROOT / "task43_rsd_lightcone_x25_jaxpower_rrdeconv_rrnran300k_fulldiscrete_lorentzian.pdf",
    )
    args = parser.parse_args()
    if int(args.burnin) >= int(args.nsteps):
        raise ValueError("--burnin must be smaller than --nsteps")
    output_npz = args.output_prefix.with_suffix(".npz")
    output_json = args.output_prefix.with_suffix(".json")
    if any(path.exists() for path in (output_npz, output_json, args.plot)):
        raise FileExistsError("immutable lightcone closure output exists; use a new tag")

    started = time.perf_counter()
    cpus = set_affinity(int(args.threads))
    inputs = load_inputs(args.summary_path, args.covariance_path)
    mean_xi = np.mean(inputs["xi"], axis=0)
    mean_rr = np.mean(inputs["rr"], axis=0)
    zeff_mean = float(np.mean(inputs["zeff"]))
    windows = {"mean": load_window(args.mean_window_path)}
    windows.update(
        {
            phase: load_window(
                phase_window_path(
                    phase,
                    int(args.nsub_window),
                    int(args.window_seed_base),
                    root=args.phase_window_root,
                )
            )
            for phase in PHASES
        }
    )
    theory_cache_path = build_cache(
        zeff=zeff_mean,
        boxsize=2000.0,
        kmax=5.0,
        ells=(0, 2),
        cosmology="abacus_c000",
    )
    exact_model = FullDiscreteRSDModel(theory_cache_path, nmu=int(args.nmu))
    fast_model = FastWindowRSDModel(
        exact_model, windows, sigma_step=float(args.sigma_grid_step)
    )
    surrogate_validation = fast_model.validate()
    print(
        json.dumps(
            {
                "stage": "surrogate_validation",
                "status": surrogate_validation["status"],
                "sigma_grid_step_mpc_h": surrogate_validation["sigma_grid_step_mpc_h"],
                "base_max_relative_l2": surrogate_validation["base"]["max_relative_l2"],
                "formal_gic_max_relative_l2": surrogate_validation["formal_gic_max_relative_l2"],
            },
            sort_keys=True,
        ),
        flush=True,
    )
    if surrogate_validation["status"] != "pass":
        raise RuntimeError(f"FullDiscrete likelihood surrogate failed validation: {surrogate_validation}")

    nominal: dict[str, dict[str, Any]] = {}
    nominal_chains: dict[str, np.ndarray] = {}
    nominal_logp: dict[str, np.ndarray] = {}
    nominal_maps: dict[str, np.ndarray] = {}
    convergence: dict[str, Any] = {}
    for imodel, model in enumerate(MODELS):
        fit, chain, logp, model_map = fit_one(
            inputs=inputs,
            xi_by_ell=mean_xi,
            rr=mean_rr,
            fast_model=fast_model,
            window_key="mean",
            model=model,
            ells=(0,),
            smin=50.0,
            nwalkers=int(args.nwalkers),
            nsteps=int(args.nsteps),
            burnin=int(args.burnin),
            seed=int(args.seed) + 100 * imodel,
            optimizer_only=False,
        )
        nominal[model] = fit
        nominal_chains[model] = chain
        nominal_logp[model] = logp
        nominal_maps[model] = model_map
        convergence[model] = convergence_diagnostics(
            chain,
            names=list(fit["parameter_names"]),
            nwalkers=int(args.nwalkers),
            nsteps=int(args.nsteps),
            burnin=int(args.burnin),
        )
        print(
            json.dumps(
                {
                    "stage": "nominal_mcmc",
                    "model": model,
                    "fnl_q50": fit["fnl_loc"]["q50"],
                    "chi2_single": fit["optimizer"]["chi2"],
                },
                sort_keys=True,
            ),
            flush=True,
        )

    scale_fits: list[dict[str, Any]] = []
    scale_maps: dict[str, np.ndarray] = {}
    for ells in ELL_SETS:
        for smin in SMIN_SCAN:
            for imodel, model in enumerate(MODELS):
                fit, _, _, model_map = fit_one(
                    inputs=inputs,
                    xi_by_ell=mean_xi,
                    rr=mean_rr,
                    fast_model=fast_model,
                    window_key="mean",
                    model=model,
                    ells=ells,
                    smin=float(smin),
                    nwalkers=8,
                    nsteps=2,
                    burnin=0,
                    seed=int(args.seed) + 1000 + 100 * imodel + int(smin),
                    optimizer_only=True,
                )
                record = profile_record("x25_mean", model, fit)
                record.update({"ells": [int(value) for value in ells], "smin_mpc_h": float(smin)})
                scale_fits.append(record)
                scale_maps[f"{model}_ell{''.join(map(str, ells))}_smin{int(smin):03d}"] = model_map

    phase_profiles: list[dict[str, Any]] = []
    for iphase, phase in enumerate(PHASES):
        for imodel, model in enumerate(MODELS):
            fit, _, _, _ = fit_one(
                inputs=inputs,
                xi_by_ell=inputs["xi"][iphase],
                rr=inputs["rr"][iphase],
                fast_model=fast_model,
                window_key=phase,
                model=model,
                ells=(0,),
                smin=50.0,
                nwalkers=8,
                nsteps=2,
                burnin=0,
                seed=int(args.seed) + 5000 + 100 * iphase + imodel,
                optimizer_only=True,
            )
            phase_profiles.append(profile_record(phase, model, fit))
        print(json.dumps({"stage": "phase_profiles", "phase": phase, "status": "done"}), flush=True)

    primary = nominal[PRIMARY_MODEL]
    primary_posterior = primary["fnl_loc"]
    primary_sigma = 0.5 * (float(primary_posterior["q84"]) - float(primary_posterior["q16"]))
    null_ratio = abs(float(primary_posterior["q50"])) / primary_sigma
    primary_context = primary["task43_fit_context"]
    mean_chi2 = len(PHASES) * float(primary_context["chi2_map_single_covariance"])
    mean_dof = int(primary_context["dof_nominal"])
    mean_pte = float(chi2_distribution.sf(mean_chi2, mean_dof))

    primary_profiles = [row for row in phase_profiles if row["model"] == PRIMARY_MODEL]
    ensemble_chi2 = float(sum(row["chi2"] for row in primary_profiles))
    ensemble_dof = int(sum(row["dof"] for row in primary_profiles))
    ensemble_pte = float(chi2_distribution.sf(ensemble_chi2, ensemble_dof))

    reference = next(
        row for row in scale_fits
        if row["model"] == PRIMARY_MODEL and row["ells"] == [0] and row["smin_mpc_h"] == 50.0
    )
    sigma_ref = {
        name: 0.5 * (float(primary[name]["q84"]) - float(primary[name]["q16"]))
        for name in ("fnl_loc", "b1", "sigma_s")
    }
    nested = [
        row for row in scale_fits
        if row["model"] == PRIMARY_MODEL and row["ells"] == [0] and row["smin_mpc_h"] in (80.0, 100.0, 120.0)
    ]
    nested_shifts = {
        str(int(row["smin_mpc_h"])): {
            name: float(abs(row["theta"][name] - reference["theta"][name]) / sigma_ref[name])
            for name in sigma_ref
        }
        for row in nested
    }
    max_scale_shift = max(value for row in nested_shifts.values() for value in row.values())
    convergence_gates = {
        f"{model}_{key}": bool(value)
        for model in MODELS
        for key, value in convergence[model]["gates"].items()
    }
    primary_gates = {
        "null_abs_median_over_sigma68_single_below_0p3": bool(null_ratio < 0.3),
        "mean_pte_Cmean_above_0p05": bool(mean_pte > 0.05),
        "phase_ensemble_profile_pte_above_0p05": bool(ensemble_pte > 0.05),
        "nested_scale_shift_max_below_0p3sigma_ref": bool(max_scale_shift < 0.3),
        **convergence_gates,
    }
    diagnostic_validation_status = "pass" if all(primary_gates.values()) else "validation_failed"
    status = "validation_failed"
    scatter = covariance_scatter_diagnostics(inputs)

    args.plot.parent.mkdir(parents=True, exist_ok=True)
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.backends.backend_pdf import PdfPages

    temporary_plot = args.plot.with_name(f".{args.plot.name}.{os.getpid()}.tmp.pdf")
    with PdfPages(temporary_plot) as pdf:
        figure, axes = plt.subplots(2, 2, figsize=(11.0, 7.5), sharex="col")
        formal_joint = next(
            row for row in scale_fits
            if row["model"] == PRIMARY_MODEL and row["ells"] == [0, 2] and row["smin_mpc_h"] == 80.0
        )
        joint_map = scale_maps["formal_gic_ell02_smin080"]
        joint_mask = inputs["s"] >= 80.0
        n_joint = int(np.count_nonzero(joint_mask))
        covariance_sigma = np.sqrt(np.diag(inputs["covariance"]))
        for iell, ell in enumerate((0, 2)):
            axes[0, iell].errorbar(
                inputs["s"],
                inputs["s"] ** 2 * mean_xi[iell],
                yerr=inputs["s"] ** 2 * covariance_sigma[iell * inputs["s"].size:(iell + 1) * inputs["s"].size] / np.sqrt(len(PHASES)),
                fmt="o",
                ms=3,
                label="x25 mean +/- analytic SEM",
            )
            prediction = joint_map[iell * n_joint:(iell + 1) * n_joint]
            axes[0, iell].plot(inputs["s"][joint_mask], inputs["s"][joint_mask] ** 2 * prediction, lw=1.5, label="formal-GIC joint fit, smin=80")
            axes[0, iell].axvline(80.0, color="0.3", ls="--", lw=0.8)
            axes[0, iell].set_ylabel(rf"$s^2\xi_{ell}(s)$")
            axes[0, iell].legend(frameon=False, fontsize=8)
            residual = (
                mean_xi[iell, joint_mask] - prediction
            ) / (covariance_sigma[iell * inputs["s"].size:(iell + 1) * inputs["s"].size][joint_mask] / np.sqrt(len(PHASES)))
            axes[1, iell].axhline(0.0, color="0.4", lw=0.8)
            axes[1, iell].plot(inputs["s"][joint_mask], residual, "o-", ms=3, lw=0.8)
            axes[1, iell].set(xlabel=r"$s\,[h^{-1}{\rm Mpc}]$", ylabel=r"residual / $\sigma_{\rm mean}$")
        figure.suptitle(
            f"Task 4.3.2 radial-LOS x25 diagnostic: {diagnostic_validation_status}; "
            f"joint fNL={formal_joint['theta']['fnl_loc']:.1f}"
        )
        figure.tight_layout()
        pdf.savefig(figure)
        plt.close(figure)

        figure, axes = plt.subplots(3, 1, figsize=(8.0, 8.4), sharex=True)
        styles = {
            ("no_gic", (0,)): ("#777777", "o", "no GIC, xi0"),
            ("formal_gic", (0,)): ("#1f4e79", "o", "formal GIC, xi0"),
            ("formal_gic", (0, 2)): ("#b22222", "s", "formal GIC, xi0+xi2"),
        }
        for (model, ells), (color, marker, label) in styles.items():
            rows = [row for row in scale_fits if row["model"] == model and tuple(row["ells"]) == ells]
            for iaxis, name in enumerate(("fnl_loc", "b1", "sigma_s")):
                axes[iaxis].plot(
                    [row["smin_mpc_h"] for row in rows],
                    [row["theta"][name] for row in rows],
                    color=color,
                    marker=marker,
                    label=label if iaxis == 0 else None,
                )
                axes[iaxis].set_ylabel(name)
        axes[0].axhline(0.0, color="0.4", lw=0.8)
        axes[0].legend(frameon=False, fontsize=8)
        axes[-1].set_xlabel(r"$s_{\min}\,[h^{-1}{\rm Mpc}]$")
        figure.suptitle(f"Frozen scale scan; max primary nested shift={max_scale_shift:.2f} sigma_ref")
        figure.tight_layout()
        pdf.savefig(figure)
        plt.close(figure)

        figure, axes = plt.subplots(2, 1, figsize=(8.5, 6.7))
        phase_fnl = np.asarray([row["theta"]["fnl_loc"] for row in primary_profiles])
        phase_pte = np.asarray([row["pte"] for row in primary_profiles])
        axes[0].axhline(0.0, color="0.4", lw=0.8)
        axes[0].plot(np.arange(len(PHASES)), phase_fnl / primary_sigma, "o-")
        axes[0].set(ylabel=r"profile $f_{\rm NL}/\sigma_{\rm single}$", xticks=np.arange(len(PHASES)), xticklabels=PHASES)
        axes[0].tick_params(axis="x", rotation=90, labelsize=7)
        axes[1].hist(phase_pte, bins=np.linspace(0.0, 1.0, 11), histtype="stepfilled", alpha=0.65)
        axes[1].axvline(0.05, color="#b22222", ls="--")
        axes[1].set(xlabel="per-phase formal-GIC profile PTE", ylabel="count")
        figure.suptitle(f"Phase audit: aggregate PTE={ensemble_pte:.3g}")
        figure.tight_layout()
        pdf.savefig(figure)
        plt.close(figure)

        figure, axes = plt.subplots(1, 2, figsize=(10.5, 4.2))
        ratio = np.asarray(scatter["sample_std_over_cov_sigma"])
        axes[0].plot(inputs["s"], ratio[: inputs["s"].size], "o-", label="xi0")
        axes[0].plot(inputs["s"], ratio[inputs["s"].size :], "s-", label="xi2")
        axes[0].axhline(1.0, color="0.4", lw=0.8)
        axes[0].set(xlabel=r"$s\,[h^{-1}{\rm Mpc}]$", ylabel="sample std / analytic sigma")
        axes[0].legend(frameon=False)
        axes[1].hist(scatter["phase_chi2_about_empirical_mean_full_xi02"], bins=10, alpha=0.7)
        axes[1].axvline(scatter["expected_phase_chi2_mean_about_empirical_mean"], color="#b22222", ls="--", label="expected mean")
        axes[1].set(xlabel="analytic-precision chi2 about empirical mean", ylabel="count")
        axes[1].legend(frameon=False)
        figure.suptitle("Single-lightcone covariance versus x25 scatter")
        figure.tight_layout()
        pdf.savefig(figure)
        plt.close(figure)
    temporary_plot.replace(args.plot)

    save_payload: dict[str, Any] = {
        "s": inputs["s"],
        "s_edges": inputs["s_edges"],
        "ells": np.asarray([0, 2], dtype="i4"),
        "phases": np.asarray(PHASES),
        "xi_multipoles_by_phase": inputs["xi"],
        "xi_multipoles_mean": mean_xi,
        "covariance_single_realization": inputs["covariance"],
        "covariance_of_mean": inputs["covariance"] / len(PHASES),
        "phase_profile_formalgic_theta": np.asarray(
            [[row["theta"][name] for name in ("fnl_loc", "b1", "sigma_s")] for row in primary_profiles]
        ),
        "phase_profile_formalgic_chi2": np.asarray([row["chi2"] for row in primary_profiles]),
    }
    for model in MODELS:
        save_payload[f"{model}_chain_flat"] = nominal_chains[model]
        save_payload[f"{model}_log_probability_flat"] = nominal_logp[model]
        save_payload[f"{model}_model_map"] = nominal_maps[model]
    for key, value in scale_maps.items():
        save_payload[f"scale_model_map_{key}"] = value
    atomic_savez(output_npz, **save_payload)

    no_gic_fnl = nominal["no_gic"]["fnl_loc"]
    formal_fnl = nominal["formal_gic"]["fnl_loc"]
    no_formal_shift = (
        float(formal_fnl["q50"]) - float(no_gic_fnl["q50"])
    ) / primary_sigma
    payload = {
        "task": "task43_fit_rsd_lightcone_x25",
        "status": status,
        "diagnostic_execution_status": "pass",
        "diagnostic_validation_status": diagnostic_validation_status,
        "classification": "production radial-LOS diagnostic closure after failed rawbox gate; cannot rescue or greenlight Task44",
        "phases": list(PHASES),
        "nphase": len(PHASES),
        "cpu_affinity": cpus,
        "model": {
            "name": "FullDiscrete shell-averaged Kaiser x squared-Lorentzian FoG",
            "p_fixed": 1.0,
            "primary_gic": PRIMARY_MODEL,
            "comparison_models": list(MODELS),
            "fit_ells_primary": [0],
            "xi2_role": "mandatory joint-fit and residual diagnostic, not part of the primary posterior",
            "zeff_mean": zeff_mean,
            "phase_profile_theory_redshift_policy": "common x25 mean zeff; each phase retains its own formal-GIC random window",
            "theory_cache_path": str(theory_cache_path),
            "theory_cache_sha256": sha256_file(theory_cache_path),
            "likelihood_surrogate_validation": surrogate_validation,
        },
        "covariance_policy": {
            "path": str(args.covariance_path),
            "sha256": sha256_file(args.covariance_path),
            "definition": "JAXpower RSD-multipole Gaussian survey-window covariance after RR(s,mu) deconvolution",
            "data_basis": "raw weighted Landy-Szalay xi, matching the established Task44 baseline",
            "quoted_posterior": "C_single, never divided by 25",
            "mean_goodness_of_fit": "C_mean=C_single/25",
            "phase_profiles": "C_single",
            "eigenvalue_min": float(inputs["covariance_eigenvalues"][0]),
            "condition_number": float(inputs["covariance_eigenvalues"][-1] / inputs["covariance_eigenvalues"][0]),
            "source_meta": inputs["covariance_meta"],
            "x25_scatter_comparison": scatter,
        },
        "formal_gic_window_policy": {
            "mean_window_path": str(args.mean_window_path),
            "mean_window_sha256": sha256_file(args.mean_window_path),
            "definition": "equal arithmetic mean of the 25 phase-specific W2(k) windows because the measured x25 mean is equally phase-weighted",
            "phase_nsub": int(args.nsub_window),
            "phase_seed_base": int(args.window_seed_base),
        },
        "nominal_mean_smin50_xi0": {model: compact_fit(fit) for model, fit in nominal.items()},
        "mcmc_convergence": convergence,
        "no_gic_to_formal_gic_fnl_shift_sigma_formal": float(no_formal_shift),
        "scale_fits": scale_fits,
        "phase_profiles": phase_profiles,
        "phase_ensemble_primary": {"model": PRIMARY_MODEL, "chi2": ensemble_chi2, "dof": ensemble_dof, "pte": ensemble_pte},
        "mean_goodness_primary": {"model": PRIMARY_MODEL, "chi2_Cmean": mean_chi2, "dof": mean_dof, "pte": mean_pte},
        "scale_stability": {
            "reference": "formal-GIC xi0 smin=50 optimizer; sigma_ref is nominal C_single posterior half-width",
            "nested_shifts_sigma_ref": nested_shifts,
            "max_shift_sigma_ref": float(max_scale_shift),
        },
        "primary_null_abs_median_over_sigma68_single": float(null_ratio),
        "primary_gates": primary_gates,
        "rawbox_dependency": {
            "status": "validation_failed",
            "consequence": "Even a passing lightcone diagnostic would not authorize a Task44 science claim until the rawbox covariance/model-shape failure is resolved.",
        },
        "task44_greenlight": False,
        "conditional_followups": {
            "radial_ic": "not_run",
            "hybrid_gsm": "not_run",
            "reason": "pre-registered base rawbox closure gate failed",
        },
        "inputs": {
            "summary": str(args.summary_path),
            "summary_sha256": sha256_file(args.summary_path),
            "measurements": [
                str(
                    measurement_path(
                        phase,
                        None if args.manifest is None else read_jsonl(args.manifest),
                    )
                )
                for phase in PHASES
            ],
            "manifest": None if args.manifest is None else str(args.manifest),
            "phase_window_root": str(args.phase_window_root),
        },
        "output_npz": str(output_npz),
        "output_npz_sha256": sha256_file(output_npz),
        "output_plot_pdf": str(args.plot),
        "output_plot_pdf_sha256": sha256_file(args.plot),
        "elapsed_sec": time.perf_counter() - started,
    }
    atomic_write_json(output_json, jsonable(payload))
    print(
        json.dumps(
            {
                "status": status,
                "diagnostic_execution_status": "pass",
                "diagnostic_validation_status": diagnostic_validation_status,
                "gates": primary_gates,
                "task44_greenlight": False,
                "output": str(output_json),
            },
            sort_keys=True,
        )
    )
    if diagnostic_validation_status != "pass":
        raise SystemExit(2)


if __name__ == "__main__":
    main()
