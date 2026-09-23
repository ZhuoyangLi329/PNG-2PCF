#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Fit Task44 LRG redshift-space P0(k) with a window-forward PNG model."""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from pathlib import Path
from typing import Any, Callable

for _name in (
    "OMP_NUM_THREADS",
    "MKL_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
):
    os.environ.setdefault(_name, "1")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

PROJECT_ROOT = Path("/pscratch/sd/l/lzy/PNG-2PCF-model-to-observe")
CODE_DIR = PROJECT_ROOT / "codes" / "task44"
if str(CODE_DIR) not in sys.path:
    sys.path.insert(0, str(CODE_DIR))

from task44_fit_validation import validate_covariance
from task44_config import SAMPLE_CONFIGS  # noqa: E402
from task44_pk_common import (  # noqa: E402
    SN0_SCALE,
    atomic_savez,
    covariance_diagnostics,
    ensure_sample_pk_dirs,
    ensure_pk_dirs,
    p_fixed_for_sample,
    pk_paths,
    to_jsonable,
    write_json,
)
from task44_rsd_theory import DELTA_C, FOG_MODELS, fog_description, growth_rate_approx  # noqa: E402

TASK43_DIR = PROJECT_ROOT / "codes" / "task43"
if str(TASK43_DIR) not in sys.path:
    sys.path.insert(0, str(TASK43_DIR))
from task43_theory_template import COSMOLOGY_CHOICES, DEFAULT_COSMOLOGY, build_template_arrays, load_task41  # noqa: E402


PARAM_NAMES = ("fnl_loc", "b1", "sigma_s", "sn0")
DEFAULT_PRIORS = {
    "fnl_loc": (-500.0, 500.0),
    "b1": (0.5, 5.0),
    "sigma_s": (0.0, 30.0),
    "sn0": (-1.0, 1.0),
}


class FitData:
    def __init__(self, payload: Path):
        self.payload = Path(payload)
        with np.load(self.payload, allow_pickle=False) as data:
            self.k_obs = np.asarray(data["k_obs"], dtype="f8")
            self.k_edges = np.asarray(data["k_edges"], dtype="f8")
            self.pk_data = np.asarray(data["pk_data"], dtype="f8")
            self.covariance = np.asarray(data["covariance"], dtype="f8")
            self.shotnoise_out = np.asarray(data["shotnoise_mean"], dtype="f8")
            self.shotnoise_scalar = float(np.asarray(data["shotnoise_mean_scalar"]).item())
            self.window_matrix = np.asarray(data["window_matrix"], dtype="f8")
            self.theory_k = np.asarray(data["theory_k"], dtype="f8")
            self.theory_ell = np.asarray(data["theory_ell"], dtype="i8")
            self.zeff = float(np.asarray(data["zeff"]).item())
            self.p0 = float(np.asarray(data["p0"]).item())
            self.kmin_fit_observed = float(np.asarray(data["kmin_fit_observed"]).item())
            self.kmax_fit = float(np.asarray(data["kmax_fit"]).item())
            self.sn0_scale = float(np.asarray(data["sn0_scale"]).item()) if "sn0_scale" in data.files else SN0_SCALE
            self.summary = json.loads(str(np.asarray(data["summary_json"]).item())) if "summary_json" in data.files else {}
            self.sample = str(np.asarray(data["sample"]).item()) if "sample" in data.files else str(self.summary.get("sample", "lrg2"))
            self.realization = (
                str(np.asarray(data["realization"]).item())
                if "realization" in data.files
                else str(self.summary.get("realization", "ph000_HODv4"))
            )
        self.sample_label = str(self.summary.get("sample_label", self.sample))
        if self.window_matrix.size == 0:
            raise ValueError("payload has no window_matrix; Task44 LRG P(k) fit requires window forward modeling")
        if self.window_matrix.shape[0] != self.pk_data.size:
            raise ValueError(f"window rows={self.window_matrix.shape[0]} but data size={self.pk_data.size}")
        if self.window_matrix.shape[1] != self.theory_k.size:
            raise ValueError(f"window columns={self.window_matrix.shape[1]} but theory size={self.theory_k.size}")
        if self.covariance.shape != (self.pk_data.size, self.pk_data.size):
            raise ValueError(f"covariance shape={self.covariance.shape} but data size={self.pk_data.size}")
        if not np.all(np.isfinite(self.pk_data)):
            raise ValueError("pk_data contains non-finite values")
        if np.any(~np.isfinite(self.covariance)):
            raise ValueError("covariance contains non-finite values")


class RSDTheoryCache:
    def __init__(self, data: FitData, template: dict[str, np.ndarray], *, nmu: int, fog_model: str):
        self.k = np.asarray(data.theory_k, dtype="f8")
        self.ell = np.asarray(data.theory_ell, dtype="i8")
        self.ells = tuple(sorted({int(v) for v in self.ell}))
        self.f_growth = growth_rate_approx(float(data.zeff))
        self.fog_model = str(fog_model)
        kbase = np.asarray(template["k"], dtype="f8")
        logkbase = np.log10(kbase)
        positive = self.k > 0.0
        self.positive = positive
        self.alpha = np.zeros_like(self.k)
        self.pk_dd = np.zeros_like(self.k)
        self.alpha[positive] = np.interp(np.log10(self.k[positive]), logkbase, np.asarray(template["alpha"], dtype="f8"))
        self.pk_dd[positive] = np.interp(np.log10(self.k[positive]), logkbase, np.asarray(template["pk_dd"], dtype="f8"))
        self.mu, self.wmu = np.polynomial.legendre.leggauss(int(nmu))
        self.mu2 = self.mu * self.mu
        self.legendre_by_ell: dict[int, np.ndarray] = {}
        for ell in self.ells:
            coeff = np.zeros(int(ell) + 1, dtype="f8")
            coeff[int(ell)] = 1.0
            self.legendre_by_ell[int(ell)] = np.polynomial.legendre.legval(self.mu, coeff)

    def vector(
        self,
        *,
        fnl_loc: float,
        b1: float,
        sigma_s: float,
        sn0_power: float,
        p_fixed: float,
        window_theory_kmin: float | None,
    ) -> np.ndarray:
        out = np.zeros_like(self.k, dtype="f8")
        good = self.positive.copy()
        if window_theory_kmin is not None:
            good &= self.k >= float(window_theory_kmin)
        if not np.any(good):
            return out
        bphi = 2.0 * DELTA_C * (float(b1) - float(p_fixed))
        amp = float(b1) + float(fnl_loc) * bphi * self.alpha[good]
        k_good = self.k[good]
        pk_dd = self.pk_dd[good]
        k_mu_sigma2 = (k_good[:, None] * self.mu[None, :] * float(sigma_s)) ** 2
        if self.fog_model == "lorentzian":
            damp = 1.0 / (1.0 + 0.5 * k_mu_sigma2) ** 2
        elif self.fog_model == "gaussian":
            damp = np.exp(-k_mu_sigma2)
        else:
            raise ValueError(f"unknown fog_model {self.fog_model!r}; choices are {FOG_MODELS}")
        pk_mu = pk_dd[:, None] * (amp[:, None] + self.f_growth * self.mu2[None, :]) ** 2 * damp
        good_indices = np.flatnonzero(good)
        ell_good = self.ell[good]
        for ell in self.ells:
            leg = self.legendre_by_ell[int(ell)]
            pole = 0.5 * (2 * int(ell) + 1) * np.sum(self.wmu[None, :] * pk_mu * leg[None, :], axis=1)
            if int(ell) == 0:
                pole = pole + float(sn0_power)
            sel = ell_good == int(ell)
            out[good_indices[sel]] = pole[sel]
        return out


def parse_prior(text: str) -> tuple[float, float]:
    lo, hi = (float(x) for x in str(text).split(",", 1))
    if not lo < hi:
        raise ValueError(f"bad prior {text!r}")
    return lo, hi


def theory_vector(
    data: FitData,
    theory_cache: RSDTheoryCache,
    *,
    fnl_loc: float,
    b1: float,
    sigma_s: float,
    sn0: float,
    p_fixed: float,
    sn0_scale: float,
    fog_model: str,
    window_theory_kmin: float | None,
) -> np.ndarray:
    del data, fog_model
    return theory_cache.vector(
        fnl_loc=float(fnl_loc),
        b1=float(b1),
        sigma_s=float(sigma_s),
        sn0_power=float(sn0) * float(sn0_scale),
        p_fixed=float(p_fixed),
        window_theory_kmin=window_theory_kmin,
    )


def model_pk(
    data: FitData,
    theory_cache: RSDTheoryCache,
    theta: dict[str, float],
    *,
    p_fixed: float,
    sn0_scale: float,
    fog_model: str,
    window_theory_kmin: float | None,
) -> np.ndarray:
    theory = theory_vector(
        data,
        theory_cache,
        fnl_loc=theta["fnl_loc"],
        b1=theta["b1"],
        sigma_s=theta["sigma_s"],
        sn0=theta["sn0"],
        p_fixed=p_fixed,
        sn0_scale=sn0_scale,
        fog_model=fog_model,
        window_theory_kmin=window_theory_kmin,
    )
    return data.window_matrix @ theory


def make_precision(cov: np.ndarray, covariance_floor: float) -> tuple[np.ndarray, dict[str, Any]]:
    # 对原矩阵检查有限值/对称性；显式 floor 之后再验证实际求逆矩阵的正定与秩。
    cov = np.asarray(cov, dtype="f8")
    if cov.ndim != 2 or cov.shape[0] != cov.shape[1] or not cov.size or not np.all(np.isfinite(cov)):
        raise ValueError("covariance must be finite, square and non-empty")
    if np.max(np.abs(cov - cov.T)) > 1.0e-10 * max(np.max(np.abs(cov)), np.finfo(float).tiny):
        raise ValueError("covariance is not symmetric")
    if not np.isfinite(covariance_floor) or covariance_floor < 0:
        raise ValueError("covariance_floor must be finite and non-negative")
    cov = 0.5 * (cov + cov.T)
    diag = np.diag(cov)
    if np.any(diag <= 0.0):
        raise ValueError("covariance has non-positive diagonal")
    floor = float(covariance_floor) * float(np.median(diag))
    cov_use = cov.copy()
    if floor > 0.0:
        cov_use.flat[:: cov_use.shape[0] + 1] += floor
    hard_gate = validate_covariance(cov_use, rcond=1.0e-10)
    precision = np.linalg.pinv(cov_use, rcond=1.0e-10)
    meta = covariance_diagnostics(cov_use)
    meta.update(hard_gate)
    meta.update(
        {
            "precision": "pinv analytic jaxpower covariance",
            "hartlap": 1.0,
            "percival_error_factor": 1.0,
            "diagonal_floor_added": float(floor),
            "covariance_floor_fraction": float(covariance_floor),
        }
    )
    return precision, meta


def pack_theta(values: np.ndarray, names: tuple[str, ...] = PARAM_NAMES) -> dict[str, float]:
    return {name: float(value) for name, value in zip(names, np.asarray(values, dtype="f8"), strict=True)}


def log_prior(values: np.ndarray, names: tuple[str, ...], priors: dict[str, tuple[float, float]]) -> float:
    for name, value in zip(names, np.asarray(values, dtype="f8"), strict=True):
        lo, hi = priors[name]
        if not (lo <= value <= hi):
            return -np.inf
    return 0.0


def chi2(
    values: np.ndarray,
    data: FitData,
    theory_cache: RSDTheoryCache,
    precision: np.ndarray,
    names: tuple[str, ...],
    *,
    p_fixed: float,
    sn0_scale: float,
    fog_model: str,
    window_theory_kmin: float | None,
) -> float:
    theta = pack_theta(values, names)
    model = model_pk(
        data,
        theory_cache,
        theta,
        p_fixed=p_fixed,
        sn0_scale=sn0_scale,
        fog_model=fog_model,
        window_theory_kmin=window_theory_kmin,
    )
    diff = data.pk_data - model
    return float(diff @ precision @ diff)


def make_log_prob(
    data: FitData,
    theory_cache: RSDTheoryCache,
    precision: np.ndarray,
    names: tuple[str, ...],
    *,
    p_fixed: float,
    sn0_scale: float,
    fog_model: str,
    window_theory_kmin: float | None,
    priors: dict[str, tuple[float, float]],
) -> Callable[[np.ndarray], float]:
    def log_prob(values: np.ndarray) -> float:
        lp = log_prior(values, names, priors)
        if not np.isfinite(lp):
            return -np.inf
        return lp - 0.5 * chi2(
            values,
            data,
            theory_cache,
            precision,
            names,
            p_fixed=p_fixed,
            sn0_scale=sn0_scale,
            fog_model=fog_model,
            window_theory_kmin=window_theory_kmin,
        )

    return log_prob


def find_initial_point(
    data: FitData,
    theory_cache: RSDTheoryCache,
    precision: np.ndarray,
    names: tuple[str, ...],
    *,
    p_fixed: float,
    sn0_scale: float,
    fog_model: str,
    window_theory_kmin: float | None,
    priors: dict[str, tuple[float, float]],
) -> tuple[np.ndarray, dict[str, Any]]:
    from scipy.optimize import minimize

    if str(data.sample).lower().startswith("qso"):
        base_starts = [
            {"fnl_loc": 100.0, "b1": 2.85, "sigma_s": 6.5, "sn0": 0.0},
            {"fnl_loc": 150.0, "b1": 2.9, "sigma_s": 7.0, "sn0": 0.0},
            {"fnl_loc": 50.0, "b1": 2.75, "sigma_s": 6.0, "sn0": 0.1},
            {"fnl_loc": 0.0, "b1": 2.8, "sigma_s": 5.5, "sn0": 0.0},
            {"fnl_loc": 200.0, "b1": 3.0, "sigma_s": 8.0, "sn0": -0.1},
        ]
    else:
        base_starts = [
            {"fnl_loc": 100.0, "b1": 1.95, "sigma_s": 6.8, "sn0": 0.0},
            {"fnl_loc": 150.0, "b1": 1.95, "sigma_s": 7.0, "sn0": 0.0},
            {"fnl_loc": 50.0, "b1": 1.9, "sigma_s": 6.0, "sn0": 0.1},
            {"fnl_loc": 0.0, "b1": 2.0, "sigma_s": 5.0, "sn0": 0.0},
            {"fnl_loc": 200.0, "b1": 1.8, "sigma_s": 9.0, "sn0": -0.1},
        ]
    bounds = [priors[name] for name in names]
    best = None
    for start in base_starts:
        x0 = np.asarray([start[name] for name in names], dtype="f8")
        for i, name in enumerate(names):
            lo, hi = priors[name]
            x0[i] = np.clip(x0[i], lo + 1.0e-8, hi - 1.0e-8)
        result = minimize(
            lambda x: chi2(
                x,
                data,
                theory_cache,
                precision,
                names,
                p_fixed=p_fixed,
                sn0_scale=sn0_scale,
                fog_model=fog_model,
                window_theory_kmin=window_theory_kmin,
            ),
            x0=x0,
            method="L-BFGS-B",
            bounds=bounds,
            options={"maxiter": 3000, "ftol": 1.0e-10},
        )
        if best is None or float(result.fun) < float(best.fun):
            best = result
    assert best is not None
    theta = pack_theta(best.x, names)
    return np.asarray(best.x, dtype="f8"), {
        "success": bool(best.success),
        "message": str(best.message),
        "chi2": float(best.fun),
        "point": theta,
    }


def initialize_walkers(center: np.ndarray, names: tuple[str, ...], nwalkers: int, seed: int, priors: dict[str, tuple[float, float]]) -> np.ndarray:
    rng = np.random.default_rng(int(seed))
    scale_map = {"fnl_loc": max(8.0, 0.04 * max(1.0, abs(center[0]))), "b1": 0.03, "sigma_s": 0.4, "sn0": 0.04}
    scales = np.asarray([scale_map[name] for name in names], dtype="f8")
    walkers = center[None, :] + rng.normal(scale=scales[None, :], size=(int(nwalkers), len(names)))
    for i, name in enumerate(names):
        lo, hi = priors[name]
        walkers[:, i] = np.clip(walkers[:, i], lo + 1.0e-8, hi - 1.0e-8)
    return walkers


def summarize_samples(samples: np.ndarray, names: tuple[str, ...]) -> dict[str, dict[str, float]]:
    out: dict[str, dict[str, float]] = {}
    for i, name in enumerate(names):
        vals = np.asarray(samples[:, i], dtype="f8")
        q025, q16, q50, q84, q975 = np.quantile(vals, [0.025, 0.1586552539, 0.5, 0.8413447461, 0.975])
        out[name] = {
            "mean": float(np.mean(vals)),
            "std": float(np.std(vals, ddof=1)),
            "median": float(q50),
            "q025": float(q025),
            "q16": float(q16),
            "q84": float(q84),
            "q975": float(q975),
            "err_low": float(q50 - q16),
            "err_high": float(q84 - q50),
        }
    return out


def best_sample(
    samples: np.ndarray,
    log_prob: np.ndarray,
    names: tuple[str, ...],
    *,
    data: FitData,
    theory_cache: RSDTheoryCache,
    precision: np.ndarray,
    p_fixed: float,
    sn0_scale: float,
    fog_model: str,
    window_theory_kmin: float | None,
) -> dict[str, Any]:
    imax = int(np.nanargmax(log_prob))
    values = np.asarray(samples[imax], dtype="f8")
    theta = pack_theta(values, names)
    model = model_pk(
        data,
        theory_cache,
        theta,
        p_fixed=p_fixed,
        sn0_scale=sn0_scale,
        fog_model=fog_model,
        window_theory_kmin=window_theory_kmin,
    )
    diff = data.pk_data - model
    return {
        "log_prob": float(log_prob[imax]),
        "chi2": float(diff @ precision @ diff),
        "ndof": int(data.pk_data.size - len(names)),
        "point": theta,
        "pk_model_best": model,
        "pk_residual_best": diff,
    }


def plot_fit(data: FitData, best: dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    model = np.asarray(best["pk_model_best"], dtype="f8")
    sigma = np.sqrt(np.diag(data.covariance))
    fig, (ax, rx) = plt.subplots(2, 1, figsize=(8.4, 6.8), sharex=True, gridspec_kw={"height_ratios": [3, 1]})
    ax.errorbar(data.k_obs, data.pk_data, yerr=sigma, fmt="o", ms=4, color="black", capsize=2, label=f"{data.sample_label} P0")
    ax.plot(data.k_obs, model, color="#b24b36", lw=2.0, label="best fit")
    ax.set_ylabel(r"$P_0(k)$")
    ax.legend(frameon=False)
    ax.grid(alpha=0.25)
    rx.axhline(0.0, color="0.4", lw=1.0)
    rx.errorbar(data.k_obs, data.pk_data - model, yerr=sigma, fmt="o", ms=4, color="black", capsize=2)
    rx.set_xlabel(r"$k\,[h\,{\rm Mpc}^{-1}]$")
    rx.set_ylabel("res.")
    rx.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Fit Task44 LRG P0(k) with a redshift-space window-forward PNG model.")
    parser.add_argument("--payload", type=Path, required=True)
    parser.add_argument("--label", type=str, default=None)
    parser.add_argument("--sample", choices=tuple(sorted(SAMPLE_CONFIGS)), default=None)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--plot-dir", type=Path, default=None)
    parser.add_argument("--target-evals", type=int, default=160_000)
    parser.add_argument("--nwalkers", type=int, default=32)
    parser.add_argument("--burn-fraction", type=float, default=0.15)
    parser.add_argument("--seed", type=int, default=20260706)
    parser.add_argument("--p-fixed", type=float, default=None)
    parser.add_argument("--sn0-scale", type=float, default=None)
    parser.add_argument("--fnl-prior", default="-500,500")
    parser.add_argument("--b1-prior", default="0.5,5")
    parser.add_argument("--sigma-s-prior", default="0,30")
    parser.add_argument("--sn0-prior", default="-1,1")
    parser.add_argument("--fog-model", choices=list(FOG_MODELS), default="lorentzian")
    parser.add_argument("--nmu", type=int, default=128, help="Gauss-Legendre mu quadrature points for RSD multipoles.")
    parser.add_argument("--cosmology", choices=COSMOLOGY_CHOICES, default=DEFAULT_COSMOLOGY)
    parser.add_argument("--template-kmax", type=float, default=20.0)
    parser.add_argument("--covariance-floor", type=float, default=0.0)
    parser.add_argument(
        "--window-theory-kmin",
        type=float,
        default=None,
        help="If omitted, use payload kmin_fit_observed. This low-k cutoff is applied to theory vector before W(P).",
    )
    parser.add_argument("--disable-window-theory-kmin", action="store_true")
    parser.add_argument(
        "--show-progress",
        action="store_true",
        help="Show the per-step tqdm meter. Disabled by default to keep long-run logs compact.",
    )
    parser.add_argument(
        "--skip-fit-plot",
        action="store_true",
        help=(
            "Do not emit the fitter's standalone diagnostic PDF. Use this when a downstream "
            "audited plotting stage owns the final, curated PDF products."
        ),
    )
    parser.add_argument(
        "--optimizer-only",
        action="store_true",
        help=(
            "Write the deterministic multi-start MAP result without running emcee. "
            "This is reserved for pre-registered scale-cut audits; primary constraints "
            "must use the normal long-chain mode."
        ),
    )
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> None:
    import emcee

    args = parse_args()
    ensure_pk_dirs()
    data = FitData(Path(args.payload))
    sample = args.sample or data.sample
    paths = ensure_sample_pk_dirs(sample)
    names = PARAM_NAMES
    if int(args.nwalkers) < 2 * len(names):
        raise ValueError("nwalkers must be >= 2 * ndim")
    label = args.label or f"{Path(args.payload).stem}_free_sn0_sigmas"
    output_dir = Path(args.output_dir) if args.output_dir is not None else paths["fits"]
    plot_dir = Path(args.plot_dir) if args.plot_dir is not None else paths["plots"]
    out_dir = output_dir / label
    summary_path = out_dir / f"task44_pk_{sample}_{label}_fit_summary.json"
    samples_path = out_dir / f"task44_pk_{sample}_{label}_fit_samples.npz"
    plot_path = plot_dir / f"task44_pk_{sample}_{label}_fit.pdf"
    if summary_path.exists() and samples_path.exists() and not args.overwrite:
        print(f"[skip] {summary_path}")
        return

    priors = {
        "fnl_loc": parse_prior(args.fnl_prior),
        "b1": parse_prior(args.b1_prior),
        "sigma_s": parse_prior(args.sigma_s_prior),
        "sn0": parse_prior(args.sn0_prior),
    }
    sn0_scale = float(args.sn0_scale) if args.sn0_scale is not None else float(data.sn0_scale)
    p_fixed = p_fixed_for_sample(sample) if args.p_fixed is None else float(args.p_fixed)
    precision, cov_meta = make_precision(data.covariance, covariance_floor=float(args.covariance_floor))
    if args.disable_window_theory_kmin:
        window_theory_kmin = None
    elif args.window_theory_kmin is None:
        window_theory_kmin = float(data.kmin_fit_observed)
    else:
        window_theory_kmin = float(args.window_theory_kmin)

    task41 = load_task41()
    positive_k = data.theory_k[data.theory_k > 0.0]
    kmin_template = min(1.0e-5, float(np.min(positive_k)) * 0.5)
    k_template = np.logspace(np.log10(kmin_template), np.log10(float(args.template_kmax)), 20000)
    template, cosmology_meta = build_template_arrays(task41, k_template, z=float(data.zeff), cosmology=str(args.cosmology))
    theory_cache = RSDTheoryCache(data, template, nmu=int(args.nmu), fog_model=str(args.fog_model))

    center, opt_meta = find_initial_point(
        data,
        theory_cache,
        precision,
        names,
        p_fixed=float(p_fixed),
        sn0_scale=sn0_scale,
        fog_model=str(args.fog_model),
        window_theory_kmin=window_theory_kmin,
        priors=priors,
    )
    log_prob = make_log_prob(
        data,
        theory_cache,
        precision,
        names,
        p_fixed=float(p_fixed),
        sn0_scale=sn0_scale,
        fog_model=str(args.fog_model),
        window_theory_kmin=window_theory_kmin,
        priors=priors,
    )
    if args.optimizer_only:
        nsteps = 0
        burnin = 0
        elapsed = 0.0
        flat_samples = np.asarray(center, dtype="f8")[None, :]
        flat_logp = np.asarray([log_prob(center)], dtype="f8")
        # A one-point MAP audit has no posterior interval.  Keep the familiar
        # summary schema, but mark every uncertainty field as unavailable so a
        # downstream plotter cannot accidentally present MAP points as chains.
        params = {
            name: {
                "mean": float(center[index]),
                "std": None,
                "median": float(center[index]),
                "q025": None,
                "q16": None,
                "q84": None,
                "q975": None,
                "err_low": None,
                "err_high": None,
            }
            for index, name in enumerate(names)
        }
        acceptance_mean = None
        acceptance_min = None
        acceptance_max = None
    else:
        nsteps = int(math.ceil(float(args.target_evals) / float(args.nwalkers)))
        burnin = int(math.floor(float(args.burn_fraction) * float(nsteps)))
        walkers = initialize_walkers(center, names, int(args.nwalkers), int(args.seed), priors)
        sampler = emcee.EnsembleSampler(int(args.nwalkers), len(names), log_prob)
        t0 = time.time()
        sampler.run_mcmc(
            walkers,
            nsteps,
            progress=bool(args.show_progress),
            skip_initial_state_check=True,
        )
        elapsed = time.time() - t0
        chain = sampler.get_chain()
        logp = sampler.get_log_prob()
        burnin = min(burnin, max(0, chain.shape[0] - 1))
        flat_samples = chain[burnin:, :, :].reshape((-1, len(names)))
        flat_logp = logp[burnin:, :].reshape((-1,))
        params = summarize_samples(flat_samples, names)
        acceptance_mean = float(np.mean(sampler.acceptance_fraction))
        acceptance_min = float(np.min(sampler.acceptance_fraction))
        acceptance_max = float(np.max(sampler.acceptance_fraction))
    best = best_sample(
        flat_samples,
        flat_logp,
        names,
        data=data,
        theory_cache=theory_cache,
        precision=precision,
        p_fixed=float(p_fixed),
        sn0_scale=sn0_scale,
        fog_model=str(args.fog_model),
        window_theory_kmin=window_theory_kmin,
    )
    # 最终展示由独立审计绘图器统一负责时，可以只保存 MCMC 数值产物，避免
    # 在 active plot 目录额外留下一个未纳入最终图合同的诊断 PDF。
    if not args.skip_fit_plot:
        plot_fit(data, best, plot_path)

    out_dir.mkdir(parents=True, exist_ok=True)
    atomic_savez(
        samples_path,
        param_names=np.asarray(names),
        samples=np.asarray(flat_samples, dtype="f8"),
        log_prob=np.asarray(flat_logp, dtype="f8"),
        k_obs=np.asarray(data.k_obs, dtype="f8"),
        pk_data=np.asarray(data.pk_data, dtype="f8"),
        covariance=np.asarray(data.covariance, dtype="f8"),
        pk_model_best=np.asarray(best["pk_model_best"], dtype="f8"),
        burnin=np.asarray(burnin, dtype="i8"),
        nsteps=np.asarray(nsteps, dtype="i8"),
        nwalkers=np.asarray(args.nwalkers, dtype="i8"),
        optimizer_only=np.asarray(bool(args.optimizer_only)),
        sample=np.asarray(str(sample)),
        realization=np.asarray(str(data.realization)),
    )
    summary = {
        "task": "task44_fit_pk_lrg2",
        "status": "done",
        "label": label,
        "sample": str(sample),
        "sample_label": str(data.sample_label),
        "realization": str(data.realization),
        "payload": str(data.payload),
        "config": {
            "observable": "redshift-space P0 only",
            "parameter_names": list(names),
            "free_parameters": list(names),
            "fixed_parameters": {},
            "sn0_policy": "free",
            "sigma_s_policy": "free_redshift_space",
            "p_fixed": float(p_fixed),
            "sn0_scale": float(sn0_scale),
            "sn0_units_note": "model adds sn0 * sn0_scale to P0 theory before window convolution",
            "fog_model": str(args.fog_model),
            "fog_description": fog_description(str(args.fog_model)),
            "nmu": int(args.nmu),
            "f_growth": float(growth_rate_approx(float(data.zeff))),
            "window_forward_model": "observed P0 = W_{0,{0,2,4}} @ P_theory_{0,2,4}",
            "window_theory_kmin": None if window_theory_kmin is None else float(window_theory_kmin),
            "window_theory_kmin_note": (
                "Low-k cutoff applied to the theory vector before W(P). By default this equals the observed-bin fit kmin."
            ),
            "no_ric_amr_gic": True,
            "priors": {key: list(value) for key, value in priors.items()},
            "target_evals": int(args.target_evals),
            "nwalkers": int(args.nwalkers),
            "nsteps": int(nsteps),
            "burnin": int(burnin),
            "optimizer_only": bool(args.optimizer_only),
            "progress_shown": bool(args.show_progress),
            "cosmology": str(args.cosmology),
            "cosmology_meta": cosmology_meta,
        },
        "parameters": params,
        "maximum_posterior_sample": best,
        "optimizer_initial_center": opt_meta,
        "covariance": cov_meta,
        "data": {
            "ndata": int(data.pk_data.size),
            "k_min": float(np.min(data.k_obs)),
            "k_max": float(np.max(data.k_obs)),
            "k_edge_min": float(np.min(data.k_edges[:, 0])),
            "k_edge_max": float(np.max(data.k_edges[:, 1])),
            "dk_min": float(np.min(data.k_edges[:, 1] - data.k_edges[:, 0])),
            "dk_max": float(np.max(data.k_edges[:, 1] - data.k_edges[:, 0])),
            "kmin_fit_observed": float(data.kmin_fit_observed),
            "kmax_fit": float(data.kmax_fit),
            "zeff": float(data.zeff),
            "p0": float(data.p0),
            "shotnoise_scalar": float(data.shotnoise_scalar),
        },
        "input_summary": data.summary,
        "diagnostics": {
            "acceptance_fraction_mean": acceptance_mean,
            "acceptance_fraction_min": acceptance_min,
            "acceptance_fraction_max": acceptance_max,
            "elapsed_sec": float(elapsed),
        },
        "paths": {
            "summary_json": str(summary_path),
            "samples_npz": str(samples_path),
            "fit_pdf": None if args.skip_fit_plot else str(plot_path),
        },
    }
    write_json(summary_path, to_jsonable(summary))
    print(f"[write] {samples_path}")
    if not args.skip_fit_plot:
        print(f"[write] {plot_path}")
    print(f"[write] {summary_path}")
    fnl = params["fnl_loc"]
    if args.optimizer_only:
        print(
            f"[map] fNL={fnl['median']:.2f} chi2={best['chi2']:.2f}/{best['ndof']}",
            flush=True,
        )
    else:
        print(
            f"[fit] fNL={fnl['median']:.2f} -{fnl['err_low']:.2f} +{fnl['err_high']:.2f} "
            f"chi2={best['chi2']:.2f}/{best['ndof']}",
            flush=True,
        )


if __name__ == "__main__":
    main()
