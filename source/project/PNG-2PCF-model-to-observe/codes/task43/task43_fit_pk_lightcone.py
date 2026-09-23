#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Fit Task43 lightcone P(k) with a geometry-window forward model.

Main convention follows the corrected Task4.2 P(k) line:

- real space, so sigmas is fixed to 0;
- sn0 is free with desilike-style scale 1e4;
- the lsstypes/desilike-refactor window model is W(P_theory), with jaxpower
  covariance carrying Poisson terms through WS/SW/SS;
- no RIC, no AMR, no GIC.
"""

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
CODE_DIR = PROJECT_ROOT / "codes" / "task43"
if str(CODE_DIR) not in sys.path:
    sys.path.insert(0, str(CODE_DIR))

from task43_pk_common import (  # noqa: E402
    DELTA_C,
    P_FIXED,
    PK_FIT_DIR,
    PK_PLOT_DIR,
    SIGMAS_FIXED,
    SN0_SCALE,
    atomic_savez,
    covariance_diagnostics,
    to_jsonable,
    write_json,
)
from task43_finite_mock_corrections import covariance_corrections  # noqa: E402
from task43_theory_template import COSMOLOGY_CHOICES, DEFAULT_COSMOLOGY, build_template_arrays, load_task41  # noqa: E402


BASE_PARAM_NAMES = ("fnl_loc", "b1", "sn0")
DEFAULT_PRIORS = {
    "fnl_loc": (-500.0, 500.0),
    "b1": (0.5, 5.0),
    "sn0": (-1.0, 1.0),
}


class FitData:
    def __init__(self, payload: Path):
        self.payload = payload
        with np.load(payload, allow_pickle=False) as data:
            self.k_obs = np.asarray(data["k_obs"], dtype="f8")
            self.k_edges = np.asarray(data["k_edges"], dtype="f8")
            self.pk_data = np.asarray(data["pk_mean"], dtype="f8")
            self.covariance = np.asarray(data["covariance"], dtype="f8")
            self.shotnoise_out = np.asarray(data["shotnoise_mean"], dtype="f8")
            self.shotnoise_scalar = float(np.asarray(data["shotnoise_mean_scalar"]).item())
            self.window_matrix = np.asarray(data["window_matrix"], dtype="f8")
            self.theory_k = np.asarray(data["theory_k"], dtype="f8")
            self.theory_ell = np.asarray(data["theory_ell"], dtype="i8")
            self.zeff = float(np.asarray(data["zeff"]).item()) if "zeff" in data.files else np.nan
            self.p0 = float(np.asarray(data["p0"]).item()) if "p0" in data.files else np.nan
            self.kmin_fit_observed = float(np.asarray(data["kmin_fit_observed"]).item())
            self.kmax_fit = float(np.asarray(data["kmax_fit"]).item())
            self.sn0_scale = float(np.asarray(data["sn0_scale"]).item()) if "sn0_scale" in data.files else SN0_SCALE
            self.summary = json.loads(str(np.asarray(data["summary_json"]).item())) if "summary_json" in data.files else {}
        if self.window_matrix.size == 0:
            raise ValueError("payload has no window_matrix; Task43 lightcone P(k) fit requires window forward modeling")
        if self.window_matrix.shape[0] != self.pk_data.size:
            raise ValueError(f"window rows={self.window_matrix.shape[0]} but data size={self.pk_data.size}")
        if self.window_matrix.shape[1] != self.theory_k.size:
            raise ValueError(f"window columns={self.window_matrix.shape[1]} but theory size={self.theory_k.size}")
        if self.covariance.shape != (self.pk_data.size, self.pk_data.size):
            raise ValueError(f"covariance shape={self.covariance.shape} but data size={self.pk_data.size}")
        if not np.all(np.isfinite(self.pk_data)):
            raise ValueError("pk_data contains non-finite values")
        # 默认保持历史 geometry-only 模型；只有显式传入 --ric-operator 时才
        # 装载独立 matrix。这样旧命令和旧结果不会被本轮修改静默改变。
        self.ric_matrix = np.zeros_like(self.window_matrix)
        self.ric_meta: dict[str, Any] | None = None

    def load_radial_ric_operator(self, path: Path) -> None:
        """装载与当前 payload theory grid 严格一致的 single-term RIC matrix。

        operator 保存的是正的 auto response；``model_pk`` 统一执行
        ``W_geom P - W_RIC P``。同一 theory vector 中已经含 free sn0，因此
        stochastic constant 也按相同 radial response 投影到低 k。
        """
        with np.load(path, allow_pickle=False) as operator:
            theory_k = np.asarray(operator["pk_theory_k"], dtype="f8")
            theory_ell = np.asarray(operator["pk_theory_ell"], dtype="i8")
            matrix = np.asarray(operator["pk_ric_matrix"], dtype="f8")
            meta = json.loads(str(np.asarray(operator["meta_json"]).item()))
        if theory_k.shape != self.theory_k.shape or not np.allclose(theory_k, self.theory_k, rtol=0.0, atol=1.0e-14):
            raise ValueError(f"RIC operator {path} 的 theory_k 与 P(k) payload 不一致")
        if theory_ell.shape != self.theory_ell.shape or not np.array_equal(theory_ell, self.theory_ell):
            raise ValueError(f"RIC operator {path} 的 theory_ell 与 P(k) payload 不一致")
        if matrix.shape != self.window_matrix.shape:
            raise ValueError(f"RIC matrix shape={matrix.shape}，geometry window shape={self.window_matrix.shape}")
        self.ric_matrix = matrix
        self.ric_meta = {"path": str(path), "audit": meta}


def parse_prior(text: str) -> tuple[float, float]:
    lo, hi = (float(x) for x in str(text).split(",", 1))
    if not lo < hi:
        raise ValueError(f"bad prior {text!r}")
    return lo, hi


def interp_logk(k_query: np.ndarray, k_base: np.ndarray, y_base: np.ndarray) -> np.ndarray:
    return np.interp(np.log10(k_query), np.log10(k_base), y_base)


def png_realspace_pk(
    k: np.ndarray,
    template: dict[str, np.ndarray],
    *,
    fnl_loc: float,
    b1: float,
    p_fixed: float,
) -> np.ndarray:
    k = np.asarray(k, dtype="f8")
    out = np.zeros_like(k)
    mask = k > 0.0
    if not np.any(mask):
        return out
    alpha = interp_logk(k[mask], template["k"], template["alpha"])
    pk_dd = interp_logk(k[mask], template["k"], template["pk_dd"])
    bphi = 2.0 * DELTA_C * (float(b1) - float(p_fixed))
    bias = float(b1) + bphi * float(fnl_loc) * alpha
    out[mask] = bias * bias * pk_dd
    return out


def theory_vector(
    data: FitData,
    template: dict[str, np.ndarray],
    *,
    fnl_loc: float,
    b1: float,
    sn0: float,
    p_fixed: float,
    sn0_scale: float,
    window_theory_kmin: float | None,
) -> np.ndarray:
    out = np.zeros_like(data.theory_k, dtype="f8")
    ell0 = data.theory_ell == 0
    out[ell0] = png_realspace_pk(data.theory_k[ell0], template, fnl_loc=fnl_loc, b1=b1, p_fixed=p_fixed)
    out[ell0] += float(sn0) * float(sn0_scale)
    if window_theory_kmin is not None:
        out[data.theory_k < float(window_theory_kmin)] = 0.0
    return out


def model_pk(
    data: FitData,
    template: dict[str, np.ndarray],
    theta: dict[str, float],
    *,
    p_fixed: float,
    sn0_scale: float,
    window_theory_kmin: float | None,
) -> np.ndarray:
    theory = theory_vector(
        data,
        template,
        fnl_loc=theta["fnl_loc"],
        b1=theta["b1"],
        sn0=theta["sn0"],
        p_fixed=p_fixed,
        sn0_scale=sn0_scale,
        window_theory_kmin=window_theory_kmin,
    )
    # radial normalization 已经包含 global normalization，所以 RIC branch
    # 只减同一个 radial auto response，不再加入 standalone GIC/sigma_W2。
    return data.window_matrix @ theory - data.ric_matrix @ theory


def make_precision(
    cov: np.ndarray,
    covariance_floor: float,
    *,
    covariance_nmock: int | None = None,
    nparams: int = len(BASE_PARAM_NAMES),
) -> tuple[np.ndarray, dict[str, Any]]:
    cov = 0.5 * (np.asarray(cov, dtype="f8") + np.asarray(cov, dtype="f8").T)
    diag = np.diag(cov)
    if np.any(diag <= 0.0):
        raise ValueError("covariance has non-positive diagonal")
    floor = float(covariance_floor) * float(np.median(diag))
    cov_use = cov.copy()
    if floor > 0.0:
        cov_use.flat[:: cov_use.shape[0] + 1] += floor
    precision = np.linalg.pinv(cov_use, rcond=1.0e-10)
    meta = covariance_diagnostics(cov_use)
    if covariance_nmock is None:
        correction = {
            "hartlap": 1.0,
            "percival_m1_variance": 1.0,
            "percival_error_factor": 1.0,
            "finite_mock_correction": False,
        }
        precision_label = "pinv analytic jaxpower covariance"
    else:
        correction = covariance_corrections(
            nmock=int(covariance_nmock), ndata=int(cov_use.shape[0]), nparams=int(nparams)
        )
        correction["finite_mock_correction"] = True
        precision *= float(correction["hartlap"])
        precision_label = "Hartlap * pinv finite-mock sample covariance"
    meta.update(correction)
    meta.update(
        {
            "precision": precision_label,
            "diagonal_floor_added": float(floor),
            "covariance_floor_fraction": float(covariance_floor),
        }
    )
    return precision, meta


def param_names(sn0_policy: str) -> tuple[str, ...]:
    if sn0_policy != "free":
        raise ValueError("Task43 P(k) requires free sn0; fixed sn0=0 is not a supported analysis mode.")
    return BASE_PARAM_NAMES


def pack_theta(values: np.ndarray, names: tuple[str, ...], *, sn0_policy: str) -> dict[str, float]:
    theta = {"fnl_loc": 0.0, "b1": 2.5, "sn0": 0.0}
    for name, value in zip(names, np.asarray(values, dtype="f8"), strict=True):
        theta[name] = float(value)
    return theta


def log_prior(values: np.ndarray, names: tuple[str, ...], priors: dict[str, tuple[float, float]]) -> float:
    for name, value in zip(names, np.asarray(values, dtype="f8"), strict=True):
        lo, hi = priors[name]
        if not (lo <= value <= hi):
            return -np.inf
    return 0.0


def chi2(
    values: np.ndarray,
    data: FitData,
    template: dict[str, np.ndarray],
    precision: np.ndarray,
    names: tuple[str, ...],
    *,
    p_fixed: float,
    sn0_scale: float,
    sn0_policy: str,
    window_theory_kmin: float | None,
) -> float:
    theta = pack_theta(values, names, sn0_policy=sn0_policy)
    model = model_pk(data, template, theta, p_fixed=p_fixed, sn0_scale=sn0_scale, window_theory_kmin=window_theory_kmin)
    diff = data.pk_data - model
    return float(diff @ precision @ diff)


def make_log_prob(
    data: FitData,
    template: dict[str, np.ndarray],
    precision: np.ndarray,
    names: tuple[str, ...],
    *,
    p_fixed: float,
    sn0_scale: float,
    sn0_policy: str,
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
            template,
            precision,
            names,
            p_fixed=p_fixed,
            sn0_scale=sn0_scale,
            sn0_policy=sn0_policy,
            window_theory_kmin=window_theory_kmin,
        )

    return log_prob


def find_initial_point(
    data: FitData,
    template: dict[str, np.ndarray],
    precision: np.ndarray,
    names: tuple[str, ...],
    *,
    p_fixed: float,
    sn0_scale: float,
    sn0_policy: str,
    window_theory_kmin: float | None,
    priors: dict[str, tuple[float, float]],
) -> tuple[np.ndarray, dict[str, Any]]:
    from scipy.optimize import minimize

    base_starts = [
        {"fnl_loc": 0.0, "b1": 2.5, "sn0": 0.0},
        {"fnl_loc": 50.0, "b1": 2.5, "sn0": 0.0},
        {"fnl_loc": -50.0, "b1": 2.5, "sn0": 0.0},
        {"fnl_loc": 0.0, "b1": 2.2, "sn0": 0.2},
        {"fnl_loc": 0.0, "b1": 2.8, "sn0": -0.2},
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
                template,
                precision,
                names,
                p_fixed=p_fixed,
                sn0_scale=sn0_scale,
                sn0_policy=sn0_policy,
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
    theta = pack_theta(best.x, names, sn0_policy=sn0_policy)
    return np.asarray(best.x, dtype="f8"), {
        "success": bool(best.success),
        "message": str(best.message),
        "chi2": float(best.fun),
        "point": theta,
    }


def initialize_walkers(center: np.ndarray, names: tuple[str, ...], nwalkers: int, seed: int, priors: dict[str, tuple[float, float]]) -> np.ndarray:
    rng = np.random.default_rng(int(seed))
    scale_map = {"fnl_loc": max(5.0, 0.03 * max(1.0, abs(center[0]))), "b1": 0.025, "sn0": 0.04}
    scales = np.asarray([scale_map[name] for name in names], dtype="f8")
    walkers = center[None, :] + rng.normal(scale=scales[None, :], size=(int(nwalkers), len(names)))
    for i, name in enumerate(names):
        lo, hi = priors[name]
        walkers[:, i] = np.clip(walkers[:, i], lo + 1.0e-8, hi - 1.0e-8)
    return walkers


def summarize_samples(
    samples: np.ndarray,
    names: tuple[str, ...],
    *,
    percival_error_factor: float = 1.0,
) -> dict[str, dict[str, float]]:
    out: dict[str, dict[str, float]] = {}
    for i, name in enumerate(names):
        vals = np.asarray(samples[:, i], dtype="f8")
        q025, q16, q50, q84, q975 = np.quantile(vals, [0.025, 0.1586552539, 0.5, 0.8413447461, 0.975])
        err_low = float(q50 - q16)
        err_high = float(q84 - q50)
        out[name] = {
            "mean": float(np.mean(vals)),
            "std": float(np.std(vals, ddof=1)),
            "median": float(q50),
            "q025": float(q025),
            "q16": float(q16),
            "q84": float(q84),
            "q975": float(q975),
            "err_low": err_low,
            "err_high": err_high,
            "err_low_percival": float(err_low * percival_error_factor),
            "err_high_percival": float(err_high * percival_error_factor),
            "std_percival": float(np.std(vals, ddof=1) * percival_error_factor),
        }
    if "sn0" not in out:
        out["sn0"] = {
            "mean": 0.0,
            "std": 0.0,
            "median": 0.0,
            "q025": 0.0,
            "q16": 0.0,
            "q84": 0.0,
            "q975": 0.0,
            "err_low": 0.0,
            "err_high": 0.0,
        }
    out["sigmas"] = {
        "mean": SIGMAS_FIXED,
        "std": 0.0,
        "median": SIGMAS_FIXED,
        "q025": SIGMAS_FIXED,
        "q16": SIGMAS_FIXED,
        "q84": SIGMAS_FIXED,
        "q975": SIGMAS_FIXED,
        "err_low": 0.0,
        "err_high": 0.0,
    }
    return out


def best_sample(
    samples: np.ndarray,
    log_prob: np.ndarray,
    names: tuple[str, ...],
    *,
    data: FitData,
    template: dict[str, np.ndarray],
    precision: np.ndarray,
    p_fixed: float,
    sn0_scale: float,
    sn0_policy: str,
    window_theory_kmin: float | None,
) -> dict[str, Any]:
    imax = int(np.nanargmax(log_prob))
    values = np.asarray(samples[imax], dtype="f8")
    theta = pack_theta(values, names, sn0_policy=sn0_policy)
    theory = theory_vector(
        data,
        template,
        fnl_loc=theta["fnl_loc"],
        b1=theta["b1"],
        sn0=theta["sn0"],
        p_fixed=p_fixed,
        sn0_scale=sn0_scale,
        window_theory_kmin=window_theory_kmin,
    )
    geometry_model = data.window_matrix @ theory
    ric_correction = data.ric_matrix @ theory
    model = geometry_model - ric_correction
    diff = data.pk_data - model
    return {
        "log_prob": float(log_prob[imax]),
        "chi2": float(diff @ precision @ diff),
        "ndof": int(data.pk_data.size - len(names)),
        "point": theta,
        "pk_model_best": model,
        "pk_geometry_model_best": geometry_model,
        "pk_ric_correction_best": ric_correction,
        "pk_residual_best": diff,
    }


def plot_fit(data: FitData, best: dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    model = np.asarray(best["pk_model_best"], dtype="f8")
    sigma = np.sqrt(np.diag(data.covariance))
    fig, (ax, rx) = plt.subplots(2, 1, figsize=(8.4, 6.8), sharex=True, gridspec_kw={"height_ratios": [3, 1]})
    ax.errorbar(data.k_obs, data.pk_data, yerr=sigma, fmt="o", ms=4, color="black", capsize=2, label="Task43 mean P0")
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
    parser = argparse.ArgumentParser(description="Fit Task43 lightcone P(k) with a window-forward PNG model.")
    parser.add_argument("--payload", type=Path, required=True)
    parser.add_argument(
        "--ric-operator",
        type=Path,
        default=None,
        help="Optional compiled radial single-term operator; default keeps geometry-only model.",
    )
    parser.add_argument("--label", type=str, default=None)
    parser.add_argument("--output-dir", type=Path, default=PK_FIT_DIR)
    parser.add_argument("--plot-dir", type=Path, default=PK_PLOT_DIR)
    parser.add_argument("--target-evals", type=int, default=120_000)
    parser.add_argument("--nwalkers", type=int, default=18)
    parser.add_argument("--burn-fraction", type=float, default=0.10)
    parser.add_argument("--seed", type=int, default=20260706)
    parser.add_argument("--p-fixed", type=float, default=P_FIXED)
    parser.add_argument("--sn0-policy", choices=("free",), default="free")
    parser.add_argument("--sn0-scale", type=float, default=None)
    parser.add_argument("--fnl-prior", default="-500,500")
    parser.add_argument("--b1-prior", default="0.5,5")
    parser.add_argument("--sn0-prior", default="-1,1")
    parser.add_argument("--cosmology", choices=COSMOLOGY_CHOICES, default=DEFAULT_COSMOLOGY)
    parser.add_argument("--template-kmax", type=float, default=20.0)
    parser.add_argument("--covariance-floor", type=float, default=0.0)
    parser.add_argument(
        "--covariance-nmock",
        type=int,
        default=None,
        help=(
            "Number of independent mocks used for a sample covariance. If set, apply "
            "Hartlap to the likelihood precision and record Percival m1 for parameter errors."
        ),
    )
    parser.add_argument(
        "--window-theory-kmin",
        type=float,
        default=None,
        help=(
            "Diagnostic only: set theory-vector entries with window theory k below this value to zero "
            "before applying W(P). Do not confuse this with observed-bin fit kmin."
        ),
    )
    parser.add_argument(
        "--no-plot",
        action="store_true",
        help=(
            "只写 posterior samples/summary，不生成 best-fit PDF；用于不应向 "
            "plots/task43 活跃目录增加诊断图的受控 A/B。"
        ),
    )
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument(
        "--progress",
        action="store_true",
        help="Show emcee progress bar; disabled by default to keep login-node logs compact.",
    )
    return parser.parse_args()


def main() -> None:
    import emcee

    args = parse_args()
    data = FitData(Path(args.payload))
    if args.ric_operator is not None:
        data.load_radial_ric_operator(Path(args.ric_operator))
    names = param_names(str(args.sn0_policy))
    if int(args.nwalkers) < 2 * len(names):
        raise ValueError("nwalkers must be >= 2 * ndim")
    label = args.label or f"{Path(args.payload).stem}_{args.sn0_policy}_sn0"
    out_dir = Path(args.output_dir) / label
    plot_dir = Path(args.plot_dir)
    summary_path = out_dir / f"task43_pk_lightcone_{label}_fit_summary.json"
    samples_path = out_dir / f"task43_pk_lightcone_{label}_fit_samples.npz"
    plot_path = plot_dir / f"task43_pk_lightcone_{label}_fit.pdf"
    out_dir.mkdir(parents=True, exist_ok=True)
    if not bool(args.no_plot):
        plot_dir.mkdir(parents=True, exist_ok=True)
    if summary_path.exists() and samples_path.exists() and not args.overwrite:
        print(f"[skip] {summary_path}")
        return

    priors = {
        "fnl_loc": parse_prior(args.fnl_prior),
        "b1": parse_prior(args.b1_prior),
        "sn0": parse_prior(args.sn0_prior),
    }
    sn0_scale = float(args.sn0_scale) if args.sn0_scale is not None else float(data.sn0_scale)
    precision, cov_meta = make_precision(
        data.covariance,
        covariance_floor=float(args.covariance_floor),
        covariance_nmock=args.covariance_nmock,
        nparams=len(names),
    )

    task41 = load_task41()
    kmin_template = min(1.0e-5, float(np.min(data.theory_k[data.theory_k > 0.0])) * 0.5)
    k_template = np.logspace(np.log10(kmin_template), np.log10(float(args.template_kmax)), 20000)
    template, cosmology_meta = build_template_arrays(task41, k_template, z=float(data.zeff), cosmology=str(args.cosmology))

    center, opt_meta = find_initial_point(
        data,
        template,
        precision,
        names,
        p_fixed=float(args.p_fixed),
        sn0_scale=sn0_scale,
        sn0_policy=str(args.sn0_policy),
        window_theory_kmin=None if args.window_theory_kmin is None else float(args.window_theory_kmin),
        priors=priors,
    )
    nsteps = int(math.ceil(float(args.target_evals) / float(args.nwalkers)))
    burnin = int(math.floor(float(args.burn_fraction) * float(nsteps)))
    walkers = initialize_walkers(center, names, int(args.nwalkers), int(args.seed), priors)
    # emcee 还会使用 NumPy 的全局随机状态产生 proposal；除了 walker 初值
    # 之外也显式固定该状态，使不同 kth,min A/B 可以被确定性复现。
    np.random.seed(int(args.seed))
    log_prob = make_log_prob(
        data,
        template,
        precision,
        names,
        p_fixed=float(args.p_fixed),
        sn0_scale=sn0_scale,
        sn0_policy=str(args.sn0_policy),
        window_theory_kmin=None if args.window_theory_kmin is None else float(args.window_theory_kmin),
        priors=priors,
    )
    sampler = emcee.EnsembleSampler(int(args.nwalkers), len(names), log_prob)
    t0 = time.time()
    sampler.run_mcmc(walkers, nsteps, progress=bool(args.progress), skip_initial_state_check=True)
    elapsed = time.time() - t0
    chain = sampler.get_chain()
    logp = sampler.get_log_prob()
    burnin = min(burnin, max(0, chain.shape[0] - 1))
    flat_samples = chain[burnin:, :, :].reshape((-1, len(names)))
    flat_logp = logp[burnin:, :].reshape((-1,))
    params = summarize_samples(
        flat_samples,
        names,
        percival_error_factor=float(cov_meta["percival_error_factor"]),
    )
    best = best_sample(
        flat_samples,
        flat_logp,
        names,
        data=data,
        template=template,
        precision=precision,
        p_fixed=float(args.p_fixed),
        sn0_scale=sn0_scale,
        sn0_policy=str(args.sn0_policy),
        window_theory_kmin=None if args.window_theory_kmin is None else float(args.window_theory_kmin),
    )
    # Task43 活跃图目录当前只保留用户指定的两张 P(k)-2PCF 主图；低层级
    # cutoff A/B 可以仅保存机器可读 posterior，避免重新堆积诊断 PDF。
    if not bool(args.no_plot):
        plot_fit(data, best, plot_path)

    atomic_savez(
        samples_path,
        param_names=np.asarray(names),
        samples=np.asarray(flat_samples, dtype="f8"),
        log_prob=np.asarray(flat_logp, dtype="f8"),
        k_obs=np.asarray(data.k_obs, dtype="f8"),
        pk_data=np.asarray(data.pk_data, dtype="f8"),
        covariance=np.asarray(data.covariance, dtype="f8"),
        pk_model_best=np.asarray(best["pk_model_best"], dtype="f8"),
        pk_geometry_model_best=np.asarray(best["pk_geometry_model_best"], dtype="f8"),
        pk_ric_correction_best=np.asarray(best["pk_ric_correction_best"], dtype="f8"),
        burnin=np.asarray(burnin, dtype="i8"),
        nsteps=np.asarray(chain.shape[0], dtype="i8"),
        nwalkers=np.asarray(args.nwalkers, dtype="i8"),
    )
    summary = {
        "task": "task43_fit_pk_lightcone",
        "status": "done",
        "label": label,
        "payload": str(data.payload),
        "config": {
            "parameter_names": list(names),
            "free_parameters": list(names),
            "fixed_parameters": {"sigmas": SIGMAS_FIXED},
            "sn0_policy": str(args.sn0_policy),
            "sn0_policy_note": "Task43 P(k) uses free sn0, matching Task4.2 P(k); fixed sn0=0 is not a supported analysis mode.",
            "sigmas_policy": "fixed_zero_real_space",
            "p_fixed": float(args.p_fixed),
            "sn0_scale": float(sn0_scale),
            "sn0_units_note": "model adds sn0 * sn0_scale to P0 before window convolution",
            "window_forward_model": (
                "W_geom(P_theory) - W_RIC(P_theory)" if data.ric_meta is not None else "W_geom(P_theory)"
            ),
            "radial_singleterm_ric": data.ric_meta,
            "radial_singleterm_note": (
                "single IC^(rad,rad) auto term; no density-RIC cross terms; no extra global sigma_W2; "
                "the same theory vector, including free sn0*1e4, enters W_geom and W_RIC"
                if data.ric_meta is not None
                else None
            ),
            "window_theory_kmin": None if args.window_theory_kmin is None else float(args.window_theory_kmin),
            "window_theory_kmin_note": (
                "If set, this is a diagnostic low-k cutoff applied to the theory vector before W(P); "
                "it is not the observed-bin fit kmin."
            ),
            "no_ric_amr_gic": bool(data.ric_meta is None),
            "standalone_gic": False,
            "priors": {key: list(value) for key, value in priors.items()},
            "target_evals": int(args.target_evals),
            "nwalkers": int(args.nwalkers),
            "nsteps": int(chain.shape[0]),
            "burnin": int(burnin),
            "seed": int(args.seed),
            "random_seed_policy": (
                "same seed controls walker initialization and emcee global "
                "NumPy proposal state"
            ),
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
            "kmin_fit_observed": float(data.kmin_fit_observed),
            "kmax_fit": float(data.kmax_fit),
            "zeff": float(data.zeff),
            "p0": float(data.p0),
            "shotnoise_scalar": float(data.shotnoise_scalar),
        },
        "input_summary": data.summary,
        "diagnostics": {
            "acceptance_fraction_mean": float(np.mean(sampler.acceptance_fraction)),
            "acceptance_fraction_min": float(np.min(sampler.acceptance_fraction)),
            "acceptance_fraction_max": float(np.max(sampler.acceptance_fraction)),
            "elapsed_sec": float(elapsed),
        },
        "paths": {
            "summary_json": str(summary_path),
            "samples_npz": str(samples_path),
            "fit_pdf": None if bool(args.no_plot) else str(plot_path),
        },
    }
    write_json(summary_path, to_jsonable(summary))
    print(f"[write] {samples_path}")
    if not bool(args.no_plot):
        print(f"[write] {plot_path}")
    print(f"[write] {summary_path}")


if __name__ == "__main__":
    main()
