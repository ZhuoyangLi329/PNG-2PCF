#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Run a minimal Task43 MCMC closure fit for one measured xi0."""

from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
from typing import Any

for _name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_name, "1")

import emcee
import numpy as np
from scipy.optimize import minimize

from task43_config import BOX_SIZE, FIT_DIR, K_FUND, PROJECT_ROOT
from task43_fkp_zeff import load_fkp_summary, path_with_weight_tag, total_weight_from_summary
from task43_finite_mock_corrections import covariance_corrections
from task43_theory_template import COSMOLOGY_CHOICES, DEFAULT_COSMOLOGY, build_template_arrays, load_task41


K_CUT_WINDOW = 0.5


def load_covariance(
    *,
    mode: str,
    xi: np.ndarray,
    rr: np.ndarray,
    covariance_path: Path | None,
    covariance_key: str,
    fit_target: str,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Load or construct a covariance matrix."""
    if mode == "diagonal-smoke":
        # This is only a chain/debug covariance. It is intentionally conservative
        # and must not be used as the final Task43 fNL covariance.
        xi_ref = np.mean(xi, axis=0) if xi.ndim == 2 else xi
        floor = max(2.0e-4, 0.05 * float(np.nanmax(np.abs(xi_ref))))
        sigma = np.maximum(0.25 * np.abs(xi_ref), floor)
        cov = np.diag(sigma**2)
        return cov, {
            "mode": mode,
            "warning": "debug covariance only; final fNL constraints must use an external validated single-realization covariance",
            "sigma_floor": floor,
        }
    if mode == "npz":
        if covariance_path is None:
            raise ValueError("--covariance-path is required for --covariance-mode npz")
        data = np.load(covariance_path, allow_pickle=True)
        if isinstance(data, np.lib.npyio.NpzFile):
            if covariance_key == "auto":
                preferred_keys = ("covariance_single_realization", "covariance", "covariance_of_mean", "cov", "value")
                selected_key = next((key for key in preferred_keys if key in data.files), None)
            else:
                selected_key = covariance_key if covariance_key in data.files else None
            if selected_key is None:
                raise KeyError(f"no requested covariance key in {covariance_path}; available keys={data.files}")
            cov = np.asarray(data[selected_key], dtype="f8")
        else:
            selected_key = "array_or_dict"
            obj = data.item() if getattr(data, "shape", None) == () else data
            if isinstance(obj, dict) and "value" in obj:
                cov = np.asarray(obj["value"], dtype="f8")
            else:
                cov = np.asarray(obj, dtype="f8")
        meta = {"mode": mode, "path": str(covariance_path), "key": str(selected_key), "fit_target": fit_target}
        if str(selected_key) in {"covariance", "covariance_of_mean"}:
            meta["warning"] = (
                "Task43 lightcone fNL convention is covariance_single_realization. "
                "This selected key may be a covariance-of-mean builder byproduct; "
                "pass --covariance-key covariance_single_realization when available."
            )
        return cov, meta
    raise ValueError(f"unknown covariance mode: {mode}")


def build_theory_context(
    zeff: float,
    *,
    kmax: float,
    ndense: int,
    boxsize: float = BOX_SIZE,
    cosmology: str = DEFAULT_COSMOLOGY,
) -> dict[str, Any]:
    task41 = load_task41()
    kfund = 2.0 * np.pi / float(boxsize)
    volume = float(boxsize) ** 3
    k_template = np.geomspace(min(1.0e-4, kfund / 2.0), max(1.0, kmax * 1.2), 4000)
    template, cosmology_meta = build_template_arrays(task41, k_template, z=float(zeff), cosmology=str(cosmology))
    qmax = int(np.ceil((kmax / kfund) ** 2))
    nmax = int(np.ceil(np.sqrt(qmax)))
    gq = task41.gq_fft(qmax, nmax)
    g_nz, k_eff = task41.precompute_rebin_cache(gq, kfund, kmax, dk_factor=0.1)
    k_dense = np.geomspace(max(1.0e-4, kfund / 5.0), kmax, int(ndense))
    return {
        "task41": task41,
        "template": template,
        "cosmology": str(cosmology),
        "cosmology_meta": cosmology_meta,
        "g_nz": g_nz,
        "k_eff": k_eff,
        "k_dense": k_dense,
        "boxsize": float(boxsize),
        "volume": volume,
        "kfund": kfund,
    }


def _to_jsonable(obj: Any) -> Any:
    """Convert numpy/path objects to JSON-safe values."""
    if isinstance(obj, dict):
        return {str(key): _to_jsonable(value) for key, value in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_to_jsonable(value) for value in obj]
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, Path):
        return str(obj)
    if obj is None or isinstance(obj, (str, int, float, bool)):
        return obj
    return str(obj)


def _shell_j0_average(k_values: np.ndarray, s_lo: np.ndarray, s_hi: np.ndarray) -> np.ndarray:
    """Average j0(k s) over spherical shells [s_lo, s_hi]."""
    kval = np.asarray(k_values, dtype="f8")[:, None]
    s_lo = np.asarray(s_lo, dtype="f8")[None, :]
    s_hi = np.asarray(s_hi, dtype="f8")[None, :]
    shell_vol_no4pi = (s_hi**3 - s_lo**3) / 3.0
    out = np.ones((kval.shape[0], s_lo.shape[1]), dtype="f8")
    nonzero = kval[:, 0] > 0.0
    if np.any(nonzero):
        k = kval[nonzero]
        upper = np.sin(k * s_hi) - k * s_hi * np.cos(k * s_hi)
        lower = np.sin(k * s_lo) - k * s_lo * np.cos(k * s_lo)
        out[nonzero] = (upper - lower) / (k**3 * shell_vol_no4pi)
    return out


def load_or_build_formal_gic_window(
    *,
    theory: dict[str, Any],
    window_path: Path,
    random_path: Path | None,
    fkp_summary_path: Path | None,
    p0: float | None,
    n_subsample: int,
    seed: int,
    nthreads: int,
    force: bool,
) -> dict[str, Any]:
    """
    Load or build the fixed random-window W2(k) used by formal GIC.

    The cache stores W2 on the same `k_eff` grid used by FullDiscrete, so the
    GIC constant is `sigma_W^2 = sum_q g_q P(k_q) W2(k_q) / V_box`.
    """
    k_eff = np.asarray(theory["k_eff"], dtype="f8")
    if window_path.exists() and not force:
        data = np.load(window_path, allow_pickle=False)
        cached_k = np.asarray(data["k_eff"], dtype="f8")
        if cached_k.shape == k_eff.shape and np.allclose(cached_k, k_eff, rtol=0.0, atol=1.0e-14):
            meta = json.loads(str(np.asarray(data["meta_json"]).item()))
            expected_p0 = None if p0 is None else float(p0)
            cached_p0 = meta.get("weighting", {}).get("p0")
            if cached_p0 != expected_p0:
                raise ValueError(
                    f"{window_path} has cached p0={cached_p0}, but current p0={expected_p0}; "
                    "use a P0-tagged cache or --force-formal-gic-window"
                )
            return {"k_eff": cached_k, "w2": np.asarray(data["w2"], dtype="f8"), "meta": meta}
        if random_path is None:
            raise ValueError(f"{window_path} k grid does not match current theory and no random path was provided")

    if random_path is None:
        raise ValueError("--formal-gic-random-path is required when the W2 cache is absent or stale")
    if not random_path.exists():
        raise FileNotFoundError(f"missing formal GIC random catalog: {random_path}")

    from pycorr import TwoPointCorrelationFunction

    random = np.load(random_path, allow_pickle=False)
    xyz = np.column_stack(
        [
            np.asarray(random["X"], dtype="f8"),
            np.asarray(random["Y"], dtype="f8"),
            np.asarray(random["Zcart"], dtype="f8"),
        ]
    )
    base_weight = np.asarray(random["WEIGHT"], dtype="f8") if "WEIGHT" in random.files else np.ones(xyz.shape[0], dtype="f8")
    fkp_summary = load_fkp_summary(fkp_summary_path) if fkp_summary_path is not None else None
    if fkp_summary is not None:
        if "Z" not in random.files:
            raise KeyError(f"{random_path} has no Z column needed for FKP weighting")
        random_z = np.asarray(random["Z"], dtype="f8")
    else:
        random_z = np.zeros(xyz.shape[0], dtype="f8")
    weight = total_weight_from_summary(random_z, base_weight, fkp_summary, p0=p0)
    n_random = int(xyz.shape[0])
    n_sub = n_random if int(n_subsample) <= 0 else min(n_random, int(n_subsample))
    rng = np.random.default_rng(int(seed))
    if n_sub < n_random:
        choice = rng.choice(n_random, size=n_sub, replace=False)
        xyz_sub = xyz[choice]
        weight_sub = weight[choice]
    else:
        xyz_sub = xyz
        weight_sub = weight

    spans = np.max(xyz, axis=0) - np.min(xyz, axis=0)
    smax_diag = float(np.sqrt(np.sum(spans**2)))
    s_edges = np.unique(
        np.concatenate(
            [
                np.arange(0.5, 80.0, 2.0),
                np.arange(80.0, 600.0, 10.0),
                np.arange(600.0, 2000.0, 25.0),
                np.arange(2000.0, smax_diag + 200.0, 100.0),
            ]
        )
    )
    if s_edges[-1] < smax_diag:
        s_edges = np.append(s_edges, smax_diag + 100.0)

    result = TwoPointCorrelationFunction(
        mode="s",
        edges=s_edges,
        data_positions1=xyz_sub.T,
        data_weights1=weight_sub,
        randoms_positions1=xyz_sub.T,
        randoms_weights1=weight_sub,
        estimator="natural",
        nthreads=int(nthreads),
    )
    rr_raw = np.asarray(result.R1R2.wcounts, dtype="f8")
    sumw = float(np.sum(weight_sub))
    sumw2 = float(np.sum(weight_sub * weight_sub))
    pair_norm = float(sumw * sumw - sumw2)
    if pair_norm <= 0.0:
        raise RuntimeError("formal GIC weighted pair normalization is non-positive")
    pair_prob = rr_raw / pair_norm
    coverage = float(np.sum(pair_prob))
    if coverage <= 0.0:
        raise RuntimeError("formal GIC random pair coverage is zero")
    pair_prob_norm = pair_prob / coverage

    w2 = np.zeros_like(k_eff)
    active = np.flatnonzero((k_eff > 0.0) & (k_eff <= K_CUT_WINDOW))
    batch_size = 512
    for start in range(0, active.size, batch_size):
        ids = active[start : start + batch_size]
        avg_j0 = _shell_j0_average(k_eff[ids], s_edges[:-1], s_edges[1:])
        w2[ids] = avg_j0 @ pair_prob_norm
    w2[k_eff > K_CUT_WINDOW] = 0.0

    meta = {
        "method": "formal_gic_random_pair_kernel",
        "formula": "sigma_W2 = sum_q g_q P(k_q) W2(k_q) / V_box",
        "theory_boxsize": float(theory["boxsize"]),
        "theory_volume": float(theory["volume"]),
        "theory_kfund": float(theory["kfund"]),
        "random_path": str(random_path),
        "n_random_total": n_random,
        "n_subsample": int(n_sub),
        "seed": int(seed),
        "nthreads": int(nthreads),
        "smax_diag": smax_diag,
        "n_s_edges": int(s_edges.size),
        "coverage": coverage,
        "pair_norm": pair_norm,
        "k_cut_window": float(K_CUT_WINDOW),
        "w2_kmin": float(w2[0]),
        "w2_min_active": float(np.min(w2[active])) if active.size else float("nan"),
        "w2_max_active": float(np.max(w2[active])) if active.size else float("nan"),
        "window_path": str(window_path),
        "weighting": {
            "scheme": "WEIGHT_TOTAL=WEIGHT*WEIGHT_FKP" if fkp_summary is not None else "catalog_WEIGHT",
            "p0": None if p0 is None else float(p0),
            "fkp_summary_path": None if fkp_summary_path is None else str(fkp_summary_path),
            "weight_min": float(np.min(weight_sub)),
            "weight_max": float(np.max(weight_sub)),
            "weight_sum": sumw,
            "weight_sum2": sumw2,
        },
    }
    window_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        window_path,
        k_eff=k_eff,
        w2=w2,
        s_edges=s_edges,
        rr_raw=rr_raw,
        pair_prob=pair_prob,
        meta_json=np.asarray(json.dumps(_to_jsonable(meta), ensure_ascii=False)),
    )
    return {"k_eff": k_eff, "w2": w2, "meta": meta}


def model_xi(
    theta: np.ndarray,
    s: np.ndarray,
    rr: np.ndarray,
    theory: dict[str, Any],
    *,
    model: str,
    p_fixed: float,
    sn0_fixed: float,
) -> np.ndarray:
    fnl_loc, b1 = theta
    task41 = theory["task41"]
    pk = task41.evaluate_realspace_png_pk(
        theory["k_dense"],
        theory["template"],
        fnl_loc=float(fnl_loc),
        b1=float(b1),
        p_fixed=float(p_fixed),
        sn0=float(sn0_fixed),
    )
    xi = task41.fast_discrete_xi(s, theory["g_nz"], theory["k_eff"], theory["k_dense"], pk)
    if model == "no_gic":
        return xi
    if model == "ic_constant":
        weights = np.asarray(rr, dtype="f8")
        const = float(np.sum(weights * xi) / np.sum(weights))
        return xi - const
    raise ValueError(f"unknown model: {model}")


def summarize_chain(
    samples: np.ndarray,
    log_prob: np.ndarray,
    names: list[str],
    *,
    percival_error_factor: float = 1.0,
) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for i, name in enumerate(names):
        q16, q50, q84 = np.percentile(samples[:, i], [16, 50, 84])
        err_low = float(q50 - q16)
        err_high = float(q84 - q50)
        out[name] = {
            "q16": float(q16),
            "q50": float(q50),
            "q84": float(q84),
            "mean": float(np.mean(samples[:, i])),
            "std": float(np.std(samples[:, i])),
            "err_low": err_low,
            "err_high": err_high,
            "err_low_percival": float(err_low * percival_error_factor),
            "err_high_percival": float(err_high * percival_error_factor),
            "std_percival": float(np.std(samples[:, i]) * percival_error_factor),
        }
    imax = int(np.argmax(log_prob))
    out["map"] = {name: float(samples[imax, i]) for i, name in enumerate(names)}
    out["map"]["log_prob"] = float(log_prob[imax])
    return out


def load_radial_ric_operator(
    path: Path,
    *,
    theory: dict[str, Any],
    selected_s: np.ndarray,
) -> dict[str, Any]:
    """读取并严格对齐 radial single-term 2PCF basis。

    新 operator 已在与当前 FullDiscrete 完全相同的 k_eff/volume/cosmology
    口径上预投影了三个 PNG basis。这里不重新估计 window，只检查 k grid 和
    s-bin center 后选择拟合范围，防止误把不同 L、不同 binning 的 cache 混用。
    """
    with np.load(path, allow_pickle=False) as data:
        cached_k = np.asarray(data["k_eff"], dtype="f8")
        target_edges = np.asarray(data["target_s_edges"], dtype="f8")
        cached_volume = float(np.asarray(data["volume"]).item())
        meta = json.loads(str(np.asarray(data["meta_json"]).item()))
        cached_basis = {
            "pk_dd": np.asarray(data["xi_basis_pk_dd"], dtype="f8"),
            "alpha_pk_dd": np.asarray(data["xi_basis_alpha_pk_dd"], dtype="f8"),
            "alpha2_pk_dd": np.asarray(data["xi_basis_alpha2_pk_dd"], dtype="f8"),
        }
    if cached_k.shape != np.asarray(theory["k_eff"]).shape or not np.allclose(
        cached_k, theory["k_eff"], rtol=0.0, atol=1.0e-14
    ):
        raise ValueError(f"radial RIC operator {path} 的 k_eff 与当前 FullDiscrete grid 不同")
    if not np.isclose(cached_volume, float(theory["volume"]), rtol=0.0, atol=1.0e-6):
        raise ValueError(f"radial RIC operator volume={cached_volume} 与当前 {theory['volume']} 不同")
    centers = 0.5 * (target_edges[:-1] + target_edges[1:])
    indices = []
    for value in np.asarray(selected_s, dtype="f8"):
        found = np.flatnonzero(np.isclose(centers, value, rtol=0.0, atol=1.0e-10))
        if found.size != 1:
            raise ValueError(f"radial RIC operator 中找不到唯一 s center={value}")
        indices.append(int(found[0]))
    selected = {name: values[np.asarray(indices, dtype="i8")] for name, values in cached_basis.items()}
    return {
        "path": str(path),
        "basis": selected,
        "meta": meta,
        "selected_indices": indices,
        "selected_s": [float(v) for v in selected_s],
    }


def run_one_model(
    *,
    model: str,
    s: np.ndarray,
    s_bin_edges: np.ndarray | None,
    xi: np.ndarray,
    rr: np.ndarray,
    cov: np.ndarray,
    theory: dict[str, Any],
    formal_gic_window: dict[str, Any] | None,
    radial_ric_operator: dict[str, Any] | None,
    p_fixed: float,
    sn0_fixed: float,
    free_sn0: bool,
    sn0_prior: tuple[float, float],
    png_order: str,
    xi_kernel: str,
    nwalkers: int,
    nsteps: int,
    burnin: int,
    seed: int,
    precision_scale: float = 1.0,
    percival_error_factor: float = 1.0,
) -> tuple[dict[str, Any], np.ndarray, np.ndarray]:
    precision = float(precision_scale) * np.linalg.pinv(cov, rcond=1.0e-10)
    task41 = theory["task41"]
    xi_kernel_mode = str(xi_kernel)
    if xi_kernel_mode == "center":
        arg = np.outer(theory["k_eff"], s)
        xi_kernel_matrix = np.ones_like(arg)
        nonzero = arg != 0.0
        xi_kernel_matrix[nonzero] = np.sin(arg[nonzero]) / arg[nonzero]
        xi_kernel_meta = {"mode": xi_kernel_mode, "description": "j0 evaluated at bin centers"}
    elif xi_kernel_mode == "shell-averaged":
        if s_bin_edges is None:
            raise ValueError("--xi-kernel shell-averaged requires s_edges in the xi summary")
        edges = np.asarray(s_bin_edges, dtype="f8")
        if edges.shape != (s.size, 2):
            raise ValueError(f"s_bin_edges shape {edges.shape} does not match selected bins {s.size}")
        xi_kernel_matrix = _shell_j0_average(theory["k_eff"], edges[:, 0], edges[:, 1])
        xi_kernel_meta = {
            "mode": xi_kernel_mode,
            "description": "volume-averaged j0 over each radial shell",
            "s_lower_edges": [float(v) for v in edges[:, 0]],
            "s_upper_edges": [float(v) for v in edges[:, 1]],
        }
    else:
        raise ValueError(f"unknown xi_kernel: {xi_kernel_mode}")

    def xi_from_pk_dense(pk_dense: np.ndarray) -> np.ndarray:
        weight = theory["g_nz"] * task41.interp_logk(theory["k_eff"], theory["k_dense"], pk_dense)
        return (weight @ xi_kernel_matrix) / float(theory["volume"])

    pk_dd_dense = task41.interp_logk(theory["k_dense"], theory["template"]["k"], theory["template"]["pk_dd"])
    alpha_dense = task41.interp_logk(theory["k_dense"], theory["template"]["k"], theory["template"]["alpha"])
    xi_pk_dd = xi_from_pk_dense(pk_dd_dense)
    xi_alpha_pk_dd = xi_from_pk_dense(alpha_dense * pk_dd_dense)
    xi_alpha2_pk_dd = xi_from_pk_dense(alpha_dense * alpha_dense * pk_dd_dense)
    xi_sn0 = xi_from_pk_dense(np.ones_like(theory["k_dense"])) if (free_sn0 or sn0_fixed != 0.0) else np.zeros_like(xi_pk_dd)

    # radial operator 保存的是正的 IC^(rad,rad) auto response；single-term
    # approximation 在 evaluate_model 中统一做减法。2PCF 主线固定 sn0=0，
    # 因此这里刻意不为 radial branch 发明额外 stochastic prescription。
    if radial_ric_operator is not None:
        radial_basis = radial_ric_operator["basis"]
        radial_pk_dd = np.asarray(radial_basis["pk_dd"], dtype="f8")
        radial_alpha_pk_dd = np.asarray(radial_basis["alpha_pk_dd"], dtype="f8")
        radial_alpha2_pk_dd = np.asarray(radial_basis["alpha2_pk_dd"], dtype="f8")
        if any(array.shape != xi_pk_dd.shape for array in (radial_pk_dd, radial_alpha_pk_dd, radial_alpha2_pk_dd)):
            raise ValueError("radial RIC basis shape 与当前 2PCF data vector 不一致")
    else:
        radial_pk_dd = np.zeros_like(xi_pk_dd)
        radial_alpha_pk_dd = np.zeros_like(xi_pk_dd)
        radial_alpha2_pk_dd = np.zeros_like(xi_pk_dd)

    def sigma_from_pk_dense(pk_dense: np.ndarray) -> float:
        if formal_gic_window is None:
            return 0.0
        w2 = np.asarray(formal_gic_window["w2"], dtype="f8")
        p_eff = task41.interp_logk(theory["k_eff"], theory["k_dense"], pk_dense)
        return float(np.sum(theory["g_nz"] * p_eff * w2) / float(theory["volume"]))

    sigma_pk_dd = sigma_from_pk_dense(pk_dd_dense)
    sigma_alpha_pk_dd = sigma_from_pk_dense(alpha_dense * pk_dd_dense)
    sigma_alpha2_pk_dd = sigma_from_pk_dense(alpha_dense * alpha_dense * pk_dd_dense)
    sigma_sn0 = sigma_from_pk_dense(np.ones_like(theory["k_dense"])) if (free_sn0 or sn0_fixed != 0.0) else 0.0

    if str(png_order) not in {"full", "linear"}:
        raise ValueError(f"unknown png_order: {png_order}")

    def coefficients(theta: np.ndarray) -> tuple[float, float, float, float]:
        fnl_loc, b1 = theta[:2]
        sn0 = float(theta[2]) if free_sn0 else float(sn0_fixed)
        bphi = 2.0 * 1.686 * (float(b1) - float(p_fixed))
        fnl_bphi = float(fnl_loc) * bphi
        quadratic = fnl_bphi**2 if str(png_order) == "full" else 0.0
        return float(b1) ** 2, 2.0 * float(b1) * fnl_bphi, quadratic, sn0

    def base_model_xi(theta: np.ndarray) -> np.ndarray:
        c0, c1, c2, csn = coefficients(theta)
        return (
            c0 * xi_pk_dd
            + c1 * xi_alpha_pk_dd
            + c2 * xi_alpha2_pk_dd
            + csn * xi_sn0
        )

    def formal_sigma_w2(theta: np.ndarray) -> float:
        c0, c1, c2, csn = coefficients(theta)
        return float(c0 * sigma_pk_dd + c1 * sigma_alpha_pk_dd + c2 * sigma_alpha2_pk_dd + csn * sigma_sn0)

    def radial_auto_response(theta: np.ndarray) -> np.ndarray:
        """按与 no-IC model 相同的 full-PNG 系数组合 radial correction。"""
        c0, c1, c2, csn = coefficients(theta)
        if csn != 0.0:
            raise ValueError("Task43 radial single-term 2PCF 当前只支持物理主线 fixed sn0=0")
        return c0 * radial_pk_dd + c1 * radial_alpha_pk_dd + c2 * radial_alpha2_pk_dd

    def evaluate_model(theta: np.ndarray) -> np.ndarray:
        base_xi = base_model_xi(theta)
        if model == "no_gic":
            return base_xi
        if model == "formal_gic":
            if formal_gic_window is None:
                raise ValueError("formal_gic requires a fixed W2 window cache")
            return base_xi - formal_sigma_w2(theta)
        if model == "radial_singleterm":
            if radial_ric_operator is None:
                raise ValueError("radial_singleterm requires --radial-ric-operator")
            # radial normalization 已自动包含 global normalization；此处绝不
            # 再减 formal_sigma_w2，避免 global IC double counting。
            return base_xi - radial_auto_response(theta)
        if xi.ndim == 1:
            weights_1d = np.asarray(rr, dtype="f8")
            const_1d = float(np.sum(weights_1d * base_xi) / np.sum(weights_1d))
            return base_xi - const_1d
        weights = np.asarray(rr, dtype="f8")
        const = np.sum(weights * base_xi[None, :], axis=1) / np.sum(weights, axis=1)
        return base_xi[None, :] - const[:, None]

    def log_prior(theta: np.ndarray) -> float:
        fnl_loc, b1 = theta[:2]
        if not (-500.0 <= fnl_loc <= 500.0 and 0.2 <= b1 <= 10.0):
            return -np.inf
        if free_sn0:
            sn0 = float(theta[2])
            if not (float(sn0_prior[0]) <= sn0 <= float(sn0_prior[1])):
                return -np.inf
            return 0.0
        return 0.0

    def chi2(theta: np.ndarray) -> float:
        diff = xi - evaluate_model(theta)
        if diff.ndim == 1:
            return float(diff @ precision @ diff)
        return float(np.einsum("ij,jk,ik->", diff, precision, diff))

    def chi2_per_realization(theta: np.ndarray) -> np.ndarray:
        diff = xi - evaluate_model(theta)
        if diff.ndim == 1:
            return np.array([float(diff @ precision @ diff)])
        return np.einsum("ij,jk,ik->i", diff, precision, diff)

    def log_prob(theta: np.ndarray) -> float:
        lp = log_prior(theta)
        if not np.isfinite(lp):
            return -np.inf
        return lp - 0.5 * chi2(theta)

    def objective(theta: np.ndarray) -> float:
        lp = log_prior(theta)
        if not np.isfinite(lp):
            return 1.0e100
        return chi2(theta)

    x0 = np.array([0.0, 3.0, float(sn0_fixed)], dtype="f8") if free_sn0 else np.array([0.0, 3.0], dtype="f8")
    opt = minimize(objective, x0=x0, method="Nelder-Mead")
    center = np.asarray(opt.x if opt.success else x0, dtype="f8")
    center[0] = np.clip(center[0], -400.0, 400.0)
    center[1] = np.clip(center[1], 0.5, 8.0)
    if free_sn0:
        center[2] = np.clip(center[2], float(sn0_prior[0]), float(sn0_prior[1]))

    rng = np.random.default_rng(seed)
    if free_sn0:
        sn0_width = max(float(sn0_prior[1]) - float(sn0_prior[0]), 1.0)
        scale = np.array([10.0, 0.05, min(500.0, 0.02 * sn0_width)], dtype="f8")
    else:
        scale = np.array([10.0, 0.05], dtype="f8")
    p0 = center[None, :] + rng.normal(scale=scale, size=(nwalkers, center.size))
    p0[:, 0] = np.clip(p0[:, 0], -490.0, 490.0)
    p0[:, 1] = np.clip(p0[:, 1], 0.25, 9.5)
    if free_sn0:
        p0[:, 2] = np.clip(p0[:, 2], float(sn0_prior[0]) + 1.0e-8, float(sn0_prior[1]) - 1.0e-8)
    # emcee 的 stretch-move proposal 还会读取 NumPy 全局随机状态；显式固定
    # 它，保证 dchi=1/2/4、10k/50k/200k 的 matched posterior A/B 不被
    # 不同 proposal realization 污染。
    np.random.seed(int(seed))
    sampler = emcee.EnsembleSampler(nwalkers, center.size, log_prob)
    sampler.run_mcmc(p0, nsteps, progress=False)
    chain = sampler.get_chain(discard=burnin, flat=True)
    logp = sampler.get_log_prob(discard=burnin, flat=True)
    param_names = ["fnl_loc", "b1", "sn0"] if free_sn0 else ["fnl_loc", "b1"]
    summary = summarize_chain(
        chain,
        logp,
        param_names,
        percival_error_factor=float(percival_error_factor),
    )
    map_theta = np.array([summary["map"][name] for name in param_names], dtype="f8")
    prediction_map = np.asarray(evaluate_model(map_theta), dtype="f8")
    residual_map = np.asarray(xi - prediction_map, dtype="f8")
    chi2_each = chi2_per_realization(map_theta)
    summary.update(
        {
            "model": model,
            "parameter_names": param_names,
            "priors": {
                "fnl_loc": [-500.0, 500.0],
                "b1": [0.2, 10.0],
                "sn0": [float(sn0_prior[0]), float(sn0_prior[1])] if free_sn0 else None,
            },
            "png_order": str(png_order),
            "xi_kernel": xi_kernel_meta,
            "data": {
                "fit_target": "all-realizations" if xi.ndim == 2 else "mean",
                "nreal": int(xi.shape[0]) if xi.ndim == 2 else 1,
                "nbins": int(s.size),
                "data_vector_size": int(xi.size),
                "chi2_map_total": float(np.sum(chi2_each)),
                "chi2_map_per_realization": [float(v) for v in chi2_each],
                "prediction_map": prediction_map.tolist(),
                "prediction_map_shape": [int(v) for v in prediction_map.shape],
                "residual_map": residual_map.tolist(),
                "residual_map_shape": [int(v) for v in residual_map.shape],
            },
            "optimizer": {
                "success": bool(opt.success),
                "x": [float(v) for v in np.asarray(opt.x).ravel()],
                "chi2": float(opt.fun),
            },
            "mcmc": {
                "nwalkers": int(nwalkers),
                "nsteps": int(nsteps),
                "burnin": int(burnin),
                "seed": int(seed),
                "random_seed_policy": "same seed controls walker initialization and emcee global random state",
                "nsamples": int(chain.shape[0]),
                "ndim": int(chain.shape[1]),
                "mean_acceptance_fraction": float(np.mean(sampler.acceptance_fraction)),
            },
        }
    )
    if model == "formal_gic":
        opt_theta = np.asarray(opt.x, dtype="f8")
        map_theta = np.asarray([summary["map"][name] for name in param_names], dtype="f8")
        summary["formal_gic"] = {
            "sigma_w2_optimizer": float(formal_sigma_w2(opt_theta)),
            "sigma_w2_map": float(formal_sigma_w2(map_theta)),
            "window_meta": None if formal_gic_window is None else formal_gic_window.get("meta", {}),
        }
    if model == "radial_singleterm":
        opt_theta = np.asarray(opt.x, dtype="f8")
        map_theta = np.asarray([summary["map"][name] for name in param_names], dtype="f8")
        correction_map = radial_auto_response(map_theta)
        summary["radial_singleterm"] = {
            "approximation": "single IC^(rad,rad) auto term; density-RIC cross terms omitted",
            "fit_formula": "xi_model = xi_noIC - IC_rad_rad",
            "extra_global_sigma_w2": False,
            "correction_optimizer": [float(v) for v in radial_auto_response(opt_theta)],
            "correction_map": [float(v) for v in correction_map],
            "operator": {key: value for key, value in radial_ric_operator.items() if key != "basis"},
        }
    return summary, chain, logp


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--xi-path", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=FIT_DIR / "smoke_topN20000_x5")
    parser.add_argument("--fit-target", choices=["all-realizations", "mean"], default="all-realizations")
    parser.add_argument("--covariance-mode", choices=["diagonal-smoke", "npz"], default="diagonal-smoke")
    parser.add_argument("--covariance-path", type=Path, default=None)
    parser.add_argument("--covariance-key", type=str, default="auto")
    parser.add_argument(
        "--covariance-nmock",
        type=int,
        default=None,
        help=(
            "Number of independent mocks used for a sample covariance. If set, apply "
            "Hartlap to the likelihood precision and record Percival m1 for parameter errors."
        ),
    )
    parser.add_argument("--rmin", type=float, default=80.0)
    parser.add_argument("--rmax", type=float, default=350.0)
    parser.add_argument("--models", type=str, default="no_gic,formal_gic")
    parser.add_argument(
        "--radial-ric-operator",
        type=Path,
        default=None,
        help="Compiled task43_ric_operator_*.npz used by model=radial_singleterm.",
    )
    parser.add_argument("--formal-gic-window-path", type=Path, default=None)
    parser.add_argument("--formal-gic-random-path", type=Path, default=None)
    parser.add_argument("--fkp-summary", type=Path, default=None)
    parser.add_argument("--p0", type=float, default=None)
    parser.add_argument("--output-tag", type=str, default=None)
    parser.add_argument("--formal-gic-window-nsub", type=int, default=100000)
    parser.add_argument("--formal-gic-window-seed", type=int, default=20260629)
    parser.add_argument("--formal-gic-window-nthreads", type=int, default=1)
    parser.add_argument("--force-formal-gic-window", action="store_true")
    parser.add_argument("--cosmology", choices=COSMOLOGY_CHOICES, default=DEFAULT_COSMOLOGY)
    parser.add_argument("--theory-boxsize", type=float, default=None)
    parser.add_argument("--p-fixed", type=float, default=1.1)
    parser.add_argument("--png-order", choices=["full", "linear"], default="full")
    parser.add_argument("--sn0-fixed", type=float, default=0.0)
    parser.add_argument("--free-sn0", action="store_true")
    parser.add_argument("--sn0-prior", type=float, nargs=2, default=(-10000.0, 10000.0))
    parser.add_argument("--kmax", type=float, default=5.0)
    parser.add_argument("--ndense", type=int, default=60000)
    parser.add_argument(
        "--xi-kernel",
        choices=["center", "shell-averaged"],
        default="center",
        help="Projection kernel for xi_model: bin-center j0(ks), or shell-averaged j0 over each s bin.",
    )
    parser.add_argument("--nwalkers", type=int, default=32)
    parser.add_argument("--nsteps", type=int, default=800)
    parser.add_argument("--burnin", type=int, default=200)
    parser.add_argument("--seed", type=int, default=20260628)
    args = parser.parse_args()

    ndim = 3 if bool(args.free_sn0) else 2
    if int(args.nsteps) <= 0:
        raise ValueError("--nsteps must be positive")
    if int(args.burnin) < 0 or int(args.burnin) >= int(args.nsteps):
        raise ValueError("--burnin must satisfy 0 <= burnin < nsteps")
    if int(args.nwalkers) < 2 * ndim:
        raise ValueError(f"--nwalkers must be at least 2*ndim={2 * ndim} for ndim={ndim}")

    if args.fkp_summary is not None and args.p0 is None:
        raise ValueError("--p0 is required when --fkp-summary is provided")
    if args.fkp_summary is None and args.p0 is not None and float(args.p0) != 0.0:
        raise ValueError("nonzero --p0 requires --fkp-summary")
    data = np.load(args.xi_path)
    s_all = np.asarray(data["s"], dtype="f8")
    s_edges_all = np.asarray(data["s_edges"], dtype="f8") if "s_edges" in data.files else None
    xi_mean_all = np.asarray(data["xi0"], dtype="f8")
    rr_mean_all = np.asarray(data["RR"], dtype="f8")
    mask = (s_all >= float(args.rmin)) & (s_all <= float(args.rmax))
    s = s_all[mask]
    if str(args.xi_kernel) == "shell-averaged":
        if s_edges_all is None:
            raise KeyError(f"{args.xi_path} has no s_edges needed for --xi-kernel shell-averaged")
        if s_edges_all.size != s_all.size + 1:
            raise ValueError(f"s_edges size {s_edges_all.size} is inconsistent with s size {s_all.size}")
        s_bin_edges = np.column_stack([s_edges_all[:-1][mask], s_edges_all[1:][mask]])
    else:
        s_bin_edges = None
    if args.fit_target == "all-realizations":
        if "xi0_all" not in data.files:
            raise KeyError(f"{args.xi_path} has no xi0_all; run task43_summarize_xi.py on multiple realizations")
        if "RR_all" not in data.files:
            raise KeyError(f"{args.xi_path} has no RR_all; run task43_summarize_xi.py with per-realization counts")
        xi = np.asarray(data["xi0_all"], dtype="f8")[:, mask]
        rr = np.asarray(data["RR_all"], dtype="f8")[:, mask]
        if xi.ndim != 2 or rr.shape != xi.shape:
            raise ValueError(f"unexpected all-realization shapes: xi={xi.shape} rr={rr.shape}")
        nreal_fit = int(xi.shape[0])
    else:
        xi = xi_mean_all[mask]
        rr = rr_mean_all[mask]
        nreal_fit = int(np.asarray(data["nreal"]).item()) if "nreal" in data.files else 1
    cov_full, cov_meta = load_covariance(
        mode=args.covariance_mode,
        xi=xi,
        rr=rr,
        covariance_path=args.covariance_path,
        covariance_key=str(args.covariance_key),
        fit_target=str(args.fit_target),
    )
    cov = cov_full[np.ix_(mask, mask)] if cov_full.shape == (s_all.size, s_all.size) else cov_full
    if cov.shape != (s.size, s.size):
        raise ValueError(f"covariance shape {cov.shape} does not match selected data size {s.size}")
    if args.covariance_nmock is None:
        finite_mock = {
            "hartlap": 1.0,
            "percival_m1_variance": 1.0,
            "percival_error_factor": 1.0,
            "finite_mock_correction": False,
        }
    else:
        finite_mock = covariance_corrections(
            nmock=int(args.covariance_nmock), ndata=int(s.size), nparams=int(ndim)
        )
        finite_mock["finite_mock_correction"] = True
    cov_meta.update(finite_mock)
    cov_meta["precision"] = (
        "Hartlap * pinv finite-mock sample covariance"
        if args.covariance_nmock is not None
        else "pinv input covariance"
    )

    zeff = float(data["zeff"])
    if args.theory_boxsize is not None:
        boxsize = float(args.theory_boxsize)
        boxsize_source = "--theory-boxsize"
    elif "boxsize" in data.files:
        boxsize = float(np.asarray(data["boxsize"]).item())
        boxsize_source = "xi_npz:boxsize"
    else:
        boxsize = float(BOX_SIZE)
        boxsize_source = "task43_config.BOX_SIZE"
    theory = build_theory_context(
        zeff,
        kmax=float(args.kmax),
        ndense=int(args.ndense),
        boxsize=boxsize,
        cosmology=str(args.cosmology),
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)

    model_names = [item.strip() for item in str(args.models).split(",") if item.strip()]
    allowed_models = {"no_gic", "formal_gic", "ic_constant", "radial_singleterm"}
    unknown_models = sorted(set(model_names) - allowed_models)
    if unknown_models:
        raise ValueError(f"unknown model(s): {unknown_models}; allowed={sorted(allowed_models)}")
    formal_gic_window = None
    if "formal_gic" in model_names:
        window_path = args.formal_gic_window_path
        if window_path is None:
            window_path = args.output_dir / "task43_formal_gic_window_w2.npz"
            window_path = path_with_weight_tag(window_path, p0=args.p0, output_tag=args.output_tag)
        formal_gic_window = load_or_build_formal_gic_window(
            theory=theory,
            window_path=window_path,
            random_path=args.formal_gic_random_path,
            fkp_summary_path=args.fkp_summary,
            p0=args.p0,
            n_subsample=int(args.formal_gic_window_nsub),
            seed=int(args.formal_gic_window_seed),
            nthreads=int(args.formal_gic_window_nthreads),
            force=bool(args.force_formal_gic_window),
        )
    radial_ric_operator = None
    if "radial_singleterm" in model_names:
        if args.radial_ric_operator is None:
            raise ValueError("models 包含 radial_singleterm 时必须提供 --radial-ric-operator")
        if bool(args.free_sn0) or float(args.sn0_fixed) != 0.0:
            raise ValueError("Task43 radial single-term 2PCF 正式口径固定 sn0=0")
        radial_ric_operator = load_radial_ric_operator(
            Path(args.radial_ric_operator),
            theory=theory,
            selected_s=s,
        )

    summaries: list[dict[str, Any]] = []
    for imodel, model in enumerate(model_names):
        summary, chain, logp = run_one_model(
            model=model,
            s=s,
            s_bin_edges=s_bin_edges,
            xi=xi,
            rr=rr,
            cov=cov,
            theory=theory,
            formal_gic_window=formal_gic_window,
            radial_ric_operator=radial_ric_operator,
            p_fixed=float(args.p_fixed),
            sn0_fixed=float(args.sn0_fixed),
            free_sn0=bool(args.free_sn0),
            sn0_prior=(float(args.sn0_prior[0]), float(args.sn0_prior[1])),
            png_order=str(args.png_order),
            xi_kernel=str(args.xi_kernel),
            nwalkers=int(args.nwalkers),
            nsteps=int(args.nsteps),
            burnin=int(args.burnin),
            seed=int(args.seed) + imodel,
            precision_scale=float(finite_mock["hartlap"]),
            percival_error_factor=float(finite_mock["percival_error_factor"]),
        )
        np.savez_compressed(
            args.output_dir / f"task43_mcmc_{model}_samples.npz",
            samples=chain,
            log_prob=logp,
            prediction_map=np.asarray(summary["data"]["prediction_map"], dtype="f8"),
            residual_map=np.asarray(summary["data"]["residual_map"], dtype="f8"),
        )
        summaries.append(summary)

    out = {
        "status": "done",
        "task": "task43",
        "xi_path": str(args.xi_path),
        "phase": str(data["phase"]),
        "sim_name": str(data["sim_name"]),
        "zeff": zeff,
        "fit_target": str(args.fit_target),
        "fit_range": {
            "rmin": float(args.rmin),
            "rmax": float(args.rmax),
            "nbins": int(s.size),
            "nreal": nreal_fit,
            "data_vector_size": int(xi.size),
            "s_centers": [float(v) for v in s],
        },
        "covariance": cov_meta,
        "theory": {
            "kmax": float(args.kmax),
            "ndense": int(args.ndense),
            "boxsize": float(theory["boxsize"]),
            "boxsize_source": boxsize_source,
            "volume": float(theory["volume"]),
            "kfund": float(theory["kfund"]),
            "k_eff_min": float(np.min(theory["k_eff"])),
            "k_eff_max": float(np.max(theory["k_eff"])),
            "k_eff_size": int(np.asarray(theory["k_eff"]).size),
            "k_dense_min": float(np.min(theory["k_dense"])),
            "k_dense_max": float(np.max(theory["k_dense"])),
            "k_dense_size": int(np.asarray(theory["k_dense"]).size),
            "p_fixed": float(args.p_fixed),
            "png_order": str(args.png_order),
            "xi_kernel": str(args.xi_kernel),
            "sn0_fixed": float(args.sn0_fixed),
            "sn0_policy": "free" if bool(args.free_sn0) else "fixed",
            "sn0_prior": [float(v) for v in args.sn0_prior],
            "cosmology": str(args.cosmology),
            "cosmology_meta": theory.get("cosmology_meta", {}),
            "models": model_names,
            "formal_gic_window": None if formal_gic_window is None else formal_gic_window.get("meta", {}),
            "radial_ric_operator": (
                None
                if radial_ric_operator is None
                else {key: value for key, value in radial_ric_operator.items() if key != "basis"}
            ),
            "weighting": {
                "p0": None if args.p0 is None else float(args.p0),
                "fkp_summary_path": None if args.fkp_summary is None else str(args.fkp_summary),
                "output_tag": args.output_tag,
            },
        },
        "models": summaries,
    }
    summary_path = args.output_dir / "task43_minimal_closure_mcmc_summary.json"
    summary_path.write_text(json.dumps(out, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"[done] wrote {summary_path}")
    for summary in summaries:
        fnl = summary["fnl_loc"]
        print(f"[{summary['model']}] fNL={fnl['q50']:.2f} -{fnl['q50']-fnl['q16']:.2f} +{fnl['q84']-fnl['q50']:.2f}")


if __name__ == "__main__":
    main()
