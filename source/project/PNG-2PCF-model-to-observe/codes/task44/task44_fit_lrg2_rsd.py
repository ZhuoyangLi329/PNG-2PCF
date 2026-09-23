#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Fit Task44 LRG-bin xi0 with an RSD monopole PNG model."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
from typing import Any

for _name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_name, "1")

import emcee
import numpy as np
from scipy.optimize import minimize
from scipy.special import spherical_jn

from task44_config import (
    BOX_SIZE,
    DEFAULT_RANDOM_CATALOG,
    DEFAULT_XI,
    DEFAULT_ZEFF,
    FIT_DIR,
    P_FIXED_DEFAULT,
    P0_DEFAULT,
    SAMPLE,
    SAMPLE_CONFIGS,
    lrg_bin_config,
    write_json,
)
from task44_rsd_theory import DELTA_C, DEFAULT_COSMOLOGY, FOG_MODELS, build_theory_context, fog_description, mu_moments


K_CUT_WINDOW = 0.5
PINV_RCOND = 1.0e-10
ALLOWED_MODELS = {"no_gic", "formal_gic"}
PRECISION_MODES = {"inverse-covariance", "rascalc-corrected"}


def file_sha256(path: Path) -> str:
    """Return a content hash suitable for binding fit outputs to an input file."""

    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def parse_ells(text: str) -> tuple[int, ...]:
    values = tuple(int(item.strip()) for item in str(text).split(",") if item.strip())
    if not values:
        raise ValueError("--fit-ells must contain at least one multipole")
    if values[0] != 0:
        raise ValueError(f"--fit-ells must start with 0 so the monopole/GIC convention is explicit; got {values}")
    if any(ell < 0 or ell % 2 for ell in values):
        raise ValueError(f"Task44 fits require non-negative even multipoles; got {values}")
    return values


def _jsonable(obj: Any) -> Any:
    if isinstance(obj, dict):
        return {str(k): _jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_jsonable(v) for v in obj]
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


def _shell_jell_average_numeric(k_values: np.ndarray, s_lo: np.ndarray, s_hi: np.ndarray, ell: int, *, nquad: int = 24) -> np.ndarray:
    kval = np.asarray(k_values, dtype="f8")[:, None, None]
    s_lo = np.asarray(s_lo, dtype="f8")[None, :, None]
    s_hi = np.asarray(s_hi, dtype="f8")[None, :, None]
    xg, wg = np.polynomial.legendre.leggauss(int(nquad))
    radius = 0.5 * (s_hi - s_lo) * xg[None, None, :] + 0.5 * (s_hi + s_lo)
    weights = 0.5 * (s_hi - s_lo) * wg[None, None, :] * radius**2
    denom = (s_hi[:, :, 0] ** 3 - s_lo[:, :, 0] ** 3) / 3.0
    return np.sum(weights * spherical_jn(int(ell), kval * radius), axis=2) / denom


def xi_multipole_phase(ell: int) -> float:
    return -1.0 if (int(ell) // 2) % 2 else 1.0


def build_xi_kernel(k_values: np.ndarray, s: np.ndarray, s_edges: np.ndarray, mode: str, *, ell: int = 0) -> tuple[np.ndarray, dict[str, Any]]:
    """Build the xi projection kernel used to compare theory with binned 2PCF."""
    ell = int(ell)
    phase = xi_multipole_phase(ell)
    if mode == "center":
        arg = np.outer(np.asarray(k_values, dtype="f8"), np.asarray(s, dtype="f8"))
        kernel = phase * spherical_jn(ell, arg)
        return kernel, {
            "mode": mode,
            "ell": ell,
            "phase_i_ell": phase,
            "description": f"i^ell j_ell evaluated at measured bin centers; no shell-bin averaging",
        }
    if mode == "shell-averaged":
        edges = np.asarray(s_edges, dtype="f8")
        if edges.shape != (np.asarray(s).size, 2):
            raise ValueError(f"s_edges shape {edges.shape} does not match selected bins {np.asarray(s).size}")
        kernel = _shell_j0_average(k_values, edges[:, 0], edges[:, 1]) if ell == 0 else _shell_jell_average_numeric(k_values, edges[:, 0], edges[:, 1], ell)
        return phase * kernel, {
            "mode": mode,
            "ell": ell,
            "phase_i_ell": phase,
            "description": f"volume-averaged i^ell j_ell over each radial shell; ell={ell}",
            "s_lower_edges": [float(v) for v in edges[:, 0]],
            "s_upper_edges": [float(v) for v in edges[:, 1]],
        }
    raise ValueError(f"unknown xi_kernel mode {mode!r}; choices are center,shell-averaged")


def edge_pairs(edges: np.ndarray) -> np.ndarray:
    arr = np.asarray(edges, dtype="f8")
    if arr.ndim == 2:
        return arr
    return np.column_stack([arr[:-1], arr[1:]])


def load_xi(path: Path, rmin: float, rmax: float, ells: tuple[int, ...]) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, dict[str, Any], dict[int, np.ndarray]]:
    data = np.load(path, allow_pickle=False)
    s_all = np.asarray(data["s"], dtype="f8")
    rr_all = np.asarray(data["RR"], dtype="f8")
    edges_all = edge_pairs(np.asarray(data["s_edges"], dtype="f8"))
    mask = (s_all >= float(rmin)) & (s_all <= float(rmax))
    if not np.any(mask):
        raise ValueError(f"no xi bins in [{rmin}, {rmax}]")
    meta = json.loads(str(np.asarray(data["meta_json"]).item())) if "meta_json" in data.files else {}
    xi_by_ell: dict[int, np.ndarray] = {}
    if "xi_multipoles" in data.files and "ells" in data.files:
        file_ells = tuple(int(v) for v in np.asarray(data["ells"]).ravel())
        xi_multipoles = np.asarray(data["xi_multipoles"], dtype="f8")
        if xi_multipoles.shape != (len(file_ells), s_all.size):
            raise ValueError(f"xi_multipoles shape {xi_multipoles.shape} inconsistent with ells={file_ells} and s={s_all.size}")
        for ell in ells:
            if int(ell) not in file_ells:
                raise KeyError(f"{path} has multipoles {file_ells}, not requested ell={ell}")
            xi_by_ell[int(ell)] = xi_multipoles[file_ells.index(int(ell)), mask]
    else:
        if tuple(int(v) for v in ells) != (0,):
            raise KeyError(f"{path} has no xi_multipoles/ells keys; cannot fit ells={ells}")
        xi_by_ell[0] = np.asarray(data["xi0"], dtype="f8")[mask]
    xi_vector = np.concatenate([xi_by_ell[int(ell)] for ell in ells])
    return s_all[mask], xi_vector, rr_all[mask], edges_all[mask], meta, xi_by_ell


def covariance_health(cov: np.ndarray) -> dict[str, Any]:
    sym = 0.5 * (np.asarray(cov, dtype="f8") + np.asarray(cov, dtype="f8").T)
    diag = np.diag(sym)
    if np.any(diag <= 0.0):
        return {"diag_positive": False}
    sigma = np.sqrt(diag)
    corr = sym / np.outer(sigma, sigma)
    eig = np.linalg.eigvalsh(sym)
    corr_eig = np.linalg.eigvalsh(0.5 * (corr + corr.T))
    eig_positive = bool(eig[0] > 0.0)
    corr_eig_positive = bool(corr_eig[0] > 0.0)
    condition = float(eig[-1] / eig[0]) if eig_positive else np.inf
    corr_condition = float(corr_eig[-1] / corr_eig[0]) if corr_eig_positive else np.inf
    return {
        "diag_positive": True,
        "eig_positive": eig_positive,
        "corr_eig_positive": corr_eig_positive,
        "sigma_min": float(np.min(sigma)),
        "sigma_max": float(np.max(sigma)),
        "eig_min": float(eig[0]),
        "eig_max": float(eig[-1]),
        "condition_number": condition,
        "corr_eig_min": float(corr_eig[0]),
        "corr_eig_max": float(corr_eig[-1]),
        "corr_condition_number": corr_condition,
    }


def precision_diagnostics(cov: np.ndarray, *, rcond: float = PINV_RCOND) -> dict[str, Any]:
    eig = np.linalg.eigvalsh(0.5 * (np.asarray(cov, dtype="f8") + np.asarray(cov, dtype="f8").T))
    cutoff = float(rcond) * float(np.max(eig))
    return {
        "pinv_rcond": float(rcond),
        "pinv_cutoff": cutoff,
        "effective_rank": int(np.count_nonzero(eig > cutoff)),
        "nbins": int(eig.size),
        "eig_min": float(eig[0]),
        "eig_max": float(eig[-1]),
    }


def load_rascalc_corrected_precision(
    path: Path | None,
    *,
    covariance_mode: str,
    cov: np.ndarray,
    cov_meta: dict[str, Any],
    s: np.ndarray,
    xi: np.ndarray,
    ells: tuple[int, ...],
    rmin: float,
    rmax: float,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Load RascalC's inverse-bias-corrected precision without sub-selecting it.

    The correction matrix ``D`` is derived for the complete data vector.  A
    principal sub-block of the corrected precision is *not* the precision of a
    shortened data vector, so this path deliberately accepts only the native
    30-bin monopole product and uses the complete stored matrix.
    """

    if str(covariance_mode) != "npz" or path is None:
        raise ValueError(
            "--precision-mode rascalc-corrected requires --covariance-mode npz "
            "and a --covariance-path containing RascalC precision products"
        )
    if tuple(int(value) for value in ells) != (0,):
        raise ValueError(
            "RascalC corrected precision is currently defined only for the full "
            "30-bin monopole vector (--fit-ells 0); precision sub-blocks are forbidden"
        )

    with np.load(path, allow_pickle=False) as data:
        required = {
            "s",
            "covariance_single_realization",
            "rascalc_full_theory_precision",
            "rascalc_full_theory_D_matrix",
        }
        missing = sorted(required - set(data.files))
        if missing:
            raise KeyError(
                f"{path} is not a corrected-precision RascalC product; missing keys={missing}"
            )

        s_cov = np.asarray(data["s"], dtype="f8")
        selected = np.flatnonzero((s_cov >= float(rmin)) & (s_cov <= float(rmax)))
        full_ids = np.arange(s_cov.size, dtype="i8")
        if (
            s_cov.ndim != 1
            or s_cov.size != 30
            or selected.size != 30
            or not np.array_equal(selected, full_ids)
        ):
            raise ValueError(
                "--precision-mode rascalc-corrected requires the complete native "
                f"30-bin radial selection; file_nbins={s_cov.size}, selected={selected.size}, "
                f"range=[{rmin}, {rmax}]. Directly slicing a corrected precision sub-block "
                "is forbidden; rerun RascalC/post-processing for the desired data vector."
            )
        s_fit = np.asarray(s, dtype="f8")
        if s_fit.shape != (30,) or not np.allclose(s_fit, s_cov, rtol=0.0, atol=1.0e-12):
            raise ValueError(
                "fit bins do not exactly match the complete 30-bin RascalC precision grid; "
                "no precision sub-selection or reordering is allowed"
            )
        if np.asarray(xi).shape != (30,) or np.asarray(cov).shape != (30, 30):
            raise ValueError(
                f"corrected precision requires xi=(30,) and covariance=(30,30); "
                f"got xi={np.asarray(xi).shape}, covariance={np.asarray(cov).shape}"
            )
        if str(cov_meta.get("key")) != "covariance_single_realization":
            raise ValueError(
                "corrected precision must accompany covariance_single_realization from "
                f"the same NPZ; selected covariance key={cov_meta.get('key')!r}"
            )

        covariance_stored = np.asarray(data["covariance_single_realization"], dtype="f8")
        precision_raw = np.asarray(data["rascalc_full_theory_precision"], dtype="f8")
        d_matrix = np.asarray(data["rascalc_full_theory_D_matrix"], dtype="f8")

    expected_shape = (30, 30)
    for name, matrix in (
        ("covariance_single_realization", covariance_stored),
        ("rascalc_full_theory_precision", precision_raw),
        ("rascalc_full_theory_D_matrix", d_matrix),
    ):
        if matrix.shape != expected_shape:
            raise ValueError(f"{path} key {name} has shape {matrix.shape}, expected {expected_shape}")
        if not np.all(np.isfinite(matrix)):
            raise ValueError(f"{path} key {name} contains non-finite entries")
    if not np.allclose(np.asarray(cov, dtype="f8"), covariance_stored, rtol=1.0e-12, atol=0.0):
        raise ValueError(
            "the covariance used by the fit is not the complete covariance_single_realization "
            "stored beside the RascalC corrected precision"
        )

    # RascalC's finite-integration correction can leave tiny numerical
    # asymmetry.  A quadratic likelihood only depends on the symmetric part,
    # which is what we validate and use below.
    precision = 0.5 * (precision_raw + precision_raw.T)
    precision_eigenvalues = np.linalg.eigvalsh(precision)
    if not np.all(np.isfinite(precision_eigenvalues)):
        raise ValueError("symmetrized RascalC corrected precision has non-finite eigenvalues")
    if float(precision_eigenvalues[0]) <= 0.0:
        raise ValueError(
            "symmetrized RascalC corrected precision is not SPD: "
            f"min_eigenvalue={precision_eigenvalues[0]:.6e}"
        )
    rank_cutoff = float(PINV_RCOND) * float(precision_eigenvalues[-1])
    effective_rank = int(np.count_nonzero(precision_eigenvalues > rank_cutoff))
    algebraic_rank = int(np.linalg.matrix_rank(precision))
    if effective_rank != 30 or algebraic_rank != 30:
        raise ValueError(
            "symmetrized RascalC corrected precision is not full rank: "
            f"effective_rank={effective_rank}, algebraic_rank={algebraic_rank}, nbins=30, "
            f"rank_cutoff={rank_cutoff:.6e}"
        )

    naive_precision = np.linalg.pinv(np.asarray(cov, dtype="f8"), rcond=PINV_RCOND)
    delta = precision - naive_precision
    naive_frobenius = float(np.linalg.norm(naive_precision))
    naive_absmax = float(np.max(np.abs(naive_precision)))
    d_eigenvalues = np.linalg.eigvals(d_matrix)
    d_spectral_radius = float(np.max(np.abs(d_eigenvalues)))
    raw_asymmetry = float(np.max(np.abs(precision_raw - precision_raw.T)))
    raw_scale = max(float(np.max(np.abs(precision))), 1.0e-300)
    d_scale = max(float(np.max(np.abs(d_matrix))), 1.0e-300)
    metadata: dict[str, Any] = {
        "mode": "rascalc-corrected",
        "source": f"{path}::rascalc_full_theory_precision (symmetrized)",
        "matrix_key": "rascalc_full_theory_precision",
        "d_matrix_source": f"{path}::rascalc_full_theory_D_matrix",
        "selection_policy": "complete unsliced 30-bin monopole RascalC data vector",
        "selected_bin_count": 30,
        "full_native_bin_count": 30,
        "symmetrized_before_use": True,
        "finite": True,
        "spd": True,
        "full_rank": True,
        "effective_rank": effective_rank,
        "algebraic_rank": algebraic_rank,
        "rank_rcond": float(PINV_RCOND),
        "rank_cutoff": rank_cutoff,
        "eigenvalue_min": float(precision_eigenvalues[0]),
        "eigenvalue_max": float(precision_eigenvalues[-1]),
        "condition_number": float(precision_eigenvalues[-1] / precision_eigenvalues[0]),
        "raw_asymmetry_max_abs": raw_asymmetry,
        "raw_asymmetry_relative_max": float(raw_asymmetry / raw_scale),
        "d_spectral_radius": d_spectral_radius,
        "d_eigenvalue_real_min": float(np.min(np.real(d_eigenvalues))),
        "d_eigenvalue_real_max": float(np.max(np.real(d_eigenvalues))),
        "d_eigenvalue_imag_max_abs": float(np.max(np.abs(np.imag(d_eigenvalues)))),
        "d_asymmetry_relative_max": float(np.max(np.abs(d_matrix - d_matrix.T)) / d_scale),
        "corrected_vs_naive_precision_relative_frobenius": float(
            np.linalg.norm(delta) / max(naive_frobenius, 1.0e-300)
        ),
        "corrected_vs_naive_precision_relative_max_abs": float(
            np.max(np.abs(delta)) / max(naive_absmax, 1.0e-300)
        ),
        "naive_precision_definition": f"numpy.linalg.pinv(covariance, rcond={PINV_RCOND:.1e})",
    }
    return precision, metadata


def load_covariance(
    path: Path | None,
    *,
    mode: str,
    covariance_key: str = "covariance_single_realization",
    xi: np.ndarray,
    ells: tuple[int, ...],
    rmin: float,
    rmax: float,
    max_corr_condition: float,
    allow_ill_conditioned: bool,
) -> tuple[np.ndarray, dict[str, Any]]:
    if mode == "diagonal-smoke":
        floor = max(2.0e-4, 0.05 * float(np.nanmax(np.abs(xi))))
        sigma = np.maximum(0.25 * np.abs(xi), floor)
        return np.diag(sigma**2), {"mode": mode, "sigma_floor": floor}
    if mode != "npz":
        raise ValueError(f"unknown covariance mode {mode!r}")
    if path is None:
        raise ValueError("--covariance-path is required for --covariance-mode npz")
    data = np.load(path, allow_pickle=False)
    meta = json.loads(str(np.asarray(data["meta_json"]).item())) if "meta_json" in data.files and str(np.asarray(data["meta_json"]).item()) else {}
    s_cov = np.asarray(data["s"], dtype="f8")
    ids = np.flatnonzero((s_cov >= float(rmin)) & (s_cov <= float(rmax)))

    def selected_rr_window_metadata(indices: list[int] | np.ndarray) -> dict[str, Any]:
        if "rr_window_inverse" not in data.files:
            return {}
        indices = np.asarray(indices, dtype="i8")
        inv_all = np.asarray(data["rr_window_inverse"], dtype="f8")
        if inv_all.ndim != 2 or inv_all.shape[0] != inv_all.shape[1]:
            return {}
        if indices.size == 0 or int(np.max(indices)) >= inv_all.shape[0]:
            return {}
        out: dict[str, Any] = {
            "data_window_basis": "RR-deconvolved",
            "_rr_window_inverse_selected": inv_all[np.ix_(indices, indices)],
        }
        if "rr_window_matrix" in data.files:
            window_all = np.asarray(data["rr_window_matrix"], dtype="f8")
            if window_all.shape == inv_all.shape:
                out["_rr_window_matrix_selected"] = window_all[np.ix_(indices, indices)]
        return out

    def covariance_subset_by_ells(cov_all: np.ndarray, cov_ells: tuple[int, ...], *, key: str, warning: str | None = None) -> tuple[np.ndarray, dict[str, Any]]:
        indices: list[int] = []
        for ell in ells:
            if int(ell) not in cov_ells:
                raise KeyError(f"{path} covariance multipoles are {cov_ells}, not requested ell={ell}")
            offset = cov_ells.index(int(ell)) * s_cov.size
            indices.extend((offset + ids).tolist())
        if len(indices) != xi.size:
            raise ValueError(f"covariance/data vector mismatch: selected cov ids={len(indices)}, xi bins={xi.size}")
        cov = np.asarray(cov_all, dtype="f8")[np.ix_(indices, indices)]
        health = covariance_health(cov)
        if not bool(health.get("diag_positive", False)):
            raise ValueError(f"covariance has non-positive diagonal entries: {path}")
        if not bool(health.get("eig_positive", False)) or not bool(health.get("corr_eig_positive", False)):
            raise ValueError(
                f"covariance is not positive definite: eig_min={health.get('eig_min')} "
                f"corr_eig_min={health.get('corr_eig_min')} path={path}"
            )
        corr_condition = float(health["corr_condition_number"])
        if corr_condition > float(max_corr_condition) and not bool(allow_ill_conditioned):
            raise ValueError(
                f"refusing ill-conditioned multipole covariance: corr_condition={corr_condition:.3e} "
                f"> {float(max_corr_condition):.3e}; path={path}. "
                "Pass --allow-ill-conditioned-covariance for a diagnostic-only run."
            )
        precision = precision_diagnostics(cov)
        if int(precision["effective_rank"]) != int(precision["nbins"]):
            raise ValueError(
                f"multipole covariance precision is rank-deficient at pinv_rcond={precision['pinv_rcond']}: "
                f"effective_rank={precision['effective_rank']} nbins={precision['nbins']} path={path}"
            )
        out_meta: dict[str, Any] = {
            "mode": mode,
            "path": str(path),
            "key": key,
            "ells": [int(v) for v in ells],
            "covariance_ells": [int(v) for v in cov_ells],
            "health": health,
            "precision": precision,
        }
        out_meta.update(selected_rr_window_metadata(indices))
        if warning:
            out_meta["warning"] = warning
        return cov, out_meta

    if tuple(int(v) for v in ells) != (0,) and "covariance_xi_multipoles" in data.files:
        key = "covariance_xi_multipoles"
        cov_all = np.asarray(data[key], dtype="f8")
        cov_ells = tuple(int(v) for v in meta.get("theory", {}).get("ells", (0, 2, 4)))
        return covariance_subset_by_ells(
            cov_all,
            cov_ells,
            key=key,
            warning="raw multipole covariance block; RR(s,mu) deconvolution is not applied to this block",
        )
    if str(covariance_key) not in data.files:
        raise KeyError(
            f"{path} has no requested covariance key {covariance_key!r}; "
            "Task44 fits must select their likelihood covariance explicitly."
        )
    key = str(covariance_key)
    cov_all = np.asarray(data[key], dtype="f8")
    if tuple(int(v) for v in ells) != (0,):
        cov_ells = tuple(int(v) for v in np.asarray(data["ells"]).ravel()) if "ells" in data.files else tuple(int(v) for v in meta.get("rr_window", {}).get("ells", ells))
        if cov_all.shape == (s_cov.size * len(cov_ells), s_cov.size * len(cov_ells)):
            return covariance_subset_by_ells(cov_all, cov_ells, key=key)
    if ids.size != xi.size:
        raise ValueError(f"covariance bin mismatch: cov ids={ids.size}, xi bins={xi.size}")
    cov = cov_all[np.ix_(ids, ids)]
    health = covariance_health(cov)
    if not bool(health.get("diag_positive", False)):
        raise ValueError(f"covariance has non-positive diagonal entries: {path}")
    if not bool(health.get("eig_positive", False)) or not bool(health.get("corr_eig_positive", False)):
        raise ValueError(
            f"covariance is not positive definite: eig_min={health.get('eig_min')} "
            f"corr_eig_min={health.get('corr_eig_min')} path={path}"
        )
    corr_condition = float(health["corr_condition_number"])
    if corr_condition > float(max_corr_condition) and not bool(allow_ill_conditioned):
        raise ValueError(
            f"refusing ill-conditioned covariance: corr_condition={corr_condition:.3e} "
            f"> {float(max_corr_condition):.3e}; path={path}. "
            "Rebuild the Task44 covariance with the Task43 full-window RR-deconvolved path, "
            "or pass --allow-ill-conditioned-covariance for a diagnostic-only run."
        )
    precision = precision_diagnostics(cov)
    if int(precision["effective_rank"]) != int(precision["nbins"]):
        raise ValueError(
            f"covariance precision is rank-deficient at pinv_rcond={precision['pinv_rcond']}: "
            f"effective_rank={precision['effective_rank']} nbins={precision['nbins']} path={path}"
        )
    out_meta = {"mode": mode, "path": str(path), "key": key, "health": health, "precision": precision}
    out_meta.update(selected_rr_window_metadata(ids))
    return cov, out_meta


def public_covariance_meta(cov_meta: dict[str, Any]) -> dict[str, Any]:
    return {key: value for key, value in cov_meta.items() if not str(key).startswith("_")}


def apply_data_window_treatment(
    xi: np.ndarray,
    xi_by_ell: dict[int, np.ndarray],
    *,
    s: np.ndarray,
    ells: tuple[int, ...],
    cov_meta: dict[str, Any],
    treatment: str,
) -> tuple[np.ndarray, dict[int, np.ndarray], dict[str, Any]]:
    treatment = str(treatment)
    inv_window = cov_meta.get("_rr_window_inverse_selected")
    if treatment == "raw":
        return xi, xi_by_ell, {"mode": treatment, "applied": False, "reason": "requested raw data vector"}
    if inv_window is None:
        if treatment == "deconvolve-required":
            raise ValueError("requested RR-window data deconvolution, but covariance file has no rr_window_inverse")
        return xi, xi_by_ell, {"mode": treatment, "applied": False, "reason": "covariance file has no rr_window_inverse"}

    inv = np.asarray(inv_window, dtype="f8")
    if inv.shape != (xi.size, xi.size):
        raise ValueError(f"rr_window_inverse shape {inv.shape} does not match xi vector length {xi.size}")
    xi_raw = np.asarray(xi, dtype="f8")
    xi_deconvolved = inv @ xi_raw
    nbins = np.asarray(s).size
    xi_by_ell_deconvolved = {
        int(ell): xi_deconvolved[iell * nbins : (iell + 1) * nbins]
        for iell, ell in enumerate(ells)
    }
    delta = xi_deconvolved - xi_raw
    meta: dict[str, Any] = {
        "mode": treatment,
        "applied": True,
        "operation": "xi_deconvolved = rr_window_inverse @ xi_raw",
        "rr_window_inverse_shape": [int(v) for v in inv.shape],
        "delta_rms": float(np.sqrt(np.mean(delta**2))),
        "delta_max_abs": float(np.max(np.abs(delta))),
        "raw_norm": float(np.linalg.norm(xi_raw)),
        "deconvolved_norm": float(np.linalg.norm(xi_deconvolved)),
    }
    window = cov_meta.get("_rr_window_matrix_selected")
    if window is not None:
        w = np.asarray(window, dtype="f8")
        meta.update(
            {
                "rr_window_condition_number": float(np.linalg.cond(w)),
                "rr_window_diag_min": float(np.min(np.diag(w))),
                "rr_window_diag_max": float(np.max(np.diag(w))),
                "rr_window_offdiag_absmax": float(np.max(np.abs(w - np.diag(np.diag(w))))),
            }
        )
    return xi_deconvolved, xi_by_ell_deconvolved, meta


def load_or_build_w2(
    *,
    theory: dict[str, Any],
    random_path: Path,
    window_path: Path,
    nsub: int,
    seed: int,
    nthreads: int,
    p0: float,
    boxsize: float,
    force: bool,
    trust_cache: bool,
) -> dict[str, Any]:
    k_eff = np.asarray(theory["k_eff"], dtype="f8")
    def with_cache_meta(meta: dict[str, Any], *, reused: bool) -> dict[str, Any]:
        out = dict(meta)
        out["cache"] = {
            "path": str(window_path),
            "reused": bool(reused),
            "requested_nthreads": int(nthreads),
            "stored_nthreads": int(meta.get("nthreads", nthreads)),
            "trust_cache": bool(trust_cache),
        }
        return out

    if window_path.exists() and not force:
        data = np.load(window_path, allow_pickle=False)
        cached_k = np.asarray(data["k_eff"], dtype="f8")
        if cached_k.shape == k_eff.shape and np.allclose(cached_k, k_eff, rtol=0.0, atol=1.0e-14):
            meta = json.loads(str(np.asarray(data["meta_json"]).item()))
            if trust_cache:
                return {"k_eff": cached_k, "w2": np.asarray(data["w2"], dtype="f8"), "meta": with_cache_meta(meta, reused=True)}
            stale_reasons = []
            if str(meta.get("random_path")) != str(random_path):
                stale_reasons.append("random_path")
            weighting = meta.get("weighting", {})
            if weighting.get("scheme") != "catalog_WEIGTHTOTAL" and weighting.get("scheme") != "catalog_WEIGHT_TOTAL":
                stale_reasons.append("weighting_scheme")
            if weighting.get("p0") is None or abs(float(weighting.get("p0")) - float(p0)) > 1.0e-10:
                stale_reasons.append("p0")
            theory_meta = meta.get("theory", {})
            if theory_meta.get("boxsize") is None or abs(float(theory_meta.get("boxsize")) - float(boxsize)) > 1.0e-10:
                stale_reasons.append("boxsize")
            if theory_meta.get("kfund") is None or abs(float(theory_meta.get("kfund")) - float(theory["kfund"])) > 1.0e-14:
                stale_reasons.append("kfund")
            if int(meta.get("n_subsample", -1)) != (int(nsub) if int(nsub) > 0 else int(meta.get("n_random_total", -2))):
                stale_reasons.append("nsub")
            if int(meta.get("seed", -1)) != int(seed):
                stale_reasons.append("seed")
            if not stale_reasons:
                return {"k_eff": cached_k, "w2": np.asarray(data["w2"], dtype="f8"), "meta": with_cache_meta(meta, reused=True)}
            print(f"[w2] stale cache {window_path}; rebuilding because {','.join(stale_reasons)}", flush=True)
    from pycorr import TwoPointCorrelationFunction

    random = np.load(random_path, allow_pickle=False)
    xyz = np.column_stack([random["X"], random["Y"], random["Zcart"]]).astype("f8")
    weight = np.asarray(random["WEIGHT_TOTAL"], dtype="f8")
    n_random = int(xyz.shape[0])
    n_sub = n_random if int(nsub) <= 0 else min(n_random, int(nsub))
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
    pair_prob = rr_raw / pair_norm
    coverage = float(np.sum(pair_prob))
    pair_prob_norm = pair_prob / coverage
    w2 = np.zeros_like(k_eff)
    active = np.flatnonzero((k_eff > 0.0) & (k_eff <= K_CUT_WINDOW))
    for start in range(0, active.size, 512):
        ids = active[start : start + 512]
        avg_j0 = _shell_j0_average(k_eff[ids], s_edges[:-1], s_edges[1:])
        w2[ids] = avg_j0 @ pair_prob_norm
    meta = {
        "method": "task44_formal_gic_random_pair_kernel",
        "formula": "sigma_W2 = sum_q g_q P0_rsd(k_q) W2(k_q) / V_box",
        "random_path": str(random_path),
        "n_random_total": n_random,
        "n_subsample": int(n_sub),
        "seed": int(seed),
        "nthreads": int(nthreads),
        "coverage": coverage,
        "pair_norm": pair_norm,
        "k_cut_window": float(K_CUT_WINDOW),
        "weight_min": float(np.min(weight_sub)),
        "weight_max": float(np.max(weight_sub)),
        "weighting": {"scheme": "catalog_WEIGHT_TOTAL", "p0": float(p0)},
        "theory": {"boxsize": float(boxsize), "kfund": float(theory["kfund"])},
    }
    window_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        window_path,
        k_eff=k_eff,
        w2=w2,
        s_edges=s_edges,
        rr_raw=rr_raw,
        pair_prob=pair_prob,
        meta_json=np.asarray(json.dumps(meta, sort_keys=True)),
    )
    return {"k_eff": k_eff, "w2": w2, "meta": with_cache_meta(meta, reused=False)}


def summarize_chain(samples: np.ndarray, log_prob: np.ndarray, names: list[str]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for i, name in enumerate(names):
        q16, q50, q84 = np.percentile(samples[:, i], [16, 50, 84])
        out[name] = {
            "q16": float(q16),
            "q50": float(q50),
            "q84": float(q84),
            "mean": float(np.mean(samples[:, i])),
            "std": float(np.std(samples[:, i])),
        }
    imax = int(np.argmax(log_prob))
    out["map"] = {name: float(samples[imax, i]) for i, name in enumerate(names)}
    out["map"]["log_prob"] = float(log_prob[imax])
    return out


def run_one_model(
    *,
    model: str,
    s: np.ndarray,
    xi: np.ndarray,
    rr: np.ndarray,
    s_edges: np.ndarray,
    cov: np.ndarray,
    precision_override: np.ndarray | None,
    precision_override_meta: dict[str, Any] | None,
    theory: dict[str, Any],
    formal_gic_window: dict[str, Any] | None,
    p_fixed: float,
    fit_ells: tuple[int, ...],
    xi_kernel: str,
    xi_projector: str,
    projection_kmin: float | None,
    fog_model: str,
    sn0_fixed: float,
    free_sn0: bool,
    sn0_scale: float,
    sn0_treatment: str,
    sn0_prior: tuple[float, float],
    sigma_s_prior: tuple[float, float],
    sigma_s_fixed: float | None,
    b1_gaussian_prior: tuple[float, float] | None,
    nwalkers: int,
    nsteps: int,
    burnin: int,
    seed: int,
    optimizer_only: bool,
) -> tuple[dict[str, Any], np.ndarray, np.ndarray, np.ndarray]:
    precision_meta = precision_diagnostics(cov, rcond=PINV_RCOND)
    if precision_override is None:
        # Keep the historical JAXpower/general-covariance behavior exactly:
        # no corrected-precision keys are required or inspected in this mode.
        precision = np.linalg.pinv(cov, rcond=PINV_RCOND)
        precision_source = f"numpy.linalg.pinv(covariance_used, rcond={PINV_RCOND:.1e})"
        precision_meta.update(
            {
                "mode": "inverse-covariance",
                "source": precision_source,
                "precision_override": False,
            }
        )
    else:
        precision_raw = np.asarray(precision_override, dtype="f8")
        if precision_raw.shape != np.asarray(cov).shape:
            raise ValueError(
                f"precision override shape {precision_raw.shape} does not match covariance {np.asarray(cov).shape}"
            )
        if not np.all(np.isfinite(precision_raw)):
            raise ValueError("precision override contains non-finite entries")
        precision = 0.5 * (precision_raw + precision_raw.T)
        precision_eigenvalues = np.linalg.eigvalsh(precision)
        rank_cutoff = float(PINV_RCOND) * float(precision_eigenvalues[-1])
        effective_rank = int(np.count_nonzero(precision_eigenvalues > rank_cutoff))
        if float(precision_eigenvalues[0]) <= 0.0 or effective_rank != precision.shape[0]:
            raise ValueError(
                "precision override failed run_one_model SPD/full-rank validation: "
                f"eig_min={precision_eigenvalues[0]:.6e}, effective_rank={effective_rank}, "
                f"ndata={precision.shape[0]}"
            )
        if precision_override_meta is None or not str(precision_override_meta.get("source", "")):
            raise ValueError("precision override requires provenance metadata with a non-empty source")
        precision_source = str(precision_override_meta["source"])
        precision_meta = dict(precision_override_meta)
        precision_meta["precision_override"] = True
    task41 = theory["task41"]
    k_dense = np.asarray(theory["k_dense"], dtype="f8")
    xi_projector = str(xi_projector).lower()
    kernels: dict[int, np.ndarray] = {}
    xi_kernel_meta_by_ell: dict[int, dict[str, Any]] = {}
    for ell in fit_ells:
        kernels[int(ell)], xi_kernel_meta_by_ell[int(ell)] = build_xi_kernel(theory["k_eff"], s, s_edges, str(xi_kernel), ell=int(ell))
    continuous_meta: dict[str, Any] | None = None
    if xi_projector == "continuous-hankel":
        kmin_eff = float(np.min(k_dense) if projection_kmin is None else projection_kmin)
        cmask = k_dense >= kmin_eff
        if np.count_nonzero(cmask) < 3:
            raise ValueError(f"continuous projector has too few k samples above projection_kmin={kmin_eff}")
        k_cont = k_dense[cmask]
        kernel_cont_by_ell: dict[int, np.ndarray] = {}
        xi_kernel_meta_cont_by_ell: dict[int, dict[str, Any]] = {}
        for ell in fit_ells:
            kernel_cont_by_ell[int(ell)], xi_kernel_meta_cont_by_ell[int(ell)] = build_xi_kernel(k_cont, s, s_edges, str(xi_kernel), ell=int(ell))
        dk = np.empty_like(k_cont)
        dk[0] = 0.5 * (k_cont[1] - k_cont[0])
        dk[-1] = 0.5 * (k_cont[-1] - k_cont[-2])
        dk[1:-1] = 0.5 * (k_cont[2:] - k_cont[:-2])
        weights_cont = dk * k_cont**2 / (2.0 * np.pi**2)
        continuous_meta = {
            "projection_kmin": kmin_eff,
            "projection_kmax": float(k_cont[-1]),
            "nk": int(k_cont.size),
            "integration": "trapezoid on theory k_dense grid",
            "xi_kernel_by_ell": {str(int(ell)): xi_kernel_meta_cont_by_ell[int(ell)] for ell in fit_ells},
        }
    elif xi_projector != "discrete-box":
        raise ValueError(f"unknown xi_projector {xi_projector!r}; choices are discrete-box,continuous-hankel")
    pk_dd_dense = task41.interp_logk(k_dense, theory["template"]["k"], theory["template"]["pk_dd"])
    alpha_dense = task41.interp_logk(k_dense, theory["template"]["k"], theory["template"]["alpha"])
    ones_dense = np.ones_like(k_dense)
    fog_model = str(fog_model).lower()
    fog_label = fog_description(fog_model)
    sn0_treatment = str(sn0_treatment).lower()
    if sn0_treatment not in {"contact", "finite-k-constant"}:
        raise ValueError(f"unknown sn0_treatment={sn0_treatment!r}")
    sn0_prior_lo, sn0_prior_hi = float(sn0_prior[0]), float(sn0_prior[1])
    if free_sn0 and not (np.isfinite(sn0_prior_lo) and np.isfinite(sn0_prior_hi) and sn0_prior_lo < sn0_prior_hi):
        raise ValueError(f"free sn0 requires a finite ordered --sn0-prior; got {sn0_prior!r}")
    sn0_identifiability = (
        "prior_only_contact_term_for_s_gt_0"
        if sn0_treatment == "contact"
        else "likelihood_constrained_by_diagnostic_finite_k_ringing"
    )
    sigma_s_fixed_value = None if sigma_s_fixed is None else float(sigma_s_fixed)
    if sigma_s_fixed_value is not None:
        if not np.isfinite(sigma_s_fixed_value) or sigma_s_fixed_value < 0.0:
            raise ValueError(f"--sigma-s-fixed must be finite and non-negative; got {sigma_s_fixed!r}")
    elif not (
        np.isfinite(float(sigma_s_prior[0]))
        and np.isfinite(float(sigma_s_prior[1]))
        and float(sigma_s_prior[0]) < float(sigma_s_prior[1])
    ):
        raise ValueError(f"free sigma_s requires a finite ordered prior; got {sigma_s_prior!r}")
    b1_gaussian_prior_value = (
        None
        if b1_gaussian_prior is None
        else (float(b1_gaussian_prior[0]), float(b1_gaussian_prior[1]))
    )
    if b1_gaussian_prior_value is not None:
        b1_prior_mean, b1_prior_sigma = b1_gaussian_prior_value
        if not (np.isfinite(b1_prior_mean) and np.isfinite(b1_prior_sigma) and b1_prior_sigma > 0.0):
            raise ValueError(
                "--b1-gaussian-prior requires a finite mean and a strictly positive sigma; "
                f"got {b1_gaussian_prior!r}"
            )

    sampled_names = ["fnl_loc", "b1"]
    if sigma_s_fixed_value is None:
        sampled_names.append("sigma_s")
    if free_sn0:
        sampled_names.append("sn0")
    output_names = ["fnl_loc", "b1", "sigma_s"] + (["sn0"] if free_sn0 else [])

    def expand_sampled_theta(theta_sampled: np.ndarray) -> np.ndarray:
        """Insert fixed parameters into the historical full parameter ordering."""

        values = np.asarray(theta_sampled, dtype="f8").ravel()
        if values.size != len(sampled_names):
            raise ValueError(
                f"sampled theta has size {values.size}, expected {len(sampled_names)} for {sampled_names}"
            )
        by_name = {name: float(values[i]) for i, name in enumerate(sampled_names)}
        full = [
            by_name["fnl_loc"],
            by_name["b1"],
            sigma_s_fixed_value if sigma_s_fixed_value is not None else by_name["sigma_s"],
        ]
        if free_sn0:
            full.append(by_name["sn0"])
        return np.asarray(full, dtype="f8")
    mu_quad: np.ndarray | None = None
    wmu_quad: np.ndarray | None = None
    mu2_quad: np.ndarray | None = None
    legendre_by_ell: dict[int, np.ndarray] = {}
    if tuple(int(v) for v in fit_ells) != (0,):
        mu_quad, wmu_quad = np.polynomial.legendre.leggauss(96)
        mu2_quad = mu_quad * mu_quad
        for ell in fit_ells:
            coeff = np.zeros(int(ell) + 1, dtype="f8")
            coeff[int(ell)] = 1.0
            legendre_by_ell[int(ell)] = np.polynomial.legendre.legval(mu_quad, coeff)

    def pk0_dense(theta: np.ndarray) -> np.ndarray:
        fnl_loc, b1, sigma_s = theta[:3]
        sn0 = float(theta[3]) if free_sn0 else float(sn0_fixed)
        sn0_power = float(sn0_scale) * sn0 if sn0_treatment == "finite-k-constant" else 0.0
        bphi = 2.0 * DELTA_C * (float(b1) - float(p_fixed))
        amp = float(b1) + float(fnl_loc) * bphi * alpha_dense
        i0, i2, i4 = mu_moments(k_dense, float(sigma_s), nmu=96, fog_model=fog_model)
        f_growth = float(theory["f_growth"])
        return pk_dd_dense * (amp * amp * i0 + 2.0 * amp * f_growth * i2 + f_growth**2 * i4) + sn0_power * ones_dense

    def pkells_dense(theta: np.ndarray) -> dict[int, np.ndarray]:
        if tuple(int(v) for v in fit_ells) == (0,):
            return {0: pk0_dense(theta)}
        if mu_quad is None or wmu_quad is None or mu2_quad is None:
            raise RuntimeError("internal error: multipole quadrature is not initialized")
        fnl_loc, b1, sigma_s = theta[:3]
        sn0 = float(theta[3]) if free_sn0 else float(sn0_fixed)
        sn0_power = float(sn0_scale) * sn0 if sn0_treatment == "finite-k-constant" else 0.0
        bphi = 2.0 * DELTA_C * (float(b1) - float(p_fixed))
        amp = float(b1) + float(fnl_loc) * bphi * alpha_dense
        k_mu_sigma2 = (k_dense[:, None] * mu_quad[None, :] * float(sigma_s)) ** 2
        if fog_model == "lorentzian":
            damp = 1.0 / (1.0 + 0.5 * k_mu_sigma2) ** 2
        elif fog_model == "gaussian":
            damp = np.exp(-k_mu_sigma2)
        else:
            raise ValueError(f"unknown fog_model {fog_model!r}; choices are {FOG_MODELS}")
        f_growth = float(theory["f_growth"])
        pk_mu = pk_dd_dense[:, None] * (amp[:, None] + f_growth * mu2_quad[None, :]) ** 2 * damp
        out: dict[int, np.ndarray] = {}
        for ell in fit_ells:
            pole = 0.5 * (2 * int(ell) + 1) * np.sum(wmu_quad[None, :] * pk_mu * legendre_by_ell[int(ell)][None, :], axis=1)
            if int(ell) == 0:
                pole = pole + sn0_power * ones_dense
            out[int(ell)] = np.asarray(pole, dtype="f8")
        return out

    def xi_from_pkells(pks_dense: dict[int, np.ndarray]) -> np.ndarray:
        parts = []
        for ell in fit_ells:
            pk_dense = pks_dense[int(ell)]
            if xi_projector == "continuous-hankel":
                parts.append((weights_cont * pk_dense[cmask]) @ kernel_cont_by_ell[int(ell)])
            else:
                p_eff = task41.interp_logk(theory["k_eff"], k_dense, pk_dense)
                parts.append(((theory["g_nz"] * p_eff) @ kernels[int(ell)]) / float(theory["volume"]))
        return np.concatenate(parts)

    def sigma_w2(pk_dense: np.ndarray) -> float:
        if formal_gic_window is None:
            return 0.0
        if xi_projector == "continuous-hankel":
            wk = np.asarray(formal_gic_window["k_eff"], dtype="f8")
            w2_raw = np.asarray(formal_gic_window["w2"], dtype="f8")
            w2_cont = np.interp(k_cont, wk, w2_raw, left=float(w2_raw[0]), right=0.0)
            return float(np.sum(weights_cont * pk_dense[cmask] * w2_cont))
        p_eff = task41.interp_logk(theory["k_eff"], k_dense, pk_dense)
        w2 = np.asarray(formal_gic_window["w2"], dtype="f8")
        return float(np.sum(theory["g_nz"] * p_eff * w2) / float(theory["volume"]))

    # Exact acceleration for the large-scale sigma_s=0 monopole test.  The
    # Kaiser+PNG monopole is a linear combination of Pdd, Pdd*alpha, and
    # Pdd*alpha^2, so interpolation, FullDiscrete projection, and formal GIC
    # can be applied to these bases once before MCMC.  This changes no theory
    # or prior and is disabled for every historical free-FoG path.
    fast_sigma0_basis = bool(
        sigma_s_fixed_value == 0.0
        and tuple(int(value) for value in fit_ells) == (0,)
        and xi_projector == "discrete-box"
        and not free_sn0
        and sn0_treatment == "contact"
    )
    fast_xi_basis: tuple[np.ndarray, ...] | None = None
    fast_gic_basis: tuple[float, ...] | None = None
    if fast_sigma0_basis:
        dense_basis = (
            pk_dd_dense,
            pk_dd_dense * alpha_dense,
            pk_dd_dense * alpha_dense**2,
        )
        effective_basis = tuple(
            task41.interp_logk(theory["k_eff"], k_dense, basis) for basis in dense_basis
        )
        fast_xi_basis = tuple(
            ((theory["g_nz"] * basis) @ kernels[0]) / float(theory["volume"])
            for basis in effective_basis
        )
        if formal_gic_window is not None:
            w2_basis = np.asarray(formal_gic_window["w2"], dtype="f8")
            fast_gic_basis = tuple(
                float(np.sum(theory["g_nz"] * basis * w2_basis) / float(theory["volume"]))
                for basis in effective_basis
            )
        else:
            fast_gic_basis = (0.0, 0.0, 0.0)

    def evaluate_fast_sigma0(theta: np.ndarray) -> np.ndarray:
        if fast_xi_basis is None or fast_gic_basis is None:
            raise RuntimeError("internal error: sigma_s=0 basis acceleration is not initialized")
        fnl_loc, b1 = map(float, theta[:2])
        png_bias_amplitude = fnl_loc * 2.0 * DELTA_C * (b1 - float(p_fixed))
        f_growth = float(theory["f_growth"])
        coefficients = (
            b1**2 + (2.0 / 3.0) * b1 * f_growth + (1.0 / 5.0) * f_growth**2,
            2.0 * png_bias_amplitude * (b1 + f_growth / 3.0),
            png_bias_amplitude**2,
        )
        base = sum(coefficient * basis for coefficient, basis in zip(coefficients, fast_xi_basis, strict=True))
        if model == "no_gic":
            return np.asarray(base, dtype="f8")
        if model == "formal_gic":
            gic = sum(coefficient * basis for coefficient, basis in zip(coefficients, fast_gic_basis, strict=True))
            return np.asarray(base - gic, dtype="f8")
        raise ValueError(f"unknown model {model!r}")

    def evaluate(theta: np.ndarray) -> np.ndarray:
        if fast_sigma0_basis:
            return evaluate_fast_sigma0(theta)
        pks = pkells_dense(theta)
        base = xi_from_pkells(pks)
        if model == "no_gic":
            return base
        if model == "formal_gic":
            corrected = np.asarray(base, dtype="f8").copy()
            corrected[: s.size] -= sigma_w2(pks[0])
            return corrected
        raise ValueError(f"unknown model {model!r}")

    def log_prior(theta: np.ndarray) -> float:
        fnl_loc, b1, sigma_s = theta[:3]
        if not (-500.0 <= fnl_loc <= 500.0 and 0.2 <= b1 <= 10.0):
            return -np.inf
        if sigma_s_fixed_value is None and not (
            float(sigma_s_prior[0]) <= sigma_s <= float(sigma_s_prior[1])
        ):
            return -np.inf
        if free_sn0:
            sn0 = float(theta[3])
            if not (sn0_prior_lo <= sn0 <= sn0_prior_hi):
                return -np.inf
        if b1_gaussian_prior_value is None:
            return 0.0
        b1_prior_mean, b1_prior_sigma = b1_gaussian_prior_value
        return -0.5 * ((float(b1) - b1_prior_mean) / b1_prior_sigma) ** 2

    def chi2(theta: np.ndarray) -> float:
        diff = xi - evaluate(theta)
        return float(diff @ precision @ diff)

    def log_prob(theta: np.ndarray) -> float:
        lp = log_prior(theta)
        if not np.isfinite(lp):
            return -np.inf
        return lp - 0.5 * chi2(theta)

    def objective(theta: np.ndarray) -> float:
        lp = log_prior(theta)
        if not np.isfinite(lp):
            return 1.0e100
        return chi2(theta) - 2.0 * lp

    contact_free_sn0 = bool(free_sn0 and sn0_treatment == "contact")
    sn0_prior_mid = 0.5 * (sn0_prior_lo + sn0_prior_hi) if free_sn0 else float(sn0_fixed)
    x0_values = {
        "fnl_loc": 100.0,
        "b1": 2.2 if b1_gaussian_prior_value is None else b1_gaussian_prior_value[0],
        "sigma_s": 3.0,
        "sn0": sn0_prior_mid,
    }
    x0_sampled = np.asarray([x0_values[name] for name in sampled_names], dtype="f8")
    optimizer_names = sampled_names[:-1] if contact_free_sn0 else list(sampled_names)
    x0_optimizer = x0_sampled[:-1] if contact_free_sn0 else x0_sampled

    def optimizer_to_sampled(theta_optimizer: np.ndarray) -> np.ndarray:
        values = np.asarray(theta_optimizer, dtype="f8")
        if contact_free_sn0:
            return np.concatenate([values, np.asarray([sn0_prior_mid], dtype="f8")])
        return values

    def objective_optimizer(theta_optimizer: np.ndarray) -> float:
        return objective(expand_sampled_theta(optimizer_to_sampled(theta_optimizer)))

    opt = minimize(objective_optimizer, x0=x0_optimizer, method="Nelder-Mead")
    # 优化解、输出解与 walker 初始化互不共享内存；禁止初始化裁剪篡改 MAP。
    optimizer_raw = np.array(opt.x, dtype="f8", copy=True)
    optimizer_sampled = np.array(optimizer_to_sampled(optimizer_raw), dtype="f8", copy=True)
    optimizer_logp = float(log_prob(expand_sampled_theta(optimizer_sampled)))
    if not np.isfinite(optimizer_logp):
        raise RuntimeError("optimizer returned a non-finite/out-of-prior solution; no fit exported")
    if optimizer_only and not opt.success:
        raise RuntimeError(f"optimizer-only fit did not converge: {opt.message}; no MAP exported")
    center_sampled = np.array(
        optimizer_sampled if opt.success else optimizer_to_sampled(x0_optimizer), dtype="f8", copy=True
    )
    for index, name in enumerate(sampled_names):
        if name == "fnl_loc":
            center_sampled[index] = np.clip(center_sampled[index], -400.0, 400.0)
        elif name == "b1":
            center_sampled[index] = np.clip(center_sampled[index], 0.5, 8.0)
        elif name == "sigma_s":
            center_sampled[index] = np.clip(
                center_sampled[index],
                float(sigma_s_prior[0]) + 1.0e-6,
                float(sigma_s_prior[1]) - 1.0e-6,
            )
        elif name == "sn0":
            center_sampled[index] = (
                sn0_prior_mid
                if sn0_treatment == "contact"
                else np.clip(center_sampled[index], sn0_prior_lo + 1.0e-8, sn0_prior_hi - 1.0e-8)
            )

    def log_prob_sampled(theta_sampled: np.ndarray) -> float:
        return log_prob(expand_sampled_theta(theta_sampled))

    if optimizer_only:
        sampled_chain = optimizer_sampled[None, :].copy()
        logp = np.asarray([log_prob_sampled(optimizer_sampled)], dtype="f8")
        sampler_acceptance: float | None = None
    else:
        rng = np.random.default_rng(int(seed))
        sn0_width = max(sn0_prior_hi - sn0_prior_lo, 1.0)
        scale_by_name = {
            "fnl_loc": 10.0,
            "b1": 0.05,
            "sigma_s": 0.4,
            "sn0": min(0.5, 0.02 * sn0_width),
        }
        scale = np.asarray([scale_by_name[name] for name in sampled_names], dtype="f8")
        p0 = center_sampled[None, :] + rng.normal(
            scale=scale, size=(int(nwalkers), center_sampled.size)
        )
        for index, name in enumerate(sampled_names):
            if name == "fnl_loc":
                p0[:, index] = np.clip(p0[:, index], -490.0, 490.0)
            elif name == "b1":
                p0[:, index] = np.clip(p0[:, index], 0.25, 9.5)
            elif name == "sigma_s":
                p0[:, index] = np.clip(
                    p0[:, index],
                    float(sigma_s_prior[0]) + 1.0e-6,
                    float(sigma_s_prior[1]) - 1.0e-6,
                )
            elif name == "sn0":
                if sn0_treatment == "contact":
                    p0[:, index] = rng.uniform(
                        sn0_prior_lo + 1.0e-8,
                        sn0_prior_hi - 1.0e-8,
                        size=int(nwalkers),
                    )
                else:
                    p0[:, index] = np.clip(
                        p0[:, index], sn0_prior_lo + 1.0e-8, sn0_prior_hi - 1.0e-8
                    )
        sampler = emcee.EnsembleSampler(int(nwalkers), center_sampled.size, log_prob_sampled)
        sampler.run_mcmc(p0, int(nsteps), progress=False)
        sampled_chain = sampler.get_chain(discard=int(burnin), flat=True)
        logp = sampler.get_log_prob(discard=int(burnin), flat=True)
        sampler_acceptance = float(np.mean(sampler.acceptance_fraction))
    chain = np.empty((sampled_chain.shape[0], len(output_names)), dtype="f8")
    for index, name in enumerate(output_names):
        if name == "sigma_s" and sigma_s_fixed_value is not None:
            chain[:, index] = sigma_s_fixed_value
        else:
            chain[:, index] = sampled_chain[:, sampled_names.index(name)]
    summary = summarize_chain(chain, logp, output_names)
    map_theta = np.array([summary["map"][name] for name in output_names], dtype="f8")
    model_map = evaluate(map_theta)
    summary.update(
        {
            "model": model,
            "parameter_names": output_names,
            "sampled_parameter_names": sampled_names,
            "fixed_parameters": (
                {"sigma_s": sigma_s_fixed_value} if sigma_s_fixed_value is not None else {}
            ),
            "priors": {
                "fnl_loc": [-500.0, 500.0],
                "b1": [0.2, 10.0],
                "b1_gaussian": (
                    None
                    if b1_gaussian_prior_value is None
                    else {
                        "mean": b1_gaussian_prior_value[0],
                        "sigma": b1_gaussian_prior_value[1],
                        "normalization_included": False,
                    }
                ),
                "sigma_s": (
                    None
                    if sigma_s_fixed_value is not None
                    else [float(sigma_s_prior[0]), float(sigma_s_prior[1])]
                ),
                "sn0": [sn0_prior_lo, sn0_prior_hi] if free_sn0 else None,
            },
            "theory": {
                "space": "redshift",
                "multipoles": [int(v) for v in fit_ells],
                "p_fixed": float(p_fixed),
                "f_growth": float(theory["f_growth"]),
                "growth_rate_method": theory["growth_rate_method"],
                "fog": fog_label,
                "fog_model": fog_model,
                "sigma_s_policy": "fixed" if sigma_s_fixed_value is not None else "free",
                "sigma_s_fixed": sigma_s_fixed_value,
                "likelihood_acceleration": (
                    "exact preprojection of Pdd, Pdd*alpha, and Pdd*alpha^2 bases"
                    if fast_sigma0_basis
                    else "none"
                ),
                "xi_kernel_by_ell": {str(int(ell)): xi_kernel_meta_by_ell[int(ell)] for ell in fit_ells},
                "xi_projector": {
                    "mode": xi_projector,
                    "boxsize": float(theory["boxsize"]) if xi_projector == "discrete-box" else None,
                    "kfund": float(theory["kfund"]) if xi_projector == "discrete-box" else None,
                    "continuous": continuous_meta,
                },
                "sn0_fixed": float(sn0_fixed),
                "sn0_policy": "free" if free_sn0 else "fixed",
                "sn0_treatment": sn0_treatment,
                "sn0_identifiability": sn0_identifiability,
                "sn0_scale": float(sn0_scale),
                "sn0_units": (
                    "desilike P(k) reports dimensionless sn0 and adds sn0/nbar to P0(k); "
                    "in xi(s>0), the physical stochastic constant is a zero-lag contact term. "
                    "finite-k-constant is retained only as an explicit diagnostic."
                ),
            },
            "data": {"nradial_bins": int(s.size), "ndata_vector": int(xi.size), "chi2_map": float(chi2(map_theta))},
            "precision": precision_meta,
            "precision_source": precision_source,
            "precision_used": precision,
            "optimizer": {
                "success": bool(opt.success),
                "optimized_parameter_names": optimizer_names,
                "x": [float(v) for v in expand_sampled_theta(optimizer_sampled)],
                "raw_x": [float(v) for v in optimizer_raw.ravel()],
                "walker_initialization_center": [float(v) for v in expand_sampled_theta(center_sampled)],
                "message": str(opt.message),
                "chi2": float(chi2(expand_sampled_theta(optimizer_sampled))),
                "prior_penalty_relative": float(-2.0 * log_prior(expand_sampled_theta(optimizer_sampled))),
                "negative_2_log_posterior_relative": float(objective(expand_sampled_theta(optimizer_sampled))),
            },
            "mcmc": {
                "nwalkers": int(nwalkers),
                "nsteps": int(nsteps),
                "burnin": int(burnin),
                "seed": int(seed),
                "nsamples": int(chain.shape[0]),
                "ndim_sampled": int(len(sampled_names)),
                "sampled_parameter_names": sampled_names,
                "optimizer_only": bool(optimizer_only),
                "mean_acceptance_fraction": sampler_acceptance,
            },
        }
    )
    if model == "formal_gic" and formal_gic_window is not None:
        summary["formal_gic"] = {
            "sigma_w2_map": float(sigma_w2(pk0_dense(map_theta))),
            "window": formal_gic_window["meta"],
        }
    return summary, chain, logp, model_map


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--xi-path", type=Path, default=DEFAULT_XI)
    parser.add_argument("--zeff-path", type=Path, default=DEFAULT_ZEFF)
    parser.add_argument("--zeff-override", type=float, default=None)
    parser.add_argument("--sample", choices=sorted(SAMPLE_CONFIGS), default=SAMPLE)
    parser.add_argument("--covariance-path", type=Path, default=None)
    parser.add_argument(
        "--covariance-key",
        type=str,
        default="covariance_single_realization",
        help="Explicit NPZ key used by the likelihood, e.g. covariance_likelihood_total.",
    )
    parser.add_argument("--covariance-mode", choices=["npz", "diagonal-smoke"], default="npz")
    parser.add_argument(
        "--precision-mode",
        choices=sorted(PRECISION_MODES),
        default="inverse-covariance",
        help=(
            "inverse-covariance preserves the historical pinv(covariance) likelihood. "
            "rascalc-corrected uses the symmetrized RascalC full_theory_precision from "
            "the same NPZ and is allowed only for the complete unsliced 30-bin monopole."
        ),
    )
    parser.add_argument("--max-cov-corr-condition", type=float, default=1.0e5)
    parser.add_argument("--allow-ill-conditioned-covariance", action="store_true")
    parser.add_argument("--output-dir", type=Path, default=FIT_DIR / "lrg2_rsd_monopole")
    parser.add_argument("--models", type=str, default="formal_gic")
    parser.add_argument("--fit-ells", type=str, default="0", help="Comma-separated xi multipoles to fit, e.g. 0 or 0,2.")
    parser.add_argument("--rmin", type=float, default=80.0)
    parser.add_argument("--rmax", type=float, default=350.0)
    parser.add_argument("--random-path", type=Path, default=DEFAULT_RANDOM_CATALOG)
    parser.add_argument("--formal-gic-window-path", type=Path, default=None)
    parser.add_argument("--formal-gic-window-nsub", type=int, default=200000)
    parser.add_argument("--formal-gic-window-seed", type=int, default=20260704)
    parser.add_argument("--formal-gic-window-nthreads", type=int, default=8)
    parser.add_argument("--force-formal-gic-window", action="store_true")
    parser.add_argument("--trust-formal-gic-window-cache", action="store_true")
    parser.add_argument("--p-fixed", type=float, default=None, help="PNG bias p parameter. Defaults to the selected --sample value.")
    parser.add_argument("--p0", type=float, default=P0_DEFAULT)
    parser.add_argument("--theory-boxsize", type=float, default=BOX_SIZE)
    parser.add_argument("--kmax", type=float, default=5.0)
    parser.add_argument("--ndense", type=int, default=60000)
    parser.add_argument("--cosmology", type=str, default=DEFAULT_COSMOLOGY)
    parser.add_argument(
        "--xi-kernel",
        choices=["center", "shell-averaged"],
        default="shell-averaged",
        help="Projection kernel for xi_model: bin-center j0(ks), or shell-averaged j0 over each s bin.",
    )
    parser.add_argument(
        "--xi-projector",
        choices=["discrete-box", "continuous-hankel"],
        default="discrete-box",
        help="Mode sum used for P(k)->xi. continuous-hankel is a diagnostic for lightcone/DESI-style comparisons.",
    )
    parser.add_argument("--projection-kmin", type=float, default=None)
    parser.add_argument("--sigma-s-min", type=float, default=0.0)
    parser.add_argument("--sigma-s-max", type=float, default=30.0)
    parser.add_argument(
        "--sigma-s-fixed",
        type=float,
        default=None,
        help="Fix sigma_s exactly and remove it from the optimizer/MCMC dimensions.",
    )
    parser.add_argument(
        "--b1-gaussian-prior",
        type=float,
        nargs=2,
        default=None,
        metavar=("MEAN", "SIGMA"),
        help=(
            "Optional normalized-shape Gaussian prior on b1; the additive normalization is "
            "irrelevant for posterior sampling.  Intended for explicitly labelled conditional diagnostics."
        ),
    )
    parser.add_argument(
        "--fog-model",
        choices=list(FOG_MODELS),
        default="lorentzian",
        help="FoG damping convention. 'lorentzian' matches desilike PNGTracerSpectrum2Poles auto spectra.",
    )
    parser.add_argument("--sn0-fixed", type=float, default=0.0)
    parser.add_argument("--free-sn0", action="store_true")
    parser.add_argument(
        "--sn0-scale",
        type=float,
        default=1.0,
        help="Multiplier converting the reported sn0 nuisance to the additive P0(k) term; desilike's default PNG convention uses 10000.",
    )
    parser.add_argument(
        "--sn0-treatment",
        choices=["contact", "finite-k-constant"],
        default="contact",
        help=(
            "How to map a P(k)-space constant stochastic term into xi(s). "
            "'contact' treats it as a zero-lag term with no effect for s>0. "
            "'finite-k-constant' keeps the old diagnostic truncated-k ringing model."
        ),
    )
    parser.add_argument("--sn0-prior", type=float, nargs=2, default=(-1.0, 1.0))
    parser.add_argument(
        "--data-window-treatment",
        choices=["deconvolve-if-available", "deconvolve-required", "raw"],
        default="raw",
        help="Diagnostic transform for the measured xi vector when the covariance file stores an RR-window inverse.",
    )
    parser.add_argument("--nwalkers", type=int, default=48)
    parser.add_argument("--nsteps", type=int, default=1200)
    parser.add_argument("--burnin", type=int, default=300)
    parser.add_argument("--optimizer-only", action="store_true", help="Diagnostic mode: write the optimizer MAP without running MCMC.")
    parser.add_argument("--seed", type=int, default=20260704)
    args = parser.parse_args()
    if int(args.formal_gic_window_nthreads) > 8:
        raise ValueError("--formal-gic-window-nthreads must be <=8 for the Task44 login-node CPU policy")

    bin_config = lrg_bin_config(args.sample)
    p_fixed = float(bin_config["p_fixed"] if args.p_fixed is None else args.p_fixed)
    fit_ells = parse_ells(str(args.fit_ells))
    zeff_data = np.load(args.zeff_path, allow_pickle=False)
    zeff_from_file = float(np.asarray(zeff_data["zeff"]).item())
    zeff = zeff_from_file if args.zeff_override is None else float(args.zeff_override)
    s, xi, rr, s_edges, xi_meta, xi_by_ell = load_xi(args.xi_path, float(args.rmin), float(args.rmax), fit_ells)
    cov, cov_meta = load_covariance(
        args.covariance_path,
        mode=str(args.covariance_mode),
        covariance_key=str(args.covariance_key),
        xi=xi,
        ells=fit_ells,
        rmin=float(args.rmin),
        rmax=float(args.rmax),
        max_corr_condition=float(args.max_cov_corr_condition),
        allow_ill_conditioned=bool(args.allow_ill_conditioned_covariance),
    )
    covariance_provenance: dict[str, Any] | None = None
    if args.covariance_path is not None:
        covariance_path_resolved = Path(args.covariance_path).resolve(strict=True)
        covariance_stat = covariance_path_resolved.stat()
        covariance_provenance = {
            "path": str(covariance_path_resolved),
            "sha256": file_sha256(covariance_path_resolved),
            "size_bytes": int(covariance_stat.st_size),
            "mtime_ns_at_load": int(covariance_stat.st_mtime_ns),
        }
    precision_override: np.ndarray | None = None
    precision_override_meta: dict[str, Any] | None = None
    if str(args.precision_mode) == "rascalc-corrected":
        precision_override, precision_override_meta = load_rascalc_corrected_precision(
            args.covariance_path,
            covariance_mode=str(args.covariance_mode),
            cov=cov,
            cov_meta=cov_meta,
            s=s,
            xi=xi,
            ells=fit_ells,
            rmin=float(args.rmin),
            rmax=float(args.rmax),
        )
    xi_raw = np.asarray(xi, dtype="f8").copy()
    xi_by_ell_raw = {int(ell): np.asarray(values, dtype="f8").copy() for ell, values in xi_by_ell.items()}
    xi, xi_by_ell, data_window_meta = apply_data_window_treatment(
        xi,
        xi_by_ell,
        s=s,
        ells=fit_ells,
        cov_meta=cov_meta,
        treatment=str(args.data_window_treatment),
    )
    theory = build_theory_context(
        zeff,
        kmax=float(args.kmax),
        ndense=int(args.ndense),
        boxsize=float(args.theory_boxsize),
        cosmology=str(args.cosmology),
    )
    model_names = [item.strip() for item in str(args.models).split(",") if item.strip()]
    unknown_models = sorted(set(model_names) - ALLOWED_MODELS)
    if unknown_models:
        raise ValueError(f"Task44 science fits only allow {sorted(ALLOWED_MODELS)}; got {unknown_models}")
    formal_window: dict[str, Any] | None = None
    if "formal_gic" in model_names:
        window_path = args.formal_gic_window_path
        if window_path is None:
            window_path = args.output_dir / (
                f"task44_{args.sample}_formal_gic_window_nsub{args.formal_gic_window_nsub}_"
                f"seed{args.formal_gic_window_seed}.npz"
            )
        formal_window = load_or_build_w2(
            theory=theory,
            random_path=args.random_path,
            window_path=window_path,
            nsub=int(args.formal_gic_window_nsub),
            seed=int(args.formal_gic_window_seed),
            nthreads=int(args.formal_gic_window_nthreads),
            p0=float(args.p0),
            boxsize=float(args.theory_boxsize),
            force=bool(args.force_formal_gic_window),
            trust_cache=bool(args.trust_formal_gic_window_cache),
        )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    summaries: dict[str, Any] = {}
    chains: dict[str, np.ndarray] = {}
    logps: dict[str, np.ndarray] = {}
    model_maps: dict[str, np.ndarray] = {}
    for i, model in enumerate(model_names):
        summary, chain, logp, model_map = run_one_model(
            model=model,
            s=s,
            xi=xi,
            rr=rr,
            s_edges=s_edges,
            cov=cov,
            precision_override=precision_override,
            precision_override_meta=precision_override_meta,
            theory=theory,
            formal_gic_window=formal_window,
            p_fixed=p_fixed,
            fit_ells=fit_ells,
            xi_kernel=str(args.xi_kernel),
            xi_projector=str(args.xi_projector),
            projection_kmin=None if args.projection_kmin is None else float(args.projection_kmin),
            fog_model=str(args.fog_model),
            sn0_fixed=float(args.sn0_fixed),
            free_sn0=bool(args.free_sn0),
            sn0_scale=float(args.sn0_scale),
            sn0_treatment=str(args.sn0_treatment),
            sn0_prior=(float(args.sn0_prior[0]), float(args.sn0_prior[1])),
            sigma_s_prior=(float(args.sigma_s_min), float(args.sigma_s_max)),
            sigma_s_fixed=None if args.sigma_s_fixed is None else float(args.sigma_s_fixed),
            b1_gaussian_prior=(
                None
                if args.b1_gaussian_prior is None
                else (float(args.b1_gaussian_prior[0]), float(args.b1_gaussian_prior[1]))
            ),
            nwalkers=int(args.nwalkers),
            nsteps=int(args.nsteps),
            burnin=int(args.burnin),
            seed=int(args.seed) + 100 * i,
            optimizer_only=bool(args.optimizer_only),
        )
        summaries[model] = summary
        chains[f"{model}_chain"] = chain
        logps[f"{model}_log_prob"] = logp
        model_maps[f"{model}_model_map"] = model_map
        print(
            "[fit] {model} fNL={fnl:.2f} -{lo:.2f} +{hi:.2f} b1={b1:.3f} sigma_s={sig:.3f}".format(
                model=model,
                fnl=summary["fnl_loc"]["q50"],
                lo=summary["fnl_loc"]["q50"] - summary["fnl_loc"]["q16"],
                hi=summary["fnl_loc"]["q84"] - summary["fnl_loc"]["q50"],
                b1=summary["b1"]["q50"],
                sig=summary["sigma_s"]["q50"],
            )
        )

    if model_names:
        first_summary = summaries[model_names[0]]
        precision_used = np.asarray(first_summary["precision_used"], dtype="f8")
        precision_source = str(first_summary["precision_source"])
        precision_record = dict(first_summary["precision"])
    else:
        # This is unreachable for normal science calls but keeps output
        # construction explicit if an empty --models string is supplied.
        precision_used = (
            np.linalg.pinv(cov, rcond=PINV_RCOND)
            if precision_override is None
            else np.asarray(precision_override, dtype="f8")
        )
        precision_source = (
            f"numpy.linalg.pinv(covariance_used, rcond={PINV_RCOND:.1e})"
            if precision_override_meta is None
            else str(precision_override_meta["source"])
        )
        precision_record = (
            {"mode": "inverse-covariance", "source": precision_source}
            if precision_override_meta is None
            else dict(precision_override_meta)
        )

    payload = {
        "task": "task44_fit_lrg_bin_rsd",
        "sample": str(args.sample),
        "sample_label": str(bin_config["label"]),
        "xi_path": str(args.xi_path),
        "zeff_path": str(args.zeff_path),
        "zeff_from_file": float(zeff_from_file),
        "zeff_override": None if args.zeff_override is None else float(args.zeff_override),
        "covariance": public_covariance_meta(cov_meta),
        "covariance_provenance": covariance_provenance,
        "precision_mode": str(args.precision_mode),
        "precision_source": precision_source,
        "precision_used": precision_used,
        "precision_diagnostics": precision_record,
        "data_window_treatment": data_window_meta,
        "random_path": str(args.random_path),
        "fit_range": {
            "rmin": float(args.rmin),
            "rmax": float(args.rmax),
            "nradial_bins": int(s.size),
            "ndata_vector": int(xi.size),
            "fit_ells": [int(v) for v in fit_ells],
        },
        "zeff": float(zeff),
        "p0": float(args.p0),
        "p_fixed": p_fixed,
        "theory": {
            "boxsize": float(theory["boxsize"]),
            "kfund": float(theory["kfund"]),
            "kmax": float(args.kmax),
            "ndense": int(args.ndense),
            "cosmology": str(args.cosmology),
            "cosmology_meta": theory["cosmology_meta"],
            "f_growth": float(theory["f_growth"]),
            "growth_rate_method": theory["growth_rate_method"],
            "xi_kernel": str(args.xi_kernel),
            "xi_projector": str(args.xi_projector),
            "projection_kmin": None if args.projection_kmin is None else float(args.projection_kmin),
            "fog_model": str(args.fog_model),
            "fog": fog_description(str(args.fog_model)),
            "sigma_s_policy": "fixed" if args.sigma_s_fixed is not None else "free",
            "sigma_s_fixed": None if args.sigma_s_fixed is None else float(args.sigma_s_fixed),
            "sn0_fixed": float(args.sn0_fixed),
            "sn0_policy": "free" if bool(args.free_sn0) else "fixed",
            "sn0_treatment": str(args.sn0_treatment),
            "sn0_identifiability": (
                "prior_only_contact_term_for_s_gt_0"
                if str(args.sn0_treatment) == "contact"
                else "likelihood_constrained_by_diagnostic_finite_k_ringing"
            ),
            "sn0_scale": float(args.sn0_scale),
            "sn0_units": (
                "desilike P(k) reports dimensionless sn0 and adds sn0/nbar to P0(k); "
                "in xi(s>0), the physical stochastic constant is a zero-lag contact term. "
                "finite-k-constant is retained only as an explicit diagnostic."
            ),
            "sn0_prior": [float(v) for v in args.sn0_prior],
        },
        "xi_meta": xi_meta,
        "sampling": {
            "nwalkers": int(args.nwalkers),
            "nsteps": int(args.nsteps),
            "burnin": int(args.burnin),
            "seed": int(args.seed),
            "optimizer_only": bool(args.optimizer_only),
            "b1_gaussian_prior": (
                None
                if args.b1_gaussian_prior is None
                else {
                    "mean": float(args.b1_gaussian_prior[0]),
                    "sigma": float(args.b1_gaussian_prior[1]),
                    "interpretation": "external conditional prior; no P(k)-xi cross-covariance included",
                }
            ),
        },
        "models": summaries,
    }
    out_json = args.output_dir / f"task44_{args.sample}_rsd_fit_summary.json"
    out_npz = args.output_dir / f"task44_{args.sample}_rsd_fit_chains.npz"
    write_json(out_json, _jsonable(payload))
    save_payload: dict[str, Any] = {
        "s": s,
        "ells": np.asarray(fit_ells, dtype="i8"),
        "xi_vector": xi,
        "xi_vector_raw": xi_raw,
        "covariance": cov,
        "precision_used": precision_used,
        "precision_source": np.asarray(precision_source),
        "precision_mode": np.asarray(str(args.precision_mode)),
        "fit_nwalkers": np.asarray(int(args.nwalkers), dtype="i8"),
        "fit_nsteps": np.asarray(int(args.nsteps), dtype="i8"),
        "fit_burnin": np.asarray(int(args.burnin), dtype="i8"),
        "fit_seed": np.asarray(int(args.seed), dtype="i8"),
    }
    if covariance_provenance is not None:
        save_payload.update(
            {
                "covariance_path": np.asarray(covariance_provenance["path"]),
                "covariance_sha256": np.asarray(covariance_provenance["sha256"]),
                "covariance_size_bytes": np.asarray(
                    covariance_provenance["size_bytes"], dtype="i8"
                ),
            }
        )
    for ell in fit_ells:
        save_payload[f"xi{int(ell)}"] = xi_by_ell[int(ell)]
        save_payload[f"xi{int(ell)}_raw"] = xi_by_ell_raw[int(ell)]
    if tuple(int(v) for v in fit_ells) == (0,):
        save_payload["xi0"] = xi
    np.savez_compressed(out_npz, **save_payload, **chains, **logps, **model_maps)
    print(f"[done] wrote {out_json}")
    print(f"[done] wrote {out_npz}")


if __name__ == "__main__":
    main()
