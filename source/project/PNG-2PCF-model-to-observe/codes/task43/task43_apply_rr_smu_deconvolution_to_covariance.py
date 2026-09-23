#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Apply an RR(s,mu) LS-window deconvolution to a Task43 xi covariance."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np

from task43_rsd_common import atomic_savez, atomic_write_json


def _jsonable(obj: Any) -> Any:
    if isinstance(obj, dict):
        return {str(key): _jsonable(value) for key, value in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_jsonable(value) for value in obj]
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


def correlation_from_covariance(cov: np.ndarray) -> np.ndarray:
    sigma = np.sqrt(np.clip(np.diag(cov), 0.0, np.inf))
    denom = np.outer(sigma, sigma)
    out = np.zeros_like(cov)
    np.divide(cov, denom, out=out, where=denom > 0.0)
    return out


def nearest_spd(matrix: np.ndarray, *, floor_fraction: float) -> tuple[np.ndarray, dict[str, Any]]:
    sym = 0.5 * (np.asarray(matrix, dtype="f8") + np.asarray(matrix, dtype="f8").T)
    evals, evecs = np.linalg.eigh(sym)
    max_eval = float(np.max(evals))
    floor = float(max(float(floor_fraction) * max_eval, 1.0e-30))
    floored = np.maximum(evals, floor)
    cov = (evecs * floored[None, :]) @ evecs.T
    cov = 0.5 * (cov + cov.T)
    return cov, {
        "raw_min_eigenvalue": float(np.min(evals)),
        "raw_max_eigenvalue": max_eval,
        "floor_eigenvalue": floor,
        "n_floored": int(np.count_nonzero(evals < floor)),
        "condition_number_after_floor": float(np.linalg.cond(cov)),
    }


def edge_pairs(edges: np.ndarray) -> np.ndarray:
    edges = np.asarray(edges, dtype="f8")
    if edges.ndim == 2:
        return edges
    return np.column_stack([edges[:-1], edges[1:]])


def parse_ells(text: str) -> tuple[int, ...]:
    values = tuple(int(item.strip()) for item in str(text).split(",") if item.strip())
    if not values:
        raise ValueError("ell list must not be empty")
    if any(ell < 0 or ell % 2 for ell in values):
        raise ValueError(f"ell list must contain non-negative even multipoles; got {values}")
    return values


def covariance_summary(cov: np.ndarray) -> dict[str, Any]:
    sym = 0.5 * (np.asarray(cov, dtype="f8") + np.asarray(cov, dtype="f8").T)
    diag = np.diag(sym)
    corr = correlation_from_covariance(sym)
    eig = np.linalg.eigvalsh(sym)
    eig_corr = np.linalg.eigvalsh(0.5 * (corr + corr.T))
    return {
        "shape": [int(v) for v in sym.shape],
        "sigma": np.sqrt(diag).tolist(),
        "sigma_min": float(np.sqrt(np.min(diag))),
        "sigma_max": float(np.sqrt(np.max(diag))),
        "eig_min": float(eig[0]),
        "eig_max": float(eig[-1]),
        "condition_number": float(np.linalg.cond(sym)),
        "corr_eig_min": float(eig_corr[0]),
        "corr_eig_max": float(eig_corr[-1]),
        "corr_condition_number": float(np.linalg.cond(corr)),
        "corr_offdiag_absmax": float(np.max(np.abs(corr - np.eye(corr.shape[0])))),
    }


def source_covariance_summary(path: Path, meta: dict[str, Any]) -> dict[str, Any]:
    keys = (
        "sample",
        "sample_label",
        "covariance_kind",
        "warning",
        "data_catalog",
        "random_catalog",
        "xi_path",
        "zeff_path",
        "p0",
        "zeff",
        "nreal_for_covariance_of_mean",
        "window",
        "projection",
        "subsample_correction",
        "k_grid",
        "theory",
        "covariance_flags",
        "diagnostics_single",
    )
    summary = {"path": str(path)}
    summary.update({key: meta.get(key) for key in keys if key in meta})
    return summary


def scatter_comparison(xi_summary_path: Path | None, s_edges: np.ndarray, cov: np.ndarray) -> dict[str, Any] | None:
    if xi_summary_path is None or not xi_summary_path.exists():
        return None
    data = np.load(xi_summary_path, allow_pickle=False)
    s_all = np.asarray(data["s"], dtype="f8")
    s = 0.5 * (edge_pairs(s_edges)[:, 0] + edge_pairs(s_edges)[:, 1])
    ids = []
    for value in s:
        match = np.flatnonzero(np.isclose(s_all, value, rtol=0.0, atol=1.0e-8))
        if match.size != 1:
            return {"available": False, "reason": f"missing s={value} in xi summary"}
        ids.append(int(match[0]))
    if "xi_multipoles_by_phase" in data.files:
        xi_all = np.asarray(data["xi_multipoles_by_phase"], dtype="f8")
        summary_ells = tuple(int(value) for value in np.asarray(data["ells"]).ravel())
        expected_nell = int(np.asarray(cov).shape[0] // len(ids))
        if xi_all.ndim != 3 or expected_nell * len(ids) != np.asarray(cov).shape[0]:
            return {"available": False, "reason": "xi multipole summary and covariance dimensions differ"}
        if xi_all.shape[1] < expected_nell or len(summary_ells) < expected_nell:
            return {"available": False, "reason": "xi summary has too few multipoles"}
        xi = np.concatenate([xi_all[:, iell, ids] for iell in range(expected_nell)], axis=1)
        selected_ells = list(summary_ells[:expected_nell])
    elif "xi0_all" in data.files:
        xi = np.asarray(data["xi0_all"], dtype="f8")[:, ids]
        selected_ells = [0]
    else:
        return {"available": False, "reason": "xi summary has neither xi_multipoles_by_phase nor xi0_all"}
    if np.asarray(cov).shape != (xi.shape[1], xi.shape[1]):
        return {"available": False, "reason": "selected xi vector and covariance dimensions differ"}
    mean = np.mean(xi, axis=0)
    precision = np.linalg.pinv(cov, rcond=1.0e-10)
    chi = np.einsum("ij,jk,ik->i", xi - mean, precision, xi - mean)
    scatter = np.cov(xi, rowvar=False, ddof=1)
    sigma_cov = np.sqrt(np.diag(cov))
    sigma_scatter = np.sqrt(np.diag(scatter))
    return {
        "available": True,
        "xi_summary_path": str(xi_summary_path),
        "nreal": int(xi.shape[0]),
        "nbins": int(xi.shape[1]),
        "ells": selected_ells,
        "expected_chi2_mean_about_sample_mean": float((xi.shape[0] - 1) / xi.shape[0] * xi.shape[1]),
        "chi2_about_empirical_mean": [float(v) for v in chi],
        "chi2_mean_per_realization": float(np.mean(chi)),
        "chi2_min": float(np.min(chi)),
        "chi2_max": float(np.max(chi)),
        "sample_std_over_cov_sigma": (sigma_scatter / sigma_cov).tolist(),
        "sample_std_over_cov_sigma_median": float(np.median(sigma_scatter / sigma_cov)),
        "sample_std_over_cov_sigma_min": float(np.min(sigma_scatter / sigma_cov)),
        "sample_std_over_cov_sigma_max": float(np.max(sigma_scatter / sigma_cov)),
        "scatter_covariance": covariance_summary(scatter),
    }


def build_rr_window_matrix(
    *,
    rr_path: Path,
    s_edges: np.ndarray,
    ells: tuple[int, ...],
    ellsin: tuple[int, ...],
    kind: str,
    fkp_norm: float | None,
    resolution: int,
) -> np.ndarray:
    from lsstypes.types import Count2, compute_RR2_window

    rr_data = np.load(rr_path, allow_pickle=False)
    rr_counts = np.asarray(rr_data["rr_counts"], dtype="f8")
    rr_norm = np.asarray(rr_data["rr_norm"], dtype="f8")
    rr_s = np.asarray(rr_data["s"], dtype="f8")
    rr_mu = np.asarray(rr_data["mu"], dtype="f8")
    rr_s_edges = edge_pairs(np.asarray(rr_data["s_edge_pairs"] if "s_edge_pairs" in rr_data.files else rr_data["s_edges"], dtype="f8"))
    mu_edges_raw = np.asarray(rr_data["mu_edges"], dtype="f8")
    rr_mu_edges = edge_pairs(mu_edges_raw)
    norm_scale = 1.0 if fkp_norm is None else float(fkp_norm)
    rr = Count2(
        counts=rr_counts,
        norm=rr_norm * norm_scale,
        s=rr_s,
        s_edges=rr_s_edges,
        mu=rr_mu,
        mu_edges=rr_mu_edges,
        coords=["s", "mu"],
    )
    window = compute_RR2_window(rr, edges=edge_pairs(s_edges), ells=ells, ellsin=ellsin, kind=str(kind), resolution=int(resolution))
    return np.asarray(window.value(), dtype="f8")


def select_multipole_covariance(cov: np.ndarray, *, s_size: int, input_ells: tuple[int, ...], output_ells: tuple[int, ...]) -> np.ndarray:
    cov = np.asarray(cov, dtype="f8")
    if cov.shape != (s_size * len(input_ells), s_size * len(input_ells)):
        raise ValueError(f"multipole covariance shape {cov.shape} inconsistent with s_size={s_size} input_ells={input_ells}")
    indices: list[int] = []
    for ell in output_ells:
        if int(ell) not in input_ells:
            raise KeyError(f"input covariance has ells={input_ells}, not requested ell={ell}")
        offset = input_ells.index(int(ell)) * s_size
        indices.extend(range(offset, offset + s_size))
    return cov[np.ix_(indices, indices)]


def transform_covariance(cov: np.ndarray, inv_window: np.ndarray, floor_fraction: float) -> tuple[np.ndarray, dict[str, Any]]:
    raw = inv_window @ np.asarray(cov, dtype="f8") @ inv_window.T
    return nearest_spd(raw, floor_fraction=float(floor_fraction))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--covariance-path", type=Path, required=True)
    parser.add_argument("--rr-smu-path", type=Path, required=True)
    parser.add_argument("--output-prefix", type=Path, required=True)
    parser.add_argument("--xi-summary", type=Path, default=None)
    parser.add_argument("--rr-window-kind", choices=["RR", "RR/RR"], default="RR")
    parser.add_argument("--rr-fkp-norm", type=float, default=None)
    parser.add_argument(
        "--rr-fkp-norm-json",
        type=Path,
        default=None,
        help="Read recommended_rr_fkp_norm_nx_window from the RR-window JSON sidecar.",
    )
    parser.add_argument("--ells", type=str, default="0", help="Output/configuration-space multipoles to deconvolve, e.g. 0 or 0,2.")
    parser.add_argument("--ellsin", type=str, default=None, help="Input multipoles for RR window. Defaults to --ells.")
    parser.add_argument("--resolution", type=int, default=1)
    parser.add_argument("--floor-fraction", type=float, default=1.0e-10)
    args = parser.parse_args()

    if args.rr_fkp_norm is not None and args.rr_fkp_norm_json is not None:
        raise ValueError("use only one of --rr-fkp-norm and --rr-fkp-norm-json")
    rr_fkp_norm_source: str | None = None
    if args.rr_fkp_norm_json is not None:
        rr_norm_metadata = json.loads(args.rr_fkp_norm_json.read_text(encoding="utf-8"))
        if rr_norm_metadata.get("status") != "done":
            raise RuntimeError(f"RR normalization sidecar is not complete: {args.rr_fkp_norm_json}")
        key = "recommended_rr_fkp_norm_nx_window"
        if key not in rr_norm_metadata:
            raise KeyError(f"{args.rr_fkp_norm_json} has no {key}")
        args.rr_fkp_norm = float(rr_norm_metadata[key])
        rr_fkp_norm_source = f"{args.rr_fkp_norm_json}::{key}"

    ells = parse_ells(str(args.ells))
    ellsin = parse_ells(str(args.ellsin)) if args.ellsin is not None else ells
    if ellsin != ells:
        raise ValueError("This RR-deconvolution helper currently requires square multipole windows: --ellsin must equal --ells")
    cov_data = np.load(args.covariance_path, allow_pickle=False)
    s_edges = np.asarray(cov_data["s_edges"], dtype="f8")
    rr_window = build_rr_window_matrix(
        rr_path=args.rr_smu_path,
        s_edges=s_edges,
        ells=ells,
        ellsin=ellsin,
        kind=str(args.rr_window_kind),
        fkp_norm=args.rr_fkp_norm,
        resolution=int(args.resolution),
    )
    inv_window = np.linalg.inv(rr_window)

    preferred = ["covariance_single_realization", "covariance", "covariance_of_mean"]
    transformed: dict[str, np.ndarray] = {}
    spd: dict[str, Any] = {}
    cov_meta = json.loads(str(np.asarray(cov_data["meta_json"]).item())) if "meta_json" in cov_data.files and str(np.asarray(cov_data["meta_json"]).item()) else {}
    if ells != (0,) and "covariance_xi_multipoles" in cov_data.files:
        input_ells = tuple(int(v) for v in cov_meta.get("theory", {}).get("ells", (0, 2, 4)))
        selected = select_multipole_covariance(
            cov_data["covariance_xi_multipoles"],
            s_size=np.asarray(cov_data["s"]).size,
            input_ells=input_ells,
            output_ells=ellsin,
        )
        transformed["covariance_single_realization"], spd["covariance_single_realization"] = transform_covariance(selected, inv_window, float(args.floor_fraction))
    else:
        for key in preferred + ["covariance_WW", "covariance_WS", "covariance_SS"]:
            if key in cov_data.files:
                transformed[key], spd[key] = transform_covariance(cov_data[key], inv_window, float(args.floor_fraction))

    single_key = "covariance_single_realization" if "covariance_single_realization" in transformed else "covariance"
    single = transformed[single_key]
    out_npz = args.output_prefix.with_suffix(".npz")
    out_json = args.output_prefix.with_suffix(".json")
    if out_npz.exists() or out_json.exists():
        raise FileExistsError(f"immutable RR-deconvolved covariance output exists: {out_npz} / {out_json}")
    out_npz.parent.mkdir(parents=True, exist_ok=True)
    save_payload = {
        "s": np.asarray(cov_data["s"], dtype="f8"),
        "s_edges": s_edges,
        "ells": np.asarray(ells, dtype="i8"),
        "ellsin": np.asarray(ellsin, dtype="i8"),
        "rr_window_matrix": rr_window,
        "rr_window_inverse": inv_window,
        "correlation": correlation_from_covariance(single),
        "meta_json": np.asarray(""),
        "source_meta_json": np.asarray(json.dumps(cov_meta, sort_keys=True)),
    }
    save_payload.update(transformed)
    if "covariance_single_realization" in transformed and "nreal" in cov_data.files:
        nreal = int(np.asarray(cov_data["nreal"]).item())
        save_payload["covariance"] = transformed["covariance_single_realization"] / float(nreal)
        save_payload["covariance_of_mean"] = transformed["covariance_single_realization"] / float(nreal)
        save_payload["nreal"] = np.array(nreal, dtype="i8")
    for key in ("zeff", "p0", "covariance_kind"):
        if key in cov_data.files:
            save_payload[key] = cov_data[key]
    payload = {
        "status": "done",
        "task": "task43_apply_rr_smu_deconvolution_to_covariance",
        "warning": "Diagnostic covariance repair. Requires validation against realization scatter before fNL constraints.",
        "input_covariance": str(args.covariance_path),
        "source_covariance": source_covariance_summary(args.covariance_path, cov_meta),
        "input_rr_smu": str(args.rr_smu_path),
        "output_npz": str(out_npz),
        "rr_window": {
            "kind": str(args.rr_window_kind),
            "fkp_norm": None if args.rr_fkp_norm is None else float(args.rr_fkp_norm),
            "fkp_norm_source": rr_fkp_norm_source,
            "ells": [int(v) for v in ells],
            "ellsin": [int(v) for v in ellsin],
            "resolution": int(args.resolution),
            "shape": [int(v) for v in rr_window.shape],
            "diag_min": float(np.min(np.diag(rr_window))),
            "diag_max": float(np.max(np.diag(rr_window))),
            "condition_number": float(np.linalg.cond(rr_window)),
            "offdiag_absmax": float(np.max(np.abs(rr_window - np.diag(np.diag(rr_window))))),
        },
        "covariance_single_realization": covariance_summary(single),
        "spd": spd,
        "scatter_comparison": scatter_comparison(args.xi_summary, s_edges, single),
    }
    save_payload["meta_json"] = np.asarray(json.dumps(payload, sort_keys=True))
    atomic_savez(out_npz, **save_payload)
    atomic_write_json(out_json, _jsonable(payload))
    print(f"[done] wrote {out_npz}")
    if payload["scatter_comparison"]:
        sc = payload["scatter_comparison"]
        print(
            "[scatter] chi2_mean={:.2f} expected={:.2f} sigma_ratio_med={:.2f}".format(
                sc["chi2_mean_per_realization"],
                sc["expected_chi2_mean_about_sample_mean"],
                sc["sample_std_over_cov_sigma_median"],
            )
        )


if __name__ == "__main__":
    main()
