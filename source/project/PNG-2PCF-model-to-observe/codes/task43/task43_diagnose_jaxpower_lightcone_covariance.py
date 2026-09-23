#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Diagnose Task43 lightcone jaxpower covariance normalization.

This script is a reproducible version of the covariance-debug checks used for
the Task43 lightcone `M_min=1.4e13` closure test.  It does not produce an fNL
constraint.  It builds the same FKP survey window as the covariance script,
records weight/alpha/window normalization diagnostics, decomposes Cxi into
WW/WS/SS terms, and optionally compares the result to measured realization
scatter from an xi summary.
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

import numpy as np

from task43_config import PROJECT_ROOT, S_EDGES, SUMMARY_DIR, read_jsonl
from task43_fkp_zeff import load_fkp_summary, total_weight_from_summary
from task43_make_jaxpower_lightcone_covariance import (
    build_theory_poles,
    covariance_diagnostics,
    infer_mesh_attrs,
    load_positions_weights,
    make_edges,
)
from task43_make_rawbox_z0p725_mmin1p3e13_gaussian_covariance import correlation_from_covariance, nearest_spd
from task43_theory_template import COSMOLOGY_CHOICES, DEFAULT_COSMOLOGY


ARCHIVED_TASK43_OUTPUTS = (
    PROJECT_ROOT
    / "old_doc_codes"
    / "task4_task44_cleanup_20260707T061844Z"
    / "moved"
    / "outputs"
    / "task43_outputs"
)


def resolve_task43_input(path: Path) -> Path:
    """Resolve paths moved by the 2026-07-07 Task43 output cleanup."""
    path = Path(path)
    if path.exists():
        return path
    marker = Path("outputs") / "task43_outputs"
    try:
        relative = path.relative_to(PROJECT_ROOT / marker)
    except ValueError:
        raise FileNotFoundError(path) from None
    archived = ARCHIVED_TASK43_OUTPUTS / relative
    if archived.exists():
        return archived
    raise FileNotFoundError(f"missing active and archived Task43 input: {path}; {archived}")


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


def _moments(weight: np.ndarray) -> dict[str, float | int]:
    weight = np.asarray(weight, dtype="f8")
    return {
        "n": int(weight.size),
        "sum_w": float(np.sum(weight)),
        "sum_w2": float(np.sum(weight * weight)),
        "sum_w4": float(np.sum(weight**4)),
        "mean_w": float(np.mean(weight)),
        "mean_w2": float(np.mean(weight * weight)),
        "min_w": float(np.min(weight)),
        "max_w": float(np.max(weight)),
    }


def full_catalog_weight_moments(path: Path, *, fkp_summary: dict[str, np.ndarray], p0: float) -> dict[str, Any]:
    data = np.load(path, allow_pickle=False)
    z = np.asarray(data["Z"], dtype="f8")
    base_weight = np.asarray(data["WEIGHT"], dtype="f8") if "WEIGHT" in data.files else np.ones(z.size, dtype="f8")
    weight = total_weight_from_summary(z, base_weight, fkp_summary, p0=float(p0))
    return _moments(weight)


def analytic_window_limits(fkp_summary: dict[str, np.ndarray], *, p0: float) -> dict[str, float]:
    """Continuous n(z) estimate of the monopole window small-s limits."""
    z_edges = np.asarray(fkp_summary["z_edges"], dtype="f8")
    nbar = np.asarray(fkp_summary["nbar"], dtype="f8")
    volume = np.asarray(fkp_summary["volume_shell"], dtype="f8")
    p0_values = np.asarray(fkp_summary["p0_values"], dtype="f8")
    fkp_weights = np.asarray(fkp_summary["fkp_weights"], dtype="f8")
    match = np.flatnonzero(np.isclose(p0_values, float(p0), rtol=0.0, atol=1.0e-10))
    if match.size != 1:
        raise ValueError(f"P0={p0} not found in FKP summary")
    w = fkp_weights[int(match[0])]
    i_w2 = float(np.sum(volume * (nbar * w) ** 2))
    i_w4 = float(np.sum(volume * (nbar * w) ** 4))
    i_s2 = float(np.sum(volume * (nbar * w * w) ** 2))
    veff_ww = i_w2**2 / i_w4
    return {
        "z_min": float(z_edges[0]),
        "z_max": float(z_edges[-1]),
        "volume_shell_sum": float(np.sum(volume)),
        "i_w2_int_n2_w2": i_w2,
        "i_w4_int_n4_w4": i_w4,
        "i_s2_int_n2_w4": i_s2,
        "ww_s0_expected_int_w4_over_iw2sq": float(i_w4 / i_w2**2),
        "ss_s0_expected_int_s2_over_iw2sq": float(i_s2 / i_w2**2),
        "veff_ww": float(veff_ww),
        "density_eff_from_ww_ss": float(np.sqrt(i_w4 / i_s2)),
    }


def _first_window_leaf(tree: Any) -> Any:
    fields = list(getattr(tree, "fields", []))
    if fields:
        return tree.get(fields[0]), fields[0]
    leaf = next(iter(tree))
    return leaf, None


def summarize_window2(window2: Any) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for kind in ("WW", "WS", "SW", "SS"):
        try:
            tree = window2.get(kind)
        except Exception as exc:  # pragma: no cover - defensive for API drift
            out[kind] = {"available": False, "error": str(exc)}
            continue
        leaf, field = _first_window_leaf(tree)
        kind_out: dict[str, Any] = {"available": True, "field": None if field is None else list(field)}
        for ell in getattr(leaf, "ells", (0,)):
            pole = leaf.get(ell)
            s = np.asarray(pole.coords("s"), dtype="f8")
            value = np.asarray(pole.value(), dtype="f8")
            norm = np.asarray(pole.norm, dtype="f8") if hasattr(pole, "norm") else np.array([])
            finite = np.isfinite(value)
            ell_key = f"ell{int(ell)}"
            kind_out[ell_key] = {
                "n": int(value.size),
                "s_first": float(s[0]) if s.size else None,
                "s_last": float(s[-1]) if s.size else None,
                "value_first": float(value[0]) if value.size else None,
                "value_second": float(value[1]) if value.size > 1 else None,
                "value_min": float(np.min(value[finite])) if np.any(finite) else None,
                "value_max": float(np.max(value[finite])) if np.any(finite) else None,
                "norm_first": float(np.ravel(norm)[0]) if norm.size else None,
            }
        out[kind] = kind_out
    return out


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
        "condition_number": float(eig[-1] / eig[0]) if eig[0] > 0 else None,
        "corr_eig_min": float(eig_corr[0]),
        "corr_eig_max": float(eig_corr[-1]),
        "corr_condition_number": float(eig_corr[-1] / eig_corr[0]) if eig_corr[0] > 0 else None,
        "corr_offdiag_absmax": float(np.max(np.abs(corr - np.eye(corr.shape[0])))),
        "corr_min": float(np.min(corr)),
        "corr_max": float(np.max(corr)),
    }


def scatter_comparison(xi_summary_path: Path, s_edges: np.ndarray, cov: np.ndarray) -> dict[str, Any] | None:
    if xi_summary_path is None or not xi_summary_path.exists():
        return None
    data = np.load(xi_summary_path, allow_pickle=False)
    if "xi0_all" not in data.files:
        return {"available": False, "reason": "xi summary has no xi0_all"}
    s_all = np.asarray(data["s"], dtype="f8")
    s = 0.5 * (s_edges[:-1] + s_edges[1:])
    ids = []
    for value in s:
        match = np.flatnonzero(np.isclose(s_all, value, rtol=0.0, atol=1.0e-8))
        if match.size != 1:
            return {"available": False, "reason": f"missing s={value} in xi summary"}
        ids.append(int(match[0]))
    xi = np.asarray(data["xi0_all"], dtype="f8")[:, ids]
    scatter = np.cov(xi, rowvar=False, ddof=1)
    mean = np.mean(xi, axis=0)
    precision = np.linalg.pinv(cov, rcond=1.0e-10)
    chi = np.einsum("ij,jk,ik->i", xi - mean, precision, xi - mean)
    eig, vec = np.linalg.eigh(0.5 * (cov + cov.T))
    mode_var = ((xi - mean) @ vec / np.sqrt(eig)).var(axis=0, ddof=1)
    return {
        "available": True,
        "xi_summary_path": str(xi_summary_path),
        "nreal": int(xi.shape[0]),
        "scatter_covariance": covariance_summary(scatter),
        "chi2_about_empirical_mean": [float(v) for v in chi],
        "chi2_sum": float(np.sum(chi)),
        "chi2_mean_per_realization": float(np.mean(chi)),
        "empirical_variance_in_cov_eigenmodes": [float(v) for v in mode_var],
    }


def parse_int_tuple(text: str) -> tuple[int, ...]:
    values = tuple(int(item.strip()) for item in str(text).split(",") if item.strip())
    if not values:
        raise ValueError("empty integer tuple")
    return values


def parse_flag_tuple(text: str) -> tuple[str, ...]:
    allowed = {"smooth", "fftlog"}
    flags = tuple(item.strip() for item in str(text).split(",") if item.strip())
    if not flags:
        raise ValueError("empty covariance flags")
    unknown = sorted(set(flags) - allowed)
    if unknown:
        raise ValueError(f"unknown covariance flags: {unknown}")
    return flags


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, default=SUMMARY_DIR.parent / "manifests" / "task43_mmin1p4e13_x25.jsonl")
    parser.add_argument("--fkp-summary", type=Path, default=SUMMARY_DIR / "task43_fkp_zeff_mmin1p4e13_x25.npz")
    parser.add_argument("--xi-summary", type=Path, default=None)
    parser.add_argument("--output-prefix", type=Path, default=SUMMARY_DIR / "jaxpower_covariance_lightcone_mmin1p4e13_ph000_diagnostic")
    parser.add_argument("--phase-index", type=int, default=0)
    parser.add_argument("--p0", type=float, default=10000.0)
    parser.add_argument("--max-random", type=int, default=20000)
    parser.add_argument("--max-data", type=int, default=10000)
    parser.add_argument("--subsample-seed", type=int, default=20260702)
    parser.add_argument("--no-rescale-subsample", action="store_true")
    parser.add_argument("--meshsize", type=int, default=64)
    parser.add_argument("--mesh-pad", type=float, default=400.0)
    parser.add_argument("--window-s-max", type=float, default=350.0)
    parser.add_argument("--window-ds", type=float, default=1.0)
    parser.add_argument("--window-basis", choices=["histogram", "bessel"], default="histogram")
    parser.add_argument("--window-ells", type=str, default="0")
    parser.add_argument("--interpolate-window", action="store_true")
    parser.add_argument("--interpolate-window-ncoords", type=int, default=8192)
    parser.add_argument("--interpolate-window-log10-min", type=float, default=-2.0)
    parser.add_argument("--interpolate-window-log10-max", type=float, default=8.0)
    parser.add_argument("--covariance-flags", type=str, default="smooth")
    parser.add_argument("--s-edge-min", type=float, default=80.0)
    parser.add_argument("--s-edge-max", type=float, default=350.0)
    parser.add_argument("--s-edge-step", type=float, default=30.0)
    parser.add_argument("--kmin", type=float, default=1.0e-4)
    parser.add_argument("--kmax", type=float, default=3.0001)
    parser.add_argument("--dk", type=float, default=0.002)
    parser.add_argument("--los", type=str, default="local")
    parser.add_argument("--resampler", type=str, default="cic")
    parser.add_argument("--interlacing", type=int, default=1)
    parser.add_argument("--b1-cov", type=float, default=2.5)
    parser.add_argument("--fnl-cov", type=float, default=0.0)
    parser.add_argument("--p-fixed", type=float, default=1.1)
    parser.add_argument("--sn0-fixed", type=float, default=0.0)
    parser.add_argument("--cosmology", choices=COSMOLOGY_CHOICES, default=DEFAULT_COSMOLOGY)
    parser.add_argument("--floor-fraction", type=float, default=1.0e-10)
    args = parser.parse_args()

    t0 = time.perf_counter()
    rows = read_jsonl(args.manifest)
    row = rows[int(args.phase_index)]
    halo_catalog_path = resolve_task43_input(Path(row["halo_catalog_path"]))
    random_catalog_path = resolve_task43_input(Path(row["random_catalog_path"]))
    fkp_summary = load_fkp_summary(args.fkp_summary)
    p0_values = np.asarray(fkp_summary["p0_values"], dtype="f8")
    zeff_values = np.asarray(fkp_summary["zeff_random_auto"], dtype="f8")
    p0_match = np.flatnonzero(np.isclose(p0_values, float(args.p0), rtol=0.0, atol=1.0e-10))
    if p0_match.size != 1:
        raise ValueError(f"P0={args.p0} not found in {args.fkp_summary}")
    zeff = float(zeff_values[int(p0_match[0])])
    nbar = np.asarray(fkp_summary["nbar"], dtype="f8")
    nbar_shot = float(np.mean(nbar))

    rescale = not bool(args.no_rescale_subsample)
    data_xyz, data_weight, data_meta = load_positions_weights(
        halo_catalog_path,
        fkp_summary=fkp_summary,
        p0=float(args.p0),
        max_rows=args.max_data,
        seed=int(args.subsample_seed) + 1,
        rescale_subsample=rescale,
    )
    random_xyz, random_weight, random_meta = load_positions_weights(
        random_catalog_path,
        fkp_summary=fkp_summary,
        p0=float(args.p0),
        max_rows=args.max_random,
        seed=int(args.subsample_seed) + 2,
        rescale_subsample=rescale,
    )

    import jax

    jax.config.update("jax_enable_x64", True)
    from jaxpower import (
        BinMesh2CorrelationPoles,
        FKPField,
        ParticleField,
        compute_fkp2_covariance_window,
        compute_spectrum2_covariance,
        interpolate_window_function,
    )
    from jaxpower.cov2 import matrix_project_to_correlation

    mattrs, mesh_meta = infer_mesh_attrs([data_xyz, random_xyz], meshsize=int(args.meshsize), pad=float(args.mesh_pad))
    data_field = ParticleField(data_xyz, weights=data_weight, attrs=mattrs, exchange=False)
    random_field = ParticleField(random_xyz, weights=random_weight, attrs=mattrs, exchange=False)
    fkp = FKPField(data_field, random_field)
    window_s_edges = make_edges(0.0, float(args.window_s_max), float(args.window_ds))
    window_basis = None if args.window_basis == "histogram" else str(args.window_basis)
    window_ells = parse_int_tuple(args.window_ells)
    covariance_flags = parse_flag_tuple(args.covariance_flags)
    window_bin = BinMesh2CorrelationPoles(mattrs, edges=window_s_edges, ells=window_ells, basis=window_basis)
    window2 = compute_fkp2_covariance_window(
        fkp,
        bin=window_bin,
        los=str(args.los),
        resampler=str(args.resampler),
        interlacing=int(args.interlacing),
    )
    interpolation_meta = {"applied": False}
    if args.interpolate_window:
        coords = np.logspace(
            float(args.interpolate_window_log10_min),
            float(args.interpolate_window_log10_max),
            int(args.interpolate_window_ncoords),
        )
        window2 = window2.map(lambda window: interpolate_window_function(window, coords=coords), level=1)
        interpolation_meta = {
            "applied": True,
            "coords": "logspace",
            "log10_min": float(args.interpolate_window_log10_min),
            "log10_max": float(args.interpolate_window_log10_max),
            "ncoords": int(args.interpolate_window_ncoords),
        }

    k_edges = make_edges(float(args.kmin), float(args.kmax), float(args.dk))
    poles, theory_meta = build_theory_poles(
        k_edges=k_edges,
        zeff=zeff,
        b1_cov=float(args.b1_cov),
        fnl_cov=float(args.fnl_cov),
        p_fixed=float(args.p_fixed),
        sn0_fixed=float(args.sn0_fixed),
        nbar_shot=nbar_shot,
        cosmology=str(args.cosmology),
    )
    cov_parts_pk = compute_spectrum2_covariance(window2, poles, flags=covariance_flags, return_type="list")
    s_edges = make_edges(float(args.s_edge_min), float(args.s_edge_max), float(args.s_edge_step))
    project = np.asarray(matrix_project_to_correlation(s_edges, poles), dtype="f8")
    part_names = ("WW", "WS", "SS")
    data_count_scale = float(data_meta["n_total"]) / float(data_meta["n_used"])
    cov_parts_xi_raw = {}
    for name, cov_pk in zip(part_names, cov_parts_pk, strict=True):
        cov_parts_xi_raw[name] = project @ np.asarray(cov_pk.value(), dtype="f8") @ project.T
    cov_parts_xi = {
        "WW": cov_parts_xi_raw["WW"],
        "WS": cov_parts_xi_raw["WS"] / data_count_scale,
        "SS": cov_parts_xi_raw["SS"] / data_count_scale**2,
    }
    uncorrected_cov = sum(cov_parts_xi_raw.values())
    corrected_cov = sum(cov_parts_xi.values())
    cov_single, spd_meta = nearest_spd(corrected_cov, floor_fraction=float(args.floor_fraction))
    corr_single = correlation_from_covariance(cov_single)

    full_data_moments = full_catalog_weight_moments(halo_catalog_path, fkp_summary=fkp_summary, p0=float(args.p0))
    full_random_moments = full_catalog_weight_moments(random_catalog_path, fkp_summary=fkp_summary, p0=float(args.p0))
    used_data_moments = _moments(data_weight)
    used_random_moments = _moments(random_weight)
    weights = {
        "full_data": full_data_moments,
        "full_random": full_random_moments,
        "used_data": used_data_moments,
        "used_random": used_random_moments,
        "alpha_sumw_full": float(full_data_moments["sum_w"] / full_random_moments["sum_w"]),
        "alpha_sumw_used": float(used_data_moments["sum_w"] / used_random_moments["sum_w"]),
        "alpha_sumw2_full": float(full_data_moments["sum_w2"] / full_random_moments["sum_w2"]),
        "alpha_sumw2_used": float(used_data_moments["sum_w2"] / used_random_moments["sum_w2"]),
    }

    component_summary = {}
    diag_total = np.diag(corrected_cov)
    for name in part_names:
        cov = cov_parts_xi[name]
        component_summary[name] = covariance_summary(cov)
        component_summary[name]["diag_fraction_of_total"] = (np.diag(cov) / diag_total).tolist()
        component_summary[name]["raw_before_data_thinning_correction"] = covariance_summary(cov_parts_xi_raw[name])

    out_npz = args.output_prefix.with_suffix(".npz")
    out_json = args.output_prefix.with_suffix(".json")
    out_npz.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        out_npz,
        s=0.5 * (s_edges[:-1] + s_edges[1:]),
        s_edges=s_edges,
        k_edges=k_edges,
        covariance_single_realization=cov_single,
        corrected_covariance_single_realization=corrected_cov,
        uncorrected_covariance_single_realization=uncorrected_cov,
        correlation=correlation_from_covariance(cov_single),
        covariance_WW=cov_parts_xi["WW"],
        covariance_WS=cov_parts_xi["WS"],
        covariance_SS=cov_parts_xi["SS"],
        covariance_WW_raw=cov_parts_xi_raw["WW"],
        covariance_WS_raw=cov_parts_xi_raw["WS"],
        covariance_SS_raw=cov_parts_xi_raw["SS"],
        projection_matrix=project,
        zeff=np.array(zeff, dtype="f8"),
        p0=np.array(float(args.p0), dtype="f8"),
    )
    payload = {
        "status": "done",
        "task": "task43_diagnose_jaxpower_lightcone_covariance",
        "purpose": "debug normalization/window/FKP/shot/projection for the Task43 lightcone jaxpower covariance",
        "output_npz": str(out_npz),
        "manifest": str(args.manifest),
        "phase": row["phase"],
        "phase_index": int(args.phase_index),
        "halo_catalog_path": str(halo_catalog_path),
        "random_catalog_path": str(random_catalog_path),
        "catalog_path_resolution": "active path if present, otherwise 2026-07-07 archived Task43 path",
        "fkp_summary": str(args.fkp_summary),
        "p0": float(args.p0),
        "zeff": zeff,
        "data": data_meta,
        "random": random_meta,
        "weights": weights,
        "analytic_window_limits_from_nz": analytic_window_limits(fkp_summary, p0=float(args.p0)),
        "mesh": mesh_meta,
        "window": {
            "interface": "jaxpower.compute_fkp2_covariance_window",
            "basis": args.window_basis,
            "ells": [int(ell) for ell in window_ells],
            "los": str(args.los),
            "s_edges": [float(v) for v in window_s_edges],
            "resampler": str(args.resampler),
            "interlacing": int(args.interlacing),
            "interpolation": interpolation_meta,
            "small_s_summary": summarize_window2(window2),
        },
        "projection": {
            "interface": "jaxpower.cov2.matrix_project_to_correlation",
            "s_edges": [float(v) for v in s_edges],
            "ells": [0],
            "note": "No window_deconvolution is applied in this diagnostic.",
        },
        "subsample_correction": {
            "reason": "jaxpower S shot-window terms use data-weight products; thinning the data catalog preserves W but biases S by N_data_total/N_data_used.",
            "data_count_scale": float(data_count_scale),
            "applied_to": {
                "WW": 1.0,
                "WS": float(1.0 / data_count_scale),
                "SS": float(1.0 / data_count_scale**2),
            },
        },
        "k_grid": {
            "k_edges": [float(v) for v in k_edges],
            "kmin": float(args.kmin),
            "kmax": float(args.kmax),
            "dk": float(args.dk),
            "nk": int(k_edges.size - 1),
            "note": "Use kmax~3 and fine dk for xi covariance. Earlier kmax<=0.8/dk~0.02 diagnostics were low-rank and invalid for fNL constraints.",
        },
        "theory": theory_meta,
        "covariance_flags": list(covariance_flags),
        "components": component_summary,
        "uncorrected_sum_covariance": covariance_summary(uncorrected_cov),
        "corrected_sum_covariance": covariance_summary(corrected_cov),
        "spd": spd_meta,
        "covariance_single_realization": covariance_diagnostics(cov_single),
        "scatter_comparison": scatter_comparison(args.xi_summary, s_edges, cov_single),
        "runtime_sec": float(time.perf_counter() - t0),
        "warning": "Diagnostic only. Do not use as final fNL covariance until normalization/projection checks pass.",
    }
    out_json.write_text(json.dumps(_jsonable(payload), indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"[done] wrote {out_json}")
    print(
        "[diag] sigma={:.3e}..{:.3e} corr_cond={:.3g} offdiag={:.3g}".format(
            payload["corrected_sum_covariance"]["sigma_min"],
            payload["corrected_sum_covariance"]["sigma_max"],
            payload["corrected_sum_covariance"]["corr_condition_number"],
            payload["corrected_sum_covariance"]["corr_offdiag_absmax"],
        )
    )


if __name__ == "__main__":
    main()
