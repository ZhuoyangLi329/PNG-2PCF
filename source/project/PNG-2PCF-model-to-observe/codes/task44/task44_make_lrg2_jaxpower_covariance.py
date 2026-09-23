#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Build a Task44 LRG-bin fNL=0 Gaussian survey-window covariance with jaxpower."""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Any

for _name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_name, "1")

import numpy as np

from task44_config import (
    BOX_SIZE,
    COV_DIR,
    DEFAULT_DATA_CATALOG,
    DEFAULT_RANDOM_CATALOG,
    DEFAULT_XI,
    DEFAULT_ZEFF,
    K_FUND,
    P0_DEFAULT,
    SAMPLE,
    SAMPLE_CONFIGS,
    S_EDGES,
    covariance_prefix,
    format_float_tag,
    lrg_bin_config,
    write_json,
)
from task44_rsd_theory import FOG_MODELS, evaluate_rsd_png_monopole_pk, evaluate_rsd_png_multipole_pk, fog_description, growth_rate_approx


PROJECT_ROOT = Path("/pscratch/sd/l/lzy/PNG-2PCF-model-to-observe")
TASK43_DIR = PROJECT_ROOT / "codes" / "task43"
if str(TASK43_DIR) not in sys.path:
    sys.path.insert(0, str(TASK43_DIR))

from task43_theory_template import COSMOLOGY_CHOICES, DEFAULT_COSMOLOGY, build_template_arrays, load_task41  # noqa: E402


def make_edges(vmin: float, vmax: float, step: float) -> np.ndarray:
    edges = np.arange(float(vmin), float(vmax) + 0.5 * float(step), float(step), dtype="f8")
    if edges[-1] < float(vmax):
        edges = np.append(edges, float(vmax))
    edges[0] = float(vmin)
    edges[-1] = float(vmax)
    return edges


def make_piecewise_window_edges(
    *,
    smax: float,
    ds: float,
    low_s_max: float | None,
    high_ds: float | None,
) -> dict[str, Any]:
    """Build a window-separation grid with an optional coarser large-s tail."""

    if low_s_max is None or high_ds is None:
        edges = make_edges(0.0, float(smax), float(ds))
        return {
            "edges": edges,
            "meta": {
                "mode": "uniform",
                "s_min": 0.0,
                "s_max": float(smax),
                "ds": float(ds),
                "nbins": int(edges.size - 1),
            },
        }
    low_s_max = float(low_s_max)
    high_ds = float(high_ds)
    if not 0.0 < low_s_max < float(smax):
        raise ValueError("--window-low-s-max must satisfy 0 < low_s_max < window_s_max")
    if not high_ds >= float(ds):
        raise ValueError("--window-high-ds must be >= --window-ds")
    low_edges = make_edges(0.0, low_s_max, float(ds))
    high_edges = make_edges(low_s_max, float(smax), high_ds)
    edges = np.concatenate([low_edges[:-1], high_edges])
    keep = np.concatenate([[True], np.diff(edges) > 1.0e-12])
    edges = edges[keep]
    edges[0] = 0.0
    edges[-1] = float(smax)
    return {
        "edges": edges,
        "meta": {
            "mode": "piecewise_large_s_tail",
            "s_min": 0.0,
            "s_max": float(smax),
            "ds": float(ds),
            "low_s_max": low_s_max,
            "high_ds": high_ds,
            "nbins": int(edges.size - 1),
        },
    }


def make_k_edges(
    *,
    kmin: float,
    kmax: float,
    dk: float,
    low_k_max: float | None = None,
    low_k_dk: float | None = None,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Build a possibly refined low-k grid for steep PNG covariance spectra."""

    if low_k_max is None or low_k_dk is None:
        edges = make_edges(float(kmin), float(kmax), float(dk))
        return edges, {"mode": "uniform", "kmin": float(kmin), "kmax": float(kmax), "dk": float(dk), "nk": int(edges.size - 1)}

    low_k_max = float(low_k_max)
    low_k_dk = float(low_k_dk)
    if not float(kmin) < low_k_max < float(kmax):
        raise ValueError("--low-k-max must satisfy kmin < low_k_max < kmax")
    if not (0.0 < low_k_dk <= float(dk)):
        raise ValueError("--low-k-dk must be positive and <= --dk")

    low_edges = make_edges(float(kmin), low_k_max, low_k_dk)
    high_edges = make_edges(low_k_max, float(kmax), float(dk))
    edges = np.concatenate([low_edges[:-1], high_edges])
    # Remove accidental duplicates from roundoff while preserving order.
    keep = np.concatenate([[True], np.diff(edges) > 1.0e-12])
    edges = edges[keep]
    edges[0] = float(kmin)
    edges[-1] = float(kmax)
    return edges, {
        "mode": "piecewise_low_k_refined",
        "kmin": float(kmin),
        "kmax": float(kmax),
        "dk": float(dk),
        "low_k_max": low_k_max,
        "low_k_dk": low_k_dk,
        "nk": int(edges.size - 1),
    }


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


def covariance_diagnostics(cov: np.ndarray) -> dict[str, Any]:
    sym = 0.5 * (np.asarray(cov, dtype="f8") + np.asarray(cov, dtype="f8").T)
    eig = np.linalg.eigvalsh(sym)
    diag = np.diag(sym)
    corr = correlation_from_covariance(sym)
    return {
        "shape": [int(v) for v in sym.shape],
        "min_eigenvalue": float(np.min(eig)),
        "max_eigenvalue": float(np.max(eig)),
        "condition_number": float(np.linalg.cond(sym)),
        "diag_min": float(np.min(diag)),
        "diag_max": float(np.max(diag)),
        "sigma_min": float(np.sqrt(np.min(diag))),
        "sigma_max": float(np.sqrt(np.max(diag))),
        "corr_min": float(np.min(corr)),
        "corr_max": float(np.max(corr)),
        "corr_nearest_offdiag_max_abs": float(np.max(np.abs(corr - np.eye(corr.shape[0])))),
    }


def infer_mesh_attrs(positions: list[np.ndarray], *, meshsize: int, pad: float) -> tuple[Any, dict[str, Any]]:
    from jaxpower import MeshAttrs

    mins = np.min(np.vstack([np.min(pos, axis=0) for pos in positions]), axis=0)
    maxs = np.max(np.vstack([np.max(pos, axis=0) for pos in positions]), axis=0)
    center = 0.5 * (mins + maxs)
    span = maxs - mins
    boxsize_scalar = float(np.max(span) + 2.0 * float(pad))
    mattrs = MeshAttrs(meshsize=[int(meshsize)] * 3, boxsize=[boxsize_scalar] * 3, boxcenter=center)
    meta = {
        "meshsize": int(meshsize),
        "boxsize": [boxsize_scalar] * 3,
        "boxcenter": [float(v) for v in center],
        "catalog_min": [float(v) for v in mins],
        "catalog_max": [float(v) for v in maxs],
        "catalog_span": [float(v) for v in span],
        "pad": float(pad),
    }
    return mattrs, meta


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


def infer_p0_from_catalog_path(path: Path) -> float | None:
    """Infer an FKP P0 value from Task44 catalog tags such as ``fkpP050000``."""

    marker = "fkpP0"
    text = str(path)
    start = text.rfind(marker)
    if start < 0:
        return None
    tail = text[start + len(marker) :]
    token = []
    for char in tail:
        if char.isdigit() or char in {"p", "m"}:
            token.append(char)
        else:
            break
    if not token:
        return None
    try:
        return float("".join(token).replace("p", ".").replace("m", "-"))
    except ValueError:
        return None


def load_catalog(
    path: Path,
    *,
    max_rows: int | None,
    seed: int,
    rescale_subsample: bool,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, Any]]:
    path = Path(path)
    is_hdf5 = path.suffix.lower() in {".h5", ".hdf5"}
    if is_hdf5:
        import h5py

        data = h5py.File(path, "r")
    else:
        data = np.load(path, allow_pickle=False)
    n_total = int(data["Z"].shape[0])
    if max_rows is not None and int(max_rows) > 0 and int(max_rows) < n_total:
        rng = np.random.default_rng(int(seed))
        choice = np.sort(rng.choice(n_total, size=int(max_rows), replace=False))
    else:
        choice = slice(None)
    try:
        xyz = np.column_stack(
            [
                np.asarray(data["X"][choice], dtype="f8"),
                np.asarray(data["Y"][choice], dtype="f8"),
                np.asarray(data["Zcart"][choice], dtype="f8"),
            ]
        )
        weight_sel = np.asarray(data["WEIGHT_TOTAL"][choice], dtype="f8")
        nx_sel = np.asarray(data["NX"][choice], dtype="f8")
        z_sel = np.asarray(data["Z"][choice], dtype="f8")
    finally:
        data.close()
    n_used = int(weight_sel.size)
    scale = float(n_total) / float(n_used) if (rescale_subsample and n_used > 0) else 1.0
    weight_scaled = weight_sel * scale
    meta = {
        "path": str(path),
        "n_total": n_total,
        "n_used": n_used,
        "subsample_seed": int(seed),
        "subsample_weight_rescale": bool(rescale_subsample),
        "subsample_weight_scale": scale,
        "weight_sum_unscaled": float(np.sum(weight_sel)),
        "weight_sum_scaled": float(np.sum(weight_scaled)),
        "weight_min_scaled": float(np.min(weight_scaled)),
        "weight_max_scaled": float(np.max(weight_scaled)),
        "nx_mean_used": float(np.mean(nx_sel)),
        "nx_min_used": float(np.min(nx_sel)),
        "nx_max_used": float(np.max(nx_sel)),
        "format": "hdf5" if is_hdf5 else "npz",
        "z_min_used": float(np.min(z_sel)),
        "z_max_used": float(np.max(z_sel)),
    }
    return xyz, weight_scaled, nx_sel, meta


def build_theory_poles(
    *,
    k_edges: np.ndarray,
    zeff: float,
    b1_cov: float,
    fnl_cov: float,
    p_fixed: float,
    sigma_s_cov: float,
    nbar_shot: float,
    cosmology: str,
    theory_ells: tuple[int, ...],
    fog_model: str,
) -> tuple[Any, dict[str, Any]]:
    from jaxpower import Mesh2SpectrumPole, Mesh2SpectrumPoles

    if not theory_ells or theory_ells[0] != 0:
        raise ValueError("theory_ells must start with 0 because Task44 fits xi0 only")
    task41 = load_task41()
    k = 0.5 * (k_edges[:-1] + k_edges[1:])
    template_k = np.geomspace(max(1.0e-5, np.min(k_edges[k_edges > 0]) * 0.5), max(1.0, np.max(k_edges) * 1.5), 5000)
    template, cosmology_meta = build_template_arrays(task41, template_k, z=float(zeff), cosmology=str(cosmology))
    f_growth = growth_rate_approx(float(zeff))
    pks = evaluate_rsd_png_multipole_pk(
        k,
        template,
        fnl_loc=float(fnl_cov),
        b1=float(b1_cov),
        p_fixed=float(p_fixed),
        f_growth=f_growth,
        sigma_s=float(sigma_s_cov),
        ells=theory_ells,
        sn0=0.0,
        fog_model=str(fog_model),
    )
    pk0_check = evaluate_rsd_png_monopole_pk(
        k,
        template,
        fnl_loc=float(fnl_cov),
        b1=float(b1_cov),
        p_fixed=float(p_fixed),
        f_growth=f_growth,
        sigma_s=float(sigma_s_cov),
        sn0=0.0,
        fog_model=str(fog_model),
    )
    shot = 1.0 / float(nbar_shot)
    k_edge_pairs = np.column_stack([k_edges[:-1], k_edges[1:]])
    poles = []
    for ell in theory_ells:
        pk = np.asarray(pks[int(ell)], dtype="f8")
        shot_arr = np.full_like(pk, shot) if int(ell) == 0 else np.zeros_like(pk)
        poles.append(
            Mesh2SpectrumPole(
                k=k,
                k_edges=k_edge_pairs,
                num_raw=pk + shot_arr,
                num_shotnoise=shot_arr,
                norm=np.ones_like(pk),
                ell=int(ell),
            )
        )
    denom = np.maximum(1.0, np.abs(pk0_check))
    pk0_rel = float(np.max(np.abs(np.asarray(pks[0], dtype="f8") - pk0_check) / denom))
    meta = {
        "model": "RSD multipole covariance theory",
        "zeff": float(zeff),
        "b1_cov": float(b1_cov),
        "fnl_cov": float(fnl_cov),
        "p_fixed": float(p_fixed),
        "sigma_s_cov": float(sigma_s_cov),
        "fog_model": str(fog_model),
        "fog": fog_description(str(fog_model)),
        "f_growth": float(f_growth),
        "ells": [int(ell) for ell in theory_ells],
        "cosmology": str(cosmology),
        "cosmology_meta": cosmology_meta,
        "nbar_shot": float(nbar_shot),
        "shot_noise_1_over_nbar": float(shot),
        "pk0_quadrature_vs_moment_max_rel": pk0_rel,
        "pk_min_by_ell": {str(int(ell)): float(np.min(pks[int(ell)])) for ell in theory_ells},
        "pk_max_by_ell": {str(int(ell)): float(np.max(pks[int(ell)])) for ell in theory_ells},
        "ptotal0_min": float(np.min(np.asarray(pks[0], dtype="f8") + shot)),
        "ptotal0_max": float(np.max(np.asarray(pks[0], dtype="f8") + shot)),
    }
    return Mesh2SpectrumPoles(poles, ells=theory_ells), meta


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA_CATALOG)
    parser.add_argument("--random", type=Path, default=DEFAULT_RANDOM_CATALOG)
    parser.add_argument("--zeff-path", type=Path, default=DEFAULT_ZEFF)
    parser.add_argument("--xi-path", type=Path, default=DEFAULT_XI)
    parser.add_argument("--sample", choices=sorted(SAMPLE_CONFIGS), default=SAMPLE)
    parser.add_argument("--output-prefix", type=Path, default=None)
    parser.add_argument(
        "--output-compression",
        choices=["compressed", "none"],
        default="compressed",
        help="Use uncompressed npz output when walltime is dominated by compression.",
    )
    parser.add_argument("--p0", type=float, default=None)
    parser.add_argument("--max-random", type=int, default=100000)
    parser.add_argument("--max-data", type=int, default=50000)
    parser.add_argument("--subsample-seed", type=int, default=20260704)
    parser.add_argument("--no-rescale-subsample", action="store_true")
    parser.add_argument("--meshsize", type=int, default=64)
    parser.add_argument("--mesh-pad", type=float, default=400.0)
    parser.add_argument("--window-s-max", type=float, default=350.0)
    parser.add_argument("--window-ds", type=float, default=1.0)
    parser.add_argument(
        "--window-low-s-max",
        type=float,
        default=None,
        help="Optional transition scale for a piecewise window grid; use --window-ds below it.",
    )
    parser.add_argument(
        "--window-high-ds",
        type=float,
        default=None,
        help="Optional coarser window grid spacing above --window-low-s-max.",
    )
    parser.add_argument("--window-basis", choices=["histogram", "bessel"], default="histogram")
    parser.add_argument("--window-ells", type=str, default="0,2,4,6,8")
    parser.add_argument("--theory-ells", type=str, default="0,2,4")
    parser.add_argument("--interpolate-window", action="store_true")
    parser.add_argument("--interpolate-window-ncoords", type=int, default=8192)
    parser.add_argument("--interpolate-window-log10-min", type=float, default=-2.0)
    parser.add_argument("--interpolate-window-log10-max", type=float, default=8.0)
    parser.add_argument("--covariance-flags", type=str, default="smooth,fftlog")
    parser.add_argument("--s-edge-min", type=float, default=float(S_EDGES[0]))
    parser.add_argument("--s-edge-max", type=float, default=float(S_EDGES[-1]))
    parser.add_argument("--s-edge-step", type=float, default=float(np.median(np.diff(S_EDGES))))
    parser.add_argument(
        "--kmin",
        type=float,
        default=float(K_FUND),
        help="Minimum theory k for covariance P multipoles. Default is the Abacus L=2000 box fundamental mode.",
    )
    parser.add_argument("--kmax", type=float, default=3.0001)
    parser.add_argument("--dk", type=float, default=0.002)
    parser.add_argument(
        "--low-k-max",
        type=float,
        default=None,
        help="Optional transition k for a refined low-k covariance grid.",
    )
    parser.add_argument(
        "--low-k-dk",
        type=float,
        default=None,
        help="Optional low-k bin width used below --low-k-max; keeps --dk above it.",
    )
    parser.add_argument("--los", type=str, default="local")
    parser.add_argument("--resampler", type=str, default="cic")
    parser.add_argument("--interlacing", type=int, default=1)
    parser.add_argument("--b1-cov", type=float, default=2.2)
    parser.add_argument("--fnl-cov", type=float, default=0.0)
    parser.add_argument("--p-fixed", type=float, default=None)
    parser.add_argument("--sigma-s-cov", type=float, default=0.0)
    parser.add_argument(
        "--fog-model",
        choices=list(FOG_MODELS),
        default="lorentzian",
        help="FoG damping convention for covariance theory poles.",
    )
    parser.add_argument("--cosmology", choices=COSMOLOGY_CHOICES, default=DEFAULT_COSMOLOGY)
    parser.add_argument("--floor-fraction", type=float, default=1.0e-10)
    args = parser.parse_args()

    t0 = time.perf_counter()
    bin_config = lrg_bin_config(args.sample)
    if args.p_fixed is None:
        args.p_fixed = float(bin_config["p_fixed"])
    if args.p0 is None:
        args.p0 = infer_p0_from_catalog_path(args.data)
        if args.p0 is None:
            args.p0 = infer_p0_from_catalog_path(args.random)
        if args.p0 is None:
            args.p0 = float(P0_DEFAULT)
    if args.output_prefix is None:
        args.output_prefix = covariance_prefix(
            args.sample,
            rrdeconv=False,
            p_fixed=float(args.p_fixed),
            b1_cov=float(args.b1_cov),
            sigma_s_cov=float(args.sigma_s_cov),
        )
    s_edges = make_edges(float(args.s_edge_min), float(args.s_edge_max), float(args.s_edge_step))
    with np.load(args.xi_path, allow_pickle=False) as xi_payload:
        if "s_edges" not in xi_payload.files:
            raise KeyError(f"{args.xi_path} has no s_edges for covariance projection validation")
        xi_s_edges = np.asarray(xi_payload["s_edges"], dtype="f8")
    if not np.array_equal(s_edges, xi_s_edges):
        raise ValueError(
            "covariance projection edges do not match measured xi edges: "
            f"requested={s_edges.tolist()}, xi={xi_s_edges.tolist()}"
        )
    zeff_data = np.load(args.zeff_path, allow_pickle=False)
    zeff = float(np.asarray(zeff_data["zeff"]).item())
    rescale = not bool(args.no_rescale_subsample)
    data_xyz, data_weight, data_nx, data_meta = load_catalog(
        args.data,
        max_rows=args.max_data,
        seed=int(args.subsample_seed) + 1,
        rescale_subsample=rescale,
    )
    random_xyz, random_weight, random_nx, random_meta = load_catalog(
        args.random,
        max_rows=args.max_random,
        seed=int(args.subsample_seed) + 2,
        rescale_subsample=rescale,
    )
    nbar_shot = float(np.mean(random_nx))

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
    window_edges_payload = make_piecewise_window_edges(
        smax=float(args.window_s_max),
        ds=float(args.window_ds),
        low_s_max=args.window_low_s_max,
        high_ds=args.window_high_ds,
    )
    window_s_edges = np.asarray(window_edges_payload["edges"], dtype="f8")
    window_s_edge_meta = dict(window_edges_payload["meta"])
    window_basis = None if args.window_basis == "histogram" else str(args.window_basis)
    window_ells = parse_int_tuple(args.window_ells)
    theory_ells = parse_int_tuple(args.theory_ells)
    required_window_ells = tuple(range(0, 2 * max(theory_ells) + 1, 2))
    missing_window_ells = [ell for ell in required_window_ells if ell not in window_ells]
    if missing_window_ells:
        raise ValueError(
            "window_ells must include all even multipoles up to "
            f"2 * max(theory_ells)={2 * max(theory_ells)}; missing {missing_window_ells}"
        )
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
    k_edges, k_grid_meta = make_k_edges(
        kmin=float(args.kmin),
        kmax=float(args.kmax),
        dk=float(args.dk),
        low_k_max=args.low_k_max,
        low_k_dk=args.low_k_dk,
    )
    if np.isclose(float(k_edges[0]), float(K_FUND), rtol=0.0, atol=1.0e-14):
        k_grid_meta.update(
            {
                "physical_kmin": "abacus_box_fundamental",
                "box_size": float(BOX_SIZE),
                "kfund": float(K_FUND),
                "note": "The first k edge is 2*pi/2000, the smallest nonzero mode present in the Abacus initial conditions.",
            }
        )
    poles, theory_meta = build_theory_poles(
        k_edges=k_edges,
        zeff=zeff,
        b1_cov=float(args.b1_cov),
        fnl_cov=float(args.fnl_cov),
        p_fixed=float(args.p_fixed),
        sigma_s_cov=float(args.sigma_s_cov),
        nbar_shot=nbar_shot,
        cosmology=str(args.cosmology),
        theory_ells=theory_ells,
        fog_model=str(args.fog_model),
    )
    cov_pk_parts = compute_spectrum2_covariance(window2, poles, flags=covariance_flags, return_type="list")
    cov_pk_part_values = {
        name: np.asarray(cov_pk.value(), dtype="f8")
        for name, cov_pk in zip(("WW", "WS", "SS"), cov_pk_parts, strict=True)
    }
    project = np.asarray(matrix_project_to_correlation(s_edges, poles), dtype="f8")
    data_count_scale = float(data_meta["n_total"]) / float(data_meta["n_used"])
    cov_xi_parts_raw_full = {name: project @ value @ project.T for name, value in cov_pk_part_values.items()}
    ns = int(s_edges.size - 1)
    xi0_slice = slice(0, ns)
    cov_xi_parts_raw = {name: value[xi0_slice, xi0_slice] for name, value in cov_xi_parts_raw_full.items()}
    cov_xi_parts = {
        "WW": cov_xi_parts_raw["WW"],
        "WS": cov_xi_parts_raw["WS"] / data_count_scale,
        "SS": cov_xi_parts_raw["SS"] / data_count_scale**2,
    }
    raw_cov = sum(cov_xi_parts.values())
    cov_single, spd_meta = nearest_spd(raw_cov, floor_fraction=float(args.floor_fraction))
    corr = correlation_from_covariance(cov_single)
    out_npz = args.output_prefix.with_suffix(".npz")
    out_json = args.output_prefix.with_suffix(".json")
    out_npz.parent.mkdir(parents=True, exist_ok=True)
    meta = {
        "status": "done",
        "task": "task44_make_lrg_bin_jaxpower_covariance",
        "sample": str(args.sample),
        "sample_label": str(bin_config["label"]),
        "covariance_kind": f"jaxpower_gaussian_survey_window_xi0_fnlcov{format_float_tag(float(args.fnl_cov))}",
        "warning": (
            f"Gaussian fNL_cov={float(args.fnl_cov):.6g} survey-window covariance for Task44 LRG bin; "
            "not connected-trispectrum or full RascalC LS covariance."
        ),
        "data_catalog": str(args.data),
        "random_catalog": str(args.random),
        "xi_path": str(args.xi_path),
        "zeff_path": str(args.zeff_path),
        "p0": float(args.p0),
        "zeff": float(zeff),
        "nreal_for_covariance_of_mean": 1,
        "data": data_meta,
        "random": random_meta,
        "mesh": mesh_meta,
        "window": {
            "interface": "jaxpower.compute_fkp2_covariance_window",
            "basis": str(args.window_basis),
            "ells": [int(ell) for ell in window_ells],
            "los": str(args.los),
            "s_edges": [float(v) for v in window_s_edges],
            "s_edge_grid": window_s_edge_meta,
            "resampler": str(args.resampler),
            "interlacing": int(args.interlacing),
            "interpolation": interpolation_meta,
        },
        "projection": {
            "interface": "jaxpower.cov2.matrix_project_to_correlation",
            "s_edges": [float(v) for v in s_edges],
            "theory_ells": [int(ell) for ell in theory_ells],
            "selected_ell": 0,
            "selected_block": [0, int(ns)],
        },
        "subsample_correction": {
            "data_count_scale": float(data_count_scale),
            "applied_to": {"WW": 1.0, "WS": float(1.0 / data_count_scale), "SS": float(1.0 / data_count_scale**2)},
        },
        "k_grid": k_grid_meta,
        "theory": theory_meta,
        "covariance_flags": list(covariance_flags),
        "spd": spd_meta,
        "output_compression": str(args.output_compression),
        "runtime_sec": float(time.perf_counter() - t0),
    }
    savez = np.savez_compressed if str(args.output_compression) == "compressed" else np.savez
    savez(
        out_npz,
        s=0.5 * (s_edges[:-1] + s_edges[1:]),
        s_edges=s_edges,
        covariance=cov_single,
        covariance_of_mean=cov_single,
        covariance_single_realization=cov_single,
        correlation=corr,
        covariance_pk=cov_pk_part_values["WW"] + cov_pk_part_values["WS"] / data_count_scale + cov_pk_part_values["SS"] / data_count_scale**2,
        covariance_pk_WW=cov_pk_part_values["WW"],
        covariance_pk_WS=cov_pk_part_values["WS"] / data_count_scale,
        covariance_pk_SS=cov_pk_part_values["SS"] / data_count_scale**2,
        covariance_WW=cov_xi_parts["WW"],
        covariance_WS=cov_xi_parts["WS"],
        covariance_SS=cov_xi_parts["SS"],
        covariance_xi_multipoles=cov_xi_parts_raw_full["WW"]
        + cov_xi_parts_raw_full["WS"] / data_count_scale
        + cov_xi_parts_raw_full["SS"] / data_count_scale**2,
        projection_matrix=project,
        zeff=np.array(float(zeff), dtype="f8"),
        nreal=np.array(1, dtype="i8"),
        p0=np.array(float(args.p0), dtype="f8"),
        covariance_kind=np.asarray(f"jaxpower_gaussian_survey_window_xi0_fnlcov{format_float_tag(float(args.fnl_cov))}"),
        meta_json=np.asarray(json.dumps(meta, sort_keys=True)),
    )
    payload = dict(meta)
    payload["output_npz"] = str(out_npz)
    payload["diagnostics_single"] = covariance_diagnostics(cov_single)
    write_json(out_json, payload)
    print(f"[done] wrote {out_npz}")
    print(f"[diag] sigma={payload['diagnostics_single']['sigma_min']:.3e}..{payload['diagnostics_single']['sigma_max']:.3e}")


if __name__ == "__main__":
    main()
