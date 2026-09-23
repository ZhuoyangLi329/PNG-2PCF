#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Build a Task43 lightcone xi0 Gaussian window covariance with jaxpower."""

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

from task43_config import DEFAULT_MANIFEST, S_EDGES, SUMMARY_DIR, read_jsonl
from task43_fkp_zeff import load_fkp_summary, total_weight_from_summary
from task43_make_rawbox_z0p725_mmin1p3e13_gaussian_covariance import correlation_from_covariance, nearest_spd
from task43_theory_template import COSMOLOGY_CHOICES, DEFAULT_COSMOLOGY, build_template_arrays, load_task41


def make_edges(vmin: float, vmax: float, step: float) -> np.ndarray:
    edges = np.arange(float(vmin), float(vmax) + 0.5 * float(step), float(step), dtype="f8")
    if edges[-1] < float(vmax):
        edges = np.append(edges, float(vmax))
    edges[0] = float(vmin)
    edges[-1] = float(vmax)
    return edges


def covariance_diagnostics(cov: np.ndarray) -> dict[str, Any]:
    sym = 0.5 * (cov + cov.T)
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


def load_positions_weights(
    path: Path,
    *,
    fkp_summary: dict[str, np.ndarray],
    p0: float,
    max_rows: int | None,
    seed: int,
    rescale_subsample: bool,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    data = np.load(path, allow_pickle=False)
    z = np.asarray(data["Z"], dtype="f8")
    base_weight = np.asarray(data["WEIGHT"], dtype="f8") if "WEIGHT" in data.files else np.ones(z.size, dtype="f8")
    weight = total_weight_from_summary(z, base_weight, fkp_summary, p0=float(p0))
    n_total = int(z.size)
    if max_rows is not None and int(max_rows) > 0 and int(max_rows) < n_total:
        rng = np.random.default_rng(int(seed))
        choice = np.sort(rng.choice(n_total, size=int(max_rows), replace=False))
    else:
        choice = slice(None)
    xyz = np.column_stack(
        [
            np.asarray(data["X"][choice], dtype="f8"),
            np.asarray(data["Y"][choice], dtype="f8"),
            np.asarray(data["Zcart"][choice], dtype="f8"),
        ]
    )
    weight_sel = np.asarray(weight[choice], dtype="f8")
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
        "z_min_used": float(np.min(np.asarray(z[choice], dtype="f8"))),
        "z_max_used": float(np.max(np.asarray(z[choice], dtype="f8"))),
    }
    return xyz, weight_scaled, meta


def infer_mesh_attrs(
    positions: list[np.ndarray],
    *,
    meshsize: int,
    pad: float,
) -> tuple[Any, dict[str, Any]]:
    from jaxpower import MeshAttrs

    mins = np.min(np.vstack([np.min(pos, axis=0) for pos in positions]), axis=0)
    maxs = np.max(np.vstack([np.max(pos, axis=0) for pos in positions]), axis=0)
    center = 0.5 * (mins + maxs)
    span = maxs - mins
    boxsize_scalar = float(np.max(span) + 2.0 * float(pad))
    boxcenter = center
    mattrs = MeshAttrs(meshsize=[int(meshsize)] * 3, boxsize=[boxsize_scalar] * 3, boxcenter=boxcenter)
    meta = {
        "meshsize": int(meshsize),
        "boxsize": [boxsize_scalar] * 3,
        "boxcenter": [float(v) for v in boxcenter],
        "catalog_min": [float(v) for v in mins],
        "catalog_max": [float(v) for v in maxs],
        "catalog_span": [float(v) for v in span],
        "pad": float(pad),
    }
    return mattrs, meta


def build_theory_poles(
    *,
    k_edges: np.ndarray,
    zeff: float,
    b1_cov: float,
    fnl_cov: float,
    p_fixed: float,
    sn0_fixed: float,
    nbar_shot: float,
    cosmology: str,
) -> tuple[Any, dict[str, Any]]:
    from jaxpower import Mesh2SpectrumPole, Mesh2SpectrumPoles

    task41 = load_task41()
    k = 0.5 * (k_edges[:-1] + k_edges[1:])
    template_k = np.geomspace(max(1.0e-5, np.min(k_edges[k_edges > 0]) * 0.5), max(1.0, np.max(k_edges) * 1.5), 5000)
    template, cosmology_meta = build_template_arrays(task41, template_k, z=float(zeff), cosmology=str(cosmology))
    pk = task41.evaluate_realspace_png_pk(
        k,
        template,
        fnl_loc=float(fnl_cov),
        b1=float(b1_cov),
        p_fixed=float(p_fixed),
        sn0=float(sn0_fixed),
    )
    shot = 1.0 / float(nbar_shot)
    pole0 = Mesh2SpectrumPole(
        k=k,
        k_edges=np.column_stack([k_edges[:-1], k_edges[1:]]),
        num_raw=pk + shot,
        num_shotnoise=np.full_like(pk, shot),
        norm=np.ones_like(pk),
        ell=0,
    )
    meta = {
        "zeff": float(zeff),
        "b1_cov": float(b1_cov),
        "fnl_cov": float(fnl_cov),
        "p_fixed": float(p_fixed),
        "sn0_fixed": float(sn0_fixed),
        "cosmology": str(cosmology),
        "cosmology_meta": cosmology_meta,
        "nbar_shot": float(nbar_shot),
        "shot_noise_1_over_nbar": float(shot),
        "pk_min": float(np.min(pk)),
        "pk_max": float(np.max(pk)),
        "ptotal_min": float(np.min(pk + shot)),
        "ptotal_max": float(np.max(pk + shot)),
        "shot_policy": "Mesh2SpectrumPole num_raw=P_h(k)+1/nbar_shot, num_shotnoise=1/nbar_shot, so pole.value() is P_h(k); FKP WS/SS windows carry survey shot terms.",
    }
    return Mesh2SpectrumPoles([pole0], ells=[0]), meta


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, default=SUMMARY_DIR.parent / "manifests" / "task43_mmin1p4e13_x25.jsonl")
    parser.add_argument("--fkp-summary", type=Path, default=SUMMARY_DIR / "task43_fkp_zeff_mmin1p4e13_x25.npz")
    parser.add_argument("--xi-summary", type=Path, default=None)
    parser.add_argument("--output-prefix", type=Path, default=SUMMARY_DIR / "jaxpower_covariance_lightcone_mmin1p4e13_ph000")
    parser.add_argument("--phase-index", type=int, default=0)
    parser.add_argument("--p0", type=float, default=10000.0)
    parser.add_argument("--max-random", type=int, default=100000)
    parser.add_argument("--max-data", type=int, default=50000)
    parser.add_argument("--subsample-seed", type=int, default=20260702)
    parser.add_argument("--no-rescale-subsample", action="store_true")
    parser.add_argument("--meshsize", type=int, default=64)
    parser.add_argument("--mesh-pad", type=float, default=400.0)
    parser.add_argument("--window-s-max", type=float, default=350.0)
    parser.add_argument("--window-ds", type=float, default=1.0)
    parser.add_argument("--s-edge-min", type=float, default=float(S_EDGES[0]))
    parser.add_argument("--s-edge-max", type=float, default=float(S_EDGES[-1]))
    parser.add_argument("--s-edge-step", type=float, default=float(np.median(np.diff(S_EDGES))))
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
        Path(row["halo_catalog_path"]),
        fkp_summary=fkp_summary,
        p0=float(args.p0),
        max_rows=args.max_data,
        seed=int(args.subsample_seed) + 1,
        rescale_subsample=rescale,
    )
    random_xyz, random_weight, random_meta = load_positions_weights(
        Path(row["random_catalog_path"]),
        fkp_summary=fkp_summary,
        p0=float(args.p0),
        max_rows=args.max_random,
        seed=int(args.subsample_seed) + 2,
        rescale_subsample=rescale,
    )

    import jax

    jax.config.update("jax_enable_x64", True)
    from jaxpower import BinMesh2CorrelationPoles, FKPField, ParticleField, compute_fkp2_covariance_window, compute_spectrum2_covariance
    from jaxpower.cov2 import matrix_project_to_correlation

    mattrs, mesh_meta = infer_mesh_attrs([data_xyz, random_xyz], meshsize=int(args.meshsize), pad=float(args.mesh_pad))
    data_field = ParticleField(data_xyz, weights=data_weight, attrs=mattrs, exchange=False)
    random_field = ParticleField(random_xyz, weights=random_weight, attrs=mattrs, exchange=False)
    fkp = FKPField(data_field, random_field)
    window_s_edges = make_edges(0.0, float(args.window_s_max), float(args.window_ds))
    window_bin = BinMesh2CorrelationPoles(mattrs, edges=window_s_edges, ells=(0,))
    window2 = compute_fkp2_covariance_window(
        fkp,
        bin=window_bin,
        los=str(args.los),
        resampler=str(args.resampler),
        interlacing=int(args.interlacing),
    )
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
    cov_pk_parts = compute_spectrum2_covariance(window2, poles, flags=("smooth",), return_type="list")
    cov_pk_part_values = {
        name: np.asarray(cov_pk.value(), dtype="f8")
        for name, cov_pk in zip(("WW", "WS", "SS"), cov_pk_parts, strict=True)
    }
    s_edges = make_edges(float(args.s_edge_min), float(args.s_edge_max), float(args.s_edge_step))
    project = np.asarray(matrix_project_to_correlation(s_edges, poles), dtype="f8")
    data_count_scale = float(data_meta["n_total"]) / float(data_meta["n_used"])
    cov_xi_parts_raw = {name: project @ value @ project.T for name, value in cov_pk_part_values.items()}
    cov_xi_parts = {
        "WW": cov_xi_parts_raw["WW"],
        "WS": cov_xi_parts_raw["WS"] / data_count_scale,
        "SS": cov_xi_parts_raw["SS"] / data_count_scale**2,
    }
    raw_cov = sum(cov_xi_parts.values())
    cov_single, spd_meta = nearest_spd(raw_cov, floor_fraction=float(args.floor_fraction))
    nreal = 25
    if args.xi_summary is not None and args.xi_summary.exists():
        xi_data = np.load(args.xi_summary, allow_pickle=False)
        nreal = int(np.asarray(xi_data["nreal"]).item()) if "nreal" in xi_data.files else nreal
    cov_mean = cov_single / float(nreal)
    corr_mean = correlation_from_covariance(cov_mean)
    out_npz = args.output_prefix.with_suffix(".npz")
    out_json = args.output_prefix.with_suffix(".json")
    out_npz.parent.mkdir(parents=True, exist_ok=True)
    meta = {
        "status": "done",
        "covariance_kind": "jaxpower_gaussian_survey_window_xi0",
        "warning": "Gaussian survey-window covariance only; does not include connected trispectrum and is not RascalC full LS covariance.",
        "manifest": str(args.manifest),
        "phase": row["phase"],
        "phase_index": int(args.phase_index),
        "halo_catalog_path": str(row["halo_catalog_path"]),
        "random_catalog_path": str(row["random_catalog_path"]),
        "fkp_summary": str(args.fkp_summary),
        "p0": float(args.p0),
        "zeff": float(zeff),
        "nreal_for_covariance_of_mean": int(nreal),
        "data": data_meta,
        "random": random_meta,
        "mesh": mesh_meta,
        "window": {
            "interface": "jaxpower.compute_fkp2_covariance_window",
            "los": str(args.los),
            "s_edges": [float(v) for v in window_s_edges],
            "ells": [0],
            "resampler": str(args.resampler),
            "interlacing": int(args.interlacing),
        },
        "projection": {
            "interface": "jaxpower.cov2.matrix_project_to_correlation",
            "s_edges": [float(v) for v in s_edges],
            "ells": [0],
            "note": "Direct C_P -> C_xi0 projection. RR-window LS deconvolution is not applied in this script.",
        },
        "subsample_correction": {
            "reason": "jaxpower S shot-window terms use data-weight products; thinning the data catalog preserves W but biases S by N_data_total/N_data_used.",
            "data_count_scale": float(data_count_scale),
            "applied_to": {
                "WW": 1.0,
                "WS": float(1.0 / data_count_scale),
                "SS": float(1.0 / data_count_scale**2),
            },
            "note": "No correction is needed when the full data catalog is used; the factors then equal 1.",
        },
        "k_grid": {
            "k_edges": [float(v) for v in k_edges],
            "kmin": float(args.kmin),
            "kmax": float(args.kmax),
            "dk": float(args.dk),
            "nk": int(k_edges.size - 1),
            "note": "Task43 covariance-fix grid. Earlier kmax<=0.8/dk~0.02 grids produced a low-rank xi covariance and must not be used for fNL constraints.",
        },
        "theory": theory_meta,
        "spd": spd_meta,
        "runtime_sec": float(time.perf_counter() - t0),
    }
    np.savez_compressed(
        out_npz,
        s=0.5 * (s_edges[:-1] + s_edges[1:]),
        s_edges=s_edges,
        covariance=cov_mean,
        covariance_of_mean=cov_mean,
        covariance_single_realization=cov_single,
        correlation=corr_mean,
        covariance_pk=cov_pk_part_values["WW"] + cov_pk_part_values["WS"] / data_count_scale + cov_pk_part_values["SS"] / data_count_scale**2,
        covariance_pk_WW=cov_pk_part_values["WW"],
        covariance_pk_WS=cov_pk_part_values["WS"] / data_count_scale,
        covariance_pk_SS=cov_pk_part_values["SS"] / data_count_scale**2,
        covariance_WW=cov_xi_parts["WW"],
        covariance_WS=cov_xi_parts["WS"],
        covariance_SS=cov_xi_parts["SS"],
        covariance_WW_raw=cov_xi_parts_raw["WW"],
        covariance_WS_raw=cov_xi_parts_raw["WS"],
        covariance_SS_raw=cov_xi_parts_raw["SS"],
        projection_matrix=project,
        zeff=np.array(float(zeff), dtype="f8"),
        nreal=np.array(int(nreal), dtype="i8"),
        p0=np.array(float(args.p0), dtype="f8"),
        covariance_kind=np.asarray("jaxpower_gaussian_survey_window_xi0"),
        meta_json=np.asarray(json.dumps(meta, sort_keys=True)),
    )
    payload = dict(meta)
    payload["output_npz"] = str(out_npz)
    payload["diagnostics_single"] = covariance_diagnostics(cov_single)
    payload["diagnostics_mean"] = covariance_diagnostics(cov_mean)
    out_json.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"[done] wrote {out_npz}")
    print(f"[diag] single sigma={payload['diagnostics_single']['sigma_min']:.3e}..{payload['diagnostics_single']['sigma_max']:.3e} eigmin={payload['diagnostics_single']['min_eigenvalue']:.3e}")
    print(f"[warn] {payload['warning']}")


if __name__ == "__main__":
    main()
