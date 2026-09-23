#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Build Task43 lightcone P(k) Gaussian survey-window covariance with jaxpower."""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Any

for _name in (
    "OMP_NUM_THREADS",
    "MKL_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
):
    os.environ.setdefault(_name, "1")

import numpy as np

PROJECT_ROOT = Path("/pscratch/sd/l/lzy/PNG-2PCF-model-to-observe")
CODE_DIR = PROJECT_ROOT / "codes" / "task43"
DESI_CLUSTERING_ROOT = Path("/pscratch/sd/l/lzy/desi-clustering")
for _path in (CODE_DIR, DESI_CLUSTERING_ROOT):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

from task43_config import read_jsonl  # noqa: E402
from task43_make_jaxpower_lightcone_covariance import build_theory_poles  # noqa: E402
from task43_pk_common import (  # noqa: E402
    DEFAULT_FKP_SUMMARY,
    DEFAULT_MANIFEST_MMIN1P4,
    P_FIXED,
    PK_COV_DIR,
    atomic_savez,
    covariance_diagnostics,
    ensure_pk_dirs,
    infer_mesh_attrs_from_catalogs,
    load_fkp_summary,
    load_phase_catalog,
    make_k_edges,
    mesh_attrs_for_jaxpower,
    p0_output_tag,
    phase_index,
    select_rows,
    to_jsonable,
    write_json,
)
from task43_theory_template import COSMOLOGY_CHOICES, DEFAULT_COSMOLOGY  # noqa: E402


def output_prefix(row: dict[str, Any], *, tag: str, meshsize: int, kmax: float, dk: float) -> Path:
    kt = f"kmax{float(kmax):.3f}_dk{float(dk):.3f}".replace(".", "p")
    return PK_COV_DIR / tag / f"task43_pk_cov_{row['phase']}_mesh{int(meshsize)}_{kt}"


def p0_index(fkp_summary: dict[str, np.ndarray], p0: float) -> int:
    p0_values = np.asarray(fkp_summary["p0_values"], dtype="f8")
    match = np.flatnonzero(np.isclose(p0_values, float(p0), rtol=0.0, atol=1.0e-10))
    if match.size != 1:
        raise ValueError(f"P0={p0} not found in fkp_summary p0_values={p0_values}")
    return int(match[0])


def compute_covariance_mesh2_spectrum_compat(get_data_randoms, *, theory: Any, mattrs: dict[str, Any]) -> dict[str, Any]:
    """Compute P(k) covariance with the desi-clustering/jaxpower recipe.

    The local desi-clustering helper computes the correct jaxpower covariance,
    then tries to relabel ``covariance.observable.fields``.  With the lsstypes
    version in the login-node environment, a plain Mesh2SpectrumPoles observable
    has no ``fields`` property, so the helper raises after the expensive
    covariance has already been built.  This compatibility wrapper mirrors the
    helper up to the jaxpower covariance call and leaves the observable labels
    untouched.
    """
    from clustering_statistics.spectrum2_tools import prepare_jaxpower_particles
    from jaxpower import (
        FKPField,
        compute_fkp2_covariance_window,
        compute_spectrum2_covariance,
        create_sharding_mesh,
        interpolate_window_function,
    )

    fields = [1]
    with create_sharding_mesh(meshsize=mattrs.get("meshsize", None)):
        all_particles = prepare_jaxpower_particles(get_data_randoms, mattrs=mattrs, add_randoms=["IDS"])
        all_fkp = [FKPField(particles["data"], particles["randoms"]) for particles in all_particles]
        mattrs_obj = all_fkp[0].attrs
        kw = dict(edges={"step": mattrs_obj.cellsize.min()}, basis="bessel")
        kw.update(los="local", fields=fields, split=[(42, fkp.randoms.extra["IDS"]) for fkp in all_fkp])
        kw_paint = dict(resampler="tsc", interlacing=3, compensate=True)
        windows = compute_fkp2_covariance_window(all_fkp, **kw, **kw_paint)
        coords = np.logspace(-2, 8, 8 * 1024)
        windows = windows.map(lambda window: interpolate_window_function(window, coords=coords), level=1)
    covariance_parts = compute_spectrum2_covariance(windows, theory, flags=["smooth", "fftlog"], return_type="list")
    return {"raw_parts": covariance_parts, "window_covariance_mesh2_correlation": windows}


def build_covariance(row: dict[str, Any], args: argparse.Namespace) -> dict[str, Any]:
    import jax

    jax.config.update("jax_enable_x64", True)

    t0 = time.perf_counter()
    fkp_summary = load_fkp_summary(args.fkp_summary)
    ip0 = p0_index(fkp_summary, float(args.p0))
    zeff = float(np.asarray(fkp_summary["zeff_random_auto"], dtype="f8")[ip0])
    nbar_shot = float(np.mean(np.asarray(fkp_summary["nbar"], dtype="f8")))
    tag = p0_output_tag(float(args.p0), args.tag)
    prefix = output_prefix(row, tag=tag, meshsize=int(args.meshsize), kmax=float(args.kmax), dk=float(args.dk))
    out_npz = prefix.with_suffix(".npz")
    out_json = prefix.with_suffix(".json")
    if out_npz.exists() and out_json.exists() and not args.overwrite:
        print(f"[skip] {out_npz}", flush=True)
        return {"status": "skip_existing", "path": str(out_npz)}

    data_cat, data_meta = load_phase_catalog(
        Path(row["halo_catalog_path"]),
        fkp_summary=fkp_summary,
        p0=float(args.p0),
        max_rows=args.max_data,
        seed=int(args.seed) + 1000 * phase_index(row["phase"]) + 1,
        rescale_subsample=False,
        add_targetid=False,
    )
    random_cat, random_meta = load_phase_catalog(
        Path(row["random_catalog_path"]),
        fkp_summary=fkp_summary,
        p0=float(args.p0),
        max_rows=args.max_random,
        seed=int(args.seed) + 1000 * phase_index(row["phase"]) + 2,
        rescale_subsample=False,
        add_targetid=True,
    )
    mesh_meta = infer_mesh_attrs_from_catalogs([data_cat, random_cat], meshsize=int(args.meshsize), pad=float(args.mesh_pad))
    mattrs = mesh_attrs_for_jaxpower(mesh_meta)
    k_edges = make_k_edges(float(args.kmin), float(args.kmax), float(args.dk))
    theory, theory_meta = build_theory_poles(
        k_edges=k_edges,
        zeff=zeff,
        b1_cov=float(args.b1_cov),
        fnl_cov=float(args.fnl_cov),
        p_fixed=float(args.p_fixed),
        sn0_fixed=float(args.sn0_fixed),
        nbar_shot=nbar_shot,
        cosmology=str(args.cosmology),
    )

    def get_data_randoms() -> dict[str, dict[str, np.ndarray]]:
        return {"data": data_cat, "randoms": random_cat}

    result = compute_covariance_mesh2_spectrum_compat(get_data_randoms, theory=theory, mattrs=mattrs)
    cov_parts = result["raw_parts"] if isinstance(result, dict) else result
    if len(cov_parts) != 3:
        raise ValueError(f"expected WW/WS/SS covariance parts, got {len(cov_parts)}")
    cov_part_values = {
        name: np.asarray(cov_part.value(), dtype="f8")
        for name, cov_part in zip(("WW", "WS", "SS"), cov_parts, strict=True)
    }
    data_count_scale = float(data_meta["n_total"]) / float(data_meta["n_used"])
    cov_part_factors = {"WW": 1.0, "WS": 1.0 / data_count_scale, "SS": 1.0 / data_count_scale**2}
    cov_part_values_corrected = {
        name: cov_part_values[name] * cov_part_factors[name]
        for name in ("WW", "WS", "SS")
    }
    covariance = sum(cov_part_values_corrected.values())
    covariance = 0.5 * (covariance + covariance.T)
    cov_obj = cov_parts[0].clone(value=covariance)
    cov_h5 = prefix.with_suffix(".h5")
    try:
        cov_obj.write(cov_h5)
    except Exception as exc:
        print(f"[warn] could not write lsstypes covariance h5: {exc}", flush=True)
        cov_h5 = None

    k = np.asarray(theory.get(0).coords("k"), dtype="f8")
    theory_edges = np.asarray(theory.get(0).edges("k"), dtype="f8")
    elapsed = time.perf_counter() - t0
    is_subsampled = bool(data_meta["n_used"] != data_meta["n_total"] or random_meta["n_used"] != random_meta["n_total"])
    summary = {
        "task": "task43_make_pk_covariance_jaxpower",
        "status": "done",
        "phase": row["phase"],
        "sim_name": row["sim_name"],
        "covariance_kind": "jaxpower_gaussian_survey_window_pk0",
        "warning": (
            "Gaussian survey-window covariance only. If max-data/max-random were used, "
            "this is a smoke/debug covariance and must not be used as the final fit covariance."
        ),
        "is_subsampled": is_subsampled,
        "halo_catalog_path": row["halo_catalog_path"],
        "random_catalog_path": row["random_catalog_path"],
        "fkp_summary": str(args.fkp_summary),
        "p0": float(args.p0),
        "zeff": float(zeff),
        "nbar_shot_mean": float(nbar_shot),
        "mesh": mesh_meta,
        "data": data_meta,
        "random": random_meta,
        "theory": theory_meta,
        "subsample_correction": {
            "data_count_scale": float(data_count_scale),
            "applied_to": {name: float(cov_part_factors[name]) for name in ("WW", "WS", "SS")},
            "note": "Matches the Task43 xi0 / Task44 P0 jaxpower covariance treatment: WW + WS/N_subscale + SS/N_subscale^2.",
        },
        "k_edges": [float(v) for v in k_edges],
        "kmin_measure": float(args.kmin),
        "kmax_measure": float(args.kmax),
        "dk": float(args.dk),
        "los": "local",
        "p_fixed": float(args.p_fixed),
        "sigmas_fixed": 0.0,
        "sn0_fixed": float(args.sn0_fixed),
        "sn0_fixed_note": "covariance fiducial residual stochastic term only; the Task43 P(k) fit default keeps sn0 free.",
        "diagnostics": covariance_diagnostics(covariance),
        "output_npz": str(out_npz),
        "output_json": str(out_json),
        "covariance_h5": None if cov_h5 is None else str(cov_h5),
        "cpu_thread_limits": {name: os.environ.get(name) for name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS")},
        "elapsed_sec": float(elapsed),
    }
    atomic_savez(
        out_npz,
        k_obs=k,
        k_edges=theory_edges,
        covariance_single_realization=covariance,
        covariance=covariance,
        covariance_full_WW=cov_part_values_corrected["WW"],
        covariance_full_WS=cov_part_values_corrected["WS"],
        covariance_full_SS=cov_part_values_corrected["SS"],
        covariance_full_uncorrected_WW=cov_part_values["WW"],
        covariance_full_uncorrected_WS=cov_part_values["WS"],
        covariance_full_uncorrected_SS=cov_part_values["SS"],
        data_count_scale=np.asarray(float(data_count_scale), dtype="f8"),
        p0=np.asarray(float(args.p0), dtype="f8"),
        zeff=np.asarray(float(zeff), dtype="f8"),
        phase=np.asarray(row["phase"]),
        phase_index=np.asarray(phase_index(row["phase"]), dtype="i8"),
        b1_cov=np.asarray(float(args.b1_cov), dtype="f8"),
        fnl_cov=np.asarray(float(args.fnl_cov), dtype="f8"),
        p_fixed=np.asarray(float(args.p_fixed), dtype="f8"),
        nbar_shot_mean=np.asarray(float(nbar_shot), dtype="f8"),
        is_subsampled=np.asarray(is_subsampled),
        summary_json=np.asarray(json.dumps(to_jsonable(summary), sort_keys=True)),
    )
    write_json(out_json, summary)
    print(
        f"[write] {out_npz} phase={row['phase']} shape={covariance.shape} "
        f"subsampled={is_subsampled} elapsed={elapsed:.1f}s",
        flush=True,
    )
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build Task43 lightcone P(k) covariance with jaxpower.")
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST_MMIN1P4)
    parser.add_argument("--fkp-summary", type=Path, default=DEFAULT_FKP_SUMMARY)
    parser.add_argument("--index", type=int, default=0)
    parser.add_argument("--phase", type=str, default=None)
    parser.add_argument("--p0", type=float, default=10000.0)
    parser.add_argument("--tag", type=str, default=None)
    parser.add_argument("--meshsize", type=int, default=128)
    parser.add_argument("--mesh-pad", type=float, default=400.0)
    parser.add_argument("--kmin", type=float, default=0.001)
    parser.add_argument("--kmax", type=float, default=0.3001)
    parser.add_argument("--dk", type=float, default=0.002)
    parser.add_argument("--b1-cov", type=float, default=2.5)
    parser.add_argument("--fnl-cov", type=float, default=0.0)
    parser.add_argument("--p-fixed", type=float, default=P_FIXED)
    parser.add_argument("--sn0-fixed", type=float, default=0.0)
    parser.add_argument("--cosmology", choices=COSMOLOGY_CHOICES, default=DEFAULT_COSMOLOGY)
    parser.add_argument("--max-data", type=int, default=None, help="Smoke/debug only; final covariance should use full data.")
    parser.add_argument("--max-random", type=int, default=None, help="Smoke/debug only; final covariance should use full random.")
    parser.add_argument("--seed", type=int, default=20260706)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    ensure_pk_dirs()
    rows = read_jsonl(args.manifest)
    selected = select_rows(rows, index=args.index, phase=args.phase)
    if len(selected) != 1:
        raise ValueError("covariance builder expects exactly one phase/index")
    print(
        f"[task43-pk-cov] phase={selected[0]['phase']} p0={args.p0} mesh={args.meshsize} "
        f"k={args.kmin:g}..{args.kmax:g} dk={args.dk:g}",
        flush=True,
    )
    build_covariance(selected[0], args)


if __name__ == "__main__":
    main()
