#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Measure Task43 halo-lightcone P0(k) with desi-clustering/jaxpower."""

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
from task43_pk_common import (  # noqa: E402
    DEFAULT_FKP_SUMMARY,
    DEFAULT_MANIFEST_MMIN1P4,
    PK_MEASURE_DIR,
    atomic_savez,
    column_edges,
    ensure_pk_dirs,
    extract_spectrum_arrays,
    extract_window_arrays,
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


def output_prefix(row: dict[str, Any], *, tag: str, meshsize: int, kmax: float, dk: float) -> Path:
    kt = f"kmax{float(kmax):.3f}_dk{float(dk):.3f}".replace(".", "p")
    return PK_MEASURE_DIR / tag / f"task43_pk_{row['phase']}_mesh{int(meshsize)}_{kt}"


def measure_one(row: dict[str, Any], args: argparse.Namespace) -> dict[str, Any]:
    import jax

    jax.config.update("jax_enable_x64", True)
    from clustering_statistics import spectrum2_tools

    t0 = time.perf_counter()
    fkp_summary = load_fkp_summary(args.fkp_summary)
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
        rescale_subsample=bool(args.rescale_subsample),
        add_targetid=False,
    )
    random_cat, random_meta = load_phase_catalog(
        Path(row["random_catalog_path"]),
        fkp_summary=fkp_summary,
        p0=float(args.p0),
        max_rows=args.max_random,
        seed=int(args.seed) + 1000 * phase_index(row["phase"]) + 2,
        rescale_subsample=bool(args.rescale_subsample),
        add_targetid=True,
    )
    mesh_meta = infer_mesh_attrs_from_catalogs([data_cat, random_cat], meshsize=int(args.meshsize), pad=float(args.mesh_pad))
    mattrs = mesh_attrs_for_jaxpower(mesh_meta)
    k_edges = make_k_edges(float(args.kmin), float(args.kmax), float(args.dk))

    def get_data_randoms() -> dict[str, dict[str, np.ndarray]]:
        return {"data": data_cat, "randoms": random_cat}

    spectrum = spectrum2_tools.compute_mesh2_spectrum(
        get_data_randoms,
        mattrs=mattrs,
        edges=column_edges(k_edges),
        ells=(0,),
        los=str(args.los),
        optimal_weights=None,
        norm={"cellsize": float(args.norm_cellsize)},
    )
    arrays = extract_spectrum_arrays(spectrum, ell=0)
    window_arrays: dict[str, np.ndarray] = {}
    window_path = None
    if args.window_method is not None:
        window = spectrum2_tools.compute_window_mesh2_spectrum(
            get_data_randoms,
            spectrum=spectrum,
            optimal_weights=None,
            method=str(args.window_method),
        )
        window_arrays = extract_window_arrays(window)
        window_path = prefix.with_name(prefix.name + f"_window_{args.window_method}.h5")
        try:
            window["raw"].write(window_path) if isinstance(window, dict) else window.write(window_path)
        except Exception as exc:
            print(f"[warn] could not write lsstypes window h5: {exc}", flush=True)
            window_path = None

    elapsed = time.perf_counter() - t0
    summary = {
        "task": "task43_measure_pk_jaxpower",
        "status": "done",
        "phase": row["phase"],
        "sim_name": row["sim_name"],
        "halo_catalog_path": row["halo_catalog_path"],
        "random_catalog_path": row["random_catalog_path"],
        "fkp_summary": str(args.fkp_summary),
        "p0": float(args.p0),
        "weight_policy": "INDWEIGHT = WEIGHT * WEIGHT_FKP(P0=10000 by default)",
        "estimator": "desi-clustering spectrum2_tools.compute_mesh2_spectrum",
        "window_policy": "geometry-only compute_window_mesh2_spectrum; no RIC, no AMR, no GIC",
        "has_window": bool(args.window_method is not None),
        "window_method": None if args.window_method is None else str(args.window_method),
        "window_h5": None if window_path is None else str(window_path),
        "los": str(args.los),
        "ells": [0],
        "mesh": mesh_meta,
        "k_edges": [float(v) for v in k_edges],
        "kmin_measure": float(args.kmin),
        "kmax_measure": float(args.kmax),
        "dk": float(args.dk),
        "norm_cellsize": float(args.norm_cellsize),
        "data": data_meta,
        "random": random_meta,
        "rescale_subsample": bool(args.rescale_subsample),
        "cpu_thread_limits": {name: os.environ.get(name) for name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS")},
        "elapsed_sec": float(elapsed),
        "output_npz": str(out_npz),
        "output_json": str(out_json),
        "shotnoise_mean": float(np.nanmean(arrays["shotnoise"])),
        "shotnoise_min": float(np.nanmin(arrays["shotnoise"])),
        "shotnoise_max": float(np.nanmax(arrays["shotnoise"])),
    }
    atomic_savez(
        out_npz,
        **arrays,
        **window_arrays,
        phase=np.asarray(row["phase"]),
        phase_index=np.asarray(phase_index(row["phase"]), dtype="i8"),
        p0=np.asarray(float(args.p0), dtype="f8"),
        meshsize=np.asarray(int(args.meshsize), dtype="i8"),
        k_edges_requested=np.asarray(k_edges, dtype="f8"),
        data_n_used=np.asarray(int(data_meta["n_used"]), dtype="i8"),
        random_n_used=np.asarray(int(random_meta["n_used"]), dtype="i8"),
        data_n_total=np.asarray(int(data_meta["n_total"]), dtype="i8"),
        random_n_total=np.asarray(int(random_meta["n_total"]), dtype="i8"),
        has_window=np.asarray(bool(args.window_method is not None)),
        summary_json=np.asarray(json.dumps(to_jsonable(summary), sort_keys=True)),
    )
    write_json(out_json, summary)
    print(
        f"[write] {out_npz} phase={row['phase']} ndata={data_meta['n_used']} "
        f"nran={random_meta['n_used']} elapsed={elapsed:.1f}s",
        flush=True,
    )
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Measure Task43 lightcone P0(k) with jaxpower on CPU.")
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST_MMIN1P4)
    parser.add_argument("--fkp-summary", type=Path, default=DEFAULT_FKP_SUMMARY)
    parser.add_argument("--index", type=int, default=None)
    parser.add_argument("--phase", type=str, default=None)
    parser.add_argument("--p0", type=float, default=10000.0)
    parser.add_argument("--tag", type=str, default=None)
    parser.add_argument("--meshsize", type=int, default=256)
    parser.add_argument("--mesh-pad", type=float, default=400.0)
    parser.add_argument("--kmin", type=float, default=0.001)
    parser.add_argument("--kmax", type=float, default=0.3001)
    parser.add_argument("--dk", type=float, default=0.002)
    parser.add_argument("--los", type=str, default="local")
    parser.add_argument("--norm-cellsize", type=float, default=10.0)
    parser.add_argument("--window-method", choices=("smooth", "exact"), default=None)
    parser.add_argument("--max-data", type=int, default=None, help="Smoke/debug only; production should use full data.")
    parser.add_argument("--max-random", type=int, default=None, help="Smoke/debug only; production should use full random.")
    parser.add_argument("--rescale-subsample", action="store_true", help="Smoke/debug only; preserves total weight when subsampling.")
    parser.add_argument("--seed", type=int, default=20260706)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    ensure_pk_dirs()
    rows = read_jsonl(args.manifest)
    selected = select_rows(rows, index=args.index, phase=args.phase)
    print(
        f"[task43-pk] rows={len(selected)} p0={args.p0} mesh={args.meshsize} "
        f"k={args.kmin:g}..{args.kmax:g} dk={args.dk:g} window={args.window_method}",
        flush=True,
    )
    for row in selected:
        measure_one(row, args)


if __name__ == "__main__":
    main()

