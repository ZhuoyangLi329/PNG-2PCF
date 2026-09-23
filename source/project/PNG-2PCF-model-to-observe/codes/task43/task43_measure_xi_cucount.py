#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Measure Task43 halo-lightcone xi0 with cucount and Landy-Szalay."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

import numpy as np

from task43_config import DEFAULT_MANIFEST, S_EDGES, read_jsonl
from task43_fkp_zeff import (
    compute_zeff_diagnostics,
    get_fiducial_cosmology,
    load_fkp_summary,
    path_with_weight_tag,
    total_weight_from_summary,
)


def select_rows(rows: list[dict[str, Any]], *, index: int | None, phase: str | None) -> list[dict[str, Any]]:
    if index is not None and phase is not None:
        raise ValueError("provide only one of --index or --phase")
    if index is not None:
        return [rows[int(index)]]
    if phase is not None:
        selected = [row for row in rows if row["phase"] == phase]
        if not selected:
            raise ValueError(f"phase not found in manifest: {phase}")
        return selected
    return rows


def positions_from_npz(
    path: Path,
    *,
    fkp_summary: dict[str, np.ndarray] | None,
    p0: float | None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    data = np.load(path)
    xyz = np.column_stack([data["X"], data["Y"], data["Zcart"]]).astype("f4")
    z = np.asarray(data["Z"], dtype="f8")
    base_weight = np.asarray(data["WEIGHT"], dtype="f8") if "WEIGHT" in data.files else np.ones(z.size, dtype="f8")
    weight = total_weight_from_summary(z, base_weight, fkp_summary, p0=p0)
    return xyz, z, weight


def build_s_edges(edge_min: float, edge_max: float, edge_step: float) -> np.ndarray:
    if edge_max <= edge_min:
        raise ValueError("--s-edge-max must exceed --s-edge-min")
    if edge_step <= 0:
        raise ValueError("--s-edge-step must be positive")
    return np.arange(edge_min, edge_max + 0.5 * edge_step, edge_step, dtype="f8")


def measure_one(
    row: dict[str, Any],
    *,
    s_edges: np.ndarray,
    force: bool,
    fkp_summary: dict[str, np.ndarray] | None,
    fkp_summary_path: Path | None,
    p0: float | None,
    output_tag: str | None,
    cosmo: Any,
    backend: str,
    nthreads: int,
) -> dict[str, Any]:
    out = path_with_weight_tag(Path(row["xi_path"]), p0=p0, output_tag=output_tag)
    if out.exists() and not force:
        print(f"[skip] existing {out}")
        return {"status": "skip_existing", "path": str(out)}

    halo_path = Path(row["halo_catalog_path"])
    random_path = Path(row["random_catalog_path"])
    if not halo_path.exists():
        raise FileNotFoundError(f"missing halo catalog: {halo_path}")
    if not random_path.exists():
        raise FileNotFoundError(f"missing random catalog: {random_path}")

    if backend == "jax":
        import jax

        jax.config.update("jax_enable_x64", True)
        from cucount.jax import BinAttrs, MeshAttrs, Particles, WeightAttrs
    elif backend == "numpy":
        from cucount.numpy import BinAttrs, MeshAttrs, Particles, WeightAttrs
    else:
        raise ValueError(f"unknown cucount backend: {backend}")
    from cucount.types import count2
    from lsstypes import Count2Correlation

    t0 = time.perf_counter()
    data_xyz, data_z, data_w = positions_from_npz(halo_path, fkp_summary=fkp_summary, p0=p0)
    random_xyz, random_z, random_w = positions_from_npz(random_path, fkp_summary=fkp_summary, p0=p0)
    ndata = int(data_xyz.shape[0])
    nrandom = int(random_xyz.shape[0])
    if ndata < 2 or nrandom < 2:
        raise RuntimeError(f"insufficient points: ndata={ndata} nrandom={nrandom}")

    battrs = BinAttrs(s=np.asarray(s_edges, dtype="f8"))
    if backend == "jax":
        d_particles = Particles(data_xyz, weights=data_w.astype("f4"), exchange=True)
        r_particles = Particles(random_xyz, weights=random_w.astype("f4"), exchange=True)
        count_kwargs: dict[str, Any] = {}
    else:
        d_particles = Particles(data_xyz, weights=data_w.astype("f8"))
        r_particles = Particles(random_xyz, weights=random_w.astype("f8"))
        count_kwargs = {"nthreads": int(nthreads)}
    mattrs = MeshAttrs(d_particles, r_particles, battrs=battrs, periodic=False)
    wattrs = WeightAttrs()

    count_t0 = time.perf_counter()
    dd = count2(d_particles, d_particles, battrs=battrs, mattrs=mattrs, wattrs=wattrs, **count_kwargs)["weight"]
    dr = count2(d_particles, r_particles, battrs=battrs, mattrs=mattrs, wattrs=wattrs, **count_kwargs)["weight"]
    rr = count2(r_particles, r_particles, battrs=battrs, mattrs=mattrs, wattrs=wattrs, **count_kwargs)["weight"]
    corr = Count2Correlation(estimator="landyszalay", DD=dd, DS=dr, SD=dr, SS=rr, RR=rr)
    xi0 = np.asarray(corr.value(), dtype="f8")
    s = np.asarray(corr.coords("s"), dtype="f8")
    dd_value = np.asarray(dd.value(), dtype="f8")
    dr_value = np.asarray(dr.value(), dtype="f8")
    rr_value = np.asarray(rr.value(), dtype="f8")
    if backend == "jax":
        jax.block_until_ready(xi0)
    count_elapsed = time.perf_counter() - count_t0
    zeff = compute_zeff_diagnostics(
        data_z,
        data_w,
        random_z,
        random_w,
        cosmo=cosmo,
        zrange=(float(row["zmin"]), float(row["zmax"])),
    )

    out.parent.mkdir(parents=True, exist_ok=True)
    weighting_meta = {
        "scheme": "WEIGHT_TOTAL=WEIGHT*WEIGHT_FKP" if fkp_summary is not None else "catalog_WEIGHT",
        "p0": None if p0 is None else float(p0),
        "fkp_summary_path": None if fkp_summary_path is None else str(fkp_summary_path),
        "output_tag": output_tag,
        "data_weight_min": float(np.min(data_w)),
        "data_weight_max": float(np.max(data_w)),
        "random_weight_min": float(np.min(random_w)),
        "random_weight_max": float(np.max(random_w)),
    }
    np.savez_compressed(
        out,
        s=s,
        s_edges=np.asarray(s_edges, dtype="f8"),
        xi0=xi0,
        DD=dd_value,
        DR=dr_value,
        RR=rr_value,
        ndata=np.array(ndata, dtype="i8"),
        nrandom=np.array(nrandom, dtype="i8"),
        phase=np.asarray(row["phase"]),
        sim_name=np.asarray(row["sim_name"]),
        zmin=np.array(float(row["zmin"]), dtype="f8"),
        zmax=np.array(float(row["zmax"]), dtype="f8"),
        zeff=np.array(float(zeff["zeff_random_auto"]), dtype="f8"),
        zeff_random_auto=np.array(float(zeff["zeff_random_auto"]), dtype="f8"),
        zeff_data_auto=np.array(float(zeff["zeff_data_auto"]), dtype="f8"),
        zeff_data_random_cross=np.array(float(zeff["zeff_data_random_cross"]), dtype="f8"),
        zeff_data_mean=np.array(float(zeff["data_mean_z"]), dtype="f8"),
        zeff_random_mean=np.array(float(zeff["random_mean_z"]), dtype="f8"),
        zeff_data_weighted_mean=np.array(float(zeff["data_weighted_mean_z"]), dtype="f8"),
        zeff_random_weighted_mean=np.array(float(zeff["random_weighted_mean_z"]), dtype="f8"),
        data_weight_min=np.array(float(np.min(data_w)), dtype="f8"),
        data_weight_max=np.array(float(np.max(data_w)), dtype="f8"),
        random_weight_min=np.array(float(np.min(random_w)), dtype="f8"),
        random_weight_max=np.array(float(np.max(random_w)), dtype="f8"),
        p0=np.array(np.nan if p0 is None else float(p0), dtype="f8"),
        weighting_meta_json=np.asarray(json.dumps(weighting_meta, sort_keys=True)),
        halo_catalog_path=np.asarray(str(halo_path)),
        random_catalog_path=np.asarray(str(random_path)),
        engine=np.asarray(f"cucount_{backend}"),
        estimator=np.asarray("landy_szalay"),
        backend=np.asarray(backend),
        nthreads=np.array(int(nthreads), dtype="i8"),
        status=np.asarray("done"),
        count_elapsed_sec=np.array(count_elapsed, dtype="f8"),
        elapsed_sec=np.array(time.perf_counter() - t0, dtype="f8"),
    )
    print(f"[done] {row['phase']} xi bins={len(xi0)} ndata={ndata} nrandom={nrandom} path={out}")
    print(f"[check] xi finite={bool(np.all(np.isfinite(xi0)))} RR>0={bool(np.all(rr_value > 0))}")
    return {"status": "done", "path": str(out)}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--index", type=int, default=None)
    parser.add_argument("--phase", type=str, default=None)
    parser.add_argument("--s-edge-min", type=float, default=float(S_EDGES[0]))
    parser.add_argument("--s-edge-max", type=float, default=float(S_EDGES[-1]))
    parser.add_argument("--s-edge-step", type=float, default=float(np.median(np.diff(S_EDGES))))
    parser.add_argument("--fkp-summary", type=Path, default=None)
    parser.add_argument("--p0", type=float, default=None)
    parser.add_argument("--output-tag", type=str, default=None)
    parser.add_argument("--cosmology", choices=["DESI", "AbacusSummit"], default="DESI")
    parser.add_argument("--backend", choices=["jax", "numpy"], default="jax")
    parser.add_argument("--nthreads", type=int, default=1)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    if args.fkp_summary is not None and args.p0 is None:
        raise ValueError("--p0 is required when --fkp-summary is provided")
    if args.fkp_summary is None and args.p0 is not None and float(args.p0) != 0.0:
        raise ValueError("nonzero --p0 requires --fkp-summary")
    fkp_summary = load_fkp_summary(args.fkp_summary) if args.fkp_summary is not None else None
    cosmo = get_fiducial_cosmology(args.cosmology)
    rows = read_jsonl(args.manifest)
    selected = select_rows(rows, index=args.index, phase=args.phase)
    s_edges = build_s_edges(args.s_edge_min, args.s_edge_max, args.s_edge_step)
    print(f"[task43] xi rows={len(selected)} bins={len(s_edges)-1} manifest={args.manifest}")
    for row in selected:
        measure_one(
            row,
            s_edges=s_edges,
            force=bool(args.force),
            fkp_summary=fkp_summary,
            fkp_summary_path=args.fkp_summary,
            p0=args.p0,
            output_tag=args.output_tag,
            cosmo=cosmo,
            backend=args.backend,
            nthreads=int(args.nthreads),
        )


if __name__ == "__main__":
    main()
