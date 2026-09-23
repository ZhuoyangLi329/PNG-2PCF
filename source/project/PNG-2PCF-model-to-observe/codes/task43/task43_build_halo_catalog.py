#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Build Task43 halo data catalogs from AbacusSummit halo lightcones."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

import numpy as np

from task43_config import DEFAULT_MANIFEST, OBSERVER_ORIGIN_POLICY, read_jsonl


def select_rows(rows: list[dict[str, Any]], *, index: int | None, phase: str | None) -> list[dict[str, Any]]:
    """Select manifest rows by index or phase."""
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


def _empty_candidates() -> dict[str, np.ndarray]:
    return {
        "ninterp": np.empty(0, dtype="u4"),
        "redshift": np.empty(0, dtype="f4"),
        "pos": np.empty((0, 3), dtype="f4"),
        "origin_code": np.empty(0, dtype="i1"),
        "shell_index": np.empty(0, dtype="i2"),
    }


def _append_top(
    current: dict[str, np.ndarray],
    new: dict[str, np.ndarray],
    *,
    target_count: int,
) -> dict[str, np.ndarray]:
    """Append candidates and keep only top target_count by N_interp."""
    if len(new["ninterp"]) == 0:
        return current
    if len(new["ninterp"]) > target_count:
        idx = np.argpartition(new["ninterp"], -target_count)[-target_count:]
        new = {key: value[idx] for key, value in new.items()}
    if len(current["ninterp"]) == 0:
        combined = new
    else:
        combined = {
            "ninterp": np.concatenate([current["ninterp"], new["ninterp"]]),
            "redshift": np.concatenate([current["redshift"], new["redshift"]]),
            "pos": np.concatenate([current["pos"], new["pos"]]),
            "origin_code": np.concatenate([current["origin_code"], new["origin_code"]]),
            "shell_index": np.concatenate([current["shell_index"], new["shell_index"]]),
        }
    if len(combined["ninterp"]) > target_count:
        idx = np.argpartition(combined["ninterp"], -target_count)[-target_count:]
        combined = {key: value[idx] for key, value in combined.items()}
    return combined


def _append_all(current: dict[str, np.ndarray], new: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    """Append all candidates passing an explicit threshold."""
    if len(new["ninterp"]) == 0:
        return current
    if len(current["ninterp"]) == 0:
        return new
    return {
        "ninterp": np.concatenate([current["ninterp"], new["ninterp"]]),
        "redshift": np.concatenate([current["redshift"], new["redshift"]]),
        "pos": np.concatenate([current["pos"], new["pos"]]),
        "origin_code": np.concatenate([current["origin_code"], new["origin_code"]]),
        "shell_index": np.concatenate([current["shell_index"], new["shell_index"]]),
    }


def _target_counts_by_zshell(z_edges: np.ndarray, *, target_count: int) -> np.ndarray:
    """Allocate target counts proportional to comoving shell volume."""
    chi_edges = _cosmo_distance(np.asarray(z_edges, dtype="f8"))
    weights = np.diff(chi_edges**3)
    raw = float(target_count) * weights / np.sum(weights)
    counts = np.floor(raw).astype(int)
    remainder = int(target_count) - int(np.sum(counts))
    if remainder > 0:
        order = np.argsort(raw - counts)[::-1]
        counts[order[:remainder]] += 1
    if np.any(counts <= 0):
        raise ValueError(f"non-positive z-shell target count: {counts}")
    return counts


def _selection_z_edges(row: dict[str, Any], *, zmin: float, zmax: float) -> np.ndarray:
    if "selection_z_edges" in row:
        edges = np.asarray(row["selection_z_edges"], dtype="f8")
    else:
        n_zshells = int(row.get("selection_n_zshells", 1))
        edges = np.linspace(zmin, zmax, n_zshells + 1, dtype="f8")
    if edges.ndim != 1 or edges.size < 2:
        raise ValueError(f"bad selection_z_edges: {edges}")
    if not np.all(np.diff(edges) > 0):
        raise ValueError(f"selection_z_edges must be strictly increasing: {edges}")
    if edges[0] < zmin - 1.0e-8 or edges[-1] > zmax + 1.0e-8:
        raise ValueError(f"selection_z_edges outside z range: {edges} vs {zmin}..{zmax}")
    return edges


def _ra_dec_from_observer_xyz(xyz: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Convert observer-centric Cartesian coordinates to RA/DEC and radius."""
    radius = np.linalg.norm(xyz, axis=1)
    ra = np.degrees(np.arctan2(xyz[:, 1], xyz[:, 0])) % 360.0
    dec = np.degrees(np.arcsin(np.clip(xyz[:, 2] / radius, -1.0, 1.0)))
    return ra.astype("f8"), dec.astype("f8"), radius.astype("f8")


def _cosmo_distance(z: np.ndarray) -> np.ndarray:
    """Return AbacusSummit fiducial comoving radial distance in Mpc/h."""
    from cosmoprimo.fiducial import AbacusSummit

    cosmo = AbacusSummit(0)
    return np.asarray(cosmo.comoving_radial_distance(z), dtype="f8")


def build_one(row: dict[str, Any], *, chunk_size: int, force: bool) -> dict[str, Any]:
    """Build one phase halo catalog."""
    import asdf

    out = Path(row["halo_catalog_path"])
    meta_out = Path(row["halo_metadata_path"])
    if out.exists() and meta_out.exists() and not force:
        print(f"[skip] existing {out}")
        return {"status": "skip_existing", "path": str(out)}

    t0 = time.perf_counter()
    target_count_raw = row.get("target_count")
    target_count = None if target_count_raw is None else int(target_count_raw)
    zmin = float(row["zmin"])
    zmax = float(row["zmax"])
    selection_mode = str(row.get("selection_mode", "fixed_count_top_N_interp"))
    use_mass_threshold = selection_mode == "fixed_mass_threshold_ninterp"
    use_zshell_selection = selection_mode == "fixed_count_top_N_interp_zshell_volume"
    if use_mass_threshold:
        mass_threshold_hmsun = float(row["mass_threshold_hmsun"])
        z_edges = np.array([zmin, zmax], dtype="f8")
        target_counts = np.empty(0, dtype=int)
        candidates = _empty_candidates()
        ninterp_threshold_requested: int | None = None
        particle_mass_hmsun: float | None = None
    elif use_zshell_selection:
        if target_count is None:
            raise ValueError("target_count is required for fixed_count_top_N_interp_zshell_volume")
        z_edges = _selection_z_edges(row, zmin=zmin, zmax=zmax)
        target_counts = _target_counts_by_zshell(z_edges, target_count=target_count)
        candidate_bins = [_empty_candidates() for _ in range(z_edges.size - 1)]
    else:
        if target_count is None:
            raise ValueError("target_count is required for fixed_count_top_N_interp")
        z_edges = np.array([zmin, zmax], dtype="f8")
        target_counts = np.array([target_count], dtype=int)
        candidates = _empty_candidates()
    shell_counts: list[dict[str, Any]] = []
    observer_origin: np.ndarray | None = None
    header_redshifts: list[float] = []

    for ishell, path_str in enumerate(row["shell_paths"]):
        path = Path(path_str)
        if not path.exists():
            raise FileNotFoundError(path)
        shell_t0 = time.perf_counter()
        with asdf.open(path, lazy_load=True) as af:
            header = af.tree.get("header", {})
            if use_mass_threshold:
                this_particle_mass = float(header["ParticleMassHMsun"])
                if particle_mass_hmsun is None:
                    particle_mass_hmsun = this_particle_mass
                    ninterp_threshold_requested = int(np.ceil(mass_threshold_hmsun / particle_mass_hmsun))
                elif not np.isclose(particle_mass_hmsun, this_particle_mass, rtol=0.0, atol=1.0e-6):
                    raise RuntimeError(f"ParticleMassHMsun changed: {particle_mass_hmsun} vs {this_particle_mass} in {path}")
            origins = np.asarray(header["LightConeOrigins"], dtype="f8").reshape(-1, 3)
            this_origin = origins[0]
            if observer_origin is None:
                observer_origin = this_origin
            elif not np.allclose(observer_origin, this_origin, rtol=0.0, atol=1.0e-6):
                raise RuntimeError(f"observer origin changed: {observer_origin} vs {this_origin} in {path}")
            header_redshifts.append(float(header.get("Redshift", np.nan)))
            data = af["data"]
            zarr = data["redshift_interp"]
            narr = data["N_interp"]
            posarr = data["pos_interp"]
            originarr = data["origin"]
            nobj = len(zarr)
            n_z_selected = 0
            n_chunks = 0
            for start in range(0, nobj, chunk_size):
                stop = min(start + chunk_size, nobj)
                z = np.asarray(zarr[start:stop], dtype="f4")
                mask = (z > zmin) & (z < zmax)
                if not np.any(mask):
                    n_chunks += 1
                    continue
                n_z_selected += int(np.count_nonzero(mask))
                ninterp = np.asarray(narr[start:stop], dtype="u4")[mask]
                pos = np.asarray(posarr[start:stop], dtype="f4")[mask]
                origin_code = np.asarray(originarr[start:stop], dtype="i1")[mask]
                if use_mass_threshold:
                    if ninterp_threshold_requested is None:
                        raise RuntimeError("internal error: missing mass threshold")
                    keep = ninterp >= int(ninterp_threshold_requested)
                    if not np.any(keep):
                        n_chunks += 1
                        continue
                    ninterp = ninterp[keep]
                    pos = pos[keep]
                    origin_code = origin_code[keep]
                    redshift = z[mask][keep]
                else:
                    redshift = z[mask]
                new = {
                    "ninterp": ninterp,
                    "redshift": redshift,
                    "pos": pos,
                    "origin_code": origin_code,
                    "shell_index": np.full(ninterp.size, ishell, dtype="i2"),
                }
                if use_mass_threshold:
                    candidates = _append_all(candidates, new)
                elif use_zshell_selection:
                    z_selected = new["redshift"]
                    bin_index = np.searchsorted(z_edges, z_selected, side="right") - 1
                    bin_index = np.clip(bin_index, 0, z_edges.size - 2)
                    for ibin in np.unique(bin_index):
                        submask = bin_index == ibin
                        sub = {key: value[submask] for key, value in new.items()}
                        candidate_bins[int(ibin)] = _append_top(
                            candidate_bins[int(ibin)],
                            sub,
                            target_count=int(target_counts[int(ibin)]),
                        )
                else:
                    candidates = _append_top(candidates, new, target_count=target_count)
                n_chunks += 1
            shell_counts.append(
                {
                    "path": str(path),
                    "n_total": int(nobj),
                    "n_z_selected": int(n_z_selected),
                    "n_chunks": int(n_chunks),
                    "header_redshift": float(header.get("Redshift", np.nan)),
                    "elapsed_sec": float(time.perf_counter() - shell_t0),
                }
            )

    if observer_origin is None:
        raise RuntimeError("no observer origin found")
    z_shell_summary: list[dict[str, Any]] = []
    if use_mass_threshold:
        if particle_mass_hmsun is None or ninterp_threshold_requested is None:
            raise RuntimeError("missing particle mass for mass-threshold selection")
    elif use_zshell_selection:
        parts = []
        for ibin, cand in enumerate(candidate_bins):
            nsel = int(len(cand["ninterp"]))
            ntarget = int(target_counts[ibin])
            if nsel < ntarget:
                raise RuntimeError(f"z-shell {ibin} selected only {nsel} halos, target={ntarget}")
            threshold_i = int(np.min(cand["ninterp"]))
            z_shell_summary.append(
                {
                    "index": int(ibin),
                    "zmin": float(z_edges[ibin]),
                    "zmax": float(z_edges[ibin + 1]),
                    "target_count": ntarget,
                    "selected_count": nsel,
                    "ninterp_threshold": threshold_i,
                }
            )
            parts.append(cand)
        candidates = {
            key: np.concatenate([part[key] for part in parts])
            for key in ("ninterp", "redshift", "pos", "origin_code", "shell_index")
        }
    if len(candidates["ninterp"]) == 0:
        raise RuntimeError(f"no halos selected for {row['phase']} in {zmin}<z<{zmax}")

    order = np.argsort(candidates["ninterp"])[::-1]
    candidates = {key: value[order] for key, value in candidates.items()}
    selected_count = int(len(candidates["ninterp"]))
    threshold = int(candidates["ninterp"][-1])

    xyz = np.asarray(candidates["pos"], dtype="f8") - observer_origin[None, :]
    ra, dec, radius = _ra_dec_from_observer_xyz(xyz)
    chi = _cosmo_distance(np.asarray(candidates["redshift"], dtype="f8"))
    frac_err = radius / chi - 1.0
    direction = xyz / radius[:, None]
    direction_min = np.nanmin(direction, axis=0)
    direction_max = np.nanmax(direction, axis=0)

    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        out,
        RA=ra,
        DEC=dec,
        Z=np.asarray(candidates["redshift"], dtype="f4"),
        X=np.asarray(xyz[:, 0], dtype="f4"),
        Y=np.asarray(xyz[:, 1], dtype="f4"),
        Zcart=np.asarray(xyz[:, 2], dtype="f4"),
        WEIGHT=np.ones(selected_count, dtype="f4"),
        halo_mass_proxy=np.asarray(candidates["ninterp"], dtype="u4"),
        N_interp=np.asarray(candidates["ninterp"], dtype="u4"),
        realization=np.full(selected_count, int(row["phase_index"]), dtype="i2"),
        phase=np.asarray(row["phase"]),
        origin_code=np.asarray(candidates["origin_code"], dtype="i1"),
        shell_index=np.asarray(candidates["shell_index"], dtype="i2"),
        observer_origin=np.asarray(observer_origin, dtype="f8"),
    )
    meta = {
        "status": "done",
        "task": "task43",
        "phase": row["phase"],
        "sim_name": row["sim_name"],
        "zmin": zmin,
        "zmax": zmax,
        "target_count": target_count,
        "selected_count": selected_count,
        "selection_mode": selection_mode,
        "selection_tag": row.get("selection_tag", ""),
        "selection_z_edges": z_edges.tolist(),
        "selection_target_counts": target_counts.astype(int).tolist(),
        "z_shell_summary": z_shell_summary,
        "ninterp_threshold": threshold,
        "observer_origin": observer_origin.tolist(),
        "observer_origin_policy": OBSERVER_ORIGIN_POLICY,
        "header_redshifts": header_redshifts,
        "shell_counts": shell_counts,
        "coord_check_radius_over_chi_minus_one": {
            "mean": float(np.mean(frac_err)),
            "median": float(np.median(frac_err)),
            "p99_abs": float(np.quantile(np.abs(frac_err), 0.99)),
            "max_abs": float(np.max(np.abs(frac_err))),
        },
        "direction_min": direction_min.tolist(),
        "direction_max": direction_max.tolist(),
        "positive_octant_gate": bool(np.all(direction_min >= -1.0e-5)),
        "ra_range": [float(np.min(ra)), float(np.max(ra))],
        "dec_range": [float(np.min(dec)), float(np.max(dec))],
        "radius_range": [float(np.min(radius)), float(np.max(radius))],
        "output_path": str(out),
        "elapsed_sec": float(time.perf_counter() - t0),
    }
    if use_mass_threshold:
        meta["mass_threshold_hmsun"] = mass_threshold_hmsun
        meta["particle_mass_hmsun"] = float(particle_mass_hmsun)
        meta["ninterp_threshold_requested"] = int(ninterp_threshold_requested)
        meta["mass_threshold_actual_hmsun"] = float(int(ninterp_threshold_requested) * float(particle_mass_hmsun))
    meta_out.parent.mkdir(parents=True, exist_ok=True)
    meta_out.write_text(json.dumps(meta, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"[done] {row['phase']} selected={selected_count} N_interp>={threshold} path={out}")
    print(f"[check] p99_abs(radius/chi-1)={meta['coord_check_radius_over_chi_minus_one']['p99_abs']:.3e}")
    return {"status": "done", "path": str(out), "metadata_path": str(meta_out)}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--index", type=int, default=None)
    parser.add_argument("--phase", type=str, default=None)
    parser.add_argument("--chunk-size", type=int, default=1_000_000)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    rows = read_jsonl(args.manifest)
    selected = select_rows(rows, index=args.index, phase=args.phase)
    print(f"[task43] halo catalog rows={len(selected)} manifest={args.manifest}")
    for row in selected:
        build_one(row, chunk_size=int(args.chunk_size), force=bool(args.force))


if __name__ == "__main__":
    main()
