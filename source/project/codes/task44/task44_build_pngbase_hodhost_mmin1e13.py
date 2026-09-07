#!/usr/bin/env python3
"""Build audited M>=1e13 Msun/h HOD-host halo catalogs for c300/c302.

Run the ``build`` mode with the kirisame environment because Abacus ASDF
files require the Abacus ``blsc`` codec and the official cleaned CompaSO
loader.  The resulting NPZ and ASCII files contain one unit-weight row per
unique HOD host that passes the cleaned halo-mass cut.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import time
from pathlib import Path
from typing import Any

import h5py
import numpy as np

from task44_pngbase_hodhost_mmin1e13_common import (
    BOX_SIZE,
    BOX_VOLUME,
    CATALOGS,
    HALO_MASS_MIN_HMSUN,
    REDSHIFT,
    SUMMARY_DIR,
    atomic_savez,
    atomic_write_json,
    ensure_output_dirs,
    get_spec,
    host_ascii_metadata_path,
    host_ascii_path,
    host_catalog_metadata_path,
    host_catalog_path,
    load_host_catalog,
    raw_halo_files,
    sha256_file,
)


def _finite_summary(values: np.ndarray) -> dict[str, float]:
    array = np.asarray(values, dtype="f8")
    if array.size == 0 or np.any(~np.isfinite(array)):
        raise ValueError("summary values must be non-empty and finite")
    return {
        "min": float(np.min(array)),
        "q01": float(np.quantile(array, 0.01)),
        "median": float(np.median(array)),
        "mean": float(np.mean(array)),
        "q99": float(np.quantile(array, 0.99)),
        "max": float(np.max(array)),
    }


def _validated_existing(path: Path, metadata_path: Path) -> bool:
    if not path.is_file() or not metadata_path.is_file():
        return False
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    return metadata.get("status") == "pass" and metadata.get("output_sha256") == sha256_file(path)


def build_one(tag: str, *, overwrite: bool) -> None:
    from abacusnbody.data.compaso_halo_catalog import CompaSOHaloCatalog
    import abacusnbody

    ensure_output_dirs()
    spec = get_spec(tag)
    output = host_catalog_path(tag)
    metadata_path = host_catalog_metadata_path(tag)
    if not overwrite and _validated_existing(output, metadata_path):
        print(f"[skip] validated host catalog {output}", flush=True)
        return
    if not overwrite and (output.exists() or metadata_path.exists()):
        raise FileExistsError(f"partial/unvalidated host output exists: {output} / {metadata_path}")

    started = time.perf_counter()
    hod_path = Path(spec.path)
    with h5py.File(hod_path, "r") as h5:
        hod_id = np.asarray(h5["HALO_ID"], dtype="i8")
        hod_mass = np.asarray(h5["MASS"], dtype="f8")
        is_central = np.asarray(h5["ISCENTRAL"], dtype="i8") == 1
        central_id = np.asarray(hod_id[is_central], dtype="i8")
        central_position = np.column_stack(
            [np.asarray(h5[name], dtype="f8")[is_central] for name in ("X", "Y", "Z")]
        )

    host_id, first, inverse, occupation = np.unique(
        hod_id,
        return_index=True,
        return_inverse=True,
        return_counts=True,
    )
    host_mass_hod = np.asarray(hod_mass[first], dtype="f8")
    hod_mass_constant_within_host = bool(np.array_equal(hod_mass, host_mass_hod[inverse]))
    if not hod_mass_constant_within_host:
        raise RuntimeError(f"HOD MASS is not constant within HALO_ID groups for {tag}")
    del inverse, hod_mass

    matched_ids: list[np.ndarray] = []
    matched_n: list[np.ndarray] = []
    matched_position_centered: list[np.ndarray] = []
    shard_rows: list[dict[str, Any]] = []
    particle_mass: float | None = None
    source_boxsize: float | None = None
    raw_paths = raw_halo_files(tag)
    for shard_index, path in enumerate(raw_paths):
        shard_started = time.perf_counter()
        catalog = CompaSOHaloCatalog(
            str(path),
            fields=["N", "x_L2com", "id"],
            cleaned=True,
        )
        header = catalog.header
        this_particle_mass = float(header["ParticleMassHMsun"])
        boxsize_key = "BoxSizeHMpc" if "BoxSizeHMpc" in header else "BoxSize"
        this_boxsize = float(header[boxsize_key])
        if particle_mass is None:
            particle_mass = this_particle_mass
            source_boxsize = this_boxsize
        if not np.isclose(this_particle_mass, particle_mass, rtol=0.0, atol=1.0e-6):
            raise RuntimeError(f"particle mass changed in {path}")
        if not np.isclose(this_boxsize, source_boxsize, rtol=0.0, atol=1.0e-12):
            raise RuntimeError(f"box size changed in {path}")

        halos = catalog.halos
        raw_id_u64 = np.asarray(halos["id"], dtype="u8")
        if raw_id_u64.size and int(np.max(raw_id_u64)) > np.iinfo("i8").max:
            raise RuntimeError(f"halo ID exceeds signed int64 range in {path}")
        raw_id = raw_id_u64.astype("i8", copy=False)
        raw_n = np.asarray(halos["N"], dtype="i8")
        raw_position = np.asarray(halos["x_L2com"], dtype="f4")
        location = np.searchsorted(host_id, raw_id)
        in_range = location < host_id.size
        clipped = np.minimum(location, host_id.size - 1)
        is_hod_host = in_range & (host_id[clipped] == raw_id)
        keep = is_hod_host & (raw_n > 0)
        matched_ids.append(np.asarray(raw_id[keep], dtype="i8"))
        matched_n.append(np.asarray(raw_n[keep], dtype="i8"))
        matched_position_centered.append(np.asarray(raw_position[keep], dtype="f4"))
        shard_rows.append(
            {
                "index": shard_index,
                "path": str(path),
                "size_bytes": int(path.stat().st_size),
                "mtime_ns": int(path.stat().st_mtime_ns),
                "n_raw_cleaned_table": int(raw_id.size),
                "n_cleaned_positive": int(np.count_nonzero(raw_n > 0)),
                "n_hod_host_matches_positive": int(np.count_nonzero(keep)),
                "elapsed_sec": time.perf_counter() - shard_started,
            }
        )
        print(
            f"[read] {tag} shard={shard_index:02d} raw={raw_id.size} "
            f"host_matches={np.count_nonzero(keep)}",
            flush=True,
        )
        del catalog, halos, raw_id_u64, raw_id, raw_n, raw_position

    if particle_mass is None or source_boxsize is None:
        raise RuntimeError(f"no raw halo shards loaded for {tag}")
    all_id = np.concatenate(matched_ids)
    all_n = np.concatenate(matched_n)
    all_position_centered = np.concatenate(matched_position_centered)
    order = np.argsort(all_id, kind="stable")
    all_id = all_id[order]
    all_n = all_n[order]
    all_position_centered = all_position_centered[order]
    del matched_ids, matched_n, matched_position_centered, order

    ids_match_exactly = bool(np.array_equal(all_id, host_id))
    if not ids_match_exactly:
        missing = np.setdiff1d(host_id, all_id)
        duplicated = int(all_id.size - np.unique(all_id).size)
        raise RuntimeError(
            f"cleaned raw/HOD host mapping failed for {tag}: "
            f"matched={all_id.size}, expected={host_id.size}, missing={missing.size}, duplicates={duplicated}"
        )
    raw_mass = np.asarray(all_n, dtype="f8") * float(particle_mass)
    # Use an explicit independent output buffer.  The legacy kirisame stack
    # (NumPy 1.26 under its current Python build) can recycle the left operand
    # for this large subtraction expression, which would corrupt raw_mass.
    mass_difference = np.empty_like(raw_mass)
    np.subtract(raw_mass, host_mass_hod, out=mass_difference)
    mass_max_abs_difference = float(np.max(np.abs(mass_difference)))
    mass_mismatch_count = int(np.count_nonzero(mass_difference))
    mass_matches_hod_exactly = bool(
        raw_mass.shape == host_mass_hod.shape
        and mass_mismatch_count == 0
        and np.all(np.isfinite(raw_mass))
        and np.all(np.isfinite(host_mass_hod))
    )
    if not mass_matches_hod_exactly:
        raise RuntimeError(
            f"official cleaned raw mass does not reproduce HOD MASS for {tag}: "
            f"max_abs={mass_max_abs_difference}"
        )

    particle_n_min = int(math.ceil(HALO_MASS_MIN_HMSUN / float(particle_mass)))
    selection_by_n = all_n >= particle_n_min
    selection_by_mass = raw_mass >= HALO_MASS_MIN_HMSUN
    selection_by_hod_mass = host_mass_hod >= HALO_MASS_MIN_HMSUN
    n_vs_rawmass_cut_mismatch_count = int(np.count_nonzero(selection_by_n != selection_by_mass))
    n_vs_hodmass_cut_mismatch_count = int(np.count_nonzero(selection_by_n != selection_by_hod_mass))
    threshold_masks_identical = bool(
        selection_by_n.shape == selection_by_mass.shape == selection_by_hod_mass.shape
        and n_vs_rawmass_cut_mismatch_count == 0
        and n_vs_hodmass_cut_mismatch_count == 0
    )
    if not threshold_masks_identical:
        discrepant_n = all_n[selection_by_n != selection_by_mass]
        raise RuntimeError(
            f"N/raw-mass/HOD-mass threshold masks differ for {tag}: "
            f"Nmin={particle_n_min}, N-vs-raw={n_vs_rawmass_cut_mismatch_count}, "
            f"N-vs-HOD={n_vs_hodmass_cut_mismatch_count}, "
            f"counts(N/raw/HOD)={np.count_nonzero(selection_by_n)}/"
            f"{np.count_nonzero(selection_by_mass)}/{np.count_nonzero(selection_by_hod_mass)}, "
            f"raw_mass_minmax={np.min(raw_mass)}/{np.max(raw_mass)}, "
            f"hod_mass_minmax={np.min(host_mass_hod)}/{np.max(host_mass_hod)}, "
            f"cut={HALO_MASS_MIN_HMSUN}, discrepant_N={np.unique(discrepant_n)[:10].tolist()}"
        )

    central_location = np.searchsorted(host_id, central_id)
    central_id_mapping_exact = bool(
        np.all(central_location < host_id.size)
        and np.array_equal(host_id[central_location], central_id)
    )
    if not central_id_mapping_exact:
        raise RuntimeError(f"central HALO_ID mapping failed for {tag}")
    central_raw_position = np.asarray(all_position_centered[central_location], dtype="f8")
    central_delta = np.abs(central_raw_position - central_position)
    central_delta = np.minimum(central_delta, BOX_SIZE - central_delta)
    central_position_max_abs = float(np.max(central_delta))
    central_positions_exact = bool(central_position_max_abs == 0.0)
    if not central_positions_exact:
        raise RuntimeError(
            f"official cleaned positions do not reproduce HOD centrals for {tag}: "
            f"max periodic difference={central_position_max_abs} Mpc/h"
        )

    selected_id = np.asarray(all_id[selection_by_n], dtype="i8")
    selected_n = np.asarray(all_n[selection_by_n], dtype="i8")
    selected_mass = np.asarray(raw_mass[selection_by_n], dtype="f8")
    selected_occupation = np.asarray(occupation[selection_by_n], dtype="i4")
    selected_position_centered = np.asarray(all_position_centered[selection_by_n], dtype="f4")
    # Keep shifted coordinates in float64.  Casting back to float32 can round a
    # valid value just below L up to exactly L, which violates [0,L) and can
    # change periodic-box pair assignment at the boundary.
    selected_position = np.mod(selected_position_centered.astype("f8") + BOX_SIZE / 2.0, BOX_SIZE)
    ndata = int(selected_id.size)
    if ndata == 0:
        raise RuntimeError(f"M>=1e13 selection is empty for {tag}")

    output.parent.mkdir(parents=True, exist_ok=True)
    atomic_savez(
        output,
        position=selected_position,
        position_centered=selected_position_centered,
        halo_id=selected_id,
        N_cleaned=selected_n,
        mass_hmsun=selected_mass,
        hod_occupation=selected_occupation,
        ndata=np.asarray(ndata, dtype="i8"),
        nbar=np.asarray(ndata / BOX_VOLUME, dtype="f8"),
        boxsize=np.asarray(BOX_SIZE, dtype="f8"),
        volume=np.asarray(BOX_VOLUME, dtype="f8"),
        redshift=np.asarray(REDSHIFT, dtype="f8"),
        particle_mass_hmsun=np.asarray(particle_mass, dtype="f8"),
        particle_n_min=np.asarray(particle_n_min, dtype="i8"),
        halo_mass_min_hmsun=np.asarray(HALO_MASS_MIN_HMSUN, dtype="f8"),
        realization=np.asarray(spec.realization),
        fnl=np.asarray(spec.fnl, dtype="f8"),
    )
    output_sha256 = sha256_file(output)
    gates = {
        "seventeen_raw_shards": len(raw_paths) == 17,
        "official_compaso_cleaning_enabled": True,
        "hod_mass_constant_within_host": hod_mass_constant_within_host,
        "all_unique_hod_hosts_map_once_to_cleaned_raw_halos": ids_match_exactly,
        "cleaned_raw_mass_matches_hod_mass_exactly": mass_matches_hod_exactly,
        "N_rawmass_hodmass_threshold_masks_identical": threshold_masks_identical,
        "particle_n_min_is_987": particle_n_min == 987,
        "N986_below_and_N987_above_mass_cut": bool(
            (particle_n_min - 1) * particle_mass < HALO_MASS_MIN_HMSUN
            and particle_n_min * particle_mass >= HALO_MASS_MIN_HMSUN
        ),
        "central_positions_match_exactly": central_positions_exact,
        "positive_positions_inside_box": bool(np.all(selected_position >= 0.0) and np.all(selected_position < BOX_SIZE)),
        "selected_ids_unique": int(np.unique(selected_id).size) == ndata,
        "selected_mass_cut_exact": bool(np.all(selected_mass >= HALO_MASS_MIN_HMSUN)),
        "output_nonempty": ndata > 0,
    }
    status = "pass" if all(gates.values()) else "review"
    metadata = {
        "task": "task44_build_pngbase_hodhost_mmin1e13",
        "status": status,
        "tag": tag,
        "realization": spec.realization,
        "catalog_baseline_fnl": spec.fnl,
        "source_hod_hdf5": str(hod_path),
        "source_hod_hdf5_sha256": sha256_file(hod_path),
        "raw_halo_sources": shard_rows,
        "reader": {
            "class": "abacusnbody.data.compaso_halo_catalog.CompaSOHaloCatalog",
            "abacusnbody_version": str(getattr(abacusnbody, "__version__", "unknown")),
            "cleaned": True,
            "fields": ["N", "x_L2com", "id"],
        },
        "selection": {
            "population": "unique HALO_ID values present in the exact HOD-MAP LRG catalog",
            "weight": "one per selected host halo",
            "mass_definition": "M = N_cleaned * ParticleMassHMsun from official cleaned CompaSO loader",
            "halo_mass_min_hmsun": HALO_MASS_MIN_HMSUN,
            "comparison": ">=",
            "particle_mass_hmsun": float(particle_mass),
            "particle_n_min": particle_n_min,
            "effective_minimum_mass_hmsun": float(particle_n_min * particle_mass),
            "largest_rejected_particle_mass_hmsun": float((particle_n_min - 1) * particle_mass),
            "not_a_complete_mass_threshold_catalog": True,
            "interpretation": "mass-cut subset of actual occupied HOD hosts, not all CompaSO halos above the threshold",
        },
        "counts": {
            "ngal": int(hod_id.size),
            "n_unique_hod_hosts_before_mass_cut": int(host_id.size),
            "n_selected_hod_hosts": ndata,
            "n_centrals": int(central_id.size),
            "n_satellite_only_hosts_before_mass_cut": int(host_id.size - np.unique(central_id).size),
            "nbar_selected_h3_mpc3": float(ndata / BOX_VOLUME),
        },
        "geometry": {
            "type": "full periodic cube",
            "space": "real",
            "boxsize_mpc_h": BOX_SIZE,
            "volume_mpc_h3": BOX_VOLUME,
            "coordinate_transform": "official centered x_L2com -> mod(x + L/2, L)",
        },
        "mass_selected_hmsun": _finite_summary(selected_mass),
        "N_cleaned_selected": _finite_summary(selected_n),
        "hod_occupation_selected": _finite_summary(selected_occupation),
        "cross_checks": {
            "mass_max_abs_difference_hmsun": mass_max_abs_difference,
            "mass_nonzero_difference_count": mass_mismatch_count,
            "N_vs_rawmass_cut_mismatch_count": n_vs_rawmass_cut_mismatch_count,
            "N_vs_hodmass_cut_mismatch_count": n_vs_hodmass_cut_mismatch_count,
            "central_position_max_abs_periodic_mpc_h": central_position_max_abs,
        },
        "gates": gates,
        "output": str(output),
        "output_sha256": output_sha256,
        "elapsed_sec": time.perf_counter() - started,
    }
    atomic_write_json(metadata_path, metadata)
    if status != "pass":
        raise RuntimeError(f"host catalog gates require review for {tag}: {gates}")
    print(
        f"[done] {tag} selected_hosts={ndata} nbar={ndata / BOX_VOLUME:.8g} "
        f"Nmin={particle_n_min} output={output}",
        flush=True,
    )


def build_ascii(tag: str, *, overwrite: bool) -> None:
    ensure_output_dirs()
    arrays, catalog_metadata = load_host_catalog(tag)
    output = host_ascii_path(tag)
    metadata_path = host_ascii_metadata_path(tag)
    if not overwrite and _validated_existing(output, metadata_path):
        print(f"[skip] validated ASCII host catalog {output}", flush=True)
        return
    if not overwrite and (output.exists() or metadata_path.exists()):
        raise FileExistsError(f"partial/unvalidated ASCII output exists: {output} / {metadata_path}")
    position = np.asarray(arrays["position"], dtype="f8")
    temporary = output.with_name(f".{output.name}.{os.getpid()}.tmp")
    started = time.perf_counter()
    np.savetxt(
        temporary,
        position,
        fmt="%.10f %.10f %.10f",
        header="X Y Z [Mpc/h]; Mcleaned>=1e13 Msun/h unique occupied HOD hosts; periodic [0,2000)",
        comments="# ",
    )
    temporary.replace(output)
    metadata = {
        "task": "task44_build_pngbase_hodhost_mmin1e13_ascii",
        "status": "pass",
        "tag": tag,
        "source_npz": str(host_catalog_path(tag)),
        "source_npz_sha256": catalog_metadata["output_sha256"],
        "ndata": int(position.shape[0]),
        "format": "ASCII X Y Z, unit weight implicit",
        "output": str(output),
        "output_sha256": sha256_file(output),
        "elapsed_sec": time.perf_counter() - started,
    }
    atomic_write_json(metadata_path, metadata)
    print(f"[done] {tag} ASCII rows={position.shape[0]} output={output}", flush=True)


def write_combined_audit(*, overwrite: bool) -> None:
    ensure_output_dirs()
    output = SUMMARY_DIR / "task44_pngbase_hodhost_mmin1e13_catalog_audit.json"
    if output.is_file() and not overwrite:
        print(f"[skip] {output}", flush=True)
        return
    catalogs: dict[str, Any] = {}
    for tag in CATALOGS:
        path = host_catalog_path(tag)
        metadata_path = host_catalog_metadata_path(tag)
        arrays, metadata = load_host_catalog(tag)
        ascii_path = host_ascii_path(tag)
        ascii_metadata_path = host_ascii_metadata_path(tag)
        ascii_metadata = json.loads(ascii_metadata_path.read_text(encoding="utf-8"))
        if ascii_metadata.get("output_sha256") != sha256_file(ascii_path):
            raise RuntimeError(f"ASCII hash validation failed: {ascii_path}")
        catalogs[tag] = {
            "catalog_metadata": metadata,
            "npz": str(path),
            "npz_sha256": sha256_file(path),
            "ascii": str(ascii_path),
            "ascii_sha256": sha256_file(ascii_path),
            "ndata": int(np.asarray(arrays["ndata"]).item()),
        }
    gates = {
        "both_catalogs_pass": all(row["catalog_metadata"]["status"] == "pass" for row in catalogs.values()),
        "same_physical_mass_min": all(
            np.isclose(
                row["catalog_metadata"]["selection"]["halo_mass_min_hmsun"],
                HALO_MASS_MIN_HMSUN,
                rtol=0.0,
                atol=0.0,
            )
            for row in catalogs.values()
        ),
        "same_integer_particle_cut": len(
            {row["catalog_metadata"]["selection"]["particle_n_min"] for row in catalogs.values()}
        )
        == 1,
        "all_product_hashes_valid": True,
    }
    payload = {
        "task": "task44_pngbase_hodhost_mmin1e13_catalog_audit",
        "status": "pass" if all(gates.values()) else "review",
        "selection": "unique occupied HOD hosts with official cleaned CompaSO mass >= 1e13 Msun/h",
        "gates": gates,
        "catalogs": catalogs,
    }
    atomic_write_json(output, payload)
    if payload["status"] != "pass":
        raise RuntimeError(gates)
    print(f"[done] combined host audit {output}", flush=True)


def parse_tags(values: list[str] | None) -> list[str]:
    return list(CATALOGS) if not values else [get_spec(value).tag for value in values]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=("build", "ascii", "audit", "all"))
    parser.add_argument("--tag", action="append", choices=tuple(CATALOGS))
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    tags = parse_tags(args.tag)
    if args.mode in ("build", "all"):
        for tag in tags:
            build_one(tag, overwrite=bool(args.overwrite))
    if args.mode in ("ascii", "all"):
        for tag in tags:
            build_ascii(tag, overwrite=bool(args.overwrite))
    if args.mode in ("audit", "all"):
        if set(tags) != set(CATALOGS):
            raise ValueError("combined audit requires both c300 and c302")
        write_combined_audit(overwrite=bool(args.overwrite))


if __name__ == "__main__":
    main()
