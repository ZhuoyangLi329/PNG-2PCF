#!/usr/bin/env python3
"""Build paired real/RSD octant snapshot-shell pseudo-lightcones."""

from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path
from typing import Any

import numpy as np

from task43_pngbase_pseudolc_common import (
    BOX_SIZE_MPC_H,
    COSMOLOGIES,
    FNL_BY_COSMOLOGY,
    MASS_THRESHOLD_HMSUN,
    OBSERVER_MPC_H,
    PHASES,
    SEED_BY_PHASE,
    SNAPSHOT_REDSHIFT,
    SPACE_WINDOWS,
    catalog_path,
    halo_dir,
    sim_name,
)
from task43_rsd_common import (
    apply_radial_rsd,
    atomic_savez,
    atomic_write_json,
    finite_summary,
    sha256_file,
    velocity_kms_per_mpc_h,
)


class DistanceToRedshift:
    def __init__(self, zmax: float = 3.0, ngrid: int = 600_001) -> None:
        from cosmoprimo.fiducial import AbacusSummit

        self.z = np.linspace(0.0, float(zmax), int(ngrid), dtype="f8")
        self.chi = np.asarray(AbacusSummit(0).comoving_radial_distance(self.z), dtype="f8")
        if np.any(~np.isfinite(self.chi)) or not np.all(np.diff(self.chi) > 0.0):
            raise RuntimeError("distance grid is not finite and strictly monotonic")

    def redshift(self, radius: np.ndarray) -> np.ndarray:
        values = np.asarray(radius, dtype="f8")
        if values.size and (float(np.min(values)) < self.chi[0] or float(np.max(values)) > self.chi[-1]):
            raise ValueError("radius lies outside the inverse-distance grid")
        return np.interp(values, self.chi, self.z)

    def distance(self, redshift: float) -> float:
        return float(np.interp(float(redshift), self.z, self.chi))


def radec(relative: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    xyz = np.asarray(relative, dtype="f8")
    radius = np.linalg.norm(xyz, axis=1)
    direction = xyz / radius[:, None]
    return (
        np.degrees(np.arctan2(direction[:, 1], direction[:, 0])) % 360.0,
        np.degrees(np.arcsin(np.clip(direction[:, 2], -1.0, 1.0))),
    )


def validated(path: Path) -> bool:
    metadata_path = path.with_suffix(".json")
    if not (path.is_file() and metadata_path.is_file()):
        return False
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    return metadata.get("status") == "pass" and metadata.get("output_sha256") == sha256_file(path)


def append_selected(parts: dict[str, list[np.ndarray]], values: dict[str, np.ndarray], mask: np.ndarray) -> None:
    if np.any(mask):
        for key in parts:
            parts[key].append(np.asarray(values[key])[mask])


def concatenate(parts: dict[str, list[np.ndarray]]) -> dict[str, np.ndarray]:
    if not parts["halo_id"]:
        raise RuntimeError("pseudo-lightcone selection is empty")
    values = {key: np.concatenate(chunks) for key, chunks in parts.items()}
    order = np.argsort(values["halo_id"], kind="stable")
    return {key: value[order] for key, value in values.items()}


def build_one(cosmology: str, phase: str) -> dict[str, Any]:
    from abacusnbody.data.compaso_halo_catalog import CompaSOHaloCatalog
    import abacusnbody

    outputs = {space: catalog_path(cosmology, phase, space) for space in SPACE_WINDOWS}
    if all(validated(path) for path in outputs.values()):
        print(f"[skip] validated {cosmology} {phase} real/RSD catalogs", flush=True)
        return {"status": "skip_validated", "cosmology": cosmology, "phase": phase}
    for path in outputs.values():
        if (path.exists() or path.with_suffix(".json").exists()) and not validated(path):
            raise FileExistsError(f"partial or unvalidated output exists: {path}")

    source_paths = sorted(halo_dir(cosmology, phase).glob("halo_info_*.asdf"))
    if len(source_paths) != 17:
        raise RuntimeError(f"expected 17 halo shards, found {len(source_paths)} in {halo_dir(cosmology, phase)}")
    started = time.perf_counter()
    inverse = DistanceToRedshift()
    observer = np.asarray(OBSERVER_MPC_H, dtype="f8").reshape(1, 3)
    chi = {space: tuple(inverse.distance(z) for z in window) for space, window in SPACE_WINDOWS.items()}
    keys = (
        "halo_id", "n_cleaned", "position_real", "position_rsd", "z_real", "z_observed",
        "velocity_los", "displacement",
    )
    parts = {space: {key: [] for key in keys} for space in SPACE_WINDOWS}
    shard_audit: list[dict[str, Any]] = []
    particle_mass: float | None = None
    threshold_n: int | None = None
    velocity_conversion: float | None = None
    n_cleaned_total = n_mass_total = n_positive_total = 0
    displacement_extrema: list[float] = []

    for index, path in enumerate(source_paths):
        shard_started = time.perf_counter()
        catalog = CompaSOHaloCatalog(
            str(path), fields=["N", "x_L2com", "v_L2com", "id"], cleaned=True
        )
        header = catalog.header
        this_mass = float(header["ParticleMassHMsun"])
        this_box = float(header.get("BoxSizeHMpc", header.get("BoxSize")))
        this_conversion = velocity_kms_per_mpc_h(header)
        if particle_mass is None:
            particle_mass = this_mass
            threshold_n = int(math.ceil(MASS_THRESHOLD_HMSUN / particle_mass))
            velocity_conversion = this_conversion
        if not np.isclose(this_mass, particle_mass, rtol=0.0, atol=1.0e-6):
            raise RuntimeError(f"particle mass changed in {path}")
        if not np.isclose(this_box, BOX_SIZE_MPC_H, rtol=0.0, atol=1.0e-12):
            raise RuntimeError(f"box size changed in {path}: {this_box}")
        if not np.isclose(this_conversion, velocity_conversion, rtol=0.0, atol=1.0e-12):
            raise RuntimeError(f"velocity conversion changed in {path}")

        halos = catalog.halos
        n_all = np.asarray(halos["N"], dtype="u4")
        mass_mask = n_all >= int(threshold_n)
        halo_id = np.asarray(halos["id"], dtype="u8")[mass_mask]
        position = np.asarray(halos["x_L2com"], dtype="f8")[mass_mask]
        velocity = np.asarray(halos["v_L2com"], dtype="f8")[mass_mask]
        n_cleaned = n_all[mass_mask]
        codes = np.zeros(halo_id.size, dtype="i1")
        mapped = apply_radial_rsd(
            position,
            velocity,
            observer,
            codes,
            np.full(halo_id.size, float(velocity_conversion), dtype="f8"),
        )
        positive = np.all(mapped["position_real_relative"] >= 0.0, axis=1)
        z_real = inverse.redshift(mapped["radius_real"])
        z_observed = inverse.redshift(mapped["radius_rsd"])
        values = {
            "halo_id": halo_id,
            "n_cleaned": n_cleaned,
            "position_real": mapped["position_real_relative"],
            "position_rsd": mapped["position_rsd_relative"],
            "z_real": z_real,
            "z_observed": z_observed,
            "velocity_los": mapped["velocity_los"],
            "displacement": mapped["displacement"],
        }
        real_mask = positive & (mapped["radius_real"] > chi["real"][0]) & (mapped["radius_real"] < chi["real"][1])
        rsd_mask = positive & (mapped["radius_rsd"] > chi["rsd"][0]) & (mapped["radius_rsd"] < chi["rsd"][1])
        append_selected(parts["real"], values, real_mask)
        append_selected(parts["rsd"], values, rsd_mask)
        n_cleaned_total += int(n_all.size)
        n_mass_total += int(np.count_nonzero(mass_mask))
        n_positive_total += int(np.count_nonzero(positive))
        if mapped["displacement"].size:
            displacement_extrema.append(float(np.max(np.abs(mapped["displacement"]))))
        shard_audit.append(
            {
                "index": index,
                "path": str(path),
                "size_bytes": int(path.stat().st_size),
                "n_cleaned_table": int(n_all.size),
                "n_mass_selected": int(halo_id.size),
                "n_positive_octant": int(np.count_nonzero(positive)),
                "n_real_selected": int(np.count_nonzero(real_mask)),
                "n_rsd_selected": int(np.count_nonzero(rsd_mask)),
                "elapsed_sec": time.perf_counter() - shard_started,
            }
        )
        print(
            f"[shard] {cosmology} {phase} {index:02d} mass={halo_id.size} "
            f"real={np.count_nonzero(real_mask)} rsd={np.count_nonzero(rsd_mask)}",
            flush=True,
        )
        del catalog, halos, n_all, halo_id, position, velocity, n_cleaned, mapped

    if particle_mass is None or threshold_n is None or velocity_conversion is None:
        raise RuntimeError("failed to read pngbase halo headers")
    result: dict[str, Any] = {"status": "pass", "cosmology": cosmology, "phase": phase, "outputs": {}}
    for space, path in outputs.items():
        if validated(path):
            metadata = json.loads(path.with_suffix(".json").read_text(encoding="utf-8"))
            result["outputs"][space] = {
                "path": str(path),
                "n_selected": int(metadata["counts"]["n_selected"]),
                "resumed": True,
            }
            continue
        selected = concatenate(parts[space])
        chosen_position = selected["position_real"] if space == "real" else selected["position_rsd"]
        chosen_z = selected["z_real"] if space == "real" else selected["z_observed"]
        ra, dec = radec(chosen_position)
        zmin, zmax = SPACE_WINDOWS[space]
        radius = np.linalg.norm(chosen_position, axis=1)
        ndata = int(chosen_z.size)
        angular_gate = bool(np.all(chosen_position >= -1.0e-10))
        redshift_gate = bool(np.all((chosen_z > zmin) & (chosen_z < zmax)))
        id_gate = int(np.unique(selected["halo_id"]).size) == ndata
        finite_gate = bool(np.all(np.isfinite(chosen_position)) and np.all(np.isfinite(chosen_z)))
        angle_delta = np.max(
            np.abs(
                chosen_position / radius[:, None]
                - selected["position_real"] / np.linalg.norm(selected["position_real"], axis=1)[:, None]
            )
        )
        angle_gate = bool(float(angle_delta) < 1.0e-10)
        gates = {
            "seventeen_raw_shards": len(source_paths) == 17,
            "official_compaso_cleaning_enabled": True,
            "mass_threshold_nonempty": ndata > 0,
            "positive_octant_gate": angular_gate,
            "strict_redshift_gate": redshift_gate,
            "selected_halo_ids_unique": id_gate,
            "finite_coordinates_and_redshifts": finite_gate,
            "radial_rsd_preserves_angles": angle_gate,
        }
        status = "pass" if all(gates.values()) else "fail"
        atomic_savez(
            path,
            RA=ra.astype("f8"),
            DEC=dec.astype("f8"),
            Z=chosen_z.astype("f4"),
            X=chosen_position[:, 0].astype("f4"),
            Y=chosen_position[:, 1].astype("f4"),
            Zcart=chosen_position[:, 2].astype("f4"),
            X_REAL=selected["position_real"][:, 0].astype("f4"),
            Y_REAL=selected["position_real"][:, 1].astype("f4"),
            ZCART_REAL=selected["position_real"][:, 2].astype("f4"),
            Z_GEOM_REAL=selected["z_real"].astype("f4"),
            Z_OBSERVED=selected["z_observed"].astype("f4"),
            VLOS_KMS=selected["velocity_los"].astype("f4"),
            RSD_DISPLACEMENT_MPC_H=selected["displacement"].astype("f4"),
            N_cleaned=selected["n_cleaned"].astype("u4"),
            halo_id=selected["halo_id"].astype("u8"),
            WEIGHT=np.ones(ndata, dtype="f4"),
            realization=np.full(ndata, PHASES.index(phase), dtype="i2"),
            phase=np.asarray(phase),
            cosmology=np.asarray(cosmology),
            injected_fnl=np.asarray(FNL_BY_COSMOLOGY[cosmology], dtype="f8"),
            observer_mpc_h=np.asarray(OBSERVER_MPC_H, dtype="f8"),
        )
        metadata = {
            "task": "task43_build_pngbase_pseudolc_catalog",
            "status": status,
            "cosmology": cosmology,
            "phase": phase,
            "sim_name": sim_name(cosmology, phase),
            "initial_condition_seed": SEED_BY_PHASE[phase],
            "injected_fnl": FNL_BY_COSMOLOGY[cosmology],
            "catalog_role": f"snapshot-shell pseudo-lightcone in {space} space",
            "snapshot_redshift": SNAPSHOT_REDSHIFT,
            "selection": f"positive octant and {zmin} < {'z_geom' if space == 'real' else 'z_observed'} < {zmax}",
            "geometry": {
                "observer_mpc_h": list(OBSERVER_MPC_H),
                "mother_box_size_mpc_h": BOX_SIZE_MPC_H,
                "replication": False,
                "angular_footprint": "observer-relative X,Y,Z >= 0 (one octant)",
                "radial_distance_mpc_h": list(chi[space]),
            },
            "mapping": (
                "identity real-space observer-relative coordinates"
                if space == "real"
                else "local radial s=x+rhat*(v.rhat)/(VelZSpace_to_kms/BoxSize)"
            ),
            "mass_selection": {
                "definition": "N_cleaned * ParticleMassHMsun >= 1.4e13 Msun/h",
                "particle_mass_hmsun": particle_mass,
                "threshold_n": threshold_n,
                "effective_minimum_mass_hmsun": threshold_n * particle_mass,
            },
            "reader": {
                "class": "abacusnbody.data.compaso_halo_catalog.CompaSOHaloCatalog",
                "abacusnbody_version": str(getattr(abacusnbody, "__version__", "unknown")),
                "cleaned": True,
                "fields": ["N", "x_L2com", "v_L2com", "id"],
                "position_units": "Mpc/h centered on the mother box",
                "velocity_units": "km/s",
            },
            "counts": {
                "n_cleaned_table_all_shards": n_cleaned_total,
                "n_mass_selected_all_shards": n_mass_total,
                "n_mass_and_positive_octant": n_positive_total,
                "n_selected": ndata,
            },
            "z_observed": finite_summary(chosen_z),
            "z_geom_real": finite_summary(selected["z_real"]),
            "velocity_los_kms": finite_summary(selected["velocity_los"]),
            "displacement_mpc_h": finite_summary(selected["displacement"]),
            "maximum_abs_displacement_all_mass_selected_mpc_h": max(displacement_extrema),
            "velocity_kms_per_mpc_h": velocity_conversion,
            "angle_direction_max_abs_difference": float(angle_delta),
            "positive_octant_gate": angular_gate,
            "gates": gates,
            "source_shards": shard_audit,
            "output_path": str(path),
            "output_sha256": sha256_file(path),
            "elapsed_sec": time.perf_counter() - started,
        }
        atomic_write_json(path.with_suffix(".json"), metadata)
        if status != "pass":
            raise RuntimeError(f"{cosmology} {phase} {space} catalog gates failed: {gates}")
        result["outputs"][space] = {"path": str(path), "n_selected": ndata}
    print(json.dumps(result, sort_keys=True), flush=True)
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cosmology", choices=COSMOLOGIES, required=True)
    parser.add_argument("--phase", choices=PHASES, required=True)
    args = parser.parse_args()
    build_one(args.cosmology, args.phase)


if __name__ == "__main__":
    main()
