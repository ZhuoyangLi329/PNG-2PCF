#!/usr/bin/env python3
"""Build physical-RSD and zero-velocity bridge lightcone catalogs together."""

from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path
from typing import Any

import numpy as np

from task43_rsd_common import (
    MASS_THRESHOLD_HMSUN,
    PHASES,
    apply_official_lightcone_fallback,
    apply_radial_rsd,
    atomic_savez,
    atomic_write_json,
    finite_summary,
    sha256_file,
    velocity_kms_per_mpc_h,
)


LEGACY_ROOT = Path(
    "/pscratch/sd/l/lzy/PNG-2PCF-model-to-observe/old_doc_codes/"
    "task4_task44_cleanup_20260707T061844Z/moved/outputs/task43_outputs/halo_catalogs"
)


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def select_row(rows: list[dict[str, Any]], phase: str) -> dict[str, Any]:
    matches = [row for row in rows if row["phase"] == phase]
    if len(matches) != 1:
        raise ValueError(f"expected one manifest row for {phase}, found {len(matches)}")
    return matches[0]


def legacy_catalog_path(phase: str) -> Path:
    return LEGACY_ROOT / f"halo_lightcone_AbacusSummit_base_c000_{phase}_z0p6_0p8_mmin1p4e13.npz"


def radec(relative_position: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    xyz = np.asarray(relative_position, dtype="f8")
    radius = np.linalg.norm(xyz, axis=1)
    if xyz.ndim != 2 or xyz.shape[1] != 3 or np.any(~np.isfinite(radius)) or np.any(radius <= 0.0):
        raise ValueError("invalid observer-relative lightcone coordinates")
    direction = xyz / radius[:, None]
    ra = np.degrees(np.arctan2(direction[:, 1], direction[:, 0])) % 360.0
    dec = np.degrees(np.arcsin(np.clip(direction[:, 2], -1.0, 1.0)))
    return ra, dec, radius


class DistanceToRedshift:
    """High-resolution monotonic inverse of the Abacus fiducial distance."""

    def __init__(self, zmax: float = 1.2, ngrid: int = 600_001) -> None:
        from cosmoprimo.fiducial import AbacusSummit

        self.z = np.linspace(0.0, float(zmax), int(ngrid), dtype="f8")
        self.chi = np.asarray(AbacusSummit(0).comoving_radial_distance(self.z), dtype="f8")
        if np.any(~np.isfinite(self.chi)) or not np.all(np.diff(self.chi) > 0.0):
            raise RuntimeError("fiducial distance grid is not finite and monotonic")

    def __call__(self, radius: np.ndarray) -> np.ndarray:
        values = np.asarray(radius, dtype="f8")
        if np.any(values < self.chi[0]) or np.any(values > self.chi[-1]):
            raise ValueError("radius lies outside inverse-distance grid")
        return np.interp(values, self.chi, self.z)


def empty_parts(keys: tuple[str, ...]) -> dict[str, list[np.ndarray]]:
    return {key: [] for key in keys}


def append_selected(target: dict[str, list[np.ndarray]], values: dict[str, np.ndarray], mask: np.ndarray) -> None:
    if np.any(mask):
        for key in target:
            target[key].append(np.asarray(values[key])[mask])


def concatenate_and_sort(parts: dict[str, list[np.ndarray]]) -> dict[str, np.ndarray]:
    first = next(iter(parts))
    if not parts[first]:
        raise RuntimeError("no selected lightcone objects")
    values = {key: np.concatenate(value) for key, value in parts.items()}
    order = np.argsort(values["ninterp"])[::-1]
    return {key: value[order] for key, value in values.items()}


def validated_existing(output: Path, metadata_path: Path) -> bool:
    if not (output.is_file() and metadata_path.is_file()):
        return False
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    return metadata.get("status") == "pass" and metadata.get("output_sha256") == sha256_file(output)


def duplicate_count(shell: np.ndarray, index_halo: np.ndarray, origin: np.ndarray) -> int:
    keys = np.empty(shell.size, dtype=[("shell", "i2"), ("index", "i8"), ("origin", "i1")])
    keys["shell"], keys["index"], keys["origin"] = shell, index_halo, origin
    return int(keys.size - np.unique(keys).size)


def legacy_bridge_metrics(path: Path | None, bridge: dict[str, np.ndarray]) -> dict[str, Any]:
    result: dict[str, Any] = {"path": str(path) if path is not None else None, "available": bool(path and path.is_file())}
    if path is None or not path.is_file():
        result["pass"] = None
        return result
    with np.load(path, allow_pickle=False) as legacy:
        comparisons = {
            "N_interp": np.asarray(bridge["ninterp"], dtype="u4"),
            "Z": np.asarray(bridge["z_source"], dtype="f4"),
            "X": np.asarray(bridge["position_relative"][:, 0], dtype="f4"),
            "Y": np.asarray(bridge["position_relative"][:, 1], dtype="f4"),
            "Zcart": np.asarray(bridge["position_relative"][:, 2], dtype="f4"),
            "origin_code": np.asarray(bridge["origin_code"], dtype="i1"),
            "shell_index": np.asarray(bridge["shell_index"], dtype="i2"),
        }
        result["shape_equal"] = bool(len(legacy["Z"]) == len(bridge["z_source"]))
        equal: dict[str, bool] = {}
        max_abs: dict[str, float | None] = {}
        for key, current in comparisons.items():
            old = np.asarray(legacy[key])
            equal[key] = bool(old.shape == current.shape and np.array_equal(old, current))
            max_abs[key] = (
                float(np.max(np.abs(old.astype("f8") - current.astype("f8"))))
                if old.shape == current.shape
                else None
            )
        result["array_equal"] = equal
        result["max_abs"] = max_abs
        result["pass"] = bool(result["shape_equal"] and all(equal.values()))
    return result


def build_one(row: dict[str, Any], *, chunk_size: int) -> dict[str, Any]:
    import asdf

    primary_output = Path(row["lightcone_catalog_path"])
    primary_metadata_path = Path(row["lightcone_metadata_path"])
    bridge_output = Path(row["lightcone_zero_velocity_catalog_path"])
    bridge_metadata_path = Path(row["lightcone_zero_velocity_metadata_path"])
    primary_valid = validated_existing(primary_output, primary_metadata_path)
    bridge_valid = validated_existing(bridge_output, bridge_metadata_path)
    if primary_valid and bridge_valid:
        print(f"[skip] validated paired lightcone catalogs for {row['phase']}")
        return {"status": "skip_validated", "phase": row["phase"]}
    for output, metadata, valid in (
        (primary_output, primary_metadata_path, primary_valid),
        (bridge_output, bridge_metadata_path, bridge_valid),
    ):
        if (output.exists() or metadata.exists()) and not valid:
            raise FileExistsError(f"partial or unvalidated output exists: {output} / {metadata}")

    started = time.perf_counter()
    zmin_observed = float(row["zmin_observed"])
    zmax_observed = float(row["zmax_observed"])
    if not (0.0 <= zmin_observed < zmax_observed):
        raise ValueError(f"invalid observed-redshift interval {(zmin_observed, zmax_observed)}")
    mass_threshold_hmsun = float(row.get("mass_threshold_hmsun", MASS_THRESHOLD_HMSUN))
    inverse_distance = DistanceToRedshift(zmax=max(1.2, zmax_observed + 0.1))
    primary_parts = empty_parts(
        (
            "ninterp", "z_source", "z_real", "z_observed", "position_real_relative",
            "position_rsd_relative", "velocity_los", "displacement", "origin_code",
            "origin_index", "fallback", "shell_index", "index_halo",
        )
    )
    bridge_parts = empty_parts(
        ("ninterp", "z_source", "position_relative", "origin_code", "shell_index", "index_halo")
    )
    shell_audit: list[dict[str, Any]] = []
    all_source_radius_min, all_source_radius_max = np.inf, -np.inf
    all_displacement_abs_max = 0.0
    all_zsource_min, all_zsource_max = np.inf, -np.inf
    total_rows = total_mass_selected = total_fallback_mass_selected = 0
    total_angular_rejected_after_radial_selection = 0
    particle_mass: float | None = None
    threshold_n: int | None = None
    reference_origins: np.ndarray | None = None

    source_paths = [Path(path) for path in row["lightcone_source_paths"]]
    if not source_paths or any(not path.is_file() for path in source_paths):
        raise RuntimeError(f"{row['phase']} requires one or more existing lightcone shells")

    for shell_index, path in enumerate(source_paths):
        shell_started = time.perf_counter()
        shell_mass = shell_primary = shell_bridge = shell_fallback = shell_angular_rejected = 0
        with asdf.open(path, lazy_load=True, memmap=False) as af:
            header = af["header"]
            header_redshift = float(header.get("Redshift", np.nan))
            this_mass = float(header["ParticleMassHMsun"])
            this_origins = np.asarray(header["LightConeOrigins"], dtype="f8").reshape(-1, 3)
            this_conversion = velocity_kms_per_mpc_h(header)
            if particle_mass is None:
                particle_mass = this_mass
                threshold_n = int(math.ceil(mass_threshold_hmsun / particle_mass))
                reference_origins = this_origins
            if not np.isclose(this_mass, particle_mass, rtol=0.0, atol=1.0e-6):
                raise RuntimeError(f"ParticleMassHMsun changed in {path}")
            if reference_origins is None or this_origins.shape != reference_origins.shape or not np.allclose(
                this_origins, reference_origins, rtol=0.0, atol=1.0e-6
            ):
                raise RuntimeError(f"LightConeOrigins changed in {path}")
            data = af["data"]
            nobj = int(len(data["N_interp"]))
            total_rows += nobj
            for start in range(0, nobj, int(chunk_size)):
                stop = min(start + int(chunk_size), nobj)
                ninterp_all = np.asarray(data["N_interp"][start:stop], dtype="u4")
                mass_mask = ninterp_all >= int(threshold_n)
                if not np.any(mass_mask):
                    continue
                ninterp = ninterp_all[mass_mask]
                z_source = np.asarray(data["redshift_interp"][start:stop], dtype="f4")[mass_mask]
                position_interp = np.asarray(data["pos_interp"][start:stop], dtype="f4")[mass_mask]
                velocity_interp = np.asarray(data["vel_interp"][start:stop], dtype="f4")[mass_mask]
                position_average = np.asarray(data["pos_avg"][start:stop], dtype="f4")[mass_mask]
                velocity_average = np.asarray(data["vel_avg"][start:stop], dtype="f4")[mass_mask]
                origin_code = np.asarray(data["origin"][start:stop], dtype="i1")[mass_mask]
                index_halo = np.asarray(data["index_halo"][start:stop], dtype="i8")[mass_mask]
                position_clean, velocity_clean, fallback = apply_official_lightcone_fallback(
                    position_interp, velocity_interp, position_average, velocity_average, origin_code
                )
                mapped = apply_radial_rsd(
                    position_clean,
                    velocity_clean,
                    this_origins,
                    origin_code,
                    np.full(ninterp.size, this_conversion, dtype="f8"),
                    velocity_scale=float(row["rsd_velocity_scale"]),
                )
                chi_selection_min = float(
                    np.interp(float(row["zmin_observed"]), inverse_distance.z, inverse_distance.chi)
                )
                chi_selection_max = float(
                    np.interp(float(row["zmax_observed"]), inverse_distance.z, inverse_distance.chi)
                )
                radial_selection_mask = (mapped["radius_rsd"] > chi_selection_min) & (
                    mapped["radius_rsd"] < chi_selection_max
                )
                angular_mask = np.all(mapped["position_real_relative"] >= 0.0, axis=1)
                primary_mask = radial_selection_mask & angular_mask
                n_angular_rejected = int(np.count_nonzero(radial_selection_mask & ~angular_mask))
                if np.any(primary_mask):
                    selected_values = {
                        "ninterp": ninterp[primary_mask],
                        "z_source": z_source[primary_mask],
                        "z_real": inverse_distance(mapped["radius_real"][primary_mask]),
                        "z_observed": inverse_distance(mapped["radius_rsd"][primary_mask]),
                        "position_real_relative": mapped["position_real_relative"][primary_mask],
                        "position_rsd_relative": mapped["position_rsd_relative"][primary_mask],
                        "velocity_los": mapped["velocity_los"][primary_mask],
                        "displacement": mapped["displacement"][primary_mask],
                        "origin_code": origin_code[primary_mask],
                        "origin_index": mapped["origin_index"][primary_mask],
                        "fallback": fallback[primary_mask],
                        "shell_index": np.full(np.count_nonzero(primary_mask), shell_index, dtype="i2"),
                        "index_halo": index_halo[primary_mask],
                    }
                    for key in primary_parts:
                        primary_parts[key].append(selected_values[key])

                bridge_mask = (z_source > float(row["zmin_observed"])) & (z_source < float(row["zmax_observed"]))
                append_selected(
                    bridge_parts,
                    {
                        "ninterp": ninterp,
                        "z_source": z_source,
                        "position_relative": position_interp.astype("f8") - this_origins[0][None, :],
                        "origin_code": origin_code,
                        "shell_index": np.full(ninterp.size, shell_index, dtype="i2"),
                        "index_halo": index_halo,
                    },
                    bridge_mask,
                )

                nmass = int(ninterp.size)
                shell_mass += nmass
                shell_fallback += int(np.count_nonzero(fallback))
                shell_primary += int(np.count_nonzero(primary_mask))
                shell_angular_rejected += n_angular_rejected
                shell_bridge += int(np.count_nonzero(bridge_mask))
                total_mass_selected += nmass
                total_fallback_mass_selected += int(np.count_nonzero(fallback))
                total_angular_rejected_after_radial_selection += n_angular_rejected
                all_source_radius_min = min(all_source_radius_min, float(np.min(mapped["radius_real"])))
                all_source_radius_max = max(all_source_radius_max, float(np.max(mapped["radius_real"])))
                all_displacement_abs_max = max(
                    all_displacement_abs_max, float(np.max(np.abs(mapped["displacement"])))
                )
                all_zsource_min = min(all_zsource_min, float(np.min(z_source)))
                all_zsource_max = max(all_zsource_max, float(np.max(z_source)))
        shell_audit.append(
            {
                "path": str(path),
                "header_redshift": header_redshift,
                "velocity_kms_per_mpc_h": float(this_conversion),
                "n_total": nobj,
                "n_mass_selected": shell_mass,
                "n_fallback_mass_selected": shell_fallback,
                "n_primary_observed_selected": shell_primary,
                "n_angular_rejected_after_radial_selection": shell_angular_rejected,
                "n_bridge_source_selected": shell_bridge,
                "elapsed_sec": time.perf_counter() - shell_started,
            }
        )
        print(
            f"[shell] {row['phase']} {path.parent.name} mass={shell_mass} "
            f"primary={shell_primary} bridge={shell_bridge} fallback={shell_fallback}",
            flush=True,
        )

    if particle_mass is None or threshold_n is None or reference_origins is None:
        raise RuntimeError("failed to read lightcone headers")
    primary = concatenate_and_sort(primary_parts)
    bridge = concatenate_and_sort(bridge_parts)
    nprimary, nbridge = int(primary["ninterp"].size), int(bridge["ninterp"].size)
    ra, dec, radius_rsd = radec(primary["position_rsd_relative"])
    ra_real, dec_real, _ = radec(primary["position_real_relative"])
    angle_max_abs_deg = float(max(np.max(np.abs(ra - ra_real)), np.max(np.abs(dec - dec_real))))
    zero_map = apply_radial_rsd(
        primary["position_real_relative"],
        np.zeros_like(primary["position_real_relative"]),
        np.zeros((reference_origins.shape[0], 3), dtype="f8"),
        primary["origin_index"],
        np.ones(nprimary, dtype="f8"),
        velocity_scale=0.0,
    )
    zero_velocity_max_abs = float(
        np.max(np.abs(zero_map["position_rsd_relative"] - primary["position_real_relative"]))
    )
    chi_min = float(np.interp(float(row["zmin_observed"]), inverse_distance.z, inverse_distance.chi))
    chi_max = float(np.interp(float(row["zmax_observed"]), inverse_distance.z, inverse_distance.chi))
    source_chi_min = float(np.interp(all_zsource_min, inverse_distance.z, inverse_distance.chi))
    source_chi_max = float(np.interp(all_zsource_max, inverse_distance.z, inverse_distance.chi))
    lower_margin, upper_margin = chi_min - source_chi_min, source_chi_max - chi_max
    source_coverage_gate = bool(
        lower_margin > all_displacement_abs_max and upper_margin > all_displacement_abs_max
    )
    primary_duplicates = duplicate_count(
        primary["shell_index"], primary["index_halo"], primary["origin_code"]
    )
    positive_octant_gate = bool(
        np.all(primary["position_rsd_relative"] / radius_rsd[:, None] >= -1.0e-12)
    )
    migrated_in = int(
        np.count_nonzero(
            (primary["z_source"] <= zmin_observed) | (primary["z_source"] >= zmax_observed)
        )
    )
    legacy_path_value = row.get("legacy_bridge_catalog_path")
    if legacy_path_value:
        legacy_path: Path | None = Path(str(legacy_path_value))
    elif np.isclose(zmin_observed, 0.6) and np.isclose(zmax_observed, 0.8):
        legacy_path = legacy_catalog_path(str(row["phase"]))
    else:
        legacy_path = None
    legacy_metrics = legacy_bridge_metrics(legacy_path, bridge)
    primary_status = "pass" if (
        zero_velocity_max_abs == 0.0
        and angle_max_abs_deg < 1.0e-10
        and source_coverage_gate
        and primary_duplicates == 0
        and positive_octant_gate
    ) else "fail"
    bridge_status = "pass" if legacy_metrics.get("pass") in (True, None) else "fail"

    if not primary_valid:
        atomic_savez(
            primary_output,
            RA=ra.astype("f8"), DEC=dec.astype("f8"), Z=primary["z_observed"].astype("f4"),
            X=primary["position_rsd_relative"][:, 0].astype("f4"),
            Y=primary["position_rsd_relative"][:, 1].astype("f4"),
            Zcart=primary["position_rsd_relative"][:, 2].astype("f4"),
            X_REAL=primary["position_real_relative"][:, 0].astype("f4"),
            Y_REAL=primary["position_real_relative"][:, 1].astype("f4"),
            ZCART_REAL=primary["position_real_relative"][:, 2].astype("f4"),
            Z_SOURCE=primary["z_source"].astype("f4"),
            Z_GEOM_REAL=primary["z_real"].astype("f4"),
            VLOS_KMS=primary["velocity_los"].astype("f4"),
            RSD_DISPLACEMENT_MPC_H=primary["displacement"].astype("f4"),
            WEIGHT=np.ones(nprimary, dtype="f4"),
            halo_mass_proxy=primary["ninterp"].astype("u4"), N_interp=primary["ninterp"].astype("u4"),
            realization=np.full(nprimary, int(row["phase_index"]), dtype="i2"), phase=np.asarray(row["phase"]),
            origin_code=primary["origin_code"].astype("i1"), origin_index=primary["origin_index"].astype("i1"),
            fallback_used=primary["fallback"].astype(bool), shell_index=primary["shell_index"].astype("i2"),
            index_halo=primary["index_halo"].astype("i8"), observer_origins=reference_origins.astype("f8"),
        )
        primary_metadata = {
            "task": "task43_build_rsd_lightcone_catalog",
            "status": primary_status,
            "phase": row["phase"],
            "sim_name": row["sim_name"],
            "catalog_role": "physical_radial_rsd_observed_z_selection",
            "mapping": "local radial: s=x+rhat*(v.rhat)/(VelZSpace_to_kms/BoxSize)",
            "fallback_policy": "nonzero pos_avg uses pos_avg/vel_avg; observer index=origin modulo n_origins",
            "selection": f"{row['zmin_observed']} < z_observed < {row['zmax_observed']}",
            "n_total_source_rows": total_rows,
            "n_mass_selected_all_shells": total_mass_selected,
            "n_selected": nprimary,
            "n_angular_rejected_after_radial_selection": total_angular_rejected_after_radial_selection,
            "n_migrated_in_from_source_z_outside_window": migrated_in,
            "n_fallback_mass_selected": total_fallback_mass_selected,
            "n_fallback_selected": int(np.count_nonzero(primary["fallback"])),
            "fallback_fraction_selected": float(np.mean(primary["fallback"])),
            "mass_threshold_hmsun": mass_threshold_hmsun,
            "particle_mass_hmsun": particle_mass,
            "threshold_n": threshold_n,
            "observer_origins_mpc_h": reference_origins.tolist(),
            "source_z_mass_selected_range": [all_zsource_min, all_zsource_max],
            "source_radius_mass_selected_range_mpc_h": [all_source_radius_min, all_source_radius_max],
            "source_coverage_margin_mpc_h": {"lower": lower_margin, "upper": upper_margin},
            "maximum_abs_displacement_all_sources_mpc_h": all_displacement_abs_max,
            "source_coverage_gate": source_coverage_gate,
            "zero_velocity_max_abs_mpc_h": zero_velocity_max_abs,
            "angle_invariance_max_abs_deg": angle_max_abs_deg,
            "duplicate_shell_index_origin_count": primary_duplicates,
            "positive_octant_gate": positive_octant_gate,
            "z_source_minus_z_geom_real": finite_summary(primary["z_source"] - primary["z_real"]),
            "z_observed": finite_summary(primary["z_observed"]),
            "velocity_los_kms": finite_summary(primary["velocity_los"]),
            "displacement_mpc_h": finite_summary(primary["displacement"]),
            "shells": shell_audit,
            "output_path": str(primary_output),
            "output_sha256": sha256_file(primary_output),
            "elapsed_sec": time.perf_counter() - started,
        }
        atomic_write_json(primary_metadata_path, primary_metadata)

    bridge_ra, bridge_dec, _ = radec(bridge["position_relative"])
    if not bridge_valid:
        atomic_savez(
            bridge_output,
            RA=bridge_ra.astype("f8"), DEC=bridge_dec.astype("f8"), Z=bridge["z_source"].astype("f4"),
            X=bridge["position_relative"][:, 0].astype("f4"),
            Y=bridge["position_relative"][:, 1].astype("f4"),
            Zcart=bridge["position_relative"][:, 2].astype("f4"), WEIGHT=np.ones(nbridge, dtype="f4"),
            halo_mass_proxy=bridge["ninterp"].astype("u4"), N_interp=bridge["ninterp"].astype("u4"),
            realization=np.full(nbridge, int(row["phase_index"]), dtype="i2"), phase=np.asarray(row["phase"]),
            origin_code=bridge["origin_code"].astype("i1"), shell_index=bridge["shell_index"].astype("i2"),
            index_halo=bridge["index_halo"].astype("i8"), observer_origin=reference_origins[0].astype("f8"),
        )
        bridge_metadata = {
            "task": "task43_build_rsd_lightcone_catalog",
            "status": bridge_status,
            "phase": row["phase"],
            "catalog_role": "zero_velocity_legacy_implementation_bridge",
            "mapping": "velocity scale zero; pos_interp; first observer; source-z selection",
            "n_selected": nbridge,
            "mass_threshold_hmsun": mass_threshold_hmsun,
            "particle_mass_hmsun": particle_mass,
            "threshold_n": threshold_n,
            "legacy_bridge": legacy_metrics,
            "output_path": str(bridge_output),
            "output_sha256": sha256_file(bridge_output),
            "elapsed_sec": time.perf_counter() - started,
        }
        atomic_write_json(bridge_metadata_path, bridge_metadata)

    result = {
        "status": "pass" if primary_status == bridge_status == "pass" else "fail",
        "phase": row["phase"],
        "n_primary": nprimary,
        "n_zero_bridge": nbridge,
        "primary_status": primary_status,
        "bridge_status": bridge_status,
    }
    print(json.dumps(result, sort_keys=True))
    if result["status"] != "pass":
        raise SystemExit(2)
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--phase", choices=PHASES, required=True)
    parser.add_argument("--chunk-size", type=int, default=1_000_000)
    args = parser.parse_args()
    build_one(select_row(read_jsonl(args.manifest), args.phase), chunk_size=int(args.chunk_size))


if __name__ == "__main__":
    main()
