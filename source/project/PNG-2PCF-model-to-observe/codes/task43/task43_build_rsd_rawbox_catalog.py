#!/usr/bin/env python3
"""Build mass-matched real/RSD rawbox catalogs for Task 4.3.2."""

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
    apply_plane_parallel_rsd,
    atomic_savez,
    atomic_write_json,
    finite_summary,
    header_boxsize,
    sha256_file,
)


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def select_row(rows: list[dict[str, Any]], phase: str) -> dict[str, Any]:
    matches = [row for row in rows if row["phase"] == phase]
    if len(matches) != 1:
        raise ValueError(f"expected one manifest row for {phase}, found {len(matches)}")
    return matches[0]


def legacy_catalog_path(phase: str) -> Path:
    return Path(
        "/pscratch/sd/l/lzy/PNG-2PCF-model-to-observe/plots/outputs/task43_outputs/"
        "ezmock_rawbox_z0p725_mmin1p4e13/halo_catalogs/"
        f"halo_AbacusSummit_base_c000_{phase}_z0p725_mmin1p4e13.npz"
    )


def build_one(row: dict[str, Any], *, chunk_size: int, force: bool) -> dict[str, Any]:
    import asdf

    output = Path(row["rawbox_catalog_path"])
    metadata_path = Path(row["rawbox_metadata_path"])
    if output.exists() or metadata_path.exists():
        if not force:
            if output.is_file() and metadata_path.is_file():
                metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
                if metadata.get("status") == "pass" and metadata.get("output_sha256") == sha256_file(output):
                    print(f"[skip] validated existing {output}")
                    return metadata
            raise FileExistsError(f"partial or unvalidated output exists: {output} / {metadata_path}")

    started = time.perf_counter()
    positions: list[np.ndarray] = []
    velocities: list[np.ndarray] = []
    counts: list[np.ndarray] = []
    source_paths = [Path(path) for path in row["rawbox_source_paths"]]
    if len(source_paths) != 34 or any(not path.is_file() for path in source_paths):
        raise RuntimeError(f"{row['phase']} requires exactly 34 existing rawbox slabs")

    particle_mass: float | None = None
    boxsize: float | None = None
    velocity_factor: float | None = None
    threshold_n: int | None = None
    slab_audit: list[dict[str, Any]] = []
    n_total = 0
    for path in source_paths:
        slab_started = time.perf_counter()
        with asdf.open(path, lazy_load=True, memmap=False) as af:
            header = af["header"]
            this_mass = float(header["ParticleMassHMsun"])
            this_box = header_boxsize(header)
            this_velocity_factor = float(header["VelZSpace_to_kms"])
            if particle_mass is None:
                particle_mass = this_mass
                boxsize = this_box
                velocity_factor = this_velocity_factor
                threshold_n = int(math.ceil(MASS_THRESHOLD_HMSUN / particle_mass))
            if not np.isclose(this_mass, particle_mass, rtol=0.0, atol=1.0e-6):
                raise RuntimeError(f"ParticleMassHMsun changed in {path}")
            if not np.isclose(this_box, boxsize, rtol=0.0, atol=1.0e-10):
                raise RuntimeError(f"boxsize changed in {path}")
            if not np.isclose(this_velocity_factor, velocity_factor, rtol=0.0, atol=1.0e-8):
                raise RuntimeError(f"VelZSpace_to_kms changed in {path}")
            data = af["data"]
            narr = data["N"]
            xarr = data["x_L2com"]
            varr = data["v_L2com"]
            nobj = int(len(narr))
            n_total += nobj
            n_selected_slab = 0
            for start in range(0, nobj, int(chunk_size)):
                stop = min(start + int(chunk_size), nobj)
                n = np.asarray(narr[start:stop], dtype="u4")
                keep = n >= int(threshold_n)
                if not np.any(keep):
                    continue
                x_internal = np.asarray(xarr[start:stop], dtype="f4")[keep]
                v_internal = np.asarray(varr[start:stop], dtype="f4")[keep]
                position = np.mod((x_internal.astype("f8") + 0.5) * float(boxsize), float(boxsize)).astype("f4")
                velocity = (v_internal.astype("f8") * float(velocity_factor)).astype("f4")
                positions.append(position)
                velocities.append(velocity)
                counts.append(n[keep])
                n_selected_slab += int(position.shape[0])
            slab_audit.append(
                {
                    "path": str(path),
                    "n_total": nobj,
                    "n_selected": n_selected_slab,
                    "elapsed_sec": time.perf_counter() - slab_started,
                }
            )

    if particle_mass is None or boxsize is None or velocity_factor is None or threshold_n is None or not positions:
        raise RuntimeError(f"failed to build {row['phase']} rawbox catalog")
    position_real = np.concatenate(positions).astype("f4", copy=False)
    velocity_kms = np.concatenate(velocities).astype("f4", copy=False)
    halo_count = np.concatenate(counts).astype("u4", copy=False)
    position_rsd, displacement = apply_plane_parallel_rsd(
        position_real,
        velocity_kms,
        velocity_kms_per_mpc_h_value=float(velocity_factor) / float(boxsize),
        boxsize_mpc_h=float(boxsize),
        los_axis=int(row["rsd_plane_parallel_los_axis"]),
    )
    position_rsd = position_rsd.astype("f4")
    displacement = displacement.astype("f4")
    position_zero, displacement_zero = apply_plane_parallel_rsd(
        position_real,
        np.zeros_like(velocity_kms),
        velocity_kms_per_mpc_h_value=float(velocity_factor) / float(boxsize),
        boxsize_mpc_h=float(boxsize),
        los_axis=int(row["rsd_plane_parallel_los_axis"]),
    )
    zero_velocity_max_abs = float(np.max(np.abs(position_zero - position_real)))
    if zero_velocity_max_abs != 0.0 or np.any(displacement_zero != 0.0):
        raise RuntimeError("zero-velocity rawbox mapping is not exact")

    legacy_path = legacy_catalog_path(str(row["phase"]))
    legacy_bridge: dict[str, Any] = {"path": str(legacy_path), "available": legacy_path.is_file()}
    if legacy_path.is_file():
        with np.load(legacy_path, allow_pickle=False) as legacy:
            legacy_position = np.asarray(legacy["POSITION"], dtype="f4")
            legacy_n = np.asarray(legacy["N"], dtype="u4")
        legacy_bridge.update(
            {
                "shape_equal": legacy_position.shape == position_real.shape,
                "counts_equal": bool(np.array_equal(legacy_n, halo_count)),
                "position_max_abs_mpc_h": (
                    float(np.max(np.abs(legacy_position - position_real)))
                    if legacy_position.shape == position_real.shape
                    else None
                ),
            }
        )
        legacy_bridge["pass"] = bool(
            legacy_bridge["shape_equal"]
            and legacy_bridge["counts_equal"]
            and legacy_bridge["position_max_abs_mpc_h"] == 0.0
        )
    else:
        legacy_bridge["pass"] = None

    atomic_savez(
        output,
        POSITION_REAL=position_real,
        POSITION_RSD=position_rsd,
        VELOCITY_KMS=velocity_kms,
        VLOS_KMS=velocity_kms[:, int(row["rsd_plane_parallel_los_axis"])],
        RSD_DISPLACEMENT_MPC_H=displacement,
        N=halo_count,
        phase=np.asarray(row["phase"]),
        sim_name=np.asarray(row["sim_name"]),
        redshift=np.asarray(0.725, dtype="f8"),
        boxsize=np.asarray(boxsize, dtype="f8"),
        mass_threshold_hmsun=np.asarray(MASS_THRESHOLD_HMSUN, dtype="f8"),
        threshold_n=np.asarray(threshold_n, dtype="i8"),
        velocity_conversion_kms_per_mpc_h=np.asarray(float(velocity_factor) / float(boxsize), dtype="f8"),
    )
    metadata = {
        "task": "task43_build_rsd_rawbox_catalog",
        "status": "pass" if legacy_bridge.get("pass") in (True, None) else "fail",
        "phase": row["phase"],
        "sim_name": row["sim_name"],
        "n_slabs": len(source_paths),
        "n_total": n_total,
        "n_selected": int(position_real.shape[0]),
        "boxsize_mpc_h": float(boxsize),
        "particle_mass_hmsun": float(particle_mass),
        "mass_threshold_hmsun": MASS_THRESHOLD_HMSUN,
        "threshold_n": int(threshold_n),
        "velocity_factor_header": float(velocity_factor),
        "velocity_kms_per_mpc_h": float(velocity_factor) / float(boxsize),
        "mapping": "plane_parallel_z: s_z=x_z+v_z/(VelZSpace_to_kms/BoxSize), periodic wrap",
        "position_field": "x_L2com",
        "velocity_field": "v_L2com * VelZSpace_to_kms",
        "zero_velocity_max_abs_mpc_h": zero_velocity_max_abs,
        "legacy_bridge": legacy_bridge,
        "velocity_norm_kms": finite_summary(np.linalg.norm(velocity_kms.astype("f8"), axis=1)),
        "vlos_kms": finite_summary(velocity_kms[:, int(row["rsd_plane_parallel_los_axis"])]),
        "displacement_mpc_h": finite_summary(displacement),
        "slabs": slab_audit,
        "output_path": str(output),
        "output_sha256": sha256_file(output),
        "elapsed_sec": time.perf_counter() - started,
    }
    atomic_write_json(metadata_path, metadata)
    print(json.dumps({"status": metadata["status"], "phase": row["phase"], "n_selected": metadata["n_selected"], "output": str(output)}, sort_keys=True))
    if metadata["status"] != "pass":
        raise SystemExit(2)
    return metadata


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--phase", choices=PHASES, required=True)
    parser.add_argument("--chunk-size", type=int, default=1_000_000)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    build_one(select_row(read_jsonl(args.manifest), args.phase), chunk_size=int(args.chunk_size), force=bool(args.force))


if __name__ == "__main__":
    main()

