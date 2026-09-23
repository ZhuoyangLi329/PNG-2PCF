#!/usr/bin/env python3
"""Shared, side-effect-free utilities for the Task 4.3.2 RSD validation."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import Any, Mapping

import numpy as np


PROJECT_ROOT = Path("/pscratch/sd/l/lzy/PNG-2PCF-model-to-observe")
ABACUS_ROOT = Path("/global/cfs/cdirs/desi/public/cosmosim/AbacusSummit")
LIGHTCONE_ROOT = ABACUS_ROOT / "halo_light_cones"
SIM_PREFIX = "AbacusSummit_base_c000"
PHASES = tuple(f"ph{index:03d}" for index in range(25))

OUTPUT_ROOT = PROJECT_ROOT / "outputs" / "task43_outputs" / "rsd_validation"
PLOT_ROOT = PROJECT_ROOT / "plots" / "task43" / "rsd_validation"
LOG_ROOT = PROJECT_ROOT / "codes" / "logs" / "task43" / "rsd_validation"

MASS_THRESHOLD_HMSUN = 1.4e13
BOX_SIZE_MPC_H = 2000.0
ZMIN = 0.6
ZMAX = 0.8
P0_FKP = 10000.0
P_FIXED = 1.0
S_EDGES = np.arange(30.0, 360.0, 10.0, dtype="f8")
SMIN_SCAN = (30, 40, 50, 80, 100, 120)
RAWBOX_XI_PRIMARY_S_RANGE = (50.0, 350.0)
RAWBOX_XI_PRIMARY_BAO_EXCLUSION = (80.0, 120.0)
LIGHTCONE_SHELLS = ("z0.575", "z0.650", "z0.725", "z0.800")


def rawbox_xi_primary_mask(centers_mpc_h: np.ndarray) -> np.ndarray:
    """Return the canonical rawbox xi fit mask with the BAO region removed.

    The interval convention is lower-inclusive and upper-exclusive.  For the
    Task43 10 Mpc/h shells this retains centers 55, 65, 75 and 125..345 while
    excluding 85, 95, 105 and 115 Mpc/h.
    """

    centers = np.asarray(centers_mpc_h, dtype="f8")
    if centers.ndim != 1 or centers.size == 0 or not np.all(np.isfinite(centers)):
        raise ValueError("separation centers must be a nonempty finite one-dimensional array")
    if np.any(np.diff(centers) <= 0.0):
        raise ValueError("separation centers must increase strictly")
    smin, smax = RAWBOX_XI_PRIMARY_S_RANGE
    bao_min, bao_max = RAWBOX_XI_PRIMARY_BAO_EXCLUSION
    fit_range = (centers >= smin) & (centers < smax)
    bao_region = (centers >= bao_min) & (centers < bao_max)
    return fit_range & ~bao_region


def sim_name(phase: str) -> str:
    """Return the canonical AbacusSummit simulation name."""
    if phase not in PHASES:
        raise ValueError(f"unknown phase {phase!r}")
    return f"{SIM_PREFIX}_{phase}"


def rawbox_halo_dir(phase: str) -> Path:
    return ABACUS_ROOT / sim_name(phase) / "halos" / "z0.725" / "halo_info"


def lightcone_shell_path(phase: str, shell: str) -> Path:
    return LIGHTCONE_ROOT / sim_name(phase) / shell / "lc_halo_info.asdf"


def sha256_file(path: Path, *, chunk_bytes: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while block := stream.read(int(chunk_bytes)):
            digest.update(block)
    return digest.hexdigest()


def _json_default(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    raise TypeError(f"not JSON serializable: {type(value).__name__}")


def atomic_write_json(path: Path, payload: Mapping[str, Any]) -> None:
    """Atomically write deterministic, human-readable JSON."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        temporary.write_text(
            json.dumps(payload, indent=2, sort_keys=True, default=_json_default) + "\n",
            encoding="utf-8",
        )
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def atomic_savez(path: Path, *, compressed: bool = True, **arrays: Any) -> None:
    """Atomically write an NPZ without relying on NumPy filename rewriting."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    writer = np.savez_compressed if compressed else np.savez
    try:
        with temporary.open("wb") as stream:
            writer(stream, **arrays)
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def header_boxsize(header: Mapping[str, Any]) -> float:
    for key in ("BoxSizeHMpc", "BoxSize"):
        if key in header:
            value = float(header[key])
            if np.isfinite(value) and value > 0.0:
                return value
    raise KeyError("header contains neither a valid BoxSizeHMpc nor BoxSize")


def velocity_kms_per_mpc_h(header: Mapping[str, Any]) -> float:
    """Return the Abacus redshift-space conversion in km/s per (Mpc/h)."""
    boxsize = header_boxsize(header)
    value = float(header["VelZSpace_to_kms"]) / boxsize
    if not np.isfinite(value) or value <= 0.0:
        raise ValueError(f"invalid velocity conversion {value}")
    return value


def apply_plane_parallel_rsd(
    position_mpc_h: np.ndarray,
    velocity_kms: np.ndarray,
    *,
    velocity_kms_per_mpc_h_value: float,
    boxsize_mpc_h: float,
    los_axis: int = 2,
    velocity_scale: float = 1.0,
) -> tuple[np.ndarray, np.ndarray]:
    """Apply a periodic plane-parallel RSD displacement along one axis."""
    position = np.asarray(position_mpc_h, dtype="f8")
    velocity = np.asarray(velocity_kms, dtype="f8")
    if position.ndim != 2 or position.shape[1] != 3 or velocity.shape != position.shape:
        raise ValueError(f"bad position/velocity shapes: {position.shape}, {velocity.shape}")
    axis = int(los_axis)
    if axis not in (0, 1, 2):
        raise ValueError(f"los_axis must be 0, 1, or 2; got {los_axis}")
    conversion = float(velocity_kms_per_mpc_h_value)
    boxsize = float(boxsize_mpc_h)
    if not np.isfinite(conversion) or conversion <= 0.0:
        raise ValueError(f"invalid velocity conversion {conversion}")
    if not np.isfinite(boxsize) or boxsize <= 0.0:
        raise ValueError(f"invalid boxsize {boxsize}")
    displacement = float(velocity_scale) * velocity[:, axis] / conversion
    shifted = np.array(position, copy=True)
    shifted[:, axis] = np.mod(shifted[:, axis] + displacement, boxsize)
    return shifted, displacement


def apply_radial_rsd(
    position_absolute_mpc_h: np.ndarray,
    velocity_kms: np.ndarray,
    observer_origins_mpc_h: np.ndarray,
    origin_code: np.ndarray,
    velocity_kms_per_mpc_h_by_object: np.ndarray,
    *,
    velocity_scale: float = 1.0,
) -> dict[str, np.ndarray]:
    """Apply the AbacusHOD local-radial RSD convention to lightcone objects."""
    position = np.asarray(position_absolute_mpc_h, dtype="f8")
    velocity = np.asarray(velocity_kms, dtype="f8")
    origins = np.asarray(observer_origins_mpc_h, dtype="f8").reshape(-1, 3)
    codes = np.asarray(origin_code, dtype="i8")
    conversion = np.asarray(velocity_kms_per_mpc_h_by_object, dtype="f8")
    if position.ndim != 2 or position.shape[1] != 3 or velocity.shape != position.shape:
        raise ValueError(f"bad position/velocity shapes: {position.shape}, {velocity.shape}")
    nobject = position.shape[0]
    if codes.shape != (nobject,) or conversion.shape != (nobject,):
        raise ValueError(f"bad code/conversion shapes: {codes.shape}, {conversion.shape}")
    if origins.shape[0] < 1:
        raise ValueError("at least one observer origin is required")
    if np.any(~np.isfinite(conversion)) or np.any(conversion <= 0.0):
        raise ValueError("velocity conversions must be finite and positive")
    origin_index = np.mod(codes, origins.shape[0])
    origin = origins[origin_index]
    relative_real = position - origin
    radius_real = np.linalg.norm(relative_real, axis=1)
    if np.any(~np.isfinite(radius_real)) or np.any(radius_real <= 0.0):
        raise ValueError("observer-relative radii must be finite and positive")
    direction = relative_real / radius_real[:, None]
    velocity_los = np.einsum("ij,ij->i", velocity, direction)
    displacement = float(velocity_scale) * velocity_los / conversion
    position_rsd = position + displacement[:, None] * direction
    relative_rsd = position_rsd - origin
    radius_rsd = np.linalg.norm(relative_rsd, axis=1)
    return {
        "position_rsd_absolute": position_rsd,
        "position_real_relative": relative_real,
        "position_rsd_relative": relative_rsd,
        "radius_real": radius_real,
        "radius_rsd": radius_rsd,
        "direction": direction,
        "velocity_los": velocity_los,
        "displacement": displacement,
        "origin_index": origin_index.astype("i1"),
    }


def apply_official_lightcone_fallback(
    position_interp_mpc_h: np.ndarray,
    velocity_interp_kms: np.ndarray,
    position_average_mpc_h: np.ndarray,
    velocity_average_kms: np.ndarray,
    origin_code: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Apply the cleaned-halo fallback used by the Abacus lightcone loader.

    The installed ``CompaSOHaloCatalog`` loader defines availability as any
    non-zero component of ``pos_avg``. Available snapshot averages replace the
    interpolated position and velocity. Observer lookup independently remains
    ``origin % n_origins`` in :func:`apply_radial_rsd`.
    """
    position_interp = np.asarray(position_interp_mpc_h, dtype="f8")
    velocity_interp = np.asarray(velocity_interp_kms, dtype="f8")
    position_average = np.asarray(position_average_mpc_h, dtype="f8")
    velocity_average = np.asarray(velocity_average_kms, dtype="f8")
    codes = np.asarray(origin_code, dtype="i8")
    shape = position_interp.shape
    if (
        position_interp.ndim != 2
        or shape[1] != 3
        or velocity_interp.shape != shape
        or position_average.shape != shape
        or velocity_average.shape != shape
        or codes.shape != (shape[0],)
    ):
        raise ValueError("lightcone fallback arrays have incompatible shapes")
    fallback = np.any(position_average, axis=1)
    position = np.array(position_interp, copy=True)
    velocity = np.array(velocity_interp, copy=True)
    position[fallback] = position_average[fallback]
    velocity[fallback] = velocity_average[fallback]
    return position, velocity, fallback


def finite_summary(values: np.ndarray) -> dict[str, float]:
    array = np.asarray(values, dtype="f8")
    if array.size == 0 or np.any(~np.isfinite(array)):
        raise ValueError("summary input must be non-empty and finite")
    return {
        "min": float(np.min(array)),
        "q01": float(np.quantile(array, 0.01)),
        "median": float(np.median(array)),
        "mean": float(np.mean(array)),
        "q99": float(np.quantile(array, 0.99)),
        "max": float(np.max(array)),
        "std": float(np.std(array, ddof=1)) if array.size > 1 else 0.0,
    }
