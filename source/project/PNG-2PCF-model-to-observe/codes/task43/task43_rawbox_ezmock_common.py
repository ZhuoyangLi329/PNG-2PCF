#!/usr/bin/env python3
"""Shared paths and small I/O helpers for the Task43 EZmock raw-box target."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

import numpy as np


PROJECT_ROOT = Path("/pscratch/sd/l/lzy/PNG-2PCF-model-to-observe")
ABACUS_ROOT = Path("/global/cfs/cdirs/desi/public/cosmosim/AbacusSummit")
SIM_PREFIX = "AbacusSummit_base_c000"
PHASES = tuple(f"ph{index:03d}" for index in range(25))
SNAPSHOT = "z0.725"
SNAPSHOT_TAG = "z0p725"
SELECTION_TAG = "mmin1p4e13"
MASS_THRESHOLD_HMSUN = 1.4e13
BOX_SIZE = 2000.0
K_FUND = 2.0 * np.pi / BOX_SIZE

OUTPUT_ROOT = PROJECT_ROOT / "outputs" / "task43_outputs" / f"ezmock_rawbox_{SNAPSHOT_TAG}_{SELECTION_TAG}"
CATALOG_DIR = OUTPUT_ROOT / "halo_catalogs"
ASCII_DIR = OUTPUT_ROOT / "fcfc_catalogs"
XI_DIR = OUTPUT_ROOT / "xi_fcfc"
PK_DIR = OUTPUT_ROOT / "pk_jaxpower"
CONFIG_DIR = OUTPUT_ROOT / "fcfc_configs"
PAIR_DIR = OUTPUT_ROOT / "fcfc_pairs"
LOG_DIR = OUTPUT_ROOT / "logs"
SUMMARY_DIR = OUTPUT_ROOT / "summary"


def sim_name(phase: str) -> str:
    if phase not in PHASES:
        raise ValueError(f"unknown phase: {phase}")
    return f"{SIM_PREFIX}_{phase}"


def halo_info_dir(phase: str) -> Path:
    return ABACUS_ROOT / sim_name(phase) / "halos" / SNAPSHOT / "halo_info"


def halo_info_paths(phase: str) -> list[Path]:
    return sorted(halo_info_dir(phase).glob("halo_info_*.asdf"))


def catalog_path(phase: str) -> Path:
    return CATALOG_DIR / f"halo_{sim_name(phase)}_{SNAPSHOT_TAG}_{SELECTION_TAG}.npz"


def catalog_metadata_path(phase: str) -> Path:
    return catalog_path(phase).with_suffix(".json")


def ascii_path(phase: str) -> Path:
    return ASCII_DIR / f"halo_{sim_name(phase)}_{SNAPSHOT_TAG}_{SELECTION_TAG}_xyz.txt"


def pk_path(phase: str, meshsize: int = 400) -> Path:
    return PK_DIR / f"pk0_{sim_name(phase)}_{SNAPSHOT_TAG}_{SELECTION_TAG}_mesh{int(meshsize)}.npz"


def pk_metadata_path(phase: str, meshsize: int = 400) -> Path:
    return pk_path(phase, meshsize).with_suffix(".json")


def xi_path(phase: str) -> Path:
    return XI_DIR / f"xi0_{sim_name(phase)}_{SNAPSHOT_TAG}_{SELECTION_TAG}_s50_550_ds10_fcfc.npz"


def xi_metadata_path(phase: str) -> Path:
    return xi_path(phase).with_suffix(".json")


def ensure_dirs() -> None:
    for path in (CATALOG_DIR, ASCII_DIR, XI_DIR, PK_DIR, CONFIG_DIR, PAIR_DIR, LOG_DIR, SUMMARY_DIR):
        path.mkdir(parents=True, exist_ok=True)


def to_jsonable(value: Any) -> Any:
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, dict):
        return {str(key): to_jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [to_jsonable(item) for item in value]
    return str(value)


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    tmp.write_text(json.dumps(to_jsonable(payload), indent=2, sort_keys=True) + "\n", encoding="utf-8")
    tmp.replace(path)


def atomic_savez(path: Path, **arrays: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.{os.getpid()}.tmp.npz")
    try:
        np.savez_compressed(tmp, **arrays)
        tmp.replace(path)
    except Exception:
        tmp.unlink(missing_ok=True)
        raise


def set_cpu_affinity(max_threads: int) -> list[int]:
    """Restrict this process and all children to at most ``max_threads`` CPUs."""
    if not 1 <= int(max_threads) <= 12:
        raise ValueError(f"threads must be in [1, 12], got {max_threads}")
    available = sorted(os.sched_getaffinity(0))
    selected = available[: min(int(max_threads), len(available))]
    if not selected:
        raise RuntimeError("empty CPU affinity set")
    os.sched_setaffinity(0, selected)
    return selected


def load_catalog(phase: str) -> tuple[np.ndarray, dict[str, Any]]:
    path = catalog_path(phase)
    meta_path = catalog_metadata_path(phase)
    if not path.exists() or not meta_path.exists():
        raise FileNotFoundError(f"missing raw-box catalog or metadata for {phase}: {path}")
    with np.load(path, allow_pickle=False) as data:
        position = np.asarray(data["POSITION"], dtype="f8")
    metadata = json.loads(meta_path.read_text(encoding="utf-8"))
    if position.ndim != 2 or position.shape[1] != 3:
        raise ValueError(f"bad POSITION shape in {path}: {position.shape}")
    if position.shape[0] != int(metadata["n_selected"]):
        raise ValueError(f"catalog/metadata count mismatch for {phase}")
    return position, metadata
