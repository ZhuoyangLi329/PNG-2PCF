#!/usr/bin/env python3
"""Task43 EZmock RSD lightcone covariance production contract."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import Any

import numpy as np

PROJECT_ROOT = Path("/pscratch/sd/l/lzy/PNG-2PCF-model-to-observe")
OUTPUT_ROOT = PROJECT_ROOT / "outputs/task43_outputs/ezmock_rsd_lightcone_covariance_z0p4_0p8_x1000_common50"
MANIFEST_DIR = OUTPUT_ROOT / "manifests"
CONFIG_DIR = OUTPUT_ROOT / "ezmock_configs"
RAWBOX_DIR = OUTPUT_ROOT / "rawbox_catalogs"
LIGHTCONE_DIR = OUTPUT_ROOT / "lightcone_catalogs"
RANDOM_DIR = OUTPUT_ROOT / "common_random"
FCFC_CONFIG_DIR = OUTPUT_ROOT / "fcfc_configs"
FCFC_PAIR_DIR = OUTPUT_ROOT / "fcfc_pairs"
XI_DIR = OUTPUT_ROOT / "xi_fcfc"
PK_DIR = OUTPUT_ROOT / "pk_jaxpower"
LOG_DIR = OUTPUT_ROOT / "logs"
SUMMARY_DIR = OUTPUT_ROOT / "summary"
TMP_DIR = OUTPUT_ROOT / "tmp"

MANIFEST = MANIFEST_DIR / "task43_ezmock_rsd_covariance_x1000_fixampF_common50.jsonl"
MANIFEST_AUDIT = MANIFEST.with_suffix(".json")
COMMON_RANDOM = RANDOM_DIR / "common_random_zobs0p4_0p8_x50.npz"
COMMON_RANDOM_META = COMMON_RANDOM.with_suffix(".json")
COMMON_RANDOM_HDF5 = RANDOM_DIR / "common_random_zobs0p4_0p8_x50_fcfc.h5"
COMMON_RANDOM_HDF5_META = COMMON_RANDOM_HDF5.with_suffix(".json")
COMMON_RR = FCFC_PAIR_DIR / "RR_common50_zobs0p4_0p8_s30_350_ds10.txt"
COMMON_RR_META = COMMON_RR.with_suffix(".json")
# Full s-mu RR cache required for FCFC xi0+xi2.
MU_BIN_NUM = 120
COMMON_RR_SMU = FCFC_PAIR_DIR / "RR_common50_zobs0p4_0p8_s30_350_ds10_mu120.bin"
COMMON_RR_SMU_META = COMMON_RR_SMU.with_suffix(".json")

EZMOCK_BINARY = PROJECT_ROOT / "EZmock/EZmock-1.0.0/EZmock"
LINEAR_PK = PROJECT_ROOT / "outputs/task43_outputs/ezmock_calibration_rawbox_z0p725_mmin1p4e13/linear_pk/abacus_c000_linear_matter_pk_z0p725_desilike_cosmoprimo.dat"
FCFC_BINARY = PROJECT_ROOT / "refcode/FCFC-main/FCFC_2PT"
ABACUS_FKP_DIR = PROJECT_ROOT / "outputs/task43_outputs/rsd_validation/lightcone_boxsafe_zobs0p4_0p8/fkp"

NREAL = 1000
SEED_BASE = 600000
FIX_AMPLITUDE = False
ATTACH_PARTICLE = True
BOX_SIZE = 2000.0
NGRID = 320
DEFAULT_NTRACER = 1_512_000
REDSHIFT = 0.725
ZMIN = 0.4
ZMAX = 0.8
RHO_C = 1.14
RHO_EXP = 5.0
PDF_BASE = 0.25
SIGMA_V = 0.0
RAND_GENERATOR = 1
PK_INTERP_LOG = True
INVERT_PHASE = False
BAO_ENHANCE = 0.0
OMEGA_M = 0.3137721026737606
OMEGA_NU = 0.0014197664745152646
P0_FKP = 10_000.0
RANDOM_MULTIPLIER = 50
TARGET_NDATA = 591_308

S_EDGES = np.arange(30.0, 360.0, 10.0, dtype="f8")
KMIN = 0.001
KMAX = 0.3001
DK = 0.002
PK_MESHSIZE = 256
PK_MESH_PAD = 400.0
N_PK_FINE = 150
FIT_KMAX = 0.08
FIT_SMIN = 50.0
FIT_SMAX = 350.0
BAO_MASK = (80.0, 120.0)

def ensure_dirs() -> None:
    for path in (MANIFEST_DIR, CONFIG_DIR, RAWBOX_DIR, LIGHTCONE_DIR, RANDOM_DIR,
                 FCFC_CONFIG_DIR, FCFC_PAIR_DIR, XI_DIR, PK_DIR, LOG_DIR, SUMMARY_DIR, TMP_DIR):
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

def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    with tmp.open("w", encoding="utf-8") as stream:
        for row in rows:
            stream.write(json.dumps(to_jsonable(row), sort_keys=True) + "\n")
    tmp.replace(path)

def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]

def atomic_savez(path: Path, **arrays: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.{os.getpid()}.tmp.npz")
    np.savez_compressed(tmp, **arrays)
    tmp.replace(path)

def sha256(path: Path, chunk_bytes: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(chunk_bytes), b""):
            digest.update(block)
    return digest.hexdigest()

def row_paths(index: int, seed: int) -> dict[str, str]:
    token = f"ezmock_m{int(index):04d}_seed{int(seed)}"
    return {
        "ezmock_config_path": str(CONFIG_DIR / f"{token}.conf"),
        "ezmock_log_path": str(LOG_DIR / f"{token}.log"),
        "rawbox_catalog_path": str(RAWBOX_DIR / f"{token}_raw6.dat"),
        "lightcone_catalog_path": str(LIGHTCONE_DIR / f"{token}_zobs0p4_0p8.npz"),
        "lightcone_metadata_path": str(LIGHTCONE_DIR / f"{token}_zobs0p4_0p8.json"),
        "random_catalog_path": str(COMMON_RANDOM),
        "random_metadata_path": str(COMMON_RANDOM_META),
        "pk_path": str(PK_DIR / f"{token}_p02_mesh{PK_MESHSIZE}.npz"),
        "xi_path": str(XI_DIR / f"{token}_xi02_s30_350_ds10.npz"),
    }

# Compatibility aliases for shared FCFC/JAX measurement drivers.
ABACUS_FKP_SUMMARY = RANDOM_DIR / "mean_abacus_rsd_fkp_summary.npz"
P0 = P0_FKP
COMMON_RANDOM_SIZE = TARGET_NDATA * RANDOM_MULTIPLIER
