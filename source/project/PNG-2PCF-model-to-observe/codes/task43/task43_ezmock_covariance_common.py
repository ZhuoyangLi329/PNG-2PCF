#!/usr/bin/env python3
"""Shared contract and atomic I/O for Task43 EZmock covariance production."""

from __future__ import annotations

import hashlib
import json
import logging
import os
from pathlib import Path
from typing import Any

import numpy as np


class _JaxCpuOnlyCudaProbeFilter(logging.Filter):
    """Drop only the optional CUDA-plugin traceback on CPU-only workers."""

    def filter(self, record: logging.LogRecord) -> bool:
        message = record.getMessage()
        return not (
            record.name == "jax._src.xla_bridge"
            and "Jax plugin configuration error" in message
            and "jax_plugins.xla_cuda12.initialize()" in message
        )


def install_jax_cpu_only_log_filter() -> None:
    """Keep CPU-backend failures visible while silencing CUDA probe noise."""
    logger = logging.getLogger("jax._src.xla_bridge")
    if not any(isinstance(item, _JaxCpuOnlyCudaProbeFilter) for item in logger.filters):
        logger.addFilter(_JaxCpuOnlyCudaProbeFilter())


PROJECT_ROOT = Path("/pscratch/sd/l/lzy/PNG-2PCF-model-to-observe")
CODE_DIR = PROJECT_ROOT / "codes" / "task43"
OUTPUT_ROOT = PROJECT_ROOT / "outputs/task43_outputs/ezmock_lightcone_covariance_x1000_s50_350_common50"
PLOT_DIR = PROJECT_ROOT / "plots/task43/ezmock_lightcone_covariance"

MANIFEST_DIR = OUTPUT_ROOT / "manifests"
CONFIG_DIR = OUTPUT_ROOT / "ezmock_configs"
RAWBOX_DIR = OUTPUT_ROOT / "tmp/rawbox_catalogs"
LIGHTCONE_DIR = OUTPUT_ROOT / "lightcone_catalogs"
COMMON_RANDOM_DIR = OUTPUT_ROOT / "common_random"
FCFC_CONFIG_DIR = OUTPUT_ROOT / "fcfc_configs"
FCFC_PAIR_DIR = OUTPUT_ROOT / "fcfc_pairs"
XI_DIR = OUTPUT_ROOT / "xi_fcfc"
PK_DIR = OUTPUT_ROOT / "pk_jaxpower"
LOG_DIR = OUTPUT_ROOT / "logs"
SUMMARY_DIR = OUTPUT_ROOT / "summary"
TMP_DIR = OUTPUT_ROOT / "tmp"

MANIFEST = MANIFEST_DIR / "task43_ezmock_covariance_x1000_fixampF_common50_s50_350.jsonl"
MANIFEST_AUDIT = MANIFEST.with_suffix(".json")
ABUNDANCE_AUDIT = SUMMARY_DIR / "task43_ezmock_covariance_abundance_pilot.json"
COMMON_RANDOM = COMMON_RANDOM_DIR / "task43_common_random_ezmock_geometric_shell_x50_n16577250.npz"
COMMON_RANDOM_META = COMMON_RANDOM.with_suffix(".json")
COMMON_RANDOM_HDF5 = COMMON_RANDOM_DIR / "task43_common_random_ezmock_geometric_shell_x50_n16577250_fcfc.h5"
COMMON_RANDOM_HDF5_META = COMMON_RANDOM_HDF5.with_suffix(".json")
COMMON_RR = FCFC_PAIR_DIR / "RR_common50_ezmock_geometric_shell_s50_350_ds10.txt"
COMMON_RR_META = COMMON_RR.with_suffix(".json")

ABACUS_FKP_SUMMARY = PROJECT_ROOT / "outputs/task43_outputs/summary/task43_fkp_zeff_mmin1p4e13_x25.npz"
ABACUS_XI_SUMMARY = PROJECT_ROOT / "outputs/task43_outputs/summary/task43_mean_xi_mmin1p4e13_x25_s50_350_ds10_fkpP010000.npz"
if not ABACUS_XI_SUMMARY.is_file():
    ABACUS_XI_SUMMARY = PROJECT_ROOT / "outputs/task43_outputs/rmax_scan/summary/task43_mean_xi_mmin1p4e13_x25_s50_550_ds10_fkpP010000.npz"
ABACUS_PK_PAYLOAD = PROJECT_ROOT / "outputs/task43_outputs/pk_lightcone/summary/task43_pk_lightcone_mmin1p4e13_x25_fkpP010000_desi_rebin_kmax0p10_payload.npz"
LINEAR_PK = PROJECT_ROOT / "outputs/task43_outputs/ezmock_calibration_rawbox_z0p725_mmin1p4e13/linear_pk/abacus_c000_linear_matter_pk_z0p725_desilike_cosmoprimo.dat"
EZMOCK_BINARY = PROJECT_ROOT / "EZmock/EZmock-1.0.0/EZmock"
FCFC_BINARY = PROJECT_ROOT / "refcode/FCFC-main/FCFC_2PT_HDF5"

NREAL = 1000
SEED_BASE = 432_000
SEEDS = tuple(SEED_BASE + index + 1 for index in range(NREAL))
PHASES = tuple(f"ph{1000 + index:04d}" for index in range(NREAL))
FIX_AMPLITUDE = False
ATTACH_PARTICLE = True

BOX_SIZE = 2000.0
NGRID = 320
DEFAULT_NTRACER = 1_391_337
REDSHIFT = 0.725
ZMIN = 0.6
ZMAX = 0.8
RHO_C = 1.14
RHO_EXP = 5.0
PDF_BASE = 0.25
SIGMA_V = 0.0
RAND_GENERATOR = 1
PK_INTERP_LOG = True
INVERT_PHASE = False
BAO_ENHANCE = 0.0
OMEGA_M_NON_NEUTRINO = 0.3137721026737606
OMEGA_NU = 0.0014197664745152646

P0 = 10_000.0
TARGET_NDATA = 331_545
COMMON_RANDOM_MULTIPLIER = 50
COMMON_RANDOM_SIZE = TARGET_NDATA * COMMON_RANDOM_MULTIPLIER
COMMON_RANDOM_SEED = 43_250_001
S_EDGES = np.arange(50.0, 350.0 + 10.0, 10.0, dtype="f8")
N_XI_BINS = S_EDGES.size - 1
N_PK_BINS = 15
JOINT_DIMENSION = N_XI_BINS + N_PK_BINS
PK_MESHSIZE = 256
PK_MESH_PAD = 400.0
KMIN = 0.001
KMAX = 0.3001
DK = 0.002


def ensure_dirs() -> None:
    for path in (
        MANIFEST_DIR, CONFIG_DIR, RAWBOX_DIR, LIGHTCONE_DIR, COMMON_RANDOM_DIR,
        FCFC_CONFIG_DIR, FCFC_PAIR_DIR, XI_DIR, PK_DIR, LOG_DIR, SUMMARY_DIR,
        TMP_DIR, PLOT_DIR,
    ):
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
    try:
        np.savez_compressed(tmp, **arrays)
        tmp.replace(path)
    except Exception:
        tmp.unlink(missing_ok=True)
        raise


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def row_paths(phase: str, seed: int) -> dict[str, str]:
    token = f"ezmock_{phase}_seed{seed}"
    return {
        "ezmock_config_path": str(CONFIG_DIR / f"{token}.conf"),
        "ezmock_log_path": str(LOG_DIR / f"{token}.log"),
        "rawbox_catalog_path": str(RAWBOX_DIR / f"{token}.dat"),
        "halo_catalog_path": str(LIGHTCONE_DIR / f"lightcone_{token}_z0p6_0p8.npz"),
        "halo_metadata_path": str(LIGHTCONE_DIR / f"lightcone_{token}_z0p6_0p8.json"),
        "random_catalog_path": str(COMMON_RANDOM),
        "random_metadata_path": str(COMMON_RANDOM_META),
        "rr_path": str(COMMON_RR),
        "xi_path": str(XI_DIR / f"xi0_{token}_common50_s50_350_ds10_fcfc.npz"),
        "pk_path": str(PK_DIR / f"pk0_{token}_common50_mesh256_kmax0p300_dk0p002.npz"),
    }
