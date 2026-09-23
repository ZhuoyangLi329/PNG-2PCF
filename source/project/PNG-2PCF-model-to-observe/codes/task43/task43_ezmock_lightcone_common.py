#!/usr/bin/env python3
"""Task43 EZmock lightcone 验证流程的共享常量和轻量 I/O 工具。

执行大纲
--------
1. 集中冻结本轮 validation 的 EZmock、几何、random、FKP 与测量口径。
2. 提供 manifest、JSON 和 NPZ 的原子写入工具，避免中断时留下半文件。
3. 提供统一的路径函数，隔离 rawbox calibration、lightcone validation 和
   将来的 covariance production。

本文件只定义配置和小工具，不启动 EZmock、FCFC 或 jaxpower。
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

import numpy as np


PROJECT_ROOT = Path("/pscratch/sd/l/lzy/PNG-2PCF-model-to-observe")
CODE_DIR = PROJECT_ROOT / "codes" / "task43"
OUTPUT_ROOT = (
    PROJECT_ROOT
    / "outputs"
    / "task43_outputs"
    / "ezmock_lightcone_z0p6_0p8_mmin1p4e13_x25_fixedamp_validation"
)
PLOT_DIR = PROJECT_ROOT / "plots" / "task43" / "ezmock_lightcone"

MANIFEST_DIR = OUTPUT_ROOT / "manifests"
EZMOCK_CONFIG_DIR = OUTPUT_ROOT / "ezmock_configs"
RAWBOX_DIR = OUTPUT_ROOT / "rawbox_catalogs"
LIGHTCONE_DIR = OUTPUT_ROOT / "lightcone_catalogs"
RANDOM_DIR = OUTPUT_ROOT / "randoms"
FCFC_CONFIG_DIR = OUTPUT_ROOT / "fcfc_configs"
FCFC_PAIR_DIR = OUTPUT_ROOT / "fcfc_pairs"
XI_DIR = OUTPUT_ROOT / "xi_fcfc"
PK_DIR = OUTPUT_ROOT / "pk_jaxpower"
WINDOW_DIR = OUTPUT_ROOT / "windows"
SUMMARY_DIR = OUTPUT_ROOT / "summary"
LOG_DIR = OUTPUT_ROOT / "logs"

MANIFEST = MANIFEST_DIR / "task43_ezmock_lightcone_fixedamp_x10.jsonl"
AUDIT_MANIFEST = MANIFEST_DIR / "task43_ezmock_lightcone_fixedamp_x10.json"

ABACUS_RMAX_MANIFEST = (
    PROJECT_ROOT
    / "outputs/task43_outputs/rmax_scan/manifests/task43_rmax_scan_mmin1p4e13_x25.jsonl"
)
ABACUS_FKP_SUMMARY = (
    PROJECT_ROOT
    / "outputs/task43_outputs/summary/task43_fkp_zeff_mmin1p4e13_x25.npz"
)
ABACUS_XI_CUCOUNT_SUMMARY = (
    PROJECT_ROOT
    / "outputs/task43_outputs/rmax_scan/summary"
    / "task43_mean_xi_mmin1p4e13_x25_s50_550_ds10_fkpP010000.npz"
)
ABACUS_PK_PAYLOAD = (
    PROJECT_ROOT
    / "outputs/task43_outputs/pk_lightcone/summary"
    / "task43_pk_lightcone_mmin1p4e13_x25_fkpP010000_desi_rebin_payload.npz"
)

LINEAR_PK = (
    PROJECT_ROOT
    / "outputs/task43_outputs/ezmock_calibration_rawbox_z0p725_mmin1p4e13"
    / "linear_pk/abacus_c000_linear_matter_pk_z0p725_desilike_cosmoprimo.dat"
)
EZMOCK_BINARY = PROJECT_ROOT / "EZmock/EZmock-1.0.0/EZmock"
FCFC_BINARY = PROJECT_ROOT / "refcode/FCFC-main/FCFC_2PT"

# 本轮是均值/measurement validation，用户明确要求固定振幅保持开启。
# 将来正式 covariance production 必须建立另一条 FIX_AMPLITUDE=F 流程。
NREAL = 10
SEED_BASE = 431000
SEEDS = tuple(SEED_BASE + index for index in range(1, NREAL + 1))
PHASES = tuple(f"ph{100 + index:03d}" for index in range(NREAL))
FIX_AMPLITUDE = True

BOX_SIZE = 2000.0
NGRID = 320
NTRACER = 1_297_050
REDSHIFT = 0.725
ZMIN = 0.6
ZMAX = 0.8
RANDOM_MULTIPLIER = 25
P0 = 10_000.0

RHO_C = 1.14
RHO_EXP = 5.0
PDF_BASE = 0.25
SIGMA_V = 0.0
ATTACH_PARTICLE = True
RAND_GENERATOR = 1
PK_INTERP_LOG = True
INVERT_PHASE = False
BAO_ENHANCE = 0.0

OMEGA_M_NON_NEUTRINO = 0.3137721026737606
OMEGA_NU = 0.0014197664745152646

S_EDGES = np.arange(50.0, 550.0 + 10.0, 10.0, dtype="f8")
KMIN = 0.001
KMAX = 0.3001
DK = 0.002
PK_MESHSIZE = 256
PK_MESH_PAD = 400.0
PK_TAG = "ezmock_fixedamp_x10_mmin1p4e13_x25_fkpP010000"


def ensure_dirs() -> None:
    """创建本轮验证所需目录，不触碰其他 Task43 产物。"""
    for path in (
        MANIFEST_DIR,
        EZMOCK_CONFIG_DIR,
        RAWBOX_DIR,
        LIGHTCONE_DIR,
        RANDOM_DIR,
        FCFC_CONFIG_DIR,
        FCFC_PAIR_DIR,
        XI_DIR,
        PK_DIR,
        WINDOW_DIR,
        SUMMARY_DIR,
        LOG_DIR,
        PLOT_DIR,
    ):
        path.mkdir(parents=True, exist_ok=True)


def to_jsonable(value: Any) -> Any:
    """递归把 Path/NumPy 对象转换为 JSON 可序列化对象。"""
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
    """原子写入带缩进的 JSON 文件。"""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    tmp.write_text(
        json.dumps(to_jsonable(payload), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    tmp.replace(path)


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    """原子写入 JSONL manifest。"""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    with tmp.open("w", encoding="utf-8") as stream:
        for row in rows:
            stream.write(json.dumps(to_jsonable(row), sort_keys=True) + "\n")
    tmp.replace(path)


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    """读取非空 JSONL 行。"""
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as stream:
        for line in stream:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def atomic_savez(path: Path, **arrays: Any) -> None:
    """原子写入压缩 NPZ。"""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.{os.getpid()}.tmp.npz")
    try:
        np.savez_compressed(tmp, **arrays)
        tmp.replace(path)
    except Exception:
        tmp.unlink(missing_ok=True)
        raise


def row_paths(phase: str, seed: int) -> dict[str, str]:
    """返回一个 realization 的标准输入/输出路径字典。"""
    stem = f"ezmock_{phase}_seed{int(seed)}"
    return {
        "ezmock_config_path": str(EZMOCK_CONFIG_DIR / f"{stem}.conf"),
        "rawbox_catalog_path": str(RAWBOX_DIR / f"{stem}.dat"),
        "halo_catalog_path": str(LIGHTCONE_DIR / f"lightcone_{stem}_z0p6_0p8.npz"),
        "halo_metadata_path": str(LIGHTCONE_DIR / f"lightcone_{stem}_z0p6_0p8.json"),
        "random_catalog_path": str(RANDOM_DIR / f"random_{stem}_z0p6_0p8_x25.npz"),
        "random_metadata_path": str(RANDOM_DIR / f"random_{stem}_z0p6_0p8_x25.json"),
        "xi_path": str(XI_DIR / f"xi0_{stem}_z0p6_0p8_x25_s50_550_ds10_fcfc.npz"),
        "ezmock_log_path": str(LOG_DIR / f"ezmock_{phase}_seed{int(seed)}.log"),
    }

