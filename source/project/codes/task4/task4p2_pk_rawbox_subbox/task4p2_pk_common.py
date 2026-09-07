#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Task4.2 P(k): Task4.2 rawbox/subbox P(k) 公共工具。

代码大纲：
1. 定义 Task4.2 P(k) 的统一路径、k-range、参数口径。
2. 读取当前 Task47 r50 2PCF 主线使用的 realization/subbox 输入。
3. 提供 rawbox P(k) 文件读取、subbox catalog/random 生成和 covariance 修正。
4. 提供线性 PNG P(k) basis 与 bin-average/window-forward 的基础函数。

本文件只放轻量工具，不直接跑重任务。
"""

from __future__ import annotations

import hashlib
import json
import math
import os
from pathlib import Path
from typing import Any

import numpy as np


PROJECT_ROOT = Path("/pscratch/sd/l/lzy/PNG-2PCF-model-to-observe")
OUTPUT_ROOT = PROJECT_ROOT / "outputs" / "task4_outputs" / "task4p2_pk_rawbox_subbox"
PLOT_ROOT = PROJECT_ROOT / "plots" / "task4" / "task4p2_pk_rawbox_subbox"

TASK47_ROOT = PROJECT_ROOT / "outputs" / "task47_outputs" / "fnl100_norsd_r50"
TASK22_SUBBOX_ROOT = PROJECT_ROOT / "outputs" / "task22_outputs" / "cubic_subboxes_fnl100_norsd"
RAWBOX_PK_DIR = PROJECT_ROOT / "outputs" / "task18_outputs" / "fnl100_box_realspace_pk" / "pk"

FNL_TAG = "fnl100"
LSUB_LIST = (1500, 1000, 750)
PARAM_NAMES = ("fnl_loc", "b1", "sn0")

P_FIXED = 1.2
SIGMAS_FIXED = 0.0
DELTA_C = 1.686
L_BOX = 3000.0
V_BOX = L_BOX**3
K_FUND = 2.0 * math.pi / L_BOX
K_MIN_MODEL = K_FUND
K_MAX_FIT = 0.08

# Task4.2 使用的 FastPM-L3 halo catalog 是单一 z=1 snapshot，而不是
# z=0.4--1.0 cut-sky 几何标签所暗示的 lightcone effective redshift。
# 这里集中定义 snapshot redshift，避免 P(k) 与 2PCF 各自硬编码后再次漂移。
# 原始模拟说明：FastPM-L3 从 z=99 演化到 z=1，包含 100 组 fNL=0/100
# matched realizations（MNRAS 543, 2078；FastPM-L3 simulation description）。
FASTPM_SNAPSHOT_Z = 1.0

# 沿用已有 rawbox P(k) 文件的 bin：0.002-0.005, 0.005-0.008, ...
K_EDGE_START = 0.002
K_BIN_DK = 0.003


def ensure_dir(path: Path) -> None:
    """创建目录。"""
    path.mkdir(parents=True, exist_ok=True)


def to_jsonable(obj: Any) -> Any:
    """把 numpy/Path 等对象递归转成 JSON 可写对象。"""
    if isinstance(obj, (str, int, float, bool)) or obj is None:
        return obj
    if isinstance(obj, Path):
        return str(obj)
    if isinstance(obj, np.generic):
        return obj.item()
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, dict):
        return {str(key): to_jsonable(value) for key, value in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [to_jsonable(value) for value in obj]
    return str(obj)


def write_json(path: Path, payload: dict[str, Any]) -> None:
    """写 JSON summary。"""
    ensure_dir(path.parent)
    path.write_text(json.dumps(to_jsonable(payload), indent=2, sort_keys=True), encoding="utf-8")


def atomic_savez(path: Path, **arrays: Any) -> None:
    """先写临时 npz，再原子替换目标文件。"""
    ensure_dir(path.parent)
    tmp = path.with_name(f".{path.name}.{os.getpid()}.tmp.npz")
    try:
        np.savez_compressed(tmp, **arrays)
        tmp.replace(path)
    except Exception:
        try:
            tmp.unlink()
        except FileNotFoundError:
            pass
        raise


def realization_label(realization: int) -> str:
    """把 realization 编号格式化成 N002。"""
    return f"N{int(realization):03d}"


def subbox_label(subbox_id: int) -> str:
    """把 subbox 编号格式化成 B000。"""
    return f"B{int(subbox_id):03d}"


def deterministic_seed(*parts: object) -> int:
    """稳定随机种子，低 63 bit，便于写入 int64。"""
    payload = "|".join(str(part) for part in parts).encode("utf-8")
    digest = hashlib.sha256(payload).digest()
    return int.from_bytes(digest[:8], "little", signed=False) >> 1


def task47_rawbox_npz() -> Path:
    """当前 Task4.2 主线 rawbox 2PCF summary npz。"""
    return TASK47_ROOT / "rawbox" / "task47_fnl100_rawbox_mean_xi_s50_350.npz"


def task47_subbox_npz(lsub: int) -> Path:
    """当前 Task4.2 主线 subbox 2PCF summary npz。"""
    return TASK47_ROOT / "subboxes" / f"L{int(lsub)}" / "summary" / f"mean_xi_L{int(lsub)}_r20_individual.npz"


def subbox_catalog_dir(lsub: int) -> Path:
    """Task22 已切好的非周期 subbox catalog 目录。"""
    return TASK22_SUBBOX_ROOT / f"L{int(lsub)}" / "subboxes"


def subbox_catalog_path(lsub: int, realization: int, subbox_id: int) -> Path:
    """单个 subbox catalog 路径。"""
    return (
        subbox_catalog_dir(lsub)
        / f"subbox_{FNL_TAG}_{realization_label(realization)}_L{int(lsub)}_{subbox_label(subbox_id)}.npz"
    )


def collect_subbox_catalogs(lsub: int, limit: int | None = None) -> list[Path]:
    """列出某个 L 的所有 subbox catalog。"""
    paths = sorted(subbox_catalog_dir(lsub).glob(f"subbox_{FNL_TAG}_N*_L{int(lsub)}_B*.npz"))
    if limit is not None:
        paths = paths[: int(limit)]
    if not paths:
        raise FileNotFoundError(f"没有找到 subbox catalog: {subbox_catalog_dir(lsub)}")
    return paths


def parse_subbox_path(path: Path) -> tuple[int, int]:
    """从 subbox 文件名解析 realization 和 subbox_id。"""
    stem = path.stem
    realization = int(stem.split("_N")[-1].split("_")[0])
    subbox_id = int(stem.split("_B")[-1])
    return realization, subbox_id


def load_subbox_catalog(path: Path) -> dict[str, Any]:
    """读取一个 subbox catalog 的局部坐标和元数据。"""
    with np.load(path) as data:
        return {
            "pos_local": np.asarray(data["pos_local"], dtype="f8"),
            "ndata": int(np.asarray(data["ndata"]).item()),
            "realization": int(np.asarray(data["realization"]).item()),
            "subbox_id": int(np.asarray(data["subbox_id"]).item()),
            "lsub": int(round(float(np.asarray(data["lsub"]).item()))),
            "origin_global": np.asarray(data["origin_global"], dtype="f8"),
        }


def build_uniform_random(lsub: int, realization: int, subbox_id: int, ndata: int, random_multiplier: float) -> tuple[np.ndarray, int, int]:
    """
    为非周期 cube 生成 uniform Cartesian random。

    random_multiplier 是 P(k) 测量专用参数；默认脚本里可设得比 2PCF 的 20x 小，
    但权重始终为 1，不引入 FKP science weight。
    """
    nrandom = int(round(float(ndata) * float(random_multiplier)))
    seed = deterministic_seed("task4p2", "pk_uniform_random", FNL_TAG, lsub, realization, subbox_id, f"{random_multiplier:.6g}")
    rng = np.random.default_rng(seed)
    pos = rng.random((nrandom, 3), dtype=np.float32) * np.float32(lsub)
    return np.asarray(pos, dtype="f8"), nrandom, seed


def rawbox_realizations_from_task47() -> np.ndarray:
    """读取当前 rawbox 2PCF 主线实际使用的 realizations。"""
    with np.load(task47_rawbox_npz()) as data:
        return np.asarray(data["realizations"], dtype="i8")


def rawbox_pk_path(realization: int) -> Path:
    """已有 no-RSD rawbox P(k) 文件路径。"""
    return RAWBOX_PK_DIR / f"pk_realspace_{FNL_TAG}_{realization_label(realization)}.dat"


def load_rawbox_pk_file(path: Path) -> dict[str, np.ndarray]:
    """读取一个 rawbox P(k) ASCII 文件。"""
    data = np.loadtxt(path, comments="#", dtype="f8")
    return {
        "kcen": np.asarray(data[:, 0], dtype="f8"),
        "kmin": np.asarray(data[:, 1], dtype="f8"),
        "kmax": np.asarray(data[:, 2], dtype="f8"),
        "kavg": np.asarray(data[:, 3], dtype="f8"),
        "nmode": np.asarray(data[:, 4], dtype="f8"),
        "pk0": np.asarray(data[:, 5], dtype="f8"),
    }


def default_k_edges() -> np.ndarray:
    """返回与 rawbox P(k) 对齐的 k-bin edges，最后一个 edge 为 0.08。"""
    nbin = int(round((K_MAX_FIT - K_EDGE_START) / K_BIN_DK))
    edges = K_EDGE_START + K_BIN_DK * np.arange(nbin + 1, dtype="f8")
    if not np.isclose(edges[-1], K_MAX_FIT):
        raise RuntimeError(f"k edges 未落到 kmax={K_MAX_FIT}: last={edges[-1]}")
    return edges


def column_edges(edges_1d: np.ndarray) -> np.ndarray:
    """jaxpower 使用的二列边界格式。"""
    edges_1d = np.asarray(edges_1d, dtype="f8")
    return np.column_stack([edges_1d[:-1], edges_1d[1:]])


def select_fit_bins(kmin: np.ndarray, kmax: np.ndarray) -> np.ndarray:
    """选择与 parent-box k>=2pi/Lbox mode 有交集且 kmax<=0.08 的 fit bins。"""
    kmin = np.asarray(kmin, dtype="f8")
    kmax = np.asarray(kmax, dtype="f8")
    return np.nonzero((kmax > K_MIN_MODEL + 1.0e-15) & (kmax <= K_MAX_FIT + 1.0e-12))[0].astype("i8")


def covariance_corrections(nmock: int, ndata: int, nparams: int = len(PARAM_NAMES)) -> dict[str, float]:
    """计算 Hartlap precision correction 与 Percival error correction。"""
    if nmock <= ndata + 4:
        raise ValueError(f"Nmock={nmock} 对 Ndata={ndata} 太少，无法稳定估计 covariance correction")
    hartlap = (nmock - ndata - 2.0) / (nmock - 1.0)
    a = 2.0 / ((nmock - ndata - 1.0) * (nmock - ndata - 4.0))
    b = (nmock - ndata - 2.0) / ((nmock - ndata - 1.0) * (nmock - ndata - 4.0))
    m1 = (1.0 + b * (ndata - nparams)) / (1.0 + a + b * (nparams + 1.0))
    return {
        "hartlap": float(hartlap),
        "percival_m1_variance": float(m1),
        "percival_error_factor": float(math.sqrt(m1)),
        "A": float(a),
        "B": float(b),
    }


def hartlap_precision(cov: np.ndarray, nmock: int, nparams: int = len(PARAM_NAMES)) -> tuple[np.ndarray, dict[str, Any]]:
    """返回 Hartlap 修正后的 precision 和 covariance 元数据。"""
    cov = np.asarray(cov, dtype="f8")
    corrections = covariance_corrections(nmock=int(nmock), ndata=int(cov.shape[0]), nparams=int(nparams))
    precision = corrections["hartlap"] * np.linalg.pinv(cov, rcond=1.0e-10)
    meta = {
        **corrections,
        "nmock": int(nmock),
        "ndata": int(cov.shape[0]),
        "nparams": int(nparams),
        "condition_number": float(np.linalg.cond(cov)),
        "used_covariance_divided_by_nmock": False,
    }
    return precision, meta


def interp_logk(k_query: np.ndarray, k_base: np.ndarray, y_base: np.ndarray) -> np.ndarray:
    """在 log10(k) 上插值；k_query 必须为正。"""
    return np.interp(np.log10(k_query), np.log10(k_base), y_base)


def evaluate_pk_basis(k: np.ndarray, template: dict[str, np.ndarray], *, kmin_model: float = K_MIN_MODEL) -> dict[str, np.ndarray]:
    """
    在给定 k 上评估线性 PNG P(k) 四个 basis。

    返回 basis:
    - pkdd
    - gamma_pkdd, gamma=2 delta_c alpha(k)
    - gamma2_pkdd
    - sn0，常数 1
    """
    k = np.asarray(k, dtype="f8")
    out = {
        "pkdd": np.zeros_like(k),
        "gamma_pkdd": np.zeros_like(k),
        "gamma2_pkdd": np.zeros_like(k),
        "sn0": np.ones_like(k),
    }
    mask = k >= float(kmin_model) - 1.0e-15
    if np.any(mask):
        alpha = interp_logk(k[mask], template["k"], template["alpha"])
        pkdd = interp_logk(k[mask], template["k"], template["pk_dd"])
        gamma = 2.0 * DELTA_C * alpha
        out["pkdd"][mask] = pkdd
        out["gamma_pkdd"][mask] = gamma * pkdd
        out["gamma2_pkdd"][mask] = gamma * gamma * pkdd
    return out


def pk_coefficients(fnl_loc: float, b1: float, sn0: float, p_fixed: float = P_FIXED) -> tuple[float, float, float, float]:
    """返回四个 P(k) basis 的系数。"""
    fnl_loc = float(fnl_loc)
    b1 = float(b1)
    return (
        b1**2,
        2.0 * b1 * (b1 - float(p_fixed)) * fnl_loc,
        (b1 - float(p_fixed)) ** 2 * fnl_loc**2,
        float(sn0),
    )


def combine_pk_basis(basis: dict[str, np.ndarray], fnl_loc: float, b1: float, sn0: float, p_fixed: float = P_FIXED) -> np.ndarray:
    """用四个 basis 快速组合 tracer P(k)。"""
    c0, c1, c2, csn = pk_coefficients(fnl_loc, b1, sn0, p_fixed=p_fixed)
    return c0 * basis["pkdd"] + c1 * basis["gamma_pkdd"] + c2 * basis["gamma2_pkdd"] + csn * basis["sn0"]


def gq_enumerate(qmax: int) -> np.ndarray:
    """枚举 parent periodic box 低 k 离散壳层简并度。"""
    nmax = int(np.ceil(np.sqrt(qmax))) + 1
    gq = np.zeros(qmax + 1, dtype=np.int64)
    for nx in range(-nmax, nmax + 1):
        for ny in range(-nmax, nmax + 1):
            for nz in range(-nmax, nmax + 1):
                if nx == 0 and ny == 0 and nz == 0:
                    continue
                q = nx * nx + ny * ny + nz * nz
                if q <= qmax:
                    gq[q] += 1
    return gq


def build_parent_binavg_matrix(kmin: np.ndarray, kmax: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """
    为 rawbox P(k) fit 构造 parent box 离散 bin-average 矩阵。

    返回：
    - unique_k: 所有进入拟合的 parent-box 离散 k
    - matrix: shape=(nbin, nk)，每行对相应 bin 做 mode-count average
    """
    kmin = np.asarray(kmin, dtype="f8")
    kmax = np.asarray(kmax, dtype="f8")
    qmax = int(np.floor((float(np.max(kmax)) / K_FUND) ** 2)) + 1
    gq = gq_enumerate(qmax)
    qnz = np.nonzero(gq[1:])[0] + 1
    kvals = K_FUND * np.sqrt(qnz.astype("f8"))
    weights = gq[qnz].astype("f8")
    used_k: list[float] = []
    rows: list[np.ndarray] = []
    for lo, hi in zip(kmin, kmax, strict=True):
        mask = (kvals >= lo - 1.0e-15) & (kvals < hi - 1.0e-15)
        if not np.any(mask):
            raise RuntimeError(f"bin [{lo},{hi}) 没有 parent-box mode")
        used_k.extend(kvals[mask].tolist())
    unique_k = np.unique(np.asarray(used_k, dtype="f8"))
    index = {float(k): i for i, k in enumerate(unique_k)}
    for lo, hi in zip(kmin, kmax, strict=True):
        mask = (kvals >= lo - 1.0e-15) & (kvals < hi - 1.0e-15)
        row = np.zeros(unique_k.size, dtype="f8")
        norm = float(np.sum(weights[mask]))
        for k, w in zip(kvals[mask], weights[mask], strict=True):
            row[index[float(k)]] += float(w) / norm
        rows.append(row)
    return unique_k, np.vstack(rows)
