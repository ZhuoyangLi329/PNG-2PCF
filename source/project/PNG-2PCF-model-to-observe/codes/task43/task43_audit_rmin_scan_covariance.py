#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""审计 rmin-scan 的 RR、raw jaxpower 与 RR-deconvolved covariance。

执行逻辑大纲：
1. 检查三阶段产品均严格对应 30--350 的 32-bin 网格。
2. 检查 RR 为正、covariance finite/symmetric/SPD 且没有 eigenvalue flooring。
3. 把新矩阵的 bins[2:32] 与权威旧 50--350 产品逐层桥接。
4. 检查 25-phase scatter metadata，并把关键数值原子写入 JSON audit。
"""

from __future__ import annotations

import argparse
import json
import os
import tempfile
from pathlib import Path
from typing import Any

import numpy as np


ROOT = Path("/pscratch/sd/l/lzy/PNG-2PCF-model-to-observe")
SCAN_ROOT = ROOT / "outputs/task43_outputs/rmin_scan"
MOVED_ROOT = ROOT / "plots/outputs/task43_outputs"
COV_DIR = SCAN_ROOT / "covariance"

RR_STEM = "task43_rr_smu_ph000_nran100k_seed20260702_s30_350_ds10_nmu20"
RAW_STEM = "jaxpower_covariance_lightcone_mmin1p4e13_ph000_fitcov_bessel_interp_smoothfftlog_mesh64_nran100k_ndata50k_pad400_win3600_ds2_k0001_3000_dk002_p1p0_s30_350_ds10"
FINAL_STEM = "jaxpower_covariance_lightcone_mmin1p4e13_ph000_fitcov_bessel_interp_smoothfftlog_rrdeconv_fkpNorm4p8925e10_mesh64_nran100k_ndata50k_pad400_win3600_ds2_k0001_3000_dk002_p1p0_s30_350_ds10"

DEFAULT_OLD_RR = MOVED_ROOT / "summary/task43_rr_smu_window_smoke_ph000_nran100k_s50_350_ds10_nmu20_midpoint.npz"
DEFAULT_OLD_RAW = MOVED_ROOT / "summary/jaxpower_covariance_lightcone_mmin1p4e13_ph000_fitcov_bessel_interp_smoothfftlog_mesh64_nran100k_ndata50k_pad400_win3600_ds2_k0001_3000_dk002_p1p0_s50_350_ds10.npz"
DEFAULT_OLD_FINAL = MOVED_ROOT / "summary/jaxpower_covariance_lightcone_mmin1p4e13_ph000_fitcov_bessel_interp_smoothfftlog_rrdeconv_mesh64_nran100k_ndata50k_pad400_win3600_ds2_k0001_3000_dk002_p1p0_s50_350_ds10.npz"


def atomic_json(path: Path, payload: dict[str, Any]) -> None:
    """原子写入 audit JSON，避免中断留下半文件。"""

    path.parent.mkdir(parents=True, exist_ok=True)
    handle, temporary = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(handle, "w", encoding="utf-8") as stream:
            json.dump(payload, stream, indent=2, sort_keys=True)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    except Exception:
        Path(temporary).unlink(missing_ok=True)
        raise


def relative_frobenius(left: np.ndarray, right: np.ndarray) -> float:
    """返回两个同 shape 数组的相对 Frobenius 差。"""

    denominator = max(float(np.linalg.norm(right)), np.finfo("f8").tiny)
    return float(np.linalg.norm(left - right) / denominator)


def covariance_diagnostics(matrix: np.ndarray, expected_size: int) -> dict[str, float]:
    """验证 covariance 并返回 eigenvalue 与条件数诊断。

    参数：matrix 为待审计矩阵，expected_size 为预期维数。
    返回：包含最小/最大特征值与 condition number 的字典。
    """

    value = np.asarray(matrix, dtype="f8")
    if value.shape != (expected_size, expected_size) or not np.all(np.isfinite(value)):
        raise RuntimeError(f"covariance shape/finite 错误：{value.shape}")
    if not np.allclose(value, value.T, rtol=1.0e-11, atol=1.0e-15):
        raise RuntimeError("covariance 不对称")
    eig = np.linalg.eigvalsh(0.5 * (value + value.T))
    condition = float(np.linalg.cond(value))
    if eig[0] <= 0.0 or not np.isfinite(condition) or condition >= 1.0e12:
        raise RuntimeError(f"covariance 非 SPD 或病态：eigmin={eig[0]}, cond={condition}")
    return {"eig_min": float(eig[0]), "eig_max": float(eig[-1]), "condition_number": condition}


def main() -> None:
    """读取三阶段新旧产品，执行全部 bridge 与数值 gate。"""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rr", type=Path, default=COV_DIR / f"{RR_STEM}.npz")
    parser.add_argument("--raw", type=Path, default=COV_DIR / f"{RAW_STEM}.npz")
    parser.add_argument("--final", type=Path, default=COV_DIR / f"{FINAL_STEM}.npz")
    parser.add_argument("--old-rr", type=Path, default=DEFAULT_OLD_RR)
    parser.add_argument("--old-raw", type=Path, default=DEFAULT_OLD_RAW)
    parser.add_argument("--old-final", type=Path, default=DEFAULT_OLD_FINAL)
    parser.add_argument("--output", type=Path, default=SCAN_ROOT / "audits/task43_rmin_scan_covariance_audit.json")
    args = parser.parse_args()

    for path in (args.rr, args.raw, args.final, args.old_rr, args.old_raw, args.old_final):
        if not path.is_file() or path.stat().st_size == 0:
            raise FileNotFoundError(path)

    new_edges = np.arange(30.0, 351.0, 10.0)
    old_edges = np.arange(50.0, 351.0, 10.0)
    bridges: dict[str, float] = {}
    with np.load(args.rr, allow_pickle=False) as new, np.load(args.old_rr, allow_pickle=False) as old:
        if not np.array_equal(np.asarray(new["s_edges"], dtype="f8"), new_edges):
            raise RuntimeError("新 RR edge 不是 30--350")
        if not np.array_equal(np.asarray(old["s_edges"], dtype="f8"), old_edges):
            raise RuntimeError("旧 RR edge 不是 50--350")
        counts = np.asarray(new["rr_counts"], dtype="f8")
        norm = np.asarray(new["rr_norm"], dtype="f8")
        if counts.shape != (32, 20) or norm.shape != counts.shape or np.any(np.sum(counts, axis=1) <= 0.0):
            raise RuntimeError("新 RR shape 或正定性错误")
        bridges["rr_counts"] = relative_frobenius(counts[2:], np.asarray(old["rr_counts"], dtype="f8"))
        bridges["rr_norm"] = relative_frobenius(norm[2:], np.asarray(old["rr_norm"], dtype="f8"))

    with np.load(args.raw, allow_pickle=False) as new, np.load(args.old_raw, allow_pickle=False) as old:
        if not np.array_equal(np.asarray(new["s_edges"], dtype="f8"), new_edges):
            raise RuntimeError("raw covariance edge 不是 30--350")
        raw_cov = np.asarray(new["covariance_single_realization"], dtype="f8")
        raw_diag = covariance_diagnostics(raw_cov, 32)
        bridges["raw_covariance"] = relative_frobenius(
            raw_cov[2:, 2:], np.asarray(old["covariance_single_realization"], dtype="f8")
        )
        bridges["raw_projection"] = relative_frobenius(
            np.asarray(new["projection_matrix"], dtype="f8")[2:],
            np.asarray(old["projection_matrix"], dtype="f8"),
        )

    with np.load(args.final, allow_pickle=False) as new, np.load(args.old_final, allow_pickle=False) as old:
        if not np.array_equal(np.asarray(new["s_edges"], dtype="f8"), new_edges):
            raise RuntimeError("final covariance edge 不是 30--350")
        final_cov = np.asarray(new["covariance_single_realization"], dtype="f8")
        final_diag = covariance_diagnostics(final_cov, 32)
        window = np.asarray(new["rr_window_matrix"], dtype="f8")
        if window.shape != (32, 32) or np.any(np.diag(window) <= 0.0):
            raise RuntimeError("RR deconvolution window shape/对角错误")
        offdiag = float(np.max(np.abs(window - np.diag(np.diag(window)))))
        if offdiag > 1.0e-14:
            raise RuntimeError(f"ell=0 RR window 出现非对角项：{offdiag}")
        bridges["final_covariance"] = relative_frobenius(
            final_cov[2:, 2:], np.asarray(old["covariance_single_realization"], dtype="f8")
        )
        bridges["rr_window"] = relative_frobenius(
            window[2:, 2:], np.asarray(old["rr_window_matrix"], dtype="f8")
        )
        meta = json.loads(str(np.asarray(new["meta_json"]).item()))

    raw_meta = json.loads(args.raw.with_suffix(".json").read_text(encoding="utf-8"))
    final_meta = json.loads(args.final.with_suffix(".json").read_text(encoding="utf-8"))
    raw_floor = int(raw_meta.get("spd", {}).get("n_floored", -1))
    final_floor = int(final_meta.get("spd", {}).get("covariance_single_realization", {}).get("n_floored", -1))
    scatter = meta.get("scatter_comparison") or final_meta.get("scatter_comparison")
    if raw_floor != 0 or final_floor != 0:
        raise RuntimeError(f"covariance 使用了 eigenvalue flooring：raw={raw_floor}, final={final_floor}")
    if not isinstance(scatter, dict) or scatter.get("available") is not True or int(scatter.get("nreal", -1)) != 25:
        raise RuntimeError(f"25-phase scatter audit 缺失：{scatter}")
    if any((not np.isfinite(value)) or value > 1.0e-8 for value in bridges.values()):
        raise RuntimeError(f"旧 50--350 bridge 失败：{bridges}")

    payload = {
        "status": "pass",
        "task": "task43_audit_rmin_scan_covariance",
        "contract": {"s_edges": new_edges.tolist(), "nbins": 32, "covariance_key": "covariance_single_realization"},
        "raw_diagnostics": raw_diag,
        "final_diagnostics": final_diag,
        "rr_window_offdiag_absmax": offdiag,
        "eigenvalue_flooring": {"raw": raw_floor, "final": final_floor},
        "bridge_relative_frobenius": bridges,
        "scatter_comparison": scatter,
        "paths": {"rr": str(args.rr), "raw": str(args.raw), "final": str(args.final)},
    }
    atomic_json(args.output, payload)
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
