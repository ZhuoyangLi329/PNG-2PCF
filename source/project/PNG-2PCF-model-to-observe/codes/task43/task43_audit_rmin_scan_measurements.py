#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""审计 Task43 rmin-scan 2PCF，并桥接既有 50--350 测量。

执行逻辑大纲：
1. 从新旧 manifest 逐 phase 定位 NPZ，不递归扫描大型输出目录。
2. 检查新测量严格为 30--350 的 32 bins、weighted LS、P0=10000。
3. 将新 bins 2:32 与旧 rmax-scan bins 0:30 逐数组比较。
4. 用旧 jaxpower covariance sigma 把 xi 差异标准化，并原子写 audit JSON。
"""

from __future__ import annotations

import argparse
import json
import os
import tempfile
from pathlib import Path
from typing import Any

import numpy as np

from task43_fkp_zeff import path_with_weight_tag


ROOT = Path("/pscratch/sd/l/lzy/PNG-2PCF-model-to-observe")
NEW_ROOT = ROOT / "outputs/task43_outputs/rmin_scan"
MOVED_ROOT = ROOT / "plots/outputs/task43_outputs"
DEFAULT_MANIFEST = NEW_ROOT / "manifests/task43_rmin_scan_mmin1p4e13_x25.jsonl"
DEFAULT_OLD_MANIFEST = MOVED_ROOT / "rmax_scan/manifests/task43_rmax_scan_mmin1p4e13_x25.jsonl"
DEFAULT_OLD_COV = MOVED_ROOT / "summary/jaxpower_covariance_lightcone_mmin1p4e13_ph000_fitcov_bessel_interp_smoothfftlog_rrdeconv_mesh64_nran100k_ndata50k_pad400_win3600_ds2_k0001_3000_dk002_p1p0_s50_350_ds10.npz"
DEFAULT_OUTPUT = NEW_ROOT / "audits/task43_rmin_scan_measurement_audit.json"


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    """读取 JSONL 并返回有序字典列表。"""

    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def atomic_json(path: Path, payload: dict[str, Any]) -> None:
    """把小型 JSON audit 原子写到目标路径。"""

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


def measurement_path(row: dict[str, Any]) -> Path:
    """按统一 P0/tag 规则把 manifest 的基础输出名变成实测文件名。"""

    return path_with_weight_tag(Path(str(row["xi_path"])), p0=10000.0, output_tag="fkpP010000")


def archived_old_measurement_path(row: dict[str, Any]) -> Path:
    """把旧 manifest 中迁移前的逻辑路径映射到当前只读归档目录。

    旧 manifest 是不可变的历史产物，其中 ``xi_path`` 仍记录迁移前的
    ``outputs/task43_outputs`` 根目录；这里只复用其文件名，并显式拼到当前
    ``plots/outputs`` 归档根，避免修改旧 manifest 或复制数百 GB 历史数据。
    """

    logical_path = measurement_path(row)
    return MOVED_ROOT / "rmax_scan/xi_cucount" / logical_path.name


def main() -> None:
    """执行逐 phase 合约检查、旧区间 bridge 和汇总审计。"""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--old-manifest", type=Path, default=DEFAULT_OLD_MANIFEST)
    parser.add_argument("--old-covariance", type=Path, default=DEFAULT_OLD_COV)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--indices", type=str, default="0-24", help="例如 0 或 0-24。")
    parser.add_argument("--require-all", action="store_true")
    args = parser.parse_args()

    new_rows = read_jsonl(args.manifest)
    old_rows = read_jsonl(args.old_manifest)
    if len(new_rows) != 25 or len(old_rows) != 25:
        raise RuntimeError("新旧 manifest 都必须包含 25 个 phase")
    if "-" in args.indices:
        lo, hi = (int(value) for value in args.indices.split("-", 1))
        indices = list(range(lo, hi + 1))
    else:
        indices = [int(args.indices)]
    if args.require_all and indices != list(range(25)):
        raise RuntimeError("--require-all 要求 --indices 0-24")

    with np.load(args.old_covariance, allow_pickle=False) as cov_data:
        old_cov = np.asarray(cov_data["covariance_single_realization"], dtype="f8")
        old_edges = np.asarray(cov_data["s_edges"], dtype="f8")
    if old_cov.shape != (30, 30) or not np.array_equal(old_edges, np.arange(50.0, 351.0, 10.0)):
        raise RuntimeError("旧 covariance 不是权威的 50--350 30-bin 产品")
    sigma = np.sqrt(np.diag(old_cov))

    expected_edges = np.arange(30.0, 351.0, 10.0)
    expected_centers = 0.5 * (expected_edges[:-1] + expected_edges[1:])
    rows_audit: list[dict[str, Any]] = []
    all_standardized: list[np.ndarray] = []
    missing: list[int] = []
    for index in indices:
        new_row, old_row = new_rows[index], old_rows[index]
        if new_row["phase"] != old_row["phase"] or new_row["phase"] != f"ph{index:03d}":
            raise RuntimeError(f"phase 顺序错误：index={index}")
        new_path = measurement_path(new_row)
        old_path = archived_old_measurement_path(old_row)
        if not new_path.is_file():
            missing.append(index)
            continue
        if not old_path.is_file():
            raise FileNotFoundError(old_path)

        with np.load(new_path, allow_pickle=False) as new, np.load(old_path, allow_pickle=False) as old:
            s = np.asarray(new["s"], dtype="f8")
            edges = np.asarray(new["s_edges"], dtype="f8")
            arrays = {key: np.asarray(new[key], dtype="f8") for key in ("xi0", "DD", "DR", "RR")}
            old_arrays = {key: np.asarray(old[key], dtype="f8")[:30] for key in ("xi0", "DD", "DR", "RR")}
            metadata_ok = (
                str(np.asarray(new["engine"]).item()) == "cucount_jax"
                and str(np.asarray(new["estimator"]).item()) == "landy_szalay"
                and np.isclose(float(np.asarray(new["p0"]).item()), 10000.0, rtol=0.0, atol=1.0e-10)
                and str(np.asarray(new["phase"]).item()) == f"ph{index:03d}"
            )
        if not np.array_equal(edges, expected_edges) or not np.array_equal(s, expected_centers):
            raise RuntimeError(f"{new_path} 的径向坐标不满足 32-bin contract")
        if not metadata_ok or any(value.shape != (32,) for value in arrays.values()):
            raise RuntimeError(f"{new_path} 的 metadata 或数组 shape 错误")
        if not all(np.all(np.isfinite(value)) for value in arrays.values()) or np.any(arrays["RR"] <= 0.0):
            raise RuntimeError(f"{new_path} 含非有限值或非正 RR")

        delta = arrays["xi0"][2:] - old_arrays["xi0"]
        standardized = delta / sigma
        all_standardized.append(standardized)
        rows_audit.append(
            {
                "index": index,
                "phase": f"ph{index:03d}",
                "new_path": str(new_path),
                "old_path": str(old_path),
                "bridge_xi_max_abs_sigma": float(np.max(np.abs(standardized))),
                "bridge_xi_rms_sigma": float(np.sqrt(np.mean(standardized**2))),
                "bridge_pair_arrays_max_abs": {
                    key: float(np.max(np.abs(arrays[key][2:] - old_arrays[key]))) for key in ("DD", "DR", "RR")
                },
            }
        )

    if args.require_all and missing:
        raise RuntimeError(f"缺少新测量 indices={missing}")
    if not rows_audit:
        raise RuntimeError("没有可审计的新测量")
    stacked = np.vstack(all_standardized)
    max_abs = float(np.max(np.abs(stacked)))
    rms = float(np.sqrt(np.mean(stacked**2)))
    payload = {
        "status": "pass" if max_abs < 0.05 else "fail",
        "task": "task43_audit_rmin_scan_measurements",
        "indices_requested": indices,
        "nrows_audited": len(rows_audit),
        "missing_indices": missing,
        "measurement_contract": {"s_edges": expected_edges.tolist(), "nbins": 32, "p0": 10000.0},
        "bridge_contract": "new bins[2:32] versus old rmax-scan bins[0:30], normalized by old jaxpower sigma",
        "bridge_gate_max_abs_sigma": 0.05,
        "bridge_xi_max_abs_sigma": max_abs,
        "bridge_xi_rms_sigma": rms,
        "rows": rows_audit,
    }
    atomic_json(args.output, payload)
    if payload["status"] != "pass":
        raise RuntimeError(f"2PCF bridge 未通过：max_abs_sigma={max_abs}")
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
