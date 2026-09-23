#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""比较 rmin-scan jaxpower covariance 的 kmax=3 与 kmax=4 A/B。

执行逻辑大纲：
1. 读取两份 32-bin RR-deconvolved covariance，检查坐标、有限性和 SPD。
2. 比较新增中心 35/45 Mpc/h 的 sigma、完整 correlation 和 covariance。
3. 读取两份 Fisher JSON，比较三个 rmin case 的 marginalized sigma(b1)。
4. 按预注册阈值写 audit JSON；失败时明确报错，不静默选择某一版本。
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
SCAN = ROOT / "outputs/task43_outputs/rmin_scan"
COV = SCAN / "covariance"
BASE_STEM = "jaxpower_covariance_lightcone_mmin1p4e13_ph000_fitcov_bessel_interp_smoothfftlog_rrdeconv_fkpNorm4p8925e10_mesh64_nran100k_ndata50k_pad400_win3600_ds2_k0001_3000_dk002_p1p0_s30_350_ds10"
TEST_STEM = "jaxpower_covariance_lightcone_mmin1p4e13_ph000_fitcov_bessel_interp_smoothfftlog_rrdeconv_fkpNorm4p8925e10_mesh64_nran100k_ndata50k_pad400_win3600_ds2_k0001_4000_dk002_p1p0_s30_350_ds10"


def atomic_json(path: Path, payload: dict[str, Any]) -> None:
    """原子写入小型审计 JSON。"""

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


def load_covariance(path: Path) -> tuple[np.ndarray, np.ndarray]:
    """读取并验证 32-bin covariance，返回 covariance 与 correlation。"""

    with np.load(path, allow_pickle=False) as data:
        edges = np.asarray(data["s_edges"], dtype="f8")
        covariance = np.asarray(data["covariance_single_realization"], dtype="f8")
    if not np.array_equal(edges, np.arange(30.0, 351.0, 10.0)) or covariance.shape != (32, 32):
        raise RuntimeError(f"covariance contract 错误：{path}")
    covariance = 0.5 * (covariance + covariance.T)
    eig = np.linalg.eigvalsh(covariance)
    if eig[0] <= 0.0 or not np.all(np.isfinite(covariance)):
        raise RuntimeError(f"covariance 非 finite/SPD：{path}, eigmin={eig[0]}")
    sigma = np.sqrt(np.diag(covariance))
    correlation = covariance / np.outer(sigma, sigma)
    return covariance, correlation


def main() -> None:
    """执行 covariance/Fisher support A/B 并应用固定阈值。"""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-covariance", type=Path, default=COV / f"{BASE_STEM}.npz")
    parser.add_argument("--test-covariance", type=Path, default=COV / f"{TEST_STEM}.npz")
    parser.add_argument("--base-fisher", type=Path, default=SCAN / "fisher/task43_rmin_scan_fisher_information.json")
    parser.add_argument("--test-fisher", type=Path, default=SCAN / "fisher/task43_rmin_scan_fisher_information_kmax4.json")
    parser.add_argument("--output", type=Path, default=SCAN / "audits/task43_rmin_scan_covariance_kmax3_vs4.json")
    args = parser.parse_args()

    base_cov, base_corr = load_covariance(args.base_covariance)
    test_cov, test_corr = load_covariance(args.test_covariance)
    sigma_base = np.sqrt(np.diag(base_cov))
    sigma_test = np.sqrt(np.diag(test_cov))
    sigma_ratio = sigma_test / sigma_base
    relative_covariance = float(np.linalg.norm(test_cov - base_cov) / np.linalg.norm(base_cov))
    corr_max = float(np.max(np.abs(test_corr - base_corr)))

    base_fisher = json.loads(args.base_fisher.read_text(encoding="utf-8"))
    test_fisher = json.loads(args.test_fisher.read_text(encoding="utf-8"))
    fisher_ratio: dict[str, float] = {}
    for rmin in (30, 40, 50):
        base = float(base_fisher["xi"]["cases"][str(rmin)]["sigma_marginalized"]["b1"])
        test = float(test_fisher["xi"]["cases"][str(rmin)]["sigma_marginalized"]["b1"])
        fisher_ratio[str(rmin)] = test / base

    gates = {
        "new_bin_sigma_within_2percent": bool(np.max(np.abs(sigma_ratio[:2] - 1.0)) < 0.02),
        "correlation_absmax_below_0p02": bool(corr_max < 0.02),
        "fisher_b1_all_rmin_within_2percent": bool(
            max(abs(value - 1.0) for value in fisher_ratio.values()) < 0.02
        ),
    }
    payload = {
        "status": "pass" if all(gates.values()) else "fail",
        "task": "task43_compare_rmin_covariance_support",
        "comparison": "same covariance setup; only theory covariance kmax changes 3.0001 -> 4.0001 h/Mpc",
        "new_bin_sigma_ratio_k4_over_k3": {"s35": float(sigma_ratio[0]), "s45": float(sigma_ratio[1])},
        "sigma_ratio_all_min": float(np.min(sigma_ratio)),
        "sigma_ratio_all_max": float(np.max(sigma_ratio)),
        "relative_frobenius_covariance": relative_covariance,
        "correlation_difference_absmax": corr_max,
        "fisher_sigma_b1_ratio_k4_over_k3": fisher_ratio,
        "gates": gates,
        "inputs": {"base_covariance": str(args.base_covariance), "test_covariance": str(args.test_covariance)},
    }
    atomic_json(args.output, payload)
    print(json.dumps(payload, indent=2, sort_keys=True))
    if payload["status"] != "pass":
        raise RuntimeError(f"covariance k-support A/B 未通过：{gates}")


if __name__ == "__main__":
    main()
