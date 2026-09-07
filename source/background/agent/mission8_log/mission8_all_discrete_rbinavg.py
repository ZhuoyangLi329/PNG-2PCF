#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Mission 8：原始全离散的径向 bin 平均修正
=======================================

任务目标
--------
以“原始全离散”作为基准，只做一个物理上最直接的修正：

1. pcf 数据是径向 bin 平均，不是单点 `xi(r_center)`；
2. 因此理论端更合理的比较对象应当是每个 `[s_min, s_max]` bin 内的体积平均。

原始全离散写法：

    xi_i = (1/V) Σ_q g_q P(k_q) j0(k_q s_cen,i)

本脚本测试的修正版：

    xi_i^bin = (1/V) Σ_q g_q P(k_q) J_bin(k_q; s_min,i, s_max,i)

其中

    J_bin(k; r1, r2)
      = 3/(r2^3-r1^3) ∫_{r1}^{r2} r^2 j0(kr) dr

这是一个不含自由参数、直接由 pcf 测量定义决定的修正。

说明
----
1. 这里仍然不区分 PNG / 非 PNG，仍然是“总 P(k) 全离散”。
2. 只比较：
   - ExpWindow（原有参考）
   - AllDiscreteCenter（原始全离散）
   - AllDiscreteBinAvg（严格径向 bin 平均）
"""

from __future__ import annotations

import csv
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.fft import irfft, next_fast_len, rfft


THIS_DIR = Path(__file__).resolve().parent
MISSION5_DIR = THIS_DIR.parent / "mission5_log"
if str(MISSION5_DIR) not in sys.path:
    sys.path.insert(0, str(MISSION5_DIR))
if str(THIS_DIR) not in sys.path:
    sys.path.insert(0, str(THIS_DIR))

import mission8_exp_window_box_finish as m8finish
import task5_ir_window_solution_3gpc_multitype as m5


KMIN_GLOBAL = 1.0e-4
KMAX_INT = 20.0
FFTLOG_N = m5.FFTLOG_N
FFTLOG_PADDING = m5.FFTLOG_PADDING
LARGE_SCALE_MIN = m5.LARGE_SCALE_MIN

OUT_METRIC_CSV = THIS_DIR / "mission8_all_discrete_rbinavg_metrics.csv"
OUT_SUMMARY_MD = THIS_DIR / "mission8_all_discrete_rbinavg_summary_20260313.md"
OUT_FIG = THIS_DIR / "mission8_all_discrete_rbinavg_compare.png"


@dataclass
class BoxConfig:
    """
    单个盒长配置。
    """

    tag: str
    box_size: float
    pk100_glob: str
    pcf100_glob: str
    pk0_glob: str
    pcf0_glob: str
    rid_min: int
    rid_max: int
    drop_zero_std: bool
    x_power: float
    q_chunk: int


BOXES = [
    BoxConfig(
        tag="1Gpc",
        box_size=1000.0,
        pk100_glob="/pscratch/sd/l/lzy/1gpc-UNIT-fnl100-fastpm/pk_masscut/pk_rsd_N*.dat",
        pcf100_glob="/pscratch/sd/l/lzy/1gpc-UNIT-fnl100-fastpm/pcf_masscut/pcf_rsd_N*.dat",
        pk0_glob="/pscratch/sd/l/lzy/1gpc-UNIT-fnl100-fastpm/pk_masscut_fnl0/pk_rsd_N*.dat",
        pcf0_glob="/pscratch/sd/l/lzy/1gpc-UNIT-fnl100-fastpm/pcf_masscut_fnl0/pcf_rsd_N*.dat",
        rid_min=1,
        rid_max=50,
        drop_zero_std=True,
        x_power=4.0,
        q_chunk=400000,
    ),
    BoxConfig(
        tag="3Gpc",
        box_size=3000.0,
        pk100_glob="/pscratch/sd/l/lzy/3gpc-UNIT-fastpm/pk_masscut/pk_rsd_3gpc_fnl100_N*.dat",
        pcf100_glob="/pscratch/sd/l/lzy/3gpc-UNIT-fastpm/pcf_masscut/pcf_rsd_3gpc_fnl100_N*.dat",
        pk0_glob="/pscratch/sd/l/lzy/3gpc-UNIT-fastpm/pk_masscut/pk_rsd_3gpc_fnl0_N*.dat",
        pcf0_glob="/pscratch/sd/l/lzy/3gpc-UNIT-fastpm/pcf_masscut/pcf_rsd_3gpc_fnl0_N*.dat",
        rid_min=2,
        rid_max=80,
        drop_zero_std=False,
        x_power=12.0,
        q_chunk=250000,
    ),
]


def compute_shell_degeneracy_fft(qmax: int, nmax: int) -> np.ndarray:
    """
    用 FFT 卷积精确计算三维离散 shell 简并度 g_q。
    """
    a = np.zeros(qmax + 1, dtype=np.float32)
    a[0] = 1.0
    n = np.arange(1, nmax + 1, dtype=np.int64)
    sq = np.square(n)
    sq = sq[sq <= qmax]
    a[sq] = 2.0

    nfft = next_fast_len(3 * qmax + 1)
    print(f"[INFO] degeneracy FFT: qmax={qmax}, nfft={nfft}")
    fa = rfft(a, n=nfft)
    conv = irfft(fa * fa * fa, n=nfft)
    return np.rint(conv[: qmax + 1]).astype(np.int64)


def spherical_bessel_j0(x: np.ndarray | float) -> np.ndarray:
    """
    球贝塞尔函数 j0(x)=sin(x)/x。
    """
    x = np.asarray(x, dtype=float)
    out = np.ones_like(x)
    mask = x != 0.0
    out[mask] = np.sin(x[mask]) / x[mask]
    return out


def j0_volume_bin_average(kvals: np.ndarray, rmin: np.ndarray, rmax: np.ndarray) -> np.ndarray:
    """
    计算体积平均后的 bin kernel:

        J_bin(k; r1, r2) = 3/(r2^3-r1^3) ∫ r^2 j0(kr) dr

    对应解析表达式：

        J_bin = 3 * [sin(kr2)-sin(kr1)-kr2 cos(kr2)+kr1 cos(kr1)]
                / [k^3 (r2^3-r1^3)]
    """
    kvals = np.asarray(kvals, dtype=float)
    rmin = np.asarray(rmin, dtype=float)
    rmax = np.asarray(rmax, dtype=float)

    kr1 = np.outer(kvals, rmin)
    kr2 = np.outer(kvals, rmax)
    denom = np.outer(kvals**3, rmax**3 - rmin**3)
    numer = np.sin(kr2) - np.sin(kr1) - kr2 * np.cos(kr2) + kr1 * np.cos(kr1)
    out = 3.0 * numer / denom
    return out


def load_pcf_bin_edges(pcf_glob: str, rid_min: int, rid_max: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    读取 pcf 的 s_cen / s_min / s_max 网格。
    """
    pcf_map = m5.list_realization_files(pcf_glob)
    common = [rid for rid in sorted(pcf_map) if rid_min <= rid <= rid_max]
    if not common:
        raise RuntimeError(f"未找到 pcf realization: {pcf_glob}")
    tables = [np.loadtxt(pcf_map[rid], comments="#") for rid in common]
    scen_list = [arr[:, 0] for arr in tables]
    smin_list = [arr[:, 1] for arr in tables]
    smax_list = [arr[:, 2] for arr in tables]
    m5.assert_same_grid(scen_list, "pcf_scen")
    m5.assert_same_grid(smin_list, "pcf_smin")
    m5.assert_same_grid(smax_list, "pcf_smax")
    return scen_list[0].copy(), smin_list[0].copy(), smax_list[0].copy()


def xi_center_and_binavg_for_two_models(
    s_center: np.ndarray,
    smin: np.ndarray,
    smax: np.ndarray,
    gq: np.ndarray,
    k_fund: float,
    k_dense: np.ndarray,
    p_dense_a: np.ndarray,
    p_dense_b: np.ndarray,
    box_size: float,
    q_chunk: int,
) -> Dict[str, np.ndarray]:
    """
    在同一轮 q-shell 扫描中，同时计算两个功率谱模型的：

    1. 原始 center-eval 全离散
    2. 严格 r-bin average 全离散
    """
    volume = float(box_size) ** 3
    xi_a_center = np.zeros_like(s_center, dtype=np.float64)
    xi_b_center = np.zeros_like(s_center, dtype=np.float64)
    xi_a_bin = np.zeros_like(s_center, dtype=np.float64)
    xi_b_bin = np.zeros_like(s_center, dtype=np.float64)

    q_nonzero = np.nonzero(gq[1:])[0] + 1
    for start in range(0, q_nonzero.size, q_chunk):
        q_block = q_nonzero[start : start + q_chunk]
        kvals = k_fund * np.sqrt(q_block.astype(np.float64))
        pvals_a = np.interp(kvals, k_dense, p_dense_a)
        pvals_b = np.interp(kvals, k_dense, p_dense_b)
        weights_a = gq[q_block].astype(np.float64) * pvals_a
        weights_b = gq[q_block].astype(np.float64) * pvals_b

        j0_center = spherical_bessel_j0(np.outer(kvals, s_center))
        j0_bin = j0_volume_bin_average(kvals, smin, smax)

        xi_a_center += np.sum(weights_a[:, None] * j0_center, axis=0) / volume
        xi_b_center += np.sum(weights_b[:, None] * j0_center, axis=0) / volume
        xi_a_bin += np.sum(weights_a[:, None] * j0_bin, axis=0) / volume
        xi_b_bin += np.sum(weights_b[:, None] * j0_bin, axis=0) / volume

        if start == 0 or (start // q_chunk) % 50 == 0:
            print(f"[INFO] shell-sum progress: {start + q_block.size} / {q_nonzero.size}")

    return {
        "a_center": xi_a_center,
        "b_center": xi_b_center,
        "a_bin": xi_a_bin,
        "b_bin": xi_b_bin,
    }


def build_metric_row(
    method: str,
    s: np.ndarray,
    xi_data: np.ndarray,
    xi_std: np.ndarray,
    xi_model: np.ndarray,
    box: str,
    sample: str,
) -> Dict[str, object]:
    """
    统一包装一行指标。
    """
    row = dict(m5.compute_alignment_metrics(s, xi_data, xi_std, xi_model, tag=method))
    row["box"] = box
    row["sample"] = sample
    return row


def main() -> None:
    """
    执行原始全离散的径向 bin 平均修正测试。
    """
    print("[INFO] Mission8 all-discrete r-bin-average start")
    all_rows: List[Dict[str, object]] = []
    plot_data: Dict[str, Dict[str, object]] = {}

    for cfg in BOXES:
        print(f"[INFO] ===== box {cfg.tag} =====")
        k_fund = 2.0 * np.pi / cfg.box_size
        qmax = int(np.floor((KMAX_INT / k_fund) ** 2))
        nmax = int(np.floor(KMAX_INT / k_fund))

        s100_cen, s100_min, s100_max = load_pcf_bin_edges(cfg.pcf100_glob, cfg.rid_min, cfg.rid_max)
        s0_cen, s0_min, s0_max = load_pcf_bin_edges(cfg.pcf0_glob, cfg.rid_min, cfg.rid_max)
        if not (np.allclose(s100_cen, s0_cen) and np.allclose(s100_min, s0_min) and np.allclose(s100_max, s0_max)):
            raise ValueError(f"{cfg.tag}: fnl100 与 fnl0 的 pcf bin 网格不一致")

        data100_raw = m5.load_mock_data(f"{cfg.tag}_fnl100", cfg.pk100_glob, cfg.pcf100_glob, cfg.rid_min, cfg.rid_max)
        data0_raw = m5.load_mock_data(f"{cfg.tag}_fnl0", cfg.pk0_glob, cfg.pcf0_glob, cfg.rid_min, cfg.rid_max)
        if cfg.drop_zero_std:
            data100 = m8finish.filter_zero_std_pk_bins(data100_raw, tag=f"{cfg.tag} fnl100")
            data0 = m8finish.filter_zero_std_pk_bins(data0_raw, tag=f"{cfg.tag} fnl0")
        else:
            data100 = data100_raw
            data0 = data0_raw

        fit100 = m5.fit_best_pk(data100)
        fit0 = m5.fit_best_pk(data0)
        bestfit100 = fit100["bestfit"]
        bestfit0 = fit0["bestfit"]
        print(f"[INFO] {cfg.tag} bestfit100 = {bestfit100}")
        print(f"[INFO] {cfg.tag} bestfit0   = {bestfit0}")

        k_fft = np.geomspace(KMIN_GLOBAL / FFTLOG_PADDING, KMAX_INT * FFTLOG_PADDING, FFTLOG_N)
        p100_fft = m5.build_theory_p0(k_fft, bestfit100)
        p0_fft = m5.build_theory_p0(k_fft, bestfit0)
        xi100_exp = m8finish.evaluate_exp_window_model(
            s_data=data100.scen,
            k_grid=k_fft,
            p0_grid=p100_fft,
            k_fund=k_fund,
            x_power=cfg.x_power,
        )
        xi0_exp = m8finish.evaluate_exp_window_model(
            s_data=data0.scen,
            k_grid=k_fft,
            p0_grid=p0_fft,
            k_fund=k_fund,
            x_power=cfg.x_power,
        )

        print(f"[INFO] {cfg.tag}: compute full-shell degeneracy")
        gq = compute_shell_degeneracy_fft(qmax, nmax)
        k_dense = np.geomspace(k_fund, KMAX_INT, 300000)
        p100_dense = m5.build_theory_p0(k_dense, bestfit100)
        p0_dense = m5.build_theory_p0(k_dense, bestfit0)

        xi_map = xi_center_and_binavg_for_two_models(
            s_center=data100.scen,
            smin=s100_min,
            smax=s100_max,
            gq=gq,
            k_fund=k_fund,
            k_dense=k_dense,
            p_dense_a=p100_dense,
            p_dense_b=p0_dense,
            box_size=cfg.box_size,
            q_chunk=cfg.q_chunk,
        )

        rows_here = [
            build_metric_row(f"{cfg.tag}_fnl100_ExpWindow", data100.scen, data100.xi_mean, data100.xi_std, xi100_exp, cfg.tag, "fnl100"),
            build_metric_row(f"{cfg.tag}_fnl100_AllDiscreteCenter", data100.scen, data100.xi_mean, data100.xi_std, xi_map["a_center"], cfg.tag, "fnl100"),
            build_metric_row(f"{cfg.tag}_fnl100_AllDiscreteBinAvg", data100.scen, data100.xi_mean, data100.xi_std, xi_map["a_bin"], cfg.tag, "fnl100"),
            build_metric_row(f"{cfg.tag}_fnl0_ExpWindow", data0.scen, data0.xi_mean, data0.xi_std, xi0_exp, cfg.tag, "fnl0"),
            build_metric_row(f"{cfg.tag}_fnl0_AllDiscreteCenter", data0.scen, data0.xi_mean, data0.xi_std, xi_map["b_center"], cfg.tag, "fnl0"),
            build_metric_row(f"{cfg.tag}_fnl0_AllDiscreteBinAvg", data0.scen, data0.xi_mean, data0.xi_std, xi_map["b_bin"], cfg.tag, "fnl0"),
        ]
        all_rows.extend(rows_here)

        plot_data[cfg.tag] = {
            "fnl100": {
                "s": data100.scen,
                "xi_data": data100.xi_mean,
                "xi_std": data100.xi_std,
                "exp": xi100_exp,
                "disc_center": xi_map["a_center"],
                "disc_bin": xi_map["a_bin"],
            },
            "fnl0": {
                "s": data0.scen,
                "xi_data": data0.xi_mean,
                "xi_std": data0.xi_std,
                "exp": xi0_exp,
                "disc_center": xi_map["b_center"],
                "disc_bin": xi_map["b_bin"],
            },
        }

    keys: List[str] = []
    seen = set()
    for row in all_rows:
        for key in row:
            if key not in seen:
                seen.add(key)
                keys.append(key)
    with OUT_METRIC_CSV.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        writer.writerows([{k: row.get(k, "") for k in keys} for row in all_rows])
    print(f"[INFO] saved metrics: {OUT_METRIC_CSV}")

    fig, axes = plt.subplots(4, 2, figsize=(12.8, 16.1))
    panels = [("1Gpc", "fnl100"), ("1Gpc", "fnl0"), ("3Gpc", "fnl100"), ("3Gpc", "fnl0")]
    for row_idx, (tag, sample) in enumerate(panels):
        item = plot_data[tag][sample]
        s = item["s"]

        ax = axes[row_idx, 0]
        ax.errorbar(s, s**2 * item["xi_data"], yerr=s**2 * item["xi_std"],
                    fmt="o", ms=2.8, capsize=2, color="black", label="Measured")
        ax.plot(s, s**2 * item["exp"], "-", lw=1.5, color="tab:red", label="Exp window")
        ax.plot(s, s**2 * item["disc_center"], "-", lw=1.5, color="tab:green", label="All discrete center")
        ax.plot(s, s**2 * item["disc_bin"], "-", lw=1.8, color="tab:orange", label="All discrete r-bin avg")
        ax.axvline(LARGE_SCALE_MIN, color="gray", lw=1.0, ls="--", alpha=0.8)
        ax.set_title(f"{tag} {sample}: curves")
        ax.set_xlabel(r"$r\,[\mathrm{Mpc}/h]$")
        ax.set_ylabel(r"$r^2 \xi_0(r)$")
        ax.legend(fontsize=8)

        axr = axes[row_idx, 1]
        r2_data = s**2 * item["xi_data"]
        r2_std = np.maximum(s**2 * item["xi_std"], 1e-12)
        for arr, color, label, marker in [
            (item["exp"], "tab:red", "Exp", "s"),
            (item["disc_center"], "tab:green", "Disc center", "^"),
            (item["disc_bin"], "tab:orange", "Disc binavg", "d"),
        ]:
            resid = (r2_data - s**2 * arr) / r2_std
            axr.plot(s, resid, marker + "-", ms=2.7, lw=1.2, color=color, label=label)
        axr.axhline(0.0, color="black", lw=1.0)
        axr.axhline(1.0, color="gray", lw=0.8, ls="--", alpha=0.7)
        axr.axhline(-1.0, color="gray", lw=0.8, ls="--", alpha=0.7)
        axr.axvline(LARGE_SCALE_MIN, color="gray", lw=1.0, ls="--", alpha=0.8)
        axr.set_title(f"{tag} {sample}: residuals")
        axr.set_xlabel(r"$r\,[\mathrm{Mpc}/h]$")
        axr.set_ylabel(r"$(Data-Model)/\sigma$")
        axr.legend(fontsize=8)

    fig.tight_layout()
    fig.savefig(OUT_FIG, bbox_inches="tight")
    plt.close(fig)
    print(f"[INFO] saved figure: {OUT_FIG}")

    with OUT_SUMMARY_MD.open("w", encoding="utf-8") as f:
        f.write("# Mission 8：原始全离散的径向 bin 平均修正\n\n")
        f.write("本次完全保留“总 P(k) 全离散”框架，只把理论端从 `j0(k s_cen)` 改成了与 pcf 测量定义一致的 bin 体积平均 kernel。\n\n")
        f.write("其中 `mean_sigma < 0` 表示模型整体偏高，`mean_sigma > 0` 表示模型整体偏低。\n\n")
        f.write("| Box | Sample | Method | mean_abs_sigma | chi2_ndof | mean_sigma |\n")
        f.write("|---|---|---|---:|---:|---:|\n")
        for tag in ["1Gpc", "3Gpc"]:
            for sample in ["fnl100", "fnl0"]:
                for method in ["ExpWindow", "AllDiscreteCenter", "AllDiscreteBinAvg"]:
                    key = f"{tag}_{sample}_{method}"
                    row = next(row for row in all_rows if row["method"] == key)
                    f.write(
                        f"| {tag} | {sample} | {method} | "
                        f"{float(row['mean_abs_sigma']):.4f} | "
                        f"{float(row['chi2_ndof']):.4f} | "
                        f"{float(row['mean_sigma']):.4f} |\n"
                    )
        f.write("\n## 物理解释\n\n")
        f.write("1. pcf 文件本身提供的是 `s_min, s_max` 区间平均后的 `xi_0`，而不是点值。\n")
        f.write("2. 因此用 `j0(k s_cen)` 只是近似；更自洽的做法应当是对每个 r-bin 做体积平均。\n")
        f.write("3. 这一修正不引入任何额外经验参数，所以如果它改善了结果，物理解释是最直接的。\n")
    print(f"[INFO] saved summary: {OUT_SUMMARY_MD}")
    print("[INFO] Mission8 all-discrete r-bin-average done")


if __name__ == "__main__":
    main()
