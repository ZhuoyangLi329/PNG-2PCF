#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Mission 8：PNG 低-k 离散，高-k 连续/忽略，对比 k_split=0.05
=============================================================

按用户最新要求，统一采用下面的拆分：

    xi_total = xi_ref_continuous(no PNG)
             + xi_png_discrete(k <= k_split)
             + xi_png_continuous(k > k_split)   [方案A]

并额外测试：

    xi_total = xi_ref_continuous(no PNG)
             + xi_png_discrete(k <= k_split)    [方案B]

其中固定：

    k_split = 0.05 h/Mpc

说明
----
1. no-PNG 主体部分不加 IR 窗，只在 FFTLog 的积分边界做平滑截断。
2. PNG 的低-k 离散部分只离散到 k_split，不再全离散到 kmax=20。
3. 方案A/B 都会对 fnl100 与 fnl0 分别测试，重点看 mean_sigma 的系统偏高/偏低。
"""

from __future__ import annotations

import csv
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List

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
K_SPLIT = 0.05
FFTLOG_N = m5.FFTLOG_N
FFTLOG_PADDING = m5.FFTLOG_PADDING
LARGE_SCALE_MIN = m5.LARGE_SCALE_MIN

OUT_METRIC_CSV = THIS_DIR / "mission8_pngsplit_ksplit005_metrics.csv"
OUT_SUMMARY_MD = THIS_DIR / "mission8_pngsplit_ksplit005_summary_20260313.md"
OUT_FIG = THIS_DIR / "mission8_pngsplit_ksplit005_compare.png"


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
    ),
]


def spherical_bessel_j0(x: np.ndarray | float) -> np.ndarray:
    """
    球贝塞尔函数 j0(x)=sin(x)/x。
    """
    x = np.asarray(x, dtype=float)
    out = np.ones_like(x)
    mask = x != 0.0
    out[mask] = np.sin(x[mask]) / x[mask]
    return out


def compute_shell_degeneracy_fft(qmax: int, nmax: int) -> np.ndarray:
    """
    用 FFT 卷积精确计算三维离散 shell 的简并度 g_q。
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


def xi_from_discrete_shell_sum(
    s_data: np.ndarray,
    gq: np.ndarray,
    k_fund: float,
    k_dense: np.ndarray,
    p_dense: np.ndarray,
    box_size: float,
) -> np.ndarray:
    """
    用离散 shell 求和计算低-k 的 xi_0(r)。
    """
    volume = float(box_size) ** 3
    q_nonzero = np.nonzero(gq[1:])[0] + 1
    if q_nonzero.size == 0:
        return np.zeros_like(s_data, dtype=np.float64)

    kvals = k_fund * np.sqrt(q_nonzero.astype(np.float64))
    pvals = np.interp(kvals, k_dense, p_dense)
    weights = gq[q_nonzero].astype(np.float64) * pvals
    return np.sum(weights[:, None] * spherical_bessel_j0(np.outer(kvals, s_data)), axis=0) / volume


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


def xi_ref_continuous_no_png(s_data: np.ndarray, k_grid: np.ndarray, p_ref_grid: np.ndarray) -> np.ndarray:
    """
    无 PNG 主体部分：不加 IR 窗，只保留积分边界平滑。
    """
    taper = m5.build_log_taper_window(
        k_grid,
        kmin=KMIN_GLOBAL,
        kmax=KMAX_INT,
        frac=m5.EDGE_TAPER_FRAC,
    )
    r_grid, xi_grid = m5.xi_fftlog_from_effective_p0(k_grid, p_ref_grid * taper)
    return m5.interp_xi_to_s(s_data, r_grid, xi_grid)


def xi_png_continuous_highk(
    s_data: np.ndarray,
    k_grid: np.ndarray,
    delta_png_grid: np.ndarray,
) -> np.ndarray:
    """
    PNG 高-k 连续项：只保留 k > k_split 的部分。
    """
    taper = m5.build_log_taper_window(
        k_grid,
        kmin=K_SPLIT,
        kmax=KMAX_INT,
        frac=m5.EDGE_TAPER_FRAC,
    )
    r_grid, xi_grid = m5.xi_fftlog_from_effective_p0(k_grid, delta_png_grid * taper)
    return m5.interp_xi_to_s(s_data, r_grid, xi_grid)


def main() -> None:
    """
    执行 k_split=0.05 的两种 PNG-split 方案。
    """
    print("[INFO] Mission8 pngsplit ksplit=0.05 start")
    all_rows: List[Dict[str, object]] = []
    plot_data: Dict[str, Dict[str, object]] = {}

    for cfg in BOXES:
        print(f"[INFO] ===== box {cfg.tag} =====")
        k_fund = 2.0 * np.pi / cfg.box_size
        qmax_low = int(np.floor((K_SPLIT / k_fund) ** 2))
        nmax_low = int(np.floor(K_SPLIT / k_fund))
        print(f"[INFO] {cfg.tag}: k_fund={k_fund:.6f}, qmax_low={qmax_low}, nmax_low={nmax_low}")

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

        bestfit100_ref = dict(bestfit100)
        bestfit100_ref["fnl_loc"] = 0.0
        bestfit0_ref = dict(bestfit0)
        bestfit0_ref["fnl_loc"] = 0.0

        print(f"[INFO] {cfg.tag} bestfit100 = {bestfit100}")
        print(f"[INFO] {cfg.tag} bestfit0   = {bestfit0}")

        k_fft = np.geomspace(KMIN_GLOBAL / FFTLOG_PADDING, KMAX_INT * FFTLOG_PADDING, FFTLOG_N)
        k_dense_low = np.geomspace(k_fund, K_SPLIT, 80000)

        p100_fft = m5.build_theory_p0(k_fft, bestfit100)
        p100_ref_fft = m5.build_theory_p0(k_fft, bestfit100_ref)
        p0_fft = m5.build_theory_p0(k_fft, bestfit0)
        p0_ref_fft = m5.build_theory_p0(k_fft, bestfit0_ref)

        delta100_fft = p100_fft - p100_ref_fft
        delta0_fft = p0_fft - p0_ref_fft

        p100_dense_low = m5.build_theory_p0(k_dense_low, bestfit100)
        p100_ref_dense_low = m5.build_theory_p0(k_dense_low, bestfit100_ref)
        p0_dense_low = m5.build_theory_p0(k_dense_low, bestfit0)
        p0_ref_dense_low = m5.build_theory_p0(k_dense_low, bestfit0_ref)

        delta100_dense_low = p100_dense_low - p100_ref_dense_low
        delta0_dense_low = p0_dense_low - p0_ref_dense_low

        xi100_ref = xi_ref_continuous_no_png(data100.scen, k_fft, p100_ref_fft)
        xi0_ref = xi_ref_continuous_no_png(data0.scen, k_fft, p0_ref_fft)

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

        gq_low = compute_shell_degeneracy_fft(qmax_low, nmax_low)

        xi100_png_low = xi_from_discrete_shell_sum(
            s_data=data100.scen,
            gq=gq_low,
            k_fund=k_fund,
            k_dense=k_dense_low,
            p_dense=delta100_dense_low,
            box_size=cfg.box_size,
        )
        xi0_png_low = xi_from_discrete_shell_sum(
            s_data=data0.scen,
            gq=gq_low,
            k_fund=k_fund,
            k_dense=k_dense_low,
            p_dense=delta0_dense_low,
            box_size=cfg.box_size,
        )

        xi100_png_high = xi_png_continuous_highk(data100.scen, k_fft, delta100_fft)
        xi0_png_high = xi_png_continuous_highk(data0.scen, k_fft, delta0_fft)

        xi100_split_with_high = xi100_ref + xi100_png_low + xi100_png_high
        xi0_split_with_high = xi0_ref + xi0_png_low + xi0_png_high

        xi100_split_no_high = xi100_ref + xi100_png_low
        xi0_split_no_high = xi0_ref + xi0_png_low

        rows_here = [
            build_metric_row(f"{cfg.tag}_fnl100_ExpWindow", data100.scen, data100.xi_mean, data100.xi_std, xi100_exp, cfg.tag, "fnl100"),
            build_metric_row(f"{cfg.tag}_fnl100_PNGSplit_k005", data100.scen, data100.xi_mean, data100.xi_std, xi100_split_with_high, cfg.tag, "fnl100"),
            build_metric_row(f"{cfg.tag}_fnl100_PNGSplit_k005_NoHigh", data100.scen, data100.xi_mean, data100.xi_std, xi100_split_no_high, cfg.tag, "fnl100"),
            build_metric_row(f"{cfg.tag}_fnl0_ExpWindow", data0.scen, data0.xi_mean, data0.xi_std, xi0_exp, cfg.tag, "fnl0"),
            build_metric_row(f"{cfg.tag}_fnl0_PNGSplit_k005", data0.scen, data0.xi_mean, data0.xi_std, xi0_split_with_high, cfg.tag, "fnl0"),
            build_metric_row(f"{cfg.tag}_fnl0_PNGSplit_k005_NoHigh", data0.scen, data0.xi_mean, data0.xi_std, xi0_split_no_high, cfg.tag, "fnl0"),
        ]
        all_rows.extend(rows_here)

        plot_data[cfg.tag] = {
            "fnl100": {
                "s": data100.scen,
                "xi_data": data100.xi_mean,
                "xi_std": data100.xi_std,
                "exp": xi100_exp,
                "split": xi100_split_with_high,
                "split_no_high": xi100_split_no_high,
            },
            "fnl0": {
                "s": data0.scen,
                "xi_data": data0.xi_mean,
                "xi_std": data0.xi_std,
                "exp": xi0_exp,
                "split": xi0_split_with_high,
                "split_no_high": xi0_split_no_high,
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

    fig, axes = plt.subplots(4, 2, figsize=(12.4, 16.0))
    panels = [("1Gpc", "fnl100"), ("1Gpc", "fnl0"), ("3Gpc", "fnl100"), ("3Gpc", "fnl0")]
    for row_idx, (tag, sample) in enumerate(panels):
        item = plot_data[tag][sample]
        s = item["s"]

        ax = axes[row_idx, 0]
        ax.errorbar(s, s**2 * item["xi_data"], yerr=s**2 * item["xi_std"],
                    fmt="o", ms=2.8, capsize=2, color="black", label="Measured")
        ax.plot(s, s**2 * item["exp"], "-", lw=1.5, color="tab:red", label="Exp window")
        ax.plot(s, s**2 * item["split"], "-", lw=1.8, color="tab:orange", label="Split k=0.05")
        ax.plot(s, s**2 * item["split_no_high"], "-", lw=1.6, color="tab:green", label="Split k=0.05 no high")
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
            (item["split"], "tab:orange", "Split", "d"),
            (item["split_no_high"], "tab:green", "SplitNoHigh", "^"),
        ]:
            resid = (r2_data - s**2 * arr) / r2_std
            axr.plot(s, resid, marker + "-", ms=2.8, lw=1.2, color=color, label=label)
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
        f.write("# Mission 8：k_split=0.05 的 PNG-split 两方案\n\n")
        f.write("本次固定 `k_split=0.05`，比较三种方法：\n")
        f.write("- `ExpWindow`\n")
        f.write("- `PNGSplit_k005 = xi_ref_cont + xi_png_disc(k<0.05) + xi_png_cont(k>0.05)`\n")
        f.write("- `PNGSplit_k005_NoHigh = xi_ref_cont + xi_png_disc(k<0.05)`\n\n")
        f.write("其中 `mean_sigma < 0` 表示模型整体偏高，`mean_sigma > 0` 表示模型整体偏低。\n\n")
        f.write("| Box | Sample | Method | mean_abs_sigma | chi2_ndof | mean_sigma |\n")
        f.write("|---|---|---|---:|---:|---:|\n")
        for tag in ["1Gpc", "3Gpc"]:
            for sample in ["fnl100", "fnl0"]:
                for method in ["ExpWindow", "PNGSplit_k005", "PNGSplit_k005_NoHigh"]:
                    key = f"{tag}_{sample}_{method}"
                    row = next(row for row in all_rows if row["method"] == key)
                    f.write(
                        f"| {tag} | {sample} | {method} | "
                        f"{float(row['mean_abs_sigma']):.4f} | "
                        f"{float(row['chi2_ndof']):.4f} | "
                        f"{float(row['mean_sigma']):.4f} |\n"
                    )
        f.write("\n## 重点看 fnl=0\n\n")
        for tag in ["1Gpc", "3Gpc"]:
            row_exp = next(row for row in all_rows if row["method"] == f"{tag}_fnl0_ExpWindow")
            row_split = next(row for row in all_rows if row["method"] == f"{tag}_fnl0_PNGSplit_k005")
            row_nohigh = next(row for row in all_rows if row["method"] == f"{tag}_fnl0_PNGSplit_k005_NoHigh")
            f.write(
                f"- {tag}: ExpWindow mean_sigma={float(row_exp['mean_sigma']):.4f}, "
                f"Split mean_sigma={float(row_split['mean_sigma']):.4f}, "
                f"SplitNoHigh mean_sigma={float(row_nohigh['mean_sigma']):.4f}\n"
            )
    print(f"[INFO] saved summary: {OUT_SUMMARY_MD}")
    print("[INFO] Mission8 pngsplit ksplit=0.05 done")


if __name__ == "__main__":
    main()
