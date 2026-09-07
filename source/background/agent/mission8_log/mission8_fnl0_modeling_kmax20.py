#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Mission 8：fnl=0 的 2PCF 建模图（kmax=20）
=========================================

任务目标
--------
用户当前只要求一件事：

1. 保持 `kmax=20`；
2. 看 `fnl=0` 时的 2PCF 建模曲线；
3. 同时检查不同建模方法的系统偏差方向。

因此本脚本统一对 `1Gpc` 和 `3Gpc` 的 `fnl=0` 做：

- best-fit P0(k)
- Baseline 建模
- ExpWindow 建模
- AllDiscrete 建模

最后输出一张并排图和一份简短指标表。
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
FFTLOG_N = m5.FFTLOG_N
FFTLOG_PADDING = m5.FFTLOG_PADDING
LARGE_SCALE_MIN = m5.LARGE_SCALE_MIN

OUT_FIG = THIS_DIR / "mission8_fnl0_modeling_kmax20.png"
OUT_METRIC_CSV = THIS_DIR / "mission8_fnl0_modeling_kmax20_metrics.csv"
OUT_SUMMARY_MD = THIS_DIR / "mission8_fnl0_modeling_kmax20_summary_20260313.md"


@dataclass
class BoxConfig:
    """
    单个盒长配置。
    """

    tag: str
    box_size: float
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
        pk0_glob="/pscratch/sd/l/lzy/3gpc-UNIT-fastpm/pk_masscut/pk_rsd_3gpc_fnl0_N*.dat",
        pcf0_glob="/pscratch/sd/l/lzy/3gpc-UNIT-fastpm/pcf_masscut/pcf_rsd_3gpc_fnl0_N*.dat",
        rid_min=2,
        rid_max=80,
        drop_zero_std=False,
        x_power=12.0,
        q_chunk=250000,
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


def xi_from_full_discrete_shell_sum(
    s_data: np.ndarray,
    gq: np.ndarray,
    k_fund: float,
    k_dense: np.ndarray,
    p_dense: np.ndarray,
    box_size: float,
    q_chunk: int,
) -> np.ndarray:
    """
    全离散 shell 求和：

        xi_0(r) = (1/V) * sum_{q>=1, k_q<=kmax} g_q P(k_q) j0(k_q r)
    """
    volume = float(box_size) ** 3
    xi = np.zeros_like(s_data, dtype=np.float64)
    q_nonzero = np.nonzero(gq[1:])[0] + 1

    for start in range(0, q_nonzero.size, q_chunk):
        q_block = q_nonzero[start : start + q_chunk]
        kvals = k_fund * np.sqrt(q_block.astype(np.float64))
        pvals = np.interp(kvals, k_dense, p_dense)
        weights = gq[q_block].astype(np.float64) * pvals
        xi += np.sum(weights[:, None] * spherical_bessel_j0(np.outer(kvals, s_data)), axis=0) / volume
        if start == 0 or (start // q_chunk) % 50 == 0:
            print(f"[INFO] shell-sum progress: {start + q_block.size} / {q_nonzero.size}")
    return xi


def build_metric_row(
    method: str,
    s: np.ndarray,
    xi_data: np.ndarray,
    xi_std: np.ndarray,
    xi_model: np.ndarray,
    box: str,
) -> Dict[str, object]:
    """
    统一包装一行指标。
    """
    row = dict(m5.compute_alignment_metrics(s, xi_data, xi_std, xi_model, tag=method))
    row["box"] = box
    return row


def main() -> None:
    """
    画 fnl=0, kmax=20 的 2PCF 建模图。
    """
    print("[INFO] Mission8 fnl0 modeling kmax20 start")
    all_rows: List[Dict[str, object]] = []
    plot_data: Dict[str, Dict[str, object]] = {}

    for cfg in BOXES:
        print(f"[INFO] ===== box {cfg.tag} fnl0 =====")
        k_fund = 2.0 * np.pi / cfg.box_size
        qmax = int(np.floor((KMAX_INT / k_fund) ** 2))
        nmax = int(np.floor(KMAX_INT / k_fund))

        data_raw = m5.load_mock_data(
            f"{cfg.tag}_fnl0",
            cfg.pk0_glob,
            cfg.pcf0_glob,
            cfg.rid_min,
            cfg.rid_max,
        )
        if cfg.drop_zero_std:
            data = m8finish.filter_zero_std_pk_bins(data_raw, tag=f"{cfg.tag} fnl0")
        else:
            data = data_raw

        fit = m5.fit_best_pk(data)
        bestfit = fit["bestfit"]
        print(f"[INFO] {cfg.tag} fnl0 bestfit = {bestfit}")

        k_fft = np.geomspace(KMIN_GLOBAL / FFTLOG_PADDING, KMAX_INT * FFTLOG_PADDING, FFTLOG_N)
        p_fft = m5.build_theory_p0(k_fft, bestfit)
        xi_baseline = m5.evaluate_unwindowed_with_kmin(
            data.scen,
            k_fft,
            p_fft,
            kmin=k_fund,
            tag=f"{cfg.tag}_fnl0_baseline",
        )
        xi_exp = m8finish.evaluate_exp_window_model(
            s_data=data.scen,
            k_grid=k_fft,
            p0_grid=p_fft,
            k_fund=k_fund,
            x_power=cfg.x_power,
        )

        print(f"[INFO] {cfg.tag}: compute full discrete degeneracy for fnl0")
        gq = compute_shell_degeneracy_fft(qmax, nmax)
        k_dense = np.geomspace(k_fund, KMAX_INT, 300000)
        p_dense = m5.build_theory_p0(k_dense, bestfit)
        xi_discrete = xi_from_full_discrete_shell_sum(
            s_data=data.scen,
            gq=gq,
            k_fund=k_fund,
            k_dense=k_dense,
            p_dense=p_dense,
            box_size=cfg.box_size,
            q_chunk=cfg.q_chunk,
        )

        rows_here = [
            build_metric_row(f"{cfg.tag}_fnl0_Baseline", data.scen, data.xi_mean, data.xi_std, xi_baseline, cfg.tag),
            build_metric_row(f"{cfg.tag}_fnl0_ExpWindow", data.scen, data.xi_mean, data.xi_std, xi_exp, cfg.tag),
            build_metric_row(f"{cfg.tag}_fnl0_AllDiscrete", data.scen, data.xi_mean, data.xi_std, xi_discrete, cfg.tag),
        ]
        all_rows.extend(rows_here)

        plot_data[cfg.tag] = {
            "s": data.scen,
            "xi_data": data.xi_mean,
            "xi_std": data.xi_std,
            "baseline": xi_baseline,
            "exp": xi_exp,
            "discrete": xi_discrete,
            "metrics": {row["method"]: row for row in rows_here},
        }

    keys: List[str] = []
    seen = set()
    for row in all_rows:
        for key in row.keys():
            if key not in seen:
                seen.add(key)
                keys.append(key)
    with OUT_METRIC_CSV.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        writer.writerows([{k: row.get(k, "") for k in keys} for row in all_rows])
    print(f"[INFO] saved metrics: {OUT_METRIC_CSV}")

    fig, axes = plt.subplots(2, 2, figsize=(12.4, 9.5))
    for col, tag in enumerate(["1Gpc", "3Gpc"]):
        item = plot_data[tag]
        s = item["s"]
        ax = axes[0, col]
        ax.errorbar(
            s,
            s**2 * item["xi_data"],
            yerr=s**2 * item["xi_std"],
            fmt="o",
            ms=3.0,
            capsize=2,
            color="black",
            label="Measured fnl=0",
        )
        ax.plot(s, s**2 * item["baseline"], "-", lw=1.6, color="tab:blue", label="Baseline")
        ax.plot(s, s**2 * item["exp"], "-", lw=1.6, color="tab:red", label="Exp window")
        ax.plot(s, s**2 * item["discrete"], "-", lw=1.8, color="tab:green", label="All discrete")
        ax.axvline(LARGE_SCALE_MIN, color="gray", lw=1.0, ls="--", alpha=0.8)
        ax.set_title(f"{tag}: fnl=0, kmax=20")
        ax.set_xlabel(r"$r\,[\mathrm{Mpc}/h]$")
        ax.set_ylabel(r"$r^2 \xi_0(r)$")
        text = "\n".join([
            f"Baseline mean_sigma={item['metrics'][f'{tag}_fnl0_Baseline']['mean_sigma']:.3f}",
            f"Exp mean_sigma={item['metrics'][f'{tag}_fnl0_ExpWindow']['mean_sigma']:.3f}",
            f"Disc mean_sigma={item['metrics'][f'{tag}_fnl0_AllDiscrete']['mean_sigma']:.3f}",
        ])
        ax.text(
            0.02,
            0.98,
            text,
            transform=ax.transAxes,
            va="top",
            ha="left",
            fontsize=8.4,
            bbox=dict(facecolor="white", alpha=0.82, edgecolor="gray"),
        )
        ax.legend(fontsize=8)

        axr = axes[1, col]
        r2_data = s**2 * item["xi_data"]
        r2_std = np.maximum(s**2 * item["xi_std"], 1e-12)
        for arr, color, label, marker in [
            (item["baseline"], "tab:blue", "Baseline", "o"),
            (item["exp"], "tab:red", "ExpWindow", "s"),
            (item["discrete"], "tab:green", "AllDiscrete", "^"),
        ]:
            resid = (r2_data - s**2 * arr) / r2_std
            axr.plot(s, resid, marker + "-", ms=2.8, lw=1.2, color=color, label=label)
        axr.axhline(0.0, color="black", lw=1.0)
        axr.axhline(1.0, color="gray", lw=0.8, ls="--", alpha=0.7)
        axr.axhline(-1.0, color="gray", lw=0.8, ls="--", alpha=0.7)
        axr.axvline(LARGE_SCALE_MIN, color="gray", lw=1.0, ls="--", alpha=0.8)
        axr.set_title(f"{tag}: residual direction")
        axr.set_xlabel(r"$r\,[\mathrm{Mpc}/h]$")
        axr.set_ylabel(r"$(Data-Model)/\sigma$")
        axr.legend(fontsize=8)

    fig.tight_layout()
    fig.savefig(OUT_FIG, bbox_inches="tight")
    plt.close(fig)
    print(f"[INFO] saved figure: {OUT_FIG}")

    with OUT_SUMMARY_MD.open("w", encoding="utf-8") as f:
        f.write("# Mission 8：fnl=0 的 2PCF 建模图（kmax=20）\n\n")
        f.write("这次只画 `fnl=0` 的 2PCF 建模图，`kmax` 固定为 `20`，并统一比较三种方法：Baseline / ExpWindow / AllDiscrete。\n\n")
        f.write("| Box | Method | mean_abs_sigma | chi2_ndof | mean_sigma |\n")
        f.write("|---|---|---:|---:|---:|\n")
        for tag in ["1Gpc", "3Gpc"]:
            for method in [f"{tag}_fnl0_Baseline", f"{tag}_fnl0_ExpWindow", f"{tag}_fnl0_AllDiscrete"]:
                row = next(row for row in all_rows if row["method"] == method)
                f.write(
                    f"| {tag} | {method.split('_')[-1]} | "
                    f"{float(row['mean_abs_sigma']):.4f} | "
                    f"{float(row['chi2_ndof']):.4f} | "
                    f"{float(row['mean_sigma']):.4f} |\n"
                )
        f.write("\n")
        f.write("其中 `mean_sigma < 0` 表示模型整体偏高，`mean_sigma > 0` 表示模型整体偏低。\n")
    print(f"[INFO] saved summary: {OUT_SUMMARY_MD}")
    print("[INFO] Mission8 fnl0 modeling kmax20 done")


if __name__ == "__main__":
    main()
