#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Mission 8：直接用现有测量 P(k) 做全离散建模
===========================================

任务目标
--------
不再先做 desilike best-fit，而是直接把当前已有 `pk_masscut` 的测量均值 P(k)
通过全离散方法变成 2PCF，并和原始“best-fit 输入的全离散”比较。

这里先不重测 k-bin，而是用现有输出做一个 sanity check：

1. 如果“直接测量 P(k)”本身就比理论 best-fit 更好，
   那么再去重测更合理的 k-bin 才有价值；
2. 如果连这一步都不提升，那么单纯换 k-bin 可能不是主因。

实现两种无拟合输入口径：

1. `MeasuredPkInterp`：对测量均值 P(k) 做线性插值后取 P(k_q)
2. `MeasuredPkStep`：按测量 bin 的 `[kmin, kmax]` 做分段常数取值
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

OUT_METRIC_CSV = THIS_DIR / "mission8_measured_pk_direct_existingbins_metrics.csv"
OUT_SUMMARY_MD = THIS_DIR / "mission8_measured_pk_direct_existingbins_summary_20260313.md"
OUT_FIG = THIS_DIR / "mission8_measured_pk_direct_existingbins_compare.png"


@dataclass
class BoxConfig:
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
    x = np.asarray(x, dtype=float)
    out = np.ones_like(x)
    mask = x != 0.0
    out[mask] = np.sin(x[mask]) / x[mask]
    return out


def build_metric_row(method: str, s: np.ndarray, xi_data: np.ndarray, xi_std: np.ndarray, xi_model: np.ndarray, box: str, sample: str) -> Dict[str, object]:
    row = dict(m5.compute_alignment_metrics(s, xi_data, xi_std, xi_model, tag=method))
    row["box"] = box
    row["sample"] = sample
    return row


def evaluate_measured_pk_at_shells_interp(kvals: np.ndarray, kcen: np.ndarray, pmean: np.ndarray) -> np.ndarray:
    return np.interp(kvals, kcen, pmean, left=pmean[0], right=pmean[-1])


def evaluate_measured_pk_at_shells_step(kvals: np.ndarray, kmin: np.ndarray, kmax: np.ndarray, pmean: np.ndarray) -> np.ndarray:
    idx = np.searchsorted(kmax, kvals, side="left")
    idx = np.clip(idx, 0, len(pmean) - 1)
    # 对落在首 bin 左侧或末 bin右侧的情况做边界夹取
    return pmean[idx]


def xi_from_measured_pk_direct(
    s_data: np.ndarray,
    gq: np.ndarray,
    k_fund: float,
    kcen: np.ndarray,
    kmin: np.ndarray,
    kmax: np.ndarray,
    pmean: np.ndarray,
    box_size: float,
    q_chunk: int,
) -> Dict[str, np.ndarray]:
    volume = float(box_size) ** 3
    xi_interp = np.zeros_like(s_data, dtype=np.float64)
    xi_step = np.zeros_like(s_data, dtype=np.float64)
    q_nonzero = np.nonzero(gq[1:])[0] + 1

    for start in range(0, q_nonzero.size, q_chunk):
        q_block = q_nonzero[start : start + q_chunk]
        kvals = k_fund * np.sqrt(q_block.astype(np.float64))

        p_interp = evaluate_measured_pk_at_shells_interp(kvals, kcen, pmean)
        p_step = evaluate_measured_pk_at_shells_step(kvals, kmin, kmax, pmean)
        j0 = spherical_bessel_j0(np.outer(kvals, s_data))

        xi_interp += np.sum((gq[q_block].astype(np.float64) * p_interp)[:, None] * j0, axis=0) / volume
        xi_step += np.sum((gq[q_block].astype(np.float64) * p_step)[:, None] * j0, axis=0) / volume

        if start == 0 or (start // q_chunk) % 50 == 0:
            print(f"[INFO] shell-sum progress: {start + q_block.size} / {q_nonzero.size}")

    return {"interp": xi_interp, "step": xi_step}


def main() -> None:
    print("[INFO] Mission8 measured-Pk direct-existingbins start")
    all_rows: List[Dict[str, object]] = []
    plot_data: Dict[str, Dict[str, object]] = {}

    for cfg in BOXES:
        print(f"[INFO] ===== box {cfg.tag} =====")
        k_fund = 2.0 * np.pi / cfg.box_size
        qmax = int(np.floor((KMAX_INT / k_fund) ** 2))
        nmax = int(np.floor(KMAX_INT / k_fund))

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
        xi100_exp = m8finish.evaluate_exp_window_model(data100.scen, k_fft, p100_fft, k_fund, cfg.x_power)
        xi0_exp = m8finish.evaluate_exp_window_model(data0.scen, k_fft, p0_fft, k_fund, cfg.x_power)

        print(f"[INFO] {cfg.tag}: compute full-shell degeneracy")
        gq = compute_shell_degeneracy_fft(qmax, nmax)

        # 原始 best-fit 输入的全离散，作为对照基准
        k_dense = np.geomspace(k_fund, KMAX_INT, 300000)
        p100_dense = m5.build_theory_p0(k_dense, bestfit100)
        p0_dense = m5.build_theory_p0(k_dense, bestfit0)
        xi100_best = xi_from_measured_pk_direct(data100.scen, gq, k_fund, k_dense, k_dense, k_dense, p100_dense, cfg.box_size, cfg.q_chunk)["interp"]
        xi0_best = xi_from_measured_pk_direct(data0.scen, gq, k_fund, k_dense, k_dense, k_dense, p0_dense, cfg.box_size, cfg.q_chunk)["interp"]

        pmean100 = np.mean(data100.p0_mocks, axis=0)
        pmean0 = np.mean(data0.p0_mocks, axis=0)
        xi100_meas = xi_from_measured_pk_direct(data100.scen, gq, k_fund, data100.kcen, data100.kmin, data100.kmax, pmean100, cfg.box_size, cfg.q_chunk)
        xi0_meas = xi_from_measured_pk_direct(data0.scen, gq, k_fund, data0.kcen, data0.kmin, data0.kmax, pmean0, cfg.box_size, cfg.q_chunk)

        rows_here = [
            build_metric_row(f"{cfg.tag}_fnl100_ExpWindow", data100.scen, data100.xi_mean, data100.xi_std, xi100_exp, cfg.tag, "fnl100"),
            build_metric_row(f"{cfg.tag}_fnl100_AllDiscreteBestfit", data100.scen, data100.xi_mean, data100.xi_std, xi100_best, cfg.tag, "fnl100"),
            build_metric_row(f"{cfg.tag}_fnl100_MeasuredPkInterp", data100.scen, data100.xi_mean, data100.xi_std, xi100_meas['interp'], cfg.tag, "fnl100"),
            build_metric_row(f"{cfg.tag}_fnl100_MeasuredPkStep", data100.scen, data100.xi_mean, data100.xi_std, xi100_meas['step'], cfg.tag, "fnl100"),
            build_metric_row(f"{cfg.tag}_fnl0_ExpWindow", data0.scen, data0.xi_mean, data0.xi_std, xi0_exp, cfg.tag, "fnl0"),
            build_metric_row(f"{cfg.tag}_fnl0_AllDiscreteBestfit", data0.scen, data0.xi_mean, data0.xi_std, xi0_best, cfg.tag, "fnl0"),
            build_metric_row(f"{cfg.tag}_fnl0_MeasuredPkInterp", data0.scen, data0.xi_mean, data0.xi_std, xi0_meas['interp'], cfg.tag, "fnl0"),
            build_metric_row(f"{cfg.tag}_fnl0_MeasuredPkStep", data0.scen, data0.xi_mean, data0.xi_std, xi0_meas['step'], cfg.tag, "fnl0"),
        ]
        all_rows.extend(rows_here)

        plot_data[cfg.tag] = {
            "fnl100": {"s": data100.scen, "xi_data": data100.xi_mean, "xi_std": data100.xi_std, "exp": xi100_exp, "best": xi100_best, "interp": xi100_meas["interp"], "step": xi100_meas["step"]},
            "fnl0": {"s": data0.scen, "xi_data": data0.xi_mean, "xi_std": data0.xi_std, "exp": xi0_exp, "best": xi0_best, "interp": xi0_meas["interp"], "step": xi0_meas["step"]},
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

    fig, axes = plt.subplots(4, 2, figsize=(13.0, 16.4))
    panels = [("1Gpc", "fnl100"), ("1Gpc", "fnl0"), ("3Gpc", "fnl100"), ("3Gpc", "fnl0")]
    for row_idx, (tag, sample) in enumerate(panels):
        item = plot_data[tag][sample]
        s = item["s"]
        ax = axes[row_idx, 0]
        ax.errorbar(s, s**2 * item["xi_data"], yerr=s**2 * item["xi_std"], fmt="o", ms=2.8, capsize=2, color="black", label="Measured")
        ax.plot(s, s**2 * item["exp"], "-", lw=1.5, color="tab:red", label="Exp window")
        ax.plot(s, s**2 * item["best"], "-", lw=1.5, color="tab:green", label="AllDiscrete bestfit")
        ax.plot(s, s**2 * item["interp"], "-", lw=1.7, color="tab:orange", label="MeasuredPk interp")
        ax.plot(s, s**2 * item["step"], "-", lw=1.7, color="tab:blue", label="MeasuredPk step")
        ax.axvline(LARGE_SCALE_MIN, color="gray", lw=1.0, ls="--", alpha=0.8)
        ax.set_title(f"{tag} {sample}: curves")
        ax.set_xlabel(r"$r\,[\\mathrm{Mpc}/h]$")
        ax.set_ylabel(r"$r^2 \\xi_0(r)$")
        ax.legend(fontsize=8)

        axr = axes[row_idx, 1]
        r2_data = s**2 * item["xi_data"]
        r2_std = np.maximum(s**2 * item["xi_std"], 1e-12)
        for arr, color, label, marker in [
            (item["exp"], "tab:red", "Exp", "s"),
            (item["best"], "tab:green", "Bestfit", "^"),
            (item["interp"], "tab:orange", "MeasInterp", "d"),
            (item["step"], "tab:blue", "MeasStep", "o"),
        ]:
            resid = (r2_data - s**2 * arr) / r2_std
            axr.plot(s, resid, marker + "-", ms=2.6, lw=1.1, color=color, label=label)
        axr.axhline(0.0, color="black", lw=1.0)
        axr.axhline(1.0, color="gray", lw=0.8, ls="--", alpha=0.7)
        axr.axhline(-1.0, color="gray", lw=0.8, ls="--", alpha=0.7)
        axr.axvline(LARGE_SCALE_MIN, color="gray", lw=1.0, ls="--", alpha=0.8)
        axr.set_title(f"{tag} {sample}: residuals")
        axr.set_xlabel(r"$r\,[\\mathrm{Mpc}/h]$")
        axr.set_ylabel(r"$(Data-Model)/\\sigma$")
        axr.legend(fontsize=8)

    fig.tight_layout()
    fig.savefig(OUT_FIG, bbox_inches="tight")
    plt.close(fig)
    print(f"[INFO] saved figure: {OUT_FIG}")

    with OUT_SUMMARY_MD.open("w", encoding="utf-8") as f:
        f.write("# Mission 8：直接用现有测量 P(k) 做全离散建模\n\n")
        f.write("本次不经过理论拟合，直接把当前已有的测量均值 `P(k)` 送入全离散求和。\n\n")
        f.write("测试了两种无拟合口径：\n")
        f.write("- `MeasuredPkInterp`：对测量 P(k) 做线性插值\n")
        f.write("- `MeasuredPkStep`：按测量 bin 做分段常数\n\n")
        f.write("| Box | Sample | Method | mean_abs_sigma | chi2_ndof | mean_sigma |\n")
        f.write("|---|---|---|---:|---:|---:|\n")
        for tag in ["1Gpc", "3Gpc"]:
            for sample in ["fnl100", "fnl0"]:
                subset = [row for row in all_rows if row["box"] == tag and row["sample"] == sample]
                subset = sorted(subset, key=lambda row: float(row["mean_abs_sigma"]))
                for row in subset:
                    f.write(
                        f"| {tag} | {sample} | {row['method']} | "
                        f"{float(row['mean_abs_sigma']):.4f} | "
                        f"{float(row['chi2_ndof']):.4f} | "
                        f"{float(row['mean_sigma']):.4f} |\n"
                    )
                f.write("\n")
    print(f"[INFO] saved summary: {OUT_SUMMARY_MD}")
    print("[INFO] Mission8 measured-Pk direct-existingbins done")


if __name__ == "__main__":
    main()
