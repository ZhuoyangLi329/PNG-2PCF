#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Mission 8：把离散壳层的 delta 函数改成窄高斯
=============================================

任务目标
--------
用户要求回到“总 P(k) 全离散”的旧框架，不再区分 PNG / 非 PNG。

在这个框架下，原来的径向离散壳层写法本质上对应：

    P_eff(k) = Σ_q C_q δ(k-k_q)

其中

    C_q = (2π² / V) * g_q * P(k_q) / k_q²

现在测试把每个 delta 壳层改成一个非常窄、单位归一的高斯：

    δ(k-k_q)  ->  G_sigma(k-k_q)

并观察 2PCF 建模会往哪边变化。

数值设定
--------
为了既保持“非常窄”，又能看到趋势，这里测试两组盒长自适应宽度：

    sigma_k = alpha * k_f
    alpha in {0.10, 0.25}

其中 k_f = 2π/L。

说明
----
1. alpha=0.10 基本对应“极窄高斯”，应非常接近原始全离散结果。
2. alpha=0.25 仍然比相邻低-k shell 的间隔小很多，但能更清楚看出平滑后趋势。
3. 本脚本同时给出 fnl100 / fnl0 的结果。
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
GAUSS_ALPHA_LIST = [0.10, 0.25]

OUT_METRIC_CSV = THIS_DIR / "mission8_all_discrete_gaussian_delta_metrics.csv"
OUT_SUMMARY_MD = THIS_DIR / "mission8_all_discrete_gaussian_delta_summary_20260313.md"
OUT_FIG = THIS_DIR / "mission8_all_discrete_gaussian_delta_compare.png"


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


def xi_from_full_discrete_with_gaussian_options(
    s_data: np.ndarray,
    gq: np.ndarray,
    k_fund: float,
    k_dense: np.ndarray,
    p_dense: np.ndarray,
    box_size: float,
    q_chunk: int,
    alpha_list: List[float],
) -> Dict[str, np.ndarray]:
    """
    在一次 q-shell 扫描中同时计算：

    1. 原始全离散 delta-shell 结果
    2. 若干个窄高斯 shell 结果

    数学说明
    --------
    若把每个径向 delta 壳层替换为

        G_sigma(k-k_q) = exp[-(k-k_q)^2 / (2 sigma^2)] / (sqrt(2π) sigma)

    并把积分范围近似延拓到 (-∞, +∞)，那么单个壳层对 xi_0 的贡献可以精确写成：

        xi_q^G(r) = (g_q P_q / V) * exp[-sigma^2 r^2 / 2]
                    * [ j0(k_q r) + (sigma^2 / k_q^2) cos(k_q r) ]

    当 sigma << k_q 时，负 k 一侧的高斯尾巴可以忽略，
    这个公式就是“窄高斯逼近 delta 壳层”的高精度表达。
    """
    volume = float(box_size) ** 3
    xi_map: Dict[str, np.ndarray] = {"AllDiscrete": np.zeros_like(s_data, dtype=np.float64)}
    for alpha in alpha_list:
        xi_map[f"Gauss_a{int(round(alpha * 100)):03d}kf"] = np.zeros_like(s_data, dtype=np.float64)

    q_nonzero = np.nonzero(gq[1:])[0] + 1
    s_row = s_data[None, :]

    for start in range(0, q_nonzero.size, q_chunk):
        q_block = q_nonzero[start : start + q_chunk]
        kvals = k_fund * np.sqrt(q_block.astype(np.float64))
        pvals = np.interp(kvals, k_dense, p_dense)
        weights = gq[q_block].astype(np.float64) * pvals

        kr = np.outer(kvals, s_data)
        j0 = spherical_bessel_j0(kr)
        xi_map["AllDiscrete"] += np.sum(weights[:, None] * j0, axis=0) / volume

        coskr = np.cos(kr)
        k2 = np.square(kvals)[:, None]
        for alpha in alpha_list:
            sigma = alpha * k_fund
            sigma2 = sigma * sigma
            damp = np.exp(-0.5 * sigma2 * s_row * s_row)
            kernel = damp * (j0 + sigma2 * coskr / k2)
            key = f"Gauss_a{int(round(alpha * 100)):03d}kf"
            xi_map[key] += np.sum(weights[:, None] * kernel, axis=0) / volume

        if start == 0 or (start // q_chunk) % 50 == 0:
            print(f"[INFO] shell-sum progress: {start + q_block.size} / {q_nonzero.size}")

    return xi_map


def main() -> None:
    """
    执行窄高斯 delta-shell 测试。
    """
    print("[INFO] Mission8 all-discrete gaussian-delta start")
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

        xi100_map = xi_from_full_discrete_with_gaussian_options(
            s_data=data100.scen,
            gq=gq,
            k_fund=k_fund,
            k_dense=k_dense,
            p_dense=p100_dense,
            box_size=cfg.box_size,
            q_chunk=cfg.q_chunk,
            alpha_list=GAUSS_ALPHA_LIST,
        )
        xi0_map = xi_from_full_discrete_with_gaussian_options(
            s_data=data0.scen,
            gq=gq,
            k_fund=k_fund,
            k_dense=k_dense,
            p_dense=p0_dense,
            box_size=cfg.box_size,
            q_chunk=cfg.q_chunk,
            alpha_list=GAUSS_ALPHA_LIST,
        )

        rows_here = [
            build_metric_row(f"{cfg.tag}_fnl100_ExpWindow", data100.scen, data100.xi_mean, data100.xi_std, xi100_exp, cfg.tag, "fnl100"),
            build_metric_row(f"{cfg.tag}_fnl100_AllDiscrete", data100.scen, data100.xi_mean, data100.xi_std, xi100_map["AllDiscrete"], cfg.tag, "fnl100"),
            build_metric_row(f"{cfg.tag}_fnl0_ExpWindow", data0.scen, data0.xi_mean, data0.xi_std, xi0_exp, cfg.tag, "fnl0"),
            build_metric_row(f"{cfg.tag}_fnl0_AllDiscrete", data0.scen, data0.xi_mean, data0.xi_std, xi0_map["AllDiscrete"], cfg.tag, "fnl0"),
        ]
        for alpha in GAUSS_ALPHA_LIST:
            suffix = f"a{int(round(alpha * 100)):03d}kf"
            rows_here.append(
                build_metric_row(
                    f"{cfg.tag}_fnl100_GaussDelta_{suffix}",
                    data100.scen,
                    data100.xi_mean,
                    data100.xi_std,
                    xi100_map[f"Gauss_{suffix}"],
                    cfg.tag,
                    "fnl100",
                )
            )
            rows_here.append(
                build_metric_row(
                    f"{cfg.tag}_fnl0_GaussDelta_{suffix}",
                    data0.scen,
                    data0.xi_mean,
                    data0.xi_std,
                    xi0_map[f"Gauss_{suffix}"],
                    cfg.tag,
                    "fnl0",
                )
            )
        all_rows.extend(rows_here)

        plot_data[cfg.tag] = {
            "fnl100": {"s": data100.scen, "xi_data": data100.xi_mean, "xi_std": data100.xi_std, "exp": xi100_exp, "all_disc": xi100_map["AllDiscrete"], "gauss": xi100_map},
            "fnl0": {"s": data0.scen, "xi_data": data0.xi_mean, "xi_std": data0.xi_std, "exp": xi0_exp, "all_disc": xi0_map["AllDiscrete"], "gauss": xi0_map},
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
        ax.errorbar(s, s**2 * item["xi_data"], yerr=s**2 * item["xi_std"],
                    fmt="o", ms=2.8, capsize=2, color="black", label="Measured")
        ax.plot(s, s**2 * item["exp"], "-", lw=1.5, color="tab:red", label="Exp window")
        ax.plot(s, s**2 * item["all_disc"], "-", lw=1.6, color="tab:green", label="All discrete")
        for alpha, color in zip(GAUSS_ALPHA_LIST, ["tab:orange", "tab:blue"]):
            suffix = f"a{int(round(alpha * 100)):03d}kf"
            ax.plot(s, s**2 * item["gauss"][f"Gauss_{suffix}"], "-", lw=1.6, color=color, label=f"Gauss {alpha:.2f} kf")
        ax.axvline(LARGE_SCALE_MIN, color="gray", lw=1.0, ls="--", alpha=0.8)
        ax.set_title(f"{tag} {sample}: curves")
        ax.set_xlabel(r"$r\,[\mathrm{Mpc}/h]$")
        ax.set_ylabel(r"$r^2 \xi_0(r)$")
        ax.legend(fontsize=8)

        axr = axes[row_idx, 1]
        r2_data = s**2 * item["xi_data"]
        r2_std = np.maximum(s**2 * item["xi_std"], 1e-12)
        curve_list = [
            (item["exp"], "tab:red", "Exp", "s"),
            (item["all_disc"], "tab:green", "AllDisc", "^"),
        ]
        for alpha, color, marker in zip(GAUSS_ALPHA_LIST, ["tab:orange", "tab:blue"], ["d", "o"]):
            suffix = f"a{int(round(alpha * 100)):03d}kf"
            curve_list.append((item["gauss"][f"Gauss_{suffix}"], color, f"Gauss {alpha:.2f}kf", marker))
        for arr, color, label, marker in curve_list:
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
        f.write("# Mission 8：全离散壳层的 delta -> 窄高斯 测试\n\n")
        f.write("本次回到“总 P(k) 全离散”框架，不区分 PNG / 非 PNG。\n\n")
        f.write("测试了三种壳层表示：\n")
        f.write("- `AllDiscrete`: 原始 delta-shell\n")
        f.write("- `GaussDelta_a010kf`: `sigma_k = 0.10 k_f`\n")
        f.write("- `GaussDelta_a025kf`: `sigma_k = 0.25 k_f`\n\n")
        f.write("其中 `mean_sigma < 0` 表示模型整体偏高，`mean_sigma > 0` 表示模型整体偏低。\n\n")
        f.write("| Box | Sample | Method | mean_abs_sigma | chi2_ndof | mean_sigma |\n")
        f.write("|---|---|---|---:|---:|---:|\n")
        order = [
            "ExpWindow",
            "AllDiscrete",
            "GaussDelta_a010kf",
            "GaussDelta_a025kf",
        ]
        for tag in ["1Gpc", "3Gpc"]:
            for sample in ["fnl100", "fnl0"]:
                for method in order:
                    key = f"{tag}_{sample}_{method}"
                    row = next(row for row in all_rows if row["method"] == key)
                    f.write(
                        f"| {tag} | {sample} | {method} | "
                        f"{float(row['mean_abs_sigma']):.4f} | "
                        f"{float(row['chi2_ndof']):.4f} | "
                        f"{float(row['mean_sigma']):.4f} |\n"
                    )
        f.write("\n## 解释提示\n\n")
        f.write("1. 当高斯足够窄时，结果应连续逼近原始全离散 delta-shell。\n")
        f.write("2. 如果 `alpha=0.10` 与原始全离散几乎一致，就说明“只把 delta 换成极窄高斯”本身不会改变结论。\n")
        f.write("3. `alpha=0.25` 若出现明显变化，则代表壳层的径向展宽开始真正影响 2PCF。\n")
    print(f"[INFO] saved summary: {OUT_SUMMARY_MD}")
    print("[INFO] Mission8 all-discrete gaussian-delta done")


if __name__ == "__main__":
    main()
