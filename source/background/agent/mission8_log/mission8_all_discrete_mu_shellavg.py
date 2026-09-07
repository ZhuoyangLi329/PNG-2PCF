#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Mission 8：原始全离散的低-k 离散 mu-shell 平均修正
===================================================

物理动机
--------
原始全离散方法默认把离散壳层上的功率直接取成连续单极矩 `P0(k_q)`：

    xi(r) = (1/V) Σ_q g_q P0(k_q) j0(k_q r)

但对于 RSD 来说，真实功率是 `P(k, mu)`，而 `P0(k)` 只是连续球面平均。
在有限盒的低-k 壳层里，离散模式方向集合并不等于连续球面平均，
因此更物理的做法是：

1. 对最低若干个离散 shell，显式枚举该 shell 内所有模式方向；
2. 用 `P0 + P2 L2 + P4 L4` 近似重建 `P(k,mu)`；
3. 在每个低-k shell 内对离散 mu 做平均，得到 `P_shell(q)`；
4. 更高 k 处继续回到原始 `P0(k_q)` 全离散。

这个修正没有引入新的经验窗口，只是把低-k RSD 的离散角向平均做得更自洽。
"""

from __future__ import annotations

import csv
import math
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
import mission8_mu_modesum_p024 as mmu
import task5_ir_window_solution_3gpc_multitype as m5


KMIN_GLOBAL = 1.0e-4
KMAX_INT = 20.0
FFTLOG_N = m5.FFTLOG_N
FFTLOG_PADDING = m5.FFTLOG_PADDING
LARGE_SCALE_MIN = m5.LARGE_SCALE_MIN
K_MU_CUT_LIST = [0.05, 0.10, 0.15, 0.20]

OUT_METRIC_CSV = THIS_DIR / "mission8_all_discrete_mu_shellavg_metrics.csv"
OUT_SUMMARY_MD = THIS_DIR / "mission8_all_discrete_mu_shellavg_summary_20260313.md"
OUT_FIG = THIS_DIR / "mission8_all_discrete_mu_shellavg_compare.png"


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


def build_lowq_shellavg_p(
    box_size: float,
    k_mu_cut: float,
    poles: Dict[str, np.ndarray],
    k_grid: np.ndarray,
) -> np.ndarray:
    """
    对 q<=q_cut 的低-k shell，显式做离散 mu 平均，得到每个 q 的 P_shell(q)。
    """
    k_fund = 2.0 * np.pi / float(box_size)
    q_cut = int(np.floor((k_mu_cut / k_fund) ** 2))
    shell_p = np.full(q_cut + 1, np.nan, dtype=np.float64)

    shells, mode_records = mmu.enumerate_modes_by_shell(box_size=box_size, qmax=q_cut)
    for shell in shells:
        shell_modes = [rec for rec in mode_records if rec.q == shell.q]
        kvals = np.asarray([rec.k_value for rec in shell_modes], dtype=float)
        muvals = np.asarray([rec.mu for rec in shell_modes], dtype=float)
        p_shell = float(np.mean(mmu.reconstruct_pkmu_from_poles(kvals, muvals, poles, k_grid)))
        shell_p[shell.q] = p_shell

    return shell_p


def xi_full_discrete_with_lowq_override(
    s_data: np.ndarray,
    gq: np.ndarray,
    k_fund: float,
    k_dense: np.ndarray,
    p_dense: np.ndarray,
    p_lowq_override: np.ndarray,
    box_size: float,
    q_chunk: int,
) -> np.ndarray:
    """
    全离散求和；对于 q<=q_cut 的壳层，使用低-k 离散 mu 平均后的 P_shell(q) 覆盖原始 P0(k_q)。
    """
    volume = float(box_size) ** 3
    xi = np.zeros_like(s_data, dtype=np.float64)
    q_nonzero = np.nonzero(gq[1:])[0] + 1
    q_cut = len(p_lowq_override) - 1

    for start in range(0, q_nonzero.size, q_chunk):
        q_block = q_nonzero[start : start + q_chunk]
        kvals = k_fund * np.sqrt(q_block.astype(np.float64))
        pvals = np.interp(kvals, k_dense, p_dense)

        mask_low = q_block <= q_cut
        if np.any(mask_low):
            pvals[mask_low] = p_lowq_override[q_block[mask_low]]

        weights = gq[q_block].astype(np.float64) * pvals
        xi += np.sum(weights[:, None] * spherical_bessel_j0(np.outer(kvals, s_data)), axis=0) / volume

        if start == 0 or (start // q_chunk) % 50 == 0:
            print(f"[INFO] shell-sum progress: {start + q_block.size} / {q_nonzero.size}")
    return xi


def main() -> None:
    """
    执行低-k 离散 mu-shell 平均修正测试。
    """
    print("[INFO] Mission8 all-discrete mu-shellavg start")
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

        print(f"[INFO] {cfg.tag}: build P0/P2/P4")
        poles100 = mmu.build_theory_p024(k_fft, bestfit100)
        poles0 = mmu.build_theory_p024(k_fft, bestfit0)

        print(f"[INFO] {cfg.tag}: compute full-shell degeneracy")
        gq = compute_shell_degeneracy_fft(qmax, nmax)
        k_dense = np.geomspace(k_fund, KMAX_INT, 300000)
        p100_dense = m5.build_theory_p0(k_dense, bestfit100)
        p0_dense = m5.build_theory_p0(k_dense, bestfit0)

        xi100_all = xi_full_discrete_with_lowq_override(
            s_data=data100.scen,
            gq=gq,
            k_fund=k_fund,
            k_dense=k_dense,
            p_dense=p100_dense,
            p_lowq_override=np.array([np.nan]),
            box_size=cfg.box_size,
            q_chunk=cfg.q_chunk,
        )
        xi0_all = xi_full_discrete_with_lowq_override(
            s_data=data0.scen,
            gq=gq,
            k_fund=k_fund,
            k_dense=k_dense,
            p_dense=p0_dense,
            p_lowq_override=np.array([np.nan]),
            box_size=cfg.box_size,
            q_chunk=cfg.q_chunk,
        )

        rows_here = [
            build_metric_row(f"{cfg.tag}_fnl100_ExpWindow", data100.scen, data100.xi_mean, data100.xi_std, xi100_exp, cfg.tag, "fnl100"),
            build_metric_row(f"{cfg.tag}_fnl100_AllDiscrete", data100.scen, data100.xi_mean, data100.xi_std, xi100_all, cfg.tag, "fnl100"),
            build_metric_row(f"{cfg.tag}_fnl0_ExpWindow", data0.scen, data0.xi_mean, data0.xi_std, xi0_exp, cfg.tag, "fnl0"),
            build_metric_row(f"{cfg.tag}_fnl0_AllDiscrete", data0.scen, data0.xi_mean, data0.xi_std, xi0_all, cfg.tag, "fnl0"),
        ]

        xi100_curves: Dict[str, np.ndarray] = {"AllDiscrete": xi100_all}
        xi0_curves: Dict[str, np.ndarray] = {"AllDiscrete": xi0_all}

        for k_mu_cut in K_MU_CUT_LIST:
            print(f"[INFO] {cfg.tag}: low-k mu shellavg cut = {k_mu_cut:.3f}")
            shell100 = build_lowq_shellavg_p(cfg.box_size, k_mu_cut, poles100, k_fft)
            shell0 = build_lowq_shellavg_p(cfg.box_size, k_mu_cut, poles0, k_fft)

            xi100_mu = xi_full_discrete_with_lowq_override(
                s_data=data100.scen,
                gq=gq,
                k_fund=k_fund,
                k_dense=k_dense,
                p_dense=p100_dense,
                p_lowq_override=shell100,
                box_size=cfg.box_size,
                q_chunk=cfg.q_chunk,
            )
            xi0_mu = xi_full_discrete_with_lowq_override(
                s_data=data0.scen,
                gq=gq,
                k_fund=k_fund,
                k_dense=k_dense,
                p_dense=p0_dense,
                p_lowq_override=shell0,
                box_size=cfg.box_size,
                q_chunk=cfg.q_chunk,
            )

            tagcut = f"k{int(round(1000 * k_mu_cut)):03d}"
            xi100_curves[tagcut] = xi100_mu
            xi0_curves[tagcut] = xi0_mu
            row100 = build_metric_row(f"{cfg.tag}_fnl100_MuShellAvg_{tagcut}", data100.scen, data100.xi_mean, data100.xi_std, xi100_mu, cfg.tag, "fnl100")
            row0 = build_metric_row(f"{cfg.tag}_fnl0_MuShellAvg_{tagcut}", data0.scen, data0.xi_mean, data0.xi_std, xi0_mu, cfg.tag, "fnl0")
            row100["k_mu_cut"] = k_mu_cut
            row0["k_mu_cut"] = k_mu_cut
            rows_here.extend([row100, row0])

        all_rows.extend(rows_here)
        plot_data[cfg.tag] = {
            "fnl100": {"s": data100.scen, "xi_data": data100.xi_mean, "xi_std": data100.xi_std, "exp": xi100_exp, "curves": xi100_curves},
            "fnl0": {"s": data0.scen, "xi_data": data0.xi_mean, "xi_std": data0.xi_std, "exp": xi0_exp, "curves": xi0_curves},
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

    fig, axes = plt.subplots(4, 2, figsize=(13.2, 16.6))
    curve_colors = {
        "AllDiscrete": "tab:green",
        "k050": "tab:orange",
        "k100": "tab:blue",
        "k150": "tab:purple",
        "k200": "tab:brown",
    }
    panels = [("1Gpc", "fnl100"), ("1Gpc", "fnl0"), ("3Gpc", "fnl100"), ("3Gpc", "fnl0")]
    for row_idx, (tag, sample) in enumerate(panels):
        item = plot_data[tag][sample]
        s = item["s"]

        ax = axes[row_idx, 0]
        ax.errorbar(s, s**2 * item["xi_data"], yerr=s**2 * item["xi_std"], fmt="o", ms=2.8, capsize=2, color="black", label="Measured")
        ax.plot(s, s**2 * item["exp"], "-", lw=1.5, color="tab:red", label="Exp window")
        for name, arr in item["curves"].items():
            label = "All discrete" if name == "AllDiscrete" else f"mu-shellavg {name}"
            ax.plot(s, s**2 * arr, "-", lw=1.5, color=curve_colors.get(name, "gray"), label=label)
        ax.axvline(LARGE_SCALE_MIN, color="gray", lw=1.0, ls="--", alpha=0.8)
        ax.set_title(f"{tag} {sample}: curves")
        ax.set_xlabel(r"$r\,[\mathrm{Mpc}/h]$")
        ax.set_ylabel(r"$r^2 \xi_0(r)$")
        ax.legend(fontsize=7.5)

        axr = axes[row_idx, 1]
        r2_data = s**2 * item["xi_data"]
        r2_std = np.maximum(s**2 * item["xi_std"], 1e-12)
        axr.plot(s, (r2_data - s**2 * item["exp"]) / r2_std, "s-", ms=2.7, lw=1.2, color="tab:red", label="Exp")
        for name, arr in item["curves"].items():
            marker = "^" if name == "AllDiscrete" else "d"
            label = "AllDisc" if name == "AllDiscrete" else name
            resid = (r2_data - s**2 * arr) / r2_std
            axr.plot(s, resid, marker + "-", ms=2.5, lw=1.1, color=curve_colors.get(name, "gray"), label=label)
        axr.axhline(0.0, color="black", lw=1.0)
        axr.axhline(1.0, color="gray", lw=0.8, ls="--", alpha=0.7)
        axr.axhline(-1.0, color="gray", lw=0.8, ls="--", alpha=0.7)
        axr.axvline(LARGE_SCALE_MIN, color="gray", lw=1.0, ls="--", alpha=0.8)
        axr.set_title(f"{tag} {sample}: residuals")
        axr.set_xlabel(r"$r\,[\mathrm{Mpc}/h]$")
        axr.set_ylabel(r"$(Data-Model)/\sigma$")
        axr.legend(fontsize=7.5)

    fig.tight_layout()
    fig.savefig(OUT_FIG, bbox_inches="tight")
    plt.close(fig)
    print(f"[INFO] saved figure: {OUT_FIG}")

    with OUT_SUMMARY_MD.open("w", encoding="utf-8") as f:
        f.write("# Mission 8：低-k 离散 mu-shell 平均修正\n\n")
        f.write("本次以原始全离散为基准，只在最低若干个 shell 上，把连续单极矩 `P0(k_q)` 改成了离散方向集合上的 `P(k,mu)` 平均。\n\n")
        f.write("测试的低-k 截止为：`0.05, 0.10, 0.15, 0.20 h/Mpc`。\n\n")
        f.write("其中 `mean_sigma < 0` 表示模型整体偏高，`mean_sigma > 0` 表示模型整体偏低。\n\n")
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
        f.write("## 物理解释\n\n")
        f.write("1. 连续理论的 `P0(k)` 是连续角平均；有限盒低-k shell 的模式方向却是离散的。\n")
        f.write("2. 若低-k RSD 的角向结构重要，则在离散 shell 上直接用 `P0(k_q)` 会有系统偏差。\n")
        f.write("3. 本方法只修正这个低-k 离散角平均口径，不引入额外窗口函数。\n")
    print(f"[INFO] saved summary: {OUT_SUMMARY_MD}")
    print("[INFO] Mission8 all-discrete mu-shellavg done")


if __name__ == "__main__":
    main()
