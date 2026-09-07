#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Mission 8：把 no-PNG 主体积分下限改成 k_base=2pi/L
===================================================

本脚本复用 `mission8_pngsplit_ksplit005.py` 的全部 PNG 处理方式，
只改一件事：

1. no-PNG 主体部分的连续积分下限不再取 `1e-4`
2. 改成盒子基模 `k_base = 2pi/L`

也就是测试：

    xi_total = xi_ref_continuous(k > k_base, no PNG)
             + xi_png_discrete(k <= 0.05)
             + xi_png_continuous(k > 0.05)      [方案A]

以及

    xi_total = xi_ref_continuous(k > k_base, no PNG)
             + xi_png_discrete(k <= 0.05)       [方案B]

目的就是检查：用户提出把 no-PNG 主体从 `k_base` 开始积以后，
之前那种系统偏高是否会缓解。
"""

from __future__ import annotations

import csv
import sys
from pathlib import Path
from typing import Dict, List

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


THIS_DIR = Path(__file__).resolve().parent
if str(THIS_DIR) not in sys.path:
    sys.path.insert(0, str(THIS_DIR))

import mission8_pngsplit_ksplit005 as base
import task5_ir_window_solution_3gpc_multitype as m5
import mission8_exp_window_box_finish as m8finish


OUT_METRIC_CSV = THIS_DIR / "mission8_pngsplit_ksplit005_kbase_metrics.csv"
OUT_SUMMARY_MD = THIS_DIR / "mission8_pngsplit_ksplit005_kbase_summary_20260313.md"
OUT_FIG = THIS_DIR / "mission8_pngsplit_ksplit005_kbase_compare.png"


def xi_ref_continuous_from_kbase(s_data: np.ndarray, k_grid: np.ndarray, p_ref_grid: np.ndarray, k_base: float) -> np.ndarray:
    """
    无 PNG 主体部分：从 k_base=2pi/L 开始做连续积分。
    """
    taper = m5.build_log_taper_window(
        k_grid,
        kmin=k_base,
        kmax=base.KMAX_INT,
        frac=m5.EDGE_TAPER_FRAC,
    )
    r_grid, xi_grid = m5.xi_fftlog_from_effective_p0(k_grid, p_ref_grid * taper)
    return m5.interp_xi_to_s(s_data, r_grid, xi_grid)


def main() -> None:
    """
    跑 k_base 版本。
    """
    print("[INFO] Mission8 pngsplit ksplit=0.05 with k_base start")
    all_rows: List[Dict[str, object]] = []
    plot_data: Dict[str, Dict[str, object]] = {}

    for cfg in base.BOXES:
        print(f"[INFO] ===== box {cfg.tag} =====")
        k_fund = 2.0 * np.pi / cfg.box_size
        qmax_low = int(np.floor((base.K_SPLIT / k_fund) ** 2))
        nmax_low = int(np.floor(base.K_SPLIT / k_fund))
        print(f"[INFO] {cfg.tag}: k_base={k_fund:.6f}, qmax_low={qmax_low}, nmax_low={nmax_low}")

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

        k_fft = np.geomspace(base.KMIN_GLOBAL / base.FFTLOG_PADDING, base.KMAX_INT * base.FFTLOG_PADDING, base.FFTLOG_N)
        k_dense_low = np.geomspace(k_fund, base.K_SPLIT, 80000)

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

        xi100_ref = xi_ref_continuous_from_kbase(data100.scen, k_fft, p100_ref_fft, k_fund)
        xi0_ref = xi_ref_continuous_from_kbase(data0.scen, k_fft, p0_ref_fft, k_fund)

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

        gq_low = base.compute_shell_degeneracy_fft(qmax_low, nmax_low)

        xi100_png_low = base.xi_from_discrete_shell_sum(
            s_data=data100.scen,
            gq=gq_low,
            k_fund=k_fund,
            k_dense=k_dense_low,
            p_dense=delta100_dense_low,
            box_size=cfg.box_size,
        )
        xi0_png_low = base.xi_from_discrete_shell_sum(
            s_data=data0.scen,
            gq=gq_low,
            k_fund=k_fund,
            k_dense=k_dense_low,
            p_dense=delta0_dense_low,
            box_size=cfg.box_size,
        )

        xi100_png_high = base.xi_png_continuous_highk(data100.scen, k_fft, delta100_fft)
        xi0_png_high = base.xi_png_continuous_highk(data0.scen, k_fft, delta0_fft)

        xi100_split = xi100_ref + xi100_png_low + xi100_png_high
        xi0_split = xi0_ref + xi0_png_low + xi0_png_high
        xi100_split_no_high = xi100_ref + xi100_png_low
        xi0_split_no_high = xi0_ref + xi0_png_low

        rows_here = [
            base.build_metric_row(f"{cfg.tag}_fnl100_ExpWindow", data100.scen, data100.xi_mean, data100.xi_std, xi100_exp, cfg.tag, "fnl100"),
            base.build_metric_row(f"{cfg.tag}_fnl100_PNGSplit_k005_kbase", data100.scen, data100.xi_mean, data100.xi_std, xi100_split, cfg.tag, "fnl100"),
            base.build_metric_row(f"{cfg.tag}_fnl100_PNGSplit_k005_kbase_NoHigh", data100.scen, data100.xi_mean, data100.xi_std, xi100_split_no_high, cfg.tag, "fnl100"),
            base.build_metric_row(f"{cfg.tag}_fnl0_ExpWindow", data0.scen, data0.xi_mean, data0.xi_std, xi0_exp, cfg.tag, "fnl0"),
            base.build_metric_row(f"{cfg.tag}_fnl0_PNGSplit_k005_kbase", data0.scen, data0.xi_mean, data0.xi_std, xi0_split, cfg.tag, "fnl0"),
            base.build_metric_row(f"{cfg.tag}_fnl0_PNGSplit_k005_kbase_NoHigh", data0.scen, data0.xi_mean, data0.xi_std, xi0_split_no_high, cfg.tag, "fnl0"),
        ]
        all_rows.extend(rows_here)

        plot_data[cfg.tag] = {
            "fnl100": {"s": data100.scen, "xi_data": data100.xi_mean, "xi_std": data100.xi_std, "exp": xi100_exp, "split": xi100_split, "split_no_high": xi100_split_no_high},
            "fnl0": {"s": data0.scen, "xi_data": data0.xi_mean, "xi_std": data0.xi_std, "exp": xi0_exp, "split": xi0_split, "split_no_high": xi0_split_no_high},
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
        ax.errorbar(s, s**2 * item["xi_data"], yerr=s**2 * item["xi_std"], fmt="o", ms=2.8, capsize=2, color="black", label="Measured")
        ax.plot(s, s**2 * item["exp"], "-", lw=1.5, color="tab:red", label="Exp window")
        ax.plot(s, s**2 * item["split"], "-", lw=1.8, color="tab:orange", label="Split kbase")
        ax.plot(s, s**2 * item["split_no_high"], "-", lw=1.6, color="tab:green", label="Split kbase no high")
        ax.axvline(base.LARGE_SCALE_MIN, color="gray", lw=1.0, ls="--", alpha=0.8)
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
        axr.axvline(base.LARGE_SCALE_MIN, color="gray", lw=1.0, ls="--", alpha=0.8)
        axr.set_title(f"{tag} {sample}: residuals")
        axr.set_xlabel(r"$r\,[\mathrm{Mpc}/h]$")
        axr.set_ylabel(r"$(Data-Model)/\sigma$")
        axr.legend(fontsize=8)

    fig.tight_layout()
    fig.savefig(OUT_FIG, bbox_inches="tight")
    plt.close(fig)
    print(f"[INFO] saved figure: {OUT_FIG}")

    with OUT_SUMMARY_MD.open("w", encoding="utf-8") as f:
        f.write("# Mission 8：k_base 起积的 PNG-split(k_split=0.05)\n\n")
        f.write("本次只改了 no-PNG 主体的积分下限：从 `1e-4` 改成 `k_base = 2pi/L`。\n\n")
        f.write("| Box | Sample | Method | mean_abs_sigma | chi2_ndof | mean_sigma |\n")
        f.write("|---|---|---|---:|---:|---:|\n")
        for tag in ["1Gpc", "3Gpc"]:
            for sample in ["fnl100", "fnl0"]:
                for method in ["ExpWindow", "PNGSplit_k005_kbase", "PNGSplit_k005_kbase_NoHigh"]:
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
            row_split = next(row for row in all_rows if row["method"] == f"{tag}_fnl0_PNGSplit_k005_kbase")
            row_nohigh = next(row for row in all_rows if row["method"] == f"{tag}_fnl0_PNGSplit_k005_kbase_NoHigh")
            f.write(
                f"- {tag}: ExpWindow mean_sigma={float(row_exp['mean_sigma']):.4f}, "
                f"Split_kbase mean_sigma={float(row_split['mean_sigma']):.4f}, "
                f"Split_kbase_NoHigh mean_sigma={float(row_nohigh['mean_sigma']):.4f}\n"
            )
    print(f"[INFO] saved summary: {OUT_SUMMARY_MD}")
    print("[INFO] Mission8 pngsplit ksplit=0.05 with k_base done")


if __name__ == "__main__":
    main()
