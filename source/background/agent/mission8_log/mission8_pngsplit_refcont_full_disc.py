#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Mission 8：连续 no-PNG 主体 + 全离散 PNG 增量
=============================================

任务目标
--------
按用户最新要求，把 P(k)->xi(r) 的建模拆成两部分：

1. 无 PNG 主体部分：
   - 取 best-fit desilike 理论，但把 `fnl_loc` 单独置为 0；
   - 在 `k in (1e-4, 20)` 上直接做连续积分。

2. PNG 增量部分：
   - 定义为
         ΔP_png(k) = P_total_bestfit(k) - P_ref(fnl->0, same other params)
   - 这一项继续采用“全离散 shell 求和”做到 `kmax=20`。

所以总模型写成

    xi_model(r) = xi_ref_continuous(r) + xi_png_discrete(r)

这个构造的一个重要性质是：
若数据本身是 fnl=0，则 PNG 增量应当很小，模型自然退化到连续 reference，
从而避免“整个 P(k) 都离散化”带来的系统性偏高。
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

OUT_METRIC_CSV = THIS_DIR / "mission8_pngsplit_refcont_full_disc_metrics.csv"
OUT_SUMMARY_MD = THIS_DIR / "mission8_pngsplit_refcont_full_disc_summary_20260313.md"
OUT_FIG = THIS_DIR / "mission8_pngsplit_refcont_full_disc_compare.png"


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
    用全离散 shell 求和计算 xi_0(r)。
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
    sample: str,
) -> Dict[str, object]:
    """
    统一包装一行指标。
    """
    row = dict(m5.compute_alignment_metrics(s, xi_data, xi_std, xi_model, tag=method))
    row["box"] = box
    row["sample"] = sample
    return row


def continuous_ref_xi(s_data: np.ndarray, k_grid: np.ndarray, p_ref_grid: np.ndarray) -> np.ndarray:
    """
    连续 no-PNG 主体项：不加 IR 窗，只在积分边界做平滑截断。

    说明
    ----
    用户这里强调“无 PNG 的主体部分不需要 IR window”，
    因为它在大尺度并不发散。

    因此这里的做法是：
    - 不对低-k 施加任何额外抑制；
    - 只把积分范围限制在 `1e-4 < k < 20`；
    - 为避免 FFTLog 在边界处出现硬截断振铃，对上下边界做一个纯数值用的 smooth taper。
    """
    taper = m5.build_log_taper_window(
        k_grid,
        kmin=KMIN_GLOBAL,
        kmax=KMAX_INT,
        frac=m5.EDGE_TAPER_FRAC,
    )
    r_grid, xi_grid = m5.xi_fftlog_from_effective_p0(k_grid, p_ref_grid * taper)
    return m5.interp_xi_to_s(s_data, r_grid, xi_grid)


def main() -> None:
    """
    运行“连续 ref + 全离散 PNG 增量”的新方案。
    """
    print("[INFO] Mission8 png-split ref-cont + full-disc start")
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

        bestfit100_ref = dict(bestfit100)
        bestfit100_ref["fnl_loc"] = 0.0
        bestfit0_ref = dict(bestfit0)
        bestfit0_ref["fnl_loc"] = 0.0

        print(f"[INFO] {cfg.tag} bestfit100 = {bestfit100}")
        print(f"[INFO] {cfg.tag} bestfit0   = {bestfit0}")

        k_fft = np.geomspace(KMIN_GLOBAL / FFTLOG_PADDING, KMAX_INT * FFTLOG_PADDING, FFTLOG_N)
        k_dense = np.geomspace(k_fund, KMAX_INT, 300000)

        p100_fft = m5.build_theory_p0(k_fft, bestfit100)
        p100_ref_fft = m5.build_theory_p0(k_fft, bestfit100_ref)
        p0_fft = m5.build_theory_p0(k_fft, bestfit0)
        p0_ref_fft = m5.build_theory_p0(k_fft, bestfit0_ref)

        p100_dense = m5.build_theory_p0(k_dense, bestfit100)
        p100_ref_dense = m5.build_theory_p0(k_dense, bestfit100_ref)
        p0_dense = m5.build_theory_p0(k_dense, bestfit0)
        p0_ref_dense = m5.build_theory_p0(k_dense, bestfit0_ref)

        xi100_ref_cont = continuous_ref_xi(data100.scen, k_fft, p100_ref_fft)
        xi0_ref_cont = continuous_ref_xi(data0.scen, k_fft, p0_ref_fft)

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

        print(f"[INFO] {cfg.tag}: compute full-shell degeneracy once for total and PNG parts")
        gq = compute_shell_degeneracy_fft(qmax, nmax)

        delta100_dense = p100_dense - p100_ref_dense
        delta0_dense = p0_dense - p0_ref_dense

        xi100_png_disc = xi_from_full_discrete_shell_sum(
            s_data=data100.scen,
            gq=gq,
            k_fund=k_fund,
            k_dense=k_dense,
            p_dense=delta100_dense,
            box_size=cfg.box_size,
            q_chunk=cfg.q_chunk,
        )
        xi0_png_disc = xi_from_full_discrete_shell_sum(
            s_data=data0.scen,
            gq=gq,
            k_fund=k_fund,
            k_dense=k_dense,
            p_dense=delta0_dense,
            box_size=cfg.box_size,
            q_chunk=cfg.q_chunk,
        )

        xi100_split = xi100_ref_cont + xi100_png_disc
        xi0_split = xi0_ref_cont + xi0_png_disc

        xi100_all_disc = xi_from_full_discrete_shell_sum(
            s_data=data100.scen,
            gq=gq,
            k_fund=k_fund,
            k_dense=k_dense,
            p_dense=p100_dense,
            box_size=cfg.box_size,
            q_chunk=cfg.q_chunk,
        )
        xi0_all_disc = xi_from_full_discrete_shell_sum(
            s_data=data0.scen,
            gq=gq,
            k_fund=k_fund,
            k_dense=k_dense,
            p_dense=p0_dense,
            box_size=cfg.box_size,
            q_chunk=cfg.q_chunk,
        )

        rows_here = [
            build_metric_row(f"{cfg.tag}_fnl100_ExpWindow", data100.scen, data100.xi_mean, data100.xi_std, xi100_exp, cfg.tag, "fnl100"),
            build_metric_row(f"{cfg.tag}_fnl100_PNGSplitFullDisc", data100.scen, data100.xi_mean, data100.xi_std, xi100_split, cfg.tag, "fnl100"),
            build_metric_row(f"{cfg.tag}_fnl100_AllDiscrete", data100.scen, data100.xi_mean, data100.xi_std, xi100_all_disc, cfg.tag, "fnl100"),
            build_metric_row(f"{cfg.tag}_fnl0_ExpWindow", data0.scen, data0.xi_mean, data0.xi_std, xi0_exp, cfg.tag, "fnl0"),
            build_metric_row(f"{cfg.tag}_fnl0_PNGSplitFullDisc", data0.scen, data0.xi_mean, data0.xi_std, xi0_split, cfg.tag, "fnl0"),
            build_metric_row(f"{cfg.tag}_fnl0_AllDiscrete", data0.scen, data0.xi_mean, data0.xi_std, xi0_all_disc, cfg.tag, "fnl0"),
        ]
        all_rows.extend(rows_here)

        plot_data[cfg.tag] = {
            "fnl100": {
                "s": data100.scen,
                "xi_data": data100.xi_mean,
                "xi_std": data100.xi_std,
                "exp": xi100_exp,
                "split": xi100_split,
                "all_disc": xi100_all_disc,
            },
            "fnl0": {
                "s": data0.scen,
                "xi_data": data0.xi_mean,
                "xi_std": data0.xi_std,
                "exp": xi0_exp,
                "split": xi0_split,
                "all_disc": xi0_all_disc,
            },
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

    fig, axes = plt.subplots(4, 2, figsize=(12.6, 16.2))
    panels = [
        ("1Gpc", "fnl100"),
        ("1Gpc", "fnl0"),
        ("3Gpc", "fnl100"),
        ("3Gpc", "fnl0"),
    ]
    for row_idx, (tag, sample) in enumerate(panels):
        item = plot_data[tag][sample]
        s = item["s"]

        ax = axes[row_idx, 0]
        ax.errorbar(s, s**2 * item["xi_data"], yerr=s**2 * item["xi_std"],
                    fmt="o", ms=2.8, capsize=2, color="black", label="Measured")
        ax.plot(s, s**2 * item["exp"], "-", lw=1.6, color="tab:red", label="Exp window")
        ax.plot(s, s**2 * item["split"], "-", lw=1.8, color="tab:orange", label="Ref-cont + PNG-disc")
        ax.plot(s, s**2 * item["all_disc"], "-", lw=1.5, color="tab:green", label="All discrete")
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
            (item["all_disc"], "tab:green", "AllDisc", "^"),
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
        f.write("# Mission 8：连续 no-PNG 主体 + 全离散 PNG 增量\n\n")
        f.write("本次按新的加法分解实现：\n\n")
        f.write("`xi_total = xi_ref_continuous(fnl->0, 1e-4<k<20) + xi_png_discrete(full shells to kmax=20)`\n\n")
        f.write("其中 `mean_sigma < 0` 表示模型整体偏高，`mean_sigma > 0` 表示模型整体偏低。\n\n")
        f.write("| Box | Sample | Method | mean_abs_sigma | chi2_ndof | mean_sigma |\n")
        f.write("|---|---|---|---:|---:|---:|\n")
        for tag in ["1Gpc", "3Gpc"]:
            for sample in ["fnl100", "fnl0"]:
                for method in ["ExpWindow", "PNGSplitFullDisc", "AllDiscrete"]:
                    key = f"{tag}_{sample}_{method}"
                    row = next(row for row in all_rows if row["method"] == key)
                    f.write(
                        f"| {tag} | {sample} | {method} | "
                        f"{float(row['mean_abs_sigma']):.4f} | "
                        f"{float(row['chi2_ndof']):.4f} | "
                        f"{float(row['mean_sigma']):.4f} |\n"
                    )
        f.write("\n## 核心检查\n\n")
        for tag in ["1Gpc", "3Gpc"]:
            row_exp = next(row for row in all_rows if row["method"] == f"{tag}_fnl0_ExpWindow")
            row_split = next(row for row in all_rows if row["method"] == f"{tag}_fnl0_PNGSplitFullDisc")
            row_disc = next(row for row in all_rows if row["method"] == f"{tag}_fnl0_AllDiscrete")
            f.write(
                f"- {tag} fnl=0: ExpWindow mean_sigma={float(row_exp['mean_sigma']):.4f}, "
                f"Split mean_sigma={float(row_split['mean_sigma']):.4f}, "
                f"AllDiscrete mean_sigma={float(row_disc['mean_sigma']):.4f}\n"
            )
    print(f"[INFO] saved summary: {OUT_SUMMARY_MD}")
    print("[INFO] Mission8 png-split ref-cont + full-disc done")


if __name__ == "__main__":
    main()
