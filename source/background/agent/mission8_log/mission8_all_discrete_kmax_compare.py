#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Mission 8：全离散方法的 kmax=12 vs kmax=20 对比
================================================

任务目标
--------
用户指出一个重要现象：

1. 不能只看 `mean|Δ/σ|`；
2. 全离散模型并不像 baseline 那样“系统偏低”，而更像是“系统偏高/偏另一侧”；
3. 希望在“方法不变”的前提下，只把全离散求和的 `kmax` 从 20 改到 12，
   再看看 1Gpc 和 3Gpc 会发生什么。

因此本脚本严格保持“全离散 shell 求和”方法不变，只修改：

    kmax: 20.0  ->  12.0

然后把新结果和已有的 `kmax=20` 做并列比较。

代码大纲
--------
1. 读取 1Gpc / 3Gpc 的测量数据，并拟合得到 best-fit P0(k)。
2. 用和之前完全相同的“全离散 shell 求和”方法，重新计算 `kmax=12`。
3. 从已有 CSV 里读取 `kmax=20` 的结果。
4. 输出对比表、图像和 markdown 总结，重点强调：
   - mean_abs_sigma
   - chi2_ndof
   - mean_sigma（系统偏高/偏低方向）
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
from scipy.fft import next_fast_len, rfft, irfft


# ============================================================
# 复用模块
# ============================================================

THIS_DIR = Path(__file__).resolve().parent
MISSION5_DIR = THIS_DIR.parent / "mission5_log"
if str(MISSION5_DIR) not in sys.path:
    sys.path.insert(0, str(MISSION5_DIR))
if str(THIS_DIR) not in sys.path:
    sys.path.insert(0, str(THIS_DIR))

import task5_ir_window_solution_3gpc_multitype as m5
import mission8_exp_window_box_finish as m8finish


# ============================================================
# 常数与路径
# ============================================================

KMAX_NEW = 12.0
KMAX_OLD = 20.0
KMIN_GLOBAL = 1.0e-4
FFTLOG_N = m5.FFTLOG_N
FFTLOG_PADDING = m5.FFTLOG_PADDING

OUT_METRIC_CSV = THIS_DIR / "mission8_all_discrete_kmax12_compare_metrics.csv"
OUT_SUMMARY_MD = THIS_DIR / "mission8_all_discrete_kmax12_compare_summary_20260313.md"
OUT_FIG = THIS_DIR / "mission8_all_discrete_kmax12_compare.png"

OLD_METRIC_1GPC = THIS_DIR / "mission8_all_discrete_1gpc_metrics.csv"
OLD_METRIC_3GPC = THIS_DIR / "mission8_all_discrete_3gpc_metrics.csv"


@dataclass
class BoxConfig:
    """
    单个盒长的配置。
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
    ),
]


# ============================================================
# 数学工具
# ============================================================

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
    用 FFT 卷积计算三维离散 shell 的简并度 g_q。
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

    公式
    ----------
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
    return xi


def build_metric_row(method: str, s: np.ndarray, xi_data: np.ndarray, xi_std: np.ndarray, xi_model: np.ndarray) -> Dict[str, object]:
    """
    包装一行标准指标。
    """
    return dict(m5.compute_alignment_metrics(s, xi_data, xi_std, xi_model, tag=method))


def load_old_metric(csv_path: Path, method_name: str) -> Dict[str, float]:
    """
    从已有 kmax=20 结果文件中读取指定方法的指标。
    """
    with csv_path.open(newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    for row in rows:
        if row["method"] == method_name:
            return {k: float(v) if k not in {"method"} else v for k, v in row.items()}
    raise KeyError(f"未在 {csv_path} 中找到方法 {method_name}")


def main() -> None:
    """
    执行 kmax=12 vs 20 的全离散对比。
    """
    print("[INFO] Mission8 all-discrete kmax12 compare start")
    all_rows: List[Dict[str, object]] = []
    plot_data: Dict[str, Dict[str, object]] = {}

    for cfg in BOXES:
        print(f"[INFO] ===== box {cfg.tag} =====")
        k_fund = 2.0 * np.pi / cfg.box_size
        qmax = int(np.floor((KMAX_NEW / k_fund) ** 2))
        nmax = int(np.floor(KMAX_NEW / k_fund))
        q_chunk = 400000 if cfg.tag == "1Gpc" else 250000

        data100_raw = m5.load_mock_data(f"{cfg.tag}_fnl100", cfg.pk100_glob, cfg.pcf100_glob, cfg.rid_min, cfg.rid_max)
        if cfg.drop_zero_std:
            data100 = m8finish.filter_zero_std_pk_bins(data100_raw, tag=f"{cfg.tag} fnl100")
        else:
            data100 = data100_raw

        fit100 = m5.fit_best_pk(data100)
        bestfit100 = fit100["bestfit"]
        print(f"[INFO] {cfg.tag} bestfit100 = {bestfit100}")

        # 先构造 baseline / exp 参照曲线，仅用于图上对照。
        k_fft = np.geomspace(KMIN_GLOBAL / FFTLOG_PADDING, KMAX_NEW * FFTLOG_PADDING, FFTLOG_N)
        p_fft = m5.build_theory_p0(k_fft, bestfit100)
        xi_baseline = m5.evaluate_unwindowed_with_kmin(data100.scen, k_fft, p_fft, kmin=k_fund, tag=f"{cfg.tag}_baseline_kmax12")
        xi_exp = m8finish.evaluate_exp_window_model(
            s_data=data100.scen,
            k_grid=k_fft,
            p0_grid=p_fft,
            k_fund=k_fund,
            x_power=4.0 * (cfg.box_size / 1000.0),
        )

        print(f"[INFO] {cfg.tag}: compute degeneracy for kmax={KMAX_NEW}")
        gq = compute_shell_degeneracy_fft(qmax, nmax)

        # 稠密理论曲线只需要到 kmax=12。
        k_dense = np.geomspace(k_fund, KMAX_NEW, 220000)
        p_dense = m5.build_theory_p0(k_dense, bestfit100)

        xi_discrete12 = xi_from_full_discrete_shell_sum(
            s_data=data100.scen,
            gq=gq,
            k_fund=k_fund,
            k_dense=k_dense,
            p_dense=p_dense,
            box_size=cfg.box_size,
            q_chunk=q_chunk,
        )

        # 读取已有 kmax=20 指标
        old_csv = OLD_METRIC_1GPC if cfg.tag == "1Gpc" else OLD_METRIC_3GPC
        old_method = f"{cfg.tag}_AllDiscrete"
        old_metric = load_old_metric(old_csv, old_method)

        new_metric = build_metric_row(f"{cfg.tag}_AllDiscrete_kmax12", data100.scen, data100.xi_mean, data100.xi_std, xi_discrete12)
        new_metric["box"] = cfg.tag
        new_metric["kmax"] = KMAX_NEW
        new_metric["kmax20_mean_abs_sigma"] = old_metric["mean_abs_sigma"]
        new_metric["kmax20_chi2_ndof"] = old_metric["chi2_ndof"]
        new_metric["kmax20_mean_sigma"] = old_metric["mean_sigma"]
        new_metric["delta_mean_abs_sigma_12_minus_20"] = float(new_metric["mean_abs_sigma"]) - old_metric["mean_abs_sigma"]
        new_metric["delta_mean_sigma_12_minus_20"] = float(new_metric["mean_sigma"]) - old_metric["mean_sigma"]
        all_rows.append(new_metric)

        plot_data[cfg.tag] = {
            "s": data100.scen,
            "xi_data": data100.xi_mean,
            "xi_std": data100.xi_std,
            "baseline12": xi_baseline,
            "exp12": xi_exp,
            "discrete12": xi_discrete12,
            "old_metric20": old_metric,
            "new_metric12": new_metric,
        }

    # 写 CSV
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

    # 作图：两列分别是 1Gpc / 3Gpc；下排看残差方向
    fig, axes = plt.subplots(2, 2, figsize=(12.2, 9.4))
    for col, tag in enumerate(["1Gpc", "3Gpc"]):
        item = plot_data[tag]
        s = item["s"]
        ax = axes[0, col]
        ax.errorbar(s, s**2 * item["xi_data"], yerr=s**2 * item["xi_std"],
                    fmt="o", ms=3.0, capsize=2, color="black", label="Measured mean")
        ax.plot(s, s**2 * item["baseline12"], "-", lw=1.5, color="tab:blue", label="Baseline (kmax=12)")
        ax.plot(s, s**2 * item["exp12"], "-", lw=1.5, color="tab:red", label="Exp window (kmax=12)")
        ax.plot(s, s**2 * item["discrete12"], "-", lw=1.8, color="tab:green", label="All discrete (kmax=12)")
        ax.axvline(m5.LARGE_SCALE_MIN, color="gray", lw=1.0, ls="--", alpha=0.8)
        ax.set_title(f"{tag}: all-discrete kmax=12")
        ax.set_xlabel(r"$r\,[\mathrm{Mpc}/h]$")
        ax.set_ylabel(r"$r^2\xi_0(r)$")
        txt = (
            f"kmax=20: mean_sigma={item['old_metric20']['mean_sigma']:.3f}\n"
            f"kmax=12: mean_sigma={float(item['new_metric12']['mean_sigma']):.3f}"
        )
        ax.text(0.02, 0.98, txt, transform=ax.transAxes, va="top", ha="left", fontsize=8.5,
                bbox=dict(facecolor="white", alpha=0.82, edgecolor="gray"))
        ax.legend(fontsize=8)

        axr = axes[1, col]
        r2_data = s**2 * item["xi_data"]
        r2_std = np.maximum(s**2 * item["xi_std"], 1e-12)
        for arr, color, label, marker in [
            (item["baseline12"], "tab:blue", "Baseline12", "o"),
            (item["exp12"], "tab:red", "Exp12", "s"),
            (item["discrete12"], "tab:green", "AllDiscrete12", "^"),
        ]:
            resid = (r2_data - s**2 * arr) / r2_std
            axr.plot(s, resid, marker + "-", ms=2.8, lw=1.2, color=color, label=label)
        axr.axhline(0.0, color="black", lw=1.0)
        axr.axhline(1.0, color="gray", lw=0.8, ls="--", alpha=0.7)
        axr.axhline(-1.0, color="gray", lw=0.8, ls="--", alpha=0.7)
        axr.axvline(m5.LARGE_SCALE_MIN, color="gray", lw=1.0, ls="--", alpha=0.8)
        axr.set_title(f"{tag}: residual direction")
        axr.set_xlabel(r"$r\,[\mathrm{Mpc}/h]$")
        axr.set_ylabel(r"$(Data-Model)/\sigma$")
        axr.legend(fontsize=8)

    fig.tight_layout()
    fig.savefig(OUT_FIG, bbox_inches="tight")
    plt.close(fig)
    print(f"[INFO] saved figure: {OUT_FIG}")

    with OUT_SUMMARY_MD.open("w", encoding="utf-8") as f:
        f.write("# Mission 8：全离散方法的 kmax=12 vs kmax=20 对比\n\n")
        f.write("这次严格保持“全离散 shell 求和”方法不变，只把求和上限从 `kmax=20` 改成 `kmax=12`。\n\n")
        f.write("用户特别强调不能只看平均绝对误差，而要看偏差方向，因此下表会同时列出：\n")
        f.write("- `mean_abs_sigma`\n")
        f.write("- `chi2_ndof`\n")
        f.write("- `mean_sigma`（看系统偏高/偏低）\n\n")

        f.write("| Box | kmax=20 mean_abs | kmax=20 mean_sigma | kmax=12 mean_abs | kmax=12 mean_sigma | Δmean_sigma(12-20) |\n")
        f.write("|---|---:|---:|---:|---:|---:|\n")
        for row in all_rows:
            f.write(
                f"| {row['box']} | {float(row['kmax20_mean_abs_sigma']):.4f} | {float(row['kmax20_mean_sigma']):.4f} | "
                f"{float(row['mean_abs_sigma']):.4f} | {float(row['mean_sigma']):.4f} | {float(row['delta_mean_sigma_12_minus_20']):.4f} |\n"
            )

        f.write("\n## 判断\n\n")
        for row in all_rows:
            box = str(row["box"])
            mean20 = float(row["kmax20_mean_sigma"])
            mean12 = float(row["mean_sigma"])
            if abs(mean12) < abs(mean20):
                trend = "系统偏差被压小了"
            else:
                trend = "系统偏差没有变小"
            f.write(
                f"- {box}: `mean_sigma` 从 {mean20:.4f} 变到 {mean12:.4f}，说明 {trend}。\n"
            )

        f.write("\n## 解释\n\n")
        f.write("1. 这里没有改变任何方法学，只改了全离散求和的 `kmax`。\n")
        f.write("2. 因此如果偏差方向确实发生了变化，就可以直接归因于高-k 离散尾部的贡献，而不是别的建模改动。\n")
        f.write("3. 这一步主要是为了回答：全离散模型在之前出现的“系统偏高/偏另一侧”，是否可以通过压低 `kmax` 缓解。\n")

    print(f"[INFO] saved summary: {OUT_SUMMARY_MD}")
    print("[INFO] Mission8 all-discrete kmax12 compare done")


if __name__ == "__main__":
    main()
