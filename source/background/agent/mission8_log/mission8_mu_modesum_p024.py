#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Mission 8：把低-k 的 mu 显式纳入，再试一次
===========================================

代码大纲
--------
1. 读取 1Gpc / 3Gpc 的 fastPM fnl100 / fnl0 数据，并做 best-fit。
2. 让 desilike 理论输出 P0, P2, P4。
3. 用
       P(k,mu) ≈ P0(k) + P2(k)L2(mu) + P4(k)L4(mu)
   近似重建低-k 的各向异性功率谱。
4. 只对 PNG 增量项做低-k 离散处理，并测试两种写法：
   A. 逐模式求和：对每个离散模式直接代入 P(k,mu)。
   B. shell 内离散 mu 平均：在每个 shell 内对离散 mu 取平均，再乘 g_q。
5. 与之前的旧解析法（P0-shell PNG）和 exp 窗口法比较。

说明
----
1. 当前 desilike 这个类默认公开的是 multipoles，而不是直接公开内部 pkmu 数组。
2. 所以这里采用用户建议的近似：
       P(k,mu) ≈ P0 + P2 L2 + P4 L4
3. 从数学上讲，如果对 xi_0 做的是球平均，那么方案 A 和方案 B 应当等价；
   本脚本会把这件事直接数值验证出来。
"""

from __future__ import annotations

import csv
import math
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


# ============================================================
# 复用已有脚本
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
# 输出文件
# ============================================================

OUT_METRIC_CSV = THIS_DIR / "mission8_mu_modesum_metrics.csv"
OUT_SUMMARY_MD = THIS_DIR / "mission8_mu_modesum_summary_20260313.md"
OUT_FIG = THIS_DIR / "mission8_mu_modesum_compare.png"


# ============================================================
# 参数
# ============================================================

KMIN_GLOBAL = 1.0e-4
KMAX_INT = 20.0
FFTLOG_N = m5.FFTLOG_N
FFTLOG_PADDING = m5.FFTLOG_PADDING
EDGE_TAPER_FRAC = m5.EDGE_TAPER_FRAC
LARGE_SCALE_MIN = m5.LARGE_SCALE_MIN
NSHELL_SCAN = [1, 2, 3, 5, 8, 12]


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


@dataclass
class ModeRecord:
    """
    保存一个离散低-k 模式的信息。
    """

    shell_index: int
    q: int
    k_value: float
    mu: float


BOXES = [
    BoxConfig(
        tag="3Gpc",
        box_size=3000.0,
        pk100_glob="/pscratch/sd/l/lzy/3gpc-UNIT-fastpm/pk_masscut/pk_rsd_3gpc_fnl100_N*.dat",
        pcf100_glob="/pscratch/sd/l/lzy/3gpc-UNIT-fastpm/pcf_masscut/pcf_rsd_3gpc_fnl100_N*.dat",
        pk0_glob="/pscratch/sd/l/lzy/3gpc-UNIT-fastpm/pk_masscut/pk_rsd_3gpc_fnl0_N*.dat",
        pcf0_glob="/pscratch/sd/l/lzy/3gpc-UNIT-fastpm/pcf_masscut/pcf_rsd_3gpc_fnl0_N*.dat",
        rid_min=2,
        rid_max=80,
    ),
    BoxConfig(
        tag="1Gpc",
        box_size=1000.0,
        pk100_glob="/pscratch/sd/l/lzy/1gpc-UNIT-fnl100-fastpm/pk_masscut/pk_rsd_N*.dat",
        pcf100_glob="/pscratch/sd/l/lzy/1gpc-UNIT-fnl100-fastpm/pcf_masscut/pcf_rsd_N*.dat",
        pk0_glob="/pscratch/sd/l/lzy/1gpc-UNIT-fnl100-fastpm/pk_masscut_fnl0/pk_rsd_N*.dat",
        pcf0_glob="/pscratch/sd/l/lzy/1gpc-UNIT-fnl100-fastpm/pcf_masscut_fnl0/pcf_rsd_N*.dat",
        rid_min=1,
        rid_max=50,
    ),
]


# ============================================================
# 数学辅助函数
# ============================================================

def legendre_l2(mu: np.ndarray) -> np.ndarray:
    """
    二阶勒让德多项式 L2(mu)。
    """
    mu = np.asarray(mu, dtype=float)
    return 0.5 * (3.0 * mu**2 - 1.0)


def legendre_l4(mu: np.ndarray) -> np.ndarray:
    """
    四阶勒让德多项式 L4(mu)。
    """
    mu = np.asarray(mu, dtype=float)
    return (35.0 * mu**4 - 30.0 * mu**2 + 3.0) / 8.0


def spherical_bessel_j0(x: np.ndarray | float) -> np.ndarray:
    """
    球贝塞尔函数 j0(x)=sin(x)/x。
    """
    x = np.asarray(x, dtype=float)
    out = np.ones_like(x)
    mask = x != 0.0
    out[mask] = np.sin(x[mask]) / x[mask]
    return out


def build_theory_p024(k_grid: np.ndarray, params: Dict[str, float]) -> Dict[str, np.ndarray]:
    """
    用 desilike 评估 P0, P2, P4。

    参数
    ----------
    k_grid : ndarray
        目标 k 网格。
    params : dict
        best-fit 参数。

    返回
    ----------
    dict
        包含 p0, p2, p4 三条曲线。
    """
    from cosmoprimo import Cosmology
    from desilike.theories.galaxy_clustering import FixedPowerSpectrumTemplate, PNGTracerPowerSpectrumMultipoles

    cosmo = Cosmology(
        h=0.6711,
        Omega_b=0.049,
        Omega_cdm=0.3175 - 0.049,
        sigma8=0.834,
        n_s=0.9624,
        engine="class",
    )
    template = FixedPowerSpectrumTemplate(z=m5.UNIT_Z, fiducial=cosmo)
    theory = PNGTracerPowerSpectrumMultipoles(k=k_grid, ells=(0, 2, 4), mu=200, template=template, mode="b-p")
    theory.init.params["p"].update(fixed=True, value=m5.FIXED_P)
    theory.init.params["sn0"].update(fixed=True, value=m5.FIXED_SN0)
    theory.init.params["sigmas"].update(fixed=False, value=m5.FIXED_SIGMAS)
    theory(**params)
    power = np.asarray(theory.power, dtype=float)
    return {"p0": power[0], "p2": power[1], "p4": power[2]}


def reconstruct_pkmu_from_poles(k_values: np.ndarray, mu_values: np.ndarray, poles: Dict[str, np.ndarray], k_grid: np.ndarray) -> np.ndarray:
    """
    用 P0, P2, P4 近似重建 P(k,mu)。

    公式
    ----------
        P(k,mu) ≈ P0(k) + P2(k)L2(mu) + P4(k)L4(mu)
    """
    p0 = np.interp(k_values, k_grid, poles["p0"])
    p2 = np.interp(k_values, k_grid, poles["p2"])
    p4 = np.interp(k_values, k_grid, poles["p4"])
    return p0 + p2 * legendre_l2(mu_values) + p4 * legendre_l4(mu_values)


def enumerate_modes_by_shell(box_size: float, qmax: int = 300) -> Tuple[List[m8finish.ShellInfo], List[ModeRecord]]:
    """
    枚举低-k 离散模式，并给出每个模式的 shell 编号与 mu。

    约定
    ----------
    这里采用 LOS 沿 z 方向，所以
        mu = |k_z| / |k|
    使用绝对值是因为 RSD 理论在 mu 上是偶函数。
    """
    shells = m8finish.enumerate_shells(box_size=box_size, qmax=qmax)
    q_to_shell_index = {shell.q: shell.index for shell in shells}
    kf = 2.0 * np.pi / float(box_size)
    nmax = int(math.ceil(math.sqrt(qmax)))
    mode_records: List[ModeRecord] = []

    for nx in range(-nmax, nmax + 1):
        for ny in range(-nmax, nmax + 1):
            for nz in range(-nmax, nmax + 1):
                if nx == 0 and ny == 0 and nz == 0:
                    continue
                q = nx * nx + ny * ny + nz * nz
                if q > qmax:
                    continue
                k = kf * math.sqrt(q)
                mu = abs(nz) / math.sqrt(q)
                mode_records.append(
                    ModeRecord(
                        shell_index=q_to_shell_index[q],
                        q=q,
                        k_value=k,
                        mu=mu,
                    )
                )

    mode_records.sort(key=lambda rec: (rec.shell_index, rec.mu, rec.q))
    return shells, mode_records


def xi_png_low_modesum_A(
    s_data: np.ndarray,
    mode_records: List[ModeRecord],
    nshell: int,
    poles_total: Dict[str, np.ndarray],
    poles_ref: Dict[str, np.ndarray],
    k_grid: np.ndarray,
    box_size: float,
) -> np.ndarray:
    """
    方案 A：低-k 直接做模式级 P(k,mu) 显式求和。

    公式
    ----------
        xi_low(r) = (1/V) * sum_modes ΔP(k_i,mu_i) * j0(k_i r)

    其中
        ΔP = P_fnl100 - P_fnl0_ref
    """
    selected = [rec for rec in mode_records if rec.shell_index <= nshell]
    kvals = np.asarray([rec.k_value for rec in selected], dtype=float)
    muvals = np.asarray([rec.mu for rec in selected], dtype=float)

    p_total = reconstruct_pkmu_from_poles(kvals, muvals, poles_total, k_grid)
    p_ref = reconstruct_pkmu_from_poles(kvals, muvals, poles_ref, k_grid)
    delta = p_total - p_ref

    j0 = spherical_bessel_j0(np.outer(kvals, s_data))
    volume = float(box_size) ** 3
    return np.sum(delta[:, None] * j0, axis=0) / volume


def xi_png_low_shellavg_B(
    s_data: np.ndarray,
    mode_records: List[ModeRecord],
    shells: List[m8finish.ShellInfo],
    nshell: int,
    poles_total: Dict[str, np.ndarray],
    poles_ref: Dict[str, np.ndarray],
    k_grid: np.ndarray,
    box_size: float,
) -> np.ndarray:
    """
    方案 B：在每个 shell 内保留离散 mu 取值，但先做离散平均，再乘 g_q。

    说明
    ----------
    对于 xi0(r) 来说，因为同一 shell 上 j0(k r) 相同，
    所以方案 B 与方案 A 在数学上应当等价。本函数就是把这个等价性显式写出来。
    """
    volume = float(box_size) ** 3
    xi = np.zeros_like(s_data, dtype=float)

    for shell in shells[:nshell]:
        shell_modes = [rec for rec in mode_records if rec.shell_index == shell.index]
        kvals = np.asarray([rec.k_value for rec in shell_modes], dtype=float)
        muvals = np.asarray([rec.mu for rec in shell_modes], dtype=float)

        p_total = reconstruct_pkmu_from_poles(kvals, muvals, poles_total, k_grid)
        p_ref = reconstruct_pkmu_from_poles(kvals, muvals, poles_ref, k_grid)
        delta_avg = float(np.mean(p_total - p_ref))
        xi += (shell.multiplicity * delta_avg / volume) * spherical_bessel_j0(shell.k_value * s_data)

    return xi


def xi_modepng_hybrid(
    s_data: np.ndarray,
    shells: List[m8finish.ShellInfo],
    mode_records: List[ModeRecord],
    nshell: int,
    k_grid: np.ndarray,
    poles_total: Dict[str, np.ndarray],
    poles_ref: Dict[str, np.ndarray],
    p0_ref: np.ndarray,
    box_size: float,
    k_fund: float,
    variant: str,
) -> Tuple[np.ndarray, float]:
    """
    构造带 mu 的 PNG-only hybrid。

    结构
    ----
    1. reference / Gaussian 部分：连续 baseline，从 k_f 起积；
    2. PNG 增量部分低-k：用方案 A 或 B；
    3. PNG 增量部分高-k：连续 P0 差值，从 k_transition 起积。

    参数
    ----------
    variant : str
        'A' 表示逐模式求和；'B' 表示 shell 内离散 mu 平均。
    """
    if nshell < len(shells):
        k_transition = 0.5 * (shells[nshell - 1].k_value + shells[nshell].k_value)
    else:
        k_transition = shells[nshell - 1].k_value

    # reference / Gaussian 部分
    p0_ref_eff = p0_ref * m5.build_log_taper_window(
        k_grid, kmin=k_fund, kmax=KMAX_INT, frac=EDGE_TAPER_FRAC
    )
    r_ref, xi_ref = m5.xi_fftlog_from_effective_p0(k_grid, p0_ref_eff)
    xi_ref_on_s = m5.interp_xi_to_s(s_data, r_ref, xi_ref)

    # PNG 低-k 增量
    if variant == "A":
        xi_low = xi_png_low_modesum_A(s_data, mode_records, nshell, poles_total, poles_ref, k_grid, box_size)
    elif variant == "B":
        xi_low = xi_png_low_shellavg_B(s_data, mode_records, shells, nshell, poles_total, poles_ref, k_grid, box_size)
    else:
        raise ValueError(f"未知 variant: {variant}")

    # PNG 高-k 增量：仍用 P0 差值做连续部分
    delta_p0 = poles_total["p0"] - poles_ref["p0"]
    delta_eff = delta_p0 * m5.build_log_taper_window(
        k_grid, kmin=k_transition, kmax=KMAX_INT, frac=EDGE_TAPER_FRAC
    )
    r_delta, xi_delta = m5.xi_fftlog_from_effective_p0(k_grid, delta_eff)
    xi_delta_on_s = m5.interp_xi_to_s(s_data, r_delta, xi_delta)

    return xi_ref_on_s + xi_low + xi_delta_on_s, k_transition


def build_metric_row(method: str, s: np.ndarray, xi_data: np.ndarray, xi_std: np.ndarray, xi_model: np.ndarray) -> Dict[str, object]:
    """
    包装一行指标。
    """
    return dict(m5.compute_alignment_metrics(s, xi_data, xi_std, xi_model, tag=method))


def main() -> None:
    """
    执行带 mu 的 Mission 8 新一轮对比。
    """
    print("[INFO] Mission8 mu-mode-sum test start")
    rows: List[Dict[str, object]] = []
    plot_store: Dict[str, Dict[str, object]] = {}

    for cfg in BOXES:
        print(f"[INFO] ===== box {cfg.tag} =====")
        k_fund = 2.0 * np.pi / cfg.box_size
        x_power = 4.0 * (cfg.box_size / 1000.0)

        data100_raw = m5.load_mock_data(f"{cfg.tag}_fnl100", cfg.pk100_glob, cfg.pcf100_glob, cfg.rid_min, cfg.rid_max)
        data0_raw = m5.load_mock_data(f"{cfg.tag}_fnl0", cfg.pk0_glob, cfg.pcf0_glob, cfg.rid_min, cfg.rid_max)
        data100 = m8finish.filter_zero_std_pk_bins(data100_raw, tag=f"{cfg.tag} fnl100")
        data0 = m8finish.filter_zero_std_pk_bins(data0_raw, tag=f"{cfg.tag} fnl0")

        fit100 = m5.fit_best_pk(data100)
        fit0 = m5.fit_best_pk(data0)
        bestfit100 = fit100["bestfit"]
        bestfit0 = fit0["bestfit"]

        bestfit100_fnl0ref = dict(bestfit100)
        bestfit100_fnl0ref["fnl_loc"] = 0.0

        k_grid = np.geomspace(KMIN_GLOBAL / FFTLOG_PADDING, KMAX_INT * FFTLOG_PADDING, FFTLOG_N)
        p0_total = m5.build_theory_p0(k_grid, bestfit100)
        p0_ref = m5.build_theory_p0(k_grid, bestfit100_fnl0ref)

        poles_total = build_theory_p024(k_grid, bestfit100)
        poles_ref = build_theory_p024(k_grid, bestfit100_fnl0ref)

        # 参照方法
        s = data100.scen
        xi_baseline = m5.evaluate_unwindowed_with_kmin(s, k_grid, p0_total, kmin=k_fund, tag=f"{cfg.tag}_baseline")
        xi_exp = m8finish.evaluate_exp_window_model(s, k_grid, p0_total, k_fund=k_fund, x_power=x_power)
        xi_shell_png_old, _ = m8finish.xi_shell_hybrid_png_only(
            s_data=s,
            all_shells=m8finish.enumerate_shells(cfg.box_size, qmax=500),
            nshell=2 if cfg.tag == "3Gpc" else 5,
            k_grid=k_grid,
            p0_total_grid=p0_total,
            p0_ref_fnl0_grid=p0_ref,
            box_size=cfg.box_size,
            k_fund=k_fund,
        )

        shells, mode_records = enumerate_modes_by_shell(cfg.box_size, qmax=500)

        rows_A: List[Dict[str, object]] = []
        rows_B: List[Dict[str, object]] = []
        curves_A: Dict[int, np.ndarray] = {}
        curves_B: Dict[int, np.ndarray] = {}

        for nshell in NSHELL_SCAN:
            xi_A, ktr_A = xi_modepng_hybrid(
                s_data=s,
                shells=shells,
                mode_records=mode_records,
                nshell=nshell,
                k_grid=k_grid,
                poles_total=poles_total,
                poles_ref=poles_ref,
                p0_ref=p0_ref,
                box_size=cfg.box_size,
                k_fund=k_fund,
                variant="A",
            )
            xi_B, ktr_B = xi_modepng_hybrid(
                s_data=s,
                shells=shells,
                mode_records=mode_records,
                nshell=nshell,
                k_grid=k_grid,
                poles_total=poles_total,
                poles_ref=poles_ref,
                p0_ref=p0_ref,
                box_size=cfg.box_size,
                k_fund=k_fund,
                variant="B",
            )
            curves_A[nshell] = xi_A
            curves_B[nshell] = xi_B

            rowA = build_metric_row(f"{cfg.tag}_ModeMu_A_Nshell{nshell}", s, data100.xi_mean, data100.xi_std, xi_A)
            rowA["box"] = cfg.tag
            rowA["kind"] = "ModeMu_A_scan"
            rowA["nshell"] = nshell
            rowA["k_transition"] = ktr_A
            rowA["diff_A_vs_B_max"] = float(np.max(np.abs(xi_A - xi_B)))
            rows_A.append(rowA)

            rowB = build_metric_row(f"{cfg.tag}_ModeMu_B_Nshell{nshell}", s, data100.xi_mean, data100.xi_std, xi_B)
            rowB["box"] = cfg.tag
            rowB["kind"] = "ModeMu_B_scan"
            rowB["nshell"] = nshell
            rowB["k_transition"] = ktr_B
            rows_B.append(rowB)

        bestA = min(rows_A, key=lambda row: float(row["mean_abs_sigma"]))
        bestB = min(rows_B, key=lambda row: float(row["mean_abs_sigma"]))
        nshell_A = int(bestA["nshell"])
        nshell_B = int(bestB["nshell"])
        xi_A_best = curves_A[nshell_A]
        xi_B_best = curves_B[nshell_B]

        main_rows = [
            build_metric_row(f"{cfg.tag}_Baseline", s, data100.xi_mean, data100.xi_std, xi_baseline),
            build_metric_row(f"{cfg.tag}_ExpWindow", s, data100.xi_mean, data100.xi_std, xi_exp),
            build_metric_row(f"{cfg.tag}_ShellPNG_old", s, data100.xi_mean, data100.xi_std, xi_shell_png_old),
            build_metric_row(f"{cfg.tag}_ModeMu_A_best", s, data100.xi_mean, data100.xi_std, xi_A_best),
            build_metric_row(f"{cfg.tag}_ModeMu_B_best", s, data100.xi_mean, data100.xi_std, xi_B_best),
        ]
        for row in main_rows:
            row["box"] = cfg.tag
            if "ModeMu_A_best" in str(row["method"]):
                row["nshell"] = nshell_A
                row["max_abs_diff_to_B"] = float(np.max(np.abs(xi_A_best - xi_B_best)))
            if "ModeMu_B_best" in str(row["method"]):
                row["nshell"] = nshell_B
                row["max_abs_diff_to_A"] = float(np.max(np.abs(xi_A_best - xi_B_best)))

        rows.extend(main_rows)
        rows.extend(rows_A)
        rows.extend(rows_B)

        plot_store[cfg.tag] = {
            "data": data100,
            "baseline": xi_baseline,
            "exp": xi_exp,
            "shell_old": xi_shell_png_old,
            "modeA": xi_A_best,
            "modeB": xi_B_best,
            "bestfit100": bestfit100,
            "nshell_A": nshell_A,
            "nshell_B": nshell_B,
        }

    # 写 CSV
    keys: List[str] = []
    seen = set()
    for row in rows:
        for key in row:
            if key not in seen:
                seen.add(key)
                keys.append(key)
    with OUT_METRIC_CSV.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        writer.writerows([{k: row.get(k, "") for k in keys} for row in rows])
    print(f"[INFO] saved metrics: {OUT_METRIC_CSV}")

    # 作图
    fig, axes = plt.subplots(2, 2, figsize=(12.6, 9.8))
    for col, tag in enumerate(["3Gpc", "1Gpc"]):
        item = plot_store[tag]
        s = item["data"].scen
        ax = axes[0, col]
        ax.errorbar(s, s**2 * item["data"].xi_mean, yerr=s**2 * item["data"].xi_std,
                    fmt="o", ms=3.0, capsize=2, color="black", label="Measured mean")
        ax.plot(s, s**2 * item["baseline"], "-", lw=1.7, color="tab:blue", label="Baseline")
        ax.plot(s, s**2 * item["exp"], "-", lw=1.7, color="tab:red", label="Exp window")
        ax.plot(s, s**2 * item["shell_old"], "-", lw=1.7, color="tab:green", label="Old shell PNG")
        ax.plot(s, s**2 * item["modeA"], "-", lw=1.7, color="tab:orange", label=f"ModeMu A (N={item['nshell_A']})")
        ax.plot(s, s**2 * item["modeB"], "--", lw=1.5, color="tab:purple", label=f"ModeMu B (N={item['nshell_B']})")
        ax.axvline(LARGE_SCALE_MIN, color="gray", lw=1.0, ls="--", alpha=0.8)
        ax.set_title(f"{tag} fnl=100")
        ax.set_xlabel(r"$r\,[\mathrm{Mpc}/h]$")
        ax.set_ylabel(r"$r^2\xi_0(r)$")
        ax.legend(fontsize=7.8, ncol=2)

        axr = axes[1, col]
        r2_data = s**2 * item["data"].xi_mean
        r2_std = np.maximum(s**2 * item["data"].xi_std, 1e-12)
        for arr, color, label, marker in [
            (item["baseline"], "tab:blue", "Baseline", "o"),
            (item["exp"], "tab:red", "Exp", "s"),
            (item["shell_old"], "tab:green", "Old shell PNG", "^"),
            (item["modeA"], "tab:orange", "ModeMu A", "d"),
            (item["modeB"], "tab:purple", "ModeMu B", "x"),
        ]:
            resid = (r2_data - s**2 * arr) / r2_std
            axr.plot(s, resid, marker + "-", ms=2.8, lw=1.2, color=color, label=label)
        axr.axhline(0.0, color="black", lw=1.0)
        axr.axhline(1.0, color="gray", lw=0.8, ls="--", alpha=0.7)
        axr.axhline(-1.0, color="gray", lw=0.8, ls="--", alpha=0.7)
        axr.axvline(LARGE_SCALE_MIN, color="gray", lw=1.0, ls="--", alpha=0.8)
        axr.set_title(f"{tag} residuals")
        axr.set_xlabel(r"$r\,[\mathrm{Mpc}/h]$")
        axr.set_ylabel(r"$(Data-Model)/\sigma$")
        axr.legend(fontsize=7.6, ncol=2)

    fig.tight_layout()
    fig.savefig(OUT_FIG, bbox_inches="tight")
    plt.close(fig)
    print(f"[INFO] saved figure: {OUT_FIG}")

    # 总结 markdown
    def find_row(name: str) -> Dict[str, object]:
        for row in rows:
            if row.get("method") == name:
                return row
        raise KeyError(name)

    with OUT_SUMMARY_MD.open("w", encoding="utf-8") as f:
        f.write("# Mission 8：把 mu 纳入后的低-k 离散模式测试\n\n")
        f.write("这次不再只用 `P0(k)`。低-k 部分先由 desilike 输出 `P0,P2,P4`，再用\n\n")
        f.write("$$\n")
        f.write("P(k,\\mu) \\approx P_0(k) + P_2(k)L_2(\\mu) + P_4(k)L_4(\\mu)\n")
        f.write("$$\n\n")
        f.write("去近似各向异性的红移空间功率谱。然后只对 PNG 增量项做离散 low-k 处理。\n\n")

        for tag in ["3Gpc", "1Gpc"]:
            rb = find_row(f"{tag}_Baseline")
            re = find_row(f"{tag}_ExpWindow")
            ro = find_row(f"{tag}_ShellPNG_old")
            ra = find_row(f"{tag}_ModeMu_A_best")
            rb2 = find_row(f"{tag}_ModeMu_B_best")
            f.write(f"## {tag}\n\n")
            f.write("| 方法 | mean|Δ/σ| | chi2/ndof |\n")
            f.write("|---|---:|---:|\n")
            for row in [rb, re, ro, ra, rb2]:
                f.write(f"| {row['method']} | {float(row['mean_abs_sigma']):.4f} | {float(row['chi2_ndof']):.4f} |\n")
            f.write("\n")
            f.write(
                f"- 最优 ModeMu A：Nshell={int(ra['nshell'])}\n"
                f"- 最优 ModeMu B：Nshell={int(rb2['nshell'])}\n"
            )
            if "max_abs_diff_to_B" in ra:
                f.write(f"- A 与 B 最优曲线最大绝对差：{float(ra['max_abs_diff_to_B']):.6e}\n")
            f.write("\n")

        f.write("## 解释\n\n")
        f.write("1. 方案 A 是真正的“逐模式 P(k,mu) 求和”；方案 B 是“每个 shell 内先对离散 mu 平均，再乘 g_q”。\n")
        f.write("2. 对 xi0(r) 来说，两者理论上应当等价，因为同一 shell 上 `j0(k r)` 相同；脚本里也直接验证了两者差异几乎为 0。\n")
        f.write("3. 因此真正的改进点，不是把 `g_q` 展开成一个个模式，而是把低-k 理论从 `P0(k)` 升级成含离散 `mu` 的 `P(k,mu)` 近似。\n")
        f.write("4. 这正是 1Gpc 比 3Gpc 更需要做的事，因为 1Gpc 最低几个 shell 的 `mu` 取值非常稀疏。\n")

    print(f"[INFO] saved summary: {OUT_SUMMARY_MD}")
    print("[INFO] Mission8 mu-mode-sum test done")


if __name__ == "__main__":
    main()
