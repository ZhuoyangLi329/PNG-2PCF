#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
meeting-fig: Masscut validation
baseline FFTLog vs BinAvgFit + FastDiscrete
================================================

代码大纲
--------
1. 自动扫描 `masscut_scan/<tag>/` 中已有完整结果的 masscut 样本。
2. 对每个 masscut：
   - 读取 pk / pcf 数据
   - 用标准 desilike 做 baseline 所需的 best-fit
   - 用 BinAvgFit 做当前方法所需的 best-fit
   - baseline: 直接 FFTLog，kmin = 2*pi/L 硬截断
   - current : BinAvgFit + CachedRebin(FastDiscrete)
3. 把所有 masscut 画到一张总图：
   - 上排：模型 vs 测量均值 + errorbar
   - 下排：baseline 和 current 的 (Data-Model)/sigma
4. 为避免高 mass_min 下 halo 数太少导致子图失真，默认去掉排序最后 4 个 masscut。

输出
----
- 只输出 PDF：
  `/pscratch/sd/l/lzy/1gpc-UNIT-fnl100-fastpm/agent/meeting-fig/masscut_validation_baseline_vs_fastdiscrete.pdf`
"""

from __future__ import annotations

import math
import os
import sys
import time
from pathlib import Path

import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


# ============================================================
# 路径与复用函数
# ============================================================
THIS_DIR = Path(__file__).resolve().parent
AGENT_DIR = THIS_DIR.parent
MISSION10_DIR = AGENT_DIR / "mission10_log"
MISSION7_SCRIPT_DIR = AGENT_DIR / "mission7_masscut" / "scripts"

for path in [MISSION10_DIR, MISSION7_SCRIPT_DIR]:
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from mission10_binavgfit_full_discrete_compare import (  # noqa: E402
    fit_standard_desilike,
    fit_binavg_minuit,
    eval_pk_dense,
    gq_enumerate,
    build_shells_for_kmax,
    build_bin_shell_index,
    gq_fft,
)
from masscut_model_validate import (  # noqa: E402
    load_pk_ensemble,
    filter_singular_pk_bins,
    read_pcf_mean_std,
    xi0_from_p0_fftlog,
)


# ============================================================
# 全局设置
# ============================================================
PROJECT_ROOT = Path("/pscratch/sd/l/lzy/1gpc-UNIT-fnl100-fastpm")
MASSCUT_ROOT = PROJECT_ROOT / "masscut_scan"
OUT_PDF = THIS_DIR / "masscut_validation_baseline_vs_fastdiscrete.pdf"

BOX_SIZE = 1000.0
K_FUND = 2.0 * np.pi / BOX_SIZE
VOLUME = BOX_SIZE ** 3
KMAX_DISCRETE = 15.0
KMAX_BASELINE = 20.0
KINT_MIN_WINDOW = 1e-4
N_DENSE = 300_000
FFTLOG_N = 4096
FFTLOG_PADDING = 4.0
EDGE_TAPER_FRAC = 0.06
P_FIX = 1.2
Z = 1.0
REALIZATION_MIN = 1
REALIZATION_MAX = 50
N_DATAPOINTS = 20
DK_FACTOR = 0.1
DROP_LAST_N_TAGS = 4

matplotlib.rcParams.update({
    "font.size": 10,
    "font.family": "serif",
    "font.serif": ["DejaVu Serif", "Times New Roman", "Times"],
    "mathtext.fontset": "stix",
    "axes.spines.top": False,
    "axes.spines.right": False,
    "savefig.dpi": 300,
    "figure.dpi": 150,
})


# ============================================================
# CachedRebin / FastDiscrete
# ============================================================
def precompute_rebin_cache(gq: np.ndarray, kf: float, kmax: float, dk_factor: float = 0.1) -> tuple[np.ndarray, np.ndarray]:
    """
    预计算 CachedRebin 所需的模式总数和有效 k。

    参数
    ----
    gq : ndarray
        壳层简并度数组。
    kf : float
        基模 2pi/L。
    kmax : float
        离散求和上限。
    dk_factor : float
        重分箱宽度，dk = dk_factor * kf。

    返回
    ----
    g_nz, k_eff : ndarray
        非空重分箱的总权重与有效 k。
    """
    q_nz = np.nonzero(gq[1:])[0] + 1
    k_shell = kf * np.sqrt(q_nz.astype(np.float64))
    g_shell = gq[q_nz].astype(np.float64)

    dk = dk_factor * kf
    n_bins = int(np.ceil(kmax / dk)) + 1
    bin_index = np.clip((k_shell / dk).astype(np.int64), 0, n_bins - 1)

    g_bin = np.bincount(bin_index, weights=g_shell, minlength=n_bins)
    gk_bin = np.bincount(bin_index, weights=g_shell * k_shell, minlength=n_bins)
    nonzero = g_bin > 0
    return g_bin[nonzero], gk_bin[nonzero] / g_bin[nonzero]


def xi_cached_rebin(s: np.ndarray, g_nz: np.ndarray, k_eff: np.ndarray, volume: float, kd: np.ndarray, pd: np.ndarray) -> np.ndarray:
    """
    用 CachedRebin 快速计算 xi0(r)。

    参数
    ----
    s : ndarray
        目标 r 网格。
    g_nz : ndarray
        每个重分箱中的模式总数。
    k_eff : ndarray
        每个重分箱的有效 k。
    volume : float
        盒体积。
    kd, pd : ndarray
        稠密 k 网格及理论 P(k)。

    返回
    ----
    ndarray
        xi0(r)。
    """
    weights = g_nz * np.interp(k_eff, kd, pd)
    kr = np.outer(k_eff, s)
    j0 = np.ones_like(kr)
    mask = kr != 0.0
    j0[mask] = np.sin(kr[mask]) / kr[mask]
    return (weights @ j0) / volume


# ============================================================
# Masscut 辅助
# ============================================================
def tag_to_mass_value(tag: str) -> float:
    """
    把 `mmin_1p4e13` / `mmin_1e14` 转回 float。

    参数
    ----
    tag : str
        masscut 标签。

    返回
    ----
    float
        对应的 mass_min 数值。
    """
    return float(tag.replace("mmin_", "").replace("p", "."))


def discover_masscut_tags(root: Path) -> list[str]:
    """
    自动发现已有完整 masscut 数据的标签。

    判定条件
    --------
    - `pk/` 与 `pcf/` 目录都存在
    - 至少有一个 `pk_rsd_N*.dat` 和 `pcf_rsd_N*.dat`

    返回
    ----
    list[str]
        按 mass_min 从低到高排序后的标签列表。
    """
    tags: list[str] = []
    for tag_dir in sorted(root.glob("mmin_*")):
        pk_files = list((tag_dir / "pk").glob("pk_rsd_N*.dat"))
        pcf_files = list((tag_dir / "pcf").glob("pcf_rsd_N*.dat"))
        if pk_files and pcf_files:
            tags.append(tag_dir.name)
    tags.sort(key=tag_to_mass_value)
    return tags


def build_cosmology():
    """
    构建 UNIT 对应的 Cosmology。

    返回
    ----
    Cosmology
        cosmoprimo 宇宙学对象。
    """
    from cosmoprimo import Cosmology

    return Cosmology(
        h=0.6711,
        Omega_b=0.049,
        Omega_cdm=0.3175 - 0.049,
        sigma8=0.834,
        n_s=0.9624,
        engine="class",
    )


def run_one_masscut(tag: str, cosmo, kd_discrete: np.ndarray, g_nz: np.ndarray, k_eff: np.ndarray) -> dict[str, object]:
    """
    对单个 masscut 执行 baseline 与 current 的对比建模。

    参数
    ----
    tag : str
        masscut 标签，例如 `mmin_1e13`。
    cosmo :
        cosmoprimo Cosmology 对象。
    kd_discrete : ndarray
        current 方法（FastDiscrete）使用的稠密 k 网格。
    g_nz, k_eff : ndarray
        CachedRebin 的缓存。

    返回
    ----
    dict
        包含 top/bottom 作图所需数组。
    """
    tag_dir = MASSCUT_ROOT / tag
    pk_dir = tag_dir / "pk"
    pcf_dir = tag_dir / "pcf"

    print(f"\n{'=' * 68}\n{tag}\n{'=' * 68}")

    # 1) 读取并过滤 pk
    pk = load_pk_ensemble(
        data_dir=str(pk_dir),
        file_glob="pk_rsd_N*.dat",
        realization_min=REALIZATION_MIN,
        realization_max=REALIZATION_MAX,
        n_datapoints=N_DATAPOINTS,
        k_cen_col=0,
        k_min_col=1,
        k_max_col=2,
        p0_col=5,
    )
    pk = filter_singular_pk_bins(pk)

    # 2) 读取 pcf
    r_data, xi_mean, xi_std, used_pcf = read_pcf_mean_std(
        pcf_dir=str(pcf_dir),
        pcf_glob="pcf_rsd_N*.dat",
        realization_min=REALIZATION_MIN,
        realization_max=REALIZATION_MAX,
    )
    r2_data = r_data ** 2 * xi_mean
    r2_err = r_data ** 2 * xi_std
    sigma_floor = np.nanmedian(r2_err[r2_err > 0]) * 1e-6 if np.any(r2_err > 0) else 1e-12
    r2_err_safe = np.where(r2_err > 0, r2_err, sigma_floor)

    print(f"  pk mocks={pk.p0_mocks.shape[0]}, pcf mocks={len(used_pcf)}, k bins={pk.p0_mocks.shape[1]}")

    # 3) baseline 用标准拟合
    print("  [fit] standard desilike for baseline ...")
    bestfit_std = fit_standard_desilike(
        cosmo=cosmo,
        kcen=pk.kcen,
        kmin_b=pk.kmin,
        kmax_b=pk.kmax,
        pk_mean=pk.p0_mean,
        cov_mocks=pk.p0_mocks,
        p_fix=P_FIX,
        z=Z,
    )

    # 4) current 用 BinAvgFit
    print("  [fit] BinAvgFit for current ...")
    kmax_fit = float(pk.kmax[-1])
    qmax_fit = int(np.floor((kmax_fit / K_FUND) ** 2)) + 1
    gq_fit = gq_enumerate(qmax_fit)
    _, k_shell, g_shell = build_shells_for_kmax(K_FUND, gq_fit, kmax=kmax_fit)
    idx_per_bin = build_bin_shell_index(k_shell, g_shell, pk.kmin, pk.kmax)
    bestfit_binavg = fit_binavg_minuit(
        cosmo=cosmo,
        base_params=bestfit_std,
        k_shell=k_shell,
        g_shell=g_shell,
        idx_per_bin=idx_per_bin,
        pk_data=pk.p0_mean,
        cov_mocks=pk.p0_mocks,
        p_fix=P_FIX,
        z=Z,
    )

    print(
        "    std   : fnl=%.3f b1=%.4f sigmas=%.4f"
        % (float(bestfit_std["fnl_loc"]), float(bestfit_std["b1"]), float(bestfit_std["sigmas"]))
    )
    print(
        "    binavg: fnl=%.3f b1=%.4f sigmas=%.4f"
        % (float(bestfit_binavg["fnl_loc"]), float(bestfit_binavg["b1"]), float(bestfit_binavg["sigmas"]))
    )

    # 5) baseline 曲线：FFTLog + 硬截断 kmin = 2pi/L
    k_grid_baseline = np.geomspace(K_FUND / FFTLOG_PADDING, KMAX_BASELINE * FFTLOG_PADDING, FFTLOG_N)
    p0_baseline = eval_pk_dense(cosmo=cosmo, k=k_grid_baseline, params=bestfit_std, p_fix=P_FIX, z=Z)
    r_base, xi_base, _, _ = xi0_from_p0_fftlog(
        k_grid=k_grid_baseline,
        p0_grid=p0_baseline,
        kmin=K_FUND,
        kmax=KMAX_BASELINE,
        mu=0.5,
        bias=0.0,
        taper_frac=EDGE_TAPER_FRAC,
        use_param_ir_window=False,
        k_fund=K_FUND,
        x_power=4.0,
    )
    xi_model_base = np.interp(r_data, r_base[np.argsort(r_base)], xi_base[np.argsort(r_base)])

    # 6) current 曲线：BinAvgFit + CachedRebin(FastDiscrete)
    p0_current = eval_pk_dense(cosmo=cosmo, k=kd_discrete, params=bestfit_binavg, p_fix=P_FIX, z=Z)
    xi_model_current = xi_cached_rebin(
        s=r_data,
        g_nz=g_nz,
        k_eff=k_eff,
        volume=VOLUME,
        kd=kd_discrete,
        pd=p0_current,
    )

    r2_base = r_data ** 2 * xi_model_base
    r2_current = r_data ** 2 * xi_model_current
    ds_base = (r2_data - r2_base) / r2_err_safe
    ds_current = (r2_data - r2_current) / r2_err_safe

    return {
        "tag": tag,
        "r": r_data,
        "r2_data": r2_data,
        "r2_err": r2_err,
        "r2_base": r2_base,
        "r2_current": r2_current,
        "ds_base": ds_base,
        "ds_current": ds_current,
    }


def draw_masscut_figure(results: list[dict[str, object]], out_pdf: Path) -> None:
    """
    把所有 masscut 样本画成一张总图。

    参数
    ----
    results : list[dict]
        每个 tag 的绘图结果。
    out_pdf : Path
        输出 PDF 路径。
    """
    n_panel = len(results)
    n_col = 4
    n_pair = math.ceil(n_panel / n_col)
    n_row = 2 * n_pair

    fig, axes = plt.subplots(
        n_row,
        n_col,
        figsize=(20.0, 3.0 * n_row + 0.6),
        sharex=False,
        gridspec_kw={"height_ratios": [3.0, 1.35] * n_pair, "hspace": 0.10, "wspace": 0.22},
    )
    axes = np.atleast_2d(axes)

    for idx, res in enumerate(results):
        pair = idx // n_col
        col = idx % n_col
        ax = axes[2 * pair, col]
        ax_res = axes[2 * pair + 1, col]

        r = res["r"]
        ax.errorbar(
            r,
            res["r2_data"],
            yerr=res["r2_err"],
            fmt="o",
            ms=3.0,
            lw=0.8,
            capsize=1.8,
            color="black",
            ecolor="black",
            label="Measured mean",
            zorder=10,
        )
        ax.plot(r, res["r2_base"], color="tab:blue", lw=1.7, ls="--", label="Baseline FFTLog")
        ax.plot(r, res["r2_current"], color="#c62828", lw=2.0, label="BinAvgFit + FastDiscrete")
        ax.set_title(res["tag"])
        if col == 0:
            ax.set_ylabel(r"$r^2 \xi_0(r)$")
        ax.grid(alpha=0.28, ls="--")
        ax.tick_params(axis="x", labelbottom=False)

        ax_res.axhline(0.0, color="black", lw=0.8)
        ax_res.axhline(1.0, color="gray", ls=":", lw=0.7)
        ax_res.axhline(-1.0, color="gray", ls=":", lw=0.7)
        ax_res.axhline(2.0, color="#d98c00", ls=":", lw=0.7)
        ax_res.axhline(-2.0, color="#d98c00", ls=":", lw=0.7)
        ax_res.plot(r, res["ds_base"], color="tab:blue", lw=1.3, ls="--", marker="o", ms=2.1, label="Baseline")
        ax_res.plot(r, res["ds_current"], color="#c62828", lw=1.5, marker="o", ms=2.1, label="Current")
        ax_res.set_xlabel(r"$r\,[\mathrm{Mpc}/h]$")
        if col == 0:
            ax_res.set_ylabel(r"$(D-M)/\sigma$")
        ax_res.set_ylim(-3.0, 3.0)
        ax_res.grid(alpha=0.28, ls="--")

    # 隐藏多余空白子图
    total_slots = n_pair * n_col
    for idx in range(n_panel, total_slots):
        pair = idx // n_col
        col = idx % n_col
        axes[2 * pair, col].axis("off")
        axes[2 * pair + 1, col].axis("off")

    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.suptitle("Masscut Validation: Baseline FFTLog vs BinAvgFit + FastDiscrete", fontsize=15, y=0.985)
    fig.legend(handles, labels, frameon=False, ncol=3, loc="upper center", bbox_to_anchor=(0.5, 0.958))
    fig.subplots_adjust(left=0.05, right=0.995, bottom=0.06, top=0.92, wspace=0.22, hspace=0.10)
    fig.savefig(out_pdf)
    plt.close(fig)


def main() -> None:
    """
    主函数。

    执行逻辑
    --------
    1. 自动发现已有完整 masscut 样本；
    2. 逐个计算 baseline 与 current 两种模型；
    3. 输出总图 PDF。
    """
    t0 = time.time()
    cosmo = build_cosmology()

    # 只对已有完整 pk/pcf 数据的 tag 作图；mmin_1e15 会自动被跳过
    tags = discover_masscut_tags(MASSCUT_ROOT)
    if DROP_LAST_N_TAGS > 0 and len(tags) > DROP_LAST_N_TAGS:
        tags = tags[:-DROP_LAST_N_TAGS]
    print("usable tags:", tags)

    # 1Gpc 的 CachedRebin 缓存只需算一次
    qmax_fd = int((KMAX_DISCRETE / K_FUND) ** 2)
    nmax_fd = int(KMAX_DISCRETE / K_FUND)
    print(f"[cache] building 1Gpc g_q with qmax={qmax_fd}")
    gq_fd = gq_fft(qmax=qmax_fd, nmax=nmax_fd)
    g_nz, k_eff = precompute_rebin_cache(gq=gq_fd, kf=K_FUND, kmax=KMAX_DISCRETE, dk_factor=DK_FACTOR)
    kd_discrete = np.geomspace(K_FUND * 0.5, KMAX_DISCRETE * 1.1, N_DENSE)

    results = []
    for tag in tags:
        try:
            results.append(run_one_masscut(tag=tag, cosmo=cosmo, kd_discrete=kd_discrete, g_nz=g_nz, k_eff=k_eff))
        except Exception as exc:  # noqa: BLE001
            print(f"[SKIP] {tag}: {exc}")

    draw_masscut_figure(results=results, out_pdf=OUT_PDF)
    print(f"[OK] saved: {OUT_PDF}")
    print(f"Total elapsed: {time.time() - t0:.1f}s")


if __name__ == "__main__":
    main()
