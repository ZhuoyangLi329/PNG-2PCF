#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
meeting-fig: 当前最佳 2PCF 理论建模方法下的 fnl 对比图
=====================================================

代码大纲
--------
1. 读取当前项目“最佳建模方法”所需的数值配置；
2. 复用 mission10/11 的核心思想：
   - 用 desilike 的 PNGTracerPowerSpectrumMultipoles 计算理论 P0(k)
   - 用 CachedRebin / FastDiscrete 近似 FullDiscrete 求和得到 xi0(r)
3. 固定参数 `L=3000, b1=2, p=1, sigmas=0, sn0=0, z=1`；
4. 分别计算 `fnl=0, +100, -100` 的理论 2PCF monopole；
5. 画单张主图：`r^2 * xi0(r)`；
6. 只保存 PNG，并把 3Gpc 的 CachedRebin 缓存保存到本地文件，便于复用。

逻辑关系
--------
- `load_r_grid_from_measurement` 负责读取 3Gpc 测量数据的 r bin；
- `precompute_rebin_cache` / `load_or_build_cached_rebin` 负责当前方法的离散求和缓存；
- `fast_discrete_xi0` 负责把理论 P0(k) 转成 xi0(r)；
- `main` 负责组装参数、计算三条理论曲线并画图。

说明
----
- 这里没有做 BinAvgFit，因为你现在要的是“固定参数的纯理论曲线对比”，
  不涉及用测量功率谱去拟合参数。当前最佳方法在这种场景下对应的是：
      参数 -> P_model(k) -> FastDiscrete / CachedRebin -> xi0(r)
- 为了与当前 3Gpc 建模口径保持一致，这里采用：
      kmax = 15 h/Mpc
      z    = 1.0
      sigmas = 0
      sn0    = 0
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


# ============================================================
# 一、路径与 mission10 复用函数
# ============================================================
THIS_DIR = Path(__file__).resolve().parent
AGENT_DIR = THIS_DIR.parent
MISSION10_DIR = AGENT_DIR / "mission10_log"
if str(MISSION10_DIR) not in sys.path:
    sys.path.insert(0, str(MISSION10_DIR))

from mission10_binavgfit_full_discrete_compare import eval_pk_dense, gq_fft  # noqa: E402


# ============================================================
# 二、全局参数
# ============================================================
BOX_SIZE = 3000.0
UNIT_Z = 1.0
B1_VALUE = 2.0
P_VALUE = 1.0
SIGMAS_VALUE = 0.0
SN0_VALUE = 0.0
KMAX_DISCRETE = 15.0
N_DENSE = 300_000
DK_FACTOR = 0.1

R_GRID_SOURCE = Path("/pscratch/sd/l/lzy/3gpc-UNIT-fastpm/pcf_masscut/pcf_rsd_3gpc_fnl0_N2.dat")

OUT_PNG = THIS_DIR / "theory_2pcf_current_method_L3000_b2_p1_fnl_compare.png"
CACHE_NPZ = THIS_DIR / "cache_cachedrebin_L3000_kmax15_dk0p1.npz"

SERIES = [
    {"fnl": -100.0, "label": r"$f_{\mathrm{NL}}=-100$", "color": "blue"},
    {"fnl": 0.0, "label": r"$f_{\mathrm{NL}}=0$", "color": "gray"},
    {"fnl": 100.0, "label": r"$f_{\mathrm{NL}}=100$", "color": "#d62728"},
]

matplotlib.rcParams.update({
    "font.size": 11,
    "font.family": "serif",
    "font.serif": ["DejaVu Serif", "Times New Roman", "Times"],
    "mathtext.fontset": "stix",
    "axes.spines.top": True,
    "axes.spines.right": True,
    "savefig.dpi": 300,
    "figure.dpi": 150,
})


def load_r_grid_from_measurement(path: Path) -> np.ndarray:
    """
    从已有 3Gpc 测量 2PCF 文件中读取 r bin 中心。

    参数
    ----
    path : Path
        任意一个 3Gpc pcf 文件路径。

    返回
    ----
    ndarray
        r 网格（一维数组，单位 Mpc/h）。
    """
    arr = np.loadtxt(path, comments="#")
    if arr.ndim != 2 or arr.shape[1] < 1:
        raise ValueError(f"PCF 文件格式异常：{path}")
    return np.asarray(arr[:, 0], dtype=np.float64)


def precompute_rebin_cache(gq: np.ndarray, kf: float, kmax: float, dk_factor: float = 0.1) -> tuple[np.ndarray, np.ndarray]:
    """
    预计算 CachedRebin 所需的模式总数与有效 k。

    背景
    ----
    当前最佳 2PCF 数值建模方法的预测阶段是：
        xi0(r) = (1 / V) * sum_q g_q * P_model(k_q) * j0(k_q r)

    直接逐壳层求和在 3Gpc, kmax=15 时会非常慢，所以 mission11 用
    “k-rebinning / CachedRebin” 把相近的壳层先压缩到更少的 bin。

    参数
    ----
    gq : ndarray
        壳层简并度数组，gq[q] 表示满足 n_x^2 + n_y^2 + n_z^2 = q 的整数三元组个数。
    kf : float
        基模 2π/L。
    kmax : float
        最大离散求和波数。
    dk_factor : float
        重分箱宽度，定义为 dk = dk_factor * kf。

    返回
    ----
    tuple[ndarray, ndarray]
        `g_nz, k_eff`：
        - g_nz: 每个非空重分箱里的总模式数
        - k_eff: 每个非空重分箱里的加权平均 k
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


def load_or_build_cached_rebin(box_size: float, kmax: float, dk_factor: float, cache_file: Path) -> dict[str, np.ndarray | float]:
    """
    读取或生成 CachedRebin 的缓存文件。

    参数
    ----
    box_size : float
        盒长 L。
    kmax : float
        离散求和最大 k。
    dk_factor : float
        重分箱宽度系数。
    cache_file : Path
        缓存文件路径（npz）。

    返回
    ----
    dict
        包含 `kf`, `g_nz`, `k_eff` 三项。
    """
    kf = 2.0 * np.pi / box_size

    if cache_file.exists():
        cache = np.load(cache_file)
        cache_kf = float(cache["kf"])
        cache_kmax = float(cache["kmax"])
        cache_dk_factor = float(cache["dk_factor"])
        if (
            np.isclose(cache_kf, kf, rtol=0.0, atol=1e-15)
            and np.isclose(cache_kmax, kmax, rtol=0.0, atol=1e-12)
            and np.isclose(cache_dk_factor, dk_factor, rtol=0.0, atol=1e-12)
        ):
            print(f"[cache] load from {cache_file}")
            return {
                "kf": cache_kf,
                "g_nz": np.asarray(cache["g_nz"], dtype=np.float64),
                "k_eff": np.asarray(cache["k_eff"], dtype=np.float64),
            }

    print(f"[cache] build for L={box_size:.0f}, kmax={kmax:.1f}, dk_factor={dk_factor}")
    qmax = int((kmax / kf) ** 2)
    nmax = int(kmax / kf)
    gq = gq_fft(qmax=qmax, nmax=nmax)
    g_nz, k_eff = precompute_rebin_cache(gq=gq, kf=kf, kmax=kmax, dk_factor=dk_factor)

    np.savez_compressed(
        cache_file,
        kf=np.array(kf, dtype=np.float64),
        kmax=np.array(kmax, dtype=np.float64),
        dk_factor=np.array(dk_factor, dtype=np.float64),
        g_nz=np.asarray(g_nz, dtype=np.float64),
        k_eff=np.asarray(k_eff, dtype=np.float64),
    )
    print(f"[cache] saved to {cache_file}")
    print(f"[cache] rebinned non-empty bins = {len(g_nz)}")
    return {"kf": kf, "g_nz": g_nz, "k_eff": k_eff}


def fast_discrete_xi0(s: np.ndarray, g_nz: np.ndarray, k_eff: np.ndarray, volume: float, kd: np.ndarray, pd: np.ndarray) -> np.ndarray:
    """
    用 CachedRebin / FastDiscrete 方法快速计算 xi0(r)。

    参数
    ----
    s : ndarray
        目标 r 网格。
    g_nz : ndarray
        每个非空重分箱里的总模式数。
    k_eff : ndarray
        每个非空重分箱里的有效 k。
    volume : float
        盒子体积 L^3。
    kd : ndarray
        稠密 k 网格。
    pd : ndarray
        理论 P0(kd)。

    返回
    ----
    ndarray
        对应 s 网格上的 xi0(r)。
    """
    weights = g_nz * np.interp(k_eff, kd, pd)
    kr = np.outer(k_eff, s)
    j0 = np.ones_like(kr)
    mask = kr != 0.0
    j0[mask] = np.sin(kr[mask]) / kr[mask]
    return (weights @ j0) / volume


def build_theory_params(fnl_value: float, b1_value: float, sigmas_value: float) -> dict[str, float]:
    """
    组装 desilike 理论模型所需的参数字典。

    参数
    ----
    fnl_value : float
        局域型 PNG 参数 fnl_loc。
    b1_value : float
        线性 bias 参数 b1。
    sigmas_value : float
        Fingers-of-God / damping 参数 sigmas。

    返回
    ----
    dict[str, float]
        可直接传给 `eval_pk_dense` 的参数字典。
    """
    return {
        "fnl_loc": float(fnl_value),
        "b1": float(b1_value),
        "sigmas": float(sigmas_value),
        "sn0": float(SN0_VALUE),
    }


def main() -> None:
    """
    主函数。

    执行步骤
    --------
    1. 加载 3Gpc 的 r 网格；
    2. 读取或创建 CachedRebin 缓存；
    3. 构造当前项目统一使用的 fiducial cosmology；
    4. 计算 fnl=-100/0/100 的理论 P0(k)；
    5. 用 FastDiscrete 求出 xi0(r)；
    6. 画图并保存。
    """
    from cosmoprimo import Cosmology

    r_grid = load_r_grid_from_measurement(R_GRID_SOURCE)
    volume = BOX_SIZE ** 3

    cache = load_or_build_cached_rebin(
        box_size=BOX_SIZE,
        kmax=KMAX_DISCRETE,
        dk_factor=DK_FACTOR,
        cache_file=CACHE_NPZ,
    )

    cosmo = Cosmology(
        h=0.6711,
        Omega_b=0.049,
        Omega_cdm=0.3175 - 0.049,
        sigma8=0.834,
        n_s=0.9624,
        engine="class",
    )

    kd = np.geomspace(float(cache["kf"]) * 0.5, KMAX_DISCRETE * 1.1, N_DENSE)

    curves: list[dict[str, np.ndarray | float | str]] = []
    for series in SERIES:
        params = build_theory_params(
            fnl_value=float(series["fnl"]),
            b1_value=B1_VALUE,
            sigmas_value=SIGMAS_VALUE,
        )
        pd = eval_pk_dense(
            cosmo=cosmo,
            k=kd,
            params=params,
            p_fix=P_VALUE,
            z=UNIT_Z,
        )
        xi = fast_discrete_xi0(
            s=r_grid,
            g_nz=np.asarray(cache["g_nz"], dtype=np.float64),
            k_eff=np.asarray(cache["k_eff"], dtype=np.float64),
            volume=volume,
            kd=kd,
            pd=pd,
        )
        curves.append({
            "fnl": float(series["fnl"]),
            "label": str(series["label"]),
            "color": str(series["color"]),
            "r2xi": r_grid ** 2 * xi,
        })
        print(
            "[curve] fnl=%+.0f: xi0(first)=%.6e, xi0(last)=%.6e"
            % (float(series["fnl"]), float(xi[0]), float(xi[-1]))
        )

    fig, ax = plt.subplots(figsize=(11.0, 6.3))

    for curve in curves:
        ax.plot(
            r_grid,
            np.asarray(curve["r2xi"]),
            lw=2.4,
            color=str(curve["color"]),
            label=str(curve["label"]),
        )

    ax.set_xlabel(r"$r\,[\mathrm{Mpc}/h]$", fontsize=22)
    ax.set_ylabel(r"$r^2 \xi_0(r)$", fontsize=22)
    ax.set_xlim(float(r_grid.min()), 250.0)
    ax.spines["top"].set_visible(True)
    ax.spines["right"].set_visible(True)
    ax.tick_params(axis="both", which="major", labelsize=22)
    ax.grid(alpha=0.28, ls="--")
    ax.legend(frameon=False, ncol=1, loc="best", fontsize=22)

    fig.tight_layout()
    fig.savefig(OUT_PNG)
    plt.close(fig)

    print(f"[OK] saved png: {OUT_PNG}")


if __name__ == "__main__":
    main()
