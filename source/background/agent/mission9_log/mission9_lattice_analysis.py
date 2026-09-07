#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Mission 9 探索（A部分）：格点模式超额分析
==========================================

代码大纲
--------
1. 枚举三维离散 shell (k_q, g_q)，范围覆盖 3Gpc 和 1Gpc 盒子的数据 k-bin。
2. 计算累积模式数 N_disc(k) vs 连续近似 N_cont(k) = 4π/3 (k/k_f)^3。
3. 计算逐 shell 的"权重比"：discrete weight / continuous weight。
4. 用玩具 P(k) = A + B/k^2 估算格点超额对 xi_0 的影响。
5. 将离散模式映射到测量 P(k) 的 k-bin，分析每个 bin 内的模式加权平均。
6. 所有图和数据保存到 mission9_log 目录。

核心假说
--------
全离散求和系统性高估 xi_0 的原因：
(a) 低 k 处离散格点的模式数 g_q 超过连续近似 4πk^2Δk/k_f^3，
    在 PNG (1/k^2) 权重下这一超额被放大。
(b) P(k) 拟合时在 bin center 比较模型与数据，但数据的 bin-average
    是模式加权的，对于弯曲的 P(k) 这引入了系统偏差。

输入
----
- 无需 desilike，纯 numpy 分析。
- 需要 3Gpc 测量 P(k) 文件来确定 k-bin 结构。

输出
----
- mission9_lattice_excess_3gpc.png  : 格点超额图
- mission9_mode_weighted_bins.png   : 逐 bin 模式加权分析
- mission9_lattice_analysis.md      : 数值结论摘要
"""

import os
import sys
import glob
import re
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from collections import defaultdict

# ============================================================
# 参数
# ============================================================
OUT_DIR = os.path.dirname(os.path.abspath(__file__))

# 两个盒子
BOXES = {
    "3Gpc": {"L": 3000.0, "pk_glob": "/pscratch/sd/l/lzy/3gpc-UNIT-fastpm/pk_masscut/pk_rsd_3gpc_fnl100_N*.dat"},
    "1Gpc": {"L": 1000.0, "pk_glob": "/pscratch/sd/l/lzy/1gpc-UNIT-fnl100-fastpm/pk/pk_rsd_N*.dat"},
}

# 分析参数
QMAX_FOR_ANALYSIS = 2000   # 枚举到的最大 q（覆盖低 k 区域）
KMAX_GLOBAL = 15.0         # 全局 kmax
N_DATAPOINTS = 20          # 使用前多少个 k-bin


# ============================================================
# 辅助函数
# ============================================================

def enumerate_shells_direct(nmax, qmax=None):
    """
    直接枚举 (nx, ny, nz) 来统计每个 q = nx^2+ny^2+nz^2 的简并度 g_q。

    参数
    ----------
    nmax : int
        每个方向的扫描上限 [-nmax, nmax]。
    qmax : int, optional
        只保留 q <= qmax 的 shell。

    返回
    ----------
    dict : {q: g_q} 映射
    """
    shell_count = defaultdict(int)
    for nx in range(-nmax, nmax + 1):
        for ny in range(-nmax, nmax + 1):
            for nz in range(-nmax, nmax + 1):
                if nx == 0 and ny == 0 and nz == 0:
                    continue
                q = nx * nx + ny * ny + nz * nz
                if qmax is not None and q > qmax:
                    continue
                shell_count[q] += 1
    return dict(shell_count)


def read_pk_kbins(filepath, n_datapoints=20):
    """
    读取单个功率谱文件，返回 (kcen, kmin, kmax)。

    参数
    ----------
    filepath : str
        功率谱文件路径。
    n_datapoints : int
        使用前多少行。

    返回
    ----------
    tuple(ndarray, ndarray, ndarray)
        kcen, kmin, kmax 数组。
    """
    arr = np.loadtxt(filepath, comments='#')
    sl = slice(0, n_datapoints)
    return arr[sl, 0], arr[sl, 1], arr[sl, 2]


def parse_rid(fp):
    """从文件名解析 realization 编号。"""
    m = re.search(r'N([0-9]+)', os.path.basename(fp))
    return int(m.group(1)) if m else -1


# ============================================================
# 主分析
# ============================================================

def main():
    results_md = []
    results_md.append("# Mission 9：格点模式超额分析\n")

    for box_name, box_info in BOXES.items():
        L = box_info["L"]
        kf = 2.0 * np.pi / L
        print(f"\n{'='*60}")
        print(f"盒子: {box_name}, L={L}, k_f={kf:.6f}")
        print(f"{'='*60}")

        # 确定需要枚举的 nmax
        nmax = int(np.ceil(np.sqrt(QMAX_FOR_ANALYSIS))) + 1
        print(f"枚举 nmax={nmax}, qmax={QMAX_FOR_ANALYSIS}...")
        shell_dict = enumerate_shells_direct(nmax, qmax=QMAX_FOR_ANALYSIS)

        # 排序得到 (q, g_q) 数组
        q_arr = np.array(sorted(shell_dict.keys()), dtype=np.int64)
        g_arr = np.array([shell_dict[q] for q in q_arr], dtype=np.int64)
        k_arr = kf * np.sqrt(q_arr.astype(float))

        n_shells = len(q_arr)
        print(f"找到 {n_shells} 个非零 shell, q 范围 [{q_arr[0]}, {q_arr[-1]}]")
        print(f"k 范围 [{k_arr[0]:.6f}, {k_arr[-1]:.6f}] h/Mpc")

        # ----- 1. 累积模式数 vs 连续近似 -----
        cum_g = np.cumsum(g_arr)
        cum_cont = (4.0 * np.pi / 3.0) * (q_arr.astype(float)) ** 1.5
        excess_ratio = cum_g / cum_cont

        print("\n--- 累积模式数（前 15 个 shell）---")
        print(f"{'q':>5s} {'k/kf':>10s} {'g_q':>8s} {'Σg':>8s} {'N_cont':>10s} {'Σg/N_cont':>10s}")
        for i in range(min(15, n_shells)):
            print(f"{q_arr[i]:>5d} {np.sqrt(q_arr[i]):>10.4f} {g_arr[i]:>8d} "
                  f"{cum_g[i]:>8d} {cum_cont[i]:>10.2f} {excess_ratio[i]:>10.4f}")

        results_md.append(f"\n## {box_name} (L={L}, k_f={kf:.6f})\n")
        results_md.append("### 累积模式数 vs 连续近似\n")
        results_md.append(f"| q | k/k_f | g_q | Σg | N_cont | Σg/N_cont |\n")
        results_md.append(f"|---|-------|-----|------|--------|----------|\n")
        for i in range(min(20, n_shells)):
            results_md.append(
                f"| {q_arr[i]} | {np.sqrt(q_arr[i]):.4f} | {g_arr[i]} | "
                f"{cum_g[i]} | {cum_cont[i]:.1f} | {excess_ratio[i]:.4f} |\n"
            )

        # ----- 2. 格点超额图 -----
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        fig.suptitle(f"{box_name}: Lattice Mode Excess Analysis", fontsize=14)

        # 2a) 累积模式数
        ax = axes[0, 0]
        ax.plot(np.sqrt(q_arr[:200]), cum_g[:200], 'b-', lw=1.5, label=r"Discrete $\sum g_q$")
        ax.plot(np.sqrt(q_arr[:200]), cum_cont[:200], 'r--', lw=1.5, label=r"Continuous $\frac{4\pi}{3}q^{3/2}$")
        ax.set_xlabel(r"$k/k_f = \sqrt{q}$")
        ax.set_ylabel("Cumulative mode count")
        ax.legend()
        ax.set_title("Cumulative mode count")
        ax.grid(alpha=0.3)

        # 2b) 超额比
        ax = axes[0, 1]
        ax.plot(np.sqrt(q_arr[:200]), excess_ratio[:200], 'k-', lw=1.0)
        ax.axhline(1.0, color='red', ls='--', lw=1)
        ax.set_xlabel(r"$k/k_f = \sqrt{q}$")
        ax.set_ylabel(r"$\sum g_q / N_{cont}$")
        ax.set_title("Excess ratio (cumulative)")
        ax.set_ylim(0.8, 1.6)
        ax.grid(alpha=0.3)

        # 2c) 用 1/k^2 权重的累积贡献
        # 模拟 PNG：P(k) ~ C + B/k^2，只关注 B/k^2 部分
        # 离散贡献: Σ g_q * (1/k_q^2) = Σ g_q / (kf^2 q)
        # 连续贡献: ∫ k^2/(2π^2) * (1/k^2) dk = ∫ dk/(2π^2) = Δk/(2π^2)
        png_weight_disc = np.cumsum(g_arr / q_arr.astype(float)) / (kf ** 2)  # 归一化因子
        # 对应的连续近似：∫_0^{k} dk * (1/k^2) * k^2/(2π^2) = k/(2π^2)
        # 但这不对，因为 P(k)=1/k^2 时 ∫ k^2/(2π^2) * (1/k^2) j0(kr) dk = ∫ j0(kr)/(2π^2) dk
        # 累积（不含 j0）：∫_0^K dk/(2π^2) = K/(2π^2)
        # 对于离散版本：(1/V) Σ g_q * (1/k_q^2) = Σ g_q/(V kf^2 q)
        # 由于 V = L^3 = (2π/kf)^3:
        # (1/V) Σ g_q/(kf^2 q) = kf^3/(2π)^3 * Σ g_q/(kf^2 q) = kf/(8π^3) * Σ g_q/q
        # 连续版本：∫_{kf}^{K} dk/(2π^2) = (K-kf)/(2π^2)
        # 比较这两个就能看到 PNG 权重下的超额

        cum_png_disc = kf / (8 * np.pi**3) * np.cumsum(g_arr / q_arr.astype(float))
        K_vals = k_arr
        cum_png_cont = (K_vals - kf) / (2 * np.pi**2)
        cum_png_cont = np.maximum(cum_png_cont, 0)

        ax = axes[1, 0]
        ax.plot(np.sqrt(q_arr[:200]), cum_png_disc[:200], 'b-', lw=1.5, label="Discrete (PNG weight)")
        ax.plot(np.sqrt(q_arr[:200]), cum_png_cont[:200], 'r--', lw=1.5, label="Continuous (PNG weight)")
        ax.set_xlabel(r"$k/k_f = \sqrt{q}$")
        ax.set_ylabel(r"Cumulative $1/k^2$-weighted contribution")
        ax.legend()
        ax.set_title(r"PNG-weighted cumulative (P(k)$\propto 1/k^2$)")
        ax.grid(alpha=0.3)

        # 2d) PNG权重下的超额比
        ax = axes[1, 1]
        with np.errstate(divide='ignore', invalid='ignore'):
            png_ratio = np.where(cum_png_cont > 0, cum_png_disc / cum_png_cont, np.nan)
        ax.plot(np.sqrt(q_arr[:200]), png_ratio[:200], 'k-', lw=1.0)
        ax.axhline(1.0, color='red', ls='--', lw=1)
        ax.set_xlabel(r"$k/k_f = \sqrt{q}$")
        ax.set_ylabel("Discrete / Continuous (PNG weight)")
        ax.set_title(r"PNG-weighted excess ratio")
        ax.set_ylim(0.5, 2.5)
        ax.grid(alpha=0.3)

        plt.tight_layout()
        fig_path = os.path.join(OUT_DIR, f"mission9_lattice_excess_{box_name.lower()}.png")
        plt.savefig(fig_path, dpi=150)
        plt.close()
        print(f"\n图已保存: {fig_path}")

        # ----- 3. K-bin 内的模式分布分析 -----
        pk_files = sorted(glob.glob(box_info["pk_glob"]), key=parse_rid)
        if not pk_files:
            print(f"警告: 未找到 {box_name} 的 pk 文件")
            continue

        kcen, kmin, kmax_bins = read_pk_kbins(pk_files[0], N_DATAPOINTS)
        print(f"\n数据 k-bin 结构: {len(kcen)} bins, k=[{kcen[0]:.5f}, {kcen[-1]:.5f}]")

        results_md.append(f"\n### K-bin 内模式分析\n")
        results_md.append("| bin | k_center | k_min | k_max | N_modes | k_mode_weighted | Δk/k_center | N_excess |\n")
        results_md.append("|-----|----------|-------|-------|---------|-----------------|-------------|----------|\n")

        bin_stats = []
        for ib in range(len(kcen)):
            # 找到落在此 bin 内的离散模式
            mask = (k_arr >= kmin[ib]) & (k_arr < kmax_bins[ib])
            k_in_bin = k_arr[mask]
            g_in_bin = g_arr[mask]
            n_modes_in_bin = np.sum(g_in_bin)

            if n_modes_in_bin > 0:
                k_mw = np.sum(g_in_bin * k_in_bin) / n_modes_in_bin  # 模式加权平均 k
                # 连续近似模式数: ∫_{kmin}^{kmax} 4πk^2/kf^3 dk = 4π/(3kf^3)*(kmax^3-kmin^3)
                n_cont = 4 * np.pi / (3 * kf**3) * (kmax_bins[ib]**3 - kmin[ib]**3)
                n_excess = n_modes_in_bin / n_cont if n_cont > 0 else np.nan
            else:
                k_mw = kcen[ib]
                n_cont = 0
                n_excess = np.nan

            dk_over_k = (k_mw - kcen[ib]) / kcen[ib] if kcen[ib] > 0 else 0

            bin_stats.append({
                'bin': ib,
                'kcen': kcen[ib],
                'kmin': kmin[ib],
                'kmax': kmax_bins[ib],
                'n_modes': n_modes_in_bin,
                'k_mw': k_mw,
                'dk_over_k': dk_over_k,
                'n_excess': n_excess,
                'n_shells': int(np.sum(mask)),
            })

            results_md.append(
                f"| {ib} | {kcen[ib]:.5f} | {kmin[ib]:.5f} | {kmax_bins[ib]:.5f} | "
                f"{n_modes_in_bin} | {k_mw:.5f} | {dk_over_k:+.4f} | {n_excess:.3f} |\n"
            )

        # 3b) 画 bin 内模式统计
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        fig.suptitle(f"{box_name}: Mode Distribution within K-bins", fontsize=14)

        bins_arr = np.arange(len(bin_stats))
        n_modes_arr = np.array([b['n_modes'] for b in bin_stats])
        dk_arr = np.array([b['dk_over_k'] for b in bin_stats])
        n_excess_arr = np.array([b['n_excess'] for b in bin_stats])

        # 模式数 per bin
        ax = axes[0, 0]
        ax.bar(bins_arr, n_modes_arr, color='steelblue', alpha=0.7)
        ax.set_xlabel("Bin index")
        ax.set_ylabel("Number of modes in bin")
        ax.set_title("Mode count per k-bin")
        ax.grid(alpha=0.3)

        # k偏移
        ax = axes[0, 1]
        ax.plot(bins_arr, dk_arr * 100, 'o-', color='darkred', ms=4)
        ax.axhline(0, color='gray', ls='--')
        ax.set_xlabel("Bin index")
        ax.set_ylabel(r"$(k_{mw} - k_{center})/k_{center}$ [%]")
        ax.set_title("Mode-weighted k offset from bin center")
        ax.grid(alpha=0.3)

        # 模式数超额比
        ax = axes[1, 0]
        ax.plot(bins_arr, n_excess_arr, 'o-', color='darkgreen', ms=4)
        ax.axhline(1.0, color='red', ls='--')
        ax.set_xlabel("Bin index")
        ax.set_ylabel(r"$N_{disc}/N_{cont}$")
        ax.set_title("Mode count: discrete / continuous")
        ax.grid(alpha=0.3)

        # 对 PNG 影响的估计：Δk/k * 2 ≈ P(k) 的相对偏差
        # 对于 P(k) ~ B/k^2，δP/P = -2 δk/k
        ax = axes[1, 1]
        pk_bias_pct = -2 * dk_arr * 100  # P(k) ∝ 1/k^2 时的相对偏差
        ax.plot(bins_arr, pk_bias_pct, 's-', color='purple', ms=4)
        ax.axhline(0, color='gray', ls='--')
        ax.set_xlabel("Bin index")
        ax.set_ylabel(r"Estimated $\delta P_{model}/P$ [%] from k-offset")
        ax.set_title(r"Estimated model bias (assuming $P \propto 1/k^2$)")
        ax.grid(alpha=0.3)

        plt.tight_layout()
        fig_path = os.path.join(OUT_DIR, f"mission9_mode_weighted_bins_{box_name.lower()}.png")
        plt.savefig(fig_path, dpi=150)
        plt.close()
        print(f"图已保存: {fig_path}")

        # 打印详细统计
        print(f"\n--- K-bin 模式统计 ({box_name}) ---")
        print(f"{'bin':>4s} {'k_cen':>8s} {'N_modes':>8s} {'N_shells':>9s} {'k_mw':>8s} "
              f"{'Δk/k%':>8s} {'N_disc/cont':>11s}")
        for b in bin_stats:
            print(f"{b['bin']:>4d} {b['kcen']:>8.5f} {b['n_modes']:>8d} {b['n_shells']:>9d} "
                  f"{b['k_mw']:>8.5f} {b['dk_over_k']*100:>+8.3f} {b['n_excess']:>11.3f}")

        # ----- 4. PNG 权重下的估算 -----
        # 用 P(k)=C/k^2 估算模式加权 vs 中心值的偏差对 fnl 拟合的影响
        results_md.append(f"\n### PNG 偏差估计\n")
        results_md.append("对于 P(k) ∝ 1/k^2 部分，模式加权平均偏差对 best-fit 的影响：\n\n")

        total_bias_contribution = 0
        for b in bin_stats[:5]:  # 只看前 5 个低 k bin
            if b['n_modes'] > 0:
                mask = (k_arr >= b['kmin']) & (k_arr < b['kmax'])
                k_in = k_arr[mask]
                g_in = g_arr[mask]
                # 1/k^2 的模式加权平均 vs 中心值
                pk_mw = np.sum(g_in / k_in**2) / np.sum(g_in)
                pk_center = 1.0 / b['kcen']**2
                bias_pct = (pk_mw / pk_center - 1) * 100
                total_bias_contribution += bias_pct
                results_md.append(f"- bin {b['bin']}: <1/k²>_mw / (1/k²_center) - 1 = {bias_pct:+.2f}%\n")
                print(f"bin {b['bin']}: <1/k²>_mw / (1/k²_center) - 1 = {bias_pct:+.2f}%")

        results_md.append(f"\n前 5 bin 累积偏差: {total_bias_contribution:+.2f}%\n")
        print(f"\n前 5 bin 累积偏差贡献: {total_bias_contribution:+.2f}%")

    # ----- 5. 保存总结 -----
    results_md.append("\n## 结论\n")
    results_md.append("见上方各盒子的数值分析结果。核心关注：\n")
    results_md.append("1. 在 q < 10 (最低 k-shell) 处，离散模式数显著超过连续近似\n")
    results_md.append("2. 这一超额在 PNG 1/k^2 权重下被放大\n")
    results_md.append("3. K-bin 内的模式加权平均 k 与 bin center 的偏移可能导致 P(k) 拟合偏差\n")

    md_path = os.path.join(OUT_DIR, "mission9_lattice_analysis.md")
    with open(md_path, 'w') as f:
        f.writelines(results_md)
    print(f"\n总结已保存: {md_path}")


if __name__ == "__main__":
    main()
