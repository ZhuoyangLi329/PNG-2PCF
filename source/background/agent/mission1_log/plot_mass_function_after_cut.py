#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
脚本名称
--------
plot_mass_function_after_cut.py

代码大纲（执行逻辑）
--------------------
1) 扫描可用 realization：
   - 1Gpc: /pscratch/.../fofs/fof_N*/fof_0.5000/LL-0.200
   - 3Gpc fnl0/fnl100: /global/cfs/.../reduced_catalogs/{tag}/halos_fastpm_N*.gz
2) 对每个 realization 读取 halo 质量，应用统一质量筛选 [1.4e13, 1e16] Msun/h，
   计算该 realization 的质量函数 dn/dlog10M。
3) 在每个样本集合内（1Gpc、3Gpc fnl0、3Gpc fnl100）对所有 realization 做平均。
4) 只输出一张 PNG 对比图，不额外导出 CSV/TXT。
"""

from __future__ import annotations

from pathlib import Path
import re

import numpy as np
import matplotlib.pyplot as plt
import bigfile


# =====================
# 参数区
# =====================
OUT_DIR = Path("/pscratch/sd/l/lzy/1gpc-UNIT-fnl100-fastpm/agent/mission1_log")
OUT_DIR.mkdir(parents=True, exist_ok=True)
OUT_PNG = OUT_DIR / "mass_function_after_cut_compare.png"

# 统一质量范围（按任务书修补要求）
MASS_MIN = 1.4e13
MASS_MAX = 1e16

# 体积
VOLUME_1GPC = 1000.0**3
VOLUME_3GPC = 3000.0**3

# 1Gpc 质量来源
M_PART_1GPC = 9.9775e9
FOF_ROOT_1GPC = Path("/pscratch/sd/l/lzy/1gpc-UNIT-fnl100-fastpm/fofs")
FOF_REL_PATH = Path("fof_0.5000/LL-0.200")

# 3Gpc 质量来源
CAT_ROOT_3GPC = Path("/global/cfs/cdirs/desi/mocks/UNIT/fastpm_3gpc_fnl/reduced_catalogs")

# mass function bins
NBINS = 24
MASS_EDGES = np.geomspace(MASS_MIN, MASS_MAX, NBINS + 1)
MASS_CEN = np.sqrt(MASS_EDGES[:-1] * MASS_EDGES[1:])
DLOG10M = np.log10(MASS_EDGES[1:]) - np.log10(MASS_EDGES[:-1])


def parse_seed_from_name(name: str) -> int:
    """
    从文件/目录名中提取 realization 编号。

    参数
    ----
    name : str
        例如 fof_N27 或 halos_fastpm_N27.gz。

    返回
    ----
    int
        realization 编号。
    """
    match = re.search(r"_N(\d+)", name)
    if not match:
        raise ValueError(f"无法从名字中解析 realization: {name}")
    return int(match.group(1))


def list_1gpc_catalogs() -> list[Path]:
    """
    列出 1Gpc 可用的 FOF catalog 路径（按 realization 升序）。

    返回
    ----
    list[Path]
        每个元素是 LL-0.200 的 bigfile 目录路径。
    """
    catalogs: list[Path] = []
    for d in sorted(FOF_ROOT_1GPC.glob("fof_N*"), key=lambda p: parse_seed_from_name(p.name)):
        cat = d / FOF_REL_PATH
        if cat.exists():
            catalogs.append(cat)
    return catalogs


def list_3gpc_catalogs(tag: str) -> list[Path]:
    """
    列出 3Gpc 某个 tag 的可用 catalog 文本路径（按 realization 升序）。

    参数
    ----
    tag : str
        'fnl0' 或 'fnl100'。

    返回
    ----
    list[Path]
        对应 tag 的 halos_fastpm_N*.gz 路径列表。
    """
    base = CAT_ROOT_3GPC / tag
    files = sorted(base.glob("halos_fastpm_N*.gz"), key=lambda p: parse_seed_from_name(p.name))
    return [p for p in files if p.exists()]


def hmf_counts_from_mass(mass: np.ndarray) -> np.ndarray:
    """
    对单个 realization 的质量数组做统一质量筛选并统计质量 bin 计数。

    参数
    ----
    mass : np.ndarray
        质量数组，单位 Msun/h。

    返回
    ----
    np.ndarray
        每个质量 bin 的计数，shape=(NBINS,)。
    """
    mask = (mass >= MASS_MIN) & (mass <= MASS_MAX)
    counts, _ = np.histogram(mass[mask], bins=MASS_EDGES)
    return counts.astype(np.float64)


def hmf_counts_1gpc_one(cat_path: Path) -> np.ndarray:
    """
    读取 1Gpc 单个 realization 的 bigfile（Length），并得到质量函数计数。

    参数
    ----
    cat_path : Path
        例如 .../fof_N1/fof_0.5000/LL-0.200

    返回
    ----
    np.ndarray
        该 realization 在统一质量筛选后的各 bin 计数。
    """
    bf = bigfile.BigFile(str(cat_path))
    length = bf["Length"][:].astype(np.float64)
    mass = length * M_PART_1GPC
    return hmf_counts_from_mass(mass)


def hmf_counts_3gpc_one(cat_path: Path) -> np.ndarray:
    """
    读取 3Gpc 单个 realization 的文本 catalog（第7列质量），并得到质量函数计数。

    参数
    ----
    cat_path : Path
        例如 .../halos_fastpm_N2.gz

    返回
    ----
    np.ndarray
        该 realization 在统一质量筛选后的各 bin 计数。
    """
    mass = np.loadtxt(cat_path, usecols=(6,), dtype=np.float64)
    if mass.size == 0:
        return np.zeros(NBINS, dtype=np.float64)
    if mass.max() < 1e10:
        mass *= 1e10
    return hmf_counts_from_mass(mass)


def mean_hmf_for_group(catalogs: list[Path], volume: float, loader: str) -> tuple[np.ndarray, np.ndarray]:
    """
    对一组 realization 计算平均质量函数。

    参数
    ----
    catalogs : list[Path]
        该组全部 realization 文件列表。
    volume : float
        盒子体积，单位 (Mpc/h)^3。
    loader : str
        '1gpc' 或 '3gpc'，用于选择读取函数。

    返回
    ----
    tuple[np.ndarray, np.ndarray]
        (hmf_mean, hmf_std)，两者 shape 都是 (NBINS,)。
    """
    if not catalogs:
        raise RuntimeError(f"没有可用 catalog: loader={loader}")

    all_hmf: list[np.ndarray] = []
    ncat = len(catalogs)
    for i, cat in enumerate(catalogs, start=1):
        if loader == "1gpc":
            counts = hmf_counts_1gpc_one(cat)
        elif loader == "3gpc":
            counts = hmf_counts_3gpc_one(cat)
        else:
            raise ValueError(f"未知 loader: {loader}")

        hmf = counts / (volume * DLOG10M)
        all_hmf.append(hmf)

        # 进度打印：保证长作业可监控
        seed = parse_seed_from_name(cat.name if loader == "3gpc" else cat.parent.parent.name)
        print(f"[{loader}] {i:3d}/{ncat:3d} done: N{seed}")

    arr = np.vstack(all_hmf)  # shape=(Nreal, NBINS)
    hmf_mean = arr.mean(axis=0)
    hmf_std = arr.std(axis=0, ddof=1) if arr.shape[0] > 1 else np.zeros_like(hmf_mean)
    return hmf_mean, hmf_std


def main() -> None:
    """主流程：全 realization 平均，并输出一张 PNG。"""
    cats_1gpc = list_1gpc_catalogs()
    cats_3gpc_fnl0 = list_3gpc_catalogs("fnl0")
    cats_3gpc_fnl100 = list_3gpc_catalogs("fnl100")

    print("=== realization 统计 ===")
    print(f"1Gpc  catalog 数量: {len(cats_1gpc)}")
    print(f"3Gpc fnl0   数量: {len(cats_3gpc_fnl0)}")
    print(f"3Gpc fnl100 数量: {len(cats_3gpc_fnl100)}")

    h1, s1 = mean_hmf_for_group(cats_1gpc, VOLUME_1GPC, loader="1gpc")
    h0, s0 = mean_hmf_for_group(cats_3gpc_fnl0, VOLUME_3GPC, loader="3gpc")
    h100, s100 = mean_hmf_for_group(cats_3gpc_fnl100, VOLUME_3GPC, loader="3gpc")

    # 仅输出一个 PNG：上面画平均 HMF，下面画相对 fnl100 的比值
    fig, axes = plt.subplots(
        2,
        1,
        figsize=(8.6, 8.4),
        sharex=True,
        gridspec_kw={"height_ratios": [3, 1.4]},
    )

    ax = axes[0]
    ax.loglog(MASS_CEN, h1, "o-", ms=4, lw=1.4, label=f"1Gpc mean ({len(cats_1gpc)} realizations)")
    ax.loglog(MASS_CEN, h0, "s-", ms=4, lw=1.4, label=f"3Gpc fnl0 mean ({len(cats_3gpc_fnl0)})")
    ax.loglog(MASS_CEN, h100, "^-", ms=4, lw=1.4, label=f"3Gpc fnl100 mean ({len(cats_3gpc_fnl100)})")

    # 轻量误差带：用 realization 间标准差表示散布（不是误差棒拟合）
    ax.fill_between(MASS_CEN, np.maximum(h1 - s1, 1e-30), h1 + s1, alpha=0.15)
    ax.fill_between(MASS_CEN, np.maximum(h0 - s0, 1e-30), h0 + s0, alpha=0.15)
    ax.fill_between(MASS_CEN, np.maximum(h100 - s100, 1e-30), h100 + s100, alpha=0.15)

    ax.set_ylabel(r"$\langle dn/d\log_{10}M \rangle\ [h^3\,\mathrm{Mpc}^{-3}]$")
    ax.set_title("Halo Mass Function After Unified Mass Cut [1.4e13, 1e16]")
    ax.grid(True, which="both", ls="--", alpha=0.3)
    ax.legend()

    ax = axes[1]
    ratio_1 = h1 / np.maximum(h100, 1e-30)
    ratio_0 = h0 / np.maximum(h100, 1e-30)
    ax.semilogx(MASS_CEN, ratio_1, "o-", ms=3.8, lw=1.2, label="1Gpc / 3Gpc fnl100")
    ax.semilogx(MASS_CEN, ratio_0, "s-", ms=3.8, lw=1.2, label="3Gpc fnl0 / fnl100")
    ax.axhline(1.0, color="k", lw=1.0, ls="--")
    ax.set_xlabel(r"$M\ [M_{\odot}/h]$")
    ax.set_ylabel("Ratio")
    ax.grid(True, which="both", ls="--", alpha=0.3)
    ax.legend()

    fig.tight_layout()
    fig.savefig(OUT_PNG, bbox_inches="tight", dpi=180)
    plt.close(fig)

    print(f"saved: {OUT_PNG}")


if __name__ == "__main__":
    main()
