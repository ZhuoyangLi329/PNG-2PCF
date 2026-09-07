#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Mission 8：三维周期盒离散 k-shell 枚举工具
==========================================

代码大纲
--------
1. 定义三维周期盒的基模 k_f = 2*pi/L。
2. 枚举整数三元组 (n_x, n_y, n_z)，构造 q = n_x^2 + n_y^2 + n_z^2。
3. 统计每个 q 对应的离散 shell 简并度 g_q。
4. 提供一个 Legendre 三平方定理检查函数，说明哪些 q 根本不可能出现。
5. 提供从“连续 best-fit P(k)”采样到“离散 shell 功率谱”的辅助函数。
6. 提供用离散 shell 直接计算 xi_0(r) 的函数。

本脚本的目的
------------
Mission 8 不是继续调一个经验 IR 窗口，而是要从有限盒离散 Fourier 模的角度，
解释为什么当前窗口法会有效。因此这个脚本专门服务于以下研究步骤：

1. 明确三维盒子里最低几个非零 k-shell 的位置；
2. 明确每个 shell 的模式数 g_q；
3. 为后续构造“低 k 离散求和 + 高 k 连续 FFTLog”的 hybrid 模型提供基础函数。
"""

from __future__ import annotations

import argparse
import math
from collections import defaultdict
from dataclasses import dataclass
from typing import Callable, Iterable

import numpy as np


@dataclass
class ShellRecord:
    """
    保存一个离散 k-shell 的基本信息。

    参数
    ----------
    q : int
        三平方和 q = n_x^2 + n_y^2 + n_z^2。
    multiplicity : int
        该 q 对应的整数三元组总数，也就是 shell 简并度 g_q。
    k_over_kf : float
        该 shell 的模长与基模之比，即 sqrt(q)。
    k_value : float
        该 shell 的实际波数，单位与 k_f 一致。
    representatives : list[tuple[int, int, int]]
        该 shell 的若干代表性整数三元组，用于人工检查。
    """

    q: int
    multiplicity: int
    k_over_kf: float
    k_value: float
    representatives: list[tuple[int, int, int]]


def k_fundamental(box_size: float) -> float:
    """
    计算周期盒基模 k_f = 2*pi/L。

    参数
    ----------
    box_size : float
        盒长 L，单位通常为 Mpc/h。

    返回
    ----------
    float
        基模 k_f，单位通常为 h/Mpc。
    """
    return 2.0 * np.pi / float(box_size)


def is_sum_of_three_squares(q: int) -> bool:
    """
    用 Legendre 三平方定理判断 q 是否能写成三个平方和。

    定理内容
    ----------
    正整数 q 能表示成三个整数平方和，当且仅当：
    q 不是 4^a * (8b + 7) 的形式。

    参数
    ----------
    q : int
        待检查的正整数。

    返回
    ----------
    bool
        True 表示 q 可以写成三个平方和，因此存在对应的 k-shell；
        False 表示该 q 对应的 shell 在三维整数格点上根本不存在。
    """
    if q < 0:
        return False

    n = q
    while n % 4 == 0 and n > 0:
        n //= 4
    return n % 8 != 7


def enumerate_shells(
    box_size: float,
    nmax: int,
    qmax: int | None = None,
    keep_examples: int = 6,
) -> list[ShellRecord]:
    """
    枚举三维周期盒中的离散 k-shell。

    参数
    ----------
    box_size : float
        盒长 L。
    nmax : int
        整数三元组扫描上限。每个方向会遍历 [-nmax, nmax]。
    qmax : int or None
        若给定，仅保留 q <= qmax 的 shell。
    keep_examples : int
        每个 shell 保存多少个代表性三元组。

    返回
    ----------
    list[ShellRecord]
        按 q 从小到大排序的 shell 信息列表。

    说明
    ----------
    这里统计的是完整整数格点中的模式数，包含：
    1. 坐标置换；
    2. 正负号变化；
    因此 multiplicity 就是有限盒离散 Fourier 模在该 shell 上的实际模式总数。
    """
    kf = k_fundamental(box_size)
    shell_map: dict[int, list[tuple[int, int, int]]] = defaultdict(list)

    for nx in range(-nmax, nmax + 1):
        for ny in range(-nmax, nmax + 1):
            for nz in range(-nmax, nmax + 1):
                if nx == 0 and ny == 0 and nz == 0:
                    continue

                q = nx * nx + ny * ny + nz * nz
                if qmax is not None and q > qmax:
                    continue

                shell_map[q].append((nx, ny, nz))

    records: list[ShellRecord] = []
    for q in sorted(shell_map):
        modes = shell_map[q]
        records.append(
            ShellRecord(
                q=q,
                multiplicity=len(modes),
                k_over_kf=math.sqrt(q),
                k_value=math.sqrt(q) * kf,
                representatives=modes[:keep_examples],
            )
        )
    return records


def shell_average_power(
    shell_records: Iterable[ShellRecord],
    pofk: Callable[[np.ndarray], np.ndarray] | Callable[[float], float],
) -> np.ndarray:
    """
    在每个离散 shell 的 k_q 位置采样给定的连续 P(k)。

    参数
    ----------
    shell_records : Iterable[ShellRecord]
        已经枚举好的离散 shell 列表。
    pofk : callable
        连续功率谱模型函数，输入 k，输出 P(k)。

    返回
    ----------
    ndarray
        每个 shell 对应的 P(k_q) 采样值。

    说明
    ----------
    当前 Mission 8 第一阶段的重点是离散结构本身，因此这里先采用最简单的
    “在 shell 半径处直接采样连续 P(k)” 近似。后续如果需要更精细地对接测量，
    可以再改成 bin 平均或角向平均。
    """
    kvals = np.asarray([record.k_value for record in shell_records], dtype=float)
    return np.asarray(pofk(kvals), dtype=float)


def radial_delta_coefficients(
    shell_records: Iterable[ShellRecord],
    shell_powers: np.ndarray,
    box_size: float,
) -> np.ndarray:
    """
    计算径向 delta-shell 功率谱的系数 A_q。

    目标关系
    ----------
    希望把离散球平均功率谱写成：

        P_disc(k) = sum_q A_q * delta_D(k - k_q)

    并使得标准 Hankel 关系

        xi_0(r) = int dk k^2/(2*pi^2) P_disc(k) j0(k r)

    精确等价于有限盒离散求和

        xi_0(r) = (1/V) * sum_q g_q * P_q * j0(k_q r)

    则必须取

        A_q = (2*pi^2 / V) * g_q * P_q / k_q^2

    参数
    ----------
    shell_records : Iterable[ShellRecord]
        离散 shell 列表。
    shell_powers : ndarray
        每个 shell 的功率谱值 P_q。
    box_size : float
        盒长 L。

    返回
    ----------
    ndarray
        每个 shell 对应的 delta-shell 系数 A_q。
    """
    volume = float(box_size) ** 3
    kvals = np.asarray([record.k_value for record in shell_records], dtype=float)
    gvals = np.asarray([record.multiplicity for record in shell_records], dtype=float)
    return (2.0 * np.pi**2 / volume) * gvals * shell_powers / np.square(kvals)


def spherical_bessel_j0(x: np.ndarray | float) -> np.ndarray:
    """
    计算 j0(x) = sin(x) / x，并在 x=0 处返回 1。

    参数
    ----------
    x : ndarray or float
        输入自变量。

    返回
    ----------
    ndarray
        j0(x) 的数值结果。
    """
    x = np.asarray(x, dtype=float)
    out = np.ones_like(x)
    mask = x != 0.0
    out[mask] = np.sin(x[mask]) / x[mask]
    return out


def xi0_from_discrete_shells(
    r: np.ndarray,
    shell_records: Iterable[ShellRecord],
    shell_powers: np.ndarray,
    box_size: float,
) -> np.ndarray:
    """
    直接用有限盒离散 shell 求和计算 xi_0(r)。

    公式
    ----------
        xi_0(r) = (1/V) * sum_q g_q * P_q * j0(k_q r)

    参数
    ----------
    r : ndarray
        需要计算的距离数组。
    shell_records : Iterable[ShellRecord]
        离散 shell 列表。
    shell_powers : ndarray
        每个 shell 的功率谱值 P_q。
    box_size : float
        盒长 L。

    返回
    ----------
    ndarray
        与 r 同长度的 xi_0(r)。
    """
    volume = float(box_size) ** 3
    kvals = np.asarray([record.k_value for record in shell_records], dtype=float)
    gvals = np.asarray([record.multiplicity for record in shell_records], dtype=float)

    r = np.asarray(r, dtype=float)
    kr = np.outer(kvals, r)
    j0 = spherical_bessel_j0(kr)
    weighted_sum = (gvals * shell_powers)[:, None] * j0
    return np.sum(weighted_sum, axis=0) / volume


def toy_png_like_power(k: np.ndarray, amplitude: float = 1.0, eps: float = 1.0e-8) -> np.ndarray:
    """
    一个简单的玩具低 k 功率谱，仅用于测试离散 shell 求和的数值行为。

    形式
    ----------
        P(k) = amplitude / (k^2 + eps)

    这个函数没有物理完整性，只是为了模仿“低 k 权重很强”的情况，
    便于直观看出离散 shell 求和与连续 IR 积分在行为上的差异。
    """
    k = np.asarray(k, dtype=float)
    return amplitude / (np.square(k) + eps)


def print_shell_table(records: list[ShellRecord], max_rows: int) -> None:
    """
    以纯文本方式打印前若干个 shell，方便在终端快速查看。
    """
    print("index  q   sqrt(q)      multiplicity   k[h/Mpc]      theorem_ok   representatives")
    for i, record in enumerate(records[:max_rows], start=1):
        theorem_ok = is_sum_of_three_squares(record.q)
        print(
            f"{i:>5d} {record.q:>3d} {record.k_over_kf:>10.6f} "
            f"{record.multiplicity:>14d} {record.k_value:>12.8f} "
            f"{str(theorem_ok):>11s}   {record.representatives}"
        )


def main() -> None:
    """
    命令行入口。

    运行功能
    ----------
    1. 枚举并打印前若干个离散 shell；
    2. 打印一个简单的“三平方定理缺失 shell”检查；
    3. 用玩具 P(k) 计算一组离散 xi_0(r)，方便后续研究时做 sanity check。
    """
    parser = argparse.ArgumentParser(description="Enumerate discrete k-shells in a periodic 3D box.")
    parser.add_argument("--box-size", type=float, default=3000.0, help="盒长 L，默认 3000 Mpc/h。")
    parser.add_argument("--nmax", type=int, default=8, help="整数三元组扫描上限，默认 8。")
    parser.add_argument("--qmax", type=int, default=60, help="保留的最大 q，默认 60。")
    parser.add_argument("--rows", type=int, default=18, help="打印前多少个 shell。")
    args = parser.parse_args()

    kf = k_fundamental(args.box_size)
    print(f"Box size L = {args.box_size:.1f} Mpc/h")
    print(f"Fundamental mode k_f = 2*pi/L = {kf:.8f} h/Mpc")
    print("")

    records = enumerate_shells(box_size=args.box_size, nmax=args.nmax, qmax=args.qmax)
    print_shell_table(records, max_rows=args.rows)
    print("")

    print("First missing q values predicted by Legendre theorem:")
    missing = [q for q in range(1, args.qmax + 1) if not is_sum_of_three_squares(q)]
    print(missing[:15])
    print("")

    # 做一个很轻量的 toy model 检查：前若干个 shell 对 xi_0(r) 的贡献是否数值有限。
    toy_records = records[: min(12, len(records))]
    toy_p = shell_average_power(toy_records, lambda kvals: toy_png_like_power(kvals, amplitude=1.0))
    r = np.array([20.0, 50.0, 100.0, 200.0, 400.0], dtype=float)
    xi = xi0_from_discrete_shells(r, toy_records, toy_p, args.box_size)

    print("Toy discrete-shell xi_0(r) using first shells and P(k)=1/k^2:")
    for rr, xx in zip(r, xi):
        print(f"r = {rr:7.1f} Mpc/h   xi_0 = {xx:.8e}")


if __name__ == "__main__":
    main()
