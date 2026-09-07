#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Mission 8：toy IR 检查
======================

目标
----
用一个故意带强 IR 权重的玩具功率谱，直接比较：

1. 连续 Hankel 积分在不同 kmin 下得到的 xi_0(r)；
2. 有限盒离散 shell 求和得到的 xi_0(r)。

如果连续结果随着 kmin 持续抬升，而离散结果保持有限，
就能很直观地说明 Mission 8 的核心判断：

    当前的 IR 问题，并不是有限盒真实离散模本身发散，
    而是把低 k 部分误近似成连续谱之后产生的。
"""

from __future__ import annotations

import numpy as np

from enumerate_k_shells import enumerate_shells, xi0_from_discrete_shells


def spherical_bessel_j0(x: np.ndarray) -> np.ndarray:
    """
    计算球贝塞尔函数 j0(x) = sin(x) / x。
    """
    x = np.asarray(x, dtype=float)
    out = np.ones_like(x)
    mask = x != 0.0
    out[mask] = np.sin(x[mask]) / x[mask]
    return out


def toy_power_ir_strong(k: np.ndarray, amplitude: float = 1.0, eps: float = 1.0e-12) -> np.ndarray:
    """
    构造一个低 k 明显发散的玩具功率谱。

    形式
    ----------
        P(k) = A / (k^3 + eps)

    说明
    ----------
    这个模型不是为了模拟真实宇宙学，而是专门用来制造一个
    “连续积分会对 kmin 极其敏感”的 IR 场景。
    """
    k = np.asarray(k, dtype=float)
    return amplitude / (np.power(k, 3) + eps)


def xi0_continuous(
    r: float,
    kmin: float,
    kmax: float,
    nk: int = 200000,
) -> float:
    """
    直接用连续 Hankel 积分计算 toy xi_0(r)。

    公式
    ----------
        xi_0(r) = int dk k^2/(2*pi^2) P(k) j0(k r)

    实现细节
    ----------
    这里使用对数 k 网格 + 梯形积分。
    对 toy IR 行为来说，重点不是高精度，而是比较不同 kmin 的系统趋势。
    """
    k = np.geomspace(kmin, kmax, nk)
    p = toy_power_ir_strong(k)
    integrand = (k**2 / (2.0 * np.pi**2)) * p * spherical_bessel_j0(k * r)
    return float(np.trapz(integrand, k))


def main() -> None:
    """
    运行 toy 检查，并打印连续积分与离散求和的对比结果。
    """
    box_size = 3000.0
    kmax = 20.0
    r_values = np.array([100.0, 200.0, 400.0], dtype=float)

    print(f"Toy comparison for L = {box_size:.1f} Mpc/h")
    print("Toy spectrum: P(k) = 1 / k^3")
    print("")

    # 连续积分：逐步降低 kmin，观察 xi_0(r) 是否不断抬升。
    kmins = [2.0e-3, 1.0e-3, 5.0e-4, 1.0e-4, 1.0e-5]
    for r in r_values:
        print(f"[Continuous integral] r = {r:.1f} Mpc/h")
        for kmin in kmins:
            xi = xi0_continuous(r=r, kmin=kmin, kmax=kmax)
            print(f"  kmin = {kmin:>8.1e}  ->  xi_0 = {xi:.8e}")
        print("")

    # 离散 shell：只要盒长固定，最低非零模就是 k_f，不会出现 k->0 的连续 IR 面积。
    records = enumerate_shells(box_size=box_size, nmax=80, qmax=300)
    shell_powers = np.array([toy_power_ir_strong(record.k_value) for record in records], dtype=float)
    xi_discrete = xi0_from_discrete_shells(r_values, records, shell_powers, box_size=box_size)

    print("[Discrete shell sum]")
    for r, xi in zip(r_values, xi_discrete):
        print(f"  r = {r:>6.1f} Mpc/h  ->  xi_0 = {xi:.8e}")


if __name__ == "__main__":
    main()
