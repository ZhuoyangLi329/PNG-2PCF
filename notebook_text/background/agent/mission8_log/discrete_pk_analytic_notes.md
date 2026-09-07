# Notebook source mirror

Original: `source/background/agent/mission8_log/discrete_pk_analytic_notes.ipynb`

Outputs remain in original notebook.

## Cell 1 (markdown)

# Mission 8：离散功率谱与 2PCF 的解析笔记

本笔记用于完成任务书 `##8 窗口方法的解析解释`。

当前第一阶段目标：

1. 明确有限盒中的离散 Fourier 模与球平均功率谱的关系；
2. 枚举三维盒子最低若干个离散 `k-shell`；
3. 为后续把离散 `P(k)` 写成 delta-shell 形式、并推导 `xi(r)` 的解析求和式做准备。


## Cell 2 (markdown)

## 代码大纲

A. 定义盒长 `L` 与基模 `k_f = 2*pi/L`

B. 枚举整数三元组 `(n_x, n_y, n_z)`，统计 `q = n_x^2 + n_y^2 + n_z^2`

C. 输出前若干个非零离散 shell 的：
- `q`
- `k / k_f = sqrt(q)`
- 对应的简并度 `g_q`

D. 后续在此基础上补充：
- 一维离散 `P(k)` 的解析形式
- 三维 delta-shell `P(k)` 的解析形式
- `P(k) -> xi(r)` 的解析求和式与 continuous / window 模型对比


## Cell 3 (code)

```python
# ============================================================
# Mission 8 第一阶段：枚举三维盒子的离散 k-shell
#
# 说明：
# 1. 这里先不直接碰真实功率谱数据，而是先把几何结构理清楚。
# 2. 三维周期盒中允许的波矢为 k = k_f * (n_x, n_y, n_z)。
# 3. 球平均功率谱只依赖 |k|，因此关键量是 q = n_x^2 + n_y^2 + n_z^2。
# 4. 同一个 q 可能由多个整数三元组给出，它们共同构成一个离散 shell。
# 5. 这些 shell 的位置与简并度，就是后续离散 P(k) 解析建模的基础。
# ============================================================

import math
from collections import defaultdict

import numpy as np
import pandas as pd


# 盒长先按 mission 8 主目标的 3Gpc 口径设置；单位均为 Mpc/h。
L = 3000.0
k_f = 2.0 * np.pi / L

print(f'L = {L:.1f} Mpc/h')
print(f'k_f = 2*pi/L = {k_f:.8f} h/Mpc')


def enumerate_k_shells(nmax, qmax=None):
    """
    枚举三维周期盒中离散 k-shell 的整数结构。

    参数
    ----------
    nmax : int
        枚举整数三元组时，每个方向扫描的最大绝对值。
        实际扫描范围为 [-nmax, nmax]。
    qmax : int or None
        若不为 None，则仅保留 q <= qmax 的 shell。

    返回
    ----------
    pandas.DataFrame
        每一行对应一个非零 shell，包含：
        - q: q = n_x^2 + n_y^2 + n_z^2
        - multiplicity: 该 q 对应的整数三元组个数
        - k_over_kf: sqrt(q)
        - representatives: 若干代表性整数三元组

    说明
    ----------
    这里统计的是完整整数格点上的简并度，包含正负号和坐标置换。
    这正是有限盒离散 Fourier 模天然具有的模式数。
    """
    shell_map = defaultdict(list)

    for nx in range(-nmax, nmax + 1):
        for ny in range(-nmax, nmax + 1):
            for nz in range(-nmax, nmax + 1):
                if nx == 0 and ny == 0 and nz == 0:
                    continue

                q = nx * nx + ny * ny + nz * nz
                if qmax is not None and q > qmax:
                    continue

                shell_map[q].append((nx, ny, nz))

    rows = []
    for q in sorted(shell_map):
        modes = shell_map[q]
        rows.append({
            'q': q,
            'multiplicity': len(modes),
            'k_over_kf': math.sqrt(q),
            'k_value': math.sqrt(q) * k_f,
            'representatives': modes[:6],
        })

    return pd.DataFrame(rows)


# 这里先枚举前一些较低 q 的 shell，用来确认低 k 结构。
shell_df = enumerate_k_shells(nmax=6, qmax=40)
shell_df.head(15)

```

## Cell 4 (markdown)

## 当前观察目标

运行上面的枚举后，需要重点确认：

1. 最低几个非零 shell 是否依次对应 `q = 1, 2, 3, 4, 5, ...`；
2. 哪些整数不能写成三个平方和，因此对应 shell 会缺失；
3. 每个 shell 的简并度 `g_q` 如何进入球平均离散 `P(k)` 的系数；
4. 后续从 delta-shell 形式做 Hankel 变换时，最终系数是否自然变成 `j_0(k_q r)` 的离散求和。

下一步会在本笔记里补上：

$$
P_{\mathrm{disc}}(k) = \sum_{q \in \mathcal{Q}} A_q\,\delta_D(k-k_f\sqrt{q})
$$

以及与有限盒离散 Fourier 求和形式完全对应的 `xi(r)` 公式。

