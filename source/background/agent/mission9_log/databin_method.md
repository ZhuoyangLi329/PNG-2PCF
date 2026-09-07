# DataBin 方法：原理、实现与验证

## 1. 背景问题

在有限体积模拟盒（边长 $L$）中，从 best-fit 功率谱 $P_0(k)$ 建模两点相关函数 $\xi_0(r)$ 时，
现有方法面临 IR 发散问题（$f_\text{NL}\neq 0$ 时 $P(k)\propto 1/k^2$）。

**现有方法对比：**

| 方法 | 公式 | 优点 | 缺点 |
|---|---|---|---|
| Baseline | $\xi_0 = \int_{k_f}^{k_\max} \frac{k^2}{2\pi^2} P_\text{model}(k) j_0(kr)\,dk$ | 简单 | 大尺度系统性偏低 |
| ExpWindow | 同上，但 $k_\min=10^{-4}$，加 IR 窗口 | 效果好 | 需要经验参数 $x(L)$ |
| FullDiscrete | $\xi_0 = \frac{1}{V}\sum_q g_q\, P_\text{model}(k_q)\, j_0(k_q r)$ | 无参数、物理清晰 | 略有系统偏差（$\sim -0.09\sigma$）|

## 2. DataBin 方法原理

### 2.1 核心想法

FullDiscrete 的 $\sim -0.09\sigma$ 偏差来源于：best-fit 连续模型 $P_\text{model}(k)$ 在各离散 $k_q$ 处
与真实集合平均 $\langle P(k_q)\rangle$ 之间的微小系统性差异。

这种差异的根源是功率谱拟合过程：

1. 测量的 $P(k)$ 是按 $k$-bin 平均的：$\bar{P}_\text{bin} = \frac{\sum_{q\in\text{bin}} g_q P_\text{true}(k_q)}{\sum_{q\in\text{bin}} g_q}$
2. 拟合时，模型在 bin center 处评估：$P_\text{model}(k_\text{center})$
3. 对于弯曲的 $P(k)$（尤其是 PNG 的 $1/k^2$），$P_\text{model}(k_\text{center}) \neq \bar{P}_\text{bin}$

**DataBin 的解决方案：** 在低 $k$（数据覆盖范围内），直接使用测量的 bin 平均值 $\bar{P}_\text{bin}$
替代模型 $P_\text{model}(k_q)$，从而消除 model-vs-data mismatch。

### 2.2 公式

$$\xi_0(r) = \frac{1}{V}\sum_q g_q \cdot P_\text{eff}(k_q) \cdot j_0(k_q r)$$

其中有效功率谱为：

$$P_\text{eff}(k_q) = \begin{cases}
\bar{P}_{\text{data},i} & \text{if } k_q \in [k_{\min,i},\, k_{\max,i}),\quad i = 1,\ldots, N_\text{fit} \\
P_\text{model}(k_q)     & \text{otherwise}
\end{cases}$$

- $\bar{P}_{\text{data},i}$ = 第 $i$ 个 $k$-bin 的测量功率谱均值
- $N_\text{fit}$ = 拟合时使用的 $k$-bin 数量（通常 $\sim 20$）
- $P_\text{model}(k_q)$ = desilike best-fit 连续模型
- $g_q$ = 整数格点 $n_x^2+n_y^2+n_z^2=q$ 的简并度（用 FFT 卷积精确计算）
- $k_q = k_f\sqrt{q}$，$k_f = 2\pi/L$
- $V = L^3$

### 2.3 物理意义

1. **低 $k$ 区域**（$k < k_\text{data,max}$）：
   直接使用测量值，规避了理论模型在低 $k$ 处的拟合偏差。
   测量值本身就是盒内所有模式的准确平均，不需要任何参数。

2. **高 $k$ 区域**（$k > k_\text{data,max}$）：
   使用 best-fit 理论模型外推。由于 $j_0(kr)$ 在 $kr\gg 1$ 时高频振荡，
   高 $k$ 对大尺度 $\xi_0(r)$ 的净贡献几乎为零（kmax 不敏感性的原因）。

3. **离散模式结构**：
   保留了有限盒中实际存在的 Fourier 模式（$g_q$ 精确计数），
   比连续积分更准确地描述了有限体积效应。

### 2.4 为什么 $N_\text{fit}$ 必须有限

如果将 **所有** $k$-bin（包括高 $k$）都替换为测量值，会产生问题：
- 每个 bin 内所有模式被赋予相同的 $P$ 值（阶梯函数近似）
- 窄 bin 在宽 $k$ 范围内制造大量不连续点
- $j_0$ 卷积将这些不连续点转化为 $\xi_0$ 的振荡伪影

因此，$N_\text{fit}$ 应限制在拟合范围内（$\sim 20$ 个 bin），超出的 $k$ 使用光滑模型。

## 3. 实现步骤

### Step 1: 功率谱拟合
与现有流程完全一致：用 desilike + Minuit 拟合前 $N_\text{fit}$ 个 $k$-bin 的均值，
得到 best-fit 参数（$f_\text{NL}$, $b_1$, $\sigma_s$ 等）。

### Step 2: 计算 $g_q$
对给定盒长 $L$ 和截断 $k_\text{max}$：
```python
# FFT 卷积精确计算 g_q
qmax = int((kmax / kf) ** 2)
nmax = int(kmax / kf)
a = zeros(qmax+1); a[0] = 1
a[n^2 for n=1..nmax] = 2
g = round(IFFT(FFT(a)^3))
```

### Step 3: 构造 $P_\text{eff}(k)$
```python
def P_eff(k_array):
    # 默认使用理论模型（已在密集 k 网格上预计算并插值）
    P_out = interp(k_array, k_dense, P_model_dense)
    # 数据覆盖范围内替换为测量值
    for i_bin in range(N_fit):
        mask = (k_array >= kmin[i_bin]) & (k_array < kmax[i_bin])
        P_out[mask] = P_data_mean[i_bin]
    return P_out
```

### Step 4: 离散求和
```python
xi_0 = zeros(len(s))
for q in nonzero_shells:
    k_q = kf * sqrt(q)
    xi_0 += g[q] * P_eff(k_q) * j0(k_q * s) / V
```
（实际实现中按 shell 分块求和以控制内存。）

## 4. 验证结果

### 4.1 FastPM 样本
| 方法 | 3Gpc fnl100 mean|Δ/σ| | 1Gpc fnl100 mean|Δ/σ| |
|---|---|---|
| ExpWindow | 0.498 | 0.231 |
| FullDiscrete | 0.643 | 0.277 |
| **DataBin** | **0.441** | **0.208** |

### 4.2 Quijote N-body (L=1Gpc, 500 realizations)
| $f_\text{NL}^\text{true}$ | DataBin(20) | FullDiscrete | ExpWindow |
|---|---|---|---|
| 0 | **0.090** | 0.107 | 0.136 |
| 50 | **0.080** | 0.100 | 0.137 |
| 100 | 0.093 | **0.089** | 0.129 |

DataBin 在低 $f_\text{NL}$ ($\leq 50$) 时一致最优；
在高 $f_\text{NL}$ ($\geq 75$) 时与 FullDiscrete 接近。

## 5. 与现有方法的关系

- **DataBin = FullDiscrete + 低 $k$ 数据校正**：
  当 $N_\text{fit}=0$（不替换任何 bin），DataBin 退化为 FullDiscrete。
  当 $N_\text{fit}$ 增大，更多低 $k$ 模式使用测量值。

- **vs ExpWindow**：
  ExpWindow 在 $k < k_f$ 加经验窗口函数来补偿 baseline 的低估。
  DataBin 则直接在数据范围内使用测量值，不引入经验参数。

- **vs Baseline (连续 FFTLog)**：
  Baseline 使用连续模式密度 $k^2/(2\pi^2)$ 近似有限盒的离散结构。
  DataBin 使用精确的离散模式计数 $g_q$。

## 6. 局限性

1. 需要多个 realization 的 $P(k)$ 均值（单个 realization 的噪声可能影响效果）
2. $N_\text{fit}$ 的选择取决于拟合范围（通常由数据 k-binning 决定，不是自由参数）
3. 在数据 bin 较宽时（如 fastPM 的 $\Delta k = 0.003$），bin 内常数近似的误差更大
4. 高 $f_\text{NL}$ 时系统偏差略有增加（$\sim +0.05\sigma$）
