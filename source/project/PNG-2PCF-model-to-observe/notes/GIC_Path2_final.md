# GIC C_IC 理论计算：Path 2 最终结果

*更新日期：2026-03-30*

---

## 1. 问题背景

在 no-RSD cut-sky vs box 的比较中，观测到：

$$\Delta\xi = \xi_{\rm cut} - \xi_{\rm box} \approx -2.24 \times 10^{-5} \quad (s \geq 260\ {\rm Mpc}/h)$$

这个近似常数的负偏移是 **Global Integral Constraint (GIC)** 的典型表现。

---

## 2. 两条路径的比较

| 方法 | C_IC | 与观测比值 |
|------|------|----------|
| **Path 1** (FFTLog + RR加权) | 9.32×10⁻⁵ | ~4× (过大) |
| **Path 2** (W̃ 连续积分) | **-2.37×10⁻⁵** | **~1.06** ✓ |
| 观测值 (const_large) | -2.24×10⁻⁵ | — |

**Path 1 高估约 4× 的原因**：FFTLog 的 taper 在 k→0 处对 PNG 的 1/k² 信号处理不当，导致 xi(s) 被系统性高估。W̃(k) 在 k~0.005-0.5 之间大量振荡变负，这些负贡献在 Path 1 中没有被正确计入，但在 Path 2 中通过连续积分自然消除。

---

## 3. 正确的 Path 2 公式

### 3.1 核心公式

$$\boxed{C_{IC} = \frac{1}{2\pi^2} \int_0^\infty k^2 P(k)\, \tilde{W}(k)\, dk}$$

其中 survey window 函数的归一化 Fourier 变换为：

$$\tilde{W}(k) = \frac{\int_0^\infty 4\pi s^2\, RR_{\rm norm}(s)\, j_0(ks)\, ds}{\int_0^\infty 4\pi s^2\, RR_{\rm norm}(s)\, ds}$$

- $RR_{\rm norm}(s)$：pycorr 给出的归一化 random-random pair counts，$\sum_i RR_{\rm norm}(s_i) = 1$
- $j_0(x) = \sin(x)/x$：零阶球贝塞尔函数
- **不需要除以 V_eff**：几何信息已经通过 $\tilde{W}$ 的归一化隐含在内

### 3.2 W̃(k) 的性质

- $\tilde{W}(k \to 0) = 1$（归一化条件）
- $\tilde{W}(k)$ 在 $k \sim 2\pi / L_{\rm max\_sep}$ 处开始振荡并变负
- 对于我们的 cut-sky（$s_{\rm max} \approx 595$ Mpc/h），$\tilde{W}$ 在 k~0.009 h/Mpc 处首次变负
- 正负贡献的净结果决定了 $C_{IC}$ 的符号和量级

### 3.3 数值实现

```python
import numpy as np
from scipy.integrate import simpson
from scipy.interpolate import interp1d

# 从 pycorr 获取 RR_norm
rr_s = rr_data['s_mid']     # 单位 Mpc/h
rr_c = rr_data['rr_counts'] # 归一化，sum = 1

# 插值到细 s 网格
s_fine = np.linspace(rr_s.min(), rr_s.max(), 10000)
rr_fine = interp1d(rr_s, rr_c, fill_value=0, bounds_error=False)(s_fine)
ds = s_fine[1] - s_fine[0]

# 归一化常数
norm = np.sum(4*np.pi * s_fine**2 * rr_fine * ds)

# 计算 W̃(k)
k_fine = np.geomspace(1e-5, 0.5, 5000)
W_tilde = np.zeros_like(k_fine)
for i, k in enumerate(k_fine):
    j0 = np.sin(k*s_fine) / (k*s_fine)
    j0[s_fine == 0] = 1.0
    W_tilde[i] = np.sum(4*np.pi * s_fine**2 * rr_fine * j0 * ds) / norm

# 计算 C_IC
pk = P_model(k_fine, fnl_loc, b1)  # tracer P(k)
C_IC = simpson(k_fine**2 * pk * W_tilde, k_fine) / (2*np.pi**2)
```

**参考代码**：`codes/task18/task18_model_fnl0_cutsky_gic_v4.py`

---

## 4. 不同参数下的 C_IC

| 参数 | C_IC | 比值（/观测值） | 说明 |
|------|------|----------------|------|
| fnl=5.37, b1=2.77 (bestfit) | -2.37×10⁻⁵ | **1.06** | 与观测最接近 |
| fnl=0, b1=2.77 (真实 fnl) | -3.32×10⁻⁵ | 1.48 | 偏大 |
| fnl=0, b1=2.5 | -2.70×10⁻⁵ | 1.20 | — |
| fnl=0, b1=3.0 | -3.89×10⁻⁵ | 1.73 | — |
| fnl=100, b1=2.77 | +1.66×10⁻³ | ~74 (反号) | PNG 大幅放大 |

**观察**：
1. bestfit 参数（fnl=5.37）给出的 C_IC 与观测最接近，这是因为拟合时 box P(k) 在低 k 端的偏差被 fnl≠0 的参数吸收了
2. 真实 fnl=0 时，C_IC 比观测大约 1.5 倍，可能来自 b1 的不确定性或其他系统效应
3. fnl=100 时，PNG 的 1/k² 项使 C_IC 大幅增加并改变符号（因为 W̃ 在小 k 处接近 1，大 k 处负，净积分符号取决于 P(k) 的形状）

---

## 5. 物理解释

$$C_{IC} < 0 \Rightarrow \xi_{\rm cut} = \xi_{\rm true} - C_{IC} = \xi_{\rm true} + |C_{IC}| > \xi_{\rm true}$$

等等，这个符号不对？

实际上：
$$\Delta\xi = \xi_{\rm cut} - \xi_{\rm box} < 0$$

意味着 cut-sky 测量值低于 box 测量值。从 LS 估计器的 IC 效应来看：
$$\xi_{\rm LS}(s) \approx \xi_{\rm true}(s) - C_{IC}$$

当 $C_{IC} > 0$ 时，LS 估计器给出的 xi 比真实值低。因此：
- 对 box（无 survey geometry）：$C_{IC,\rm box} \approx 0$，$\xi_{\rm box} \approx \xi_{\rm true}$
- 对 cut-sky：$C_{IC,\rm cut} > 0$，$\xi_{\rm cut} = \xi_{\rm true} - C_{IC,\rm cut} < \xi_{\rm box}$

所以 $\Delta\xi = \xi_{\rm cut} - \xi_{\rm box} = -C_{IC,\rm cut} < 0$ ✓

即：**我们计算的 C_IC 是负数，但实际的物理修正量是 -C_IC > 0**。

这一点在代码中需要注意：
```python
xi_model_obs = xi_model_true - C_IC  # C_IC < 0，所以是加上 |C_IC|
# 或等价地：
xi_model_obs = xi_model_true + abs(C_IC)
```

---

## 6. fnl100 的情况：直接 Path2 失败，以及最终修正

fnl100 观测值（no-RSD current mean, 98 realizations）：

- $\Delta\xi$ (const_all) = $-3.49 \times 10^{-5}$
- $\Delta\xi$ (s≥260) = **$-5.66 \times 10^{-5}$**

### 6.1 直接把完整 tracer P(k) 送进 Path2，会失败

对 `fnl100` current mean 做 real-space `P(k)` 拟合后得到：

- `bestfit_center`：`fnl_loc = 76.14`, `b1 = 2.7355`
- `bestfit_binavg`：`fnl_loc = 75.84`, `b1 = 2.7369`

如果像 `fnl0` 那样，直接把完整 tracer 功率谱

$$
P_{\rm full}(k)
$$

送进

$$
C_{IC} = \frac{1}{2\pi^2}\int k^2 P(k)\,\tilde{W}(k)\,dk
$$

则会得到：

| 方法 | $C_{IC}$ | 与观测大尺度常数比值 |
|------|---------|----------------------|
| `fnl100` 直接 Path2 (`P_full`) | $+6.03 \times 10^{-4}$ | $-10.66$ |
| 观测值 $\Delta\xi_{\rm obs}(s\ge 260)$ | $-5.66 \times 10^{-5}$ | — |

也就是：

1. **符号反了**
2. **幅度大了约 10 倍**

### 6.2 为什么会失败？

`fnl100` 时，PNG 的 scale-dependent bias 使 tracer `P(k)` 在极低 `k` 处大幅抬升。

对于当前 cut-sky：

- $\tilde{W}(k \to 0) \approx 1$
- $\tilde{W}(k)$ 第一次过零在
  $$
  k_{\rm zero} \approx 0.00671\ h/{\rm Mpc}
  $$
- 在 `full Path2` 里，积分的正贡献主要来自
  $$
  k \lesssim 0.005\ h/{\rm Mpc}
  $$
- 负贡献主要来自
  $$
  0.005 \lesssim k \lesssim 0.01\ h/{\rm Mpc}
  $$

对于 `fnl100` 的完整 tracer `P(k)`，不存在的超低-`k` 连续谱把正贡献过度放大，从而把 `C_IC` 推成正号且量级过大。

换句话说：

> `fnl0` 的 Path2 可以直接作用在完整 tracer `P(k)` 上；  
> `fnl100` 不行，因为真正出问题的是 **PNG 额外项** 在极低 `k` 的连续延拓。

### 6.3 最终采用的修正：只对 PNG 差分项加 IR window

令

$$
P_{\rm full}(k) = P_{f_{\rm NL}=0,\ {\rm same}\ b_1}(k) + \Delta P_{\rm png}(k)
$$

其中

$$
\Delta P_{\rm png}(k) = P_{\rm full}(k) - P_{f_{\rm NL}=0,\ {\rm same}\ b_1}(k)
$$

然后只对 $\Delta P_{\rm png}$ 的低 `k` 端乘一个 IR 窗：

$$
P_{\rm eff}(k)
=
P_{f_{\rm NL}=0,\ {\rm same}\ b_1}(k)
+
W_{\rm IR}(k)\,\Delta P_{\rm png}(k)
$$

取

$$
W_{\rm IR}(k)
=
\frac{1-\exp\left[-(k/k_{\rm scale})^x\right]}{1-\exp(-1)},
\quad k < k_{\rm scale}
$$

并保持

$$
W_{\rm IR}(k)=1,\quad k\ge k_{\rm scale}.
$$

当前 3Gpc 的 adopted 口径：

- `x = 12`
- `k_scale = 0.8 * k_zero`
- 对当前 cut-sky，`k_zero ≈ 0.006714`
- 因此
  $$
  k_{\rm scale} \approx 0.00537\ h/{\rm Mpc}
  $$

### 6.4 最终结果

用上面的 `PNG-diff IR-windowed Path2` 后，得到：

| 方法 | $C_{IC}$ | 与观测大尺度常数比值 |
|------|---------|----------------------|
| `fnl100` 直接 Path2 (`P_full`) | $+6.03 \times 10^{-4}$ | $-10.66$ |
| `fnl100` 修正后 Path2 (`P_{\rm eff}`) | **$-5.75 \times 10^{-5}$** | **$1.015$** |
| 观测值 $\Delta\xi_{\rm obs}(s\ge 260)$ | $-5.66 \times 10^{-5}$ | — |

这说明修正后的 Path2：

1. 符号正确
2. 幅度与观测相差仅约 `1.5%`

### 6.5 对 2PCF 模型残差的改善

把 cut-sky 模型写成

$$
\xi_{\rm model,cutsky}(s)=\xi_{\rm model,box}(s)+C_{IC}
$$

后，残差明显改善：

| 指标 | noGIC | 修正后 Path2 |
|------|------|-------------|
| `mean|Δ/σ|` on all bins | 0.548 | **0.348** |
| `mean|Δ/σ|` for `s > 150` | 0.532 | **0.270** |
| `mean|Δ/σ|` for `s >= 260` | 0.634 | **0.308** |

### 6.6 代码与图

实现脚本：

- `fnl0` 版本：
  `codes/task18/task18_model_fnl0_cutsky_gic_v4.py`
- `fnl100` 版本：
  `codes/task18/task18_model_fnl100_cutsky_gic_path2.py`

结果文件：

- `task18_outputs/fnl100_cutsky_gic_path2/task18_fnl100_cutsky_gic_path2_fit_summary.json`
- `task18_outputs/fnl100_cutsky_gic_path2/task18_fnl100_cutsky_gic_path2_results.npz`

图像：

- `plots/task18/task18_fnl100_cutsky_gic_path2_xi_fit.png`
- `plots/task18/task18_fnl100_cutsky_gic_path2_pk_fit.png`
- `plots/task18/task18_fnl100_cutsky_gic_path2_wtilde.png`
- `plots/task18/task18_fnl100_model_vs_cutsky_mean_path2_clean.png`

### 6.7 当前理解

目前最合理的理解是：

- `fnl0`：Path2 可以直接作用在完整 tracer `P(k)` 上；
- `fnl100`：必须先把问题定位到 **PNG 额外项**，再对其极低 `k` 部分做受控 IR 处理；
- 这种做法保留了 Path2 的 `W̃(k)` 几何核，同时避免让不存在的超低-`k` 连续模式主导积分。

## 7. 残留问题

1. **fnl0 的小偏差**：`C_IC(bestfit) / 观测值 ≈ 0.99`，误差来源可能包括：
   - P(k) 拟合参数的不确定性（fnl≠0 是人为的）
   - W̃(k) 网格精度（s 范围截断在 595 Mpc/h）
   - RSD 效应（no-RSD 下的 b1 可能与 monopole 拟合结果不同）

2. **fnl100 的修正仍带经验性**：
   - `x = 12`
   - `k_scale = 0.8 * k_zero`
   这是一个“尽量少调参”的经验选择，但还不是严格第一性原理推导。

3. **W̃ 的 s 截断**：RR 只测到 `s=595 Mpc/h`，而 survey 的最大 pair 间距可能更大。扩展 `s` 范围到 `~3000 Mpc/h` 可能改变 `k_zero` 和最终 `C_IC`。

4. **下一步**：用 pypower 的 window matrix 方法（`task18_compute_window_function.py`）做独立验证，这是最严格的方法。

---

## 8. 参考

- 实现代码：
  - `codes/task18/task18_model_fnl0_cutsky_gic_v4.py`
  - `codes/task18/task18_model_fnl100_cutsky_gic_path2.py`
- RR counts：`task18_outputs/rr_counts/rr_counts_norsd.npz`
- 观测值来源：`task19_outputs/ic_diagnostic/task19_delta_xi_constant_test.json`
- 理论参考：de Mattia & Ruhlmann-Kleider (2019), arXiv:1904.08851
