# Global Integral Constraint (GIC) 文献调研笔记

## 目录
1. [引言与问题背景](#1-引言与问题背景)
2. [GIC 的物理来源](#2-gic-的物理来源)
3. [功率谱空间的 GIC 修正](#3-功率谱空间的-gic-修正)
4. [配置空间 2PCF 的 GIC 效应](#4-配置空间-2pcf-的-gic-效应)
5. [Landy-Szalay 估计器与 IC 的关系](#5-landy-szalay-估计器与-ic-的关系)
6. [DESI 分析中的 IC 处理](#6-desi-分析中的-ic-处理)
7. [PNG 对 GIC 的影响](#7-png-对-gic-的影响)
8. [实用计算方法](#8-实用计算方法)
9. [参考文献](#9-参考文献)

---

## 1. 引言与问题背景

### 1.1 问题描述

在我们从 cubic box N-body simulation 转向 cut-sky mock 的过程中，发现了一个关键的系统性差异：

- **Box 中**：2PCF 测量使用周期边界条件，没有几何限制
- **Cut-sky 中**：使用 Landy-Szalay 估计器，受有限巡天体积限制

观测到的差异为：
- **fnl0**: Δξ = ξ_cut - ξ_box ≈ -2.2 × 10⁻⁵ (大尺度平均值)
- **fnl100**: Δξ = ξ_cut - ξ_box ≈ -5.7 × 10⁻⁵ (大尺度平均值)

这个负偏移在配置空间中近似为常数，是 **Global Integral Constraint (GIC)** 的典型表现。

### 1.2 核心问题

1. **GIC 的物理来源是什么？**
2. **能否从第一性原理预测这个常数？**
3. **PNG (f_NL ≠ 0) 如何影响 GIC？**
4. **在功率谱和 2PCF 建模中如何考虑这个效应？**

---

## 2. GIC 的物理来源

### 2.1 根本原因

GIC 源于**有限巡天体积中平均密度必须从数据本身估计**这一事实。

在真实的宇宙中，我们不知道真实的空间平均密度，只能从巡天数据中估计。这导致：

$$\int d^3r \, \delta(\mathbf{r}) W(\mathbf{r}) = 0$$

其中 $W(\mathbf{r})$ 是巡天的选择函数，$\delta$ 是密度反差。

### 2.2 FKP 场形式

Following de Mattia & Ruhlmann-Kleider (2019, arXiv:1904.08851)，FKP 场定义为：

$$F(\mathbf{r}) = n_g(\mathbf{r}) - \alpha n_s(\mathbf{r})$$

其中 $n_g(\mathbf{r})$ 是观测星系密度，$n_s(\mathbf{r})$ 是随机目录密度（通过 Poisson 采样和选择函数加权）。

观测的星系密度为：

$$n_g(\mathbf{r}) = W(\mathbf{r}) \{1 + \delta(\mathbf{r})\}$$

其中 $W(\mathbf{r})$ 是选择函数，$\delta(\mathbf{r})$ 是密度反差。

### 2.3 带 IC 约束的密度场

施加 global IC 约束后，观测的密度场变为：

$$\delta_{cic}(\mathbf{r}) = W(\mathbf{r}) \left\{ \delta(\mathbf{r}) - \int d^3x \, W_{ic}(\mathbf{x}) \delta(\mathbf{x}) \epsilon_{ic}(\mathbf{r}, \mathbf{x}) \right\} \tag{2.7}$$

其中：
- 对于 global IC：$\epsilon_{glo}(\mathbf{r}, \mathbf{x}) = 1$
- 对于 radial IC：$\epsilon_{rad}(\mathbf{r}, \mathbf{x})$ 有更复杂的形式

选择函数为：

$$W_{ic}(\mathbf{r}) = W(\mathbf{r}) \int d^3x \, W(\mathbf{x}) \epsilon_{ic}(\mathbf{r}, \mathbf{x}) \tag{2.8}$$

---

## 3. 功率谱空间的 GIC 修正

### 3.1 Window 函数矩阵

在 Fourier 空间，带 IC 约束的功率谱观测值可以写成：

$$\hat{P}^{obs}_\ell(k) = (W_{\ell\ell'})_{ij} P_{\ell'j} - (W^{GIC}_{\ell\ell'})_{ij} P^{theo}_{\ell'j} \tag{A.4}$$

其中 $W_{\ell\ell'}$ 是 window 函数矩阵，描述巡天几何对功率谱的卷积效应。

### 3.2 GIC 的 Window 矩阵

GIC 贡献的 window 矩阵为：

$$\left(W^{GIC}_{\ell\ell'}\right)_{ij} = \frac{(W_{\ell 0})_{i0}}{(W_{00})_{00}} (W_{0\ell'})_{0j} \tag{A.5}$$

### 3.3 简化形式

文献中常见的简化形式：

$$P_{cic}(k) = P_c(k) - \frac{P_c(0)}{1} |\tilde{W}(k)|^2 \tag{2.23}$$

其中 $\tilde{W}(k)$ 是选择函数的 Fourier 变换，且归一化为 $|\tilde{W}(k)|^2 = 1$ 当 $k \ll 1\,h/\mathrm{Mpc}$。

**重要**：这个简化形式只保留了完整的 3 项 GIC 修正中的最后一项 (IC_glo,glo)。

### 3.4 完整的 3 项修正

完整的 GIC 修正包含 3 项（de Mattia & Ruhlmann-Kleider 2019 的 eq. 2.20）：

$$\xi_{cic}(s) = \text{term (2.9a)} - \text{term (2.9b)} - \text{term (2.9c)} + \text{term (2.9d)}$$

其中：
- **(2.9a)**: $\int d^3x \, W(\mathbf{x})W(\mathbf{x}+\mathbf{s}) \xi(s)$ — 真实 2PCF 乘 window
- **(2.9b) & (2.9c)**: cross-terms — density 与 IC 项的交叉关联
- **(2.9d)**: IC 项的自相关 — 在大尺度趋于常数

---

## 4. 配置空间 2PCF 的 GIC 效应

### 4.1 Landy-Szalay 估计器

Landy-Szalay 估计器定义为：

$$\hat{\xi}(s) = \frac{DD(s) - 2DR(s) + RR(s)}{RR(s)}$$

如果省略除以 $RR(s)$ 这一步，得到的就是 eq. 2.7 中描述的量。

### 4.2 2PCF 的 GIC 效应形式

在配置空间，GIC 对 2PCF 的影响可以分解为：

$$\xi_{cic}(s) = \xi^{true}(s) - 2\,IC^{\delta,glo}(s) + IC^{glo,glo}(s)$$

其中每一项都是 $\xi(r)$ 与 window 函数的卷积。

### 4.3 IC 项的表达式

**Density-IC cross term (2.9b):**

$$IC^{\delta,glo}_\ell(s) = \frac{2\ell + 1}{4\pi} \int d\Omega_s \int \Delta^2 d\Delta \int d^3x \, W(\mathbf{x})W(\mathbf{x}+\mathbf{s}) W_{glo}(\mathbf{x}+\Delta) \xi_p(\Delta) L_\ell(\hat{\eta}_s \cdot \hat{s}) L_p(\hat{\eta}_\Delta \cdot \hat{\Delta}) \epsilon_{glo}$$

**IC-IC auto-correlation term (2.9d):**

$$IC^{glo,glo}_\ell(s) = \int \Delta^2 d\Delta \sum_p \frac{4\pi}{2p+1} \xi_p(\Delta) W_{glo,glo}^{\ell p}(s, \Delta)$$

### 4.4 简化的常数近似

当 $s$ 很大时，window 函数的自相关 $W_{glo,glo}(s)$ 趋于常数，导致：

$$\xi_{cic}(s) \approx \xi^{true}(s) - C_{IC}$$

其中：

$$C_{IC} \approx \frac{\int d^3r \, \xi(\mathbf{r}) W_{glo}(\mathbf{r})}{V_{eff}}$$

**关键洞察**：这个常数与 $P(k \to 0)$ 直接相关！

---

## 5. Landy-Szalay 估计器与 IC 的关系

### 5.1 估计器中的 IC

Landy-Szalay 估计器中的 random-random pair counts $RR(s)$ 隐含包含了 window 函数信息：

$$RR(s) \propto \int d^3r_1 d^3r_2 \, W(\mathbf{r}_1) W(\mathbf{r}_2) \delta_D(\mathbf{r}_1 - \mathbf{r}_2 - \mathbf{s})$$

这使得 LS 估计器在大尺度天然包含了 IC 效应。

### 5.2 IC 的实用近似公式

从理论推导，IC 常数可以用 RR pair counts 近似为：

$$C_{IC} \approx \frac{\sum_i RR(s_i) \, \xi^{model}(s_i)}{\sum_i RR(s_i)}$$

这个公式说明：
1. **IC 不是任意的常数**
2. **可以从理论和 RR pair counts 预测**
3. **依赖于 window 函数的几何**

---

## 6. DESI 分析中的 IC 处理

### 6.1 Global IC 在 DESI 中的重要性

根据 DESI DR1 PNG 测量文章 (arXiv:2411.17623)：

> "The GIC contribution is **directly proportional to $P_{\ell=0}(k \to 0)$**, such that this contribution depends on the value of $f^{loc}_{NL}$ used to evaluate the theory."
>
> "The convolved monopole of the power spectrum converges at very large scales to a **non-zero value** such that the contribution of the GIC is **4 (resp. 10) times larger** for a situation with $f^{loc}_{NL} = 20$ compared to $f^{loc}_{NL} = 0$."

### 6.2 DESI 的结论

对于 DESI DR1 LRG 和 QSO 样本：
- **LRG** (0.4 < z < 1.1): GIC 贡献在最大尺度与 $\Delta f_{NL} = 1$ 的信号可比
- **QSO** (0.8 < z < 3.1): GIC 贡献即使在 $f_{NL} = 20$ 时也可忽略
- **最终选择**: 在数据拟合中**忽略 GIC 贡献**，因为它不显著影响测量精度

### 6.3 Radial IC (RIC)

DESI 还考虑了 **Radial Integral Constraint (RIC)**，源于 redshift shuffling 方法：
- 随机目录的红移直接从数据目录中采样
- 这压低了径向模式
- 通过修改 window 函数 $W \to W - W_{RIC}$ 来修正

### 6.4 为什么 DESI 可以忽略 GIC？

1. **体积足够大**: DESI 的有效体积远大于我们的 toy cut-sky
2. **k_min 足够大**: DESI 使用 $k_{min} = 0.006\, h/\mathrm{Mpc}$，避开了 GIC 主导的大尺度
3. **灵敏度足够高**: $f_{NL}$ 的灵敏度约 14.5，GIC 贡献远小于统计误差

---

## 7. PNG 对 GIC 的影响

### 7.1 核心关系

**GIC ∝ P(k → 0)**

由于 local PNG 在大尺度产生 scale-dependent bias:

$$P(k) \propto 1 + f_{NL} \cdot \frac{2\Delta_\Phi}{k^2 T(k)}$$

导致 $P(k \to 0) \to \infty$，使得 GIC 效应被显著放大。

### 7.2 量化关系

从 DESI 的 Figure 30：
- $f_{NL} = 0$: GIC 贡献作为基准
- $f_{NL} = 10$: GIC 贡献增大约 4 倍 (QSO) 或更多 (LRG)
- $f_{NL} = 20$: GIC 贡献增大约 10 倍

### 7.3 对我们分析的启示

我们的 toy cut-sky:
- **fnl0**: C_IC ≈ -2.2 × 10⁻⁵
- **fnl100**: C_IC ≈ -5.7 × 10⁻⁵ (约 2.5 倍差异)

这与 $P(k \to 0)$ 随 $f_{NL}$ 增大的预期一致。

### 7.4 关键未解问题

对于 **非高斯场**，GIC 的完整理论公式是否仍然适用？

de Mattia & Ruhlmann-Kleider (2019) 的推导基于二阶统计量（功率谱/2PCF），假设场是平稳的。对于 PNG 增强的大尺度信号：

1. **高阶关联函数的贡献？** 可能需要考虑三点点函数等
2. **非高斯的 window 卷积？** 目前的 window 矩阵框架假设高斯随机场
3. **Large-scale PNG 的 1/k² 效应？** 可能导致 GIC 的尺度依赖更复杂

---

## 8. 实用计算方法

### 8.1 从功率谱预测 IC

1. **计算 window 函数矩阵** $W_{\ell\ell'}$（从 random catalog）
2. **计算理论功率谱** $P^{theo}(k)$（包含 $f_{NL}$）
3. **代入公式** $(W^{GIC}_{\ell\ell'})_{ij}$ 得到 GIC 修正
4. **Hankel 变换** 得到 2PCF 空间的 IC 效应

### 8.2 从 2PCF/RR pair counts 预测 IC

$$C_{IC} \approx \frac{\sum_i RR(s_i) \cdot \xi^{model}(s_i)}{\sum_i RR(s_i)}$$

需要：
- **RR pair counts**: 从 pycorr 的 TwoPointCorrelationFunction 结果中获取
- **理论模型** $\xi^{model}(s)$: 从建模代码 (model_3_20_fast.ipynb) 得到

### 8.3 实用修正策略

**策略 1: 经验修正**
- 用模拟测量 $C_{IC} = \xi_{cut} - \xi_{box}$
- 直接应用到观测数据

**策略 2: 理论预测 + 模拟校准**
- 计算理论预测的 IC
- 用模拟检验预测准确性
- 必要时做小修正

**策略 3: 在拟合中包含 IC 作为 nuisance parameter**
- 把 $C_{IC}$ 当作自由参数
- 或用 theory prior 约束其范围

### 8.4 下一步验证计划

1. **保存 RR pair counts**: 修改 pycorr 测量代码，保存 RR(s)
2. **计算理论 $\xi^{model}(s)$**: 使用 model_3_20_fast.ipynb 的 FullDiscrete 方法
3. **预测 $C_{IC}$**: 用上面的公式计算
4. **比较预测 vs 观测**: 检验 fnl0 和 fnl100 的 IC 常数是否符合理论预期

---

## 9. 参考文献

### 核心文献

1. **de Mattia & Ruhlmann-Kleider (2019)**, *"Integral constraints in spectroscopic surveys"*, arXiv:1904.08851
   - 最完整的 GIC/RIC 理论框架
   - 包含 radial 和 angular IC 的推广

2. **Beutler & McDonald (2021)**, *"Power spectrum multipoles with window function decoupling"*, arXiv:2106.06324
   - Window 函数矩阵的实用计算方法
   - Power spectrum 多极矩分析

3. **DESI Collaboration (2024)**, *"DESI 2024 VI: Cosmological Constraints from BAO and RSD"*, arXiv:2404.03002
   - DESI 完整的 BAO + RSD 分析
   - Window matrix 框架的实际应用

4. **DESI Collaboration (2024)**, *"DESI 2024 VII: Primordial Non-Gaussianity"*, arXiv:2411.17623
   - PNG 测量的完整分析
   - **关键**: GIC 与 $f_{NL}$ 的关系
   - Appendix A.6 详细讨论 GIC

### 经典文献

5. **Peacock & Nicholson (1991)**, *"The clustering of radio galaxies"*, MNRAS 253, 307
   - 最早的 IC 系统性讨论之一

6. **Hamilton (1993)**, *"Measuring omega with peculiar velocities"*, MNRAS 262, 717
   - IC 的早期理论处理

7. **Landy & Szalay (1993)**, *"Statistical and computational conccepts of correlation functions"*, ApJ 412, 64
   - Landy-Szalay 估计器的原始文献

8. **Rimes & Hamilton (2005)**, *"Uncorrelated modes of the spatial power spectrum"*, MNRAS 358, L25
   - 信息量分析的 IC 讨论

### Window Function 相关

9. **Ross et al. (2017)**, *"The clustering of galaxies in SDSS-III BOSS"*, MNRAS 464, 1168
   - Window function 的实用计算

10. **Wilson et al. (2017)**, *"The clustering of galaxies in the completed SDSS-IV"*, MNRAS 470, 3872
    - Window matrix 在 BAO 分析中的应用

---

## 附录 A: 公式汇总

### 功率谱空间

$$P^{obs}_\ell(k) = (W_{\ell\ell'})_{ij} P_{\ell'j} - \frac{(W_{\ell 0})_{i0}}{(W_{00})_{00}} (W_{0\ell'})_{0j} P^{theo}_{\ell'j}$$

### 配置空间 (简化)

$$\xi^{obs}(s) \approx \xi^{true}(s) - C_{IC}$$

其中 $C_{IC} \propto P(k \to 0)$。

### IC 常数近似

$$C_{IC} \approx \frac{\sum_i RR(s_i) \cdot \xi^{model}(s_i)}{\sum_i RR(s_i)}$$

---

## 附录 B: 关键数值对比

| 样本 | C_IC (观测) | $P(k \to 0)$ 相对比例 | DESI 预期 |
|------|-------------|------------------------|-----------|
| fnl0 | -2.2 × 10⁻⁵ | 1× | 可忽略 |
| fnl100 | -5.7 × 10⁻⁵ | ~2.5× | 可能不可忽略 |
| DESI LRG f_NL=20 | — | ~4× | 接近可忽略 |
| DESI QSO f_NL=20 | — | ~10× | 可忽略 |

---

*笔记整理日期: 2026-03-29*
*基于 arXiv:1904.08851, arXiv:2411.17623, arXiv:2106.06324 等文献*
