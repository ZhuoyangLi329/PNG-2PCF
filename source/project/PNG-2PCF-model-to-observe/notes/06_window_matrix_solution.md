# 从 Full-Discrete 2PCF 建模到真实 Survey：Window 效应的处理方案

*文档整理日期：2026-03-30*

---

## 1. 问题的核心

### 1.1 你的方法链条

你的 Full-Discrete 方法论的核心是：

```
P_theory(k_q) [离散 k_q = k_f √q]
    → Full-Discrete 求和
    → ξ_theory(s)

P_data(k_q) [离散 k_q]
    → Full-Discrete 求和
    → ξ_data(s)

→ 对比两者，拟合 cosmology 参数
```

关键假设：**P_data 的每个 bin 必须是"干净"的离散模式，不被 window 污染。**

| | Box (已解决) | 真实 Survey (待解决) |
|---|---|---|
| $k$ 网格 | 周期边界，$k_f = 2\pi/L$，离散正交 | 被 window 卷积污染，模式耦合 |
| $g_q$ 多重数 | 可精确枚举 $q = n_x^2 + n_y^2 + n_z^2$ | 无法定义 |
| $P(k)$ 拟合 | ✅ 直接做 | ❌ 无法直接做 |

### 1.2 为什么这是根本问题

Box 里，你测量到一套**离散的、可枚举的** $P(k_q)$ 数据点，然后：
1. 用这套离散 $P(k_q)$ 拟合 $b_1, f_{\rm NL}, p$ 等参数
2. 用这套参数代入同样的离散 $k_q$ 做 Full-Discrete 求和
3. 得到 $\xi_{\rm theory}(s)$，和测量的 $\xi_{\rm data}(s)$ 对比

到了真实 survey，你**测量到的不是 $P(k_q)$，而是**：

$$\hat{P}(k) = \int \frac{d^3k'}{(2\pi)^3} \, |W(\mathbf{k}-\mathbf{k}')|^2 \, P(k')$$

Survey window $W(\mathbf{r})$ 把**不同 $k$ 模式耦合**在一起了。每个 $\hat{P}(k)$ bin 里都包含了无穷多个离散 $k_q$ 的混合。**你无法反解出"纯净的" $P(k_q)$。**

---

## 2. 解决思路：Window Matrix 前向建模

### 2.1 核心思想

**你的 Full-Discrete 求和的 $k$ 端不需要改变。Window 效应应该作用在"把理论变成观测"的那一步，而不是作用在"观测反推理论"的反解上。**

换句话说：
- **理论端永远有干净的离散 $k_q$**：因为理论是你自己定义的，$k_f$ 和 $g_q$ 是你设定的
- **Window 效应是"污染观测端"的**：应该用前向建模的方式，把干净的理论 $P$ 卷积上 window，变成"污染后的预测"，然后和观测数据比较

### 2.2 数学框架

**Step 1: 干净的理论端（不变）**

$$\mathbf{p} = \left[P(k_1), P(k_2), \ldots, P(k_{N_q})\right]^T$$

由你的理论模型给出（desilike + PNG bias）。

**Step 2: Window Matrix**

从 random catalog 计算 Window Matrix $W_{ik}$：

$$\hat{p}_i = \sum_k W_{ik} \cdot p_k$$

其中：
- $\hat{p}_i$：观测空间第 $i$ 个 $k$ bin 的功率谱（被 window 污染）
- $W_{ik}$：**Window Matrix**，编码了 survey geometry 如何把真实功率谱"污染"成观测功率谱

$W_{ik}$ 的计算方法：
1. 取 random catalog，加权后做 FFT，得到 $|Q(\mathbf{k})|^2$
2. 按观测的 $k$ binning 规则做投影
3. 从 $N_k$ 个"只有 $k_k = 1$ 其他为 0"的输入分别计算响应，构建矩阵

**Step 3: 前向卷积**

对每个 cosmology 参数候选：

$$\hat{\mathbf{p}}^{\rm pred} = \mathbf{W} \cdot \mathbf{p}$$

这给出了"如果用这套参数，在当前 survey geometry 下会观测到什么"的预测。

**Step 4: 变换到配置空间**

$$\hat{\xi}^{\rm pred}(s) = \mathbf{H} \cdot \hat{\mathbf{p}}^{\rm pred} + C_{\rm IC}(\mathbf{p})$$

其中 $\mathbf{H}$ 是 Hankel 变换矩阵：

$$H_{qs} = \frac{g_q}{g_q^{\rm total}} \cdot \frac{\sin(k_q s)}{k_q s}$$

你的 Full-Discrete 求和就是计算 $\mathbf{H} \cdot \hat{\mathbf{p}}^{\rm pred}$。

加上 $C_{\rm IC}$ 修正（你已有的 Path2 结果）。

**Step 5: 拟合**

$$\chi^2 = \sum_{i,j} \left[\hat{\xi}_{\rm data}(s_i) - \hat{\xi}^{\rm pred}(s_i)\right] C_{ij}^{-1} \left[\hat{\xi}_{\rm data}(s_j) - \hat{\xi}^{\rm pred}(s_j)\right]$$

### 2.3 矩阵的维度

| 矩阵 | 维度 | 说明 |
|------|------|------|
| $\mathbf{p}$ | $N_q \sim 10^4$ | 离散 $k_q$ 的理论功率谱 |
| $\mathbf{W}$ | $N_k^{\rm obs} \times N_q$ | Window Matrix（从 random 计算） |
| $\mathbf{H}$ | $N_s \times N_k^{\rm obs}$ | Hankel 变换矩阵 |
| $\hat{\mathbf{p}}^{\rm pred}$ | $N_k^{\rm obs}$ | 前向卷积后的观测空间功率谱 |
| $\hat{\xi}^{\rm pred}$ | $N_s$ | 最终的配置空间预测 |

关键加速点：$\mathbf{W}$ 和 $\mathbf{H}$ 都是**常数矩阵**（不依赖 cosmology 参数）。每次 likelihood 计算只需：
- $\hat{\mathbf{p}}^{\rm pred} = \mathbf{W} \cdot \mathbf{p}$（矩阵乘向量）
- $\hat{\xi}^{\rm pred} = \mathbf{H} \cdot \hat{\mathbf{p}}^{\rm pred} + C_{\rm IC}$

这两个矩阵预计算一次，之后每步 MCMC 只需两次矩阵乘法，计算量极小。

---

## 3. 与你当前工作的对应关系

### 3.1 Window Matrix vs GIC Path2

| | 你的 GIC Path2 | Window Matrix 方案 |
|---|---|---|
| **处理对象** | $k=0$ mode 的常数偏移 | 所有 $k$ 模式的尺度耦合 |
| **效应** | IC 产生的常数 $C_{\rm IC}$ | Window 产生的 mode coupling |
| **公式** | $C_{\rm IC} = \frac{1}{2\pi^2}\int k^2 P(k)\tilde{W}(k)dk$ | $\hat{\mathbf{p}} = \mathbf{W} \cdot \mathbf{p}$ |
| **信息来源** | RR pair counts ($\tilde{W}(k)$) | Random catalog 的 FFT ($W_{ik}$) |
| **复杂度** | 简单（一个常数偏移） | 复杂（完整矩阵） |
| **必要程度** | ⚠️ GIC 在真实 survey 中不是重点 | **✅ 核心：解决 $P(k)$ 拟合的问题** |

> **关键纠正**：GIC 的常数偏移在真实 survey 中**不重要**，因为你的最终比较是在 $\xi(s)$ 空间做的，而 $\xi(s)$ 空间的 window 效应远比一个常数偏移复杂。你的 Full-Discrete 框架在真实 survey 中的真正障碍是：没有干净的 $P(k_q)$ 可以拟合。

### 3.2 Window Matrix 的物理含义

Window Matrix 编码了两件事：

1. **Mode coupling（模式耦合）**：真实 $k_q$ 处的功率会"泄漏"到邻近的观测 $k$ bin
2. **Survey volume 的有限性**：最大尺度模式被截断

对 PNG 分析来说：
- PNG 的 $1/k^2$ 信号主要在**极低 $k$** 端
- Window Matrix 告诉你：极低 $k$ 端的模式如何被 survey 的有限体积"混合"到可观测的 $k$ 范围
- 这直接影响了 $P(k)$ 的低 $k$ 行为，进而影响 Full-Discrete 求和的 $\xi(s)$ 预测

### 3.3 你的当前工作处于哪个阶段

| 你目前完成的工作 | 对应 Window Matrix 框架的哪部分 |
|---|---|
| Cut-sky mock 生成 | 定义 survey geometry（angular mask + radial selection） |
| LS 估计器测量 $\xi_{\rm cutsky}$ | $\hat{\xi}_{\rm data}$ 已就绪 |
| GIC Path2 ($C_{\rm IC}$) | 只处理了 $k=0$ 偏移，**不完整** |
| P(k) 拟合 (desilike) | **还没有**：Box 里用的是干净 $k_q$，survey 里不能用同样的方法 |
| Window Matrix | **待做** |

---

## 4. 实操方案

### 4.1 在 Cut-sky Mock 上验证

在真实数据上应用之前，先在 cut-sky mock 上验证 Window Matrix 方法：

**目标**：用 Window Matrix 前向建模，在**不使用 box 真实 $\xi$ 的情况下**，仅从 mock 的 random catalog 和理论模型，预测出和 cut-sky 测量一致的 $\hat{\xi}^{\rm pred}(s)$。

**步骤**：

1. **计算 cut-sky 的 Window Matrix $\mathbf{W}^{\rm cutsky}$**
   - 从 cut-sky 的 random catalog 做 FFT
   - 用 pypower 的 `WindowTwoPointCounter.powersep` 或类似功能
   - 输出矩阵维度：$N_k^{\rm obs} \times N_q$

2. **构建干净的理论 $\mathbf{p}^{\rm box}$**
   - 用 box 的真实 $P(k_q)$ 作为 ground truth
   - 或用 desilike 模型参数生成

3. **前向卷积**
   $$\hat{\mathbf{p}}^{\rm pred} = \mathbf{W}^{\rm cutsky} \cdot \mathbf{p}^{\rm box}$$

4. **Hankel 变换**
   $$\hat{\xi}^{\rm pred}(s) = \mathbf{H} \cdot \hat{\mathbf{p}}^{\rm pred} + C_{\rm IC}$$

5. **对比**
   - $\hat{\xi}^{\rm pred}(s)$ vs $\xi_{\rm cutsky}(s)$（用 LS 估计器测量的）
   - 如果吻合 → Window Matrix 方法有效
   - 如果不吻合 → 检查 $C_{\rm IC}$ 是否足够（可能需要更精细的 window 修正）

**验证成功的标准**：
$$\chi^2 / {\rm ndof} \approx 1$$
$$\left|\hat{\xi}^{\rm pred}(s) - \xi_{\rm cutsky}(s)\right| / \sigma \lesssim 1$$

### 4.2 代码实现

DESI/pypower 里已经有 Window Matrix 的计算工具：
- `pypower` 的 `PowerSpectrum晴window` 相关功能
- `nbodykit` 的 `ConvolvedFFTPower`
- `desilike` 的 window 处理模块

建议流程：
```
Mock random catalog
    → pypower.compute_window_matrix()
    → W_matrix
    → 前向卷积 + Full-Discrete → ξ_pred
    → 和 mock data 的 ξ 比较
```

### 4.3 下一步：从 Cut-sky Mock 到真实 Survey

Cut-sky mock 上的 Window Matrix 验证成功后，过渡到真实 Survey 的路径是：

| | Cut-sky Mock | 真实 Survey |
|---|---|---|
| Random catalog | 从 box 构造，精确匹配 geometry | DESI 提供的 randoms，已匹配 |
| Window Matrix | 从 mock randoms 计算 | 从 DESI randoms 计算 |
| $\xi_{\rm data}$ | LS 估计器测量 | LS 估计器测量 |
| $C_{\rm IC}$ | mock 上标定 | 用 $\mathbf{W}$ 计算 |
| 验证 | ✅ 可以用 box ground truth 验证 | ❌ 无法验证，但信任 mock 结果 |

**注意**：真实 Survey 和 Cut-sky Mock 的关键区别在于红移演化：
- Cut-sky：固定红移 snapshot（或简单范围 $0.4 < z < 1.0$）
- 真实 Survey：$b_1(z)$、$f(z)$、$\bar{n}(z)$ 都随红移变化

需要在 effective redshift 近似下处理，或对每个红移 bin 分别计算 Window Matrix。

---

## 5. 关键注意事项

### 5.1 为什么 GIC 常数偏移"不重要"

在 $\xi(s)$ 空间，survey window 的效应**远不是**一个简单的常数偏移：

- Window 的 Hankel 变换 $H_{qs}$ 在 $s$ 空间不是 delta 函数
- 不同 $s$ 处的 $\xi(s)$ 受到不同 $k_q$ 组合的加权混合
- $C_{\rm IC}$ 只是 $k=0$ 的极限行为，在 $s \gtrsim 100$ Mpc/h 时近似成立

所以 GIC Path2 在 cut-sky mock 上是**有用的诊断工具**，但不是真实 Survey 分析的最终方案。

### 5.2 P(k) 拟合的替代路径

如果你不想完全走 Window Matrix 这条路，还有另一种思路：

**直接在 mock 上建立 $\xi(s)$ 的响应模板**：

1. 在 mock 上，用 $f_{\rm NL} = 0$ 的 box 计算 $\xi_0(s)$（clean baseline）
2. 计算不同 $f_{\rm NL}$ 值对应的 $\Delta\xi_{f_{\rm NL}}(s)$ 模板
3. 真实数据的 $\xi_{\rm obs}(s)$ 和这些模板做 $\chi^2$ 拟合

这本质上是 Brown et al. (2024) ConKer 方法的思想——**不做 $P(k)$ 参数拟合，而是直接做 $\xi$ 模板拟合**。

但这和你当前"从 $P(k)$ 出发建模 $\xi$"的方法论有出入。Window Matrix 前向建模是保持你方法论一致性的自然延伸。

### 5.3 Window Matrix 的计算代价

Window Matrix $\mathbf{W}$ 的维度可能很大：
- $N_q \sim 10^4$（你的 Full-Discrete 的离散 $k_q$ 数量）
- $N_k^{\rm obs} \sim 50$（观测 $k$ bins）

直接计算 $N_k^{\rm obs} \times N_q$ 矩阵需要大量 mock FFTs。

更高效的做法：用随机矢量方法（randomized SVD / Hessian 方法）压缩 Window Matrix：
- 生成 $N_{\rm rand} \ll N_q$ 个随机矢量
- 对每个矢量计算 window 卷积响应
- 用这些响应构建低秩近似 $\tilde{\mathbf{W}}$

这在 DESI 的 BAO 分析中是标准做法。

---

## 6. 总结

| 问题 | 解决方案 |
|------|---------|
| 真实 survey 没有干净的 $k_q$ 做 $P(k)$ 拟合 | **不用反推：用 Window Matrix 前向卷积** |
| Window 效应如何处理 | $\hat{\mathbf{p}}^{\rm pred} = \mathbf{W} \cdot \mathbf{p}$ |
| Full-Discrete 的 $k$ 端要不要改 | **不改**：$\mathbf{p}$ 永远用你自己定义的离散 $k_q$ |
| GIC Path2 的角色 | **诊断工具**：帮助理解 window 效应；不是最终分析方案 |
| 下一步做什么 | **在 cut-sky mock 上验证 Window Matrix 前向建模** |

**核心公式**：

$$\boxed{\hat{\xi}^{\rm pred}(s) = \mathbf{H} \cdot \mathbf{W} \cdot \mathbf{p}(b_1, f_{\rm NL}, p) + C_{\rm IC}(\mathbf{p})}$$

其中 $\mathbf{H}$ 和 $\mathbf{W}$ 是常数矩阵，$\mathbf{p}$ 由 cosmology 参数决定。

---

## 7. 参考

- DESI 2024 fNL analysis: arXiv:2411.17623 (Chaussidon et al.)
- Window Matrix methodology: Beutler & McDonald (2021), arXiv:2106.06324
- Brown et al. (2024) ConKer: arXiv:2403.18789
- Integral Constraint: de Mattia & Ruhlmann-Kleider (2019), arXiv:1904.08851
