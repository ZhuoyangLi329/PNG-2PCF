# 路径2 (连续积分法) 理论文档

## 1. 目标

路径2尝试**不依赖 FFTLog**，直接从连续积分计算 $C_{IC}$，避免 FFTLog 的 taper 近似带来的误差。

## 2. 理论背景

### 2.1 配置空间的 IC 修正

Landy-Szalay 估计器在大尺度（$s \gg 0$）的修正项为：

$$\xi_{LS}(s) \approx \frac{DD - 2DR + RR}{RR} = \xi_{\rm true}(s) - C_{IC}$$

其中 $C_{IC}$ 是 Global Integral Constraint 贡献的常数偏移：

$$C_{IC} = \frac{\int d^3r_1\, d^3r_2\, W(\mathbf{r}_1)W(\mathbf{r}_2)\, \xi(|\mathbf{r}_1-\mathbf{r}_2|)}{\left[\int d^3r\, W(\mathbf{r})\right]^2}$$

在实践中，$W(\mathbf{r})$ 由 random catalog 采样，所以用 pair count 形式：

$$C_{IC} = \frac{\int RR(s)\, \xi(s)\, ds}{\int RR(s)\, ds}$$

其中 $RR(s)$ 是 random-random pair count density。

### 2.2 功率谱空间的对应关系

从 Fourier 变换关系（采用宇宙学标准约定）：

$$\xi(s) = \frac{1}{2\pi^2}\int_0^\infty k^2 dk\, P(k)\, \frac{\sin(ks)}{ks}$$

代入配置空间公式，得到 Fourier 空间的等价形式：

$$C_{IC} = \frac{\int_0^\infty k^2 dk\, P(k)\, \tilde{W}(k)}{2\pi^2\, \int RR(s)\, ds}$$

其中 $\tilde{W}(k) = \int_4\pi s^2 ds\, RR_{\rm norm}(s)\, j_0(ks)$ 是 window 函数的球对称 Fourier 变换（$j_0$ 是球贝塞尔函数）。

**关键性质**：当 $k \to 0$ 时，$j_0(0) = 1$，所以 $\tilde{W}(k \to 0) = 1$（归一化条件）。

### 2.3 简化公式

对于 $k \ll 1/L_{\rm survey}$（即 $k$ 远小于 survey 最大尺度对应的波数），$\tilde{W}(k) \approx 1$。于是：

$$C_{IC} \approx \frac{1}{2\pi^2\, \int RR(s)\, ds}\int_0^\infty k^2 P(k)\, dk$$

注意：
- 这个公式在 $k \to 0$ 时 $k^2 P(k) \to$ 常数（对于有 PNG 的 $P(k) \propto 1/k^2$），积分**收敛**
- 但需要正确的**归一化因子** $2\pi^2 \int RR(s)\, ds$

## 3. 具体计算步骤

### 3.1 从 RR pair counts 计算几何因子

pycorr 的 `RR_norm` 满足 $\sum_i RR_{\rm norm}(s_i) = 1$，其中 $RR_{\rm norm}(s_i)$ 是归一化到 1 的 pair density。

**问题**：$RR_{\rm norm}(s)$ 的单位是什么？

从 pycorr 的定义：
$$RR_{\rm norm}(s)\, ds = \frac{2\, RR_{\rm raw}(s)}{N_R(N_R-1)}$$

其中 $RR_{\rm raw}(s)$ 是原始的 pair count，$N_R$ 是 random catalog 的总数。

$RR_{\rm raw}(s)$ 的定义：
$$RR_{\rm raw}(s)\, ds = d^3r_1\, d^3r_2\, \delta_D(|\mathbf{r}_1-\mathbf{r}_2| - s)\, n_R(\mathbf{r}_1)\, n_R(\mathbf{r}_2)$$

所以 $RR_{\rm raw}$ 的量纲是 $[{\rm length}]^{-2}$。

归一化后 $RR_{\rm norm}$ 的量纲也是 $[{\rm length}]^{-2}$。

### 3.2 体积因子的推导

从 $C_{IC}$ 的配置空间定义：
$$C_{IC} = \frac{\int RR(s)\, \xi(s)\, ds}{\int RR(s)\, ds}$$

- 分子：$\int RR(s)\, ds$ 的量纲是 $[{\rm length}]^{-2}$，乘以无量纲的 $\xi(s)$
- 分母：$\int RR(s)\, ds$ 的量纲也是 $[{\rm length}]^{-2}$

所以 $C_{IC}$ 本身就是无量纲的 ✓

从 Fourier 变换推导：
$$C_{IC} = \frac{\int_0^\infty k^2 P(k)\, dk}{2\pi^2\, V_{\rm eff}}$$

其中有效体积为：
$$V_{\rm eff} = \frac{\left[\int d^3r\, W(\mathbf{r})\right]^2}{\int d^3r\, W^2(\mathbf{r})}$$

对于均匀 $W(\mathbf{r})$ 和球对称 survey 几何：
$$V_{\rm eff} = \Omega_{\rm ang} \cdot D_{\rm comoving}^3 / 3$$

其中 $\Omega_{\rm ang}$ 是角向覆盖面积（steradians），$D_{\rm comoving}$ 是 comoving 深度。

## 4. 实现细节

### 4.1 代码流程

```python
def compute_gic_continuous_integral(P_model, rr_s_mid, rr_counts):
    """
    1. 构建 k 网格（从 k~1e-4 到 k~0.5 h/Mpc，覆盖 k^2 P(k) 的主要贡献范围）
    2. 在 k 网格上评估 P(k)（使用 bestfit fnl, b1）
    3. 计算 integral k^2 P(k) dk（用 Simpson 积分）
    4. 计算 survey 的有效体积 V_eff
    5. 计算 C_IC = integral / (2 * pi^2 * V_eff)
    """

    # k 网格
    k_grid = np.geomspace(1e-4, 0.5, 5000)  # 对数等间距
    pk = P_model(k_grid)

    # 积分
    P_int = simpson(k_grid**2 * pk, k_grid)  # ∫ k^2 P(k) dk

    # 有效体积
    from astropy.cosmology import FlatLambdaCDM
    cosmo = FlatLambdaCDM(H0=67.11, Om0=0.315192)
    chi_04 = cosmo.comoving_distance(0.4).value  # Mpc
    chi_10 = cosmo.comoving_distance(1.0).value

    ang_sr = 10000.0 * (np.pi/180)**2  # Y3 footprint ~ 10000 deg^2
    V_eff = ang_sr * (chi_10**3 - chi_04**3) / 3.0

    # C_IC
    C_IC = P_int / (2 * np.pi**2) / V_eff
    return C_IC
```

### 4.2 关键数值

| 量 | 数值 |
|---|---|
| $\int k^2 P(k) dk$ | ~100 (无量纲*) |
| $V_{\rm eff}$ (cone) | ~3.6×10¹⁰ Mpc³ |
| $2\pi^2$ | ~19.74 |
| $C_{IC}$ (连续积分) | ~5×10⁻¹¹ (错误) |

*注：这里的 P(k) 量纲是 [(Mpc/h)³]，k 的量纲是 [h/Mpc]，所以 k²P 的积分无量纲。

## 5. 当前问题

### 5.1 与路径1的不一致

路径1（RR 加权 FFTLog）给出 $C_{IC} \approx 9.3 \times 10^{-5}$，路径2（连续积分）给出 $C_{IC} \approx 5 \times 10^{-11}$，相差约 $10^6$ 倍。

### 5.2 可能的原因

1. **有效体积计算错误**：V_eff 应该由 randoms 的分布决定，而不是 survey 的几何体积
2. **归一化因子错误**：$2\pi^2$ 的幂次可能不对
3. **RR_norm 的物理意义理解错误**：RR_norm 可能是某种不同的归一化

### 5.3 诊断方向

最直接的方法是：检查路径1和路径2的等价性。

路径1：
$$C_{IC} = \sum_i RR_{\rm norm}(s_i)\, \xi(s_i)$$

路径2（尝试）：
$$C_{IC} = \frac{\int_0^\infty k^2 P(k)\, dk}{2\pi^2\, V_{\rm eff}}$$

如果两者等价，应该有：
$$V_{\rm eff} = \frac{\int_0^\infty k^2 P(k)\, dk}{2\pi^2\, \sum_i RR_{\rm norm}(s_i)\, \xi(s_i)}$$

从数值结果：
$$\frac{9.3\times10^{-5}}{5\times10^{-11}} \approx 1.9\times10^6$$

这意味着 $V_{\rm eff}$ 的正确值应该是：
$$V_{\rm eff}^{\rm true} = \frac{100}{2\pi^2 \times 9.3\times10^{-5}} \approx 1.7\times10^5\ {\rm Mpc}^3$$

但我们的 survey 体积是 $V \approx 3.6\times10^{10}\ {\rm Mpc}^3$，相差 $10^5$ 倍！

这个因子 $10^5$ 可能来自：
- RR_norm 的单位转换（h/Mpc vs Mpc）
- 某个缺失的 $h^3$ 或 $h^{-3}$ 因子

## 6. 下一步

1. **检查量纲**：确认 P(k)、k、s 的单位一致性
2. **检查 V_eff**：用 randoms 的实际分布计算正确的有效体积
3. **数值验证**：用简单的 test case（已知 P(k) 和 RR）验证公式

## 7. 参考文献

- de Mattia & Ruhlmann-Kleider (2019), arXiv:1904.08851
- DESI Collaboration (2024), arXiv:2411.17623
- Feldman, Kaiser & Peacock (1994), arXiv:astro-ph/9304022
