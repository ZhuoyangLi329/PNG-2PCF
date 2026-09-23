# Global Integral Constraint（GIC）详解：从有限巡天平均密度到 PNG 2PCF

> 更新日期：2026-07-13  
> 适用范围：有限体积 galaxy survey、Landy--Szalay 2PCF、FKP/Yamamoto 功率谱、local PNG 大尺度分析  
> 文档定位：这是一份从文献推导到本项目 Task 4.2/4.3/4.4 实现的教学与口径说明；若与早期探索笔记冲突，以本文和最新 task audit 为准。

## 0. 三十秒版本

真实宇宙的平均星系密度未知，有限巡天只能用本次观测的数据估计它。这个操作会强制加权后的观测密度场满足

$$
\int d^3x\,F(\mathbf x)=0,
$$

因此巡天内部的平均过密度模式被投影掉。这就是 **Global Integral Constraint（GIC，全局积分约束）**。

如果把归一化 survey window 记为 $W_N$，则最直观的场级表达式是

$$
\bar\delta_W=\int d^3y\,W_N(\mathbf y)\delta(\mathbf y),
\qquad
\delta_{\rm est}(\mathbf x)=\delta(\mathbf x)-\bar\delta_W.
$$

二点函数不是简单地“减一次平均值”，而是两个括号相乘：

$$
\boxed{
\langle\delta_{\rm est,1}\delta_{\rm est,2}\rangle
=\xi_{12}-\langle\delta_1\bar\delta_W\rangle
-\langle\bar\delta_W\delta_2\rangle
+\langle\bar\delta_W^2\rangle .}
$$

因此完整 GIC 包含两个 **density--IC cross terms** 和一个 **IC--IC auto term**。常见的

$$
\xi_{\rm obs}(s)\simeq\xi(s)-\sigma_W^2
$$

是有用的传统单项近似，不是任意 survey geometry 下的完整答案。

对本项目保留的 full local-PNG 模型，$f_{\rm NL}^2$ 项使 $P_h(k)\sim k^{-3}$，标准 Hankel 积分在 $k\to0$ 处呈对数发散。GIC 投影使低 $k$ kernel 从 $j_0(ks)\to1$ 变为一个在 $k=0$ 处为零的组合，因而让可观测 2PCF 对任意低 $k$ cutoff 收敛。

---

## 1. 先分清三个容易混淆的概念

| 概念 | 来源 | 数学作用 | 是否等于 GIC |
|---|---|---|---|
| Survey window convolution | 天空 footprint、$n(z)$、完备度和权重 | 混合不同 $k$ 和不同 multipoles | 否 |
| Global integral constraint | 用同一份有限巡天数据估计全局平均密度 | 投影掉一个或若干个加权平均模式 | 是 |
| Finite-box / mother-box cutoff | simulation 周期母盒只含离散非零模式 | 不存在 $k<2\pi/L_{\rm box}$ 的模式 | 否 |

普通 window effect 在场级是“把真实场乘上选择函数”；GIC 则来自 window 的归一化振幅 $\alpha$ 也由数据决定。前者造成 mode coupling，后者额外强制某个加权积分为零。两者在实际 estimator 中同时出现，但不能把它们当成同一个效应。

有限母盒 cutoff 也不是 GIC。它是在 mock 中真实缺少超母盒模式；GIC 是即使真实宇宙含有这些模式，有限巡天用自身估计平均密度后也无法区分其中近似常数的部分。

---

## 2. 从 FKP 场严格看 GIC 怎样产生

### 2.1 数据、random 和归一化

de Mattia & Ruhlmann-Kleider（2019）从 FKP 场出发：

$$
F(\mathbf x)=n_g(\mathbf x)-\alpha n_s(\mathbf x),
$$

其中 $n_g$ 是带权数据密度，$n_s$ 是采样 survey selection function 的带权 random 密度。归一化由同一份数据决定：

$$
\alpha
=\frac{\int d^3x\,n_g(\mathbf x)}{\int d^3x\,n_s(\mathbf x)}
=\frac{\sum_{g}w_g}{\sum_{r}w_r}.
$$

把无 clustering 时的期望带权密度记为 $W(\mathbf x)$，则

$$
n_g(\mathbf x)=W(\mathbf x)[1+\delta(\mathbf x)],
\qquad n_s(\mathbf x)\propto W(\mathbf x).
$$

定义归一化 window

$$
W_N(\mathbf x)=\frac{W(\mathbf x)}{\int d^3y\,W(\mathbf y)},
\qquad \int d^3x\,W_N(\mathbf x)=1,
$$

可得

$$
\boxed{
F(\mathbf x)=W(\mathbf x)
\left[\delta(\mathbf x)-\bar\delta_W\right],
\qquad
\bar\delta_W=\int d^3y\,W_N(\mathbf y)\delta(\mathbf y).}
$$

于是

$$
\int d^3x\,F(\mathbf x)=0.
$$

这不是关于真实宇宙场 $\delta$ 的物理定律，而是 estimator 的定义。它只说明：我们构造出来的观测场对 window-weighted constant mode 没有响应。

### 2.2 一个最直观的长波例子

设真实场中有一个波长远大于巡天尺寸的模式。在整个 footprint 内，它看起来几乎是一个常数正偏置。因为我们不知道宇宙真实平均密度，会把这个正偏置解释成“本巡天的平均数密度略高”，然后在 $\alpha n_s$ 中减掉。于是这个超长波模式的大部分信号消失。

因此 GIC 的本质可以概括为：

> **有限巡天无法同时从自身数据中估计平均密度，又保留对同一个平均模式的 clustering 响应。**

### 2.3 每个 chunk 单独归一化会更强

如果 NGC、SGC 或多个 imaging region 各自使用独立的 $\alpha_c$，那么每个区域都满足自己的零平均约束：

$$
\int_{\mathcal V_c}d^3x\,F(\mathbf x)=0.
$$

这不再只投影一个全局模式，而是每个 chunk 各投影一个平均模式，所以 IC 通常更强。分析中必须记录 $\alpha$ 是全 footprint 一个，还是分 region、分 tracer、分 redshift bin 计算。

---

## 3. 为什么二点函数会出现四项

### 3.1 场级展开

令

$$
\delta'_i=\delta(\mathbf x_i)-\bar\delta_W.
$$

则

$$
\begin{aligned}
\langle\delta'_1\delta'_2\rangle
={}&\langle\delta_1\delta_2\rangle
-\langle\delta_1\bar\delta_W\rangle
-\langle\bar\delta_W\delta_2\rangle
+\langle\bar\delta_W^2\rangle\\
={}&\xi(\mathbf x_1-\mathbf x_2)-C_1-C_2+\sigma_W^2,
\end{aligned}
$$

其中

$$
C_i=\int d^3y\,W_N(\mathbf y)\,
\xi(\mathbf x_i-\mathbf y),
$$

以及

$$
\boxed{
\sigma_W^2
=\int d^3y\,d^3z\,
W_N(\mathbf y)W_N(\mathbf z)\xi(\mathbf y-\mathbf z).}
$$

$\sigma_W^2$ 是 window 内平均密度 $\bar\delta_W$ 的方差。它依赖理论功率谱、survey geometry、选择函数和全部分析权重。

### 3.2 对固定 separation 做 pair average

定义某个 separation bin $a$ 的 random-pair window：

$$
RR_a
=\int d^3x_1d^3x_2\,
W_1W_2\,\Theta_a(|\mathbf x_1-\mathbf x_2|),
$$

其中 $\Theta_a$ 选择该 bin。除以 $RR_a$ 后，偶数 multipoles 的完整结构可概括为

$$
\boxed{
\xi^{\rm cic}_\ell(s)
=\xi^{\rm c}_\ell(s)
-IC^{\delta,{\rm ic}}_\ell(s)
-IC^{{\rm ic},\delta}_\ell(s)
+IC^{{\rm ic},{\rm ic}}_\ell(s).}
$$

这对应 de Mattia & Ruhlmann-Kleider（2019）的式（2.20）：

- $\xi^{\rm c}$：普通 density--density 项及 survey window；
- 两个负号项：density 与被投影平均模式的交叉相关；
- 最后一个正号项：被投影平均模式的自相关。

对于对称 pair average 和偶数 multipoles，两个 cross terms 常相等。若把它们记成 $B(s)$，把 auto term 记成 $C$，就得到便于理解的形式

$$
\boxed{\xi_{\rm full}(s)=\xi(s)-2B(s)+C.}
$$

关键是 $B(s)$ 一般依赖 separation 和几何，不保证严格等于常数 $C$。

### 3.3 Landy--Szalay 为什么没有自动消除 GIC

Landy--Szalay estimator 为

$$
\hat\xi(s)=\frac{DD(s)-2DR(s)+RR(s)}{RR(s)}.
$$

$RR(s)$ 很好地校正了“靠近边界的点少了多少配对”这类普通几何 pair loss。但是 $DD$ 和 $DR$ 中使用的密度场已经通过数据决定的 $\alpha$ 做了零平均投影，所以除以 $RR$ 不能把被投影掉的模式重新创造出来。

换句话说：

- $RR$ 除法处理普通 pair geometry；
- GIC 来自平均密度的 data-dependent normalization；
- 两者相关，但不是同一操作。

---

## 4. 传统常数近似：$\xi-\sigma_W^2$

### 4.1 它从哪里来

许多配置空间分析使用

$$
\boxed{\xi_{\rm obs}(s)\simeq\xi(s)-C_{\rm IC}.}
$$

为了满足 estimator 的加权积分为零，可取

$$
C_{\rm IC}
=\frac{\sum_a RR_a\,\xi_a^{\rm model}}
{\sum_a RR_a}.
$$

若 $RR_a$ 是与测量完全一致的加权 pair counts，且求和覆盖 window 中全部 separation support，这个 $C_{\rm IC}$ 才是相应 window 下的平均相关，即 $\sigma_W^2$。只在拟合区间（例如 $50<s<350\,h^{-1}{\rm Mpc}$）内求和一般不满足这个条件。该近似可理解为用一个常数代表完整的 $-2B(s)+C$：

$$
-2B(s)+C\simeq-C.
$$

这要求在相关尺度上 $B(s)\simeq C$。它在某些几何和尺度范围内很好，但不是数学恒等式。

### 4.2 用 RR kernel 在 $k$ 空间计算

对各向同性 monopole，可从 random pairs 构造

$$
W_2(k)
=\frac{\sum_a RR_a\,\overline{j_0}(k;a)}
{\sum_a RR_a},
$$

其中 $\overline{j_0}(k;a)$ 是和测量 radial bin 完全一致的 shell average。于是

$$
\boxed{
\sigma_W^2(\theta)
=\frac{1}{2\pi^2}\int_0^\infty dk\,k^2
P_0(k;\theta)W_2(k).}
$$

这里 $\theta$ 包含 $f_{\rm NL}$、$b_1$、$p$、RSD/FoG 参数和可能的 stochastic 参数。因此 PNG 分析中的 $\sigma_W^2$ 通常必须在每个 likelihood 点更新，不能只在 fiducial cosmology 算一次后永远固定。

一个真正由同一 window 自相关得到的 $W_2$ 应满足

$$
W_2(0)=1.
$$

在理想连续采样下，它等价于归一化 $|\widetilde W(\mathbf k)|^2$ 的方向平均，因此应为非负。若一个被称为 $W_2$ 的数值 kernel 大幅振荡到负值，应检查它究竟是 window auto-power、某个带符号 response，还是归一化与离散 Hankel 变换出了问题。

### 4.3 为什么我们把它叫 single-term approximation

de Mattia & Ruhlmann-Kleider 回顾的传统 Fourier 近似为

$$
P^{\rm cic}(k)=P^{\rm c}(k)-P^{\rm c}(0)|\widetilde W(k)|^2.
$$

作者指出，该式只用一个有效修正项，而完整推导有三个 IC 项。他们也发现，在其展示的 CMASS-like global-IC 例子中，单项近似和完整结果很接近。

因此合适的表述是：

> $\xi-\sigma_W^2$ 有明确文献背景，是一个 parameter-dependent single-term GIC approximation；它不是 complete GIC 的同义词。

---

## 5. Fourier 空间与 window matrix 表述

### 5.1 普通 window convolution

忽略 multipole 记号时，普通 survey window 给出

$$
P^{\rm conv}(\mathbf k)
=\int\frac{d^3q}{(2\pi)^3}
|\widetilde W(\mathbf k-\mathbf q)|^2P(\mathbf q).
$$

实际分析把它离散为

$$
\mathbf p^{\rm conv}=\mathbf W\,\mathbf p^{\rm theory}.
$$

$\mathbf W$ 混合不同 $k$ bins，也会混合 $P_0,P_2,P_4,\ldots$。

### 5.2 GIC 作为额外的投影矩阵

DESI 的 power-spectrum 实现写成

$$
\boxed{
\mathbf p^{\rm obs}
=(\mathbf W-\mathbf W^{\rm GIC})\mathbf p^{\rm theory}.}
$$

对其采用的 global-IC matrix convention，

$$
\left(W^{\rm GIC}_{\ell\ell'}\right)_{ij}
=\frac{(W_{\ell0})_{i0}}{(W_{00})_{00}}
\,(W_{0\ell'})_{0j}.
$$

这显示 GIC 不是在测量之后随手减一个固定数字，而是一个作用于完整理论向量的线性 response。对 PNG 而言，理论向量随 $f_{\rm NL}$ 改变，所以 GIC response 也随参数改变。

### 5.3 $k\to0$ 的必要性质

由于观测 FKP 场满足 $\int F=0$，完整 GIC 模型应使观测 monopole 在 $k\to0$ 时趋于零。若 forward-convolved 模型在 $k=0$ 仍对 constant mode 有非零响应，说明 IC projection、window normalization 或 chunk 定义不一致。

---

## 6. 为什么 local PNG 让 GIC 特别重要

### 6.1 scale-dependent bias

local PNG 的大尺度 bias 可写成

$$
\Delta b(k,z)
=f_{\rm NL}^{\rm loc}\frac{b_\phi(z)}{\mathcal M(k,z)},
$$

其中

$$
b_\phi=2\delta_c(b_1-p),
\qquad
\mathcal M(k,z)\propto k^2T(k)D(z).
$$

所以 $k\to0$ 时

$$
\Delta b(k)\propto k^{-2}.
$$

完整 tracer power 为

$$
P_h(k)=[b_1+\Delta b(k)]^2P_m(k).
$$

在 $P_m(k)\propto k^{n_s}$ 且 $n_s\simeq1$ 时：

| 项 | 低 $k$ 标度 | 对标准 $\xi$ 积分的影响 |
|---|---:|---|
| Gaussian：$b_1^2P_m$ | $k^{n_s}$ | 收敛 |
| linear PNG：$2b_1\Delta bP_m$ | $k^{n_s-2}\sim k^{-1}$ | $k^2P\sim k$，收敛 |
| quadratic PNG：$(\Delta b)^2P_m$ | $k^{n_s-4}\sim k^{-3}$ | $k^2P\sim k^{-1}$，对数发散 |

因此本项目的 2PCF 红外问题来自保留 full PNG square 后的 $f_{\rm NL}^2$ 项。如果某个分析只保留 $f_{\rm NL}$ 线性项，形式上的对数发散不会出现，但大尺度信号仍然对 window 与 IC 极其敏感。

### 6.2 GIC 怎样使 2PCF 收敛

标准 monopole Hankel 变换为

$$
\xi_0(s)=\frac{1}{2\pi^2}
\int_0^\infty dk\,k^2P_0(k)j_0(ks).
$$

在 single-term GIC 图像下，变成

$$
\boxed{
\xi_0^{\rm GIC}(s)=\frac{1}{2\pi^2}
\int_0^\infty dk\,k^2P_0(k)
[\overline j_0(k;s)-W_2(k)].}
$$

当 $k\to0$，

$$
\overline j_0(k;s)=1+\mathcal O(k^2),
\qquad
W_2(k)=1+\mathcal O(k^2),
$$

所以

$$
\overline j_0-W_2=\mathcal O(k^2).
$$

原来 $f_{\rm NL}^2$ 项的 integrand 近似为 $k^{-1}$；乘上这个 $k^2$ cancellation 后变成约 $k^{+1}$，因此

$$
\int_0^{k_{\min}}dk\,k
\propto k_{\min}^2
$$

并在 $k_{\min}\to0$ 时收敛。这就是 `gic_kmin_convergence.pdf` 中低 $k_{\min}$ 曲线重合的解析原因。

### 6.3 这不等于“GIC 恢复了所有超巡天信息”

GIC 只是让理论预测与 estimator 的不可观测模式一致，并使可观测组合的红外行为良好。被数据归一化投影掉的平均模式本身仍不可从这一个 tracer auto-correlation 中恢复。若要利用超巡天信息，需要外部 mean-density calibration、多 tracer、与 CMB lensing 交叉相关或显式 super-sample response 等额外信息。

---

## 7. GIC、RIC 与 AIC 的统一理解

de Mattia & Ruhlmann-Kleider 使用一个通用 projector：

$$
\delta^{\rm cic}(\mathbf x)
=W(\mathbf x)\left[
\delta(\mathbf x)
-\int d^3y\,W_{\rm ic}(\mathbf y)
\epsilon_{\rm ic}(\mathbf x,\mathbf y)\delta(\mathbf y)
\right].
$$

不同 IC 的区别只在于“在哪个子空间内估计并减去平均模式”。

### 7.1 Global IC（GIC）

$$
\epsilon_{\rm glo}(\mathbf x,\mathbf y)=1.
$$

整个 footprint 或每个独立 chunk 估计一个平均密度。它主要压制比 survey 更大的模式。

### 7.2 Radial IC（RIC）

当 random 的 radial distribution 由数据自身估计，例如 shuffled redshifts 或很细的 $n(z)$ bin reweighting，定义

$$
\epsilon_{\rm rad}(\mathbf x,\mathbf y)
=
\begin{cases}
1,& \chi_x,\chi_y\text{ 位于同一径向 bin},\\
0,& \text{其他}.
\end{cases}
$$

于是每个径向 bin 中、跨该 bin angular footprint 平均的 overdensity 被压低。需要特别强调：标准 binned/shuffled RIC 不是“每条独立视线都各自积分为零”；它是由 random 的径向归一化区域定义的约束。

RIC 自动包含 global normalization，因为把所有 radial bins 合并后就回到 GIC。因此一个已正确实现的 RIC 模型通常不应再额外减一次独立 $\sigma_W^2$，否则可能 double count。

完整 RIC 仍然有三项：

$$
\xi^{\rm RIC}
=\xi^{\rm c}
-IC^{\delta,{\rm rad}}
-IC^{{\rm rad},\delta}
+IC^{{\rm rad},{\rm rad}}.
$$

它通常比 GIC 更具尺度依赖和各向异性，因此可能明显影响 quadrupole，也可能模仿 monopole 中的 PNG 大尺度信号。

### 7.3 Angular IC（AIC）

如果 imaging-systematics regression、angular pixel normalization 或其他 angular cleaning 从数据中拟合并移除天空模式，就会产生 AIC。它投影的是 angular subspace，而不是单一全局常数。

DESI DR1 PNG power-spectrum 分析把这些效应写成

$$
\mathbf W\rightarrow
\mathbf W-\mathbf W^{\rm RIC}-\mathbf W^{\rm AIC}.
$$

这是一种前向建模：先把理论作用上与数据处理一致的 mode projection，再与测量比较。

---

## 8. FKP 权重与 random catalog 在 GIC 中扮演什么角色

### 8.1 有效 window 包含所有分析权重

在实际分析中，$W$ 不是只有几何 mask。它近似包含

$$
W(\mathbf x)
\propto
\bar n(\mathbf x),
w_{\rm FKP}(\mathbf x),
w_{\rm sys}(\mathbf x),
w_{\rm comp}(\mathbf x)\cdots.
$$

其中

$$
w_{\rm FKP}(\mathbf x)=
\frac{1}{1+\bar n(\mathbf x)P_0}.
$$

改变 $P_0$ 会改变不同红移区域对平均模式与 pair window 的贡献，因此也会改变 $W_2$、$\sigma_W^2$ 和 RIC/AIC kernel。不能用 unweighted random 计算 window，却用 FKP-weighted DD/DR/RR 做测量。

### 8.2 random 越密只会减小数值噪声

增加 random 数量能更精确地采样选择函数和 RR kernel，但不会消除 GIC。即使 random 无限密，只要 $\alpha$ 仍由有限数据估计，积分约束仍然存在。

### 8.3 random 的生成策略决定你施加了哪种 IC

| random radial policy | 典型约束 |
|---|---|
| 独立已知的真实 $n(z)$ | 主要是 GIC |
| 从数据 redshifts shuffled/resampled | RIC + GIC |
| 用数据在细 $z$ bins 内重加权 random | binned RIC + GIC |
| 每个 chunk 单独做上述操作 | 每个 chunk 各自施加更强约束 |

因此 catalog provenance 是理论模型的一部分，而不是单纯的数据工程细节。

---

## 9. GIC 与 FullDiscrete 的关系

### 9.1 FullDiscrete 回答的是 simulation support 问题

周期母盒允许的模式为

$$
\mathbf k_{\mathbf n}=\frac{2\pi}{L}\mathbf n,
\qquad \mathbf n\in\mathbb Z^3,
$$

且通常排除 $\mathbf n=0$。FullDiscrete 2PCF 使用这些实际存在的离散非零模式及其 degeneracy 求和，而不是把连续 $P(k)$ 任意延伸到 $k=0$。

这解决的是“这批 finite-box mocks 实际包含哪些初始模式”。它并没有自动模拟 survey 用数据估计 mean density 的操作。

### 9.2 在 finite-mock closure 中两者都要匹配

若 lightcone 来自 $L=2000\,h^{-1}{\rm Mpc}$ 的 Abacus mother box，则 closure test 的理论不能假装存在 $k<2\pi/L$ 的独立模式。同时，测量使用 matching random 和 data-dependent normalization，又会施加 GIC 或 RIC。

所以 finite-mock closure 的逻辑是

$$
\text{mother-box mode support}
+\text{survey window}
+\text{estimator IC projection}.
$$

### 9.3 真实 survey 没有 mother-box cutoff

真实宇宙没有一个已知的 simulation mother box。理论 $k$ grid 应向足够低的 $k$ 延伸，并用完整 window/IC response 做 $k_{\min}$ convergence。不能把某个 mock 的 $2\pi/L_{\rm box}$ 直接当成真实 survey 的物理 cutoff。

---

## 10. 与本项目当前实现的对应关系

### 10.1 Task 4.2：cubic subbox

Task 4.2 的 `formal-GIC` 使用 parameter-dependent cubic-window $\sigma_W^2(\theta)$，并在每个 likelihood 点更新。它是经过数值收敛检查的传统 single-term GIC approximation。

稳妥表述：

> **Parameter-dependent cubic-window single-term GIC correction combined with the FullDiscrete mother-box mode support.**

它不应被自动称为 de Mattia & Ruhlmann-Kleider 三项公式的完整实现，除非另有 full-vs-single-term A/B 验证。

### 10.2 Task 4.3：Abacus halo lightcone

Task 4.3 的 random redshifts 从每个 phase 的 halo redshifts 有放回重采样，属于 shuffled/data-resampled radial policy，因此测量含 RIC。

最新路线使用 $\Delta\chi=2\,h^{-1}{\rm Mpc}$ binned approximation 和 factorized radial kernel。当前代码口径为一个 **radial single-term approximation**：

$$
\text{model}=\text{noIC}-\text{radial single-term response}.
$$

它不再额外减 global $\sigma_W^2$，因为 radial normalization 已包含 global normalization；但两个 density--RIC cross terms 尚未加入。因此不能表述为完整的

$$
-IC^{\delta,{\rm rad}}
-IC^{{\rm rad},\delta}
+IC^{{\rm rad},{\rm rad}}.
$$

相关入口：

- `codes/task43/task43_ric_singleterm.py`
- `codes/task43/task43_build_ric_factorized_kernel.py`
- `codes/task43/task43_make_ric_factorized_operator.py`
- `outputs/task43_outputs/ric_singleterm/audits/task43_ric_singleterm_audit.json`

### 10.3 Task 4.4：Abacus LRG lightcone

Task 4.4 现有 `formal-GIC` 理论收敛测试使用 FKP-weighted LRG-all RR kernel：

$$
\xi_{0}^{\rm formal}(s)
=\frac{1}{2\pi^2}\int dk\,k^2P_0(k)
[\overline j_0(k;s)-W_2(k)].
$$

它很好地展示了 local-PNG $f_{\rm NL}^2$ 红外项如何在 $j_0-W_2$ 中相消，但从文献分类上仍是 parameter-dependent RR-window single-term treatment。

对于外部 LSScat random 是否还隐含完整 RIC，必须追溯其 radial redshift generation 和 normalization provenance；仅凭 random 与 data 的 $n(z)$ 相似，不能判断完整 RIC 已被建模。

### 10.4 项目中推荐统一使用的术语

| 当前实现 | 推荐名称 | 不建议名称 |
|---|---|---|
| $\xi-\sigma_W^2(\theta)$ | parameter-dependent single-term GIC approximation | complete GIC |
| $j_0-W_2$ 连续积分 | RR-window single-term GIC treatment | exact survey IC |
| Task 4.3 factorized radial response | radial single-term approximation | full RIC |
| de Mattia 三项 kernel 全部实现 | complete GIC/RIC model | — |

---

## 11. 从测量到 likelihood 的推荐实现流程

### Step 1：先审计 estimator

必须明确：

1. 使用 Landy--Szalay、Yamamoto 还是其他 estimator；
2. $\alpha$ 在哪些区域分别计算；
3. random 的 angular/radial selection 如何生成；
4. 是否 shuffled data redshifts；
5. data/random 使用哪些总权重；
6. separation、$k$ 和 multipole binning；
7. shot-noise subtraction convention。

没有这些信息，无法唯一确定 IC projector。

### Step 2：构造与测量完全一致的 window

window 必须使用与 DD/DR/RR 或 FKP field 相同的：

- footprint；
- $n(z)$；
- FKP/systematic/completeness weights；
- NGC/SGC 或 chunk splitting；
- radial/angular normalization policy。

### Step 3：选择模型精度层级

1. **诊断层**：$\xi-C_{\rm IC}$ 常数修正；
2. **single-term 层**：parameter-dependent $W_2$ 或有效 window-matrix response；
3. **完整层**：显式计算 density--IC 两个 cross kernels 和 IC--IC auto kernel；
4. **RIC/AIC 层**：把实际 data-derived selection projector 一并加入。

论文或报告必须说明采用哪一层。

### Step 4：在 likelihood 内更新参数依赖

对 local PNG，低 $k$ 功率随 $f_{\rm NL}$ 和 $b_\phi$ 强烈变化。若使用 $C_{\rm IC}$、$W^{\rm GIC}P$ 或 RIC/AIC response，应对每个参数点作用于当前理论向量，而不是把 fiducial correction 当作固定 data offset。

### Step 5：让 covariance 与 estimator 匹配

正确的 mean model 不代表 covariance 自动正确。IC 会改变可观测 mode subspace；analytic covariance、mock covariance 或 RascalC 输入都应与相同 window、weight、RR basis 和 normalization 对齐。

---

## 12. 必做的数值验证

### 12.1 Constant-field null test

输入 $\delta(\mathbf x)=\delta_0$ 后，IC-corrected field 应严格为零。若不为零，通常是 $\alpha$、window normalization 或 chunk definition 错误。

### 12.2 $W_2(0)=1$

RR auto-kernel 在 $k=0$ 必须归一化到 1。该测试应在任何插值、subsampling 和 shell averaging 之后再次执行。

### 12.3 RR-weighted integral constraint

当 $a$ 覆盖 window 的完整 pair-separation support 时，配置空间预测应满足与 estimator 对应的离散条件：

$$
\sum_a RR_a\,\xi_a^{\rm obs}\simeq0,
$$

允许的误差由 random sampling 和数值截断决定。只对 likelihood 的拟合区间求和通常不会等于零，不能把它误判为 GIC 失败。

### 12.4 $k_{\min}$ convergence

逐步降低理论积分下限，并比较每个 data bin 的变化相对 covariance sigma。对 full local-PNG + GIC 的低 $k$ residual，预期渐近收敛约为 $k_{\min}^{n_s+1}\simeq k_{\min}^{2}$。

### 12.5 Random-subsample convergence

使用至少两个独立 seed 和多档 random 数量，检查：

- $W_2(k)$；
- $\sigma_W^2$；
- 每 bin 模型变化；
- posterior 中心和宽度。

### 12.6 Full-vs-single-term A/B

保持 data、covariance、fit range、theory support 和参数先验完全相同，只切换

$$
\xi-C
\quad\text{vs}\quad
\xi-2B(s)+C.
$$

报告最大 $|\Delta m_i|/\sigma_i$、$\Delta\chi^2$、$\Delta f_{\rm NL}$ 和误差宽度比。

### 12.7 GIC-only vs shuffled/RIC A/B

同一 mock 分别使用独立平滑 $n(z)$ random 与 data-shuffled random 测量。前者主要测试 GIC，后者测试 RIC；用 matched theory 检查完整 radial response。

---

## 13. 常见误解与纠正

### 误解 1：Landy--Szalay 已经除以 RR，所以没有 GIC

错误。RR 修正普通 pair geometry，但不能恢复被 data-dependent mean normalization 投影掉的模式。

### 误解 2：random 足够多，GIC 就消失

错误。无限 random 只让 selection function 的数值表示更精确；只要平均密度仍从有限数据估计，GIC 就存在。

### 误解 3：GIC 就是把 $k<2\pi/L$ 切掉

错误。hard cutoff 删除一段 Fourier modes；GIC 是一个由 estimator 和 window 定义的投影，通常造成 mode coupling 和参数依赖 response。

### 误解 4：$\xi-\sigma_W^2$ 就是完整 GIC

错误。它是常用且可能很好的 single-term approximation；完整结果还含两个 density--IC cross terms。

### 误解 5：RIC 是每条视线单独减平均

通常错误。binned/shuffled RIC 按 random normalization 使用的 radial bins 和 angular chunks 定义；必须从 catalog construction 确定 projector。

### 误解 6：DESI 某次分析忽略 GIC，所以 GIC 普遍不重要

错误。Chaussidon et al. 的 DESI DR1 power-spectrum 分析是在其样本体积、误差和 fiducial $k_{\min}=0.006\,h\,{\rm Mpc}^{-1}$ 下判断 GIC 可忽略；同一论文同时显示 GIC 对较小体积 LRG 比 QSO 更大，并且显式建模了 RIC/AIC。这个结论不能直接移植到更小体积、更低 $k_{\min}$ 或 full-$f_{\rm NL}^2$ 的 2PCF 分析。

### 误解 7：GIC 使积分收敛，就证明 single-term 模型在所有尺度精确

错误。$k\to0$ cancellation 是必要的红外性质；它不自动证明有限 $k$、有限 $s$ 下遗漏的 cross-term shape 可以忽略。

---

## 14. 报告中可以怎样简短介绍

### 中文一句话

> 有限巡天必须用自身数据估计平均密度，因此会投影掉 survey-window 内的平均模式；我们将这一 integral-constraint response 前向作用于 PNG 理论，使可观测 2PCF 在 $k_{\min}\to0$ 时收敛。

### 英文简洁版

> **Estimating the mean density from the survey projects out the window-weighted mean mode. We forward-model this integral-constraint response, rendering the observable PNG 2PCF infrared safe.**

### 若当前只展示 single-term 模型

> **We use a parameter-dependent RR-window single-term approximation, $\xi_0^{\rm obs}(s)\simeq\xi_0(s)-\sigma_W^2(\theta)$, rather than the complete density--IC cross-term formalism.**

### 若介绍 Task 4.3

> **Because the random redshifts are resampled from each data realization, the measurement contains a radial integral constraint; our current result uses a numerically converged radial single-term approximation.**

---

## 15. 参考文献与本地资料

### 核心文献

1. A. de Mattia & V. Ruhlmann-Kleider, *Integral constraints in spectroscopic surveys*, JCAP 08 (2019) 036, [arXiv:1904.08851](https://arxiv.org/abs/1904.08851). 本地 PDF：[`1904.08851_integral_constraints_spectroscopic_surveys.pdf`](../observelearn/ref/integral_constraint/1904.08851_integral_constraints_spectroscopic_surveys.pdf)。完整 GIC/RIC/AIC 场级与二点函数推导的主要来源。
2. E. Chaussidon et al., *Constraining primordial non-Gaussianity with DESI 2024 LRG and QSO samples*, [arXiv:2411.17623](https://arxiv.org/abs/2411.17623). 本地 PDF：[`2411.17623_DESI_DR1_PNG.pdf`](../observelearn/ref/integral_constraint/2411.17623_DESI_DR1_PNG.pdf)。DESI PNG 分析中 GIC、RIC、AIC window-matrix 处理的主要实例。
3. F. Beutler & P. McDonald, *Unified galaxy power spectrum measurements from 6dFGS, BOSS, and eBOSS*, [arXiv:2106.06324](https://arxiv.org/abs/2106.06324). 本地 PDF：[`2106.06324_window_function_matrix.pdf`](../observelearn/ref/integral_constraint/2106.06324_window_function_matrix.pdf)。window matrix 与 multipole forward convolution 的实用参考。
4. S. D. Landy & A. S. Szalay, *Bias and variance of angular correlation functions*, ApJ 412 (1993) 64. Landy--Szalay estimator 的经典来源。
5. H. A. Feldman, N. Kaiser & J. A. Peacock, *Power-spectrum analysis of three-dimensional redshift surveys*, ApJ 426 (1994) 23. FKP field 与 FKP weighting 的经典来源。

### 与本项目 2PCF PNG 路线相关

6. Z. Brown et al., *Constraining primordial non-Gaussianity from the large scale structure two-point and three-point correlation functions*, [arXiv:2403.18789](https://arxiv.org/abs/2403.18789). 本地 PDF：[`2403.18789v1.pdf`](../observelearn/ref/2403.18789v1.pdf)。配置空间 PNG 分析背景；其建模路线与本文的解析 FullDiscrete 路线不同。
7. 项目教学 notebook：[`de_mattia_2019_gic_2pcf_vs_current_model.ipynb`](de_mattia_2019_gic_2pcf_vs_current_model.ipynb)。用于查看论文完整三项结构与 Task 4.3 历史实现的对应关系。
8. 当前任务总说明：[`agent/task.md`](../agent/task.md)，重点查看 Task 4.2、4.3、4.4 的最新 audit 记录。

### 阅读顺序建议

1. 先读本文第 2--4 节，建立“数据估计平均密度 $\to$ mode projection $\to$ 四项二点函数”的主线；
2. 再读 de Mattia & Ruhlmann-Kleider 的第 2 节，核对完整公式；
3. 读 DESI DR1 PNG 论文第 4.4 节和附录 A.6，理解 window-matrix 实现；
4. 最后对照 Task 4.3 radial single-term audit，区分当前数值收敛结论与完整物理模型完成度。
