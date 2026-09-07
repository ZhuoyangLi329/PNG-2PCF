# Mission 8 第一阶段研究记录：离散功率谱的解析形式

## 1. 这一阶段要回答什么

窗口法之所以看起来“有效”，我目前认为不是因为它真的修正了某个低 `k` 物理，而是因为它在数值上部分模仿了**有限盒中的离散 Fourier 模**。

所以第一阶段先不碰 `kbin` 细节，而是只做下面三件事：

1. 先在一维写清楚离散 `P(k)` 的解析形式；
2. 再推广到三维周期盒；
3. 把三维离散 `P(k)` 对应的 `xi(r)` 写成明确的离散求和式。

如果这一步成立，那么后面 Mission 8 的主要问题就会变成：

> 当前连续 Hankel 积分中的 IR 问题，是否本质上来自把“盒子里本来只有离散低 `k` 模”的东西，错误地近似成了从 `k=0` 开始的连续谱？


## 2. 先固定 Fourier 约定

为了避免后面所有 `2pi` 和体积因子混乱，我先固定一套与有限盒周期边界最自然的约定。

对三维周期盒 `V = L^3`，允许的波矢为

$$
\mathbf{k} = \frac{2\pi}{L}(n_x,n_y,n_z) \equiv k_f(n_x,n_y,n_z),\qquad n_i \in \mathbb{Z}.
$$

定义离散 Fourier 展开

$$
\delta(\mathbf{x})=\sum_{\mathbf{k}} \delta_{\mathbf{k}} e^{i\mathbf{k}\cdot\mathbf{x}},
$$

其中

$$
\delta_{\mathbf{k}} = \frac{1}{V}\int_V d^3x\ \delta(\mathbf{x})e^{-i\mathbf{k}\cdot\mathbf{x}}.
$$

在这个约定下，平移不变性给出

$$
\langle \delta_{\mathbf{k}}\delta_{\mathbf{k}'}^* \rangle
= \frac{P(\mathbf{k})}{V}\,\delta^K_{\mathbf{k},\mathbf{k}'}.
$$

这就是有限盒中最常见的结果：**每个离散模的方差与连续功率谱 `P(k)` 成正比，但多一个 `1/V`。**


## 3. 一维：离散功率谱的解析形式

### 3.1 一维离散模

在一维长度为 `L` 的周期区间中，

$$
k_n = n k_f,\qquad k_f = \frac{2\pi}{L},\qquad n\in \mathbb{Z}.
$$

若定义

$$
\delta(x)=\sum_n \delta_n e^{ik_n x},
$$

则一维两点函数为

$$
\xi(x)=\langle \delta(0)\delta(x)\rangle
= \sum_n \langle |\delta_n|^2 \rangle e^{ik_n x}
= \frac{1}{L}\sum_n P_{1D}(k_n)e^{ik_n x}.
$$

如果零模去掉，而且 `P_{1D}(k)` 是偶函数，那么也可以写成

$$
\xi(x)=\frac{2}{L}\sum_{n=1}^{\infty} P_{1D}(k_n)\cos(k_n x).
$$

### 3.2 把它写成 delta function 的连续表示

一维连续 Fourier 关系是

$$
\xi(x)=\int_{-\infty}^{\infty}\frac{dk}{2\pi}P_{1D}(k)e^{ikx}.
$$

为了让这个积分精确退化成上面的离散和，只需定义

$$
P_{1D}^{\mathrm{disc}}(k)=\frac{2\pi}{L}\sum_{n\in\mathbb{Z}}P_{1D}(k_n)\delta_D(k-k_n).
$$

代回去立刻得到

$$
\xi(x)=\frac{1}{L}\sum_n P_{1D}(k_n)e^{ik_n x}.
$$

因此在一维中，离散功率谱的“解析形式”非常直接：它就是一列位于 `k = n k_f` 的 delta spike。

这个结果有一个非常重要的含义：

> 在有限盒里，`0 < |k| < k_f` 本来就没有任何模。因此一维离散理论里根本不存在“把 `kmin` 向 0 压低导致额外 IR 面积不断长出来”的问题。


## 4. 三维：真正 relevant 的不是 `n k_f`，而是离散 shell

三维周期盒里允许的波矢是

$$
\mathbf{k}=k_f(n_x,n_y,n_z).
$$

但对球平均的单极功率谱来说，真正 relevant 的不是向量本身，而是其模长

$$
k = |\mathbf{k}| = k_f \sqrt{q},\qquad q = n_x^2+n_y^2+n_z^2.
$$

因此三维球平均功率谱的非零支持不在

$$
k_f, 2k_f, 3k_f, \dots
$$

这样的等间距点上，而在

$$
k_f,\ \sqrt{2}k_f,\ \sqrt{3}k_f,\ 2k_f,\ \sqrt{5}k_f,\ \sqrt{6}k_f,\dots
$$

这些由“三平方和”决定的离散 shell 上。

更进一步，不是每个整数 `q` 都能写成三个平方和。根据 Legendre 三平方定理：

$$
q = 4^a(8b+7)
$$

形式的整数不能写成三个平方和。

所以像 `q=7, 15, 23, 28, ...` 这样的 shell 在三维离散盒子里根本不存在。

这点非常关键，因为它说明：

> 三维有限盒的最低几个低 `k` shell 既是离散的、又是不均匀分布的，而且很多半径位置根本没有模式。


## 5. 三维相关函数的离散求和式

从离散 Fourier 展开直接计算：

$$
\xi(\mathbf{r})=\langle \delta(\mathbf{x})\delta(\mathbf{x}+\mathbf{r})\rangle
= \sum_{\mathbf{k}}\frac{P(\mathbf{k})}{V}e^{i\mathbf{k}\cdot\mathbf{r}}.
$$

如果再对 `r` 的方向做球平均，就有

$$
\frac{1}{4\pi}\int d\Omega_r\, e^{i\mathbf{k}\cdot\mathbf{r}} = j_0(kr),
$$

因此得到单极相关函数

$$
\xi_0(r)=\frac{1}{V}\sum_{\mathbf{k}}P(\mathbf{k})j_0(kr).
$$

若同一个 shell 上的功率只依赖模长 `k_q = k_f\sqrt{q}`，记其球平均值为 `\bar P_q`，并记该 shell 的简并度为 `g_q`，则

$$
\xi_0(r)=\frac{1}{V}\sum_{q\in\mathcal{Q}} g_q\,\bar P_q\, j_0(k_q r).
$$

这就是我认为 Mission 8 最核心的解析结果。

它告诉我们：

1. 低 `k` 部分不是连续积分；
2. 它是一个**有限盒离散 shell 的球贝塞尔求和**；
3. 只要每个 shell 的 `\bar P_q` 是有限的，那么 `xi_0(r)` 在数学上天然有限；
4. 所谓连续模型中的 `IR divergence`，在有限盒离散表达里并不会以同样形式出现。


## 6. 如何把三维离散 shell 写成径向 delta function

标准连续单极 Hankel 关系是

$$
\xi_0(r)=\int_0^\infty \frac{k^2 dk}{2\pi^2} P_0(k) j_0(kr).
$$

如果我们希望这个连续积分**严格等价**于上面的离散 shell 求和式

$$
\xi_0(r)=\frac{1}{V}\sum_q g_q \bar P_q j_0(k_q r),
$$

那么只需定义一个“径向 delta-shell 功率谱”

$$
P_{0,\mathrm{disc}}(k)
= \frac{2\pi^2}{V}\sum_{q\in\mathcal{Q}}
\frac{g_q \bar P_q}{k_q^2}\,\delta_D(k-k_q).
$$

把它代回 Hankel 关系：

$$
\int_0^\infty \frac{k^2 dk}{2\pi^2} P_{0,\mathrm{disc}}(k) j_0(kr)
= \frac{1}{V}\sum_q g_q \bar P_q j_0(k_q r),
$$

正好成立。

因此，Mission 8 任务书里想写的那种

$$
P(k)=P(k_f)\delta_D(k-k_f)+P(\sqrt{2}k_f)\delta_D(k-\sqrt{2}k_f)+\cdots
$$

在数学上是完全可行的；只是**精确系数不是简单的 `P(k_q)` 本身，而是要带上 `g_q / k_q^2` 和体积归一化因子**。

如果把这些因子漏掉，后续的 `xi(r)` 系数会错。


## 7. 这对窗口法意味着什么

上面的结果给了一个很直接的解释框架：

1. 当前连续模型做的是

$$
\xi_0(r)=\int_{k_{\min}}^{k_{\max}}\frac{k^2dk}{2\pi^2}P_0(k)j_0(kr),
$$

并把 `P_0(k)` 当作在低 `k` 连续存在。

2. 但有限盒真实情况是

$$
\xi_0(r)=\frac{1}{V}\sum_q g_q \bar P_q j_0(k_q r),
$$

最低几个低 `k` 只有极少数离散 shell，而且它们的位置也不是均匀的。

3. 因此当 PNG 项让 `P(k)` 在低 `k` 有很强权重时，连续积分近似会系统性夸大“`k \to 0` 这段本来不存在的连续面积”。

所以我目前的判断是：

> `IR window` 的主要作用，很可能不是在模拟真实的低 `k` 物理，而是在数值上把“连续积分虚构出来的超低 `k` 贡献”压回到更接近有限盒离散 shell 的水平。

这正是 Mission 8 后续要进一步量化验证的东西。


## 8. 下一步怎么做

基于第一阶段推导，下一步我会做两件具体的数值研究：

### 8.1 枚举真实 3Gpc 盒子的最低若干个离散 shell

输出：

1. `q`
2. `k_q = k_f sqrt(q)`
3. `g_q`
4. 前几个 shell 对应的实际 `k` 数值

这会告诉我们 3Gpc 下最低几个离散模究竟稀疏到什么程度。

### 8.2 搭一个最小的 hybrid 原型

把

$$
\xi_0(r)=\xi_{\mathrm{low,disc}}(r)+\xi_{\mathrm{high,cont}}(r)
$$

作为 Mission 8 的核心候选模型：

1. 前几个低 `k` shell 用离散求和；
2. 更高 `k` 仍用现有连续 best-fit `P(k)` 做 FFTLog。

如果这个原型能在没有经验窗口的情况下接近当前最好结果，那么窗口法就有了明确的“有限盒离散模代理”解释。


## 9. 一个直接的 toy 数值检查

为了验证上面的解释框架，我又做了一个非常直接的 toy 对比：

1. 连续模型：用

$$
P(k)=\frac{1}{k^3}
$$

做 Hankel 积分，并改变 `kmin`；
2. 离散模型：在 `L=3000 Mpc/h` 的盒子里，用真实离散 shell

$$
\xi_0(r)=\frac{1}{V}\sum_q g_q P(k_q)j_0(k_q r)
$$

直接求和。

这个 toy `P(k) ~ 1/k^3` 不是为了模拟完整物理，而是专门制造一个“连续积分对 IR 极敏感”的例子。

数值结果如下：

### 9.1 连续积分

在 `r=200 Mpc/h` 时，

- `kmin = 2e-3` 时，`xi_0 = 6.85e-02`
- `kmin = 1e-3` 时，`xi_0 = 1.03e-01`
- `kmin = 5e-4` 时，`xi_0 = 1.38e-01`
- `kmin = 1e-4` 时，`xi_0 = 2.08e-01`
- `kmin = 1e-5` 时，`xi_0 = 2.20e-01`

也就是说，只要不断向更小 `kmin` 积分，`xi_0` 就会持续抬升。

### 9.2 离散 shell 求和

同样在 `L=3000 Mpc/h` 的盒子里，离散求和给出：

- `r = 100` 时，`xi_0 = 1.19e-01`
- `r = 200` 时，`xi_0 = 8.06e-02`
- `r = 400` 时，`xi_0 = 4.72e-02`

这个结果有两个关键点：

1. **离散和本身是有限的**；
2. 它的数量级更接近 `kmin ~ k_f = 2pi/L` 的连续结果，而不是 `kmin -> 0` 的连续结果。

所以这个 toy 检查支持了我当前的判断：

> 真实有限盒中的低 `k` 贡献更像“若干离散 shell 的有限求和”，而不是“从 0 开始的一段连续 IR 面积”。

这也说明为什么经验 IR 窗口会有用：它是在数值上把后者往前者的行为上拉。
