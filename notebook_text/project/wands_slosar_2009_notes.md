# Notebook source mirror

Original: `source/project/wands_slosar_2009_notes.ipynb`

Outputs remain in original notebook.

## Cell 1 (markdown)

# Wands & Slosar (2009) 阅读笔记
## `Scale-dependent bias from primordial non-Gaussianity in general relativity`

**论文**: David Wands, Anže Slosar, 2009, arXiv:0902.1084  
**笔记目的**: 解释这篇文章到底推导了什么、解决了什么疑问、以及它和我们当前 `localPNGmodel` 项目的关系。

---

## 这篇文章一句话在干什么？

这篇文章想回答一个非常具体的问题：

> local primordial non-Gaussianity 产生的星系/halo 的 `scale-dependent bias`
> 在接近或超出 Hubble scale 时，会不会因为广义相对论而改变其
> `1/k^2` 形式？

作者的结论是：

1. **不会。**
2. 关键不是随便选一个 gauge 下的密度扰动，而是要使用
   **comoving-orthogonal gauge** 中的密度扰动。
3. 在这个变量下，Poisson 方程的形式和 Newtonian 情况保持一致，
   所以 PNG 诱导的 `\Delta b \propto 1/k^2` 在超视界尺度也不自动消失。
4. 这会导致理论上的 biased tracer 相关函数出现 **formal divergence**，
   但实际巡天测量的相关函数因为要减去 survey mean，因此仍然是有限的。


## Cell 2 (markdown)

## 1. 阅读这篇文章前，需要先记住什么？

这篇文章默认你已经接受 2008 年左右那条著名结果：
$$
\Delta b(k)\propto \frac{f_{\rm NL}}{k^2 T(k)}.
$$
更具体地，作者在引言里写的形式是
$$
\Delta b = \frac{2 f_{\rm NL}}{\alpha(k)}\frac{\partial \ln n}{\partial \ln \sigma_8},
\tag{10}
$$
其中
$$
\alpha(k)=\frac{2 c^2 T(k) D(z)}{3 \Omega_m H_0^2} k^2.
\tag{11}
$$
对 universal mass function，可化成
$$
\Delta b
= f_{\rm NL}(b_G-1)\,
\frac{3\delta_* \Omega_m H_0^2}{c^2 k^2 T(k) D(z)}.
\tag{12}
$$
所以问题的核心就是：

- `1/k^2` 是不是只是 Newtonian 近似的产物？
- 如果真正用 GR，在超大尺度会不会变成别的东西？

Wands & Slosar 的答案就是：**不是 Newtonian 偶然产物；在正确变量上，GR 不会把这个大尺度形式改掉。**


## Cell 3 (markdown)

## 2. 论文结构总览

这篇文章其实非常短，逻辑也很清楚，分成四步：

### 第一步：写出 GR 一阶约束方程

作者从一般标量扰动的 FRW 度规出发，写出能量约束和动量约束，
再把它们组合成一个 **GR 版 Poisson 方程**。

### 第二步：指出“Poisson 方程长什么样”取决于你用哪个 gauge 的密度扰动

也就是说：

- 如果你拿 **longitudinal gauge** 的密度扰动去看，
  大尺度上会出现看起来像 GR 修正的项。
- 但如果你拿 **comoving-orthogonal gauge** 的密度扰动去看，
  那么 Poisson 方程仍然保持 Newtonian 形式。

### 第三步：用 spherical collapse 说明

真正和 halo collapse / bias 相关的那一个密度扰动，
应该是 **comoving-orthogonal gauge** 里的密度扰动，
而不是 longitudinal gauge 里的局域密度。

### 第四步：讨论一个看似尴尬的结果

既然 `\Delta b \propto 1/k^2` 在 GR 中也不改，
那么 biased tracer 的功率谱在 `k\to 0` 时就会增强得非常厉害，
进而 formal correlation function 会发散。

作者说：

- 这不是观测上的灾难；
- 因为真实 survey 会从自身数据估计平均密度；
- 所以真正测到的是“减去了 survey mean 的相关函数”。


## Cell 4 (markdown)

## 3. 第二节：GR 版 Poisson 方程到底怎么来的？

### 3.1 一般标量扰动度规

作者从最一般的标量扰动 FRW 度规开始：
$$
ds^2 = a^2\Big\{(1+2A)d\eta^2 - 2(\partial_i B) dx^i d\eta
- \big[(1-2\psi)\delta_{ij}+2(\partial_i\partial_j E)\big] dx^i dx^j \Big\}.
\tag{13}
$$
这里：

- `A, B, \psi, E` 都是 gauge-dependent；
- `\delta\rho` 也同样是 gauge-dependent。

所以你不能一上来就问“GR 下的密度和势是什么关系”，
因为**你必须先说清楚你在谈哪个 gauge 的密度**。

---

### 3.2 能量约束 + 动量约束

一阶 Einstein 约束方程写成：
$$
-3\mathcal H (\psi' + \mathcal H A)
+ \partial^2 [\psi + \mathcal H(E'-B)]
= 4\pi G a^2 \delta\rho,
\tag{14}
$$
$$
\partial_i(\psi' + \mathcal H A)
= -4\pi G a^2 \partial_i(\delta q).
\tag{15}
$$
这里的 `\delta q` 是**动量密度的标量势**，也就是
3-momentum 的标量部分：
$$
\partial_i(\delta q)
$$
就是空间动量密度扰动。

这是这篇文章最重要的记号之一。

---

### 3.3 组合成 GR Poisson 方程

把 (14) 和 (15) 组合起来，可以得到
$$
\partial^2[\psi + \mathcal H(E'-B)]
= 4\pi G a^2 \big[\delta\rho - 3\mathcal H \delta q\big].
\tag{16}
$$
然后作者做了两个识别：
$$
-\phi_N = \psi_\ell \equiv \psi + \mathcal H(E'-B),
\tag{17}
$$
$$
\delta\rho_c \equiv \delta\rho - 3\mathcal H \delta q.
\tag{18}
$$
于是式 (16) 变成
$$
\partial^2(-\phi_N) = 4\pi G a^2 \delta\rho_c.
$$
这就是论文的第一核心结论：

> **如果你把 Newtonian potential 识别为 longitudinal gauge 的势，
> 同时把密度识别为 comoving-orthogonal gauge 的密度，**
> **那么 Poisson 方程的形式和 Newtonian 完全一样。**

---

### 3.4 为什么这一步这么关键？

因为许多“GR 会在大尺度修正 `1/k^2`”的直觉，
来自于把 **longitudinal gauge 的局域密度** 直接拿来塞进 Poisson 方程。

作者马上举例说明：

在 longitudinal gauge 中，大尺度上会有
$$
3\mathcal H^2 \phi_N - \partial^2 \phi_N
= 4\pi G a^2 \delta\rho_\ell.
\tag{19}
$$
如果再取超视界极限 `|\partial^2 \phi_N| \ll \mathcal H^2 |\phi_N|`，
就得到
$$
\delta_\ell(x)\simeq -2\phi_N(x).
\tag{20}
$$
这看上去就像：

- 大尺度密度和势成局域比例；
- 于是好像不再有 `1/k^2`。

但作者说：**这并不是计算 bias 时应该使用的那个密度变量。**


## Cell 5 (markdown)

## 4. 第三节：为什么 spherical collapse 选中了 comoving-orthogonal density？

这是整篇文章最关键的部分。

作者不是停留在“不同 gauge 有不同公式”，而是进一步问：

> halo/galaxy bias 到底应该由哪个密度扰动来决定？

他们的回答是：看 **spherical collapse**。

---

### 4.1 内外两个 FRW 区域

外部未扰动区域：
$$
H^2 = \left(\frac{\dot a}{a}\right)^2 = \frac{8\pi G\rho_0}{3 a^3},
\tag{21}
$$
内部 overdense 闭宇宙区域：
$$
\tilde H^2 = \left(\frac{\dot{\tilde a}}{\tilde a}\right)^2
= \frac{8\pi G\rho_0}{3\tilde a^3} - \frac{K}{\tilde a^2}.
\tag{22}
$$
这是标准 spherical collapse 设置。

参数解写成：
$$
\tilde a = \frac{1}{2K}(1-\cos\theta),
\tag{23}
$$
$$
t = \frac{1}{2K^{3/2}}(\theta-\sin\theta).
\tag{24}
$$
小 `\theta` 展开后，
线性 overdensity 为
$$
\delta
= \frac{a^3}{\tilde a^3} - 1
\simeq -3\frac{\delta a}{a}
\simeq \frac{3}{20}\theta^2
\simeq \frac{3}{5}Ka.
\tag{26}
$$
当内区塌缩时得到标准阈值
$$
\delta_* = \frac{3}{5}\left(\frac{3\pi}{2}\right)^{2/3}\approx 1.68.
\tag{27}
$$
到这里都还是你熟悉的 Newtonian spherical collapse 结果。

---

### 4.2 关键问题：这个 `\delta` 到底是哪一个 gauge 的 `\delta`？

作者把内区闭 FRW 度规写成
$$
ds^2 = dt^2 - \tilde a^2\left[\frac{dr^2}{1-Kr^2}+r^2 d\Omega^2\right],
\tag{28}
$$
在 `Kr^2\ll 1` 时改写成近似 Cartesian 形式：
$$
ds^2 = dt^2 - \tilde a^2\Big[\delta_{ij}+Kr^2(\partial_i r)(\partial_j r)\Big]dx^i dx^j.
\tag{29}
$$
再和一般扰动度规 (13) 对比，得到
$$
A=0,\tag{30}
$$
$$
B=0,\tag{31}
$$
$$
\psi = \frac{Kr^2}{4} - \frac{\delta a}{a},\tag{32}
$$
$$
E = \frac{Kr^4}{16}.\tag{33}
$$
此时立刻有两个重要结论：

1. `B=0`
2. 没有 peculiar velocity，因此 `\delta q = 0`

也就是说，这个 spherical collapse 设置天然就在
**comoving-orthogonal gauge** 里。

---

### 4.3 为什么它还同时是 synchronous？

作者又用动量守恒写道：
$$
A_c = -\frac{\delta P_c}{\rho + P}.
\tag{34}
$$
对于 pressureless matter，`\delta P_c = 0`，于是
$$
A_c = 0.
$$
所以这个 gauge 不只是 comoving-orthogonal，
同时还是 synchronous。

于是作者得出决定性结论：

> spherical collapse 问题真正使用的线性密度扰动，
> 就是 **comoving-orthogonal gauge** 下的密度扰动 `\delta_c`。

换句话说：

- collapse threshold `\delta_* \approx 1.68`
- halo formation
- peak-background split
- bias

这些东西如果要和 GR 接轨，
应该接到 `\delta_c` 上，而不是 `\delta_\ell` 上。

---

### 4.4 于是大结论是什么？

因为：

1. spherical collapse 选中的密度是 `\delta_c`
2. `\delta_c` 和 `\phi_N` 满足的是式 (16) 那个 Newtonian 形式的 Poisson 方程

所以作者认为：

> **GR 不会在超视界尺度改变 PNG scale-dependent bias 的 `1/k^2` 形式。**

这就是整篇文章的主结论。


## Cell 6 (markdown)

## 5. 第三节 A：如果你硬要在 longitudinal gauge 里看，会发生什么？

作者并不是说 longitudinal gauge 不合法，
而是说：

> 你当然可以在 longitudinal gauge 里重写同一个物理过程，
> 但那时“塌缩判据”会变成非局域的。

他们做了一个坐标变换，把同一个 spherical collapse 配置改写到 longitudinal gauge：
$$
ds^2 = a_\ell^2\Big[(1-2\phi_N)d\eta_\ell^2 - (1+2\phi_N)(dr_\ell^2+r_\ell^2 d\Omega^2)\Big],
\tag{36}
$$
并得到
$$
\phi_N = \frac{3}{20}K(r_0^2-r_\ell^2).
\tag{37}
$$
这时 longitudinal gauge 的密度对比度变成
$$
\delta_\ell = \frac{3}{5}K a_\ell
\left[1+\frac{1}{2}\frac{(r_0^2-r_\ell^2)}{r_H^2}\right].
\tag{38}
$$
中心处的塌缩阈值则变成
$$
\delta_{\ell *}\big|_{r=0}
= \frac{3}{5}\left(\frac{3\pi}{2}\right)^{2/3}
\left[1+\frac{1}{2}\frac{r_0^2}{r_H^2}\right].
\tag{39}
$$
这说明了什么？

- 在 comoving gauge 里，collapse threshold 是局域的，只是一个常数 `1.68`
- 在 longitudinal gauge 里，collapse threshold 依赖 overdensity 的尺度 `r_0`

也就是说，longitudinal gauge 中虽然密度在大尺度上看起来和势成局域关系，
但**真正的 collapse criterion 自己变成了非局域量**。

作者进一步写出一般关系：
$$
\delta_\ell = \left(1 - 3\mathcal H^2 \partial^{-2}\right)\delta.
\tag{40}
$$
这正是“longitudinal gauge 的大尺度局域感”其实被
`\partial^{-2}` 的非局域性偷偷搬回来了。

所以作者想表达的是：

> 你不能通过“换到另一个 gauge，看见局域关系了”
> 就得出“scale-dependent bias 消失了”的结论。


## Cell 7 (markdown)

## 6. 第四节：为什么相关函数会 formal divergence？

如果 `\Delta b \propto 1/k^2` 在 GR 中也不改，
那么 biased tracer 的功率谱在小 `k` 下会怎样？

相关函数定义为
$$
\xi(r)=\frac{1}{(2\pi)^3}\int d^3k\, P(k)e^{i\mathbf{k}\cdot \mathbf{r}}
\tag{41}
$$
在各向同性下写成
$$
\xi(r)=\frac{1}{2\pi^2}\int P(k)\frac{\sin kr}{kr}k^2 dk.
\tag{42}
$$
如果暗物质本身在小 `k` 时
$$
P_{\rm dm}(k)\propto k^{n_s},
$$
而 biased tracer 满足
$$
P(k)\sim (b+\Delta b)^2 P_{\rm dm}(k),
\qquad
\Delta b \propto k^{-2},
$$
那么小 `k` 下有
$$
P(k)\propto k^{n_s-4}.
$$
于是 integrand 变成
$$
P(k)k^2 \propto k^{n_s-2}.
$$
因为 `n_s \approx 1`，这个积分在 `k\to 0` 时发散。

所以作者明确承认：

> **formal correlation function 对 every `r` 都发散。**

这一步非常重要，因为它说明作者并没有回避这个问题，
而是正面承认它。


## Cell 8 (markdown)

## 7. 发散为什么不是观测灾难？作者如何 regularize？

作者的回答非常朴素，也非常关键：

### 7.1 真实 survey 不知道“宇宙平均密度”

在真实巡天里，我们不是拿“理论上绝对平均密度”去定义扰动，
而是从 survey 自己的数据估计平均密度。

所以观测到的扰动其实是
$$
\tilde\delta(\mathbf{x}) = \delta(\mathbf{x}) - \bar\delta,
\tag{43}
$$
其中
$$
\bar\delta = \int d^3x\, W(\mathbf{x})\, \delta(\mathbf{x}),
\tag{44}
$$
`W(\mathbf{x})` 是归一化的 survey window。

---

### 7.2 测到的是减去 survey mean 之后的相关函数

于是测量相关函数是
$$
\tilde\xi(r)
= \int d^3x\, W(\mathbf{x})\, \tilde\delta(\mathbf{x})\tilde\delta(\mathbf{x}+\mathbf{r}),
\tag{45}
$$
取 ensemble average 后得到
$$
\langle \tilde\xi(r)\rangle
= \xi(r) - \langle \bar\delta^2\rangle.
\tag{46}
$$
而
$$
\langle \bar\delta^2\rangle
= \frac{1}{(2\pi)^3}\int d^3k\, |W_k(k)|^2 P(k)
\equiv \sigma_W^2.
\tag{47}
$$
所以最终
$$
\langle \tilde\xi(r)\rangle = \xi(r) - \sigma_W^2.
\tag{48}
$$
这一步的物理意思非常直接：

> **比 survey 还大的模式，只会抬高整个 survey 的整体背景，
> 但不会被你当成“可测的两点相关”。**

---

### 7.3 top-hat 例子

对半径 `R` 的 top-hat window，
作者把它写成
$$
\sigma^2(R)
= \frac{1}{2\pi^2}\int d^3k\, P(k) k^2 T^2(kR),
\tag{49}
$$
进而可写出
$$
\tilde\xi(r)
= \frac{1}{2\pi^2}\int P(k)
\left[\frac{\sin kr}{kr} - T^2(kR)\right]k^2 dk.
\tag{50}
$$
关键在于小 `k` 展开时，
括号里的量是 `O(k^2)`，
于是整体 integrand 变成
$$
\propto k^{n_s},
$$
从而收敛。

这就是论文对“formal divergence 怎么和真实观测相容”的答案。


## Cell 9 (markdown)

## 8. 这篇文章真正得到的结论，按强弱分层

### 8.1 最核心结论

**结论 A**

在正确变量上，也就是 **comoving-orthogonal density perturbation** 上，
GR 不会在超视界尺度改写 PNG 诱导的 `scale-dependent bias` 的 `1/k^2` 形式。

---

### 8.2 方法论结论

**结论 B**

不能因为 longitudinal gauge 里出现
$$
3\mathcal H^2 \phi_N - \nabla^2 \phi_N = 4\pi G a^2 \delta\rho_\ell
$$
就断言“GR 修正会把 `1/k^2` 截断掉”。

原因是：

- 那个 gauge 下 density 的定义变了；
- 更糟的是，collapse criterion 自己也变得非局域；
- 所以你不能直接拿它来代替 spherical collapse / halo bias 使用的密度变量。

---

### 8.3 观测结论

**结论 C**

theoretical biased-tracer correlation function 的 formal divergence
并不意味着可观测量发散；
真实 survey 的 mean subtraction / finite window 会把它 regularize。

---

### 8.4 对 GR 修正的态度

**结论 D**

这篇文章不是说“GR 在大尺度完全不重要”，
而是说：

> **至少对 local PNG 的 scale-dependent bias 这个问题，
> 单靠 GR 不会自动把 `1/k^2` 改成有限常数。**


## Cell 10 (markdown)

## 9. 这篇文章和我们当前 `localPNGmodel` 项目的关系

这篇文章和我们当前的项目有直接冲突点，也有直接启发。

### 9.1 它直接挑战哪种思路？

它直接挑战下面这种思路：

> “因为 Newtonian gauge 中超视界 `\Phi/\delta` 不再是 `1/k^2`，
> 所以只要把 Jeong 的 `M(k)` 从 `k^2` 改成 `k^2 + O(\mathcal H^2)`，
> 就能物理地修复 PNG galaxy power spectrum 的发散。”

Wands & Slosar 会说：

- 你不能把 **用来定义 observed/longitudinal density 的关系**
  直接拿去替换
- **决定 halo collapse 的 comoving density kernel**

因为 spherical collapse 选中的变量是 `\delta_c`，
而 `\delta_c` 与 `\phi_N` 的关系仍然保持 Newtonian 形式。

---

### 9.2 它又给了我们什么启发？

它给了非常重要的一条替代路线：

> 如果你的真正麻烦是 `2PCF / Hankel transform` 发散，
> 那么你应该优先在 **finite-volume observable**
> 层面做 regularization，
> 而不是优先把 PNG bias kernel 改写成一个 GR 截断形式。

更具体地说，它支持我们去考虑：

1. survey window
2. mean-density subtraction
3. integral constraint
4. observed galaxy number counts 而不是 bare tracer field

---

### 9.3 它和更现代 GR 文献怎么接起来？

这篇 2009 文章主要做的是：

- 证明 `1/k^2` 不会因为 GR 自动消失；
- 指出 formal divergence 需要在 survey observable 层面理解。

而更现代的文献，例如

- Grimm et al. 2020
- Castorina & Di Dio 2023

继续往前走，研究的是：

- 完整 observed galaxy power spectrum estimator
- 全部 GR 投影项
- IR divergence cancellation
- equivalence principle / Weinberg adiabatic mode

所以你可以把 Wands & Slosar 看成这个问题的早期基石：

> **它告诉你：不要太早把 `1/k^2` 改掉。**
>  
> **真正该去修的，很可能是 observable。**


## Cell 11 (markdown)

## 10. 我对这篇文章的工作流总结

如果把论文逻辑压缩成最短的“推导链条”，就是：

1. 从 GR 线性约束方程出发，推出
$$
\nabla^2 \phi_N \leftrightarrow \delta\rho_c
$$
这个 Newtonian 形式在 `comoving-orthogonal gauge` 中仍然成立。

2. 用 spherical collapse 证明 halo collapse 相关的线性密度变量，
   正是 `\delta_c`。

3. 因而 PNG scale-dependent bias 的 `1/k^2` 形式在 GR 中不被改写。

4. 其后果是 formal correlation function 发散。

5. 但真实观测通过 finite survey window 和 mean subtraction
   测到的是 regularized correlation function。

---

## 11. 读完这篇以后最该记住的三句话

### 句子 1

**这篇文章不是在说“大尺度 GR 修正不存在”，而是在说“不能把它们粗暴解释成会截断 PNG bias 的局域修正”。**

### 句子 2

**决定 halo bias 的密度变量是 comoving-orthogonal density，而不是任意 gauge 下你看到的局域密度。**

### 句子 3

**formal divergence 不等于 observable divergence；真正观测到的是 finite-window、mean-subtracted 的量。**

---

## 12. 如果后面继续深挖，建议紧接着读什么？

建议顺序：

1. **Jeong, Schmidt & Hirata (2012)**  
   看 bias 应该如何在 relativistic galaxy clustering 里定义。

2. **Yoo (2010)**  
   看 observed galaxy power spectrum 为什么会被误读成 PNG。

3. **Camera, Santos & Maartens (2015)**  
   看 PNG 和 fully relativistic galaxy number counts 如何联合 forecast。

4. **Castorina & Di Dio (2023)**  
   看完整 observed power spectrum estimator 的 IR-safe 处理。

---

## 13. 给当前项目的操作性提醒

如果后面你问我“这篇文章对我们现在的 `squeeze-limit` 修改意味着什么”，
我会优先围绕下面两件事来回答：

1. 你的 `M(k)` 被替换的那个对象，到底是不是 spherical collapse / bias 理论里应当替换的对象？
2. 你想修复的到底是“bare bias term 的理论发散”，还是“真实可测 `2PCF` 的 Hankel 发散”？

这两件事，在 Wands & Slosar 看来，不是同一个问题。

