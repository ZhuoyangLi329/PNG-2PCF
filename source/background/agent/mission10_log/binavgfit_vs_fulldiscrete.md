# Mission 10 笔记：BinAvgFit+FullDiscrete 相对原始 FullDiscrete 的差别（详细版）

本文件回答你问的“我目前的方法相对于全离散（FullDiscrete）的差别是什么”。  
结论先讲清楚：

> **两者在 `pk→2pcf` 的离散求和公式完全相同；唯一差别在 `P_model` 的 best-fit 是怎么从 binned 的 `P_data` 拟合出来的。**  
> 新方法把“bin-average vs 点值（bin center）不自洽”从拟合阶段解决掉，从而在不使用 `P_measure` 参与 `pk→2pcf` 的前提下，让 FullDiscrete 的 2PCF 结果逼近 DataBin。

相关脚本与图：

- 脚本：[mission10_binavgfit_full_discrete_compare.py](/pscratch/sd/l/lzy/1gpc-UNIT-fnl100-fastpm/agent/mission10_log/mission10_binavgfit_full_discrete_compare.py)
- 对比图：`mission10_binavgfit_fd_vs_databin_fulldiscrete.png`

## 1. 原始 FullDiscrete 的 pipeline（Mission 8/9 的“全离散”）

原始 FullDiscrete 的完整 pipeline（以 monopole 为例）可以拆成两步：

### 1.1 第一步：拟合 P(k) 得到 best-fit 的连续模型 P_model(k)

在 Mission 9 的实现里，拟合用的是 `desilike` 的标准 `PowerSpectrumMultipolesObservable`，其默认做法等价于：

1. 数据输入是 **binned 的** `P_data_bin(i)`（即 pk 文件里每个 k-bin 的均值）。
2. 理论在 **bin center** 上取值：`P_model(k_center,i)`。
3. 用 `P_model(k_center,i)` 去对齐 `P_data_bin(i)` 并做 χ² 最小化，得到 `θ_best`。

写成公式就是：
$$
P_{\rm data,bin}(i)\ \xleftarrow[\text{fit}]{\text{compare}}\ P_{\rm model}(k_{{\rm center},i};\theta).
$$

### 1.2 第二步：用离散壳层求和从 P_model 得到 ξ0(r)

有限盒长度 `L`，基模 `k_f = 2\pi/L`。离散壳层：
$$
k_q = k_f\sqrt{q},\qquad q=n_x^2+n_y^2+n_z^2,
$$
简并度 `g_q = #{(n_x,n_y,n_z)\in\mathbb{Z}^3: n_x^2+n_y^2+n_z^2=q}`。

离散求和（FullDiscrete）：
$$
\xi_0^{\rm FD}(r)=\frac{1}{V}\sum_{q,\,k_q\le k_{\max}} g_q\,P_{\rm model}(k_q;\theta_{\rm best})\,j_0(k_q r),
\qquad V=L^3,\ j_0(x)=\frac{\sin x}{x}.
$$

这个第二步是“干净的 forward model”：只要有 `θ_best` 和 `L`，就能算 `ξ(r)`。

## 2. 我的新方法：BinAvgFit + FullDiscrete（差别在哪）

新方法同样分两步，但 **只改第一步（拟合阶段）**。第二步（离散求和）完全不变。

### 2.1 新方法改动点：拟合时用 bin-average 的理论，而不是 bin center 的点值

关键事实：测量的 `P_data_bin(i)` 并不是某个点 `k_center` 的真值，而更接近 bin 内离散模式的加权平均：
$$
P_{\rm data,bin}(i)\approx
\frac{\sum_{q\in{\rm bin}\ i} g_q\,P_{\rm true}(k_q)}{\sum_{q\in{\rm bin}\ i} g_q}.
$$

因此如果理论也做成同样定义的 bin-average，就得到：
$$
\bar P_{\rm model,bin}(i;\theta)=
\frac{\sum_{q\in{\rm bin}\ i} g_q\,P_{\rm model}(k_q;\theta)}{\sum_{q\in{\rm bin}\ i} g_q}.
$$

**BinAvgFit** 的拟合目标变成：
$$
P_{\rm data,bin}(i)\ \xleftarrow[\text{fit}]{\text{compare}}\ \bar P_{\rm model,bin}(i;\theta),
$$
也就是用离散模式计数 `g_q` 把理论先做成“跟数据同口径”的 bin-average，再做 χ²。

这一步解决的是：标准拟合在做
$$
P_{\rm model}(k_{{\rm center},i}) \approx P_{\rm data,bin}(i)
$$
但随后 FullDiscrete 在求和时用的却是 `P_model(k_q)`；当 `P(k)` 在 bin 内弯曲（PNG 的 `1/k^2` 在最低几个 bin 特别夸张）时：
$$
P_{\rm model}(k_{{\rm center},i}) \neq \bar P_{\rm model,bin}(i),
$$
于是“拟合出来的 `θ_best`”与“离散求和真正需要的低 k 值”天然不一致，导致 FullDiscrete 产生稳定偏差（Mission 9 看到的 `~ -0.09σ` 级别）。

### 2.2 新方法第二步：仍然用标准 FullDiscrete（不喂任何 P_measure）

拟合得到 `θ_best^{\rm BinAvgFit}` 后，2PCF 仍按标准 FullDiscrete：
$$
\xi_0^{\rm new}(r)=\frac{1}{V}\sum_{q,\,k_q\le k_{\max}} g_q\,P_{\rm model}(k_q;\theta_{\rm best}^{\rm BinAvgFit})\,j_0(k_q r).
$$

这里 **没有** 像 DataBin 那样把 `P_data_bin(i)` 再塞回 `P_eff(k_q)`，因此它仍是严格的 forward model。

## 3. 两个方法“数学上看起来很像但结果差很大”的根因对照

为了把“差别到底在哪里”说得最清楚，可以把所有差异都压缩成一句话：

> 原始 FullDiscrete 用的是 `θ_best`，这个 `θ_best` 来自匹配 `P_model(k_center)`；  
> 新方法用的是 `θ_best`，但它来自匹配 `bin-average` 的 `\bar P_model,bin`。  
> **离散求和公式本身一字不改。**

因此差别不是：

- 不是 `g_q` 的算法（两者都用同样的壳层计数）；  
- 不是 `kmax` 选取（脚本统一用 `kmax=15`）；  
- 不是 `pk→2pcf` 的核（`j0` 一样）；  
- 也不是“额外的 IR 窗口”。

差别只在：

- 原始 FullDiscrete 的拟合理论是点值 `P_model(k_center)`；
- 新方法的拟合理论是离散模式加权的 `\bar P_model,bin`。

## 4. 实现层面的具体差别（代码怎么做的）

### 4.1 原始 FullDiscrete（std fit）怎么做

脚本中对应 `fit_standard_desilike()`：

- 用 `desilike` 的 `TracerPowerSpectrumMultipolesObservable`
- 理论 `PNGTracerPowerSpectrumMultipoles` 在 `modes=k_center` 上评估
- `MinuitProfiler` 得到 best-fit。

### 4.2 BinAvgFit 怎么做（为什么没直接用 desilike）

`desilike` 的 observable 在这个项目里是围绕“bin center”这一默认口径构建的。  
为了最小改动实现 BinAvgFit，我直接写了一个 χ²：

1. 先按盒长 `L` 得到 `k_f`，只枚举到拟合上限 `kmax_fit=kmax_bin_last`：
   $$
   q_{\max}^{\rm fit}=\left\lfloor (k_{\max}^{\rm fit}/k_f)^2\right\rfloor.
   $$
2. 用小 `qmax_fit` 的 **直接枚举**计算 `g_q`（因为只到拟合区间，q 很小，枚举比 FFT 更省事）。
3. 把每个 bin 对应的壳层索引找出来：`k_q ∈ [kmin_i,kmax_i)`。
4. 对每次参数 `(fnl_loc,b1,sigmas)`，在这些 `k_q` 上评估 `P_model(k_q)`，再做 bin-average：
   $$
   \bar P_{\rm model,bin}(i)=\frac{\sum g_q P(k_q)}{\sum g_q}.
   $$
5. 用 mock pk 估计协方差 `C`，做
   $$
   \chi^2=(P_{\rm data,bin}-\bar P_{\rm model,bin})^T C^{-1}(P_{\rm data,bin}-\bar P_{\rm model,bin})
   $$
   并用 `iminuit.Minuit` 最小化得到 `θ_best^{\rm BinAvgFit}`。

### 4.3 两者的 FullDiscrete 求和阶段完全一致

两者都用 FFT 卷积法计算大 q 的 `g_q`，然后做
$$
\xi_0(r) = \frac{1}{V}\sum_q g_q P_{\rm model}(k_q) j_0(k_q r).
$$

## 5. 数值上 best-fit 参数发生了什么变化（体现差别来自拟合阶段）

这部分是非常直接的证据：只要拟合阶段不同，best-fit 就会移动；而离散求和阶段未变。

### 5.1 3Gpc fnl100（FastPM）

- 标准拟合：`fnl=77.069, b1=2.789629, sigmas=2.187603`
- BinAvgFit：`fnl=71.991, b1=2.805644, sigmas=3.563991`

### 5.2 1Gpc fnl100（FastPM）

- 标准拟合：`fnl=108.574, b1=2.754300, sigmas=2.243869`
- BinAvgFit：`fnl=101.153, b1=2.773076, sigmas=3.843243`

这种移动的方向是合理的：PNG 的 `1/k^2` 让“bin-average 的有效 k”与 `k_center` 差别更敏感，因此会直接反映到 `fnl` 和相关平滑参数 `sigmas` 上。

## 6. 2PCF 效果：新方法相对 FullDiscrete 的改进幅度

对比指标与 Mission 9 一致（对 `r^2\xi_0` 的 `mean(|Δ/σ|)` 和 `mean(Δ/σ)`）：

### 6.1 3Gpc fnl100

- FullDiscrete(std fit): `0.6430`, `-0.0849`
- **BinAvgFit+FullDiscrete (new)**: `0.4595`, `+0.1622`
- DataBin（仅作参照）: `0.4414`, `+0.1805`

这里新方法已经非常接近 DataBin，说明在 3Gpc 场景下，“FullDiscrete vs DataBin 的主要差别”确实就是拟合阶段的不自洽。

### 6.2 1Gpc fnl100

- FullDiscrete(std fit): `0.2770`, `-0.0862`
- **BinAvgFit+FullDiscrete (new)**: `0.2265`, `+0.0127`
- DataBin（仅作参照）: `0.2078`, `+0.0471`

1Gpc 下新方法也显著优于原 FullDiscrete，但仍略逊于 DataBin。一个合理的解释是：在 1Gpc 中最低几个 bin 的离散 shell 更稀疏（某些 bin 模式数少，甚至出现空 bin），PNG 的 IR 权重更集中在极少数 mode 上，这会让 “bin-average 拟合” 仍然没法完全复现 DataBin 那种“直接用测量 bin 值锚定低 k”的效果。

## 7. 和 DataBin 的关系（强调：新方法不依赖 P_measure 做 pk→2pcf）

DataBin 的关键点是：在 `pk→2pcf` 的求和里，低 k 直接替换为 `P_data_bin(i)`：
$$
P_{\rm eff}(k_q)=P_{\rm data,bin}(i)\quad (k_q\in{\rm fit\ bins}).
$$

这使得 DataBin 不是纯 forward model（没有 `P_measure` 就没法做）。

而 BinAvgFit+FullDiscrete 的 `pk→2pcf` 阶段始终只用 `P_model(k_q)`，因此：

- 你如果只有 `P_model`（比如理论预测、或者 mock 的 best-fit 参数），依然可以跑；
- 并且它在 FastPM 的实际数据上已经“数值上逼近”了 DataBin（尤其是 3Gpc）。

## 8. 下一步建议（如果你希望继续压缩 1Gpc 与 DataBin 的差距）

（这一段不是必须做，但我把可行方向写在这里，方便你决定要不要继续。）

- 继续保持 forward model 的前提下，可以在拟合阶段把理论 `\bar P_model,bin` 做得更接近测量定义：
  - 在 bin-average 时纳入测量时的实际权重（比如网格窗口、MAS、具体 mode 计数定义等）。
  - 或者把 “bin 内 mode 的分布” 做得更细（例如对每个 shell 再按角向/μ 做加权，若测量定义里隐含了这种权重）。
- 如果允许 “联合似然 P+ξ” 且能处理 `Cov(P,ξ)`，则 DataBin 可被解释为一种数据压缩；但这会显著增加统计建模复杂度，不建议作为当前主线。
