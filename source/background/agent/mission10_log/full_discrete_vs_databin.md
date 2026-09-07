# Mission 10 笔记：FullDiscrete 与 DataBin 的区别，以及“不用测量 P(k)”时怎么办

本笔记回答任务书 `##10 全离散/databin方法的区别` 的两问，并把 Mission 8/9 的结论串起来，明确两种方法“数学形式很近但结果差异大”的根源。

## 0. 背景回顾（Mission 8/9 已完成部分在干什么）

- **Baseline（连续积分）**：把有限盒的低 `k` 当成连续谱密度 `k^2/(2π^2)`，从 `k_min=k_f=2π/L` 积分到 `k_max` 得到 `ξ0(r)`。  
  现实是：PNG `P(k)~1/k^2` 在 IR 会放大低 `k` 权重，连续近似在大尺度会出现系统性问题（偏低/不收敛等）。

- **ExpWindow（经验 IR 窗口）**：在 `k<k_f` 对功率谱乘一个经验窗 `W(k)` 把 IR 面积“压回去”，在数值上能很好对齐 2PCF，但引入经验参数（如 `x(L)`）。

- **FullDiscrete（全离散）**：尊重有限盒的离散傅里叶模式结构，用离散壳层求和替代连续积分：

  - 基模 `k_f=2π/L`，离散壳层模长
    \[
    k_q = k_f\sqrt{q},\qquad q=n_x^2+n_y^2+n_z^2
    \]
  - 壳层简并度 `g_q` 是满足 `n_x^2+n_y^2+n_z^2=q` 的整数三元组个数（Mission 8/9 用 FFT 卷积精确算）。
  - 单极相关函数
    \[
    \xi_0(r)=\frac{1}{V}\sum_{q} g_q\,P_{\rm model}(k_q)\,j_0(k_q r),\qquad V=L^3,\;j_0(x)=\frac{\sin x}{x}.
    \]

Mission 9 的关键发现是：FullDiscrete 已经比 Baseline 更“盒子自洽”，但仍有一个稳定的小偏差（典型量级 `~0.09σ`），而 **DataBin** 能进一步显著改善。

## 1. 两个方法的“真正区别”是什么？

### 1.1 FullDiscrete

**输入**：仅依赖理论功率谱 `P_model(k)`（由 `desilike` 拟合 `P0(k)` 得到的 best-fit 连续模型）。  
**做法**：在每个离散壳层直接取 `P_model(k_q)` 进入求和：
\[
\xi_0^{\rm FD}(r)=\frac{1}{V}\sum_{q} g_q\,P_{\rm model}(k_q)\,j_0(k_q r).
\]

它是一个干净的 forward model：给定参数（`fnl,b1,...`）就能算出 `ξ0(r)`，**不需要把任何观测量再塞回模型里**。

### 1.2 DataBin（Mission 9）

**输入**：`P_model(k)` 之外，还显式使用“测量得到的 binned 功率谱均值” `\bar P_{\rm data,bin}(i)`（通常就是拟合时用的前 `N_fit≈20` 个 k-bin 的均值）。  

**做法**：在低 `k` 的拟合 bin 范围内，用测量的 bin 平均值替代理论值；高 `k` 仍用 `P_model` 外推：
\[
\xi_0^{\rm DB}(r)=\frac{1}{V}\sum_{q} g_q\,P_{\rm eff}(k_q)\,j_0(k_q r),
\]
\[
P_{\rm eff}(k_q)=
\begin{cases}
\bar P_{\rm data,bin}(i), & k_q\in [k_{\min,i},k_{\max,i})\ \ (i\le N_{\rm fit})\\
P_{\rm model}(k_q), & {\rm otherwise}.
\end{cases}
\]

因此，DataBin 可以被概括为：

> **DataBin = FullDiscrete + 低 k 用数据校正（bin-average 替换）**

两者的数学求和骨架完全一样，差异只在于 `P_eff`：  
FullDiscrete 用 `P_model(k_q)`；DataBin 在低 `k` 用 `P_data_bin`（常数阶梯）。

## 2. 为什么形式相近，但数值结果差异很大？

核心不是 `g_q` 或 `j0`，而是 **“功率谱拟合/测量的 bin 平均” 与 “求和时逐壳层取值” 之间的不自洽**，在 PNG 的 `1/k^2` 权重下被放大。

### 2.1 测量的 P(k) 不是点值，而是 bin-average

对有限盒离散模式，测到的第 `i` 个 bin 的功率谱均值本质上是离散模式的加权平均：
\[
\bar P_{\rm data,bin}(i)=
\frac{\sum_{q\in {\rm bin}\ i} g_q\,P_{\rm true}(k_q)}{\sum_{q\in {\rm bin}\ i} g_q}.
\]

而我们在拟合时（Mission 9 的诊断假设）通常用的是：
\[
P_{\rm model}(k_{{\rm center},i}),
\]
即在 bin center 评估模型来对齐 `\bar P_{\rm data,bin}(i)`。

当 `P(k)` 在 bin 内很“弯”（PNG 的 `1/k^2` 在最低几个 bin 尤其明显）时：
\[
P_{\rm model}(k_{{\rm center},i})\neq
\frac{\sum_{q\in i} g_q\,P_{\rm model}(k_q)}{\sum_{q\in i} g_q}\equiv \bar P_{\rm model,bin}(i).
\]

这会带来一条关键链式后果：

1. 拟合得到的 best-fit 参数，使得 `P_model(k_center,i)` 更接近 `\bar P_{\rm data,bin}(i)`；
2. 但 FullDiscrete 在求和时用的是 `P_model(k_q)`，它在低 `k` 的离散 `k_q` 上与 `\bar P_{\rm data,bin}(i)` 存在系统偏差；
3. 这个偏差经过 `g_q`（模式数）和 `j0(k_q r)`（大尺度对低 k 更敏感）的加权后，变成 `ξ0(r)` 的稳定偏差（Mission 9 看到的 `~0.09σ`）。

DataBin 之所以好，是因为它跳过了这条不自洽：它在低 k 直接把 `P_eff` 设成“测量定义下正确的 bin-average”。

### 2.2 为什么 PNG 情况更严重

对于 `P(k)~1/k^2`，bin 内 mode-weighted 的 `<1/k^2>` 与 “在 bin center 的 `1/k_center^2`” 会有几个百分点到十几个百分点的差别（Mission 9 的 `mission9_lattice_analysis.md` 已给过数值表）。  
这类差别对 `ξ0(r)` 的大尺度非常敏感，因此 DataBin 和 FullDiscrete 的差距会显著。

## 3. 回答任务书问题 1：没有 P_measure 时 DataBin 还能用吗？

### 3.1 严格意义下：**不能直接用**

因为 DataBin 的定义就是在低 `k` 把 `P_eff` 替换成观测量 `\bar P_{\rm data,bin}`。如果你的目标是构造一个**纯 forward model**：

> 给定参数 → 输出 `ξ0_model(r)`（不能显式依赖观测的 `P(k)`）

那 DataBin 会让 `ξ0_model(r)` 变成“部分由数据决定”，这在单独用 `ξ` 做似然时是不自洽的（等价于把数据塞回理论）。

### 3.2 例外：只有在“联合使用 P(k) 数据”的框架里才可能成立

如果你的推断同时把 `P(k)` 也作为数据（联合似然 `L(P,ξ|θ)`），那么把低 k 的 `P_bin` 当成被数据强约束的量（或者显式引入 `P_bin` 作为 nuisance 再边缘化），DataBin 才能被解释为一种数据压缩/重构步骤。  
但这种做法必须处理 `P` 与 `ξ` 的协方差，否则容易 double-count 信息。

## 4. 回答任务书问题 2：不能用的话怎么办？FullDiscrete 还有哪些“与测量无关”的改进空间？

任务书允许我们用测量 `P(k)` 去拟合得到 best-fit `P_model`，但希望 `pk→2pcf` 的映射本身是“理论可计算”的，不需要再喂 `P_measure`。

这里给出两个直接可落地、且不依赖 `P_measure` 的改进方向（只依赖：盒长 `L`、你对 `P(k)` 的 binning 定义、以及理论模型 `P_model`）。

### 4.1 推荐方案：在“功率谱拟合阶段”做 bin-average theory（把不自洽从源头消掉）

把理论预测从“bin center 点值”改为“按离散模式做 bin-average”：
\[
\bar P_{\rm model,bin}(i)=
\frac{\sum_{q\in i} g_q\,P_{\rm model}(k_q)}{\sum_{q\in i} g_q}.
\]

然后用 `\bar P_{\rm model,bin}(i)` 去拟合 `\bar P_{\rm data,bin}(i)`，而不是用 `P_model(k_center,i)`。

这样 best-fit 参数会自动吸收“bin-center vs bin-average”的差异；随后你再用 FullDiscrete
\[
\xi_0(r)=\frac{1}{V}\sum_q g_q P_{\rm model}(k_q) j_0(k_q r)
\]
去算 `ξ0`，就不会再出现那种稳定的 model-vs-data mismatch（至少应显著减弱）。

这一步是 **测量无关** 的：对任何给定参数集，`\bar P_model,bin(i)` 都可计算；它只是把理论映射改成与数据定义一致。

### 4.2 可选方案：把 DataBin “去数据化”，变成 ModelBin（只用 P_model 生成低 k 的阶梯近似）

定义一个只依赖模型的 bin-average：
\[
\bar P_{\rm model,bin}(i)=
\frac{\sum_{q\in i} g_q\,P_{\rm model}(k_q)}{\sum_{q\in i} g_q}.
\]

然后在离散求和里把低 `k` 的 `P_model(k_q)` 换成 `\bar P_model,bin(i)`：
\[
P_{\rm eff}(k_q)=
\begin{cases}
\bar P_{\rm model,bin}(i), & k_q\in{\rm low\ k\ bins}\\
P_{\rm model}(k_q), & {\rm otherwise}.
\end{cases}
\]

这就是 DataBin 的“数据替换版”，它是一个合法的 forward model（不含 `P_measure`），同时也继承了 DataBin 的直觉：低 k 处用 bin-average 表达更接近测量定义。

注意：
- 高 k 用阶梯函数会引入 `j0` 卷积振荡伪影（Mission 9 已验证），所以只应对拟合范围内有限的低 k bins 做这一近似。

### 4.3 与 Mission 11 的关系（下一步）

Mission 11 希望把离散求和等价地写成连续形式 `P_re,con(k)=w(k,L) P_eff(k)`。  
如果能把 `w(k,L)` 做成简单解析函数，那么 `w` 就可以被看作一个“与测量无关”的盒子重整化窗口，从而把 DataBin/FullDiscrete 的差异以解析方式吸收进理论管线。

## 5. 最简结论（给老板的两句回答）

1. **差别**：FullDiscrete 用 `P_model(k_q)`；DataBin 在低 k bins 用 `P_data_bin` 替换 `P_model(k_q)`，用数据直接修正了“bin-average vs bin-center”的不自洽，因此效果更好。
2. **能否在没有 P_measure 时使用 DataBin**：如果要求 `θ→ξ_model` 的纯 forward model，则不能；替代方案是把理论也按离散模式做 bin-average（用于拟合与/或用于求和），从源头消掉这类 mismatch。

## 6. 数值验证：BinAvgFit+FullDiscrete 能否逼近 DataBin？

我已经按 4.1 节“推荐方案”做了一个直接的数值验证（FastPM 的 3Gpc/1Gpc fnl100），并把 **2PCF 对比图**画出来了：

- 脚本：[mission10_binavgfit_full_discrete_compare.py](/pscratch/sd/l/lzy/1gpc-UNIT-fnl100-fastpm/agent/mission10_log/mission10_binavgfit_full_discrete_compare.py)
- 输出图：`mission10_binavgfit_fd_vs_databin_fulldiscrete.png`

指标（与 Mission 9 一致，统计 `r^2\xi_0` 的 `mean(|Δ/σ|)` 和 `mean(Δ/σ)`）：

### 3Gpc fnl100

- FullDiscrete(std fit): `0.6430`, `-0.0849`
- DataBin: `0.4414`, `+0.1805`
- **BinAvgFit+FullDiscrete (new)**: `0.4595`, `+0.1622`

这里新方法已经非常接近 DataBin，说明 FullDiscrete 与 DataBin 的主要差别确实来自“拟合阶段的 bin-average 不自洽”。

### 1Gpc fnl100

- FullDiscrete(std fit): `0.2770`, `-0.0862`
- DataBin: `0.2078`, `+0.0471`
- **BinAvgFit+FullDiscrete (new)**: `0.2265`, `+0.0127`

1Gpc 下新方法也明显优于原 FullDiscrete，但仍略逊于 DataBin，暗示 1Gpc 情况除了 bin-average 不自洽外，可能还叠加了更强的 IR 敏感性或更“尖锐”的离散结构效应（这部分可作为后续 Mission 11 的解析化切入点）。
