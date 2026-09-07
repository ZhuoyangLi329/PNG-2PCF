# Rawbox 的 P(k)–2PCF 不一致：实现审计、物理解释与最小判别实验

## 0. 结论与验证边界

本报告按 `给GPT5.6pro.md` 审阅，仅讨论 rawbox。审阅起点为 `273753d05e5fe143f61b8edd646a0e8593ba36ea`；不修改原始代码、测量、缓存、拟合或正式图。

**结论不是“只修一个 bug 就会一致”，也不是“P 和 ξ 本来就应该不一致”。目前有三个层次：**

1. **联合推断已经存在可以判定的实现错误。** 对混合量纲 covariance 做全局特征值 floor，改变了原来的 ξ 边块，即使 cross=0 也使 ξ 权重严重下降。因此旧 joint 的信息增益和“联合结果靠近 P”的物理解释不成立。这是仓库已有审计的发现，本次核查并给出不依赖量纲的替代实现，不把它冒充新发现。
2. **独立拟合确实仍有未解决的形状/协方差问题。** 混合矩阵 bug 不能解释单独 P、ξ 的全部差异。实空间的失败说明 RSD 坐标映射不是唯一原因；P02 的零边界与 ξ02 的非零阻尼又说明问题不等同于 P0 的宽后验。
3. **最值得检验的物理假设：同一个简化 sigma_s 正在吸收不同的缺失形状。** ξ 对 BAO 展宽和宽波数范围敏感，P02 对角向非线性更敏感；把它们都压进线性 Kaiser × 单一 Lorentzian，不一定得到同一个有效参数。此假设有证据支持，但尚不能宣称已经归因定量完成。

**本次实际运行：**独立的 NumPy/SciPy 实现及 14 项测试，包括 50,000 次独立复模式功率抽样、解析角矩、解析壳平均核、量纲不变性、低 k 格点几何、重分箱归属错误与审计脚本的合成缓存端到端测试。结果见 `local_test_results.json`。

**未运行：**原始 rawbox NPZ 的本地重拟合、原 catalog 的 FCFC/jaxpower 重测、长链 MCMC、仓库 PDF 图像逐图核验。当前读取能力取得了源码和审计 JSON，但未成功把原始二进制测量/图像转到计算环境。因此下文原始拟合值来自仓库 JSON；列出的图是供复核的对应产物，不表示我已经逐图看过。测试通过不等于新的物理模型已通过。

## 1. 对项目目标和方法的理解

项目用 local PNG 的尺度依赖 bias 约束 fNL，要求建立不依赖测量 P 回填的纯 forward 2PCF 模型。当前约定为

\[
A(k)=b_1+f_{\rm NL}\,2\delta_c(b_1-p)\alpha(k),\qquad\delta_c=1.686,
\]
\[
P^s(k,\mu)=P_{dd}^{\rm lin}(k)[A(k)+f\mu^2]^2
\left[1+\frac{(k\mu\sigma_s)^2}{2}\right]^{-2}.
\]

这里沿用代码的 alpha 约定，即低 k 的逆传递映射，而不是把 alpha 换成其倒数。P0 的残余 stochastic 按当前实现可另加 `sn0 * 1e4`；不能把无量纲 sn0 直接当作 P 的有量纲常数。

BinAvgFit 的必要性在于测量是按实际离散模式平均，不是 bin-center 点值。FullDiscrete 则处理有限周期盒的非零模和径向壳层简并度；球壳平均核还必须匹配 pair-count 的体积平均。两者解决的分别是测量定义和变换定义，**不负责修正错误的物理谱形状**。

本轮核心 Abacus 是 L=2000 Mpc/h、z=0.725、25 相位、mmin=1.4e13 的 halo rawbox。P 的 16 个拟合区间是有间隙的窄 bins，并非连续覆盖 0.003–0.095；ξ 的模板求和则到 k=3。实空间最简控制只有 fNL、b1，没有速度 FoG 参数。旧样本或其他分支的 sigma_s 不能直接拿来解释这个控制。

有限周期盒剔除零模与无限体积 PNG 相关函数的红外问题有关，但不是任意选择 kmin、经验 ExpWindow 或 survey IC 的理由。Wands–Slosar 讨论的是均值定义和红外行为；本次不将 survey 窗口修复迁移进 rawbox。[L1]

## 2. 先把不同“差异”量化分开

下表中区间来自原审计的 16/50/84 分位数，使用 **C_single**；均值形状检验使用 **C_mean=C_single/25**。后者不是把所有非高斯后验区间机械除以 5。表中 χ² 尚未按本报告重新优化或修复 covariance。

| 对照 | b1 后验中位数 | sigma_s 中位数和 68% 区间 | 均值 χ² / 自由度 | 原始证据 |
|---|---:|---:|---:|---|
| 实空间 P0，16 bins | 2.6501 [2.6354,2.6650] | 不拟合 | 45.234 / 14 | [R1] |
| 实空间 ξ0，s≥50 | 2.5416 [2.4899,2.5923] | 不拟合 | 331.279 / 28 | [R2],[R3] |
| 实空间 ξ0，s≥120 | 2.3682 [1.9433,2.8517] | 不拟合 | 12.025 / 21 | [R2],[R3] |
| RSD P0，free sn0 | 2.5327 | 6.489 [2.261,10.180] | 10.946 / 12 | [R4] |
| RSD ξ0，s≥50 | 2.5532 | 8.843 [7.855,9.884] | 162.980 / 27 | [R4] |
| RSD P0+P2 | 2.5760 | 0.984 [0.313,1.807] | 73.879 / 28 | [R5] |
| RSD ξ0+ξ2，smin=50/80 | 2.5425 | 7.791 [7.067,8.545] | 297.279 / 54 | [R5] |

重要解读：

- **P0 的 sigma_s 后验本来就很宽。** 把 P0 与 ξ0 的宽窄差异描述为 P02 与 ξ02 那样的大冲突，会混淆关键诊断。加入 P2 后才出现强烈的零阻尼偏好。
- P02 的 MAP sigma_s 为 `4.2414e-8`，实质在零边界。后验中位数约 1 不等于测出了非零速度弥散。
- 实空间 P0 自己的均值检验 PTE 也只有 `3.73696e-5`，不能把它当作已经精确验证的 b1 真值。ξ0 s≥50 的拒绝更严重，但不是唯一失败者。
- s≥120 的 ξ0 虽然 PTE≈0.939，MAP b1≈2.7435，**边际中位数却是 2.3682 且区间很宽**。这反映后验几何和信息损失，不能表述为“去掉小尺度后精确恢复 b1=2.74”。
- 两个统计量使用同批相位，中心差不能用独立误差平方和直接换算显著性；sigma_s 还有边界和先验体积效应。需要 paired 或条件残差检验。

## 3. 证据表：哪些已证实，哪些待判别

代码简称的完整路径见文末。图均位于 `source/project/plots/task43/rsd_validation/`，仅列对应产物。

| 状态/优先级 | 定位 | 结论与影响范围 | 对应图/复核产物 |
|---|---|---|---|
| 已证实 / P0 | [C1] joint 矩阵组装；[R6] `raw_floor` | 混合量纲 floor 改写 ξ 边块，旧 joint 失效；不解释独立 marginal 全部差异 | `task43_rsd_rawbox_joint_p02xi02_contours.pdf` |
| 已证实 / P0 | [C1] `pk_pole_cov`,`cross_block`；[C2] `precompute_rebin_cache` | 聚合中心重新归类到窄 P bins，分子 mode set 与分母不一致 | 本区 `radial_rebin_membership_geometry` |
| 已证实操作不同 / P1 | [C3] `FullDiscreteRSDModel.evaluate`；[C4] `ExactPeriodicPk0Model` | ξ 连续 mu 积分与 P 离散 mu 不是同一个 angular operator；参数影响尚未量化 | 本区 `first_bin`；原 P02/ξ02 图 |
| 已证实数值风险 / P1 | [C3] GL64；[C5] FastRSDModel | 高 k sigma 的连续角积分存在可测误差；共享近似的 surrogate 测试不能发现它 | 本区 `I0_quadrature_relative_errors` |
| 已证实 / P0 | [C6] 三个 realspace 脚本 `fit_with` | 更换 covariance 后没有在最终 metric 下重新优化，保存的 MAP 不是同一 likelihood 的 MAP | realspace 的 check/contours 图 |
| 已有数据拒绝 / P1 | [R7] `x25_scatter_comparison` | 模型均值以外还有 covariance 形状问题，不是只改 mean model 就能解决 | `task43_rsd_rawbox_x25_fulldiscrete_lorentzian.pdf` |
| 待独立验证 / P1 | [C1] quadrant scales | 四象限经验 rescaling 不能替代模式归一化；需完整 PSD 和留出校准 | canonical singular values 与 whitened scatter |
| 已证实统计非正则 / P1 | [R5] P02 `nominal.theta` | 零 sigma 的对称导数消失，局部 Fisher 不能代表边界后验 | sigma profile，而非仅 corner 图 |
| 强物理假设 / P2 | [R1]–[R5]；线性模板 | BAO 展宽、非线性 bias 与 density–velocity 项被 sigma_s/b1 吸收 | 分段残差、BAO 与 broadband 分解图 |
| 待检查，勿先判 bug | [C3] `shell_jell_kernel` | shell-average 已实现；j2 数值求积需独立收敛检查，不能误说还在用 bin center | 解析核与原核的 whitened 差 |
| 风险，非已测主因 | [C3] `build_cache`；[C8] | cache 身份未完整包含 cosmology/solver/growth/数值规则；当前 Ωm^0.55 与模板 growth 要核对 | 新 manifest 与缓存身份报告 |

## 4. 联合 covariance：首先修正，但不要误归因

### 4.1 为什么原 floor 在数学上不合法

已有 [R6] 在 cross=0 下给出

```
lambda_max = 4.269509520005424e9
lambda_min = 1.0042022457664528e-9
floor      = 4.269509520005424e-5
n_floored  = 57
median xi sigma inflation = 15.4000867140
```

P 与 ξ 的单位不同。把整个矩阵的原始数值特征值拿来比较，结果会随 P 单位改变。这里 floor 甚至高于 ξ 最大特征值，因而不是“少数浮点噪声方向的稳定化”，而是把 ξ 信息大幅抹去。

更直接的必要条件是：保持两个 marginal 模型、协方差及参数域不变，且 cross=0 时，

\[
\min_\theta[\chi_P^2(\theta)+\chi_\xi^2(\theta)]\geq
\min\chi_P^2+\min\chi_\xi^2.
\]

原四向 naive joint 最小值约 3.1418，却小于独立最小值之和 14.8463；单极对照也违反。这不能归因于物理互补性或 cosmic variance cancellation。

### 4.2 修复原则

固定 C 的情况下，设 D=diag(sqrt(Cii))，R=D^-1 C D^-1，R=LLᵀ，统一使用 `r_white = solve(L,D^-1 r)`。优化和 MCMC 都调用同一个实现。**不得再在混合量纲 C 上 floor，也不得把报错替换为更大的 ridge。**

本区 `GaussianMetric` 验证 R 的正定性；`assemble_covariance` 保留原边块。若存在真正重复的线性信息，应显式定义支持子空间和有效秩，而不是伪造噪声。若 cross 使矩阵不正定，应修 cross 或校准方式。

注意：已保存 marginal 审计中 `precision_meta.scaled_eigenvalues.n_below_floor=0`。不能据源码里存在 pinv/eigenfloor 就声称这些具体结果已经丢了模式。当前单独 ξ 的小 floor 也远低于其最小特征值。确定错误是混合量纲 joint，不是“所有特征值处理都已经造成同样污染”。

### 4.3 不应武断加一个因子 2，也不应把 P2 再乘 2

源码 `angular_totals` 的 GL weights 求的是完整 `integral_-1^1`，不是半区间平均。各向同性极限给出 2 T²，径向公式里已经有 Gaussian 模式配对所需的 2。P2 模型实际执行的是 `5 * mean(P L2)`，这也是正确的 estimator prefactor；部分注释的 2.5 不是可执行公式。

可从同一个全 ± 模式向量构造所有统计量的行算子 W：

\[
m=W P_q,\qquad C_G=2W\,\mathrm{diag}(T_q^2)W^T.
\]

其中 Tq 是每个模式的总功率，零模排除、偶多极、± 两成员都在模式表里。这样自动保证 PSD，并统一 marginal/cross 的归一化。本区提供小规模版本和抽样测试。它尚未包含真实 FCFC 的所有有限 N、mu-bin 及 mesh alias 修正。

对于已校准 cross，检查

\[
\rho_i=\mathrm{svd}(L_\xi^{-1}C_{\xi P}L_P^{-T})_i.
\]

这里 L 是对应 covariance 的 Cholesky（代码用等价的标准化形式）。固定正定边块时，完整 PSD 需要所有 rho≤1；严格正定需要 <1。

## 5. 新的可复现线索：重分箱不保留 P-bin 归属

[C2] 把径向壳层按 `dk=0.1*kf` 聚合成 G 和 keff。这适用于平滑积分加速，但 [C1] 随后按 keff 是否落进另一组窄 P bins 决定整个聚合组的归属。原来跨边界的模式被一整组移入或移出。

**先聚合再选择 ≠ 先按实际模式选择再聚合。** 当前协方差分母又使用原测量 Nmodes²，导致分子、分母不是同一 mode set。

本次按原几何规则独立枚举得到：

| P 区间 h/Mpc | 精确模式数 | keff 归类的聚合数 | 相对计数差 |
|---|---:|---:|---:|
| [0.017,0.019) | 234 | 258 | +10.26% |
| [0.037,0.039) | 1262 | 1094 | −13.31% |
| [0.061,0.063) | 2994 | 3402 | +13.63% |
| [0.085,0.087) | 6000 | 5352 | −10.80% |

这些数值是**模式归属差**，不是断言 covariance 对角差恰等于该比例，更不是 b1 或 sigma_s 已移动该百分比。T²、Legendre、核的权重还需实际计算。

修复：P 的拟合低 k 区间先按原始 q 或三维模式确定 bin_id，再构建 bin-aware 的累计权重。PP 与 XP 使用相同选择。P0 独立 exact-mode covariance 与 P02 的这段缓存协方差代码不是同一条路径，不能一概而论。

## 6. FullDiscrete 的角向部分并不全离散

在 L=2000 的首个 [0.003,0.005) bin，只有 q=1 与 q=2 的 18 个模式：

\[
\langle\mu^2\rangle=1/3,\quad\langle\mu^4\rangle=2/9,\quad
\langle\mu^6\rangle=1/6,
\]

而连续球平均是 1/3、1/5、1/7。取 bin 内 P_lin 常数、sigma=0 的纯几何 toy，

\[
P_2^{disc}/P_{lin}=5bf/3+25f^2/36,
\quad P_2^{cont}/P_{lin}=4bf/3+4f^2/7.
\]

b=2.55、f=0.81 时分别为 3.898125 和 3.128914。对 P0 的相对影响小得多。**这不是首 bin 的真实谱预测，也不是整个后验有 25% 偏差；低模噪声很大且 P_lin 在壳间变化。** 它证明连续角向不是恒等替换，尤其四极不能未经检验就称 full discrete。

推荐在低 k 使用真实 mu_q 的均值及 covariance；高 k 可使用已收敛的连续近似。比较低 k 修正时，两侧必须使用同一组未压缩径向节点，否则把角向误差和 CachedRebin 误差混在一起。本区 audit 仅给出至 0.095 的隔离诊断；正式使用前扫描转换波数。

## 7. 积分、缓存、优化和边界

### 7.1 连续角积分可以直接消除求积误差

对 x=k sigma，定义

\[
I_n^{(p)}(x)=\frac12\int_{-1}^{1}\frac{\mu^{2n}d\mu}{(1+x^2\mu^2/2)^p}
=\frac{{}_2F_1(p,n+1/2;n+3/2;-x^2/2)}{2n+1}.
\]

signal 使用 p=2，signal² 使用 p=4。加上 shot 交叉项后可精确构造完整 T² 的多极积分，不能只计算 Psignal²。

本次 GL64 对 I0 的相对误差：x=24 为 −0.861263%，x=90 为 −64.970471%。GL256 在 x=90 仍约 −0.571720%。这些是局部积分误差，**不是 ξ 总误差**。高 k 权重和振荡抵消可能大幅减小总影响，须计算 Δmᵀ C_mean^-1 Δm。只用相同 GL64 的 exact/surrogate 相互比较无法检测此问题。

本区解析角矩与独立自适应积分最大相对差约 1.14e-13。高 k 数值准确也不代表线性模型在 k=3 物理正确。

### 7.2 shell-average 已有，不重复推荐 center→shell 作为新发现

提供的解析核可作为独立参考：

\[
\int x^2j_0(x)dx=\sin x-x\cos x,
\]
\[
\int x^2j_2(x)dx=3\,Si(x)+x\cos x-4\sin x.
\]

以端点差除以 k³(rh³−rl³)/3，再乘 i^ell；小参数采用级数。与自适应积分最大绝对差约 1.54e-14。现有 j2 求积是否影响原始拟合必须另测，不能仅凭节点数判错。

### 7.3 最终 covariance 更换后必须 refit

实空间三个脚本先用 C(b1=2.5) 优化，再用 C(b1_hat) 报 χ²或跑 MCMC，但没有重新优化。在最终冻结 covariance 下重做 MAP，并保存该预测。若真的每一步都使用 C(theta)，似然还需 logdet C(theta)；固定 plug-in covariance 则不需要随 theta 加 logdet。

缓存身份至少包括 cosmology/solver、z、matter species、growth、L、kmax、ells、s_edges、rebin 与求积规则、模板和代码 hash。现在文件路径不足以区分这些内容；SHA 只证明文件未变，不证明其物理身份符合当前请求。当前 growth 用 Ωm(z)^0.55 近似而模板可选不同 cosmology；这是需量化的风险，不是已确认的主要偏差来源。

### 7.4 sigma_s=0 不是常规高斯估计点

模型依赖 sigma_s²，因此零点一阶导数消失。中央差分 Fisher 在这里病态；巨大 Fisher error 与有限 MCMC 区间并不自动说明 sampler 坏了。

优化/扫描可以改用 lambda=sigma_s²，但保持原 uniform-sigma 先验时，必须使用 `p(lambda) ∝ 1/(2 sqrt(lambda))`。不要无声换成 uniform-lambda。报告 profile、边界概率、分位数及先验敏感性，不用单个对称误差解释“测得 sigma≈1”。需要有符号形状补偿时引入有符号 counterterm，而不是允许物理速度方差为负。

## 8. 为什么物理模型本身值得改，而非只改变换

### 8.1 有限数据向量不具备完整傅里叶信息等价性

只有完整、可逆、同 covariance 的线性变换才保持似然信息。这里 P 是稀疏的 16 bins，ξ 是有限 r 区间，既不互为可逆变换，也不覆盖相同 k。不能把 `smin=50` 严格等同于 `kmax=0.06`。

设模型缺失 Δd，局部参数偏移近似为

\[
\delta\theta=(J^TC^{-1}J)^{-1}J^TC^{-1}\Delta d.
\]

P 和 ξ 的算子、J、C、尺度范围不同，缺失形状投影到 b1、sigma_s 的方向就不同。这是可检验解释，不是允许任意不一致的借口：在共同模式注入和正确模型下仍必须恢复参数与覆盖率。

### 8.2 最优先的物理假设：BAO 展宽与 FoG 被绑成一个参数

实空间 s≥50 已严重失败，说明密度模板、bias/stochastic 或算子/covariance 至少一项不充分。非线性大尺度位移会展宽 BAO，存在于实空间，不应全部归给 LOS 小尺度速度。[L2]

在当前模型中，sigma_s 同时改变 smooth broadband 和 wiggles。ξ 的 BAO 形状偏差可能把它推向较大值；P02 在其低 k 和角向信息约束下却可能要求零附近。这与“缺失项被不同有效参数吸收”相容，尚需分离实验。

更完整 RSD 还包含不同的 density-density、density-velocity、velocity-velocity 谱及模式耦合；把单一 nonlinear P 塞进 Kaiser 不等于这些项都得到修复。[L3]

### 8.3 最小可判别模型阶梯

先做实空间，逐级增加，不一次放开全部 nuisance：

\[
P_{IR}(k)=P_{nw}(k)+e^{-k^2\Sigma_{BAO}^2/2}P_w(k),
\]
\[
P_h^{real}(k)=A(k)^2P_{IR}(k)+c_0 k^2P_{lin}(k)+N_0.
\]

第一式在这里是诊断性 dewiggling 模板，不冒充完整 IR-resummed one-loop 理论。先仅增加或独立标定 Σ_BAO，再检验有符号 c0。c0 的单位为长度平方；N0 是低 k residual stochastic，不能把它理解为所有 k 恒定的 halo exclusion 模型。

RSD 的最小扩展可写成

\[
P_h^s(k,\mu)=D_{FoG}(k\mu\sigma_v)
[A(k)+f\mu^2]^2\{P_{nw}+e^{-k^2\Sigma^2(\mu)/2}P_w\}
+k^2P_{lin}(k)(c_0+c_2\mu^2)+N_0,
\]
\[
\Sigma^2(\mu)=(1-\mu^2)\Sigma_\perp^2+\mu^2\Sigma_\parallel^2.
\]

先固定或外部标定位移关系，只放开一个 BAO 尺度；不要同时任意放开 Σperp、Σparallel、sigma_v、c0、c2、多个 stochastic 系数。若最小扩展仍失败，再转向包含 bias operators 与 density–velocity 耦合的 TNS/EFT 框架，而非无限增加经验多项式。[L2,L3]

所有新增项必须经相同的离散模式/壳平均算子生成 P 与 ξ。必须检验 fNL=0 和非零注入、fNL 与 nuisance 的退化，以及外部留出尺度预测。当前 Gaussian truth 下改变 p 不改变 fNL=0 的确定性谱，所以调 p 不是修复 Gaussian b1/sigma_s 差异的首选。

若可取得同相位 matter modes，用 P_hm/P_mm 的低 k 平台约束 b1，用 `P_hh-P_hm^2/P_mm` 检验 residual stochastic，能比只看 halo auto 更直接地区分 bias 与 noise。校准需独立相位/留出；用同份测量 P 回填 ξ 不再是独立纯 forward 检验。

### 8.4 stochastic 与高 k 尾部要按估计器处理

理想完整变换中的常数对应 contact 项；有限 hard cutoff 的常数却产生 `N0[sin(Kr)-Kr cos(Kr)]/(2π²r³)` 的非零振荡尾。因此不能把 P 的低 k fitted N0 无条件延伸到 k=3，再把数值振铃称为 halo ξ 信号。

完整周期非零模还涉及零模减除和有限 N pair normalization；真实 FCFC 归一化必须单独匹配。P shot 已减不代表 covariance 中 shot 可以删除。ξ 的 Poisson covariance 高频部分也应检验闭合/尾部收敛，而不是因为 signal 衰减就假定 covariance 尾也可忽略。

## 9. 协方差政策

[R7] 给出 64 维 ξ 向量围绕经验均值的平均 phase χ²=119.1887；在当前固定正确 covariance 假设下期望为 `(24/25)*64=61.44`。这个统计量不依赖理论均值模型，所以不能只靠改 BAO mean model 来解释。

同时 empirical/Gaussian 单-bin sigma 中位数约 1.105，ξ0 约 1.041，ξ2 约 1.182；这不支持把整个矩阵武断乘 2。相关结构、离散角向、非 Gaussian/非 Poisson、有限 N 或 estimator 近似都需要区分。[R7,L4]

Abacus x25 的完整 sample covariance 秩最多 24，89 维 joint 更不可能直接求逆。Hartlap 不会把秩亏矩阵变成可逆；不要把 analytic covariance 当成 Wishart sample covariance 机械修正。适用策略是场级 analytic 基线 + 独立验证的少量校准参数，或预注册低维压缩，再用独立/留出 mocks 检验。只有25个样本时，cross 象限按同批样本拟合且相关系数不高，不能视为高精度物理定标。

Quijote 的500相位可在检查有效维数和独立性后使用 sample covariance 及相应有限 mock 修正；它的规则不自动适用于 Abacus x25。FastPM 也须按自己的 Nmock、z、p、shell 版本处理。单相位 HOD 不能从自身估计可信高维 covariance。不要混用不同样本设置来宣布 rawbox 已闭合。[L5,L6]

## 10. 第一阶段最多五个最小实验

以下阈值是建议预注册的数值验收门，不是从现有结果拟合出来的物理结论。优先复用现成向量/cache；重测大 catalog 放在确有 estimator 证据之后。

### 实验 1：不改物理，修复 likelihood 与比较对象

**固定：**原 data、单独 C_PP/C_ξξ、模型、先验和尺度。**只改：**joint raw floor，统一 whitening；实空间在最终固定 covariance 下 refit。

**输入：**[R1]–[R5] 的原向量/缓存。**输出：**原始与新 block 比较、单位缩放测试、cross=0 χ²、各模型 MAP 和 sigma² profile。

**门：**cross=0 边块保持，χ²相加和单位变换相对误差 <1e-10；joint minimum 不低于独立 minima sum（优化容差内）；MAP likelihood 与 MCMC likelihood 对同 theta 完全相同。

**解释：**未通过则不解释任何新 joint posterior。通过后若 marginal 仍不同，这是预期的：joint bug 本来不解释 marginal 的全部问题。先用 cross=0 做测试，不把它当真实联合科学 likelihood。

### 实验 2：同一模式算子与数值积分闭合

**固定：**同一 A(k)、f、sigma、L、P bins、s edges 与 covariance metric。**依次只改：**bin-aware mode set；低 k 离散 mu；GL→解析角矩；原 j2→解析核；rebin/sigma 插值/UV 截断精度。

**输入：**原 theory NPZ 和任一 P02 NPZ；本区 synthetic 模式。**输出：**每个 P bin 的 exact/rebinned Nmodes、P与ξ预测差、白化 Δm、分 k 段累计贡献、k-switch 与精度扫描图。

**门：**bin counts 严格匹配；共同模式纯线性投影与 mode covariance 通过机器精度/Monte Carlo 误差；每项实际预测误差建议 `Δmᵀ C_mean^-1 Δm<0.01`，并在代表性的参数网格和更严格设置下稳定。不能只看 Δξ/ξ，因零交叉会发散。

**解释：**若低 k/求积修复显著改变 shape 或参数，先修实现再研究物理；若都远低于门而失配不变，可降低它们的主因优先级。band-limited ξ 注入闭合只证明该算子，不证明未滤波 FCFC；后者若仍可疑，再用一个小型可控 catalog 校验 mu-bin、self-pair、normalization 和 mesh response。

### 实验 3：不依赖 mean model 的 covariance 检验

**固定：**每相位数据，残差减经验均值；采用实验2一致算子。**只改：**Gaussian 构造与少量有物理依据的校准，而不是任意四象限缩放。

**输入：**25相位 ξ/P向量及共同模式 Monte Carlo。**输出：**白化残差方差、固定低维模式的分布、cross canonical rho、经验与模型的有限样本置信带。

**门：**完整 PSD；固定的低维 summaries 在独立模拟/留出抽样置信区间内。检验119.19相对61.44的异常是否消失，不能只比较 diagonal median。25相位不够检验所有89维方向，明确不报告完整经验逆矩阵。

**解释：**若 mean-independent 异常持续，covariance 仍不可信，暂停参数 tension 显著性；若通过，才用其量化模型失配。多个 bins/模式的检验需全局校准，避免逐点挑异常。

### 实验 4：最小物理阶梯，先实空间后 RSD

**固定：**实验1–3通过的算子和 covariance，原 PNG 参数/先验。**依次只加：**P 的 residual N0 对照、单个 BAO 展宽尺度、单个有符号 k²P 项；再到 RSD 的独立速度/密度形状。

**输入：**同批实/RSD向量；可选独立 matter 谱。**输出：**BAO区与smooth residual、共同 b1/sigma profile、被 P 稀疏选择遗漏的 bins 和 k>0.095 的预测、被 ξ 拟合排除的 r bins 预测。

**门：**均值 shape 的整体 goodness-of-fit 在经 covariance/边界校准后不再被拒绝，且留出数据改善；建议预注册全局5%拒绝门并避免反复调参挑通过。fNL bias 与覆盖率在注入样本上验证。

**解释：**若独立 BAO 展宽显著降低 ξ 对 sigma_s 的要求而 P2 不再贴边，支持“有效阻尼混合”的假设；若失败，不把更大误差棒当解决，检查 nonlinear bias/density–velocity 项。排除 BAO 后区间变宽导致重叠不构成因果证明，要比较同等信息损失控制。

### 实验 5：真正的 paired 参数一致性和正式重画门

**固定：**通过前述检验的模型/协方差/先验。**比较：**共享参数与分别拟合、不同合法尺度选择、Gaussian 与非零 PNG 注入。

**输入：**paired phase vectors，必要时独立mock扩充。**输出：**相位级 Δtheta 分布、条件残差、共享/分开模型的 likelihood ratio 与模拟标定、fNL覆盖率、最终corner。

使用

\[
r_{\xi|P}=r_\xi-C_{\xi P}C_{PP}^{-1}r_P,\quad
S=C_{\xi\xi}-C_{\xi P}C_{PP}^{-1}C_{P\xi}.
\]

共享固定 covariance 下 χ²_joint=χ²_P+r_condᵀ S^-1 r_cond。**门：**条件残差和 paired 参数差通过预注册的全局检验；边界下用模拟标定，不能直接套通常的 Wilks/高斯 sigma。完整 fNL 结论需覆盖率检验，不能只看Rhat。

**解释：**共同模式正确模型的注入失败仍是实现/统计问题；注入通过而真实数据失败更支持模型缺项。不同压缩造成误差宽度不同是允许的，但不允许错误覆盖率或系统性均值失配。

## 11. 交给执行 agent 的任务书

新增 run 目录和完整 manifest，保留历史产物。先运行本区14项测试，再运行 `audit_rawbox.py` 指向原 theory cache；缺文件或 checksum 不符直接报错，不自动重建替代输入。

修改入口顺序：

1. [C1] 与单极 joint 的 assemble：去量纲敏感 floor；接入共同 `GaussianMetric`；cross=0 自动化下界测试。
2. [C6] 三个 realspace 入口：最终 covariance 冻结后再次 `fit_with`，保存最终 MAP 和一致的 likelihood 结果。
3. [C1] `pk_pole_cov/cross_block`：先 exact/bin-aware selection，再累计；核对 actual Nmodes。优先取消经验 quadrant scales 做归一化检验，不把取消后的结果无验证升级成正式科学 covariance。
4. [C3]/[C5]：引入解析角矩和独立核校验，建立低 k 格点角矩/模式权重；cache 身份包含所有物理与数值设置。
5. 在未修改 C_mean/C_single 约定下执行上面五组实验，记录每项只改变了什么；只在算子、数值、covariance、形状和覆盖率门都有证据后，重画正式限制。

本区代码的职责：`rawbox_numerics.py` 是可接入构件；`audit_rawbox.py` 是只读诊断，默认 fNL=0、b1=2.55、sigma_s=8；低 k 角修正只到0.095，不自动替换正式ξ。它不等价于新的生产pipeline，也没有替你跑MCMC。

## 12. 来源索引与图像待核验清单

### 原始仓库来源

- [R1](../source/project/outputs/task43_outputs/rsd_validation/rawbox/realspace_check/audits/task43_rsd_rawbox_realspace_pk_check.json)：`map,posterior,chi2_mean,pte_mean`。
- [R2](../source/project/outputs/task43_outputs/rsd_validation/rawbox/realspace_check/audits/task43_rsd_rawbox_realspace_check.json)：`results.smin50/smin120,sanity_xi2_real`。
- [R3](../source/project/outputs/task43_outputs/rsd_validation/rawbox/realspace_check/audits/task43_rsd_rawbox_realspace_mcmc.json)：`results.*.b1/fNL`。
- [R4](../source/project/outputs/task43_outputs/rsd_validation/rawbox/joint_pkxi/audits/task43_rsd_rawbox_joint_pkxi_summary.json)：`results.p0_marginal,xi0_marginal_s50`（按实际对象键定位）。
- [R5](../source/project/outputs/task43_outputs/rsd_validation/rawbox/joint_p02xi02/audits/task43_rsd_rawbox_joint_4way_summary.json)：`p02_marginal,xi02_marginal` 下的 `nominal,posterior,precision_meta`。
- [R6](../review_data/joint_covariance_reproduction.json)：量纲floor与χ²下界复现。
- [R7](../source/project/outputs/task43_outputs/rsd_validation/rawbox/closure/task43_rsd_rawbox_x25_fulldiscrete_lorentzian.json)：`covariance_policy.x25_scatter_comparison`。
- [C1](../source/project/codes/task43/task43_rsd_rawbox_joint_4way.py)：P02、cross与joint组装。
- [C2](../source/project/codes/task4/task41_rawbox_norsd_fnl100_profiler.py)：格点简并度与 `precompute_rebin_cache`。
- [C3](../source/project/codes/task43/task43_rsd_model.py)：cache、shell核、FullDiscreteRSDModel。
- [C4](../source/project/codes/task43/task43_fit_rsd_rawbox_pk0_vs_xi0_smin50.py)：精确模式P0与16-bin选择。
- [C5](../source/project/codes/task43/task43_fit_rsd_rawbox_x25.py)：FastRSDModel及periodic covariance。
- C6：[check](../source/project/codes/task43/task43_rsd_rawbox_realspace_check.py)、[mcmc](../source/project/codes/task43/task43_rsd_rawbox_realspace_mcmc.py)、[pk_check](../source/project/codes/task43/task43_rsd_rawbox_realspace_pk_check.py)。
- [C7](../source/project/codes/task43/task43_measure_rsd_rawbox_xi_fcfc.py)：pair定义；[P02测量](../source/project/codes/task43/task43_measure_rsd_rawbox_p02_jaxpower.py)：paint、noise、多极和P0桥接。
- [C8](../source/project/codes/task44/task44_rsd_theory.py)：growth、analytic monopole moments与共享理论构造。
- 历史：[Mission10](../source/background/agent/mission10_log/full_discrete_vs_databin.md)、[Mission11](../source/background/agent/mission11_log/mission11_acceleration_results.md)、[Mission12](../source/background/agent/mission12_log/mission12_results.md)。DataBin 的数据回填不能当独立 forward 验证；rebin 加速成功不能证明跨不连续P-bin选择也安全。

### 论文（方法依据，不是本仓库已通过这些理论的验证）

- L1：[Wands & Slosar 2009, 0902.1084](https://arxiv.org/abs/0902.1084)：PNG尺度依赖bias、相关函数红外问题及均值定义；仓库保留v2。
- L2：[Senatore & Zaldarriaga, 1404.5954](https://arxiv.org/abs/1404.5954)：位移效应、IR resummation 与 BAO相关函数。
- L3：[Taruya, Nishimichi & Saito 2010, 1006.0699](https://arxiv.org/abs/1006.0699)：density–velocity耦合与改进RSD谱。
- L4：[Grieb et al., 1509.04293](https://arxiv.org/abs/1509.04293)：多极P/ξ的Gaussian covariance及bin平均。
- L5：[Hartlap et al., astro-ph/0608064](https://arxiv.org/abs/astro-ph/0608064)：sample inverse bias，不能修复秩亏。
- L6：[Percival et al., 1312.4841](https://arxiv.org/abs/1312.4841)：有限mock covariance噪声向参数误差传播。

### 应由执行 agent 补齐的图像证据

对应实空间 `task43_rsd_rawbox_realspace_{pk0_check,xi0_check,pk0_vs_xi0_contours}.pdf`，RSD `task43_rsd_rawbox_x25_fulldiscrete_lorentzian.pdf`，单极与四向 joint contours。逐图确认图中样本、mask、C_single/C_mean、参数定义与JSON一一对应；旧joint图明确标记失效，不覆盖留存。另输出新实验的白化残差和模型分量图，不能只交新的corner。
