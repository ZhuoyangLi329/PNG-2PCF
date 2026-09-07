# Rawbox 的 P(k)–2PCF 不一致：实现审计、物理解释与执行方案

## 0. 核心判断与验证边界

按 `给GPT5.6pro.md` 审阅，仅讨论 rawbox。审阅起点为 `273753d05e5fe143f61b8edd646a0e8593ba36ea`；不修改原始代码、测量、缓存、拟合或正式图。

**不能把结论压缩成“只修一个 bug”或“P 与 ξ 本来就不需要一致”。现有问题有三个层次：**

1. **联合推断存在确定的实现错误。** 混合量纲 covariance 的全局 eigenvalue floor 改变了原 ξ 边块，即使 cross=0 也严重削弱 ξ 权重。旧 joint 的信息增益和“联合结果靠近 P”的物理解释因此不成立。这是仓库已有审计的发现，本次核查并提供替代实现，不冒充新发现。
2. **独立拟合仍有未解决的形状与协方差问题。** joint bug 不解释 marginal 的全部差异。实空间失败说明 RSD 坐标映射不是唯一原因；P02 的零边界又不同于 P0 的宽后验。
3. **最值得检验的物理假设：同一个简化 sigma_s 正在吸收不同的缺失形状。** ξ 对 BAO 展宽和宽波数范围敏感，P02 对角向非线性敏感；线性 Kaiser × 单一 Lorentzian 未必给出可跨尺度共享的有效阻尼。该解释尚未定量证实。

**实际完成：**源码与原审计 JSON 的交叉审阅、论文相关方法核对、14项独立实现测试。测试包括50,000次独立复模式功率抽样、解析角矩、解析壳平均核、量纲不变性、格点几何、重分箱归属和合成缓存端到端审计。见 `local_test_results.json`。

**没有完成：**原始 rawbox NPZ 的本地重拟合、原 catalog 的 FCFC/jaxpower 重测、长链 MCMC、仓库 PDF 图像逐图核验。当前成功取得源码与审计文本，但未能把原始二进制数据/图像转入计算环境。因此拟合数值引用原 JSON；下文图名是待复核产物，不表示已逐图看过。实现测试通过不等于新物理模型通过。

## 1. 项目目标与方法

项目用 local PNG 的尺度依赖 bias 约束 fNL，希望得到不依赖测量 P 回填的纯 forward 2PCF 模型。沿用当前代码约定：

\[
A(k)=b_1+f_{\rm NL}\,2\delta_c(b_1-p)\alpha(k),\quad\delta_c=1.686,
\]
\[
P^s(k,\mu)=P_{dd}^{lin}(k)[A(k)+f\mu^2]^2
\left[1+\frac{(k\mu\sigma_s)^2}{2}\right]^{-2}.
\]

alpha 是代码中的逆传递映射，不应反转其定义。P0 的 residual stochastic 可另加 `sn0 * 1e4`；不能把无量纲 sn0 直接当有量纲功率。

BinAvgFit 匹配实际离散模式平均，而非 bin-center 点值。FullDiscrete 匹配有限周期盒的非零模及径向壳层简并度；球壳平均核匹配 pair-count 的体积平均。**这些解决测量/变换定义，不自动修复物理谱形状。**

本轮核心 Abacus 是 L=2000 Mpc/h、z=0.725、25相位、mmin=1.4e13 的 halo rawbox。P拟合是有间隙的16个窄 bins，并非连续覆盖0.003–0.095；ξ模板求和到 k=3。实空间最简控制只有 fNL、b1，不拟合速度FoG。

有限周期盒去零模涉及 PNG 相关函数的红外与均值定义；这不是任意改变 kmin、经验 ExpWindow 或迁移 survey IC 的理由。[L1] 本轮不推进 lightcone/survey 修复。

## 2. 原始结果：先分清不同差异

区间为原审计的16/50/84分位数，使用 **C_single**；均值形状检验使用 **C_mean=C_single/25**。后者不是把所有非高斯后验区间机械除以5。以下χ²尚未按本报告修复或重新优化。

| 对照 | b1后验中位数 | sigma_s中位数及68%区间 | 均值χ²/自由度 | 来源 |
|---|---:|---:|---:|---|
| 实空间P0，16 bins | 2.6501 [2.6354,2.6650] | 不拟合 | 45.234/14 | R1 |
| 实空间ξ0，s≥50 | 2.5416 [2.4899,2.5923] | 不拟合 | 331.279/28 | R2,R3 |
| 实空间ξ0，s≥120 | 2.3682 [1.9433,2.8517] | 不拟合 | 12.025/21 | R2,R3 |
| RSD P0，free sn0 | 2.5327 | 6.489 [2.261,10.180] | 10.946/12 | R4 |
| RSD ξ0，s≥50 | 2.5532 | 8.843 [7.855,9.884] | 162.980/27 | R4 |
| RSD P0+P2 | 2.5760 | 0.984 [0.313,1.807] | 73.879/28 | R5 |
| RSD ξ0+ξ2，smin=50/80 | 2.5425 | 7.791 [7.067,8.545] | 297.279/54 | R5 |

关键解读：

- **P0的sigma_s本来就很宽。** P0–ξ0与P02–ξ02不是同一种强度的问题；加入P2才出现强烈的零阻尼偏好。
- P02的MAP sigma_s为 `4.2414e-8`，实质在零边界。中位数约1不等于测出了非零速度弥散。
- 实空间P0自己的均值PTE也只有 `3.73696e-5`，不能把其b1当作精确验证的真值。ξ0 s≥50拒绝更严重，但不是唯一失败者。
- s≥120的ξ0虽PTE≈0.939、MAP b1≈2.7435，边际中位数却是2.3682且区间很宽。这是后验几何与信息损失，不能称为“精确恢复b1=2.74”。
- 同批相位下不能用独立误差平方和直接给中心差定显著性；sigma_s还有边界和先验体积效应。应使用paired或条件残差检验。

## 3. 证据表与优先级

代码和JSON的完整路径见末尾来源索引。对应图位于 `source/project/plots/task43/rsd_validation/`；图像尚待实际核验。

| 状态/优先级 | 文件与函数/对象 | 判断及影响范围 | 对应产物 |
|---|---|---|---|
| 已证实/P0 | C1 joint组装；R6 `raw_floor` | 混合量纲floor改写ξ边块，旧joint失效；不解释全部marginal差异 | `task43_rsd_rawbox_joint_p02xi02_contours.pdf` |
| 已证实/P0 | C1 `pk_pole_cov`,`cross_block`；C2 `precompute_rebin_cache` | 聚合中心重新归类窄P bins，分子mode set与分母不一致 | 本区 `radial_rebin_membership_geometry` |
| 已证实操作不同/P1 | C3 `FullDiscreteRSDModel.evaluate`；C4 `ExactPeriodicPk0Model` | 连续mu与离散mu不相同；参数影响尚待量化 | 本区 `first_bin` |
| 已证实数值风险/P1 | C3 GL64；C5 FastRSDModel | 高k sigma角积分有误差，共享近似的surrogate测试检不出 | 本区 `I0_quadrature_relative_errors` |
| 已证实/P0 | C6 realspace脚本 `fit_with`及其调用 | 更新covariance后未重新优化，MAP和最终likelihood不一致 | realspace check/contours |
| 原数据拒绝/P1 | R7 phase scatter统计 | covariance形状问题不依赖理论均值，不能只改mean model | 原x25 closure JSON/图 |
| 待验证/P1 | C1 quadrant scales | 四象限经验rescaling不能代替模式归一化；需PSD与独立校准 | canonical rho/白化scatter |
| 已证实非正则/P1 | R5 P02 `nominal.theta` | sigma=0导数消失，局部Fisher不能代表边界后验 | sigma² profile |
| 强物理假设/P2 | R1–R5；线性模板 | BAO展宽、非线性bias与density–velocity项被b1/sigma吸收 | BAO/broadband分解残差 |
| 待检查，不先判bug | C3 `shell_jell_kernel` | shell-average已经实现；j2求积影响须独立测 | 解析核与原核白化差 |
| 风险，非已测主因 | C3 `build_cache`；C8 | cache身份未完整包含cosmology/solver/growth/数值规则 | 新manifest |

## 4. 联合covariance：必须先修，但不能误归因

### 4.1 原floor为什么不合法

R6在cross=0下给出：

```
lambda_max = 4.269509520005424e9
lambda_min = 1.0042022457664528e-9
floor      = 4.269509520005424e-5
n_floored  = 57
median xi sigma inflation = 15.4000867140
```

P与ξ单位不同。对整个原始数值矩阵的特征值做比较，结果随P单位改变。这里floor高于ξ最大特征值，故不是修复少数浮点方向，而是大幅抹去ξ信息。

保持原marginal模型、协方差及参数域不变，cross=0必须满足

\[
\min_\theta[\chi_P^2(\theta)+\chi_\xi^2(\theta)]\geq
\min\chi_P^2+\min\chi_\xi^2.
\]

原四向naive joint最小值约3.1418，小于独立最小值之和14.8463；单极也违反。这不能解释为物理互补或cosmic variance cancellation。

### 4.2 修复原则与排除错误指控

固定C时，D=diag(sqrt(Cii))，R=D^-1 C D^-1，R=LLᵀ，统一使用 `r_white=solve(L,D^-1 r)`。优化与MCMC调用同一实现。`GaussianMetric`验证R；`assemble_covariance`保留边块。

**不要再做混合量纲floor，也不要把报错改成更大ridge。** 真正线性冗余应显式定义支持子空间和有效秩；cross造成非正定则修cross或其校准。

已保存marginal的 `precision_meta.scaled_eigenvalues.n_below_floor=0`。不能只因源码有pinv/eigenfloor，就断言这些具体结果已丢模式。单独ξ的小floor也远低于其最小特征值。确定错误是混合joint，不是“所有矩阵处理都已被同样污染”。

### 4.3 不能武断再加一个2，也不能把P2再乘2

`angular_totals`计算完整 `integral_-1^1`，不是半区间平均。各向同性极限是2T²，径向公式已包含Gaussian模式配对的2。P2实际执行 `5*mean(P L2)`，也是正确prefactor；有些注释里的2.5不是执行公式。

可从同一个全±非零模式向量建立行算子W：

\[
m=WP_q,\qquad C_G=2W\,\mathrm{diag}(T_q^2)W^T.
\]

Tq是总功率，偶多极且±两成员都在表中。此构造自动PSD并统一marginal/cross。本区给出小规模实现与抽样测试；不自动包括真实FCFC的有限N、mu-bin、mesh alias修正。

固定正定边块时，令L为各边块Cholesky，完整PSD要求

\[
\rho_i=\mathrm{svd}(L_\xi^{-1}C_{\xi P}L_P^{-T})_i\leq1;
\]

严格正定要求<1。代码以等价标准化形式计算。四象限经验放大后尤其应检查此条件。

## 5. 新线索：CachedRebin不保留P-bin归属

C2按 `dk=0.1*kf` 聚合成G和keff；C1再按keff是否落入另一组窄P bins，决定整个聚合组的归属。跨边界模式会整组移入/移出，分母却仍用原测量Nmodes²。

**先聚合再选择，不等于先按实际模式选择再聚合。**

本次按源码运算顺序 `kf=2*pi/L; dk=0.1*kf` 独立枚举得到：

| P区间 h/Mpc | 精确模式数 | keff归类的聚合数 | 相对计数差 |
|---|---:|---:|---:|
| [0.037,0.039) | 1262 | 1094 | −13.31% |
| [0.045,0.047) | 1656 | 1752 | +5.80% |
| [0.053,0.055) | 2570 | 2378 | −7.47% |
| [0.085,0.087) | 6000 | 5352 | −10.80% |

这些是**模式归属差**，不是covariance对角差恰等于该比例，更不是参数移动比例。T²、Legendre与核还需加权。

浮点截断分组在整数边界对运算顺序敏感；复现时须保留原 `kf`、`dk` 和 `astype(int)` 顺序，不能把代数等价重排默认为逐bin数值相同。这里以上表及保存JSON为准；无论这些边界细节，跨不同bin体系的整组归类都不是合法的精确P-bin选择。

修复：P低k区域先按原q或三维模式确定bin_id，再构建bin-aware累计权重。PP与XP必须使用相同选择。P0独立exact-mode covariance与P02这段缓存代码不是同一路径，不能一概而论。最后仍要从原NPZ实际核对Nmodes，而不是把本几何复现称为原数组已重跑。

## 6. FullDiscrete的角向并不全离散

L=2000的首个[0.003,0.005) bin只有q=1、2的18个模式：

\[
\langle\mu^2\rangle=1/3,\quad\langle\mu^4\rangle=2/9,\quad
\langle\mu^6\rangle=1/6,
\]

连续球平均则为1/3、1/5、1/7。取bin内P_lin常数、sigma=0的几何toy：

\[
P_2^{disc}/P_{lin}=5bf/3+25f^2/36,
\quad P_2^{cont}/P_{lin}=4bf/3+4f^2/7.
\]

b=2.55、f=0.81时为3.898125和3.128914；P0相对影响小得多。**这不是实际首bin谱预测，更不是整体后验有25%偏差**：实际谱在壳间变化，低模噪声也很大。它证明尤其四极不能未经检验就把连续角向当恒等替代。

低k使用真实mu_q的均值与covariance，高k连续近似须扫描转换波数。隔离角向误差时两侧用相同未压缩径向节点，避免混入rebin误差。本区audit只计算到0.095的角修正诊断，不自动应用到正式ξ。

## 7. 数值、缓存、优化与边界

### 7.1 连续角积分可以解析处理

令x=k sigma：

\[
I_n^{(p)}(x)=\frac12\int_{-1}^{1}\frac{\mu^{2n}d\mu}{(1+x^2\mu^2/2)^p}
=\frac{{}_2F_1(p,n+1/2;n+3/2;-x^2/2)}{2n+1}.
\]

signal用p=2、signal²用p=4，另加shot交叉项才能得到完整T²。

GL64对I0的相对误差：x=24为−0.861263%，x=90为−64.970471%；GL256在x=90仍约−0.571720%。**这是局部积分误差，不是ξ总误差。** 应计算真实权重后的Δmᵀ C_mean^-1 Δm。相同GL64的exact/surrogate比较发现不了共享误差。

本区解析角矩与独立自适应积分最大相对差约1.14e-13。高k积分准确仍不意味着线性模型在k=3物理有效。

### 7.2 shell-average已经实现，不重复推荐center→shell作为新发现

提供独立解析参考：

\[
\int x^2j_0(x)dx=\sin x-x\cos x,
\qquad
\int x^2j_2(x)dx=3Si(x)+x\cos x-4\sin x.
\]

端点差除以k³(rh³−rl³)/3，再乘i^ell；小参数用级数。与自适应积分最大绝对差约1.54e-14。现有j2求积是否影响原拟合需实测，不能仅凭节点数判错。

### 7.3 最终covariance更新后必须refit

C6先用C(b1=2.5)优化，再用C(b1_hat)报χ²或跑MCMC，却未重新优化。应在最终冻结的C下重做MAP并保存该预测。若每一步都用C(theta)，似然还要logdet C(theta)；固定plug-in covariance不需要随theta加此项。

cache身份至少包括cosmology/solver、z、matter species、growth、L、kmax、ells、s_edges、rebin/求积规则、模板和代码hash。SHA只保证文件没变，不保证符合当前物理请求。当前growth近似Ωm(z)^0.55与可选模板cosmology需核对，但尚未证明是主要偏差。

### 7.4 sigma=0不是常规高斯估计点

模型依赖sigma²，所以零点一阶导数消失，中央差分Fisher病态。巨大Fisher error与有限MCMC区间不自动说明sampler坏了。

优化可用lambda=sigma²；保持uniform-sigma先验时必须用 `p(lambda)∝1/(2sqrt(lambda))`，不可无声换成uniform-lambda。应报告profile、边界、分位数与先验敏感性。需有符号形状补偿时用counterterm，而不是负的物理速度方差。

## 8. 物理模型为什么值得改

### 8.1 当前有限向量并不保持完整傅里叶信息等价

只有完整、可逆、同covariance的线性变换才保持似然信息。稀疏16个P bins和有限r的ξ不互为可逆变换，也不覆盖同样k；`smin=50`不严格等于某个kmax。

模型缺失Δd时，局部参数偏移约为

\[
\delta\theta=(J^TC^{-1}J)^{-1}J^TC^{-1}\Delta d.
\]

算子、J、C、尺度不同，使缺失形状投影到b1/sigma的方向不同。这是可检验解释，不是允许任意失配的借口：共同模式与正确模型的注入仍必须恢复参数和覆盖率。

### 8.2 优先物理假设：BAO展宽与FoG被绑成一个参数

实空间s≥50失败，说明密度模板、bias/stochastic、算子/covariance至少一项不充分。非线性大尺度位移在实空间也展宽BAO，不该全归为LOS小尺度速度。[L2]

现有sigma同时改变smooth broadband和wiggles。ξ的BAO误差可能把sigma推高；P02低k/角向信息却要求接近零。这支持“有效参数吸收不同缺项”的检验方向，尚非定论。

更完整RSD涉及不同density-density、density-velocity、velocity-velocity谱和模式耦合。把单一nonlinear P塞进Kaiser不等于解决这些项。[L3]

### 8.3 最小模型阶梯与过拟合控制

先实空间、逐级增加，不一次放开所有nuisance：

\[
P_{IR}(k)=P_{nw}(k)+e^{-k^2\Sigma_{BAO}^2/2}P_w(k),
\]
\[
P_h^{real}(k)=A(k)^2P_{IR}(k)+c_0k^2P_{lin}(k)+N_0.
\]

这是诊断性dewiggling模板，不冒充完整IR-resummed one-loop理论。先独立标定或增加一个Σ_BAO，再检验有符号c0（单位长度平方）。N0是低k残余stochastic，不代表所有k恒定的halo exclusion。

RSD最小扩展：

\[
P_h^s=D_{FoG}(k\mu\sigma_v)[A+f\mu^2]^2
\{P_{nw}+e^{-k^2\Sigma^2(\mu)/2}P_w\}
+k^2P_{lin}(c_0+c_2\mu^2)+N_0,
\]
\[
\Sigma^2(\mu)=(1-\mu^2)\Sigma_\perp^2+\mu^2\Sigma_\parallel^2.
\]

先固定或外部标定位移关系，只放开一个BAO尺度；不要同时任意放开Σperp、Σparallel、sigma_v、c0、c2和多项noise。若仍失败，再升级含bias operators与density–velocity耦合的TNS/EFT，而非无限增加经验多项式。[L2,L3]

所有新增项经相同离散/壳平均算子生成P与ξ。检验fNL=0及非零注入、PNG–nuisance退化和留出尺度预测。Gaussian truth时改变p不改变fNL=0的确定性谱，故调p不是修复Gaussian b1/sigma差异的首选。

可选同相位matter modes：用P_hm/P_mm低k平台约束b1，用 `P_hh-P_hm^2/P_mm` 区分residual stochastic。校准需独立或留出相位；把同份测量P回填ξ不再是独立纯forward检验。

### 8.4 stochastic与高k尾必须匹配估计器

理想完整变换的常数是contact项；hard cutoff却产生

\[
\xi_{N_0}^{K}(r)=N_0[\sin(Kr)-Kr\cos(Kr)]/(2\pi^2r^3).
\]

不能把P低k的N0无条件外推到k=3，再把数值尾称作halo信号。完整周期非零模还涉及零模减除与有限N pair normalization，应匹配FCFC实际定义。

P已减shot不代表covariance能删shot；ξ的Poisson covariance高频部分也需尾部/解析闭合检验，signal衰减不保证covariance尾同样可忽略。

## 9. 协方差政策

R7的64维ξ向量围绕经验均值，平均phase χ²=119.1887；固定正确covariance假设下期望 `(24/25)*64=61.44`。该统计量不依赖理论mean，故仅改BAO mean不能解释。

empirical/Gaussian单-bin sigma中位数约1.105，ξ0约1.041、ξ2约1.182；不支持把整个矩阵武断乘2。应区分相关结构、离散角向、非Gaussian/非Poisson、有限N和estimator近似。[R7,L4]

Abacus x25 sample covariance秩最多24，不能逆64维ξ或89维joint。Hartlap不能修秩亏；也不能给analytic covariance机械套Wishart sample correction。可用场级analytic基线+少量独立校准，或预注册低维压缩后检验。四象限cross从同25相位回归且矩阵相关不高，不能当高精度物理定标。

Quijote500相位可在检查独立性/维数后采用sample covariance及有限mock修正，规则不自动迁移到x25。FastPM须保留自己的Nmock、z、p与shell版本。单相位HOD不能自估高维covariance。[L5,L6]

## 10. 第一阶段：最多五个最小判别实验

阈值为建议预注册的验收门，不是从结果调出来的结论。先复用现成向量/cache，再决定是否重测大catalog。

### 实验1：不改物理，修likelihood与比较对象

**固定：**原data、marginal C、模型、先验、尺度。**只改：**joint raw floor与统一whitening；实空间在最终C下refit。

**输入/输出：**原向量和cache；输出block差、单位缩放、cross=0 χ²、MAP与sigma² profile。

**门：**边块保留；χ²相加及单位变换相对误差<1e-10；joint minimum不低于独立minima sum（优化容差内）；MAP与MCMC对同theta的likelihood相同。

**解释：**未通过则不解释新joint。通过而marginal仍异，并不奇怪：joint bug本来不解释全部marginal。cross=0只用于验算，不作为真实联合科学likelihood。

### 实验2：共同模式算子与数值闭合

**固定：**A(k)、f、sigma、L、P bins、s edges与metric。**逐项只改：**bin-aware mode set、低k离散mu、解析角矩、解析j2核、rebin/插值/UV精度。

**输入/输出：**原theory NPZ、一个P02 NPZ与合成模式；输出exact/rebinned Nmodes、P/ξ差、白化Δm、分k贡献及k-switch扫描图。

**门：**计数严格匹配；共同模式投影与Gaussian covariance在数值/抽样误差内闭合；实际预测误差建议 `Δmᵀ C_mean^-1 Δm<0.01`，在代表性参数网格和更严设置下稳定。不能只看Δξ/ξ的零交叉发散。

**解释：**修复显著改变预测则先解决实现；修复远低于门而失配不变则降低其主因优先级。band-limited ξ注入只证明该算子，不证明未滤波FCFC；若仍可疑，再用小型可控catalog核验mu-bin、self-pair、normalization与mesh response。

### 实验3：不依赖mean model的covariance测试

**固定：**每相位数据，残差减经验均值，采用实验2算子。**只改：**Gaussian构造及少量有物理依据的校准，不任意缩放四象限。

**输入/输出：**25相位向量与模式MC；输出白化方差、固定低维投影分布、canonical rho、有限样本置信带。

**门：**PSD；预注册低维统计在独立模拟/留出抽样置信带内；明确检验119.19相对61.44的异常，不只比较diagonal median。25样本不足验证所有89维方向，不报告完整经验逆。

**解释：**mean-independent异常持续则暂停tension显著性。通过后再用该C量化mean失配。多模式检验需全局校准，避免逐点挑异常。

### 实验4：最小物理阶梯，先实后红移

**固定：**通过实验1–3的算子/C、PNG约定。**逐项只加：**P residual N0对照、一个BAO展宽尺度、一个有符号k²P项，再到RSD的独立速度/密度形状。

**输入/输出：**实/RSD向量，可选独立matter谱；输出BAO/smooth残差、共同b1/sigma profile、P遗漏bins与k>0.095以及ξ排除r bins的留出预测。

**门：**整体均值GOF经covariance/边界校准后不再拒绝，且留出预测改善；建议预注册全局5%拒绝门，避免反复挑通过。检验fNL偏差与覆盖率。

**解释：**独立BAO展宽降低ξ对sigma的要求且P2不再贴边，支持有效阻尼混合假设；失败则查nonlinear bias/density–velocity项。删除BAO后仅因误差变宽而重叠不是证明，要做同等信息损失控制。

### 实验5：paired参数一致性与正式重画门

**固定：**通过上述检验的模型、C与先验。**比较：**共享/分别参数、合法尺度选择、Gaussian与非零PNG注入。

**输入/输出：**paired phase vectors，必要时独立mock；输出Δtheta分布、条件残差、共享/分开likelihood ratio的模拟标定、覆盖率和最终corner。

\[
r_{\xi|P}=r_\xi-C_{\xi P}C_{PP}^{-1}r_P,
\quad S=C_{\xi\xi}-C_{\xi P}C_{PP}^{-1}C_{P\xi}.
\]

固定共享C下 `chi2_joint=chi2_P+r_condᵀ S^-1 r_cond`。**门：**条件残差/paired差通过预注册全局检验；边界下用模拟标定，不直接套Wilks或独立高斯sigma。fNL结论须覆盖率，不只Rhat。

**解释：**正确模型注入失败仍是实现/统计问题；注入通过而真实数据失败更支持物理缺项。不同压缩的误差宽度可不同，但不允许错误覆盖率或系统性均值失配。

## 11. 执行agent交接

新增run目录和manifest，保留历史产物。先跑本区14项测试，再让 `audit_rawbox.py` 读原theory cache；缺文件/checksum不符直接报错，不自动重建替代。

修改顺序：

1. C1与单极joint的assemble：去混合量纲floor、共用GaussianMetric、加入cross=0下界测试。
2. C6三个实空间入口：最终C冻结后再次fit，保存最终MAP和一致likelihood。
3. C1 `pk_pole_cov/cross_block`：exact/bin-aware选择后累计，核对Nmodes；先取消经验scales做归一化检验，不无验证升级为正式C。
4. C3/C5：解析角矩、独立核测试、低k格点角向；补齐cache身份。
5. 依次完成五组实验，逐项记录只改了什么；算子、数值、C、形状、覆盖率都有证据后才重画正式限制。

`rawbox_numerics.py`是构件；`audit_rawbox.py`是只读诊断，默认fNL=0、b1=2.55、sigma=8，低k角修正仅到0.095。它们不是已验证生产pipeline，也没有运行MCMC。

## 12. 来源与图像核验清单

### 仓库数据与代码

- [R1](../source/project/outputs/task43_outputs/rsd_validation/rawbox/realspace_check/audits/task43_rsd_rawbox_realspace_pk_check.json)：`map,posterior,chi2_mean,pte_mean`。
- [R2](../source/project/outputs/task43_outputs/rsd_validation/rawbox/realspace_check/audits/task43_rsd_rawbox_realspace_check.json)：`results.smin50/smin120,sanity_xi2_real`。
- [R3](../source/project/outputs/task43_outputs/rsd_validation/rawbox/realspace_check/audits/task43_rsd_rawbox_realspace_mcmc.json)：`results.*.b1/fNL`。
- [R4](../source/project/outputs/task43_outputs/rsd_validation/rawbox/joint_pkxi/audits/task43_rsd_rawbox_joint_pkxi_summary.json)：查找 `p0_marginal,xi0_marginal_s50` 对象内的 `nominal,posterior`。
- [R5](../source/project/outputs/task43_outputs/rsd_validation/rawbox/joint_p02xi02/audits/task43_rsd_rawbox_joint_4way_summary.json)：`p02_marginal,xi02_marginal` 的 `nominal,posterior,precision_meta`。
- [R6](../review_data/joint_covariance_reproduction.json)：量纲floor与χ²下界。
- [R7](../source/project/outputs/task43_outputs/rsd_validation/rawbox/closure/task43_rsd_rawbox_x25_fulldiscrete_lorentzian.json)：x25 phase scatter与covariance诊断。
- [C1](../source/project/codes/task43/task43_rsd_rawbox_joint_4way.py)：P02、cross和joint组装。
- [C2](../source/project/codes/task4/task41_rawbox_norsd_fnl100_profiler.py)：`precompute_rebin_cache`与格点简并度。
- [C3](../source/project/codes/task43/task43_rsd_model.py)：cache、shell核、FullDiscrete。
- [C4](../source/project/codes/task43/task43_fit_rsd_rawbox_pk0_vs_xi0_smin50.py)：exact-mode P0与16 bins。
- [C5](../source/project/codes/task43/task43_fit_rsd_rawbox_x25.py)：FastRSDModel与periodic covariance。
- C6：[check](../source/project/codes/task43/task43_rsd_rawbox_realspace_check.py)、[mcmc](../source/project/codes/task43/task43_rsd_rawbox_realspace_mcmc.py)、[pk_check](../source/project/codes/task43/task43_rsd_rawbox_realspace_pk_check.py)。
- [C7](../source/project/codes/task43/task43_measure_rsd_rawbox_xi_fcfc.py)：pair定义；[P02测量](../source/project/codes/task43/task43_measure_rsd_rawbox_p02_jaxpower.py)：paint、noise、多极、P0桥接。
- [C8](../source/project/codes/task44/task44_rsd_theory.py)：growth、角矩与共享理论。
- 历史：[Mission10](../source/background/agent/mission10_log/full_discrete_vs_databin.md)、[Mission11](../source/background/agent/mission11_log/mission11_acceleration_results.md)、[Mission12](../source/background/agent/mission12_log/mission12_results.md)。DataBin回填不等于独立forward验证；rebin加速成功不证明跨不连续P-bin选择安全。

### 方法论文

- L1：[Wands & Slosar 2009, 0902.1084](https://arxiv.org/abs/0902.1084)：PNG bias、相关函数红外与均值；仓库保留v2。
- L2：[Senatore & Zaldarriaga, 1404.5954](https://arxiv.org/abs/1404.5954)：位移、IR resummation、BAO相关函数。
- L3：[Taruya, Nishimichi & Saito 2010, 1006.0699](https://arxiv.org/abs/1006.0699)：density–velocity耦合与RSD谱。
- L4：[Grieb et al., 1509.04293](https://arxiv.org/abs/1509.04293)：多极P/ξ Gaussian covariance及bin平均。
- L5：[Hartlap et al., astro-ph/0608064](https://arxiv.org/abs/astro-ph/0608064)：sample inverse bias，不能修秩亏。
- L6：[Percival et al., 1312.4841](https://arxiv.org/abs/1312.4841)：有限mock covariance噪声传播。

这些论文提供方法依据，不代表本仓库已经通过对应理论的验证，也不将不同版本数值无声替代历史实验。

### 待执行agent逐图核验

实空间 `task43_rsd_rawbox_realspace_{pk0_check,xi0_check,pk0_vs_xi0_contours}.pdf`，RSD `task43_rsd_rawbox_x25_fulldiscrete_lorentzian.pdf`，单极与四向joint contours。核对样本、mask、C_single/C_mean、参数和JSON的一一对应；旧joint图标记失效而不覆盖。另输出新实验的白化残差与模型分量图，不能只交新的corner。
