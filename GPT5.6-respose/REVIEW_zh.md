# Rawbox 的 P(k)–2PCF 不一致：独立审阅、可检验解释与执行任务书

## 结论先行

**当前不是一个“给 2PCF 找到正确变换公式就全部解决”的问题。至少要分别处理：联合协方差实现错误、有限盒估计器/数值操作不一致，以及最简物理模板在不同权重下得到不同有效参数。**

最有把握的判断如下。

1. 原四向 joint 的量纲敏感 eigenvalue floor 确实错误。已有单极/四向 joint 的信息增益结论应暂停引用。但这个步骤不在独立 P、xi 拟合中，不能用它解释独立探针的全部差异。
2. “P 和 xi 的 sigma_s 始终严重冲突”需要细分：P0 单独的宽后验与 xi0 明显重叠；P02 与 xi02 的差异才十分突出，而且 P02 的 MAP 接近 sigma_s=0 边界。
3. 实空间不含 FoG/Kaiser 仍有均值形状失败，排除了“只修 RSD 坐标就全部解决”的解释。当前实空间 P0 本身也没有通过均值形状检验，不能视为无偏真值。
4. 本次独立几何复算确认了一个应优先修改的实现：先做 CachedRebin，再用聚合中心重新归入窄 P 箱，会改变成员模式数；四向 P/cross 协方差的分子与实际 Nmodes 分母因此不再使用同一组模式。
5. BAO/非线性 bias、随机项和更完整速度模型是重要物理方向，但尚未证明哪一个单独占主导。必须先完成共同模式、角向、核和协方差闭合测试，再用最小模型增量判断。

本报告限定在 rawbox。没有设计 lightcone、survey window 或 RIC/GIC 的补救。下文“已证实”指代码逻辑、已有审计或明确标注的本次数值测试；不等于整个生产管线已经验证。

## 1. 已做与未做

已阅读交接文件、README/方法与结果导读，追踪 Abacus catalog/测量、FullDiscrete/快速插值、P 精确模式平均、xi covariance、实空间检查和联合拟合代码，核对结果 JSON；同时对照 Mission 10–12、notebook 源码镜像和其他样本的配置差异。相关论文用于核对低 k、Gaussian covariance、BAO/RSD 的理论依据，而不是替代本项目验证。

**本次实际运行了 14 项确定性/合成测试，全部通过。**包括 50,000 次独立模式功率抽样、单位缩放、条件似然恒等式、解析角矩、解析壳核、首箱角度、CachedRebin 成员计数以及审计脚本的合成缓存流程。完整数值见 [local_test_results.json](local_test_results.json)。

**没有运行原始 rawbox NPZ 的本地审计、catalog 重测或 MCMC；也没有逐图完成仓库 PDF 的视觉核验。**连接器成功提供了源码、图索引和 JSON，但原始二进制数组/图未能进入当前运行环境。下文拟合数值来自仓库原结果；“图定位”是执行代理复核入口，不声称已经看过图像像素。代码的合成缓存测试不能当作原始缓存复算。

交付的是可运行的诊断与修复构件及任务书，不是已经证明解决参数偏差的新生产模型。

## 2. 对项目目标和方法的理解

目标是在固定背景宇宙学下，从 local PNG 的尺度依赖 bias 限制 fNL，并把已有 P(k) 模型变成可信的配置空间前向模型。代码中的 alpha 常是 1/M，而不是 M 本身：

\[
\Delta b=f_{\rm NL}b_\phi\alpha(k),\qquad b_\phi=2\delta_c(b_1-p),\quad \delta_c=1.686.
\]

当前 RSD 模板为

\[
P^s(k,\mu)=P_{\rm lin}(k)[b_1+f_{\rm NL}b_\phi\alpha(k)+f\mu^2]^2
[1+(k\mu\sigma_s)^2/2]^{-2}.
\]

P 侧某些实验另外拟合 S0=SN0_SCALE*sn0；Abacus RSD 的数值尺度为 10^4，不能把 sn0 的数字直接解释成物理功率。

BinAvgFit 要在拟合阶段对预测做模式平均，而不是最后画图改中心点。FullDiscrete 保留周期盒 k=2πn/L 的离散性、排除零模，并以体积壳平均核投影：

\[
\xi_{\ell j}=V^{-1}\sum_q g_q P_\ell(k_q)K_{\ell j}(k_q),\quad
K_{\ell j}=i^\ell\frac{\int_{s_-}^{s_+}s^2j_\ell(ks)ds}{\int_{s_-}^{s_+}s^2ds}.
\]

需要纠正一个容易被简写误导的点：PNG 的无限体积 IR 问题不是简单地说“整个 P∝k^-2”。平方 bias 项使 P∝k^(n_s−4)，xi 被积项∝k^(n_s−2)。有限盒非零模求和解决的是这个定义问题，不自动解决非线性谱、BAO、估计器与似然问题。[Wands–Slosar](https://arxiv.org/abs/0902.1084)；仓库 [背景](../docs/01_background.md)。

当前 Abacus 使用 L=2000 Mpc/h、z=0.725、c000、25 个相位、p=1。P 是稀疏的 16 个 k 区间，至 0.095 h/Mpc；xi 主模型求和至 3 h/Mpc，xi0 常取 s≥50，xi2 常取 s≥80。这些向量不是有限维双射。不能把 s_min 等同于一个严格的 π/s_min 波数截断，也不能只截理论、不截测量和协方差来冒充原来的 xi。

样本口径不能混用：Quijote 每个 tag 的 500 相位、FastPM 的 z=1/p=1.2/98 盒、Abacus 的 x25 Gaussian C，以及 HOD-MAP 的单 ph000/连续 49 箱，具有不同的参数和误差定义。Quijote 的较宽后验相容，不反驳 Abacus 均值形状失败；FastPM shell-average 后相容改善，也不等于绝对无偏回收输入 fNL。参见 [结果地图](../docs/04_results.md)。

## 3. 先把实际差异定量分开

以下 sigma68=(q84−q16)/2，只是边际区间摘要，不假设分布严格 Gaussian。RSD 表取 joint 审计中的**独立 marginal**结果；这些 marginal 没经过错误的 joint assemble，仍需另审计各自 covariance。

| 独立拟合 | b1 中位数 ± sigma68 | sigma_s 中位数 ± sigma68 | chi²(C_mean)/名义自由度 |
|---|---:|---:|---:|
| RSD P0，free sn0 | 2.53274 ± 0.05205 | 6.4891 ± 3.9597 | 10.9464/12 |
| RSD xi0，s≥50 | 2.55317 ± 0.05334 | 8.8433 ± 1.0149 | 162.9797/27 |
| RSD P02 | 2.57604 ± 0.04560 | 0.9844 ± 0.7472 | 73.8786/28 |
| RSD xi02，s0≥50,s2≥80 | 2.54252 ± 0.05300 | 7.7905 ± 0.7389 | 297.2792/54 |

来源：[单极原审计](../source/project/outputs/task43_outputs/rsd_validation/rawbox/joint_pkxi/audits/task43_rsd_rawbox_joint_pkxi_summary.json) 的 p0_marginal、xi0_marginal_s50；[四向原审计](../source/project/outputs/task43_outputs/rsd_validation/rawbox/joint_p02xi02/audits/task43_rsd_rawbox_joint_4way_summary.json) 的独立探针项。四向不同项的精确键以原 JSON 为准，勿把 joint 项抄成 marginal。

因此，P0 的 sigma_s 宽度大，不能只看中位数不同就宣称强冲突。P0 MAP 的 sigma_s≈8.248，与 xi0 MAP≈8.771 也相近。P02 MAP 则约 4.24×10^-8，边界效应是真正需要注意的事情。b1 在 RSD 中的单盒边际区间明显重叠；不要把 sigma_s 的显著分离套在所有参数上。

独立 P02 的均值检验也不能沿用 P0 的“通过”。按原有固定 Gaussian C 及名义自由度，P02 的形式 PTE≈5.25×10^-6，xi02≈2.51×10^-35；边界和 covariance 不确定性意味着它们不是已经标定好的精确拒绝概率，但绝不能把这两行描述成均值形状都已通过。

| 实空间最简控制 | b1 MAP | b1 后验中位数 [q16,q84] | chi²(C_mean)/自由度 |
|---|---:|---:|---:|
| P0，固定 residual sn0=0 | 2.64906 | 2.65011 [2.63541,2.66504] | 45.2344/14 |
| xi0，s≥50 | 2.54484 | 2.54157 [2.48991,2.59231] | 331.2791/28 |
| xi0，s≥120 | 2.74355 | 2.36816 [1.94330,2.85174] | 12.0251/21 |

来源：[P 实空间审计](../source/project/outputs/task43_outputs/rsd_validation/rawbox/realspace_check/audits/task43_rsd_rawbox_realspace_pk_check.json)、[xi 实空间检验](../source/project/outputs/task43_outputs/rsd_validation/rawbox/realspace_check/audits/task43_rsd_rawbox_realspace_check.json)、[xi 实空间 MCMC](../source/project/outputs/task43_outputs/rsd_validation/rawbox/realspace_check/audits/task43_rsd_rawbox_realspace_mcmc.json)。

当前这组实空间实验根本没有自由 sigma_s。需要解释的是 b1/形状及 fNL，而不是“真实速度弥散测得多少”。s≥120 的宽区间和显著 MAP–中位数差说明，良好 PTE 可能同时伴随信息损失与退化；不能将 MAP b1=2.74 当成精确物理偏置。

以上后验采用 C_single 是项目明确选择的一盒精度展示；均值检验采用 C_mean=C_single/25。不要把这个有意约定误报成除 25 的 bug，但必须把它与“25 盒联合观测的真实后验”区别开。共同相位的参数差需要配对校准，不应用独立误差平方和直接算显著性。

## 4. 证据表与修复优先级

源码前缀 C=`source/project/codes/task43/`；结果前缀 R=`source/project/outputs/task43_outputs/rsd_validation/`。图前缀 F=`source/project/plots/task43/rsd_validation/`。图入口另见 [FIGURES](../docs/FIGURES.md)；表中图仅定位，未逐图视觉核验。

| 级别 | 判断 | 文件/函数或结果键 | 图定位/影响范围 |
|---|---|---|---|
| P0，已证实 | raw dimensional eigenfloor 改变 xi 边块 | C/task43_rsd_rawbox_joint_4way.py 的 assemble；review_data/joint_covariance_reproduction.json | F/task43_rsd_rawbox_joint_p02xi02_contours.pdf；原 joint 结论失效 |
| P0，本次几何复现 | 聚合中心重新归箱改变成员集合 | C/task43_rsd_rawbox_joint_4way.py::{pk_pole_cov,cross_block}；task4/task41_rawbox_norsd_fnl100_profiler.py::precompute_rebin_cache | 四向 P/cross covariance；不能泛化为所有 P-only 代码均有此问题 |
| P1，代码已证实，影响待量化 | 实空间更新 C 后未在新 C 下重新优化 | C/task43_rsd_rawbox_realspace_{check,mcmc,pk_check}.py::main/fit_with | realspace_check 的 MAP、保存 prediction 与 MCMC 目标不同 |
| P1，数学不等价，影响待量化 | P 真离散角度 vs xi 连续角积分 | ExactPeriodicPk0Model 与 FullDiscreteRSDModel.evaluate | RSD P02/xi02；最小模式尤其值得查 |
| P1，本次标量数值复现 | GL64 高 k sigma 角积分不充分 | FullDiscreteRSDModel.evaluate、FastRSDModel 的参考基准 | Fast 与“exact”相符不等于真实角积分收敛 |
| P1，已有数据证据 | xi 均值 shape 与 Gaussian precision 尚未通过 | R/rawbox/closure/task43_rsd_rawbox_x25_fulldiscrete_lorentzian.json (Gaussian covariance/phase scatter diagnostics) | F/task43_rsd_rawbox_closure_x25.pdf |
| P2，强怀疑 | FoG 在 xi 中补偿 BAO/broadband，非普适参数 | 上述四组 marginal，加实空间 s50 失败 | 需实验4，不是已确定唯一原因 |
| P2，强怀疑 | 固定 residual sn0=0 与线性 bias 使 P 的 b1 吸收形状误差 | 实空间 P 模型与 RSD free-sn0 对照 | F/task43_rsd_rawbox_realspace_pk0_check.pdf |
| P2，未知主因大小 | 非 Gaussian/非 Poisson covariance、插值/UV、pair/paint误差 | covariance 构造、FCFC 和 jaxpower 测量脚本 | 需共同模式及样本校准，不直接经验乘2 |

### 4.1 必须先修的 joint floor

原规则在有量纲混合矩阵上用 max eigenvalue×10^-14 作下限。仓库已保存的 cross=0 复现给出：最大特征值 4.26950952×10^9，下限 4.26950952×10^-5；57 个 xi 方向全部被抬升；xi 单 bin sigma 的中位数变成原来的 15.4001 倍。该 15.4 是**仓库已有离线复现**，不是本次原缓存复算。[原复现 JSON](../review_data/joint_covariance_reproduction.json)。

一个完全不依赖物理解释的必要条件是：保留边块、cross=0、同一模型和参数域时，

\[
\min_\theta(\chi_P^2+\chi_\xi^2)\ge\min\chi_P^2+\min\chi_\xi^2.
\]

原四向记录为 3.1418 < 14.8463，单极也有相同类型违反。它揭示的不是 cosmic variance cancellation，而是似然被改变，或至少不同结果未使用同一目标函数。

修复：先 D=diag(sqrt(Cii))，R=D^-1 C D^-1，验证 R 再 Cholesky。上游已经改变的边块不能靠下游标准化救回。cross=0 必须逐元素保留边块。rho_max=svd(Lx^-1 Cxp Lp^-T) 的最大值必须≤1；超过就拒绝矩阵并修构造/校准。真正精确线性冗余应显式定义支持子空间，不能任意增噪声使矩阵满秩。

原 standalone xi 的相关矩阵最小特征值约0.02765，不接近 10^-12 级阈值；不能说“所有 xi 偏差都由原特征值截断造成”。MCMC 收敛只说明链采样了某个目标，不证明目标正确。

### 4.2 新增线索：CachedRebin 与窄箱成员不交换

CachedRebin 把临近 k_q 先合成 G_B 和 k_eff,B。对平滑全谱求和，这可以是良好近似；但对窄 P 区间加指示函数时，

\[
\sum_q g_q\,1_i(k_q)F_q\ne\sum_B G_B\,1_i(k_{\rm eff,B})F(k_{\rm eff,B}).
\]

本次按源代码相同 dk=0.1 k_f 的几何规则独立重算：

| P 箱 h/Mpc | 真模式数 | 用聚合中心归箱 | 相对差 |
|---|---:|---:|---:|
| [0.045,0.047) | 1656 | 1752 | +5.80% |
| [0.037,0.039) | 1262 | 1094 | −13.31% |
| [0.061,0.063) | 2994 | 2754 | -8.02% |
| [0.085,0.087) | 6000 | 5352 | −10.80% |

完整 16 箱见测试 JSON。这是确定的成员计数差，不是已经测出的 covariance 总偏差或 b1 偏移；权重 T² 随 k 变化，必须用原谱重算。也没有直接与测量 NPZ 的 nmodes 做本地比较。

应对 P 和 cross 块直接按原始离散模式/bin ID 求和，或在每个观测 bin 内单独 rebin，绝不能跨边界再仅靠中心重新归类。原 xi 全谱快速求和可以继续保留，但必须单独验证精度。模式数闭合测试应成为构建缓存的硬门。

### 4.3 角向离散性：不是一个缺失的固定系数

L=2000 的首 P 箱包含18个模式，本次复算得到 <mu²>=1/3、<mu⁴>=2/9、<mu⁶>=1/6，而连续值为1/3、1/5、1/7。

在仅作说明的“箱内 Plin 常数、sigma=0”模型中，

\[
P_2^{\rm discrete}/P_{\rm lin}=5bf/3+25f^2/36,
\quad P_2^{\rm cont}/P_{\rm lin}=4bf/3+4f^2/7.
\]

b=2.55,f=0.81 时为3.898125与3.128914，差约24.6%。**不是实际 P2 首箱已经偏了24.6%**：真实谱、FoG和径向权重不同；它证明连续角平均不能作为低模数盒子的恒等操作。xi0 的同一 toy 改变量小得多，不能据此宣布角向离散性解释实空间 b1 全部偏差。

现行 P2 的 5×模式平均归一化已经修过，不应再次改成2.5。2.5只适用于权重和为2的连续积分。对 real-space xi2≈0 的检查也不足以证明 RSD 高阶角矩及 covariance 正确。

### 4.4 连续角积分自身也要收敛

令 x=k sigma，I_n^(m)=∫_0^1 mu^(2n)/(1+x²mu²/2)^m dmu。解析式为

\[
I_n^{(m)}=\frac{{}_2F_1(m,n+1/2;n+3/2;-x^2/2)}{2n+1}.
\]

信号用 m=2，信号平方用 m=4。P2 需要到 n=3；P2–P2 covariance 需要到 n=6。仓库 task44 已有单极解析矩，可推广复用。

本次标量测试：GL64 对 I0 的相对误差在 x=24 为−0.861263%，x=90 为−64.9705%；即使 GL256，在 x=90 仍有−0.57172%。但这**不是 xi 的总体误差**。要在真实谱、shell 核和 C_mean 中测量 delta_m，而不能直接把这些百分数贴在 b1/sigma_s 上。

提供的解析矩与自适应积分最大相对差1.14×10^-13。壳核也可用 F0(x)=sin x−x cos x、F2(x)=3Si(x)+x cos x−4sin x 的解析原函数，结合小 x 展开；本次核最大绝对差1.54×10^-14。快速 sigma 插值对同一个 GL64 参考的验证不能发现参考自身的偏差。

### 4.5 实空间 MAP 与最后使用的协方差不一致

三份实空间脚本先用 covariance(2.5) 找 best，再设 covariance(best.b1)，但没有第二次优化；MCMC 使用更新后的固定 C，保存的 MAP/prediction 却仍来自旧 C。应在最终冻结 C 下重新求 MAP，再保存预测、chi²与链起点。不能凭注释“two-step/refit”判断已做过 refit。

如果真正让 C 随每个 theta 改变，就应使用 r^T C(theta)^-1 r + log det C(theta) 的完整定义；本轮更简单的办法是明确冻结最终 C 并重拟合。这个 bug 的实际位移尚未测量，不应声称它足以解释约0.1的实空间 b1 差。

## 5. 为什么 b1 与 sigma_s 会系统地不同：物理解释与判别

底层完整傅里叶与配置空间有确定联系，不等于截断、稀疏分箱后的两个向量含相同信息。即使数值实现完美，模型遗漏 delta_m 时，局部线性近似中的参数偏移仍为

\[
\delta\theta\simeq(J^TC^{-1}J)^{-1}J^TC^{-1}\delta m.
\]

不同探针的 J、C 和保留模式不同，得到不同的“最佳有效参数”并不矛盾；这也不意味着可以接受偏差而无需处理。

**实空间的首要候选：非线性/尺度依赖 bias 与随机项。**固定 S0=0 时，P 的 b1 会尝试吸收真实 residual power 或谱形变化。RSD 的 free-sn0 和实空间的 fixed-sn0 不是严格只开关速度项的 A/B。应先在相同真实空间配置释放一个低 k 常数随机项，观察 b1 的稳定性及留出箱预测。正的遗漏随机功率可能把 P 的有效 b1 推高，方向与当前差异相容，但其真实幅度尚未测得。

**第二个候选：线性 BAO 与 broadband 不足。**xi s50–120 对 BAO/局部曲率有敏感性，删掉该区间后形状检验缓解却丢失很多信息。它支持“误差与这些尺度相关”，不单独证明 BAO damping 是唯一解释。FoG 参数乘整个谱时，可同时调整 BAO与平滑成分；xi 偏好的 sigma_s 可能正在补偿未建模的位移/非线性，而 P02 在其波数权重下更愿意贴0边界。

**RSD 的第三个候选：简单 Kaiser×单参数 FoG 不足。**密度–速度非线性耦合不会一般地化为同一个常数 sigma_s。Taruya–Nishimichi–Saito 的密度/速度耦合修正提供的是后续物理扩展依据，不是允许直接把未校准的复杂模型称为正确答案。[TNS](https://arxiv.org/abs/1006.0699)。

### 建议的最小扩展，而不是一次加十个参数

先将 BAO 与 FoG 的作用拆开：

\[
P_{\rm IR}(k,\mu)=P_{\rm nw}(k)+e^{-k^2\Sigma^2(\mu)/2}P_{\rm w}(k),
\quad \Sigma^2(\mu)=(1-\mu^2)\Sigma_\perp^2+\mu^2\Sigma_\parallel^2.
\]

第一阶段先固定位移模板/各向异性关系，只释放一个 BAO damping 幅度；sigma_FoG 独立保留。统一的诊断模板可以写成

\[
P^s_{\rm trial}=D_{\rm FoG}(k\mu\sigma_F)
\{[b_1+f_{\rm NL}b_\phi\alpha+f\mu^2]^2P_{\rm IR}
+c_0k^2P_{\rm lin}+c_2k^2\mu^2P_{\rm lin}\}+S_0+S_2k^2.
\]

这只是按层级添加的诊断 ansatz：先独立比较 S0、一个 BAO 参数、一个形状反项；不是将所有 c0,c2,S2 同时放开。需要保持 P 与 xi 使用同一 underlying spectrum，再进行各自的正确估计器投影。完整生产模型可在证据支持时升级为有一致 bias/速度项的微扰或 EFT 模板。BAO 位移与反项背景见 [Vlah 等](https://arxiv.org/abs/1509.02120)。

**不能把低 k 的 k² 反项或常数随机项无条件外推到 k=3。**必须有明确 UV 完备化/平滑延拓和理论误差预算，或构造一致的带限统计量。否则新模型在 xi 上的改善可能只是截断振铃。

检查过拟合/PNG退化：记录 fNL–b1–形状参数相关、固定参数真值的 synthetic recovery、留出波数/距离预测和参数随 cut 的漂移。不得通过事后调 p 或给 xi 添任意常数来强迫与 P 相等。高 s cut 的好 PTE 与宽先验驱动后验不是充分成功标准。

另一个有判别力的后续观测量是现成 matter 场可用时的 P_hm/P_mm 和 P_hh−P_hm²/P_mm：在 Gaussian 大尺度控制中，它能分离 cross-bias 与 auto stochasticity。当前报告没有这些数组，不能声称已经测得。

### 常数项的特别警告

完整无限体积变换中的全 k 常数对应 s=0 接触项；周期盒去掉零模还带空间常数 −S0/V。有限截断则有振铃，N²/N(N−1) 的 pair 归一化与 Poisson 自对项可能改变/抵消相应偏置。必须从同一个估计器推导，不能把 −S0/V 或 Hankel 截断尾项随意添到 xi。

尤其，低 k 拟合到的 residual stochastic power 不是已证明在所有 k 都恒定。xi 在非零 s 的均值不写常数，也不意味着 covariance 可以忽略噪声：Gaussian C 使用总 T=P_signal+P_noise 的平方。

## 6. 从同一模式推导协方差，避免经验因子替代理论

对于包含完整 ±k 的模式列表和偶对称权重，定义

\[
W_{P_{\ell i},q}=(2\ell+1)1_i(k_q)L_\ell(\mu_q)/N_i,
\quad W_{\xi_{\ell j},q}=(2\ell+1)L_\ell(\mu_q)K_{\ell j}(k_q)/V.
\]

Gaussian、无额外模式耦合近似下，

\[
C_{AB}=2\sum_qW_{Aq}W_{Bq}T_q^2.
\]

这个 Gram 构造使 P、xi、cross 三块相容，并保证半正定；独立半空间写法需要相应调整，不能混用模式数。转写成径向 g_q 及积分_-1^1 dmu 时，积分权重和为2，已经包含从角平均换成角积分的因子，不要再乘一次2。Grieb 等提供相关多极与壳平均公式的外部核对。[Grieb](https://arxiv.org/abs/1509.04293)。

代码的共同模式算子描述的是明确有限频带的场相关函数，不自动包含 FCFC 的有限 mu-bin 投影、N(N−1)、mesh paint/interlacing/alias；真正测量闭合还要匹配这些定义。Gaussian 模式测试不是对 connected trispectrum 或非 Poisson stochastic covariance 的验证。

原 covariance 的 phase 检验：xi02 的平均 chi²_about_empirical_mean=119.1887，正确期望基准为64×24/25=61.44；而 xi0 对角 sigma 比中位数约1.0405，xi2约1.1819。因此“误差棒接近，所以整个 precision 正确”不成立，也不能直接从119/61≈1.94断言漏了一个因子2。

**样本政策：**Abacus x25 的 89 维 sample C 最高秩24；用作低维投影/散布诊断、结构化 shrinkage 与交叉验证，不直接逆。Quijote 500 和 FastPM 98 的 sample precision 可以在独立样本与维数条件满足时考虑 Hartlap/Percival；这些不是给任意解析 C 的默认乘数，也不是模型误差修正。[Hartlap](https://arxiv.org/abs/astro-ph/0608064)、[Percival](https://arxiv.org/abs/1312.4841)。单相位 HOD 不能伪造经验 full covariance。

四个 cross 象限系数约1.03、2.51、2.34、1.72来自同25相位，不能当成已知常数。先用共同估计器归一化检验，再低维校准/留出测试；必须报告这些系数的不确定性与 rho_max。不要先把无效 C floor 成 SPD 再宣称 joint 可信。

条件残差是比叠画两条边际轮廓更直接的诊断：

\[
r_{\xi|P}=r_\xi-C_{\xi P}C_{PP}^{-1}r_P,\quad
C_{\xi|P}=C_{\xi\xi}-C_{\xi P}C_{PP}^{-1}C_{P\xi}.
\]

在有效固定 C 下，chi²_joint=chi²_P+chi²_cond；代码测试了这一恒等式。

## 7. 第一阶段最多五个最小判别实验

以下阈值是建议预注册的工程/科学门，不是从当前数据调出来的“通用正确值”。第一阶段以现成向量、缓存和合成模式为主，不要求重测大 catalog。每项必须新建 run 目录，不覆盖冻结结果。

### 实验1：似然恒等式和最终 metric 统一（先做）

固定数据、模型、参数域、各边块与 cuts；只替换 assemble/precision，并在最终冻结 C 下重新求 MAP。读取原 C 和原 MAP，先不用长链。

产物：原/新 marginal blocks、相关矩阵特征谱、rho_max、单位重标度测试、随机参数点的 chi²_MAP 与 MCMC loglike 一致性、cross=0 最小值下界检查；图为白化残差与边块变化。相对单位变化误差门可设10^-10，cross=0 边块保持到浮点精度。若原 cross 违反 PSD，停止 joint 而不是继续放宽 floor。

通过意味着“数值目标函数自洽”，不意味着形状正确；不通过时先修实现，不进入模型优劣比较。独立探针结果若未变而 joint 变了，正是预期，并不能证明 independent tension 已消失。

### 实验2：共同模式的数值/估计器闭合

固定 underlying P(k,mu)、b/f/sigma/fNL 与 covariance 定义；一次只改变一个离散操作。先列出真 bin IDs/Nmodes，对照 CachedRebin；再用同一未压缩低 k 径向节点比较离散 mu 与连续 mu；再将 GL64 替换解析角矩；最后检查解析壳核、径向 dk、sigma 插值及 UV cutoff。

输入：现成 theory cache、测量 nmodes/k_edges；合成模式功率，不需要新 halo catalog。低 k 显式模式上限逐步提高，例如0.03/0.06/0.10/0.15，直到角向补偿变化收敛；不得只截掉第一箱当作修复。PNG 合成控制同时覆盖0和非零值。UV与径向步长扫描应独立，不将求和 kmax 当成拟合 kmax。

产物：计数比、各操作的 delta_m、delta_m^T C_mean^-1 delta_m、线性化 delta_theta、按 k 累积的核贡献；图为误差来源分解和累计响应。建议数值误差预算门 delta_chi²_mean<0.1，且目标参数移动<0.05 sigma_single，在预设参数网格上通过。

共同频带 Gaussian 模式闭合是强基础门；实际 FCFC 全谱闭合若仍有差异，再要求一致 band-pass 测量或独立小 catalog 对照，检查 mu bins、shot/pairs、mesh。全部数值项小仍有均值残差，则物理模型/真实 covariance 成为重点；若某项大，修完必须重新做原数据拟合。

### 实验3：先在实空间分离随机项与 bias/BAO 形状

固定同25相位、cosmology、p、真实空间坐标、最终冻结 C 和原始 cuts；基线无 FoG。先只释放 P 的 S0；下一组才添加单个 BAO damping 或单个 k²P 型参数，不同时开放全部。

输入：现成 P0/xi0 向量与模型缓存。产物：b1/fNL MAP与分位数、C_mean 白化残差、BAO区域与非BAO区域的残差、留出 bins 的预测评分、随 cuts 的稳定性。所有低 k 新项的 xi 延拓必须明确，不能把常数只在数值截断内 Hankel 后当成物理修复。

如果 S0 使 P 的 b1 稳定而 xi 的 BAO残差仍在，说明至少两类问题并存；如果 BAO自由度降低训练chi²却不改善留出预测或使 fNL 先验主导，不算通过。建议以预注册留出预测/校准后的 PTE及 synthetic fNL recovery共同决定，而非要求两条轮廓视觉重合。

### 实验4：解释 RSD 的 sigma_s，而不是强行共享一个有效参数

只在实验1–2通过后，固定数值算子和最终 metric；分别比较原 Kaiser×FoG 与“独立 BAO位移 + FoG”，必要时才加入最小 density–velocity 修正。依次查看 P0、P02、xi0、xi02；不以错误 joint 作为裁判。

首先只做 sigma profile：优化 u=sigma_s²≥0，避免在 sigma=0处对称导数为0的 Fisher退化。MCMC 若改变量，必须保留先验：原 uniform sigma 对应 pi(u)∝u^-1/2，而不是无声改成 uniform u。用原 sigma 采样也可以。

产物：profile delta_chi²(sigma)、各多极的白化残差、BAO/FoG参数相关、同 cut 的预测评分与 fNL漂移。若独立 BAO参数使 sigma_F 恢复跨探针一致且改善留出形状，支持“原 FoG 代偿 BAO”；若不改善，继续 density–velocity/bias 与 covariance诊断，不能仅凭有更多参数让 chi²下降就宣称解决。

### 实验5：配对参数差与协方差可靠性

固定候选模型、算子和 cuts；只比较预定义的 covariance候选（正确 Gaussian、少参数结构化校准/收缩），用25相位做低维投影和交叉验证，不估计89维自由 precision。

每相位在同一固定 metric 下作 P/xi profile，保存 delta_theta^a=theta_P^a−theta_xi^a；报告配对均值、散布和参数协方差，不假设 P与xi独立。优先对 fNL,b1 的低维差做检验；sigma边界使用带边界的合成校准/重采样，不硬套 Gaussian z值。

产物：配对参数差散点、白化 phase scatter、cross canonical correlations、条件残差、留出预测。通过要求均值残差与phase scatter同时在预设校准区间内，且参数回收偏差满足预定预算，例如<0.1 sigma_single。若仅放大 C 才“通过”，同时记录精度损失和模型残差结构，不能声称均值模型已正确。

## 8. 给执行 agent 的修改清单与禁止事项

首先运行本目录单元测试，然后读取原缓存执行 --cache 审计。接着修改生产源码的以下节点，但写入新的版本/分支和独立输出路径：

- `task43_rsd_rawbox_joint_4way.py` 与 `task43_rsd_rawbox_joint_pkxi.py`：删除有量纲 whole-matrix floor；保持边块，统一 metric，诊断 rho/Schur；不要恢复未经验证的象限常数。
- 四向 `pk_pole_cov/cross_block`：改成真实成员模式或保留bin ID的分组求和，添加 counts 对照。
- `task43_rsd_model.py::evaluate` 和快速模型参考：先推广解析角矩，保留sigma插值加速但改用真正独立的参考；另做低 k方向修正收敛测试。
- 实空间三份 `fit_with/main`：在最终冻结 C 下重新优化并保存同一目标的 MAP/预测/chi²。
- 拟合层：对 sigma=0 边界采用 profile/正确先验，不把伪逆 Fisher 的零方差当成高精度；保存有效秩和参数可识别性。
- 缓存层：key或元数据验证加入 cosmology、growth、kernel/bin edges、数值精度与代码版本。现 `build_cache` 的可复用检查主要看 status/hash；当前缓存有 c000 标记不等于任意换宇宙学复用都安全。没有证据表明本组数据实际用错了 cosmology。

正式重画约束之前必须完成：似然同一性与单位不变性、原始 Nmodes闭合、数值误差预算、mean/phase双检验、共同相位参数差校准、链收敛与先验边界说明。图上分开标注 C_single 约束展示和 C_mean 形状检验；错误 joint 图只能标历史/失效，不再引用0.5%或0.9%增益作物理结论。

**不要做：**单靠改 IR 窗口追着数据调；DataBin 将测得 P 喂回 standalone xi理论；认为提高 MCMC步数能修复 likelihood；直接逆x25的89维样本协方差；任意乘2或整体抬高误差来让PTE过关；将有限带宽xi与原始FCFC xi混称；将本报告合成测试通过写成原始物理验证通过。

## 9. 交付代码与证据边界

[rawbox_diagnostics.py](rawbox_diagnostics.py) 提供 GaussianMetric、保留边块的 assemble、cross canonical correlations、条件残差、解析 Lorentzian角矩、解析壳平均核、低 k共同模式算子、Gaussian Gram covariance、rebin几何复算，以及原 theory cache 的只读审计入口。读取缓存时校验SHA256、cosmology与几何；禁止输出覆盖；不导入NERSC绝对路径脚本，不自动重建缓存。

原缓存审计覆盖：cross=0 floor、单位敏感性、原缓存bin计数、解析/象限定标cross的rho、GL64→解析角矩和缓存核→解析核的预测改变量。它没有自动实现低 k角向修正、完整UV/PNG网格、真实测量nmodes核验或科学重拟合；这些明确留给实验2及执行代理。它还固定当前fiducial b1=2.55,sigma=8,fNL=0,nbar=0.000162131295，不冒充通用生产配置。

[test_rawbox_diagnostics.py](test_rawbox_diagnostics.py) 的14项测试当前全部通过：单位chi²相对误差2.22×10^-16，解析角矩最大相对误差1.14×10^-13，解析核最大绝对误差1.54×10^-14；50,000次Gaussian模式功率抽样的协方差最大偏离约2.07个近似标准误。Monte Carlo标准误表达式只是诊断尺度，不是对非Gaussian field covariance的证明。

需要补齐的原始执行产物：实际缓存审计JSON/NPZ、测量nmodes比对、原图视觉核验、各相位P/xi同口径profile、最终C下的MAP、数值/UV扫描和候选物理模型留出预测。完成这些之前，最稳妥的科学表述是：**联合结果存在确定的数值失真；独立探针还存在估计器/协方差和均值模板问题，不能把某一个修复预先宣布为全部原因。**
