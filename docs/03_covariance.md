# 03 协方差具体如何得到

## 1. 经验sample covariance：Quijote、FastPM及小样本诊断

\[
C_{ij}=\frac1{N-1}\sum_a(d_i^{(a)}-\bar d_i)(d_j^{(a)}-\bar d_j).
\]

Quijote的每个PNG tag有500个realization，各自估计C_sample；不能将不同fNL的covariance差别当作完全不存在。`task45_quijote_ultranest.py::hartlap_precision`（以实际函数名索引为准）调用找回的`task42_quijote_lcp50_profiler.py::covariance_corrections`。

Hartlap为\(\alpha_H=(N-p-2)/(N-1)\)，修正sample inverse的期望偏差，依赖独立Gaussian/Wishart近似；Percival类因子用于有限mock导致的参数误差传播。它们不是消除模型误差的因子，不能套到任意非独立样本，也不应机械套到确定性的解析C上。具体宽度或方差乘什么因子，要检查脚本实现而非只看summary写有“Percival”。

FastPM rawbox主结果使用对应盒样本的估计与修正，原始z=1摘要已保存。subbox parent-cluster问题不是本轮应用范围；不要把子盒个数当作rawbox独立相位数。

Abacus x25的完整89维sample C最多秩24。它可以帮助诊断对角尺度、投影/压缩方向、cross块和模型残差；不能作为满秩89维likelihood直接取逆。事后粗分箱通过仅是诊断，不能包装成预注册模型通过。

## 2. 周期Gaussian covariance：Abacus当前主族

概念上由Gaussian场的Wick contraction得到P-mode功率方差，再以同一估计器权重投影到P_ell/xi_ell。

典型连续/全模式约定下\(Var(P_i)\sim2(P+1/\bar n)^2/N_i\)；但N_i究竟计全空间k、独立k对还是半空间，决定因子2的位置。多极还带\((2\ell+1)(2\ell'+1)\)、角积分/平均和xi的Bessel核及体积归一化。**本仓库恰好存在不同块记账约定的疑点，不能只用这条概念式宣布源码正确。**

代码路线：
- `task43_fit_rsd_rawbox_pk0_vs_xi0_smin50.py::ExactPeriodicPk0Model.gaussian_covariance`
- `task43_fit_rsd_rawbox_x25.py::periodic_gaussian_covariance`
- `task43_rsd_rawbox_joint_4way.py::{angular_totals,pk_pole_cov,cross_block}`

当前RSD covariance冻结在fiducial b1=2.55、sigma_s=8、fNL=0并含Poisson噪声。参数后验变化不意味着每步都重新计算covariance；请明确哪个实验冻结在哪个fiducial。

## 3. 联合cross covariance

拼接\(C=\begin{pmatrix}C_{PP}&C_{P\xi}\\C_{\xi P}&C_{\xi\xi}\end{pmatrix}\)。cross的模式级推导应与两个对角块使用同一个母场、角度权重和模式对约定。当前单极测试用prefactor括号及25相定标；四向再逐象限定标。

这些是混合的解析/经验诊断，并非独立的大样本协方差校准。约1.03、2.51、2.34、1.72的四个象限系数应触发归一化/有限样本检查，不是可不加解释的基本常数。

## 4. HOD-MAP的stochastic covariance

`task44_fit_pngbase_hodmap_rawbox.py`与`task44_fit_pngbase_pk_xi_consistency.py`包含周期Gaussian covariance，以及把P拟合的residual stochastic power传播到xi covariance的诊断。s>0的xi mean没有常数contact项，不意味着xi covariance可以忽略stochastic power。读取对应consistency_diagnostics摘要，区分旧Poisson-only与修正结果。

## 5. C_single与C_mean

x25 mean用于降低测量噪声；若想报告一份盒子/survey的约束精度，后验可使用C_single。若检验mean是否被模型准确描述，需C_mean=C_single/25，或者适当的逐phase统计。一个chi²=6.52(C_single)对应约162.98(C_mean)，不能随意挑更好看的一个。

原始failure audit：sigma对角中位比1.041，但analytic precision作用到xi02 phase scatter平均chi²=119.19、预期61.44。因此“误差棒差不多”并未验证相关结构。记录的正式科学状态仍为失败且协方差/模型形状未区分。

## 6. 数值逆矩阵也是科学定义的一部分

本次复核发现上游raw covariance floor抬升xi方向，之后再做correlation-normalized inverse无法恢复原信息。比较原始/最终边际块、特征谱、有效秩、单位缩放不变性和cross=0可加性，比单看Cholesky成功或MCMC收敛更有意义。

一般的修复候选是先归一化到相关矩阵，再按明确的物理/统计规则处理SPD与精度。但不能自动修复非SPD矩阵来掩盖错误cross归一化，也不能把预期降秩的同模变换当成满秩任意加噪声。请先推导预期秩，再定义似然。

## 本轮不纳入的应用方法

原项目有JAXpower survey Gaussian/RR去卷积、RascalC和EZmock lightcone covariance路线；本包只保留rawbox EZmock标定及必要文献，不导出这些后续survey应用代码/结果。GPT5.6pro应把重点放在周期盒经验、解析、混合cross和有限样本问题。
