# 修订审阅：冻结 P(k)，只改进 ξ 的 RSD 映射

日期：2026-09-08。源码审阅固定在 `e6f217b3e4890a22ac866b0a840ac9d6ae00056a`。

## 0. 本轮约束与修订结论

本报告按用户最新要求，**不更换、不重写现有 DESI-PNG 风格的 P(k) 模型**。P 的模板、分箱、参数定义、先验、已存结果及其共享缓存均作为冻结参考。此前报告中要求先升级 P 物理模型的建议，不再是本轮执行路线。这里只新增 ξ 分支的诊断和候选模型，不修改原生产代码。

结论分三层：

1. **现有 ξ 的线性 RSD 加法在连续角向、平行视线的理论层面没有发现原则性错误。** 它先构造 `[B(k)+f mu^2]^2 P_L(k)`，乘 FoG，求多极，然后用带 `i^ell` 的球壳平均核变换。它不是把配置空间 ξ 错误地直接乘 `(1+beta mu_s^2)^2`。当 FoG 关闭时，这就是 Kaiser 线性 RSD 的标准配置空间实现。
2. **GSM 值得作为独立的 ξ 候选，但不是另一套更正确的线性 Kaiser。** GSM 在严格线性振幅极限必须恢复 Kaiser；其新增能力是对实空间到红移空间的映射保留高阶效应，并允许有分离/方向依赖的成对速度统计。它是成对速度 PDF 的 Gaussian 闭合，不是精确的非线性理论，也不等同于“密度和速度都是 Gaussian 时的精确映射”。
3. **优先实验改为：先固定 ξ 的实空间输入，检验 RSD 映射本身；随后才判断是否需要改 ξ 的实空间输入或速度矩模型。** P(k) 全程不动。若新 GSM 更好，旧 P 的 `sigma_s` 与 GSM 的额外成对速度参数不应继续强制同名共享。

本区代码是可执行的数学参考，不是已验证的新 rawbox 生产模型。本次实际运行了 **11 项新测试，全部通过**，不是前一轮14项测试的再次宣称。没有原始 rawbox 拟合、MCMC 或 catalog 重测；没有以未看的仓库图作证据。

## 1. 逐函数核查：当前究竟如何加入 RSD？

| 原始文件/函数 | 实际操作 | 本轮判断 |
|---|---|---|
| `task43_rsd_model.py::FullDiscreteRSDModel.evaluate` | `amplitude=b1+fnl*bphi*alpha`；`pkmu=pk_dd*(amplitude+f_growth*mu2)^2*damping` | PNG bias 在角积分内部，Kaiser 交叉项和 f² 项均在；没有发现把实空间 ξ 直接乘角因子的错误 |
| 同文件的多极投影 | `0.5*(2ell+1)*sum(wmu*pkmu*Lell)` | 连续角积分归一化正确；`mu` 是 Fourier 模式方向，不是配置空间分离方向 |
| `shell_jell_kernel` | 对半径作体积平均，并乘 `(-1)^(ell/2)` | 包括四极的负号；不能把缺少 `i²` 当作现有 bug |
| `task43_fit_rsd_rawbox_x25.py::FastRSDModel` | 六项 b1/fNL 多项式系数，加 sigma 的 spline | 是同一个 Kaiser×Lorentzian 模型的代理，不是独立 GSM；换成 GSM 后不能原样复用这六项基底 |
| `task43_fit_rsd_rawbox_pk0_vs_xi0_smin50.py::ExactPeriodicPk0Model` | 从 `exact_xi` 读取 `k_eff,pk_dd,alpha,f_growth,boxsize` | 修改共享 ξ cache 可能间接改变 P！新 ξ 模型必须使用独立 context/cache/driver |
| `task43_measure_rsd_rawbox_xi_fcfc.py` | 先用 RSD catalog 做 DD；FCFC 输出有限 mu bins 的 ξ0/ξ2 | GSM 应在红移空间映射后再投影/球壳平均，最终匹配 FCFC 的实际 mu-bin 权重 |

现有代码先在完整 `P(k,mu)` 上施加 FoG，再提取 ell=0,2。因此只保存两个多极本身并不证明遗漏了高阶多极对 FoG 的作用。若另写配置空间卷积基准，则需保留 Kaiser 的 ell=4，并在卷积之后重新投影；不能先只留 ξ0/ξ2 再随意卷积。

另一个已有近似是：P 使用真实格点 mu，而 ξ 先连续角积分、再径向离散。它应在 ξ-only 算子测试中量化，但它与 GSM 的物理闭合不是同一问题。GSM 不会自动修复有限盒角离散、binning 或缓存误差。

## 2. 纯线性 RSD 有什么更直接的配置空间写法？

取 `u=v/(aH)`，速度位移与坐标使用同一长度单位；Fourier 约定为 `exp(i k.x)`。无速度偏置时：

\[
\delta_g^s=\delta_g-\partial_z u_z,
\qquad\delta_g=B(k)\delta_m,
\qquad u_i(k)=i f k_i\delta_m(k)/k^2.
\]

所以

\[
\delta_g^s(k)=[B(k)+f\mu_k^2]\delta_m(k),
\qquad B(k)=b_1+f_{\rm NL}\,2\delta_c(b_1-p)\alpha(k).
\]

关闭 FoG 后，连续角向的三个多极必须为

\[
P_0=[B^2+2Bf/3+f^2/5]P_L,
\]
\[
P_2=[4Bf/3+4f^2/7]P_L,
\qquad P_4=8f^2P_L/35.
\]

这些式子这里只用于构建/验证 ξ，不是替换 P 拟合。随后 `xi_ell=i^ell integral k^2 P_ell j_ell/(2pi^2)`，或使用相同有限模式与 shell average。

### Hamilton 形式可作为独立线性检查，但不能错用常数 beta

在 fNL=0、常数 bias、连续各向同性极限，令 `beta=f/b1`，则

\[
\xi_0^s=(1+2\beta/3+\beta^2/5)\xi_g,
\]
\[
\xi_2^s=(4\beta/3+4\beta^2/7)(\xi_g-\bar\xi_g),
\]
\[
\xi_4^s=(8\beta^2/35)(\xi_g+5\bar\xi_g/2-7\bar{\bar\xi}_g/2),
\]

其中 `bar xi=3 r^-3 integral_0^r x^2 xi(x) dx`，`barbar xi=5 r^-5 integral_0^r x^4 xi(x) dx`。这是另一种表达同一个线性算子的方式，不会凭空改变相同输入下的限制。

本项目带有 k 依赖的 PNG bias。**不能对包含 PNG 的完整 ξ_g 继续使用常数 beta=f/b1。** 正确的混合形式是分别定义来自 `P_hh=B^2 P_L`、`P_hm=B P_L`、`P_mm=P_L` 的 j0 变换：

\[
\xi_0^s=\xi_{hh}+2f\xi_{hm}/3+f^2\xi_{mm}/5,
\]
\[
\xi_2^s=4f(\xi_{hm}-\bar\xi_{hm})/3
+4f^2(\xi_{mm}-\bar\xi_{mm})/7,
\]
\[
\xi_4^s=8f^2(\xi_{mm}+5\bar\xi_{mm}/2-7\bar{\bar\xi}_{mm}/2)/35.
\]

这也说明：**仅知道 tracer 的实空间 ξ，并不足以无条件确定 RSD；还需要密度—速度关系。** Gaussian 常数 bias 的情形可以借助额外假设补齐，PNG 的 k 依赖使这种区分尤其重要。

Hamilton 公式是连续角向表达；对有限立方格点的严格基准，应使用完整非零 modes，而不能把连续 kmin 截止宣称为精确有限盒等价。

## 3. 当前 FoG 在配置空间对应什么？

现有

\[
D(t)=[1+\sigma_s^2t^2/2]^{-2},\qquad t=k_z,
\]

对应一个与分离无关的 LOS 卷积核。令 `a=sigma_s/sqrt(2)`：

\[
F(w)=\frac{1+|w|/a}{4a}\exp(-|w|/a),
\qquad\int Fdw=1,
\qquad\operatorname{Var}(w)=4a^2=2\sigma_s^2.
\]

因此，数学上它相当于对 Kaiser 配置空间相关函数作固定 LOS 卷积：

\[
\xi_D^s(s_\perp,s_\parallel)
=\int dw\,F(w)\xi_K^s(s_\perp,s_\parallel-w).
\]

Gaussian 选项 `exp[-(kz sigma_s)^2]` 也对应方差 `2 sigma_s^2`。

这里 F 是附加、独立、位置无关平滑的数学核，**不是完整的、pair-weighted 的真实成对速度 PDF**。因而不能从旧 `sigma_s≈8` 推断 GSM 的总成对速度标准差也应为8；即使只匹配这个额外平滑的二阶矩，也应是 `sqrt(2)*sigma_s`。GSM 中相干流动的方差另有贡献。

在同一输入及数值定义下，直接 Fourier 加 FoG 和这个卷积应相符。若相符却仍不拟合数据，问题不是“傅里叶空间不允许加 RSD”，而是固定核的模型表达力/输入不足。GSM 允许 PDF 随真实分离变化，是另一个候选统计模型。

## 4. GSM 正确的接入方式

### 4.1 从实空间 pair conservation 开始

对于平行视线，令观测坐标为 `(s_perp,s_parallel)`，真实 LOS 分离为 y，`r=sqrt(s_perp^2+y^2)`，`mu_r=y/r`，相对位移 `Delta u_z=(v2z-v1z)/(aH)`：

\[
1+\xi^s(s_\perp,s_\parallel)
=\int dy\,[1+\xi^r(r)]
\mathcal P(\Delta u_z=s_\parallel-y\mid\mathbf r).
\]

这是由 pair conservation 得到的映射；PDF 必须是给定真实分离、以同一 tracer 的 pairs 加权的条件分布。GSM 用 Gaussian 近似此 PDF：

\[
1+\xi^s=\int dy\,
\frac{1+\xi^r(r)}{\sqrt{2\pi\Sigma_{12}^2(r,\mu_r)}}
\exp\!\left[-\frac{(s_\parallel-y-\mu_r v_{12}(r))^2}
{2\Sigma_{12}^2(r,\mu_r)}\right],
\]

\[
\Sigma_{12}^2(r,\mu_r)=\mu_r^2\sigma_r^2(r)
+(1-\mu_r^2)\sigma_t^2(r)+\sigma_{*,\rm pair}^2.
\]

本报告的速度符号是 `Delta u=u2-u1`，径向 infall 为负。`sigma_r^2` 是沿 pair 分离方向的中心方差；`sigma_t^2` 是垂直 pair 分离方向的**一个**分量的中心方差，不是两个分量之和，也不是固定 LOS 的方差。

### 4.2 必须避免的接线错误

- 输入的是**未加 Kaiser、未加 FoG 的实空间 ξ**。不能把当前 `FullDiscreteRSDModel.evaluate` 的 ξ_RSD 再送入 GSM。
- 不能在 GSM 输出上再乘 Kaiser；相干 RSD 已由 mean infall 和 variance gradients 生成。
- 必须积分 `1+xi`，最后减1。只卷积 xi 会丢失背景在速度梯度下产生的 RSD。
- 不能把 `integral kernel dy` 再归一化成1：PDF 在固定 r 下对位移归一化，不代表沿着积分路径 r(y) 的 kernel 积分为1；强制归一化会改变模型。
- 指数里的角度必须使用真实 `mu_r=y/r`；最终多极投影才使用观测 `mu_s=s_parallel/s`。
- 方差必须是中心方差；原始二阶矩应减掉均值平方。不能将均值平方重复加入，或把双横向分量漏除以2。
- 已使用实测完整成对方差时，第一版令 `sigma_*,pair=0`；否则容易重复加入同一速度散布。只有经检验的未解析额外项才作为独立 nuisance。
- 映射完成后，再按观测的 s 壳层和 mu bins 平均。不要先把输入 ξ 压成最终观测 bins，再作非线性 streaming。

这些是新 GSM 应遵守的条件，**不是说当前代码已经犯了所有这些错误**。当前源码并没有 GSM。

### 4.3 单位从现有 catalog 的 metadata 读取

当前 `task43_build_rsd_rawbox_catalog.py` 已保存 `POSITION_REAL`、`VELOCITY_KMS`、`RSD_DISPLACEMENT_MPC_H` 和 `velocity_conversion_kms_per_mpc_h`。

直接用 `u=v_kms/velocity_conversion_kms_per_mpc_h` 可与原 RSD mapping 对齐；pair 位移用两者之差。不要重复乘 h、a 或 (1+z)。以原 catalog 的 RSD_DISPLACEMENT 做逐对象桥接，比另猜 `aH` 数值更稳妥。

## 5. 为什么 GSM 必须恢复 Kaiser？其线性输入是什么？

GSM 所需的不是“ξ(r)+一个常数 sigma”，而是至少四个分离函数：

`xi_hh(r), v12(r), sigma_r^2(r), sigma_t^2(r)`。

无速度偏置、线性 matter 场、`P_hm=B(k)P_L(k)` 时，连续极限的领先阶输入为：

\[
\xi_{hh}^{(1)}(r)=\frac1{2\pi^2}\int dk\,k^2B(k)^2P_L(k)j_0(kr),
\]
\[
v_{12}^{(1)}(r)=-\frac f{\pi^2}\int dk\,kB(k)P_L(k)j_1(kr),
\]
\[
\Psi_\perp(r)=\frac{f^2}{2\pi^2}\int dk\,P_L(k)\frac{j_1(kr)}{kr},
\]
\[
\Psi_\parallel(r)=\frac{f^2}{2\pi^2}\int dk\,P_L(k)
[j_0(kr)-2j_1(kr)/(kr)],
\]
\[
\sigma_u^2=\frac{f^2}{6\pi^2}\int dk\,P_L(k),
\quad\sigma_r^{2,(1)}=2(\sigma_u^2-\Psi_\parallel),
\quad\sigma_t^{2,(1)}=2(\sigma_u^2-\Psi_\perp).
\]

这里 v12 是严格领先阶；精确 pair-weighted 定义包含除以 `1+xi_hh`。保留该分母是部分高阶重求和，须与同阶分子和中心方差处理配套，不能既除一次又在别处重复除、也不能把混合阶数模型称为精确线性。

把所有谱振幅乘 epsilon 后，展开 pair mapping：

\[
\xi_s^{(1)}(s_\perp,s_\parallel)
=\xi_r^{(1)}(s)
-\partial_{s_\parallel}m_1^{(1)}
+\tfrac12\partial_{s_\parallel}^2m_2^{(1)}.
\]

三项分别产生 `B^2 P_L`、`2 B f mu_k^2 P_L`、`f^2 mu_k^4 P_L`。**仅有平均 infall 仍不够；分离/角度依赖的二阶矩不可缺少。** 把方差设成常数，就丢失线性 f² 项。只卷积 xi 而不带背景1，在严格振幅展开中连领先的 infall 项也会丢失。

这与 Fisher 1995 及后续配置空间推导一致。本次测试还在同一有限 +/- mode 集上，通过数值微分独立验证了上式，而不是只对同一个 Fourier 公式做代数重写。

### 5.1 为什么用线性输入的 GSM 仍可能不同于线性 Kaiser？

因为将线性输入代入完整 Gaussian 指数并完成积分，会保留若干 `O(P_L^2)` 及更高阶映射项。它在有限振幅下不再是严格线性理论。Reid & White 区分了完整映射与线性展开；这种差异是 GSM 可检验的价值，而不是“发现另一种线性定律”。

不要把只用了线性输入的 GSM 称为已包含所有非线性动力学。理论输入可进一步用 CLPT-GSM 的一致 ξ/平均速度/方差预测替换，且 P 的既定拟合器不需要因此改动。优先验证 moment 模型，再谈高阶非Gaussian PDF；更复杂闭合未必能弥补错误的输入矩。

## 6. PNG 的接入不能只改 ξ(r)

沿用冻结 P 的 `B(k)` 和 `b_phi` 约定。在领先响应下：

- ξ_hh 使用 `B(k)^2 P_L`；
- mean infall 由 `P_hm=B(k)P_L` 决定；
- 无速度偏置时，领先速度 covariance 来自 matter velocity，不再乘 `B^2`。

若只给 ξ_hh 加 PNG、却将 mean infall 固定为 fNL=0，会遗漏 `2 f mu^2 Delta B P_L`，在线性极限就不能与基准 Kaiser-PNG 对齐。

上述做法只保留当前 scale-dependent-bias 的领先响应，不构成完整非Gaussian streaming。PNG 还能改变 pair-weighted 高阶矩/偏度；Gaussian 初始场的 CLPT 或 GSM 校验不能自动推广到所有非零 fNL。Schmidt 2010 与 Edgeworth streaming 文献支持保留这一边界。

建议先在 c000/fNL=0 建立映射验证，再对已有非零 PNG 样本和合成注入检验响应；不能只看 fNL=0 下的更好拟合，就宣布 PNG 模型已经有效。

## 7. 有限周期盒不能在改用 GSM 时丢掉

低模导致的 PNG 问题仍应按原周期盒定义处理，不能新增经验红外窗。

真正模式级的领先输入可写成（所有求和含完整 +/- 非零 modes）：

\[
\xi_r(\mathbf r)=V^{-1}\sum B_n^2P_n\cos(\mathbf k_n\cdot\mathbf r),
\]
\[
m_{1z}(\mathbf r)=-2fV^{-1}\sum B_nP_n\frac{k_{nz}}{k_n^2}
\sin(\mathbf k_n\cdot\mathbf r),
\]
\[
m_{2zz}(\mathbf r)=2f^2V^{-1}\sum P_n\frac{k_{nz}^2}{k_n^4}
[1-\cos(\mathbf k_n\cdot\mathbf r)].
\]

这些式子保留共同有限模，并严格恢复该有限模式下的线性 RSD。有限立方盒的方向依赖不是完全径向的；仅用四个径向函数的 GSM 仍是各向同性近似，必须量化，而不能再称为全离散精确闭合。

周期 LOS 映射应积分 y∈[-L/2,L/2)，并用 periodized PDF：`sum_m G(s_parallel-y+mL-mean; variance)`。在 s远小于L/2、位移尾远小于L 时，非周期 LOS 积分可能足够准确，但必须用尾部/镜像测试确认。**镜像尾很小不表示低 k 模式可以忽略**；二者是不同误差。

本区 `gsm_point` 是无限 LOS 参考积分，仅用于明确标识的数学测试；它本身没有实现生产用周期镜像闭合。`LinearModeMoments` 则保留有限模式，用于线性恒等式测试，不声称已经将两者变成完整生产有限盒 GSM。

## 8. 五组 ξ-only 判别实验；P(k) 始终冻结

### 实验1：线性/卷积等价性，不拟合参数

固定旧 cache、B(k)、f、同一数值域和壳层。先令 FoG=0，比较现有 ξ 与解析线性多极；再比较完整模式级线性 streaming 展开。测试 fNL=0 和非零 B(k)。最后打开旧 FoG，比较 Fourier 乘 D 与其解析 LOS 卷积。

目标：确认是实现/算子误差，还是模型类本身不足。数值验收建议固定旧 Cmean 作为度量，要求加密后的 `Delta xi^T Cmean^-1 Delta xi < 0.01`；这个阈值是预先提出的数值预算，不是已取得的 rawbox 结果。不能把一边完整格点、一边连续球面之间的差误称为积分不收敛。

### 实验2：实测输入的 GSM 映射闭合——最能隔离本轮问题

从同一原始 REAL catalog 获得 ξ_real(r) 和 pair-weighted v12、sigma_r²、sigma_t²。使用已保存 VELOCITY_KMS 和转换因子，不重测/修改 P。先选少量校准 phases，保留未用于校准的 phases。

将这组实测输入送入 GSM，预测相同 catalog 的 ξ_RSD0/2；额外方差第一版设0。对比完整 s,mu 面及最终测量 bins。若失败，进一步比较实测条件PDF积分与Gaussian PDF积分：前者通过而后者失败，才有较直接证据指向Gaussian闭合/高阶矩；两者都失败应先查单位、符号、归一化、表格范围和periodicity。

这一步是 oracle/诊断闭合，**不是纯理论 forward 验证**。使用实测输入当然可能较好，不能将其当作已获得独立理论预测能力。

### 实验3：固定 ξ_real，只更换速度矩预测

在实验2通过的同一 ξ_real 下，依次比较实测 velocity moments、领先线性 moments、一致的CLPT或受控扰动 moments。一次只改一个输入类别。

若实测矩通过、线性矩失败，瓶颈在矩预测/阶数，而不一定是GSM积分公式。若线性矩GSM已比旧固定FoG改善，仍须通过留出phases而非增加任意拟合函数。不要在这一阶段重新调整P基准的b1和sigma_s以追求对齐。

### 实验4：实空间输入的2×2拆分

对 ξ 输入（现有理论/实测）和 moments 输入（理论/实测）组成2×2比较。这样可以分辨是 ξ_real 形状、velocity moments，还是Gaussian映射本身造成残差。

现有实空间 ξ 已存在失配，因此GSM成功不能保证旧ξ_real正确。只有证据指向density输入时，才在**新ξ分支内部**增加经独立检验的BAO/实空间模型；不要求改变既定P拟合模型。这个分支可以拥有与P模板不同的准线性重求和，但必须保留共同大尺度b1与PNG响应定义。

### 实验5：参数和PNG验证，最后才运行长链

先固定fNL=0和P参考参数作预测检验，再释放ξ自己的b1、fNL及至多一个明确的额外成对方差。报告sigma_*,pair而非沿用旧sigma_s名称。检查fNL正/负注入、尺度切割和先验，按配对phases估计参数差，不能将P/ξ当独立观测相加误差。

可以报告新ξ模型与冻结P参考的b1/fNL相容性；不要要求新总pair variance与旧FoG sigma_s数值相等。暂不构造共享同一sigma参数的新joint。

## 9. 执行 agent 的具体接入要求

1. 新增 `XiLinearRSDModel` 与 `XiGSMModel`/独立 driver；不要修改原 P 类及其读取的旧 `FullDiscreteRSDModel` cache。新输出另建目录，附旧P输入/代码哈希，证明冻结参考未变。
2. GSM 的输入 provider 至少返回 ξ_real、signed mean、central variance，并记录是线性、扰动还是实测矩。不得从已有 ξ_RSD 的三个拟合参数推断完整 moments。
3. pair moments 必须按 REAL separation、同一 tracer 选择、相同 pair weights 累积。横向二阶矩按两个分量之和的一半处理；中心化和噪声预算须明确。大catalog不要用无约束N²循环；对受控pair抽样或专门计数器报告采样误差。
4. 表格必须覆盖 LOS 积分真正访问的 r，而非只覆盖最终 smin~smax；插值不要产生负方差或 `1+xi<0`。禁止静默clip来掩盖不合理输入。分离域/LOS尾不足应报错或扩大测量。
5. 用观测 `mu_s` 和体积权重投影，匹配 FCFC 的 mu-bin 定义。保存不同积分分辨率、最大位移和周期镜像测试；`f=0` 并不代表关闭旧非零FoG，零RSD测试需所有速度项都为0。
6. **不能复用 FastRSDModel 的六项 b1/fNL 多项式代理。** GSM 的指数、方差和pair normalization引入新的非线性参数依赖。先用直算验证，再建立另一个有独立留出点的surrogate。
7. 前几项A/B用固定C作为比较度量，避免边换mean边调covariance。GSM若成为正式模型，ξ covariance及参数覆盖率另行验证；oracle输入与ξ_RSD同源，闭合残差的不确定性要包含这种相关性。x25高维残差仍不能直接求经验逆矩阵。
8. 现有joint权重问题保留为后续联合分析前置条件，但不再抢占本轮“ξ如何正确加RSD”的主问题。

## 10. 本次实际数值验证

运行：

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 \
  python GPT5.6-respose/xi_rsd_followup/test_xi_rsd_reference.py
```

本次结果为11 tests、0 failures、0 errors，NumPy 2.3.5、SciPy 1.17.0。`validation.json`记录脚本SHA256。

- 用数值微分检验有限模式 `xi_r - d m1 + (1/2)d²m2 = Kaiser`，最大绝对差 `1.41494e-10`。
- 将谱振幅epsilon依次减半，`|xi_GSM/epsilon-xi_Kaiser(epsilon=1)|`依次约为 `3.8590e-5, 1.9260e-5, 9.6214e-6, 4.8085e-6`，符合GSM与线性Kaiser只从更高阶开始不同。
- 故意将pair variance改成常数，数值结果恢复了“real+infall”，但缺少应有的f²项；故意只stream ξ，则领先速度贡献都丢失。这是针对错误接法的负控制，不是对现有代码的错误指控。
- 旧Lorentzian的LOS核归一化为1；sigma_s=8时方差为128，而非64；其Fourier变换与原D的最大绝对差 `3.33e-16`。
- 验证了PNG形状bias在k内部的线性投影、GSM正负LOS对称性、积分尾加密、真实mu和单横向方差约定、零速度和壳平均、非法方差拒绝。

这些测试没有验证真实rawbox的拟合改进幅度。下一份科学结果应来自上述实验2/3的实测矩闭合与留出检验，而不是把数学测试通过写成“GSM解决了sigma张力”。

## 11. 一手文献与源码

论文用于确认公式与适用边界，不以别的样本精度替代本仓库验证。

- Fisher (1995), *On the Validity of the Streaming Model for the Redshift-Space Correlation Function in the Linear Regime*: https://arxiv.org/abs/astro-ph/9412081 。mean streaming与分离依赖的dispersion如何共同恢复线性RSD。
- Reid & White (2011), *Towards an accurate model of the redshift space clustering of halos in the quasilinear regime*, §§2.2,2.5,4,5，特别是式(7)–(18)、(25)、(26)：https://arxiv.org/abs/1105.4165 ；期刊全文 https://academic.oup.com/mnras/article/417/3/1913/1089713 。GSM与严格线性、Gaussian场精确结果的区别；实测输入闭合和理论矩预测需分开。
- Jeong, Dai, Kamionkowski & Szalay (2015), *The redshift-space galaxy two-point correlation function and baryon acoustic oscillations*, §2.2：https://academic.oup.com/mnras/article/449/3/3312/2893059 。明确展示二阶矩导数产生f²项。
- Wang, Reid & White (2014), *An analytic model for redshift-space distortions*: https://arxiv.org/abs/1306.1804 。CLPT预测ξ和速度矩，再接GSM，而非只对ξ加Gaussian平滑。
- Scoccimarro (2004), *Redshift-Space Distortions, Pairwise Velocities and Nonlinearities*: https://arxiv.org/abs/astro-ph/0407214 。完整pairwise映射与phenomenological damping/PDF解释边界。
- Kuruvilla & Porciani (2018), *On the streaming model for redshift-space distortions*: https://academic.oup.com/mnras/article/479/2/2256/5042948 。pair ordering、条件PDF及Gaussian cumulant闭合。
- Schmidt (2010), *Large-scale Velocities and Primordial Non-Gaussianity*: https://arxiv.org/abs/1005.4063 。PNG的尺度依赖bias及成对速度分布高阶响应。
- Uhlemann, Kopp & Haugg (2015), *Edgeworth streaming model for redshift space distortions*: https://arxiv.org/abs/1503.08837 。何时需要高于Gaussian的闭合，以及输入矩准确性的限制。

原始源码均固定到以下根路径，避免审阅期间main变化：
`https://github.com/ZhuoyangLi329/PNG-2PCF/tree/e6f217b3e4890a22ac866b0a840ac9d6ae00056a/source/project/codes/task43/`

重点文件：`task43_rsd_model.py`、`task43_fit_rsd_rawbox_x25.py`、`task43_fit_rsd_rawbox_pk0_vs_xi0_smin50.py`、`task43_build_rsd_rawbox_catalog.py`、`task43_measure_rsd_rawbox_xi_fcfc.py`。

**最终建议：冻结P作为既定参考；为ξ新增正确的pair-conserving GSM候选，先验证线性极限与实测moment闭合，再验证理论moment和PNG响应。不要将GSM包装成对Kaiser线性理论的纠错，也不要继续把两个不同定义的sigma强制等同。**
