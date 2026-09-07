# 本轮核验结论与执行交接

本文件补充本目录已有报告，不覆盖审阅期间更新的同名代码。对应独立本地包 `GPT5.6-respose-verified.zip`，各文件 SHA256 与实跑结果见 [LOCAL_VERIFIED_2872ea42.json](LOCAL_VERIFIED_2872ea42.json)。完整本地报告包含证据矩阵、最简模型、五组判别实验及逐函数修改清单。

## 一、核心结论

**确定的联合 covariance 错误、P/xi 离散算子不一致、独立均值形状失败，是三个不同层面的问题。** 不能用“rawbox没有survey窗口”推导“剩下的都只能是2PCF物理模型错误”，也不能用修好joint宣称独立xi形状已经通过。

尤其要区分 P0 与 P02。以下来自原始 independent marginal 审计，不是本次新拟合；sigma68 是分位区间半宽而非高斯近似的保证。

| 数据向量 | b1中位数 | sigma_s中位数 | sigma68(sigma_s) | chi²_mean/dof |
|---|---:|---:|---:|---:|
| P0，free sn0 | 2.5327 | 6.4891 | 3.9597 | 10.9464/12 |
| xi0，s>=50 | 2.5532 | 8.8433 | 1.0149 | 162.9797/27 |
| P0+P2 | 2.5760 | 0.9844 | 0.7472 | 73.8786/28 |
| xi0@s>=50 + xi2@s>=80 | 2.5425 | 7.7905 | 0.7389 | 297.2792/54 |

P0 的 sigma_s MAP 约8.25，xi0约8.77，P0的68%区间约[2.26,10.18]。**明显的零边界行为来自加入P2以后**：P02的MAP约4.24e-8，不能把约0.98的边际中位数当成对非零速度色散的高斯检测。

实空间最简控制没有 sigma_s 参数。P0与xi0(s>=50)的b1中位数约2.6501、2.5416，均值chi²/dof约45.23/14、331.28/28，说明RSD映射不是唯一原因。xi的s>=120控制缓解形状问题，但b1分位区间约[1.943,2.852]，非常宽；不能说偏置已精确闭合。

## 二、已确认的实现问题

### 1. joint 的原始量纲 floor

仓库既有 `review_data/joint_covariance_reproduction.json` 给出 cross=0 时lambda_max约4.27e9、floor约4.27e-5，全部57个xi方向被抬升，xi单-bin sigma中位数增大15.4倍。这项原数组复现不是本次新运行；本次独立合成测试确认该算法不满足单位不变性。

cross=0且边块不变时，任何theta上必须有chi²_joint=chi²_P+chi²_xi，因此joint最小值不能低于独立最小值之和。现存单极下界6.9570而naive joint仅0.6863；四向下界14.8463而naive joint仅3.1418。旧joint“仅增益0.5%/0.9%”不能作为物理结论。

修复：在相关矩阵标准化后做Cholesky，MAP/MCMC共用固定metric；不自动floor/pinv。保持原边块，检查白化cross的最大奇异值<=1。非SPD先查cross/算子，真实线性冗余则显式定义支持子空间。现有其他normalized precision的n_below_floor=0，不能声称它们也已触发同样错误。

### 2. CachedRebin后按中心分配窄P bins，模式归属改变

原源码顺序为 `kf=2*np.pi/L; dk=.1*kf`。本次按相同顺序枚举格点：

| P bin | 精确模式数 | 聚合中心归类数 | 差别 |
|---|---:|---:|---:|
| [0.037,0.039) | 1262 | 1094 | -13.31% |
| [0.061,0.063) | 2994 | 2754 | -8.02% |
| [0.085,0.087) | 6000 | 5352 | -10.80% |

`task43_rsd_rawbox_joint_4way.py::{pk_pole_cov,cross_block}` 的分母仍使用测量精确Nmodes，但分子按聚合后的k_eff归类。这是实际算法不匹配；不是已经测得的参数偏移。修复应先按测量P-bin边缘选择原模式，再在bin内聚合，而不是加epsilon。另见 [NUMERICAL_CORRECTION.md](NUMERICAL_CORRECTION.md)。

### 3. real-space 更换C后没有在最终metric下refit

`task43_rsd_rawbox_realspace_{check,mcmc,pk_check}.py` 的控制先拟合，再更新covariance，但继续保存先前MAP预测；MCMC使用新metric。应在最终冻结C下重新多起点优化，并分别记录MAP、链内最佳点、median。它可能有贡献，但尚未证明能解释全部b1偏移。

## 三、已量化的数值线索，尚未量化其参数影响

P模型使用实际离散mu；xi模型使用连续mu积分后再径向离散。首个[0.003,0.005) bin含18模式，<mu²>=1/3、<mu⁴>=2/9、<mu⁶>=1/6，而后两者的连续值为1/5、1/7。常数径向Plin、无FoG的toy中，P2/Plin的离散值为5bf/3+25f²/36，连续值为4bf/3+4f²/7。b=2.55、f=.81时分别3.898125与3.128914；这是定位toy，不是整个测量P2有24.6%偏差。

统一xi算子应使用逐模式 `(2ell+1)/V * sum[P(k,mu) Lell(mu) Kell(k,s)]`。可先做低k精确角修正并扫描k_switch，不能在kmax=3暴力枚举完整立方体。真空间各向同性xi0不受该角平均差异影响，所以此项不能解释其全部偏移。

Lorentzian角矩有解析表达式：

\[
I_n^{(m)}(x)=\int_0^1\frac{\mu^{2n}}{(1+x^2\mu^2/2)^m}d\mu
=\frac{{}_2F_1(m,n+1/2;n+3/2;-x^2/2)}{2n+1}.
\]

signal用m=2、signal² covariance用m=4。本次实算GL64在x=k*sigma=24与90时对I0分别低估0.8613%和64.97%。kmax=3、sigma=8/30恰对应这些值。**这是角矩误差，不是xi总误差**；应通过真实谱与核评估delta_xi^T C_mean^-1 delta_xi及参数导数误差。

ell2体积壳核可用原函数 `3Si(x)+x cos(x)-4sin(x)` 的端点差，小x用级数；避免径向固定节点求积。解析角矩、壳核与自适应积分的最大误差约1.14e-13（相对）和1.54e-14（绝对）。

当前angular_totals是[-1,1]积分，不是平均，常数已给2；没有理由再全体乘2。正确P2模式平均系数是5，当前实际代码已使用5，旧2.5注释不能作为重复修正依据。

## 四、物理模型该怎样改

P的16个bins有空档；xi(s>=50)对应宽k核且理论积分到3。它们不是严格双射，smin也不是硬kmax。遗漏形状delta_d产生的参数偏移近似为 `(J^T C^-1 J)^-1 J^T C^-1 delta_d`；不同核与权重会得到不同有效b1。

最值得验证的是：当前单一FoG同时吸收了BAO平滑、非线性density–velocity结构和偏置形状。xi的sigma~8可能在补偿缺少的BAO阻尼；P2可能需要当前只会抑制功率的FoG无法提供的形状，所以sigma被推到零。此解释是待检验假设，不是已证明的主因。

最小真空间诊断模板：

\[
P_{hh}^r=A(k)^2[P_{nw}+e^{-k^2\Sigma^2/2}P_w]
+c_\nabla k^2P_{lin}+N_0.
\]

先固定fNL=0，单独测试N0，再Sigma，最后一个有符号k²方向，不同时全部释放。Sigma是BAO位移平滑，不等同FoG速度。RSD再测试一个signed k²mu²方向，或独立冻结的P_delta_delta、P_delta_theta、P_theta_theta，分开跑；不要把signed counterterm叫作负速度方差。

sigma=0处对sigma的导数为零，Fisher可能奇异。profile可用lambda=sigma²；采样若保留uniform-sigma prior，则lambda先验必须按Jacobian转换，不能悄悄换先验。

P已减Poisson不保证残余N0=0。常数随机项理想完整变换是contact，去零模周期和还涉及- N0/V及实际pair normalization；不能随意在xi(s>0)加自由常数，也不能把低k的N0硬外推到3制造振铃。xi mean不含contact，不代表xi covariance可忽略随机功率。

这些新增方向只作最小辨别，不宣称等于完整one-loop/TNS模型。需要共同高k completion、UV敏感性检查、留出预测和fNL注入回收。

## 五、执行顺序：五组最小实验

1. 冻结数据/模型，修metric、cross=0恒等式、单位不变性、cross正定性，最终C下重新MAP。未通过前不讨论joint。
2. 固定理论参数，逐项A/B比较membership、离散角、解析角矩和shell核。建议预注册累计数值delta chi²_mean<0.01、参数导数投影偏移<0.1sigma_mean的门，不只看相对曲线误差。
3. 真空间fNL=0控制，按N0、BAO平滑、k²偏置的顺序辨别，预注册kmax与smin扫描；用未参与选择的尺度/相位检验形状。
4. RSD分别跟踪P0→P02、xi0→xi02，画sigma² profile和分尺度残差；测试一个signed角向counterterm或独立density–velocity模板，判断零边界来源。
5. 用配对相位校准参数差与条件残差 `r_xi|P=r_xi-C_xiP C_PP^-1 r_P`。25相sample C秩最多24，不能直接逆57/89维；使用预注册低维投影或经独立检验的shrinkage。确认coverage和均值形状后才恢复正式joint信息增益。

原始关键证据位于 `source/project/outputs/task43_outputs/rsd_validation/rawbox/` 下的 `joint_pkxi/audits/`、`joint_p02xi02/audits/`、`realspace_check/audits/` 与 `closure/`，以及 `review_data/joint_covariance_reproduction.json`。文献背景：Wands–Slosar arXiv:0902.1084、Grieb arXiv:1509.04293、Taruya–Nishimichi–Saito arXiv:1006.0699、Hartlap astro-ph/0608064、Percival arXiv:1312.4841。

## 验证边界

本次14项独立数值/合成测试全部通过，含50,000次独立复模式功率抽样和合成缓存driver测试。**没有在原NPZ上重跑审计，没有catalog/FCFC/jaxpower重测，没有原始或改进物理模型MCMC，也没有仓库PDF逐页视觉核验。** 原拟合数字来自原审计JSON；15.4倍是仓库已有原数组复现。不能把本次数值测试当作物理问题已经解决。
