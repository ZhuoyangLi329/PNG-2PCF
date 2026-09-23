# 完整 RIC 的 shot 与数值审计笔记

## 文献和作者实现

- de Mattia & Ruhlmann-Kleider 2019，§3.3、§4.2：<https://arxiv.org/html/1904.08851v3#S3.SS3>。cross/auto shot 各以 correlated pair 总权重归一化；模型加实测 Poisson 幅度，另保留常数 galaxy stochastic nuisance。文中总 data+random shot 套用径向核依赖高 random density 近似，脚注明确说明。
- Tamone et al. eBOSS ELG，§5.2：<https://arxiv.org/html/2007.09009#S5.SS2>。在 mu 中除 RR 后再投影；shot 也必须进入同样的 LS 响应。
- 作者固定 commit 的 `Real3PCFBinnedShotNoise` 与 `Real4PCFBinnedShotNoise` 已按独立 A/B 目录重写成低层两点计数桥。无新的可调 RIC 幅度，原 sn0 维持冻结模型的常数 stochastic nuisance 约定；这不是声称所有非 Poisson 随机性已经建模。

## 生产估计量的关键事实

生产 `task43_measure_rsd_lightcone_xi.py` 显式调用 `count2(data_particles,data_particles)`、`count2(random_particles,random_particles)`，即两份 positional arguments。

已直接核对当前生产环境的 `cucount.types.count2` 和 `compute_norm2`：只有 `len(particles)==1` 才视作 autocorr 并在 norm 中减 sum(w²)。这里实际 norm 是 W²，s>=30 的分离选择自行排除几何 self-pair。不能未经核对套用 N(N-1) 归一化的常见公式。

源路径：`/global/common/software/desi/users/adematti/perlmutter/cosmodesiconda/20260321-1.0.0/code/cucount/main/lib/python3.12/site-packages/cucount/types.py`。

halo random 使用 `rng.choice(data_z,replace=True)`，phi 在 [0,pi/2] 均匀、cos(theta) 在 [0,1] 均匀，FKP 是生产 dz=.01 分段常数。P 使用全部 25 倍 random；xi 将 25 个 N_R=N_D blocks 分别算 LS 后平均。

## 有限 random 的 leading-order shot 分工

条件于 data 的径向分布重新抽样 random，data 噪声受径向投影 Qr；random 自身的条件抽样噪声仅受总体归一化 Qg。归一化 weighted counts 的噪声协方差在 leading 1/N 阶为

    Qr D Qr^T / Ndata + Qg D Qg^T / Nrandom.

因此 shot-subtracted P 的有限分离项使用 data amplitude × radial shot template，加 random amplitude × global shot template。对于 xi 也分别处理，random amplitude 要按每个真实 block 的总权重和平方权重计算。

- P data amplitude = sum(w_D²)/I2。
- P random amplitude = alpha² sum(w_R²)/I2，alpha=sum(w_D)/sum(w_R)。两者之和已与生产 `shotnoise_ell0` 核对。
- xi data amplitude（概率矩单位）=sum(w_D²)/sum(w_D)²。
- xi random amplitude（概率矩单位）=mean_blocks[sum(w_R,b²)/sum(w_R,b)²]。
- ξ 响应的转换使用与聚类同一 40-mu-bin RR 除法；完整 Qr 已含 data 的 GIC，不能再叠旧 scalar Cw。

`task432_full_ric_bootstrap_shot.py` 独立生成 32768 次 multinomial data+实际径向 bootstrap random，使用非均匀径向权重、实际局部 P LOS 和 midpoint LS 分母。N_R/N_D=1、25 的八个 z-score 最大绝对值 1.941，通过 5-sigma gate。它还量化实际随机 RR 分母相对固定 RR 的均值改变。该检验是噪声分工闭合，不是实际 halo selection 的所有误差验证。

## 目前数值检查

- 作者 cross/auto 原始核与独立 dense-Q 计算匹配约 3e-16。
- shot 作者计数与独立逐对求和匹配 <1e-12；k=0 Poisson 消除误差 <7e-14。
- 解析 Bessel 原函数和精确 P-band Hankel 已通过独立 scipy.quad 校验。
- P 外层和内层分别用 8/16 点积分，三个测试点的完整 covariance 误差约 1e-21。
- xi 内层 shell 平均相对仅取中心，误差约 0.00066–0.00081（C_single 二次型）。正式响应采用 shell 平均。
- 随机子集 N8192/dchi10 -> N16384/dchi4 的 ξ 差异二次型约 .03；后者 -> N32768/dchi2 约 .011–.013。不能只凭此声称已达到最终误差标准，正在独立 scrambled Sobol 取样验证。
- 当前完整 74 维固定参数模型变化二次型约 .20–.35，包含正确分工的固定 shot；这是模型变化大小，不是拟合 χ² 或拟合质量结论。

正式拟合仍待几何收敛、P 输入 LOS、生产 RR/phase 几何桥、物理网格插值误差检查。不要提前报告 joint b1 已改善或已进入两个单项之间。

## 2026-09-22 更新：既有 sn0 与后验精度

已有 sn0 定义为 underlying constant P 的 stochastic 项，单位仍为 sn0×1e4，喵～与Poisson不同，它的接触协方差权重为(nbar w)^2 P，因此在按nbar抽取的积分点上需将w²替换为nbar w²，再按连续I2/实测I2=1.01061393归一化，喵～

sn0=1时若忽略xi的RIC响应，其在完整C_single度量的误差二次型约0.025，超出0.01门限；sn0=.5时约0.00625，喵～因此正式xi-only使用既有sn0，保持原[-1,1]先验，没有增加独立RIC幅度，喵～

前置几何检查在四个代表点均通过，但首轮30k链的后验检查在xi-only 99% b1分位点（fNL=-26.4591,b1=2.671575）得到独立xi核差二次型0.01094589，喵～没有放宽门限，增加seed4322291与4322333两套N32768积分；最终xi使用四scramble平均，并将两套独立双核均值的差异重新要求<0.01，喵～所有已完成MCMC链保留，最新正式入口以formal_mean4的postflight.json为准，喵～

P采用原两个独立scramble平均；后验审计另检查P核差、sigma spline与直接理论的一致性，喵～最终还从冻结1000×74 mock stack独立重建covariance，并核对P/xi独立接口与joint对应分量相同，喵～
