# Task432 hybrid + GIC：当前正式结果

本轮补齐此前 hybrid lightcone 2PCF 模型遗漏的 GIC。正式模型采用 9.18 的各向同性 formal-GIC 常数处方，由当前 hybrid 理论计算窗口均值；这不是一般各向异性 cross-window 的完整算子。

## 固定的数据、协方差和参数口径

- 9.18 数据：25 个 Abacus phase 的均值，0.4<z_obs<0.8；P0 13 点 + P2 9 点，xi0/xi2 各 26 点，joint 共 74 点。
- kmax=0.08 h/Mpc，P2 kmin=0.015 h/Mpc；xi 中心 55–345 Mpc/h，排除 80≤s<120 的拟合 bins。
- jaxpower 与 EZmock-1000 使用各自此前冻结的 C_single，均不除以 25，保留完整 P–xi 交叉协方差；1000 mocks 的编号为 0…999。
- P-only 模型、数据、协方差和链复用原结果；2PCF 与 joint 独立重采样。
- 2PCF 自由参数 fNL、b1；joint 另有仅用于 P 的 sigma_s_P、sn0；GIC 不增加自由参数。
- 先验 fNL∈[-500,500]，b1∈[0.5,5]，sigma_s_P∈[0,30]，sn0∈[-1,1]。
- EZmock：Hartlap 修正用于 Gaussian 似然，Percival 因子用于区间和 contour 的一致缩放；原始链保留，展示样本不冒充独立采样结果。

## GIC 实现和适用范围

每个 phase 的原 RR pair_prob 独立归一化后，对 25 个 phase 等权平均；在所有窗口距离上以 r² 壳权重积分 hybrid xi0，得到 C_W(fNL,b1)，再令 xi0_pred=xi0_hybrid−C_W，xi2_pred=xi2_hybrid。拟合中的 BAO mask 不进入窗口积分。

GSM 内部累积量 r 表从原来的 <600 Mpc/h 延长至约 3996 Mpc/h，保留原有节点；窗口距离覆盖 0.5–3500 Mpc/h。约 83.5% 的随机点对权重在 600 Mpc/h 以上，这一比例不是 GIC 数值贡献比例。原拟合区间仍在 bin 中心求值。

窗口几何、权重和常数修正处方沿用 9.18；C_W 的理论内容与当前连续 hybrid 一致，不混入旧 Kaiser/FoG 的 GIC 数值或独立自由幅度。旧窗口核的离散 j0 重建作为归一化校验。当前 hybrid 原有的 plin 延拓、alpha 在输入 k 边界外的常数延拓以及 FFT Tukey taper 均保留；本轮不等同于重新实现 FullDiscrete 模型。

## 结果

下表中心为后验中位数，括号为 68% 区间；ML 单独列出，避免与边缘后验混淆。PDF 中 fNL 注释沿用 9.18 风格，以 ML 为中心、q16/q84 为区间端点。

| covariance | probe | b1 median [q16, q84] | b1 ML | fNL median [q16, q84] | chi2 / dof |
|---|---|---|---|---|---|
| jaxpower | joint | 2.383411 [2.320764, 2.446017] | 2.383227 | -6.367 [-32.145, 16.901] | 6.4222 / 70 |
| jaxpower | p02 | 2.412972 [2.316992, 2.509841] | 2.417747 | -13.339 [-47.045, 19.634] | 1.0969 / 18 |
| jaxpower | xi02 | 2.428992 [2.324372, 2.531865] | 2.433224 | -6.052 [-33.757, 20.557] | 5.1135 / 50 |
| ezmock1000 | joint | 2.382067 [2.313726, 2.450103] | 2.383164 | -3.384 [-29.372, 20.915] | 4.1798 / 70 |
| ezmock1000 | p02 | 2.414844 [2.316627, 2.513658] | 2.419496 | -13.098 [-45.872, 19.464] | 1.1106 / 18 |
| ezmock1000 | xi02 | 2.417981 [2.305109, 2.528265] | 2.423165 | -3.305 [-30.875, 23.835] | 3.0732 / 50 |

## joint b1 检查

- jaxpower：中位数是否居于 P 与 xi 之间：False；ML 是否居中：False；joint−P 中位数=-0.029561，joint−xi=-0.045581。
- ezmock1000：中位数是否居于 P 与 xi 之间：False；ML 是否居中：False；joint−P 中位数=-0.032776，joint−xi=-0.035913。

P 和 xi 有相关性，joint 中心不必在两条边缘后验中心之间；位置判断是结果报告，不作为迫使模型或采样向某个值移动的条件。

## 数值验证与采样

- RR 核重建最大绝对误差 5.551e-17。
- 上游标量与批量实现最大差异 1.249e-16。
- 加密 GIC 壳积分、streaming 积分、角积分和 r 表的最大协方差加权模型误差 2.036e-06；测试包含先验角点。
- 后验采样：64 walkers，至少 30000 steps，burn-in 5000；Rhat<1.01，postburn length>50 tau，前后半链中位数差<0.1 sigma。
- 后验区域以直接 hybrid+GIC 在随机点、参数尾部、ML 点复核 emulator；直接优化 ML 后，再用加倍积分精度检查。
- 积分精度与插值精度的检验值是模型差异的 δmᵀC⁻¹δm，不是拟合残差的 chi2；两者在 JSON 中分别记录。
- CPU-only 任务在 login11、taskset 6–13 内运行；网格最多 6 个单线程进程，MCMC 最多 4 个单线程进程。

## 推荐入口与溯源

代码：`codes/task432/task432_hybrid_gic.py`、`codes/task432/task432_hybrid_gic_0918_contract.py`。
当前 HybridGIC.evaluate() 与 prediction() 均包含 GIC；拟合完成后增加 evaluate 接口覆盖，避免继承旧类时漏掉 GIC，api_audit.json 已确认预测与冻结直接计算在机器精度内一致。执行时源代码快照与当前接口补全快照分别保留。
共享窗口审计：`outputs/task43_outputs/rsd_validation/task432_model_repair/hybrid_gic_window/window_audit.json`。
当前结果索引：`outputs/task43_outputs/rsd_validation/task432_model_repair/hybrid_gic_current_results.json`。

执行顺序：共享 window、grid 只需一次；每个 covariance 运行 prepare → xi02 / joint → postflight → plot。已完成的目录禁止重复覆盖，脚本主动拒绝覆盖正式产物。

旧版遗漏 GIC 的链和 PDF 保留，增加替代状态标记；P-only 通过显式软链接复用，未重复拷贝大型链。本轮无大批量移动、删除。
临时 smoke 记录在共享目录 debug/，网格检查点和执行日志在 logs/，源代码快照在 provenance/；推荐从当前索引开始读取，而非递归扫描全部历史输出。
