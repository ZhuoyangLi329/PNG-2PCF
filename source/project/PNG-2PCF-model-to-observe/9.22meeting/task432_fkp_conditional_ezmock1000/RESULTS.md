# Full-RIC 后的 FKP 与条件模式探索

本轮结论：在预定义的关键低维模式中，未发现 EZmock1000 协方差形状显著失配的证据；三个预选 halo phase 的 FKP 配对也没有把 joint b1 拉回两侧之间。已有结果更支持继续定位平均模型/观测算子的相干残差，不能把这一结论写成“完整 covariance 已验证正确”或“FKP 在所有 phase 下严格为零”。

## 条件模式校准

固定 fiducial (fNL,b1,sigma_P,sn0)=(0,2.42,2.2,0.08)，仅使用模型导数与原 EZ covariance 定义模式，不按 halo 残差挑选。前两项为 nuisance 局部边缘化后的 joint-minus-P、joint-minus-xi b1 响应；再加入 joint b1 和两个条件 canonical 模式，并按原 C 正交化。

25 个 halo phase 在五个模式上的方差/EZ预测比：0.5912, 0.6176, 1.3233, 0.9407, 1.3428。它们都落在每次抽25个EZ得到的95%波动范围。包含模式间相关的完整 covariance-shape 检验，上尾重采样频率分别为2维 0.1712、5维 0.7606；因此不能仅凭交叉相关推动中心位移就断言协方差有错。

两个局部线性 b1 对比的实际均值为 -0.03507、-0.03945。EZ 25-realization均值噪声对二维位移统计量的重采样尾率约 0.0159，说明该方向的平均残差值得继续查。它依赖固定切线模型、选定covariance与phase独立假设，不是当前非线性/有界后验的精确tension p值，更不能转换成确定的几sigma。正式拟合始终用C_single；除25仅用于均值噪声诊断。

## FKP 配对实验

预选 ph000、ph012、ph024；同一 data 与 random 位置、同一 mesh/LOS/bin/mask，仅将自身nbar得到的FKP换成其余24phase平均nbar得到的FKP。两侧 data/random 同步换权，分别重算实际 I2、Poisson shot、P smooth window、完整 cross+auto IC，以及既有sn0的IC响应。FKP参考仍有有限24phase误差，不是假定已知的无噪声选择函数。

P用相同4N random；xi每侧用相同两个N-sized blocks，先算每块LS再平均。xi保留原有中心值动力学约定；每个权重分支重算RR，并用于IC响应的LS除法。它不是生产25-block重新测量，也没有替换正式数据。

25个phase的计数加权FKP相对扰动RMS范围为 1.337%—2.419%；三个pilot覆盖典型幅度并包含较大扰动phase，但这不能代替完整25phase平均。

扣除权重对应的普通窗口和full-IC模型变化后，三phase平均局部响应：

- joint-minus-P b1 对比：-0.001321，phase SEM=0.001635。
- joint-minus-xi b1 对比：-0.002975，phase SEM=0.000157。
- P、xi、joint 各自b1局部响应：+0.000773, +0.002428, -0.000548。

phase之间共享参考目录，以上SEM只是pilot散布描述；有限random敏感性另外记录在每个phase的逐block响应中，不能将三个phase或两个blocks当作正式协方差校准。

用这一固定fiducial模型增量做一次敏感性优化，得到下表；这不是新权重下完整25phase数据与新协方差的正式拟合，也不是新的MCMC后验。

| 拟合 | 原 full-RIC MAP b1 | 固定增量诊断 MAP b1 | 变化 |
| --- | ---: | ---: | ---: |
| P-only | 2.413459 | 2.414221 | +0.000762 |
| xi-only | 2.422866 | 2.425208 | +0.002342 |
| joint | 2.378192 | 2.377713 | -0.000480 |

诊断 joint 是否位于两侧之间：否。该pilot没有缓解主要中心位移，不支持把“逐phase自估FKP”列为目前的首要解释。joint的raw χ²从4.00615变为4.03504，拟合质量基本没有区别。xi-only的既有sn0仍在原先验上界，局部线性响应和有界MAP只作敏感性对照。

## 数值验证与保留问题

原始74维数据、EZ1000 stack/covariance、正式full-RIC engine的hash均未变；模式正交、共同参数方向消除和独立本地重算均通过。P的Poisson项逐分支独立核对解析权重公式，CPU xi与原生产同blocks的桥接通过。

每phase两套独立积分检查权重差分；ph000保留16384点加密参照，其余8192点用于配对增量。最大独立IC差分二次型为 0.000350465，最大b1对比响应差为 0.000353732，分别低于事先设置的0.001/0.001诊断门限。这里只保证FKP增量对约0.04的目标位移有足够判别力，不升级为新绝对模型的全后验验证。

当前证据没有触发预留的EZ random×FKP配对扩展分支，因此本轮没有重测匹配流程的新covariance。固定公共random与halo shuffled-random流程的完整covariance等价性仍未证明；25个halo对低维covariance的检验能力有限。

下一步优先考虑真实halo宽带/尺度依赖随机项、P与xi的平均物理模型及lightcone随红移的有效权重。现成halo-matter谱在本轮有界检索中未找到；公开z=0.5和0.8目录包含field_rv_A和halo_rv_A，可以成为后续独立bias锚定的入口。目录存在不等于粒子完整性已认证；还需处理实际lightcone的红移演化与同一halo选择。

## 入口与文件

- `task432_fullRIC_FKP_conditional_diagnostics.pdf`：四页诊断，科学图仅PDF。
- `conditional_modes.json/.npz`、`lowdim_covariance_calibration.json`：预定义模式、25halo/1000EZ低维校准。
- `paired_fkp_results.json/.npz`：数据/模型配对增量、逐phase响应、诊断MAP。
- `postflight.json`：冻结输入、积分精度、shot、CPU桥接审计。
- `external_bias_inventory.json`：外部bias文件入口及25phase FKP扰动幅度。
- `source/`：本轮复现代码；大目录/窗口/核保留远端计算目录。

NERSC计算目录：`/pscratch/sd/l/lzy/PNG-2PCF-model-to-observe/outputs/task43_outputs/rsd_validation/task432_model_repair/fkp_conditional_diagnostics`。

已有full-RIC正式结果仍在 `9.22meeting/task432_hybrid_fullRIC_ezmock1000_9.18contract`。本轮是独立原因诊断，不改变正式参数口径。
