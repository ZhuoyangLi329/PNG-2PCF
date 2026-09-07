# 给 GPT5.6pro：请帮助我们解决 rawbox 的 P(k)–2PCF 不一致

## 你需要解决的科学问题

我们用大尺度结构的 scale-dependent bias 限制 local PNG 的 fNL。项目希望把已有的 P(k) 模型变成可靠的配置空间2PCF模型。此前发展了 BinAvgFit + FullDiscrete，并做过周期盒验证；现在高精度的实空间/RSD rawbox测试仍出现曲线形状、nuisance参数和部分fNL中心不一致。

**请以提供下一步指导为目标，而不只复述材料。** 希望你独立批判现有模型、实现、统计方法和结论，指出哪些问题已经能判定，哪些仍有多个解释，并给出优先级明确的最小判别实验。你的建议将交给能SSH到NERSC运行代码的执行agent实现。

你的审阅范围到rawbox为止。暂不设计lightcone、survey window、RIC/GIC补救流程。无窗周期盒中已存在的问题，应先在盒内理解。论文涉及观测的部分只作背景，不要求本轮迁移真实观测。

## 审阅前先区分四类“不一致”

1. **变换/估计器一致性**：同一输入模式通过准确的离散变换，能否重现同一catalog的测量定义？bin-center、shell-average、离散角度、mode count、shot noise、FFT paint的约定是否匹配？
2. **物理模型形状一致性**：连续理论的P、xi是否能同时解释均值曲线、BAO与broadband？P的低k拟合通过，不保证完整Hankel/离散变换后的xi模型通过。
3. **参数一致性**：边际fNL、b1、sigma_s、p的差异有多大？共同catalog意味着中心差不能按独立误差直接定显著性。
4. **统计/数值一致性**：C_single与C_mean、经验与Gaussian covariance、矩阵秩、模式对计数、单位缩放、联合likelihood是否自洽？

请不要把其中一种通过当作其他几种均通过。也不要把某个代码bug修正视为物理模型问题已全部解决。

## 推荐的三轮阅读

### 第一轮：建立整体理解

按README的01–05顺序读，随后看图册的“Abacus rawbox当前诊断”和“Quijote/FastPM历史验证”。重点掌握：

- 为什么PNG使无限体积xi的低k项形式发散；有限周期盒去零模与离散模求和解决的是什么问题。
- BinAvgFit为什么必须在拟合P时就应用，不能仅在最后画图换成bin average。
- FullDiscrete为什么需要球壳平均核；理论求和上限与P拟合上限为何不能混同。
- Quijote、FastPM、Abacus x25和单相位HOD样本在z、L、p、sn0及covariance上的差异。

然后读原始发展材料：

- `source/background/agent/mission10_log/full_discrete_vs_databin.md`
- `source/background/agent/mission10_log/binavgfit_vs_fulldiscrete.md`
- `source/background/agent/mission10_log/mission10_binavgfit_test_results.md`
- `source/background/agent/mission11_log/mission11_acceleration_results.md`
- `source/background/agent/mission12_log/mission12_results.md`
- `source/background/pk-pcf-model/model_3_20_fast.ipynb`（或notebook_text下对应源码镜像）

早期ExpWindow、hybrid和databin探索均保留在mission3–12里，属于历史证据；不要未经核对重新推荐已经失败的配置，也不要把历史笔记中的“已完成”当作当前科学验证状态。

### 第二轮：逐条追踪当前Abacus证据

从以下代码入口追到输入、模型、likelihood和JSON：

1. `source/project/codes/task43/task43_build_rsd_rawbox_catalog.py`：实/RSD坐标、速度单位、周期wrap。
2. `task43_measure_rsd_rawbox_pk0_jaxpower.py`与`task43_measure_rsd_rawbox_p02_jaxpower.py`：mesh400、LOS=z、TSC/interlacing/补偿、noise、多极和P0桥接。
3. `task43_measure_rsd_rawbox_xi_fcfc.py`、`task43_summarize_rsd_rawbox_x25.py`：pair计数、shell与多极归一化。
4. `task43_rsd_model.py`：FullDiscreteRSDModel与shell_jell_kernel。
5. `task43_fit_rsd_rawbox_pk0_vs_xi0_smin50.py`：ExactPeriodicPk0Model；逐离散模k、mu与16-bin选择。
6. `task43_fit_rsd_rawbox_x25.py`：FastRSDModel、periodic_gaussian_covariance、均值/逐phase门。
7. `task43_rsd_rawbox_realspace_{check,mcmc,pk_check}.py`：同批实空间最简模板控制。
8. `task43_rsd_rawbox_joint_pkxi.py`和`task43_rsd_rawbox_joint_4way.py`：联合covariance和原始floor规则。

实际数值从`source/project/outputs/task43_outputs/rsd_validation/rawbox/`内的`comparison/`、`closure/`、`realspace_check/audits/`、`joint_pkxi/audits/`、`joint_p02xi02/audits/`读取；不要只引用本说明或旧任务书。`summary/`含25相实/RSD xi向量，`pk/`含P测量；理论缓存为同级rsd_validation/theory中的z0.725文件。

### 第三轮：对照其他样本与协方差约定

- Quijote：task45_quijote_ultranest.py及其已找回的task42核心依赖。数据有500相位，C_sample、Hartlap/Percival与当前x25 Gaussian covariance不同。目录名LCp50/LCp100是PNG标签，不是本仓库纳入了lightcone。
- FastPM：task41及task4p2共享源码中的rawbox分支，z必须为1、p=1.2、实空间sigma_s=0；当前shell-average结果与旧center-kernel结果分清。
- PNG-base HOD-MAP：task44_fit_pngbase_hodmap_rawbox.py、task44_fit_pngbase_pk_xi_consistency.py及原始三个pair-counter审计。P为连续49 bins，xi主口径smin30；不要套用Abacus x25的稀疏16-bin配置。
- 其他HOD/host-halo尺度诊断和旧图在图册扩展组，宁可保留供你判断，但它们不自动替代任务书冻结结果。

## 目前最值得独立检查的疑点

### 已有可重复证据：joint矩阵的量纲敏感floor

运行 `python tools/audit_joint_covariance.py`。工具仅需NumPy，读取原缓存及原函数AST，不执行NERSC入口。已保存输出在`review_data/joint_covariance_reproduction.json`。

它复现四向cross=0矩阵中的57个xi方向全部被floor抬升，xi单-bin sigma中位数增大约15.4倍；只改变P单位并转换回来就会改变xi块。现成naive联合chi²还低于独立最小值之和。这不是跨相关可以解释的。

请检验该推理和工具是否正确，然后指导修复与重新比较。原有“联合仅增益0.5%/0.9%所以盒内无新信息”的物理结论目前不能接受。请同时检查其余pinv/eigenfloor是否存在类似的秩或单位问题；不要因为同一个工具输出通过就给整个covariance实现背书。

### 仍未解决的物理/统计问题

- 实空间s≥50均值xi形状也失败，s≥120缓解但后验很宽；RSD坐标映射不可能是唯一原因。
- P0与xi0的fNL约束接近，xi0均值shape仍被拒绝。Gaussian precision对phase scatter的检验也失败。
- P02偏好sigma_s约1，xi02偏好约7.8，P侧MAP贴近零边界。相同“sigma_s”是否能作为跨不同尺度/算子的一致物理参数？在何种模型中才应共享？
- P的精确离散mu平均与xi的连续mu多极积分+离散径向壳层不是完全同一角度操作。它们在最低模式处是否会产生可见差异？
- 常数stochastic项在s>0的理想完整变换中为contact项，但实际有限数值求和、pair normalization及covariance中应如何处理？P侧free sn0与xi最简模板是否公平？
- x25经验cross按象限定标，有限样本噪声多大？mode-pair因子2与四极prefactor能否从统一的场级估计器推导，避免经验定标代替归一化？

## 希望你返回的报告形式

1. **核心判断**：哪些P–xi不一致是真正矛盾，哪些是不同尺度、先验、covariance或边际化导致的预期差别。
2. **证据表**：每条判断给“文件路径+函数/JSON键+图”，区分已证实、强怀疑和未知。
3. **优先级清单**：首先修复的实现问题，再做的模型/统计检验；说明因果关系。
4. **最多五个第一阶段最小实验**：为每项明确固定什么、只改变什么、输入产物、输出图/指标、通过/拒绝标准、两种结果分别意味着什么。尽量复用现成向量和缓存，先不用重测大catalog。
5. **模型改进建议**：如建议BAO damping、非线性/尺度依赖bias、FoG自由度或更完整RSD，请给最简方程、参数意义和过拟合/PNG退化的检查办法，不能只说“换更好的模型”。
6. **协方差建议**：明确经验、解析Gaussian、压缩/缩减、有限mock修正分别在哪组样本可用；不要直接逆x25的高维sample covariance。
7. **交给执行agent的任务书**：应修改哪些函数，先做哪些不会破坏已有结果的检查，以及满足什么证据后才允许重画正式约束。

请主动反驳我们的错误解释。若资料不足，列明缺少的具体数组/定义/运行记录，避免把未知写成肯定结论。不要声称运行了未运行的MCMC，也不要因曲线或后验“看起来不错”就宣布问题解决。
