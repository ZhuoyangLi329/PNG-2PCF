# Mission 9: 探索方向总览

## 已完成的实验
1. ✅ 格点模式超额分析 → 低 k 确实超额 30-50%
2. ✅ ContWeight（连续权重替代 g_q）→ 失败，Voronoi cell 不合理
3. ✅ kmax 扫描 → FullDiscrete 完全不依赖 kmax，偏差 ≈ -0.086σ
4. 🔄 Mode-weighted fitting → 运行中

## 进行中
- **Mode-weighted fitting**: 用 k_mw 代替 bin center 做 P(k) 拟合
  - 动机: 3Gpc 第一个 bin 的 k_mw 比 k_center 高 9.5%，对 1/k² 影响约 16%
  - 测试: 3Gpc fnl100 + 3Gpc fnl0

## 待探索的 Ideas

### Idea A: 完整 mode-weighted observable (如果简单版有效)
不只是改 k 位置，而是对每组参数在每个 bin 内做完整的离散模式加权平均：
P_model_bin = Σ_{q in bin} g_q * P_model(k_q) / Σ_{q in bin} g_q
这需要自定义 desilike observable，但更准确。

### Idea B: 考虑 P(k) 的 bin-averaging 效应
测量的 P(k) 是 bin 内所有离散模式的加权平均。如果 P(k) 在 bin 内有曲率，
bin center 处的模型值 ≠ bin 平均值。这个差异可以被解析计算（知道 P(k) 的
二阶导数和 bin 内模式分布）。

### Idea C: FullDiscrete + 解析偏差校正
如果 -0.086σ 偏差来自 mode-weighted averaging 效应，
可以解析地计算校正量并加到 FullDiscrete 结果上。
这比重新拟合更轻量，且物理清晰。

### Idea D: 直接用测量 P(k) 的离散求和（不依赖模型）
如果测量 P(k) 能延伸到足够高的 k（通过更精细的 grid），
可以直接用测量值做离散求和，完全绕过 desilike 模型。
(Mission 8 尝试过但因 k 覆盖不够而失败；需要重新测量到更高 k)

### Idea E: 有限体积效应的理论校正
在有限盒子中，功率谱估计受到 super-sample variance 的影响。
对于 PNG，super-sample modes 通过 mode coupling 修改 sub-box 的
bias 和功率谱。这是一个已知的理论效应（参见 Li et al. 2014），
可能解释了残余偏差。

### Idea F: 非线性校正
当前的 desilike 模型基于线性/准线性理论。在小尺度上，
非线性效应（特别是与 PNG bias 的耦合）可能需要额外校正。
但考虑到 FullDiscrete 不依赖 kmax（高 k 贡献可忽略），
这可能不是主要问题。

### Idea G: 各向异性效应 (RSD)
FullDiscrete 只用了 P0(k) 的球平均。但在有限离散格点上，
不同 (nx,ny,nz) 三元组对应不同的 μ = kz/|k|。
低 k 处 μ 分布是离散的（不是均匀分布），这可能引入偏差。
(Mission 8 测试过 P0+P2+P4，发现影响小，但那是在不同口径下)
