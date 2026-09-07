# Mission 10: BinAvgFit+FullDiscrete 额外测试结果汇总（FastPM + Quijote）

本文件汇总你追加要求的测试：

- FastPM：1Gpc 的 fnl=100/0，以及 3Gpc 的 fnl=0（并保留 3Gpc fnl=100 作为参照）
- Quijote N-body：不同 fnl（0,20,30,50,75,100）

统一对比三种方法：

- **FullDiscrete(std fit)**：标准 `desilike` 拟合（bin center）+ FullDiscrete 求和
- **DataBin**：
  - FastPM：替换拟合范围内全部 bin（ndp=20）
  - Quijote：只替换前 20 个 bin（避免高 k 阶梯导致振荡）
- **BinAvgFit+FullDiscrete (new)**：用 bin-average theory 拟合 + FullDiscrete 求和（forward model）

指标口径：对 `r^2\xi_0` 计算 `mean(|Δ/σ|)` 与 `mean(Δ/σ)`（与 Mission 9 相同）。

## 1. FastPM

输出图：

- [mission10_fastpm_binavgfit_fd_vs_databin_fulldiscrete.png](/pscratch/sd/l/lzy/1gpc-UNIT-fnl100-fastpm/agent/mission10_log/mission10_fastpm_binavgfit_fd_vs_databin_fulldiscrete.png)

结果表：

| case | FullDiscrete(std) | DataBin | BinAvgFit+FD (new) |
|---|---:|---:|---:|
| 3Gpc fnl100 | 0.6430, -0.0849 | 0.4414, +0.1805 | **0.4595, +0.1622** |
| 3Gpc fnl0 | 0.6373, +0.0209 | **0.4626, +0.0355** | 0.5790, +0.1036 |
| 1Gpc fnl100 | 0.2770, -0.0862 | **0.2078, +0.0471** | 0.2265, +0.0127 |
| 1Gpc fnl0 | 0.2638, +0.0229 | **0.2129, +0.0197** | 0.2458, +0.0111 |

（每个单元格格式：`mean(|Δ/σ|), mean(Δ/σ)`）

## 2. Quijote N-body（L=1Gpc）

输出图（低/高 fnl 分开）：

- [mission10_quijote_binavgfit_compare_lowfnl.png](/pscratch/sd/l/lzy/1gpc-UNIT-fnl100-fastpm/agent/mission10_log/mission10_quijote_binavgfit_compare_lowfnl.png)
- [mission10_quijote_binavgfit_compare_highfnl.png](/pscratch/sd/l/lzy/1gpc-UNIT-fnl100-fastpm/agent/mission10_log/mission10_quijote_binavgfit_compare_highfnl.png)

结果表（完整表格见下方链接）：

- [mission10_quijote_binavgfit_results.md](/pscratch/sd/l/lzy/1gpc-UNIT-fnl100-fastpm/agent/mission10_log/mission10_quijote_binavgfit_results.md)

从结果上看：

- Quijote 上 **FullDiscrete 本身已经很准**（`mean(|Δ/σ|)~0.09-0.11`），BinAvgFit 改拟合口径后对 2PCF 改善很小（有时略好、有时略差）。
- Quijote 的 **DataBin(20)** 在低 fnl（0-50）确实能进一步改善，但在高 fnl（尤其 100）并不总是更优（可能与“把数据阶梯化后引入的卷积结构”有关）。

