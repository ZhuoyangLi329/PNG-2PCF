# Mission 8 方法总总结

本文档汇总 Mission 8 中实际尝试过的主要方法、各自的物理动机、代表性结果，以及最后保留/放弃的判断。  
为避免重复，这里不逐脚本复述实现细节，而是按“方法分支”组织。

## 0. 参照口径

- `Baseline`：连续 `P(k)` 积分，从 `k_f=2pi/L` 起积到 `kmax=20`。
- `ExpWindow`：在 `k<k_f` 加经验 `exp` IR 窗口，
  `W(k)=[1-exp(-(k/k_f)^x)]/[1-exp(-1)]`，`k>=k_f` 取 `1`。
- Mission 8 的主要目标不是重新证明窗口法有效，而是寻找更有解析解释、最好更少经验性的离散建模方案。

## 1. 离散 shell 的解析表达

- 先完成了三维周期盒中离散壳层的解析推导：
  `k_q = k_f * sqrt(q)`，其中 `q = n_x^2 + n_y^2 + n_z^2`。
- 对应的单极相关函数自然写成
  `xi_0(r) = (1/V) * sum_q g_q * P_q * j_0(k_q r)`。
- 若写成一维径向 delta-shell 形式，则 `P_disc(k)` 的系数必须带上 `g_q / k_q^2` 与体积归一化，而不能只是简单地写成 `P(k_q) delta(k-k_q)`。
- 这一步给出了 Mission 8 的理论出发点：有限盒的低 `k` 结构本来就不是连续谱，而是稀疏离散壳层。

参考文件：
- [mission8_summary_20260313.md](/pscratch/sd/l/lzy/1gpc-UNIT-fnl100-fastpm/agent/mission8_log/mission8_summary_20260313.md)
- [discrete_pk_analytic_notes.ipynb](/pscratch/sd/l/lzy/1gpc-UNIT-fnl100-fastpm/agent/mission8_log/discrete_pk_analytic_notes.ipynb)

## 2. 低-k 离散 shell + 高-k 连续积分的 hybrid

- 最早测试的解析替代方案是：最低若干个离散 shell 显式求和，更高 `k` 仍用连续 FFTLog。
- 动机是：窗口法有效，可能是因为它在数值上模拟了最前几个离散壳层，而不是因为真的存在某个物理窗口。
- 结果：
  - 3Gpc 下，最优 `ShellHybrid_best` 可达到 `mean_abs_sigma = 0.1939`，已经接近 `ExpWindow = 0.1790`。
  - 但随着离散 shell 数继续增加，结果会迅速变差。
- 结论：
  - 离散结构确实重要；
  - 但“只把前几个 shell 离散化”的最简单拼接，还不能完全替代经验窗口。

参考文件：
- [mission8_summary_20260313.md](/pscratch/sd/l/lzy/1gpc-UNIT-fnl100-fastpm/agent/mission8_log/mission8_summary_20260313.md)
- [mission8_discrete_shell_metrics.csv](/pscratch/sd/l/lzy/1gpc-UNIT-fnl100-fastpm/agent/mission8_log/mission8_discrete_shell_metrics.csv)

## 3. 只对 PNG 增量项做 shell-hybrid

- 在总 `P(k)` 直接做 hybrid 之外，还测试了只对 PNG 增量项做低-k 离散化，而 Gaussian 主体继续连续处理。
- 这是比“总功率谱整体离散化”更合理的解析版本，因为真正 problematic 的是 PNG 的 IR 行为。
- 结果：
  - 3Gpc 下 `ShellHybridPNG_best = 0.1276`，明显优于 `ExpWindow = 0.2076`。
  - 1Gpc 下 `ShellHybridPNG_best = 0.3005`，明显差于 `ExpWindow = 0.0818`。
- 结论：
  - 3Gpc 中，PNG-only 的离散处理是有希望的；
  - 1Gpc 中，它仍不能替代更有效的经验窗口。

参考文件：
- [mission8_expwindow_box_summary_20260313.md](/pscratch/sd/l/lzy/1gpc-UNIT-fnl100-fastpm/agent/mission8_log/mission8_expwindow_box_summary_20260313.md)

## 4. 原始全离散：从最低非零模到 kmax=20 全部离散求和

- 这是后来用户明确要求作为基准保留的方法。
- 具体做法：
  1. 先用测量 `P0(k)` 做 best-fit，得到连续理论 `P0_best(k)`；
  2. 计算所有离散壳层 `k_q` 与简并度 `g_q`；
  3. 直接做
     `xi_0(r) = (1/V) * sum_q g_q * P0_best(k_q) * j_0(k_q r)`；
  4. 求和到 `kmax=20`。
- 代表性结果：
  - 1Gpc fnl100：`mean_abs_sigma = 0.1674`，`mean_sigma = -0.1668`
  - 1Gpc fnl0：`0.1296`
  - 3Gpc fnl100：`0.4739`
  - 3Gpc fnl0：`0.3903`
- 结论：
  - 它不是数值指标最优的方法；
  - 但它是最干净、最少经验参数、最容易解释的离散建模基准。

参考文件：
- [mission8_all_discrete_1gpc_summary_20260313.md](/pscratch/sd/l/lzy/1gpc-UNIT-fnl100-fastpm/agent/mission8_log/mission8_all_discrete_1gpc_summary_20260313.md)
- [mission8_all_discrete_3gpc_summary_20260313.md](/pscratch/sd/l/lzy/1gpc-UNIT-fnl100-fastpm/agent/mission8_log/mission8_all_discrete_3gpc_summary_20260313.md)

## 5. 保持原始全离散不变，仅改求和上限 kmax=20 -> 12

- 用户后来指出不能只看平均绝对误差，还要看偏差方向。
- 因此专门测试了：在方法完全不变时，仅把离散求和上限从 `20` 降到 `12`。
- 结果：
  - 1Gpc：`mean_abs_sigma 0.1674 -> 0.1655`，`mean_sigma -0.1668 -> -0.1641`
  - 3Gpc：`mean_abs_sigma 0.4739 -> 0.4699`，`mean_sigma -0.4739 -> -0.4699`
- 结论：
  - 高 `k` 离散尾部确实会轻微加重“系统偏高”；
  - 但改善非常有限，不足以从根本上解释全离散与窗口法的差异。

参考文件：
- [mission8_all_discrete_kmax12_compare_summary_20260313.md](/pscratch/sd/l/lzy/1gpc-UNIT-fnl100-fastpm/agent/mission8_log/mission8_all_discrete_kmax12_compare_summary_20260313.md)

## 6. r-bin 平均修正：j0(k s_cen) -> 对 r-bin 做体积平均

- 这条线的动机是：pcf 文件本身给的是 bin 平均，而不是点值，所以 `j0(k s_cen)` 在定义上并不完全自洽。
- 因此把 kernel 改成了对每个 `r` bin 的体积平均版本。
- 结果几乎不变：
  - 1Gpc fnl100：`0.1674 -> 0.1681`
  - 3Gpc fnl100：`0.4739 -> 0.4730`
- 结论：
  - 这项修正物理上合理；
  - 但数值影响极小，不是 Mission 8 主误差源。

参考文件：
- [mission8_all_discrete_rbinavg_summary_20260313.md](/pscratch/sd/l/lzy/1gpc-UNIT-fnl100-fastpm/agent/mission8_log/mission8_all_discrete_rbinavg_summary_20260313.md)

## 7. 把 mu 纳入：用 P0+P2+P4 近似 P(k,mu)

- 用户明确要求把 `mu` 纳入再试。
- 因此构造了两种版本：
  - ModeMu A：逐模式 `P(k,mu)` 求和；
  - ModeMu B：先在每个 shell 内做离散 `mu` 平均，再乘 `g_q`。
- 理论上二者对 `xi0(r)` 应等价，数值上也确实差到机器精度。
- 结果与旧的 `ShellPNG_old` 几乎完全一样：
  - 3Gpc：`0.1276`
  - 1Gpc：`0.3005`
- 结论：
  - 漏掉低-k 离散 `mu` 结构，不是 1Gpc 解析法不够好的主因。
  - 对完整 cubic shell 平均后，`P2` 项基本抵消，`P4` 的 PNG 增量又很小，所以结果不变。

参考文件：
- [mission8_mu_modesum_summary_20260313.md](/pscratch/sd/l/lzy/1gpc-UNIT-fnl100-fastpm/agent/mission8_log/mission8_mu_modesum_summary_20260313.md)

## 8. mode-count preserving k-cell

- 这条线试图把每个 delta-shell 改成有限宽度 radial cell，但 cell 体积严格由 `g_q` 固定，不引入自由参数。
- 动机是：原始 delta-shell 也许过于尖锐，而真实有限盒测量应该对应有限相空间体积。
- 结果明显更差：
  - 1Gpc fnl100：`0.8031`
  - 1Gpc fnl0：`0.4104`
  - 3Gpc fnl100：`0.5177`
- 结论：
  - 这条“无自由参数的 k-cell 修正”没有改善；
  - 原始 delta-shell 近似反而更好。

参考文件：
- [mission8_all_discrete_modecount_cell_summary_20260313.md](/pscratch/sd/l/lzy/1gpc-UNIT-fnl100-fastpm/agent/mission8_log/mission8_all_discrete_modecount_cell_summary_20260313.md)

## 9. 用窄高斯逼近 delta-shell

- 用户提出：若把每个 delta-shell 用很窄的高斯函数逼近，会怎样。
- 测试了 `sigma_k = 0.10 k_f` 和 `0.25 k_f` 两档。
- 结果：
  - 1Gpc fnl100：
    - 原始全离散：`0.1674`
    - `0.10 k_f`：`0.1413`
    - `0.25 k_f`：`0.1058`
  - 但 `0.25 k_f` 会把 `mean_sigma` 从负值翻到正值，即从“系统偏高”变成“系统偏低/偏另一侧”。
- 结论：
  - 数值上它能改善平均误差；
  - 但它本质是人为平滑壳层，而不是更可信的盒模建模，所以没有作为主方法保留。

参考文件：
- [mission8_all_discrete_gaussian_delta_summary_20260313.md](/pscratch/sd/l/lzy/1gpc-UNIT-fnl100-fastpm/agent/mission8_log/mission8_all_discrete_gaussian_delta_summary_20260313.md)

## 10. PNG split：连续 no-PNG 主体 + 离散 PNG 增量

- 用户提出过把 `P(k)` 分成两部分：
  - no-PNG 主体：连续 FFTLog 积分；
  - PNG 增量：离散求和。
- Mission 8 中实际试了几种拆法。

### 10.1 full discrete PNG 增量

- 形式：
  `xi_total = xi_ref_continuous(fnl->0) + xi_png_discrete(full shells)`
- 结果：
  - 1Gpc fnl100：`0.2782`
  - 1Gpc fnl0：`0.2229`
  - 3Gpc fnl100：`0.4758`
- 结论：
  - 整体差于原始全离散，也差于窗口法。

参考文件：
- [mission8_pngsplit_refcont_full_disc_summary_20260313.md](/pscratch/sd/l/lzy/1gpc-UNIT-fnl100-fastpm/agent/mission8_log/mission8_pngsplit_refcont_full_disc_summary_20260313.md)

### 10.2 只对 k<0.05 的 PNG 做离散化

- 形式：
  - `PNGSplit_k005 = xi_ref_cont + xi_png_disc(k<0.05) + xi_png_cont(k>0.05)`
  - `PNGSplit_k005_NoHigh = xi_ref_cont + xi_png_disc(k<0.05)`
- 结果：
  - 1Gpc fnl100：`0.4116` / `0.3481`
  - 3Gpc fnl100：`0.5301` / `0.4743`
- 结论：
  - 仍然没有比全离散或窗口法更好。

参考文件：
- [mission8_pngsplit_ksplit005_summary_20260313.md](/pscratch/sd/l/lzy/1gpc-UNIT-fnl100-fastpm/agent/mission8_log/mission8_pngsplit_ksplit005_summary_20260313.md)

### 10.3 no-PNG 主体从 k_base 起积

- 这是用户又要求改的一版：no-PNG 主体不从 `1e-4` 起积，而改成从 `k_base = 2pi/L` 起积。
- 结果更差，尤其 1Gpc：
  - 1Gpc fnl100：`0.7501` / `0.7742`
  - 1Gpc fnl0：`0.9046` / `0.9059`
- 结论：
  - 这条分解路线可以排除。

参考文件：
- [mission8_pngsplit_ksplit005_kbase_summary_20260313.md](/pscratch/sd/l/lzy/1gpc-UNIT-fnl100-fastpm/agent/mission8_log/mission8_pngsplit_ksplit005_kbase_summary_20260313.md)

## 11. 直接用测量 P(k) 做全离散建模

这条路线的目标是尽量减少对 `desilike` 连续模型的依赖。

### 11.1 直接用现有旧 bins 的 measured P(k)

- 初始测试直接用已有 `pk_masscut` 的测量均值 `P(k)` 做离散求和。
- 旧数据实际只到 `k~0.3`，却被外推到更高 `k`，结果灾难性失败：
  - 1Gpc fnl100：`~157.7`
  - 3Gpc fnl100：`~261.0`
- 结论：
  - 不是“measured P(k)”思路本身必错，而是旧 `k` 覆盖范围远远不够。

参考文件：
- [mission8_measured_pk_direct_existingbins_metrics.csv](/pscratch/sd/l/lzy/1gpc-UNIT-fnl100-fastpm/agent/mission8_log/mission8_measured_pk_direct_existingbins_metrics.csv)

### 11.2 只积到旧数据自己的 kmax_data≈0.299

- 为排除“错误高-k 外延”，又做了只积到 `kmax_data=0.299` 的版本。
- 结果依然很差：
  - 1Gpc fnl100 `MeasuredPkStep_kdata = 2.3456`
  - 1Gpc fnl0 `MeasuredPkStep_kdata = 2.4510`
- 结论：
  - 当前旧 bins 的数据上限太低，无法用于直接 `P(k)->2PCF` 建模。

参考文件：
- [mission8_measured_pk_1gpc_cutkdata_summary_20260313.md](/pscratch/sd/l/lzy/1gpc-UNIT-fnl100-fastpm/agent/mission8_log/mission8_measured_pk_1gpc_cutkdata_summary_20260313.md)

### 11.3 重新测量 1Gpc P(k)，名义 kmax=5，实际到 kmax_data≈3.217

- 后来重新提交了 `sbatch` 任务，用更细 `k` bin 和更高 `kmax` 重测 `1Gpc` 的均值 `P(k)`。
- 由于 `GRID=1024` 的 Nyquist 限制，最终可用数据上限是
  `kmax_data = 3.216991`，不是名义上的 `5`。
- 结果：
  - 1Gpc fnl100：
    - `MeasuredPkStep_kdata = 0.1634`
    - `MeasuredPkInterp_kdata = 1.9598`
  - 1Gpc fnl0：
    - `MeasuredPkStep_kdata = 0.1706`
    - `MeasuredPkInterp_kdata = 0.9375`
- 结论：
  - `step` 版在 `fnl100` 上略优于原始全离散 `0.1674`，但 `mean_sigma = +0.0840`，呈现系统偏低/另一侧偏差；
  - 在 `fnl0` 上它反而明显差于原始全离散；
  - `interp` 版可以直接判为失败。

参考文件：
- [mission8_measured_pk_1gpc_k5_summary_20260313.md](/pscratch/sd/l/lzy/1gpc-UNIT-fnl100-fastpm/agent/mission8_log/mission8_measured_pk_1gpc_k5_summary_20260313.md)
- [mission8_remeasure_k5_status_20260313.md](/pscratch/sd/l/lzy/1gpc-UNIT-fnl100-fastpm/agent/mission8_log/mission8_remeasure_k5_status_20260313.md)

## 12. Mission 8 的总体判断

### 12.1 哪些方法可以明确排除

- 旧 bins 的 measured `P(k)` 直接建模；
- mode-count preserving k-cell；
- 各种 PNG-split 连续/离散拼接；
- 仅靠 r-bin 平均修正；
- 仅靠把 `mu` 纳入 `P0+P2+P4`。

### 12.2 哪些方法有一定启发，但没有作为最终基准保留

- 总 `P(k)` 的 shell-hybrid：说明前几个离散壳层确实重要；
- PNG-only shell-hybrid：3Gpc 下很强，但 1Gpc 下仍不理想；
- 窄高斯逼近 delta-shell：数值上能改善 1Gpc，但会引入额外平滑，不够干净；
- 重测 measured `P(k)` 后的 step 版：1Gpc fnl100 有轻微改进，但系统偏差方向不理想，且 fnl0 不过关。

### 12.3 当前最稳的离散基准

- 如果要求“少经验参数、物理解释最直观、行为最稳定”，当前应保留的离散基准仍是：
  **原始全离散 best-fit + shell sum 到 kmax=20**
- 它的核心公式是
  `xi_0(r) = (1/V) * sum_q g_q * P0_best(k_q) * j_0(k_q r)`。
- 这也是后来在 `task.md` 的 `##9自由探索` 下保留的版本。

### 12.4 Mission 8 最重要的科学认识

- 窗口法之所以有效，不太像是因为窗口本身有明确物理意义；
- 更合理的解释是：它在数值上部分模拟了“有限盒最低几个 `k` 壳层是离散而非连续”的事实；
- 但在 1Gpc fastPM 上，想用一个完全无经验参数、又显著优于当前原始全离散基准的方法，Mission 8 里还没有找到。
