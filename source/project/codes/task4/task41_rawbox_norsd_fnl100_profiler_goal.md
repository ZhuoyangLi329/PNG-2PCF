# Task 4.1 Goal: raw-box no-RSD fnl100 2PCF profiler validation

## 目标

验证 `3Gpc fastPM fnl100 no-RSD raw periodic box` 中，当前
`BinAvgFit + FullDiscrete` 的 2PCF likelihood 是否能无偏限制 PNG 参数。

本轮只做 profiler，不做 MCMC。参数口径固定为：

- free: `fnl_loc`, `b1`
- fixed: `p = 1.1`, `sn0 = 0`
- fixed: cosmology, `z = 1`, `Lbox = 3000 Mpc/h`
- no-RSD: 不使用 `sigmas`，不使用 Kaiser/FoG 项

这里的“无偏”不是直接要求回收 simulation 输入 `fnl=100`，而是要求 2PCF
profiler 回收与同一口径 P(k) BinAvgFit 一致的参数。原因是 halo PNG response、
FOF halo finder 和固定 `p` 的选择会让 P(k) best-fit 的 `fnl_loc` 不一定等于 100。

## 数据

主数据：

- xi mean/cov: `outputs/task171_outputs/box_mean_current_norsd_fnl100/task171_current_fnl100_box_vs_cutsky_mean_xi.npz`
- P(k): `outputs/task18_outputs/fnl100_box_realspace_pk/pk/pk_realspace_fnl100_N*.dat`
- realization: `N002-N099`，共 98 个

协方差使用 `fnl100` 的 98 个 realization。主 fit 对象虽然是 ensemble mean
2PCF，但这是为了压低数据向量噪声来检查方法；用于 profiler/MCMC 参数限制的
likelihood covariance 必须使用单个 3Gpc raw box 的 sample covariance：
`C = C_sample`，不是 `C_sample / Nmock`。

`C_sample / Nmock` 只能作为 theory-mean diagnostic，用来放大检查模型均值残差，
不能作为当前参数限制的主 covariance。

## P(k) reference 的定义

`P(k) reference fnl_loc` 不是 simulation 输入的 `fNL=100`，也不是 2PCF fit
结果。它是用同一批 fnl100 raw periodic boxes 的 real-space P(k) 均值，在固定
`p=1.1`、`sn0=0`、cosmology fixed 的口径下重新拟合得到的 P(k) best-fit。

具体步骤：

1. 读取与 xi 完全相同的 98 个 realization (`N002-N099`)：
   `pk_realspace_fnl100_N*.dat`。
2. 使用前 20 个 P(k) bins，即约 `k=0.0035-0.0605 h/Mpc`，
   bin edges 为 `kmin=0.002` 到 `kmax=0.062 h/Mpc`。
3. 数据向量为 98 个 P(k) 的均值；权重矩阵使用这 98 个 P(k) 的 sample covariance。
   covariance 整体是否除以 `Nmock` 不改变 best-fit 位置。
4. 理论模型为 no-RSD real-space PNG tracer：

```text
P_h(k) = [b1 + 2 delta_c (b1 - p) fnl_loc alpha(k)]^2 P_dd(k) + sn0
```

其中 `delta_c=1.686`，`p=1.1`，`sn0=0`。

5. 先做 bin-center fit，用 `P_h(kcen)` 给 BinAvgFit 提供初值。
6. 主 reference 使用 BinAvgFit：对每个 measured P(k) bin，枚举 periodic box 的
   离散 mode shells `k_q = k_f sqrt(q)` 及 degeneracy `g_q`，并计算

```text
P_model,bin(i) = sum_{q in bin i} g_q P_h(k_q) / sum_{q in bin i} g_q
```

然后最小化 `chi2 = (P_mean - P_model,bin)^T C_pk^{-1} (P_mean - P_model,bin)`。

本轮得到：

- bin-center fit: `fnl_loc = 71.4773`, `b1 = 2.7355047`
- BinAvgFit reference: `fnl_loc = 71.2115`, `b1 = 2.7369448`

## r-bin 选择

当前 xi 网格为 `s = 85, 95, ..., 345 Mpc/h`，共 27 个点。由于 `Nmock=98`，
主结果最多使用 20 个数据点，并应用 Hartlap/Percival 修正。

本轮 profiler 使用：

- fiducial: 最大的 20 个点，约 `155-345 Mpc/h`
- robustness: 最大的 18/16/14 个点
- broad-step20: `85-345 Mpc/h` 每隔一个 bin 取点，约 14 个点

这个选择参考 DESI 配置空间 PNG mock 工作中保留大尺度信息的思路：
Brown et al. 2403.18789 使用 `2PCF: 50-380 Mpc/h, Δs=10`，并测试
`smax=200/260/320/380`；DESI Fourier-space PNG 分析也强调只用大尺度。

## 协方差修正

对每个 data vector 长度 `Nb`：

```text
alpha_H = (Nmock - Nb - 2) / (Nmock - 1)
Cinv = alpha_H * inv(Csample / Nmock)
```

参数误差报告时乘 Percival correction：

```text
A = 2 / [(Nmock - Nb - 1)(Nmock - Nb - 4)]
B = (Nmock - Nb - 2) / [(Nmock - Nb - 1)(Nmock - Nb - 4)]
M1 = [1 + B (Nb - Np)] / [1 + A + B (Np + 1)]
sigma_corrected = sqrt(M1) * sigma_profiler
```

其中 `Np = 2`。

## 验收标准

本轮 profiler 通过的最低标准：

1. fiducial r-range 的 2PCF profiler best-fit 与同口径 P(k) BinAvgFit 参考值一致：
   `|Δfnl| < 1 sigma_corrected` 且 `|Δb1| < 1 sigma_corrected`。
2. robustness r-range 中 `fnl_loc` 没有随 `rmin` 单调大幅漂移；
   经验阈值为相对 fiducial 漂移不超过 `0.5 sigma_fnl,fid`。
3. Minuit `migrad` 有效，参数不贴边，fiducial `chi2/ndof` 不异常。
4. 输出 JSON/NPZ 记录输入、参数、r-bin、Hartlap/Percival 因子、best-fit 和
   与 P(k) 参考点的 pull。

## 失败时的 debug 顺序

若 profiler 失败，按以下顺序排查：

1. **代码一致性**：固定 `p=1.1` 后，P(k) BinAvgFit 和 2PCF model 是否使用完全相同的
   `P_h(k)` 公式、`alpha(k)`、`kmax`、`K_FUND` 和 `FullDiscrete` cache。
2. **r-bin 问题**：逐步抬高 `rmin`，比较 largest20/18/16/14；如果小尺度导致漂移，
   先把小尺度排除出主结果。
3. **协方差噪声**：检查 covariance condition number、Hartlap 因子、参数误差的
   Percival 放大；尝试 SVD cutoff 或更少 bins。
4. **参考点问题**：确认 P(k) BinAvgFit 在 `p=1.1` 下本身稳定，不是 P(k) reference
   被错误拟合。
5. **FullDiscrete 截断**：扫描 `kmax=15/20`，确认 2PCF 大尺度曲线稳定。

## 如果持续失败，如何反思理论

若上述 debug 后仍无法让 2PCF 回收 P(k) reference，应认真考虑：

1. `P(k) -> xi(r)` 的 FullDiscrete model 仍缺少某个 finite-box 或 estimator 效应；
   raw periodic estimator 下不应有 GIC，但可能有 binning/finite-mode residual。
2. 固定 `p=1.1` 可能不是这个 halo sample 的有效 PNG response；2PCF 和 P(k)
   对 `bphi * fnl` 的响应若不一致，需要改为先测 `bfnl_loc`。
3. 2PCF 的大尺度 bins 高度相关，98 mocks 对 20 bins 的 covariance 仍可能不足；
   参数 pull 可能被 covariance 噪声主导。
4. no-RSD real-space template 与实际 halo catalog 的质量选择、shot noise 或 halo exclusion
   在 `r ~ 100-200 Mpc/h` 仍有非线性 residual。

## 本轮执行结果

实现脚本：

- `codes/task4/task41_rawbox_norsd_fnl100_profiler.py`

主要输出：

- strict mean-covariance: `outputs/task4_outputs/rawbox_norsd_fnl100_profiler_p1p1/task41_rawbox_norsd_fnl100_profiler_summary_kmax15_debug.json`
- kmax robustness: `outputs/task4_outputs/rawbox_norsd_fnl100_profiler_p1p1/task41_rawbox_norsd_fnl100_profiler_summary_kmax20.json`
- single-survey covariance: `outputs/task4_outputs/rawbox_norsd_fnl100_profiler_p1p1/task41_rawbox_norsd_fnl100_profiler_summary_single_kmax15.json`

P(k) BinAvgFit reference 在 `p=1.1` 下为：

- `fnl_loc = 71.2115`
- `b1 = 2.7369448`
- `chi2 = 1.45` for 18 dof

strict mean-covariance (`C_sample / Nmock`) 下，fiducial `largest20`
(`155-345 Mpc/h`) 没有通过验收：

- 2PCF best-fit: `fnl_loc = 65.57 +/- 2.23`
- 相对 P(k) reference: `pull_fnl = -2.53`
- best-fit `chi2/dof = 1.69`
- P(k) reference 处 `chi2/dof = 4.47`

`kmax_discrete = 20` 不能修复该偏差：

- `largest20`: `fnl_loc = 64.89 +/- 2.19`
- 相对 P(k) reference: `pull_fnl = -2.88`

主口径 single-survey covariance (`C_sample`) 下，fiducial `largest20` 通过统计误差口径：

- 2PCF best-fit: `fnl_loc = 65.55 +/- 22.06`
- 相对 P(k) reference: `pull_fnl = -0.26`

主口径 rlist 稳定性：

| rlist | s range [Mpc/h] | Nbin | fnl_loc | sigma_fnl | pull vs P(k) ref | Delta vs largest20 |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| largest20 | 155-345 | 20 | 65.55 | 22.06 | -0.26 | 0.00 |
| largest18 | 175-345 | 18 | 67.10 | 29.10 | -0.14 | +1.55 |
| largest16 | 195-345 | 16 | 58.59 | 30.67 | -0.41 | -6.96 |
| largest14 | 215-345 | 14 | 59.09 | 38.76 | -0.31 | -6.46 |
| broad_step20 | 85-345 | 14 | 85.93 | 9.89 | +1.49 | +20.39 |

结论：对 DESI-style large-scale nested rlist (`rmin >= 155 Mpc/h`)，`fnl_loc`
中心值相对 fiducial 的漂移最大约 `0.32 sigma_fid`，满足当前稳定性验收。
`broad_step20` 把 `85-145 Mpc/h` 的小尺度信息混进来后中心值明显上移，不应作为
本轮 no-RSD 线性 PNG 2PCF 主 rlist。

附加 diagnostic：`C_sample / Nmock` 下会看到 P(k) reference 的 xi 均值残差被放大，
这是 theory-mean stress test 未通过，不应替代主 likelihood 口径。下一轮如果要研究
这个 residual，应优先检查 measured P(k) 与 measured xi 是否在同一
FullDiscrete/Fourier convention 下互相一致，再决定是修正理论映射、改变 P(k)
reference 拟合权重，还是改用 `bfnl_loc` 作为直接响应参数。
