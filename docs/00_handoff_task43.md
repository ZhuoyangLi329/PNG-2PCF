# Task 4.3 交接说明

## 1. 科学背景

目标是检验已经在周期盒中验证的 local-PNG 2PCF 建模，能否在非立方体的 lightcone mock 中工作。local PNG 通过

\[
\Delta b(k,z) \propto f_{\rm NL}\,\alpha(k,z),\qquad \alpha(k,z)\propto k^{-2}T(k)^{-1},
\]

增强大尺度信号。周期盒中可以使用离散母盒模式和 FullDiscrete 2PCF；lightcone 中还要处理 angular/radial window、RSD、FKP、Landy--Szalay、GIC/RIC 以及 P--xi covariance。

## 2. 固定数据合同

- AbacusSummit `base_c000`, `ph000--ph024`，当前 box-safe lightcone 为 `0.4 < z_obs < 0.8`。
- 25 phase 的观测曲线取算术平均。
- P 侧：`P0` 全部 15 个基准 bins，`P2` 使用 `k >= 0.015 h/Mpc` 的 11 个 bins。
- xi 侧：`50 <= s < 350 Mpc/h`，删除 `80 <= s < 120 Mpc/h`，即删除中心 `85,95,105,115`，保留 xi0/xi2 各 26 个 bins。
- 合同总维度为 `22 + 52 = 74`，P/xi joint 使用完整 cross block。
- posterior、error bars 和 likelihood 使用 `C_single`，不除以 25；均值 shape 检验另用 `C_single/25` 或 phase-level 统计。
- P 侧 `sn0` 是自由的；9.18 xi contact term 固定为零，full RIC 后 xi-only 也需要 marginalize 现有 `sn0` 的投影响应。

## 3. 9.18 基准

入口：

- `source/project/PNG-2PCF-model-to-observe/codes/task43/task43_run_lightcone_joint_baomask_v1.py`
- `source/project/PNG-2PCF-model-to-observe/codes/task43/task43_fit_rsd_lightcone_x25.py`
- `source/project/PNG-2PCF-model-to-observe/codes/task43/task43_rsd_model.py`

xi 理论是有限母盒的 shell-averaged FullDiscrete Kaiser × squared-Lorentzian FoG：

\[
P^s(k,\mu)=P_{dd}(k)\,[b_1+q\alpha(k)+f\mu^2]^2
\left[1+\tfrac12(k\mu\sigma_s)^2\right]^{-2},
\]

\[
\xi_\ell(s)=V^{-1}\sum_q g_q P_\ell(k_q)\,\overline{j_\ell(k_qs)}.
\]

P 侧和 xi 侧共享 `sigma_s`；P 侧另外有 `sn0`。formal GIC 是由 RR/window 产生的 xi0 标量修正，xi2 不变。

9.18 的 rawbox/lightcone 诊断显示：P0 形状可以闭合，但 xi0/xi2 的宽带形状和 phase mean 检验存在严重问题，sigma_s 在 P 与 xi 之间有张力，所以该模型不能直接作为最终 science 模型。

## 4. 9.22 新模型

主要入口：

- `codes/task432/task432_lightcone_png_velocileptors.py`
- `codes/task432/task432_png_velocileptors_gsm.py`
- `codes/task432/task432_hybrid_gic.py`
- `codes/task432/task432_full_ric_*.py`

9.22 将 xi RSD 理论替换为 velocileptors Gaussian CLPT/GSM baseline，并保留 minimal PNG response：

- GSM 由 linear matter spectrum 构建 real-space density、pairwise velocity、velocity dispersion cumulants；
- 用 Gaussian streaming mapping 得到 redshift-space xi multipoles；
- `b1_L=b1_E-1`；`b2, bs, b3`、EFT alpha 和 `s2fog` 固定为零；
- PNG density response 为 `Delta P = (2*b1*q*alpha + q^2*alpha^2)*P_L`；
- 额外加入 leading density--velocity response `Delta v ~ -2*q*alpha*P_L/k`；
- velocity--velocity response 暂时仍采用 Gaussian CLPT/GSM baseline；
- xi-only 参数变为 `(fNL,b1)`，joint 中 `sigma_s_P,sn0` 只作用在 P 侧。

## 5. GIC/RIC 递进

第一版 hybrid 没有 GIC，已经被替代。第二版 `hybrid_gic` 用当前 hybrid xi0 在完整 RR 距离支持上计算 `C_W(fNL,b1)`，并只从 xi0 减去该确定性常数，不增加自由 amplitude。

最新 `fullRIC` 进一步使用 eBOSS/pywindow 风格的径向投影：

\[
\Xi_{RIC}=\Xi-\Pi\Xi-\Xi\Pi^T+\Pi\Xi\Pi^T,
\]

显式加入两个 density--RIC cross、RIC auto、data/random shot 响应和现有 `sn0` 的投影。P 使用 endpoint/local LOS，xi 使用 midpoint LOS，xi 在 `(s,mu)` 单元中 RR-normalize 后再投影。GIC 只计一次，不与 full RIC 重复相加。

## 6. 当前数值结果

9.22 full RIC / EZmock1000：

| probe | b1 median | fNL median and 68% interval | raw chi2 |
|---|---:|---:|---:|
| P02 | 2.4098 | -6.49 [-39.99, 27.15] | 0.895 |
| xi02 | 2.4185 | 1.49 [-26.78, 29.63] | 2.734 |
| joint | 2.3765 | 3.05 [-23.76, 28.28] | 4.006 |

采样和直接模型 gate 通过，但 joint b1 仍低于 P-only 和 xi-only，说明 full RIC 没有解决 joint 中心偏离问题。结果应被视为方法诊断，不是最终 PNG 测量。

## 7. 希望 GPT 重点审阅

1. 9.18 FullDiscrete 与 9.22 GSM 的 xi mean 是否使用了完全一致的 bin、LOS、shell average 和 window 合同。
2. minimal PNG response 是否遗漏 velocity--velocity、higher-bias 或 nonlinear PNG terms。
3. full RIC 的 midpoint/endpoint 几何、RR normalization、shot response 和 `sn0` projection 是否与数据 estimator 一致。
4. P/xi covariance cross block 是否对应同一 mean model、FKP、random/shuffling 和 finite-mock contract。
5. joint b1 位于边缘结果之外是否可由 cross covariance、非线性后验几何或 model mismatch 解释。
6. 下一步应优先做哪一个可证伪实验：相同 GSM 同时生成 P/xi、独立 sigma、完整 xi window convolution、或经验多 realization covariance。
