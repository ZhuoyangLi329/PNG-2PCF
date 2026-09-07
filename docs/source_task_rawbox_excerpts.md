# 原任务书rawbox摘录

源：`/pscratch/sd/l/lzy/PNG-2PCF-model-to-observe/agent/task.md`，读取于2026-09-07。

历史原文，含部分与其他geometry的对照措辞；后续survey应用段落未纳入。最新joint因果解读受到本轮离线数值复核挑战，不能照抄。

## 原文件第303–350行

    #4 simulation中的严格测试
    ##4.1 raw box 理论 2PCF BinAvgFit + FullDiscrete建模方法的理论测试
    目标：先在最干净的 periodic raw box / simulation 统计量中验证理论方法本身，
    不加入 window、背景模式、GIC 或 lightcone 效应。理论 2PCF 用同一套
    P(k) 模型经 `BinAvgFit + FullDiscrete` 变换得到；“无偏”定义为 2PCF
    posterior/constraint 回收同口径 P(k) reference，而不是直接回收 simulation 输入值。

    已完成的严格测试：

    1. `Quijote 1Gpc halo RSD boxes`
       - 数据目录：`/pscratch/sd/l/lzy/pks_2pcfs`
       - 当前实际存在 tag：`fid, LCp50, LCp100`，每个都有 500 个 P(k)
         和 500 个 2PCF realization。
       - 当前 active 入口：`codes/task4/task45_quijote_ultranest.py`
         的 `--free-sn0` 口径，以及
         `codes/task4/task45_run_free_sn0_kmax_scan_30k_parallel.py` /
         `codes/task4/task45_plot_free_sn0_kmax_overlay.py`。
       - 旧 `task42 profiler` 结果只作为诊断，不再是 active 4.1 结论；
         已移动到 `old_doc_codes/task4_profiler_results_20260619T054252Z/`。
       - fixed-sn0 基础 run 的 P(k) fit range：`kcen <= 0.08 h/Mpc`；
         当前 free-sn0 主图展示的是 P(k) `kmax_fit=0.10` overlay。
       - 2PCF rlist：`rmax=350 Mpc/h`，最低可从 edge `rmin=50 Mpc/h`
         开始，即包含 center `55 Mpc/h` bin。
       - fixed-sn0 基础 run：free `fnl_loc, b1, sigmas`；
         fixed `p=1.2, sn0=0`。
       - 6.24 meeting 使用的 4.1 主诊断图是 free-sn0 kmax scan：
         `plots/6.24meeting/4p1_diagnostic_task45_quijote_ultranest_free_sn0_pk_kmax_scan_allfnl.pdf`。
         当前已复制到 curated Task4 plot 入口：
         `plots/task4/important_4p1_4p2/4p1_main_free_sn0_pk_kmax_scan_allfnl.pdf`。
         对应 manifest 为
         `outputs/task4_outputs/quijote_ultranest_free_sn0_kmax_overlay_30k/task45_quijote_ultranest_free_sn0_kmax_overlay_30k_allfnl_meeting_manifest.json`，
         base/kmax 输入 manifest 的 `parameter_names` 包含
         `fnl_loc,b1,sigmas,sn0`，`sn0_policy="free"`，`sn0` prior 为
         `[-1,1]`；因此这张 meeting 主图不能被引用为 fixed `sn0=0`
         的验证图。
       - 重要结论：若 2PCF 包含 `50-100 Mpc/h` 小尺度点，`sigmas`
         是 PNG 限制中的关键 nuisance，不能固定到 P(k) 给出的近零值；
         P(k) 在 `k<=0.08` 下基本不约束 `sigmas`。
       - 使用各 tag 自己的 500-realization `C_sample` covariance 后，
         P(k) BinAvgFit 与 2PCF UltraNest posterior 一致：

       | tag | P(k) `kmax=0.10` | 2PCF `55-345` | 2PCF `65-345` | 2PCF `85-345` |
       | --- | ---: | ---: | ---: | ---: |
       | fid | -3.1 -52.5 +48.3 | -11.1 -67.1 +68.6 | -9.1 -68.4 +70.5 | -14.8 -86.2 +75.7 |
       | LCp50 | 50.1 -59.0 +61.1 | 52.9 -79.4 +82.9 | 47.5 -76.2 +93.6 | 45.3 -84.3 +116.9 |
       | LCp100 | 100.5 -66.1 +66.4 | 109.9 -88.4 +109.1 | 110.0 -93.2 +117.4 | 104.1 -89.2 +136.5 |

       curated 主图：

## 原文件第362–379行

    目标：在 FastPM-L3 `fNL=100`、no-RSD cubic rawbox/subbox 上，先用
    periodic rawbox 建立 P(k)/2PCF reference，再测试有限 cube 的 formal GIC
    是否能让 subbox 约束回到 rawbox 口径。显示范围统一写成 bin edges
    `r=50--350 Mpc/h`，实际测量中心是 `55,65,...,345 Mpc/h`。

    数据与统一理论口径：
    - FastPM-L3 是 `L=3000 Mpc/h` 的单一 `z=1` snapshot，不是 redshift
      shell；原模拟从 `z=99` 演化到 `z=1`，有 matched `fNL=0/100`
      realizations。当前 `fNL=100` rawbox P(k)/2PCF 使用共同的 98 个有效
      realizations。
    - 固定 `p=1.2`；no-RSD 因此固定 `sigmas=0`。
    - 2PCF 使用 FullDiscrete，母盒 `kfund=2pi/3000`、
      `kmax_discrete=15 h/Mpc`；subbox formal-GIC 在每个 likelihood 点更新
      parameter-dependent cubic-window `sigma_W^2`。
    - 2PCF 在这些非零 separation 上不实际约束 contact-like `sn0`，固定它
      没有问题；P(k) 会明显受常数项影响，正式结果必须 marginalize
      `free sn0`。
    - P(k) 只拟合 `kmax=0.08 h/Mpc`。rawbox 使用 POWSPEC 已减 Poisson

## 原文件第400–400行

    | rawbox `L3000` | `76.6 -10.8 +11.0` | free |

## 原文件第406–424行

      `s=55,65,...` bin center 直接求值。当前已从 centers 严格恢复
      `50,60,...,350 Mpc/h` edges，并用解析 volume-shell-averaged `j0`
      重建 FullDiscrete 三个 PNG basis。
    - 按物理口径固定 2PCF `sn0=0,sigmas=0,p=1.2`，只采样
      `fnl_loc,b1`。rawbox、L1500 no/formal-GIC、L1000 no/formal-GIC
      五条链均使用 `32 walkers x 8000 steps, burnin=1000`；post-burn 长度
      均超过 `207 tau`，split stability 全部通过。
    - 当前 shell-average 2PCF 与正式 P(k) 结果为：

    | sample | 2PCF no-GIC `fNL_loc` | 2PCF formal-GIC `fNL_loc` | P(k) own-kmin `fNL_loc` |
    | --- | ---: | ---: | ---: |
    | rawbox `L3000` | `82.7 -10.9 +10.9` | N/A | `76.6 -10.8 +11.0` |
    | `L1500` | `75.5 -23.3 +24.0` | `82.1 -25.2 +26.1` | `77.8 -35.9 +38.2` |
    | `L1000` | `63.6 -42.4 +42.9` | `86.4 -51.7 +55.3` | `93.3 -87.9 +112.8` |

    - shell averaging 把 rawbox center-kernel profile 的
      `fNL=94.49, chi2_H=55.96` 改为 MCMC
      `fNL=82.73 -10.88/+10.92`、MAP `chi2_H=31.36`。P(k)/2PCF 的
      中心与宽度现在相容，但 rawbox ensemble-mean residual 仍明显大于完美

## 原文件第913–916行

### 4.3.1 EZmock rawbox 标定、lightcone 验证与阶段 covariance
	2026-07-20 已从 `AbacusSummit_base_c000_ph000..ph024` 的 `z0.725` rawbox 按 `N>=6638`（`Mmin=1.4e13 h^-1 Msun`）提取 25 个 real-space halo catalog，位置使用 `x_L2com` 转到 `Lbox=2000 Mpc/h` 周期盒。
	P(k) 使用登录节点 jaxpower CPU 周期盒 estimator（`mesh=400`），2PCF 使用 FCFC `DD/@@-1`（`s=50..550 Mpc/h, ds=10`），两者严格串行且均不超过 8 核。
	逐相位结果及均值/散布汇总在 `outputs/task43_outputs/ezmock_rawbox_z0p725_mmin1p4e13/`；人工标定采用 `rho_c=1.14, rho_exp=5, pdf_base=0.25, sigma_v=0`、`attach_particle=T`、`Lbox=2000 Mpc/h`、`Ngrid=320`、`Ntracer=1297050`。

## 原文件第935–936行

	- 2026-08-16 rawbox x25 已全部完成 paired real/RSD FCFC `xi0,xi2`；旧实空间 estimator bridge 的全相位最大差为 `1.0e-14`（ASCII 末位舍入）。冻结的 `xi0,smin=50` FullDiscrete Kaiser x squared-Lorentzian-FoG 闭合得到 `fNL=-1.70 ± 13.99`，故 null-bias 门通过（`|median|/sigma=0.121`），MCMC 的 split-Rhat、`>50 tau` 与 half-chain 门也全部通过；但用 single-box periodic Gaussian covariance 做均值检验得到 `chi2=162.98/27, PTE=2.17e-21`，逐 phase profile 聚合为 `1288.70/675`，且 nested-cut 最大漂移为 `2.17 sigma_ref`，所以正式状态必须是 `validation_failed`。结果与三页 PDF 分别在 `outputs/task43_outputs/rsd_validation/rawbox/closure/task43_rsd_rawbox_x25_fulldiscrete_lorentzian.{json,npz}` 和 `plots/task43/rsd_validation/task43_rsd_rawbox_x25_fulldiscrete_lorentzian.pdf`。
	- 同日失败归因审计显示：xi0 的实测/analytic 单-bin sigma 中位比仅 `1.041`，但 analytic precision 对 x25 的 full `xi0+xi2` phase scatter 给出平均 chi2 `119.19`、预期 `61.44`（`1.94x`），经验与 analytic correlation 的最大差为 `0.531`；换成实测对角误差的 post-failure 诊断则为 `PTE=0.081`，10/15 维粗粒化经验 covariance 也未拒绝模型。paired real-space 线性模板在 `smin<120` 同样失败，而 `smin=120` 才通过，说明当前证据同时包含 analytic covariance/precision 失配和非线性 BAO/broadband 模型形状风险，不能把失败唯一归因于 RSD 映射。机器审计为 `outputs/task43_outputs/rsd_validation/audits/task43_rsd_rawbox_x25_failure_audit.json`；rawbox gate 未通过，不能给 4.4 模型绿灯，后续 lightcone 仅作为 window/GIC/covariance 鉴别诊断，不得包装为补救后的 science claim。

## 原文件第940–941行

	- 2026-08-16 按用户要求将当前 active 比较收窄为 **rawbox、monopole only**：2PCF 只保留 `xi0,smin=50,smax=350`，功率谱只保留冻结的 15-bin `P0(k),kmax=0.10 h/Mpc`；不把 `ell=2`、`smin` 扫描或 lightcone 混入本轮结论。25 个 RSD `P0` 已用与 4.3.1 相同的 jaxpower CPU periodic estimator（`mesh=400`）串行测完；real-space `P0` 在逐相位 `POSITION` bitwise equality 后原样复用，25 相位中仅 ph001 有 1 个 float32 边界坐标按周期盒严格执行 `x mod L`。`P0` 用 exact parent-mode bin average Kaiser x squared-Lorentzian FoG、single-box periodic Gaussian covariance和 free `sn0` 得到 `fNL=1.18 (-20.58,+19.66)`、`b1=2.529 (-0.056,+0.057)`、`sigma_s=6.39 (-4.21,+3.75) Mpc/h`、`sn0=0.514 (-0.297,+0.326)`，x25 均值残差为 `chi2=10.70/11, PTE=0.468`，长链收敛门全部通过。对应 `xi0,smin=50` 为 `fNL=-1.70 (-14.40,+13.57)`、`b1=2.553 (-0.055,+0.053)`、`sigma_s=8.87 (-0.99,+1.03) Mpc/h`，但其均值 shape 仍为 `chi2=162.98/27`。因此两种 monopole 的 `fNL` 中心只差 `2.88`（忽略同源相关时约 `0.12 sigma`），都无明显 null bias；真正分歧是 `P0` shape 闭合而 `xi0` shape 不闭合，故仍不能给 4.4 RSD 2PCF 模型绿灯。机器结果为 `outputs/task43_outputs/rsd_validation/rawbox/comparison/task43_rsd_rawbox_x25_pk0_vs_xi0_smin50_l0only_longchain.{json,npz}`，仅含 `P0/xi0` 的两页拟合 PDF 为 `plots/task43/rsd_validation/task43_rsd_rawbox_x25_pk0_vs_xi0_smin50_l0only_longchain_fullrange.pdf`；共同参数 `fNL,b1,sigma_s` 的 68%/95% triangle contour 为 `plots/task43/rsd_validation/task43_rsd_rawbox_x25_pk0_vs_xi0_smin50_l0only_contours.pdf`。
	- 2026-08-16 rawbox `P0` 的主尺度下限进一步按母盒几何修正：`L=2000 Mpc/h` 的 `kfund=2pi/L=0.003142 h/Mpc`，因此在原 lightcone-matched 15-bin A/B 前加入有效的 `[0.003,0.005]` bin，形成 16-bin rawbox primary；该 bin 含 18 个离散 modes，mode-average `k=0.00400912 h/Mpc`，显式母盒枚举与 jaxpower 的 mode count/mean-k 均通过机器精度 gate。`kmin=0.003` 长链得到 `fNL=-0.91 (-13.89,+12.43)`、`b1=2.532 (-0.051,+0.052)`、`sigma_s=6.58 (-4.30,+3.67) Mpc/h`、`sn0=0.523 (-0.308,+0.321)`，x25 均值 shape 为 `chi2=10.95/12, PTE=0.534`，MCMC 全部收敛门通过；相对旧 `kmin=0.005` A/B，`fNL` 中心只移动 `-2.09=0.104 sigma_old`，但 `sigma68` 从 `20.12` 降至 `13.16`（保留原宽度的 `65.4%`）。与 `xi0,smin=50` 的 `fNL=-1.70 (-14.40,+13.57)` 中心只差 `0.79`（忽略同源相关时 `0.041 sigma`），故二者的 PNG 中心和宽度现已几乎一致；`xi0` 的 broadband shape failure 仍独立存在。该 16-bin 结果取代 15-bin 版本成为 active rawbox primary，机器结果为 `outputs/task43_outputs/rsd_validation/rawbox/comparison/task43_rsd_rawbox_x25_pk0_kmin0p003_vs_xi0_smin50_l0only_longchain.{json,npz}`，拟合 PDF 为 `plots/task43/rsd_validation/task43_rsd_rawbox_x25_pk0_kmin0p003_vs_xi0_smin50_l0only_longchain.pdf`，68%/95% contour 为 `plots/task43/rsd_validation/task43_rsd_rawbox_x25_pk0_kmin0p003_vs_xi0_smin50_l0only_contours.pdf`。

## 原文件第1803–1867行

    - 2026-08-21 完成师姐提供的两个 PNG-base HOD-MAP LRG catalog 的
      周期盒测量，并在同日纠正 P0 数据点口径。c300/c302 都是单一
      `ZSNAP=0.5`、`L=2000 Mpc/h` 的完整实空间周期盒，不是 lightcone，
      盒内没有红移演化。当前 `X/Y/Z` 未施加 RSD；没有 random、FKP、
      survey window、GIC、RIC 或 AIC。严格使用
      `V=L^3=8e9 (Mpc/h)^3`、
      `kfund=2*pi/L=0.00314159265 h/Mpc` 和 AbacusSummit c000
      Planck2018 背景；分别固定 catalog 基准 `fNL=30/100` 推断 `p`，
      不要求两个不同 fixed-fNL 基准下的 `p` 相等。

      P0 用 mesh 400、TSC、interlacing 3、compensated 测量。正式
      likelihood 使用周期盒自身的全部 49 个连续 `Delta k=0.002 h/Mpc`
      bins：首箱 `[0.003,0.005)`，末箱 `[0.099,0.101)`，末箱中心
      `k=0.100 h/Mpc`；最低箱含 18 个离散 modes。只删首箱的
      敏感性版本仍是同一周期网格上的连续 48 bins。此前把 Task43
      lightcone-matched sparse 16 bins 当作这里的 primary 是错误口径；
      该结果仅保留 provenance，不再进入 active 汇总或图。

      xi0 在 `s_edges=30..350, ds=10`、ordered `N(N-1)` 归一化下，
      用 FCFC、pycorr/Corrfunc、CuCount 三种 pair counter 独立复测；
      最大 `|Delta xi0|<9e-11`，因此曲线起伏不是单个 estimator 的实现 bug。
      xi0(s>0) mean 中常数 stochastic 项仍为零延迟 contact term；P0
      拟合出的 residual stochastic power 则进入 xi0 Gaussian covariance。

      corrected fixed-baseline-fNL/free-p 长链结果：

      | catalog | fixed `fNL` | `P0 p`（49 bins） | `xi0 p, smin=30` |
      | --- | ---: | ---: | ---: |
      | c300 | 30 | `0.555 -0.465 +0.489` | `-0.135 -0.490 +0.509` |
      | c302 | 100 | `1.034 -0.169 +0.177` | `0.923 -0.169 +0.177` |

      MAP chi2/dof 为 P0 `53.39/46、47.42/46`，xi0 smin=30
      `48.81/30、44.59/30`。
      chi-square 只作拟合诊断，不作为 MCMC 约束否决门禁。

      按用户要求固定 `p=1`、自由 `fNL in [-500,500]` 后：

      | catalog baseline | `P0 fNL`（49 bins） | `xi0 fNL, smin=30` |
      | --- | ---: | ---: |
      | c300 (`fNL=30`) | `46.92 -18.59 +17.85` | `73.96 -20.12 +19.76` |
      | c302 (`fNL=100`) | `95.84 -22.11 +21.97` | `109.91 -22.47 +22.23` |

      smin=50 链仅保留为历史缓存，不进入当前表格、机器审计、状态输出或最终图。

      P0 与 xi0 来自同一 ph000 realization，当前没有完整 cross-covariance，
      所以中心差只能描述，不能按独立误差解释成正式张力。c300/c302 还共享
      ph000 初始相位并分别重调 HOD，不能作为独立 likelihood 相乘。

      连续 48-bin 最低模态敏感性链给 c300/c302
      `p=0.086 -0.642 +0.659 / 0.895 -0.217 +0.223`，相对 49-bin
      primary 移动 `-0.98/-0.80 sigma`；它们不替代主结果。

      xi0 理论使用标准完整 Hankel 变换。缓存中的 `k=5 h/Mpc` 只是
      数值积分上限，不是数据拟合的 kmax；不得用任意硬截断、Fourier
      分段或所谓 band matching 对它作物理解读。当前审计不包含这类分析。

      权威入口为
      `codes/task44/run_task44_pngbase_hodmap_rawbox_login.sh`。机器汇总为
      `outputs/task44_outputs/pngbase_hodmap_rawbox_z0p500/summary/task44_pngbase_hodmap_rawbox_fixedfnl_freep_audit.json`、
      `outputs/task44_outputs/pngbase_hodmap_rawbox_z0p500/summary/task44_pngbase_hodmap_rawbox_fixedp1_freefnl_pk_xi_audit.json` 与
      `outputs/task44_outputs/pngbase_hodmap_rawbox_z0p500/consistency_diagnostics/summary/task44_pngbase_pk_xi_consistency_final_audit.json`。
      三引擎审计为
      `outputs/task44_outputs/pngbase_hodmap_rawbox_z0p500/summary/task44_pngbase_hodmap_xi0_three_engine_s30_350_audit.json`；
      PDF 位于 `plots/task44/pngbase_hodmap_rawbox_z0p500/` 及其
      `consistency_diagnostics/` 子目录；后者新增每个 catalog 的 fixed-fNL/free-p

## 原文件第2053–2143行

	2026-09-07 rawbox 周期盒 P0+xi0 joint 控制实验（Phase 1，零窗/零IC/零泄漏）：
	- 动机：boxsafe lightcone 的 monopole joint 增益为 7.7%；周期盒是同一批
	  模式的无窗控制实验——若 P(k) 与 xi(s) 在盒子里信息冗余（Gaussian 场
	  的双射重分箱），lightcone 的联合增益就来自窗的差异化模式加权而非
	  新宇宙学信息。
	- 数据：冻结 rawbox x25（P0 16-bin 主口径 kmin=0.003；xi0 smin 50/120）。
	  协方差：两个对角块沿用已审计周期盒 Gaussian 约定
	  （ExactPeriodicPk0Model.gaussian_covariance 与
	  periodic_gaussian_covariance，fiducial b1=2.55/sigma_s=8.0）；跨块为
	  新推导的 mode 级公式
	  `C_px[i,j] = p * sum_{q in bin i} g_q <total^2 L_0>_q K_0(q,j)/(N_i V)`
	  （同一理论壳网格/核/角向机器）。两侧审计对角块的 mode-pair 记账
	  相差因子 2，因此 p in {1,sqrt2,2} 括号检验并用 25 相经验 cross
	  profile 定标：**p=1 实测 scale ratio 1.03**（corr 均值 0.0415 vs
	  0.0405），p=sqrt2/2 分别差 0.73/0.52——取 p=1 为主口径。
	- 六链 64x30000 全过门（split-Rhat<1.01、>50 tau、half-chain<0.1 sigma）：

	  | 拟合 | fNL | sigma68 | sigma_s |
	  | --- | ---: | ---: | ---: |
	  | P0 16 bins | `-0.96` | `13.03` | `6.49±3.96` |
	  | xi0 s>=50 | `-1.97` | `13.89` | `8.84±1.01` |
	  | joint s50 | `-0.44` | `12.91` | `6.87±3.88` |
	  | joint naive | | `12.29` | |
	  | joint p=sqrt2 | | `13.18` | |
	  | joint s120 | `-0.93` | `13.12` | |

	- 结论：(1) **盒子 joint 增益仅 0.9%**（vs lightcone 7.7%）——周期盒里
	  P(k) 与 xi(s) 是同批离散模式的双射重分箱，联合不产生新信息；这把
	  lightcone 联合增益的来源钉在窗/估计量的差异化模式加权与噪声结构上
	  （对实际巡天分析是真实增益，但不是新宇宙学信息）。(2) naive 惩罚
	  也近乎消失（cost ratio 1.05 vs lightcone 1.33）：盒子真实跨相关很
	  小（corr profile 均值 ~0.04，lightcone |corr|max 0.48-0.55），独立
	  假设在盒里几乎无害。(3) sigma_s 在单极水平无张力（P 侧 6.5±4.0 vs
	  xi 侧 8.8±1.0，误差重叠）。(4) 跨块 prefactor 敏感性仅 2.1%。
	  (5) 盒子 fNL 误差 ~13（lightcone ~36）：无窗模式计数红利。
	- 注意点：joint 的 chi2/single=0.6/42 异常小——跨协方差把 P/xi 相关
	  方向的残差在联合度量里折价（xi-only 的均值 shape 失败 162/27 并未
	  消失，只是被联合度量重新加权）；引用 joint chi2 时须带此说明。
	- 产物：`rawbox/joint_pkxi/{fits,audits}/`（6 链 + summary）；
	  入口 `codes/task43/task43_rsd_rawbox_joint_pkxi.py`（--smoke/--out-dir）。
	  Phase 2（P2 盒内新测 + 四向 joint 检验 sigma_s 张力是否存活）待做。
	  ⏠ fix-2（EZmock 经验协方差）仍未做；本轮经验 cross 定标是其在盒内
	  的第一次预演。

	2026-09-07 rawbox 四向 joint（P0+P2+xi0+xi2，修正版）+ 七门管线审计：
	- 执行前按用户要求做了七门独立审计：(1) P2 估计量约定（shotnoise2=0
	  全零、P0 桥 vs 审计测量 rel_l2=0.0 逐位一致）；(2) P2 数据物理性
	  （P2/P0~0.28-0.40 符合 Kaiser 预期）；(3) P2 模型 vs 数据——**抓到
	  关键 bug**：BoxP02Model 用 2.5*P*L2 后对模式取平均，而正确估计量是
	  P2=(2l+1)/2*int P L2 dmu = 5*模式平均（lightcone 版用 Gauss 权重和
	  =2=积分配 2.5 恰好对，改到模式平均时漏了 1/2），导致模型 P2 系统
	  性低估数据 2 倍、残差到 26 sigma、sigma_s 被逼到 0；修正 2.5->5.0
	  后 P2 模型与数据一致（Kaiser 比值 0.38 逐 bin 吻合）；(4) P0-P2
	  解析跨块 vs 经验（matcorr 0.72，量级小）；(5) xi 侧复现审计
	  closure；(6) 跨块象限定标（P0-xi0 1.03 复现 Phase 1，l=2 象限
	  2.3-2.5）；(7) 协方差对角块 vs 25 相经验（中位比 1.09/1.10，
	  范围 0.78-1.50）。首轮五链因子 2 bug 结果作废重跑。
	- P2 x25 新测量：task43_measure_rsd_rawbox_p02_jaxpower.py（ells=(0,2)、
	  plane-parallel LOS=z、mesh400 冻结估计量、P0 桥逐位）；P 侧 16-bin
	  primary（kmin=0.003）；P2-P2 方差带 (2l+1)^2=25 前因子（经验验证
	  sigma 比值 0.80-1.50）；跨块象限经验定标。
	- 修正版五链 64x30000 全过门（split-Rhat<1.01、>50 tau、F/M=0.99-1.00）：

	  | 拟合 | fNL | sigma68 | sigma_s | b1 |
	  | --- | ---: | ---: | ---: | ---: |
	  | P0+P2 | `0.81 -11.84 +11.02` | `11.43` | `0.98±0.75` | `2.576±0.046` |
	  | xi0@50+xi2@80 | `2.26 -13.32 +12.96` | `13.14` | `7.79±0.74` | `2.543±0.053` |
	  | joint | `0.90 -11.76 +10.98` | `11.37` | `0.99±0.75` | `2.576±0.045` |
	  | joint naive | `1.35 -11.63 +10.74` | `11.18` | `0.99±0.75` | 同 |
	  | joint half | `1.22` | `11.25` | 同 | 同 |

	- **核心判决：sigma_s 跨探针张力在无窗盒子里存活**——P 侧
	  `0.98±0.75` vs xi 侧 `7.79±0.74`（~6.5 sigma）。窗/IC/泄漏全部
	  归零后张力仍在 => 纯模型失败（Kaiser x Lorentzian-FoG FullDiscrete
	  无法同时描述两侧四极子），不是 boxsafe lightcone 的窗效应。
	  联合后 sigma_s 完全跟随 P 侧（似然被 P 块主导）。
	- 盒子 joint 增益 0.5%（vs lightcone 四向 26%）：周期盒里 P(k) 与
	  xi(s) 是同批模式的双射重分箱，joint 无新信息；naive 惩罚 1.7%
	  （lightcone 11.8%）——独立假设在无窗弱相关场景近乎无害。盒子
	  fNL 误差 ~11.4（lightcone ~31）：无窗模式计数红利。
	- 与 lightcone 对照：两侧 sigma_s 张力方向一致（P 侧低 xi 侧高），
	  盒子里 P 侧 0.98（lightcone 1.21）、xi 侧 7.79（lightcone 9.24），
	  张力幅度 6.5 sigma（lightcone ~8 sigma）——同族模型失败。
	- 产物：`rawbox/pk/task43_rsd_rawbox_p02_*`（25 相测量 npz+json）、
	  `rawbox/joint_p02xi02/{fits,audits}/`（修正版五链 + summary +
	  跨块象限定标记录）、图
	  `plots/task43/rsd_validation/task43_rsd_rawbox_joint_p02xi02_contours.pdf`
	  （灰/红/蓝/紫四链，sigma_s 分离可见，图例中位值+不对称误差）；
	  入口 `codes/task43/task43_rsd_rawbox_joint_4way.py`、
	  `task43_measure_rsd_rawbox_p02_jaxpower.py`。首轮（因子 2 bug）
	  链文件已删除作废。⏠ fix-2（EZmock 经验协方差）仍未做。
