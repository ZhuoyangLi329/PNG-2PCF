# Task43 jaxpower-only 2PCF `rmin` scan goal

## 目标

在固定 Task43 Abacus `fNL=0` lightcone、RR-deconvolved jaxpower
single-lightcone covariance 和 radial single-term RIC 的条件下，把 2PCF
下边界从 `50` 扩展到 `40/30 Mpc/h`，判断当前 2PCF 的 `b1` 误差比
P(k) 宽约 `1.6x` 是否主要来自小尺度振幅信息的缺失。

## 冻结口径

- 数据：AbacusSummit `base_c000 ph000..ph024`，real space，
  `0.6<z<0.8`，`Mmin=1.4e13 h^-1 Msun`，逐 phase matching 25x random。
- 测量：`cucount.jax` weighted Landy--Szalay，
  `WEIGHT_TOTAL=WEIGHT*WEIGHT_FKP`，`P0=10000`。
- 公共向量：`s_edges=30,40,...,350 Mpc/h`，32 bins，中心为
  `35,45,...,345 Mpc/h`。
- 嵌套拟合：固定上边界 edge `rmax=350 Mpc/h`，比较
  `rmin edge=30/40/50 Mpc/h`，分别使用 32/31/30 bins。
- covariance：只使用一份 32-bin RR-deconvolved jaxpower
  `covariance_single_realization`，三个拟合取对应嵌套子块；不除以 25，
  不使用 RascalC 或 EZmock sample covariance。
- covariance 数值设置：沿用已验证的 Task43 `mesh64/pad400`、
  `window=0..3600, ds=2`、bessel+FFTLog、`k=1e-4..3.0001, dk=0.002`、
  `fNL_cov=0,b1_cov=2.5,p=1,sn0=0`，并对新增 30--50 bins 做独立
  k-support/数值收敛检查。
- 理论：Abacus c000、mother-box `L=2000 Mpc/h` FullDiscrete、full PNG、
  `p=1`、shell-averaged xi、fixed `sn0=0`、同一个 factorized radial
  `IC^(rad,rad)` single term。
- P(k) reference：现有 `kmax=0.10 h/Mpc`、free `sn0`、jaxpower
  covariance 长链，保持完全不变。

## 执行与门槛

1. 新 manifest 只引用既有 catalog，不复制、不移动大型输入。
2. 先测 ph000 smoke，再测全 25 phases；新向量的 50--350 子段必须桥接
   已有 rmax-scan 测量。
3. 生成 32-bin RR、raw jaxpower covariance 和 RR-deconvolved covariance；
   要求 finite、symmetric、SPD、无 eigenvalue flooring，并桥接旧 30-bin
   covariance 子块。
4. 编译一个 32-bin radial-RIC operator；其中 50--350 的三条 basis 必须
   桥接旧 operator。
5. 在 MCMC 前计算 `dmu/db1`、`dmu/dfNL`、累计 Fisher 信息、导数相关和
   covariance-correlation 损失，明确 `b1` 信息来自哪些 separation bins。
6. 三条长链统一使用 `64 walkers x 20000 steps, burnin=5000`；要求所有
   参数 `length/tau>100`、split median shift `<0.05 sigma`。
7. 对新增 `30--40` 与 `40--50` blocks 计算 Schur conditional residual；
   `PTE<0.01`、明显相干残差或参数显著漂移时，不把较窄误差解释为可靠改善。
8. 生成一个 JSON、一个 CSV 和 PDF-only 汇总图，并更新 `agent/task.md`。

## 资源和输出

- 只有 cucount pair count 使用一个 shared GPU，array concurrency 固定为 1。
- covariance、operator、Fisher、MCMC 和汇总全部在登录节点运行，总 CPU
  affinity 不超过 8 核。
- 新产物只写入 `outputs/task43_outputs/rmin_scan/`、
  `plots/task43/rmin_scan/` 和 `codes/logs/task43/rmin_scan/`，不覆盖旧结果。

## 解释规则

- 若降低 `rmin` 后 `b1` 明显收紧且 conditional PTE、残差与参数稳定性均
  通过，则支持“旧 2PCF 丢失小尺度振幅信息”的解释。
- 若 `b1` 不明显收紧，则用 Fisher 分解区分 covariance correlation 与
  `b1--fNL` 退化。
- 若误差收紧但模型 gate 失败，则 `30--50 Mpc/h` 超出当前理论可靠范围，
  不能作为正式约束。
