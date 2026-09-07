# 04 数据集与结果证据地图

以下均为已有结果的阅读摘要，不是本次新拟合。误差为中位数两侧68%分位区间，除非注明sigma68。实际JSON、图和代码优先于文字。

## A. Abacus c000 x25：当前最直接的矛盾来源

L=2000 Mpc/h，z=0.725，fNL=0，Mmin=1.4e13 Msun/h。相同25相位的实/RSD测量有桥接审计。主结果根：`source/project/outputs/task43_outputs/rsd_validation/rawbox/`。

| 实验 | fNL | 均值shape | 原结果定位 |
|---|---|---|---|
| 实空间P0最简2参数，16bins | −4.95 (−11.48,+10.28) | chi²45.23/14，PTE3.74e−5 | realspace_check/audits/*realspace_pk_check.json |
| 实空间xi0 s50–350 | 9.51 (−11.15,+10.83) | chi²331.28/28，PTE1.42e−53 | realspace_check/audits/*realspace_mcmc.json及*realspace_check.json |
| 实空间xi0 s120–350 | −17.01 (−80.58,+30.57) | chi²12.03/21，PTE0.939 | 同上；MAP fNL为6.35，不等于边际中位数 |
| RSD P0，16bin primary | −0.91 (−13.89,+12.43) | chi²10.95/12，PTE0.534 | comparison/*kmin0p003*longchain.json |
| RSD xi0 s50–350 | −1.70 (−14.40,+13.57) | chi²162.98/27，PTE2.17e−21 | closure/*x25_fulldiscrete_lorentzian.json |

实空间P0未释放sn0，所以不是仅去掉Kaiser/FoG的公平参数匹配对照。实空间MCMC摘要只写acceptance和分位数，不要自动声称通过了RSD相同的Rhat/tau门。

RSD P0的最低有效箱[0.003,0.005)含18个mode，平均k=0.00400912。加入它后sigma68由旧15-bin的20.12降为13.16。**PNG中心和宽度一致，但xi形状不通过**，是单极结果应保留的结论。

failure audit保存在`source/project/outputs/task43_outputs/rsd_validation/audits/task43_rsd_rawbox_x25_failure_audit.json`：对角sigma比约1.041，完整xi02散布的precision检验约为预期的1.94倍。模型形状与covariance问题尚未独立归因。旧实空间检查的chi²与新check稍不同，因为covariance的fiducial更新；两组都保留且不能混当同次拟合。

## B. Abacus四极子与joint：原结果保留，部分解释需作废

修正版P2的系数已经为5×模式平均。旧2.5因子错误结果不在当前五链汇总中，但源码附近仍残留旧注释。

| 分探针/联合 | fNL中位数 | sigma68(fNL) | sigma_s中位数±sigma68 |
|---|---:|---:|---:|
| P0+P2 | 0.81 | 11.43 | 0.984±0.747 |
| xi0@s50+xi2@s80 | 2.26 | 13.14 | 7.791±0.739 |
| 保存的四向joint | 0.90 | 11.37 | 0.992±0.749 |

原始图与JSON：`joint_p02xi02/audits/task43_rsd_rawbox_joint_4way_summary.json`、`plots/task43/rsd_validation/task43_rsd_rawbox_joint_p02xi02_contours.pdf`。

第三行不能作为正确权重下的joint结论：详见[原规则离线复核](../review_data/joint_covariance_reproduction.json)。单独两个probe不经过联合assemble步骤，其参数分离仍是重要证据；但正式张力显著性需要正确的同源协方差，而且P侧sigma_s的MAP贴0边界。

相同问题影响`joint_pkxi/`记录的单极0.9%增益。因此“周期盒P与xi有限bin向量是双射、joint无新信息”的旧论断不能据这些链成立。理论上底层完整傅里叶与配置空间的联系，不自动赋予截断/稀疏分箱向量可逆性。

## C. Quijote周期盒：历史约束层面相容

L=1 Gpc/h，fid/LCp50/LCp100各500 realizations。标签LC是fNL命名，不是lightcone应用。当前主要版本free sn0，p=1.2，P kmax=0.10；xi保留sigma_s自由。

| tag | P(k) fNL | xi中心55–345 fNL |
|---|---|---|
| fid | −3.1 (−52.5,+48.3) | −11.1 (−67.1,+68.6) |
| LCp50 | 50.1 (−59.0,+61.1) | 52.9 (−79.4,+82.9) |
| LCp100 | 100.5 (−66.1,+66.4) | 109.9 (−88.4,+109.1) |

源码task45与旧task42核心；结果实际搬到`source/project/plots/outputs/task4_outputs/quijote_*`。这些“相容”是在更宽误差与sample covariance下得到的，不与Abacus高精度mean shape失败矛盾。含s<100时sigma_s不能固定到P侧的近零值。

## D. FastPM实空间rawbox

L=3000，z=1，输入fNL=100，98个共同有效realizations，p=1.2、sigma_s=0。

- P rawbox：76.6 (−10.8,+11.0)，z=1、free sn0。
- shell-average xi rawbox：82.7 (−10.9,+10.9)。旧center-kernel约94.49，修正后MAP chi²_H由55.96变成31.36。

已找到并收录实际文件：`source/project/plots/outputs/task4_outputs/task4p2_2pcf_shellavg_fixed_sn0_z1/summaries/task4p2_rawbox_shellavg_fixed_sn0_summary.json`；P在同级`task4p2_pk_rawbox_subbox/fits/rawbox_ownkmin_free_sn0_sigmas0_z1/`。这解决了上轮只按旧outputs路径查找时报告“文件缺失”的问题。

两者相容有所改善，但输入100与后验中心的差别及均值残差仍需解释，不能把“回收同口径P参考”偷换成“绝对无偏回收模拟真值”。部分历史图含subbox对照，仅作为rawbox面板原样保留。

## E. PNG-base HOD-MAP实空间单相位

c300/c302，z=0.5，L=2000，基准fNL=30/100；X/Y/Z未施加RSD。P用连续49箱（首箱[0.003,0.005)，末箱[0.099,0.101)），xi主口径s30–350。不要与Abacus稀疏16箱混用。

| catalog | fixed fNL/free p：P p | xi p | fixed p=1：P fNL | xi fNL |
|---|---|---|---|---|
| c300 | 0.555 (−0.465,+0.489) | −0.135 (−0.490,+0.509) | 46.92 (−18.59,+17.85) | 73.96 (−20.12,+19.76) |
| c302 | 1.034 (−0.169,+0.177) | 0.923 (−0.169,+0.177) | 95.84 (−22.11,+21.97) | 109.91 (−22.47,+22.23) |

三引擎xi差<9e−11，支持测量起伏并非单一pair counter bug。P残余stochastic对xi covariance的传播诊断、HODhost与严格kmin/尺度切割结果均收录，在图册扩展组追查，不自动替换本表冻结口径。

c300/c302共享ph000并各自重调HOD；P和xi也来自同一catalog。没有正确cross covariance时，中心差只能描述，不能直接当独立的张力检验，也不能相乘获得更紧约束。
