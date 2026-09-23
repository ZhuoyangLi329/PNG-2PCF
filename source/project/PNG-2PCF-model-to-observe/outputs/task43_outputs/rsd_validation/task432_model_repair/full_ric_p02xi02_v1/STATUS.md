# eBOSS 完整 RIC /goal 执行状态

目标 active：完成 P0/P2/xi0/xi2 的完整 cross+auto、GIC/shot/LOS 记账和必要验证，再以原 EZmock1000 covariance、9.18 选择重跑三组拟合与 PDF。

## 用户最新约束

- 优先简单、直接复用 eBOSS 路线；不把 DESI 大批 mock 标定设为首轮前置。
- 2026-09-21 新授权覆盖任务书旧缺包条款：缺包时自主查找、安装，无须再次请求批准；仍优先项目隔离，避免修改共享环境。
- 用户去睡觉后要求：如网络波动断联 NERSC，至少等待 15 分钟后再尝试重连，不频繁重试。后续 SSH/SCP 使用本目录 `remote.py`，它记录 `connection_state.json` 并强制 900 秒冷却。
- CPU 工作登录节点最多 8 核（本轮全部 taskset 6-13）；只保留 PDF 科学图。

## 已完成

- /goal 已创建，未标 complete。
- 作者固定 commit 源码已下载到 NERSC 新输出目录的 `reference/`，pycute C/OpenMP 后端已编译；未修改共享环境或上游源码。
- 进程内 SciPy 旧 API aliases + 纯计数模块加载，复用真实作者 2/3/4 点计数。
- `task432_full_ric_reference.py`：Pi 幂等、常数/径向零模、分项与 QXiQT 一致到约 1e-15，四分量 Poisson 抽样检查通过。
- `task432_full_ric_backend.py`：作者 cross/auto 与独立 dense matrix 对照通过，包括 midpoint/endpoint 输出 LOS、共享及独立 A/B 积分目录；相对误差约 3e-16。
- 随机积分使用三个互不重叠子集（outer、projection A、projection B），cross 与 auto 对称化，避免有限积分目录的重复点污染。
- 正式 halo random 子集缓存 131072 点；使用生产分段常数 FKP，并核对 WEIGHT_TOTAL；已保存原 frozen data/covariance/emulator hash。
- 1024 点共享目录粗核仅用于计时（约 5 s）；8192 点、dchi=10 独立目录完整 midpoint 核约 48 s。
- xi 响应已接到 hybrid 完整 ell0/2/4，先在 40 个 mu bin 除 RR 再投影；初步 clustering-only 修正相对旧 scalar GIC 的模型差在 C_single 度量约 1，尚不是拟合结果。
- P 响应与分片常数 P->xi 解析 Bessel band 积分已写入，数值校验发现 ell4 在 x=0.3 消减误差，已将小 x 级数范围扩大至 x<1，待复测。

## 在跑/下一步

### 最新进展（覆盖下方早期 PID 清单）

- 原 PID440998 的 N16384/N32768 midpoint 与 N32768 endpoint 三套核均已完成。
- Hankel 原函数复测以及 band 积分独立 quad 对照通过。
- 新增 `task432_full_ric_shot.py`，radial N65536、global N16384、两种 LOS shot 核完成；dense pair bridge 与零模检查通过。
- 新增 `task432_full_ric_bootstrap_shot.py`：真实 conditional radial bootstrap +实际随机 LS 分母的 32768 次实验通过。有限 random 噪声必须用 global kernel，data 用 radial kernel；详见同目录 `SHOT_AND_NUMERICS.md`。
- `task432_full_ric_pilot.py` 已接通全 74 维 clustering+shot，正确使用 data/random 不同幅度。P 8/16 点积分误差 ~1e-21；xi shell averaging 与中心取值差 ~.0008。未拟合。
- N32768 对 N16384 的 ξ 核差二次型 ~.011–.013，继续降低积分噪声。
- QMC 三套核正在登录节点顺序运行：seed4322217 midpoint、seed4322257 midpoint、seed4322217 endpoint（均 N32768, dchi2, refine1）。父 PID556787，日志 `logs/qmc_convergence.log`。SSH 启动会话可能持续直到队列完成，但已 nohup；不要重复启动。
- 物理 hybrid 内层网格已发起独立 detached job，PID 文件 `logs/hybrid_inner_grid.pid`，日志 `logs/hybrid_inner_grid.log`；6 个单线程 worker，所有作业 affinity6-13。缓存名称 `geometry/hybrid_inner_refine1_q2_n600.npz`，41×46 参数网格逐行检查点。此缓存不依赖待验几何核，仍需响应/插值验证才允许采样。
- `remote.py` 已强化并发状态更新：成功连接不会覆盖另一个连接记录的失败冷却时间。到目前未遇到网络故障。

下一步优先：检查新任务是否正常、比较 QMC 独立 scramble 与 MC；检查 P 输入 LOS 与 ordinary-window bridge；量化 phase/实际 RR 差异；完成 hybrid 网格和响应插值验证，编译正式 engine 并做优化/收敛 MCMC/PDF。

远端 detached bash PID 440998（login11）按顺序：
1. n=16384, dchi=4, midpoint, ell_out=0..8。
2. n=32768, dchi=2, midpoint, ell_out=0..8。
3. n=32768, dchi=2, endpoint, ell_out=0,2。

日志 `logs/geometry_convergence.log`，PID 文件同目录 `geometry_convergence.pid`；重连先查进程/文件，不重复启动。

下一步：完成核/距离分辨率收敛、P 响应精度、真实 Poisson/已有 sn0 的投影、实际 LS random block 分母与 phase 几何误差、少量 paired 测量；通过后编译 emulator/参数 spline、运行正式 MCMC。正式拟合尚未开始。

## 路径

远端项目：`/pscratch/sd/l/lzy/PNG-2PCF-model-to-observe`

新代码：`codes/task432/task432_full_ric_{reference,backend,geometry,response}.py`

新输出：`outputs/task43_outputs/rsd_validation/task432_model_repair/full_ric_p02xi02_v1/`

原基准：`outputs/task43_outputs/rsd_validation/task432_model_repair/hybrid_ezmock1000_gic_0918_contract/`

关键审计：`smoke/reference_algebra_poisson.json`、`smoke/eboss_backend_bridge.json`、`input_manifest.json`、`geometry/random_pool_ph000.json`。

## 尚未解决、不可省略

- 当前所有核理论输入 LOS 为 midpoint；P 的输出为 endpoint。已核实 jaxpower 的 `local`=firstpoint，普通窗口还可能包括 first-order wide-angle，需量化输入 LOS 近似，不能宣布整个 P 模型严格闭合。
- 当前 pilot 仅 clustering 项；实际 shuffled data/random shot、已有 sn0 的确定响应需补齐与检验。
- xi 的生产合同是 25 random blocks 各算 LS 后平均；响应目前用连续几何 RR 的 mu 重构，有限 block 分母效应仍待检验。
- 当前 1000 EZ 使用公共固定 random，halo 使用自身 shuffled z；本轮固定 covariance 隔离均值修正，该已知系统项不等于已证明无影响。
- n1024 共享目录核只作历史计时，不是推荐正式核；结束时按任务书归档规范收尾。
