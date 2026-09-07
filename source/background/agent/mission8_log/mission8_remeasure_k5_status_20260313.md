# Mission 8：1Gpc 重测 P(k) 到 kmax=5 的当前状态

## 本轮目标

直接检验一个更物理、且更少依赖 `desilike` 外延的方案：

1. 不再把只测到低 `k` 的 `P(k)` 外推到 `k=20`。
2. 重新测量 `1Gpc` 的均值 `P(k)`，把 `k` 覆盖至少推进到 `5`。
3. 然后直接把测量均值 `P(k_q)` 代入全离散求和：

   `xi(r) = (1 / V) * sum_q g_q P(k_q) j0(k_q r)`

4. 只把求和做到测量数据本身的最后一个 `k`。

## 为什么这一步必要

此前已经做过截断扫描：

- `kmax=0.3` 时，best-fit 全离散的 `mean_abs_sigma` 约为 `1.66`
- `kmax=1.0` 时，约为 `0.43`
- `kmax=3.0` 时，约为 `0.20`
- `kmax=5.0` 时，约为 `0.18`
- `kmax=20.0` 时，约为 `0.167`

这说明如果直接用测量 `P(k)` 建模 2PCF，测量上限太低会立刻失败；当前最合理的下一步，是把实际测量 `kmax` 先推到 `5`，看是否已经足够接近原始全离散基准。

## 已提交作业

- Slurm 脚本：
  `/pscratch/sd/l/lzy/1gpc-UNIT-fnl100-fastpm/agent/mission8_log/run_1gpc_remeasure_pk_g1024_k5.slurm`
- 作业号：`50046921`
- 当前参数：
  - `BOX_SIZE = 1000`
  - `GRID_SIZE = 1024`
  - `KMIN = 0.0031415926535897933`
  - `DK = 0.0015707963267948966`
  - `KMAX = 5.0`
  - `REALIZATION = 1..50`

## 预期输出目录

- `fnl=100`
  `/pscratch/sd/l/lzy/1gpc-UNIT-fnl100-fastpm/pk_remeasure_g1024_k5`
- `fnl=0`
  `/pscratch/sd/l/lzy/1gpc-UNIT-fnl100-fastpm/pk_remeasure_g1024_k5_fnl0`

## 自动收尾脚本

为避免作业完成后还要手动接续，已经启动自动监控：

- 监控脚本：
  `/pscratch/sd/l/lzy/1gpc-UNIT-fnl100-fastpm/agent/mission8_log/watch_and_run_mission8_measured_pk_1gpc_k5.sh`
- 运行日志：
  `/pscratch/sd/l/lzy/1gpc-UNIT-fnl100-fastpm/agent/mission8_log/mission8_measured_pk_1gpc_k5_runtime.log`

该脚本会在两个输出目录各自达到 `50` 个 `pk_rsd_N*.dat` 后，自动执行：

- `/pscratch/sd/l/lzy/1gpc-UNIT-fnl100-fastpm/agent/mission8_log/mission8_measured_pk_1gpc_k5.py`

并在 `desilike` 环境内完成建模、画图和指标汇总。

## 建模脚本输出

预定输出如下：

- 指标：
  `/pscratch/sd/l/lzy/1gpc-UNIT-fnl100-fastpm/agent/mission8_log/mission8_measured_pk_1gpc_k5_metrics.csv`
- 图像：
  `/pscratch/sd/l/lzy/1gpc-UNIT-fnl100-fastpm/agent/mission8_log/mission8_measured_pk_1gpc_k5_compare.png`
- 总结：
  `/pscratch/sd/l/lzy/1gpc-UNIT-fnl100-fastpm/agent/mission8_log/mission8_measured_pk_1gpc_k5_summary_20260313.md`

## 当前状态

- `sbatch` 已提交成功；
- 队列状态：`PENDING (Priority)`；
- 自动监控已启动；
- 建模脚本已通过语法检查。
