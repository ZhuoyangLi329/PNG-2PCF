# Notebook source mirror

Original: `source/background/agent/mission3_log/mission3_log.ipynb`

Outputs remain in original notebook.

## Cell 1 (markdown)

# mission3 日志（Notebook）

记录任务 3 的关键过程与结论（按章节整理）。

## Cell 2 (markdown)

## 1. 任务目标

- 3.1：用 `desilike` 对 FastPM `fnl100` 的 P0 做拟合，得到 best-fit 参数。
- 3.2：在不用 `mcfit` 的前提下，自写 FFTLog 把 `P0(k)` 变换到 `\xi_0(r)`。
- 3.2.1：再用 `mcfit` 做一致性校验，验证自写 FFTLog 积分结果是否可靠。

## Cell 3 (markdown)

## 2. 环境与输入

- 运行环境：`conda activate desilike`。
- 主要模型文件：`/pscratch/sd/l/lzy/1gpc-UNIT-fnl100-fastpm/pk-pcf-model/model.ipynb`。
- 2PCF/FFTLog 检验脚本目录：`/pscratch/sd/l/lzy/1gpc-UNIT-fnl100-fastpm/pk-pcf-model/`。

## Cell 4 (markdown)

## 3. 3.1 功率谱拟合结果

- 读取新路径下的 P0 测量数据并完成拟合流程。
- 采用的 best-fit（用于后续 2PCF 建模）约为：
  - `fnl_loc = 122.28837057221368`
  - `b1 = 1.4799074735660578`
  - `sigmas = 3.348575085653272`
  - `p = 1.1`（固定）
  - `sn0 = 0.0`（固定）

## Cell 5 (markdown)

## 4. 3.2 自写 FFTLog 建模思路

- 目标积分：
  xi_0(r) = integral[ k^2/(2*pi^2) * P_0(k) * j_0(kr) ] dk
- 在对数 k 网格上构造 FFTLog 所需输入并执行快速汉克尔变换。
- 显式保留 `kmin` / `kmax` 控制，并在边界加平滑窗（taper）降低截断振铃。

## Cell 6 (markdown)

## 5. 3.2 代码实现要点

- 使用 `scipy.fft.fht / fhtoffset` 完成 FFTLog 核心变换。
- 使用统一的理论 P0 生成逻辑，保证与 3.1 参数定义一致。
- 输出时重点看 `r^2\xi_0(r)`，便于观察 BAO 相关形状变化。

## Cell 7 (markdown)

## 6. 3.2.1 与 mcfit 的一致性校验

- 另写独立检验脚本，分别用自写 FFTLog 与 `mcfit` 计算同一组积分。
- 对多个 `kmin / kmax` 组合做对比，确认两者曲线整体接近。
- 结论：当前自写 FFTLog 实现可用于后续 PNG 2PCF 问题复现。

## Cell 8 (markdown)

## 7. 关键图示（历史）

- `P0` 数据与 best-fit（绘到 `kmax=1`）：

![](pk_data_vs_bestfit_kmax1.png)

- `kmax=20` 下扫描 `kmin` 的 2PCF 曲线对比（FFTLog vs mcfit）：

![](fftlog_vs_mcfit_kmax20_kminscan_curves.png)

## Cell 9 (markdown)

## 8. 3.3：复现 PNG 2PCF 建模问题（kmin 扫描）

### 测试设置（更新）
- 固定 `kmax=20`。
- 按你的建议，`kmin` 从 `0.006` 开始往下扫到 `1e-5`：
  `[0.006, 2pi/L, 0.005, 0.004, 0.003, 0.002, 0.001, 5e-4, 3e-4, 1e-4, 5e-5, 3e-5, 1e-5]`，`L=1000`。
- 使用自写 FFTLog（不使用 mcfit）。
- 两组场景：
  1. A：best-fit 参数（`fnl_loc~122.29`）；
  2. B：仅把 `fnl_loc` 改为 0，其余参数与 best-fit 相同。
- 图像按你要求保持 2 子图：左 A，右 B。

### 结果图
![](task33_png_2pcf_kmin_scan.png)

### fnl=0 的大尺度收敛结论
- 我用 `r in [200, 380] Mpc/h` 作为大尺度区间。
- 参考曲线取 `kmin=1e-5`，比较 `r^2\xi_0(r)` 的绝对差 `max_abs_diff`。
- 判据结果：
  - 若要求 `max_abs_diff <= 2.0`，则 `kmin <= 0.003` 可认为收敛；
  - 若要求 `max_abs_diff <= 1.0`，则 `kmin <= 0.002` 可认为收敛；
  - 若要求 `max_abs_diff <= 0.5`，结果仍是 `kmin <= 0.002`。
- 因此一个稳妥建议是：`fnl=0` 场景下取 `kmin ~ 0.002` 或更小。
- 明细见：`/pscratch/sd/l/lzy/1gpc-UNIT-fnl100-fastpm/agent/mission3_log/task33_fnl0_convergence_summary.txt`。

## Cell 10 (markdown)

## 9. P0 负值问题过程（简要保留）

### 现象
- 在 3.1 拟合后，用 best-fit 参数外推到高 k 时，`P0(k)` 出现负值（甚至在 `k~5-10` 接近常数负值），导致 2PCF 对 `kmax` 异常敏感。

### 排查
1. 检查拟合日志发现：`sn0` 与 `sigmas` 实际处于“可变参数”状态（并未被真正固定）。
2. 其中 `sn0` 拟合到负值会在高 k 区间压低 `P0`，是出现负 `P0` 的主要原因。

### 修复
1. 在 `theory` 层显式固定：`sn0=0`（必要时连同 `sigmas` 一起固定）。
2. 在 `likelihood` 层再次显式固定，防止 profiling 过程中被重新放开。

### 结果
- 修复后 `P0(k)` 不再掉到负值；
- 2PCF 对 `kmax` 的非物理敏感性显著缓解，行为恢复到更符合直觉的状态。
