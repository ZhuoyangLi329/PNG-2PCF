# Task 4.3 lightcone RSD review package

这是 Task 4.3 的研究代码与证据快照，涵盖 AbacusSummit/EZmock 的 RSD halo lightcone、P(k)-2PCF 一致性、joint b1 位移和 GIC/RIC 建模。

## 最新状态（2026-09-29）

- **正式参考：9.22 `formal_mean4`**，hybrid + full GIC/RIC、EZmock1000、74维数据和完整 P-xi cross covariance。
- **最新探索：9.30 velocileptors xi full-RIC pilot**。代码、两套算子网格、直接MAP、短链和PDF已补齐；全部标为 `diagnostic_only`，不替换正式参考。
- 当前PNG区间覆盖零不等于模型已完成严格closure；joint中心也没有必须位于两个marginal中心之间的数学约束。

## 阅读入口

1. [交接说明与固定数据合同](docs/00_handoff_task43.md)
2. [9.18基准](docs/02_9.18_baseline.md)
3. [9.22 hybrid与full RIC](docs/03_9.22_hybrid_fullRIC.md)
4. [9.30 pipeline、结果与复现范围](docs/06_9.30_velocileptors_fullRIC.md)
5. [结果与审计索引](docs/04_results_audit.md)、[代码索引](docs/CODE_INDEX.md)、[主图索引](docs/FIGURES.md)

## 离线核验本次更新

在仓库根目录运行，仅需要Python 3.10+标准库；不会导入或执行科学pipeline：

```bash
python3 tools/verify_task432_snapshot.py
```

若环境已有NumPy，可额外验证NPZ数组、RIC投影重组和短链分位数：

```bash
python3 tools/verify_task432_snapshot.py --arrays
```

此校验检查归档一致性，不重跑模型、重新拟合或授予科学closure。科学脚本仍保留原NERSC绝对路径和环境约定；完整复跑所需数据、依赖与缺失验证见[复现说明](docs/06_9.30_velocileptors_fullRIC.md#复现范围)。

## 范围

保留原有9.18/9.22快照，并增量补充9.30。原始catalog、大型计数缓存和部分完整链保留在NERSC；仓库含明确选择的紧凑数组及证据，不是完整可移植的生产环境。

本次源文件逐字节复制，来源、大小、SHA256和排除项见[新增来源清单](provenance/task432_20260929_sources.json)。本仓库不将这些结果标记为已发表的science constraint；任何 `pass` 仅表示其明确列出的校验已通过。
