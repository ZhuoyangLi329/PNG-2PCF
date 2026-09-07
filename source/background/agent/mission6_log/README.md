# Mission 6 N-body Validation

这部分对应任务书 `##6 validation with Nbody(要重新做)`。

当前脚本：

- 主脚本：`mission6_nbody_validation_pipeline.py`
- slurm：`run_mission6_nbody_validation.slurm`

执行内容：

1. 从 `/pscratch/sd/l/lzy/pks_2pcfs` 读取 Quijote N-body 的原始测量：
   - `fnl=0` 对应 `fid`
   - `fnl=50` 对应 `LCp50`
   - `fnl=100` 对应 `LCp100`
2. 对每个 realization 分别做三节点 `CubicSpline` 插值，生成 `fnl=20/30/75` 的 `pk` 与 `pcf`。
3. 对 `fnl=0/20/30/50/75/100` 逐个做：
   - 平均功率谱 best-fit
   - FFTLog 计算 baseline / window 两种 2PCF 理论
   - 与测量均值比较
4. 输出单个 fnl 的图和 JSON，同时生成总表。

主要输出目录默认是：

- `/pscratch/sd/l/lzy/1gpc-UNIT-fnl100-fastpm/agent/mission6_log/nbody_validation`

其中包括：

- `interpolated/`：插值后的 `fnl20/30/75` 数据
- `validation/fnlXXX/`：单个 fnl 的 best-fit、图和 metrics
- `summary_metrics.json`
- `summary_metrics.tsv`
- `summary_window_metrics.png`
