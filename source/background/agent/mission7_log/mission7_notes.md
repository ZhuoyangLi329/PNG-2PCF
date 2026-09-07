# Mission 7: 不同 halo mass_min 的验证（FastPM 1Gpc fnl100）

本文件是 mission7 的“实现说明/日志”，按你的要求放在：

`/pscratch/sd/l/lzy/1gpc-UNIT-fnl100-fastpm/agent/mission7_log/`

说明：我没有在登录节点直接跑计算，也没有自动提交 sbatch（避免占用队列/资源）。这里提供可复现实验的 sbatch pipeline 与输出目录约定。

## 1) 代码位置

mission7 新增的脚本目录：

`/pscratch/sd/l/lzy/1gpc-UNIT-fnl100-fastpm/agent/mission7_masscut/`

其中：
- `masscuts_light.tsv`: 默认较轻量的 mass_min 扫描点（不含 1e12）
- `masscuts_full.tsv`: 满足 1e12..1e15 的更全扫描（含 1e12，注意可能很慢）
- `scripts/`: sbatch 脚本与建模验证脚本
  - `run_export_pos_masscut_array.slurm`: fof -> position(RSD)
  - `run_pk_masscut_array.slurm`: position -> pk
  - `run_2pcf_masscut_array.slurm`: position -> 2pcf (32 threads)
  - `run_model_validate_masscut_array.slurm`: pk best-fit + FFTLog(+IR window) -> xi0，并和测量均值对比
  - `masscut_model_validate.py`: 单个 masscut 的建模验证（命令行版，方便 array 调用）
- `submit_pipeline.sh`: 一键提交 export/pk/pcf/model 四个 array，并设置依赖关系

## 2) 输入数据

FastPM fof（fnl100）：
- `/pscratch/sd/l/lzy/1gpc-UNIT-fnl100-fastpm/fofs_fnl100`

脚本内部会使用已有 skill 的读取逻辑：
- `fastpm-halo-rsd-export/scripts/read_fastpm_rsd_batch.py`
- `fastpm-pk-2pcf/scripts/calc_powspec_fastpm.py`
- `fastpm-pk-2pcf/scripts/calc_2pcf_fastpm.py`

## 3) 输出目录（每个 masscut 一套）

每个 masscut 会写到：

`/pscratch/sd/l/lzy/1gpc-UNIT-fnl100-fastpm/masscut_scan/<tag>/`

结构：
- `position/pos_RSD_N*.txt`
- `pk/pk_rsd_N*.dat`
- `pcf/pcf_rsd_N*.dat`（以及 `.dd` / `.xi2d`）
- `model_validation/`
  - `bestfit_params.json`
  - `metrics.json`
  - `r2xi_compare.png`

## 4) 如何提交（推荐）

默认使用 `masscuts_light.tsv`（7 个点），直接跑：

```bash
bash /pscratch/sd/l/lzy/1gpc-UNIT-fnl100-fastpm/agent/mission7_masscut/submit_pipeline.sh
```

如果要用 full 扫描（含 1e12..1e15），建议先确认资源预算，然后：

```bash
MASSCUTS_TSV=/pscratch/sd/l/lzy/1gpc-UNIT-fnl100-fastpm/agent/mission7_masscut/masscuts_full.tsv \
  bash /pscratch/sd/l/lzy/1gpc-UNIT-fnl100-fastpm/agent/mission7_masscut/submit_pipeline.sh
```

## 5) 环境注意事项

- `export_pos/pk/pcf` 这三步使用当前 `python`（`/global/homes/l/lzy/anaconda3/bin/python`），并复用已有脚本。
- `model_validate` 这一步需要能 `import desilike`。slurm 脚本里默认 `conda activate desilike`。
  - 如果你实际的 desilike 环境名不同，需要改 `run_model_validate_masscut_array.slurm`。

## 6) 运行日志位置

所有 sbatch 的 stdout/err 默认写入：

`/pscratch/sd/l/lzy/1gpc-UNIT-fnl100-fastpm/agent/mission7_log/`

