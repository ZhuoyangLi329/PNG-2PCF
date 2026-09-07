# Mission 7: FastPM 不同 halo mass_min 的验证

目标：固定 `mass_max` 不动，扫描不同 `mass_min`，对 1Gpc fastpm fnl100 的 halo 样本重新生成 `position -> pk -> 2pcf`，并用现有 `pk-pcf-model/model.ipynb` 的同口径方法（Minuit best-fit + FFTLog + IR window）检查 2PCF 建模在不同质量阈值下是否仍然有效。

本目录只放“批处理脚本 + 少量配置”。实际数据输出会写到：

`/pscratch/sd/l/lzy/1gpc-UNIT-fnl100-fastpm/masscut_scan/<tag>/`

其中 `<tag>` 来自 masscuts 文件第一列，例如 `mmin_1p4e13`。

## 1. 选择 masscuts 列表

我们提供两套：

- `masscuts_light.tsv`：推荐默认（不含 `1e12`），避免生成极大的 halo 文本和非常耗时的 2PCF。
- `masscuts_full.tsv`：包含 `1e12` 到 `1e15` 的更全扫描；注意 `mass_min=1e12` 可能产生数百万 halo，`2PCF` 可能非常慢甚至不可行。

Slurm 脚本默认使用 `masscuts_light.tsv`。若要切换：

把脚本里的 `MASSCUTS_TSV=...` 改成 full，或者在提交时 `export MASSCUTS_TSV=/abs/path/masscuts_full.tsv`。

## 2. 运行顺序（必须用 sbatch，不要在登录节点跑）

1. 导出 RSD halo 位置（pos txt）：
   - `sbatch scripts/run_export_pos_masscut_array.slurm`
2. 计算功率谱 pk：
   - `sbatch scripts/run_pk_masscut_array.slurm`
3. 计算 2PCF（用 32 核）：
   - `sbatch scripts/run_2pcf_masscut_array.slurm`
4. 2PCF 建模验证（需要 `conda activate desilike`）：
   - `sbatch scripts/run_model_validate_masscut_array.slurm`

每一步都是 job array：每个 array task 对应一个 `mass_min`。

## 3. 输出

对每个 masscut，会生成：

- `masscut_scan/<tag>/position/pos_RSD_N*.txt`
- `masscut_scan/<tag>/pk/pk_rsd_N*.dat`
- `masscut_scan/<tag>/pcf/pcf_rsd_N*.dat`（以及 `.dd`/`.xi2d`）
- `masscut_scan/<tag>/model_validation/`
  - `bestfit_params.json`
  - `metrics.json`
  - `r2xi_compare.png`

Slurm stdout/err 会写到：

`/pscratch/sd/l/lzy/1gpc-UNIT-fnl100-fastpm/agent/mission7_log/`

