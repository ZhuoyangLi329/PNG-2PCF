# Mission 7: Model Validation Rerun With Normalized exp_power Window (2026-03-13)

变更点（只改这一处）：
- `exp_power` IR window 加归一化因子，使得在 `k_f` 处连续：
  - 旧：`W(k)=1-exp(-(k/k_f)^x)` (k<k_f), `W=1` (k>=k_f)
  - 新：`W(k)=[1-exp(-(k/k_f)^x)]/[1-exp(-1)]` (k<k_f), `W=1` (k>=k_f)
- 其余 FFTLog 积分范围与 mission7 当前默认保持一致：
  - `kint_min=1e-4`, `kint_max=20`, `taper_frac=0.06`, `padding=4`, `N=4096`

代码修改：
- `/pscratch/sd/l/lzy/1gpc-UNIT-fnl100-fastpm/agent/mission7_masscut/scripts/masscut_model_validate.py`
  - `build_ir_param_window()` 增加 `/(1-exp(-1))` 并 `clip` 到 [0,1]

运行方式：
- slurm array job: `50023690`（masscuts_high_run.tsv: mmin_1e13, mmin_1p4e13, mmin_1e14）
- 输出子目录：`model_validation_normexp/`（避免覆盖旧的 `model_validation/`）

产物路径：
- mmin_1e13:
  - `/pscratch/sd/l/lzy/1gpc-UNIT-fnl100-fastpm/masscut_scan/mmin_1e13/model_validation_normexp/`
- mmin_1p4e13:
  - `/pscratch/sd/l/lzy/1gpc-UNIT-fnl100-fastpm/masscut_scan/mmin_1p4e13/model_validation_normexp/`
- mmin_1e14:
  - `/pscratch/sd/l/lzy/1gpc-UNIT-fnl100-fastpm/masscut_scan/mmin_1e14/model_validation_normexp/`

每个目录包含：
- `bestfit_params.json`
- `metrics.json`
- `r2xi_compare.png`

