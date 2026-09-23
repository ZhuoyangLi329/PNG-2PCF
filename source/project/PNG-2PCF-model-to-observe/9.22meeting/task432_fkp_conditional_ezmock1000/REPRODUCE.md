# FKP / conditional diagnostics — run map

所有结果为诊断，formal full-RIC constraints 保持不变。资源绑定同一login的CPU6-13，总CPU<=8。

远端脚本位于 `codes/task432/`：

1. `task432_fkp_conditional_modes.py`（desilike Python）：复用25halo+1000EZ；预定义5个模式。
2. `task432_fkp_paired.py pk|xi --phase ph000|ph012|ph024`（cosmodesi main CPU）：同一目录与random位置，own/其余24phase平均nbar FKP；P重算I2、shot、smooth window，xi同两块LS。
3. `run_task432_fkp_ric.sh`：重算FKP权重依赖的IC模型差分；两独立scramble，首点16k为参照，其余8k。Python实现 `task432_fkp_ric_response.py`，所有核有独立文件。
4. `task432_fkp_analyze.py --require-ric`：数据减模型差分，响应及恒定模型增量诊断MAP。
5. `task432_fkp_inventory.py`：有界外部halo-matter数据入口核查，不读取大粒子文件。
6. `task432_fkp_postflight.py`：冻结输入与numerics/CPU桥接审计、生成小型交付归档。
7. 本地 `build_report.py`：四页PDF，随后使用内存JPEG视觉验收，不输出PNG。

原始目录/窗口/核只保留在远端 `outputs/task43_outputs/rsd_validation/task432_model_repair/fkp_conditional_diagnostics/paired`；本地只同步关键结果、向量、审计和源码。

配对的3phase和2LS blocks不是25phase生产替换。原EZ1000在诊断中只作固定标尺；本轮不生成匹配shuffled流程的正式covariance，也不引入新的物理自由参数。


## 复现环境与精度记录

- 远端项目根目录：`/pscratch/sd/l/lzy/PNG-2PCF-model-to-observe`。脚本复用项目已有 full-RIC 模块和冻结输入，输入hash在conditional_modes.json和delivery_manifest.json。
- 测量与IC：先source `/global/common/software/desi/users/adematti/cosmodesi_environment.sh main`；CPU绑定6-13，JAX_PLATFORMS=cpu，OMP/OPENBLAS/MKL均1。已有文件复用，避免重复测量。
- 模式、汇总与审计：使用 `/global/homes/l/lzy/anaconda3/envs/desilike/bin/python`；跨环境运行时前缀为 `env -u PYTHONPATH -u PYTHONHOME PYTHONNOUSERSITE=1 taskset -c 6-13`。IC runner已包含这项修复。
- IC每个phase两个seed：4322407和4322447。ph000/4322407使用16384，其他使用8192。验收针对两次配对差分之差：C_single二次型<0.001、每个b1对比及joint响应绝对值<0.001。三phase均通过。
- 同一host最多8 CPU总占用，不要并行在不同login各跑8核。不要为了汇总重新运行已经完成的目录/积分。
- 本地绘图使用独立venv，requirements-local.txt保存安装版本。在项目根目录依次运行 source/local_mode_audit.py、source/local_covariance_calibration.py、source/build_summary.py、source/build_report.py（source指本交付目录下source）。输出文件位于本地outputs/task432_fkp_conditional_ezmock1000。
- delivery_manifest.json校验远端13个核心文件；final_manifest.json校验包括中文报告、PDF、独立审计和源码的完整交付。pdf_quality_audit.json记录PDF SHA及逐页验收。
- 首次汇总失败仅因Python3.12的NumPy路径进入Python3.11；清除路径后汇总和49项审计通过。测量、IC Python科学源未因该环境修复改动。
- postflight恒定模式检查按phase分别保存键，避免同名kernel跨phase覆盖审计条目；最终49项均通过。
- 预览、旧runner参数及任务书备份位于远端old_doc_codes/task432_fkp_conditional；主线数据、目录、窗口、核和正式链不删除。
