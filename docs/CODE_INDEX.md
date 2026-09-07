# 代码索引

源码按原目录保存；archive/mission早期文件是历史证据。共享joint helper是明确标注的摘录。索引列出的函数名是静态解析结果，不表示已运行。

### [build_fastpm_1gpc_fnl100_current_bestfit_2pcf.py](../source/background/agent/meeting-fig/build_fastpm_1gpc_fnl100_current_bestfit_2pcf.py)

`source/background/agent/meeting-fig/build_fastpm_1gpc_fnl100_current_bestfit_2pcf.py`

生成 1Gpc FastPM fnl100 的当前方法 best-fit 2PCF 数组。

`build_cosmology`, `select_fit_bins`, `precompute_cache`, `xi_cached_rebin`, `main`

### [plot_fastpm_2pcf_validation_m11.py](../source/background/agent/meeting-fig/plot_fastpm_2pcf_validation_m11.py)

`source/background/agent/meeting-fig/plot_fastpm_2pcf_validation_m11.py`

meeting-fig: FastPM + Quijote 2PCF validation with Mission11 method

`precompute_rebin_cache`, `xi_cached_rebin`, `ensure_box_cache`, `quijote_rid_from_filename`, `load_pk_quijote`, `load_pcf_quijote`, `load_trimmed_fastpm_pk`, `fit_and_model_dataset`, `run_fastpm_dataset`, `run_quijote_dataset`, `draw_combined`, `main`

### [plot_fastpm_measured_2pcf_mean_box_compare.py](../source/background/agent/meeting-fig/plot_fastpm_measured_2pcf_mean_box_compare.py)

`source/background/agent/meeting-fig/plot_fastpm_measured_2pcf_mean_box_compare.py`

meeting-fig: FastPM fnl100 原始盒子 2PCF 均值与当前 best-fit 对比

`parse_realization_id`, `read_pcf_mean_std_from_glob`, `load_bestfit_models`, `plot_fnl100_comparison`, `main`

### [plot_fastpm_original_box_1gpc_vs_3gpc_2pcf_delta.py](../source/background/agent/meeting-fig/plot_fastpm_original_box_1gpc_vs_3gpc_2pcf_delta.py)

`source/background/agent/meeting-fig/plot_fastpm_original_box_1gpc_vs_3gpc_2pcf_delta.py`

FastPM 原始盒子 2PCF 均值：1Gpc vs 3Gpc

`XiDataset`, `realization_id`, `read_one_pcf`, `load_dataset`, `interp_to`, `plot_column`, `main`

### [plot_masscut_baseline_vs_fastdiscrete.py](../source/background/agent/meeting-fig/plot_masscut_baseline_vs_fastdiscrete.py)

`source/background/agent/meeting-fig/plot_masscut_baseline_vs_fastdiscrete.py`

meeting-fig: Masscut validation

`precompute_rebin_cache`, `xi_cached_rebin`, `tag_to_mass_value`, `discover_masscut_tags`, `build_cosmology`, `run_one_masscut`, `draw_masscut_figure`, `main`

### [plot_masscut_exp_ir_window_vs_data.py](../source/background/agent/meeting-fig/plot_masscut_exp_ir_window_vs_data.py)

`source/background/agent/meeting-fig/plot_masscut_exp_ir_window_vs_data.py`

meeting-fig: masscut measured mean + exp IR window model

`main`

### [plot_masscut_measured_2pcf_means.py](../source/background/agent/meeting-fig/plot_masscut_measured_2pcf_means.py)

`source/background/agent/meeting-fig/plot_masscut_measured_2pcf_means.py`

meeting-fig: measured 2PCF means for different masscuts

`tag_to_mass_value`, `discover_masscut_tags`, `main`

### [plot_quijote_measured_2pcf_mean_compare.py](../source/background/agent/meeting-fig/plot_quijote_measured_2pcf_mean_compare.py)

`source/background/agent/meeting-fig/plot_quijote_measured_2pcf_mean_compare.py`

meeting-fig: Quijote measured 2PCF mean comparison for fnl=0 and fnl=100

`parse_realization_id`, `read_pcf_mean_std_from_pattern`, `main`

### [plot_theory_2pcf_current_method_fnl_compare.py](../source/background/agent/meeting-fig/plot_theory_2pcf_current_method_fnl_compare.py)

`source/background/agent/meeting-fig/plot_theory_2pcf_current_method_fnl_compare.py`

meeting-fig: 当前最佳 2PCF 理论建模方法下的 fnl 对比图

`load_r_grid_from_measurement`, `precompute_rebin_cache`, `load_or_build_cached_rebin`, `fast_discrete_xi0`, `build_theory_params`, `main`

### [mission10_binavgfit_full_discrete_compare.py](../source/background/agent/mission10_log/mission10_binavgfit_full_discrete_compare.py)

`source/background/agent/mission10_log/mission10_binavgfit_full_discrete_compare.py`

Mission 10: 解决 FullDiscrete vs DataBin 的“测量依赖”问题

`_rid_from_filename`, `load_pk_multipoles`, `load_pcf`, `j0`, `met_mean_abs_sigma`, `gq_fft`, `gq_enumerate`, `build_shells_for_kmax`, `build_bin_shell_index`, `binavg_pk_from_shells`, `xi_discrete_multi`, `BoxConfig`, `fit_standard_desilike`, `fit_binavg_minuit`, `eval_pk_dense`, `main`

### [mission10_quijote_binavgfit_compare.py](../source/background/agent/mission10_log/mission10_quijote_binavgfit_compare.py)

`source/background/agent/mission10_log/mission10_quijote_binavgfit_compare.py`

Mission 10: Quijote 不同 fNL 样本上验证 BinAvgFit+FullDiscrete

`_rid_quijote`, `load_pk_quijote`, `load_pcf_quijote`, `j0`, `met_mean_abs_sigma`, `gq_fft`, `gq_enumerate`, `build_shells_for_kmax`, `build_bin_shell_index`, `binavg_pk_from_shells`, `xi_discrete_multi`, `fit_standard_desilike`, `fit_binavg_minuit`, `eval_pk_dense`, `OneResult`, `_write_results_md`, `main`

### [mission11_acceleration_benchmark.py](../source/background/agent/mission11_log/mission11_acceleration_benchmark.py)

`source/background/agent/mission11_log/mission11_acceleration_benchmark.py`

Mission 11: 全离散求和加速 — 方案对比与基准测试

`j0`, `prepare_shells`, `xi_baseline`, `xi_rebin`, `xi_rebin_o1`, `xi_rebin_o2`, `main`

### [mission11_final_validation.py](../source/background/agent/mission11_log/mission11_final_validation.py)

`source/background/agent/mission11_log/mission11_final_validation.py`

Mission 11: 全离散求和加速 — 最终验证图

`xi_exact_rebin`, `xi_cached_rebin`, `precompute_cache`, `main`

### [mission11_validation_1gpc_quijote.py](../source/background/agent/mission11_log/mission11_validation_1gpc_quijote.py)

`source/background/agent/mission11_log/mission11_validation_1gpc_quijote.py`

Mission 11: k-rebinning acceleration validation on 1Gpc fastPM + Quijote

`_rid_quijote`, `load_pk_quijote`, `load_pcf_quijote`, `precompute_cache`, `xi_exact_rebin`, `xi_cached_rebin`, `main`

### [mission12_p_scan.py](../source/background/agent/mission12_log/mission12_p_scan.py)

`source/background/agent/mission12_log/mission12_p_scan.py`

Mission 12: scan p_fix values, compare P(k) vs 2PCF fit (rmin=56, rmax=380, sigmas free)

`rid`, `load_all`, `gqenum`, `gqfft`

### [mission12_pk_vs_xi_fit.py](../source/background/agent/mission12_log/mission12_pk_vs_xi_fit.py)

`source/background/agent/mission12_log/mission12_pk_vs_xi_fit.py`

Mission 12: 功率谱拟合 vs 2PCF 拟合的 best-fit 参数对比

`_rid`, `load_pk`, `load_pcf`, `j0`, `gq_fft`, `gq_enumerate`, `precompute_rebin_cache`, `fast_discrete_xi0`, `main`

### [mission12_rmin_scan.py](../source/background/agent/mission12_log/mission12_rmin_scan.py)

`source/background/agent/mission12_log/mission12_rmin_scan.py`

Mission 12: scan rmin for 2PCF fit, rmax=380 fixed, sigmas free

`rid`, `ldpk`, `ldxi`, `gqfft`, `gqenum`, `c2pk`

### [mission12_sigmas_fixed.py](../source/background/agent/mission12_log/mission12_sigmas_fixed.py)

`source/background/agent/mission12_log/mission12_sigmas_fixed.py`

Mission 12: P(k) vs 2PCF 拟合对比 (sigmas 固定)

`_rid`, `load_pk`, `load_pcf`, `gq_fft`, `gq_enumerate`, `chi2_pk`, `chi2_xi`

### [mission12_sigmas_free.py](../source/background/agent/mission12_log/mission12_sigmas_free.py)

`source/background/agent/mission12_log/mission12_sigmas_free.py`

Mission 12: P(k) vs 2PCF fit — sigmas free

`rid`, `ldpk`, `ldxi`, `gqfft`, `gqenum`, `c2pk`, `c2xi`

### [plot_mass_function_after_cut.py](../source/background/agent/mission1_log/plot_mass_function_after_cut.py)

`source/background/agent/mission1_log/plot_mass_function_after_cut.py`

脚本名称

`parse_seed_from_name`, `list_1gpc_catalogs`, `list_3gpc_catalogs`, `hmf_counts_from_mass`, `hmf_counts_1gpc_one`, `hmf_counts_3gpc_one`, `mean_hmf_for_group`, `main`

### [fftlog_vs_mcfit_kmax20_curves.py](../source/background/agent/mission3_log/fftlog_vs_mcfit_kmax20_curves.py)

`source/background/agent/mission3_log/fftlog_vs_mcfit_kmax20_curves.py`

脚本用途

`build_log_taper_window`, `build_theory_p0`, `xi_custom_fftlog`, `xi_from_mcfit`, `main`

### [fftlog_vs_mcfit_kmax20_kminscan.py](../source/background/agent/mission3_log/fftlog_vs_mcfit_kmax20_kminscan.py)

`source/background/agent/mission3_log/fftlog_vs_mcfit_kmax20_kminscan.py`

脚本用途（3.2.1 精简检验）

`build_log_taper_window`, `build_theory_p0`, `xi_custom_fftlog`, `xi_from_mcfit`, `signed_relative_error_percent`, `main`

### [fftlog_vs_mcfit_validation.py](../source/background/agent/mission3_log/fftlog_vs_mcfit_validation.py)

`source/background/agent/mission3_log/fftlog_vs_mcfit_validation.py`

脚本大纲（执行逻辑）

`CompareMetrics`, `build_log_taper_window`, `build_theory_p0`, `xi_fftlog_custom`, `xi_mcfit_transform`, `percent_error`, `compute_metrics_for_case`, `save_csv`, `plot_heatmap`, `plot_kmin_scan_line`, `plot_case_kmin_fund`, `main`

### [task421_pk_bestfit_box_compare.py](../source/background/agent/mission4_log/task421_pk_bestfit_box_compare.py)

`source/background/agent/mission4_log/task421_pk_bestfit_box_compare.py`

脚本名称

`PkBoxData`, `parse_realization_id`, `assert_same_grid`, `load_pk_box`, `convert_bestfit_to_float_dict`, `fit_box_pk`, `format_bestfit_text`, `main`

### [task43_fnl0_pk_to_xi_check.py](../source/background/agent/mission4_log/task43_fnl0_pk_to_xi_check.py)

`source/background/agent/mission4_log/task43_fnl0_pk_to_xi_check.py`

脚本名称

`Data3GpcFnl0`, `parse_realization_id`, `assert_same_grid`, `load_data`, `convert_bestfit_to_float_dict`, `fit_best_pk`, `build_log_taper_window`, `xi_custom_fftlog`, `evaluate_xi_on_s`, `main`

### [task4_boxlength_kcut_study.py](../source/background/agent/mission4_log/task4_boxlength_kcut_study.py)

`source/background/agent/mission4_log/task4_boxlength_kcut_study.py`

脚本名称

`BoxConfig`, `BoxData`, `parse_realization_id`, `list_realization_files`, `assert_same_grid`, `load_box_data`, `subset_box_data`, `save_measurement_csv`, `plot_measurement_compare`, `convert_bestfit_to_float_dict`, `fit_box_pk`, `build_log_taper_window`, `build_theory_p0`, `xi_custom_fftlog`, `evaluate_theory_xi_on_data_grid`, `compute_residual_metrics`, `write_dict_rows_csv`, `plot_pk_fit_compare`, `plot_xi_model_compare`, `main`

### [task5_v10_fnl0_pk2xi_diffkmin_1gpc_3gpc.py](../source/background/agent/mission5_log/_replaced_archive/task5_v10_fnl0_pk2xi_diffkmin_1gpc_3gpc.py)

`source/background/agent/mission5_log/_replaced_archive/task5_v10_fnl0_pk2xi_diffkmin_1gpc_3gpc.py`

任务5 v10: 用 model.ipynb 同款方法检查 fnl0 下 best-fit PK -> 2PCF 的一致性，

`BoxConfig`, `Dataset`, `parse_realization_id`, `assert_same_grid`, `load_dataset`, `convert_bestfit_to_float_dict`, `build_fit_mask`, `fit_best_pk`, `build_log_taper_window`, `xi_custom_fftlog`, `model_xi_on_s`, `compute_metrics`, `run_one_box`, `plot_joint`, `write_summary`, `main`

### [task5_v11_fnl0_pk2xi_diffkmin_1gpc_3gpc_jointpanel.py](../source/background/agent/mission5_log/_replaced_archive/task5_v11_fnl0_pk2xi_diffkmin_1gpc_3gpc_jointpanel.py)

`source/background/agent/mission5_log/_replaced_archive/task5_v11_fnl0_pk2xi_diffkmin_1gpc_3gpc_jointpanel.py`

任务5 v11: 用 model.ipynb 同款方法检查 fnl0 下 best-fit PK -> 2PCF 的一致性，

`BoxConfig`, `Dataset`, `parse_realization_id`, `assert_same_grid`, `load_dataset`, `convert_bestfit_to_float_dict`, `build_fit_mask`, `fit_best_pk`, `build_log_taper_window`, `xi_custom_fftlog`, `model_xi_on_s`, `compute_metrics`, `run_one_box`, `plot_joint`, `write_summary`, `main`

### [task5_v9_fnl0_pk2xi_1gpc_3gpc_check.py](../source/background/agent/mission5_log/_replaced_archive/task5_v9_fnl0_pk2xi_1gpc_3gpc_check.py)

`source/background/agent/mission5_log/_replaced_archive/task5_v9_fnl0_pk2xi_1gpc_3gpc_check.py`

任务5 v9: 用 model.ipynb 同款方法检查 fnl0 下 best-fit PK -> 2PCF 的一致性。

`BoxConfig`, `Dataset`, `parse_realization_id`, `assert_same_grid`, `load_dataset`, `convert_bestfit_to_float_dict`, `fit_best_pk`, `build_log_taper_window`, `xi_custom_fftlog`, `model_xi_on_s`, `compute_metrics`, `run_one_box`, `plot_joint`, `write_summary`, `main`

### [task5_compare_1gpc_3gpc_fastpm_2pcf_fourlines.py](../source/background/agent/mission5_log/task5_compare_1gpc_3gpc_fastpm_2pcf_fourlines.py)

`source/background/agent/mission5_log/task5_compare_1gpc_3gpc_fastpm_2pcf_fourlines.py`

脚本名称

`XiDataset`, `realization_id`, `read_pcf`, `load_dataset`, `main`

### [task5_compare_fastpm_measured_vs_baseline_4panel.py](../source/background/agent/mission5_log/task5_compare_fastpm_measured_vs_baseline_4panel.py)

`source/background/agent/mission5_log/task5_compare_fastpm_measured_vs_baseline_4panel.py`

脚本名称

`build_baseline_xi_for_dataset`, `build_window_xi_for_dataset`, `plot_panel_figure`, `main`

### [task5_ir_window_solution_3gpc_multitype.py](../source/background/agent/mission5_log/task5_ir_window_solution_3gpc_multitype.py)

`source/background/agent/mission5_log/task5_ir_window_solution_3gpc_multitype.py`

脚本名称

`MockData`, `parse_realization_id`, `list_realization_files`, `assert_same_grid`, `load_mock_data`, `build_fiducial_cosmology`, `convert_bestfit_to_float_dict`, `fit_best_pk`, `build_theory_p0`, `build_log_taper_window`, `build_ir_window`, `xi_fftlog_from_effective_p0`, `interp_xi_to_s`, `compute_alignment_metrics`, `write_dict_rows_csv`, `evaluate_unwindowed_with_kmin`, `evaluate_testb_single_window`, `generate_discrete_k_modes`, `evaluate_testc_hybrid`, `plot_fig1_pk`, `plot_fig2_r2xi_compare`, `plot_fig3_residual_sigma`, `plot_fig4_fnl0_sanity`, `main`

### [task5_v12_fnl0_pk2xi_boxkmin_kmax20_jointpanel.py](../source/background/agent/mission5_log/task5_v12_fnl0_pk2xi_boxkmin_kmax20_jointpanel.py)

`source/background/agent/mission5_log/task5_v12_fnl0_pk2xi_boxkmin_kmax20_jointpanel.py`

任务5 v12: 用 model.ipynb 同款方法检查 fnl0 下 best-fit PK -> 2PCF 的一致性，

`BoxConfig`, `Dataset`, `parse_realization_id`, `assert_same_grid`, `load_dataset`, `convert_bestfit_to_float_dict`, `build_fit_mask`, `fit_best_pk`, `build_log_taper_window`, `xi_custom_fftlog`, `model_xi_on_s`, `compute_metrics`, `run_one_box`, `plot_joint`, `write_summary`, `main`

### [task5_v13_1gpc_joint_fnl0_fnl100_testb_scan.py](../source/background/agent/mission5_log/task5_v13_1gpc_joint_fnl0_fnl100_testb_scan.py)

`source/background/agent/mission5_log/task5_v13_1gpc_joint_fnl0_fnl100_testb_scan.py`

脚本名称

`fit_best_pk_safe`, `evaluate_unwindowed_1gpc`, `evaluate_testb_1gpc_single_window`, `main`

### [task5_v14_3gpc_joint_fnl0_fnl100_testb_scan.py](../source/background/agent/mission5_log/task5_v14_3gpc_joint_fnl0_fnl100_testb_scan.py)

`source/background/agent/mission5_log/task5_v14_3gpc_joint_fnl0_fnl100_testb_scan.py`

脚本名称

`fit_best_pk_safe`, `evaluate_unwindowed_3gpc`, `evaluate_testb_3gpc_single_window`, `main`

### [task5_v15_3gpc_use_1gpc_window_eval150.py](../source/background/agent/mission5_log/task5_v15_3gpc_use_1gpc_window_eval150.py)

`source/background/agent/mission5_log/task5_v15_3gpc_use_1gpc_window_eval150.py`

脚本名称

`compute_metrics_with_rmin`, `main`

### [task5_v16_3gpc_smallscale_priority_scan.py](../source/background/agent/mission5_log/task5_v16_3gpc_smallscale_priority_scan.py)

`source/background/agent/mission5_log/task5_v16_3gpc_smallscale_priority_scan.py`

任务5 v16：3Gpc 小尺度优先的联合窗口扫描（fnl0 + fnl100）

`compute_metrics_range`, `eval_window_pair`, `main`

### [task5_v17_3gpc_smallscale_deep_scan.py](../source/background/agent/mission5_log/task5_v17_3gpc_smallscale_deep_scan.py)

`source/background/agent/mission5_log/task5_v17_3gpc_smallscale_deep_scan.py`

任务5 v17：3Gpc 小尺度深扫（TESTB, direct-P window）

`compute_metric`, `eval_pair`, `render_single_curve`, `fmt_win`, `main`

### [task5_v18_sigma_scan_modelonly_dense_r.py](../source/background/agent/mission5_log/task5_v18_sigma_scan_modelonly_dense_r.py)

`source/background/agent/mission5_log/task5_v18_sigma_scan_modelonly_dense_r.py`

任务5 v18：纯模型 sigma 扫描（密集 r 网格）

`compute_xi_curve_for_sigma`, `summarize_large_scale_spread`, `main`

### [task5_v19_testc_3gpc_fnl100_check.py](../source/background/agent/mission5_log/task5_v19_testc_3gpc_fnl100_check.py)

`source/background/agent/mission5_log/task5_v19_testc_3gpc_fnl100_check.py`

任务5 v19：仅测试 TestC（3Gpc, fnl100）是否能解决 2PCF 建模偏差

`compute_metric_range`, `fmt_metric`, `main`

### [task5_v20_exp_power_L_scaling_eval150.py](../source/background/agent/mission5_log/task5_v20_exp_power_L_scaling_eval150.py)

`source/background/agent/mission5_log/task5_v20_exp_power_L_scaling_eval150.py`

任务5 v20：TESTB 固定窗口形状 exp_power(x)，测试 x 与盒长 L 的关系（评估 r>=150）

`compute_metric_rge`, `evaluate_exp_x_for_1gpc`, `evaluate_exp_x_for_3gpc`, `main`

### [task5_v21_summary_pk_xi_by_box.py](../source/background/agent/mission5_log/task5_v21_summary_pk_xi_by_box.py)

`source/background/agent/mission5_log/task5_v21_summary_pk_xi_by_box.py`

任务5 v21：按盒长分图的总结图（1Gpc / 3Gpc）

`metric_rge150`, `make_one_box_figure`, `main`

### [task5_v22_3gpc_ezcov_bestwindow_kmax008.py](../source/background/agent/mission5_log/task5_v22_3gpc_ezcov_bestwindow_kmax008.py)

`source/background/agent/mission5_log/task5_v22_3gpc_ezcov_bestwindow_kmax008.py`

任务5 v22：3Gpc 用 EZmock 协方差 + 当前最佳窗口检查 2PCF 建模

`parse_seed`, `load_ezmock_cov_for_k`, `fit_best_pk_with_external_cov`, `metric_range`, `fmt`, `main`

### [task5_v3_p_scan.py](../source/background/agent/mission5_log/task5_v3_p_scan.py)

`source/background/agent/mission5_log/task5_v3_p_scan.py`

脚本名称

`write_rows_csv`, `plot_fnl_vs_p`, `plot_metric_vs_p`, `main`

### [task5_v4_window_param_sensitivity.py](../source/background/agent/mission5_log/task5_v4_window_param_sensitivity.py)

`source/background/agent/mission5_log/task5_v4_window_param_sensitivity.py`

脚本名称

`family_sensitivity_stats`, `main`

### [task5_v5_1gpc_transfer_test.py](../source/background/agent/mission5_log/task5_v5_1gpc_transfer_test.py)

`source/background/agent/mission5_log/task5_v5_1gpc_transfer_test.py`

脚本名称

`evaluate_unwindowed_xi`, `evaluate_testb_fixed_xi`, `main`

### [task5_v6_1gpc_tanhlog_param_scan.py](../source/background/agent/mission5_log/task5_v6_1gpc_tanhlog_param_scan.py)

`source/background/agent/mission5_log/task5_v6_1gpc_tanhlog_param_scan.py`

脚本名称

`evaluate_unwindowed_xi`, `evaluate_tanhlog_window_xi`, `main`

### [task5_v7_1gpc_p_target_then_model.py](../source/background/agent/mission5_log/task5_v7_1gpc_p_target_then_model.py)

`source/background/agent/mission5_log/task5_v7_1gpc_p_target_then_model.py`

脚本名称

`evaluate_unwindowed_xi`, `evaluate_tanhlog_window_xi`, `plot_p_scan_fnl`, `plot_beta_scan`, `plot_curves_compare`, `main`

### [task5_v8_fnl0_1gpc_vs_3gpc_2pcf_compare.py](../source/background/agent/mission5_log/task5_v8_fnl0_1gpc_vs_3gpc_2pcf_compare.py)

`source/background/agent/mission5_log/task5_v8_fnl0_1gpc_vs_3gpc_2pcf_compare.py`

任务5 v8: 比较 fnl0 下 1Gpc 与 3Gpc 的 2PCF 测量均值。

`XiDataset`, `realization_id`, `read_pcf`, `load_dataset`, `main`

### [mission6_nbody_validation_pipeline.py](../source/background/agent/mission6_log/mission6_nbody_validation_pipeline.py)

`source/background/agent/mission6_log/mission6_nbody_validation_pipeline.py`

Mission 6: Quijote N-body 样本上的参数化窗口验证（重做版）。

`DatasetSpec`, `PkEnsemble`, `parse_realization_id`, `fnl_tag`, `ensure_parent`, `save_tsv`, `original_dataset_specs`, `dataset_spec_for_fnl`, `_list_realization_files`, `_load_triplet_arrays`, `_write_interpolated_file`, `build_interpolated_datasets`, `read_single_pk_file`, `load_pk_ensemble`, `filter_singular_pk_bins`, `build_pypower_data_and_mocks`, `read_pcf_mean_std`, `build_log_taper_window`, `build_ir_param_window`, `xi0_from_p0_fftlog`, `_to_jsonable_params`, `run_single_validation`, `save_summary`, `build_argparser`, `main`

### [masscut_model_validate.py](../source/background/agent/mission7_masscut/scripts/masscut_model_validate.py)

`source/background/agent/mission7_masscut/scripts/masscut_model_validate.py`

Mission 7: 不同 halo mass_min 的 2PCF 建模验证（FastPM 1Gpc fnl100）。

`parse_realization_id`, `read_single_pk_file`, `PkEnsemble`, `load_pk_ensemble`, `filter_singular_pk_bins`, `build_pypower_data_and_mocks`, `build_log_taper_window`, `build_ir_param_window`, `xi0_from_p0_fftlog`, `read_pcf_mean_std`, `_to_jsonable_params`, `main`

### [plot_masscut_model_overlay.py](../source/background/agent/mission7_masscut/scripts/plot_masscut_model_overlay.py)

`source/background/agent/mission7_masscut/scripts/plot_masscut_model_overlay.py`

Overlay "window" 2PCF modeling curves for multiple masscuts on a single figure.

`build_log_taper_window`, `build_ir_param_window_normexp`, `xi0_from_p0_fftlog`, `_discover_tags`, `_load_bestfit_params`, `_compute_xi_window_on_rgrid`, `_read_pcf_mean_std`, `main`

### [enumerate_k_shells.py](../source/background/agent/mission8_log/enumerate_k_shells.py)

`source/background/agent/mission8_log/enumerate_k_shells.py`

Mission 8：三维周期盒离散 k-shell 枚举工具

`ShellRecord`, `k_fundamental`, `is_sum_of_three_squares`, `enumerate_shells`, `shell_average_power`, `radial_delta_coefficients`, `spherical_bessel_j0`, `xi0_from_discrete_shells`, `toy_png_like_power`, `print_shell_table`, `main`

### [mission8_all_discrete_1gpc.py](../source/background/agent/mission8_log/mission8_all_discrete_1gpc.py)

`source/background/agent/mission8_log/mission8_all_discrete_1gpc.py`

Mission 8：1Gpc 的“全离散”测试

`spherical_bessel_j0`, `compute_shell_degeneracy_fft`, `xi_from_full_discrete_shell_sum`, `build_metric_row`, `main`

### [mission8_all_discrete_3gpc.py](../source/background/agent/mission8_log/mission8_all_discrete_3gpc.py)

`source/background/agent/mission8_log/mission8_all_discrete_3gpc.py`

Mission 8：3Gpc 的“全离散”测试

`spherical_bessel_j0`, `compute_shell_degeneracy_fft`, `xi_from_full_discrete_shell_sum`, `build_metric_row`, `main`

### [mission8_all_discrete_gaussian_delta.py](../source/background/agent/mission8_log/mission8_all_discrete_gaussian_delta.py)

`source/background/agent/mission8_log/mission8_all_discrete_gaussian_delta.py`

Mission 8：把离散壳层的 delta 函数改成窄高斯

`BoxConfig`, `spherical_bessel_j0`, `compute_shell_degeneracy_fft`, `build_metric_row`, `xi_from_full_discrete_with_gaussian_options`, `main`

### [mission8_all_discrete_kmax_compare.py](../source/background/agent/mission8_log/mission8_all_discrete_kmax_compare.py)

`source/background/agent/mission8_log/mission8_all_discrete_kmax_compare.py`

Mission 8：全离散方法的 kmax=12 vs kmax=20 对比

`BoxConfig`, `spherical_bessel_j0`, `compute_shell_degeneracy_fft`, `xi_from_full_discrete_shell_sum`, `build_metric_row`, `load_old_metric`, `main`

### [mission8_all_discrete_modecount_cell.py](../source/background/agent/mission8_log/mission8_all_discrete_modecount_cell.py)

`source/background/agent/mission8_log/mission8_all_discrete_modecount_cell.py`

Mission 8：用 mode-count preserving k-cell 替代 delta-shell

`BoxConfig`, `compute_shell_degeneracy_fft`, `spherical_bessel_j0`, `build_metric_row`, `cell_kernel_center`, `xi_discrete_and_modecount_cell_for_two_models`, `main`

### [mission8_all_discrete_mu_shellavg.py](../source/background/agent/mission8_log/mission8_all_discrete_mu_shellavg.py)

`source/background/agent/mission8_log/mission8_all_discrete_mu_shellavg.py`

Mission 8：原始全离散的低-k 离散 mu-shell 平均修正

`BoxConfig`, `compute_shell_degeneracy_fft`, `spherical_bessel_j0`, `build_metric_row`, `build_lowq_shellavg_p`, `xi_full_discrete_with_lowq_override`, `main`

### [mission8_all_discrete_rbinavg.py](../source/background/agent/mission8_log/mission8_all_discrete_rbinavg.py)

`source/background/agent/mission8_log/mission8_all_discrete_rbinavg.py`

Mission 8：原始全离散的径向 bin 平均修正

`BoxConfig`, `compute_shell_degeneracy_fft`, `spherical_bessel_j0`, `j0_volume_bin_average`, `load_pcf_bin_edges`, `xi_center_and_binavg_for_two_models`, `build_metric_row`, `main`

### [mission8_discrete_shell_hybrid_analysis.py](../source/background/agent/mission8_log/mission8_discrete_shell_hybrid_analysis.py)

`source/background/agent/mission8_log/mission8_discrete_shell_hybrid_analysis.py`

Mission 8：离散 shell 解析解释 + hybrid 数值验证

`ShellInfo`, `spherical_bessel_j0`, `enumerate_shells`, `save_shell_table`, `xi_low_from_shells`, `xi_shell_hybrid`, `build_metric_row`, `main`

### [mission8_exp_window_box_finish.py](../source/background/agent/mission8_log/mission8_exp_window_box_finish.py)

`source/background/agent/mission8_log/mission8_exp_window_box_finish.py`

Mission 8 收尾：按 exp_power 窗口重做 3Gpc 与 1Gpc fastPM

`BoxConfig`, `ShellInfo`, `exp_window_power`, `filter_zero_std_pk_bins`, `spherical_bessel_j0`, `enumerate_shells`, `xi_low_from_shells`, `xi_shell_hybrid`, `xi_shell_hybrid_png_only`, `evaluate_exp_window_model`, `build_metric_row`, `main`

### [mission8_fnl0_modeling_kmax20.py](../source/background/agent/mission8_log/mission8_fnl0_modeling_kmax20.py)

`source/background/agent/mission8_log/mission8_fnl0_modeling_kmax20.py`

Mission 8：fnl=0 的 2PCF 建模图（kmax=20）

`BoxConfig`, `spherical_bessel_j0`, `compute_shell_degeneracy_fft`, `xi_from_full_discrete_shell_sum`, `build_metric_row`, `main`

### [mission8_measured_pk_1gpc_cutkdata.py](../source/background/agent/mission8_log/mission8_measured_pk_1gpc_cutkdata.py)

`source/background/agent/mission8_log/mission8_measured_pk_1gpc_cutkdata.py`

Mission 8：1Gpc 直接用测量 P(k) 建模，且只积到测量 k_max

`compute_shell_degeneracy_fft`, `spherical_bessel_j0`, `build_metric_row`, `xi_from_shellsum_with_pfunc`, `main`

### [mission8_measured_pk_1gpc_k5.py](../source/background/agent/mission8_log/mission8_measured_pk_1gpc_k5.py)

`source/background/agent/mission8_log/mission8_measured_pk_1gpc_k5.py`

Mission 8：1Gpc 直接用重测 P(k)（kmax≈5）做全离散 2PCF 建模

`compute_shell_degeneracy_fft`, `spherical_bessel_j0`, `xi_from_shellsum_with_pfunc`, `build_metric_row`, `require_inputs`, `summarize_best`, `main`

### [mission8_measured_pk_direct_existingbins.py](../source/background/agent/mission8_log/mission8_measured_pk_direct_existingbins.py)

`source/background/agent/mission8_log/mission8_measured_pk_direct_existingbins.py`

Mission 8：直接用现有测量 P(k) 做全离散建模

`BoxConfig`, `compute_shell_degeneracy_fft`, `spherical_bessel_j0`, `build_metric_row`, `evaluate_measured_pk_at_shells_interp`, `evaluate_measured_pk_at_shells_step`, `xi_from_measured_pk_direct`, `main`

### [mission8_mu_modesum_p024.py](../source/background/agent/mission8_log/mission8_mu_modesum_p024.py)

`source/background/agent/mission8_log/mission8_mu_modesum_p024.py`

Mission 8：把低-k 的 mu 显式纳入，再试一次

`BoxConfig`, `ModeRecord`, `legendre_l2`, `legendre_l4`, `spherical_bessel_j0`, `build_theory_p024`, `reconstruct_pkmu_from_poles`, `enumerate_modes_by_shell`, `xi_png_low_modesum_A`, `xi_png_low_shellavg_B`, `xi_modepng_hybrid`, `build_metric_row`, `main`

### [mission8_pngsplit_ksplit005.py](../source/background/agent/mission8_log/mission8_pngsplit_ksplit005.py)

`source/background/agent/mission8_log/mission8_pngsplit_ksplit005.py`

Mission 8：PNG 低-k 离散，高-k 连续/忽略，对比 k_split=0.05

`BoxConfig`, `spherical_bessel_j0`, `compute_shell_degeneracy_fft`, `xi_from_discrete_shell_sum`, `build_metric_row`, `xi_ref_continuous_no_png`, `xi_png_continuous_highk`, `main`

### [mission8_pngsplit_ksplit005_kbase.py](../source/background/agent/mission8_log/mission8_pngsplit_ksplit005_kbase.py)

`source/background/agent/mission8_log/mission8_pngsplit_ksplit005_kbase.py`

Mission 8：把 no-PNG 主体积分下限改成 k_base=2pi/L

`xi_ref_continuous_from_kbase`, `main`

### [mission8_pngsplit_refcont_full_disc.py](../source/background/agent/mission8_log/mission8_pngsplit_refcont_full_disc.py)

`source/background/agent/mission8_log/mission8_pngsplit_refcont_full_disc.py`

Mission 8：连续 no-PNG 主体 + 全离散 PNG 增量

`BoxConfig`, `spherical_bessel_j0`, `compute_shell_degeneracy_fft`, `xi_from_full_discrete_shell_sum`, `build_metric_row`, `continuous_ref_xi`, `main`

### [toy_ir_discrete_vs_continuous.py](../source/background/agent/mission8_log/toy_ir_discrete_vs_continuous.py)

`source/background/agent/mission8_log/toy_ir_discrete_vs_continuous.py`

Mission 8：toy IR 检查

`spherical_bessel_j0`, `toy_power_ir_strong`, `xi0_continuous`, `main`

### [mission9_1gpc_kmax.py](../source/background/agent/mission9_log/mission9_1gpc_kmax.py)

`source/background/agent/mission9_log/mission9_1gpc_kmax.py`

Mission 9: 1Gpc fnl100 kmax 扫描（独立脚本）

`rid`, `load`, `load_pcf`, `j0`, `gq_fft`, `xi_disc`, `xi_fftlog`, `met`, `main`

### [mission9_databin_optimize.py](../source/background/agent/mission9_log/mission9_databin_optimize.py)

`source/background/agent/mission9_log/mission9_databin_optimize.py`

Mission 9: DataBin 优化（kmax 扫描 + sn0 组合）

`rid`, `load_pk`, `load_pcf`, `j0`, `gq_fft`, `xi_disc_pfunc`, `xi_fftlog`, `ir_window_func`, `met`, `main`

### [mission9_exploration_3gpc.py](../source/background/agent/mission9_log/mission9_exploration_3gpc.py)

`source/background/agent/mission9_log/mission9_exploration_3gpc.py`

Mission 9 探索：全离散高估原因诊断 + 通用校正方案（3Gpc fnl100）

`parse_rid`, `load_pk`, `load_pcf`, `j0`, `compute_gq`, `compute_gq_cont_vectorized`, `xi_discrete`, `xi_fftlog`, `ir_window`, `metrics`, `main`

### [mission9_fastpm_fnl0.py](../source/background/agent/mission9_log/mission9_fastpm_fnl0.py)

`source/background/agent/mission9_log/mission9_fastpm_fnl0.py`

Mission 9: FastPM fnl=0 验证 DataBin

`rid`, `load_pk`, `load_pcf`, `j0`, `gq_fft`, `xi_disc`, `xi_fftlog`, `ir_win_func`, `met`, `main`

### [mission9_fnl0_sn0free.py](../source/background/agent/mission9_log/mission9_fnl0_sn0free.py)

`source/background/agent/mission9_log/mission9_fnl0_sn0free.py`

Mission 9 支线: fnl=0 + sn0 释放 + kfit 扫描

`rid`, `load_pk`, `load_pcf`, `load_ez`, `j0`, `gq_fft`, `xi_disc`, `met`, `met_range`, `main`

### [mission9_kfit_scan.py](../source/background/agent/mission9_log/mission9_kfit_scan.py)

`source/background/agent/mission9_log/mission9_kfit_scan.py`

Mission 9 支线: 拟合 kmax 扫描 + EZmock covariance

`rid`, `load_pk_fastpm`, `load_pcf`, `load_ezmock_pk`, `j0`, `gq_fft`, `xi_disc`, `met`, `met_range`, `main`

### [mission9_kmax_scan.py](../source/background/agent/mission9_log/mission9_kmax_scan.py)

`source/background/agent/mission9_log/mission9_kmax_scan.py`

Mission 9: kmax 扫描 + 1Gpc 对比

`parse_rid`, `load_pk`, `load_pcf`, `j0`, `compute_gq`, `xi_disc_partial`, `xi_fftlog`, `ir_window`, `metrics`, `main`

### [mission9_lattice_analysis.py](../source/background/agent/mission9_log/mission9_lattice_analysis.py)

`source/background/agent/mission9_log/mission9_lattice_analysis.py`

Mission 9 探索（A部分）：格点模式超额分析

`enumerate_shells_direct`, `read_pk_kbins`, `parse_rid`, `main`

### [mission9_mcmc.py](../source/background/agent/mission9_log/mission9_mcmc.py)

`source/background/agent/mission9_log/mission9_mcmc.py`

Mission 9/12: MCMC 对比 P(k) vs 2PCF 后验分布

`rid`, `load_pk`, `load_pcf`, `load_ez`, `precompute_rebin`, `fast_xi0`, `main`

### [mission9_mw_fitting.py](../source/background/agent/mission9_log/mission9_mw_fitting.py)

`source/background/agent/mission9_log/mission9_mw_fitting.py`

Mission 9: Mode-Weighted Fitting 实验

`rid`, `load_pk`, `load_pcf`, `j0`, `gq_fft`, `xi_disc`, `met`, `enumerate_shells_for_bins`, `main`

### [mission9_new_ideas.py](../source/background/agent/mission9_log/mission9_new_ideas.py)

`source/background/agent/mission9_log/mission9_new_ideas.py`

Mission 9: 新方向探索

`rid`, `load_pk`, `load_pcf`, `j0`, `gq_fft`, `xi_disc_custom_p`, `met`, `xi_fftlog`, `ir_window`, `main`

### [mission9_pk_vs_xi_fit.py](../source/background/agent/mission9_log/mission9_pk_vs_xi_fit.py)

`source/background/agent/mission9_log/mission9_pk_vs_xi_fit.py`

Mission 9/12: P(k) 拟合 vs 2PCF 拟合 对比

`rid`, `load_pk`, `load_pcf`, `load_ez`, `gq_fft`, `precompute_rebin`, `fast_xi0`, `fast_xi0_databin`, `gq_enumerate`, `build_bin_shells`, `met`, `main`

### [mission9_plot_databin.py](../source/background/agent/mission9_log/mission9_plot_databin.py)

`source/background/agent/mission9_log/mission9_plot_databin.py`

Mission 9: DataBin 方法结果图

`rid`, `load_pk`, `load_pcf`, `j0`, `gq_fft`, `xi_disc_custom_p`, `xi_fftlog`, `ir_window_func`, `met`, `main`

### [mission9_quijote_plots.py](../source/background/agent/mission9_log/mission9_quijote_plots.py)

`source/background/agent/mission9_log/mission9_quijote_plots.py`

Mission 9: Quijote 2PCF 建模对比图

`rid`, `load_pk`, `load_pcf`, `j0`, `gq_fft`, `xi_disc`, `xi_fftlog`, `ir_win_func`, `met`, `main`

### [mission9_quijote_validation.py](../source/background/agent/mission9_log/mission9_quijote_validation.py)

`source/background/agent/mission9_log/mission9_quijote_validation.py`

Mission 9: Quijote N-body validation of DataBin method

`rid`, `load_pk_quijote`, `load_pcf_quijote`, `j0`, `gq_fft`, `xi_disc`, `xi_fftlog`, `ir_window_func`, `met`, `main`

### [read_fastpm_rsd_batch.py](../source/background/agent/skills/fastpm-halo-rsd-export/scripts/read_fastpm_rsd_batch.py)

`source/background/agent/skills/fastpm-halo-rsd-export/scripts/read_fastpm_rsd_batch.py`

批量读取 FastPM FOF halo，并导出 RSD 后位置文本。

`load_header_meta`, `apply_rsd`, `process_one_realization`, `main`

### [calc_2pcf_fastpm.py](../source/background/agent/skills/fastpm-pk-2pcf/scripts/calc_2pcf_fastpm.py)

`source/background/agent/skills/fastpm-pk-2pcf/scripts/calc_2pcf_fastpm.py`

简化版 2PCF 脚本（FCFC_2PT_BOX）。



### [calc_powspec_cov_mock.py](../source/background/agent/skills/fastpm-pk-2pcf/scripts/calc_powspec_cov_mock.py)

`source/background/agent/skills/fastpm-pk-2pcf/scripts/calc_powspec_cov_mock.py`

批量计算 cov_mock 目录下 EZmock RSD 位置文件的功率谱（POWSPEC）。

`seed_from_dirname`, `is_seed_10i_plus_5`, `split_for_array`, `run_one`, `main`

### [calc_powspec_fastpm.py](../source/background/agent/skills/fastpm-pk-2pcf/scripts/calc_powspec_fastpm.py)

`source/background/agent/skills/fastpm-pk-2pcf/scripts/calc_powspec_fastpm.py`

简化版功率谱脚本（POWSPEC）。



### [hybrid_scan.py](../source/background/pk-pcf-model/hybrid_scan.py)

`source/background/pk-pcf-model/hybrid_scan.py`

Hybrid 方案扫描：低 k 离散求和 + 高 k FFTLog 连续积分

`parse_rid`, `load_pk`, `load_pcf`, `eval_pk_grid`, `gq_enumerate`, `chi2_binavg`, `j0`, `gq_fft_fn`, `fftlog_xi0`, `hybrid_xi0`

### [base.py](../source/external/desilike/theories/galaxy_clustering/base.py)

`source/external/desilike/theories/galaxy_clustering/base.py`



`BaseTheoryPowerSpectrumMultipoles`, `BaseTheoryCorrelationFunctionMultipoles`, `BaseTheoryCorrelationFunctionFromPowerSpectrumMultipoles`, `BaseTheoryPowerSpectrumMultipolesFromWedges`, `ap_k_mu`, `ap_s_mu`, `APEffect`

### [power_template.py](../source/external/desilike/theories/galaxy_clustering/power_template.py)

`source/external/desilike/theories/galaxy_clustering/power_template.py`



`_bcast_shape`, `BasePowerSpectrumExtractor`, `BasePowerSpectrumTemplate`, `FixedPowerSpectrumTemplate`, `DirectPowerSpectrumTemplate`, `BAOExtractor`, `BAOPowerSpectrumTemplate`, `BAOPhaseShiftExtractor`, `_interp`, `BAOPhaseShiftPowerSpectrumTemplate`, `StandardPowerSpectrumExtractor`, `StandardPowerSpectrumTemplate`, `ShapeFitPowerSpectrumExtractor`, `ShapeFitPowerSpectrumTemplate`, `BandVelocityPowerSpectrumExtractor`, `BandVelocityPowerSpectrumCalculator`, `BandVelocityPowerSpectrumTemplate`, `_kernel_tophat_lowx`, `_kernel_tophat_highx`, `kernel_tophat2`, `_kernel_tophat_deriv_lowx`, `_kernel_tophat_deriv_highx`, `kernel_tophat2_deriv`, `kernel_gauss2`, `kernel_gauss2_deriv`, `integrate_sigma_r2`, `WiggleSplitPowerSpectrumExtractor`, `WiggleSplitPowerSpectrumTemplate`, `find_turn_over`, `TurnOverPowerSpectrumExtractor`, `TurnOverPowerSpectrumTemplate`, `DirectWiggleSplitPowerSpectrumTemplate`

### [primordial_non_gaussianity.py](../source/external/desilike/theories/galaxy_clustering/primordial_non_gaussianity.py)

`source/external/desilike/theories/galaxy_clustering/primordial_non_gaussianity.py`



`PNGTracerPowerSpectrumMultipoles`

### [task41_rawbox_norsd_fnl100_profiler.py](../source/project/codes/task4/task41_rawbox_norsd_fnl100_profiler.py)

`source/project/codes/task4/task41_rawbox_norsd_fnl100_profiler.py`

代码大纲（执行逻辑关系）：

`ensure_output_tree`, `realization_label`, `load_xi_mean_payload`, `load_pk_ensemble`, `build_cosmology`, `build_template_arrays`, `interp_logk`, `evaluate_realspace_png_pk`, `gq_enumerate`, `build_bin_shell_index`, `compute_binavg_pk`, `fit_pk_reference`, `gq_fft`, `precompute_rebin_cache`, `fast_discrete_xi`, `covariance_corrections`, `build_bin_selections`, `run_xi_profiler`, `evaluate_acceptance`, `main`

### [task41_scan_fastpm_mass_range.py](../source/project/codes/task4/task41_scan_fastpm_mass_range.py)

`source/project/codes/task4/task41_scan_fastpm_mass_range.py`

Scan the original FastPM halo masses used by Task4.1 inputs.

`parse_args`, `load_realizations`, `input_path`, `scan_one`, `main`

### [task45_compare_free_sn0_alltags.py](../source/project/codes/task4/task45_compare_free_sn0_alltags.py)

`source/project/codes/task4/task45_compare_free_sn0_alltags.py`

代码大纲：

`load_json`, `index_manifest`, `median`, `err_text`, `build_rows`, `aggregate`, `write_csv`, `write_markdown`, `main`

### [task45_fnl50_pk_xi_r50_emcee.py](../source/project/codes/task4/task45_fnl50_pk_xi_r50_emcee.py)

`source/project/codes/task4/task45_fnl50_pk_xi_r50_emcee.py`

Task4.1 Quijote fNL=50：重新运行 P(k) 与 2PCF r=50--350 的长链 emcee。

`FastLikelihood`, `to_jsonable`, `write_json`, `log_prior`, `log_prob_worker`, `configure_task45`, `build_projection`, `build_fast_likelihood`, `load_old_weighted_samples`, `optimize_start`, `initialize_walkers`, `equivalence_audit`, `split_rhat`, `convergence_diagnostics`, `summarize_samples`, `run_case`, `load_postburn_samples`, `constraint_text`, `shared_axis_limits`, `plot_task43_ppt_style`, `parse_args`, `main`

### [task45_plot_completed6_constraints.py](../source/project/codes/task4/task45_plot_completed6_constraints.py)

`source/project/codes/task4/task45_plot_completed6_constraints.py`

Plot a temporary constraints comparison for the six Task45 30k results that

`CompletedCase`, `weighted_quantile`, `contour_levels`, `load_case`, `parameter_ranges`, `plot_corner`, `plot_forest`, `main`

### [task45_plot_fnl50_meeting_compact.py](../source/project/codes/task4/task45_plot_fnl50_meeting_compact.py)

`source/project/codes/task4/task45_plot_fnl50_meeting_compact.py`

重画 Task4.1 fNL=50 P(k) vs 2PCF 的紧凑会议版 contour。

`sha256`, `jsonable`, `read_json`, `load_samples`, `constraint_text`, `shared_axis_limits`, `audit_pk_fit_range`, `halve_axis_height`, `main`

### [task45_plot_free_sn0_fidonly.py](../source/project/codes/task4/task45_plot_free_sn0_fidonly.py)

`source/project/codes/task4/task45_plot_free_sn0_fidonly.py`

代码大纲：

`load_json`, `load_fixed_fid`, `load_free_fid`, `param_interval`, `write_comparison_csv`, `plot_fidonly`, `write_manifest`, `main`

### [task45_plot_free_sn0_kmax_overlay.py](../source/project/codes/task4/task45_plot_free_sn0_kmax_overlay.py)

`source/project/codes/task4/task45_plot_free_sn0_kmax_overlay.py`

代码大纲（执行逻辑关系）：

`load_manifest`, `index_summaries`, `tagged_row`, `parse_args`, `parse_csv_choice`, `build_rows`, `format_constraint_text`, `plot_rows`, `main`

### [task45_plot_lcp50_meeting.py](../source/project/codes/task4/task45_plot_lcp50_meeting.py)

`source/project/codes/task4/task45_plot_lcp50_meeting.py`

Meeting version of Task45 UltraNest constraints for Quijote LCp50 only.

`load_summaries`, `padded_limits`, `plot`

### [task45_quijote_ultranest.py](../source/project/codes/task4/task45_quijote_ultranest.py)

`source/project/codes/task4/task45_quijote_ultranest.py`

代码大纲（执行逻辑关系）：

`LikelihoodContext`, `ensure_dirs`, `parse_csv`, `parse_pair`, `configure_run`, `hartlap_precision`, `weighted_quantile`, `resample_equal_weight`, `make_pk_context`, `make_xi_context`, `build_contexts`, `prior_transform`, `loglike_from_context`, `summarize_result`, `save_samples`, `run_one`, `load_sample_npz`, `plot_corner_for_summary`, `plot_forest`, `write_tables`, `parse_args`, `main`

### [task45_run_free_sn0_kmax_scan_30k_parallel.py](../source/project/codes/task4/task45_run_free_sn0_kmax_scan_30k_parallel.py)

`source/project/codes/task4/task45_run_free_sn0_kmax_scan_30k_parallel.py`

代码大纲（执行逻辑关系）：

`RunSpec`, `build_specs`, `task45_env`, `run_command_for_spec`, `summary_ncall`, `run_specs_parallel`, `collect_root`, `write_steps_table`, `make_overlay`, `parse_args`, `main`

### [task4p2_fit_2pcf_shellavg_fixed_sn0.py](../source/project/codes/task4/task4p2_pk_rawbox_subbox/task4p2_fit_2pcf_shellavg_fixed_sn0.py)

`source/project/codes/task4/task4p2_pk_rawbox_subbox/task4p2_fit_2pcf_shellavg_fixed_sn0.py`

Task4.2：用 volume shell-average 重新拟合 rawbox/subbox 2PCF。

`FitCase`, `ShellBasis`, `file_sha256`, `import_module`, `recover_shell_edges`, `build_precision`, `cluster_covariance_audit`, `load_balanced_subbox_stack`, `load_rawbox_case`, `load_subbox_cases`, `shell_j0_average`, `project_shell_basis`, `sigma_basis`, `build_shared_bases`, `coefficients`, `model_xi`, `log_prior`, `make_log_prob`, `optimize_case`, `summarize_samples`, `convergence_diagnostics`, `run_case`, `parse_args`, `main`

### [task4p2_fit_pk_windowed_model.py](../source/project/codes/task4/task4p2_pk_rawbox_subbox/task4p2_fit_pk_windowed_model.py)

`source/project/codes/task4/task4p2_pk_rawbox_subbox/task4p2_fit_pk_windowed_model.py`

Task4.2 P(k): shot-noise-aware rawbox/subbox P(k) likelihood.

`free_parameter_names`, `FitData`, `parse_prior`, `build_cosmology`, `build_template_arrays`, `hartlap_precision_local`, `load_rawbox_fit_data`, `estimate_rawbox_shotnoise`, `reconstruct_shotnoise_from_sources`, `load_subbox_fit_data`, `png_realspace_base`, `fog_multipole_coefficients`, `theory_multipoles`, `model_pk`, `log_prior`, `chi2`, `make_log_prob`, `find_initial_point`, `initialize_walkers`, `summarize_samples`, `autocorrelation_diagnostics`, `best_sample`, `plot_fit`, `parse_args`, `main`

### [task4p2_pk_common.py](../source/project/codes/task4/task4p2_pk_rawbox_subbox/task4p2_pk_common.py)

`source/project/codes/task4/task4p2_pk_rawbox_subbox/task4p2_pk_common.py`

Task4.2 P(k): Task4.2 rawbox/subbox P(k) 公共工具。

`ensure_dir`, `to_jsonable`, `write_json`, `atomic_savez`, `realization_label`, `subbox_label`, `deterministic_seed`, `task47_rawbox_npz`, `task47_subbox_npz`, `subbox_catalog_dir`, `subbox_catalog_path`, `collect_subbox_catalogs`, `parse_subbox_path`, `load_subbox_catalog`, `build_uniform_random`, `rawbox_realizations_from_task47`, `rawbox_pk_path`, `load_rawbox_pk_file`, `default_k_edges`, `column_edges`, `select_fit_bins`, `covariance_corrections`, `hartlap_precision`, `interp_logk`, `evaluate_pk_basis`, `pk_coefficients`, `combine_pk_basis`, `gq_enumerate`, `build_parent_binavg_matrix`

### [task4p2_prepare_rawbox_pk.py](../source/project/codes/task4/task4p2_pk_rawbox_subbox/task4p2_prepare_rawbox_pk.py)

`source/project/codes/task4/task4p2_pk_rawbox_subbox/task4p2_prepare_rawbox_pk.py`

Task4.2 P(k): 标准化 Task4.2 rawbox no-RSD P(k)。

`parse_args`, `main`

### [task4p2_replot_main_with_pk.py](../source/project/codes/task4/task4p2_pk_rawbox_subbox/task4p2_replot_main_with_pk.py)

`source/project/codes/task4/task4p2_pk_rawbox_subbox/task4p2_replot_main_with_pk.py`

Replot the active Task4.2 2PCF forest figure with Task4.2 P(k) results.

`read_json`, `constraint_from_parameter`, `format_constraint`, `load_twopcf`, `validate_pk_summary`, `load_pk`, `validate_chain_audit`, `add_point`, `make_plot`, `main`

### [task4p2_replot_shellavg_main_with_pk.py](../source/project/codes/task4/task4p2_pk_rawbox_subbox/task4p2_replot_shellavg_main_with_pk.py)

`source/project/codes/task4/task4p2_pk_rawbox_subbox/task4p2_replot_shellavg_main_with_pk.py`

重画 Task4.2 全 shell-average 2PCF + 正式 P(k) forest 图。

`read_json`, `validate_inputs`, `constraint`, `load_twopcf`, `main`

### [task43_audit_rsd_rawbox_x25_failure.py](../source/project/codes/task43/task43_audit_rsd_rawbox_x25_failure.py)

`source/project/codes/task43/task43_audit_rsd_rawbox_x25_failure.py`

Explain the failed x25 rawbox gate without redefining the frozen primary.

`fit_rsd`, `coarsened_empirical_tests`, `real_space_control`, `main`

### [task43_build_ezmock_rawbox_catalog.py](../source/project/codes/task43/task43_build_ezmock_rawbox_catalog.py)

`source/project/codes/task43/task43_build_ezmock_rawbox_catalog.py`

Build a mass-matched Abacus periodic raw-box catalog for EZmock calibration.

`slab_index`, `build_phase`, `main`

### [task43_build_rsd_rawbox_catalog.py](../source/project/codes/task43/task43_build_rsd_rawbox_catalog.py)

`source/project/codes/task43/task43_build_rsd_rawbox_catalog.py`

Build mass-matched real/RSD rawbox catalogs for Task 4.3.2.

`read_jsonl`, `select_row`, `legacy_catalog_path`, `build_one`, `main`

### [task43_fit_rsd_rawbox_pilot.py](../source/project/codes/task43/task43_fit_rsd_rawbox_pilot.py)

`source/project/codes/task43/task43_fit_rsd_rawbox_pilot.py`

Three-phase numerical pilot for the FullDiscrete rawbox RSD model.

`shell_kernel`, `RawboxModel`, `fit_one`, `main`

### [task43_fit_rsd_rawbox_pk0_vs_xi0_smin50.py](../source/project/codes/task43/task43_fit_rsd_rawbox_pk0_vs_xi0_smin50.py)

`source/project/codes/task43/task43_fit_rsd_rawbox_pk0_vs_xi0_smin50.py`

Fit and compare Task 4.3.2 rawbox P0(k) and xi0(s>=50).

`set_affinity`, `measurement_path`, `matching_indices`, `fit_edges_for_kmin`, `validate_canonical_edges`, `load_pk_x25`, `ExactPeriodicPk0Model`, `split_rhat`, `summarize_chain`, `fit_map`, `run_mcmc`, `load_xi0_only`, `make_plot`, `main`

### [task43_fit_rsd_rawbox_x25.py](../source/project/codes/task43/task43_fit_rsd_rawbox_x25.py)

`source/project/codes/task43/task43_fit_rsd_rawbox_x25.py`

Production x25 periodic-box closure for the Task 4.3.2 RSD model.

`set_affinity`, `measurement_path`, `load_x25`, `FastRSDModel`, `periodic_gaussian_covariance`, `vector_indices`, `data_vector`, `fit_map`, `split_rhat`, `run_mcmc`, `covariance_diagnostics`, `main`

### [task43_joint_rsd_pkxi_fit.py](../source/project/codes/task43/task43_joint_rsd_pkxi_fit.py)

`source/project/codes/task43/task43_joint_rsd_pkxi_fit.py`



`build_precision`, `summarize_chain`, `run_chain`, `fisher_covariance`

### [task43_make_rawbox_z0p725_mmin1p3e13_gaussian_covariance.py](../source/project/codes/task43/task43_make_rawbox_z0p725_mmin1p3e13_gaussian_covariance.py)

`source/project/codes/task43/task43_make_rawbox_z0p725_mmin1p3e13_gaussian_covariance.py`

Small covariance helpers retained for active Task43 lightcone scripts.

`nearest_spd`, `correlation_from_covariance`

### [task43_measure_ezmock_rawbox_pk_jaxpower.py](../source/project/codes/task43/task43_measure_ezmock_rawbox_pk_jaxpower.py)

`source/project/codes/task43/task43_measure_ezmock_rawbox_pk_jaxpower.py`

Measure the Task43 mass-matched periodic raw-box P0(k) with jaxpower.

`make_edges`, `main`

### [task43_measure_ezmock_rawbox_xi_fcfc.py](../source/project/codes/task43/task43_measure_ezmock_rawbox_xi_fcfc.py)

`source/project/codes/task43/task43_measure_ezmock_rawbox_xi_fcfc.py`

Measure the Task43 mass-matched periodic raw-box xi0(s) with FCFC.

`main`

### [task43_measure_rsd_rawbox_p02_jaxpower.py](../source/project/codes/task43/task43_measure_rsd_rawbox_p02_jaxpower.py)

`source/project/codes/task43/task43_measure_rsd_rawbox_p02_jaxpower.py`

Measure rawbox plane-parallel P0 AND P2 with the frozen Task4.3 estimator.

`output_path`, `spectrum_arrays`, `main`

### [task43_measure_rsd_rawbox_pk0_jaxpower.py](../source/project/codes/task43/task43_measure_rsd_rawbox_pk0_jaxpower.py)

`source/project/codes/task43/task43_measure_rsd_rawbox_pk0_jaxpower.py`

Measure paired real/RSD periodic rawbox P0(k) for Task 4.3.2.

`_JaxCpuOnlyCudaProbeFilter`, `read_jsonl`, `select_row`, `set_affinity`, `make_edges`, `legacy_paths`, `output_path`, `spectrum_arrays`, `main`

### [task43_measure_rsd_rawbox_xi_fcfc.py](../source/project/codes/task43/task43_measure_rsd_rawbox_xi_fcfc.py)

`source/project/codes/task43/task43_measure_rsd_rawbox_xi_fcfc.py`

Measure paired real/RSD periodic rawbox xi0 and xi2 with FCFC.

`read_jsonl`, `select_row`, `legacy_ascii_path`, `old_xi_path`, `affinity`, `ensure_rsd_ascii`, `run_fcfc`, `main`

### [task43_plot_ezmock_rawbox_mean_clustering.py](../source/project/codes/task43/task43_plot_ezmock_rawbox_mean_clustering.py)

`source/project/codes/task43/task43_plot_ezmock_rawbox_mean_clustering.py`

Plot the 25-phase mean raw-box 2PCF and power spectrum for EZmock tuning.

`sha256`, `atomic_savefig`, `main`

### [task43_plot_rsd_rawbox_pk0_vs_xi0_contours.py](../source/project/codes/task43/task43_plot_rsd_rawbox_pk0_vs_xi0_contours.py)

`source/project/codes/task43/task43_plot_rsd_rawbox_pk0_vs_xi0_contours.py`

Plot common-parameter P0-vs-xi0 contours for Task 4.3.2 rawboxes.

`density_levels`, `plot_range`, `draw_contour`, `posterior_text`, `main`

### [task43_rawbox_ezmock_common.py](../source/project/codes/task43/task43_rawbox_ezmock_common.py)

`source/project/codes/task43/task43_rawbox_ezmock_common.py`

Shared paths and small I/O helpers for the Task43 EZmock raw-box target.

`sim_name`, `halo_info_dir`, `halo_info_paths`, `catalog_path`, `catalog_metadata_path`, `ascii_path`, `pk_path`, `pk_metadata_path`, `xi_path`, `xi_metadata_path`, `ensure_dirs`, `to_jsonable`, `write_json`, `atomic_savez`, `set_cpu_affinity`, `load_catalog`

### [task43_replot_rsd_rawbox_pk0_vs_xi0_l0only.py](../source/project/codes/task43/task43_replot_rsd_rawbox_pk0_vs_xi0_l0only.py)

`source/project/codes/task43/task43_replot_rsd_rawbox_pk0_vs_xi0_l0only.py`

Replot the stored l=0 rawbox comparison without rerunning inference.

`main`

### [task43_rsd_common.py](../source/project/codes/task43/task43_rsd_common.py)

`source/project/codes/task43/task43_rsd_common.py`

Shared, side-effect-free utilities for the Task 4.3.2 RSD validation.

`sim_name`, `rawbox_halo_dir`, `lightcone_shell_path`, `sha256_file`, `_json_default`, `atomic_write_json`, `atomic_savez`, `header_boxsize`, `velocity_kms_per_mpc_h`, `apply_plane_parallel_rsd`, `apply_radial_rsd`, `apply_official_lightcone_fallback`, `finite_summary`

### [task43_rsd_model.py](../source/project/codes/task43/task43_rsd_model.py)

`source/project/codes/task43/task43_rsd_model.py`

Cached FullDiscrete Kaiser x FoG model for Task4.3.2.

`shell_jell_kernel`, `cache_path`, `build_cache`, `FullDiscreteRSDModel`, `load_or_build`

### [task43_rsd_rawbox_joint_4way.py](../source/project/codes/task43/task43_rsd_rawbox_joint_4way.py)

`source/project/codes/task43/task43_rsd_rawbox_joint_4way.py`

Periodic-box four-way joint: P0+P2 (l=0,2) + xi0+xi2 (l=0,2), RSD, x25.

`angular_totals`, `pk_pole_cov`, `cross_block`, `main`

### [task43_rsd_rawbox_joint_pkxi.py](../source/project/codes/task43/task43_rsd_rawbox_joint_pkxi.py)

`source/project/codes/task43/task43_rsd_rawbox_joint_pkxi.py`

Periodic-box joint P0(k)+xi0(s) fit: the no-window control experiment.

`mode_level_cross`, `empirical_blocks`, `main`

### [task43_rsd_rawbox_realspace_check.py](../source/project/codes/task43/task43_rsd_rawbox_realspace_check.py)

`source/project/codes/task43/task43_rsd_rawbox_realspace_check.py`

Rawbox real-space xi0 closure check: the minimal-model control.

`main`

### [task43_rsd_rawbox_realspace_mcmc.py](../source/project/codes/task43/task43_rsd_rawbox_realspace_mcmc.py)

`source/project/codes/task43/task43_rsd_rawbox_realspace_mcmc.py`

MCMC chains for the rawbox real-space xi0 check (2 params: fNL, b1).

`main`

### [task43_rsd_rawbox_realspace_pk_check.py](../source/project/codes/task43/task43_rsd_rawbox_realspace_pk_check.py)

`source/project/codes/task43/task43_rsd_rawbox_realspace_pk_check.py`

Rawbox real-space P0(k) closure: the P-side twin of the xi0 real-space check.

`main`

### [task43_run_ezmock_rawbox_trial.py](../source/project/codes/task43/task43_run_ezmock_rawbox_trial.py)

`source/project/codes/task43/task43_run_ezmock_rawbox_trial.py`

Run one five-realization fixed-amplitude EZmock calibration trial.

`parse_args`, `validate_args`, `run_logged`, `count_catalog_rows`, `ezmock_config`, `fcfc_config`, `load_target_ntracer`, `save_xi`, `summarize`, `main`

### [task43_summarize_ezmock_rawbox_clustering.py](../source/project/codes/task43/task43_summarize_ezmock_rawbox_clustering.py)

`source/project/codes/task43/task43_summarize_ezmock_rawbox_clustering.py`

Aggregate the 25 Task43 raw-box P0(k) and xi0(s) measurements.

`correlation`, `main`

### [task43_summarize_rsd_rawbox_pilot.py](../source/project/codes/task43/task43_summarize_rsd_rawbox_pilot.py)

`source/project/codes/task43/task43_summarize_rsd_rawbox_pilot.py`

Summarize the ph000--ph002 rawbox RSD model-shape pilot.

`measurement_path`, `fit_one`, `main`

### [task43_summarize_rsd_rawbox_x25.py](../source/project/codes/task43/task43_summarize_rsd_rawbox_x25.py)

`source/project/codes/task43/task43_summarize_rsd_rawbox_x25.py`

Summarize the Task 4.3.2 ph000--ph024 periodic raw-box RSD closure.

`measurement_path`, `concatenate_by_ell`, `fit_vector`, `main`

### [task43_theory_template.py](../source/project/codes/task43/task43_theory_template.py)

`source/project/codes/task43/task43_theory_template.py`

Shared Task43 theory-template helpers.

`load_task41`, `_json_value`, `cosmology_metadata`, `build_template_arrays`

### [test_task43_rsd_common.py](../source/project/codes/task43/test_task43_rsd_common.py)

`source/project/codes/task43/test_task43_rsd_common.py`

Unit tests for the Task 4.3.2 coordinate and atomic-I/O contract.

`test_plane_parallel_zero_velocity_is_exact`, `test_plane_parallel_shift_and_wrap`, `test_radial_mapping_preserves_angles_and_uses_origin_modulo`, `test_atomic_writers`, `test_official_lightcone_fallback_replaces_only_available_averages`

### [test_task43_rsd_model.py](../source/project/codes/task43/test_task43_rsd_model.py)

`source/project/codes/task43/test_task43_rsd_model.py`

Fast tests for Task4.3.2 RSD model algebra.

`test_shell_j0_low_k_limit`, `test_shell_j2_has_fourier_phase`

### [task44_audit_pngbase_hodhost_mmin1e13_strictk0p006.py](../source/project/codes/task44/task44_audit_pngbase_hodhost_mmin1e13_strictk0p006.py)

`source/project/codes/task44/task44_audit_pngbase_hodhost_mmin1e13_strictk0p006.py`

Final cross-product audit for the current-range matched halo P0/xi0 result.

`_load_json`, `_load_fit_summary_and_arrays`, `build_audit`, `main`

### [task44_audit_pngbase_hodmap_lrg_conservative.py](../source/project/codes/task44/task44_audit_pngbase_hodmap_lrg_conservative.py)

`source/project/codes/task44/task44_audit_pngbase_hodmap_lrg_conservative.py`

Audit the conservative-scale LRG-only Task44 fits and four official PDFs.

`load_fit`, `posterior`, `build_audit`, `main`

### [task44_audit_pngbase_pk_xi_consistency.py](../source/project/codes/task44/task44_audit_pngbase_pk_xi_consistency.py)

`source/project/codes/task44/task44_audit_pngbase_pk_xi_consistency.py`

Audit the corrected periodic-box P0--xi0 likelihood contract.

`read_json`, `load_variant`, `fit_record`, `engine_max_abs_delta`, `build_catalog`, `asymmetric_error`, `draw_pdf`, `main`

### [task44_build_pngbase_hodhost_mmin1e13.py](../source/project/codes/task44/task44_build_pngbase_hodhost_mmin1e13.py)

`source/project/codes/task44/task44_build_pngbase_hodhost_mmin1e13.py`

Build audited M>=1e13 Msun/h HOD-host halo catalogs for c300/c302.

`_finite_summary`, `_validated_existing`, `build_one`, `build_ascii`, `write_combined_audit`, `parse_tags`, `main`

### [task44_config.py](../source/project/codes/task44/task44_config.py)

`source/project/codes/task44/task44_config.py`

Shared constants for Task44 lightcone redshift-space tests.

`ensure_task44_dirs`, `normalize_sample`, `normalize_lrg_bin`, `normalize_realization`, `realization_config`, `sample_config`, `lrg_bin_config`, `sample_label`, `lrg_bin_label`, `sample_tracer`, `format_float_tag`, `format_p_fixed_tag`, `catalog_output_paths`, `zeff_output_path`, `xi_output_path`, `rr_smu_prefix`, `covariance_prefix`, `formal_gic_window_path`, `fit_output_dir`, `ls_data_path`, `ls_random_path`, `format_p0_tag`, `compact_random_tag`, `ez`, `comoving_distance_mpc_h`, `rdz_to_xyz`, `write_json`

### [task44_fit_pngbase_hodhost_gaussianp_freefnl.py](../source/project/codes/task44/task44_fit_pngbase_hodhost_gaussianp_freefnl.py)

`source/project/codes/task44/task44_fit_pngbase_hodhost_gaussianp_freefnl.py`

Infer fNL from matched halo P0/xi0 with a Gaussian prior on p.

`ExactPeriodicPkGaussianPModel`, `FullDiscreteXiGaussianPModel`, `bounds_for`, `prior_chi2`, `fit_map`, `run_mcmc`, `_validated_existing`, `fit_contract`, `resume_one`, `run_one`, `main`

### [task44_fit_pngbase_hodhost_pk_strictk0p006.py](../source/project/codes/task44/task44_fit_pngbase_hodhost_pk_strictk0p006.py)

`source/project/codes/task44/task44_fit_pngbase_hodhost_pk_strictk0p006.py`

Fixed-baseline-fNL/free-p fits to strict-kmin host-halo P0.

`load_measurement`, `_validated_existing`, `run_one`, `main`

### [task44_fit_pngbase_hodhost_xi_smax150.py](../source/project/codes/task44/task44_fit_pngbase_hodhost_xi_smax150.py)

`source/project/codes/task44/task44_fit_pngbase_hodhost_xi_smax150.py`

Fixed-baseline-fNL/free-p fits to Task44 LRG or occupied-host xi0.

`load_halo_xi`, `lrg_xi_contract`, `load_lrg_xi`, `slice_theory_to_fit_range`, `_validated_existing`, `run_one`, `main`

### [task44_fit_pngbase_hodmap_rawbox.py](../source/project/codes/task44/task44_fit_pngbase_hodmap_rawbox.py)

`source/project/codes/task44/task44_fit_pngbase_hodmap_rawbox.py`

Fixed-fNL/free-p inference for the two Task44 HOD-MAP periodic LRG boxes.

`shell_j0_average`, `build_theory_cache`, `load_theory`, `matching_indices`, `ExactPeriodicPkModel`, `ExactPeriodicPkFixedPModel`, `FullDiscreteXiModel`, `FullDiscreteXiFixedPModel`, `regularize_covariance`, `bounds_for`, `fit_map`, `split_rhat`, `summarize_chain`, `run_mcmc`, `load_pk_data`, `load_xi_data`, `covariance_iteration_summary`, `run_fit`, `parse_tags`, `main`

### [task44_fit_pngbase_hodmap_strictk0p006.py](../source/project/codes/task44/task44_fit_pngbase_hodmap_strictk0p006.py)

`source/project/codes/task44/task44_fit_pngbase_hodmap_strictk0p006.py`

Fixed-baseline-fNL/free-p fits to the strict-kmin galaxy P0 products.

`load_measurement`, `_validated_existing`, `run_one`, `main`

### [task44_fit_pngbase_pk_xi_consistency.py](../source/project/codes/task44/task44_fit_pngbase_pk_xi_consistency.py)

`source/project/codes/task44/task44_fit_pngbase_pk_xi_consistency.py`

Targeted likelihood diagnostics for PNG-base P0--xi0 consistency.

`FullDiscreteXiResidualStochasticCovModel`, `FullDiscreteXiFixedPResidualStochasticCovModel`, `primary_pk_summary`, `variant_prefix`, `fixed_point`, `variant_cache_contract`, `write_fit`, `run_pk_contiguous`, `run_xi_stochastic_cov`, `run_xi_fixedp1_stochastic_cov`, `main`

### [task44_fit_validation.py](../source/project/codes/task44/task44_fit_validation.py)

`source/project/codes/task44/task44_fit_validation.py`

Task44 拟合入口的数值与缓存防护。

`file_digest`, `canonical`, `validate_covariance`, `make_fit_contract`, `reusable_fit`

### [task44_measure_pngbase_hodhost_mmin1e13_pk.py](../source/project/codes/task44/task44_measure_pngbase_hodhost_mmin1e13_pk.py)

`source/project/codes/task44/task44_measure_pngbase_hodhost_mmin1e13_pk.py`

Measure strict-kmin P0 for the exact halo catalog used by halo xi0.

`_validated_existing`, `measure_one`, `main`

### [task44_measure_pngbase_hodhost_mmin1e13_xi.py](../source/project/codes/task44/task44_measure_pngbase_hodhost_mmin1e13_xi.py)

`source/project/codes/task44/task44_measure_pngbase_hodhost_mmin1e13_xi.py`

Measure M>=1e13 occupied-host halo xi0 with three independent engines.

`_load_positions`, `_validated_product`, `save_engine_product`, `_load_fcfc_pair_count`, `measure_fcfc`, `measure_pycorr`, `measure_cucount`, `load_engine`, `_old_galaxy_fcfc_path`, `compare_and_plot`, `_parse_tags`, `main`

### [task44_measure_pngbase_hodmap_rawbox.py](../source/project/codes/task44/task44_measure_pngbase_hodmap_rawbox.py)

`source/project/codes/task44/task44_measure_pngbase_hodmap_rawbox.py`

Audit and measure the two Task44 PNG-base HOD-MAP periodic LRG boxes.

`exact_first_bin`, `run_catalog_audit`, `build_ascii`, `measure_pk`, `measure_xi`, `parse_tags`, `main`

### [task44_measure_pngbase_hodmap_strictk0p006.py](../source/project/codes/task44/task44_measure_pngbase_hodmap_strictk0p006.py)

`source/project/codes/task44/task44_measure_pngbase_hodmap_strictk0p006.py`

Remeasure galaxy P0 with the strict physical mode cut k>=0.006 h/Mpc.

`explicit_lattice_statistics`, `_source_catalog_sha256`, `_validated_existing`, `measure_one`, `main`

### [task44_plot_audit_pngbase_hodhost_gaussianp_freefnl.py](../source/project/codes/task44/task44_plot_audit_pngbase_hodhost_gaussianp_freefnl.py)

`source/project/codes/task44/task44_plot_audit_pngbase_hodhost_gaussianp_freefnl.py`

Plot and audit the Gaussian-p/free-fNL matched halo P0/xi0 test.

`load_fit`, `posterior`, `flat_samples`, `_interval`, `_atomic_figure_save`, `_fit_box`, `make_bestfit`, `_set_task43_contour_style`, `make_contours`, `build_audit`, `main`

### [task44_plot_audit_pngbase_hodhost_xi_smax150.py](../source/project/codes/task44/task44_plot_audit_pngbase_hodhost_xi_smax150.py)

`source/project/codes/task44/task44_plot_audit_pngbase_hodhost_xi_smax150.py`

Plot and audit matched-catalog fixed-fNL halo P0 and halo xi0 fits.

`load_fit`, `load_full_xi_measurement`, `posterior`, `flat_samples`, `_atomic_figure_save`, `_xi_box`, `_pk_box`, `make_bestfit_measurements`, `_set_task43_contour_style`, `make_contours`, `build_audit`, `main`

### [task44_plot_audit_pngbase_strictk0p006.py](../source/project/codes/task44/task44_plot_audit_pngbase_strictk0p006.py)

`source/project/codes/task44/task44_plot_audit_pngbase_strictk0p006.py`

Plot and audit the strict-kmin fixed-fNL/free-p P0 fits.

`load_fit`, `posterior`, `flat_samples`, `_atomic_figure_save`, `_range_box`, `make_bestfit`, `_credible_density`, `make_contours`, `make_comparison`, `build_audit`, `main`

### [task44_plot_pngbase_hodmap_lrg_strictk0p006.py](../source/project/codes/task44/task44_plot_pngbase_hodmap_lrg_strictk0p006.py)

`source/project/codes/task44/task44_plot_pngbase_hodmap_lrg_strictk0p006.py`

Current-contract LRG P0/xi0 best fits and Task43-style posterior plots.

`_prefix`, `load_fit`, `load_full_xi`, `posterior`, `interval`, `flat_samples`, `save_figure`, `fit_box`, `make_bestfit`, `set_contour_style`, `make_contours`, `main`

### [task44_plot_pngbase_hodmap_rawbox.py](../source/project/codes/task44/task44_plot_pngbase_hodmap_rawbox.py)

`source/project/codes/task44/task44_plot_pngbase_hodmap_rawbox.py`

Audit and plot the Task44 fixed-fNL/free-p PNG-base raw-box fits.

`load_fit`, `selected_indices`, `posterior_row`, `build_audit`, `get_flat_samples`, `make_contour`, `make_summary_pdf`, `main`

### [task44_plot_pngbase_hodmap_rawbox_fixedp1.py](../source/project/codes/task44/task44_plot_pngbase_hodmap_rawbox_fixedp1.py)

`source/project/codes/task44/task44_plot_pngbase_hodmap_rawbox_fixedp1.py`

Audit and plot fixed-p=1/free-fNL P0--xi0 results for the PNG-base HOD boxes.

`fit_rows`, `load_diagnostic_fit`, `corrected_fit_rows`, `corrected_freep_fit_rows`, `combined_range`, `validate_fit`, `draw_triangle`, `draw_fixedfnl_bestfit`, `draw_forest`, `draw_freep_forest`, `build_catalog_record`, `main`

### [task44_pngbase_hodhost_mmin1e13_common.py](../source/project/codes/task44/task44_pngbase_hodhost_mmin1e13_common.py)

`source/project/codes/task44/task44_pngbase_hodhost_mmin1e13_common.py`

Frozen paths and numerical contracts for the Task44 halo-clustering extension.

`ensure_output_dirs`, `raw_halo_files`, `host_catalog_path`, `host_catalog_metadata_path`, `host_ascii_path`, `host_ascii_metadata_path`, `xi_engine_path`, `xi_engine_metadata_path`, `strict_pk_path`, `strict_pk_metadata_path`, `host_pk_path`, `host_pk_metadata_path`, `strict_fit_prefix`, `lrg_pk_gaussianp_fit_prefix`, `lrg_pk_conservative_fit_prefix`, `lrg_pk_conservative_gaussianp_fit_prefix`, `lrg_xi_fit_prefix`, `lrg_xi_gaussianp_fit_prefix`, `lrg_xi_extended_fit_prefix`, `lrg_xi_extended_gaussianp_fit_prefix`, `lrg_xi_conservative_fit_prefix`, `lrg_xi_conservative_gaussianp_fit_prefix`, `host_pk_fit_prefix`, `host_pk_gaussianp_fit_prefix`, `host_xi_fit_prefix`, `host_xi_gaussianp_fit_prefix`, `load_host_catalog`, `analytic_rr`

### [task44_pngbase_hodmap_rawbox_common.py](../source/project/codes/task44/task44_pngbase_hodmap_rawbox_common.py)

`source/project/codes/task44/task44_pngbase_hodmap_rawbox_common.py`

Shared immutable contract for the Task44 PNG-base HOD-MAP raw boxes.

`xi_edges`, `CatalogSpec`, `get_spec`, `ensure_output_dirs`, `to_jsonable`, `atomic_write_json`, `atomic_savez`, `sha256_file`, `set_cpu_affinity`, `catalog_audit`, `load_positions`, `ascii_catalog_path`, `ascii_metadata_path`, `pk_path`, `pk_metadata_path`, `xi_path`, `xi_metadata_path`, `theory_path`, `fit_prefix`, `legacy_lightcone_matched_fit_prefix`

### [task44_rsd_theory.py](../source/project/codes/task44/task44_rsd_theory.py)

`source/project/codes/task44/task44_rsd_theory.py`

RSD monopole theory helpers for Task44 LRG2 fNL fits.

`omega_m_z`, `growth_rate_approx`, `gaussian_mu_moments`, `lorentzian_mu_moments`, `mu_moments`, `fog_description`, `evaluate_rsd_png_monopole_pk`, `evaluate_rsd_png_multipole_pk`, `build_theory_context`

### [task44_validate_pngbase_hodmap_xi_engines.py](../source/project/codes/task44/task44_validate_pngbase_hodmap_xi_engines.py)

`source/project/codes/task44/task44_validate_pngbase_hodmap_xi_engines.py`

Cross-check Task44 periodic-box xi0 with FCFC, pycorr, and CuCount.

`engine_path`, `engine_metadata_path`, `analytic_rr`, `validated_catalog_hash`, `save_engine_product`, `load_fcfc_pair_count`, `measure_fcfc`, `measure_pycorr`, `measure_cucount`, `load_engine`, `compare_and_plot`, `parse_tags`, `main`

### [task42_plot_quijote_alltags_freesigmas_constraints.py](../source/project/old_doc_codes/task4_task44_cleanup_20260707T061844Z/moved/codes/task4/task42_plot_quijote_alltags_freesigmas_constraints.py)

`source/project/old_doc_codes/task4_task44_cleanup_20260707T061844Z/moved/codes/task4/task42_plot_quijote_alltags_freesigmas_constraints.py`

Plot free-sigmas constraints for all Quijote tags currently available locally.

`covariance_corrections`, `tag_files`, `load_rows`, `axis_limits`, `parse_args`, `plot`

### [task42_quijote_lcp50_profiler.py](../source/project/old_doc_codes/task4_task44_cleanup_20260707T061844Z/moved/codes/task4/task42_quijote_lcp50_profiler.py)

`source/project/old_doc_codes/task4_task44_cleanup_20260707T061844Z/moved/codes/task4/task42_quijote_lcp50_profiler.py`

Quijote LCp50 1Gpc halo validation trial.

`rid_from_path`, `sorted_files`, `load_pk_stack`, `load_pcf_stack`, `build_cosmology`, `gq_enumerate`, `gq_fft`, `shell_arrays`, `bin_shell_indices`, `binavg_pk`, `covariance_corrections`, `RSDModel`, `fit_pk_reference`, `precompute_rebin_cache`, `j0_matrix`, `fit_xi_profiler`, `build_rlist_selections`, `parse_selection_modes`, `main`
