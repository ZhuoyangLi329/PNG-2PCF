# Results and audit map

## 9.18 baseline

- `9.18meeting/task43_ezmock_covariance_mcmc_vs_jaxpower/task43_ezmock604_vs_jaxpower_rsd_P02xi02_joint_kmax0p08_smin50_baomask80_120_v1.json`
- 同目录 PDF：EZmock-604 empirical covariance 与 jaxpower analytic covariance 的 P02/xi02/joint 对照。

## 9.22 model progression

- `9.22meeting/task432_hybrid_ezmock1000_9.18contract/`：没有 GIC，superseded。
- `9.22meeting/task432_hybrid_ezmock1000_gic_9.18contract/`：scalar GIC。
- `9.22meeting/task432_hybrid_fullRIC_ezmock1000_9.18contract/`：当前 full RIC 结果和诊断 PDF。
- `9.22meeting/task432_hybrid_gic_joint_b1_cause_tests_ezmock1000/`：joint b1 归因实验。
- `9.22meeting/task432_fkp_conditional_ezmock1000/`：FKP/条件模式诊断。
- `9.22meeting/task432_scale_selection_ezmock1000/`：少量尺度敏感性，不替换正式基准。

## machine-readable summaries

- `outputs/task43_outputs/rsd_validation/task432_model_repair/full_ric_p02xi02_v1/formal_mean4/results.json`
- `outputs/task43_outputs/rsd_validation/task432_model_repair/full_ric_p02xi02_v1/formal_mean4/postflight.json`
- `outputs/task43_outputs/rsd_validation/task432_model_repair/hybrid_ezmock1000_gic_0918_contract/input_audit.json`
- `outputs/task43_outputs/rsd_validation/task432_model_repair/hybrid_ezmock1000_gic_0918_contract/postflight_audit.json`
- `outputs/task43_outputs/rsd_validation/task432_model_repair/hybrid_gic_current_results.json`
- `outputs/task43_outputs/rsd_validation/task432_model_repair/gsm_kaiser_limit_audit/task432_gsm_kaiser_limit_audit.json`
- `outputs/task43_outputs/rsd_validation/task432_model_repair/lightcone_png_velocileptors_gsm/task432_lightcone_png_velocileptors_gsm_map.json`

## Interpretation rule

`status=pass` means numerical/postflight gates passed. It does not mean that joint P--xi closure or a publication-quality fNL constraint has been established.
