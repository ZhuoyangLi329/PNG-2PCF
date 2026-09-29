# Figures and machine-readable figure references

## 9.18

`source/project/PNG-2PCF-model-to-observe/9.18meeting/task43_ezmock_covariance_mcmc_vs_jaxpower/` contains the baseline PDF and JSON.

## 9.22

The most important PDFs are:

- `task432_hybrid_fullRIC_ezmock1000_P02xi02_joint_kmax0p08_smin50_baomask80_120_9.18style_v1.pdf`
- `task432_hybrid_fullRIC_ezmock1000_P02xi02_joint_kmax0p08_smin50_baomask80_120_diagnostics_v1.pdf`
- `task432_hybrid_gic_ezmock1000_P02xi02_joint_kmax0p08_smin50_baomask80_120.pdf`
- `hybrid_gic_joint_b1_cause_tests_ezmock1000.pdf`
- `task432_fullRIC_FKP_conditional_diagnostics.pdf`
- `task432_scale_selection_ezmock1000_contours_and_diagnostics.pdf`

Temporary PNGs and large raster QA intermediates are not included; the repository keeps PDF as the scientific figure format.

## 9.30 diagnostic

[velocileptors full-RIC triangle PDF](../source/project/PNG-2PCF-model-to-observe/9.30meeting/task432_velocileptors_fullric/figures/task432_velocileptors_fullric_9.18style_triangle.pdf) is a short-chain sanity figure, not an updated formal constraint. The P curve reuses a formal chain; xi/joint use narrower local priors and raw-C likelihoods. Widths are not a like-for-like gain comparison.

Direct MAP and the sampled maximum-probability point are different center estimators; read `diagnostics/direct_map.json` and `diagnostic_chain_summary.json` separately.
