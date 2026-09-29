# 9.30 meeting：velocileptors/GSM xi full-RIC diagnostic

## 结论

本轮实现了独立的 velocileptors/GSM xi full-RIC pilot，并按 9.18meeting 的 triangle 页面风格生成了 PDF。结果只作 diagnostic，不替换既有正式 full-RIC 链。

## Operator

- 使用 full-RIC validated midpoint response，内部 233 个 separation shells。
- 每个 shell 使用 2-point Gauss-Legendre `r^2` 体积平均。
- 保留 velocileptors/GSM 的 `ell=(0,2,4)`。
- xi 预测分解为 raw velocileptors xi、deterministic clustering RIC、fixed Poisson shot RIC 和 `sn0` stochastic RIC response。
- P0/P2 使用既有 full-RIC P-side operator和数据向量，未修改旧 formal chain。

## Direct MAP smoke（EZmock1000 C_single）

| fit | b1 | chi2 |
|---|---:|---:|
| P-only full-RIC | 2.41346 | 0.895 |
| xi-only velocileptors + full-RIC | 2.42378 | 2.732 |
| joint，cross lambda=0 | 2.41379 | 3.759 |
| joint，cross lambda=0.5 | 2.40597 | 3.490 |
| joint，cross lambda=1 | 2.37820 | 4.031 |

The xi-only model is well behaved, while the full P-xi cross block again pulls the joint center below both marginal centers. This confirms that the RIC operator is not sufficient by itself to remove the cross-covariance/model-residual coupling.

## Short diagnostic chain

The 16-walker, 1200-step chain is intentionally short and is used only for a contour sanity check.

- P-only chain reused from the existing formal full-RIC result, `b1_MAP=2.41346`。
- xi-only chain: `b1_MAP=2.42481`, `b1_q50=2.43350`, raw `chi2_MAP=2.734`。
- joint chain: `b1_MAP=2.38676`, `b1_q50=2.38745`, raw `chi2_MAP=4.066`。

The joint center remains below both marginal centers when the full cross block is used, so the next bottleneck is covariance policy matching and the minimal PNG/GSM response, rather than the missing RIC bookkeeping alone。

## Files

- Operator grid: `outputs/task43_outputs/rsd_validation/task432_model_repair/velocileptors_fullric_v1/operator/local/velo_fullric_grid.npz`
- Smoke diagnostics: `9.30meeting/task432_velocileptors_fullric/diagnostics/smoke.json`
- Direct MAP: `9.30meeting/task432_velocileptors_fullric/diagnostics/direct_map.json`
- Chain summary: `9.30meeting/task432_velocileptors_fullric/diagnostic_chain_summary.json`
- 9.18-style contour: `9.30meeting/task432_velocileptors_fullric/figures/task432_velocileptors_fullric_9.18style_triangle.pdf`

The operator still uses the minimal PNG response already present in velocileptors/GSM; it is not a complete CLPT-PNG bias basis. The covariance random-policy mismatch between halo observed-z resampling and EZmock common-50 randoms remains open.
