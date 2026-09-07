# Mission 8：1Gpc 重测 P(k) 到 kmax≈5 的全离散建模

- 低-k 基模：`k_f = 0.00628319`
- 测量数据公共上限：`kmax_data = 3.216991`
- `fnl100` 最优方法：`1Gpc_fnl100_ExpWindow`，`mean_abs_sigma=0.0782`，`mean_sigma=0.0236`
- `fnl0` 最优方法：`1Gpc_fnl0_AllDiscreteBestfit_k20`，`mean_abs_sigma=0.1159`，`mean_sigma=-0.0565`

## 全部指标

| sample | method | mean_abs_sigma | mean_sigma | chi2_ndof |
| --- | --- | ---: | ---: | ---: |
| fnl100 | 1Gpc_fnl100_ExpWindow | 0.0782 | 0.0236 | 0.0119 |
| fnl100 | 1Gpc_fnl100_AllDiscreteBestfit_k20 | 0.1837 | -0.1837 | 0.0474 |
| fnl100 | 1Gpc_fnl100_AllDiscreteBestfit_kdata | 0.2126 | -0.1977 | 0.0669 |
| fnl100 | 1Gpc_fnl100_MeasuredPkInterp_kdata | 1.9598 | 1.9598 | 3.8814 |
| fnl100 | 1Gpc_fnl100_MeasuredPkStep_kdata | 0.1634 | 0.0840 | 0.0349 |
| fnl0 | 1Gpc_fnl0_ExpWindow | 0.1452 | 0.1268 | 0.0306 |
| fnl0 | 1Gpc_fnl0_AllDiscreteBestfit_k20 | 0.1159 | -0.0565 | 0.0195 |
| fnl0 | 1Gpc_fnl0_AllDiscreteBestfit_kdata | 0.1168 | -0.0577 | 0.0204 |
| fnl0 | 1Gpc_fnl0_MeasuredPkInterp_kdata | 0.9375 | 0.9375 | 0.8999 |
| fnl0 | 1Gpc_fnl0_MeasuredPkStep_kdata | 0.1706 | 0.0961 | 0.0376 |
