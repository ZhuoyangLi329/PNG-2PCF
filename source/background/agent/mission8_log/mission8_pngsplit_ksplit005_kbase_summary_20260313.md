# Mission 8：k_base 起积的 PNG-split(k_split=0.05)

本次只改了 no-PNG 主体的积分下限：从 `1e-4` 改成 `k_base = 2pi/L`。

| Box | Sample | Method | mean_abs_sigma | chi2_ndof | mean_sigma |
|---|---|---|---:|---:|---:|
| 1Gpc | fnl100 | ExpWindow | 0.0818 | 0.0130 | 0.0405 |
| 1Gpc | fnl100 | PNGSplit_k005_kbase | 0.7501 | 0.7130 | 0.7501 |
| 1Gpc | fnl100 | PNGSplit_k005_kbase_NoHigh | 0.7742 | 0.7003 | 0.7742 |
| 1Gpc | fnl0 | ExpWindow | 0.1256 | 0.0231 | 0.0945 |
| 1Gpc | fnl0 | PNGSplit_k005_kbase | 0.9046 | 0.8477 | 0.9046 |
| 1Gpc | fnl0 | PNGSplit_k005_kbase_NoHigh | 0.9059 | 0.8499 | 0.9059 |
| 3Gpc | fnl100 | ExpWindow | 0.2076 | 0.0495 | -0.1433 |
| 3Gpc | fnl100 | PNGSplit_k005_kbase | 0.4984 | 0.4364 | -0.4334 |
| 3Gpc | fnl100 | PNGSplit_k005_kbase_NoHigh | 0.4283 | 0.3167 | -0.4010 |
| 3Gpc | fnl0 | ExpWindow | 0.3535 | 0.1515 | -0.3535 |
| 3Gpc | fnl0 | PNGSplit_k005_kbase | 0.3102 | 0.1332 | -0.3017 |
| 3Gpc | fnl0 | PNGSplit_k005_kbase_NoHigh | 0.3043 | 0.1269 | -0.2974 |

## 重点看 fnl=0

- 1Gpc: ExpWindow mean_sigma=0.0945, Split_kbase mean_sigma=0.9046, Split_kbase_NoHigh mean_sigma=0.9059
- 3Gpc: ExpWindow mean_sigma=-0.3535, Split_kbase mean_sigma=-0.3017, Split_kbase_NoHigh mean_sigma=-0.2974
