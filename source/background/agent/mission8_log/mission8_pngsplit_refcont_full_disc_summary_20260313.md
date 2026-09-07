# Mission 8：连续 no-PNG 主体 + 全离散 PNG 增量

本次按新的加法分解实现：

`xi_total = xi_ref_continuous(fnl->0, 1e-4<k<20) + xi_png_discrete(full shells to kmax=20)`

其中 `mean_sigma < 0` 表示模型整体偏高，`mean_sigma > 0` 表示模型整体偏低。

| Box | Sample | Method | mean_abs_sigma | chi2_ndof | mean_sigma |
|---|---|---|---:|---:|---:|
| 1Gpc | fnl100 | ExpWindow | 0.0818 | 0.0130 | 0.0405 |
| 1Gpc | fnl100 | PNGSplitFullDisc | 0.2782 | 0.0901 | -0.2782 |
| 1Gpc | fnl100 | AllDiscrete | 0.1674 | 0.0404 | -0.1668 |
| 1Gpc | fnl0 | ExpWindow | 0.1256 | 0.0231 | 0.0945 |
| 1Gpc | fnl0 | PNGSplitFullDisc | 0.2229 | 0.0629 | -0.2110 |
| 1Gpc | fnl0 | AllDiscrete | 0.1296 | 0.0242 | -0.0916 |
| 3Gpc | fnl100 | ExpWindow | 0.2076 | 0.0495 | -0.1433 |
| 3Gpc | fnl100 | PNGSplitFullDisc | 0.4758 | 0.2432 | -0.4758 |
| 3Gpc | fnl100 | AllDiscrete | 0.4739 | 0.2417 | -0.4739 |
| 3Gpc | fnl0 | ExpWindow | 0.3535 | 0.1515 | -0.3535 |
| 3Gpc | fnl0 | PNGSplitFullDisc | 0.3934 | 0.1794 | -0.3934 |
| 3Gpc | fnl0 | AllDiscrete | 0.3903 | 0.1772 | -0.3903 |

## 核心检查

- 1Gpc fnl=0: ExpWindow mean_sigma=0.0945, Split mean_sigma=-0.2110, AllDiscrete mean_sigma=-0.0916
- 3Gpc fnl=0: ExpWindow mean_sigma=-0.3535, Split mean_sigma=-0.3934, AllDiscrete mean_sigma=-0.3903
