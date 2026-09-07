# Mission 8：原始全离散的径向 bin 平均修正

本次完全保留“总 P(k) 全离散”框架，只把理论端从 `j0(k s_cen)` 改成了与 pcf 测量定义一致的 bin 体积平均 kernel。

其中 `mean_sigma < 0` 表示模型整体偏高，`mean_sigma > 0` 表示模型整体偏低。

| Box | Sample | Method | mean_abs_sigma | chi2_ndof | mean_sigma |
|---|---|---|---:|---:|---:|
| 1Gpc | fnl100 | ExpWindow | 0.0818 | 0.0130 | 0.0405 |
| 1Gpc | fnl100 | AllDiscreteCenter | 0.1674 | 0.0404 | -0.1668 |
| 1Gpc | fnl100 | AllDiscreteBinAvg | 0.1681 | 0.0409 | -0.1662 |
| 1Gpc | fnl0 | ExpWindow | 0.1256 | 0.0231 | 0.0945 |
| 1Gpc | fnl0 | AllDiscreteCenter | 0.1296 | 0.0242 | -0.0916 |
| 1Gpc | fnl0 | AllDiscreteBinAvg | 0.1298 | 0.0244 | -0.0916 |
| 3Gpc | fnl100 | ExpWindow | 0.2076 | 0.0495 | -0.1433 |
| 3Gpc | fnl100 | AllDiscreteCenter | 0.4739 | 0.2417 | -0.4739 |
| 3Gpc | fnl100 | AllDiscreteBinAvg | 0.4730 | 0.2405 | -0.4730 |
| 3Gpc | fnl0 | ExpWindow | 0.3535 | 0.1515 | -0.3535 |
| 3Gpc | fnl0 | AllDiscreteCenter | 0.3903 | 0.1772 | -0.3903 |
| 3Gpc | fnl0 | AllDiscreteBinAvg | 0.3902 | 0.1770 | -0.3902 |

## 物理解释

1. pcf 文件本身提供的是 `s_min, s_max` 区间平均后的 `xi_0`，而不是点值。
2. 因此用 `j0(k s_cen)` 只是近似；更自洽的做法应当是对每个 r-bin 做体积平均。
3. 这一修正不引入任何额外经验参数，所以如果它改善了结果，物理解释是最直接的。
