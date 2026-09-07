# Mission 8：全离散壳层的 delta -> 窄高斯 测试

本次回到“总 P(k) 全离散”框架，不区分 PNG / 非 PNG。

测试了三种壳层表示：
- `AllDiscrete`: 原始 delta-shell
- `GaussDelta_a010kf`: `sigma_k = 0.10 k_f`
- `GaussDelta_a025kf`: `sigma_k = 0.25 k_f`

其中 `mean_sigma < 0` 表示模型整体偏高，`mean_sigma > 0` 表示模型整体偏低。

| Box | Sample | Method | mean_abs_sigma | chi2_ndof | mean_sigma |
|---|---|---|---:|---:|---:|
| 1Gpc | fnl100 | ExpWindow | 0.0818 | 0.0130 | 0.0405 |
| 1Gpc | fnl100 | AllDiscrete | 0.1674 | 0.0404 | -0.1668 |
| 1Gpc | fnl100 | GaussDelta_a010kf | 0.1413 | 0.0289 | -0.1269 |
| 1Gpc | fnl100 | GaussDelta_a025kf | 0.1058 | 0.0174 | 0.0587 |
| 1Gpc | fnl0 | ExpWindow | 0.1256 | 0.0231 | 0.0945 |
| 1Gpc | fnl0 | AllDiscrete | 0.1296 | 0.0242 | -0.0916 |
| 1Gpc | fnl0 | GaussDelta_a010kf | 0.1240 | 0.0226 | -0.0831 |
| 1Gpc | fnl0 | GaussDelta_a025kf | 0.1054 | 0.0176 | -0.0462 |
| 3Gpc | fnl100 | ExpWindow | 0.2076 | 0.0495 | -0.1433 |
| 3Gpc | fnl100 | AllDiscrete | 0.4739 | 0.2417 | -0.4739 |
| 3Gpc | fnl100 | GaussDelta_a010kf | 0.4753 | 0.2435 | -0.4753 |
| 3Gpc | fnl100 | GaussDelta_a025kf | 0.4825 | 0.2530 | -0.4825 |
| 3Gpc | fnl0 | ExpWindow | 0.3535 | 0.1515 | -0.3535 |
| 3Gpc | fnl0 | AllDiscrete | 0.3903 | 0.1772 | -0.3903 |
| 3Gpc | fnl0 | GaussDelta_a010kf | 0.3895 | 0.1765 | -0.3895 |
| 3Gpc | fnl0 | GaussDelta_a025kf | 0.3851 | 0.1733 | -0.3851 |

## 解释提示

1. 当高斯足够窄时，结果应连续逼近原始全离散 delta-shell。
2. 如果 `alpha=0.10` 与原始全离散几乎一致，就说明“只把 delta 换成极窄高斯”本身不会改变结论。
3. `alpha=0.25` 若出现明显变化，则代表壳层的径向展宽开始真正影响 2PCF。
