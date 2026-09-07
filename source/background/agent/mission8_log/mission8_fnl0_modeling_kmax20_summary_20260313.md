# Mission 8：fnl=0 的 2PCF 建模图（kmax=20）

这次只画 `fnl=0` 的 2PCF 建模图，`kmax` 固定为 `20`，并统一比较三种方法：Baseline / ExpWindow / AllDiscrete。

| Box | Method | mean_abs_sigma | chi2_ndof | mean_sigma |
|---|---|---:|---:|---:|
| 1Gpc | Baseline | 0.9602 | 0.9522 | 0.9602 |
| 1Gpc | ExpWindow | 0.1256 | 0.0231 | 0.0945 |
| 1Gpc | AllDiscrete | 0.1296 | 0.0242 | -0.0916 |
| 3Gpc | Baseline | 0.2641 | 0.0887 | -0.2377 |
| 3Gpc | ExpWindow | 0.3535 | 0.1515 | -0.3535 |
| 3Gpc | AllDiscrete | 0.3903 | 0.1772 | -0.3903 |

其中 `mean_sigma < 0` 表示模型整体偏高，`mean_sigma > 0` 表示模型整体偏低。
