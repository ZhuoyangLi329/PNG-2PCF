# Mission 8：mode-count preserving k-cell

本次把原始全离散的 `delta-shell` 改成了**无自由参数**的 `k-cell`：每个 shell 的连续 k 体积严格匹配它的离散模式数 `g_q`。

其中 `mean_sigma < 0` 表示模型整体偏高，`mean_sigma > 0` 表示模型整体偏低。

| Box | Sample | Method | mean_abs_sigma | chi2_ndof | mean_sigma |
|---|---|---|---:|---:|---:|
| 1Gpc | fnl100 | ExpWindow | 0.0818 | 0.0130 | 0.0405 |
| 1Gpc | fnl100 | AllDiscrete | 0.1674 | 0.0404 | -0.1668 |
| 1Gpc | fnl100 | ModeCountCell | 0.8031 | 0.6915 | -0.8031 |
| 1Gpc | fnl0 | ExpWindow | 0.1256 | 0.0231 | 0.0945 |
| 1Gpc | fnl0 | AllDiscrete | 0.1296 | 0.0242 | -0.0916 |
| 1Gpc | fnl0 | ModeCountCell | 0.4104 | 0.1913 | -0.4104 |
| 3Gpc | fnl100 | ExpWindow | 0.2076 | 0.0495 | -0.1433 |
| 3Gpc | fnl100 | AllDiscrete | 0.4739 | 0.2417 | -0.4739 |
| 3Gpc | fnl100 | ModeCountCell | 0.5177 | 0.2827 | -0.5177 |
| 3Gpc | fnl0 | ExpWindow | 0.3535 | 0.1515 | -0.3535 |
| 3Gpc | fnl0 | AllDiscrete | 0.3903 | 0.1772 | -0.3903 |
| 3Gpc | fnl0 | ModeCountCell | 0.4162 | 0.1973 | -0.4162 |

## 物理解释

1. 原始全离散把每个 shell 视为零宽度 delta；mode-count cell 则把每个 shell 视为占据有限相空间体积的 radial cell。
2. cell 体积不靠调参，而是由 `g_q` 严格固定，因此这是无自由参数的修正。
3. 若该方法优于原始全离散，就说明“delta-shell 过于尖锐，而 mode-count preserving 的有限体积描述更接近真实有限盒统计”。
