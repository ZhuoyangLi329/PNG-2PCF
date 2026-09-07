# Mission 8：k_split=0.05 的 PNG-split 两方案

本次固定 `k_split=0.05`，比较三种方法：
- `ExpWindow`
- `PNGSplit_k005 = xi_ref_cont + xi_png_disc(k<0.05) + xi_png_cont(k>0.05)`
- `PNGSplit_k005_NoHigh = xi_ref_cont + xi_png_disc(k<0.05)`

其中 `mean_sigma < 0` 表示模型整体偏高，`mean_sigma > 0` 表示模型整体偏低。

| Box | Sample | Method | mean_abs_sigma | chi2_ndof | mean_sigma |
|---|---|---|---:|---:|---:|
| 1Gpc | fnl100 | ExpWindow | 0.0818 | 0.0130 | 0.0405 |
| 1Gpc | fnl100 | PNGSplit_k005 | 0.4116 | 0.2443 | -0.2870 |
| 1Gpc | fnl100 | PNGSplit_k005_NoHigh | 0.3481 | 0.1741 | -0.2629 |
| 1Gpc | fnl0 | ExpWindow | 0.1256 | 0.0231 | 0.0945 |
| 1Gpc | fnl0 | PNGSplit_k005 | 0.2281 | 0.0653 | -0.2117 |
| 1Gpc | fnl0 | PNGSplit_k005_NoHigh | 0.2262 | 0.0642 | -0.2104 |
| 3Gpc | fnl100 | ExpWindow | 0.2076 | 0.0495 | -0.1433 |
| 3Gpc | fnl100 | PNGSplit_k005 | 0.5301 | 0.4936 | -0.4978 |
| 3Gpc | fnl100 | PNGSplit_k005_NoHigh | 0.4743 | 0.3706 | -0.4654 |
| 3Gpc | fnl0 | ExpWindow | 0.3535 | 0.1515 | -0.3535 |
| 3Gpc | fnl0 | PNGSplit_k005 | 0.3964 | 0.1949 | -0.3964 |
| 3Gpc | fnl0 | PNGSplit_k005_NoHigh | 0.3922 | 0.1880 | -0.3922 |

## 重点看 fnl=0

- 1Gpc: ExpWindow mean_sigma=0.0945, Split mean_sigma=-0.2117, SplitNoHigh mean_sigma=-0.2104
- 3Gpc: ExpWindow mean_sigma=-0.3535, Split mean_sigma=-0.3964, SplitNoHigh mean_sigma=-0.3922
