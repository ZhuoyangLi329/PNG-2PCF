# 论文阅读索引

论文PDF保留原作者与出版方版权，仅用于本私有研究审阅。下面区分原项目已有参考和本次为协方差/PNG审阅补充的文献；补充论文不表示历史代码已经依照它实现。

## 核心方法

| 论文 | 重点 | PDF |
|---|---|---|
| Wands & Slosar 2009, [0902.1084v2](https://arxiv.org/abs/0902.1084v2) | 第IV节：PNG相关函数、IR与均值定义 | [PDF](Wands_Slosar_2009_0902.1084v2.pdf) |
| Slosar et al. 2008, [0805.3580](https://arxiv.org/abs/0805.3580) | scale-dependent bias，形成历史与p | [PDF](Slosar_2008_0805.3580.pdf) |
| Brown et al., [2403.18789v1](https://arxiv.org/abs/2403.18789v1) | 配置空间PNG、2PCF/3PCF及模拟标定；与本方法不同 | [PDF](../source/project/observelearn/ref/2403.18789v1.pdf) |

## 协方差

| 论文 | 重点 | PDF |
|---|---|---|
| Hartlap et al., [astro-ph/0608064](https://arxiv.org/abs/astro-ph/0608064) | sample inverse偏差、独立样本与秩限制 | [PDF](Hartlap_2007_astro-ph_0608064.pdf) |
| Percival et al., [1312.4841](https://arxiv.org/abs/1312.4841) | covariance噪声传播到参数误差 | [PDF](Percival_2014_1312.4841.pdf) |
| Grieb et al., [1509.04293](https://arxiv.org/abs/1509.04293) | 各向异性P/xi Gaussian covariance、多极系数与bin平均 | [PDF](Grieb_2016_1509.04293.pdf) |

## 原项目保留的背景参考（本轮不推进survey应用）

- Ross et al., [1208.1491v3](https://arxiv.org/abs/1208.1491v3)：BOSS PNG与系统误差。[PDF](../source/project/observelearn/ref/1208.1491v3.pdf)
- DESI PNG, [2411.17623v2](https://arxiv.org/abs/2411.17623v2)：观测侧模型/covariance背景。[PDF](../source/project/observelearn/ref/2411.17623v2.pdf)
- DESI 2024 VI, [2404.03002v3](https://arxiv.org/abs/2404.03002v3)：BAO宇宙学分析背景。[PDF](../source/project/observelearn/ref/2404.03002v3.pdf)
- DESI DR2 II, [2503.14738v3](https://arxiv.org/abs/2503.14738v3)：BAO背景。[PDF](../source/project/observelearn/ref/2503.14738v3.pdf)
- de Mattia & Ruhlmann-Kleider, [1904.08851](https://arxiv.org/abs/1904.08851)：积分约束定义。[PDF](../source/project/observelearn/ref/integral_constraint/1904.08851_integral_constraints_spectroscopic_surveys.pdf)
- [2106.06324](https://arxiv.org/abs/2106.06324)：功率谱窗口矩阵背景。[PDF](../source/project/observelearn/ref/integral_constraint/2106.06324_window_function_matrix.pdf)

原项目PDF使用原有版本；不要将联网见到的新版本数值无声替换进历史实验解释。下载来源与SHA256见`provenance/downloaded_papers.json`及总清单。

原项目另有零字节文件 `astro-ph_9207254.pdf`，对应arXiv地址返回404。本包没有将它伪装成可读论文，已在curation清单记录排除原因。
