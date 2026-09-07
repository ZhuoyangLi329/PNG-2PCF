# PNG-2PCF：给 GPT5.6pro 的 rawbox 科学审阅包

**核心任务：诊断并提出解决 rawbox 中 P(k) 与 2PCF 不一致的可验证方案。**

这是一份为独立审阅整理的研究快照，包含早期方法探索、当前代码、原始测量向量/结果摘要、拟合产物、参考论文和大量图。不是一套已经验证通过的统一模型，也不是把所有历史脚本直接批量执行的工程。

## 按这个顺序阅读

0. **[给 GPT5.6pro 的交接与审阅任务](给GPT5.6pro.md)** — 问题、证据标准、阅读顺序、希望返回什么。
1. [背景与方法演化](docs/01_background.md)
2. [P(k)、PNG-2PCF 模型与参数推断](docs/02_models_and_inference.md)
3. [协方差方法、统计口径与风险](docs/03_covariance.md)
4. [数据集、结果和冲突的证据地图](docs/04_results.md)
5. [已知问题与建议的判别实验](docs/05_review_questions.md)
6. [运行边界、依赖与复现说明](docs/06_reproducibility.md)

查阅入口：[图册](docs/FIGURES.md) · [代码索引](docs/CODE_INDEX.md) · [文档与结果索引](docs/RESULT_INDEX.md) · [论文索引](papers/README.md) · [文件清单和哈希](provenance/FILES.json)

## 范围

- Quijote 1 Gpc 周期盒，FastPM 1/3 Gpc 周期盒；实空间与红移空间。
- Abacus c000 x25、z=0.725、L=2000 Mpc/h：paired real/RSD、P0/P2、xi0/xi2、单独及联合拟合。
- PNG-base HOD-MAP c300/c302、z=0.5 实空间周期盒，以及已存在的 host-halo/尺度诊断。
- 不纳入 lightcone/cut-sky 应用数据、结果和运行工作流。保留必要通用理论/数值依赖及背景论文；部分历史 rawbox 对比图同时画有 subbox 控制，索引中标明，仅要求审阅 rawbox 面板。

## 请先知道的两件事

**fNL 后验相近并不等于均值曲线通过验证。** x25 平均数据常用单盒 covariance 报告“一盒的精度”，而模型的均值形状检验另用 C_single/25。不要混用这两个 chi²。

**最新 joint 的“增益很小”不是可靠的物理结论。** 本次整理附带的[离线复核](review_data/joint_covariance_reproduction.json)重现了原始量纲特征值截断问题：四向 cross=0 控制中57个xi方向被抬升，xi单-bin sigma中位数变成原来的约15.4倍。原代码和历史结果均保留，尚未重跑修正后的joint。

## 文件组织

| 目录 | 内容 |
|---|---|
| `docs/` | 新写的中文导读、证据索引及原任务书rawbox摘录 |
| `source/background/` | 前身项目原始代码、过程笔记、notebook、历史图 |
| `source/project/` | 当前项目rawbox源码、测量与结果、审计、图；保留源相对路径 |
| `papers/` | 论文导航和补充方法论文PDF；原项目论文PDF在source内 |
| `notebook_text/` | notebook源码的纯文本镜像；图和执行输出仍在原notebook |
| `review_data/` | 为审阅生成的NPZ数组目录、统计反例及复核结果 |
| `tools/` | 无需NERSC的索引/完整性与joint疑点复核工具 |
| `provenance/` | 来源、SHA256、摘录/范围裁剪说明、缺失与外部依赖 |

本仓库为私有研究审阅副本。第三方论文和软件版权仍归各自权利人；未赋予它们新的许可证。原始catalog和某些巨大链/缓存未打包，具体以缺失清单和复现说明为准。
