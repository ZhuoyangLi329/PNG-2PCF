# Rawbox P(k)–2PCF 诊断与修复建议

先读 [完整中文审阅与执行任务书](REVIEW_zh.md)。本目录按请求保留 `GPT5.6-respose` 拼写，只新增诊断材料，不覆盖原生产代码、缓存、测量或拟合结果。

**核心结论：联合协方差有确定的数值错误；独立探针仍有估计器、协方差与均值模板问题。修正 joint 不等于解决独立 P–xi 偏差。** P0 的宽 sigma_s 后验与 xi0 有重叠，明显分离出现在 P02 与 xi02。实空间没有 FoG 仍失败，不能将问题全归因于速度。

## 交付文件

| 文件 | 内容 |
|---|---|
| [REVIEW_zh.md](REVIEW_zh.md) | 结果对照、证据表、因果分析、最多五个判别实验、最小物理扩展和执行门 |
| [rawbox_diagnostics.py](rawbox_diagnostics.py) | 统一固定协方差 likelihood、共同模式算子、解析角矩/壳核、只读缓存审计 |
| [test_rawbox_diagnostics.py](test_rawbox_diagnostics.py) | 14 项确定性/合成测试，含 50,000 次模式功率抽样 |
| [local_test_results.json](local_test_results.json) | 本次实际运行的测试数值及环境版本 |

本次几何复算发现：将 CachedRebin 聚合中心归入窄 P 箱会改变成员模式数；例如 `[0.037,0.039)` 从1262变成1094，差−13.31%。GL64 在 `k sigma=24` 对 Lorentzian I0 的误差为−0.8613%。这些是几何/标量误差，**不是已测出的 b1 偏移或 xi 总误差**。

## 已完成与限制

源码、说明与审计 JSON 已核对；14 项本地测试全部通过。解析角矩对自适应积分最大相对误差约1.14e-13，壳核最大绝对误差约1.54e-14，单位变换 chi² 相对误差约2.22e-16。

**未运行原始 NPZ 的本地审计、catalog 重测或 MCMC，也未逐图完成仓库 PDF 视觉核验。**拟合数字引用仓库原审计；本地代码验证使用几何/合成输入。不要把测试通过标成新的物理模型已验证。报告列出了具体需要补齐的执行产物。

## 运行

依赖 Python、NumPy、SciPy。无须导入 desilike、cosmoprimo、FCFC 或含 NERSC 绝对路径的原脚本。

```bash
# 从仓库根目录运行；会更新本地测试结果 JSON。
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 \
  python GPT5.6-respose/test_rawbox_diagnostics.py

# 读取现有原始理论缓存，不自动重建。输出路径必须尚不存在。
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 \
  python GPT5.6-respose/rawbox_diagnostics.py \
  --cache source/project/outputs/task43_outputs/rsd_validation/theory/task43_rsd_fulldiscrete_z0p725000_box2000_kmax3_ell02.npz \
  --out new-audit-run/audit.json
```

第二条命令校验缓存 SHA256、c000、z=0.725、盒长和 shell 定义，写一个 JSON 和同名 NPZ，拒绝覆盖既有输出。它审计原来的 cross=0 floor、bin计数、cross的最大典型相关系数、角积分与壳核预测差；不自动修复生产缓存或重新拟合。fiducial 固定为当前 b1=2.55、sigma_s=8、fNL=0、nbar=0.000162131295。

测量文件 nmodes 核验、低 k 角向修正、UV/PNG 参数扫描和原始模型重拟合没有包含在自动审计内，具体见报告实验2。原缓存不可用时脚本明确失败，不使用合成数据冒充替代。

## 接入拟合的最小示例

```python
from rawbox_diagnostics import GaussianMetric, assemble_covariance
from scipy.optimize import least_squares

# 需将本目录加入 Python import path。Cxp 的行是 xi，列是 P。
C = assemble_covariance(Cpp, Cxx, Cxp)
metric = GaussianMetric(C)
fit = least_squares(lambda theta: metric.residual(data - model(theta)),
                    x0, bounds=(lower, upper))
# MCMC 保留明确的先验/边界检查，再使用同一 metric。
loglike = lambda theta: -0.5 * metric.chi2(data - model(theta))
```

更换协方差后，在最终冻结 metric 下重新求 MAP。矩阵无效时应查明原因，不要将报错改为更大的 floor。共同模式 Gaussian covariance 只验证相应场模型与模式约定，不自动包含 FCFC 的 pair normalization、有限 mu-bin、paint/alias 或非 Gaussian trispectrum。

## 其他复核记录

目录中已有的 `rawbox_numerics.py`、`audit_rawbox.py`、`test_rawbox_numerics.py` 及其他复核文档保留。本页的最新测试结果对应 **test_rawbox_diagnostics.py**；不同脚本版本和 Monte Carlo 随机种子的结果不能混称同一次运行。使用本页列出的脚本可复现本次交付。
