# Rawbox P(k)–2PCF 差异审阅

## 文件与验证范围

本目录已包含以下六个文件；它们不替换原生产代码、缓存、catalog、拟合或图：

| 文件 | 用途 |
|---|---|
| [REVIEW_zh.md](REVIEW_zh.md) | 完整科学审阅、证据、模型建议、五组第一阶段实验与执行agent任务书 |
| [rawbox_numerics.py](rawbox_numerics.py) | 单位不变的固定Gaussian metric、共同模式投影、解析角矩/壳核、cross与条件残差检查 |
| [audit_rawbox.py](audit_rawbox.py) | 只读原始theory缓存，检查joint floor、模式归属、cross、角积分与壳核误差 |
| [test_rawbox_numerics.py](test_rawbox_numerics.py) | 14项合成/几何实现测试，包括模式功率Monte Carlo与审计脚本合成缓存测试 |
| [local_test_results.json](local_test_results.json) | 本次实际运行的数值测试记录；不是原始rawbox重新拟合结果 |
| README.md | 范围、运行说明及接入方法 |

起始审阅版本：`273753d05e5fe143f61b8edd646a0e8593ba36ea`。科学范围限于rawbox实空间与平行视线RSD，不推进lightcone、survey window、RIC/GIC。

**实际完成：**源码与审计JSON审查、相关论文公式核对、14项本地合成/几何测试通过。**没有完成：**原始NPZ重拟合、catalog/FCFC/jaxpower重测、长链MCMC、仓库PDF图像逐图核验。原拟合数值来自仓库审计，不能把这里的测试称为新物理管线验证通过。

## 最重要的判断

联合协方差的混合量纲floor、独立模型形状失配、参数边界和不同模式带宽必须分开处理。已有原数组审计显示cross=0时全部57个xi方向被floor抬升，xi单-bin sigma中位数增大约15.4倍；旧joint信息增益结论需要暂停。

本次额外用源码等价几何重现了窄P-bin归属问题：`[0.037,0.039)`原有1262个模式，CachedRebin后按代表k归类变成1094个，少13.31%。这是计数/分组差异，不是已经测得的参数偏移；原始缓存仍需由audit脚本核对。

P保留实际离散mu，xi目前采用连续角积分再做离散径向求和，二者不是完全同一角操作。64点Gauss–Legendre对Lorentzian I0在k*sigma_s=24时低估0.8613%，在90时低估64.97%；这不是最终xi误差，必须经过真实谱、壳核与covariance加权。

实空间无FoG也存在均值形状失败，所以不能只怪RSD坐标映射。优先物理检验是将BAO位移平滑与残余FoG分开，并逐项加入低阶bias/stochastic结构；尚未证明它们各自解释了多少b1或sigma_s偏移。

## 运行

依赖Python、NumPy和SciPy。审计脚本不导入带NERSC绝对路径的项目模块，也不需要emcee、desilike或FCFC。

```bash
# 从仓库根运行实现自检。
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 \
  python GPT5.6-respose/test_rawbox_numerics.py

# 读取原始缓存；输出位置必须不存在。
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 \
  python GPT5.6-respose/audit_rawbox.py . \
  --out /path/to/new-audit-run/audit.json
```

第二条命令另写同名NPZ，已有输出拒绝覆盖。默认原始缓存路径在脚本`CACHE`中；必须有匹配SHA256、status=pass、cosmology=abacus_c000的JSON sidecar。可用`--cache`显式指定。

增加`--measurement /path/to/existing_P02.npz`核对测量`k_edges`和`nmodes`；不提供时明确标记“未与测量比较”，不会假装完成。`--b1`、`--sigma`、`--nbar`改变审计fiducial。当前低k角向修正仅计算至0.095，用于定位而不自动修正生产xi；完整切换/UV/PNG参数扫描见报告。

## 最小接入方法

```python
from rawbox_numerics import GaussianMetric, assemble_covariance
from scipy.optimize import least_squares

# Cxp的行是xi，列是P。
C = assemble_covariance(Cpp, Cxx, Cxp)
metric = GaussianMetric(C)
fit = least_squares(lambda theta: metric.residual(data - model(theta)),
                    x0, bounds=(lower, upper))

# MCMC另行保留明确先验与边界检查，使用完全相同的metric。
loglike = lambda theta: -0.5 * metric.chi2(data - model(theta))
```

更换协方差后须在最终冻结的metric下重新求MAP。不要把正定性报错替换成更大的floor；若存在真实线性冗余，显式定义支持子空间与有效秩。25相位sample covariance的秩最多24，Hartlap不能恢复缺失秩。

实施顺序：先修joint metric与最终C下的MAP，再做同模式数值闭合，再检验实空间模型，随后RSD密度–速度/阻尼结构，最后配对参数与覆盖率验证。正式约束图须等这些门通过后再更新。
