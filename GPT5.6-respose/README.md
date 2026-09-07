# Rawbox P(k)–2PCF 差异审阅

这是按仓库任务书新增的诊断区，不覆盖原有代码、缓存、测量、拟合或图。目录名保留请求中的 `GPT5.6-respose` 拼写。

审阅起点：`273753d05e5fe143f61b8edd646a0e8593ba36ea`。范围是 Abacus rawbox 的实空间与平行视线红移空间；不推进 cutsky/lightcone 的窗口或积分约束修复。

## 先读什么

[完整审阅报告](REVIEW_zh.md)包含证据、优先级、最多五组第一阶段实验、最小物理模型、协方差政策及执行代理交接。

核心判断：**联合矩阵确有确定的实现错误；独立拟合又存在未解决的形状问题。两者必须分别修。P0 的宽后验与 P02 的边界解也不能混为一种“同样的张力”。**

本次新增的可复现线索包括：

- 当前径向 CachedRebin 分组不适合直接重新分配到窄 P(k) bins。按源代码的几何规则重算，`[0.037,0.039)` 原有 1262 个模式，却被聚合中心归类成 1094 个，少 13.31%。这是分组计数差异，不是已经测出的参数偏移。
- 64 点角积分在 `k sigma_s=24` 时将 Lorentzian 的 `I0` 矩低估 0.8613%；在 90 时低估 64.97%。这些不是 xi 总误差，需经过真实谱与核加权。
- 实空间检查更新协方差后没有重新优化，却继续保存旧 MAP 预测；P02 的均值形状检验也不能沿用 P0 的“通过”结论。

## 文件与验证范围

| 文件 | 用途 |
|---|---|
| `REVIEW_zh.md` | 科学诊断、证据与交接 |
| `rawbox_numerics.py` | 不做 eigenvalue floor 的统一 likelihood；共同模式投影；解析角矩与壳平均核；条件残差 |
| `audit_rawbox.py` | 读取原始 theory NPZ，检查 floor、模式计数、交叉块、角积分与核误差；不运行 MCMC |
| `test_rawbox_numerics.py` | 14 项确定性／合成测试，包括 50,000 次模式功率抽样与审计脚本的合成缓存端到端测试 |
| `local_test_results.json` | 本次实际运行的测试结果，不是原始 rawbox 数据重跑结果 |

**已实际完成：**源码和审计 JSON 审查、公开论文相关公式核对、14 项测试全部通过。角矩与自适应积分的最大相对差约 `1.14e-13`；壳平均核最大绝对差约 `1.54e-14`；单位变换下 chi-square 相对差约 `2.22e-16`。

**尚未完成：**原始 NPZ 审计重跑、catalog/FCFC/jaxpower 重测、长链 MCMC、仓库 PDF 图像逐图核验。当前会话成功读取了仓库文本，但未能取得原始二进制数组／图像供本地运行。因此原有拟合数值引用自仓库审计；图像外观没有被当作已核验的证据。不能把这里的单元测试称为物理模型验证通过。

## 运行

需要 Python、NumPy、SciPy；不需要导入 `desilike`、`cosmoprimo`、FCFC 或带 NERSC 绝对路径的项目脚本。

```bash
# 从仓库根目录运行实现自检。
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 \
  python GPT5.6-respose/test_rawbox_numerics.py

# 读取原始缓存，输出到一个不存在的新文件位置。
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 \
  python GPT5.6-respose/audit_rawbox.py . \
  --out /path/to/new-audit-run/audit.json
```

第二条命令另写一个同名 `.npz`，保存未被修改的边块、交叉块、精确离散 P02 协方差及预测改变量。已有输出一律拒绝覆盖。默认缓存路径写在脚本的 `CACHE` 中；缓存必须有匹配 SHA256 和 `abacus_c000` 标记的 JSON sidecar。也可用 `--cache` 指定文件。

增加 `--measurement /path/to/existing_P02.npz` 会核对该测量文件的 `k_edges` 和 `nmodes`；不提供该参数时，报告会明确写“未与测量 nmodes 比较”，不会假装完成这项核验。`--sigma`、`--b1`、`--nbar` 可改变审计 fiducial；默认复现当前 `b1=2.55, sigma_s=8` 的口径。

`audit_rawbox.py` 的低 k 角向修正目前只计算到 `0.095 h/Mpc`，用于定位问题，不自动改动最终 xi。完整 k-switch、PNG 参数与 UV 收敛扫描见报告实验 2。

## 接入拟合的最小改法

```python
from rawbox_numerics import GaussianMetric, assemble_covariance
from scipy.optimize import least_squares

# Cxp 的行是 xi、列是 P；错误/病态矩阵会报错，不会自动改变权重。
C = assemble_covariance(Cpp, Cxx, Cxp)
metric = GaussianMetric(C)
fit = least_squares(lambda theta: metric.residual(data - model(theta)),
                    x0, bounds=(lower, upper))

# MCMC 中保留原先明确的先验与边界检查，并使用同一个 metric。
loglike = lambda theta: -0.5 * metric.chi2(data - model(theta))
```

协方差更换后要在最终冻结的 metric 下重新求 MAP。不要将这里的报错替换成更大的 floor。若交叉块违反正定条件，应修正交叉协方差构造／校准；若存在真正线性冗余，应显式定义支持子空间和有效秩。

本区是可测试的诊断和修复构件，**不是已完成验证的新生产管线**。