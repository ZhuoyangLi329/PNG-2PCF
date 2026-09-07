# 独立数值核验记录与交付边界

这份记录补充 `REVIEW_zh.md`。不修改原生产代码、测量、缓存、链或正式图。仓库文件列表已确认存在完整报告、三个 Python 文件和测试摘要；因此旧 README 中“完整文件均未上传”的状态已经过时。

## 本次实际执行

在当前计算环境对聊天附件中的独立数值构件副本运行：

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python GPT5.6-respose/test_rawbox_numerics.py
```

结果：14 tests，0 failures，0 errors；NumPy 2.3.5，SciPy 1.17.0。没有运行原始 rawbox NPZ 的重拟合，没有运行 MCMC，没有重测 catalog。审计驱动器只在明确标为 synthetic_fixture 的合成缓存上做了端到端测试。

附件副本与本目录既存代码并非逐字节相同；不要把这次运行说成既存仓库版本的原样复跑。附件中的 `LOCAL_MANIFEST.json` 和下列 SHA256 确定本次实际执行版本：

| 文件 | SHA256 |
|---|---|
| rawbox_numerics.py | 382421d3c3c52835c6744875b93eebc4048995d39a7dd0073a4d342ddc2403dd |
| audit_rawbox.py | f0fd9836f0e1fba283209b13df1dc1531dae0892528dcd16ccb4bc17c184cce1 |
| test_rawbox_numerics.py | 415182cfa8ef9d0108af537f10dec60661d21e4910414430bb5a493474549b16 |
| local_test_results.json | 7755d8f10d1c0201031ac3c70d06cb7a186f2baf4ff381002a0a3d35419b7025 |

## 可复现的数值结果

1. 固定协方差经无量纲 Cholesky whitening 后，改变 P/xi 单位的 chi-square 相对误差为 2.22e-16；条件 chi-square 分解误差为0。拒绝非正定和未解析近零方向，不做自动 floor。
2. Lorentzian 解析角矩与自适应积分在所测网格的最大相对差为1.14e-13；解析体积壳平均核与积分最大绝对差为1.54e-14。
3. 50,000次独立复模式功率抽样检验共同模式 Gaussian covariance，40个独立模式，最大矩阵元偏差约1.95个近似标准误。该检验不是非Gaussian halo covariance的验证。
4. L=2000、[0.003,0.005) 的18个模式给出 mu²、mu⁴、mu⁶ 平均分别为1/3、2/9、1/6，不等于连续角积分的全部矩。径向 P 恒定的 Kaiser 算例在 b=2.55、f=0.81 时，P2/P 分别为离散3.898125、连续3.128914；不能外推成整个实际谱的偏差。
5. 源码等价的 dk=0.1*kf 重分箱几何：P bin [0.037,0.039) 从精确1262个模式变成按聚合中心归类的1094个，差-13.3122%；[0.061,0.063) 从2994变成3402，差+13.6273%。这是几何/权重问题，不是已经测到相同比例的参数偏移，也不是原始NPZ损坏。
6. 64点GL对 I0 的相对误差：k*sigma_s=24时-0.861263%，90时-64.970471%。256点在90时仍有-0.571720%。这些是角矩误差，必须经实际谱与核投影，不能当成总xi或参数误差。

## 补充两个解释边界

原始四向 JSON 的独立 P02 chi-square_single=2.9551422，xi02=11.8911664。按冻结的 C_mean=C_single/25，分别对应73.8786/28和297.2792/54；名义PTE约5.25e-6和2.51e-35。边界和协方差失配使精确PTE需要模拟校准，但至少不能把P0的通过结论沿用到P02。

常数 stochastic power 在无限体积完整变换中为contact项；周期盒去零模后更精确地为 N0[delta_periodic(r)-1/V]。去自对和N(N-1)归一化必须与零模定义一起处理。将常数谱硬积到kmax=3会造成非局域振铃，不应直接作为大尺度xi均值修复。

## 下一步

先做原缓存只读审计、likelihood逐点可加性、P-bin成员一致性和最终冻结C下的MAP；再做同模式算子闭合及实空间物理消融。原始catalog、MCMC和私有PDF图像的逐图复核仍未完成。测试通过不等于均值模型、halo covariance或正式参数约束已经通过科学验证。
