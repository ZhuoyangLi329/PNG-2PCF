# Rawbox P(k)–2PCF 差异审阅

## 交付状态

本次已完成源码与审计 JSON 审查，并在本地运行了14项合成/几何测试，全部通过。但后续创建 `rawbox_numerics.py` 的 GitHub 调用被接口安全检查拦截，**完整代码、长报告和测试 JSON 尚未成功上传本仓库**。它们通过本次 ChatGPT 对话中的 `GPT5.6-respose.zip` 附件交付。这里不再列出指向不存在文件的链接。

没有修改原生产代码、缓存、catalog、拟合结果或图。没有重跑原始 NPZ/MCMC，也没有成功取得仓库 PDF 图像逐图核验。下述旧拟合数值来自原审计 JSON；新的数值测试不是原始 rawbox 数据重拟合。

## 核心判断

联合协方差错误、独立模型形状失配、参数边界及不同模式带宽是不同问题，不能用其中一个解释全部现象。

已有 `review_data/joint_covariance_reproduction.json` 的量纲敏感 floor 确实破坏了联合比较：cross=0 时全部57个 xi 方向被抬升，xi 单-bin sigma 中位数增大约15.4倍。旧 joint 信息增益结论必须暂停；修复应采用 correlation-scaled Cholesky 和相同 MAP/MCMC metric，不是换另一个 floor。

本次额外确认了窄 P(k) bin 的重分箱问题：按生产 CachedRebin 的几何规则，真实 `[0.037,0.039)` bin 的1262个模式，聚合后按 k_eff 归类变成1094个，少13.31%；`[0.061,0.063)` 则从2994变成3402，多13.63%。这影响以原 Nmodes 为分母的相关 covariance/cross 构造，但不是已经测得的参数偏移。

P 的精确离散 mu 平均与 xi 的连续角积分不是同一算子。首个 bin 的18个模式满足 `<mu^4>=2/9`，而连续值为1/5。必须以相同径向节点隔离角向影响。

64点 Gauss–Legendre 对 Lorentzian I0 角矩在 k*sigma_s=24 时低估0.8613%，在90时低估64.97%。这是角矩误差，不是最终 xi 或参数误差；本地解析角矩实现与自适应积分最大相对差约1.14e-13。实空间脚本还存在更新 covariance 后不重新优化、继续保存旧 MAP 的流程问题。

## 对物理问题的判断

实空间最简模板已经失败，不能只怪 RSD：原审计中 P0 均值 chi-square/dof=45.23/14，xi0(s>=50)=331.28/28。s>=120 的 xi 后验很宽，不代表已精确解决 b1。

P0 的 sigma_s MAP 约8.25，xi0约8.77，两者并没有其边际中位数看起来那么不同。P02 则是 sigma_s MAP 接近零的边界解；不能把约0.98的中位数当作高斯检测。xi02约7.79的窄后验也不能脱离失败的均值形状检验解释。

优先物理假设是：不同带宽/核下，线性母谱与单一 FoG 缺失的宽带、BAO平滑及密度–速度耦合，被 b1 和 sigma_s 以不同方式吸收。需要先以实空间拆分 BAO damping、低阶宽带偏置与随机项，再在 RSD 中将 BAO平滑和残余 FoG 分开；同一母谱必须同时生成 P 和 xi。此解释仍需消融与留出检验，尚未定量证明贡献比例。

## 执行顺序

1. 冻结数据和物理，修 joint metric、窄-bin计数，验证单位不变、零cross恒等式、正定性；最终 covariance 下重新求 MAP。
2. 验证共同离散算子；分开测试角向离散、角积分、径向核、重分箱、UV尾部；报告 Delta chi-square_mean，而不仅是相对曲线误差。
3. 先修实空间形状，分别增加 BAO平滑、低阶宽带和随机项，检查留出尺度与 fNL 恢复。
4. 再区分 RSD 的真实速度效应与有效平滑，用四极留出预测和 sigma_s^2 profile 处理边界，先验变换必须显式。
5. 用配对相位、条件残差和模拟标定的共享/分离参数检验评价一致性；验证前不恢复正式 joint 信息增益图。

25相位 sample covariance 的秩最多24，不能直接逆57/89维。Hartlap/Percival不能恢复秩。现有 angular_totals 是 integral_-1^1 而非角平均，没有依据把全部 covariance 再乘2。

附件内包含 `REVIEW_zh.md`、`rawbox_numerics.py`、`audit_rawbox.py`、`test_rawbox_numerics.py`、`local_test_results.json` 和运行说明。它们是经过实现自检的诊断构件，不是已经验证的新生产管线。
