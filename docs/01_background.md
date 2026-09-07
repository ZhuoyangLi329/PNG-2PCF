# 01 背景与方法演化

目标是用galaxy/halo clustering的尺度依赖bias限制local PNG。项目当前讨论的是固定背景宇宙学下的fNL、bias和nuisance推断；不是通过P/xi再“限制2PCF这个统计量”。

写成通常的约定：

\[
\Delta b(k,z)=f_{NL}\,b_\phi/\mathcal M(k,z),\qquad b_\phi=2\delta_c(b_1-p).
\]

\(\mathcal M\propto k^2T(k)D(z)\)，故低k的PNG bias增强。代码中名为`alpha`的数组常实际储存\(1/\mathcal M\)，不能按变量名猜乘除关系；尤其要核查primordial potential的CMB/LSS与h单位。

无限体积中PNG平方项使\(P\sim k^{n_s-4}\)，xi积分的低k被积函数约为\(k^{n_s-2}\)。实际有限样本的定义必须包含有限模式或均值约束。周期盒允许\(\mathbf k=2\pi\mathbf n/L\)，不包含零模；这个事实是FullDiscrete的出发点，不是添加一个可任意调节的低k平滑窗。

## 发展顺序

1. `source/background/agent/mission3_log/`：FFTLog/mcfit、积分上下限敏感性。
2. `mission4_log/`：不同L与fNL=0对照，分离盒长与PNG效应。
3. `mission5_log/`、`mission6_log/`、`mission7_log/`：经验IR窗、N-body/mass-cut验证；属于历史探索，并非当前标准。
4. `mission8_log/`：全离散、模式角度、shell平均、混合变换与失败方案；`mission8_all_methods_summary_20260315.md`是对比入口。
5. `mission9_log/databin_method.md`：直接使用测量P分箱能作为数值桥接，但不能替代未知参数下的前向理论。
6. `mission10_log/`：在P拟合时对理论进行与测量一致的离散bin-average，再把连续best-fit理论送入FullDiscrete。
7. `mission11_log/`：保留模式简并度与核权重的加速、误差比较。
8. `mission12_log/`：直接P似然与xi似然的参数差异及sigma_s/rmin/p扫描。

核心notebook为`source/background/pk-pcf-model/model_3_20_fast.ipynb`。它是早期方法示范，里面的fixed-sn0、kmax和中心核等配置不能覆盖后来的正式shell-average/free-sn0口径。

## 文献怎么帮助审阅

- Wands–Slosar 2009：低k形式发散与有限样本定义，重点第IV节。
- Slosar et al. 2008：b_phi与形成历史/p，不要把b_phi=2δc(b1−1)视为所有HOD的无条件定理。
- Brown et al. 2403.18789：模拟标定的配置空间PNG响应及2PCF/3PCF框架。它与当前FullDiscrete方法不同，不能用它的验证直接替代本方法验证。
- Hartlap、Percival、Grieb：分别帮助检查sample precision、参数误差传播和Gaussian多极covariance。
- DESI PNG与BAO论文保留为原项目参考资料；当前不审阅它们的survey应用。

具体PDF与链接见[论文索引](../papers/README.md)。
