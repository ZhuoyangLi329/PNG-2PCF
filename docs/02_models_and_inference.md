# 02 PNG-P(k)、PNG-2PCF与参数推断

## 从理论到数据向量

实空间最简信号：\(P_g(k)=[b_1+f_{NL}b_\phi/\mathcal M(k)]^2P_{mm}^{lin}(k)\)。P侧有些分支另拟合常数残余stochastic power；它的参数sn0数值要乘代码的SN0_SCALE，不能直接当作物理功率单位。

当前Abacus RSD信号：

\[
P_g^s(k,\mu)=[b_1+f_{NL}b_\phi/\mathcal M(k)+f\mu^2]^2P_{mm}^{lin}(k)
[1+(k\mu\sigma_s)^2/2]^{-2}.
\]

源码为`source/project/codes/task43/task43_rsd_model.py::FullDiscreteRSDModel.evaluate`；Gaussian-FoG作为另一个选项存在，但不是默认结果。

**P的预测必须匹配测量平均**：\(P_{\ell,i}=(2\ell+1)N_i^{-1}\sum_{\mathbf k\in i}P(k,\mu_\mathbf k)L_\ell(\mu_\mathbf k)\)。单极就是模式平均；四极是5倍模式平均。连续积分版本的系数是5/2且积分权重和为2；不能将其原样搬到模式平均。

**xi的预测必须匹配径向shell**：\(\xi_{\ell,j}=V^{-1}\sum_qg_qP_\ell(k_q)K_{\ell,j}(k_q)\)，其中
\(K_{\ell,j}=i^\ell\int_{s_-}^{s_+}s^2j_\ell(ks)ds/\int_{s_-}^{s_+}s^2ds\)。单极有解析volume average，高阶由数值求积实现。ell=2包含i²负号；核与多极系数不可重复使用。

Abacus xi源码以连续mu积分获得P_ell，再按离散径向shell求和；P侧用实际离散mu。这是值得审阅的差异，不应预先称它们为有限bin范围下的严格双射。

## 测量端

| 分支 | P测量/处理 | xi测量/处理 |
|---|---|---|
| 早期FastPM/Quijote | 已有周期盒P文件，POWSPEC或相应原测量；读列定义见加载函数 | 已有周期盒2PCF文件 |
| Abacus rawbox x25 | jaxpower CPU，mesh400，TSC/interlacing3/compensated，LOS=z；P02扩展复核P0 | FCFC周期DD归一化；xi(s,mu)投影多极；实/RSD成对 |
| PNG-base HOD-MAP | 同类周期mesh400，P49连续bins | FCFC、pycorr/Corrfunc、CuCount三引擎；ordered N(N−1)，s30–350 |

不应把项目早期“CuCount统一标准”概述强行覆盖rawbox后来的FCFC实跑记录。mesh/assignment、粒子坐标wrap、pair归一化、bin边缘和shot-noise都要以当前case源码/JSON为准。

## 似然与自由参数

固定covariance时：\(-2\ln\mathcal L=(d-m(\theta))^TC^{-1}(d-m(\theta))\)，再乘先验得到后验。参数相关和非线性会使MAP、均值、中位数不同。固定协方差时可省略常数logdet；如果在每个参数点改变C，就必须重新考虑logdet和统计定义。

| 数据集/实验 | 自由或固定参数 | 主要区别 |
|---|---|---|
| Quijote free-sn0 | P: fNL,b1,sigma_s,sn0；p=1.2 | UltraNest；sample covariance；xi非零s无常数mean项 |
| FastPM实空间rawbox | fNL,b1；P主版本free sn0；sigma_s=0,p=1.2 | z=1；shell-average修正；98盒 |
| Abacus RSD x25 | P: fNL,b1,sigma_s,sn0；xi:前3个；p=1 | emcee长链；单盒Gaussian C；另做均值/phase形状门 |
| Abacus最简实空间检查 | fNL,b1；无FoG、无free sn0 | 不等于与RSD P自由参数完全匹配的A/B |
| PNG-base HOD-MAP | 固定catalog fNL反推p；或p=1反推fNL | 单ph000；不同HOD/cosmology标签不可视为独立样本相乘 |

P拟合范围和xi求和范围不同，是因为xi是完整理论变换后的观测量。不能任意给xi加一个与P的kmax相同的硬截断，随后把得到的另一个统计量当成原xi。可构造明确的新带通统计量作数值诊断，但必须更改测量、理论、covariance一起定义，不能暗改原分析。

精确参数先验、边界、sn0单位和scale selection见源函数与每个summary，导读不是配置文件。
