# Mission 8：把 mu 纳入后的低-k 离散模式测试

这次不再只用 `P0(k)`。低-k 部分先由 desilike 输出 `P0,P2,P4`，再用

$$
P(k,\mu) \approx P_0(k) + P_2(k)L_2(\mu) + P_4(k)L_4(\mu)
$$

去近似各向异性的红移空间功率谱。然后只对 PNG 增量项做离散 low-k 处理。

## 3Gpc

| 方法 | mean|Δ/σ| | chi2/ndof |
|---|---:|---:|
| 3Gpc_Baseline | 0.5601 | 0.3808 |
| 3Gpc_ExpWindow | 0.2076 | 0.0495 |
| 3Gpc_ShellPNG_old | 0.1276 | 0.0349 |
| 3Gpc_ModeMu_A_best | 0.1276 | 0.0349 |
| 3Gpc_ModeMu_B_best | 0.1276 | 0.0349 |

- 最优 ModeMu A：Nshell=2
- 最优 ModeMu B：Nshell=2
- A 与 B 最优曲线最大绝对差：1.626303e-19

## 1Gpc

| 方法 | mean|Δ/σ| | chi2/ndof |
|---|---:|---:|
| 1Gpc_Baseline | 2.1129 | 4.5287 |
| 1Gpc_ExpWindow | 0.0818 | 0.0130 |
| 1Gpc_ShellPNG_old | 0.3005 | 0.1318 |
| 1Gpc_ModeMu_A_best | 0.3005 | 0.1318 |
| 1Gpc_ModeMu_B_best | 0.3005 | 0.1318 |

- 最优 ModeMu A：Nshell=5
- 最优 ModeMu B：Nshell=5
- A 与 B 最优曲线最大绝对差：2.168404e-18

## 解释

1. 方案 A 是真正的“逐模式 P(k,mu) 求和”；方案 B 是“每个 shell 内先对离散 mu 平均，再乘 g_q”。
2. 对 xi0(r) 来说，两者理论上应当等价，因为同一 shell 上 `j0(k r)` 相同；脚本里也直接验证了两者差异几乎为 0。
3. 因此真正的改进点，不是把 `g_q` 展开成一个个模式，而是把低-k 理论从 `P0(k)` 升级成含离散 `mu` 的 `P(k,mu)` 近似。
4. 但这次数值结果显示，纳入 `mu` 以后和旧的 `P0-shell PNG` 几乎完全一样。原因不是代码没生效，而是对完整 cubic shell 而言，离散模式平均满足 `mean[L2(mu)] = 0`；所以 `P2` 的 PNG 增量在 shell 平均后会被抵消。
5. 另一方面，`P4` 项本来有可能留下非零净修正，但在当前 best-fit 下，PNG 引起的 `P4` 增量几乎为 0，因此最终 `P(k,mu)` 修正并没有明显偏离原先的 `P0-shell` 结果。
6. 换句话说，这次测试说明：**在当前理论和参数口径下，1Gpc 解析法差，并不是因为漏掉了低-k 的离散 `mu` 结构；至少在 `P0+P2+P4` 这个层级上，漏掉 `mu` 不是主误差源。**
