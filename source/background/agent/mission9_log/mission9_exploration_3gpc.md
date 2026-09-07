# Mission 9 自由探索 — 完整总结

## 最终结论：DataBin(N_fit) 方法

### 方法描述
在 FullDiscrete 离散求和框架中，将拟合范围内（前 N_fit 个 k-bin）的 P(k) 替换为实测 bin 平均值，
其余 k 用 best-fit 模型：

```
ξ₀(r) = (1/V) Σ_q g_q × P_eff(k_q) × j₀(k_q r)

P_eff(k_q) = P_data_bin   (k_q 在前 N_fit 个 bin 内)
           = P_model(k_q)  (其余)
```

N_fit = 拟合 P(k) 时使用的 bin 数（通常 ~20）。

### 全平台验证结果

#### FastPM (本项目样本)
| 方法 | 3Gpc fnl100 mean|Δ/σ| | 1Gpc fnl100 mean|Δ/σ| |
|---|---|---|
| Baseline | 0.835 | 1.917 |
| ExpWindow (含归一化) | 0.498 | 0.231 |
| FullDiscrete | 0.643 | 0.277 |
| **DataBin(20bins)** | **0.441** | **0.208** |

#### Quijote N-body (500 realizations, L=1Gpc)
| fnl_true | **DataBin(20)** | FullDiscrete | ExpWindow |
|---|---|---|---|
| 0 | **0.090** | 0.107 | 0.136 |
| 20 | **0.084** | 0.105 | 0.139 |
| 30 | **0.081** | 0.104 | 0.138 |
| 50 | **0.080** | 0.100 | 0.137 |
| 75 | 0.084 | **0.095** | 0.134 |
| 100 | 0.093 | **0.089** | 0.129 |

DataBin(20) 在 fnl=0~50 时最优，fnl=75-100 时与 FullDiscrete 接近。

### 重要注意事项
1. **N_fit 必须限制在拟合范围内**：如果用所有 bin（包括高 k），阶梯函数 P(k) 导致 j₀ 振荡
2. **kmax 不敏感**：DataBin 和 FullDiscrete 都完全不依赖 kmax（从 5 到 20 结果一样）
3. **ExpWindow 归一化因子**：`W(k)=[1-exp(-(k/kf)^x)]/[1-exp(-1)]`，必须包含分母

---

## 实验历程

### 失败的方向
| 方法 | 结果 | 原因 |
|---|---|---|
| ContWeight (连续权重替代 g_q) | 灾难 (1.36) | Voronoi cell 在首 shell 不合理 |
| Q_CORR (部分连续权重) | 同上 (~1.3) | 同上 |
| Mode-weighted fitting | 更差 (-0.085→-0.319) | 过度校正 fnl |
| Free sn0 | 灾难 (41.0) | sn0 微小变化导致模型崩溃 |
| PNG 缩放 | 不通用 | 最优 α 随 Lbox 变化 |
| DataBin(all bins) on quijote | 差 (0.60-0.71) | 98 bin 阶梯函数引入振荡 |

### 关键诊断发现
1. FullDiscrete **完全不依赖 kmax**
2. FullDiscrete 偏差一致 ≈ -0.086σ（3Gpc 和 1Gpc）
3. fnl=0 时 FullDiscrete 几乎完美 (+0.021σ)
4. 偏差主要来自 PNG 贡献

## 文件索引
- `mission9_lattice_analysis.py/.md` — 格点超额分析
- `mission9_exploration_3gpc.py` — 多方法对比
- `mission9_kmax_scan.py` + `mission9_1gpc_kmax.py` — kmax 扫描
- `mission9_mw_fitting.py/.md` — Mode-weighted fitting
- `mission9_new_ideas.py/.md` — DataBin + PNG 缩放 + FreeSN0
- `mission9_databin_optimize.py` — DataBin kmax 扫描
- `mission9_quijote_validation.py/.md` — Quijote N-body 验证
- `mission9_plot_databin.py` — 对比图
- `mission9_databin_results.png` — FastPM 结果图
- `mission9_quijote_validation.png` — Quijote 结果图
