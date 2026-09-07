# Mission 12: P(k) 拟合 vs 2PCF 拟合的 best-fit 对比

## 设置

- 数据: 3Gpc fastPM fnl100
- P(k) 拟合: BinAvgFit, 前 20 个 k-bin
- 2PCF 拟合: r ∈ [100.0, 350.0] Mpc/h, 21 个 r-bin
- 拟合量: r²ξ₀(r)
- 自由参数: fnl_loc, b1, sigmas (p=1.2 fix, sn0=0.0 fix)
- 2PCF model: 参数 → P_model(k) → fast_discrete_xi0(r) (k-rebinning 加速)

## Best-fit 参数

| 方法 | fnl_loc | b1 | sigmas | chi2/dof |
|---|---|---|---|---|
| StdFit(bin-center) | 77.069 | 2.789629 | 2.187603 | — |
| BinAvgFit P(k) | 71.991 | 2.805644 | 3.563991 | 0.118 |
| 2PCF fit | 67.295 | 2.971655 | 12.928060 | 0.366 |

## 耗时

- P(k) BinAvgFit: 1.9s
- 2PCF fit: 124.4s (92 calls, 1.353s/call)

## 2PCF 指标 (全 r 范围)

| 来源 | mean|Δ/σ| | mean(Δ/σ) |
|---|---|---|
| from P(k) fit | 0.4592 | +0.1629 |
| from 2PCF fit | 0.9405 | -0.8958 |
