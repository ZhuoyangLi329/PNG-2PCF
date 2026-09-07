# Mission 11: 全离散求和加速 — 最终结果

## 问题

全离散求和 (FullDiscrete) 是目前 2PCF 建模的核心步骤：

```
xi_0(r) = (1/V) sum_q g_q * P_model(k_q) * j0(k_q * r)
```

3Gpc 盒子 kmax=15 时共有约 **4270 万**个非零壳层，暴力逐壳层求和（Baseline）耗时约 **66s**（单核 Python/numpy）。

曾尝试的 Hybrid 方案（低 k 离散 + 高 k FFTLog）不可行：大尺度 xi(r) 的贡献主要来自低 k 模式，高 k 的 FFTLog 对大尺度 r 无有效贡献。

## 解决方案：k-rebinning（壳层聚合）

### 核心思想

42M 壳层的 k 值（k_q = kf√q）密集分布在 [kf, kmax] 上。
把 k 值相近的壳层合并到等间距 k-bin（bin 宽 dk = 0.1×kf）中：

- 42M 壳层聚合为 **~71,584 个 bin**（压缩 ~600 倍）
- sin 计算从 42M×28 ≈ 1.2G 次降到 71k×28 ≈ 2M 次

### 两种具体方案

**1. ExactRebin（精确 k-rebinning）**

对 42M 壳层先精确插值 P_model(k_q)，再聚合到 bin：
```
W_bin = sum_{q in bin} g_q * P(k_q)      # 精确值
k_eff = sum_{q in bin} g_q*P(k_q)*k_q / W_bin
```
每次需要对 42M 点做 interp + bincount，耗时 ~3.8s。加速 **18x**。

**2. CachedRebin（缓存 k-rebinning，推荐）**

将不依赖 P_model 的量预计算并缓存：
```
G_bin = sum_{q in bin} g_q       # 只依赖 L 和 kmax
k_eff = sum_{q in bin} g_q*k_q / G_bin
```
每次只需对 71k 个 k_eff 做 interp：
```
W_bin ≈ G_bin * P_model(k_eff)   # P 在 bin 内近似恒定
```
预计算 3s（可缓存到文件），每次求值 **0.08s**。加速 **830x**。

### 近似精度

CachedRebin 的近似来源：P_model(k) 在 bin 内不是严格恒定。
但 dk = 0.1×kf ≈ 0.0002 h/Mpc，这个尺度上 P(k) 的变化极小。

## 验证结果

测试对象：3Gpc fastPM，BinAvgFit best-fit P_model

### 3Gpc fnl100

| 方法 | 耗时 | 加速比 | vs Data mean\|Δ/σ\| | vs Data Δ/σ | vs Baseline mean\|Δξ/ξ\| |
|---|---|---|---|---|---|
| **Baseline** | 66.4s | 1x | 0.4595 | +0.1622 | — |
| **ExactRebin** | 3.75s | 18x | 0.4592 | +0.1628 | 1.56e-03 |
| **CachedRebin** | 0.08s | 831x | 0.4592 | +0.1629 | 1.56e-03 |

### 3Gpc fnl0

| 方法 | 耗时 | 加速比 | vs Data mean\|Δ/σ\| | vs Data Δ/σ | vs Baseline mean\|Δξ/ξ\| |
|---|---|---|---|---|---|
| **Baseline** | 66.2s | 1x | 0.5790 | +0.1036 | — |
| **ExactRebin** | 3.76s | 18x | 0.5786 | +0.1043 | 3.53e-04 |
| **CachedRebin** | 0.08s | 820x | 0.5787 | +0.1042 | 3.48e-04 |

**关键结论：三种方案的 vs data 指标完全一致（差异在第四位小数以下）。**

fnl0 时精度更好（mean|Δξ/ξ| ~ 3.5e-04），因为 P(k) 在低 k 没有 1/k² 的剧烈变化。

## Pipeline 耗时分解

| 阶段 | 耗时 | 可否缓存 | 说明 |
|---|---|---|---|
| g_q 计算 (FFT) | ~8s | 可缓存（只依赖 L, kmax） | 整数分拆计数 |
| G_bin, k_eff 预计算 | ~3s | 可缓存（只依赖 L, kmax, dk） | bin 聚合 |
| P_model 求值 | ~0.08s | 不可缓存 | interp + sinc + matmul |

**首次调用**（含 g_q 和预计算）：~11s
**后续调用**（只更换 P_model）：**~0.08s**

## 实现代码

推荐的生产代码模式（见 `mission11_final_validation.py`）：

```python
def precompute_cache(gq, kf, kmax, dk_factor=0.1):
    """预计算 G_nz 和 k_eff — 只依赖盒长 L 和 kmax，可序列化缓存。"""
    qnz = np.nonzero(gq[1:])[0] + 1
    kv = kf * np.sqrt(qnz.astype(float))
    g = gq[qnz].astype(float)
    dk = dk_factor * kf
    n_bins = int(np.ceil(kmax / dk)) + 1
    bin_idx = np.clip((kv / dk).astype(int), 0, n_bins - 1)
    G_bin = np.bincount(bin_idx, weights=g, minlength=n_bins)
    Gk_bin = np.bincount(bin_idx, weights=g * kv, minlength=n_bins)
    nz = G_bin > 0
    return G_bin[nz], Gk_bin[nz] / G_bin[nz]

def xi_cached_rebin(s, G_nz, k_eff, V, kd, pd):
    """每次更换 P_model 时的快速求值 — 0.08s。"""
    W = G_nz * np.interp(k_eff, kd, pd)
    arg = np.outer(k_eff, s)
    J = np.ones_like(arg)
    m = arg != 0
    J[m] = np.sin(arg[m]) / arg[m]
    return (W @ J) / V
```

## 与其他方案的对比

| 方案 | 特点 | 推荐？ |
|---|---|---|
| k-rebinning (dk=0.1kf) | 830x 加速, 0.15% 精度 | **推荐** |
| k-rebinning + O1 修正 | 同样速度, 精度无明显改善 | 不推荐 |
| k-rebinning + O2 修正 | 同上 | 不推荐 |
| Hybrid (低k离散+高k FFTLog) | 无法覆盖大尺度 r | 不可行 |
| Numba JIT | desilike 环境不可用 | 备选 |

一阶/二阶修正效果不明显的原因：计算瓶颈不在 sin（71k 次 sin 只需 0.1s），而在预处理阶段（42M 数组的 sqrt/interp/bincount），这对所有 dk 都是 ~3s。加权中心 k_eff 本身已经吸收了零阶近似的主要误差。

## 文件清单

- `mission11_acceleration_benchmark.py`: 多方案对比测试脚本
- `mission11_acceleration_benchmark.png`: 多方案对比图
- `mission11_final_validation.py`: 最终验证脚本（3Gpc fnl100 + fnl0）
- `mission11_final_validation.png`: 最终验证图
- `mission11_acceleration_results.md`: 本文件
