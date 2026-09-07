# Notebook source mirror

Original: `source/background/agent/mission4_log/mission4_log.ipynb`

Outputs remain in original notebook.

## Cell 1 (markdown)

# mission4_log：任务4调研实验报告

## 0. 任务目标
- **4.1**：比较 1Gpc 与 3Gpc（均为 fnl=100）测量得到的 2PCF 是否存在盒长依赖。
- **4.2**：在理论积分采用 `kmin = 2π/L` 的数值 cut 时，比较“模型 2PCF vs 测量均值”在 1Gpc 和 3Gpc 下的偏差强弱。

本报告对应脚本：`task4_boxlength_kcut_study.py`。


## Cell 2 (markdown)

## 1. 数据与方法

### 1.1 输入数据
- 1Gpc: `pk/pk_rsd_N*.dat`, `pcf/pcf_rsd_N*.dat`（N=1..50）
- 3Gpc: `pk/pk_rsd_3gpc_fnl100_N*.dat`, `pcf/pcf_rsd_3gpc_fnl100_N*.dat`（N=2..99）

### 1.2 任务4.1测量比较
- 先对 **all realizations** 做均值/标准差比较。
- 再对共同 realization（N=2..50）做 matched 比较，排除样本数量差异带来的影响。

### 1.3 任务4.2建模链路
1. 分别拟合 1Gpc/3Gpc 的低 k 测量 `P0(k)`（`k<=0.0635`，Minuit）。
2. 取各自 best-fit 参数构建理论 `P0(k)`。
3. 用自写 FFTLog 计算
   $$\xi_0(s)=\int \frac{k^2}{2\pi^2}P_0(k)j_0(ks)\,dk$$
   并采用 `kmin=2π/L`, `kmax=20`。
4. 与测量均值比较，计算残差指标（尤其大尺度 `s>=200`）。


## Cell 3 (markdown)

## 2. 关键数值结果

### 2.1 P0 best-fit 参数
- **1Gpc**: `fnl_loc=122.288371`, `b1=1.479907`, `sigmas=3.348575`
- **3Gpc**: `fnl_loc=75.449225`, `b1=2.782026`, `sigmas=1.045459`

### 2.2 任务4.2核心指标（大尺度 `s>=200`）
- 指标定义：`RMS[(Δ r^2 ξ)/σ_mock]`
- **1Gpc**: `2.3243`
- **3Gpc**: `0.5784`

结论：3Gpc 明显小于 1Gpc，说明在采用 `kmin=2π/L` cut 后，3Gpc 情况下模型与测量均值的差异**变小**。


## Cell 4 (markdown)

## 3. 图像结果

### 3.1 任务4.1：测量2PCF盒长比较
- 全样本：

![](task41_measurement_compare_all.png)

- 匹配样本（N=2..50）：

![](task41_measurement_compare_matched.png)

### 3.2 任务4.2：建模与测量比较
- P0 拟合：

![](task42_pk_fit_compare.png)

- 模型 2PCF（kmin=2π/L）与测量均值对比：

![](task42_xi_model_vs_measurement.png)


## Cell 5 (markdown)

## 4. 输出文件索引
- `task41_measurement_compare_all.csv`
- `task41_measurement_compare_all.png`
- `task41_measurement_compare_matched.csv`
- `task41_measurement_compare_matched.png`
- `task42_bestfit_params.csv`
- `task42_pk_fit_compare.png`
- `task42_xi_model_metrics.csv`
- `task42_xi_model_vs_data_perbin.csv`
- `task42_xi_model_vs_measurement.png`
- `task4_summary.txt`


## Cell 6 (code)

```python
import csv
from pprint import pprint

print('task42_bestfit_params.csv')
with open('task42_bestfit_params.csv', 'r', encoding='utf-8') as f:
    for row in csv.DictReader(f):
        pprint(row)

print('\n' + '='*60)
print('task42_xi_model_metrics.csv')
with open('task42_xi_model_metrics.csv', 'r', encoding='utf-8') as f:
    for row in csv.DictReader(f):
        pprint(row)

```
