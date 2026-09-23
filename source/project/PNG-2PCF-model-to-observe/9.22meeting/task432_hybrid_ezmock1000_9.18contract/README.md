> 已被包含 GIC 的正式修正版替代：/pscratch/sd/l/lzy/PNG-2PCF-model-to-observe/9.22meeting/task432_hybrid_ezmock1000_gic_9.18contract
> 本目录 xi02/joint 模型遗漏 GIC，仅保留溯源；P-only 不受影响并继续复用。

# EZmock-1000 hybrid：9.18 同口径重跑

joint b1 是否在 P 与 2PCF 之间：后验中位数=False，最大似然=False。

## 参数约束

以下区间包含 Hartlap 似然与按实际自由参数数目计算的 Percival 放大。图和 JSON 使用同一口径；原始采样链不被改写。

| 拟合 | b1 中位数 | b1 68% 区间 | b1 最大似然 | fNL 中位数 | fNL 68% 区间 |
|---|---:|---:|---:|---:|---:|
| p02 | 2.414844 | [2.316627, 2.513658] | 2.419496 | -13.0978 | [-45.8718, 19.4644] |
| xi02 | 2.421905 | [2.309747, 2.532757] | 2.426962 | -5.8275 | [-32.7383, 20.6620] |
| joint | 2.390788 | [2.323591, 2.457724] | 2.390379 | -7.5647 | [-32.4819, 15.3685] |

## 与上一轮 jaxpower hybrid 对照

| 拟合 | jaxpower b1 中位数 | EZmock-1000 b1 中位数 |
|---|---:|---:|
| p02 | 2.412972 | 2.414844 |
| xi02 | 2.433824 | 2.421905 |
| joint | 2.392282 | 2.390788 |

同一数据和模型下，仅协方差及其必要的有限 mock 修正不同；相关后验不能按彼此独立来计算偏移显著性。

## 固定约定与验证

- 按原生产校验器读取已完成的编号 0–999 共 1000 组 mock，形成 1000×74 stack 并重新计算 74×74 样本协方差。
- 数据、P 模型、hybrid 网格、先验从上一轮冻结输入逐数组复制并检查相等。
- P0/P2 为 13/9 bin，kmax=0.08 h/Mpc、P2 kmin=0.015；xi0/xi2 各 26 bin，50≤s<350 Mpc/h，剔除 80≤s<120。
- Abacus 25-phase 均值，C_single 不除以 25，保留 P–xi 交叉协方差。
- hybrid 按已批准实现运行：bin 中心求值，无 formal GIC；xi-only 为 2 参数，joint 为 4 参数，sigma_s/sn0 只作用于 P。
- 登录节点执行，三个单线程采样进程限制在同一组最多 8 CPU；未提交 Slurm。
- 三组均通过收敛检查，最大 Rhat=1.002374，最小 postburn length/tau=435.18。
- 在本次后验抽样点重新检查插值，并用直接 GSM、Nint=1200 复核 xi/joint 最大似然。

| 拟合 | Ndata | Nparam | Hartlap | Percival 宽度因子 |
|---|---:|---:|---:|---:|
| joint | 74 | 4 | 0.92492492 | 1.03442775 |
| p02 | 22 | 4 | 0.97697698 | 1.00660976 |
| xi02 | 52 | 2 | 0.94694695 | 1.02448808 |

Percival 通过围绕各参数 q50 放大展示样本来绘制一致的轮廓；这是有限样本误差修正，不是额外一次后验采样。旧 9.18 PDF 中未经 Percival 放大的轮廓没有在本次延续。

## 复现与文件卫生

推荐入口：`codes/task432/task432_hybrid_ezmock_0918_contract.py`，复用 `task432_hybrid_jaxpower_0918_contract.py` 与 `task432_hybrid_0918_postflight.py`。
执行顺序：prepare → 分别 p02、xi02、joint → postflight → plot；已有完成产物受覆盖保护。
图、同名 JSON 与本说明位于 `/pscratch/sd/l/lzy/PNG-2PCF-model-to-observe/9.22meeting/task432_hybrid_ezmock1000_9.18contract`。
链、输入及审计位于 `/pscratch/sd/l/lzy/PNG-2PCF-model-to-observe/outputs/task43_outputs/rsd_validation/task432_model_repair/hybrid_ezmock1000_0918_contract_v1`。
优先读取图的 JSON、input_audit.json、postflight_audit.json、run_manifest.json；无需扫描其他目录。
604 试跑已按用户更正停止，CANCELLED.json 标记其不是科学结果，部分链保留；旧 604 驱动脚本已归档并登记移动清单。本轮当前推荐入口为通用 EZmock 驱动脚本；公共采样脚本仅将参考协方差标签和随机种子参数化，jaxpower 默认行为保持一致。执行代码已保存于 provenance。
日志集中在 logs；完整 HDF5 链用于恢复及诊断，post-burn NPZ 用于绘图。未删除或覆盖已有科学结果。
