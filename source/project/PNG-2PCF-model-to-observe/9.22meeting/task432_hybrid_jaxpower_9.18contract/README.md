> 已被包含 GIC 的正式修正版替代：/pscratch/sd/l/lzy/PNG-2PCF-model-to-observe/9.22meeting/task432_hybrid_jaxpower_gic_9.18contract
> 本目录 xi02/joint 模型遗漏 GIC，仅保留溯源；P-only 不受影响并继续复用。

# Hybrid PNG–GSM：9.18 jaxpower 口径重跑结果

结论：joint 的 b1 后验中位数和最大似然值都低于 P-only、2PCF-only，两种中心定义下均未落在两者之间。相比原模型的下移有所缓解，但没有消失。

## 本次结果

| 拟合 | b1 中位数 | b1 的 68% 分位区间 | b1 最大似然 | fNL 中位数 | fNL 的 68% 分位区间 |
|---|---:|---:|---:|---:|---:|
| P0+P2 | 2.412972 | [2.316992, 2.509841] | 2.417747 | -13.3393 | [-47.0451, 19.6339] |
| xi0+xi2 hybrid | 2.433824 | [2.329988, 2.536657] | 2.436808 | -8.0432 | [-35.3215, 17.7762] |
| joint | 2.392282 | [2.330057, 2.454363] | 2.390391 | -11.1241 | [-36.0616, 10.6493] |

9.18 原模型 jaxpower joint 的 b1 中位数为 2.353444，本次为 2.392282，变化 +0.038838。本次 joint 比 P-only 低 0.020690，比 xi-only 低 0.041542。

三组 68% 区间明显重叠。由于它们来自相同数据并保留 P–xi 交叉协方差，不能用独立误差相加来给这种中心偏移赋予严格的显著性；joint 中心是否位于单探针中心之间是本次重点报告的诊断，不是一般统计定理。

## 固定数据与模型

- 原 9.18 jaxpower 74×74 协方差直接读取，保留交叉块；数据是 Abacus 25-phase 平均，协方差为单 realization，不除以 25。
- P0=13 bin、P2=9 bin，kmax=0.08 h/Mpc，P2 kmin=0.015 h/Mpc；xi0/xi2 各 26 bin，50≤s<350 Mpc/h，剔除 80≤s<120。
- P 模型与原先验保持一致；xi 使用已有 9.22 hybrid。xi-only 为 fNL、b1；joint 为 fNL、b1、仅 P 侧 sigma_s、sn0。
- 按本次批准的方案保留现有 hybrid 的 bin 中心求值、无 formal GIC、连续 GSM 变换。这不是补齐 shell 平均/GIC 后的新物理版本。
- 不使用 EZmock covariance，因此无 Hartlap/Percival 修正。

## 验证

- 登录节点 login11 执行，三个单线程采样进程限制在同一组 CPU 6–13；没有提交 Slurm。
- 三组均为 64 walkers × 30000 steps、burn-in 5000，三项收敛检查全部通过。
- 全部参数最大 split-Rhat=1.002742，最小 postburn length/tau=446.1，最大半链漂移=0.0167 sigma。
- P-only 后验复现原图；其最大似然 b1=2.417747、fNL=−12.022012。
- hybrid 在预检查点和实际后验抽样点对照直接 GSM；最大联合协方差加权模型误差平方为 0.007504，即约 0.087 sigma。
- xi 与 joint 的最大似然再次使用直接 GSM、Nint=1200 优化；PDF 标注采用该最大似然中心及采样链的 16%/84% 分位数。
- PDF 复用原 9.18 绘图函数、颜色和坐标范围，共 1 页；已完成逐页视觉检查，并核对 PDF SHA256 与结果 JSON 一致。

## 复现入口与产物

远程项目根目录：`/pscratch/sd/l/lzy/PNG-2PCF-model-to-observe`。

- 推荐脚本：`codes/task432/task432_hybrid_jaxpower_0918_contract.py`。
- 直接模型复核：`codes/task432/task432_hybrid_0918_postflight.py`。
- 图和完整结果 JSON：`9.22meeting/task432_hybrid_jaxpower_9.18contract/`。
- 链、输入、校验及日志：`outputs/task43_outputs/rsd_validation/task432_model_repair/hybrid_jaxpower_0918_contract_v1/`。
- 优先读取主图同名 JSON、`input_audit.json`、`postflight_audit.json`、`run_manifest.json`；无需递归扫描其他实验目录。

执行顺序：`prepare` → 分别运行 `p02`、`xi02`、`joint` → 执行 postflight 脚本 → 主脚本 `plot`。脚本默认保护已完成的结果，重跑须显式指定新的实验输出位置。

本轮只新增两份任务脚本和一个独立结果目录；原 9.18 图、原链和已有 hybrid 代码未修改。执行版本保存在 provenance，后续只给当前脚本补充中文注释；去除 docstring 后 AST 相同。日志集中到 logs；完整 HDF5 链用于恢复和诊断，post-burn NPZ 用于绘图，均保留。未删除既有文件。
