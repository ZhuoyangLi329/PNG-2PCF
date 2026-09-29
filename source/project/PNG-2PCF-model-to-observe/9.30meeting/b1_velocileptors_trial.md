# 9.30 meeting：velocileptors xi 替换的 b1 试算

本文件记录诊断性 ML 试算，不替换正式 full-RIC 后验。

## EZmock1000 C_single

| 拟合 | b1 |
|---|---:|
| P-only full-RIC | 2.41346 |
| velocileptors xi-only 敏感性模型 | 2.42646 |
| joint（velocileptors xi 替换） | 2.42239 |

结论：`2.41346 < 2.42239 < 2.42646`，joint b1 位于 P-only 与 xi-only 之间。

## jaxpower C_single 对照

| 拟合 | b1 |
|---|---:|
| P-only | 2.41137 |
| velocileptors xi-only 敏感性模型 | 2.43626 |
| joint（velocileptors xi 替换） | 2.42382 |

结论：`2.41137 < 2.42382 < 2.43626`，同样位于两侧之间。

## 限制

该试算使用已有 velocileptors xi emulator，并补接旧 Engine 的 sn0 响应；没有重新计算 velocileptors 自己的完整 deterministic full-RIC，也没有跑正式长链。因此它只能说明 xi 动力学替换可以消除 joint b1 的 outside-both 方向，不能作为正式 science 结果。
