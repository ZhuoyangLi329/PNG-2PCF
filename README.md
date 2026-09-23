# Task 4.3 lightcone RSD review package

这是给 GPT/独立审阅者使用的 Task 4.3 研究快照，重点处理 AbacusSummit/ EZmock 的 RSD lightcone、P(k)-2PCF 一致性、joint b1 偏离以及 GIC/RIC 建模。

## 先读什么

1. `docs/00_handoff_task43.md`：科学问题、数据合同、当前结论和审阅任务。
2. `docs/02_9.18_baseline.md`：9.18 的 Kaiser × Lorentzian-FoG 基准。
3. `docs/03_9.22_hybrid_and_fullRIC.md`：9.22 的 velocileptors GSM、PNG 响应和 full RIC。
4. `docs/04_results_and_audit_map.md`：权威 JSON/PDF/审计入口。
5. `docs/CODE_INDEX.md`、`docs/FIGURES.md`：代码和图的对应关系。

## 当前主问题

在相同的 25-phase lightcone 均值、single-realization covariance 和完整 P-xi cross covariance 下，P-only、xi-only 与 joint 的 `b1` 中心不一致，joint 的 `b1` 可能落在两个边缘结果之外。需要区分：

- RSD mean model 的形状错误；
- P 与 xi 使用了不同的动力学模型；
- GIC/RIC/window 算子不一致；
- covariance/cross block 的估计误差；
- finite mock correction、FKP 或 random/shuffling 合同问题。

## 范围

本包保留 9.18 基准、9.22 hybrid/full-RIC 代码、关键审计、结果摘要和 PDF 图，省略原始 catalog、巨大 HDF5 chain、完整 covariance cache 和临时日志。原始文件路径、排除项和来源说明见 `provenance/`。

本仓库不把任何结果标记为已发表的 science constraint；`pass` 只表示对应的数值、采样或 postflight gate 通过。
