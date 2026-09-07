# Mission 7 Audit: IR Window And Modeling Config Mismatch (2026-03-13)

## TL;DR
Mission7 的 `model_validation` 结果和 mission5 “建模出来的 2PCF” 不一致，不是因为 pk/pcf 测量文件不同，而是因为 **IR window 的函数族与实现细节不同**，外加 **FFTLog 积分区间 / 固定参数 p / PK 拟合 k-bin 选择** 在 mission5 的不同脚本和当前 `pk-pcf-model/model.ipynb` 之间并不完全一致。

另外，1Gpc 下 **第一个 pk k-bin (kcen=0.0035)** 在所有 realization 上 `P0=0`（k < k_fund 时没有模式），会导致协方差矩阵奇异；mission7 脚本里为保证 desilike 能拟合，自动剔除了该 bin。

## 哪些代码在用 IR window
### model.ipynb / mission7（exp_power, unnormalized）
文件：
- `/pscratch/sd/l/lzy/1gpc-UNIT-fnl100-fastpm/pk-pcf-model/model.ipynb`
- `/pscratch/sd/l/lzy/1gpc-UNIT-fnl100-fastpm/agent/mission7_masscut/scripts/masscut_model_validate.py`

IR window 定义（`build_ir_param_window`）：
- `k < k_f:  W(k) = 1 - exp(-(k/k_f)^x)`
- `k >= k_f: W(k) = 1`

注意：这里 **没有** `/(1-exp(-1))` 归一化，因此 `k -> k_f^-` 时 `W(k_f^-) = 1-exp(-1) ~ 0.632`，而 `k>=k_f` 被强制设为 1，会在 `k_f` 处有一个“单点跳变”（通常影响很小，但与 mission5 的 exp_power 定义确实不同）。

### mission5（多类型窗口族，exp_power normalized + tanh_log 等）
文件：
- `/pscratch/sd/l/lzy/1gpc-UNIT-fnl100-fastpm/agent/mission5_log/task5_ir_window_solution_3gpc_multitype.py`

其中 `build_ir_window(..., wtype="exp_power", param=a)` 定义为：
- `W = [1-exp(-(k/kf)^a)] / [1-exp(-1)]` for `k<kf`, `W=1` otherwise

并且 mission5 的建模脚本（例如 v7/v13）经常扫描并最终采用 `tanh_log(beta)`：
- `W = 0.5*(1+tanh(beta*ln(k/kf)))`（`k<kf`），`k>=kf` 仍为 1

这意味着如果你拿 mission5 的 “best beta” 曲线去对比 mission7 的 exp_power(x=4) 曲线，**一定不同**。

## mission5 与 mission7 默认数值口径也不同（会影响 2PCF）
mission7 `masscut_model_validate.py` 默认（对齐当前 model.ipynb）：
- FFTLog: `kint_min=1e-4`, `kint_max=20`, `taper_frac=0.06`, `fftlog_n=4096`, `padding=4.0`
- 理论固定参数：`fixed_p=1.2`, `sn0=0`, `sigmas` 自由
- PK 拟合数据：默认取前 `n_datapoints=20` 个 k-bin（但会自动剔除 std=0 的 bin）

而 mission5 的一些 “model.ipynb-style” 检查脚本在当时的 summary 中记录过不同口径（示例）：
- `k_int=[0.003, 1.0]`，`PK_FIT_KMAX=0.08`，且固定 `p=1.1`（见 mission5 `summary.txt` 的第九节）

所以即便都叫 “model.ipynb-style”，也可能因为 **你当时运行 notebook/脚本的参数版本不同**，导致结果不一致。

## 1Gpc 的 pk 第一个 bin 导致协方差奇异（mission7 的必要修复）
1Gpc 盒子 `k_f = 2*pi/L ~ 0.006283`。POWSPEC 输出的第一个 kcen=0.0035 小于 k_f，因此没有模式，`P0=0` 会在所有 realization 上恒为 0。

如果把它当作数据点用于 mock covariance，协方差会出现全零列/行，precision=inv(cov) 会报 `Singular matrix`。

mission7 通过 `filter_singular_pk_bins()` 自动剔除 std==0 的列来保证拟合可以跑通。

## 当前建议
请先明确你要“对齐”的目标是哪一个：
1) **对齐当前 `pk-pcf-model/model.ipynb`**：那 mission7 已经用同一套 exp_power(x)（除非你 notebook 当时用的是不同的 `kint_min/kint_max/fixed_p/n_datapoints`）。
2) **对齐 mission5 v7/v13 的“最佳窗口”**：那需要在 mission7 的 model_validation 里使用 mission5 的 `build_ir_window`（比如 `tanh_log(beta=best)` 或 mission5 的 normalized exp_power），并同时对齐 `k_int`、`PK_FIT_KMAX` 与固定参数 p。

