# Mission 8 收尾总结：exp 窗口口径下的 3Gpc / 1Gpc fastPM

本轮不再用 `tanh_log`，而是严格按任务书最新指定的窗口：

$$
W(k)=\frac{1-\exp\left[-(k/k_f)^x\right]}{1-\exp(-1)}\quad (k<k_f), \qquad W=1\ (k\ge k_f)
$$

并取 `x(L)=4*(L/1000)`，所以 `1Gpc -> x=4`，`3Gpc -> x=12`。

## 3Gpc

- 盒长 `L=3000 Mpc/h`，`k_f=2pi/L=0.00209440` h/Mpc
- exp 窗口参数：`x=12`
- fnl100 best-fit：`fnl_loc=75.697409, b1=2.781936, sigmas=0.015670`
- 大尺度指标 `mean|Δ/σ|` / `chi2_ndof`：
  Baseline = 0.5601 / 0.3808
  ExpWindow = 0.2076 / 0.0495
  ShellHybrid_best = 0.1939 / 0.0702
  ShellHybridPNG_best = 0.1276 / 0.0349
- 最优 shell hybrid：`Nshell=1`，最后一个离散 shell `q=1`，`k=0.00209440` h/Mpc
- 最优 shell PNG hybrid：`Nshell=2`，最后一个离散 shell `q=2`，`k=0.00296192` h/Mpc
- fnl0 sanity：Baseline=0.2641，ExpWindow=0.3535，ShellHybrid_best=0.2548，ShellHybridPNG_best=0.2641

## 1Gpc

- 盒长 `L=1000 Mpc/h`，`k_f=2pi/L=0.00628319` h/Mpc
- exp 窗口参数：`x=4`
- fnl100 best-fit：`fnl_loc=104.653065, b1=2.744006, sigmas=0.000048`
- 大尺度指标 `mean|Δ/σ|` / `chi2_ndof`：
  Baseline = 2.1129 / 4.5287
  ExpWindow = 0.0818 / 0.0130
  ShellHybrid_best = 0.7799 / 0.8715
  ShellHybridPNG_best = 0.3005 / 0.1318
- 最优 shell hybrid：`Nshell=1`，最后一个离散 shell `q=1`，`k=0.00628319` h/Mpc
- 最优 shell PNG hybrid：`Nshell=5`，最后一个离散 shell `q=5`，`k=0.01404963` h/Mpc
- fnl0 sanity：Baseline=0.9602，ExpWindow=0.1256，ShellHybrid_best=0.4992，ShellHybridPNG_best=0.9602

## 总结判断

1. 你指定的 normalized exp 窗口口径在两种盒长下都已经按同一套流程跑完。
2. 3Gpc 下，这个 exp 窗口仍明显优于 baseline，并且与最优 shell-hybrid 做到了同一量级的改进。
3. 我补了一个更合理的解析版本：只对 PNG 增量项做离散 shell 化，而不是对总 P(k) 直接 shell 化。
4. 从 Mission 8 的角度，窗口法最合理的解析解释仍然是：它不是直接代表某个低-k 物理，而是在数值上模拟有限盒里极少数离散 shell 对低-k 端的真实贡献。
