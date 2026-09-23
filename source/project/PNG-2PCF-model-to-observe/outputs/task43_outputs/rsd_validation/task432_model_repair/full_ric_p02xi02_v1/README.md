# Task432 full RIC — 推荐阅读入口

本轮执行 eBOSS 简化的四分量全项 IC：P0、P2、xi0、xi2；原74维数据、EZmock1000 C_single、完整交叉块、9.18 cuts全部冻结，喵～

## 当前结果与入口

正式 MCMC：远端 `outputs/task43_outputs/rsd_validation/task432_model_repair/full_ric_p02xi02_v1/formal_mean4/`，喵～加密后的三组各64 walkers、30000步、burn-in5000，全部收敛与后验数值门通过，喵～首轮mean2因尾部积分精度触发加密，保留作追溯，喵～

最终b1中位数：P-only=2.409815、xi-only=2.418489、joint=2.376509；joint仍在两者下方，喵～两份PDF共5页已逐页视觉验收，喵～

绘图与汇总入口：`task432_full_ric_deliver.py`，拟合入口：`task432_full_ric_fit.py`，后验数值检查：`task432_full_ric_postflight.py`，喵～

推荐只先读 `formal_mean4/results.json`、`formal_mean4/postflight.json`、`smoke/geometry_validation_mean4.json`、`smoke/paired_validation.json` 与最终图目录的 `RESULTS.md`，喵～不需要递归浏览整个 outputs，喵～

## 模型记账

- `backend.py` 复用作者固定 commit 的 pycute/pywindow RR、cross、auto，projector为 Qr = I-Pi_r，喵～常数模式由同一投影消去，不能再叠加旧scalar GIC，喵～
- `response.py` 用完整内层距离和ell0/2/4；P按endpoint输出，xi按midpoint、signed40mu先除RR再投影，喵～正式使用真实25phase mean RR，喵～
- 三套互相独立的Sobol目录用于outer/A/B，P核为两个独立scramble平均，xi尾部检查后增加至四个scramble平均；两套独立双核均值之间仍要求误差二次型<0.01，喵～径向宽度2 Mpc/h，1 Mpc/h加密通过；代表phase检查通过，喵～
- data Poisson噪声径向投影，conditional bootstrap的random噪声仅全局投影：Qr D Qr^T / Nd + Qg D Qg^T / Nr，喵～人工真实bootstrap与实际LS随机分母检查通过，喵～
- 原sn0的constant-P接触项使用nbar加权核；它在xi中不能全先验忽略，因此xi-only加入这个既有参数，喵～P和joint仍为4参数，xi为fNL/b1/sn0三个参数，原sn0先验[-1,1]不变，喵～RIC无新增自由振幅，喵～

## 验证的含义和局限

代数闭合、作者计数与dense sum、白噪声/Poisson、原cucount与Corrfunc CPU bridge、独立积分/径向分辨率/phase/LOS均有单独审计，喵～直接hybrid与emulator前置检查最大误差二次型约3.5e-8，喵～

一个真实halo和两个实际covariance EZmock完成paired shuffled/reference测量，四个分量均正向重合，P量级相近；xi用两个random blocks，噪声较大，只是方向/量级检查，喵～不能把C_single当paired covariance，不能声称3个realizations证明精密均值闭合，喵～

P的局部LOS输入用midpoint近似，经精确two-endpoint解析核比对误差二次型约1e-5，喵～连续Bessel IC与离散FFT仍是eBOSS式近似，喵～冻结EZ共同random covariance与halo自身shuffled random的差异仍是保留问题，喵～

## 文件归属

`source/`为本任务脚本镜像，远端统一位于`codes/task432/`，喵～`geometry/`中的候选核、`pilot/`中的优化是最终积分精度审计的可复现证据，应保留但不要当最终posterior，喵～编译文件在`pilot/mean4xi_mean2P_prodrr_stochastic/engine.npz`，正式链在manifest中锁定其hash，后续审计完成后才被用于正式结果，喵～

远端物理目录和核缓存属于重现依赖，未删除或批量移动，喵～传输压缩包仅是临时副本，可以按归档manifest回收，喵～科学图只输出PDF，不留PNG，喵～

## 用户工作约束

所有CPU作业在login11并绑定6-13，共用至多8 CPU；库单线程，明确的pair核最多8线程，喵～缺包自主查找/隔离安装，喵～所有SSH/SCP通过`remote.py`，网络故障后至少900秒再连接，喵～

## 运行环境与作者版本

推断/数值环境：`/global/homes/l/lzy/anaconda3/envs/desilike/bin/python`，喵～生产估计量CPU桥使用`source /global/common/software/desi/users/adematti/cosmodesi_environment.sh main`提供的Corrfunc/JAX环境，喵～

已实际记录推断环境版本：Python 3.11.14、NumPy 1.26.4、SciPy 1.16.3、emcee 3.1.6，喵～

作者源码固定为pycute `e29ea2ee95ba9127d00187e9e341a63a9431c437`、pywindow `9857ebef68153660454fee34c4fda60e90f7289e`，保存于远端OUT/reference；cute.so在项目源码目录中用make -j2 CC=gcc编译，喵～旧SciPy API只在Python进程内补兼容aliases，未改共享环境，喵～

理论参考：[de Mattia & Ruhlmann-Kleider 2019](https://arxiv.org/html/1904.08851v3)、[eBOSS ELG CLPT-GS 2020](https://arxiv.org/html/2007.09009#S5.SS2)，喵～

正式复现时使用`task432_full_ric_fit.py VARIANT --compiled .../pilot/mean4xi_mean2P_prodrr_stochastic/engine.npz --run-name formal_mean4 --geometry-audit .../smoke/geometry_validation_mean4.json`，喵～它会核对输入/源码manifest且不会重置旧链；新实验须指定明确的新run-name，喵～
