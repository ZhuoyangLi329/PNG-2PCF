#!/usr/bin/env python3
"""
Mission 7: 不同 halo mass_min 的 2PCF 建模验证（FastPM 1Gpc fnl100）。

========================
代码大纲（执行逻辑）
========================
1) 读取一组 masscut 的 FastPM 测量结果：
   - pk: 读取 pk_rsd_N*.dat，得到 P0(k) 的 realization 集合 + 均值
   - pcf: 读取 pcf_rsd_N*.dat，得到 xi0(r) 的 realization 集合 + 均值/误差
2) 用 desilike 构建 PNG tracer 的 P0(k) 理论模型，并用 MinuitProfiler 做 best-fit：
   - data: fastpm P0(k) 的均值
   - covariance: fastpm 各 realization 的 P0(k)（交给 desilike 做 Hartlap/Percival 修正）
3) 在高分辨率 k 网格上评估 best-fit P0(k)，再用 FFTLog (scipy.fft.fht) 计算 xi0(r)：
   - baseline: kmin = k_f = 2pi/L，只保留 FFTLog taper
   - window  : kmin = kmin_global，且对 P0(k) 额外乘 mission5 的 IR 参数化窗口 W(k)
4) 把模型插值到测量 r 网格，并计算误差指标（|delta/sigma| 的均值/最大值等），输出图和 json。

========================
输入/输出
========================
输入：
  - --pk-dir: pk 文件目录（包含 pk_rsd_N*.dat）
  - --pcf-dir: pcf 文件目录（包含 pcf_rsd_N*.dat）

输出（写到 --out-dir）：
  - bestfit_params.json
  - metrics.json
  - r2xi_compare.png

注意：
  - 本脚本依赖 desilike/cosmoprimo/pypower/scipy/matplotlib。
  - 请在能 import desilike 的环境里运行（例如 conda activate desilike）。
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np


def parse_realization_id(filepath: str) -> int:
    """
    从文件名中解析 realization 编号。

    约定：文件名中包含 'N<number>'，例如 pk_rsd_N18.dat / pcf_rsd_N18.dat。
    """
    name = os.path.basename(filepath)
    match = re.search(r"N([0-9]+)", name)
    return int(match.group(1)) if match else -1


def read_single_pk_file(
    filepath: str,
    n_datapoints: int,
    k_cen_col: int,
    k_min_col: int,
    k_max_col: int,
    p0_col: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    读取单个功率谱文件并抽取所需列。

    参数
    - filepath: 输入文件路径
    - n_datapoints: 截取前多少个 k-bin
    - k_*_col / p0_col: 列索引（与 POWSPEC 输出一致）

    返回
    - kcen, kmin, kmax, p0: shape (n_datapoints,)
    """
    arr = np.loadtxt(filepath, comments="#")
    sl = slice(0, n_datapoints)
    kcen = arr[sl, k_cen_col]
    kmin = arr[sl, k_min_col]
    kmax = arr[sl, k_max_col]
    p0 = arr[sl, p0_col]
    return kcen, kmin, kmax, p0


@dataclass(frozen=True)
class PkEnsemble:
    kcen: np.ndarray
    kmin: np.ndarray
    kmax: np.ndarray
    p0_mocks: np.ndarray  # (nmock, n_k)
    p0_mean: np.ndarray  # (n_k,)
    used_files: List[str]


def load_pk_ensemble(
    data_dir: str,
    file_glob: str,
    realization_min: int,
    realization_max: int,
    n_datapoints: int,
    k_cen_col: int,
    k_min_col: int,
    k_max_col: int,
    p0_col: int,
) -> PkEnsemble:
    """
    批量读取功率谱 realization，并构建 data/mocks 所需基础数组。
    """
    pattern = os.path.join(data_dir, file_glob)
    files = sorted(glob.glob(pattern), key=parse_realization_id)

    selected: List[str] = []
    for fp in files:
        rid = parse_realization_id(fp)
        if realization_min <= rid <= realization_max:
            selected.append(fp)

    if not selected:
        raise FileNotFoundError(
            f"未找到可用输入文件：{pattern}，范围=[{realization_min},{realization_max}]"
        )

    # 直接使用第一个文件的 k 网格作为参考
    ref_kcen, ref_kmin, ref_kmax, _ = read_single_pk_file(
        filepath=selected[0],
        n_datapoints=n_datapoints,
        k_cen_col=k_cen_col,
        k_min_col=k_min_col,
        k_max_col=k_max_col,
        p0_col=p0_col,
    )

    p0_list: List[np.ndarray] = []
    for fp in selected:
        _, _, _, p0 = read_single_pk_file(
            filepath=fp,
            n_datapoints=n_datapoints,
            k_cen_col=k_cen_col,
            k_min_col=k_min_col,
            k_max_col=k_max_col,
            p0_col=p0_col,
        )
        p0_list.append(p0)

    p0_mocks = np.asarray(p0_list, dtype=np.float64)
    p0_mean = np.mean(p0_mocks, axis=0)

    print("[PK] 样本读取完成：")
    print(f"  mock 数量 = {p0_mocks.shape[0]}")
    print(f"  k-bin 数量 = {p0_mocks.shape[1]}")
    print(f"  k 范围 = [{ref_kcen.min():.5f}, {ref_kcen.max():.5f}]")

    return PkEnsemble(
        kcen=ref_kcen,
        kmin=ref_kmin,
        kmax=ref_kmax,
        p0_mocks=p0_mocks,
        p0_mean=p0_mean,
        used_files=selected,
    )


def filter_singular_pk_bins(pk: PkEnsemble, *, rtol: float = 0.0, atol: float = 0.0) -> PkEnsemble:
    """
    过滤会导致协方差奇异（不可逆）的 k-bin。

    背景：
    - 1Gpc 盒子时，k_f = 2pi/L ~ 0.00628 h/Mpc。
      如果 POWSPEC 的 kmin 设得更低（例如 0.002），第一个 k-bin 可能没有任何模式，
      P0 会在所有 realization 上恒等于 0，从而协方差矩阵出现“全零列/行”，导致奇异。
    - desilike 在构建 precision=inv(cov) 时会直接报错 Singular matrix。

    口径：
    - 移除 std==0 的列（或近似为 0 的列）。
    - 同时移除任何非有限值列（NaN/inf）。

    参数
    - pk: 原始 PkEnsemble
    - rtol/atol: 用于判断“近似零方差”的阈值；默认严格等于 0。

    返回
    - 新的 PkEnsemble（仅保留可用 k-bin）
    """
    X = np.asarray(pk.p0_mocks, dtype=np.float64)
    # ddof=1 与后续 covariance 估计口径一致
    std = np.std(X, axis=0, ddof=1)
    finite = np.all(np.isfinite(X), axis=0) & np.isfinite(std)
    # std==0 -> 无信息、导致 cov 奇异
    nonzero = ~np.isclose(std, 0.0, rtol=rtol, atol=atol)
    keep = finite & nonzero

    if not np.all(keep):
        drop_idx = np.where(~keep)[0].tolist()
        print("[PK] 检测到可能导致协方差奇异的 k-bin，将自动剔除：")
        print(f"  drop indices = {drop_idx}")
        print(f"  kept bins = {int(np.sum(keep))} / {keep.size}")

    if int(np.sum(keep)) < 3:
        raise RuntimeError(
            f"[PK] 过滤后剩余 k-bin 过少（kept={int(np.sum(keep))}），无法进行拟合。"
        )

    return PkEnsemble(
        kcen=np.asarray(pk.kcen)[keep],
        kmin=np.asarray(pk.kmin)[keep],
        kmax=np.asarray(pk.kmax)[keep],
        p0_mocks=np.asarray(pk.p0_mocks)[:, keep],
        p0_mean=np.asarray(pk.p0_mean)[keep],
        used_files=list(pk.used_files),
    )


def build_pypower_data_and_mocks(
    kcen: np.ndarray,
    kmin: np.ndarray,
    kmax: np.ndarray,
    p0_mean_data: np.ndarray,
    p0_cov_mocks: np.ndarray,
):
    """
    把 data 与 covariance mock 封装成 desilike 可识别的 pypower 统计对象。

    返回：
      - data_ps: PowerSpectrumStatistics（只含 P0 的均值）
      - mock_ps_list: List[PowerSpectrumStatistics]（每个 mock 一条 P0，用于估计协方差）
    """
    from pypower import PowerSpectrumStatistics

    edges = np.concatenate([kmin, [kmax[-1]]])
    ells = [0]
    nmodes = 4.0 * np.pi / 3.0 * (edges[1:] ** 3 - edges[:-1] ** 3)

    poles_mean = np.asarray([p0_mean_data], dtype="f8")
    data_ps = PowerSpectrumStatistics(
        edges=edges,
        modes=kcen,
        power_nonorm=poles_mean,
        nmodes=nmodes,
        ells=ells,
        shotnoise_nonorm=0.0,
        statistic="multipole",
    )

    mock_ps_list = []
    for i in range(p0_cov_mocks.shape[0]):
        tmp = data_ps.deepcopy()
        tmp.power_nonorm.flat[...] = np.asarray([p0_cov_mocks[i]], dtype="f8").ravel()
        mock_ps_list.append(tmp)
    return data_ps, mock_ps_list


def build_log_taper_window(k_array: np.ndarray, kmin: float, kmax: float, frac: float) -> np.ndarray:
    """
    在对数 k 空间构造余弦窗，控制积分上下限并减小硬截断导致的振铃。

    说明：
    - 这是纯数值稳定性处理（FFTLog 的边界 taper），不改变物理含义，只是让边界平滑。
    """
    w = np.zeros_like(k_array, dtype=np.float64)
    lk = np.log(k_array)
    l0 = np.log(kmin)
    l1 = np.log(kmax)

    dl = frac * (l1 - l0)
    if dl <= 0:
        w[(k_array >= kmin) & (k_array <= kmax)] = 1.0
        return w

    left = l0 + dl
    right = l1 - dl

    m = (lk >= l0) & (lk < left)
    w[m] = 0.5 * (1.0 - np.cos(np.pi * (lk[m] - l0) / dl))

    m = (lk >= left) & (lk <= right)
    w[m] = 1.0

    m = (lk > right) & (lk <= l1)
    w[m] = 0.5 * (1.0 + np.cos(np.pi * (lk[m] - right) / dl))
    return w


def build_ir_param_window(k_array: np.ndarray, k_fund: float, x_power: float) -> np.ndarray:
    """
    mission5 的 exp_power IR 参数化窗口（作用于总 P0）：
      - k < k_f:  W(k) = [1 - exp(-(k/k_f)^x)] / [1 - exp(-1)]
      - k >= k_f: W(k) = 1

    说明：
    - 归一化因子 (1-exp(-1)) 使得 W(k_f^-)=1，从而与 k>=k_f 的拼接连续。
    """
    w = np.ones_like(k_array, dtype=np.float64)
    m = k_array < k_fund
    if np.any(m):
        ratio = np.clip(k_array[m] / k_fund, 0.0, None)
        norm = 1.0 - np.exp(-1.0)
        w[m] = (1.0 - np.exp(-(ratio**x_power))) / max(norm, 1e-30)
    return np.clip(w, 0.0, 1.0)


def xi0_from_p0_fftlog(
    k_grid: np.ndarray,
    p0_grid: np.ndarray,
    kmin: float,
    kmax: float,
    *,
    mu: float = 0.5,
    bias: float = 0.0,
    taper_frac: float = 0.06,
    use_param_ir_window: bool = True,
    k_fund: float | None = None,
    x_power: float | None = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    用 FFTLog 计算 xi0(r)，并支持 mission5 的参数化 IR 窗口。

    参数
    - k_grid/p0_grid: 在对数 k 网格上的 P0(k)
    - kmin/kmax: 本次积分（数值口径）上下限
    - taper_frac: FFTLog 边界平滑窗占对数区间的比例（始终启用）
    - use_param_ir_window: 是否额外乘 IR 物理窗口 W(k)
    - k_fund/x_power: IR 窗口参数（use_param_ir_window=True 时必须提供）

    返回
    - r_grid: FFTLog 输出的 r 网格
    - xi0: 对应的 xi0(r)
    - window_taper: taper 窗（用于 debug/画图）
    - window_ir: IR 窗（用于 debug/画图）
    """
    from scipy.fft import fht, fhtoffset

    # FFTLog 内部边界平滑窗（始终保留）
    window_taper = build_log_taper_window(k_grid, kmin, kmax, taper_frac)

    # 物理 IR 参数化窗口（默认启用）
    if use_param_ir_window:
        if k_fund is None or x_power is None:
            raise ValueError("use_param_ir_window=True 时必须提供 k_fund 与 x_power")
        window_ir = build_ir_param_window(k_grid, k_fund=k_fund, x_power=x_power)
    else:
        window_ir = np.ones_like(k_grid, dtype=np.float64)

    p0_eff = p0_grid * window_taper * window_ir

    # FFTLog 关键关系：k 是等比（log-均匀）网格
    dln = float(np.log(k_grid[1] / k_grid[0]))
    offset = float(fhtoffset(dln, mu=mu, initial=0.0, bias=bias))

    # Hankel 输入序列：a(k)=k^(3/2)P0(k)
    a_in = (k_grid**1.5) * p0_eff
    A_out = fht(a_in, dln=dln, mu=mu, offset=offset, bias=bias)

    # 根据 FFTLog 网格关系恢复输出 r 网格
    n = k_grid.size
    j = np.arange(n)
    j_center = (n - 1) / 2.0
    ln_kc = 0.5 * (np.log(k_grid[0]) + np.log(k_grid[-1]))
    ln_r = -ln_kc + offset + (j - j_center) * dln
    r_grid = np.exp(ln_r)

    # 还原 xi0(r)
    coef = np.sqrt(np.pi / 2.0) / (2.0 * np.pi**2)
    xi0 = coef * A_out / (r_grid**1.5)
    return r_grid, xi0, window_taper, window_ir


def read_pcf_mean_std(
    pcf_dir: str,
    pcf_glob: str,
    realization_min: int,
    realization_max: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, List[str]]:
    """
    读取 pcf realization，返回 r_grid, xi_mean, xi_std, used_files。

    输入文件格式约定（与 skill 的 FCFC 输出一致）：
      # Columns: s_cen(1) s_min(2) s_max(3) xi_0(4)
      s_cen  s_min  s_max  xi0
    """
    files_all = sorted(glob.glob(os.path.join(pcf_dir, pcf_glob)), key=parse_realization_id)
    files = [fp for fp in files_all if realization_min <= parse_realization_id(fp) <= realization_max]
    if len(files) == 0:
        raise RuntimeError(f"未找到 2PCF 输入文件：{pcf_dir}/{pcf_glob}")

    xi_list: List[np.ndarray] = []
    r_grid = None
    for fp in files:
        arr = np.loadtxt(fp, comments="#")
        if r_grid is None:
            r_grid = arr[:, 0]
        xi_list.append(arr[:, 3])

    xi_stack = np.asarray(xi_list, dtype=np.float64)
    xi_mean = np.mean(xi_stack, axis=0)
    xi_std = np.std(xi_stack, axis=0, ddof=1)
    assert r_grid is not None
    return r_grid, xi_mean, xi_std, files


def _to_jsonable_params(bestfit_params: Dict[str, Any]) -> Dict[str, float]:
    """
    desilike 的 bestfit 参数是 ParameterArray；这里转成普通 float 便于落盘。
    """
    out: Dict[str, float] = {}
    for k, v in bestfit_params.items():
        try:
            out[k] = float(np.asarray(v))
        except Exception:  # noqa: BLE001
            out[k] = float(v)
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pk-dir", required=True, help="pk 文件目录（包含 pk_rsd_N*.dat）")
    ap.add_argument("--pcf-dir", required=True, help="pcf 文件目录（包含 pcf_rsd_N*.dat）")
    ap.add_argument("--out-dir", required=True, help="输出目录")

    ap.add_argument("--pk-glob", default="pk_rsd_N*.dat", help="pk 文件 glob")
    ap.add_argument("--pcf-glob", default="pcf_rsd_N*.dat", help="pcf 文件 glob")

    ap.add_argument("--realization-min", type=int, default=1)
    ap.add_argument("--realization-max", type=int, default=50)

    ap.add_argument("--n-datapoints", type=int, default=20, help="参与 pk 拟合的前 N 个 k-bin")
    ap.add_argument("--kcen-col", type=int, default=0)
    ap.add_argument("--kmin-col", type=int, default=1)
    ap.add_argument("--kmax-col", type=int, default=2)
    ap.add_argument("--p0-col", type=int, default=5)

    # FFTLog + IR window 参数（沿用 model.ipynb 的默认口径）
    ap.add_argument("--kint-min", type=float, default=1e-4, help="window 模式的全局 kmin（mission5 口径）")
    ap.add_argument("--kint-max", type=float, default=20.0)
    ap.add_argument("--box-size", type=float, default=1000.0, help="盒长 L（决定 k_f=2pi/L）")
    ap.add_argument("--ir-x", type=float, default=None, help="IR 窗口指数 x；默认用 x(L)=4*(L/1000)")
    ap.add_argument("--fftlog-n", type=int, default=4096)
    ap.add_argument("--fftlog-padding", type=float, default=4.0)
    ap.add_argument("--fftlog-mu", type=float, default=0.5)
    ap.add_argument("--fftlog-bias", type=float, default=0.0)
    ap.add_argument("--edge-taper-frac", type=float, default=0.06)

    # 理论模型参数（沿用 notebook 固定值）
    ap.add_argument("--unit-z", type=float, default=1.0)
    ap.add_argument("--fixed-p", type=float, default=1.2)
    ap.add_argument("--fixed-sn0", type=float, default=0.0)
    ap.add_argument("--fixed-sigmas", type=float, default=0.0)

    ap.add_argument("--minuit-seed", type=int, default=66)
    ap.add_argument("--minuit-niter", type=int, default=27)

    # 误差指标的大尺度阈值
    ap.add_argument("--r-large-min", type=float, default=200.0, help="计算大尺度指标的 r 下限")

    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # ========== 1) 读取 pk ensemble ==========
    pk = load_pk_ensemble(
        data_dir=args.pk_dir,
        file_glob=args.pk_glob,
        realization_min=args.realization_min,
        realization_max=args.realization_max,
        n_datapoints=args.n_datapoints,
        k_cen_col=args.kcen_col,
        k_min_col=args.kmin_col,
        k_max_col=args.kmax_col,
        p0_col=args.p0_col,
    )

    # 关键：过滤掉会导致协方差奇异的 k-bin（例如 1Gpc 下第一个 k-bin 可能恒为 0）
    pk = filter_singular_pk_bins(pk)

    # data=均值，cov=realization 样本（交给 desilike 处理修正因子）
    data_ps, mock_ps_list = build_pypower_data_and_mocks(
        kcen=pk.kcen,
        kmin=pk.kmin,
        kmax=pk.kmax,
        p0_mean_data=pk.p0_mean,
        p0_cov_mocks=pk.p0_mocks,
    )

    # ========== 2) desilike best-fit ==========
    from cosmoprimo import Cosmology
    from desilike import setup_logging
    from desilike.likelihoods import ObservablesGaussianLikelihood
    from desilike.observables.galaxy_clustering import TracerPowerSpectrumMultipolesObservable
    from desilike.profilers import MinuitProfiler
    from desilike.theories.galaxy_clustering import FixedPowerSpectrumTemplate, PNGTracerPowerSpectrumMultipoles

    setup_logging()

    # 宇宙学设置（沿用 notebook 的 UNIT 参数）
    cosmo_unit = Cosmology(
        h=0.6711,
        Omega_b=0.049,
        Omega_cdm=0.3175 - 0.049,
        sigma8=0.834,
        n_s=0.9624,
        engine="class",
    )

    template = FixedPowerSpectrumTemplate(z=args.unit_z, fiducial=cosmo_unit)
    theory = PNGTracerPowerSpectrumMultipoles(template=template, mode="b-p")

    # 固定参数（与 notebook 保持一致）
    theory.init.params["p"].update(fixed=True, value=args.fixed_p)
    theory.init.params["sn0"].update(fixed=True, value=args.fixed_sn0)
    theory.init.params["sigmas"].update(fixed=False, value=args.fixed_sigmas)

    observable = TracerPowerSpectrumMultipolesObservable(
        data=data_ps,
        covariance=mock_ps_list,
        klim={0: [float(pk.kcen.min()), float(pk.kcen.max()), float(pk.kcen[1] - pk.kcen[0])]},
        theory=theory,
    )
    likelihood = ObservablesGaussianLikelihood(observables=[observable])
    _ = likelihood()  # 初始化并构建协方差

    # 为避免 profiling 重置参数状态，在 likelihood 层再固定一次
    likelihood.all_params["p"].update(fixed=True, value=args.fixed_p)
    likelihood.all_params["sn0"].update(fixed=True, value=args.fixed_sn0)
    likelihood.all_params["sigmas"].update(fixed=False, value=args.fixed_sigmas)

    print("[FIT] 开始 Minuit profiling...")
    profiler = MinuitProfiler(likelihood, seed=args.minuit_seed)
    profiles = profiler.maximize(niterations=args.minuit_niter)
    print("[FIT] 统计：")
    print(profiles.to_stats(tablefmt="pretty"))

    bestfit_params_raw = profiles.bestfit.choice(input=True)
    bestfit_params = _to_jsonable_params(bestfit_params_raw)
    (out_dir / "bestfit_params.json").write_text(json.dumps(bestfit_params, indent=2, sort_keys=True))
    print(f"[FIT] bestfit_params saved: {out_dir / 'bestfit_params.json'}")

    # ========== 3) FFTLog 计算 xi0 ==========
    k_fund = 2.0 * np.pi / float(args.box_size)
    x_power = float(args.ir_x) if args.ir_x is not None else float(4.0 * (args.box_size / 1000.0))

    # 3a) baseline / window 各自用独立 k 网格，避免口径混用
    baseline_kmin = k_fund
    k_grid_baseline = np.geomspace(baseline_kmin / args.fftlog_padding, args.kint_max * args.fftlog_padding, args.fftlog_n)
    k_grid_window = np.geomspace(args.kint_min / args.fftlog_padding, args.kint_max * args.fftlog_padding, args.fftlog_n)

    def build_png_theory_on_kgrid(k_grid: np.ndarray) -> np.ndarray:
        """
        在指定高分辨率 k 网格上评估 best-fit P0(k)。

        这里重新实例化 theory（指定 k 网格），避免复用拟合用的 theory 引入网格混乱。
        """
        template_fft = FixedPowerSpectrumTemplate(z=args.unit_z, fiducial=cosmo_unit)
        theory_fft = PNGTracerPowerSpectrumMultipoles(k=k_grid, template=template_fft, mode="b-p")
        theory_fft.init.params["p"].update(fixed=True, value=args.fixed_p)
        theory_fft.init.params["sn0"].update(fixed=True, value=args.fixed_sn0)
        theory_fft.init.params["sigmas"].update(fixed=False, value=args.fixed_sigmas)
        theory_fft(**bestfit_params_raw)
        return np.asarray(theory_fft.power[0], dtype=np.float64)

    p0_model_baseline = build_png_theory_on_kgrid(k_grid_baseline)
    p0_model_window = build_png_theory_on_kgrid(k_grid_window)

    r_base, xi_base, _, _ = xi0_from_p0_fftlog(
        k_grid=k_grid_baseline,
        p0_grid=p0_model_baseline,
        kmin=baseline_kmin,
        kmax=args.kint_max,
        mu=args.fftlog_mu,
        bias=args.fftlog_bias,
        taper_frac=args.edge_taper_frac,
        use_param_ir_window=False,
        k_fund=k_fund,
        x_power=x_power,
    )
    r_win, xi_win, _, _ = xi0_from_p0_fftlog(
        k_grid=k_grid_window,
        p0_grid=p0_model_window,
        kmin=args.kint_min,
        kmax=args.kint_max,
        mu=args.fftlog_mu,
        bias=args.fftlog_bias,
        taper_frac=args.edge_taper_frac,
        use_param_ir_window=True,
        k_fund=k_fund,
        x_power=x_power,
    )

    print("[FFTLog] done.")
    print(f"  baseline: kmin=k_f={k_fund:.6f}, kmax={args.kint_max}")
    print(f"  window  : kmin_global={args.kint_min:.1e}, kmax={args.kint_max}, x={x_power:.3f}")

    # ========== 4) 读取 2PCF 数据并对比 ==========
    r_data, xi_mean, xi_std, used_pcf = read_pcf_mean_std(
        pcf_dir=args.pcf_dir,
        pcf_glob=args.pcf_glob,
        realization_min=args.realization_min,
        realization_max=args.realization_max,
    )
    print(f"[PCF] files used = {len(used_pcf)}  r-range=[{r_data.min():.1f}, {r_data.max():.1f}]")

    # 插值模型到观测 r 网格
    ord_b = np.argsort(r_base)
    ord_w = np.argsort(r_win)
    xi_model_base = np.interp(r_data, r_base[ord_b], xi_base[ord_b])
    xi_model_win = np.interp(r_data, r_win[ord_w], xi_win[ord_w])

    # 使用 r^2 * xi 来定义误差指标（与 notebook 一致）
    r2_xi_data = r_data**2 * xi_mean
    r2_xi_err = r_data**2 * xi_std
    r2_xi_base = r_data**2 * xi_model_base
    r2_xi_win = r_data**2 * xi_model_win

    # 避免 sigma=0 引起发散：给一个很小的下限
    sigma_floor = np.nanmedian(r2_xi_err[r2_xi_err > 0]) * 1e-6 if np.any(r2_xi_err > 0) else 1e-12
    r2_xi_err_safe = np.where(r2_xi_err > 0, r2_xi_err, sigma_floor)

    delta_sigma_base = (r2_xi_data - r2_xi_base) / r2_xi_err_safe
    delta_sigma_win = (r2_xi_data - r2_xi_win) / r2_xi_err_safe

    # 全尺度与大尺度指标
    def _stats(x: np.ndarray, mask: np.ndarray) -> Dict[str, float]:
        xx = np.asarray(x[mask], dtype=np.float64)
        return {
            "mean_abs": float(np.nanmean(np.abs(xx))),
            "max_abs": float(np.nanmax(np.abs(xx))),
            "count": int(np.sum(mask)),
        }

    mask_all = np.isfinite(delta_sigma_win)
    mask_large = (r_data >= args.r_large_min) & mask_all

    metrics = {
        "inputs": {
            "pk_dir": args.pk_dir,
            "pcf_dir": args.pcf_dir,
            "realization_min": args.realization_min,
            "realization_max": args.realization_max,
            "n_datapoints": args.n_datapoints,
        },
        "fftlog": {
            "kint_min": args.kint_min,
            "kint_max": args.kint_max,
            "box_size": args.box_size,
            "k_fund": float(k_fund),
            "ir_x": float(x_power),
        },
        "fit": {
            "bestfit_params": bestfit_params,
        },
        "delta_sigma": {
            "baseline_all": _stats(delta_sigma_base, mask_all),
            "window_all": _stats(delta_sigma_win, mask_all),
            "baseline_large": _stats(delta_sigma_base, mask_large),
            "window_large": _stats(delta_sigma_win, mask_large),
            "r_large_min": float(args.r_large_min),
        },
    }
    (out_dir / "metrics.json").write_text(json.dumps(metrics, indent=2, sort_keys=True))
    print(f"[METRICS] saved: {out_dir / 'metrics.json'}")

    # ========== 5) 画图 ==========
    import matplotlib.pyplot as plt

    plt.figure(figsize=(10, 8))

    ax1 = plt.subplot(2, 1, 1)
    ax1.errorbar(
        r_data,
        r2_xi_data,
        yerr=r2_xi_err,
        fmt="o",
        ms=4,
        capsize=2,
        color="black",
        label=r"Measured $r^2\xi_0$ mean",
    )
    ax1.plot(r_data, r2_xi_base, "-", lw=1.8, color="tab:blue", label="Model baseline (FFTLog only)")
    ax1.plot(
        r_data,
        r2_xi_win,
        "-",
        lw=2.0,
        color="tab:red",
        label=f"Model window exp_power(x={x_power:.2f})",
    )
    ax1.set_ylabel(r"$r^2\xi_0(r)$")
    ax1.grid(True, ls="--", alpha=0.35)
    ax1.legend()

    ax2 = plt.subplot(2, 1, 2)
    ax2.axhline(0.0, color="black", ls="-", lw=1.0)
    ax2.axhline(1.0, color="red", ls="--", lw=1.0)
    ax2.axhline(-1.0, color="red", ls="--", lw=1.0)
    ax2.plot(r_data, delta_sigma_win, "o-", ms=3, lw=1.0, color="tab:red", alpha=0.9, label="(Data-Model)/sigma (window)")
    ax2.set_xlabel(r"$r\ [\mathrm{Mpc}/h]$")
    ax2.set_ylabel(r"$(Data-Model)/\sigma$ of $r^2\xi_0$")
    ax2.grid(True, ls="--", alpha=0.35)
    ax2.legend(frameon=False)

    plt.tight_layout()
    fig_path = out_dir / "r2xi_compare.png"
    plt.savefig(fig_path, dpi=180)
    print(f"[PLOT] saved: {fig_path}")


if __name__ == "__main__":
    main()
