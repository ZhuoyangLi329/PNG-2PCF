#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
任务5 v9: 用 model.ipynb 同款方法检查 fnl0 下 best-fit PK -> 2PCF 的一致性。

输出:
- 图: task5_v9_fnl0_pk2xi_1gpc_3gpc_check.png
- 文本: task5_v9_fnl0_pk2xi_1gpc_3gpc_check_summary.txt
"""

from __future__ import annotations

import glob
import os
import re
from dataclasses import dataclass
from typing import Dict, List, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.fft import fht, fhtoffset


# =====================
# 参数区（对齐 model.ipynb + 你之前要求）
# =====================
OUT_DIR = "/pscratch/sd/l/lzy/1gpc-UNIT-fnl100-fastpm/agent/mission5_log"
os.makedirs(OUT_DIR, exist_ok=True)
OUT_PNG = os.path.join(OUT_DIR, "task5_v9_fnl0_pk2xi_1gpc_3gpc_check.png")
OUT_TXT = os.path.join(OUT_DIR, "task5_v9_fnl0_pk2xi_1gpc_3gpc_check_summary.txt")

PK_FIT_KMAX = 0.08
UNIT_Z = 1.0
FIXED_P = 1.1
FIXED_SN0 = 0.0
FIXED_SIGMAS = 0.0
MINUIT_SEED = 66
MINUIT_NITER = 25

# model.ipynb 同款 FFTLog 设定
K_INT_MIN = 0.003
K_INT_MAX = 1.0
FFTLOG_N = 4096
FFTLOG_PADDING = 4.0
FFTLOG_MU = 0.5
FFTLOG_BIAS = 0.0
EDGE_TAPER_FRAC = 0.06

LARGE_SCALE_MIN = 200.0

PK_KCEN_COL = 0
PK_KMIN_COL = 1
PK_KMAX_COL = 2
PK_P0_COL = 5
PCF_SCEN_COL = 0
PCF_XI0_COL = 3


# =====================
# desilike 依赖
# =====================
try:
    from cosmoprimo import Cosmology
    from desilike import setup_logging
    from desilike.theories.galaxy_clustering import FixedPowerSpectrumTemplate, PNGTracerPowerSpectrumMultipoles
    from desilike.observables.galaxy_clustering import TracerPowerSpectrumMultipolesObservable
    from desilike.likelihoods import ObservablesGaussianLikelihood
    from desilike.profilers import MinuitProfiler
except ModuleNotFoundError as exc:
    raise SystemExit(
        "缺少 desilike 依赖，请在可用环境中运行。\n"
        f"原始报错: {repr(exc)}"
    )

setup_logging()


@dataclass
class BoxConfig:
    name: str
    box_size: float
    pk_glob: str
    pcf_glob: str
    rid_min: int
    rid_max: int


@dataclass
class Dataset:
    realizations: np.ndarray
    kcen: np.ndarray
    kmin: np.ndarray
    kmax: np.ndarray
    p0_mocks: np.ndarray
    s: np.ndarray
    xi_mocks: np.ndarray

    @property
    def nmock(self) -> int:
        return int(self.p0_mocks.shape[0])

    @property
    def p0_mean(self) -> np.ndarray:
        return np.mean(self.p0_mocks, axis=0)

    @property
    def p0_std(self) -> np.ndarray:
        return np.std(self.p0_mocks, axis=0, ddof=1)

    @property
    def xi_mean(self) -> np.ndarray:
        return np.mean(self.xi_mocks, axis=0)

    @property
    def xi_std(self) -> np.ndarray:
        return np.std(self.xi_mocks, axis=0, ddof=1)


def parse_realization_id(path: str) -> int:
    m = re.search(r"_N(\d+)\.dat$", os.path.basename(path))
    if m is None:
        raise ValueError(f"无法解析 realization: {path}")
    return int(m.group(1))


def assert_same_grid(arrays: List[np.ndarray], name: str) -> None:
    ref = arrays[0]
    for i, arr in enumerate(arrays[1:], start=1):
        if not np.allclose(ref, arr):
            raise ValueError(f"{name} 网格不一致: index={i}")


def load_dataset(cfg: BoxConfig) -> Dataset:
    pk_files = glob.glob(cfg.pk_glob)
    pcf_files = glob.glob(cfg.pcf_glob)
    if not pk_files or not pcf_files:
        raise FileNotFoundError(f"{cfg.name}: pk 或 pcf 文件为空")

    pk_map = {parse_realization_id(fp): fp for fp in pk_files}
    pcf_map = {parse_realization_id(fp): fp for fp in pcf_files}

    common = sorted(set(pk_map) & set(pcf_map))
    common = [rid for rid in common if cfg.rid_min <= rid <= cfg.rid_max]
    if not common:
        raise RuntimeError(f"{cfg.name}: 没有共同 realization")

    pk_tables = [np.loadtxt(pk_map[r], comments="#") for r in common]
    pcf_tables = [np.loadtxt(pcf_map[r], comments="#") for r in common]

    kcen_list = [t[:, PK_KCEN_COL] for t in pk_tables]
    kmin_list = [t[:, PK_KMIN_COL] for t in pk_tables]
    kmax_list = [t[:, PK_KMAX_COL] for t in pk_tables]
    assert_same_grid(kcen_list, f"{cfg.name} kcen")
    assert_same_grid(kmin_list, f"{cfg.name} kmin")
    assert_same_grid(kmax_list, f"{cfg.name} kmax")

    s_list = [t[:, PCF_SCEN_COL] for t in pcf_tables]
    assert_same_grid(s_list, f"{cfg.name} s")

    return Dataset(
        realizations=np.array(common, dtype=int),
        kcen=kcen_list[0].copy(),
        kmin=kmin_list[0].copy(),
        kmax=kmax_list[0].copy(),
        p0_mocks=np.vstack([t[:, PK_P0_COL] for t in pk_tables]),
        s=s_list[0].copy(),
        xi_mocks=np.vstack([t[:, PCF_XI0_COL] for t in pcf_tables]),
    )


def convert_bestfit_to_float_dict(bestfit: Dict[str, object]) -> Dict[str, float]:
    out: Dict[str, float] = {}
    for key, val in bestfit.items():
        out[key] = float(np.ravel(np.asarray(val))[0])
    return out


def fit_best_pk(data: Dataset) -> Dict[str, float]:
    mask = data.kcen <= PK_FIT_KMAX
    # 防止协方差奇异：剔除零方差或非有限 bin（1Gpc fnl0 的最小 k bin 为全 0）
    p0_std = data.p0_std
    mask &= np.isfinite(p0_std) & (p0_std > 0.0)
    if mask.sum() < 5:
        raise RuntimeError("低-k 拟合点太少")

    kfit = data.kcen[mask]
    p0_mean_fit = data.p0_mean[mask]
    p0_mocks_fit = data.p0_mocks[:, mask]

    cosmo_unit = Cosmology(
        h=0.6711,
        Omega_b=0.049,
        Omega_cdm=0.3175 - 0.049,
        sigma8=0.834,
        n_s=0.9624,
        engine="class",
    )

    template = FixedPowerSpectrumTemplate(z=UNIT_Z, fiducial=cosmo_unit)
    theory = PNGTracerPowerSpectrumMultipoles(template=template, mode="b-p")
    theory.init.params["p"].update(fixed=True, value=FIXED_P)
    theory.init.params["sn0"].update(fixed=True, value=FIXED_SN0)
    theory.init.params["sigmas"].update(fixed=False, value=FIXED_SIGMAS)

    dk = float(np.median(np.diff(kfit)))
    observable = TracerPowerSpectrumMultipolesObservable(
        data=p0_mean_fit,
        covariance=[row for row in p0_mocks_fit],
        klim={0: [float(kfit.min()), float(kfit.max()), dk]},
        k=kfit,
        ells=[0],
        theory=theory,
    )
    likelihood = ObservablesGaussianLikelihood(observables=[observable])
    _ = likelihood()

    likelihood.all_params["p"].update(fixed=True, value=FIXED_P)
    likelihood.all_params["sn0"].update(fixed=True, value=FIXED_SN0)
    likelihood.all_params["sigmas"].update(fixed=False, value=FIXED_SIGMAS)

    profiler = MinuitProfiler(likelihood, seed=MINUIT_SEED)
    profiles = profiler.maximize(niterations=MINUIT_NITER)
    return convert_bestfit_to_float_dict(profiles.bestfit.choice(input=True))


def build_log_taper_window(k_array: np.ndarray, kmin: float, kmax: float, frac: float) -> np.ndarray:
    w = np.zeros_like(k_array, dtype=float)
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


def xi_custom_fftlog(k_grid: np.ndarray, p0_grid: np.ndarray, kmin: float, kmax: float) -> Tuple[np.ndarray, np.ndarray]:
    p0_cut = p0_grid * build_log_taper_window(k_grid, kmin, kmax, EDGE_TAPER_FRAC)

    dln = np.log(k_grid[1] / k_grid[0])
    offset = fhtoffset(dln, mu=FFTLOG_MU, initial=0.0, bias=FFTLOG_BIAS)

    a_in = (k_grid ** 1.5) * p0_cut
    a_out = fht(a_in, dln=dln, mu=FFTLOG_MU, offset=offset, bias=FFTLOG_BIAS)

    n = k_grid.size
    j = np.arange(n)
    jc = (n - 1) / 2.0
    ln_kc = 0.5 * (np.log(k_grid[0]) + np.log(k_grid[-1]))
    r_grid = np.exp((offset - ln_kc) + (j - jc) * dln)

    const = np.sqrt(np.pi / 2.0) / (2.0 * np.pi**2)
    xi_grid = const * a_out / (r_grid ** 1.5)
    return r_grid, xi_grid


def model_xi_on_s(bestfit: Dict[str, float], s_data: np.ndarray) -> np.ndarray:
    k_grid = np.geomspace(K_INT_MIN / FFTLOG_PADDING, K_INT_MAX * FFTLOG_PADDING, FFTLOG_N)

    cosmo_unit = Cosmology(
        h=0.6711,
        Omega_b=0.049,
        Omega_cdm=0.3175 - 0.049,
        sigma8=0.834,
        n_s=0.9624,
        engine="class",
    )
    template = FixedPowerSpectrumTemplate(z=UNIT_Z, fiducial=cosmo_unit)
    theory = PNGTracerPowerSpectrumMultipoles(k=k_grid, template=template, mode="b-p")
    theory.init.params["p"].update(fixed=True, value=FIXED_P)
    theory.init.params["sn0"].update(fixed=True, value=FIXED_SN0)
    theory.init.params["sigmas"].update(fixed=False, value=FIXED_SIGMAS)
    theory(**bestfit)
    p0_grid = np.asarray(theory.power[0], dtype=float)

    r_grid, xi_grid = xi_custom_fftlog(k_grid, p0_grid, kmin=K_INT_MIN, kmax=K_INT_MAX)
    order = np.argsort(r_grid)
    return np.interp(s_data, r_grid[order], xi_grid[order])


def compute_metrics(s: np.ndarray, xi_data: np.ndarray, xi_std: np.ndarray, xi_model: np.ndarray) -> Dict[str, float]:
    r2 = s**2
    data = r2 * xi_data
    model = r2 * xi_model
    err = np.maximum(r2 * xi_std, 1e-12)

    delta_over_sigma = (model - data) / err
    mask = s >= LARGE_SCALE_MIN

    chi2 = float(np.sum(delta_over_sigma[mask] ** 2))
    ndof = int(mask.sum())
    return {
        "mean_abs_delta_sigma": float(np.mean(np.abs(delta_over_sigma[mask]))),
        "max_abs_delta_sigma": float(np.max(np.abs(delta_over_sigma[mask]))),
        "chi2_over_ndof": chi2 / max(ndof, 1),
    }


def run_one_box(cfg: BoxConfig) -> Dict[str, object]:
    data = load_dataset(cfg)
    bestfit = fit_best_pk(data)
    xi_model = model_xi_on_s(bestfit, data.s)
    metrics = compute_metrics(data.s, data.xi_mean, data.xi_std, xi_model)
    return {
        "config": cfg,
        "data": data,
        "bestfit": bestfit,
        "xi_model": xi_model,
        "metrics": metrics,
    }


def plot_joint(result_1gpc: Dict[str, object], result_3gpc: Dict[str, object]) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(13.6, 9.0), sharex="col", gridspec_kw={"height_ratios": [2.0, 1.2]})

    for col, res in enumerate([result_1gpc, result_3gpc]):
        cfg = res["config"]
        data = res["data"]
        xi_model = res["xi_model"]
        metrics = res["metrics"]

        s = data.s
        r2 = s**2
        y_data = r2 * data.xi_mean
        y_err = r2 * data.xi_std
        y_model = r2 * xi_model

        ax = axes[0, col]
        ax.errorbar(
            s,
            y_data,
            yerr=y_err,
            fmt="o",
            ms=3.3,
            capsize=2,
            color="black",
            label=f"Measured mean (N={data.nmock})",
        )
        ax.plot(s, y_model, "-", lw=2.0, color="tab:red", label="Best-fit PK -> 2PCF model")
        ax.grid(alpha=0.25)
        ax.set_title(f"{cfg.name} fnl0")
        ax.set_ylabel(r"$r^2\xi_0(r)$")
        ax.legend(loc="best", frameon=False, fontsize=9)

        text = (
            f"PK_FIT_KMAX={PK_FIT_KMAX:.3f}, p={FIXED_P:.1f}\n"
            f"r>={LARGE_SCALE_MIN:.0f}: mean|Δ/σ|={metrics['mean_abs_delta_sigma']:.2f}\n"
            f"r>={LARGE_SCALE_MIN:.0f}: χ²/ndof={metrics['chi2_over_ndof']:.2f}"
        )
        ax.text(
            0.02,
            0.98,
            text,
            transform=ax.transAxes,
            ha="left",
            va="top",
            fontsize=8.8,
            bbox=dict(boxstyle="round,pad=0.22", facecolor="white", alpha=0.80, edgecolor="0.8"),
        )

        axr = axes[1, col]
        delta_sigma = np.divide(
            y_model - y_data,
            np.maximum(y_err, 1e-12),
            out=np.zeros_like(y_data),
            where=np.maximum(y_err, 1e-12) > 0,
        )
        axr.axhline(0.0, color="black", lw=1.0, alpha=0.6)
        axr.axhline(1.0, color="gray", lw=0.8, ls="--", alpha=0.8)
        axr.axhline(-1.0, color="gray", lw=0.8, ls="--", alpha=0.8)
        axr.plot(s, delta_sigma, color="tab:blue", lw=1.8)
        axr.set_ylabel(r"$(model-data)/\sigma$", fontsize=10)
        axr.set_xlabel(r"$r\ [h^{-1}{\rm Mpc}]$")
        axr.grid(alpha=0.25)

    fig.suptitle("fnl=0: model.ipynb-style best-fit PK -> 2PCF check (1Gpc & 3Gpc)", y=0.995)
    fig.tight_layout()
    fig.savefig(OUT_PNG, dpi=180)
    plt.close(fig)


def write_summary(result_1gpc: Dict[str, object], result_3gpc: Dict[str, object]) -> None:
    def line_for(res: Dict[str, object]) -> str:
        cfg = res["config"]
        data = res["data"]
        bestfit = res["bestfit"]
        m = res["metrics"]
        return (
            f"[{cfg.name}] Nmock={data.nmock}, rid={data.realizations.min()}..{data.realizations.max()}\n"
            f"  best-fit: fnl_loc={bestfit.get('fnl_loc', np.nan):.4f}, "
            f"b1={bestfit.get('b1', np.nan):.4f}, sigmas={bestfit.get('sigmas', np.nan):.6f}\n"
            f"  r>={LARGE_SCALE_MIN:.0f}: mean|Δ/σ|={m['mean_abs_delta_sigma']:.4f}, "
            f"max|Δ/σ|={m['max_abs_delta_sigma']:.4f}, χ²/ndof={m['chi2_over_ndof']:.4f}\n"
        )

    with open(OUT_TXT, "w", encoding="utf-8") as f:
        f.write("任务5 v9: fnl0 下 1Gpc/3Gpc 的 best-fit PK -> 2PCF 一致性检查\n")
        f.write("========================================================\n")
        f.write(f"方法: 对齐 model.ipynb (PK best-fit + FFTLog, k_int=[{K_INT_MIN},{K_INT_MAX}], taper={EDGE_TAPER_FRAC})\n")
        f.write(f"拟合: PK_FIT_KMAX={PK_FIT_KMAX}, 固定 p={FIXED_P}, sn0={FIXED_SN0}, sigmas 自由\n\n")
        f.write(line_for(result_1gpc))
        f.write("\n")
        f.write(line_for(result_3gpc))
        f.write("\n")

        m1 = result_1gpc["metrics"]["mean_abs_delta_sigma"]
        m3 = result_3gpc["metrics"]["mean_abs_delta_sigma"]
        if m1 <= 1.0 and m3 <= 1.0:
            verdict = "两个盒子在大尺度上都可认为与测量均值较好一致（约 1σ 内）。"
        elif m1 <= 2.0 and m3 <= 2.0:
            verdict = "两个盒子在大尺度上基本接近测量，但存在可见偏差（约 1~2σ）。"
        else:
            verdict = "至少有一个盒子在大尺度上与测量仍有较明显偏差（超过约 2σ）。"

        f.write("结论:\n")
        f.write(f"- {verdict}\n")
        f.write(f"- 图文件: {OUT_PNG}\n")


def main() -> None:
    cfg_1gpc = BoxConfig(
        name="1Gpc",
        box_size=1000.0,
        pk_glob="/pscratch/sd/l/lzy/1gpc-UNIT-fnl100-fastpm/pk_masscut_fnl0/pk_rsd_N*.dat",
        pcf_glob="/pscratch/sd/l/lzy/1gpc-UNIT-fnl100-fastpm/pcf_masscut_fnl0/pcf_rsd_N*.dat",
        rid_min=1,
        rid_max=50,
    )
    cfg_3gpc = BoxConfig(
        name="3Gpc",
        box_size=3000.0,
        pk_glob="/pscratch/sd/l/lzy/3gpc-UNIT-fastpm/pk_masscut/pk_rsd_3gpc_fnl0_N*.dat",
        pcf_glob="/pscratch/sd/l/lzy/3gpc-UNIT-fastpm/pcf_masscut/pcf_rsd_3gpc_fnl0_N*.dat",
        rid_min=2,
        rid_max=99,
    )

    print("[INFO] running 1Gpc fnl0...")
    result_1gpc = run_one_box(cfg_1gpc)

    print("[INFO] running 3Gpc fnl0...")
    result_3gpc = run_one_box(cfg_3gpc)

    plot_joint(result_1gpc, result_3gpc)
    write_summary(result_1gpc, result_3gpc)

    print(f"[OK ] saved figure: {OUT_PNG}")
    print(f"[OK ] saved summary: {OUT_TXT}")


if __name__ == "__main__":
    main()
