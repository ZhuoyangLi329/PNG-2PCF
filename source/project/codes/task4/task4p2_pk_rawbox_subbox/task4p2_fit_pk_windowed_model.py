#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Task4.2 P(k): shot-noise-aware rawbox/subbox P(k) likelihood.

执行逻辑：
1. 读取 rawbox 或 subbox P(k) payload；rawbox 是 periodic shot-noise-subtracted P0。
2. subbox 使用 jaxpower window matrix，并按 DESI/desilike 约定前向建模：
   observable = W(P_theory + P_poisson) - P_poisson。
   observed bins 按 Lsub fundamental cut；window 输入端 theory grid 使用独立低 k cutoff。
3. 理论默认使用 no-RSD PNG real-space P(k)，可选 FoG-like sigma_s nuisance。
4. 用 emcee 采样 fnl_loc、b1、sigmas、sn0，并写 samples/summary/PDF。

红移口径：
- Task4.2 的 FastPM-L3 catalog 是 z=1 snapshot；默认模板红移必须使用共享常量
  FASTPM_SNAPSHOT_Z=1，而不是 cut-sky z-shell 的中点或 effective redshift。
- 非 z=1 的诊断只有显式传入 --allow-template-z-override 才能运行，防止两条
  statistic pipeline 再次静默使用不同红移。
"""

from __future__ import annotations

import argparse
import math
import os
import re
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

for _name in (
    "OMP_NUM_THREADS",
    "MKL_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
):
    os.environ.setdefault(_name, "1")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from task4p2_pk_common import (
    DELTA_C,
    FASTPM_SNAPSHOT_Z,
    K_MAX_FIT,
    K_MIN_MODEL,
    OUTPUT_ROOT,
    P_FIXED,
    atomic_savez,
    build_parent_binavg_matrix,
    covariance_corrections,
    interp_logk,
    load_rawbox_pk_file,
    to_jsonable,
    write_json,
)


PARAM_NAMES = ("fnl_loc", "b1", "sigmas", "sn0")
DEFAULT_PRIORS = {
    "fnl_loc": (-500.0, 500.0),
    "b1": (0.5, 5.0),
    "sigmas": (0.0, 30.0),
    "sn0": (-1.0, 1.0),
}
SN0_SCALE_DESILIKE = 1.0e4


def free_parameter_names(sigmas_policy: str) -> tuple[str, ...]:
    """返回给 covariance/Percival bookkeeping 使用的实际自由参数。"""
    if sigmas_policy == "fixed_zero":
        return tuple(name for name in PARAM_NAMES if name != "sigmas")
    return PARAM_NAMES


@dataclass
class FitData:
    """一个 likelihood 所需的固定输入。"""

    label: str
    mode: str
    k_obs: np.ndarray
    pk_data: np.ndarray
    cov: np.ndarray
    precision: np.ndarray
    covariance_meta: dict[str, object]
    source_payload: Path
    extra_meta: dict[str, object]
    rawbox_matrix: np.ndarray | None = None
    rawbox_k: np.ndarray | None = None
    window_matrix: np.ndarray | None = None
    theory_k: np.ndarray | None = None
    theory_ell: np.ndarray | None = None
    shotnoise_out: np.ndarray | None = None
    shotnoise_scalar: float = 0.0


def parse_prior(text: str) -> tuple[float, float]:
    """解析形如 lo,hi 的均匀 prior。"""
    lo, hi = (float(x) for x in str(text).split(",", 1))
    if not lo < hi:
        raise ValueError(f"prior 下界必须小于上界: {text}")
    return lo, hi


def build_cosmology():
    """构建与 Task4.2 no-RSD 理论侧一致的 cosmoprimo cosmology。"""
    from cosmoprimo import Cosmology

    return Cosmology(
        h=0.6711,
        Omega_b=0.049,
        Omega_cdm=0.3175 - 0.049,
        sigma8=0.834,
        n_s=0.9624,
        engine="class",
    )


def build_template_arrays(k_template: np.ndarray, z: float = FASTPM_SNAPSHOT_Z) -> dict[str, np.ndarray]:
    """生成线性 matter P(k) 和 local-PNG alpha(k) 模板。"""
    from desilike.theories.galaxy_clustering import FixedPowerSpectrumTemplate

    cosmo = build_cosmology()
    template = FixedPowerSpectrumTemplate(z=float(z), fiducial=cosmo, k=np.asarray(k_template, dtype="f8"))
    template()
    kin = np.asarray(template.k, dtype="f8")
    pk_dd = np.asarray(template.pk_dd, dtype="f8")
    pk_prim = cosmo.get_primordial(mode="scalar").pk_interpolator()(kin)
    pphi_prim = 9.0 / 25.0 * 2.0 * np.pi**2 / kin**3 * pk_prim / cosmo.h**3
    alpha = 1.0 / np.sqrt(pk_dd / pphi_prim)
    return {"k": kin, "pk_dd": pk_dd, "alpha": alpha}


def hartlap_precision_local(cov: np.ndarray, *, nmock: int, nparams: int) -> tuple[np.ndarray, dict[str, object]]:
    """按当前参数维度返回 Hartlap precision 和 Percival 修正元数据。"""
    cov = np.asarray(cov, dtype="f8")
    corrections = covariance_corrections(nmock=int(nmock), ndata=int(cov.shape[0]), nparams=int(nparams))
    precision = corrections["hartlap"] * np.linalg.pinv(cov, rcond=1.0e-10)
    return precision, {
        **corrections,
        "nmock": int(nmock),
        "ndata": int(cov.shape[0]),
        "nparams": int(nparams),
        "condition_number": float(np.linalg.cond(cov)),
        "used_covariance_divided_by_nmock": False,
    }


def load_rawbox_fit_data(path: Path, *, nparams: int) -> FitData:
    """读取 rawbox payload，并构造 parent-box mode-count bin-average 矩阵。"""
    with np.load(path) as data:
        kcen = np.asarray(data["kcen"], dtype="f8")
        kmin = np.asarray(data["kmin"], dtype="f8")
        kmax = np.asarray(data["kmax"], dtype="f8")
        pk_data = np.asarray(data["pk_mean"], dtype="f8")
        cov = np.asarray(data["pk_cov"], dtype="f8")
        nmock = int(np.asarray(data["realizations"]).size)
        source_files = [str(x) for x in data["source_files"]] if "source_files" in data.files else []
        rawbox_theory_kmin = (
            float(np.asarray(data["rawbox_theory_kmin"]).item())
            if "rawbox_theory_kmin" in data.files
            else float(np.asarray(data["kmin_model"]).item())
            if "kmin_model" in data.files
            else float(K_MIN_MODEL)
        )
        fit_kmin_observed = (
            float(np.asarray(data["fit_kmin_observed"]).item())
            if "fit_kmin_observed" in data.files
            else float(np.asarray(data["kmin_observed"]).item())
            if "kmin_observed" in data.files
            else float(K_MIN_MODEL)
        )
        rawbox_mode_kfund = (
            float(np.asarray(data["rawbox_mode_kfund"]).item())
            if "rawbox_mode_kfund" in data.files
            else float(np.asarray(data["parent_kfund"]).item())
            if "parent_kfund" in data.files
            else float(K_MIN_MODEL)
        )
    precision, cov_meta = hartlap_precision_local(cov, nmock=nmock, nparams=int(nparams))
    unique_k, matrix = build_parent_binavg_matrix(kmin, kmax)
    if rawbox_theory_kmin > float(np.min(unique_k)) + 1.0e-12:
        raise ValueError(
            "rawbox_theory_kmin would zero parent-box modes inside selected data bins; "
            "use fit_kmin_observed only for selecting observed bins."
        )
    shot = estimate_rawbox_shotnoise(source_files)
    return FitData(
        label="rawbox",
        mode="rawbox_parent_binavg",
        k_obs=kcen,
        pk_data=pk_data,
        cov=cov,
        precision=precision,
        covariance_meta=cov_meta,
        source_payload=path,
        rawbox_matrix=matrix,
        rawbox_k=unique_k,
        shotnoise_scalar=shot,
        extra_meta={
            "rawbox_theory_kmin": float(rawbox_theory_kmin),
            "fit_kmin_observed": float(fit_kmin_observed),
            "rawbox_mode_kfund": float(rawbox_mode_kfund),
            "kmin_model": float(rawbox_theory_kmin),
            "kmin_observed": float(fit_kmin_observed),
            "parent_kfund": float(rawbox_mode_kfund),
            "kmax_fit": float(K_MAX_FIT),
            "unique_parent_modes": int(unique_k.size),
            "poisson_shotnoise_mean_from_headers": float(shot),
        },
    )


def estimate_rawbox_shotnoise(source_files: list[str]) -> float:
    """从 POWSPEC header 读取 rawbox Poisson shot noise；读不到则返回 0。"""
    values = []
    pattern = re.compile(r"Shot noise:\s*([0-9eE+\-.]+)")
    for filename in source_files:
        path = Path(filename)
        if not path.exists():
            continue
        with path.open("r", encoding="utf-8", errors="ignore") as file:
            for line in file:
                if not line.startswith("#"):
                    break
                match = pattern.search(line)
                if match:
                    values.append(float(match.group(1)))
                    break
    return float(np.mean(values)) if values else 0.0


def reconstruct_shotnoise_from_sources(source_files: np.ndarray, fit_mask: np.ndarray | None) -> tuple[np.ndarray, float]:
    """老 payload 没有 shotnoise_mean 时，从单 subbox npz 重新计算。"""
    stack = []
    for filename in source_files:
        with np.load(str(filename)) as data:
            shot = np.asarray(data["num_shotnoise"], dtype="f8") / np.asarray(data["norm"], dtype="f8")
            stack.append(shot)
    shot_stack = np.vstack(stack)
    if fit_mask is not None and fit_mask.size == shot_stack.shape[1]:
        shot_stack = shot_stack[:, fit_mask]
    shot_mean = np.mean(shot_stack, axis=0)
    return shot_mean, float(np.mean(shot_mean))


def load_subbox_fit_data(path: Path, *, nparams: int) -> FitData:
    """读取 subbox payload，并保留 window matrix 与 shot-noise convention 信息。"""
    with np.load(path) as data:
        k_obs = np.asarray(data["k_obs"], dtype="f8")
        pk_data = np.asarray(data["pk_mean"], dtype="f8")
        cov = np.asarray(data["pk_cov"], dtype="f8")
        nmock = int(np.asarray(data["nmock_cov"]).item())
        window = np.asarray(data["window_matrix"], dtype="f8")
        theory_k = np.asarray(data["theory_k"], dtype="f8")
        theory_ell = np.asarray(data["theory_ell"], dtype="i8")
        lsub = int(np.asarray(data["lsub"]).item())
        kmin_model = float(np.asarray(data["kmin_model"]).item()) if "kmin_model" in data.files else 2.0 * math.pi / float(lsub)
        theory_kmin_model = float(np.asarray(data["theory_kmin_model"]).item()) if "theory_kmin_model" in data.files else 0.0
        if "shotnoise_mean" in data.files:
            shotnoise_out = np.asarray(data["shotnoise_mean"], dtype="f8")
            shotnoise_scalar = float(np.asarray(data["shotnoise_mean_scalar"]).item())
        else:
            fit_mask = np.asarray(data["fit_bin_mask"], dtype=bool) if "fit_bin_mask" in data.files else None
            shotnoise_out, shotnoise_scalar = reconstruct_shotnoise_from_sources(data["source_files"], fit_mask)
    if window.shape[0] != pk_data.size:
        raise ValueError(f"window 行数 {window.shape[0]} != data size {pk_data.size}")
    if window.shape[1] != theory_k.size:
        raise ValueError(f"window 列数 {window.shape[1]} != theory size {theory_k.size}")
    precision, cov_meta = hartlap_precision_local(cov, nmock=nmock, nparams=int(nparams))
    return FitData(
        label=f"L{lsub}",
        mode="subbox_window_forward_shotnoise",
        k_obs=k_obs,
        pk_data=pk_data,
        cov=cov,
        precision=precision,
        covariance_meta=cov_meta,
        source_payload=path,
        window_matrix=window,
        theory_k=theory_k,
        theory_ell=theory_ell,
        shotnoise_out=np.asarray(shotnoise_out, dtype="f8"),
        shotnoise_scalar=float(shotnoise_scalar),
        extra_meta={
            "lsub": int(lsub),
            "kmin_model": float(kmin_model),
            "kmin_observed": float(kmin_model),
            "theory_kmin_model": float(theory_kmin_model),
            "kmax_fit": float(K_MAX_FIT),
            "window_shape": list(window.shape),
            "theory_size": int(theory_k.size),
            "poisson_shotnoise_mean_scalar": float(shotnoise_scalar),
        },
    )


def png_realspace_base(k: np.ndarray, template: dict[str, np.ndarray], *, fnl_loc: float, b1: float, p_fixed: float) -> np.ndarray:
    """计算 no-RSD local-PNG tracer clustering P(k)，不含 stochastic 常数项。"""
    k = np.asarray(k, dtype="f8")
    out = np.zeros_like(k)
    mask = k >= 1.0e-12
    if np.any(mask):
        alpha = interp_logk(k[mask], template["k"], template["alpha"])
        pkdd = interp_logk(k[mask], template["k"], template["pk_dd"])
        bphi = 2.0 * DELTA_C * (float(b1) - float(p_fixed))
        bias = float(b1) + bphi * float(fnl_loc) * alpha
        out[mask] = bias * bias * pkdd
    return out


def fog_multipole_coefficients(k: np.ndarray, sigmas: float, ell: int, nmu: int = 80) -> np.ndarray:
    """DESI FoG 形式在 no-RSD 情况下的 multipole 系数。"""
    k = np.asarray(k, dtype="f8")
    if float(sigmas) == 0.0:
        return np.ones_like(k) if int(ell) == 0 else np.zeros_like(k)
    mu, weight = np.polynomial.legendre.leggauss(int(nmu))
    x = 0.5 * (float(sigmas) * k[:, None] * mu[None, :]) ** 2
    fog = 1.0 / (1.0 + x) ** 2
    if int(ell) == 0:
        leg = np.ones_like(mu)
    elif int(ell) == 2:
        leg = 0.5 * (3.0 * mu**2 - 1.0)
    elif int(ell) == 4:
        leg = (35.0 * mu**4 - 30.0 * mu**2 + 3.0) / 8.0
    else:
        raise ValueError(f"unsupported ell={ell}")
    return (2 * int(ell) + 1.0) * 0.5 * np.sum(weight[None, :] * fog * leg[None, :], axis=1)


def theory_multipoles(
    k: np.ndarray,
    ell: np.ndarray,
    template: dict[str, np.ndarray],
    *,
    fnl_loc: float,
    b1: float,
    sigmas: float,
    sn0: float,
    p_fixed: float,
    sn0_scale: float,
    kmin_model: float,
    sigmas_policy: str,
) -> np.ndarray:
    """在 window 输入网格上计算 P_ell(k)，包含 desilike-style sn0 residual。"""
    k = np.asarray(k, dtype="f8")
    ell = np.asarray(ell, dtype="i8")
    out = np.zeros_like(k)
    valid = k >= float(kmin_model) - 1.0e-15
    if not np.any(valid):
        return out
    base = png_realspace_base(k[valid], template, fnl_loc=fnl_loc, b1=b1, p_fixed=p_fixed)
    for ell_value in (0, 2, 4):
        mask = valid.copy()
        mask[valid] &= ell[valid] == ell_value
        if not np.any(mask):
            continue
        if sigmas_policy == "fixed_zero":
            coeff = np.ones(np.count_nonzero(mask)) if ell_value == 0 else np.zeros(np.count_nonzero(mask))
        elif sigmas_policy == "fog_no_rsd":
            coeff = fog_multipole_coefficients(k[mask], float(sigmas), int(ell_value))
        else:
            raise ValueError(f"unknown sigmas_policy={sigmas_policy}")
        out[mask] = png_realspace_base(k[mask], template, fnl_loc=fnl_loc, b1=b1, p_fixed=p_fixed) * coeff
    out[(ell == 0) & valid] += float(sn0) * float(sn0_scale)
    return out


def model_pk(
    data: FitData,
    template: dict[str, np.ndarray],
    theta: np.ndarray,
    *,
    p_fixed: float,
    sn0_scale: float,
    subbox_window_shotnoise: bool,
    sigmas_policy: str,
) -> np.ndarray:
    """计算与 data.pk_data 同 convention 的模型。"""
    fnl_loc, b1, sigmas, sn0 = np.asarray(theta, dtype="f8")
    if data.mode.startswith("rawbox"):
        ell = np.zeros_like(data.rawbox_k, dtype="i8")
        theory = theory_multipoles(
            data.rawbox_k,
            ell,
            template,
            fnl_loc=fnl_loc,
            b1=b1,
            sigmas=sigmas,
            sn0=sn0,
            p_fixed=p_fixed,
            sn0_scale=sn0_scale,
            kmin_model=float(data.extra_meta.get("theory_kmin_model", data.extra_meta["kmin_model"])),
            sigmas_policy=sigmas_policy,
        )
        return data.rawbox_matrix @ theory

    theory = theory_multipoles(
        data.theory_k,
        data.theory_ell,
        template,
        fnl_loc=fnl_loc,
        b1=b1,
        sigmas=sigmas,
        sn0=sn0,
        p_fixed=p_fixed,
        sn0_scale=sn0_scale,
        kmin_model=float(data.extra_meta.get("theory_kmin_model", data.extra_meta["kmin_model"])),
        sigmas_policy=sigmas_policy,
    )
    if subbox_window_shotnoise:
        shot_in = np.zeros_like(theory)
        shot_in[data.theory_ell == 0] = float(data.shotnoise_scalar)
        return data.window_matrix @ (theory + shot_in) - np.asarray(data.shotnoise_out, dtype="f8")
    return data.window_matrix @ theory


def log_prior(theta: np.ndarray, priors: dict[str, tuple[float, float]], *, sigmas_policy: str) -> float:
    """均匀 prior。"""
    values = np.asarray(theta, dtype="f8")
    for i, name in enumerate(PARAM_NAMES):
        lo, hi = priors[name]
        if sigmas_policy == "fixed_zero" and name == "sigmas":
            if abs(values[i]) > 1.0e-12:
                return -np.inf
            continue
        if not (lo <= values[i] <= hi):
            return -np.inf
    return 0.0


def chi2(
    data: FitData,
    template: dict[str, np.ndarray],
    theta: np.ndarray,
    *,
    p_fixed: float,
    sn0_scale: float,
    subbox_window_shotnoise: bool,
    sigmas_policy: str,
) -> float:
    """计算 chi2。"""
    model = model_pk(
        data,
        template,
        theta,
        p_fixed=p_fixed,
        sn0_scale=sn0_scale,
        subbox_window_shotnoise=subbox_window_shotnoise,
        sigmas_policy=sigmas_policy,
    )
    diff = data.pk_data - model
    return float(diff @ data.precision @ diff)


def make_log_prob(
    data: FitData,
    template: dict[str, np.ndarray],
    *,
    p_fixed: float,
    sn0_scale: float,
    subbox_window_shotnoise: bool,
    sigmas_policy: str,
    priors: dict[str, tuple[float, float]],
) -> Callable[[np.ndarray], float]:
    """构建 emcee log-probability。"""

    def log_prob(theta: np.ndarray) -> float:
        lp = log_prior(theta, priors, sigmas_policy=sigmas_policy)
        if not np.isfinite(lp):
            return -np.inf
        return lp - 0.5 * chi2(
            data,
            template,
            theta,
            p_fixed=p_fixed,
            sn0_scale=sn0_scale,
            subbox_window_shotnoise=subbox_window_shotnoise,
            sigmas_policy=sigmas_policy,
        )

    return log_prob


def find_initial_point(
    data: FitData,
    template: dict[str, np.ndarray],
    *,
    p_fixed: float,
    sn0_scale: float,
    subbox_window_shotnoise: bool,
    sigmas_policy: str,
    priors: dict[str, tuple[float, float]],
) -> tuple[np.ndarray, dict[str, object]]:
    """用多起点 L-BFGS-B 找 walker 初始中心。"""
    from scipy.optimize import minimize

    starts = [
        np.array([100.0, 2.6, 0.0, 0.0]),
        np.array([100.0, 2.6, 5.0, 0.3]),
        np.array([80.0, 2.4, 2.0, 0.0]),
        np.array([120.0, 2.8, 10.0, 0.5]),
        np.array([50.0, 2.2, 0.0, -0.3]),
    ]
    if sigmas_policy == "fixed_zero":
        starts = [start.copy() for start in starts]
        for start in starts:
            start[2] = 0.0
        bounds = [priors["fnl_loc"], priors["b1"], (0.0, 1.0e-12), priors["sn0"]]
    else:
        bounds = [priors[name] for name in PARAM_NAMES]
    best = None
    for start in starts:
        result = minimize(
            lambda x: chi2(
                data,
                template,
                x,
                p_fixed=p_fixed,
                sn0_scale=sn0_scale,
                subbox_window_shotnoise=subbox_window_shotnoise,
                sigmas_policy=sigmas_policy,
            ),
            x0=start,
            method="L-BFGS-B",
            bounds=bounds,
            options={"maxiter": 3000, "ftol": 1.0e-10},
        )
        if best is None or float(result.fun) < float(best.fun):
            best = result
    assert best is not None
    return np.asarray(best.x, dtype="f8"), {
        "success": bool(best.success),
        "message": str(best.message),
        "chi2": float(best.fun),
        "point": {name: float(best.x[i]) for i, name in enumerate(PARAM_NAMES)},
    }


def initialize_walkers(center: np.ndarray, nwalkers: int, seed: int, priors: dict[str, tuple[float, float]], *, sigmas_policy: str) -> np.ndarray:
    """在 MLE 附近初始化 walkers。"""
    rng = np.random.default_rng(int(seed))
    scales = np.array([max(5.0, 0.03 * abs(center[0])), 0.025, 0.5, 0.04], dtype="f8")
    p0 = center[None, :] + rng.normal(scale=scales[None, :], size=(int(nwalkers), len(PARAM_NAMES)))
    for i, name in enumerate(PARAM_NAMES):
        if sigmas_policy == "fixed_zero" and name == "sigmas":
            p0[:, i] = 0.0
            continue
        lo, hi = priors[name]
        p0[:, i] = np.clip(p0[:, i], lo + 1.0e-8, hi - 1.0e-8)
    return p0


def summarize_samples(samples: np.ndarray, percival: float) -> dict[str, dict[str, float]]:
    """计算 posterior 分位数，并给出 Percival 修正后的误差。"""
    out: dict[str, dict[str, float]] = {}
    for i, name in enumerate(PARAM_NAMES):
        vals = np.asarray(samples[:, i], dtype="f8")
        q025, q16, q50, q84, q975 = np.quantile(vals, [0.025, 0.1586552539, 0.5, 0.8413447461, 0.975])
        err_low = float(q50 - q16)
        err_high = float(q84 - q50)
        out[name] = {
            "mean": float(np.mean(vals)),
            "std": float(np.std(vals, ddof=1)),
            "median": float(q50),
            "q025": float(q025),
            "q16": float(q16),
            "q84": float(q84),
            "q975": float(q975),
            "err_low": err_low,
            "err_high": err_high,
            "err_low_percival": float(err_low * percival),
            "err_high_percival": float(err_high * percival),
            "std_percival": float(np.std(vals, ddof=1) * percival),
        }
    return out


def autocorrelation_diagnostics(
    chain: np.ndarray,
    *,
    burnin: int,
    free_parameters: tuple[str, ...],
) -> dict[str, object]:
    """
    只对实际自由参数计算 emcee integrated autocorrelation time。

    ``chain`` 的轴依次是 step、walker、PARAM_NAMES；fixed ``sigmas`` 不参与
    tau 计算，避免全零维度产生 NaN。输出同时给出 post-burn steps/tau 和近似
    有效样本量，并以每个参数至少 50 tau 作为保守的链长检查。
    """
    from emcee.autocorr import integrated_time

    indices = [PARAM_NAMES.index(name) for name in free_parameters]
    postburn = np.asarray(chain[int(burnin) :, :, :], dtype="f8")[:, :, indices]
    try:
        tau = np.asarray(integrated_time(postburn, quiet=True), dtype="f8")
    except Exception as error:  # 审计信息不应让已经完成的科学链无法落盘。
        return {
            "status": "unavailable",
            "error": repr(error),
            "free_parameters": list(free_parameters),
        }
    nsteps = int(postburn.shape[0])
    nwalkers = int(postburn.shape[1])
    by_parameter = {
        name: {
            "tau_steps": float(value),
            "postburn_steps_per_tau": float(nsteps / value),
            "effective_samples_approx": float(nsteps * nwalkers / value),
            "passes_50_tau_gate": bool(nsteps >= 50.0 * value),
        }
        for name, value in zip(free_parameters, tau, strict=True)
    }
    return {
        "status": "pass" if all(item["passes_50_tau_gate"] for item in by_parameter.values()) else "short_chain",
        "postburn_steps": nsteps,
        "nwalkers": nwalkers,
        "parameters": by_parameter,
    }


def best_sample(
    data: FitData,
    template: dict[str, np.ndarray],
    samples: np.ndarray,
    logp: np.ndarray,
    *,
    p_fixed: float,
    sn0_scale: float,
    subbox_window_shotnoise: bool,
    sigmas_policy: str,
    nfree: int,
) -> dict[str, object]:
    """记录 post-burn 最大 posterior 样本。"""
    imax = int(np.nanargmax(logp))
    theta = np.asarray(samples[imax], dtype="f8")
    model = model_pk(
        data,
        template,
        theta,
        p_fixed=p_fixed,
        sn0_scale=sn0_scale,
        subbox_window_shotnoise=subbox_window_shotnoise,
        sigmas_policy=sigmas_policy,
    )
    diff = data.pk_data - model
    return {
        "log_prob": float(logp[imax]),
        "chi2": float(diff @ data.precision @ diff),
        "ndof": int(data.pk_data.size - int(nfree)),
        "point": {name: float(theta[i]) for i, name in enumerate(PARAM_NAMES)},
        "pk_model_best": model,
        "pk_residual_best": diff,
    }


def plot_fit(data: FitData, best: dict[str, object], path: Path) -> None:
    """画 P(k) 数据、best-fit 和 residual PDF。"""
    path.parent.mkdir(parents=True, exist_ok=True)
    model = np.asarray(best["pk_model_best"], dtype="f8")
    sigma = np.sqrt(np.diag(data.cov))
    fig, (ax, rx) = plt.subplots(2, 1, figsize=(8.6, 7.0), sharex=True, gridspec_kw={"height_ratios": [3, 1]})
    ax.errorbar(data.k_obs, data.pk_data, yerr=sigma, fmt="o", ms=4, color="black", capsize=2, label="measured mean")
    ax.plot(data.k_obs, model, color="#b04a2f", lw=2.0, label="best-fit model")
    ax.set_ylabel(r"$P_0(k)$")
    ax.set_title(f"Task4.2 P(k) {data.label} P(k) fit")
    ax.grid(color="0.9")
    ax.legend(frameon=False)
    rx.axhline(0.0, color="0.4", lw=1.0)
    rx.errorbar(data.k_obs, (data.pk_data - model) / sigma, yerr=np.ones_like(sigma), fmt="o", ms=4, color="black", capsize=2)
    rx.set_ylabel(r"$\Delta/\sigma$")
    rx.set_xlabel(r"$k\,[h/{\rm Mpc}]$")
    rx.grid(color="0.9")
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    """解析命令行参数。"""
    parser = argparse.ArgumentParser(description="Fit Task4.2 P(k) P(k) with window and shot-noise convention.")
    parser.add_argument("--payload", type=Path, required=True)
    parser.add_argument("--mode", choices=("rawbox", "subbox"), required=True)
    parser.add_argument("--label", default=None)
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_ROOT / "fits")
    parser.add_argument("--target-evals", type=int, default=150_000)
    parser.add_argument("--nwalkers", type=int, default=16)
    parser.add_argument("--burn-fraction", type=float, default=0.10)
    parser.add_argument("--seed", type=int, default=20260706)
    parser.add_argument("--p-fixed", type=float, default=P_FIXED)
    parser.add_argument(
        "--template-z",
        type=float,
        default=FASTPM_SNAPSHOT_Z,
        help="FastPM-L3 snapshot redshift；Task4.2 正式口径固定为 z=1。",
    )
    parser.add_argument(
        "--allow-template-z-override",
        action="store_true",
        help="仅供受控诊断：允许 template-z 偏离共享的 FastPM snapshot redshift。",
    )
    parser.add_argument("--sn0-scale", type=float, default=SN0_SCALE_DESILIKE)
    parser.add_argument("--fnl-prior", default="-500,500")
    parser.add_argument("--b1-prior", default="0.5,5")
    parser.add_argument("--sigmas-prior", default="0,30")
    parser.add_argument("--sn0-prior", default="-1,1")
    parser.add_argument("--sigmas-policy", choices=("fixed_zero", "fog_no_rsd"), default="fixed_zero")
    parser.add_argument("--theory-kmin-model", type=float, default=None, help="override subbox window theory-grid low-k cutoff without rebuilding payload")
    parser.add_argument("--no-subbox-window-shotnoise", action="store_true")
    parser.add_argument("--quiet", action="store_true", help="关闭 emcee tqdm 进度条，适合长链后台运行。")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> None:
    """主入口。"""
    import emcee

    args = parse_args()
    if int(args.nwalkers) < 2 * len(PARAM_NAMES):
        raise ValueError("nwalkers 必须 >= 2*ndim")
    label = args.label or (args.mode if args.mode == "rawbox" else Path(args.payload).stem)
    out_dir = Path(args.output_dir) / label
    summary_path = out_dir / f"task4p2_{label}_pk_windowed_fit_summary.json"
    samples_path = out_dir / f"task4p2_{label}_pk_windowed_fit_samples.npz"
    plot_path = out_dir / f"task4p2_{label}_pk_windowed_fit.pdf"
    if summary_path.exists() and not args.overwrite:
        print(f"[skip] {summary_path}")
        return

    priors = {
        "fnl_loc": parse_prior(args.fnl_prior),
        "b1": parse_prior(args.b1_prior),
        "sigmas": parse_prior(args.sigmas_prior),
        "sn0": parse_prior(args.sn0_prior),
    }
    sigmas_policy = str(args.sigmas_policy)
    free_params = free_parameter_names(sigmas_policy)
    fixed_params = {"sigmas": 0.0} if sigmas_policy == "fixed_zero" else {}
    if not np.isclose(float(args.template_z), float(FASTPM_SNAPSHOT_Z), rtol=0.0, atol=1.0e-12):
        if not bool(args.allow_template_z_override):
            raise ValueError(
                "Task4.2 FastPM-L3 catalog 是 z=1 snapshot；若确实要做非正式红移诊断，"
                "请同时传入 --allow-template-z-override。"
            )
    k_template = np.logspace(np.log10(1.0e-5), np.log10(20.0), 20000)
    template = build_template_arrays(k_template, z=float(args.template_z))
    if args.mode == "rawbox":
        data = load_rawbox_fit_data(Path(args.payload), nparams=len(free_params))
    else:
        data = load_subbox_fit_data(Path(args.payload), nparams=len(free_params))
        if args.theory_kmin_model is not None:
            data.extra_meta["theory_kmin_model"] = float(args.theory_kmin_model)
    data.label = label
    subbox_window_shotnoise = (args.mode == "subbox") and not bool(args.no_subbox_window_shotnoise)

    center, opt_meta = find_initial_point(
        data,
        template,
        p_fixed=float(args.p_fixed),
        sn0_scale=float(args.sn0_scale),
        subbox_window_shotnoise=subbox_window_shotnoise,
        sigmas_policy=sigmas_policy,
        priors=priors,
    )
    nsteps = int(math.ceil(float(args.target_evals) / float(args.nwalkers)))
    burnin = int(math.floor(float(args.burn_fraction) * float(nsteps)))
    p0 = initialize_walkers(center, int(args.nwalkers), int(args.seed), priors, sigmas_policy=sigmas_policy)
    log_prob = make_log_prob(
        data,
        template,
        p_fixed=float(args.p_fixed),
        sn0_scale=float(args.sn0_scale),
        subbox_window_shotnoise=subbox_window_shotnoise,
        sigmas_policy=sigmas_policy,
        priors=priors,
    )
    # emcee 的 proposal RNG 默认读取 numpy 全局随机态；除了 walker 初始化外也在
    # 这里显式设种子，使相同输入与命令真正可以逐样本复现。
    np.random.seed(int(args.seed))
    sampler = emcee.EnsembleSampler(int(args.nwalkers), len(PARAM_NAMES), log_prob)
    t0 = time.time()
    sampler.run_mcmc(p0, nsteps, progress=not bool(args.quiet), skip_initial_state_check=True)
    elapsed = time.time() - t0
    chain = sampler.get_chain()
    logp = sampler.get_log_prob()
    burnin = min(burnin, max(0, chain.shape[0] - 1))
    flat_samples = chain[burnin:, :, :].reshape((-1, len(PARAM_NAMES)))
    flat_logp = logp[burnin:, :].reshape((-1,))
    autocorr = autocorrelation_diagnostics(chain, burnin=burnin, free_parameters=free_params)
    percival = float(data.covariance_meta["percival_error_factor"])
    params = summarize_samples(flat_samples, percival)
    best = best_sample(
        data,
        template,
        flat_samples,
        flat_logp,
        p_fixed=float(args.p_fixed),
        sn0_scale=float(args.sn0_scale),
        subbox_window_shotnoise=subbox_window_shotnoise,
        sigmas_policy=sigmas_policy,
        nfree=len(free_params),
    )

    atomic_savez(
        samples_path,
        param_names=np.asarray(PARAM_NAMES),
        samples=flat_samples,
        log_prob=flat_logp,
        k_obs=data.k_obs,
        pk_data=data.pk_data,
        pk_cov=data.cov,
        pk_model_best=np.asarray(best["pk_model_best"], dtype="f8"),
        burnin=np.asarray(burnin),
        nsteps=np.asarray(chain.shape[0]),
        nwalkers=np.asarray(args.nwalkers),
    )
    plot_fit(data, best, plot_path)

    summary = {
        "task": "task4p2_fit_pk_windowed_model",
        "status": "done",
        "label": label,
        "mode": data.mode,
        "payload": str(data.source_payload),
        "config": {
            "parameter_names": list(PARAM_NAMES),
            "free_parameters": list(free_params),
            "fixed_parameters": fixed_params,
            "p_fixed": float(args.p_fixed),
            "sigmas_policy": sigmas_policy,
            "sn0_scale": float(args.sn0_scale),
            "sn0_units_note": "model adds sn0 * sn0_scale to P0, matching desilike default shotnoise=1e4 convention",
            "subbox_window_shotnoise": bool(subbox_window_shotnoise),
            "subbox_window_shotnoise_formula": "W(P_theory + P_poisson) - P_poisson" if subbox_window_shotnoise else "W(P_theory)",
            "priors": {key: list(value) for key, value in priors.items()},
            "target_evals": int(args.target_evals),
            "seed": int(args.seed),
            "nwalkers": int(args.nwalkers),
            "nsteps": int(chain.shape[0]),
            "burnin": int(burnin),
            "template_z": float(args.template_z),
            "fastpm_snapshot_z": float(FASTPM_SNAPSHOT_Z),
            "template_z_override_enabled": bool(args.allow_template_z_override),
            "quiet": bool(args.quiet),
            "template_z_provenance": (
                "FastPM-L3 single snapshot evolved from z=99 to z=1; the z=0.4--1.0 "
                "directory tag describes a later CUTSKY shell, not the simulation epoch"
            ),
        },
        "parameters": params,
        "maximum_posterior_sample": best,
        "optimizer_initial_center": opt_meta,
        "covariance": data.covariance_meta,
        "data": {
            "ndata": int(data.pk_data.size),
            "k_min": float(np.min(data.k_obs)),
            "k_max": float(np.max(data.k_obs)),
        },
        "extra_meta": data.extra_meta,
        "diagnostics": {
            "acceptance_fraction_mean": float(np.mean(sampler.acceptance_fraction)),
            "acceptance_fraction_min": float(np.min(sampler.acceptance_fraction)),
            "acceptance_fraction_max": float(np.max(sampler.acceptance_fraction)),
            "elapsed_sec": float(elapsed),
            "autocorrelation": autocorr,
        },
        "paths": {
            "summary_json": str(summary_path),
            "samples_npz": str(samples_path),
            "fit_pdf": str(plot_path),
        },
    }
    write_json(summary_path, to_jsonable(summary))
    print(f"[write] {samples_path}")
    print(f"[write] {plot_path}")
    print(f"[write] {summary_path}")


if __name__ == "__main__":
    main()
