#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Task4.1 Quijote fNL=50：重新运行 P(k) 与 2PCF r=50--350 的长链 emcee。

代码大纲：
1. 复用 Task45 原始数据、sample covariance、Hartlap precision、BinAvgFit 和
   FullDiscrete likelihood 构造，严格保持主图的 free-sn0 科学口径。
2. 从 desilike 原模型中提取已经计算好的 P_dd、alpha、f、mu quadrature，并缓存
   成快速 RSD monopole evaluator；启动 MCMC 前逐 case 与原始 desilike likelihood
   做数值 A/B，要求 ``max |Delta logL| < 1e-6``。
3. 对 LCp50 的两个 case 分别运行 emcee：
   - P(k): kmax_fit = 0.10 h/Mpc；
   - 2PCF: bin edges 50--350 Mpc/h，实际 centers 55--345 Mpc/h。
4. 每条链默认 64 walkers、至少 8000 steps、burn-in 2000；每 1000 steps 检查
   autocorrelation time 和 split-Rhat，不满足收敛 gate 时自动延长到最多 20000 steps。
5. 保存 HDF backend、post-burn samples、单项 summary 和 combined manifest。
6. 用 Task4.3 ``task43_pk_vs_2pcf_s50_350_both_jaxpower_covariance.pdf``
   的 PPT 风格，绘制 ``fNL_loc, b1`` 两参数 PDF：P(k) 深炭灰、2PCF 红色，
   左上角写彩色 fNL 约束；sigmas 和 sn0 均保持自由并在成图时边缘化。

资源约束：
- 这是 CPU-only 任务，直接在登录节点运行；
- 2PCF likelihood 可使用最多 8 个 multiprocessing workers；
- 每个 worker 的 BLAS/OpenMP/JAX 线程数固定为 1。
"""

from __future__ import annotations

import argparse
import json
import math
import multiprocessing as mp
import os
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

# 必须在 numpy/JAX/desilike import 前限制数值库线程，避免 8 个 worker 各自再开线程。
for _name in (
    "OMP_NUM_THREADS",
    "MKL_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
    "JAX_NUM_THREADS",
):
    os.environ.setdefault(_name, "1")
os.environ.setdefault("XLA_FLAGS", "--xla_cpu_multi_thread_eigen=false intra_op_parallelism_threads=1")
os.environ.setdefault("JAX_PLATFORMS", "cpu")
os.environ.setdefault("JAX_PLATFORM_NAME", "cpu")
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
os.environ.setdefault("HDF5_USE_FILE_LOCKING", "FALSE")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


PROJECT_ROOT = Path("/pscratch/sd/l/lzy/PNG-2PCF-model-to-observe")
TASK4_DIR = PROJECT_ROOT / "codes/task4"
TASK4_ARCHIVE_DIR = PROJECT_ROOT / "old_doc_codes/task4_task44_cleanup_20260707T061844Z/moved/codes/task4"
if str(TASK4_ARCHIVE_DIR) not in sys.path:
    sys.path.insert(0, str(TASK4_ARCHIVE_DIR))
if str(TASK4_DIR) not in sys.path:
    sys.path.insert(0, str(TASK4_DIR))

# task45_quijote_ultranest.py 仍是 active likelihood 入口；它依赖的 task42 helper
# 已在仓库清理时归档，因此上面同时加入明确的 archive code path。
import task45_quijote_ultranest as task45  # noqa: E402


TAG = "LCp50"
CASE_ORDER = ("pk_binavg", "xi_r50_350")
PARAM_NAMES = ("fnl_loc", "b1", "sigmas", "sn0")
PLOT_PARAM_NAMES = ("fnl_loc", "b1")
PRIORS = {
    "fnl_loc": (-500.0, 500.0),
    "b1": (0.5, 5.0),
    "sigmas": (0.0, 30.0),
    "sn0": (-1.0, 1.0),
}
P_FIXED = 1.2
SN0_SCALE = 1.0e4
DELTA_C = 1.686

OUTPUT_ROOT = PROJECT_ROOT / "outputs/task4_outputs/quijote_emcee_free_sn0_fnl50_pk_xi_r50"
PLOT_ROOT = PROJECT_ROOT / "plots/task4/important_4p1_4p2/contours"
OUT_PDF = PLOT_ROOT / "4p1_fnl50_pk_kmax0p10_vs_2pcf_r50_350_contour.pdf"
OUT_MANIFEST = OUTPUT_ROOT / "task45_fnl50_pk_xi_r50_emcee_manifest.json"

PPT_COLORS = {"pk": "#2F2F2F", "twopcf": "#C44E52"}
PPT_LEGEND_FONTSIZE = 17.0
PPT_FNL_FONTSIZE = 14.0
PPT_FNL_HEADROOM = 1.38
PPT_LEGEND_GAP = 0.018
GETDIST_SMOOTH_1D = 0.35
GETDIST_SMOOTH_2D = 0.40

OLD_PK_SAMPLES = (
    PROJECT_ROOT
    / "old_doc_codes/task4_task44_cleanup_20260707T061844Z/moved/outputs/task4_outputs"
    / "quijote_ultranest_free_sn0_kmax0p10_30k/samples"
    / "task45_free_sn0_kmax0p10_30k_LCp50_pk_binavg_weighted_samples.npz"
)
OLD_XI_SAMPLES = (
    PROJECT_ROOT
    / "outputs/task4_outputs/quijote_ultranest_free_sn0_30k/samples"
    / "task45_free_sn0_30k_LCp50_xi_r50_350_weighted_samples.npz"
)
OLD_SAMPLE_PATHS = {"pk_binavg": OLD_PK_SAMPLES, "xi_r50_350": OLD_XI_SAMPLES}


@dataclass
class FastLikelihood:
    """
    一个 case 的快速、与原 desilike 数值等价的 likelihood 缓存。

    参数：
    - case: ``pk_binavg`` 或 ``xi_r50_350``；
    - data/precision: Task45 测量向量和 Hartlap precision；
    - k/alpha/pk_dd/f_growth/mu/mu_weights: 原 desilike RSD monopole 所需缓存；
    - projection: 把 P0(k) 投影到观测 P(k) bins 或 FullDiscrete xi bins 的矩阵。
    """

    case: str
    data: np.ndarray
    precision: np.ndarray
    k: np.ndarray
    alpha: np.ndarray
    pk_dd: np.ndarray
    f_growth: float
    mu: np.ndarray
    mu_weights: np.ndarray
    projection: np.ndarray
    covariance_meta: dict[str, Any]

    def pk0(self, theta: np.ndarray) -> np.ndarray:
        """计算与 Task45 desilike ``PNGTracerPowerSpectrumMultipoles`` 一致的 P0(k)。"""
        fnl_loc, b1, sigmas, sn0 = np.asarray(theta, dtype="f8")
        bphi = 2.0 * DELTA_C * (float(b1) - P_FIXED)
        amplitude = float(b1) + float(fnl_loc) * bphi * self.alpha
        mu2 = self.mu * self.mu
        fog = 1.0 / (1.0 + 0.5 * (self.k[:, None] * self.mu[None, :] * float(sigmas)) ** 2) ** 2
        pk_mu = (
            self.pk_dd[:, None]
            * (amplitude[:, None] + self.f_growth * mu2[None, :]) ** 2
            * fog
            + float(sn0) * SN0_SCALE
        )
        return np.sum(pk_mu * self.mu_weights[None, :], axis=1)

    def model_vector(self, theta: np.ndarray) -> np.ndarray:
        """把 P0(k) 投影成当前 case 的最终理论数据向量。"""
        return self.projection @ self.pk0(theta)

    def log_likelihood(self, theta: np.ndarray) -> float:
        """返回 ``-chi2/2``。"""
        residual = self.data - self.model_vector(theta)
        return -0.5 * float(residual @ self.precision @ residual)


# multiprocessing worker 通过 fork 继承这个只读缓存，避免每次 pickle 大矩阵。
ACTIVE_LIKELIHOOD: FastLikelihood | None = None


def to_jsonable(value: Any) -> Any:
    """递归转换 numpy/Path 类型，便于写 JSON。"""
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, dict):
        return {str(key): to_jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [to_jsonable(item) for item in value]
    return str(value)


def write_json(path: Path, payload: dict[str, Any]) -> None:
    """写 JSON，并确保父目录存在。"""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(to_jsonable(payload), indent=2, ensure_ascii=False), encoding="utf-8")


def log_prior(theta: np.ndarray) -> float:
    """四参数均匀先验；区间与 Task45 free-sn0 主图完全一致。"""
    values = np.asarray(theta, dtype="f8")
    for index, name in enumerate(PARAM_NAMES):
        lo, hi = PRIORS[name]
        if not (lo <= values[index] <= hi):
            return -np.inf
    return 0.0


def log_prob_worker(theta: np.ndarray) -> float:
    """emcee/multiprocessing 使用的顶层 log-probability 函数。"""
    prior = log_prior(theta)
    if not np.isfinite(prior):
        return -np.inf
    if ACTIVE_LIKELIHOOD is None:
        raise RuntimeError("ACTIVE_LIKELIHOOD 尚未初始化")
    return prior + ACTIVE_LIKELIHOOD.log_likelihood(theta)


def configure_task45() -> None:
    """把 Task45 全局配置固定到主图的 free-sn0、P(k) kmax=0.10 口径。"""
    namespace = argparse.Namespace(
        kmax_fit=0.10,
        output_label="emcee_fnl50_internal",
        free_sn0=True,
        sn0_prior=(-1.0, 1.0),
    )
    task45.configure_run(namespace)


def build_projection(context: task45.LikelihoodContext) -> np.ndarray:
    """
    构建线性投影矩阵。

    P(k) case 对每个离散壳层按 degeneracy 做 bin average；2PCF case 使用
    ``(g_cache * P0) @ j0_kernel / V`` 的 FullDiscrete 变换。
    """
    nk = int(np.asarray(context.model.theory.k).size)
    if context.case == "pk_binavg":
        matrix = np.zeros((context.data.size, nk), dtype="f8")
        g_shell = np.asarray(context.extra["g_shell"], dtype="f8")
        for ibin, indices in enumerate(list(context.extra["indices_object"])):
            indices = np.asarray(indices, dtype="i8")
            weights = g_shell[indices]
            matrix[ibin, indices] = weights / np.sum(weights)
        return matrix
    g_cache = np.asarray(context.extra["g_cache"], dtype="f8")
    kernel = np.asarray(context.extra["kernel"], dtype="f8")
    return (g_cache[:, None] * kernel / float(task45.VOLUME)).T


def build_fast_likelihood(context: task45.LikelihoodContext) -> FastLikelihood:
    """
    从原 Task45/desilike model 提取缓存，构建快速 likelihood。

    关键细节：必须使用 ``desilike.jax.interp1d``，因为其边界和插值实现与原
    PNG theory 完全相同；普通 ``numpy.interp`` 在 xi 高-k cancellation 下会造成
    可见的 Delta-logL，因此不能替代。
    """
    from desilike.jax import interp1d
    import jax.numpy as jnp

    # 先执行一次原模型，触发 template/cosmology 的 lazy calculation。
    context.model.pk0(fnl_loc=50.0, b1=2.6, sigmas=8.0, sn0=0.0)
    theory = context.model.theory
    template = theory.template
    k = np.asarray(theory.k, dtype="f8")
    kin = np.asarray(template.k, dtype="f8")
    pk_dd = np.asarray(template.pk_dd, dtype="f8")
    cosmo = template.cosmo
    pk_prim = np.asarray(cosmo.get_primordial(mode="scalar").pk_interpolator()(kin), dtype="f8")
    pphi_prim = 9.0 / 25.0 * 2.0 * np.pi**2 / kin**3 * pk_prim / cosmo.h**3
    alpha = 1.0 / np.sqrt(pk_dd / pphi_prim)
    # 原 desilike 实现会删除用于 transfer normalization 的第一个 k 点。
    kin = kin[1:]
    pk_dd = pk_dd[1:]
    alpha = alpha[1:]
    logk = jnp.log10(k)
    pk_eval = np.asarray(interp1d(logk, np.log10(kin), pk_dd), dtype="f8")
    alpha_eval = np.asarray(interp1d(logk, np.log10(kin), alpha), dtype="f8")
    return FastLikelihood(
        case=str(context.case),
        data=np.asarray(context.data, dtype="f8"),
        precision=np.asarray(context.precision, dtype="f8"),
        k=k,
        alpha=alpha_eval,
        pk_dd=pk_eval,
        f_growth=float(template.f),
        mu=np.asarray(theory.mu, dtype="f8"),
        mu_weights=np.asarray(theory.wmu[0], dtype="f8"),
        projection=build_projection(context),
        covariance_meta=dict(context.corrections),
    )


def load_old_weighted_samples(case: str) -> dict[str, np.ndarray]:
    """读取旧 UltraNest weighted samples，仅用于优化起点和 walker 初始化。"""
    path = OLD_SAMPLE_PATHS[case]
    if not path.exists():
        raise FileNotFoundError(path)
    with np.load(path, allow_pickle=False) as data:
        names = [str(name) for name in np.asarray(data["param_names"])]
        points_all = np.asarray(data["points"], dtype="f8")
        weights = np.asarray(data["weights"], dtype="f8")
        ml = np.asarray(data["maximum_likelihood_point"], dtype="f8")
    indices = [names.index(name) for name in PARAM_NAMES]
    points = points_all[:, indices]
    ml = ml[indices]
    mask = np.all(np.isfinite(points), axis=1) & np.isfinite(weights) & (weights > 0.0)
    points = points[mask]
    weights = weights[mask]
    weights /= np.sum(weights)
    return {"path": np.asarray(str(path)), "points": points, "weights": weights, "ml": ml}


def optimize_start(likelihood: FastLikelihood, old: dict[str, np.ndarray]) -> tuple[np.ndarray, dict[str, Any]]:
    """从旧最大似然点出发，用当前快速 likelihood 做一次 bounded L-BFGS-B。"""
    from scipy.optimize import minimize

    bounds = [PRIORS[name] for name in PARAM_NAMES]
    starts = [
        np.asarray(old["ml"], dtype="f8"),
        np.asarray([50.0, 2.6, 8.0, 0.0], dtype="f8"),
    ]
    best = None
    for start in starts:
        result = minimize(
            lambda values: -likelihood.log_likelihood(values),
            x0=np.asarray(start, dtype="f8"),
            method="L-BFGS-B",
            bounds=bounds,
            options={"maxiter": 2500, "ftol": 1.0e-12},
        )
        if best is None or float(result.fun) < float(best.fun):
            best = result
    assert best is not None
    return np.asarray(best.x, dtype="f8"), {
        "success": bool(best.success),
        "message": str(best.message),
        "log_likelihood": float(-best.fun),
        "point": {name: float(best.x[index]) for index, name in enumerate(PARAM_NAMES)},
    }


def initialize_walkers(
    old: dict[str, np.ndarray],
    optimum: np.ndarray,
    *,
    nwalkers: int,
    seed: int,
) -> np.ndarray:
    """
    用旧 weighted posterior 的全局形状初始化 walkers，并加入小 jitter 防止重复点。

    旧 posterior 只用于初始化，不进入新链的 posterior 权重或最终成图。
    """
    rng = np.random.default_rng(int(seed))
    points = np.asarray(old["points"], dtype="f8")
    weights = np.asarray(old["weights"], dtype="f8")
    choice = rng.choice(points.shape[0], size=int(nwalkers), replace=True, p=weights)
    walkers = points[choice].copy()
    posterior_std = np.std(points, axis=0, ddof=1)
    jitter = np.maximum(0.005 * posterior_std, np.asarray([0.1, 0.001, 0.01, 1.0e-6]))
    walkers += rng.normal(scale=jitter[None, :], size=walkers.shape)
    # 让 1/8 walkers 明确覆盖最优点附近，兼顾旧 weighted posterior 的宽尾和主峰。
    nlocal = max(2, int(nwalkers) // 8)
    walkers[:nlocal] = optimum[None, :] + rng.normal(scale=jitter[None, :], size=(nlocal, len(PARAM_NAMES)))
    for index, name in enumerate(PARAM_NAMES):
        lo, hi = PRIORS[name]
        walkers[:, index] = np.clip(walkers[:, index], lo + 1.0e-8, hi - 1.0e-8)
    return walkers


def equivalence_audit(
    context: task45.LikelihoodContext,
    likelihood: FastLikelihood,
    old: dict[str, np.ndarray],
) -> dict[str, Any]:
    """比较原始 desilike likelihood 与快速缓存 likelihood，失败则禁止跑链。"""
    slow = task45.loglike_from_context(context)
    rng = np.random.default_rng(20260711)
    points = [np.asarray(old["ml"], dtype="f8"), np.asarray([50.0, 2.6, 8.0, 0.0], dtype="f8")]
    old_points = np.asarray(old["points"], dtype="f8")
    old_weights = np.asarray(old["weights"], dtype="f8")
    for index in rng.choice(old_points.shape[0], size=4, replace=False, p=old_weights):
        points.append(old_points[int(index)])
    rows = []
    for point in points:
        slow_logl = float(slow(point))
        fast_logl = float(likelihood.log_likelihood(point))
        rows.append(
            {
                "point": {name: float(point[i]) for i, name in enumerate(PARAM_NAMES)},
                "slow_logl": slow_logl,
                "fast_logl": fast_logl,
                "delta_logl": fast_logl - slow_logl,
            }
        )
    maximum = max(abs(float(row["delta_logl"])) for row in rows)
    passed = bool(maximum < 1.0e-6)
    audit = {"passed": passed, "threshold_abs_delta_logl": 1.0e-6, "max_abs_delta_logl": maximum, "rows": rows}
    if not passed:
        raise RuntimeError(f"fast/slow likelihood equivalence audit failed: {audit}")
    return audit


def split_rhat(chain: np.ndarray) -> np.ndarray:
    """
    计算 split-Rhat。

    输入 chain shape 为 ``(nstep, nwalker, nparam)``；将每个 walker 的前后半段
    当作两条链，返回每个参数的 Gelman-Rubin Rhat。
    """
    chain = np.asarray(chain, dtype="f8")
    nstep, nwalker, nparam = chain.shape
    half = nstep // 2
    if half < 20:
        return np.full(nparam, np.inf)
    first = chain[:half].transpose(1, 0, 2)
    second = chain[-half:].transpose(1, 0, 2)
    split = np.concatenate([first, second], axis=0)
    chain_means = np.mean(split, axis=1)
    chain_variances = np.var(split, axis=1, ddof=1)
    between = half * np.var(chain_means, axis=0, ddof=1)
    within = np.mean(chain_variances, axis=0)
    var_hat = (half - 1.0) / half * within + between / half
    return np.sqrt(var_hat / within)


def convergence_diagnostics(backend: Any, burnin: int) -> dict[str, Any]:
    """从 emcee HDF backend 计算 tau、steps/tau、split-Rhat 和 acceptance gate。"""
    iteration = int(backend.iteration)
    chain = np.asarray(backend.get_chain(discard=min(int(burnin), max(0, iteration - 1))), dtype="f8")
    try:
        tau = np.asarray(backend.get_autocorr_time(discard=min(int(burnin), max(0, iteration - 1)), tol=0), dtype="f8")
        tau_error = None
    except Exception as exc:
        tau = np.full(len(PARAM_NAMES), np.nan)
        tau_error = repr(exc)
    rhat = split_rhat(chain)
    steps_after_burn = int(chain.shape[0])
    ratios = steps_after_burn / tau if np.all(np.isfinite(tau)) else np.full(len(PARAM_NAMES), np.nan)
    converged = bool(
        np.all(np.isfinite(tau))
        and np.all(ratios >= 50.0)
        and np.all(np.isfinite(rhat))
        and np.all(rhat < 1.02)
    )
    return {
        "iteration": iteration,
        "steps_after_burn": steps_after_burn,
        "autocorr_time": tau,
        "autocorr_error": tau_error,
        "steps_after_burn_over_tau": ratios,
        "split_rhat": rhat,
        "gates": {"min_steps_over_tau": 50.0, "max_split_rhat": 1.02},
        "converged": converged,
    }


def summarize_samples(samples: np.ndarray, percival: float) -> dict[str, dict[str, float]]:
    """汇总等权 post-burn MCMC 样本，并同时记录 Percival-scaled 误差。"""
    output: dict[str, dict[str, float]] = {}
    for index, name in enumerate(PARAM_NAMES):
        values = np.asarray(samples[:, index], dtype="f8")
        q025, q16, q50, q84, q975 = np.quantile(values, [0.025, 0.1586552539, 0.5, 0.8413447461, 0.975])
        err_low = float(q50 - q16)
        err_high = float(q84 - q50)
        output[name] = {
            "mean": float(np.mean(values)),
            "std": float(np.std(values, ddof=1)),
            "median": float(q50),
            "q025": float(q025),
            "q16": float(q16),
            "q84": float(q84),
            "q975": float(q975),
            "err_low": err_low,
            "err_high": err_high,
            "err_low_percival": float(err_low * percival),
            "err_high_percival": float(err_high * percival),
        }
    return output


def run_case(
    case: str,
    context: task45.LikelihoodContext,
    likelihood: FastLikelihood,
    old: dict[str, np.ndarray],
    *,
    nwalkers: int,
    min_steps: int,
    max_steps: int,
    burnin: int,
    chunk_steps: int,
    nworkers: int,
    seed: int,
    overwrite: bool,
) -> dict[str, Any]:
    """运行一个可续跑、带自动收敛检查的 emcee case。"""
    import emcee

    global ACTIVE_LIKELIHOOD
    ACTIVE_LIKELIHOOD = likelihood
    case_dir = OUTPUT_ROOT / case
    case_dir.mkdir(parents=True, exist_ok=True)
    backend_path = case_dir / "chain.h5"
    summary_path = case_dir / "summary.json"
    samples_path = case_dir / "postburn_samples.npz"
    optimum, optimizer_meta = optimize_start(likelihood, old)
    backend = emcee.backends.HDFBackend(str(backend_path))
    if overwrite or not backend_path.exists():
        backend.reset(int(nwalkers), len(PARAM_NAMES))
        initial_state = initialize_walkers(old, optimum, nwalkers=nwalkers, seed=seed)
    else:
        if backend.shape != (int(nwalkers), len(PARAM_NAMES)):
            raise ValueError(f"backend shape={backend.shape}, expected={(nwalkers, len(PARAM_NAMES))}")
        initial_state = None

    processes = 1 if case == "pk_binavg" else int(nworkers)
    if processes < 1 or processes > 8:
        raise ValueError(f"nworkers must be 1..8, got {processes}")
    t0 = time.time()
    pool_context = mp.get_context("fork")
    pool = pool_context.Pool(processes=processes) if processes > 1 else None
    try:
        sampler = emcee.EnsembleSampler(
            int(nwalkers),
            len(PARAM_NAMES),
            log_prob_worker,
            pool=pool,
            backend=backend,
        )
        state = initial_state
        diagnostics: dict[str, Any] = {}
        while int(backend.iteration) < int(max_steps):
            remaining = int(max_steps) - int(backend.iteration)
            run_steps = min(int(chunk_steps), remaining)
            state = sampler.run_mcmc(state, run_steps, progress=False, skip_initial_state_check=True)
            diagnostics = convergence_diagnostics(backend, burnin=burnin)
            acceptance = np.asarray(sampler.acceptance_fraction, dtype="f8")
            print(
                f"[progress] case={case} steps={backend.iteration} "
                f"accept={np.mean(acceptance):.3f} "
                f"tau={np.array2string(np.asarray(diagnostics['autocorr_time']), precision=1)} "
                f"ratio={np.array2string(np.asarray(diagnostics['steps_after_burn_over_tau']), precision=1)} "
                f"rhat={np.array2string(np.asarray(diagnostics['split_rhat']), precision=4)} "
                f"converged={diagnostics['converged']}",
                flush=True,
            )
            if int(backend.iteration) >= int(min_steps) and bool(diagnostics["converged"]):
                break
            state = None  # 后续 chunk 从 backend 的最后状态继续。
    finally:
        if pool is not None:
            pool.close()
            pool.join()
    elapsed = time.time() - t0

    discard = min(int(burnin), max(0, int(backend.iteration) - 1))
    flat_samples = np.asarray(backend.get_chain(discard=discard, flat=True), dtype="f8")
    flat_log_prob = np.asarray(backend.get_log_prob(discard=discard, flat=True), dtype="f8")
    diagnostics = convergence_diagnostics(backend, burnin=burnin)
    # 重新打开一个无 pool sampler 只用于读取逐 walker acceptance fraction。
    acceptance = np.asarray(
        emcee.EnsembleSampler(int(nwalkers), len(PARAM_NAMES), log_prob_worker, backend=backend).acceptance_fraction,
        dtype="f8",
    )
    imax = int(np.nanargmax(flat_log_prob))
    best_point = flat_samples[imax]
    best_model = likelihood.model_vector(best_point)
    best_residual = likelihood.data - best_model
    percival = float(likelihood.covariance_meta["percival_error_factor"])
    np.savez_compressed(
        samples_path,
        param_names=np.asarray(PARAM_NAMES),
        samples=flat_samples,
        log_prob=flat_log_prob,
        burnin=np.asarray(discard),
        nwalkers=np.asarray(nwalkers),
        nsteps=np.asarray(backend.iteration),
        acceptance_fraction=acceptance,
        autocorr_time=np.asarray(diagnostics["autocorr_time"]),
        split_rhat=np.asarray(diagnostics["split_rhat"]),
        model_map=best_model,
        residual_map=best_residual,
    )
    summary = {
        "task": "task45_fnl50_pk_xi_r50_emcee",
        "status": "done" if bool(diagnostics["converged"]) else "max_steps_without_convergence",
        "tag": TAG,
        "case": case,
        "display_fit_range": "kmax=0.10 h/Mpc" if case == "pk_binavg" else "r_edges=50-350 Mpc/h",
        "actual_xi_centers": None if case == "pk_binavg" else [55.0, 345.0],
        "parameters": summarize_samples(flat_samples, percival=percival),
        "maximum_posterior_sample": {
            "log_prob": float(flat_log_prob[imax]),
            "chi2": float(best_residual @ likelihood.precision @ best_residual),
            "point": {name: float(best_point[index]) for index, name in enumerate(PARAM_NAMES)},
        },
        "optimizer": optimizer_meta,
        "mcmc": {
            "sampler": "emcee.EnsembleSampler",
            "nwalkers": int(nwalkers),
            "nsteps": int(backend.iteration),
            "burnin": int(discard),
            "postburn_samples": int(flat_samples.shape[0]),
            "nworkers": int(processes),
            "mean_acceptance_fraction": float(np.mean(acceptance)),
            "min_acceptance_fraction": float(np.min(acceptance)),
            "max_acceptance_fraction": float(np.max(acceptance)),
            "diagnostics": diagnostics,
            "elapsed_sec": float(elapsed),
        },
        "covariance": likelihood.covariance_meta,
        "priors": PRIORS,
        "fixed": {"p": P_FIXED},
        "sn0_policy": "free",
        "initialization": {
            "old_weighted_samples": str(np.asarray(old["path"]).item()),
            "policy": "weighted-resample old UltraNest posterior plus 1-percent jitter; final posterior uses only new emcee chain",
        },
        "paths": {"backend_hdf5": backend_path, "postburn_samples": samples_path, "summary": summary_path},
    }
    write_json(summary_path, summary)
    print(f"[write] {summary_path}", flush=True)
    return summary


def load_postburn_samples(summary: dict[str, Any]) -> np.ndarray:
    """读取一个新 emcee case 的 post-burn 等权 samples。"""
    with np.load(summary["paths"]["postburn_samples"], allow_pickle=False) as data:
        names = [str(name) for name in np.asarray(data["param_names"])]
        samples = np.asarray(data["samples"], dtype="f8")
    indices = [names.index(name) for name in PLOT_PARAM_NAMES]
    return samples[:, indices]


def constraint_text(samples: np.ndarray) -> str:
    """把 posterior 的 16/50/84 分位数格式化成图内非对称误差。"""
    q16, q50, q84 = np.quantile(np.asarray(samples, dtype="f8"), [0.1586552539, 0.5, 0.8413447461])
    return rf"{q50:.1f}^{{+{q84 - q50:.1f}}}_{{-{q50 - q16:.1f}}}"


def shared_axis_limits(
    sample_sets: list[np.ndarray],
    column: int,
    *,
    include_zero: bool = False,
) -> tuple[float, float]:
    """按 Task4.3 口径由两条 posterior 共同确定紧凑且共享的坐标范围。"""
    values = np.concatenate([np.asarray(samples[:, column], dtype="f8") for samples in sample_sets])
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        raise ValueError("posterior 中没有有限样本")
    lo, hi = np.quantile(finite, [0.0025, 0.9975])
    if include_zero:
        lo = min(float(lo), 0.0)
        hi = max(float(hi), 0.0)
    width = float(hi - lo)
    if not np.isfinite(width) or width <= 0.0:
        width = 1.0
    return float(lo - 0.08 * width), float(hi + 0.08 * width)


def plot_task43_ppt_style(pk_summary: dict[str, Any], xi_summary: dict[str, Any]) -> dict[str, Any]:
    """按 Task4.3 both-jaxpower 主图风格绘制 Quijote 两参数 corner。"""
    from getdist import MCSamples, plots

    pk_samples = load_postburn_samples(pk_summary)
    xi_samples = load_postburn_samples(xi_summary)
    names = ["fnl_loc", "b1"]
    labels = [r"f_{\rm NL}", r"b_1"]
    xlim = shared_axis_limits([pk_samples, xi_samples], 0, include_zero=True)
    ylim = shared_axis_limits([pk_samples, xi_samples], 1)
    ranges = {"fnl_loc": xlim, "b1": ylim}
    settings = {
        "ignore_rows": 0,
        "fine_bins": 2048,
        "fine_bins_2D": 1024,
        "smooth_scale_1D": GETDIST_SMOOTH_1D,
        "smooth_scale_2D": GETDIST_SMOOTH_2D,
        "boundary_correction_order": 1,
        "mult_bias_correction_order": 1,
    }
    pk_label = (
        r"$P_0(k)$" "\n"
        r"$0.008\leq k\leq0.10\,h\,{\rm Mpc}^{-1}$"
    )
    xi_label = r"$\xi_0(s)$" "\n" r"$50<s<350\,h^{-1}{\rm Mpc}$"
    # 第一条 GetDist root 显示在更高图层；保持与 Task4.3 一致，让 P(k)
    # 深色轮廓位于 2PCF 红色填充之上，legend 也按 P(k)、2PCF 排列。
    pk_mc = MCSamples(samples=pk_samples, names=names, labels=labels, label=pk_label, ranges=ranges, settings=settings)
    xi_mc = MCSamples(samples=xi_samples, names=names, labels=labels, label=xi_label, ranges=ranges, settings=settings)
    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "DejaVu Sans", "Liberation Sans"],
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "font.size": 12,
            "axes.linewidth": 1.0,
            "legend.frameon": False,
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "savefig.facecolor": "white",
            "savefig.transparent": False,
        }
    )
    plotter = plots.get_subplot_plotter(width_inch=8.8)
    plotter.settings.axes_fontsize = 13.5
    plotter.settings.lab_fontsize = 17.0
    plotter.settings.legend_fontsize = PPT_LEGEND_FONTSIZE
    plotter.settings.legend_frame = False
    plotter.settings.figure_legend_frame = False
    plotter.settings.linewidth = 1.8
    plotter.settings.linewidth_contour = 1.8
    plotter.settings.alpha_filled_add = 0.30
    plotter.settings.num_plot_contours = 2
    plotter.triangle_plot(
        [pk_mc, xi_mc],
        names,
        filled=True,
        contour_colors=[PPT_COLORS["pk"], PPT_COLORS["twopcf"]],
        contour_lws=[1.8, 1.8],
        legend_labels=[pk_label, xi_label],
        legend_loc="lower left",
        param_limits=ranges,
    )
    axes = plotter.subplots
    axes[0, 0].set_xlim(*xlim)
    axes[1, 0].set_xlim(*xlim)
    axes[1, 0].set_ylim(*ylim)
    axes[1, 1].set_xlim(*ylim)

    blank_corner_x = float(axes[0, 0].get_position().x1)
    blank_corner_y = float(axes[1, 1].get_position().y1)
    legend_anchor = (blank_corner_x + PPT_LEGEND_GAP, blank_corner_y + PPT_LEGEND_GAP)
    if not plotter.fig.legends:
        raise RuntimeError("GetDist 没有创建 figure legend")
    plotter.fig.legends[-1].set_bbox_to_anchor(legend_anchor, transform=plotter.fig.transFigure)

    fnl_axis = axes[0, 0]
    ymin, ymax = fnl_axis.get_ylim()
    fnl_axis.set_ylim(ymin, ymin + (ymax - ymin) * PPT_FNL_HEADROOM)
    fnl_axis.text(
        0.035,
        0.955,
        rf"$P_0(k):\ f_{{\rm NL}}={constraint_text(pk_samples[:, 0])}$",
        transform=fnl_axis.transAxes,
        ha="left",
        va="top",
        fontsize=PPT_FNL_FONTSIZE,
        color=PPT_COLORS["pk"],
    )
    fnl_axis.text(
        0.035,
        0.825,
        rf"$\xi_0(s):\ f_{{\rm NL}}={constraint_text(xi_samples[:, 0])}$",
        transform=fnl_axis.transAxes,
        ha="left",
        va="top",
        fontsize=PPT_FNL_FONTSIZE,
        color=PPT_COLORS["twopcf"],
    )
    plotter.fig.patch.set_facecolor("white")
    PLOT_ROOT.mkdir(parents=True, exist_ok=True)
    plotter.fig.savefig(OUT_PDF, bbox_inches="tight", pad_inches=0.08, facecolor="white", transparent=False)
    plt.close(plotter.fig)
    return {
        "pdf": OUT_PDF,
        "style_reference": PROJECT_ROOT / "plots/7.13meeting/task43_pk_vs_2pcf_s50_350_both_jaxpower_covariance.pdf",
        "plot_parameters": list(PLOT_PARAM_NAMES),
        "marginalized_parameters": ["sigmas", "sn0"],
        "nuisance_note": "sigmas and sn0 remain free in both Quijote fits but are marginalized and omitted from the triangle.",
        "shared_plot_ranges": {"fnl_loc": xlim, "b1": ylim},
        "contour_and_legend_order": ["pk", "xi0_2pcf"],
        "colors": PPT_COLORS,
        "fnl_annotation_in_panel": True,
        "legend_anchor_figure_fraction": list(legend_anchor),
    }


def parse_args() -> argparse.Namespace:
    """解析 MCMC、资源和续跑参数。"""
    parser = argparse.ArgumentParser(description="Long-chain Task45 LCp50 P(k) vs xi emcee rerun")
    parser.add_argument("--nwalkers", type=int, default=64)
    parser.add_argument("--min-steps", type=int, default=8000)
    parser.add_argument("--max-steps", type=int, default=20000)
    parser.add_argument("--burnin", type=int, default=2000)
    parser.add_argument("--chunk-steps", type=int, default=500)
    parser.add_argument("--nworkers", type=int, default=8)
    parser.add_argument("--seed", type=int, default=20260711)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--plot-only", action="store_true")
    return parser.parse_args()


def main() -> None:
    """构建 likelihood、运行/续跑两条链、检查 gate，并生成 Task4.4 风格 PDF。"""
    args = parse_args()
    if args.nwalkers < 2 * len(PARAM_NAMES):
        raise ValueError("nwalkers 必须至少为 2*ndim=8")
    if args.nworkers < 1 or args.nworkers > 8:
        raise ValueError("登录节点 CPU 总占用必须限制在 1--8 workers")
    if args.burnin >= args.min_steps:
        raise ValueError("burnin 必须小于 min_steps")
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)

    if args.plot_only:
        summaries = {
            case: json.loads((OUTPUT_ROOT / case / "summary.json").read_text(encoding="utf-8"))
            for case in CASE_ORDER
        }
        plot_meta = plot_task43_ppt_style(summaries["pk_binavg"], summaries["xi_r50_350"])
        existing = json.loads(OUT_MANIFEST.read_text(encoding="utf-8")) if OUT_MANIFEST.exists() else {}
        existing["plot"] = plot_meta
        write_json(OUT_MANIFEST, existing)
        print(f"[write] {OUT_PDF}")
        return

    configure_task45()
    print("[setup] build original Task45 contexts", flush=True)
    contexts = task45.build_contexts([TAG], list(CASE_ORDER))
    summaries: dict[str, dict[str, Any]] = {}
    audits: dict[str, dict[str, Any]] = {}
    for icase, case in enumerate(CASE_ORDER):
        context = contexts[(TAG, case)]
        old = load_old_weighted_samples(case)
        likelihood = build_fast_likelihood(context)
        audit = equivalence_audit(context, likelihood, old)
        audits[case] = audit
        print(f"[audit] case={case} max_abs_delta_logl={audit['max_abs_delta_logl']:.3e}", flush=True)
        summaries[case] = run_case(
            case,
            context,
            likelihood,
            old,
            nwalkers=int(args.nwalkers),
            min_steps=int(args.min_steps),
            max_steps=int(args.max_steps),
            burnin=int(args.burnin),
            chunk_steps=int(args.chunk_steps),
            nworkers=int(args.nworkers),
            seed=int(args.seed) + 1009 * icase,
            overwrite=bool(args.overwrite),
        )

    failed = [case for case, summary in summaries.items() if summary["status"] != "done"]
    plot_meta = plot_task43_ppt_style(summaries["pk_binavg"], summaries["xi_r50_350"])
    manifest = {
        "task": "task45_fnl50_pk_xi_r50_emcee",
        "status": "done" if not failed else "convergence_failed",
        "failed_convergence_cases": failed,
        "scientific_scope": {
            "tag": TAG,
            "input_fnl": 50.0,
            "free_parameters": list(PARAM_NAMES),
            "fixed": {"p": P_FIXED},
            "pk_kmax_fit_hmpc": 0.10,
            "xi_display_edges_mpc_over_h": [50.0, 350.0],
            "xi_actual_centers_mpc_over_h": [55.0, 345.0],
        },
        "mcmc_request": {
            "nwalkers": int(args.nwalkers),
            "min_steps": int(args.min_steps),
            "max_steps": int(args.max_steps),
            "burnin": int(args.burnin),
            "chunk_steps": int(args.chunk_steps),
            "max_cpu_workers": int(args.nworkers),
            "execution": "login node, CPU-only, no Slurm",
        },
        "fast_slow_equivalence_audits": audits,
        "summaries": {case: summary["paths"]["summary"] for case, summary in summaries.items()},
        "plot": plot_meta,
    }
    write_json(OUT_MANIFEST, manifest)
    print(f"[write] {OUT_MANIFEST}", flush=True)
    print(f"[write] {OUT_PDF}", flush=True)
    if failed:
        raise SystemExit(f"convergence gate failed for {failed}; chains and diagnostics were preserved")


if __name__ == "__main__":
    main()
