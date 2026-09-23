#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Task 4.3.2 rawbox shared-vs-split sigma_s diagnostic.

代码执行大纲
------------
1. 读取现有 25 个 rawbox RSD P0/P2 与 xi0/xi2 测量，严格使用当前
   promoted BAO-mask：50 <= s < 350 且排除 80 <= s < 120。
2. 复用 Task43 已审计的 P02、xi02 模型和 periodic Gaussian covariance，
   只把 likelihood 参数从 shared sigma_s 扩展为
   ``sigma_s_P`` 与 ``sigma_s_xi``。
3. 对 P-only、xi-only、joint、joint-naive 做 matched MAP/MCMC 对照。
4. 初始诊断固定 covariance，不因 posterior 中的 sigma_s 改写 covariance，
   这样能够单独判断模型自由度和 P/xi cross-covariance 的作用。
5. 所有结果写入独立的 task432_model_repair 子树，不覆盖 Task43 原结果。

注意：这个脚本是模型诊断，不会把 split-sigma 结果自动升级为正式
science model；正式结论还需要 exact-lattice/operator 修复后的 shape closure。
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Any, Callable

for _name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_name, "1")

import emcee
import numpy as np
from scipy.optimize import least_squares


PROJECT_ROOT = Path("/pscratch/sd/l/lzy/PNG-2PCF-model-to-observe")
CODE_DIR = PROJECT_ROOT / "codes" / "task43"
OUTPUT_ROOT = PROJECT_ROOT / "outputs" / "task43_outputs" / "rsd_validation" / "task432_model_repair"
if str(CODE_DIR) not in sys.path:
    sys.path.insert(0, str(CODE_DIR))

from task43_fit_rsd_rawbox_pk0_vs_xi0_smin50 import (  # noqa: E402
    BOUNDS_HI,
    BOUNDS_LO,
    RAWBOX_FIT_EDGES,
    matching_indices,
    set_affinity,
)
from task43_fit_rsd_rawbox_x25 import FastRSDModel, S_EDGES, load_x25, periodic_gaussian_covariance  # noqa: E402
from task43_rawbox_numerics import (  # noqa: E402
    GaussianMetric,
    continuous_rsd_poles,
    exact_lattice_modes,
    mode_operators,
    shell_kernel as analytic_shell_kernel,
)
from task43_rsd_common import PHASES, atomic_savez, atomic_write_json, rawbox_xi_primary_mask  # noqa: E402
from task43_rsd_model import DELTA_C, FullDiscreteRSDModel, build_cache  # noqa: E402
from task43_rsd_rawbox_joint_4way import angular_totals, cross_block, pk_pole_cov  # noqa: E402


P02_DIR = OUTPUT_ROOT.parent / "rawbox" / "pk"
MEASUREMENT_DIR = OUTPUT_ROOT.parent / "rawbox" / "summary"


def jsonable(value: Any) -> Any:
    """递归转换 numpy/path 对象。"""

    if isinstance(value, dict):
        return {str(key): jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [jsonable(item) for item in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    return value


def atomic_json(path: Path, payload: dict[str, Any]) -> None:
    """原子写入 JSON。"""

    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(json.dumps(jsonable(payload), indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(path)


def load_p02_data() -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """读取 25 个 rawbox P02 payload，并选择冻结的 16 个 P bins。"""

    p0_stack: list[np.ndarray] = []
    p2_stack: list[np.ndarray] = []
    k_edges = nmodes = k_obs = None
    for phase in PHASES:
        path = P02_DIR / f"task43_rsd_rawbox_p02_AbacusSummit_base_c000_{phase}_mmin1p4e13_mesh400.npz"
        with np.load(path, allow_pickle=False) as payload:
            p0_stack.append(np.asarray(payload["pk0"], dtype="f8"))
            p2_stack.append(np.asarray(payload["pk2"], dtype="f8"))
            if k_edges is None:
                k_edges = np.asarray(payload["k_edges"], dtype="f8")
                nmodes = np.asarray(payload["nmodes"], dtype="f8")
                k_obs = np.asarray(payload["k"], dtype="f8")
    selected = matching_indices(np.asarray(k_edges), RAWBOX_FIT_EDGES)
    return (
        np.mean(np.stack(p0_stack), axis=0)[selected],
        np.mean(np.stack(p2_stack), axis=0)[selected],
        np.asarray(k_edges)[selected],
        np.asarray(nmodes)[selected],
        np.asarray(k_obs)[selected],
    )


def build_models_and_covariance(*, xi_angle_mode: str = "continuous", k_switch: float = 0.01) -> dict[str, Any]:
    """构造 BAO-masked rawbox data、模型和固定 joint covariance。

    ``xi_angle_mode=exact_lowk`` 时，在当前 continuous-angle xi 模型上加入
    低-k exact lattice angular operator 与 continuous operator 的差值，
    只作为 Task 4.3.2 诊断，不改动 Task43 production cache。
    """

    p0_data, p2_data, p_edges, p_nmodes, p_k = load_p02_data()
    xi, metadata_rows, _ = load_x25()
    xi0_stack = np.asarray(xi["xi0_rsd"], dtype="f8")
    xi2_stack = np.asarray(xi["xi2_rsd"], dtype="f8")
    centers = 0.5 * (S_EDGES[:-1] + S_EDGES[1:])
    mask0 = rawbox_xi_primary_mask(centers)
    mask2 = rawbox_xi_primary_mask(centers)
    xi0_data = np.mean(xi0_stack, axis=0)[mask0]
    xi2_data = np.mean(xi2_stack, axis=0)[mask2]
    nbar = float(np.mean([row["nbar_h3_mpc3_from_npz"] for row in metadata_rows]))

    cache = build_cache(zeff=0.725, boxsize=2000.0, kmax=3.0, ells=(0, 2), cosmology="abacus_c000")
    exact = FullDiscreteRSDModel(cache, nmu=96)
    xi_model = FastRSDModel(exact, sigma_step=0.05)
    if xi_angle_mode not in {"continuous", "exact_lowk"}:
        raise ValueError(f"unknown xi_angle_mode={xi_angle_mode!r}")
    lattice_context: dict[str, Any] | None = None
    if xi_angle_mode == "exact_lowk":
        if not (0.0 < float(k_switch) <= float(np.max(exact.k_eff))):
            raise ValueError(f"invalid k_switch={k_switch}")
        modes = exact_lattice_modes(exact.boxsize, np.asarray([[0.0, float(k_switch)]], dtype="f8"))
        _, operator_x = mode_operators(modes, exact.s_edges, ells=(0, 2))
        lattice_context = {
            "modes": modes,
            "operator_x": operator_x,
            "power": np.interp(np.log(modes.k), np.log(exact.k_eff), exact.pk_dd),
            "alpha": np.interp(np.log(modes.k), np.log(exact.k_eff), exact.alpha),
            "k_switch": float(k_switch),
        }

    # P0/P2 model：P2 仍使用原始 exact parent-mode machinery 和正确的 factor 5。
    from task43_fit_rsd_rawbox_pk0_vs_xi0_smin50 import ExactPeriodicPk0Model, SN0_SCALE

    pk_inner = ExactPeriodicPk0Model(exact, RAWBOX_FIT_EDGES)

    def eval_p(theta: np.ndarray, sigma_p_index: int = 2) -> np.ndarray:
        fnl, b1, sigma_p, sn0 = [float(v) for v in np.asarray(theta)[[0, 1, sigma_p_index, 4]]]
        p0 = pk_inner.evaluate(np.asarray([fnl, b1, sigma_p, sn0], dtype="f8"))
        q = fnl * 2.0 * DELTA_C * (b1 - 1.0)
        amp = b1 + q * pk_inner.alpha
        # P2 使用 ExactPeriodicPk0Model 的 parent-mode 数组，而不是
        # FullDiscreteRSDModel 的连续角向 shell cache，保持与 rawbox P2
        # production machinery 完全一致。
        damping = 1.0 / (1.0 + 0.5 * (pk_inner.k**2 * pk_inner.mu2 * sigma_p**2)) ** 2
        p_mu = pk_inner.pk_dd * (amp + pk_inner.f_growth * pk_inner.mu2) ** 2 * damping
        l2 = 0.5 * (3.0 * pk_inner.mu2 - 1.0)
        p2_modes = 5.0 * p_mu * l2
        p2_bins = np.bincount(pk_inner.bin_id, weights=p2_modes.reshape(-1), minlength=RAWBOX_FIT_EDGES.shape[0]) / pk_inner.counts
        return np.concatenate([p0, p2_bins])

    def eval_x(theta: np.ndarray, sigma_xi_index: int = 3) -> np.ndarray:
        sigma_xi = float(theta[sigma_xi_index])
        values = xi_model.evaluate(np.asarray([float(theta[0]), float(theta[1]), sigma_xi]))
        if lattice_context is not None:
            modes = lattice_context["modes"]
            amplitude = (
                float(theta[1])
                + float(theta[0]) * 2.0 * DELTA_C * (float(theta[1]) - 1.0) * lattice_context["alpha"]
            )
            damping = 1.0 / (1.0 + 0.5 * (modes.k * modes.mu * sigma_xi) ** 2) ** 2
            signal = lattice_context["power"] * (amplitude + exact.f_growth * modes.mu**2) ** 2 * damping
            exact_vector = lattice_context["operator_x"] @ signal
            poles = continuous_rsd_poles(
                modes.k,
                lattice_context["power"],
                amplitude,
                exact.f_growth,
                sigma_xi,
            )
            continuous_vector = np.concatenate(
                [
                    np.sum(poles[ell][:, None] * analytic_shell_kernel(modes.k, exact.s_edges, ell), axis=0)
                    / exact.volume
                    for ell in (0, 2)
                ]
            )
            correction = exact_vector - continuous_vector
            nbin = exact.s_edges.size - 1
            values[0] = np.asarray(values[0]) + correction[:nbin]
            values[2] = np.asarray(values[2]) + correction[nbin:]
        return np.concatenate([np.asarray(values[0])[mask0], np.asarray(values[2])[mask2]])

    # 固定 covariance 使用与 rawbox 四向 control 同样的 periodic Gaussian 结构。
    ang = angular_totals(exact, b1=2.55, sigma_s=8.0, nbar=nbar)
    c_pp = pk_pole_cov(exact, RAWBOX_FIT_EDGES, p_nmodes, ang)
    c_xx_full = periodic_gaussian_covariance(exact, b1=2.55, sigma_s=8.0, nbar=nbar)
    ids_x = np.concatenate([np.flatnonzero(mask0), exact.s_edges.size - 1 + np.flatnonzero(mask2)])
    c_xx = c_xx_full[np.ix_(ids_x, ids_x)]

    # 25 phase empirical cross 用于沿用原 rawbox joint 的 quadrant calibration。
    p_all = []
    for phase in PHASES:
        path = P02_DIR / f"task43_rsd_rawbox_p02_AbacusSummit_base_c000_{phase}_mmin1p4e13_mesh400.npz"
        with np.load(path, allow_pickle=False) as payload:
            p_all.append(np.concatenate([np.asarray(payload["pk0"])[matching_indices(np.asarray(payload["k_edges"]), RAWBOX_FIT_EDGES)], np.asarray(payload["pk2"])[matching_indices(np.asarray(payload["k_edges"]), RAWBOX_FIT_EDGES)]]))
    x_all = np.concatenate([xi0_stack[:, mask0], xi2_stack[:, mask2]], axis=1)
    empirical_cross = np.cov(np.hstack([np.stack(p_all), x_all]), rowvar=False, ddof=1)[:32, 32:]
    cross_raw = cross_block(exact, RAWBOX_FIT_EDGES, p_nmodes, ang, mask0, mask2)
    x0_n = int(np.count_nonzero(mask0))
    quadrant_scales: dict[tuple[int, int], float] = {}
    for la, p_slice in ((0, slice(0, 16)), (2, slice(16, 32))):
        for lb, x_slice in ((0, slice(0, x0_n)), (2, slice(x0_n, None))):
            empirical = empirical_cross[p_slice, x_slice].ravel()
            analytic = cross_raw[x_slice, p_slice].T.ravel()
            denom = float(analytic @ analytic)
            quadrant_scales[(la, lb)] = max(float(empirical @ analytic / denom), 0.0) if denom > 0.0 else 1.0
    c_px = cross_block(exact, RAWBOX_FIT_EDGES, p_nmodes, ang, mask0, mask2, quadrant_scales)

    def assemble(cross_scale: float) -> np.ndarray:
        covariance = np.block([[c_pp, cross_scale * c_px.T], [cross_scale * c_px, c_xx]])
        covariance = 0.5 * (covariance + covariance.T)
        eigenvalues, eigenvectors = np.linalg.eigh(covariance)
        floor = max(1.0e-14 * float(eigenvalues[-1]), 1.0e-300)
        if float(eigenvalues[0]) < floor:
            covariance = (eigenvectors * np.maximum(eigenvalues, floor)[None, :]) @ eigenvectors.T
            covariance = 0.5 * (covariance + covariance.T)
        return covariance

    data = {
        "p": np.concatenate([p0_data, p2_data]),
        "x": np.concatenate([xi0_data, xi2_data]),
    }
    return {
        "data": data,
        "models": {"p": eval_p, "x": eval_x},
        "covariances": {"p": c_pp, "x": c_xx, "joint": assemble(1.0), "joint_naive": assemble(0.0)},
        "metadata": {
            "cache": str(cache),
            "xi_angle_mode": xi_angle_mode,
            "k_switch": float(k_switch) if xi_angle_mode == "exact_lowk" else None,
            "nbar": nbar,
            "p_bins": 16,
            "xi0_bins": int(np.count_nonzero(mask0)),
            "xi2_bins": int(np.count_nonzero(mask2)),
            "xi_mask_contract": "50 <= s < 350; 80 <= s < 120 excluded",
            "quadrant_scales": {f"P{la}_xi{lb}": value for (la, lb), value in quadrant_scales.items()},
        },
    }


def precision_from_covariance(covariance: np.ndarray) -> np.ndarray:
    """在相关矩阵空间求逆，避免 P/xi 量纲跨度损坏 xi 小特征值。"""

    covariance = 0.5 * (covariance + covariance.T)
    scale = np.sqrt(np.diag(covariance))
    correlation = covariance / np.outer(scale, scale)
    eigenvalues, eigenvectors = np.linalg.eigh(0.5 * (correlation + correlation.T))
    floor = max(float(eigenvalues[-1]) * 1.0e-14, 1.0e-300)
    inverse = (eigenvectors * np.where(eigenvalues >= floor, 1.0 / np.maximum(eigenvalues, floor), 0.0)[None, :]) @ eigenvectors.T
    return inverse / np.outer(scale, scale)


def run_fit(
    label: str,
    data: np.ndarray,
    covariance: np.ndarray,
    evaluate: Callable[[np.ndarray], np.ndarray],
    bounds: tuple[np.ndarray, np.ndarray],
    names: tuple[str, ...],
    *,
    nwalkers: int,
    nsteps: int,
    burnin: int,
    seed: int,
    output_root: Path,
) -> dict[str, Any]:
    """执行一个 shared/split sigma 变体的 MAP + MCMC。"""

    precision = precision_from_covariance(covariance)
    chol = np.linalg.cholesky(covariance)

    def residual(theta: np.ndarray) -> np.ndarray:
        return np.linalg.solve(chol, data - evaluate(theta))

    ndim = len(names)
    # 不同 marginal 的参数维度和 sigma 位置不同，不能简单截断一个
    # 五维起点，否则 P-only 的 sn0 起点会被错误地设成 8 Mpc/h。
    if names == ("fNL", "b1", "sigma_s_P", "sn0"):
        starts = [
            np.asarray([0.0, 2.55, 1.0, 0.0], dtype="f8"),
            np.asarray([-20.0, 2.5, 2.0, 0.1], dtype="f8"),
            np.asarray([20.0, 2.6, 1.0, -0.1], dtype="f8"),
        ]
    elif names == ("fNL", "b1", "sigma_s_xi"):
        starts = [
            np.asarray([0.0, 2.55, 8.0], dtype="f8"),
            np.asarray([-20.0, 2.5, 6.0], dtype="f8"),
            np.asarray([20.0, 2.6, 10.0], dtype="f8"),
        ]
    elif names == ("fNL", "b1", "sigma_s", "sn0"):
        starts = [
            np.asarray([0.0, 2.55, 4.0, 0.0], dtype="f8"),
            np.asarray([-20.0, 2.5, 2.0, 0.1], dtype="f8"),
            np.asarray([20.0, 2.6, 6.0, -0.1], dtype="f8"),
        ]
    else:
        starts = [
            np.asarray([0.0, 2.55, 1.0, 8.0, 0.0], dtype="f8"),
            np.asarray([-20.0, 2.5, 2.0, 8.0, 0.1], dtype="f8"),
            np.asarray([20.0, 2.6, 1.0, 6.0, -0.1], dtype="f8"),
        ]
    solutions = [
        least_squares(residual, start, bounds=bounds, max_nfev=3000, xtol=1.0e-11, ftol=1.0e-11, gtol=1.0e-11)
        for start in starts
    ]
    best = min(solutions, key=lambda item: float(item.fun @ item.fun))
    map_theta = np.asarray(best.x, dtype="f8")
    rng = np.random.default_rng(int(seed))
    widths = np.asarray([10.0, 0.05, 0.2, 0.5, 0.05][:ndim], dtype="f8")
    initial = map_theta[None, :] + rng.normal(size=(int(nwalkers), ndim)) * widths[None, :]
    initial = np.clip(initial, bounds[0] + 1.0e-8, bounds[1] - 1.0e-8)
    np.random.seed(int(seed))
    sampler = emcee.EnsembleSampler(int(nwalkers), ndim, lambda theta: -0.5 * float(np.sum(residual(theta) ** 2)) if np.all(theta >= bounds[0]) and np.all(theta <= bounds[1]) else -np.inf)
    sampler.run_mcmc(initial, int(nsteps), progress=False, skip_initial_state_check=True)
    chain = sampler.get_chain(discard=int(burnin), flat=True)
    logp = sampler.get_log_prob(discard=int(burnin), flat=True)
    quantiles = np.percentile(chain, [16.0, 50.0, 84.0], axis=0)
    summary: dict[str, Any] = {
        "label": label,
        "parameter_names": list(names),
        "map_optimizer": map_theta.tolist(),
        "map_chi2": float(best.fun @ best.fun),
        "posterior": {
            name: {
                "q16": float(quantiles[0, index]),
                "q50": float(quantiles[1, index]),
                "q84": float(quantiles[2, index]),
                "sigma68": float(0.5 * (quantiles[2, index] - quantiles[0, index])),
            }
            for index, name in enumerate(names)
        },
        "mcmc": {"nwalkers": int(nwalkers), "nsteps": int(nsteps), "burnin": int(burnin), "seed": int(seed)},
        "acceptance_fraction_mean": float(np.mean(sampler.acceptance_fraction)),
    }
    try:
        tau = np.asarray(emcee.autocorr.integrated_time(sampler.get_chain(discard=int(burnin)), tol=0), dtype="f8")
        summary["postburn_length_over_tau"] = (sampler.get_chain(discard=int(burnin)).shape[0] / tau).tolist()
    except Exception as exc:
        summary["tau_error"] = str(exc)
    out_dir = output_root / "fits" / label
    out_dir.mkdir(parents=True, exist_ok=True)
    atomic_savez(out_dir / "samples.npz", samples=chain, log_prob=logp, map_theta=map_theta, parameter_names=np.asarray(names))
    summary["samples_path"] = str(out_dir / "samples.npz")
    return summary


def main() -> None:
    """执行 rawbox BAO-mask shared/split sigma 受控诊断。"""

    parser = argparse.ArgumentParser()
    parser.add_argument("--threads", type=int, default=6)
    parser.add_argument("--nwalkers", type=int, default=32)
    parser.add_argument("--nsteps", type=int, default=1000)
    parser.add_argument("--burnin", type=int, default=200)
    parser.add_argument("--seed", type=int, default=20260920)
    parser.add_argument(
        "--output-root",
        type=Path,
        default=OUTPUT_ROOT / "rawbox_baomask_split_sigma",
        help="独立诊断输出目录；不能指向已有 immutable production 目录。",
    )
    parser.add_argument("--xi-angle-mode", choices=("continuous", "exact_lowk"), default="continuous")
    parser.add_argument("--k-switch", type=float, default=0.01)
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    if int(args.threads) > 8:
        raise ValueError("Task432 login-node fit must use <= 8 threads")
    set_affinity(int(args.threads))
    output_root = Path(args.output_root)
    if args.smoke:
        args.nsteps, args.burnin = 300, 80

    products = build_models_and_covariance(
        xi_angle_mode=str(args.xi_angle_mode),
        k_switch=float(args.k_switch),
    )
    data_p = products["data"]["p"]
    data_x = products["data"]["x"]
    data_joint = np.concatenate([data_p, data_x])
    models = products["models"]
    cov = products["covariances"]

    lo5 = np.asarray([-500.0, 0.5, 0.0, 0.0, -1.0], dtype="f8")
    hi5 = np.asarray([500.0, 5.0, 30.0, 30.0, 1.0], dtype="f8")

    def evaluate_p(theta: np.ndarray) -> np.ndarray:
        return models["p"](np.asarray(theta, dtype="f8"))

    def evaluate_x(theta: np.ndarray) -> np.ndarray:
        return models["x"](np.asarray(theta, dtype="f8"))

    def evaluate_joint(theta: np.ndarray) -> np.ndarray:
        return np.concatenate([evaluate_p(theta), evaluate_x(theta)])

    def evaluate_joint_shared(theta: np.ndarray) -> np.ndarray:
        """当前 production shared-sigma joint 的同一 BAO-mask 重算。"""

        compact = np.asarray(theta, dtype="f8")
        full = np.asarray([compact[0], compact[1], compact[2], compact[2], compact[3]], dtype="f8")
        return evaluate_joint(full)

    # P-only: [fNL,b1,sigma_P,sigma_Xi_unused,sn0] -> compact four-parameter map.
    def p_compact(theta: np.ndarray) -> np.ndarray:
        full = np.asarray([theta[0], theta[1], theta[2], 8.0, theta[3]], dtype="f8")
        return evaluate_p(full)

    # xi-only: [fNL,b1,sigma_Xi] -> compact three-parameter map.
    def x_compact(theta: np.ndarray) -> np.ndarray:
        full = np.asarray([theta[0], theta[1], 1.0, theta[2], 0.0], dtype="f8")
        return evaluate_x(full)

    fits = {}
    fits["p_marginal"] = run_fit(
        "p_marginal", data_p, cov["p"], p_compact,
        (lo5[[0, 1, 2, 4]], hi5[[0, 1, 2, 4]]), ("fNL", "b1", "sigma_s_P", "sn0"),
        nwalkers=args.nwalkers, nsteps=args.nsteps, burnin=args.burnin, seed=args.seed, output_root=output_root,
    )
    fits["xi_marginal"] = run_fit(
        "xi_marginal", data_x, cov["x"], x_compact,
        (lo5[[0, 1, 3]], hi5[[0, 1, 3]]), ("fNL", "b1", "sigma_s_xi"),
        nwalkers=args.nwalkers, nsteps=args.nsteps, burnin=args.burnin, seed=args.seed + 1, output_root=output_root,
    )
    fits["joint_shared"] = run_fit(
        "joint_shared", data_joint, cov["joint"], evaluate_joint_shared,
        (lo5[[0, 1, 2, 4]], hi5[[0, 1, 2, 4]]), ("fNL", "b1", "sigma_s", "sn0"),
        nwalkers=args.nwalkers, nsteps=args.nsteps, burnin=args.burnin, seed=args.seed + 4, output_root=output_root,
    )
    fits["joint"] = run_fit(
        "joint", data_joint, cov["joint"], evaluate_joint,
        (lo5, hi5), ("fNL", "b1", "sigma_s_P", "sigma_s_xi", "sn0"),
        nwalkers=args.nwalkers, nsteps=args.nsteps, burnin=args.burnin, seed=args.seed + 2, output_root=output_root,
    )
    fits["joint_naive"] = run_fit(
        "joint_naive", data_joint, cov["joint_naive"], evaluate_joint,
        (lo5, hi5), ("fNL", "b1", "sigma_s_P", "sigma_s_xi", "sn0"),
        nwalkers=args.nwalkers, nsteps=args.nsteps, burnin=args.burnin, seed=args.seed + 3, output_root=output_root,
    )

    audit = {
        "task": "Task 4.3.2 rawbox BAO-masked split sigma diagnostic",
        "status": "smoke_complete" if args.smoke else "complete",
        "data_contract": products["metadata"],
        "fit_contract": {
            "baseline": "same covariance and P02/xi02 data as current rawbox diagnostic, with promoted BAO mask",
            "split_parameters": "sigma_s_P enters P0/P2; sigma_s_xi enters xi0/xi2; sn0 enters P0 only",
            "covariance_policy": "fixed diagnostic covariance; no posterior-dependent covariance update",
        },
        "fits": fits,
        "resource": {"threads": int(args.threads), "nwalkers": int(args.nwalkers), "nsteps": int(args.nsteps), "burnin": int(args.burnin)},
    }
    output_root.joinpath("audits").mkdir(parents=True, exist_ok=True)
    atomic_json(output_root / "audits" / ("task432_split_sigma_smoke.json" if args.smoke else "task432_split_sigma_summary.json"), audit)
    print(json.dumps({"status": audit["status"], "output": str(output_root / "audits")}, sort_keys=True))


if __name__ == "__main__":
    main()
