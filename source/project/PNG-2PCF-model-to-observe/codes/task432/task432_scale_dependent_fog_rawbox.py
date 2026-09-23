#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Task 4.3.2 rawbox scale-dependent xi-side FoG diagnostic.

代码执行大纲
------------
1. 复用 task432_split_sigma_rawbox.py 的 BAO-masked rawbox data/covariance
   contract 和 P-side Kaiser x squared-Lorentzian model。
2. 对 xi-side FullDiscrete theory cache 分成 k<k_split 与 k>=k_split 两段，
   分别使用 sigma_xi_low 和 sigma_xi_high 生成 shell-averaged xi0/xi2。
3. 比较 P-only、xi scale-dependent、shared constant-sigma joint 和
   scale-dependent xi joint。
4. 该模型是低维 nonlinear/FoG diagnostic，不会覆盖 shared-sigma production
   结果，也不会自动升级为 science model。
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any, Callable

for _name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_name, "1")

import emcee
import numpy as np
from scipy.interpolate import CubicSpline
from scipy.optimize import least_squares

PROJECT_ROOT = Path("/pscratch/sd/l/lzy/PNG-2PCF-model-to-observe")
CODE_DIR = PROJECT_ROOT / "codes" / "task43"
TASK432_DIR = PROJECT_ROOT / "codes" / "task432"
for _path in (CODE_DIR, TASK432_DIR):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

from task432_split_sigma_rawbox import (  # noqa: E402
    atomic_savez,
    atomic_json,
    build_models_and_covariance,
    precision_from_covariance,
)
from task43_fit_rsd_rawbox_pk0_vs_xi0_smin50 import set_affinity  # noqa: E402
from task43_rsd_common import rawbox_xi_primary_mask  # noqa: E402
from task43_fit_rsd_rawbox_x25 import S_EDGES  # noqa: E402
from task43_rsd_model import DELTA_C, FullDiscreteRSDModel, build_cache  # noqa: E402


OUTPUT_ROOT = PROJECT_ROOT / "outputs" / "task43_outputs" / "rsd_validation" / "task432_model_repair"


class ScaleDependentXiModel:
    """Two-k-band xi-side FoG model using the same FullDiscrete basis machinery."""

    def __init__(self, cache_path: Path, *, k_split: float = 0.08, sigma_step: float = 0.1) -> None:
        self.exact = FullDiscreteRSDModel(cache_path, nmu=96)
        self.k_split = float(k_split)
        self.sigma_grid = np.arange(0.0, 30.0 + 0.5 * float(sigma_step), float(sigma_step), dtype="f8")
        self.ell_values = self.exact.ell_values
        self.nbin = self.exact.s_edges.size - 1
        self.band_masks = (self.exact.k_eff < self.k_split, self.exact.k_eff >= self.k_split)
        self.splines: list[CubicSpline] = []
        powers = (np.ones_like(self.exact.mu2), self.exact.mu2, self.exact.mu2 * self.exact.mu2)
        for band_mask in self.band_masks:
            projection = [
                self.exact.g_nz[:, None] * self.exact.kernels[index] / float(self.exact.volume)
                for index in range(len(self.ell_values))
            ]
            for index in range(len(projection)):
                projection[index] = np.array(projection[index], copy=True)
                projection[index][~band_mask, :] = 0.0
            basis = np.empty((self.sigma_grid.size, len(self.ell_values), 6, self.nbin), dtype="f8")
            for isig, sigma_s in enumerate(self.sigma_grid):
                x = (self.exact.k_eff[:, None] * self.exact.mu[None, :] * float(sigma_s)) ** 2
                damping = 1.0 / (1.0 + 0.5 * x) ** 2
                for iell, ell in enumerate(self.ell_values):
                    prefactor = 0.5 * (2 * int(ell) + 1)
                    moments = [
                        prefactor
                        * np.sum(
                            self.exact.wmu[None, :]
                            * damping
                            * power[None, :]
                            * self.exact.legendre[int(ell)][None, :],
                            axis=1,
                        )
                        for power in powers
                    ]
                    vectors = np.stack(
                        [
                            self.exact.pk_dd * moments[0],
                            self.exact.pk_dd * self.exact.alpha * moments[0],
                            self.exact.pk_dd * self.exact.alpha**2 * moments[0],
                            self.exact.pk_dd * moments[1],
                            self.exact.pk_dd * self.exact.alpha * moments[1],
                            self.exact.pk_dd * moments[2],
                        ]
                    )
                    basis[isig, iell] = vectors @ projection[iell]
            self.splines.append(CubicSpline(self.sigma_grid, basis, axis=0))

    def evaluate(self, theta: np.ndarray, *, sigma_low_index: int = 2, sigma_high_index: int = 3) -> dict[int, np.ndarray]:
        """Evaluate xi0/xi2 for theta=(fNL,b1,sigma_low,sigma_high)."""

        values = np.asarray(theta, dtype="f8")
        fnl, b1 = float(values[0]), float(values[1])
        sigma_low, sigma_high = float(values[sigma_low_index]), float(values[sigma_high_index])
        q = fnl * 2.0 * DELTA_C * (b1 - 1.0)
        coefficients = np.asarray(
            [b1 * b1, 2.0 * b1 * q, q * q, 2.0 * b1 * self.exact.f_growth, 2.0 * q * self.exact.f_growth, self.exact.f_growth**2],
            dtype="f8",
        )
        low = np.asarray(self.splines[0](sigma_low), dtype="f8")
        high = np.asarray(self.splines[1](sigma_high), dtype="f8")
        result = low + high
        return {int(ell): coefficients @ result[index] for index, ell in enumerate(self.ell_values)}


def run_fit(
    label: str,
    data: np.ndarray,
    covariance: np.ndarray,
    evaluate: Callable[[np.ndarray], np.ndarray],
    bounds: tuple[np.ndarray, np.ndarray],
    names: tuple[str, ...],
    *,
    starts: list[np.ndarray],
    nwalkers: int,
    nsteps: int,
    burnin: int,
    seed: int,
    output_root: Path,
) -> dict[str, Any]:
    """Run MAP + emcee for one scale-dependent-FoG variant."""

    precision = precision_from_covariance(covariance)
    chol = np.linalg.cholesky(covariance)

    def residual(theta: np.ndarray) -> np.ndarray:
        return np.linalg.solve(chol, data - evaluate(theta))

    solutions = [
        least_squares(residual, start, bounds=bounds, max_nfev=3000, xtol=1e-11, ftol=1e-11, gtol=1e-11)
        for start in starts
    ]
    best = min(solutions, key=lambda item: float(item.fun @ item.fun))
    center = np.asarray(best.x, dtype="f8")
    rng = np.random.default_rng(int(seed))
    widths = np.asarray([10.0, 0.05, 0.2, 0.4, 0.8, 0.05][:len(names)], dtype="f8")
    initial = center[None, :] + rng.normal(size=(int(nwalkers), len(names))) * widths[None, :]
    initial = np.clip(initial, bounds[0] + 1.0e-8, bounds[1] - 1.0e-8)
    np.random.seed(int(seed))
    sampler = emcee.EnsembleSampler(
        int(nwalkers), len(names),
        lambda theta: -0.5 * float(np.sum(residual(theta) ** 2))
        if np.all(theta >= bounds[0]) and np.all(theta <= bounds[1])
        else -np.inf,
    )
    sampler.run_mcmc(initial, int(nsteps), progress=False, skip_initial_state_check=True)
    chain_by_step = sampler.get_chain(discard=int(burnin))
    chain = chain_by_step.reshape((-1, len(names)))
    logp = sampler.get_log_prob(discard=int(burnin)).reshape((-1,))
    q16, q50, q84 = np.percentile(chain, [16.0, 50.0, 84.0], axis=0)
    summary: dict[str, Any] = {
        "label": label,
        "parameter_names": list(names),
        "map_optimizer": center.tolist(),
        "map_chi2": float(best.fun @ best.fun),
        "posterior": {
            name: {
                "q16": float(q16[index]),
                "q50": float(q50[index]),
                "q84": float(q84[index]),
                "sigma68": float(0.5 * (q84[index] - q16[index])),
            }
            for index, name in enumerate(names)
        },
        "mcmc": {"nwalkers": int(nwalkers), "nsteps": int(nsteps), "burnin": int(burnin), "seed": int(seed)},
        "acceptance_fraction_mean": float(np.mean(sampler.acceptance_fraction)),
    }
    try:
        tau = np.asarray(emcee.autocorr.integrated_time(chain_by_step, tol=0), dtype="f8")
        summary["postburn_length_over_tau"] = (chain_by_step.shape[0] / tau).tolist()
    except Exception as exc:
        summary["tau_error"] = str(exc)
    output_dir = output_root / "fits" / label
    output_dir.mkdir(parents=True, exist_ok=True)
    atomic_savez(output_dir / "samples.npz", samples=chain, log_prob=logp, map_theta=center, parameter_names=np.asarray(names))
    summary["samples_path"] = str(output_dir / "samples.npz")
    return summary


def main() -> None:
    """Run the rawbox scale-dependent FoG diagnostic."""

    parser = argparse.ArgumentParser()
    parser.add_argument("--threads", type=int, default=6)
    parser.add_argument("--nwalkers", type=int, default=32)
    parser.add_argument("--nsteps", type=int, default=1000)
    parser.add_argument("--burnin", type=int, default=200)
    parser.add_argument("--seed", type=int, default=20260930)
    parser.add_argument("--k-split", type=float, default=0.08)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument(
        "--output-root",
        type=Path,
        default=OUTPUT_ROOT / "rawbox_baomask_scale_fog",
    )
    args = parser.parse_args()
    if int(args.threads) > 8:
        raise ValueError("Task432 login-node fit must use <= 8 threads")
    set_affinity(int(args.threads))
    if args.smoke:
        args.nsteps, args.burnin = 300, 80

    products = build_models_and_covariance()
    data = products["data"]
    covariance = products["covariances"]
    base_p = products["models"]["p"]
    xi_scale = ScaleDependentXiModel(products["metadata"]["cache"], k_split=float(args.k_split))
    mask0 = rawbox_xi_primary_mask(0.5 * (S_EDGES[:-1] + S_EDGES[1:]))
    # data x is already ordered as 26 xi0 + 26 xi2 under the frozen BAO mask.

    def p_model(theta: np.ndarray) -> np.ndarray:
        # base_p 的旧诊断合同是五维
        # [fNL, b1, sigma_s_P, sigma_s_xi_unused, sn0]，而本模型的
        # 六维向量是 [fNL, b1, sigma_s_P, sigma_xi_low, sigma_xi_high, sn0]。
        # 这里显式重排，避免把 sigma_xi_high 误当成 sn0。
        mapped = np.asarray([theta[0], theta[1], theta[2], 0.0, theta[5]], dtype="f8")
        return base_p(mapped)

    def x_model(theta: np.ndarray) -> np.ndarray:
        # xi_scale 只读取 [fNL, b1, sigma_xi_low, sigma_xi_high]，
        # 不使用 P-side sigma_s_P 和 sn0。
        xi_theta = np.asarray([theta[0], theta[1], theta[3], theta[4]], dtype="f8")
        values = xi_scale.evaluate(xi_theta)
        return np.concatenate([values[0][mask0], values[2][mask0]])

    def joint_model(theta: np.ndarray) -> np.ndarray:
        return np.concatenate([p_model(theta), x_model(theta)])

    lo = np.asarray([-500.0, 0.5, 0.0, 0.0, 0.0, -1.0], dtype="f8")
    hi = np.asarray([500.0, 5.0, 30.0, 30.0, 30.0, 1.0], dtype="f8")
    # Full parameter vector: fNL, b1, sigma_P, sigma_xi_low, sigma_xi_high, sn0.
    # The P-only compact vector is handled by an explicit adapter below.
    def p_compact(theta: np.ndarray) -> np.ndarray:
        full = np.asarray([theta[0], theta[1], theta[2], 0.0, 0.0, theta[3]], dtype="f8")
        return p_model(full)

    def x_compact(theta: np.ndarray) -> np.ndarray:
        full = np.asarray([theta[0], theta[1], 1.0, theta[2], theta[3], 0.0], dtype="f8")
        return x_model(full)

    fits: dict[str, Any] = {}
    fits["p_marginal"] = run_fit(
        "p_marginal", data["p"], covariance["p"], p_compact,
        (lo[[0, 1, 2, 5]], hi[[0, 1, 2, 5]]), ("fNL", "b1", "sigma_s_P", "sn0"),
        starts=[np.asarray([0.0, 2.55, 1.0, 0.0]), np.asarray([-20.0, 2.5, 2.0, 0.1]), np.asarray([20.0, 2.6, 1.0, -0.1])],
        nwalkers=args.nwalkers, nsteps=args.nsteps, burnin=args.burnin, seed=args.seed, output_root=args.output_root,
    )
    fits["xi_scale"] = run_fit(
        "xi_scale", data["x"], covariance["x"], x_compact,
        (lo[[0, 1, 3, 4]], hi[[0, 1, 3, 4]]), ("fNL", "b1", "sigma_xi_low", "sigma_xi_high"),
        starts=[np.asarray([0.0, 2.55, 2.0, 5.0]), np.asarray([-20.0, 2.5, 3.0, 6.0]), np.asarray([20.0, 2.6, 1.0, 8.0])],
        nwalkers=args.nwalkers, nsteps=args.nsteps, burnin=args.burnin, seed=args.seed + 1, output_root=args.output_root,
    )
    fits["joint_scale"] = run_fit(
        "joint_scale", np.concatenate([data["p"], data["x"]]), covariance["joint"], joint_model,
        (lo, hi), ("fNL", "b1", "sigma_s_P", "sigma_xi_low", "sigma_xi_high", "sn0"),
        starts=[np.asarray([0.0, 2.55, 1.0, 2.0, 5.0, 0.0]), np.asarray([-20.0, 2.5, 2.0, 3.0, 6.0, 0.1]), np.asarray([20.0, 2.6, 1.0, 1.0, 8.0, -0.1])],
        nwalkers=args.nwalkers, nsteps=args.nsteps, burnin=args.burnin, seed=args.seed + 2, output_root=args.output_root,
    )

    audit = {
        "task": "Task 4.3.2 rawbox BAO-masked scale-dependent xi FoG diagnostic",
        "status": "smoke_complete" if args.smoke else "complete",
        "k_split_h_mpc": float(args.k_split),
        "data_contract": products["metadata"],
        "fit_contract": {
            "P_model": "unchanged current P02 model; sigma_s_P",
            "xi_model": "FullDiscrete continuous-angle xi with separate sigma_xi_low/high by k band",
            "covariance": "fixed C_single family; no /25",
            "diagnostic_only": True,
        },
        "fits": fits,
        "resource": {"threads": int(args.threads), "nwalkers": int(args.nwalkers), "nsteps": int(args.nsteps), "burnin": int(args.burnin)},
    }
    args.output_root.joinpath("audits").mkdir(parents=True, exist_ok=True)
    atomic_json(args.output_root / "audits" / ("task432_scale_fog_smoke.json" if args.smoke else "task432_scale_fog_summary.json"), audit)
    print(json.dumps({"status": audit["status"], "output": str(args.output_root / "audits")}, sort_keys=True))


if __name__ == "__main__":
    main()
