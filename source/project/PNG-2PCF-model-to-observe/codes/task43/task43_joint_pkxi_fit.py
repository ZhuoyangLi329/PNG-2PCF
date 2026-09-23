#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Task43 joint P(k)+xi0 fit under the audited s50-350 jaxpower radial-RIC setup.

目的
====
在 4.3 实空间主图口径（P(k) 15 bins free-sn0 + xi0 30 bins fixed-sn0=0，
双侧 radial single-term RIC，single-realization covariance）下量化
P(k)+xi0 联合拟合相对单 probe 的 fNL 约束提升。

模型与 nuisance 约定
--------------------
- P(k) 侧完全复用已审计 pk 链：W_geom[P(θ)] - W_RIC[P(θ)]，
  window_theory_kmin = 2*pi/2000（mother-box），theory vector 含
  sn0*1e4 常数项。
- xi0 侧完全复用已审计 2PCF 链：FullDiscrete + shell-averaged j0 核 +
  full PNG，radial operator 减法，contact sn0=0。
- 联合拟合中 sn0 只进入 P(k) 侧（与两条边际链各自的物理口径一致；
  xi0 s>0 bin 不约束 contact term）。这一 nuisance 基差异会写入
  summary，不得静默。
- 先验：fnl [-500,500]，sn0 [-1,1]（同 pk 链）；b1 取两条审计链先验的
  交集 [0.5,5]（pk 链先验；postior 位于 ~2.5，远离边界）。

拟合变体
--------
pk_marginal   15-bin P(k)，C=C_pp（本次新 mesh64 recipe）
xi_marginal   30-bin xi0，C=C_xx_new（本次新网格 rrdeconv 版）
joint         45-bin 联合，完整含 cross block C_px
joint_naive   45-bin 联合但 cross block 置零（"独立相乘"的错误做法，
              用于量化忽略 cross-covariance 会高估多少增益）
joint_half    cross block 减半（对 cross block 精度的敏感性检查）

链与 gate
---------
统一 64 walkers x 20000 steps, burnin 5000；gate 与 pair audit 一致：
post-burn length/tau > 100 且 split median shift < 0.05 sigma。
MAP optimizer 的原始解单独保存，walker 初始化才做 clip（避免
2026-09-04 pipeline audit 指出的 optimizer 裁剪覆盖问题）。
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

import numpy as np

PROJECT_ROOT = Path("/pscratch/sd/l/lzy/PNG-2PCF-model-to-observe")
CODE_DIR = PROJECT_ROOT / "codes" / "task43"
if str(CODE_DIR) not in sys.path:
    sys.path.insert(0, str(CODE_DIR))

from task43_fit_minimal_closure import (  # noqa: E402
    _shell_j0_average,
    build_theory_context,
    load_radial_ric_operator,
)
from task43_fit_pk_lightcone import (  # noqa: E402
    FitData,
    model_pk,
    png_realspace_pk,
)
from task43_theory_template import COSMOLOGY_CHOICES, DEFAULT_COSMOLOGY, build_template_arrays, load_task41  # noqa: E402

ACTIVE_T43_ROOT = PROJECT_ROOT / "plots" / "outputs" / "task43_outputs"
JOINT_ROOT = PROJECT_ROOT / "outputs" / "task43_outputs" / "joint_pkxi_s50_350"
JOINT_FIT_DIR = JOINT_ROOT / "fits"
JOINT_AUDIT_DIR = JOINT_ROOT / "audits"

MEAN_XI_PATH = ACTIVE_T43_ROOT / "summary" / "task43_mean_xi_mmin1p4e13_x25_s50_350_ds10_fkpP010000.npz"
PK_PAYLOAD_PATH = (
    ACTIVE_T43_ROOT
    / "pk_lightcone"
    / "summary"
    / "task43_pk_lightcone_mmin1p4e13_x25_fkpP010000_desi_rebin_kmax0p10_payload.npz"
)
RIC_OPERATOR_PATH = (
    ACTIVE_T43_ROOT
    / "ric_singleterm"
    / "operators"
    / "task43_ric_factorized_operator_ph000_dchi2_nsub200000_sobol2p22_ds2_seed20260712_L2000.npz"
)

PARAM_NAMES = ("fnl_loc", "b1", "sn0")
PRIORS = {"fnl_loc": (-500.0, 500.0), "b1": (0.5, 5.0), "sn0": (-1.0, 1.0)}
WINDOW_THEORY_KMIN = 2.0 * np.pi / 2000.0  # mother-box cutoff, 与已审计 pk 链一致
XI_KMAX = 5.0
XI_NDENSE = 60000
XI_BOXSIZE = 2000.0


def to_jsonable(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): to_jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [to_jsonable(item) for item in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, Path):
        return str(value)
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    return str(value)


def atomic_write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    tmp.write_text(json.dumps(to_jsonable(payload), indent=2, sort_keys=True) + "\n", encoding="utf-8")
    tmp.replace(path)


def atomic_savez(path: Path, **arrays: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.{os.getpid()}.tmp.npz")
    try:
        np.savez_compressed(tmp, **arrays)
        tmp.replace(path)
    except Exception:
        try:
            tmp.unlink()
        except FileNotFoundError:
            pass
        raise


class JointModel:
    """P(k) 与 xi0 两侧模型向量的统一封装（各复用其审计实现）。"""

    def __init__(self, *, args: argparse.Namespace) -> None:
        # ---- P(k) 侧（复用已审计 pk 链的全部实现细节） ----
        self.pkdata = FitData(Path(args.pk_payload))
        self.pkdata.load_radial_ric_operator(Path(args.ric_operator))
        task41 = load_task41()
        kmin_template = min(1.0e-5, float(np.min(self.pkdata.theory_k[self.pkdata.theory_k > 0.0])) * 0.5)
        k_template = np.logspace(np.log10(kmin_template), np.log10(20.0), 20000)
        self.pk_template, self.pk_cosmology_meta = build_template_arrays(
            task41, k_template, z=float(self.pkdata.zeff), cosmology=str(args.cosmology)
        )
        self.pk_theta_kwargs = dict(
            p_fixed=float(args.p_fixed),
            sn0_scale=float(self.pkdata.sn0_scale),
            window_theory_kmin=WINDOW_THEORY_KMIN,
        )

        # ---- xi0 侧（复用已审计 2PCF 链的 FullDiscrete basis） ----
        xi_data = np.load(args.xi_path, allow_pickle=False)
        self.xi_s = np.asarray(xi_data["s"], dtype="f8")
        s_edges = np.asarray(xi_data["s_edges"], dtype="f8")
        self.xi_mean = np.asarray(xi_data["xi0"], dtype="f8")
        self.zeff = float(np.asarray(xi_data["zeff"]).item())
        self.xi_s_bin_edges = np.column_stack([s_edges[:-1], s_edges[1:]])

        theory = build_theory_context(
            self.zeff,
            kmax=XI_KMAX,
            ndense=XI_NDENSE,
            boxsize=XI_BOXSIZE,
            cosmology=str(args.cosmology),
        )
        radial = load_radial_ric_operator(Path(args.ric_operator), theory=theory, selected_s=self.xi_s)
        task41 = theory["task41"]
        kernel = _shell_j0_average(theory["k_eff"], self.xi_s_bin_edges[:, 0], self.xi_s_bin_edges[:, 1])

        def xi_from_pk_dense(pk_dense: np.ndarray) -> np.ndarray:
            weight = theory["g_nz"] * task41.interp_logk(theory["k_eff"], theory["k_dense"], pk_dense)
            return (weight @ kernel) / float(theory["volume"])

        pk_dd = task41.interp_logk(theory["k_dense"], theory["template"]["k"], theory["template"]["pk_dd"])
        alpha = task41.interp_logk(theory["k_dense"], theory["template"]["k"], theory["template"]["alpha"])
        basis = {
            "xi_pk_dd": xi_from_pk_dense(pk_dd),
            "xi_alpha_pk_dd": xi_from_pk_dense(alpha * pk_dd),
            "xi_alpha2_pk_dd": xi_from_pk_dense(alpha * alpha * pk_dd),
            "rad_pk_dd": np.asarray(radial["basis"]["pk_dd"], dtype="f8"),
            "rad_alpha_pk_dd": np.asarray(radial["basis"]["alpha_pk_dd"], dtype="f8"),
            "rad_alpha2_pk_dd": np.asarray(radial["basis"]["alpha2_pk_dd"], dtype="f8"),
        }
        for name in ("xi_pk_dd", "xi_alpha_pk_dd", "xi_alpha2_pk_dd"):
            if basis[name].shape != self.xi_mean.shape:
                raise ValueError(f"xi basis {name} shape {basis[name].shape} 与数据 {self.xi_mean.shape} 不一致")
        self.xi_basis = basis
        self.xi_theory_meta = {
            "kmax": XI_KMAX,
            "ndense": XI_NDENSE,
            "boxsize": XI_BOXSIZE,
            "operator": {key: value for key, value in radial.items() if key != "basis"},
        }

    # ---- 参数 -> 两侧模型向量 ----
    @staticmethod
    def coefficients(theta: np.ndarray) -> tuple[float, float, float]:
        fnl_loc, b1 = float(theta[0]), float(theta[1])
        bphi = 2.0 * 1.686 * (b1 - 1.0)
        fnl_bphi = fnl_loc * bphi
        return b1 * b1, 2.0 * b1 * fnl_bphi, fnl_bphi * fnl_bphi

    def model_pk_vector(self, theta: np.ndarray) -> np.ndarray:
        theta_dict = {"fnl_loc": float(theta[0]), "b1": float(theta[1]), "sn0": float(theta[2])}
        return model_pk(
            self.pkdata,
            self.pk_template,
            theta_dict,
            p_fixed=self.pk_theta_kwargs["p_fixed"],
            sn0_scale=self.pk_theta_kwargs["sn0_scale"],
            window_theory_kmin=self.pk_theta_kwargs["window_theory_kmin"],
        )

    def model_xi_vector(self, theta: np.ndarray) -> np.ndarray:
        c0, c1, c2 = self.coefficients(theta)
        base = (
            c0 * self.xi_basis["xi_pk_dd"]
            + c1 * self.xi_basis["xi_alpha_pk_dd"]
            + c2 * self.xi_basis["xi_alpha2_pk_dd"]
        )
        radial = (
            c0 * self.xi_basis["rad_pk_dd"]
            + c1 * self.xi_basis["rad_alpha_pk_dd"]
            + c2 * self.xi_basis["rad_alpha2_pk_dd"]
        )
        return base - radial

    def model_vector(self, theta: np.ndarray, blocks: str) -> np.ndarray:
        if blocks == "pk":
            return self.model_pk_vector(theta)
        if blocks == "xi":
            return self.model_xi_vector(theta)
        return np.concatenate([self.model_pk_vector(theta), self.model_xi_vector(theta)])

    def data_vector(self, blocks: str) -> np.ndarray:
        if blocks == "pk":
            return self.pkdata.pk_data
        if blocks == "xi":
            return self.xi_mean
        return np.concatenate([self.pkdata.pk_data, self.xi_mean])


def build_precision(joint_cov: np.ndarray, blocks: str, cross_scale: float) -> np.ndarray:
    cov = np.array(joint_cov, dtype="f8", copy=True)
    if blocks == "pk":
        cov = cov[:15, :15]
    elif blocks == "xi":
        cov = cov[15:, 15:]
    else:
        if float(cross_scale) != 1.0:
            cov[:15, 15:] *= float(cross_scale)
            cov[15:, :15] *= float(cross_scale)
    cov = 0.5 * (cov + cov.T)
    # 联合矩阵谱跨度极大（P 块特征值 ~4e8，xi 块 ~1e-9），pinv 的相对
    # rcond 会把 xi 侧特征方向整体截掉（等价于丢掉 xi 的独立信息）。
    # 因此先做 D^{-1/2} C D^{-1/2} 相关矩阵归一化，在对角为 1 的良态
    # 空间里求逆，再变换回去；负/零特征值只允许绝对极小的修复。
    scale = np.sqrt(np.diag(cov))
    if np.any(scale <= 0.0):
        raise ValueError("covariance has non-positive diagonal")
    scaled = cov / np.outer(scale, scale)
    scaled = 0.5 * (scaled + scaled.T)
    evals, evecs = np.linalg.eigh(scaled)
    if float(np.max(evals)) <= 0.0:
        raise ValueError("scaled covariance is not positive")
    floor = max(float(np.max(evals)) * 1.0e-14, 1.0e-300)
    n_floored = int(np.count_nonzero(evals < floor))
    inv_evals = np.where(evals >= floor, 1.0 / np.maximum(evals, floor), 0.0)
    inv_scaled = (evecs * inv_evals[None, :]) @ evecs.T
    return inv_scaled / np.outer(scale, scale)


def fit_one(
    *,
    label: str,
    model: JointModel,
    joint_cov: np.ndarray,
    blocks: str,
    ndim: int,
    cross_scale: float,
    nwalkers: int,
    nsteps: int,
    burnin: int,
    seed: int,
    out_dir: Path,
) -> dict[str, Any]:
    import emcee
    from scipy.optimize import minimize

    data = model.data_vector(blocks)
    precision = build_precision(joint_cov, blocks, cross_scale)

    def chi2(values: np.ndarray) -> float:
        diff = data - model.model_vector(np.asarray(values, dtype="f8"), blocks)
        return float(diff @ precision @ diff)

    def log_prior(values: np.ndarray) -> float:
        for i, name in enumerate(PARAM_NAMES[:ndim]):
            lo, hi = PRIORS[name]
            if not (lo <= float(values[i]) <= hi):
                return -np.inf
        return 0.0

    def log_prob(values: np.ndarray) -> float:
        lp = log_prior(values)
        if not np.isfinite(lp):
            return -np.inf
        return lp - 0.5 * chi2(values)

    starts = [
        np.array([0.0, 2.5, 0.0]),
        np.array([50.0, 2.5, 0.0]),
        np.array([-50.0, 2.5, 0.0]),
        np.array([0.0, 2.2, 0.2]),
        np.array([0.0, 2.8, -0.2]),
    ]
    if ndim == 2:
        starts = [np.array([0.0, 2.5]), np.array([50.0, 2.5]), np.array([-50.0, 2.5]), np.array([0.0, 2.2]), np.array([0.0, 2.8])]
    bounds = [PRIORS[name] for name in PARAM_NAMES[:ndim]]
    best = None
    for start in starts:
        result = minimize(chi2, x0=np.asarray(start, dtype="f8"), method="L-BFGS-B", bounds=bounds, options={"maxiter": 3000, "ftol": 1.0e-10})
        if best is None or float(result.fun) < float(best.fun):
            best = result
    opt_raw = np.asarray(best.x, dtype="f8").copy()  # 原始 optimizer 解单独保存，绝不回写

    rng = np.random.default_rng(int(seed))
    center = np.clip(opt_raw, [lo + 1.0e-8 for lo, _ in bounds], [hi - 1.0e-8 for _, hi in bounds])
    scales = np.array([10.0, 0.05, 0.04][:ndim], dtype="f8")
    walkers = center[None, :] + rng.normal(scale=scales[None, :], size=(int(nwalkers), ndim))
    for i, (lo, hi) in enumerate(bounds):
        walkers[:, i] = np.clip(walkers[:, i], lo + 1.0e-8, hi - 1.0e-8)
    np.random.seed(int(seed))
    sampler = emcee.EnsembleSampler(int(nwalkers), ndim, log_prob)
    t0 = time.time()
    sampler.run_mcmc(walkers, int(nsteps), progress=False, skip_initial_state_check=True)
    elapsed = time.time() - t0

    chain = sampler.get_chain(discard=int(burnin))
    flat = chain.reshape((-1, ndim))
    logp = sampler.get_log_prob(discard=int(burnin)).reshape((-1,))

    names = PARAM_NAMES[:ndim]
    summary: dict[str, Any] = {"label": label, "blocks": blocks, "cross_scale": float(cross_scale)}
    for i, name in enumerate(names):
        q16, q50, q84 = np.percentile(flat[:, i], [15.8655, 50.0, 84.1345])
        summary[name] = {
            "q16": float(q16),
            "q50": float(q50),
            "q84": float(q84),
            "sigma68": float(0.5 * (q84 - q16)),
            "err_low": float(q50 - q16),
            "err_high": float(q84 - q50),
            "mean": float(np.mean(flat[:, i])),
            "std": float(np.std(flat[:, i], ddof=1)),
        }
    imax = int(np.nanargmax(logp))
    summary["map"] = {name: float(flat[imax, i]) for i, name in enumerate(names)}
    summary["map"]["log_prob"] = float(logp[imax])
    summary["map"]["chi2"] = float(chi2(flat[imax]))
    summary["map"]["chi2_dof"] = float(chi2(flat[imax]) / (data.size - ndim))
    summary["data_size"] = int(data.size)
    summary["optimizer_raw"] = {
        "x": [float(v) for v in opt_raw],
        "chi2": float(best.fun),
        "success": bool(best.success),
        "note": "raw optimizer solution preserved unclipped; walker init used a clipped copy",
    }

    # ---- 链诊断 gate（与 pair audit 同一定义） ----
    from emcee.autocorr import integrated_time

    try:
        tau = np.asarray(integrated_time(chain, tol=0), dtype="f8")
    except Exception:
        tau = np.full(ndim, np.nan)
    length_over_tau = float(chain.shape[0]) / tau if np.all(np.isfinite(tau)) else np.full(ndim, np.nan)
    split_shifts = []
    half = flat.shape[0] // 2
    for i, name in enumerate(names):
        sigma68 = summary[name]["sigma68"]
        shift = abs(float(np.median(flat[:half, i])) - float(np.median(flat[half:, i]))) / max(sigma68, 1.0e-30)
        split_shifts.append(shift)
    diag = {
        "gate": "length/tau > 100 and split median shift < 0.05 sigma",
        "length_over_tau": [float(v) for v in np.atleast_1d(length_over_tau)],
        "length_over_tau_min": float(np.min(length_over_tau)) if np.all(np.isfinite(length_over_tau)) else None,
        "split_median_shift_sigma": [float(v) for v in split_shifts],
        "split_median_shift_sigma_max": float(np.max(split_shifts)),
        "pass": bool(
            np.all(np.isfinite(length_over_tau))
            and float(np.min(length_over_tau)) > 100.0
            and float(np.max(split_shifts)) < 0.05
        ),
        "acceptance_fraction_mean": float(np.mean(sampler.acceptance_fraction)),
        "elapsed_sec": float(elapsed),
    }
    summary["chain_diagnostics"] = diag

    # ---- Fisher 交叉检验（数值微分于 MAP） ----
    theta_map = flat[imax].copy()
    step = np.array([1.0, 0.01, 0.01][:ndim], dtype="f8")
    grads = []
    for i in range(ndim):
        tp = theta_map.copy()
        tm = theta_map.copy()
        tp[i] += step[i]
        tm[i] -= step[i]
        grads.append((model.model_vector(tp, blocks) - model.model_vector(tm, blocks)) / (2.0 * step[i]))
    gmat = np.column_stack(grads)
    fisher = gmat.T @ precision @ gmat
    try:
        cov_fisher = np.linalg.inv(fisher)
        sigma_fisher = np.sqrt(np.diag(cov_fisher))
    except np.linalg.LinAlgError:
        sigma_fisher = np.full(ndim, np.nan)
    summary["fisher_check"] = {
        "steps": [float(v) for v in step],
        "sigma_fisher": [float(v) for v in sigma_fisher],
        "sigma68_mcmc": [summary[name]["sigma68"] for name in names],
        "ratio_sigma68_over_fisher": [
            float(summary[name]["sigma68"] / sigma_f) if np.isfinite(sigma_f) else None
            for name, sigma_f in zip(names, sigma_fisher)
        ],
    }

    out_dir = out_dir / label
    atomic_savez(
        out_dir / "samples.npz",
        param_names=np.asarray(names),
        samples=flat,
        log_prob=logp,
        map_theta=theta_map,
    )
    summary["paths"] = {"samples_npz": str(out_dir / "samples.npz")}
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--joint-cov",
        type=Path,
        default=JOINT_ROOT / "covariance" / "task43_joint_cov_64_s50_350.npz",
    )
    parser.add_argument("--pk-payload", type=Path, default=PK_PAYLOAD_PATH)
    parser.add_argument("--xi-path", type=Path, default=MEAN_XI_PATH)
    parser.add_argument("--ric-operator", type=Path, default=RIC_OPERATOR_PATH)
    parser.add_argument("--cosmology", choices=COSMOLOGY_CHOICES, default=DEFAULT_COSMOLOGY)
    parser.add_argument("--p-fixed", type=float, default=1.0)
    parser.add_argument("--nwalkers", type=int, default=64)
    parser.add_argument("--nsteps", type=int, default=20000)
    parser.add_argument("--burnin", type=int, default=5000)
    parser.add_argument("--seed", type=int, default=20260905)
    parser.add_argument("--fits", type=str, default="pk_marginal,xi_marginal,joint,joint_naive,joint_half")
    parser.add_argument("--smoke", action="store_true", help="短链 smoke（64x200, burnin 50），只写 fits/smoke/")
    args = parser.parse_args()

    if args.smoke:
        args.nsteps, args.burnin = 200, 50
        fit_root = JOINT_FIT_DIR / "smoke"
    else:
        fit_root = JOINT_FIT_DIR

    joint_npz = np.load(args.joint_cov, allow_pickle=False)
    joint_cov = np.asarray(joint_npz["joint_covariance"], dtype="f8")

    model = JointModel(args=args)

    specs = {
        "pk_marginal": {"blocks": "pk", "ndim": 3, "cross_scale": 1.0},
        "xi_marginal": {"blocks": "xi", "ndim": 2, "cross_scale": 1.0},
        "joint": {"blocks": "joint", "ndim": 3, "cross_scale": 1.0},
        "joint_naive": {"blocks": "joint", "ndim": 3, "cross_scale": 0.0},
        "joint_half": {"blocks": "joint", "ndim": 3, "cross_scale": 0.5},
    }
    requested = [item.strip() for item in str(args.fits).split(",") if item.strip()]
    summaries: dict[str, Any] = {}
    for iplot, label in enumerate(requested):
        spec = specs[label]
        print(f"[fit] {label} ...", flush=True)
        summaries[label] = fit_one(
            label=label,
            model=model,
            joint_cov=joint_cov,
            blocks=spec["blocks"],
            ndim=spec["ndim"],
            cross_scale=spec["cross_scale"],
            nwalkers=int(args.nwalkers),
            nsteps=int(args.nsteps),
            burnin=int(args.burnin),
            seed=int(args.seed) + 100 * iplot,
            out_dir=fit_root,
        )
        fnl = summaries[label]["fnl_loc"]
        print(
            f"[fit] {label}: fNL={fnl['q50']:.2f} -{fnl['err_low']:.2f} +{fnl['err_high']:.2f} "
            f"sigma68={fnl['sigma68']:.2f} gate_pass={summaries[label]['chain_diagnostics']['pass']}",
            flush=True,
        )

    # ---- 增益汇总（仅完整运行时；smoke 只存原始 summary） ----
    result = {
        "task": "task43_joint_pkxi_fit",
        "status": "done",
        "smoke": bool(args.smoke),
        "config": {
            "joint_cov": str(args.joint_cov),
            "pk_payload": str(args.pk_payload),
            "xi_path": str(args.xi_path),
            "ric_operator": str(args.ric_operator),
            "window_theory_kmin": float(WINDOW_THEORY_KMIN),
            "xi_theory": model.xi_theory_meta,
            "priors": {key: list(value) for key, value in PRIORS.items()},
            "b1_prior_note": "intersection of audited pk [0.5,5] and xi [0.2,10] priors; posterior near 2.5",
            "nuisance_note": "sn0*1e4 enters the P(k) theory vector only; xi0 contact term stays 0 (audited marginal conventions)",
            "nwalkers": int(args.nwalkers),
            "nsteps": int(args.nsteps),
            "burnin": int(args.burnin),
            "seed": int(args.seed),
            "cosmology": str(args.cosmology),
            "p_fixed": float(args.p_fixed),
        },
        "fits": summaries,
    }
    if not args.smoke and "joint" in summaries and "pk_marginal" in summaries and "xi_marginal" in summaries:
        sig = {label: summaries[label]["fnl_loc"]["sigma68"] for label in ("pk_marginal", "xi_marginal", "joint")}
        if "joint_naive" in summaries:
            sig["joint_naive"] = summaries["joint_naive"]["fnl_loc"]["sigma68"]
        best_marginal = min(("pk_marginal", "xi_marginal"), key=lambda name: sig[name])
        improvement = {
            "sigma68": sig,
            "best_marginal": best_marginal,
            "improvement_vs_best_marginal": float(1.0 - sig["joint"] / sig[best_marginal]),
            "naive_improvement_vs_best_marginal": (
                float(1.0 - sig["joint_naive"] / sig[best_marginal]) if "joint_naive" in sig else None
            ),
            "cross_covariance_cost_ratio_joint_over_naive": (
                float(sig["joint"] / sig["joint_naive"]) if "joint_naive" in sig else None
            ),
            "center_fnl": {label: summaries[label]["fnl_loc"]["q50"] for label in summaries},
        }
        result["improvement"] = improvement
    out_json = (fit_root if args.smoke else JOINT_AUDIT_DIR) / ("task43_joint_pkxi_fit_smoke.json" if args.smoke else "task43_joint_pkxi_fit_summary.json")
    atomic_write_json(out_json, result)
    print(f"[done] wrote {out_json}")


if __name__ == "__main__":
    main()
