#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""汇总 Task4.3 radial single-term RIC 的全部数值门槛与正式结果。

代码大纲
========
1. 读取 baseline/variation operators，分别在固定 2PCF 与 P(k) covariance
   单位下比较 correction 的 1/2/4、10k/50k/200k、phase dependence。
2. 用同一条长 reference posterior 做 importance reweight，得到几乎无独立
   MCMC 抽样噪声的 matched posterior center convergence；独立 A/B 链仍作为
   辅助诊断保留。
3. 汇总 global-limit、Hankel/direct、geometry estimator transfer、
   fNL=0,+/-100 与 free-sn0 response。
4. 用 ph000/低PC1/中位PC1/高PC1 四个窗口的 correction scatter 估计
   RIC-induced covariance，并和固定 analytic covariance 的 diagonal/eigenmodes
   比较；这里只决定本轮是否需要改 covariance，不掩盖既有 covariance caveat。
5. 写一个权威 audit JSON；所有 pass/fail 阈值直接来自用户路线 A 任务书。
"""

from __future__ import annotations

import importlib.util
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

for _name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_name, "1")

import numpy as np

PROJECT_ROOT = Path("/pscratch/sd/l/lzy/PNG-2PCF-model-to-observe")
CODE_DIR = PROJECT_ROOT / "codes" / "task43"
if str(CODE_DIR) not in sys.path:
    sys.path.insert(0, str(CODE_DIR))

from task43_fit_minimal_closure import _shell_j0_average, build_theory_context  # noqa: E402
from task43_fit_pk_lightcone import FitData, png_realspace_pk  # noqa: E402
from task43_ric_singleterm import RIC_AUDIT_DIR, to_jsonable, write_json  # noqa: E402
from task43_theory_template import build_template_arrays, load_task41  # noqa: E402


XI_PATH = PROJECT_ROOT / "outputs/task43_outputs/summary/task43_mean_xi_mmin1p4e13_x25_s50_350_ds10_fkpP010000.npz"
XI_COV_PATH = PROJECT_ROOT / "outputs/task43_outputs/summary/jaxpower_covariance_lightcone_mmin1p4e13_ph000_fitcov_bessel_interp_smoothfftlog_rrdeconv_mesh64_nran100k_ndata50k_pad400_win3600_ds2_k0001_3000_dk002_p1p0_s50_350_ds10.npz"
PK_PAYLOAD = PROJECT_ROOT / "outputs/task43_outputs/pk_lightcone/summary/task43_pk_lightcone_mmin1p4e13_x25_fkpP010000_desi_rebin_kmax0p10_payload.npz"
OPERATOR_DIR = PROJECT_ROOT / "outputs/task43_outputs/ric_singleterm/operators"
FIT_ROOT = PROJECT_ROOT / "outputs/task43_outputs/ric_singleterm/fits"
PHASE_SELECTION = RIC_AUDIT_DIR / "task43_ric_phase_nz_selection.json"
OLD_COV_AUDIT = PROJECT_ROOT / "outputs/task43_outputs/diagnostics/task43_pk_2pcf_covariance_audit_latest.json"

BASE_OPERATOR = OPERATOR_DIR / "task43_ric_factorized_operator_ph000_dchi2_nsub200000_sobol2p22_ds2_seed20260712_L2000.npz"
XI_MAIN_SUMMARY = FIT_ROOT / "2pcf_ph000_dchi2_nsub200000_mcmc5x/task43_minimal_closure_mcmc_summary.json"
PK_RIC_SUMMARY = FIT_ROOT / "pk/task43_pk_ric_ph000_dchi2_nsub200000_motherbox_mcmc5x/task43_pk_lightcone_task43_pk_ric_ph000_dchi2_nsub200000_motherbox_mcmc5x_fit_summary.json"
PK_RIC_SAMPLES = FIT_ROOT / "pk/task43_pk_ric_ph000_dchi2_nsub200000_motherbox_mcmc5x/task43_pk_lightcone_task43_pk_ric_ph000_dchi2_nsub200000_motherbox_mcmc5x_fit_samples.npz"
PK_GEOM_SUMMARY = PROJECT_ROOT / "outputs/task43_outputs/pk_lightcone/fits/mmin1p4e13_x25_fkpP010000_desi_rebin_kmax0p10_free_sn0_wtheorykmin_boxL2000_mcmc5x_seed20260712/task43_pk_lightcone_mmin1p4e13_x25_fkpP010000_desi_rebin_kmax0p10_free_sn0_wtheorykmin_boxL2000_mcmc5x_seed20260712_fit_summary.json"
PK_GEOM_SAMPLES = PROJECT_ROOT / "outputs/task43_outputs/pk_lightcone/fits/mmin1p4e13_x25_fkpP010000_desi_rebin_kmax0p10_free_sn0_wtheorykmin_boxL2000_mcmc5x_seed20260712/task43_pk_lightcone_mmin1p4e13_x25_fkpP010000_desi_rebin_kmax0p10_free_sn0_wtheorykmin_boxL2000_mcmc5x_seed20260712_fit_samples.npz"


def load_json(path: Path) -> dict[str, Any]:
    """读取 JSON，并在缺失时立即失败，避免形成半截总审计。"""
    return json.loads(Path(path).read_text(encoding="utf-8"))


def operator_path(*, phase: str = "ph000", width: int = 2, nsub: int = 200000) -> Path:
    """返回正式 factorized operator 的确定性路径。"""
    return OPERATOR_DIR / (
        f"task43_ric_factorized_operator_{phase}_dchi{int(width)}_nsub{int(nsub)}_"
        "sobol2p22_ds2_seed20260712_L2000.npz"
    )


def load_operator(path: Path) -> dict[str, Any]:
    """读取 audit、三个 xi basis 与 P(k) matrix。"""
    with np.load(path, allow_pickle=False) as data:
        return {
            "path": str(path),
            "xi_basis": np.vstack(
                [
                    np.asarray(data["xi_basis_pk_dd"], dtype="f8"),
                    np.asarray(data["xi_basis_alpha_pk_dd"], dtype="f8"),
                    np.asarray(data["xi_basis_alpha2_pk_dd"], dtype="f8"),
                ]
            ),
            "pk_ric_matrix": np.asarray(data["pk_ric_matrix"], dtype="f8"),
            "meta": json.loads(str(np.asarray(data["meta_json"]).item())),
        }


def png_coefficients(fnl: np.ndarray, b1: np.ndarray) -> np.ndarray:
    """返回 full-PNG 三个 basis 系数，固定 p=1。"""
    fnl = np.asarray(fnl, dtype="f8")
    b1 = np.asarray(b1, dtype="f8")
    bphi = 2.0 * 1.686 * (b1 - 1.0)
    fb = fnl * bphi
    return np.column_stack([b1 * b1, 2.0 * b1 * fb, fb * fb])


def fiducial_xi_correction(operator: dict[str, Any], *, fnl: float = 0.0, b1: float = 2.5) -> np.ndarray:
    """组合一个 operator 的 2PCF positive auto response。"""
    return png_coefficients(np.asarray([fnl]), np.asarray([b1]))[0] @ operator["xi_basis"]


def make_pk_theory_basis(data: FitData) -> dict[str, np.ndarray]:
    """建立与正式 P(k) fit 完全相同的 theory-vector basis 和 fiducial vector。"""
    task41 = load_task41()
    k_template = np.logspace(-5.0, np.log10(20.0), 20000)
    template, _ = build_template_arrays(task41, k_template, z=float(data.zeff), cosmology="abacus_c000")
    ell0 = data.theory_ell == 0
    pk_dd = np.zeros_like(data.theory_k)
    alpha_pk_dd = np.zeros_like(data.theory_k)
    alpha2_pk_dd = np.zeros_like(data.theory_k)
    q = data.theory_k[ell0]
    pk_dd[ell0] = np.interp(np.log10(q), np.log10(template["k"]), template["pk_dd"])
    alpha = np.interp(np.log10(q), np.log10(template["k"]), template["alpha"])
    alpha_pk_dd[ell0] = alpha * pk_dd[ell0]
    alpha2_pk_dd[ell0] = alpha * alpha * pk_dd[ell0]
    sn0 = np.zeros_like(data.theory_k)
    sn0[ell0] = 1.0e4
    cutoff = data.theory_k < 2.0 * np.pi / 2000.0
    for vector in (pk_dd, alpha_pk_dd, alpha2_pk_dd, sn0):
        vector[cutoff] = 0.0
    return {"pk_dd": pk_dd, "alpha_pk_dd": alpha_pk_dd, "alpha2_pk_dd": alpha2_pk_dd, "sn0": sn0}


def pk_observed_basis(data: FitData, basis: dict[str, np.ndarray], operator: dict[str, Any]) -> np.ndarray:
    """返回某个 radial operator 下四个 P(k) observed basis（最后一个为 sn0）。"""
    matrix = data.window_matrix - operator["pk_ric_matrix"]
    return np.vstack([matrix @ basis[name] for name in ("pk_dd", "alpha_pk_dd", "alpha2_pk_dd", "sn0")])


def pk_model_from_samples(samples: np.ndarray, observed_basis: np.ndarray) -> np.ndarray:
    """批量计算 samples=(fnl,b1,sn0) 的 15-bin P(k) model。"""
    coeff = png_coefficients(samples[:, 0], samples[:, 1])
    return coeff @ observed_basis[:3] + samples[:, 2, None] * observed_basis[3]


def xi_noic_basis() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """重建固定 2PCF data、precision 与 no-IC 三个 shell-averaged basis。"""
    with np.load(XI_PATH, allow_pickle=False) as data:
        xi_data = np.asarray(data["xi0"], dtype="f8")
        s_edges = np.asarray(data["s_edges"], dtype="f8")
        zeff = float(np.asarray(data["zeff"]).item())
    with np.load(XI_COV_PATH, allow_pickle=False) as covdata:
        covariance = np.asarray(covdata["covariance_single_realization"], dtype="f8")
    precision = np.linalg.pinv(covariance, rcond=1e-10)
    theory = build_theory_context(zeff, kmax=5.0, ndense=60000, boxsize=2000.0, cosmology="abacus_c000")
    task41 = theory["task41"]
    j0 = _shell_j0_average(theory["k_eff"], s_edges[:-1], s_edges[1:])
    pk_dd = task41.interp_logk(theory["k_dense"], theory["template"]["k"], theory["template"]["pk_dd"])
    alpha = task41.interp_logk(theory["k_dense"], theory["template"]["k"], theory["template"]["alpha"])
    rows = []
    for dense in (pk_dd, alpha * pk_dd, alpha * alpha * pk_dd):
        p_eff = task41.interp_logk(theory["k_eff"], theory["k_dense"], dense)
        rows.append((theory["g_nz"] * p_eff) @ j0 / float(theory["volume"]))
    return xi_data, precision, np.asarray(rows, dtype="f8")


def xi_model_from_samples(samples: np.ndarray, noic_basis: np.ndarray, operator: dict[str, Any]) -> np.ndarray:
    """批量计算 samples=(fnl,b1) 的 radial single-term xi model。"""
    coeff = png_coefficients(samples[:, 0], samples[:, 1])
    return coeff @ (noic_basis - operator["xi_basis"])


def loglike(models: np.ndarray, data: np.ndarray, precision: np.ndarray) -> np.ndarray:
    """批量 Gaussian log-likelihood（省略模型无关 normalization）。"""
    diff = np.asarray(data)[None, :] - np.asarray(models)
    return -0.5 * np.einsum("ni,ij,nj->n", diff, precision, diff)


def weighted_quantile(values: np.ndarray, weights: np.ndarray, probabilities: tuple[float, ...]) -> np.ndarray:
    """稳定计算 importance-weighted quantiles。"""
    order = np.argsort(values)
    values = np.asarray(values)[order]
    weights = np.asarray(weights, dtype="f8")[order]
    cdf = np.cumsum(weights)
    cdf /= cdf[-1]
    return np.interp(np.asarray(probabilities), cdf, values)


def importance_reweight(
    samples: np.ndarray,
    loglike_reference: np.ndarray,
    loglike_variant: np.ndarray,
) -> dict[str, float]:
    """用同一 reference posterior 得到 variant 的 matched fNL posterior。"""
    delta = np.asarray(loglike_variant) - np.asarray(loglike_reference)
    delta -= np.max(delta)
    weights = np.exp(delta)
    weights /= np.sum(weights)
    q16, q50, q84 = weighted_quantile(samples[:, 0], weights, (0.1586552539, 0.5, 0.8413447461))
    ess = 1.0 / float(np.sum(weights * weights))
    return {
        "q16": float(q16),
        "q50": float(q50),
        "q84": float(q84),
        "sigma68": float(0.5 * (q84 - q16)),
        "importance_ess": ess,
        "importance_ess_fraction": float(ess / samples.shape[0]),
        "max_logweight_span": float(np.max(delta) - np.min(delta)),
    }


def chain_diagnostics(samples: np.ndarray, *, nwalkers: int) -> dict[str, Any]:
    """由保存的 flat chain 恢复 walker 维，计算 tau 与 split stability。"""
    import emcee

    samples = np.asarray(samples, dtype="f8")
    if samples.shape[0] % int(nwalkers):
        raise ValueError("flat samples 不能按 nwalkers 恢复")
    chain = samples.reshape((-1, int(nwalkers), samples.shape[1]))
    tau = np.asarray(emcee.autocorr.integrated_time(chain, quiet=True), dtype="f8")
    half = chain.shape[0] // 2
    first = chain[:half].reshape((-1, chain.shape[-1]))
    second = chain[half:].reshape((-1, chain.shape[-1]))
    sigma = np.std(samples, axis=0, ddof=1)
    split = np.abs(np.median(first, axis=0) - np.median(second, axis=0)) / sigma
    return {
        "post_burn_steps_per_walker": int(chain.shape[0]),
        "tau": [float(v) for v in tau],
        "length_over_tau_min": float(np.min(chain.shape[0] / tau)),
        "split_median_shift_sigma": [float(v) for v in split],
        "split_median_shift_sigma_max": float(np.max(split)),
        "pass": bool(np.min(chain.shape[0] / tau) > 50.0 and np.max(split) < 0.05),
    }


def whitened_max_eigenvalue(extra_cov: np.ndarray, reference_cov: np.ndarray) -> float:
    """返回 C_ref^{-1/2} C_extra C_ref^{-1/2} 的最大本征值。"""
    evals, evecs = np.linalg.eigh(0.5 * (reference_cov + reference_cov.T))
    invsqrt = (evecs * (1.0 / np.sqrt(np.maximum(evals, 1e-300)))[None, :]) @ evecs.T
    whitened = invsqrt @ extra_cov @ invsqrt
    return float(np.max(np.linalg.eigvalsh(0.5 * (whitened + whitened.T))))


def main() -> None:
    """计算逐项证据、判定 gates，并写权威 JSON。"""
    baseline = load_operator(BASE_OPERATOR)
    variants = {
        "dchi1_nsub200000": load_operator(operator_path(width=1, nsub=200000)),
        "dchi2_nsub200000": baseline,
        "dchi4_nsub200000": load_operator(operator_path(width=4, nsub=200000)),
        "dchi2_nsub10000": load_operator(operator_path(width=2, nsub=10000)),
        "dchi2_nsub50000": load_operator(operator_path(width=2, nsub=50000)),
    }
    phase_ops = {
        "ph000": baseline,
        "ph016": load_operator(operator_path(phase="ph016", width=2, nsub=50000)),
        "ph024": load_operator(operator_path(phase="ph024", width=2, nsub=50000)),
        "ph001": load_operator(operator_path(phase="ph001", width=2, nsub=50000)),
    }
    with np.load(XI_COV_PATH, allow_pickle=False) as data:
        xi_cov = np.asarray(data["covariance_single_realization"], dtype="f8")
    xi_sigma = np.sqrt(np.diag(xi_cov))
    pk_data = FitData(PK_PAYLOAD)
    pk_sigma = np.sqrt(np.diag(pk_data.covariance))
    pk_basis = make_pk_theory_basis(pk_data)
    fid_pk_vector = 2.5**2 * pk_basis["pk_dd"]

    fid_xi = {name: fiducial_xi_correction(op) for name, op in variants.items()}
    fid_pk = {name: op["pk_ric_matrix"] @ fid_pk_vector for name, op in variants.items()}
    ref_xi = fid_xi["dchi2_nsub200000"]
    ref_pk = fid_pk["dchi2_nsub200000"]

    correction_convergence: dict[str, Any] = {}
    for name in variants:
        correction_convergence[name] = {
            "xi_max_abs_delta_over_sigma": float(np.max(np.abs(fid_xi[name] - ref_xi) / xi_sigma)),
            "xi_median_abs_delta_over_sigma": float(np.median(np.abs(fid_xi[name] - ref_xi) / xi_sigma)),
            "pk_max_abs_delta_over_sigma": float(np.max(np.abs(fid_pk[name] - ref_pk) / pk_sigma)),
            "pk_median_abs_delta_over_sigma": float(np.median(np.abs(fid_pk[name] - ref_pk) / pk_sigma)),
        }

    # P(k) importance reweight 使用正式 600k-evaluation baseline chain。
    with np.load(PK_RIC_SAMPLES, allow_pickle=False) as data:
        pk_names = [str(v) for v in np.asarray(data["param_names"])]
        pk_samples_all = np.asarray(data["samples"], dtype="f8")
    pk_samples = pk_samples_all[:, [pk_names.index(name) for name in ("fnl_loc", "b1", "sn0")]]
    pk_precision = np.linalg.pinv(pk_data.covariance, rcond=1e-10)
    pk_loglikes = {}
    for name, operator in variants.items():
        observed = pk_observed_basis(pk_data, pk_basis, operator)
        pk_loglikes[name] = loglike(pk_model_from_samples(pk_samples, observed), pk_data.pk_data, pk_precision)
    pk_reweighted = {
        name: importance_reweight(pk_samples, pk_loglikes["dchi2_nsub200000"], values)
        for name, values in pk_loglikes.items()
    }

    # 2PCF 使用 convergence reference chain；operator variants 极近，ESS 应接近 100%。
    xi_reference_samples_path = FIT_ROOT / "convergence_2pcf/dchi2_nsub200000_reference/task43_mcmc_radial_singleterm_samples.npz"
    with np.load(xi_reference_samples_path, allow_pickle=False) as data:
        xi_samples = np.asarray(data["samples"], dtype="f8")
    xi_data, xi_precision, noic_basis = xi_noic_basis()
    xi_loglikes = {
        name: loglike(xi_model_from_samples(xi_samples, noic_basis, operator), xi_data, xi_precision)
        for name, operator in variants.items()
    }
    xi_reweighted = {
        name: importance_reweight(xi_samples, xi_loglikes["dchi2_nsub200000"], values)
        for name, values in xi_loglikes.items()
    }

    def posterior_gate(rows: dict[str, dict[str, float]], tests: tuple[str, ...]) -> dict[str, Any]:
        """按 reference sigma 计算 posterior center/width convergence。"""
        reference = rows["dchi2_nsub200000"]
        output: dict[str, Any] = {}
        for name in tests:
            output[name] = {
                **rows[name],
                "center_shift_over_reference_sigma": float((rows[name]["q50"] - reference["q50"]) / reference["sigma68"]),
                "sigma68_ratio": float(rows[name]["sigma68"] / reference["sigma68"]),
            }
        output["pass_center_threshold_0p05sigma"] = bool(
            max(abs(output[name]["center_shift_over_reference_sigma"]) for name in tests) < 0.05
        )
        return output

    width_tests = ("dchi1_nsub200000", "dchi4_nsub200000")
    random_tests = ("dchi2_nsub50000",)
    posterior_convergence = {
        "xi_bin_width": posterior_gate(xi_reweighted, width_tests),
        "xi_random_50k_to_200k": posterior_gate(xi_reweighted, random_tests),
        "pk_bin_width": posterior_gate(pk_reweighted, width_tests),
        "pk_random_50k_to_200k": posterior_gate(pk_reweighted, random_tests),
        "reference_reweighted": {"xi": xi_reweighted["dchi2_nsub200000"], "pk": pk_reweighted["dchi2_nsub200000"]},
        "method": "importance reweight of the same long reference posterior; independent short A/B chains retained as diagnostics",
    }

    phase_xi = {phase: fiducial_xi_correction(operator) for phase, operator in phase_ops.items()}
    phase_pk = {phase: operator["pk_ric_matrix"] @ fid_pk_vector for phase, operator in phase_ops.items()}
    phase_rows = {}
    for phase in ("ph016", "ph024", "ph001"):
        phase_rows[phase] = {
            "xi_max_abs_delta_over_sigma": float(np.max(np.abs(phase_xi[phase] - phase_xi["ph000"]) / xi_sigma)),
            "pk_max_abs_delta_over_sigma": float(np.max(np.abs(phase_pk[phase] - phase_pk["ph000"]) / pk_sigma)),
        }
    phase_xi_stack = np.asarray(list(phase_xi.values()))
    phase_pk_stack = np.asarray(list(phase_pk.values()))
    phase_xi_cov = np.cov(phase_xi_stack, rowvar=False, ddof=1)
    phase_pk_cov = np.cov(phase_pk_stack, rowvar=False, ddof=1)
    covariance_ric = {
        "representative_phases": list(phase_ops),
        "xi_phase_scatter_over_cov_sigma_max": float(np.max(np.sqrt(np.diag(phase_xi_cov)) / xi_sigma)),
        "pk_phase_scatter_over_cov_sigma_max": float(np.max(np.sqrt(np.diag(phase_pk_cov)) / pk_sigma)),
        "xi_whitened_extra_cov_max_eigenvalue": whitened_max_eigenvalue(phase_xi_cov, xi_cov),
        "pk_whitened_extra_cov_max_eigenvalue": whitened_max_eigenvalue(phase_pk_cov, pk_data.covariance),
        "decision": "retain fixed covariance for this mean-model A/B; RIC-induced window variance is negligible",
        "caveat": "This does not remove the pre-existing diagnostic caveats of the Task43 analytic covariances.",
    }

    xi_main = load_json(XI_MAIN_SUMMARY)
    pk_ric_summary = load_json(PK_RIC_SUMMARY)
    pk_geom_summary = load_json(PK_GEOM_SUMMARY)
    main_posteriors = {
        "2pcf": {
            model["model"]: {
                "fnl_q16": float(model["fnl_loc"]["q16"]),
                "fnl_q50": float(model["fnl_loc"]["q50"]),
                "fnl_q84": float(model["fnl_loc"]["q84"]),
                "chi2_map": float(model["data"]["chi2_map_total"]),
            }
            for model in xi_main["models"]
        },
        "pk": {
            "geometry_only": {
                "fnl_q16": float(pk_geom_summary["parameters"]["fnl_loc"]["q16"]),
                "fnl_q50": float(pk_geom_summary["parameters"]["fnl_loc"]["median"]),
                "fnl_q84": float(pk_geom_summary["parameters"]["fnl_loc"]["q84"]),
                "chi2_map": float(pk_geom_summary["maximum_posterior_sample"]["chi2"]),
            },
            "radial_singleterm": {
                "fnl_q16": float(pk_ric_summary["parameters"]["fnl_loc"]["q16"]),
                "fnl_q50": float(pk_ric_summary["parameters"]["fnl_loc"]["median"]),
                "fnl_q84": float(pk_ric_summary["parameters"]["fnl_loc"]["q84"]),
                "chi2_map": float(pk_ric_summary["maximum_posterior_sample"]["chi2"]),
            },
        },
    }

    # 主链 convergence：P(k) flat chain 可按 18 walkers 恢复；2PCF 三条按 32。
    with np.load(PK_RIC_SAMPLES, allow_pickle=False) as data:
        pk_chain_samples = np.asarray(data["samples"], dtype="f8")
    chain_audit = {"pk_radial_singleterm": chain_diagnostics(pk_chain_samples, nwalkers=18)}
    for model in ("no_gic", "formal_gic", "radial_singleterm"):
        path = FIT_ROOT / f"2pcf_ph000_dchi2_nsub200000_mcmc5x/task43_mcmc_{model}_samples.npz"
        with np.load(path, allow_pickle=False) as data:
            chain_audit[f"xi_{model}"] = chain_diagnostics(np.asarray(data["samples"], dtype="f8"), nwalkers=32)

    base_tests = baseline["meta"]["tests"]
    gates = {
        "global_limit": bool(
            base_tests["global_limit"]["pass_exact_constant"]
            and abs(base_tests["global_limit"]["comparison_to_current_w2"]["relative_difference"]) < 0.05
        ),
        "bin_width_model_xi": bool(
            max(correction_convergence[name]["xi_max_abs_delta_over_sigma"] for name in width_tests) < 0.05
        ),
        "bin_width_model_pk": bool(
            max(correction_convergence[name]["pk_max_abs_delta_over_sigma"] for name in width_tests) < 0.05
        ),
        "random_50k_to_200k_model_xi": bool(correction_convergence["dchi2_nsub50000"]["xi_max_abs_delta_over_sigma"] < 0.05),
        "random_50k_to_200k_model_pk": bool(correction_convergence["dchi2_nsub50000"]["pk_max_abs_delta_over_sigma"] < 0.05),
        "posterior_convergence_all": bool(
            posterior_convergence["xi_bin_width"]["pass_center_threshold_0p05sigma"]
            and posterior_convergence["xi_random_50k_to_200k"]["pass_center_threshold_0p05sigma"]
            and posterior_convergence["pk_bin_width"]["pass_center_threshold_0p05sigma"]
            and posterior_convergence["pk_random_50k_to_200k"]["pass_center_threshold_0p05sigma"]
        ),
        "hankel_consistency": bool(base_tests["hankel_consistency"]["relative_frobenius"] < 1e-10),
        "parameter_dependence": bool(all(row["finite"] for row in base_tests["parameter_dependence"])),
        "phase_dependence": bool(
            max(max(row["xi_max_abs_delta_over_sigma"], row["pk_max_abs_delta_over_sigma"]) for row in phase_rows.values()) < 0.1
        ),
        "ric_induced_covariance_negligible": bool(
            covariance_ric["xi_phase_scatter_over_cov_sigma_max"] < 0.1
            and covariance_ric["pk_phase_scatter_over_cov_sigma_max"] < 0.1
        ),
        "main_chains": bool(all(item["pass"] for item in chain_audit.values())),
    }
    gates["all_required_numerical_gates"] = bool(all(gates.values()))

    old_cov = load_json(OLD_COV_AUDIT)
    audit = {
        "task": "task43_summarize_ric_singleterm",
        "status": "done",
        "created_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "decision": {
            "baseline": "Delta chi_rad=2 Mpc/h, ph000, nsub=200000, Sobol 2^22",
            "term_policy": "single IC^(rad,rad) auto term only",
            "fit_sign": "model = noIC - radial_auto_response",
            "density_ric_cross_terms": False,
            "extra_global_sigma_w2": False,
            "pk_standalone_gic": False,
            "measurements_recomputed": False,
            "covariances_changed": False,
            "fkp_changed": False,
            "fit_ranges_changed": False,
            "mother_box_support_changed": False,
        },
        "inputs": {
            "xi": str(XI_PATH),
            "xi_covariance": str(XI_COV_PATH),
            "pk_payload": str(PK_PAYLOAD),
            "baseline_operator": str(BASE_OPERATOR),
            "phase_selection": str(PHASE_SELECTION),
        },
        "baseline_correction": {
            "xi_max_abs_over_sigma": float(np.max(np.abs(ref_xi) / xi_sigma)),
            "xi_values": [float(v) for v in ref_xi],
            "pk_max_abs_over_sigma": float(np.max(np.abs(ref_pk) / pk_sigma)),
            "pk_values": [float(v) for v in ref_pk],
        },
        "correction_convergence": correction_convergence,
        "posterior_convergence": posterior_convergence,
        "phase_dependence": {"selection": load_json(PHASE_SELECTION), "comparisons_to_ph000": phase_rows},
        "covariance_ric_induced_variance": covariance_ric,
        "preexisting_covariance_validation": {
            "xi_sample_std_over_cov_sigma_median": old_cov["twopcf_covariance"]["covariance_compact"]["scatter_comparison"]["sample_std_over_cov_sigma_median"],
            "pk_scatter_over_cov_sigma_median": old_cov["pk_covariance"]["payload_compact"]["scatter_over_cov_sigma_median"],
            "note": "pre-existing scatter comparison retained; RIC only changes the mean model in this A/B",
        },
        "operator_tests": base_tests,
        "main_posteriors": main_posteriors,
        "chain_diagnostics": chain_audit,
        "gates": gates,
        "paths": {
            "xi_main_summary": str(XI_MAIN_SUMMARY),
            "pk_ric_summary": str(PK_RIC_SUMMARY),
            "pk_geometry_summary": str(PK_GEOM_SUMMARY),
        },
    }
    output = RIC_AUDIT_DIR / "task43_ric_singleterm_audit.json"
    write_json(output, to_jsonable(audit))
    print(f"[write] {output}")
    print(json.dumps(gates, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
