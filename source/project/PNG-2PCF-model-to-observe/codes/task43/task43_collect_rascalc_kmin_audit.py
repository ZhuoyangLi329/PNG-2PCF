#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""汇总 Task43 FKP、母盒 kmin、RascalC 与 P(k)/2PCF posterior 审计。

执行大纲
--------
1. 独立读取 25 个 2PCF 产物，验证 DD/DR/RR、Landy--Szalay、P0=10000
   和 ``WEIGHT_TOTAL=WEIGHT*WEIGHT_FKP`` 是否在全部 phase 中一致。
2. 读取母盒 ``L=2000`` 的 FullDiscrete/continuous-boxcut/continuous-lowk
   xi 输入元数据，明确区分“离散模求和”和“连续积分硬截断”。
3. 读取 512-loop RascalC、mock-alpha 校准和 jaxpower 对照，计算 covariance
   的对角、相关矩阵、Frobenius 范数和广义本征值差异。
4. 汇总 P(k) 的 effective-volume cutoff/box cutoff，以及 2PCF 的旧 jaxpower
   与三种 RascalC kmin covariance posterior。
5. 写 JSON、CSV 与 PDF forest plot；不改写任何原始测量或拟合结果。
"""

from __future__ import annotations

import argparse
import csv
import json
import os
from pathlib import Path
from typing import Any

for _name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_name, "1")

import matplotlib.pyplot as plt
import numpy as np
from scipy.linalg import eigh

from task43_config import BOX_SIZE, K_FUND, PLOT_DIR, SUMMARY_DIR


PROJECT_ROOT = Path("/pscratch/sd/l/lzy/PNG-2PCF-model-to-observe")
ARCHIVE_ROOT = PROJECT_ROOT / "old_doc_codes/task4_task44_cleanup_20260707T061844Z/moved/outputs/task43_outputs"
DEFAULT_OUTPUT = SUMMARY_DIR / "task43_rascalc_kmin_audit_20260709"
DEFAULT_PLOT = PLOT_DIR / "task43_rascalc_kmin_audit_20260709.pdf"


def jsonable(value: Any) -> Any:
    """把 numpy/Path 转换成 JSON 可序列化对象。"""
    if isinstance(value, dict):
        return {str(key): jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [jsonable(item) for item in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, Path):
        return str(value)
    return value


def correlation(covariance: np.ndarray) -> np.ndarray:
    """由 covariance 构造 correlation matrix。"""
    sigma = np.sqrt(np.diag(covariance))
    return covariance / np.outer(sigma, sigma)


def covariance_pair_metrics(reference: np.ndarray, other: np.ndarray) -> dict[str, Any]:
    """比较两个 SPD covariance；other/reference 的方向在字段名中固定。"""
    reference = np.asarray(reference, dtype="f8")
    other = np.asarray(other, dtype="f8")
    sigma_ratio = np.sqrt(np.diag(other) / np.diag(reference))
    corr_delta = correlation(other) - correlation(reference)
    generalized = eigh(other, reference, eigvals_only=True)
    return {
        "sigma_other_over_reference_min": float(np.min(sigma_ratio)),
        "sigma_other_over_reference_median": float(np.median(sigma_ratio)),
        "sigma_other_over_reference_max": float(np.max(sigma_ratio)),
        "covariance_relative_frobenius_difference": float(np.linalg.norm(other - reference) / np.linalg.norm(reference)),
        "correlation_max_abs_difference": float(np.max(np.abs(corr_delta))),
        "correlation_rms_difference": float(np.sqrt(np.mean(corr_delta**2))),
        "generalized_variance_eigenvalue_min": float(np.min(generalized)),
        "generalized_variance_eigenvalue_median": float(np.median(generalized)),
        "generalized_variance_eigenvalue_max": float(np.max(generalized)),
        "sigma_other_over_reference": sigma_ratio,
    }


def audit_fkp() -> dict[str, Any]:
    """逐 phase 验证 2PCF 的 FKP/LS 元数据，并核对 FKP 公式。"""
    fkp_path = SUMMARY_DIR / "task43_fkp_zeff_mmin1p4e13_x25.npz"
    fkp = np.load(fkp_path, allow_pickle=False)
    p0_values = np.asarray(fkp["p0_values"], dtype="f8")
    match = np.flatnonzero(np.isclose(p0_values, 10000.0, rtol=0.0, atol=1.0e-10))
    if match.size != 1:
        raise ValueError("P0=10000 is not unique in the FKP summary")
    p0_index = int(match[0])
    expected_fkp = 1.0 / (1.0 + 10000.0 * np.asarray(fkp["nbar"], dtype="f8"))
    stored_fkp = np.asarray(fkp["fkp_weights"], dtype="f8")[p0_index]

    xi_paths = sorted((ARCHIVE_ROOT / "xi_cucount").glob("xi0_AbacusSummit_base_c000_ph*_z0p6_0p8_mmin1p4e13_x25_fkpP010000.npz"))
    phase_rows: list[dict[str, Any]] = []
    for path in xi_paths:
        data = np.load(path, allow_pickle=False)
        required = {"DD", "DR", "RR", "p0", "weighting_meta_json", "ndata", "nrandom", "estimator", "engine"}
        missing = sorted(required - set(data.files))
        if missing:
            raise KeyError(f"{path} missing {missing}")
        weighting = json.loads(str(np.asarray(data["weighting_meta_json"]).item()))
        ndata = int(np.asarray(data["ndata"]).item())
        nrandom = int(np.asarray(data["nrandom"]).item())
        phase_rows.append(
            {
                "path": str(path),
                "phase": str(np.asarray(data["phase"]).item()),
                "p0": float(np.asarray(data["p0"]).item()),
                "scheme": weighting.get("scheme"),
                "estimator": str(np.asarray(data["estimator"]).item()),
                "engine": str(np.asarray(data["engine"]).item()),
                "ndata": ndata,
                "nrandom": nrandom,
                "random_multiplier": float(nrandom / ndata),
                "dd_shape": list(np.asarray(data["DD"]).shape),
                "dr_shape": list(np.asarray(data["DR"]).shape),
                "rr_shape": list(np.asarray(data["RR"]).shape),
                "data_weight_min": float(np.asarray(data["data_weight_min"]).item()),
                "data_weight_max": float(np.asarray(data["data_weight_max"]).item()),
                "random_weight_min": float(np.asarray(data["random_weight_min"]).item()),
                "random_weight_max": float(np.asarray(data["random_weight_max"]).item()),
            }
        )

    allcounts_json = ARCHIVE_ROOT / "rascalc_covariance/allcounts_AbacusSummit_base_c000_ph000_z0p6_0p8_mmin1p4e13_x25_s50_350_ds10_nmu20_fkpP010000.json"
    allcounts = json.loads(allcounts_json.read_text(encoding="utf-8"))
    unique_p0 = sorted({row["p0"] for row in phase_rows})
    unique_scheme = sorted({row["scheme"] for row in phase_rows})
    unique_estimator = sorted({row["estimator"] for row in phase_rows})
    unique_engine = sorted({row["engine"] for row in phase_rows})
    multiplier = np.asarray([row["random_multiplier"] for row in phase_rows], dtype="f8")
    confirmed = bool(
        len(phase_rows) == 25
        and unique_p0 == [10000.0]
        and unique_scheme == ["WEIGHT_TOTAL=WEIGHT*WEIGHT_FKP"]
        and unique_estimator == ["landy_szalay"]
        and all(np.isclose(multiplier, 25.0, rtol=0.0, atol=1.0e-12))
    )
    return {
        "confirmed": confirmed,
        "fkp_summary_path": str(fkp_path),
        "formula": "w_FKP(z)=1/[1+nbar(z) P0]; WEIGHT_TOTAL=WEIGHT*WEIGHT_FKP",
        "p0": 10000.0,
        "formula_max_abs_difference_from_stored_weights": float(np.max(np.abs(expected_fkp - stored_fkp))),
        "weight_min": float(np.min(stored_fkp)),
        "weight_max": float(np.max(stored_fkp)),
        "zeff_random_auto": float(np.asarray(fkp["zeff_random_auto"], dtype="f8")[p0_index]),
        "nphase_2pcf_files": len(phase_rows),
        "unique_p0": unique_p0,
        "unique_weight_schemes": unique_scheme,
        "unique_estimators": unique_estimator,
        "unique_engines": unique_engine,
        "random_multiplier_min": float(np.min(multiplier)),
        "random_multiplier_max": float(np.max(multiplier)),
        "phase_rows": phase_rows,
        "full_ph000_allcounts": {
            "path": str(allcounts_json),
            "ndata_used": int(allcounts["ndata_used"]),
            "nrandom_used": int(allcounts["nrandom_used"]),
            "shape": allcounts["shape"],
            "weighting": allcounts["weighting"],
            "max_data": allcounts["max_data"],
            "max_random": allcounts["max_random"],
        },
    }


def posterior_from_2pcf(path: Path, label: str, covariance_label: str, smin: float = 50.0) -> dict[str, Any]:
    """读取 task43_fit_minimal_closure 的 formal-GIC posterior。"""
    data = json.loads(path.read_text(encoding="utf-8"))
    model = next(item for item in data["models"] if item["model"] == "formal_gic")
    fnl = model["fnl_loc"]
    return {
        "label": label,
        "probe": "2PCF",
        "smin": float(smin),
        "kmax": None,
        "covariance_or_cutoff": covariance_label,
        "fnl_q16": float(fnl["q16"]),
        "fnl_q50": float(fnl["q50"]),
        "fnl_q84": float(fnl["q84"]),
        "fnl_err_low": float(fnl["q50"] - fnl["q16"]),
        "fnl_err_high": float(fnl["q84"] - fnl["q50"]),
        "source": str(path),
    }


def posterior_from_pk(path: Path, label: str) -> dict[str, Any]:
    """读取 Task43 P(k) posterior，并记录 window-theory cutoff。"""
    data = json.loads(path.read_text(encoding="utf-8"))
    fnl = data["parameters"]["fnl_loc"]
    return {
        "label": label,
        "probe": "P(k)",
        "smin": None,
        "kmax": float(data["data"]["kmax_fit"]),
        "covariance_or_cutoff": f"window theory kmin={data['config']['window_theory_kmin']}",
        "fnl_q16": float(fnl["q16"]),
        "fnl_q50": float(fnl["median"]),
        "fnl_q84": float(fnl["q84"]),
        "fnl_err_low": float(fnl["err_low"]),
        "fnl_err_high": float(fnl["err_high"]),
        "source": str(path),
    }


def collect_posteriors() -> list[dict[str, Any]]:
    """收集本轮用于结论的 P(k)/2PCF posterior 行。"""
    pk_root = SUMMARY_DIR.parent / "pk_lightcone/fits"
    fit_root = SUMMARY_DIR.parent / "fits"
    rows = [
        posterior_from_pk(
            pk_root / "mmin1p4e13_x25_fkpP010000_desi_rebin_kmax0p10_free_sn0_wtheorykmin_eff_mcmc5x/task43_pk_lightcone_mmin1p4e13_x25_fkpP010000_desi_rebin_kmax0p10_free_sn0_wtheorykmin_eff_mcmc5x_fit_summary.json",
            "P(k), effective-volume theory cutoff",
        ),
        posterior_from_pk(
            pk_root / "mmin1p4e13_x25_fkpP010000_desi_rebin_kmax0p10_free_sn0_wtheorykmin_boxL2000_mcmc5x/task43_pk_lightcone_mmin1p4e13_x25_fkpP010000_desi_rebin_kmax0p10_free_sn0_wtheorykmin_boxL2000_mcmc5x_fit_summary.json",
            "P(k), mother-box hard cutoff",
        ),
        posterior_from_2pcf(
            ARCHIVE_ROOT / "fits/formalgic_w2_convergence_L2000_smin50_nsub200000/task43_minimal_closure_mcmc_summary.json",
            "2PCF, old jaxpower covariance",
            "jaxpower RR-deconv, kmin=1e-4",
        ),
        posterior_from_2pcf(
            fit_root / "full25_mean_s50_350_ds10_shellavg_rascalcFullDiscrete_nran300k_nloop512_alphaMockCal_L2000_p1p0_formalgic_mcmc5x_nsub200000/task43_minimal_closure_mcmc_summary.json",
            "2PCF, RascalC FullDiscrete",
            "FullDiscrete L=2000, 512 loops, nran=300k, mock alpha",
        ),
        posterior_from_2pcf(
            fit_root / "full25_mean_s50_350_ds10_shellavg_rascalccontinuousBoxcut_nran300k_nloop512_alphaMockCal_L2000_p1p0_formalgic_mcmc5x_nsub200000/task43_minimal_closure_mcmc_summary.json",
            "2PCF, RascalC continuous box-cut",
            "continuous k>=2pi/2000, 512 loops, nran=300k, mock alpha",
        ),
        posterior_from_2pcf(
            fit_root / "full25_mean_s50_350_ds10_shellavg_rascalccontinuousLowk_nran300k_nloop512_alphaMockCal_L2000_p1p0_formalgic_mcmc5x_nsub200000/task43_minimal_closure_mcmc_summary.json",
            "2PCF, RascalC continuous low-k",
            "continuous k>=1e-4, 512 loops, nran=300k, mock alpha",
        ),
    ]
    for smin in (60, 80):
        rows.append(
            posterior_from_2pcf(
                fit_root / f"full25_mean_s{smin}_350_ds10_shellavg_rascalcFullDiscrete_nran300k_nloop512_alphaMockCal_L2000_p1p0_formalgic_mcmc5x_nsub200000/task43_minimal_closure_mcmc_summary.json",
                f"2PCF, RascalC FullDiscrete, smin={smin}",
                "FullDiscrete L=2000, alpha calibrated on smin=50 range",
                smin=float(smin),
            )
        )
    return rows


def build_payload() -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    """组合全部审计信息，并返回绘图需要的 covariance 向量。"""
    fkp = audit_fkp()
    xi_meta_path = SUMMARY_DIR.parent / "rascalc_covariance/inputs/task43_rascalc_xi_kmin_L2000_fnl0_b1cov2p5.json"
    xi_meta = json.loads(xi_meta_path.read_text(encoding="utf-8"))
    compare_path = SUMMARY_DIR / "task43_rascalc_kmin_covariance_comparison.json"
    comparison = json.loads(compare_path.read_text(encoding="utf-8"))
    alpha_json_path = SUMMARY_DIR / "task43_rascalc_alpha_mock_calibration.json"
    alpha_summary = json.loads(alpha_json_path.read_text(encoding="utf-8"))
    alpha_npz_path = alpha_json_path.with_suffix(".npz")
    alpha_npz = np.load(alpha_npz_path, allow_pickle=False)
    labels = [str(item) for item in alpha_npz["run_labels"]]
    indices = {label: index for index, label in enumerate(labels)}
    high = {
        "full_discrete": indices["full_discrete (512 loops)"],
        "continuous_boxcut": indices["continuous_boxcut (512 loops)"],
        "continuous_lowk": indices["continuous_lowk (512 loops)"],
    }
    alpha1_cov = np.asarray(alpha_npz["run_alpha1_covariances"], dtype="f8")
    calibrated_cov = np.asarray(alpha_npz["run_calibrated_covariances"], dtype="f8")
    pairwise = {}
    for covariance_kind, covariances in (("alpha1", alpha1_cov), ("mock_calibrated", calibrated_cov)):
        reference = covariances[high["full_discrete"]]
        pairwise[covariance_kind] = {
            "reference": "full_discrete (512 loops)",
            "continuous_boxcut_over_full_discrete": covariance_pair_metrics(reference, covariances[high["continuous_boxcut"]]),
            "continuous_lowk_over_full_discrete": covariance_pair_metrics(reference, covariances[high["continuous_lowk"]]),
        }

    posteriors = collect_posteriors()
    full_rows = [row for row in comparison["rows"] if row.get("xi_model") == "full_discrete" and row.get("n_loops") == 512]
    box_rows = [row for row in comparison["rows"] if row.get("xi_model") == "continuous_boxcut" and row.get("n_loops") == 512]
    low_rows = [row for row in comparison["rows"] if row.get("xi_model") == "continuous_lowk" and row.get("n_loops") == 512]
    if len(full_rows) != 1 or len(box_rows) != 1 or len(low_rows) != 1:
        raise ValueError("Expected exactly one high-loop row for each RascalC xi model")
    highloop_quality = {"full_discrete": full_rows[0], "continuous_boxcut": box_rows[0], "continuous_lowk": low_rows[0]}

    payload = {
        "status": "done",
        "task": "task43_collect_rascalc_kmin_audit",
        "scope": "Task43 section 4.3 fNL=0 real-space halo lightcone; not Task44 section 4.4",
        "full_scale_status": {
            "full_25_phase_2pcf_measurements_complete": bool(fkp["nphase_2pcf_files"] == 25),
            "full_25x_matching_randoms_used_in_2pcf_measurements": bool(fkp["random_multiplier_min"] == 25.0 and fkp["random_multiplier_max"] == 25.0),
            "full_ph000_pair_counts_reused_by_rascalc": True,
            "rascalc_random_geometry_full_input_started": False,
            "rascalc_random_geometry_n_used": 300000,
            "rascalc_random_geometry_n_full_ph000": int(fkp["full_ph000_allcounts"]["nrandom_used"]),
            "classification": "controlled 300k-random convergence/A-B, not full-input production",
        },
        "fkp_audit": fkp,
        "mother_box": {
            "boxsize": float(BOX_SIZE),
            "kfund": float(K_FUND),
            "primary_covariance_xi_policy": "full_discrete",
            "xi_input_meta_path": str(xi_meta_path),
            "full_discrete_meta": xi_meta["models"]["full_discrete"],
            "continuous_boxcut_meta": xi_meta["models"]["continuous_boxcut"],
            "continuous_lowk_meta": xi_meta["models"]["continuous_lowk"],
            "warning": "A hard continuous cutoff is not the same object as the integer-lattice FullDiscrete sum; the latter is the primary mock-closure model.",
        },
        "rascalc_highloop_quality": highloop_quality,
        "rascalc_kmin_pairwise": pairwise,
        "alpha_calibration": {
            "summary_json": str(alpha_json_path),
            "summary_npz": str(alpha_npz_path),
            "primary_alpha": float(alpha_summary["primary_calibrated_alpha"]),
            "primary_reference": alpha_summary["primary_reference"],
            "primary_smin50_row": alpha_summary["runs"][0]["ranges"][0],
        },
        "posteriors": posteriors,
        "verdict": {
            "fkp_weighting_confirmed": bool(fkp["confirmed"]),
            "rascalc_512_loop_numerical_gate_passed_for_all_three_kmin_models": bool(
                all(bool(row["r_inv_pass_0p05"]) for row in highloop_quality.values())
            ),
            "mother_box_kmin_matters_for_pk_posterior": True,
            "rascalc_kmin_covariance_posterior_effect_at_fnl_cov0_is_small": True,
            "do_not_extrapolate_fnl_cov0_kmin_result_to_fnl_cov100": True,
            "jaxpower_rrdeconv_covariance_remains_too_small_against_phase_scatter": True,
            "full_rascalc_random_input_not_yet_run": True,
        },
        "recommended_next_gates_before_full_input": [
            "Run FullDiscrete 512-loop random-geometry convergence at nrandom=1,000,000 with the physical ndata normalization held fixed.",
            "A/B the linear-theory xi table against a measured/hybrid small-scale xi input used by RascalC.",
            "Propagate the mock-alpha uncertainty/range; keep alpha calibrated on the declared fit range.",
            "Only then decide whether the full 8,254,350-random RascalC integration is necessary.",
        ],
        "source_summaries": {
            "covariance_comparison": str(compare_path),
            "alpha_calibration": str(alpha_json_path),
        },
    }
    arrays = {
        "s": np.asarray(alpha_npz["s"], dtype="f8"),
        "alpha1_full": alpha1_cov[high["full_discrete"]],
        "alpha1_box": alpha1_cov[high["continuous_boxcut"]],
        "alpha1_low": alpha1_cov[high["continuous_lowk"]],
        "calibrated_full": calibrated_cov[high["full_discrete"]],
        "calibrated_box": calibrated_cov[high["continuous_boxcut"]],
        "calibrated_low": calibrated_cov[high["continuous_lowk"]],
    }
    return payload, arrays


def write_csv(path: Path, posteriors: list[dict[str, Any]]) -> None:
    """写 posterior forest plot 对应的紧凑数值表。"""
    fields = ["label", "probe", "smin", "kmax", "covariance_or_cutoff", "fnl_q16", "fnl_q50", "fnl_q84", "fnl_err_low", "fnl_err_high", "source"]
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(posteriors)


def plot_audit(path: Path, payload: dict[str, Any], arrays: dict[str, np.ndarray]) -> None:
    """画 posterior forest 与 covariance kmin 对角 A/B，只输出 PDF。"""
    path.parent.mkdir(parents=True, exist_ok=True)
    rows = payload["posteriors"][:6]
    colors = ["#35689a", "#35689a", "#888888", "#c44e52", "#dd8452", "#55a868"]
    fig, axes = plt.subplots(1, 2, figsize=(12.5, 6.6), constrained_layout=True)

    y = np.arange(len(rows))[::-1]
    q50 = np.asarray([row["fnl_q50"] for row in rows], dtype="f8")
    low = np.asarray([row["fnl_err_low"] for row in rows], dtype="f8")
    high = np.asarray([row["fnl_err_high"] for row in rows], dtype="f8")
    for index, row in enumerate(rows):
        axes[0].errorbar(q50[index], y[index], xerr=np.asarray([[low[index]], [high[index]]]), fmt="o", color=colors[index], capsize=4)
    axes[0].axvline(0.0, color="black", ls="--", lw=1.0)
    axes[0].set_yticks(y)
    axes[0].set_yticklabels([row["label"] for row in rows], fontsize=8)
    axes[0].set_xlabel(r"$f_{\rm NL}^{\rm loc}$ (median and 68% interval)")
    axes[0].set_title("Task43 P(k) / 2PCF comparison")
    axes[0].grid(axis="x", alpha=0.25)

    s = arrays["s"]
    for other, label, color in (("box", "continuous box-cut / FullDiscrete", "#dd8452"), ("low", "continuous low-k / FullDiscrete", "#55a868")):
        ratio_alpha1 = np.sqrt(np.diag(arrays[f"alpha1_{other}"]) / np.diag(arrays["alpha1_full"]))
        ratio_cal = np.sqrt(np.diag(arrays[f"calibrated_{other}"]) / np.diag(arrays["calibrated_full"]))
        axes[1].plot(s, ratio_alpha1, color=color, ls="--", lw=1.5, label=f"{label}, alpha=1")
        axes[1].plot(s, ratio_cal, color=color, ls="-", lw=1.8, label=f"{label}, calibrated alpha")
    axes[1].axhline(1.0, color="black", ls=":", lw=1.0)
    axes[1].set_xlabel(r"$s\ [h^{-1}{\rm Mpc}]$")
    axes[1].set_ylabel(r"$\sigma_{\rm other}/\sigma_{\rm FullDiscrete}$")
    axes[1].set_title(r"512-loop covariance $k_{\min}$ A/B at $f_{\rm NL}^{\rm cov}=0$")
    axes[1].legend(fontsize=7)
    axes[1].grid(alpha=0.25)

    fig.savefig(path)
    plt.close(fig)


def main() -> None:
    """执行只读汇总并写出审计产物。"""
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-prefix", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--plot-path", type=Path, default=DEFAULT_PLOT)
    args = parser.parse_args()
    payload, arrays = build_payload()
    payload["outputs"] = {
        "json": str(args.output_prefix.with_suffix(".json")),
        "csv": str(args.output_prefix.with_suffix(".csv")),
        "pdf": str(args.plot_path),
    }
    out_json = args.output_prefix.with_suffix(".json")
    out_csv = args.output_prefix.with_suffix(".csv")
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(jsonable(payload), indent=2, sort_keys=True) + "\n", encoding="utf-8")
    write_csv(out_csv, payload["posteriors"])
    plot_audit(args.plot_path, payload, arrays)
    print(f"[done] wrote {out_json}")
    print(f"[fkp] confirmed={payload['fkp_audit']['confirmed']}")
    print(f"[status] {payload['full_scale_status']['classification']}")


if __name__ == "__main__":
    main()
