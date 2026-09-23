#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""汇总 Task43 2PCF rmin=30/40/50 matched scan。

执行逻辑大纲：
1. 读取三条 raw MCMC chain，统一重算 68.27% posterior 与收敛诊断。
2. 比较 `sigma(b1)`、`sigma(fNL)`、中心和 `corr(fNL,b1)` 随 rmin 的变化。
3. 对新增低-r bins 计算 Schur conditional residual 与 mean-data PTE。
4. 联合 Fisher 诊断和固定 P(k) baseline，分类 b1 变宽的主要来源。
5. 只有 chain 与输入硬门槛通过时才原子写 JSON/CSV/PDF；图只保存 PDF。
"""

from __future__ import annotations

import argparse
import csv
import io
import json
import os
import tempfile
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import chi2 as chi2_distribution

from task43_summarize_rmax_scan import load_chain


ROOT = Path("/pscratch/sd/l/lzy/PNG-2PCF-model-to-observe")
SCAN = ROOT / "outputs/task43_outputs/rmin_scan"
MOVED = ROOT / "plots/outputs/task43_outputs"
RMINS = (30, 40, 50)
DEFAULT_XI = SCAN / "summary/task43_mean_xi_mmin1p4e13_x25_s30_350_ds10_fkpP010000.npz"
DEFAULT_COV = SCAN / "covariance/jaxpower_covariance_lightcone_mmin1p4e13_ph000_fitcov_bessel_interp_smoothfftlog_rrdeconv_fkpNorm4p8925e10_mesh64_nran100k_ndata50k_pad400_win3600_ds2_k0001_3000_dk002_p1p0_s30_350_ds10.npz"
DEFAULT_FISHER = SCAN / "fisher/task43_rmin_scan_fisher_information.json"
DEFAULT_PK_SUMMARY = MOVED / "ric_singleterm/fits/pk/task43_pk_ric_ph000_dchi2_nsub200000_motherbox_long_mcmc50k/task43_pk_ric_ph000_dchi2_nsub200000_motherbox_long_mcmc50k/task43_pk_lightcone_task43_pk_ric_ph000_dchi2_nsub200000_motherbox_long_mcmc50k_fit_summary.json"
DEFAULT_OUTPUT_JSON = SCAN / "audits/task43_jaxpower_2pcf_rmin_scan_longchain.json"
DEFAULT_OUTPUT_CSV = SCAN / "audits/task43_jaxpower_2pcf_rmin_scan_longchain.csv"
DEFAULT_OUTPUT_PDF = ROOT / "plots/task43/rmin_scan/task43_jaxpower_2pcf_rmin_scan_b1_fnl.pdf"


def atomic_text(path: Path, text: str) -> None:
    """原子写入小型文本文件。"""

    path.parent.mkdir(parents=True, exist_ok=True)
    handle, temporary = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(handle, "w", encoding="utf-8", newline="") as stream:
            stream.write(text)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    except Exception:
        Path(temporary).unlink(missing_ok=True)
        raise


def conditional_low_bin(
    residual: np.ndarray,
    covariance: np.ndarray,
    *,
    added_index: int,
    retained_indices: np.ndarray,
    nreal: int,
) -> dict[str, Any]:
    """计算一个新增低-r bin 在既有高-r bins 条件下的 Schur residual。

    参数：residual/covariance 使用完整 32-bin 坐标；added_index 为新增 bin，
    retained_indices 为已经存在的高-r bins，nreal 为 mean-data realization 数。
    返回：conditional residual、Schur variance、chi2 与 PTE。
    """

    retained = np.asarray(retained_indices, dtype="i8")
    c_aa = covariance[np.ix_(retained, retained)]
    c_ba = covariance[np.ix_([added_index], retained)]
    c_ab = c_ba.T
    c_bb = covariance[np.ix_([added_index], [added_index])]
    r_a = residual[retained]
    r_b = residual[[added_index]]
    conditional = r_b - c_ba @ np.linalg.solve(c_aa, r_a)
    schur = c_bb - c_ba @ np.linalg.solve(c_aa, c_ab)
    variance = float(schur[0, 0])
    if variance <= 0.0 or not np.isfinite(variance):
        raise RuntimeError(f"Schur variance 非正：{variance}")
    raw_chi2 = float(conditional[0] ** 2 / variance)
    scaled_chi2 = float(nreal * raw_chi2)
    pte = float(chi2_distribution.sf(scaled_chi2, 1))
    return {
        "added_index": added_index,
        "added_center": float(35.0 + 10.0 * added_index),
        "conditional_residual": float(conditional[0]),
        "schur_variance": variance,
        "raw_chi2_single_lightcone_covariance": raw_chi2,
        "mean_data_scaled_chi2": scaled_chi2,
        "nreal_scale": nreal,
        "df": 1,
        "pte": pte,
        "pass_pte_gt_0p01": bool(pte > 0.01),
    }


def save_pdf(
    path: Path,
    rows: list[dict[str, Any]],
    pk_sigma_b1: float,
    pk_sigma_fnl: float,
    fisher: dict[str, Any],
) -> None:
    """生成唯一的 PDF 汇总图，不创建 PNG。

    参数：path 为 PDF 路径，rows 为三个 rmin posterior，两个 pk_sigma 为
    固定 P(k) baseline 的参考误差，fisher 提供新增低-r bins 的局部信息份额。
    """

    path.parent.mkdir(parents=True, exist_ok=True)
    rmin = np.asarray([row["rmin"] for row in rows], dtype="f8")
    sigma_b1 = np.asarray([row["posterior"]["b1"]["sigma68"] for row in rows])
    sigma_fnl = np.asarray([row["posterior"]["fnl_loc"]["sigma68"] for row in rows])
    low_r_information = fisher["xi"]["low_r_b1_information"]
    information_fraction = np.asarray(
        [
            low_r_information["fraction_already_in_rmin50"],
            low_r_information["fraction_added_by_center45"],
            low_r_information["fraction_added_by_center35"],
        ],
        dtype="f8",
    )
    if not np.all(np.isfinite(information_fraction)) or not np.isclose(
        np.sum(information_fraction), 1.0, rtol=0.0, atol=1.0e-8
    ):
        raise RuntimeError(f"Fisher b1 信息份额未闭合：{information_fraction}")

    fig, axes = plt.subplots(1, 3, figsize=(15.2, 4.4))
    axes[0].plot(rmin, sigma_b1, "o-", color="#C44E52", lw=2.2, ms=7, label=r"$\xi_0(s)$")
    axes[0].axhline(pk_sigma_b1, color="#2F2F2F", lw=2.0, ls="--", label=r"$P_0(k)$")
    axes[0].set_xlabel(r"$r_{\min}\,[h^{-1}{\rm Mpc}]$")
    axes[0].set_ylabel(r"$\sigma(b_1)$")
    axes[0].legend(frameon=False)
    axes[1].plot(rmin, sigma_fnl, "o-", color="#C44E52", lw=2.2, ms=7, label=r"$\xi_0(s)$")
    axes[1].axhline(pk_sigma_fnl, color="#2F2F2F", lw=2.0, ls="--", label=r"$P_0(k)$")
    axes[1].set_xlabel(r"$r_{\min}\,[h^{-1}{\rm Mpc}]$")
    axes[1].set_ylabel(r"$\sigma(f_{\rm NL}^{\rm loc})$")
    axes[2].bar(
        [r"$r\geq 50$", r"$45$ bin", r"$35$ bin"],
        information_fraction,
        color=["#4C72B0", "#DD8452", "#C44E52"],
        width=0.72,
    )
    axes[2].set_ylabel(r"fraction of marginalized $b_1$ Fisher information")
    axes[2].set_ylim(0.0, max(1.0, 1.12 * float(np.max(information_fraction))))
    for index, value in enumerate(information_fraction):
        axes[2].text(index, value + 0.025, f"{100.0 * value:.1f}%", ha="center", va="bottom")
    for axis in axes[:2]:
        axis.grid(alpha=0.22)
        axis.set_xticks([30, 40, 50])
        axis.invert_xaxis()
    axes[2].grid(axis="y", alpha=0.22)
    fig.tight_layout()
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp.pdf")
    fig.savefig(temporary)
    plt.close(fig)
    temporary.replace(path)


def main() -> None:
    """运行最终 posterior、conditional residual 与解释审计。"""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--xi", type=Path, default=DEFAULT_XI)
    parser.add_argument("--covariance", type=Path, default=DEFAULT_COV)
    parser.add_argument("--fisher", type=Path, default=DEFAULT_FISHER)
    parser.add_argument("--fit-root", type=Path, default=SCAN / "fits")
    parser.add_argument("--pk-summary", type=Path, default=DEFAULT_PK_SUMMARY)
    parser.add_argument(
        "--expected-nreal",
        type=int,
        default=25,
        help="正式结果固定为 25；隔离 preliminary 子集必须显式传入其冻结 realization 数。",
    )
    parser.add_argument("--output-json", type=Path, default=DEFAULT_OUTPUT_JSON)
    parser.add_argument("--output-csv", type=Path, default=DEFAULT_OUTPUT_CSV)
    parser.add_argument("--output-pdf", type=Path, default=DEFAULT_OUTPUT_PDF)
    args = parser.parse_args()

    for path in (args.xi, args.covariance, args.fisher, args.pk_summary):
        if not path.is_file() or path.stat().st_size == 0:
            raise FileNotFoundError(path)
    with np.load(args.xi, allow_pickle=False) as data:
        centers = np.asarray(data["s"], dtype="f8")
        xi_mean = np.asarray(data["xi0"], dtype="f8")
        nreal = int(np.asarray(data["nreal"]).item())
    with np.load(args.covariance, allow_pickle=False) as data:
        covariance = np.asarray(data["covariance_single_realization"], dtype="f8")
    if args.expected_nreal < 1:
        raise ValueError("--expected-nreal 必须为正整数")
    if (
        centers.shape != (32,)
        or xi_mean.shape != (32,)
        or covariance.shape != (32, 32)
        or nreal != args.expected_nreal
    ):
        raise RuntimeError("xi/covariance/nreal 不满足 32-bin contract")

    rows: list[dict[str, Any]] = []
    residual_by_rmin: dict[int, np.ndarray] = {}
    for rmin in RMINS:
        fit_dir = args.fit_root / f"rmin{rmin}"
        summary_path = fit_dir / "task43_minimal_closure_mcmc_summary.json"
        samples_path = fit_dir / "task43_mcmc_radial_singleterm_samples.npz"
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        chain = load_chain(summary, samples_path, require_map_arrays=True)
        if not chain["diagnostics"]["pass"]:
            raise RuntimeError(f"rmin={rmin} chain 未通过收敛门槛：{chain['diagnostics']}")
        expected_nbins = (350 - rmin) // 10
        if int(summary["fit_range"]["nbins"]) != expected_nbins:
            raise RuntimeError(f"rmin={rmin} fit bins 错误")
        mask = centers >= float(rmin)
        residual = np.full(32, np.nan, dtype="f8")
        residual[mask] = np.asarray(chain["residual_map"], dtype="f8")
        residual_by_rmin[rmin] = residual
        rows.append(
            {
                "rmin": rmin,
                "rmax": 350,
                "nbins": expected_nbins,
                "posterior": chain["posterior"],
                "chain_diagnostics": chain["diagnostics"],
                "optimizer_chi2_single_cov": float(summary["models"][0]["optimizer"]["chi2"]),
                "summary": str(summary_path),
                "samples": str(samples_path),
            }
        )

    by_rmin = {row["rmin"]: row for row in rows}
    # rmin 40 新增 center=45(index=1)，条件在旧 50--350(index=2:32) 上；
    # rmin 30 再新增 center=35(index=0)，条件在旧 40--350(index=1:32) 上。
    conditional_40 = conditional_low_bin(
        residual_by_rmin[40], covariance, added_index=1, retained_indices=np.arange(2, 32), nreal=nreal
    )
    conditional_30 = conditional_low_bin(
        residual_by_rmin[30], covariance, added_index=0, retained_indices=np.arange(1, 32), nreal=nreal
    )
    by_rmin[40]["conditional_added_low_bin"] = conditional_40
    by_rmin[30]["conditional_added_low_bin"] = conditional_30

    fisher = json.loads(args.fisher.read_text(encoding="utf-8"))
    pk_summary = json.loads(args.pk_summary.read_text(encoding="utf-8"))
    pk_b1 = pk_summary["parameters"]["b1"]
    pk_fnl = pk_summary["parameters"]["fnl_loc"]
    pk_sigma_b1 = 0.5 * (float(pk_b1["err_low"]) + float(pk_b1["err_high"]))
    pk_sigma_fnl = 0.5 * (float(pk_fnl["err_low"]) + float(pk_fnl["err_high"]))
    sigma_b1_50 = by_rmin[50]["posterior"]["b1"]["sigma68"]
    sigma_b1_30 = by_rmin[30]["posterior"]["b1"]["sigma68"]
    sigma_fnl_50 = by_rmin[50]["posterior"]["fnl_loc"]["sigma68"]
    delta_b1 = by_rmin[30]["posterior"]["b1"]["q50"] - by_rmin[50]["posterior"]["b1"]["q50"]
    delta_fnl = by_rmin[30]["posterior"]["fnl_loc"]["q50"] - by_rmin[50]["posterior"]["fnl_loc"]["q50"]
    b1_shift_sigma = float(delta_b1 / sigma_b1_50)
    fnl_shift_sigma = float(delta_fnl / sigma_fnl_50)
    conditional_pass = bool(conditional_40["pass_pte_gt_0p01"] and conditional_30["pass_pte_gt_0p01"])
    stability_pass = bool(abs(b1_shift_sigma) < 0.3 and abs(fnl_shift_sigma) < 0.3)
    b1_improvement = float(1.0 - sigma_b1_30 / sigma_b1_50)
    b1_gap_denominator = sigma_b1_50 - pk_sigma_b1
    b1_gap_closed = (
        float((sigma_b1_50 - sigma_b1_30) / b1_gap_denominator)
        if abs(b1_gap_denominator) > np.finfo("f8").eps
        else float("nan")
    )
    low_r_information = fisher["xi"]["low_r_b1_information"]

    if b1_improvement >= 0.10 and conditional_pass and stability_pass:
        classification = "lower-r amplitude information materially explains the wider 2PCF b1 constraint"
    elif not conditional_pass or not stability_pass:
        classification = "lower-r bins tighten or shift the fit but fail the pre-registered model-validity gate"
    else:
        classification = "lower-r bins do not materially close the b1 gap; covariance/parameter degeneracy dominates"

    payload = {
        "status": "pass",
        "task": "task43_summarize_rmin_scan",
        "classification": classification,
        "scope": "fixed RR-deconvolved jaxpower covariance and radial single-term RIC; not science-ready",
        "nreal": nreal,
        "cases": rows,
        "pk_reference": {
            "path": str(args.pk_summary),
            "sigma_b1": pk_sigma_b1,
            "sigma_fnl": pk_sigma_fnl,
            "b1": pk_b1,
            "fnl_loc": pk_fnl,
        },
        "headline": {
            "sigma_b1_rmin50": sigma_b1_50,
            "sigma_b1_rmin30": sigma_b1_30,
            "sigma_b1_rmin30_over_rmin50": float(sigma_b1_30 / sigma_b1_50),
            "b1_improvement_fraction": b1_improvement,
            "sigma_b1_rmin50_over_pk": float(sigma_b1_50 / pk_sigma_b1),
            "sigma_b1_rmin30_over_pk": float(sigma_b1_30 / pk_sigma_b1),
            "posterior_b1_width_gap_to_pk_closed_fraction": b1_gap_closed,
            "fisher_b1_information_fraction_already_in_rmin50": float(
                low_r_information["fraction_already_in_rmin50"]
            ),
            "fisher_b1_information_fraction_added_by_center45": float(
                low_r_information["fraction_added_by_center45"]
            ),
            "fisher_b1_information_fraction_added_by_center35": float(
                low_r_information["fraction_added_by_center35"]
            ),
            "b1_median_shift_rmin30_minus50_sigma50": b1_shift_sigma,
            "fnl_median_shift_rmin30_minus50_sigma50": fnl_shift_sigma,
            "conditional_pte_center45": conditional_40["pte"],
            "conditional_pte_center35": conditional_30["pte"],
            "conditional_gate_pass": conditional_pass,
            "parameter_stability_gate_pass": stability_pass,
        },
        "fisher": fisher,
        "outputs": {"json": str(args.output_json), "csv": str(args.output_csv), "pdf": str(args.output_pdf)},
    }
    atomic_text(args.output_json, json.dumps(payload, indent=2, sort_keys=True) + "\n")

    csv_rows: list[dict[str, Any]] = []
    for row in rows:
        posterior = row["posterior"]
        conditional = row.get("conditional_added_low_bin", {})
        fisher_case = fisher["xi"]["cases"][str(row["rmin"])]
        cumulative_case = fisher["xi"]["cumulative_by_rmin"][str(row["rmin"])]
        csv_rows.append(
            {
                "rmin": row["rmin"],
                "nbins": row["nbins"],
                "fnl_q50": posterior["fnl_loc"]["q50"],
                "fnl_sigma68": posterior["fnl_loc"]["sigma68"],
                "b1_q50": posterior["b1"]["q50"],
                "b1_sigma68": posterior["b1"]["sigma68"],
                "corr_fnl_b1": posterior["corr_fnl_b1"],
                "fisher_derivative_corr_fnl_b1": fisher_case["derivative_precision_correlation"][0][1],
                "fisher_covariance_corr_sigma_ratio_b1": fisher_case["covariance_correlation_sigma_ratio"]["b1"],
                "fisher_b1_information_fraction_of_rmin30": cumulative_case[
                    "b1_information_fraction_of_rmin30"
                ],
                "conditional_pte": conditional.get("pte"),
                "chain_length_over_tau_min": row["chain_diagnostics"]["length_over_tau_min"],
                "chain_split_shift_sigma_max": row["chain_diagnostics"]["split_median_shift_sigma_max"],
            }
        )
    buffer = io.StringIO()
    writer = csv.DictWriter(buffer, fieldnames=list(csv_rows[0]))
    writer.writeheader()
    writer.writerows(csv_rows)
    atomic_text(args.output_csv, buffer.getvalue())
    save_pdf(args.output_pdf, rows, pk_sigma_b1, pk_sigma_fnl, fisher)
    if args.output_pdf.suffix.lower() != ".pdf" or not args.output_pdf.is_file():
        raise RuntimeError("PDF-only 输出 gate 失败")
    print(json.dumps(payload["headline"], indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
