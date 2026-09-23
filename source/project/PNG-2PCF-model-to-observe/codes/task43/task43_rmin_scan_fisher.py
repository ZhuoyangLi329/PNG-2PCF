#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""比较 Task43 2PCF rmin ladder 与 P(k) 的局部 Fisher 信息。

执行逻辑大纲：
1. 用和正式 likelihood 相同的 FullDiscrete、shell-average 与 radial-RIC
   operator 重建 2PCF 模型，并以有限差分求 dmu/dfNL、dmu/db1。
2. 对 rmin=30/40/50 分别计算 full-covariance 与 diagonal-covariance Fisher，
   区分参数退化和 bin-to-bin covariance correlation 的影响。
3. 用现有 P(k) payload、同一 radial-RIC operator 和 free sn0 模型计算
   13-bin kmax=0.08 与 15-bin kmax=0.10 Fisher。
4. 原子输出 JSON/CSV；本脚本只做局部信息诊断，不替代正式 MCMC。
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import tempfile
from pathlib import Path
from typing import Any, Callable

import numpy as np

from task43_fit_minimal_closure import _shell_j0_average, build_theory_context
from task43_fit_pk_lightcone import FitData, model_pk
from task43_theory_template import build_template_arrays, load_task41


ROOT = Path("/pscratch/sd/l/lzy/PNG-2PCF-model-to-observe")
SCAN = ROOT / "outputs/task43_outputs/rmin_scan"
MOVED = ROOT / "plots/outputs/task43_outputs"
DEFAULT_XI = SCAN / "summary/task43_mean_xi_mmin1p4e13_x25_s30_350_ds10_fkpP010000.npz"
DEFAULT_COV = SCAN / "covariance/jaxpower_covariance_lightcone_mmin1p4e13_ph000_fitcov_bessel_interp_smoothfftlog_rrdeconv_fkpNorm4p8925e10_mesh64_nran100k_ndata50k_pad400_win3600_ds2_k0001_3000_dk002_p1p0_s30_350_ds10.npz"
DEFAULT_OPERATOR = SCAN / "operators/task43_ric_factorized_operator_ph000_dchi2_nsub200000_sobol2p22_ds2_seed20260712_L2000_s30_350_ds10.npz"
DEFAULT_PK_PAYLOAD = MOVED / "pk_lightcone/summary/task43_pk_lightcone_mmin1p4e13_x25_fkpP010000_desi_rebin_kmax0p10_payload.npz"
DEFAULT_OUTPUT = SCAN / "fisher/task43_rmin_scan_fisher_information"


def atomic_text(path: Path, text: str) -> None:
    """把小型文本结果原子写入目标路径。"""

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


def finite_difference(
    function: Callable[[np.ndarray], np.ndarray],
    point: np.ndarray,
    steps: np.ndarray,
) -> tuple[np.ndarray, dict[str, float]]:
    """以中心有限差分计算 Jacobian，并用半步长检查数值收敛。

    参数：function 接受参数向量并返回模型向量，point 为展开点，steps 为步长。
    返回：shape=(ndata,nparam) 的 Jacobian 和逐参数半步收敛误差。
    """

    point = np.asarray(point, dtype="f8")
    steps = np.asarray(steps, dtype="f8")
    columns: list[np.ndarray] = []
    convergence: dict[str, float] = {}
    for index, step in enumerate(steps):
        direction = np.zeros_like(point)
        direction[index] = step
        coarse = (function(point + direction) - function(point - direction)) / (2.0 * step)
        direction[index] = 0.5 * step
        fine = (function(point + direction) - function(point - direction)) / step
        denominator = max(float(np.linalg.norm(fine)), np.finfo("f8").tiny)
        convergence[str(index)] = float(np.linalg.norm(coarse - fine) / denominator)
        columns.append(fine)
    return np.column_stack(columns), convergence


def fisher_summary(jacobian: np.ndarray, covariance: np.ndarray, names: list[str]) -> dict[str, Any]:
    """计算 full/diagonal covariance Fisher 与参数退化诊断。

    参数：jacobian 的列对应 names，covariance 为单 realization covariance。
    返回：Fisher、marginalized covariance、sigma、相关系数和信息损失。
    """

    covariance = 0.5 * (np.asarray(covariance, dtype="f8") + np.asarray(covariance, dtype="f8").T)
    precision = np.linalg.pinv(covariance, rcond=1.0e-12)
    fisher = jacobian.T @ precision @ jacobian
    parameter_covariance = np.linalg.pinv(fisher, rcond=1.0e-12)
    sigma = np.sqrt(np.diag(parameter_covariance))
    diagonal_precision = np.diag(1.0 / np.diag(covariance))
    diagonal_fisher = jacobian.T @ diagonal_precision @ jacobian
    diagonal_parameter_covariance = np.linalg.pinv(diagonal_fisher, rcond=1.0e-12)
    diagonal_sigma = np.sqrt(np.diag(diagonal_parameter_covariance))
    denominator = np.sqrt(np.outer(np.diag(parameter_covariance), np.diag(parameter_covariance)))
    correlation = parameter_covariance / denominator
    unmarginalized_sigma = 1.0 / np.sqrt(np.diag(fisher))
    # ``F_ij / sqrt(F_ii F_jj)`` 是两个模型导数在 covariance precision
    # 度量下的夹角；它直接量化观测空间里的模板退化，不应与 posterior
    # parameter correlation 混为一谈（两参数情形二者符号相反）。
    fisher_denominator = np.sqrt(np.outer(np.diag(fisher), np.diag(fisher)))
    derivative_correlation = fisher / fisher_denominator
    return {
        "parameter_names": names,
        "fisher": fisher.tolist(),
        "parameter_covariance": parameter_covariance.tolist(),
        "sigma_marginalized": {name: float(sigma[i]) for i, name in enumerate(names)},
        "sigma_unmarginalized": {name: float(unmarginalized_sigma[i]) for i, name in enumerate(names)},
        "parameter_correlation": correlation.tolist(),
        "derivative_precision_correlation": derivative_correlation.tolist(),
        "information_marginalized": {
            name: float(1.0 / parameter_covariance[i, i]) for i, name in enumerate(names)
        },
        "sigma_diagonal_covariance": {name: float(diagonal_sigma[i]) for i, name in enumerate(names)},
        "covariance_correlation_sigma_ratio": {
            name: float(sigma[i] / diagonal_sigma[i]) for i, name in enumerate(names)
        },
        "condition_number": float(np.linalg.cond(fisher)),
    }


def cumulative_xi_information(
    centers: np.ndarray,
    jacobian: np.ndarray,
    covariance: np.ndarray,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """按从大尺度向小尺度逐 bin 加入的顺序分解 2PCF ``b1`` 信息。

    参数：centers/jacobian/covariance 是完整 32-bin 共同坐标。
    返回：逐 rmin 的累计 Fisher 诊断，以及每个新增低-r bin 对
    ``I(b1 | fNL)=1/sigma_marg(b1)^2`` 的增量；后者包含 covariance
    correlation 和 ``fNL--b1`` 退化，因而比逐 bin ``(dmu/db1)^2/Cii``
    更接近本任务真正比较的 marginalized posterior 宽度。
    """

    centers = np.asarray(centers, dtype="f8")
    jacobian = np.asarray(jacobian, dtype="f8")
    covariance = np.asarray(covariance, dtype="f8")
    if centers.shape != (32,) or jacobian.shape != (32, 2) or covariance.shape != (32, 32):
        raise RuntimeError("累计 xi Fisher 输入不满足 32-bin/2-parameter contract")

    # 从最后两个 bins（rmin edge=330）开始，逐步加入更小 separation。
    # 单 bin 无法同时区分 fnl 与 b1，所以不把 rmin=340 当成有效 seed。
    cumulative: dict[str, Any] = {}
    ordered_cuts = list(range(330, 29, -10))
    previous_information: float | None = None
    increments: list[dict[str, Any]] = []
    for rmin in ordered_cuts:
        mask = centers >= float(rmin)
        summary = fisher_summary(jacobian[mask], covariance[np.ix_(mask, mask)], ["fnl_loc", "b1"])
        information = float(summary["information_marginalized"]["b1"])
        row = {
            "rmin_edge": rmin,
            "nbins": int(np.count_nonzero(mask)),
            "sigma_b1_marginalized": float(summary["sigma_marginalized"]["b1"]),
            "sigma_b1_unmarginalized": float(summary["sigma_unmarginalized"]["b1"]),
            "b1_information_marginalized": information,
            "derivative_precision_correlation_fnl_b1": float(
                summary["derivative_precision_correlation"][0][1]
            ),
            "covariance_correlation_sigma_ratio_b1": float(
                summary["covariance_correlation_sigma_ratio"]["b1"]
            ),
        }
        cumulative[str(rmin)] = row
        if previous_information is None:
            increments.append(
                {
                    "kind": "seed_high_range",
                    "range_edges": [330, 350],
                    "nbins": row["nbins"],
                    "b1_information_marginalized": information,
                }
            )
        else:
            delta = information - previous_information
            tolerance = 1.0e-8 * max(abs(information), abs(previous_information), 1.0)
            if delta < -tolerance:
                raise RuntimeError(
                    f"嵌套 xi Fisher 信息不单调：rmin={rmin}, delta_I_b1={delta}"
                )
            increments.append(
                {
                    "kind": "added_low_r_bin",
                    "added_bin_edges": [rmin, rmin + 10],
                    "added_bin_center": float(rmin + 5),
                    "b1_information_increment_marginalized": float(max(delta, 0.0)),
                }
            )
        previous_information = information

    total_information = float(cumulative["30"]["b1_information_marginalized"])
    if not np.isfinite(total_information) or total_information <= 0.0:
        raise RuntimeError(f"rmin=30 marginalized b1 Fisher 信息非法：{total_information}")
    for row in cumulative.values():
        row["b1_information_fraction_of_rmin30"] = float(
            row["b1_information_marginalized"] / total_information
        )
    for row in increments:
        key = "b1_information_increment_marginalized"
        if key in row:
            row["b1_information_fraction_of_rmin30"] = float(row[key] / total_information)
        else:
            row["b1_information_fraction_of_rmin30"] = float(
                row["b1_information_marginalized"] / total_information
            )
    return cumulative, increments


def build_xi_model(xi_path: Path, operator_path: Path) -> tuple[np.ndarray, Callable[[np.ndarray], np.ndarray], dict[str, Any]]:
    """重建与正式 2PCF likelihood 完全一致的 32-bin radial-RIC 模型。

    参数：xi_path 提供 shell edges/zeff，operator_path 提供 radial bases。
    返回：bin centers、二参数模型函数和 provenance metadata。
    """

    with np.load(xi_path, allow_pickle=False) as data:
        edges = np.asarray(data["s_edges"], dtype="f8")
        centers = np.asarray(data["s"], dtype="f8")
        zeff = float(np.asarray(data["zeff"]).item())
    with np.load(operator_path, allow_pickle=False) as operator:
        target_edges = np.asarray(operator["target_s_edges"], dtype="f8")
        radial = {
            "pk": np.asarray(operator["xi_basis_pk_dd"], dtype="f8"),
            "alpha": np.asarray(operator["xi_basis_alpha_pk_dd"], dtype="f8"),
            "alpha2": np.asarray(operator["xi_basis_alpha2_pk_dd"], dtype="f8"),
        }
    if not np.array_equal(edges, target_edges) or edges.size != 33:
        raise RuntimeError("xi 与 radial operator 不是共同 32-bin 网格")

    context = build_theory_context(zeff, kmax=5.0, ndense=60000, boxsize=2000.0, cosmology="abacus_c000")
    task41 = context["task41"]
    kernel = _shell_j0_average(context["k_eff"], edges[:-1], edges[1:])

    def xi_from_dense(pk_dense: np.ndarray) -> np.ndarray:
        weight = context["g_nz"] * task41.interp_logk(context["k_eff"], context["k_dense"], pk_dense)
        return (weight @ kernel) / float(context["volume"])

    pk_dd = task41.interp_logk(context["k_dense"], context["template"]["k"], context["template"]["pk_dd"])
    alpha = task41.interp_logk(context["k_dense"], context["template"]["k"], context["template"]["alpha"])
    bases = {
        "pk": xi_from_dense(pk_dd) - radial["pk"],
        "alpha": xi_from_dense(alpha * pk_dd) - radial["alpha"],
        "alpha2": xi_from_dense(alpha * alpha * pk_dd) - radial["alpha2"],
    }

    def evaluate(theta: np.ndarray) -> np.ndarray:
        fnl, b1 = np.asarray(theta, dtype="f8")
        bphi = 2.0 * 1.686 * (b1 - 1.0)
        product = fnl * bphi
        return b1 * b1 * bases["pk"] + 2.0 * b1 * product * bases["alpha"] + product * product * bases["alpha2"]

    return centers, evaluate, {"zeff": zeff, "edges": edges.tolist(), "theory": "FullDiscrete shell-average - radial auto"}


def main() -> None:
    """执行 xi/P(k) Fisher 分解并写出机器可读结果。"""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--xi", type=Path, default=DEFAULT_XI)
    parser.add_argument("--covariance", type=Path, default=DEFAULT_COV)
    parser.add_argument("--operator", type=Path, default=DEFAULT_OPERATOR)
    parser.add_argument("--pk-payload", type=Path, default=DEFAULT_PK_PAYLOAD)
    parser.add_argument("--output-prefix", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--fnl-reference", type=float, default=0.0)
    parser.add_argument("--b1-reference", type=float, default=2.5)
    args = parser.parse_args()

    for path in (args.xi, args.covariance, args.operator, args.pk_payload):
        if not path.is_file() or path.stat().st_size == 0:
            raise FileNotFoundError(path)

    centers, xi_model, xi_meta = build_xi_model(args.xi, args.operator)
    with np.load(args.covariance, allow_pickle=False) as data:
        xi_covariance = np.asarray(data["covariance_single_realization"], dtype="f8")
        if not np.array_equal(np.asarray(data["s"], dtype="f8"), centers):
            raise RuntimeError("xi covariance 与模型 centers 不一致")
    xi_jacobian, xi_convergence = finite_difference(
        xi_model,
        np.asarray([args.fnl_reference, args.b1_reference], dtype="f8"),
        np.asarray([0.1, 1.0e-3], dtype="f8"),
    )
    if max(xi_convergence.values()) > 1.0e-6:
        raise RuntimeError(f"xi finite difference 未收敛：{xi_convergence}")

    xi_cumulative, xi_information_increments = cumulative_xi_information(
        centers,
        xi_jacobian,
        xi_covariance,
    )

    xi_results: dict[str, Any] = {}
    csv_rows: list[dict[str, Any]] = []
    for rmin in (30, 40, 50):
        mask = centers >= float(rmin)
        result = fisher_summary(xi_jacobian[mask], xi_covariance[np.ix_(mask, mask)], ["fnl_loc", "b1"])
        result.update({"rmin_edge": rmin, "rmax_edge": 350, "nbins": int(np.count_nonzero(mask))})
        xi_results[str(rmin)] = result
        csv_rows.append(
            {
                "probe": "2pcf",
                "cut": rmin,
                "nbins": result["nbins"],
                "sigma_fnl": result["sigma_marginalized"]["fnl_loc"],
                "sigma_b1": result["sigma_marginalized"]["b1"],
                "sigma_b1_unmarginalized": result["sigma_unmarginalized"]["b1"],
                "sigma_b1_diagonal_cov": result["sigma_diagonal_covariance"]["b1"],
                "derivative_corr_fnl_b1": result["derivative_precision_correlation"][0][1],
                "b1_information_marginalized": result["information_marginalized"]["b1"],
            }
        )

    # 另写完整 separation ladder，便于从 CSV 直接定位哪些径向 bin 提供 b1
    # 信息；``cut`` 仍统一表示 rmin edge，不改变三档正式 case 的定义。
    for rmin in range(30, 331, 10):
        row = xi_cumulative[str(rmin)]
        csv_rows.append(
            {
                "probe": "2pcf_cumulative",
                "cut": rmin,
                "nbins": row["nbins"],
                "sigma_fnl": "",
                "sigma_b1": row["sigma_b1_marginalized"],
                "sigma_b1_unmarginalized": row["sigma_b1_unmarginalized"],
                "sigma_b1_diagonal_cov": "",
                "derivative_corr_fnl_b1": row["derivative_precision_correlation_fnl_b1"],
                "b1_information_marginalized": row["b1_information_marginalized"],
            }
        )

    pk_data = FitData(args.pk_payload)
    pk_data.load_radial_ric_operator(args.operator)
    task41 = load_task41()
    positive = pk_data.theory_k[pk_data.theory_k > 0.0]
    k_template = np.logspace(np.log10(min(1.0e-5, float(np.min(positive)) * 0.5)), np.log10(20.0), 20000)
    pk_template, _ = build_template_arrays(task41, k_template, z=pk_data.zeff, cosmology="abacus_c000")

    def pk_model(theta: np.ndarray) -> np.ndarray:
        fnl, b1, sn0 = np.asarray(theta, dtype="f8")
        return model_pk(
            pk_data,
            pk_template,
            {"fnl_loc": float(fnl), "b1": float(b1), "sn0": float(sn0)},
            p_fixed=1.0,
            sn0_scale=float(pk_data.sn0_scale),
            window_theory_kmin=2.0 * np.pi / 2000.0,
        )

    pk_jacobian, pk_convergence = finite_difference(
        pk_model,
        np.asarray([args.fnl_reference, args.b1_reference, 0.0], dtype="f8"),
        np.asarray([0.1, 1.0e-3, 1.0e-4], dtype="f8"),
    )
    if max(pk_convergence.values()) > 1.0e-6:
        raise RuntimeError(f"P(k) finite difference 未收敛：{pk_convergence}")

    pk_results: dict[str, Any] = {}
    for kmax in (0.08, 0.10):
        mask = pk_data.k_obs <= kmax + 1.0e-12
        result = fisher_summary(pk_jacobian[mask], pk_data.covariance[np.ix_(mask, mask)], ["fnl_loc", "b1", "sn0"])
        result.update({"kmax": kmax, "nbins": int(np.count_nonzero(mask))})
        pk_results[f"{kmax:.2f}"] = result
        csv_rows.append(
            {
                "probe": "pk",
                "cut": kmax,
                "nbins": result["nbins"],
                "sigma_fnl": result["sigma_marginalized"]["fnl_loc"],
                "sigma_b1": result["sigma_marginalized"]["b1"],
                "sigma_b1_unmarginalized": result["sigma_unmarginalized"]["b1"],
                "sigma_b1_diagonal_cov": result["sigma_diagonal_covariance"]["b1"],
                "derivative_corr_fnl_b1": result["derivative_precision_correlation"][0][1],
                "b1_information_marginalized": result["information_marginalized"]["b1"],
            }
        )

    payload = {
        "status": "pass",
        "task": "task43_rmin_scan_fisher",
        "classification": "local Fisher diagnostic; formal interpretation uses matched MCMC and residual gates",
        "reference_point": {"fnl_loc": args.fnl_reference, "b1": args.b1_reference, "sn0_pk": 0.0},
        "xi": {
            "metadata": xi_meta,
            "finite_difference": xi_convergence,
            "cases": xi_results,
            "cumulative_by_rmin": xi_cumulative,
            "information_increments_large_to_small": xi_information_increments,
            "low_r_b1_information": {
                "fraction_already_in_rmin50": float(
                    xi_cumulative["50"]["b1_information_fraction_of_rmin30"]
                ),
                "fraction_added_by_center45": float(
                    next(
                        row["b1_information_fraction_of_rmin30"]
                        for row in xi_information_increments
                        if row.get("added_bin_center") == 45.0
                    )
                ),
                "fraction_added_by_center35": float(
                    next(
                        row["b1_information_fraction_of_rmin30"]
                        for row in xi_information_increments
                        if row.get("added_bin_center") == 35.0
                    )
                ),
            },
        },
        "pk": {"finite_difference": pk_convergence, "cases": pk_results},
        "headline": {
            "xi_sigma_b1_ratio_rmin30_over_50": float(
                xi_results["30"]["sigma_marginalized"]["b1"] / xi_results["50"]["sigma_marginalized"]["b1"]
            ),
            "pk_sigma_b1_ratio_kmax010_over_008": float(
                pk_results["0.10"]["sigma_marginalized"]["b1"] / pk_results["0.08"]["sigma_marginalized"]["b1"]
            ),
            "xi30_over_pk010_sigma_b1": float(
                xi_results["30"]["sigma_marginalized"]["b1"] / pk_results["0.10"]["sigma_marginalized"]["b1"]
            ),
            "xi_rmin50_fraction_of_rmin30_b1_information": float(
                xi_cumulative["50"]["b1_information_fraction_of_rmin30"]
            ),
            "xi_center45_fraction_of_rmin30_b1_information": float(
                next(
                    row["b1_information_fraction_of_rmin30"]
                    for row in xi_information_increments
                    if row.get("added_bin_center") == 45.0
                )
            ),
            "xi_center35_fraction_of_rmin30_b1_information": float(
                next(
                    row["b1_information_fraction_of_rmin30"]
                    for row in xi_information_increments
                    if row.get("added_bin_center") == 35.0
                )
            ),
        },
        "inputs": {"xi": str(args.xi), "covariance": str(args.covariance), "operator": str(args.operator), "pk_payload": str(args.pk_payload)},
    }
    atomic_text(args.output_prefix.with_suffix(".json"), json.dumps(payload, indent=2, sort_keys=True) + "\n")
    fieldnames = [
        "probe",
        "cut",
        "nbins",
        "sigma_fnl",
        "sigma_b1",
        "sigma_b1_unmarginalized",
        "sigma_b1_diagonal_cov",
        "derivative_corr_fnl_b1",
        "b1_information_marginalized",
    ]
    lines: list[str] = []
    # csv 模块需要 file-like 对象；StringIO 只承载小型诊断文本。
    import io

    buffer = io.StringIO()
    writer = csv.DictWriter(buffer, fieldnames=fieldnames)
    writer.writeheader()
    writer.writerows(csv_rows)
    lines.append(buffer.getvalue())
    atomic_text(args.output_prefix.with_suffix(".csv"), "".join(lines))
    print(json.dumps(payload["headline"], indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
