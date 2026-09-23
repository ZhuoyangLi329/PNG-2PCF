#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""用 Task43 的 25 个 halo-lightcone phase 校准 RascalC ``alpha_SN``。

执行大纲
--------
1. 自动发现（或读取命令行指定的）RascalC Legendre-monopole 积分产物。
2. 从 25 个 FKP-weighted 2PCF realization 构造 sample covariance；这个
   covariance 可以秩亏，因为 RascalC 的 L1 目标不需要反演它。
3. 对 ``smin=50/60/80`` 分别调用 RascalC 官方的一参数 mock-calibration
   目标，拟合 ``C(alpha)=C4+alpha*C3+alpha^2*C2``。
4. 做 leave-one-phase-out alpha 稳定性和 held-out chi2 诊断；它们用于
   识别明显过拟合，但不应被解释成 25 个彼此独立的 alpha 测量。
5. 对主参考（最高 loops 的 FullDiscrete run）保存 calibrated covariance、
   RascalC 数值偏差修正 precision、JSON/CSV summary 和一张 PDF 诊断图。

这里有意不直接覆盖原来的 ``alpha=1`` 产物。校准结果使用独立文件名，
这样后续可以清楚地区分“积分收敛测试”和“mock-calibrated covariance”。
"""

from __future__ import annotations

import argparse
import contextlib
import csv
import io
import json
import os
from pathlib import Path
from typing import Any

# 登录节点只允许小规模 CPU 工作；在导入 numpy/scipy 前锁住数值库线程。
for _name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_name, "1")

import matplotlib.pyplot as plt
import numpy as np

from RascalC.convergence_check_extra import convergence_check_extra_splittings
from RascalC.post_process.utils import (
    add_cov_terms_single,
    compute_D_precision_matrix,
    compute_N_eff_D,
    cov_filter_legendre,
    fit_shot_noise_rescaling,
    load_matrices_single,
)
from RascalC.raw_covariance_matrices import load_raw_covariances_legendre

from task43_config import PLOT_DIR, SUMMARY_DIR


DEFAULT_RASCALC_ROOT = SUMMARY_DIR.parent / "rascalc_covariance" / "lightcone_kmin"
DEFAULT_SCATTER = SUMMARY_DIR / "task43_mean_xi_mmin1p4e13_x25_s50_350_ds10_fkpP010000.npz"
DEFAULT_OUTPUT = SUMMARY_DIR / "task43_rascalc_alpha_mock_calibration"
DEFAULT_PLOT = PLOT_DIR / "task43_rascalc_alpha_mock_calibration.pdf"
DEFAULT_CALIBRATED_DIR = SUMMARY_DIR.parent / "rascalc_covariance" / "calibrated"


def jsonable(value: Any) -> Any:
    """把 numpy、Path 和嵌套容器转换成可以写入 JSON 的普通对象。"""
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
    """由 covariance 计算相关系数矩阵。"""
    covariance = np.asarray(covariance, dtype="f8")
    sigma = np.sqrt(np.diag(covariance))
    return covariance / np.outer(sigma, sigma)


def run_sort_key(path: Path) -> tuple[int, int, str]:
    """让 FullDiscrete 最高 loops 成为主参考，再排列其余 kmin 对照。"""
    name = path.parent.name if path.name.endswith(".npz") else path.name
    if "xifull_discrete" in name:
        model_rank = 0
    elif "xicontinuous_boxcut" in name:
        model_rank = 1
    elif "xicontinuous_lowk" in name:
        model_rank = 2
    else:
        model_rank = 3
    loops = 0
    for token in name.split("_"):
        if token.startswith("nloop") and token[5:].isdigit():
            loops = int(token[5:])
            break
    return model_rank, -loops, name


def discover_runs(root: Path) -> list[Path]:
    """只发现带完整 JSON/NPZ 与 raw covariance 的成功 run。"""
    candidates: list[Path] = []
    for npz_path in root.glob("*/task43_rascalc_lightcone_covariance.npz"):
        json_path = npz_path.with_suffix(".json")
        raw_path = npz_path.parent / "rascalc_out" / "Raw_Covariance_Matrices_n30_l0.npz"
        if not json_path.exists() or not raw_path.exists():
            continue
        meta = json.loads(json_path.read_text(encoding="utf-8"))
        if meta.get("status") == "done":
            candidates.append(npz_path)
    return sorted(candidates, key=run_sort_key)


def normalize_run_path(path: Path) -> tuple[Path, Path, Path]:
    """统一用户给出的 run 目录或 summary NPZ，并返回 NPZ/JSON/raw-dir。"""
    path = Path(path)
    if path.is_dir():
        npz_path = path / "task43_rascalc_lightcone_covariance.npz"
    else:
        npz_path = path
    json_path = npz_path.with_suffix(".json")
    raw_dir = npz_path.parent / "rascalc_out"
    for required in (npz_path, json_path, raw_dir / "Raw_Covariance_Matrices_n30_l0.npz"):
        if not required.exists():
            raise FileNotFoundError(required)
    return npz_path, json_path, raw_dir


def fit_alpha_silent(
    target_covariance: np.ndarray,
    c2: np.ndarray,
    c3: np.ndarray,
    c4: np.ndarray,
    c2_samples: np.ndarray,
    c3_samples: np.ndarray,
    c4_samples: np.ndarray,
) -> float:
    """调用 RascalC 官方 optimizer，同时压掉 scipy.fmin 的逐次终端输出。"""
    with contextlib.redirect_stdout(io.StringIO()):
        alpha = fit_shot_noise_rescaling(
            target_covariance,
            c2,
            c3,
            c4,
            c2_samples,
            c3_samples,
            c4_samples,
        )
    return float(alpha)


def flatten_r_inv(convergence: dict[str, Any]) -> list[float]:
    """从 RascalC 两种 half-split 检验中取出全部 R_inv 数值。"""
    values: list[float] = []
    for split in convergence.values():
        values.extend(float(value) for value in split.get("R_inv", ()))
    return values


def covariance_summary(covariance: np.ndarray) -> dict[str, float]:
    """记录 covariance 的正定性、条件数和对角误差范围。"""
    covariance = 0.5 * (np.asarray(covariance, dtype="f8") + np.asarray(covariance, dtype="f8").T)
    eigenvalues = np.linalg.eigvalsh(covariance)
    sigma = np.sqrt(np.diag(covariance))
    return {
        "min_eigenvalue": float(eigenvalues[0]),
        "max_eigenvalue": float(eigenvalues[-1]),
        "condition_number": float(np.linalg.cond(covariance)),
        "sigma_min": float(np.min(sigma)),
        "sigma_median": float(np.median(sigma)),
        "sigma_max": float(np.max(sigma)),
    }


def scatter_summary(xi: np.ndarray, covariance: np.ndarray, precision: np.ndarray | None = None) -> dict[str, Any]:
    """比较单-lightcone covariance 与 phase scatter；不反演 sample covariance。"""
    xi = np.asarray(xi, dtype="f8")
    covariance = np.asarray(covariance, dtype="f8")
    if precision is None:
        precision = np.linalg.inv(covariance)
    residual = xi - np.mean(xi, axis=0)
    chi2 = np.einsum("ij,jk,ik->i", residual, precision, residual)
    ratio = np.std(xi, axis=0, ddof=1) / np.sqrt(np.diag(covariance))
    return {
        "nphase": int(xi.shape[0]),
        "nbins": int(xi.shape[1]),
        "expected_chi2_mean_about_sample_mean": float((xi.shape[0] - 1) / xi.shape[0] * xi.shape[1]),
        "chi2": chi2,
        "chi2_mean": float(np.mean(chi2)),
        "chi2_min": float(np.min(chi2)),
        "chi2_max": float(np.max(chi2)),
        "sample_std_over_cov_sigma": ratio,
        "sample_std_over_cov_sigma_min": float(np.min(ratio)),
        "sample_std_over_cov_sigma_median": float(np.median(ratio)),
        "sample_std_over_cov_sigma_max": float(np.max(ratio)),
    }


def calibrate_range(raw: dict[str, np.ndarray], xi: np.ndarray, start_bin: int) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    """拟合一个 smin 对应的 alpha，并返回留一诊断与 covariance 矩阵。"""
    n_total = int(xi.shape[1])
    cov_filter = cov_filter_legendre(n_total, 0, skip_r_bins=(int(start_bin), 0))
    c2, c3, c4 = load_matrices_single(raw, cov_filter, full=True)
    c2_samples, c3_samples, c4_samples = load_matrices_single(raw, cov_filter, full=False)
    xi_fit = np.asarray(xi[:, int(start_bin) :], dtype="f8")
    sample_covariance = np.cov(xi_fit, rowvar=False, ddof=1)

    alpha = fit_alpha_silent(sample_covariance, c2, c3, c4, c2_samples, c3_samples, c4_samples)
    covariance = add_cov_terms_single(c2, c3, c4, alpha)
    individual_covariances = add_cov_terms_single(c2_samples, c3_samples, c4_samples, alpha)
    d_matrix, corrected_precision = compute_D_precision_matrix(individual_covariances, covariance)
    n_eff = compute_N_eff_D(d_matrix)
    convergence = convergence_check_extra_splittings(individual_covariances)
    r_inv = flatten_r_inv(convergence)

    # 留一法同时诊断 alpha 稳定性和真正 held-out 数据向量的 chi2。
    loo_alpha: list[float] = []
    heldout_chi2: list[float] = []
    heldout_chi2_corrected: list[float] = []
    for phase_index in range(xi_fit.shape[0]):
        train = np.delete(xi_fit, phase_index, axis=0)
        train_covariance = np.cov(train, rowvar=False, ddof=1)
        alpha_train = fit_alpha_silent(train_covariance, c2, c3, c4, c2_samples, c3_samples, c4_samples)
        loo_alpha.append(alpha_train)
        covariance_train = add_cov_terms_single(c2, c3, c4, alpha_train)
        individual_train = add_cov_terms_single(c2_samples, c3_samples, c4_samples, alpha_train)
        _, corrected_precision_train = compute_D_precision_matrix(individual_train, covariance_train)
        residual = xi_fit[phase_index] - np.mean(train, axis=0)
        heldout_chi2.append(float(residual @ np.linalg.inv(covariance_train) @ residual))
        heldout_chi2_corrected.append(float(residual @ corrected_precision_train @ residual))

    scatter_inverse = scatter_summary(xi_fit, covariance)
    scatter_corrected = scatter_summary(xi_fit, covariance, corrected_precision)
    alpha_one_covariance = add_cov_terms_single(c2, c3, c4, 1.0)
    alpha_one_scatter = scatter_summary(xi_fit, alpha_one_covariance)
    ntrain = int(xi_fit.shape[0] - 1)
    heldout_expected = float(xi_fit.shape[1] * (1.0 + 1.0 / ntrain))
    row = {
        "start_bin": int(start_bin),
        "smin": float(50.0 + 10.0 * start_bin),
        "nbins": int(xi_fit.shape[1]),
        "sample_covariance_rank": int(np.linalg.matrix_rank(sample_covariance)),
        "alpha": alpha,
        "leave_one_out_alpha": loo_alpha,
        "leave_one_out_alpha_min": float(np.min(loo_alpha)),
        "leave_one_out_alpha_median": float(np.median(loo_alpha)),
        "leave_one_out_alpha_max": float(np.max(loo_alpha)),
        "leave_one_out_alpha_std": float(np.std(loo_alpha, ddof=1)),
        "heldout_chi2": heldout_chi2,
        "heldout_chi2_corrected_precision": heldout_chi2_corrected,
        "heldout_chi2_expected_mean_including_train_mean_noise": heldout_expected,
        "heldout_chi2_mean": float(np.mean(heldout_chi2)),
        "heldout_chi2_corrected_precision_mean": float(np.mean(heldout_chi2_corrected)),
        "alpha1_scatter": alpha_one_scatter,
        "calibrated_scatter_inverse_covariance": scatter_inverse,
        "calibrated_scatter_rascalc_corrected_precision": scatter_corrected,
        "covariance": covariance_summary(covariance),
        "N_eff": float(n_eff),
        "extra_convergence": convergence,
        "r_inv_values": r_inv,
        "r_inv_max": None if not r_inv else float(np.max(r_inv)),
        "r_inv_pass_0p05": bool(r_inv and np.max(r_inv) < 0.05),
    }
    matrices = {
        "c2": c2,
        "c3": c3,
        "c4": c4,
        "c2_samples": c2_samples,
        "c3_samples": c3_samples,
        "c4_samples": c4_samples,
        "sample_covariance": sample_covariance,
        "covariance": covariance,
        "individual_covariances": individual_covariances,
        "precision": corrected_precision,
        "d_matrix": d_matrix,
        "alpha_one_covariance": alpha_one_covariance,
    }
    return row, matrices


def calibrate_run(npz_path: Path, xi: np.ndarray, start_bins: list[int]) -> tuple[dict[str, Any], dict[int, dict[str, np.ndarray]]]:
    """读取一个 RascalC run，并对所有请求的 radial ranges 校准 alpha。"""
    npz_path, json_path, raw_dir = normalize_run_path(npz_path)
    meta = json.loads(json_path.read_text(encoding="utf-8"))
    raw = load_raw_covariances_legendre(str(raw_dir), 30, 0, print_function=lambda *_: None)
    model = str(meta.get("physics", {}).get("xi_model", "unknown"))
    loops = int(meta.get("settings", {}).get("n_loops", 0))
    label = f"{model} ({loops} loops)"
    rows: list[dict[str, Any]] = []
    matrices: dict[int, dict[str, np.ndarray]] = {}
    for start_bin in start_bins:
        row, matrix_row = calibrate_range(raw, xi, int(start_bin))
        rows.append(row)
        matrices[int(start_bin)] = matrix_row
    return {
        "label": label,
        "path": str(npz_path),
        "raw_dir": str(raw_dir),
        "xi_model": model,
        "n_loops": loops,
        "nrandom_monte_carlo": meta.get("random_monte_carlo", {}).get("n_used"),
        "original_alpha": meta.get("physics", {}).get("shot_noise_rescaling"),
        "ranges": rows,
    }, matrices


def write_csv(path: Path, runs: list[dict[str, Any]]) -> None:
    """写出一行一个 run/range 的紧凑表，方便以后不读大 JSON 即可比较。"""
    fields = [
        "label",
        "xi_model",
        "n_loops",
        "nrandom_monte_carlo",
        "smin",
        "nbins",
        "alpha",
        "loo_alpha_min",
        "loo_alpha_median",
        "loo_alpha_max",
        "r_inv_max",
        "alpha1_chi2_mean",
        "calibrated_chi2_mean",
        "expected_chi2_mean",
        "heldout_chi2_mean",
        "heldout_expected_chi2_mean",
    ]
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for run in runs:
            for row in run["ranges"]:
                writer.writerow(
                    {
                        "label": run["label"],
                        "xi_model": run["xi_model"],
                        "n_loops": run["n_loops"],
                        "nrandom_monte_carlo": run["nrandom_monte_carlo"],
                        "smin": row["smin"],
                        "nbins": row["nbins"],
                        "alpha": row["alpha"],
                        "loo_alpha_min": row["leave_one_out_alpha_min"],
                        "loo_alpha_median": row["leave_one_out_alpha_median"],
                        "loo_alpha_max": row["leave_one_out_alpha_max"],
                        "r_inv_max": row["r_inv_max"],
                        "alpha1_chi2_mean": row["alpha1_scatter"]["chi2_mean"],
                        "calibrated_chi2_mean": row["calibrated_scatter_inverse_covariance"]["chi2_mean"],
                        "expected_chi2_mean": row["calibrated_scatter_inverse_covariance"]["expected_chi2_mean_about_sample_mean"],
                        "heldout_chi2_mean": row["heldout_chi2_mean"],
                        "heldout_expected_chi2_mean": row["heldout_chi2_expected_mean_including_train_mean_noise"],
                    }
                )


def plot_diagnostics(path: Path, runs: list[dict[str, Any]], s: np.ndarray, xi: np.ndarray, primary_matrices: dict[int, dict[str, np.ndarray]]) -> None:
    """生成唯一一张 PDF：alpha 稳定性、误差闭合与 chi2 闭合。"""
    path.parent.mkdir(parents=True, exist_ok=True)
    colors = plt.cm.tab10(np.linspace(0.0, 0.9, len(runs)))
    fig, axes = plt.subplots(3, 1, figsize=(8.4, 10.8), constrained_layout=True)

    for color, run in zip(colors, runs, strict=True):
        smin = np.asarray([row["smin"] for row in run["ranges"]], dtype="f8")
        alpha = np.asarray([row["alpha"] for row in run["ranges"]], dtype="f8")
        lower = alpha - np.asarray([row["leave_one_out_alpha_min"] for row in run["ranges"]], dtype="f8")
        upper = np.asarray([row["leave_one_out_alpha_max"] for row in run["ranges"]], dtype="f8") - alpha
        axes[0].errorbar(smin, alpha, yerr=np.vstack([lower, upper]), marker="o", capsize=3, color=color, label=run["label"])
    axes[0].axhline(1.0, color="black", ls="--", lw=1.0)
    axes[0].set_ylabel(r"mock-calibrated $\alpha_{\rm SN}$")
    axes[0].set_xlabel(r"$s_{\min}\ [h^{-1}{\rm Mpc}]$")
    axes[0].legend(fontsize=7, ncol=2)
    axes[0].set_title("Task43 RascalC mock calibration (error bars: leave-one-phase-out range)")

    primary_start = min(primary_matrices)
    matrix = primary_matrices[primary_start]
    sample_std = np.std(xi[:, primary_start:], axis=0, ddof=1)
    axes[1].plot(s[primary_start:], sample_std / np.sqrt(np.diag(matrix["alpha_one_covariance"])), label=r"sample std / RascalC $\alpha=1$", lw=1.8)
    axes[1].plot(s[primary_start:], sample_std / np.sqrt(np.diag(matrix["covariance"])), label=r"sample std / calibrated RascalC", lw=1.8)
    axes[1].axhline(1.0, color="black", ls="--", lw=1.0)
    axes[1].set_xlabel(r"$s\ [h^{-1}{\rm Mpc}]$")
    axes[1].set_ylabel("scatter / model sigma")
    axes[1].legend(fontsize=8)

    # 每个 smin 组固定留出 8 个横轴单位，run 多时也不会互相覆盖。
    width = 8.0 / max(len(runs), 1)
    for index, (color, run) in enumerate(zip(colors, runs, strict=True)):
        rows = run["ranges"]
        x = np.arange(len(rows), dtype="f8") * 10.0 + index * width
        axes[2].bar(x, [row["alpha1_scatter"]["chi2_mean"] for row in rows], width=width, color=color, alpha=0.35)
        axes[2].bar(x, [row["calibrated_scatter_inverse_covariance"]["chi2_mean"] for row in rows], width=width, color=color, alpha=0.9, label=run["label"])
    expected = [row["calibrated_scatter_inverse_covariance"]["expected_chi2_mean_about_sample_mean"] for row in runs[0]["ranges"]]
    for group, value in enumerate(expected):
        axes[2].hlines(value, group * 10.0 - width, group * 10.0 + len(runs) * width, color="black", ls="--", lw=1.0)
    axes[2].set_xticks(np.arange(len(runs[0]["ranges"])) * 10.0 + 0.5 * (len(runs) - 1) * width)
    axes[2].set_xticklabels([f"smin={row['smin']:.0f}" for row in runs[0]["ranges"]])
    axes[2].set_ylabel(r"$\langle\chi^2\rangle$ about phase mean")
    axes[2].legend(fontsize=7, ncol=2)
    axes[2].set_title("Pale bars: alpha=1; solid bars: calibrated; dashed: finite-sample expectation")

    fig.savefig(path)
    plt.close(fig)


def main() -> None:
    """解析参数、执行校准并写出可复现产物。"""
    parser = argparse.ArgumentParser()
    parser.add_argument("--rascalc-run", type=Path, nargs="*", default=None, help="Run directories or summary NPZ files; default discovers all successful controlled runs.")
    parser.add_argument("--rascalc-root", type=Path, default=DEFAULT_RASCALC_ROOT)
    parser.add_argument("--scatter", type=Path, default=DEFAULT_SCATTER)
    parser.add_argument("--smin", type=float, nargs="+", default=(50.0, 60.0, 80.0))
    parser.add_argument("--output-prefix", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--plot-path", type=Path, default=DEFAULT_PLOT)
    parser.add_argument("--calibrated-dir", type=Path, default=DEFAULT_CALIBRATED_DIR)
    args = parser.parse_args()

    if not args.scatter.exists():
        raise FileNotFoundError(args.scatter)
    run_paths = discover_runs(args.rascalc_root) if args.rascalc_run is None else [normalize_run_path(path)[0] for path in args.rascalc_run]
    run_paths = sorted(run_paths, key=run_sort_key)
    if not run_paths:
        raise FileNotFoundError(f"No successful RascalC runs under {args.rascalc_root}")

    scatter_data = np.load(args.scatter, allow_pickle=False)
    xi = np.asarray(scatter_data["xi0_all"], dtype="f8")
    s = np.asarray(scatter_data["s"], dtype="f8")
    if xi.shape != (25, 30):
        raise ValueError(f"Expected Task43 phase stack shape (25, 30), got {xi.shape}")
    start_bins: list[int] = []
    for smin in args.smin:
        match = np.flatnonzero(np.isclose(s, float(smin) + 5.0, rtol=0.0, atol=1.0e-8))
        if match.size != 1:
            raise ValueError(f"smin={smin} does not match one Task43 lower bin edge")
        start_bins.append(int(match[0]))

    runs: list[dict[str, Any]] = []
    all_run_matrices: list[dict[int, dict[str, np.ndarray]]] = []
    primary_matrices: dict[int, dict[str, np.ndarray]] | None = None
    for index, run_path in enumerate(run_paths):
        print(f"[calibrate] {run_path}", flush=True)
        run, matrices = calibrate_run(run_path, xi, start_bins)
        runs.append(run)
        all_run_matrices.append(matrices)
        if index == 0:
            primary_matrices = matrices
    assert primary_matrices is not None

    primary_start = min(start_bins)
    primary = runs[0]
    matrix = primary_matrices[primary_start]
    primary_row = next(row for row in primary["ranges"] if row["start_bin"] == primary_start)

    # 给每个 run 单独保存一个可直接被 fit 脚本读取的 calibrated covariance。
    # 这样 posterior-level A/B 不需要从 stacked summary 中手工切片。
    args.calibrated_dir.mkdir(parents=True, exist_ok=True)
    for run, matrices in zip(runs, all_run_matrices, strict=True):
        row = next(item for item in run["ranges"] if item["start_bin"] == primary_start)
        model_tag = str(run["xi_model"]).replace(" ", "_")
        nran_tag = int(run["nrandom_monte_carlo"] or 0)
        calibrated_path = args.calibrated_dir / (
            f"task43_rascalc_{model_tag}_nran{nran_tag}_nloop{int(run['n_loops'])}_"
            f"alphaMockCal_smin{int(row['smin'])}.npz"
        )
        this_matrix = matrices[primary_start]
        standalone_meta = {
            "status": "done",
            "task": "task43_calibrate_rascalc_alpha_from_phases",
            "source_run": run["path"],
            "label": run["label"],
            "xi_model": run["xi_model"],
            "n_loops": run["n_loops"],
            "nrandom_monte_carlo": run["nrandom_monte_carlo"],
            "smin_calibration": row["smin"],
            "shot_noise_rescaling": row["alpha"],
            "leave_one_out_alpha_min": row["leave_one_out_alpha_min"],
            "leave_one_out_alpha_max": row["leave_one_out_alpha_max"],
            "r_inv_max": row["r_inv_max"],
            "warning": "Mock-calibrated from the same 25 Task43 phases; use the recorded held-out diagnostics and do not call this full-random input unless nrandom_monte_carlo=8254350.",
        }
        np.savez_compressed(
            calibrated_path,
            s=s[primary_start:],
            s_edges=np.arange(50.0 + 10.0 * primary_start, 360.0, 10.0, dtype="f8"),
            covariance_single_realization=this_matrix["covariance"],
            correlation=correlation(this_matrix["covariance"]),
            precision_rascalc_bias_corrected=this_matrix["precision"],
            shot_noise_rescaling=np.asarray(float(row["alpha"]), dtype="f8"),
            covariance_alpha1=this_matrix["alpha_one_covariance"],
            source_run=np.asarray(run["path"]),
            meta_json=np.asarray(json.dumps(jsonable(standalone_meta), sort_keys=True)),
        )
        run["calibrated_covariance_npz"] = str(calibrated_path)

    payload = {
        "status": "done",
        "task": "task43_calibrate_rascalc_alpha_from_phases",
        "method": "RascalC official mock L1 alpha calibration with 25 FKP-weighted Task43 phase realizations",
        "scatter_path": str(args.scatter),
        "sample_covariance_note": "The 25x30 sample covariance is rank deficient, but RascalC's L1 alpha objective explicitly permits a singular target covariance.",
        "validation_note": "Leave-one-phase-out rows are stability/held-out diagnostics; folds overlap and are not 25 independent calibration measurements.",
        "production_warning": "All listed RascalC integrations use a Monte-Carlo random subset unless nrandom_monte_carlo equals the full 8,254,350 catalog; check this before calling any row production/full-input.",
        "primary_reference": primary["label"],
        "primary_calibrated_alpha": primary_row["alpha"],
        "runs": runs,
        "outputs": {
            "npz": str(args.output_prefix.with_suffix(".npz")),
            "json": str(args.output_prefix.with_suffix(".json")),
            "csv": str(args.output_prefix.with_suffix(".csv")),
            "plot_pdf": str(args.plot_path),
            "standalone_calibrated_covariance_dir": str(args.calibrated_dir),
        },
    }

    out_npz = args.output_prefix.with_suffix(".npz")
    out_json = args.output_prefix.with_suffix(".json")
    out_csv = args.output_prefix.with_suffix(".csv")
    out_npz.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        out_npz,
        s=s[primary_start:],
        s_edges=np.arange(50.0 + 10.0 * primary_start, 360.0, 10.0, dtype="f8"),
        covariance_single_realization=matrix["covariance"],
        correlation=correlation(matrix["covariance"]),
        precision_rascalc_bias_corrected=matrix["precision"],
        precision_inverse_covariance=np.linalg.inv(matrix["covariance"]),
        shot_noise_rescaling=np.array(float(primary_row["alpha"]), dtype="f8"),
        sample_covariance=matrix["sample_covariance"],
        covariance_alpha1=matrix["alpha_one_covariance"],
        run_labels=np.asarray([run["label"] for run in runs]),
        run_paths=np.asarray([run["path"] for run in runs]),
        run_calibrated_alphas=np.asarray(
            [next(row for row in run["ranges"] if row["start_bin"] == primary_start)["alpha"] for run in runs],
            dtype="f8",
        ),
        run_calibrated_covariances=np.stack([matrices[primary_start]["covariance"] for matrices in all_run_matrices]),
        run_alpha1_covariances=np.stack([matrices[primary_start]["alpha_one_covariance"] for matrices in all_run_matrices]),
        run_calibrated_precisions=np.stack([matrices[primary_start]["precision"] for matrices in all_run_matrices]),
        c2=matrix["c2"],
        c3=matrix["c3"],
        c4=matrix["c4"],
        individual_theory_covariances=matrix["individual_covariances"],
        full_theory_D_matrix=matrix["d_matrix"],
        meta_json=np.asarray(json.dumps(jsonable(payload), sort_keys=True)),
    )
    out_json.write_text(json.dumps(jsonable(payload), indent=2, sort_keys=True) + "\n", encoding="utf-8")
    write_csv(out_csv, runs)
    plot_diagnostics(args.plot_path, runs, s, xi, primary_matrices)
    print(f"[done] wrote {out_json}")
    for run in runs:
        for row in run["ranges"]:
            print(
                f"[row] {run['label']} smin={row['smin']:.0f} alpha={row['alpha']:.5f} "
                f"loo=[{row['leave_one_out_alpha_min']:.5f},{row['leave_one_out_alpha_max']:.5f}] "
                f"R_inv={row['r_inv_max']:.4f} chi2={row['calibrated_scatter_inverse_covariance']['chi2_mean']:.3f}",
                flush=True,
            )


if __name__ == "__main__":
    main()
