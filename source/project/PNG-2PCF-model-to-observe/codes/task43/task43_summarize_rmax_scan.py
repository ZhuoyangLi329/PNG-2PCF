#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""汇总 Task4.3 jaxpower-only 2PCF ``s_max`` matched scan。

代码大纲
========
1. 读取同一套 50-bin xi、50x50 RR-deconvolved jaxpower covariance、50-bin
   radial single-term RIC operator，以及 ``smax=350/400/450/500/550`` 五条长链。
2. 从 raw flat samples 恢复 ``(post_steps, nwalkers, ndim)``，统一执行
   ``length/tau > 100`` 与 split-median ``< 0.05 sigma`` 收敛门槛。
3. 用统一的 Gaussian 68.27% quantiles 重算 fNL/b1 posterior，并检查 covariance
   坐标、SPD、nested 子块和所有 fit provenance。
4. 对旧权威 350 链、xi、covariance、operator 做 bridge；对相邻新增 5 bins
   计算 Schur-complement conditional residual chi2 与 mean-data PTE。
5. 只有输入、covariance、chain 和 bridge 硬门槛全部通过时才写 CSV/PDF；任何
   失败都会写 ``status=fail`` JSON，且不会新生成一张掩盖失败的图。
6. 最终解释严格限定为 fixed jaxpower diagnostic covariance 加 radial
   single-term RIC 下的相对 ``smax`` 信息变化，不升级为 science-ready 约束。
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import traceback
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

for _name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_name, "1")

import emcee
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import chi2 as chi2_distribution


PROJECT_ROOT = Path("/pscratch/sd/l/lzy/PNG-2PCF-model-to-observe")
SCAN_ROOT = PROJECT_ROOT / "outputs/task43_outputs/rmax_scan"
RMAX_VALUES = (350, 400, 450, 500, 550)
GAUSSIAN_QUANTILES = (0.1586552539, 0.5, 0.8413447461)

DEFAULT_XI = (
    SCAN_ROOT
    / "summary/task43_mean_xi_mmin1p4e13_x25_s50_550_ds10_fkpP010000.npz"
)
DEFAULT_COVARIANCE = (
    SCAN_ROOT
    / "covariance/jaxpower_covariance_lightcone_mmin1p4e13_ph000_fitcov_"
    "bessel_interp_smoothfftlog_rrdeconv_fkpNorm4p8925e10_mesh64_nran100k_"
    "ndata50k_pad400_win3600_ds2_k0001_3000_dk002_p1p0_s50_550_ds10.npz"
)
DEFAULT_OPERATOR = (
    SCAN_ROOT
    / "operators/task43_ric_factorized_operator_ph000_dchi2_nsub200000_"
    "sobol2p22_ds2_seed20260712_L2000_s50_550_ds10.npz"
)
DEFAULT_FIT_ROOT = SCAN_ROOT / "fits"

OLD_ROOT = PROJECT_ROOT / "outputs/task43_outputs"
DEFAULT_OLD_XI = (
    OLD_ROOT / "summary/task43_mean_xi_mmin1p4e13_x25_s50_350_ds10_fkpP010000.npz"
)
DEFAULT_OLD_COVARIANCE = (
    OLD_ROOT
    / "summary/jaxpower_covariance_lightcone_mmin1p4e13_ph000_fitcov_"
    "bessel_interp_smoothfftlog_rrdeconv_mesh64_nran100k_ndata50k_pad400_"
    "win3600_ds2_k0001_3000_dk002_p1p0_s50_350_ds10.npz"
)
DEFAULT_OLD_OPERATOR = (
    OLD_ROOT
    / "ric_singleterm/operators/task43_ric_factorized_operator_ph000_dchi2_"
    "nsub200000_sobol2p22_ds2_seed20260712_L2000.npz"
)
DEFAULT_OLD_FIT_DIR = (
    OLD_ROOT / "ric_singleterm/fits/2pcf_jaxpower_ph000_dchi2_nsub200000_long_mcmc20k"
)

DEFAULT_OUTPUT_JSON = (
    SCAN_ROOT / "audits/task43_jaxpower_2pcf_rmax_scan_longchain.json"
)
DEFAULT_OUTPUT_CSV = (
    SCAN_ROOT / "audits/task43_jaxpower_2pcf_rmax_scan_longchain.csv"
)
DEFAULT_OUTPUT_PDF = (
    PROJECT_ROOT
    / "plots/task43/rmax_scan/task43_jaxpower_2pcf_rmax_scan_s50_350_550.pdf"
)

SUMMARY_NAME = "task43_minimal_closure_mcmc_summary.json"
SAMPLES_NAME = "task43_mcmc_radial_singleterm_samples.npz"


class AuditError(RuntimeError):
    """表示输入或机器门槛不满足，调用方应写明确的失败审计。"""


def utc_now() -> str:
    """返回稳定的 UTC 时间字符串。"""
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def jsonable(value: Any) -> Any:
    """递归转换 Path 与 NumPy 对象，供 audit JSON 使用。"""
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.ndarray):
        return jsonable(value.tolist())
    if isinstance(value, np.generic):
        return jsonable(value.item())
    if isinstance(value, float) and not np.isfinite(value):
        return None
    if isinstance(value, dict):
        return {str(key): jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [jsonable(item) for item in value]
    return value


def require(condition: bool, message: str) -> None:
    """不满足条件时抛出带清楚上下文的审计错误。"""
    if not bool(condition):
        raise AuditError(message)


def canonical(path: str | Path) -> Path:
    """把 summary 中相对项目根目录的 provenance 路径统一成绝对路径。"""
    value = Path(path).expanduser()
    if not value.is_absolute():
        value = PROJECT_ROOT / value
    return value.resolve()


def same_path(first: str | Path, second: str | Path) -> bool:
    """比较两个可为相对路径的 provenance。"""
    return canonical(first) == canonical(second)


def read_json(path: Path) -> dict[str, Any]:
    """读取必须存在的 JSON object。"""
    require(Path(path).is_file(), f"missing JSON: {path}")
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    require(isinstance(payload, dict), f"JSON root is not an object: {path}")
    return payload


def atomic_write_json(path: Path, payload: dict[str, Any]) -> None:
    """原子写机器审计，避免中断留下半个 JSON。"""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(json.dumps(jsonable(payload), indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(path)


def atomic_write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    """原子写一行一个 smax 的扁平 CSV。"""
    require(bool(rows), "refusing to write an empty scan CSV")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    fieldnames = list(rows[0])
    with temporary.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)


def parse_fit_overrides(values: list[str]) -> dict[int, Path]:
    """解析可重复的 ``--fit-dir RMAX=PATH``。"""
    output: dict[int, Path] = {}
    for value in values:
        if "=" not in value:
            raise ValueError(f"--fit-dir must be RMAX=PATH, got {value!r}")
        raw_rmax, raw_path = value.split("=", 1)
        rmax = int(raw_rmax)
        if rmax not in RMAX_VALUES:
            raise ValueError(f"unsupported --fit-dir rmax={rmax}; expected one of {RMAX_VALUES}")
        if rmax in output:
            raise ValueError(f"duplicate --fit-dir for rmax={rmax}")
        output[rmax] = canonical(raw_path)
    return output


def fit_directories(fit_root: Path, overrides: list[str]) -> dict[int, Path]:
    """返回五档 canonical fit 目录；默认命名为 ``fits/smaxNNN``。"""
    result = {rmax: canonical(Path(fit_root) / f"smax{rmax}") for rmax in RMAX_VALUES}
    result.update(parse_fit_overrides(overrides))
    return result


def radial_model(summary: dict[str, Any]) -> dict[str, Any]:
    """取得 summary 中唯一的 radial single-term model 行。"""
    rows = [row for row in summary.get("models", []) if row.get("model") == "radial_singleterm"]
    require(len(rows) == 1, f"expected one radial_singleterm model, found {len(rows)}")
    return rows[0]


def posterior_stats(values: np.ndarray) -> dict[str, float]:
    """统一用 15.8655/50/84.1345 percentiles 汇总一维 posterior。"""
    array = np.asarray(values, dtype="f8")
    require(array.ndim == 1 and array.size > 1, f"invalid posterior array shape={array.shape}")
    require(bool(np.all(np.isfinite(array))), "posterior contains non-finite samples")
    q16, q50, q84 = np.quantile(array, GAUSSIAN_QUANTILES)
    return {
        "q16": float(q16),
        "q50": float(q50),
        "q84": float(q84),
        "err_low": float(q50 - q16),
        "err_high": float(q84 - q50),
        "sigma68": float(0.5 * (q84 - q16)),
        "mean": float(np.mean(array)),
        "std": float(np.std(array, ddof=1)),
    }


def load_chain(
    summary: dict[str, Any],
    samples_path: Path,
    *,
    require_map_arrays: bool,
) -> dict[str, Any]:
    """恢复 walker chain，计算收敛量、posterior 和保存的 MAP 数组。"""
    model = radial_model(summary)
    names = [str(name) for name in model["parameter_names"]]
    mcmc = model["mcmc"]
    nwalkers = int(mcmc["nwalkers"])
    nsteps = int(mcmc["nsteps"])
    burnin = int(mcmc["burnin"])
    post_steps = nsteps - burnin
    require(nwalkers > 0 and post_steps > 1, f"invalid MCMC dimensions in {samples_path}")
    require(Path(samples_path).is_file(), f"missing samples NPZ: {samples_path}")
    with np.load(samples_path, allow_pickle=False) as data:
        require("samples" in data.files, f"samples key missing in {samples_path}")
        samples = np.asarray(data["samples"], dtype="f8")
        log_prob = np.asarray(data["log_prob"], dtype="f8") if "log_prob" in data.files else None
        prediction_map = np.asarray(data["prediction_map"], dtype="f8") if "prediction_map" in data.files else None
        residual_map = np.asarray(data["residual_map"], dtype="f8") if "residual_map" in data.files else None
    require(samples.ndim == 2, f"flat samples are not 2D in {samples_path}: {samples.shape}")
    require(samples.shape == (post_steps * nwalkers, len(names)), (
        f"flat sample shape mismatch in {samples_path}: got {samples.shape}, "
        f"expected {(post_steps * nwalkers, len(names))}"
    ))
    require(bool(np.all(np.isfinite(samples))), f"non-finite samples in {samples_path}")
    if log_prob is not None:
        require(log_prob.shape == (samples.shape[0],), f"log_prob shape mismatch in {samples_path}")
        require(bool(np.all(np.isfinite(log_prob))), f"non-finite log_prob in {samples_path}")
    if require_map_arrays:
        require(prediction_map is not None, f"prediction_map missing in {samples_path}")
        require(residual_map is not None, f"residual_map missing in {samples_path}")

    chain = samples.reshape((post_steps, nwalkers, len(names)))
    tau = np.asarray(emcee.autocorr.integrated_time(chain, quiet=True), dtype="f8")
    half = post_steps // 2
    require(half > 0, f"post-burn chain cannot be split in {samples_path}")
    first = chain[:half].reshape((-1, len(names)))
    second = chain[half:].reshape((-1, len(names)))
    sigma = np.std(samples, axis=0, ddof=1)
    split = np.full(len(names), np.inf, dtype="f8")
    positive_sigma = sigma > 0.0
    split[positive_sigma] = (
        np.abs(np.median(first, axis=0)[positive_sigma] - np.median(second, axis=0)[positive_sigma])
        / sigma[positive_sigma]
    )
    length_over_tau = post_steps / tau
    finite_diagnostics = bool(np.all(np.isfinite(tau)) and np.all(tau > 0.0) and np.all(np.isfinite(split)))
    diagnostics_pass = bool(
        finite_diagnostics
        and float(np.min(length_over_tau)) > 100.0
        and float(np.max(split)) < 0.05
    )

    require("fnl_loc" in names and "b1" in names, f"fNL/b1 columns missing in {samples_path}: {names}")
    fnl = samples[:, names.index("fnl_loc")]
    b1 = samples[:, names.index("b1")]
    return {
        "names": names,
        "samples": samples,
        "prediction_map": prediction_map,
        "residual_map": residual_map,
        "posterior": {
            "fnl_loc": posterior_stats(fnl),
            "b1": posterior_stats(b1),
            "corr_fnl_b1": float(np.corrcoef(fnl, b1)[0, 1]),
        },
        "diagnostics": {
            "post_burn_steps_per_walker": post_steps,
            "nwalkers": nwalkers,
            "ndim": len(names),
            "tau": {name: float(value) for name, value in zip(names, tau, strict=True)},
            "length_over_tau": {
                name: float(value) for name, value in zip(names, length_over_tau, strict=True)
            },
            "length_over_tau_min": float(np.min(length_over_tau)),
            "split_median_shift_sigma": {
                name: float(value) for name, value in zip(names, split, strict=True)
            },
            "split_median_shift_sigma_max": float(np.max(split)),
            "gate": "all parameters: post-burn length/tau > 100 and split median shift < 0.05 sigma",
            "pass": diagnostics_pass,
        },
        "mcmc": {
            "nwalkers": nwalkers,
            "nsteps": nsteps,
            "burnin": burnin,
            "seed": None if "seed" not in mcmc else int(mcmc["seed"]),
            "mean_acceptance_fraction": float(mcmc.get("mean_acceptance_fraction", np.nan)),
        },
    }


def matrix_audit(matrix: np.ndarray) -> dict[str, Any]:
    """检查 covariance 或 Schur complement 的有限性、对称性与 SPD。"""
    value = np.asarray(matrix, dtype="f8")
    finite = bool(value.ndim == 2 and value.shape[0] == value.shape[1] and np.all(np.isfinite(value)))
    if not finite:
        return {"shape": list(value.shape), "finite": False, "symmetric": False, "spd": False, "pass": False}
    symmetric_value = 0.5 * (value + value.T)
    symmetry_max_abs = float(np.max(np.abs(value - value.T)))
    scale = max(float(np.max(np.abs(value))), 1.0e-300)
    symmetric = bool(symmetry_max_abs <= 1.0e-10 * scale)
    eigenvalues = np.linalg.eigvalsh(symmetric_value)
    eig_min = float(np.min(eigenvalues))
    eig_max = float(np.max(eigenvalues))
    spd = bool(eig_min > 0.0)
    sigma = np.sqrt(np.maximum(np.diag(symmetric_value), 0.0))
    positive_sigma = bool(np.all(sigma > 0.0))
    return {
        "shape": [int(v) for v in value.shape],
        "finite": finite,
        "symmetric": symmetric,
        "symmetry_max_abs": symmetry_max_abs,
        "eig_min": eig_min,
        "eig_max": eig_max,
        "condition_number": float(eig_max / eig_min) if spd else float("inf"),
        "diagonal_positive": positive_sigma,
        "sigma_min": float(np.min(sigma)),
        "sigma_max": float(np.max(sigma)),
        "spd": spd,
        "pass": bool(finite and symmetric and spd and positive_sigma),
    }


def correlation_matrix(covariance: np.ndarray) -> np.ndarray:
    """从 SPD covariance 构造 correlation matrix。"""
    sigma = np.sqrt(np.diag(np.asarray(covariance, dtype="f8")))
    return np.asarray(covariance, dtype="f8") / np.outer(sigma, sigma)


def load_xi(path: Path, *, expected_nbins: int, expected_nreal: int) -> dict[str, Any]:
    """读取 mean xi 与坐标，并恢复真实 realization 数。"""
    require(Path(path).is_file(), f"missing xi NPZ: {path}")
    with np.load(path, allow_pickle=False) as data:
        for key in ("s", "s_edges", "xi0"):
            require(key in data.files, f"{key} missing in {path}")
        s = np.asarray(data["s"], dtype="f8")
        edges = np.asarray(data["s_edges"], dtype="f8")
        xi = np.asarray(data["xi0"], dtype="f8")
        if "xi0_all" in data.files:
            nreal = int(np.asarray(data["xi0_all"]).shape[0])
        elif "nreal" in data.files:
            nreal = int(np.asarray(data["nreal"]).item())
        else:
            raise AuditError(f"cannot determine nreal from {path}")
    require(s.shape == (expected_nbins,), f"xi s shape mismatch in {path}: {s.shape}")
    require(edges.shape == (expected_nbins + 1,), f"xi s_edges shape mismatch in {path}: {edges.shape}")
    require(xi.shape == (expected_nbins,), f"xi0 shape mismatch in {path}: {xi.shape}")
    require(bool(np.all(np.isfinite(s)) and np.all(np.isfinite(edges)) and np.all(np.isfinite(xi))), (
        f"non-finite xi content in {path}"
    ))
    require(bool(np.all(np.diff(edges) > 0.0)), f"xi edges are not strictly increasing in {path}")
    require(bool(np.allclose(s, 0.5 * (edges[:-1] + edges[1:]), rtol=0.0, atol=1.0e-10)), (
        f"xi centers are not shell midpoints in {path}"
    ))
    require(nreal == expected_nreal, f"xi nreal={nreal}, expected {expected_nreal} in {path}")
    return {"path": canonical(path), "s": s, "s_edges": edges, "xi": xi, "nreal": nreal}


def load_covariance(
    path: Path,
    *,
    key: str,
    expected_s: np.ndarray,
    expected_edges: np.ndarray,
    nested_rmax_values: tuple[int, ...] = RMAX_VALUES,
) -> dict[str, Any]:
    """读取指定 covariance key，并严格核对坐标与 full/nested SPD。"""
    require(Path(path).is_file(), f"missing covariance NPZ: {path}")
    with np.load(path, allow_pickle=False) as data:
        require(key in data.files, f"covariance key={key!r} missing in {path}; keys={data.files}")
        require("s" in data.files and "s_edges" in data.files, f"covariance coordinates missing in {path}")
        covariance = np.asarray(data[key], dtype="f8")
        s = np.asarray(data["s"], dtype="f8")
        edges = np.asarray(data["s_edges"], dtype="f8")
    require(covariance.shape == (expected_s.size, expected_s.size), (
        f"covariance shape {covariance.shape} does not match {expected_s.size} xi bins in {path}"
    ))
    require(s.shape == expected_s.shape and np.allclose(s, expected_s, rtol=0.0, atol=1.0e-10), (
        f"covariance s coordinates do not match xi in {path}"
    ))
    require(edges.shape == expected_edges.shape and np.allclose(edges, expected_edges, rtol=0.0, atol=1.0e-10), (
        f"covariance s_edges do not match xi in {path}"
    ))
    full_audit = matrix_audit(covariance)
    require(full_audit["pass"], f"full covariance failed finite/symmetric/SPD gate: {full_audit}")
    nested: dict[str, Any] = {}
    for rmax in nested_rmax_values:
        mask = (s >= 50.0) & (s <= float(rmax))
        expected_size = 30 + (rmax - 350) // 10
        require(int(np.sum(mask)) == expected_size, f"rmax={rmax} selected {np.sum(mask)} covariance bins")
        indices = np.flatnonzero(mask)
        require(np.array_equal(indices, np.arange(expected_size)), (
            f"rmax={rmax} crop is not the leading nested block: {indices}"
        ))
        selected_edges = edges[: expected_size + 1]
        require(float(selected_edges[0]) == 50.0 and float(selected_edges[-1]) == float(rmax), (
            f"rmax={rmax} must denote the upper separation edge, got {selected_edges[[0, -1]]}"
        ))
        row = matrix_audit(covariance[np.ix_(mask, mask)])
        row["selected_indices"] = [int(v) for v in indices]
        row["selected_edge_min"] = float(selected_edges[0])
        row["selected_edge_max"] = float(selected_edges[-1])
        row["last_bin_center"] = float(s[indices[-1]])
        require(row["pass"], f"covariance nested subblock rmax={rmax} failed SPD gate: {row}")
        nested[str(rmax)] = row
    return {
        "path": canonical(path),
        "key": key,
        "s": s,
        "s_edges": edges,
        "matrix": covariance,
        "full_audit": full_audit,
        "nested_audits": nested,
    }


def load_operator(path: Path, *, expected_edges: np.ndarray) -> dict[str, Any]:
    """读取 2PCF radial-RIC 三个 basis 并核对 target shells。"""
    require(Path(path).is_file(), f"missing radial operator: {path}")
    with np.load(path, allow_pickle=False) as data:
        required = (
            "target_s_edges",
            "xi_basis_pk_dd",
            "xi_basis_alpha_pk_dd",
            "xi_basis_alpha2_pk_dd",
        )
        for key in required:
            require(key in data.files, f"operator key={key!r} missing in {path}")
        edges = np.asarray(data["target_s_edges"], dtype="f8")
        basis = np.column_stack(
            [
                np.asarray(data["xi_basis_pk_dd"], dtype="f8"),
                np.asarray(data["xi_basis_alpha_pk_dd"], dtype="f8"),
                np.asarray(data["xi_basis_alpha2_pk_dd"], dtype="f8"),
            ]
        )
    require(edges.shape == expected_edges.shape and np.allclose(edges, expected_edges, rtol=0.0, atol=1.0e-10), (
        f"operator target edges do not match xi in {path}"
    ))
    require(basis.shape == (expected_edges.size - 1, 3), f"operator basis shape mismatch in {path}: {basis.shape}")
    require(bool(np.all(np.isfinite(basis))), f"non-finite operator basis in {path}")
    return {"path": canonical(path), "s_edges": edges, "basis": basis}


def validate_fit_provenance(
    summary: dict[str, Any],
    *,
    rmax: int,
    expected_s: np.ndarray,
    xi_path: Path,
    covariance_path: Path,
    covariance_key: str,
    operator_path: Path,
) -> dict[str, Any]:
    """硬检查五条 fit 除 smax 外完全同一科学口径。"""
    fit_range = summary["fit_range"]
    expected_nbins = int(expected_s.size)
    require(float(fit_range["rmin"]) == 50.0 and float(fit_range["rmax"]) == float(rmax), (
        f"fit range mismatch for rmax={rmax}: {fit_range}"
    ))
    require(int(fit_range["nbins"]) == expected_nbins, f"fit nbins mismatch for rmax={rmax}")
    require(np.allclose(np.asarray(fit_range["s_centers"], dtype="f8"), expected_s, rtol=0.0, atol=1.0e-10), (
        f"fit centers mismatch for rmax={rmax}"
    ))
    require(summary.get("fit_target") == "mean", f"fit_target is not mean for rmax={rmax}")
    require(same_path(summary["xi_path"], xi_path), f"xi provenance mismatch for rmax={rmax}")
    covariance = summary["covariance"]
    require(covariance.get("mode") == "npz", f"covariance mode mismatch for rmax={rmax}")
    require(covariance.get("key") == covariance_key, f"covariance key mismatch for rmax={rmax}")
    require(same_path(covariance["path"], covariance_path), f"covariance path mismatch for rmax={rmax}")

    theory = summary["theory"]
    required_theory = bool(
        float(theory["boxsize"]) == 2000.0
        and float(theory["p_fixed"]) == 1.0
        and str(theory["png_order"]) == "full"
        and str(theory["xi_kernel"]) == "shell-averaged"
        and str(theory["sn0_policy"]) == "fixed"
        and float(theory["sn0_fixed"]) == 0.0
        and str(theory["cosmology"]) == "abacus_c000"
    )
    require(required_theory, f"theory/p/sn0/kernel/cosmology mismatch for rmax={rmax}")
    model = radial_model(summary)
    require(model["parameter_names"] == ["fnl_loc", "b1"], f"unexpected free parameters for rmax={rmax}")
    radial = model["radial_singleterm"]
    require(radial.get("extra_global_sigma_w2") is False, f"extra sigmaW2 enabled for rmax={rmax}")
    require(same_path(radial["operator"]["path"], operator_path), f"operator provenance mismatch for rmax={rmax}")
    return {
        "fit_target": "mean",
        "rmin": 50.0,
        "rmax": float(rmax),
        "nbins": expected_nbins,
        "covariance_key": covariance_key,
        "theory_boxsize": 2000.0,
        "p_fixed": 1.0,
        "png_order": "full",
        "xi_kernel": "shell-averaged",
        "sn0_policy": "fixed_zero",
        "cosmology": "abacus_c000",
        "model": "xi_noIC - IC^(rad,rad)",
    }


def fit_result(
    fit_dir: Path,
    *,
    rmax: int,
    xi: dict[str, Any],
    covariance: dict[str, Any],
    operator: dict[str, Any],
) -> dict[str, Any]:
    """读取并完整审计一档 fit。"""
    summary_path = canonical(Path(fit_dir) / SUMMARY_NAME)
    samples_path = canonical(Path(fit_dir) / SAMPLES_NAME)
    summary = read_json(summary_path)
    mask = (xi["s"] >= 50.0) & (xi["s"] <= float(rmax))
    selected_s = xi["s"][mask]
    selected_indices = np.flatnonzero(mask)
    selected_edges = xi["s_edges"][: selected_indices.size + 1]
    require(np.array_equal(selected_indices, np.arange(selected_indices.size)), (
        f"fit crop is not a leading nested block for rmax={rmax}: {selected_indices}"
    ))
    require(float(selected_edges[0]) == 50.0 and float(selected_edges[-1]) == float(rmax), (
        f"rmax={rmax} is not the selected upper separation edge: {selected_edges[[0, -1]]}"
    ))
    provenance = validate_fit_provenance(
        summary,
        rmax=rmax,
        expected_s=selected_s,
        xi_path=xi["path"],
        covariance_path=covariance["path"],
        covariance_key=covariance["key"],
        operator_path=operator["path"],
    )
    chain = load_chain(summary, samples_path, require_map_arrays=True)
    residual = np.asarray(chain["residual_map"], dtype="f8")
    prediction = np.asarray(chain["prediction_map"], dtype="f8")
    require(residual.shape == selected_s.shape, f"MAP residual shape mismatch for rmax={rmax}: {residual.shape}")
    require(prediction.shape == selected_s.shape, f"MAP prediction shape mismatch for rmax={rmax}: {prediction.shape}")
    require(bool(np.all(np.isfinite(residual)) and np.all(np.isfinite(prediction))), (
        f"non-finite MAP prediction/residual for rmax={rmax}"
    ))
    model = radial_model(summary)
    summary_residual = np.asarray(model["data"]["residual_map"], dtype="f8")
    summary_prediction = np.asarray(model["data"]["prediction_map"], dtype="f8")
    require(np.allclose(residual, summary_residual, rtol=0.0, atol=0.0), (
        f"summary/NPZ residual_map mismatch for rmax={rmax}"
    ))
    require(np.allclose(prediction, summary_prediction, rtol=0.0, atol=0.0), (
        f"summary/NPZ prediction_map mismatch for rmax={rmax}"
    ))

    submatrix = covariance["matrix"][np.ix_(mask, mask)]
    recomputed_chi2 = float(residual @ np.linalg.solve(submatrix, residual))
    stored_chi2 = float(model["data"]["chi2_map_total"])
    chi2_close = bool(np.isclose(recomputed_chi2, stored_chi2, rtol=1.0e-6, atol=1.0e-8))
    public = {
        "rmax": rmax,
        "nbins": int(selected_s.size),
        "selected_indices": [int(v) for v in selected_indices],
        "selected_edges": [float(v) for v in selected_edges],
        "selected_edge_min": float(selected_edges[0]),
        "selected_edge_max": float(selected_edges[-1]),
        "last_bin_center": float(selected_s[-1]),
        "paths": {"fit_dir": canonical(fit_dir), "summary": summary_path, "samples": samples_path},
        "provenance": provenance,
        "mcmc": chain["mcmc"],
        "chain_diagnostics": chain["diagnostics"],
        "posterior": chain["posterior"],
        "map_fit": {
            "chi2_total_saved": stored_chi2,
            "chi2_total_recomputed": recomputed_chi2,
            "saved_recomputed_close": chi2_close,
        },
        "covariance_submatrix": covariance["nested_audits"][str(rmax)],
    }
    return {
        "public": public,
        "summary": summary,
        "samples": chain["samples"],
        "residual": residual,
        "prediction": prediction,
        "mask": mask,
    }


def xi_bridge(new: dict[str, Any], old: dict[str, Any], old_covariance: np.ndarray) -> dict[str, Any]:
    """比较新 50-bin xi 的前 30 bins 与旧权威 xi。"""
    require(np.allclose(new["s"][:30], old["s"], rtol=0.0, atol=1.0e-10), "xi bridge centers mismatch")
    require(np.allclose(new["s_edges"][:31], old["s_edges"], rtol=0.0, atol=1.0e-10), (
        "xi bridge edges mismatch"
    ))
    difference = np.asarray(new["xi"][:30] - old["xi"], dtype="f8")
    old_sigma = np.sqrt(np.diag(old_covariance))
    max_sigma = float(np.max(np.abs(difference) / old_sigma))
    return {
        "max_abs_delta_xi": float(np.max(np.abs(difference))),
        "rms_delta_xi": float(np.sqrt(np.mean(difference**2))),
        "max_abs_delta_over_old_cov_sigma": max_sigma,
        "threshold_max_abs_delta_over_old_cov_sigma": 0.05,
        "pass": bool(max_sigma < 0.05),
    }


def covariance_bridge(new: np.ndarray, old: np.ndarray) -> dict[str, Any]:
    """比较新 full covariance 左上 30x30 与旧权威 covariance。"""
    test = np.asarray(new[:30, :30], dtype="f8")
    reference = np.asarray(old, dtype="f8")
    difference = test - reference
    relative_frobenius = float(np.linalg.norm(difference) / np.linalg.norm(reference))
    correlation_delta = correlation_matrix(test) - correlation_matrix(reference)
    sigma_ratio = np.sqrt(np.diag(test) / np.diag(reference))
    max_corr = float(np.max(np.abs(correlation_delta)))
    max_sigma_fraction = float(np.max(np.abs(sigma_ratio - 1.0)))
    passed = bool(relative_frobenius < 0.05 and max_corr < 0.05 and max_sigma_fraction < 0.05)
    return {
        "relative_frobenius_delta": relative_frobenius,
        "max_abs_correlation_delta": max_corr,
        "sigma_ratio_min": float(np.min(sigma_ratio)),
        "sigma_ratio_max": float(np.max(sigma_ratio)),
        "max_abs_sigma_ratio_minus_one": max_sigma_fraction,
        "thresholds": {
            "relative_frobenius_delta": 0.05,
            "max_abs_correlation_delta": 0.05,
            "max_abs_sigma_ratio_minus_one": 0.05,
        },
        "pass": passed,
    }


def operator_bridge(new: np.ndarray, old: np.ndarray) -> dict[str, Any]:
    """逐 basis 比较新 50-bin operator 前 30 行与旧 operator。"""
    names = ("pk_dd", "alpha_pk_dd", "alpha2_pk_dd")
    rows: dict[str, Any] = {}
    for index, name in enumerate(names):
        test = np.asarray(new[:30, index], dtype="f8")
        reference = np.asarray(old[:, index], dtype="f8")
        difference = test - reference
        denominator = max(float(np.sqrt(np.mean(reference**2))), 1.0e-300)
        rows[name] = {
            "max_abs_delta": float(np.max(np.abs(difference))),
            "relative_rms_delta": float(np.sqrt(np.mean(difference**2)) / denominator),
            "allclose_rtol1e-10_atol1e-14": bool(
                np.allclose(test, reference, rtol=1.0e-10, atol=1.0e-14)
            ),
        }
    return {"basis": rows, "pass": bool(all(row["allclose_rtol1e-10_atol1e-14"] for row in rows.values()))}


def posterior_bridge(new: dict[str, Any], old: dict[str, Any]) -> dict[str, Any]:
    """比较新旧 350 raw-chain fNL posterior，并执行既定 bridge gate。"""
    new_fnl = new["posterior"]["fnl_loc"]
    old_fnl = old["posterior"]["fnl_loc"]
    delta = float(new_fnl["q50"] - old_fnl["q50"])
    shift = float(delta / old_fnl["sigma68"])
    width_ratio = float(new_fnl["sigma68"] / old_fnl["sigma68"])
    center_pass = bool(abs(shift) < 0.1)
    width_pass = bool(abs(width_ratio - 1.0) < 0.05)
    return {
        "old_fnl": old_fnl,
        "new_fnl": new_fnl,
        "delta_median": delta,
        "delta_median_over_old_sigma68": shift,
        "sigma68_ratio_new_over_old": width_ratio,
        "gate": {
            "absolute_center_shift_below_0p1_sigma": center_pass,
            "width_ratio_within_5percent": width_pass,
            "pass": bool(center_pass and width_pass),
        },
    }


def conditional_residual(
    residual: np.ndarray,
    covariance: np.ndarray,
    *,
    old_size: int,
    new_size: int,
    nreal: int,
    old_rmax: int,
    new_rmax: int,
) -> dict[str, Any]:
    """计算相邻新增 block 在旧 block 条件下的 Schur residual 与 PTE。"""
    require(new_size > old_size, "conditional residual requires a non-empty new block")
    selected_covariance = np.asarray(covariance[:new_size, :new_size], dtype="f8")
    residual = np.asarray(residual[:new_size], dtype="f8")
    c_aa = selected_covariance[:old_size, :old_size]
    c_ab = selected_covariance[:old_size, old_size:new_size]
    c_ba = selected_covariance[old_size:new_size, :old_size]
    c_bb = selected_covariance[old_size:new_size, old_size:new_size]
    r_a = residual[:old_size]
    r_b = residual[old_size:new_size]
    conditional_vector = r_b - c_ba @ np.linalg.solve(c_aa, r_a)
    schur = c_bb - c_ba @ np.linalg.solve(c_aa, c_ab)
    schur = 0.5 * (schur + schur.T)
    schur_audit = matrix_audit(schur)
    require(schur_audit["pass"], f"Schur covariance failed SPD gate for {old_rmax}->{new_rmax}")
    raw_chi2 = float(conditional_vector @ np.linalg.solve(schur, conditional_vector))
    scaled_chi2 = float(nreal * raw_chi2)
    degrees_of_freedom = int(new_size - old_size)
    pte = float(chi2_distribution.sf(scaled_chi2, degrees_of_freedom))
    anomaly = bool(pte < 0.01)
    return {
        "old_rmax": old_rmax,
        "new_rmax": new_rmax,
        "old_nbins": old_size,
        "new_nbins": new_size,
        "added_nbins": degrees_of_freedom,
        "conditional_residual": [float(v) for v in conditional_vector],
        "schur_covariance_audit": schur_audit,
        "raw_chi2_cond_single_lightcone_covariance": raw_chi2,
        "mean_data_scaled_chi2_cond": scaled_chi2,
        "mean_data_scale_factor_nreal": nreal,
        "df": degrees_of_freedom,
        "pte": pte,
        "anomaly_threshold_pte": 0.01,
        "anomaly": anomaly,
        "pass": not anomaly,
    }


def csv_rows(fits: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """把 nested JSON fit rows 转为稳定的会议表格。"""
    output: list[dict[str, Any]] = []
    for fit in fits:
        fnl = fit["posterior"]["fnl_loc"]
        b1 = fit["posterior"]["b1"]
        conditional = fit.get("conditional_from_previous")
        output.append(
            {
                "rmax": fit["rmax"],
                "nbins": fit["nbins"],
                "fnl_q16": fnl["q16"],
                "fnl_q50": fnl["q50"],
                "fnl_q84": fnl["q84"],
                "fnl_err_low": fnl["err_low"],
                "fnl_err_high": fnl["err_high"],
                "fnl_sigma68": fnl["sigma68"],
                "b1_q16": b1["q16"],
                "b1_q50": b1["q50"],
                "b1_q84": b1["q84"],
                "corr_fnl_b1": fit["posterior"]["corr_fnl_b1"],
                "width_ratio_to_350": fit["comparison_to_350"]["sigma68_ratio"],
                "improvement_fraction_to_350": fit["comparison_to_350"]["improvement_fraction"],
                "center_shift_over_sigma350": fit["comparison_to_350"]["center_shift_over_sigma350"],
                "robust_improvement_gate": fit["comparison_to_350"]["robust_improvement_gate"],
                "map_chi2_total": fit["map_fit"]["chi2_total_saved"],
                "conditional_raw_chi2": "" if conditional is None else conditional["raw_chi2_cond_single_lightcone_covariance"],
                "conditional_scaled_chi2": "" if conditional is None else conditional["mean_data_scaled_chi2_cond"],
                "conditional_df": "" if conditional is None else conditional["df"],
                "conditional_pte": "" if conditional is None else conditional["pte"],
                "conditional_anomaly": "" if conditional is None else conditional["anomaly"],
                "tau_length_ratio_min": fit["chain_diagnostics"]["length_over_tau_min"],
                "split_shift_sigma_max": fit["chain_diagnostics"]["split_median_shift_sigma_max"],
                "chain_gate": fit["chain_diagnostics"]["pass"],
                "covariance_eig_min": fit["covariance_submatrix"]["eig_min"],
                "covariance_condition_number": fit["covariance_submatrix"]["condition_number"],
                "covariance_gate": fit["covariance_submatrix"]["pass"],
                "seed": fit["mcmc"]["seed"],
                "summary_path": str(fit["paths"]["summary"]),
                "samples_path": str(fit["paths"]["samples"]),
            }
        )
    return output


def render_pdf(payload: dict[str, Any], output: Path) -> None:
    """绘制 interval ladder 与相对宽度；科学 gate 失败时绝不调用。"""
    rows = payload["fits"]
    x = np.asarray([row["rmax"] for row in rows], dtype="f8")
    q50 = np.asarray([row["posterior"]["fnl_loc"]["q50"] for row in rows], dtype="f8")
    error_low = np.asarray([row["posterior"]["fnl_loc"]["err_low"] for row in rows], dtype="f8")
    error_high = np.asarray([row["posterior"]["fnl_loc"]["err_high"] for row in rows], dtype="f8")
    width_ratio = np.asarray([row["comparison_to_350"]["sigma68_ratio"] for row in rows], dtype="f8")
    nbins = [int(row["nbins"]) for row in rows]

    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "DejaVu Sans", "Liberation Sans"],
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "axes.linewidth": 1.0,
            "font.size": 11.5,
        }
    )
    figure, (upper, lower) = plt.subplots(
        2,
        1,
        figsize=(7.4, 7.2),
        sharex=True,
        gridspec_kw={"height_ratios": [1.65, 1.0], "hspace": 0.08},
    )
    color = "#C44E52"
    upper.errorbar(
        x,
        q50,
        yerr=np.vstack([error_low, error_high]),
        fmt="o-",
        color=color,
        markerfacecolor="white",
        markeredgewidth=1.6,
        linewidth=1.8,
        capsize=4.0,
        label=r"$\xi_0(s)$, radial single-term RIC",
    )
    upper.axhline(0.0, color="#777777", linestyle="--", linewidth=1.0, zorder=0)
    upper.set_ylabel(r"$f_{\rm NL}$ (68% interval)")
    upper.legend(loc="best", frameon=False)
    upper.grid(alpha=0.18)
    for xpos, center, high, count in zip(x, q50, error_high, nbins, strict=True):
        upper.annotate(
            f"{count} bins",
            (xpos, center + high),
            xytext=(0, 7),
            textcoords="offset points",
            ha="center",
            va="bottom",
            fontsize=8.5,
            color="#555555",
        )

    lower.plot(x, width_ratio, "o-", color=color, linewidth=1.8, markerfacecolor="white", markeredgewidth=1.6)
    lower.axhline(1.0, color="#555555", linestyle="-", linewidth=1.0)
    lower.axhline(0.95, color="#777777", linestyle="--", linewidth=1.0, label="5% narrower")
    lower.set_xlabel(r"$s_{\max}\ [h^{-1}{\rm Mpc}]$")
    lower.set_ylabel(r"$\sigma_{68}(s_{\max})/\sigma_{68}(350)$")
    lower.set_xticks(x)
    lower.grid(alpha=0.18)
    lower.legend(loc="best", frameon=False, fontsize=9.0)

    decision = payload["decision"]
    if decision["robust_improvement_at_550"]:
        banner = "Robust >=5% improvement at smax=550 within the tested approximation"
        banner_color = "#2F6B3F"
    else:
        banner = "No robust >=5% improvement at smax=550 within the tested approximation"
        banner_color = "#8B2E2E"
    if decision["any_conditional_residual_anomaly"]:
        banner += "; conditional residual anomaly present"
    figure.text(0.5, 0.985, banner, ha="center", va="top", fontsize=10.0, color=banner_color)
    figure.text(
        0.5,
        0.012,
        "Fixed jaxpower diagnostic covariance; single IC(rad,rad) term only; not science-ready",
        ha="center",
        va="bottom",
        fontsize=8.4,
        color="#555555",
    )
    figure.subplots_adjust(left=0.14, right=0.97, top=0.94, bottom=0.13)
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(f".{output.name}.{os.getpid()}.tmp.pdf")
    figure.savefig(temporary, format="pdf", bbox_inches="tight")
    plt.close(figure)
    temporary.replace(output)


def analyze(args: argparse.Namespace) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """执行完整只读审计并返回 JSON payload 与 CSV rows。"""
    fit_dirs = fit_directories(args.fit_root, args.fit_dir)
    xi = load_xi(canonical(args.xi_path), expected_nbins=50, expected_nreal=int(args.nreal))
    expected_edges = np.arange(50.0, 551.0, 10.0)
    require(np.allclose(xi["s_edges"], expected_edges, rtol=0.0, atol=1.0e-10), (
        f"scan xi is not s_edges=50..550 ds=10: {xi['s_edges']}"
    ))
    covariance = load_covariance(
        canonical(args.covariance_path),
        key=str(args.covariance_key),
        expected_s=xi["s"],
        expected_edges=xi["s_edges"],
    )
    operator = load_operator(canonical(args.operator_path), expected_edges=xi["s_edges"])

    internal_fits = [
        fit_result(
            fit_dirs[rmax],
            rmax=rmax,
            xi=xi,
            covariance=covariance,
            operator=operator,
        )
        for rmax in RMAX_VALUES
    ]
    common_mcmc = [
        (
            row["public"]["mcmc"]["nwalkers"],
            row["public"]["mcmc"]["nsteps"],
            row["public"]["mcmc"]["burnin"],
            row["public"]["mcmc"]["seed"],
        )
        for row in internal_fits
    ]
    require(all(item == common_mcmc[0] for item in common_mcmc), f"five fits do not share MCMC settings/seed: {common_mcmc}")
    require(common_mcmc[0][-1] is not None, "new fit summaries do not record MCMC seed")

    old_xi = load_xi(canonical(args.old_xi_path), expected_nbins=30, expected_nreal=int(args.nreal))
    old_covariance = load_covariance(
        canonical(args.old_covariance_path),
        key=str(args.covariance_key),
        expected_s=old_xi["s"],
        expected_edges=old_xi["s_edges"],
        nested_rmax_values=(350,),
    )
    old_operator = load_operator(canonical(args.old_operator_path), expected_edges=old_xi["s_edges"])
    old_summary_path = canonical(args.old_summary) if args.old_summary is not None else canonical(Path(args.old_fit_dir) / SUMMARY_NAME)
    old_samples_path = canonical(args.old_samples) if args.old_samples is not None else canonical(Path(args.old_fit_dir) / SAMPLES_NAME)
    old_summary = read_json(old_summary_path)
    old_provenance = validate_fit_provenance(
        old_summary,
        rmax=350,
        expected_s=old_xi["s"],
        xi_path=old_xi["path"],
        covariance_path=old_covariance["path"],
        covariance_key=old_covariance["key"],
        operator_path=old_operator["path"],
    )
    old_chain = load_chain(old_summary, old_samples_path, require_map_arrays=False)

    bridge_xi = xi_bridge(xi, old_xi, old_covariance["matrix"])
    bridge_covariance = covariance_bridge(covariance["matrix"], old_covariance["matrix"])
    bridge_operator = operator_bridge(operator["basis"], old_operator["basis"])
    bridge_posterior = posterior_bridge(internal_fits[0]["public"], old_chain)
    bridge = {
        "xi_first30": bridge_xi,
        "covariance_first30": bridge_covariance,
        "operator_first30": bridge_operator,
        "old_fit_provenance": old_provenance,
        "old_chain_diagnostics": old_chain["diagnostics"],
        "posterior_new350_vs_old": bridge_posterior,
    }
    bridge["pass"] = bool(
        bridge_xi["pass"]
        and bridge_covariance["pass"]
        and bridge_operator["pass"]
        and old_chain["diagnostics"]["pass"]
        and bridge_posterior["gate"]["pass"]
    )

    conditional_rows: list[dict[str, Any]] = []
    for index in range(1, len(internal_fits)):
        previous = internal_fits[index - 1]
        current = internal_fits[index]
        conditional = conditional_residual(
            current["residual"],
            covariance["matrix"],
            old_size=int(previous["public"]["nbins"]),
            new_size=int(current["public"]["nbins"]),
            nreal=int(xi["nreal"]),
            old_rmax=int(previous["public"]["rmax"]),
            new_rmax=int(current["public"]["rmax"]),
        )
        current["public"]["conditional_from_previous"] = conditional
        conditional_rows.append(conditional)
    internal_fits[0]["public"]["conditional_from_previous"] = None

    baseline_fnl = internal_fits[0]["public"]["posterior"]["fnl_loc"]
    cumulative_residual_pass = True
    for index, row in enumerate(internal_fits):
        if index > 0:
            cumulative_residual_pass = bool(cumulative_residual_pass and conditional_rows[index - 1]["pass"])
        fnl = row["public"]["posterior"]["fnl_loc"]
        width_ratio = float(fnl["sigma68"] / baseline_fnl["sigma68"])
        shift = float((fnl["q50"] - baseline_fnl["q50"]) / baseline_fnl["sigma68"])
        improvement = float(1.0 - width_ratio)
        row["public"]["comparison_to_350"] = {
            "sigma68_ratio": width_ratio,
            "improvement_fraction": improvement,
            "center_shift_over_sigma350": shift,
            "improvement_at_least_5percent": bool(improvement >= 0.05),
            "absolute_center_shift_below_0p2_sigma": bool(abs(shift) < 0.2),
            "all_conditional_residuals_through_this_rmax_pass": cumulative_residual_pass,
            "robust_improvement_gate": bool(
                index > 0 and improvement >= 0.05 and abs(shift) < 0.2 and cumulative_residual_pass
            ),
        }

    public_fits = [row["public"] for row in internal_fits]
    hard_failures: list[str] = []
    for row in public_fits:
        if not row["chain_diagnostics"]["pass"]:
            hard_failures.append(f"rmax={row['rmax']} chain convergence gate failed")
        if not row["covariance_submatrix"]["pass"]:
            hard_failures.append(f"rmax={row['rmax']} covariance submatrix gate failed")
        if not row["map_fit"]["saved_recomputed_close"]:
            hard_failures.append(f"rmax={row['rmax']} saved/recomputed MAP chi2 mismatch")
    if not bridge["pass"]:
        hard_failures.append("350 bridge gate failed")

    final_comparison = public_fits[-1]["comparison_to_350"]
    any_residual_anomaly = bool(any(row["anomaly"] for row in conditional_rows))
    robust_550 = bool(final_comparison["robust_improvement_gate"])
    if robust_550:
        classification = "robust_relative_improvement_within_fixed_model_covariance"
    elif final_comparison["improvement_at_least_5percent"] and (
        not final_comparison["absolute_center_shift_below_0p2_sigma"] or any_residual_anomaly
    ):
        classification = "tail_tension_or_model_covariance_dependent"
    elif final_comparison["improvement_fraction"] > 0.0:
        classification = "weak_relative_improvement_below_5percent"
    else:
        classification = "no_relative_improvement"
    status = "done" if not hard_failures else "fail"
    statement = (
        "Relative smax information test under one fixed RR-deconvolved jaxpower diagnostic "
        "single-lightcone covariance and the radial single IC^(rad,rad) mean-model term; "
        "density-RIC cross terms and connected covariance terms are absent, so this is not a science-ready constraint."
    )
    payload = {
        "task": "task43_summarize_rmax_scan",
        "status": status,
        "created_utc": utc_now(),
        "scope": {
            "statement": statement,
            "covariance": "fixed RR-deconvolved jaxpower diagnostic single-lightcone covariance",
            "mean_model": "xi_noIC - IC^(rad,rad)",
            "density_ric_cross_terms": False,
            "extra_global_sigma_w2": False,
            "science_ready": False,
            "caveats": [
                "theory/RIC uses mother-box k>=2pi/L while the inherited jaxpower covariance integrates from k=1e-4",
                "jaxpower covariance is Gaussian ph000 50k/100k and omits connected/full Landy-Szalay covariance terms",
                "mesh64 pad400 and ell=0 window/RR angular treatment were inherited rather than reconverged over 350--550",
                "radial RIC keeps only IC^(rad,rad); density-RIC cross terms and independent tail convergence tests are absent",
                "the five ranges are correlated nested crops of the same vector and covariance",
            ],
        },
        "experiment": {
            "rmin": 50.0,
            "rmax_values": list(RMAX_VALUES),
            "bin_width": 10.0,
            "nbins_values": [30, 35, 40, 45, 50],
            "nreal_mean": int(xi["nreal"]),
            "fit_covariance_key": covariance["key"],
            "posterior_quantile_probabilities": list(GAUSSIAN_QUANTILES),
            "posterior_quantile_policy": "all old/new raw chains recomputed identically",
            "common_mcmc": {
                "nwalkers": common_mcmc[0][0],
                "nsteps": common_mcmc[0][1],
                "burnin": common_mcmc[0][2],
                "seed": common_mcmc[0][3],
            },
        },
        "inputs": {
            "xi_50bin": xi["path"],
            "covariance_50bin": covariance["path"],
            "operator_50bin": operator["path"],
            "fit_directories": {str(key): value for key, value in fit_dirs.items()},
            "old_xi_30bin": old_xi["path"],
            "old_covariance_30bin": old_covariance["path"],
            "old_operator_30bin": old_operator["path"],
            "old_fit_summary": old_summary_path,
            "old_fit_samples": old_samples_path,
        },
        "covariance_audit": {
            "coordinates_match_xi": True,
            "full": covariance["full_audit"],
            "nested": covariance["nested_audits"],
        },
        "bridge": bridge,
        "fits": public_fits,
        "conditional_residuals": conditional_rows,
        "hard_gates": {
            "requirements": [
                "all five chains: length/tau > 100 and split median shift < 0.05 sigma",
                "full and all nested covariance subblocks are coordinate-matched SPD",
                "saved and recomputed MAP chi2 agree",
                "xi/covariance/operator/posterior 350 bridge passes",
            ],
            "failures": hard_failures,
            "pass": not hard_failures,
        },
        "decision": {
            "target_rmax": 550,
            "width_ratio_550_over_350": final_comparison["sigma68_ratio"],
            "improvement_fraction_550_over_350": final_comparison["improvement_fraction"],
            "center_shift_over_sigma350": final_comparison["center_shift_over_sigma350"],
            "improvement_at_least_5percent": final_comparison["improvement_at_least_5percent"],
            "absolute_center_shift_below_0p2_sigma": final_comparison["absolute_center_shift_below_0p2_sigma"],
            "any_conditional_residual_anomaly": any_residual_anomaly,
            "robust_improvement_at_550": robust_550,
            "classification": classification,
            "model_covariance_dependent": True,
            "interpretation_scope": statement,
        },
        "outputs": {
            "json": canonical(args.output_json),
            "csv": canonical(args.output_csv),
            "pdf": canonical(args.output_pdf),
            "pdf_written": False,
        },
    }
    return payload, csv_rows(public_fits)


def parse_args() -> argparse.Namespace:
    """定义可复现默认路径及必要 CLI overrides。"""
    parser = argparse.ArgumentParser(description="Audit and plot the Task43 jaxpower-only 2PCF rmax scan.")
    parser.add_argument("--xi-path", type=Path, default=DEFAULT_XI)
    parser.add_argument("--covariance-path", type=Path, default=DEFAULT_COVARIANCE)
    parser.add_argument("--covariance-key", type=str, default="covariance_single_realization")
    parser.add_argument("--operator-path", type=Path, default=DEFAULT_OPERATOR)
    parser.add_argument("--fit-root", type=Path, default=DEFAULT_FIT_ROOT)
    parser.add_argument(
        "--fit-dir",
        action="append",
        default=[],
        metavar="RMAX=PATH",
        help="Override one canonical fits/smaxNNN directory; repeat for multiple rmax values.",
    )
    parser.add_argument("--old-xi-path", type=Path, default=DEFAULT_OLD_XI)
    parser.add_argument("--old-covariance-path", type=Path, default=DEFAULT_OLD_COVARIANCE)
    parser.add_argument("--old-operator-path", type=Path, default=DEFAULT_OLD_OPERATOR)
    parser.add_argument("--old-fit-dir", type=Path, default=DEFAULT_OLD_FIT_DIR)
    parser.add_argument("--old-summary", type=Path, default=None)
    parser.add_argument("--old-samples", type=Path, default=None)
    parser.add_argument("--nreal", type=int, default=25)
    parser.add_argument("--output-json", type=Path, default=DEFAULT_OUTPUT_JSON)
    parser.add_argument("--output-csv", type=Path, default=DEFAULT_OUTPUT_CSV)
    parser.add_argument("--output-pdf", type=Path, default=DEFAULT_OUTPUT_PDF)
    return parser.parse_args()


def failure_payload(args: argparse.Namespace, error: BaseException) -> dict[str, Any]:
    """异常时写最小但明确的 status=fail audit，不触碰 PDF。"""
    output_pdf = canonical(args.output_pdf)
    output_csv = canonical(args.output_csv)
    return {
        "task": "task43_summarize_rmax_scan",
        "status": "fail",
        "created_utc": utc_now(),
        "error": {"type": type(error).__name__, "message": str(error), "traceback": traceback.format_exc()},
        "arguments": jsonable(vars(args)),
        "outputs": {
            "json": canonical(args.output_json),
            "csv": output_csv,
            "pdf": output_pdf,
            "pdf_written": False,
            "stale_csv_already_exists": output_csv.exists(),
            "stale_pdf_already_exists": output_pdf.exists(),
            "policy": "validation failure writes JSON only and never creates/replaces the CSV or PDF",
        },
    }


def main() -> int:
    """运行审计；成功才写三件套，失败只写清楚的 JSON 并返回非零。"""
    args = parse_args()
    output_json = canonical(args.output_json)
    try:
        payload, rows = analyze(args)
        if payload["status"] != "done":
            payload["outputs"]["stale_csv_already_exists"] = canonical(args.output_csv).exists()
            payload["outputs"]["stale_pdf_already_exists"] = canonical(args.output_pdf).exists()
            payload["outputs"]["policy"] = "hard-gate failure writes JSON only; CSV/PDF are not created or replaced"
            atomic_write_json(output_json, payload)
            print(f"[fail] hard gates failed; wrote JSON only: {output_json}", file=sys.stderr)
            return 1
        render_pdf(payload, canonical(args.output_pdf))
        atomic_write_csv(canonical(args.output_csv), rows)
        payload["outputs"]["pdf_written"] = True
        atomic_write_json(output_json, payload)
        print(f"[write] {output_json}")
        print(f"[write] {canonical(args.output_csv)}")
        print(f"[write] {canonical(args.output_pdf)}")
        return 0
    except Exception as error:  # noqa: BLE001 - audit 必须把任何失败写成机器可读状态
        payload = failure_payload(args, error)
        atomic_write_json(output_json, payload)
        print(f"[fail] {type(error).__name__}: {error}", file=sys.stderr)
        print(f"[write] failure audit: {output_json}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
