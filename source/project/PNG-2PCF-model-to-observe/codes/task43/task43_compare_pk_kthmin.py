#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""审计 Task43 P(k) 的 window-theory kth,min A/B。

代码执行大纲
============
1. 读取同 seed 重跑的 mother-box cutoff reference 与 kth,min=0.001 test。
2. 逐项检查 payload、观测 bins、covariance、参数基、先验和 MCMC 设置，
   确保除 window_theory_kmin 外没有科学口径变化。
3. 汇总 fNL、b1、sn0 posterior，并计算中心移动和误差宽度比例。
4. 从 window payload 检查两个 cutoff 实际保留了哪些离散 theory bins，
   以及新增低 k 列对 window matrix 的权重。
5. 在 reference MAP 参数不变时，只切换 cutoff，计算预测变化的
   covariance-metric 大小；同时比较 fNL 一阶响应。
6. 对保存的 MCMC 链做 autocorrelation-time 与前后半链稳定性检查。
7. 写出一个 JSON 总审计和一个紧凑 CSV，不生成图片。
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np


PROJECT_ROOT = Path("/pscratch/sd/l/lzy/PNG-2PCF-model-to-observe")
CODE_DIR = PROJECT_ROOT / "codes" / "task43"
if str(CODE_DIR) not in sys.path:
    sys.path.insert(0, str(CODE_DIR))

FIT_ROOT = PROJECT_ROOT / "outputs" / "task43_outputs" / "pk_lightcone" / "fits"
SUMMARY_DIR = PROJECT_ROOT / "outputs" / "task43_outputs" / "summary"
LABEL_PREFIX = "mmin1p4e13_x25_fkpP010000_desi_rebin_kmax0p10_free_sn0_wtheorykmin_"
DEFAULT_REFERENCE_DIR = FIT_ROOT / f"{LABEL_PREFIX}boxL2000_mcmc5x_seed20260712"
DEFAULT_TEST_DIR = FIT_ROOT / f"{LABEL_PREFIX}k001_mcmc5x_seed20260712"
DEFAULT_PAYLOAD = (
    PROJECT_ROOT
    / "outputs/task43_outputs/pk_lightcone/summary/"
    "task43_pk_lightcone_mmin1p4e13_x25_fkpP010000_"
    "desi_rebin_kmax0p10_payload.npz"
)
DEFAULT_OUTPUT = SUMMARY_DIR / "task43_pk_kthmin_0p001_ab_20260712"


def summary_path(directory: Path) -> Path:
    """根据 fit 目录名构造该目录内唯一的 Task43 P(k) summary 路径。"""
    label = directory.name
    return directory / f"task43_pk_lightcone_{label}_fit_summary.json"


def samples_path(directory: Path) -> Path:
    """根据 fit 目录名构造该目录内唯一的 Task43 P(k) samples 路径。"""
    label = directory.name
    return directory / f"task43_pk_lightcone_{label}_fit_samples.npz"


def read_json(path: Path) -> dict[str, Any]:
    """读取 JSON 并返回字典。"""
    return json.loads(path.read_text(encoding="utf-8"))


def jsonable(value: Any) -> Any:
    """递归地把 numpy 和 Path 对象转换成 JSON 原生类型。"""
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


def same_value(left: Any, right: Any) -> bool:
    """比较标量、列表和字典；浮点数采用严格到数值舍入误差的比较。"""
    if isinstance(left, dict) and isinstance(right, dict):
        return set(left) == set(right) and all(
            same_value(left[key], right[key]) for key in left
        )
    if isinstance(left, (list, tuple)) and isinstance(right, (list, tuple)):
        return len(left) == len(right) and all(
            same_value(a, b) for a, b in zip(left, right, strict=True)
        )
    if isinstance(left, (float, int)) and isinstance(right, (float, int)):
        return bool(np.isclose(float(left), float(right), rtol=0.0, atol=1.0e-12))
    return left == right


def nested_get(data: dict[str, Any], keys: tuple[str, ...]) -> Any:
    """按 keys 指定的层级读取嵌套字典字段。"""
    out: Any = data
    for key in keys:
        out = out[key]
    return out


def validate_matched(
    reference: dict[str, Any],
    test: dict[str, Any],
) -> dict[str, Any]:
    """验证两条 fit 只有 theory cutoff 不同。

    返回每一项匹配状态；任一要求项不匹配都会直接报错，避免把非受控比较
    写成科学结论。
    """
    required = {
        "payload": ("payload",),
        "parameter_names": ("config", "parameter_names"),
        "fixed_parameters": ("config", "fixed_parameters"),
        "sn0_policy": ("config", "sn0_policy"),
        "p_fixed": ("config", "p_fixed"),
        "sn0_scale": ("config", "sn0_scale"),
        "window_forward_model": ("config", "window_forward_model"),
        "priors": ("config", "priors"),
        "target_evals": ("config", "target_evals"),
        "nwalkers": ("config", "nwalkers"),
        "nsteps": ("config", "nsteps"),
        "burnin": ("config", "burnin"),
        "seed": ("config", "seed"),
        "cosmology": ("config", "cosmology"),
        "data": ("data",),
        "covariance_source": ("input_summary", "covariance", "source"),
        "covariance_path": ("input_summary", "covariance", "path"),
    }
    rows: dict[str, Any] = {}
    failures: list[str] = []
    for name, keys in required.items():
        left = nested_get(reference, keys)
        right = nested_get(test, keys)
        matched = same_value(left, right)
        rows[name] = {"matched": matched, "reference": left, "test": right}
        if not matched:
            failures.append(name)
    cutoff_ref = float(reference["config"]["window_theory_kmin"])
    cutoff_test = float(test["config"]["window_theory_kmin"])
    if np.isclose(cutoff_ref, cutoff_test, rtol=0.0, atol=1.0e-15):
        failures.append("window_theory_kmin_not_changed")
    if failures:
        raise ValueError(f"kth,min A/B 不是严格 matched：{failures}")
    return {
        "all_required_fields_matched": True,
        "fields": rows,
        "reference_cutoff": cutoff_ref,
        "test_cutoff": cutoff_test,
        "only_intended_science_change": "window_theory_kmin",
    }


def parameter_stats(summary: dict[str, Any], name: str) -> dict[str, float]:
    """读取一个参数的标准 posterior 数值并补充对称化 68% 半宽。"""
    stats = summary["parameters"][name]
    q16 = float(stats["q16"])
    q50 = float(stats["median"])
    q84 = float(stats["q84"])
    return {
        "q16": q16,
        "q50": q50,
        "q84": q84,
        "err_low": q50 - q16,
        "err_high": q84 - q50,
        "sigma68": 0.5 * (q84 - q16),
        "std": float(stats["std"]),
    }


def posterior_comparison(
    reference: dict[str, Any],
    test: dict[str, Any],
) -> dict[str, Any]:
    """比较 fNL、b1、sn0 posterior 的中心和宽度。"""
    output: dict[str, Any] = {}
    for name in ("fnl_loc", "b1", "sn0"):
        ref = parameter_stats(reference, name)
        tst = parameter_stats(test, name)
        output[name] = {
            "reference": ref,
            "test": tst,
            "delta_q50": tst["q50"] - ref["q50"],
            "delta_q50_over_reference_sigma68": (
                (tst["q50"] - ref["q50"]) / ref["sigma68"]
            ),
            "sigma68_ratio_test_over_reference": tst["sigma68"] / ref["sigma68"],
        }
    ref_chi2 = float(reference["maximum_posterior_sample"]["chi2"])
    test_chi2 = float(test["maximum_posterior_sample"]["chi2"])
    output["fit_quality"] = {
        "reference_chi2": ref_chi2,
        "test_chi2": test_chi2,
        "delta_chi2_test_minus_reference": test_chi2 - ref_chi2,
        "ndata": int(reference["data"]["ndata"]),
        "nfree": len(reference["config"]["free_parameters"]),
    }
    fnl = output["fnl_loc"]
    output["impact_gate"] = {
        "center_shift_below_0p2sigma": bool(
            abs(fnl["delta_q50_over_reference_sigma68"]) < 0.2
        ),
        "sigma68_change_below_10percent": bool(
            abs(fnl["sigma68_ratio_test_over_reference"] - 1.0) < 0.1
        ),
    }
    return output


def load_chain_diagnostics(
    path: Path,
    summary: dict[str, Any],
) -> dict[str, Any]:
    """重建 post-burn walker 链并检查自相关时间和前后半稳定性。"""
    import emcee

    with np.load(path, allow_pickle=False) as data:
        samples = np.asarray(data["samples"], dtype="f8")
        names = [str(value) for value in np.asarray(data["param_names"]).tolist()]
        nsteps = int(np.asarray(data["nsteps"]).item())
        nwalkers = int(np.asarray(data["nwalkers"]).item())
        burnin = int(np.asarray(data["burnin"]).item())
    expected = (nsteps - burnin) * nwalkers
    if samples.shape[0] != expected:
        raise ValueError(
            f"{path} samples={samples.shape[0]}，预期 post-burn={expected}"
        )
    chain = samples.reshape((nsteps - burnin, nwalkers, samples.shape[1]))
    try:
        tau = np.asarray(
            emcee.autocorr.integrated_time(chain, quiet=True),
            dtype="f8",
        )
    except Exception as exc:
        tau = np.full(samples.shape[1], np.nan, dtype="f8")
        tau_error = f"{type(exc).__name__}: {exc}"
    else:
        tau_error = None
    midpoint = chain.shape[0] // 2
    first = chain[:midpoint].reshape((-1, chain.shape[2]))
    second = chain[midpoint:].reshape((-1, chain.shape[2]))
    split: dict[str, Any] = {}
    for index, name in enumerate(names):
        q_first = np.quantile(first[:, index], [0.16, 0.5, 0.84])
        q_second = np.quantile(second[:, index], [0.16, 0.5, 0.84])
        sigma = 0.5 * (
            float(summary["parameters"][name]["q84"])
            - float(summary["parameters"][name]["q16"])
        )
        split[name] = {
            "first_half": q_first,
            "second_half": q_second,
            "delta_median_over_full_sigma68": (
                float(q_second[1] - q_first[1]) / sigma
            ),
        }
    return {
        "path": str(path),
        "nsteps_total": nsteps,
        "burnin": burnin,
        "nsteps_post_burn": int(chain.shape[0]),
        "nwalkers": nwalkers,
        "tau": {name: float(tau[i]) for i, name in enumerate(names)},
        "post_burn_steps_over_tau": {
            name: (
                float(chain.shape[0] / tau[i])
                if np.isfinite(tau[i]) and tau[i] > 0.0
                else None
            )
            for i, name in enumerate(names)
        },
        "tau_error": tau_error,
        "split_half": split,
    }


def cutoff_support(
    payload: Path,
    cutoff_ref: float,
    cutoff_test: float,
) -> dict[str, Any]:
    """报告两个 hard cutoff 在离散 window theory 网格上的实际支持。"""
    with np.load(payload, allow_pickle=False) as data:
        theory_k = np.asarray(data["theory_k"], dtype="f8")
        theory_ell = np.asarray(data["theory_ell"], dtype="i8")
        theory_edges = np.asarray(data["theory_edges"], dtype="f8")
        window = np.asarray(data["window_matrix"], dtype="f8")
    ell0 = theory_ell == 0
    keep_ref = ell0 & (theory_k >= float(cutoff_ref))
    keep_test = ell0 & (theory_k >= float(cutoff_test))
    added = keep_test & ~keep_ref
    ids = np.flatnonzero(added)

    def support_row(mask: np.ndarray, cutoff: float) -> dict[str, Any]:
        first = int(np.flatnonzero(mask)[0])
        return {
            "requested_cutoff": float(cutoff),
            "first_retained_center": float(theory_k[first]),
            "first_retained_edges": theory_edges[first],
            "retained_ell0_bins": int(np.sum(mask)),
            "removed_ell0_bins": int(np.sum(ell0 & ~mask)),
        }

    return {
        "payload": str(payload),
        "reference": support_row(keep_ref, cutoff_ref),
        "test": support_row(keep_test, cutoff_test),
        "newly_added_bin_count": int(ids.size),
        "newly_added_indices": ids,
        "newly_added_centers": theory_k[ids],
        "newly_added_edges": theory_edges[ids],
        "newly_added_window_frobenius_norm": float(np.linalg.norm(window[:, ids])),
        "newly_added_window_max_abs": (
            float(np.max(np.abs(window[:, ids]))) if ids.size else 0.0
        ),
        "implementation_note": (
            "The code zeros complete theory columns based on bin centers; "
            "the requested scalar cutoff is therefore represented by the "
            "first retained finite-width theory bin."
        ),
    }


def fixed_theta_response(
    payload: Path,
    reference: dict[str, Any],
    cutoff_ref: float,
    cutoff_test: float,
) -> dict[str, Any]:
    """在相同参数下只切换 cutoff，量化 windowed prediction 与 fNL 响应。"""
    from task43_fit_pk_lightcone import (
        FitData,
        make_precision,
        model_pk,
    )
    from task43_theory_template import (
        build_template_arrays,
        load_task41,
    )

    data = FitData(payload)
    task41 = load_task41()
    positive = data.theory_k[data.theory_k > 0.0]
    kmin_template = min(1.0e-5, float(np.min(positive)) * 0.5)
    k_template = np.logspace(
        np.log10(kmin_template),
        np.log10(20.0),
        20000,
    )
    template, _ = build_template_arrays(
        task41,
        k_template,
        z=float(data.zeff),
        cosmology=str(reference["config"]["cosmology"]),
    )
    theta = {
        key: float(value)
        for key, value in reference["maximum_posterior_sample"]["point"].items()
    }
    common = {
        "p_fixed": float(reference["config"]["p_fixed"]),
        "sn0_scale": float(reference["config"]["sn0_scale"]),
    }
    pred_ref = model_pk(
        data,
        template,
        theta,
        window_theory_kmin=cutoff_ref,
        **common,
    )
    pred_test = model_pk(
        data,
        template,
        theta,
        window_theory_kmin=cutoff_test,
        **common,
    )
    delta = pred_test - pred_ref
    precision, _ = make_precision(data.covariance, covariance_floor=0.0)
    sigma = np.sqrt(np.diag(data.covariance))

    # 中心差分只保留 fNL 的一阶响应；full-PNG 的 fNL^2 项会相消。
    theta_plus = dict(theta)
    theta_minus = dict(theta)
    theta_plus["fnl_loc"] = 1.0
    theta_minus["fnl_loc"] = -1.0
    derivatives: dict[str, np.ndarray] = {}
    for name, cutoff in (("reference", cutoff_ref), ("test", cutoff_test)):
        plus = model_pk(
            data,
            template,
            theta_plus,
            window_theory_kmin=cutoff,
            **common,
        )
        minus = model_pk(
            data,
            template,
            theta_minus,
            window_theory_kmin=cutoff,
            **common,
        )
        derivatives[name] = 0.5 * (plus - minus)
    derivative_extra = derivatives["test"] - derivatives["reference"]
    derivative_fraction = np.divide(
        derivative_extra,
        derivatives["test"],
        out=np.zeros_like(derivative_extra),
        where=np.abs(derivatives["test"]) > 0.0,
    )
    return {
        "fixed_parameters": theta,
        "k_obs": data.k_obs,
        "prediction_reference": pred_ref,
        "prediction_test": pred_test,
        "delta_prediction_test_minus_reference": delta,
        "delta_over_diagonal_sigma": delta / sigma,
        "max_abs_delta_over_diagonal_sigma": float(
            np.max(np.abs(delta / sigma))
        ),
        "delta_covariance_metric_chi2": float(delta @ precision @ delta),
        "dfnl_reference": derivatives["reference"],
        "dfnl_test": derivatives["test"],
        "dfnl_extra_test_minus_reference": derivative_extra,
        "dfnl_extra_fraction_of_test": derivative_fraction,
    }


def write_csv(
    path: Path,
    reference: dict[str, Any],
    test: dict[str, Any],
) -> None:
    """写出便于人工快速查看的 posterior CSV。"""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(
            stream,
            fieldnames=[
                "case",
                "kth_min",
                "fnl_q16",
                "fnl_q50",
                "fnl_q84",
                "b1_q16",
                "b1_q50",
                "b1_q84",
                "sn0_q16",
                "sn0_q50",
                "sn0_q84",
                "chi2_map",
            ],
        )
        writer.writeheader()
        for case, summary in (("reference_box", reference), ("test_k001", test)):
            writer.writerow(
                {
                    "case": case,
                    "kth_min": summary["config"]["window_theory_kmin"],
                    "fnl_q16": summary["parameters"]["fnl_loc"]["q16"],
                    "fnl_q50": summary["parameters"]["fnl_loc"]["median"],
                    "fnl_q84": summary["parameters"]["fnl_loc"]["q84"],
                    "b1_q16": summary["parameters"]["b1"]["q16"],
                    "b1_q50": summary["parameters"]["b1"]["median"],
                    "b1_q84": summary["parameters"]["b1"]["q84"],
                    "sn0_q16": summary["parameters"]["sn0"]["q16"],
                    "sn0_q50": summary["parameters"]["sn0"]["median"],
                    "sn0_q84": summary["parameters"]["sn0"]["q84"],
                    "chi2_map": summary["maximum_posterior_sample"]["chi2"],
                }
            )


def main() -> None:
    """执行完整 cutoff A/B 审计并写出 JSON/CSV。"""
    parser = argparse.ArgumentParser()
    parser.add_argument("--reference-dir", type=Path, default=DEFAULT_REFERENCE_DIR)
    parser.add_argument("--test-dir", type=Path, default=DEFAULT_TEST_DIR)
    parser.add_argument("--payload", type=Path, default=DEFAULT_PAYLOAD)
    parser.add_argument("--output-prefix", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()

    reference_dir = args.reference_dir.resolve()
    test_dir = args.test_dir.resolve()
    payload = args.payload.resolve()
    reference_json = summary_path(reference_dir)
    test_json = summary_path(test_dir)
    reference_npz = samples_path(reference_dir)
    test_npz = samples_path(test_dir)
    for path in (
        reference_json,
        test_json,
        reference_npz,
        test_npz,
        payload,
    ):
        if not path.exists():
            raise FileNotFoundError(path)

    reference = read_json(reference_json)
    test = read_json(test_json)
    matched = validate_matched(reference, test)
    cutoff_ref = float(matched["reference_cutoff"])
    cutoff_test = float(matched["test_cutoff"])
    comparison = posterior_comparison(reference, test)
    output = {
        "task": "task43_pk_kthmin_0p001_ab",
        "status": "done",
        "scope": (
            "Matched Task43 P(k) posterior A/B changing only the hard "
            "window-theory cutoff from the Abacus L=2000 fundamental "
            "to 0.001 h/Mpc."
        ),
        "inputs": {
            "reference_summary": str(reference_json),
            "reference_samples": str(reference_npz),
            "test_summary": str(test_json),
            "test_samples": str(test_npz),
            "payload": str(payload),
        },
        "matched_contract": matched,
        "posterior_comparison": comparison,
        "theory_grid_support": cutoff_support(
            payload,
            cutoff_ref,
            cutoff_test,
        ),
        "fixed_theta_window_response": fixed_theta_response(
            payload,
            reference,
            cutoff_ref,
            cutoff_test,
        ),
        "chain_diagnostics": {
            "reference": load_chain_diagnostics(reference_npz, reference),
            "test": load_chain_diagnostics(test_npz, test),
        },
        "interpretation_guardrails": [
            (
                "This A/B tests the current continuous theory-grid hard-cut "
                "implementation; it is not a FullDiscrete integer-lattice "
                "window convolution."
            ),
            (
                "A smaller cutoff is physically appropriate for real-data "
                "theory support only after the low-k window and integral-"
                "constraint response is converged and modeled."
            ),
            (
                "For finite-box mock closure, adding modes below the mother-"
                "box fundamental tests a deliberately mismatched infinite-"
                "volume-like model rather than recovering missing simulated "
                "modes."
            ),
        ],
    }
    prefix = args.output_prefix.resolve()
    json_path = prefix.with_suffix(".json")
    csv_path = prefix.with_suffix(".csv")
    json_path.parent.mkdir(parents=True, exist_ok=True)
    json_path.write_text(
        json.dumps(jsonable(output), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    write_csv(csv_path, reference, test)
    print(f"[write] {json_path}")
    print(f"[write] {csv_path}")
    fnl = comparison["fnl_loc"]
    print(
        "[result] "
        f"reference={fnl['reference']['q50']:.3f} "
        f"test={fnl['test']['q50']:.3f} "
        f"delta/sigma={fnl['delta_q50_over_reference_sigma68']:.3f} "
        f"sigma_ratio={fnl['sigma68_ratio_test_over_reference']:.3f}"
    )


if __name__ == "__main__":
    main()
