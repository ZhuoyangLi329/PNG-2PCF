#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""编译正八分体 factorized radial kernel 的 2PCF/P(k) 共用 operator。

代码大纲
========
1. 在当前 L=2000 FullDiscrete grid 上计算三个 PNG basis 的 xi(Delta)；内部
   Delta 使用 2 Mpc/h 解析 shell-average，避免高-k center alias。
2. 第一遍 radial-pair component 求和得到解析 p_model(s)，并以已经完成的
   pycorr exact RR marginal 做逐行 reweight。
3. 第二遍 component 求和同时得到：
   - RR-normalised 50--350 Mpc/h 2PCF correction basis；
   - 当前 15 bandpower x 981 theory-vector 的 P(k) RIC matrix。
4. P(k) matrix 用 direct component contraction 和先构造 C_rad(s) 再 Hankel
   两条独立 contraction 闭合；free sn0 与 clustering theory 使用同一 matrix。
5. 执行 global-limit、geometry reconstruction、fNL=0,+/-100、shot-noise
   audits，并写成与拟合入口兼容的 operator NPZ。
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
import re
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

for _name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_name, "1")

import numpy as np

PROJECT_ROOT = Path("/pscratch/sd/l/lzy/PNG-2PCF-model-to-observe")
CODE_DIR = PROJECT_ROOT / "codes" / "task43"
if str(CODE_DIR) not in sys.path:
    sys.path.insert(0, str(CODE_DIR))

from task43_ric_singleterm import (  # noqa: E402
    DEFAULT_PK,
    DEFAULT_XI,
    RIC_OPERATOR_DIR,
    atomic_savez,
    component_separation_probability,
    load_kernel,
    piecewise_constant_pk_xi_basis,
    shell_average_j0,
    to_jsonable,
    write_json,
)
from task43_theory_template import DEFAULT_COSMOLOGY  # noqa: E402


DEFAULT_W2 = (
    PROJECT_ROOT
    / "outputs/task43_outputs/summary"
    / "task43_formal_gic_window_mmin1p4e13_x25_ph000_fkpP010000_L2000_nsub200000_seed20260703.npz"
)


def load_minimal_closure_module() -> Any:
    """导入现有 Task4.3 FullDiscrete builder，确保不复制主理论实现。"""
    path = CODE_DIR / "task43_fit_minimal_closure.py"
    spec = importlib.util.spec_from_file_location("task43_fit_minimal_closure_for_factorized_ric", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"无法导入 {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def coefficients(fnl_loc: float, b1: float, *, p_fixed: float = 1.0) -> tuple[float, float, float]:
    """返回 full-PNG 的 Pdd、alpha Pdd、alpha^2 Pdd 三个系数。"""
    bphi = 2.0 * 1.686 * (float(b1) - float(p_fixed))
    fb = float(fnl_loc) * bphi
    return float(b1) ** 2, 2.0 * float(b1) * fb, fb * fb


def build_basis(theory: dict[str, Any]) -> dict[str, np.ndarray]:
    """在当前 k_dense 上建立与 2PCF/P(k) 拟合一致的三个 PNG basis。"""
    task41 = theory["task41"]
    pk_dd = task41.interp_logk(theory["k_dense"], theory["template"]["k"], theory["template"]["pk_dd"])
    alpha = task41.interp_logk(theory["k_dense"], theory["template"]["k"], theory["template"]["alpha"])
    return {
        "pk_dd": np.asarray(pk_dd, dtype="f8"),
        "alpha_pk_dd": np.asarray(alpha * pk_dd, dtype="f8"),
        "alpha2_pk_dd": np.asarray(alpha * alpha * pk_dd, dtype="f8"),
    }


def combine_basis(projected: dict[str, np.ndarray], *, fnl_loc: float, b1: float) -> np.ndarray:
    """用 full-PNG 系数组合三个已投影 correction basis。"""
    c0, c1, c2 = coefficients(fnl_loc, b1)
    return c0 * projected["pk_dd"] + c1 * projected["alpha_pk_dd"] + c2 * projected["alpha2_pk_dd"]


def robust_relative_rms(test: np.ndarray, reference: np.ndarray, mask: np.ndarray) -> float:
    """用 reference RMS 归一化矩阵误差，避免逐元素小分母。"""
    difference = np.asarray(test)[mask] - np.asarray(reference)[mask]
    denominator = float(np.sqrt(np.mean(np.asarray(reference)[mask] ** 2)))
    return float(np.sqrt(np.mean(difference**2)) / denominator) if denominator > 0.0 else float("nan")


def parse_args() -> argparse.Namespace:
    """定义 factorized operator 编译参数。"""
    parser = argparse.ArgumentParser(description="Compile factorized Task43 radial RIC operator.")
    parser.add_argument("--kernel", type=Path, required=True)
    parser.add_argument("--xi-path", type=Path, default=DEFAULT_XI)
    parser.add_argument("--pk-payload", type=Path, default=DEFAULT_PK)
    parser.add_argument("--w2-path", type=Path, default=DEFAULT_W2)
    parser.add_argument("--boxsize", type=float, default=2000.0)
    parser.add_argument("--kmax", type=float, default=5.0)
    parser.add_argument("--ndense", type=int, default=60000)
    parser.add_argument("--cosmology", type=str, default=DEFAULT_COSMOLOGY)
    parser.add_argument("--component-batch", type=int, default=256)
    parser.add_argument("--k-batch", type=int, default=512)
    parser.add_argument("--output-dir", type=Path, default=RIC_OPERATOR_DIR)
    parser.add_argument(
        "--output-tag",
        type=str,
        default=None,
        help="Optional filesystem-safe science tag appended to the legacy output stem.",
    )
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def sanitize_output_tag(value: str | None) -> str | None:
    """严格验证可选输出标签，避免路径穿越、隐藏文件和静默命名碰撞。"""
    if value is None:
        return None
    tag = str(value).strip()
    if not tag:
        raise ValueError("--output-tag 不能为空或只包含空白")
    if len(tag) > 96:
        raise ValueError("--output-tag 最长为 96 个字符")
    if re.fullmatch(r"[A-Za-z0-9](?:[A-Za-z0-9._-]*[A-Za-z0-9])?", tag) is None:
        raise ValueError(
            "--output-tag 只能包含 ASCII 字母、数字、点、下划线和连字符，且必须以字母或数字开头、结尾"
        )
    return tag


def output_stem(kernel: Path, *, output_tag: str | None = None) -> str:
    """operator 文件名继承 kernel 科学标签，并可追加显式 target 标签。"""
    stem = Path(kernel).stem
    if stem.startswith("task43_ric_factorized_"):
        stem = stem[len("task43_ric_factorized_") :]
    output = f"task43_ric_factorized_operator_{stem}_L2000"
    tag = sanitize_output_tag(output_tag)
    return output if tag is None else f"{output}_{tag}"


def full_discrete_xi_basis_on_shells(
    *,
    theory: dict[str, Any],
    dense_basis: dict[str, np.ndarray],
    separation_edges: np.ndarray,
    k_batch: int,
) -> dict[str, np.ndarray]:
    """把三个 dense P(k) basis 投影到完整 lightcone Delta shells。

    只保存最终三个 xi(Delta) 数组，避免生成 15882x1700x3 的永久大矩阵。
    """
    names = ("pk_dd", "alpha_pk_dd", "alpha2_pk_dd")
    task41 = theory["task41"]
    p_eff = np.vstack(
        [task41.interp_logk(theory["k_eff"], theory["k_dense"], dense_basis[name]) for name in names]
    )
    weights = p_eff * np.asarray(theory["g_nz"], dtype="f8")[None, :]
    result = np.zeros((len(names), np.asarray(separation_edges).size - 1), dtype="f8")
    for start in range(0, np.asarray(theory["k_eff"]).size, int(k_batch)):
        stop = min(np.asarray(theory["k_eff"]).size, start + int(k_batch))
        j0 = shell_average_j0(
            np.asarray(theory["k_eff"])[start:stop],
            np.asarray(separation_edges[:-1], dtype="f8"),
            np.asarray(separation_edges[1:], dtype="f8"),
        )
        result += weights[:, start:stop] @ j0
    result /= float(theory["volume"])
    return {name: result[index] for index, name in enumerate(names)}


def target_projection_weights(kernel_edges: np.ndarray, target_edges: np.ndarray, row_ratio: np.ndarray) -> np.ndarray:
    """构造 component H(s) 到实际 2PCF bins 的 outer reweight 矩阵。"""
    centers = 0.5 * (np.asarray(kernel_edges[:-1]) + np.asarray(kernel_edges[1:]))
    target = np.zeros((centers.size, np.asarray(target_edges).size - 1), dtype="f8")
    for ibin, (lo, hi) in enumerate(zip(target_edges[:-1], target_edges[1:], strict=True)):
        mask = (centers >= lo) & (centers < hi)
        target[mask, ibin] = np.asarray(row_ratio)[mask]
    return target


def main() -> None:
    """执行两遍 component contraction、数值审计和 operator 落盘。"""
    args = parse_args()
    output_tag = sanitize_output_tag(args.output_tag)
    stem = output_stem(Path(args.kernel), output_tag=output_tag)
    npz_path = Path(args.output_dir) / f"{stem}.npz"
    json_path = Path(args.output_dir) / f"{stem}_audit.json"
    if not args.overwrite:
        existing = [path for path in (npz_path, json_path) if path.exists()]
        if len(existing) == 2:
            print(f"[skip] complete operator pair already exists: {npz_path}")
            return
        if existing:
            missing = json_path if existing[0] == npz_path else npz_path
            raise FileExistsError(
                "refusing to overwrite an incomplete operator output pair without --overwrite: "
                f"existing={existing[0]}, missing={missing}"
            )

    cache = load_kernel(Path(args.kernel))
    if cache["meta"].get("kernel_kind") != "factorized_positive_octant_radial_auto":
        raise ValueError("该入口只接受 task43_ric_factorized kernel")
    sedges = np.asarray(cache["separation_edges"], dtype="f8")
    outer_exact = np.asarray(cache["outer_counts"], dtype="f8")
    outer_exact /= np.sum(outer_exact)
    component_weight = np.asarray(cache["component_weight"], dtype="f8")
    component_r1 = np.asarray(cache["component_r1"], dtype="f8")
    component_r2 = np.asarray(cache["component_r2"], dtype="f8")
    ncomponent = component_weight.size

    with np.load(args.xi_path, allow_pickle=False) as xi_data:
        target_edges = np.asarray(xi_data["s_edges"], dtype="f8")
        zeff = float(np.asarray(xi_data["zeff"]).item())
    if (
        target_edges.ndim != 1
        or target_edges.size < 2
        or not np.all(np.isfinite(target_edges))
        or np.any(np.diff(target_edges) <= 0.0)
    ):
        raise ValueError(f"invalid target s_edges in {args.xi_path}: shape={target_edges.shape}")
    closure = load_minimal_closure_module()
    theory = closure.build_theory_context(
        zeff,
        kmax=float(args.kmax),
        ndense=int(args.ndense),
        boxsize=float(args.boxsize),
        cosmology=str(args.cosmology),
    )
    dense_basis = build_basis(theory)
    xi_delta = full_discrete_xi_basis_on_shells(
        theory=theory,
        dense_basis=dense_basis,
        separation_edges=sedges,
        k_batch=int(args.k_batch),
    )
    xi_delta_matrix = np.vstack([xi_delta["pk_dd"], xi_delta["alpha_pk_dd"], xi_delta["alpha2_pk_dd"]]).T

    # 第一遍只建立解析 outer marginal；随后 exact pycorr RR 提供 row ratio。
    outer_model = np.zeros_like(outer_exact)
    for start in range(0, ncomponent, int(args.component_batch)):
        stop = min(ncomponent, start + int(args.component_batch))
        h = component_separation_probability(
            component_r1[start:stop],
            component_r2[start:stop],
            separation_edges=sedges,
            cosine_edges=cache["cosine_edges"],
            cosine_cdf=cache["cosine_cdf"],
        )
        outer_model += component_weight[start:stop] @ h
    outer_model /= np.sum(outer_model)
    row_ratio = np.zeros_like(outer_exact)
    valid = outer_model > 0.0
    row_ratio[valid] = outer_exact[valid] / outer_model[valid]
    if np.any((outer_exact > 0.0) & ~valid):
        raise RuntimeError("解析 outer model 在 exact RR 非零处出现零 probability")

    with np.load(args.pk_payload, allow_pickle=False) as pk_data:
        observed_k_edges = np.asarray(pk_data["k_edges"], dtype="f8")
        theory_k = np.asarray(pk_data["theory_k"], dtype="f8")
        theory_ell = np.asarray(pk_data["theory_ell"], dtype="i8")
        theory_edges = np.asarray(pk_data["theory_edges"], dtype="f8")
        geometry = np.asarray(pk_data["window_matrix"], dtype="f8")
    ell0 = theory_ell == 0
    centers = 0.5 * (sedges[:-1] + sedges[1:])
    j0_observed = shell_average_j0(centers, observed_k_edges[:, 0], observed_k_edges[:, 1]).T
    xi_band = piecewise_constant_pk_xi_basis(centers, theory_edges[ell0])
    target_weight = target_projection_weights(sedges, target_edges, row_ratio)
    target_norm = target_weight.T @ outer_model

    xi_numerator = np.zeros((target_edges.size - 1, 3), dtype="f8")
    pk_direct_ell0 = np.zeros((observed_k_edges.shape[0], int(np.sum(ell0))), dtype="f8")
    raw_c_s_q = np.zeros((centers.size, int(np.sum(ell0))), dtype="f8")
    j0_with_ratio = j0_observed * row_ratio[None, :]
    print(f"[components] n={ncomponent} batch={args.component_batch}", flush=True)
    for start in range(0, ncomponent, int(args.component_batch)):
        stop = min(ncomponent, start + int(args.component_batch))
        h = component_separation_probability(
            component_r1[start:stop],
            component_r2[start:stop],
            separation_edges=sedges,
            cosine_edges=cache["cosine_edges"],
            cosine_cdf=cache["cosine_cdf"],
        )
        weight = component_weight[start:stop]
        inner_xi_basis = h @ xi_delta_matrix
        outer_target = h @ target_weight
        xi_numerator += outer_target.T @ (weight[:, None] * inner_xi_basis)

        inner_pk_band = h @ xi_band
        weighted_inner_pk = weight[:, None] * inner_pk_band
        outer_pk = h @ j0_with_ratio.T
        pk_direct_ell0 += outer_pk.T @ weighted_inner_pk
        # 独立路径先构造 raw C_rad(s,q)，循环结束后再做 outer Hankel。
        raw_c_s_q += (h.T * row_ratio[:, None]) @ weighted_inner_pk

    xi_basis_matrix = xi_numerator / target_norm[:, None]
    xi_basis = {
        "pk_dd": xi_basis_matrix[:, 0],
        "alpha_pk_dd": xi_basis_matrix[:, 1],
        "alpha2_pk_dd": xi_basis_matrix[:, 2],
    }
    pair_volume = float(cache["meta"]["fkp_fourier_normalisation"]["pair_volume"])
    pk_direct_ell0 *= pair_volume
    pk_hankel_ell0 = pair_volume * (j0_observed @ raw_c_s_q)
    pk_ric_raw = np.zeros_like(geometry)
    pk_hankel_raw = np.zeros_like(geometry)
    pk_ric_raw[:, ell0] = pk_direct_ell0
    pk_hankel_raw[:, ell0] = pk_hankel_ell0

    geometry_factorized = np.zeros_like(geometry)
    geometry_factorized[:, ell0] = pair_volume * ((j0_observed * outer_exact[None, :]) @ xi_band)
    # 当前实际 P(k) estimator 含 FFT mesh 与离散 observed-k mode average，而
    # continuum pair integral 使用解析 k-shell average。用固定 covariance
    # fiducial (fNL=0,b1=2.5,sn0=0) 的 geometry-only response 推导每个
    # observed bin 的参数无关 transfer；随后同一 transfer 同时作用于 direct
    # 与 Hankel RIC matrix。它不是拟合数据的经验因子，只把 radial response
    # 放到现有 jaxpower window 的 estimator normalization 上。
    fiducial_vector = np.zeros_like(theory_k)
    fiducial_vector[ell0] = theory["task41"].interp_logk(
        theory_k[ell0],
        theory["k_dense"],
        2.5**2 * dense_basis["pk_dd"],
    )
    fiducial_vector[theory_k < 2.0 * np.pi / float(args.boxsize)] = 0.0
    geometry_fiducial = geometry @ fiducial_vector
    factorized_fiducial = geometry_factorized @ fiducial_vector
    estimator_transfer = np.ones_like(geometry_fiducial)
    nonzero_fiducial = factorized_fiducial != 0.0
    estimator_transfer[nonzero_fiducial] = geometry_fiducial[nonzero_fiducial] / factorized_fiducial[nonzero_fiducial]
    pk_ric = estimator_transfer[:, None] * pk_ric_raw
    pk_hankel = estimator_transfer[:, None] * pk_hankel_raw
    common = (theory_k >= 2.0 * np.pi / float(args.boxsize)) & (theory_k <= 0.12) & ell0
    matrix_mask = np.broadcast_to(common[None, :], geometry.shape)
    row_corr = [
        float(np.corrcoef(geometry_factorized[irow, common], geometry[irow, common])[0, 1])
        for irow in range(geometry.shape[0])
    ]
    geometry_audit = {
        "relative_rms": robust_relative_rms(geometry_factorized, geometry, matrix_mask),
        "row_correlation_min": float(np.min(row_corr)),
        "row_correlation_median": float(np.median(row_corr)),
        "row_correlations": row_corr,
        "domain": f"ell0, 2pi/{float(args.boxsize):g} <= kth <= 0.12 h/Mpc",
        "fiducial_relative_rms_before_transfer": float(
            np.linalg.norm(factorized_fiducial - geometry_fiducial) / np.linalg.norm(geometry_fiducial)
        ),
        "fiducial_relative_rms_after_transfer": float(
            np.linalg.norm(estimator_transfer * factorized_fiducial - geometry_fiducial)
            / np.linalg.norm(geometry_fiducial)
        ),
        "estimator_transfer": [float(v) for v in estimator_transfer],
        "transfer_policy": "fixed fNL=0,b1=2.5,sn0=0,mother-box-cut geometry response; no data fit",
    }
    hankel_diff = pk_ric[:, ell0] - pk_hankel[:, ell0]
    hankel_audit = {
        "max_abs": float(np.max(np.abs(hankel_diff))),
        "relative_frobenius": float(np.linalg.norm(hankel_diff) / max(np.linalg.norm(pk_ric[:, ell0]), 1e-300)),
        "paths": "direct component contraction versus raw C_rad(s,q) then outer Hankel",
    }

    # Global bin 合并后 inner separation distribution 与 exact outer RR 相同。
    global_basis_scalar = outer_exact @ xi_delta_matrix
    global_basis = {
        "pk_dd": np.full(target_edges.size - 1, global_basis_scalar[0]),
        "alpha_pk_dd": np.full(target_edges.size - 1, global_basis_scalar[1]),
        "alpha2_pk_dd": np.full(target_edges.size - 1, global_basis_scalar[2]),
    }
    global_fid = combine_basis(global_basis, fnl_loc=0.0, b1=2.5)
    w2_comparison: dict[str, Any] = {"available": False}
    if Path(args.w2_path).exists():
        with np.load(args.w2_path, allow_pickle=False) as w2data:
            wk = np.asarray(w2data["k_eff"], dtype="f8")
            w2 = np.asarray(w2data["w2"], dtype="f8")
        if wk.shape == np.asarray(theory["k_eff"]).shape and np.allclose(wk, theory["k_eff"], rtol=0.0, atol=1e-14):
            p_eff = theory["task41"].interp_logk(theory["k_eff"], theory["k_dense"], 2.5**2 * dense_basis["pk_dd"])
            old_sigma = float(np.sum(theory["g_nz"] * p_eff * w2) / float(theory["volume"]))
            new_sigma = float(np.mean(global_fid))
            w2_comparison = {
                "available": True,
                "path": str(args.w2_path),
                "old_sigma_w2": old_sigma,
                "merged_radial_sigma": new_sigma,
                "relative_difference": (new_sigma - old_sigma) / old_sigma,
            }

    parameter_rows = []
    task41 = theory["task41"]
    for fnl in (-100.0, 0.0, 100.0):
        xi_corr = combine_basis(xi_basis, fnl_loc=fnl, b1=2.5)
        c0, c1, c2 = coefficients(fnl, 2.5)
        dense = c0 * dense_basis["pk_dd"] + c1 * dense_basis["alpha_pk_dd"] + c2 * dense_basis["alpha2_pk_dd"]
        vector = np.zeros_like(theory_k)
        vector[ell0] = task41.interp_logk(theory_k[ell0], theory["k_dense"], dense)
        pcorr = pk_ric @ vector
        parameter_rows.append(
            {
                "fnl_loc": fnl,
                "xi_min": float(np.min(xi_corr)),
                "xi_max": float(np.max(xi_corr)),
                "pk_min": float(np.min(pcorr)),
                "pk_max": float(np.max(pcorr)),
                "finite": bool(np.all(np.isfinite(xi_corr)) and np.all(np.isfinite(pcorr))),
            }
        )
    sn0 = np.zeros_like(theory_k)
    sn0[ell0] = 1.0e4
    sn0_projection = pk_ric @ sn0

    nonzero_outer = outer_exact > 0.0
    outer_reconstruction = {
        "relative_rms_before_reweight": float(
            np.sqrt(np.mean((outer_model[nonzero_outer] - outer_exact[nonzero_outer]) ** 2))
            / np.sqrt(np.mean(outer_exact[nonzero_outer] ** 2))
        ),
        "max_row_ratio": float(np.max(row_ratio[nonzero_outer])),
        "min_row_ratio": float(np.min(row_ratio[nonzero_outer])),
        "exact_probability_sum": float(np.sum(outer_exact)),
        "model_probability_sum": float(np.sum(outer_model)),
    }
    audit = {
        "task": "task43_make_ric_factorized_operator",
        "status": "done",
        "created_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "scope": {
            "approximation": "single IC^(rad,rad) auto term",
            "fit_formula": "model = noIC - radial_auto_response",
            "density_ric_cross_terms": False,
            "extra_global_sigma_w2": False,
            "pk_standalone_gic": False,
            "shared_kernel": "same radial components, H_ab(s), exact RR reweight for 2PCF and P(k)",
        },
        "output_tag": output_tag,
        "target_2pcf": {
            "s_edge_min": float(target_edges[0]),
            "s_edge_max": float(target_edges[-1]),
            "nbins": int(target_edges.size - 1),
        },
        "inputs": {"kernel": str(args.kernel), "xi": str(args.xi_path), "pk_payload": str(args.pk_payload)},
        "theory": {
            "boxsize": float(theory["boxsize"]),
            "kfund": float(theory["kfund"]),
            "kmax": float(args.kmax),
            "ndense": int(args.ndense),
            "cosmology": str(args.cosmology),
            "zeff": zeff,
            "p_fixed": 1.0,
            "png_order": "full including fNL^2",
        },
        "operator_shapes": {"xi_basis": [int(target_edges.size - 1), 3], "pk_ric_matrix": list(pk_ric.shape)},
        "tests": {
            "outer_geometry_reconstruction": outer_reconstruction,
            "global_limit": {
                "max_s_dependence": float(np.max(np.abs(global_fid - np.mean(global_fid)))),
                "constant_value_fnl0_b1_2p5": float(np.mean(global_fid)),
                "pass_exact_constant": bool(np.max(np.abs(global_fid - np.mean(global_fid))) < 1e-12),
                "comparison_to_current_w2": w2_comparison,
            },
            "hankel_consistency": hankel_audit,
            "geometry_window_reconstruction": geometry_audit,
            "parameter_dependence": parameter_rows,
            "shot_noise_convention": {
                "measurement": "shot-noise-subtracted P0",
                "model": "same W_RIC acts on free residual sn0*1e4",
                "sn0_1_projection": [float(v) for v in sn0_projection],
                "finite": bool(np.all(np.isfinite(sn0_projection))),
            },
        },
        "paths": {"operator_npz": str(npz_path), "audit_json": str(json_path)},
    }
    atomic_savez(
        npz_path,
        kernel_path=np.asarray(str(args.kernel)),
        target_s_edges=target_edges,
        k_eff=np.asarray(theory["k_eff"], dtype="f8"),
        g_nz=np.asarray(theory["g_nz"], dtype="f8"),
        volume=np.asarray(float(theory["volume"]), dtype="f8"),
        xi_basis_pk_dd=np.asarray(xi_basis["pk_dd"], dtype="f8"),
        xi_basis_alpha_pk_dd=np.asarray(xi_basis["alpha_pk_dd"], dtype="f8"),
        xi_basis_alpha2_pk_dd=np.asarray(xi_basis["alpha2_pk_dd"], dtype="f8"),
        xi_global_basis_pk_dd=np.asarray(global_basis["pk_dd"], dtype="f8"),
        xi_global_basis_alpha_pk_dd=np.asarray(global_basis["alpha_pk_dd"], dtype="f8"),
        xi_global_basis_alpha2_pk_dd=np.asarray(global_basis["alpha2_pk_dd"], dtype="f8"),
        pk_theory_k=theory_k,
        pk_theory_ell=theory_ell,
        pk_ric_matrix=np.asarray(pk_ric, dtype="f8"),
        pk_ric_matrix_hankel=np.asarray(pk_hankel, dtype="f8"),
        pk_ric_matrix_continuum_raw=np.asarray(pk_ric_raw, dtype="f8"),
        pk_ric_matrix_hankel_continuum_raw=np.asarray(pk_hankel_raw, dtype="f8"),
        pk_geometry_factorized_matrix=np.asarray(geometry_factorized, dtype="f8"),
        pk_estimator_transfer=np.asarray(estimator_transfer, dtype="f8"),
        outer_exact_probability=np.asarray(outer_exact, dtype="f8"),
        outer_model_probability=np.asarray(outer_model, dtype="f8"),
        outer_row_ratio=np.asarray(row_ratio, dtype="f8"),
        meta_json=np.asarray(json.dumps(to_jsonable(audit), sort_keys=True)),
    )
    write_json(json_path, audit)
    print(f"[write] {npz_path}")
    print(f"[write] {json_path}")
    print(
        f"[audit] global={audit['tests']['global_limit']['constant_value_fnl0_b1_2p5']:.6e} "
        f"hankel_rel={hankel_audit['relative_frobenius']:.3e} geometry_rel={geometry_audit['relative_rms']:.3e}"
    )


if __name__ == "__main__":
    main()
