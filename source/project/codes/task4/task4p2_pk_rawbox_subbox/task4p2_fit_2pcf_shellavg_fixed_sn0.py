#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Task4.2：用 volume shell-average 重新拟合 rawbox/subbox 2PCF。

执行大纲：
1. 读取归档但仍是正式输入的 Task47 rawbox、L1500、L1000 2PCF mean/cov，
   从严格的 ``s=55,65,...,345`` centers 恢复 ``50,60,...,350`` shell edges。
2. 复用 Task4 的 z=1、Lbox=3000、kmax_discrete=15 FullDiscrete 与 analytic
   cubic-window 缓存；只构建一次三个 PNG P(k) basis。
3. 用每个 shell 内的体积平均 j0，而不是 bin-center j0，将三个 basis 投影到
   2PCF；formal-GIC case 再减去 parameter-dependent sigma_W^2。
4. 按用户确认的物理口径固定 2PCF sn0=0、sigmas=0、p=1.2，只采样
   ``fnl_loc,b1``；五个 case 均运行 convergence-gated emcee 长链。
5. 写每个 case 的 post-burn samples/summary、统一 summary/audit 和小型 basis
   cache；本脚本不画最终 forest 图，绘图由独立 replot 入口完成。

注意：归档目录仅作为不可变数据来源，本脚本不会修改、移动或删除其中任何文件。
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

for _name in (
    "OMP_NUM_THREADS",
    "MKL_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
):
    os.environ.setdefault(_name, "1")

import emcee
import numpy as np
from scipy.optimize import minimize

from task4p2_pk_common import FASTPM_SNAPSHOT_Z, atomic_savez, covariance_corrections, to_jsonable, write_json


PROJECT_ROOT = Path("/pscratch/sd/l/lzy/PNG-2PCF-model-to-observe")
ARCHIVE_ROOT = PROJECT_ROOT / "old_doc_codes/task4_task44_cleanup_20260707T061844Z/moved"
ARCHIVED_TASK44 = ARCHIVE_ROOT / "codes/task4/task44_fastpm_fnl100_subbox_gic_profiler.py"
RAWBOX_DATA = ARCHIVE_ROOT / "outputs/task47_outputs/fnl100_norsd_r50/rawbox/task47_fnl100_rawbox_mean_xi_s50_350.npz"
SUBBOX_DATA = (
    ARCHIVE_ROOT
    / "outputs/task47_outputs/fnl100_norsd_r50/subboxes/L{lsub}/summary/mean_xi_L{lsub}_r20_individual.npz"
)
SUBBOX_INDIVIDUAL_DIR = (
    ARCHIVE_ROOT / "outputs/task47_outputs/fnl100_norsd_r50/subboxes/L{lsub}/xi_r20_individual"
)
OUTPUT_ROOT = PROJECT_ROOT / "outputs/task4_outputs/task4p2_2pcf_shellavg_fixed_sn0_z1"
COMBINED_SUMMARY = OUTPUT_ROOT / "task4p2_2pcf_shellavg_fixed_sn0_z1_summary.json"
THEORY_CACHE = OUTPUT_ROOT / "task4p2_2pcf_shellavg_basis_cache.npz"

PARAM_NAMES = ("fnl_loc", "b1")
P_FIXED = 1.2
SN0_FIXED = 0.0
SIGMAS_FIXED = 0.0
DELTA_C = 1.686
BOX_SIZE = 3000.0
VOLUME = BOX_SIZE**3
KMAX_DISCRETE = 15.0
EXPECTED_S = np.arange(55.0, 350.0, 10.0, dtype="f8")
EXPECTED_EDGES = np.arange(50.0, 350.0 + 10.0, 10.0, dtype="f8")

PRIORS = {
    "fnl_loc": (-500.0, 500.0),
    "b1": (0.5, 5.0),
}


@dataclass
class FitCase:
    """一个 2PCF likelihood 的数据、precision 与 GIC 标签。"""

    label: str
    sample_label: str
    model_label: str
    lsub: int | None
    source_path: Path
    s: np.ndarray
    s_edges: np.ndarray
    data: np.ndarray
    covariance: np.ndarray
    precision: np.ndarray
    covariance_meta: dict[str, Any]
    cluster_meta: dict[str, Any]


@dataclass
class ShellBasis:
    """共享 shell-averaged xi basis 与单个 case 的 GIC sigma basis。"""

    xi_pkdd: np.ndarray
    xi_gamma_pkdd: np.ndarray
    xi_gamma2_pkdd: np.ndarray
    sigma_pkdd: float = 0.0
    sigma_gamma_pkdd: float = 0.0
    sigma_gamma2_pkdd: float = 0.0


def file_sha256(path: Path, chunk_bytes: int = 1024 * 1024) -> str:
    """分块计算文件 SHA256，用于把 summary 绑定到不可变输入。"""
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        while block := stream.read(int(chunk_bytes)):
            digest.update(block)
    return digest.hexdigest()


def import_module(path: Path, name: str):
    """从显式路径导入归档理论模块，不改变归档文件。"""
    spec = importlib.util.spec_from_file_location(name, Path(path))
    if spec is None or spec.loader is None:
        raise RuntimeError(f"无法导入模块: {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def recover_shell_edges(s: np.ndarray) -> np.ndarray:
    """验证 Task47 centers，并无歧义恢复每个 10 Mpc/h shell 的上下边界。"""
    s = np.asarray(s, dtype="f8")
    if s.shape != EXPECTED_S.shape or not np.allclose(s, EXPECTED_S, rtol=0.0, atol=1.0e-12):
        raise ValueError(f"Task4.2 shell centers 不是预期的 55..345: {s}")
    return np.column_stack([EXPECTED_EDGES[:-1], EXPECTED_EDGES[1:]])


def build_precision(covariance: np.ndarray, *, nmock: int) -> tuple[np.ndarray, dict[str, Any]]:
    """对 single-realization sample covariance 施加 Hartlap，并记录 Percival 因子。"""
    covariance = np.asarray(covariance, dtype="f8")
    corrections = covariance_corrections(nmock=int(nmock), ndata=int(covariance.shape[0]), nparams=len(PARAM_NAMES))
    precision = float(corrections["hartlap"]) * np.linalg.pinv(covariance, rcond=1.0e-10)
    eigenvalues = np.linalg.eigvalsh(0.5 * (covariance + covariance.T))
    if float(eigenvalues[0]) <= 0.0:
        raise ValueError(f"covariance 非正定: min_eigenvalue={eigenvalues[0]}")
    return precision, {
        **corrections,
        "nmock": int(nmock),
        "ndata": int(covariance.shape[0]),
        "nparams": len(PARAM_NAMES),
        "condition_number": float(np.linalg.cond(covariance)),
        "min_eigenvalue": float(eigenvalues[0]),
        "used_covariance_divided_by_nmock": False,
    }


def cluster_covariance_audit(stack: np.ndarray) -> dict[str, Any]:
    """
    对 balanced ``parent x subbox x bin`` stack 做母盒聚类审计。

    ``C_auto`` 是固定 subbox 位置跨独立母盒的 covariance 再对位置平均；
    ``C_cross`` 是同一母盒不同 subbox 的平均 cross-covariance。cluster
    leave-one-parent-out jackknife 在 pooled covariance 的 whitened basis 中估计
    Wishart-like 有效自由度，用来检查是否可把 Nsub 用于 Hartlap/Percival。
    """
    values = np.asarray(stack, dtype="f8")
    if values.ndim != 3:
        raise ValueError(f"cluster stack 必须是 R x m x p，实际 shape={values.shape}")
    nparent, nsub_per_parent, nbin = values.shape
    flattened = values.reshape((nparent * nsub_per_parent, nbin))
    covariance_pooled = np.cov(flattened, rowvar=False, ddof=1)

    slot_mean = np.mean(values, axis=0)
    centered = values - slot_mean[None, :, :]
    covariance_auto = sum(
        centered[:, isub, :].T @ centered[:, isub, :] / float(nparent - 1)
        for isub in range(nsub_per_parent)
    ) / float(nsub_per_parent)
    covariance_cross = np.zeros_like(covariance_auto)
    pair_count = 0
    for first in range(nsub_per_parent):
        for second in range(nsub_per_parent):
            if first == second:
                continue
            covariance_cross += centered[:, first, :].T @ centered[:, second, :] / float(nparent - 1)
            pair_count += 1
    covariance_cross = 0.5 * (covariance_cross / float(pair_count) + (covariance_cross / float(pair_count)).T)
    rho = np.diag(covariance_cross) / np.diag(covariance_auto)
    design_effect = 1.0 + float(nsub_per_parent - 1) * rho
    pooled_over_auto = np.diag(covariance_pooled) / np.diag(covariance_auto)

    # 在 pooled covariance 的 whitened basis 中，对 leave-one-parent-out
    # covariance 做 cluster jackknife；独立 Gaussian subboxes 时应回收
    # nu_eff ~ Nparent*Nsub-1。
    eigenvalues, eigenvectors = np.linalg.eigh(covariance_pooled)
    whitening = (eigenvectors / np.sqrt(eigenvalues)) @ eigenvectors.T
    leave_one_out = []
    for parent in range(nparent):
        reduced = np.delete(values, parent, axis=0).reshape(((nparent - 1) * nsub_per_parent, nbin))
        leave_one_out.append(np.cov(reduced, rowvar=False, ddof=1))
    leave_one_out = np.asarray(leave_one_out, dtype="f8")
    loo_mean = np.mean(leave_one_out, axis=0)
    whitened_delta = np.einsum("ij,rjk,kl->ril", whitening, leave_one_out - loo_mean, whitening)
    jackknife_variance = float(nparent - 1) / float(nparent) * np.sum(whitened_delta**2, axis=0)
    diag_variance = np.diag(jackknife_variance)
    offdiag_variance = jackknife_variance[~np.eye(nbin, dtype=bool)]
    nu_diag = 2.0 / diag_variance
    nu_offdiag = 1.0 / offdiag_variance

    def positive_quantiles(array: np.ndarray) -> list[float]:
        selected = np.asarray(array, dtype="f8")
        selected = selected[np.isfinite(selected) & (selected > 0.0)]
        return [float(value) for value in np.quantile(selected, [0.16, 0.5, 0.84])]

    nu_diag_q = positive_quantiles(nu_diag)
    nu_offdiag_q = positive_quantiles(nu_offdiag)
    nsub_total = int(nparent * nsub_per_parent)
    nu_calibrated = min(float(nsub_total - 1), 0.5 * (nu_diag_q[1] + nu_offdiag_q[1]))
    nmock_calibrated = int(round(nu_calibrated + 1.0))
    return {
        "policy": "balanced complete parent clusters",
        "nparent_independent": int(nparent),
        "nsubbox_per_parent": int(nsub_per_parent),
        "nsubbox_total": nsub_total,
        "same_parent_rho_per_bin": rho,
        "same_parent_rho_min": float(np.min(rho)),
        "same_parent_rho_median": float(np.median(rho)),
        "same_parent_rho_max": float(np.max(rho)),
        "design_effect_per_bin": design_effect,
        "design_effect_median": float(np.median(design_effect)),
        "design_effect_max": float(np.max(design_effect)),
        "pooled_over_independent_slot_covariance_diag": pooled_over_auto,
        "pooled_over_independent_slot_diag_min": float(np.min(pooled_over_auto)),
        "pooled_over_independent_slot_diag_median": float(np.median(pooled_over_auto)),
        "pooled_over_independent_slot_diag_max": float(np.max(pooled_over_auto)),
        "pooled_vs_independent_slot_relative_frobenius": float(
            np.linalg.norm(covariance_pooled - covariance_auto) / np.linalg.norm(covariance_auto)
        ),
        "cross_vs_auto_relative_frobenius": float(np.linalg.norm(covariance_cross) / np.linalg.norm(covariance_auto)),
        "cluster_jackknife_effective_nu_diag_q16_q50_q84": nu_diag_q,
        "cluster_jackknife_effective_nu_offdiag_q16_q50_q84": nu_offdiag_q,
        "calibrated_wishart_nu": float(nu_calibrated),
        "calibrated_nmock_for_hartlap_percival": int(nmock_calibrated),
        "calibration_cap": "nu_eff is capped at Nsub_total-1; no independence gain beyond the observed subboxes",
    }


def load_balanced_subbox_stack(lsub: int) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, Any]]:
    """读取单 subbox xi 文件，只保留拥有完整 8/27 个 subbox 的母盒。"""
    directory = Path(str(SUBBOX_INDIVIDUAL_DIR).format(lsub=int(lsub)))
    expected_subboxes = int(round((BOX_SIZE / float(lsub)) ** 3))
    records: dict[int, dict[int, np.ndarray]] = {}
    s_reference: np.ndarray | None = None
    valid_files = 0
    for path in sorted(directory.glob("*.npz")):
        with np.load(path, allow_pickle=False) as payload:
            status = str(np.asarray(payload["status"]).item()) if "status" in payload.files else "done"
            random_mode = str(np.asarray(payload["random_mode"]).item()) if "random_mode" in payload.files else ""
            if status != "done" or random_mode != "per_subbox_dynamic":
                continue
            s = np.asarray(payload["s"], dtype="f8")
            xi = np.asarray(payload["xi0"], dtype="f8")
            realization = int(np.asarray(payload["realization"]).item())
            subbox_id = int(np.asarray(payload["subbox_id"]).item())
        if s_reference is None:
            s_reference = s
        elif not np.array_equal(s, s_reference):
            raise ValueError(f"subbox s grid 不一致: {path}")
        records.setdefault(realization, {})[subbox_id] = xi
        valid_files += 1
    if s_reference is None:
        raise FileNotFoundError(f"没有有效 subbox xi 文件: {directory}")

    expected_ids = set(range(expected_subboxes))
    complete_realizations = sorted(
        realization for realization, by_subbox in records.items() if set(by_subbox) == expected_ids
    )
    if len(complete_realizations) < 2:
        raise RuntimeError(f"L{lsub} 完整母盒不足: {complete_realizations}")
    stack = np.stack(
        [np.stack([records[realization][subbox] for subbox in range(expected_subboxes)]) for realization in complete_realizations]
    )
    audit = cluster_covariance_audit(stack)
    audit.update(
        {
            "source_directory": str(directory),
            "valid_files_before_balancing": int(valid_files),
            "complete_parent_realizations": complete_realizations,
            "incomplete_parent_count": int(len(records) - len(complete_realizations)),
            "excluded_incomplete_subbox_count": int(valid_files - stack.shape[0] * stack.shape[1]),
        }
    )
    return stack, np.asarray(s_reference, dtype="f8"), np.asarray(complete_realizations, dtype="i8"), audit


def load_rawbox_case() -> FitCase:
    """读取 Task47 rawbox mean/cov，并构造 shell-average likelihood case。"""
    with np.load(RAWBOX_DATA, allow_pickle=False) as payload:
        s = np.asarray(payload["s"], dtype="f8")
        data = np.asarray(payload["mean_box"], dtype="f8")
        covariance = np.asarray(payload["cov_box"], dtype="f8")
        nmock = int(np.asarray(payload["realizations"]).size)
    precision, covariance_meta = build_precision(covariance, nmock=nmock)
    return FitCase(
        label="rawbox",
        sample_label="rawbox reference",
        model_label="rawbox",
        lsub=None,
        source_path=RAWBOX_DATA,
        s=s,
        s_edges=recover_shell_edges(s),
        data=data,
        covariance=covariance,
        precision=precision,
        covariance_meta=covariance_meta,
        cluster_meta={
            "policy": "98 distinct periodic parent realizations",
            "nparent_independent": int(nmock),
            "nsubbox_per_parent": 1,
            "nsubbox_total": int(nmock),
            "calibrated_nmock_for_hartlap_percival": int(nmock),
        },
    )


def load_subbox_cases(lsub: int) -> list[FitCase]:
    """重建 balanced parent-cluster mean/cov，并返回 no/formal-GIC 两个 case。"""
    path = Path(str(SUBBOX_DATA).format(lsub=int(lsub)))
    stack, s, complete_realizations, cluster_meta = load_balanced_subbox_stack(int(lsub))
    flattened = stack.reshape((-1, stack.shape[-1]))
    data = np.mean(flattened, axis=0)
    covariance = np.cov(flattened, rowvar=False, ddof=1)
    nmock = int(cluster_meta["calibrated_nmock_for_hartlap_percival"])
    precision, covariance_meta = build_precision(covariance, nmock=nmock)
    covariance_meta.update(
        {
            "nsubbox_observations": int(flattened.shape[0]),
            "nparent_independent": int(complete_realizations.size),
            "nmock_policy": "cluster-jackknife calibrated and capped at balanced Nsub",
        }
    )
    common = {
        "sample_label": f"L{int(lsub)}",
        "lsub": int(lsub),
        "source_path": path,
        "s": s,
        "s_edges": recover_shell_edges(s),
        "data": data,
        "covariance": covariance,
        "precision": precision,
        "covariance_meta": covariance_meta,
        "cluster_meta": cluster_meta,
    }
    return [
        FitCase(label=f"L{int(lsub)}_no_gic", model_label="no_gic", **common),
        FitCase(label=f"L{int(lsub)}_formal_gic", model_label="formal_gic", **common),
    ]


def shell_j0_average(k_values: np.ndarray, s_edges: np.ndarray) -> np.ndarray:
    """
    解析计算每个径向壳层的体积平均 j0。

    返回 shape 为 ``(nk, nshell)`` 的 kernel；公式是
    ``3 [sin(kr)-kr cos(kr)]_lo^hi / [k^3 (r_hi^3-r_lo^3)]``。
    """
    k = np.asarray(k_values, dtype="f8")[:, None]
    edges = np.asarray(s_edges, dtype="f8")
    lo = edges[:, 0][None, :]
    hi = edges[:, 1][None, :]
    shell_vol_no4pi = (hi**3 - lo**3) / 3.0
    upper = np.sin(k * hi) - k * hi * np.cos(k * hi)
    lower = np.sin(k * lo) - k * lo * np.cos(k * lo)
    return (upper - lower) / (k**3 * shell_vol_no4pi)


def project_shell_basis(theory: Any, pdense: np.ndarray, kernel: np.ndarray) -> np.ndarray:
    """把一个 dense P(k) basis 用 FullDiscrete mode weights 投影成 shell-averaged xi。"""
    task41 = theory.task41
    p_eff = task41.interp_logk(theory.k_eff, theory.k_dense, np.asarray(pdense, dtype="f8"))
    weights = np.asarray(theory.g_nz, dtype="f8") * p_eff
    return (weights @ np.asarray(kernel, dtype="f8")) / VOLUME


def sigma_basis(theory: Any, pdense: np.ndarray, lsub: int) -> float:
    """计算一个 P(k) basis 对指定 cubic window 的 sigma_W^2 贡献。"""
    task41 = theory.task41
    p_eff = task41.interp_logk(theory.k_eff, theory.k_dense, np.asarray(pdense, dtype="f8"))
    w2 = np.asarray(theory.w2_by_lsub[int(lsub)], dtype="f8")
    return float(np.sum(np.asarray(theory.g_nz, dtype="f8") * p_eff * w2) / VOLUME)


def build_shared_bases(*, n_dense: int) -> tuple[dict[int | None, ShellBasis], dict[str, Any]]:
    """构建一次 FullDiscrete/window cache，并返回 raw/no-GIC 与两个 Lsub 的 basis。"""
    task44 = import_module(ARCHIVED_TASK44, "task4p2_archived_task44_shellavg")
    task44.P_FIXED = P_FIXED
    task44.SN0_FIXED = SN0_FIXED
    if not np.isclose(float(task44.UNIT_Z), FASTPM_SNAPSHOT_Z, rtol=0.0, atol=1.0e-12):
        raise ValueError(f"归档 Task4.2 理论红移不是 z=1: {task44.UNIT_Z}")
    task41 = task44.import_task41_module()
    t0 = time.time()
    theory, theory_meta = task44.build_theory_context(task41, kmax_discrete=KMAX_DISCRETE, n_dense=int(n_dense))

    alpha = task41.interp_logk(theory.k_dense, theory.template_arrays["k"], theory.template_arrays["alpha"])
    pkdd = task41.interp_logk(theory.k_dense, theory.template_arrays["k"], theory.template_arrays["pk_dd"])
    gamma = 2.0 * DELTA_C * alpha
    dense_bases = (pkdd, gamma * pkdd, gamma**2 * pkdd)
    edges = np.column_stack([EXPECTED_EDGES[:-1], EXPECTED_EDGES[1:]])
    kernel = shell_j0_average(theory.k_eff, edges)
    xi_bases = tuple(project_shell_basis(theory, basis, kernel) for basis in dense_bases)

    # center projection 只用于量化本轮修复，不会进入任何 likelihood。
    center_bases = tuple(
        task41.fast_discrete_xi(EXPECTED_S, theory.g_nz, theory.k_eff, theory.k_dense, basis)
        for basis in dense_bases
    )
    shared = ShellBasis(*xi_bases)
    by_lsub: dict[int | None, ShellBasis] = {None: shared}
    for lsub in (1500, 1000):
        by_lsub[lsub] = ShellBasis(
            *xi_bases,
            sigma_pkdd=sigma_basis(theory, dense_bases[0], lsub),
            sigma_gamma_pkdd=sigma_basis(theory, dense_bases[1], lsub),
            sigma_gamma2_pkdd=sigma_basis(theory, dense_bases[2], lsub),
        )

    atomic_savez(
        THEORY_CACHE,
        s=EXPECTED_S,
        s_edges=edges,
        xi_pkdd=xi_bases[0],
        xi_gamma_pkdd=xi_bases[1],
        xi_gamma2_pkdd=xi_bases[2],
        xi_center_pkdd=center_bases[0],
        xi_center_gamma_pkdd=center_bases[1],
        xi_center_gamma2_pkdd=center_bases[2],
        sigma_L1500=np.asarray(
            [by_lsub[1500].sigma_pkdd, by_lsub[1500].sigma_gamma_pkdd, by_lsub[1500].sigma_gamma2_pkdd]
        ),
        sigma_L1000=np.asarray(
            [by_lsub[1000].sigma_pkdd, by_lsub[1000].sigma_gamma_pkdd, by_lsub[1000].sigma_gamma2_pkdd]
        ),
        template_z=np.asarray(FASTPM_SNAPSHOT_Z),
        p_fixed=np.asarray(P_FIXED),
        sn0_fixed=np.asarray(SN0_FIXED),
        kmax_discrete=np.asarray(KMAX_DISCRETE),
        n_dense=np.asarray(int(n_dense)),
    )
    metadata = {
        **theory_meta,
        "template_z": FASTPM_SNAPSHOT_Z,
        "xi_kernel": "volume-shell-averaged-j0",
        "shell_edges_mpc_h": EXPECTED_EDGES.tolist(),
        "sn0_policy": "fixed_contact_term",
        "sn0_fixed": SN0_FIXED,
        "sigmas_fixed": SIGMAS_FIXED,
        "basis_cache": str(THEORY_CACHE),
        "archived_theory_module": str(ARCHIVED_TASK44),
        "archived_theory_module_sha256": file_sha256(ARCHIVED_TASK44),
        "build_elapsed_sec": float(time.time() - t0),
        "center_vs_shell_basis_max_abs": {
            "pkdd": float(np.max(np.abs(xi_bases[0] - center_bases[0]))),
            "gamma_pkdd": float(np.max(np.abs(xi_bases[1] - center_bases[1]))),
            "gamma2_pkdd": float(np.max(np.abs(xi_bases[2] - center_bases[2]))),
        },
    }
    return by_lsub, metadata


def coefficients(theta: np.ndarray) -> tuple[float, float, float]:
    """把 ``fnl_loc,b1`` 转成三个 PNG basis 的系数。"""
    fnl_loc, b1 = np.asarray(theta, dtype="f8")
    return (
        float(b1**2),
        float(2.0 * b1 * (b1 - P_FIXED) * fnl_loc),
        float((b1 - P_FIXED) ** 2 * fnl_loc**2),
    )


def model_xi(case: FitCase, basis: ShellBasis, theta: np.ndarray) -> tuple[np.ndarray, float]:
    """计算一个 case 的 shell-averaged raw/no-GIC/formal-GIC 模型。"""
    c0, c1, c2 = coefficients(theta)
    xi = c0 * basis.xi_pkdd + c1 * basis.xi_gamma_pkdd + c2 * basis.xi_gamma2_pkdd
    sigma_w2 = 0.0
    if case.model_label == "formal_gic":
        sigma_w2 = c0 * basis.sigma_pkdd + c1 * basis.sigma_gamma_pkdd + c2 * basis.sigma_gamma2_pkdd
        xi = xi - sigma_w2
    return np.asarray(xi, dtype="f8"), float(sigma_w2)


def log_prior(theta: np.ndarray) -> float:
    """二维均匀先验。"""
    values = np.asarray(theta, dtype="f8")
    for index, name in enumerate(PARAM_NAMES):
        lo, hi = PRIORS[name]
        if not lo <= float(values[index]) <= hi:
            return -np.inf
    return 0.0


def make_log_prob(case: FitCase, basis: ShellBasis) -> Callable[[np.ndarray], float]:
    """构造单个 case 的 log posterior。"""
    def log_prob(theta: np.ndarray) -> float:
        prior = log_prior(theta)
        if not np.isfinite(prior):
            return -np.inf
        model, _sigma = model_xi(case, basis, theta)
        residual = case.data - model
        return float(prior - 0.5 * residual @ case.precision @ residual)

    return log_prob


def optimize_case(case: FitCase, basis: ShellBasis) -> dict[str, Any]:
    """用多起点 L-BFGS-B 找 walker 初始中心与 profile best fit。"""
    starts = ([80.0, 2.7], [100.0, 2.6], [60.0, 3.0], [120.0, 2.4], [0.0, 2.8])
    bounds = [PRIORS[name] for name in PARAM_NAMES]

    def chi2(theta: np.ndarray) -> float:
        model, _sigma = model_xi(case, basis, theta)
        residual = case.data - model
        return float(residual @ case.precision @ residual)

    results = [
        minimize(chi2, np.asarray(start, dtype="f8"), method="L-BFGS-B", bounds=bounds, options={"maxiter": 3000, "ftol": 1.0e-12})
        for start in starts
    ]
    best = min(results, key=lambda item: float(item.fun))
    return {
        "point_array": np.asarray(best.x, dtype="f8"),
        "point": {name: float(best.x[index]) for index, name in enumerate(PARAM_NAMES)},
        "chi2": float(best.fun),
        "success": bool(best.success),
        "message": str(best.message),
    }


def summarize_samples(samples: np.ndarray, percival: float) -> dict[str, dict[str, float]]:
    """计算等权 posterior 分位数与 Percival-scaled 68% 误差。"""
    output: dict[str, dict[str, float]] = {}
    for index, name in enumerate(PARAM_NAMES):
        values = np.asarray(samples[:, index], dtype="f8")
        q025, q16, q50, q84, q975 = np.quantile(values, [0.025, 0.1586552539, 0.5, 0.8413447461, 0.975])
        err_low = float(q50 - q16)
        err_high = float(q84 - q50)
        output[name] = {
            "mean": float(np.mean(values)),
            "std": float(np.std(values, ddof=1)),
            "median": float(q50),
            "q025": float(q025),
            "q16": float(q16),
            "q84": float(q84),
            "q975": float(q975),
            "err_low": err_low,
            "err_high": err_high,
            "err_low_percival": float(err_low * percival),
            "err_high_percival": float(err_high * percival),
            "std_percival": float(np.std(values, ddof=1) * percival),
        }
    return output


def convergence_diagnostics(chain: np.ndarray, burnin: int) -> dict[str, Any]:
    """计算 tau、ESS 与 split stability，并返回统一 convergence gate。"""
    postburn = np.asarray(chain[int(burnin) :, :, :], dtype="f8")
    tau = np.asarray(emcee.autocorr.integrated_time(postburn, quiet=True), dtype="f8")
    nsteps, nwalkers, _ndim = postburn.shape
    by_parameter: dict[str, Any] = {}
    for index, (name, value) in enumerate(zip(PARAM_NAMES, tau, strict=True)):
        array = postburn[:, :, index]
        half = nsteps // 2
        delta = float(np.median(array[half:]) - np.median(array[:half]))
        std = float(np.std(array, ddof=1))
        quarters = [float(np.median(part)) for part in np.array_split(array, 4, axis=0)]
        by_parameter[name] = {
            "tau_steps": float(value),
            "postburn_steps_per_tau": float(nsteps / value),
            "effective_samples_approx": float(nsteps * nwalkers / value),
            "first_vs_second_half_delta_over_std": float(delta / std),
            "quarter_medians": quarters,
            "passes_50_tau_gate": bool(nsteps >= 50.0 * value),
            "passes_half_shift_0p1sigma_gate": bool(abs(delta) <= 0.1 * std),
        }
    passed = all(
        item[gate]
        for item in by_parameter.values()
        for gate in ("passes_50_tau_gate", "passes_half_shift_0p1sigma_gate")
    )
    return {
        "status": "pass" if passed else "short_or_unstable",
        "postburn_shape": list(postburn.shape),
        "parameters": by_parameter,
    }


def run_case(
    case: FitCase,
    basis: ShellBasis,
    *,
    nwalkers: int,
    min_steps: int,
    max_steps: int,
    burnin: int,
    chunk_steps: int,
    seed: int,
) -> dict[str, Any]:
    """运行一个 convergence-gated emcee case，并原子写 samples/summary。"""
    optimizer = optimize_case(case, basis)
    center = np.asarray(optimizer.pop("point_array"), dtype="f8")
    rng = np.random.default_rng(int(seed))
    scales = np.array([max(3.0, 0.03 * abs(center[0])), 0.02], dtype="f8")
    walkers = center[None, :] + rng.normal(scale=scales, size=(int(nwalkers), len(PARAM_NAMES)))
    for index, name in enumerate(PARAM_NAMES):
        lo, hi = PRIORS[name]
        walkers[:, index] = np.clip(walkers[:, index], lo + 1.0e-8, hi - 1.0e-8)

    np.random.seed(int(seed))
    sampler = emcee.EnsembleSampler(int(nwalkers), len(PARAM_NAMES), make_log_prob(case, basis))
    state: Any = walkers
    convergence: dict[str, Any] | None = None
    t0 = time.time()
    while int(sampler.iteration) < int(max_steps):
        steps = min(int(chunk_steps), int(max_steps) - int(sampler.iteration))
        state = sampler.run_mcmc(state, steps, progress=False, skip_initial_state_check=True)
        if int(sampler.iteration) >= int(min_steps):
            convergence = convergence_diagnostics(sampler.get_chain(), burnin)
            if convergence["status"] == "pass":
                break

    chain = np.asarray(sampler.get_chain(), dtype="f8")
    log_prob = np.asarray(sampler.get_log_prob(), dtype="f8")
    convergence = convergence or convergence_diagnostics(chain, burnin)
    flat_samples = chain[int(burnin) :, :, :].reshape((-1, len(PARAM_NAMES)))
    flat_log_prob = log_prob[int(burnin) :, :].reshape((-1,))
    parameters = summarize_samples(flat_samples, float(case.covariance_meta["percival_error_factor"]))
    imax = int(np.argmax(flat_log_prob))
    best_theta = flat_samples[imax]
    best_model, sigma_w2 = model_xi(case, basis, best_theta)
    residual = case.data - best_model
    best = {
        "point": {name: float(best_theta[index]) for index, name in enumerate(PARAM_NAMES)},
        "log_prob": float(flat_log_prob[imax]),
        "chi2": float(residual @ case.precision @ residual),
        "ndof": int(case.data.size - len(PARAM_NAMES)),
        "sigma_w2": float(sigma_w2),
        "xi_model": best_model,
        "xi_residual": residual,
    }

    samples_path = OUTPUT_ROOT / "samples" / f"task4p2_{case.label}_shellavg_fixed_sn0_samples.npz"
    summary_path = OUTPUT_ROOT / "summaries" / f"task4p2_{case.label}_shellavg_fixed_sn0_summary.json"
    atomic_savez(
        samples_path,
        param_names=np.asarray(PARAM_NAMES),
        samples=flat_samples,
        log_prob=flat_log_prob,
        nwalkers=np.asarray(nwalkers),
        nsteps=np.asarray(chain.shape[0]),
        burnin=np.asarray(burnin),
        s=case.s,
        s_edges=case.s_edges,
        xi_data=case.data,
        covariance=case.covariance,
        xi_model_best=best_model,
    )
    summary = {
        "task": "task4p2_fit_2pcf_shellavg_fixed_sn0",
        "status": "pass" if convergence["status"] == "pass" else "convergence_failed",
        "label": case.label,
        "sample_label": case.sample_label,
        "model_label": case.model_label,
        "payload": {
            "path": str(case.source_path),
            "sha256": file_sha256(case.source_path),
            "lsub": case.lsub,
        },
        "selection": {
            "edge_min": 50.0,
            "edge_max": 350.0,
            "nbins": int(case.s.size),
            "s_centers": case.s,
            "s_edges": case.s_edges,
        },
        "config": {
            "parameter_names": list(PARAM_NAMES),
            "free_parameters": list(PARAM_NAMES),
            "fixed_parameters": {"p": P_FIXED, "sn0": SN0_FIXED, "sigmas": SIGMAS_FIXED},
            "priors": {name: list(bounds) for name, bounds in PRIORS.items()},
            "template_z": FASTPM_SNAPSHOT_Z,
            "xi_kernel": "volume-shell-averaged-j0",
            "kmax_discrete": KMAX_DISCRETE,
            "nwalkers": int(nwalkers),
            "min_steps": int(min_steps),
            "max_steps": int(max_steps),
            "completed_steps": int(chain.shape[0]),
            "burnin": int(burnin),
            "chunk_steps": int(chunk_steps),
            "seed": int(seed),
        },
        "parameters": parameters,
        "maximum_posterior_sample": best,
        "optimizer": optimizer,
        "covariance": case.covariance_meta,
        "parent_cluster_audit": case.cluster_meta,
        "diagnostics": {
            "convergence": convergence,
            "acceptance_fraction_mean": float(np.mean(sampler.acceptance_fraction)),
            "acceptance_fraction_min": float(np.min(sampler.acceptance_fraction)),
            "acceptance_fraction_max": float(np.max(sampler.acceptance_fraction)),
            "elapsed_sec": float(time.time() - t0),
        },
        "paths": {"samples": str(samples_path), "summary": str(summary_path)},
    }
    write_json(summary_path, to_jsonable(summary))
    print(
        f"[done] {case.label}: status={summary['status']} steps={chain.shape[0]} "
        f"fnl={parameters['fnl_loc']['median']:.3f}",
        flush=True,
    )
    return summary


def parse_args() -> argparse.Namespace:
    """解析 CPU/MCMC 参数。"""
    parser = argparse.ArgumentParser(description="Task4.2 shell-averaged fixed-sn0 2PCF rerun")
    parser.add_argument("--n-dense", type=int, default=60000)
    parser.add_argument("--nwalkers", type=int, default=32)
    parser.add_argument("--min-steps", type=int, default=8000)
    parser.add_argument("--max-steps", type=int, default=20000)
    parser.add_argument("--burnin", type=int, default=1000)
    parser.add_argument("--chunk-steps", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=20260712)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> None:
    """构建共享 basis、运行五条链并写统一审计 summary。"""
    args = parse_args()
    if int(args.nwalkers) < 2 * len(PARAM_NAMES):
        raise ValueError("nwalkers 必须至少为 2*ndim")
    if not 0 <= int(args.burnin) < int(args.min_steps) <= int(args.max_steps):
        raise ValueError("必须满足 0 <= burnin < min_steps <= max_steps")
    if COMBINED_SUMMARY.exists() and not bool(args.overwrite):
        existing = json.loads(COMBINED_SUMMARY.read_text(encoding="utf-8"))
        if existing.get("status") == "pass":
            print(f"[skip] 已有通过审计的结果: {COMBINED_SUMMARY}")
            return

    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    cases = [load_rawbox_case(), *load_subbox_cases(1500), *load_subbox_cases(1000)]
    if any(not np.array_equal(case.s, EXPECTED_S) for case in cases):
        raise ValueError("五个 case 的 s centers 不完全一致")
    bases, theory_meta = build_shared_bases(n_dense=int(args.n_dense))

    summaries: dict[str, Any] = {}
    for index, case in enumerate(cases):
        basis = bases[case.lsub] if case.model_label == "formal_gic" else bases[None]
        summaries[case.label] = run_case(
            case,
            basis,
            nwalkers=int(args.nwalkers),
            min_steps=int(args.min_steps),
            max_steps=int(args.max_steps),
            burnin=int(args.burnin),
            chunk_steps=int(args.chunk_steps),
            seed=int(args.seed) + 1009 * index,
        )

    failed = [label for label, summary in summaries.items() if summary["status"] != "pass"]
    combined = {
        "task": "task4p2_fit_2pcf_shellavg_fixed_sn0",
        "status": "pass" if not failed else "convergence_failed",
        "failed_cases": failed,
        "scientific_scope": {
            "samples": ["rawbox", "L1500", "L1000"],
            "models": ["rawbox", "no_gic", "formal_gic"],
            "display_edges_mpc_h": [50.0, 350.0],
            "actual_centers_mpc_h": [55.0, 345.0],
            "xi_kernel": "volume-shell-averaged-j0",
            "template_z": FASTPM_SNAPSHOT_Z,
            "free_parameters": list(PARAM_NAMES),
            "fixed_parameters": {"p": P_FIXED, "sn0": SN0_FIXED, "sigmas": SIGMAS_FIXED},
        },
        "mcmc": {
            "nwalkers": int(args.nwalkers),
            "min_steps": int(args.min_steps),
            "max_steps": int(args.max_steps),
            "burnin": int(args.burnin),
            "chunk_steps": int(args.chunk_steps),
            "execution": "login node, CPU-only, one process/one BLAS thread",
        },
        "theory": theory_meta,
        "summaries": summaries,
        "paths": {"combined_summary": str(COMBINED_SUMMARY), "theory_cache": str(THEORY_CACHE)},
    }
    write_json(COMBINED_SUMMARY, to_jsonable(combined))
    print(f"[write] {COMBINED_SUMMARY}", flush=True)
    print(f"[status] {combined['status']}", flush=True)
    if failed:
        raise SystemExit(f"convergence failed: {failed}")


if __name__ == "__main__":
    main()
