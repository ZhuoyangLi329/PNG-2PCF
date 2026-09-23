#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Task4.3 radial single-term RIC 的共享实现。

代码大纲
========
1. 解析 Task4.3 已归档但不可变的 random catalog，并恢复与原测量完全相同的
   ``WEIGHT_TOTAL = WEIGHT * WEIGHT_FKP``。
2. 用四个 random 位置 ``(x1, x2, y1, y2)`` 蒙特卡洛采样论文中的
   ``IC^(rad,rad)``：外层 pair 决定观测 separation，内层 pair 的两个端点
   分别被限制在与外层端点相同的 radial bin。
3. 将二维 ``p(s_outer, Delta_inner)`` cache 转换为：
   - 2PCF 的 RR-normalised、实际 shell-averaged correction；
   - P(k) 的 window-convolved response matrix。
4. 提供 global-limit、geometry-window reconstruction 和 Hankel/direct closure
   所需的通用数学工具。这里实现的始终只是 single auto term；不会悄悄加入
   两个 density--RIC cross terms，也不会再额外减一次 global sigma_W^2。

数学约定
========
对随机位置分布 ``p_W(x) propto W(x)``，radial bin 标记为 ``a(x)``。定义

``C_rad(s) = E[xi(|y1-y2|) | |x1-x2| in s,
                                  a(y1)=a(x1), a(y2)=a(x2)]``。

这已经包含 LS estimator 的 RR(s) 除法。因此 2PCF single-term 模型直接为
``xi_model(s) - C_rad(s)``。对 P(k)，则保留未除 RR 的外层 pair 权重并计算
``V_FKP E[j0(k s_outer) xi(Delta_inner)]``。两侧读取同一个二维 cache，避免
构造两套彼此不一致的经验修正。
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np


PROJECT_ROOT = Path("/pscratch/sd/l/lzy/PNG-2PCF-model-to-observe")
TASK43_ROOT = PROJECT_ROOT / "outputs" / "task43_outputs"
RIC_ROOT = TASK43_ROOT / "ric_singleterm"
RIC_CACHE_DIR = RIC_ROOT / "kernels"
RIC_OPERATOR_DIR = RIC_ROOT / "operators"
RIC_FIT_DIR = RIC_ROOT / "fits"
RIC_AUDIT_DIR = RIC_ROOT / "audits"
RIC_PLOT_DIR = PROJECT_ROOT / "plots" / "task43" / "ric_singleterm"

DEFAULT_MANIFEST = TASK43_ROOT / "manifests" / "task43_mmin1p4e13_x25.jsonl"
DEFAULT_FKP_SUMMARY = TASK43_ROOT / "summary" / "task43_fkp_zeff_mmin1p4e13_x25.npz"
DEFAULT_XI = TASK43_ROOT / "summary" / "task43_mean_xi_mmin1p4e13_x25_s50_350_ds10_fkpP010000.npz"
DEFAULT_PK = (
    TASK43_ROOT
    / "pk_lightcone"
    / "summary"
    / "task43_pk_lightcone_mmin1p4e13_x25_fkpP010000_desi_rebin_kmax0p10_payload.npz"
)

# 2026-07-07 的仓库清理只移动、没有删除这些大 catalog。manifest 中保存的是
# 移动前路径，所以新分析必须显式记录 fallback provenance。
ARCHIVE_RANDOM_ROOT = (
    PROJECT_ROOT
    / "old_doc_codes"
    / "task4_task44_cleanup_20260707T061844Z"
    / "moved"
    / "outputs"
    / "task43_outputs"
    / "randoms"
)


def to_jsonable(value: Any) -> Any:
    """递归转换 NumPy/Path 对象，确保 audit metadata 可写成 JSON。"""
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, dict):
        return {str(key): to_jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [to_jsonable(item) for item in value]
    return str(value)


def write_json(path: Path, payload: dict[str, Any]) -> None:
    """原子写 JSON，避免长任务中断后留下半个 summary。"""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    tmp.write_text(json.dumps(to_jsonable(payload), indent=2, sort_keys=True) + "\n", encoding="utf-8")
    tmp.replace(path)


def atomic_savez(path: Path, **arrays: Any) -> None:
    """原子写压缩 NPZ；失败时只清理本函数自己的临时文件。"""
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


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    """读取 Task4.3 manifest。"""
    rows: list[dict[str, Any]] = []
    with Path(path).open("r", encoding="utf-8") as stream:
        for line in stream:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def manifest_row(path: Path, phase: str) -> dict[str, Any]:
    """返回指定 phase 的唯一 manifest 行。"""
    rows = [row for row in read_jsonl(path) if str(row.get("phase")) == str(phase)]
    if len(rows) != 1:
        raise ValueError(f"manifest 中 phase={phase!r} 的行数不是 1，而是 {len(rows)}")
    return rows[0]


def resolve_archived_path(recorded_path: str | Path, *, archive_root: Path = ARCHIVE_RANDOM_ROOT) -> tuple[Path, dict[str, Any]]:
    """解析 manifest 中已被仓库清理移动的 catalog 路径。

    参数
    ----
    recorded_path
        manifest 当时记录的绝对或相对路径。
    archive_root
        只在原路径不存在时搜索的、已知且受控的归档目录。

    返回
    ----
    resolved, metadata
        实际读取路径及是否使用 archive fallback 的 provenance。
    """
    original = Path(recorded_path)
    candidates = [original]
    if not original.is_absolute():
        candidates.append(PROJECT_ROOT / original)
    for candidate in candidates:
        if candidate.exists():
            return candidate.resolve(), {
                "recorded_path": str(recorded_path),
                "resolved_path": str(candidate.resolve()),
                "archive_fallback": False,
            }
    archived = archive_root / original.name
    if archived.exists():
        return archived.resolve(), {
            "recorded_path": str(recorded_path),
            "resolved_path": str(archived.resolve()),
            "archive_fallback": True,
            "archive_reason": "2026-07-07 repository cleanup moved immutable Task43 catalogs",
        }
    raise FileNotFoundError(f"catalog 既不在原路径，也不在受控归档目录：{recorded_path}")


def load_fkp_arrays(path: Path) -> dict[str, np.ndarray]:
    """读取构造原测量 FKP weight 所需的少量数组。"""
    with np.load(path, allow_pickle=False) as data:
        return {key: np.asarray(data[key]) for key in data.files}


def fkp_weight(z: np.ndarray, summary: dict[str, np.ndarray], p0: float) -> np.ndarray:
    """复现 Task4.3 的 ``1 / (1 + nbar(z) P0)`` 插值口径。"""
    z = np.asarray(z, dtype="f8")
    centers = np.asarray(summary["z_centers"], dtype="f8")
    nbar = np.asarray(summary["nbar"], dtype="f8")
    n_of_z = np.interp(z, centers, nbar, left=nbar[0], right=nbar[-1])
    return 1.0 / (1.0 + n_of_z * float(p0))


def fkp_effective_normalisation(summary: dict[str, np.ndarray], p0: float) -> dict[str, float]:
    """计算 Fourier pair integral 的 ``S_W^2/A`` 绝对归一化。

    对 ``W=nbar*w_FKP``，有 ``S_W=integral W dV``、
    ``A=integral W^2 dV``。随机 pair 期望乘 ``S_W^2/A`` 后与当前 FKP
    P(k) estimator 同单位；Task4.3 中该量约等于 lightcone shell volume。
    """
    nbar = np.asarray(summary["nbar"], dtype="f8")
    volume = np.asarray(summary["volume_shell"], dtype="f8")
    weight = 1.0 / (1.0 + nbar * float(p0))
    sw = float(np.sum(nbar * weight * volume))
    norm = float(np.sum(nbar * nbar * weight * weight * volume))
    if sw <= 0.0 or norm <= 0.0:
        raise ValueError("FKP summary 给出了非正的 S_W 或 A")
    return {
        "s_w": sw,
        "a_fkp": norm,
        "pair_volume": sw * sw / norm,
        "shell_volume": float(np.sum(volume)),
    }


@dataclass
class RandomSubsample:
    """一次 kernel 构造所需的 random 子样本及 provenance。"""

    xyz: np.ndarray
    chi: np.ndarray
    weight: np.ndarray
    source_indices: np.ndarray
    metadata: dict[str, Any]


def load_random_subsample(
    path: Path,
    *,
    fkp_summary: dict[str, np.ndarray],
    p0: float,
    n_subsample: int,
    seed: int,
) -> RandomSubsample:
    """均匀无放回抽 random 点，再恢复每点 FKP integration weight。

    子样本本身均匀抽取，随后所有外层/内层积分按 ``WEIGHT_TOTAL`` 重采样；
    这样不会把 FKP weight 同时用在抽样概率和点权中两次。
    """
    path = Path(path)
    with np.load(path, allow_pickle=False) as data:
        n_total = int(np.asarray(data["Z"]).size)
        n_use = n_total if int(n_subsample) <= 0 else min(n_total, int(n_subsample))
        rng = np.random.default_rng(int(seed))
        if n_use < n_total:
            indices = np.sort(rng.choice(n_total, size=n_use, replace=False))
        else:
            indices = np.arange(n_total, dtype="i8")
        xyz = np.column_stack(
            [
                np.asarray(data["X"][indices], dtype="f8"),
                np.asarray(data["Y"][indices], dtype="f8"),
                np.asarray(data["Zcart"][indices], dtype="f8"),
            ]
        )
        z = np.asarray(data["Z"][indices], dtype="f8")
        base = np.asarray(data["WEIGHT"][indices], dtype="f8") if "WEIGHT" in data.files else np.ones(n_use)
    weight = base * fkp_weight(z, fkp_summary, float(p0))
    chi = np.linalg.norm(xyz, axis=1)
    metadata = {
        "path": str(path),
        "n_total": n_total,
        "n_subsample": int(n_use),
        "subsample_seed": int(seed),
        "selection": "uniform_without_replacement_then_WEIGHT_TOTAL_in_integrals",
        "weight_policy": "WEIGHT_TOTAL=WEIGHT*WEIGHT_FKP",
        "p0": float(p0),
        "weight_min": float(np.min(weight)),
        "weight_max": float(np.max(weight)),
        "weight_sum": float(np.sum(weight)),
        "chi_min": float(np.min(chi)),
        "chi_max": float(np.max(chi)),
    }
    return RandomSubsample(xyz=xyz, chi=chi, weight=weight, source_indices=indices, metadata=metadata)


def radial_bin_edges(chi: np.ndarray, width: float) -> np.ndarray:
    """以 chi=0 为固定锚点生成 phase 间可比较的 radial bins。"""
    if float(width) <= 0.0:
        return np.asarray([float(np.min(chi)) - 1.0, float(np.max(chi)) + 1.0], dtype="f8")
    lo = np.floor(float(np.min(chi)) / float(width)) * float(width)
    hi = np.ceil(float(np.max(chi)) / float(width)) * float(width)
    return np.arange(lo, hi + 0.5 * float(width), float(width), dtype="f8")


class ConditionalRadialSampler:
    """按 radial-bin 条件分布重复抽取 inner random 端点。

    每个 bin 内仍严格使用 catalog 的 ``WEIGHT_TOTAL``。接口接收与外层端点
    等长的 bin label 数组，并返回内层端点的 catalog index。
    """

    def __init__(self, labels: np.ndarray, weights: np.ndarray, rng: np.random.Generator):
        self.labels = np.asarray(labels, dtype="i8")
        self.weights = np.asarray(weights, dtype="f8")
        self.rng = rng
        self.members: dict[int, np.ndarray] = {}
        self.probabilities: dict[int, np.ndarray] = {}
        for label in np.unique(self.labels):
            ids = np.flatnonzero(self.labels == int(label))
            local = self.weights[ids]
            self.members[int(label)] = ids
            self.probabilities[int(label)] = local / np.sum(local)

    def sample(self, requested_labels: np.ndarray) -> np.ndarray:
        """对每个 requested label 独立抽一个同-bin random index。"""
        requested = np.asarray(requested_labels, dtype="i8")
        out = np.empty(requested.size, dtype="i8")
        for label in np.unique(requested):
            where = np.flatnonzero(requested == int(label))
            ids = self.members[int(label)]
            out[where] = self.rng.choice(ids, size=where.size, replace=True, p=self.probabilities[int(label)])
        return out


def _resample_bad_conditional(
    sampled: np.ndarray,
    requested_labels: np.ndarray,
    forbidden: tuple[np.ndarray, ...],
    sampler: ConditionalRadialSampler,
) -> np.ndarray:
    """去掉有限 random 子样本产生的人工同点 coincidence。"""
    sampled = np.asarray(sampled, dtype="i8").copy()
    bad = np.zeros(sampled.size, dtype=bool)
    for other in forbidden:
        bad |= sampled == np.asarray(other, dtype="i8")
    attempts = 0
    while np.any(bad):
        sampled[bad] = sampler.sample(np.asarray(requested_labels, dtype="i8")[bad])
        bad[:] = False
        for other in forbidden:
            bad |= sampled == np.asarray(other, dtype="i8")
        attempts += 1
        if attempts > 100:
            raise RuntimeError("同一 radial bin 可用点过少，无法排除人工 self coincidence")
    return sampled


def monte_carlo_joint_histogram(
    random: RandomSubsample,
    *,
    radial_width: float,
    separation_edges: np.ndarray,
    n_quadruplets: int,
    batch_size: int,
    seed: int,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    """采样 ``p(s_outer, Delta_inner)`` 的二维 histogram。

    外层两个点从全局 FKP-weighted selection 独立抽取；内层两个点分别从与
    对应外层端点相同的 radial bin 条件抽取。返回的 row sum 是 MC 外层
    separation histogram，之后可用精确 pycorr marginal 做 row reweight。
    """
    edges = np.asarray(separation_edges, dtype="f8")
    if edges.ndim != 1 or edges.size < 2 or np.any(np.diff(edges) <= 0.0):
        raise ValueError("separation_edges 必须严格递增")
    nbin = edges.size - 1
    nquad = int(n_quadruplets)
    if nquad <= 0:
        raise ValueError("n_quadruplets 必须为正")
    rng = np.random.default_rng(int(seed))
    radial_edges = radial_bin_edges(random.chi, float(radial_width))
    radial_label = np.searchsorted(radial_edges, random.chi, side="right") - 1
    radial_label = np.clip(radial_label, 0, radial_edges.size - 2)
    sampler = ConditionalRadialSampler(radial_label, random.weight, rng)
    global_prob = random.weight / np.sum(random.weight)
    flat_counts = np.zeros(nbin * nbin, dtype="i8")
    covered = 0

    for start in range(0, nquad, int(batch_size)):
        size = min(int(batch_size), nquad - start)
        outer1 = rng.choice(random.xyz.shape[0], size=size, replace=True, p=global_prob)
        outer2 = rng.choice(random.xyz.shape[0], size=size, replace=True, p=global_prob)
        same = outer1 == outer2
        while np.any(same):
            outer2[same] = rng.choice(random.xyz.shape[0], size=int(np.sum(same)), replace=True, p=global_prob)
            same = outer1 == outer2

        label1 = radial_label[outer1]
        label2 = radial_label[outer2]
        inner1 = sampler.sample(label1)
        inner1 = _resample_bad_conditional(inner1, label1, (outer1, outer2), sampler)
        inner2 = sampler.sample(label2)
        inner2 = _resample_bad_conditional(inner2, label2, (outer1, outer2, inner1), sampler)

        s_outer = np.linalg.norm(random.xyz[outer1] - random.xyz[outer2], axis=1)
        delta_inner = np.linalg.norm(random.xyz[inner1] - random.xyz[inner2], axis=1)
        si = np.searchsorted(edges, s_outer, side="right") - 1
        di = np.searchsorted(edges, delta_inner, side="right") - 1
        valid = (si >= 0) & (si < nbin) & (di >= 0) & (di < nbin)
        covered += int(np.sum(valid))
        flat = si[valid] * nbin + di[valid]
        flat_counts += np.bincount(flat, minlength=flat_counts.size)

    counts = flat_counts.reshape((nbin, nbin))
    meta = {
        "method": "weighted_random_quadruplet_radial_auto",
        "n_quadruplets_requested": nquad,
        "n_quadruplets_covered": covered,
        "coverage": float(covered / nquad),
        "batch_size": int(batch_size),
        "seed": int(seed),
        "radial_width": float(radial_width),
        "radial_edges": radial_edges,
        "n_radial_bins": int(radial_edges.size - 1),
        "min_random_per_radial_bin": int(min(ids.size for ids in sampler.members.values())),
        "max_random_per_radial_bin": int(max(ids.size for ids in sampler.members.values())),
    }
    return counts, radial_edges, meta


def exact_outer_pair_histogram(
    random: RandomSubsample,
    *,
    separation_edges: np.ndarray,
    nthreads: int,
) -> tuple[np.ndarray, dict[str, Any]]:
    """用 pycorr/Corrfunc 计算同一 random 子样本的 weighted outer RR(s)。"""
    from pycorr import TwoPointCorrelationFunction

    edges = np.asarray(separation_edges, dtype="f8").copy()
    # Corrfunc 不接受严格为 0 的第一条边；0.5 Mpc/h 以下 continuum pair 对本
    # lightcone 的低-k和 50--350 Mpc/h 2PCF 都可忽略，同时明确排除 self pair。
    if edges[0] <= 0.0:
        edges[0] = 0.5
    result = TwoPointCorrelationFunction(
        mode="s",
        edges=edges,
        data_positions1=random.xyz.T,
        data_weights1=random.weight,
        randoms_positions1=random.xyz.T,
        randoms_weights1=random.weight,
        estimator="natural",
        nthreads=int(nthreads),
    )
    counts = np.asarray(result.R1R2.wcounts, dtype="f8")
    total = float(np.sum(counts))
    if total <= 0.0:
        raise RuntimeError("pycorr outer RR histogram 总权重非正")
    return counts, {
        "method": "pycorr_weighted_R1R2",
        "nthreads": int(nthreads),
        "edge0_effective": float(edges[0]),
        "weighted_pair_sum": total,
        "coverage_policy": "all separations through survey diagonal; self pair excluded by smin=0.5",
    }


def reweight_joint_rows(joint_counts: np.ndarray, outer_counts: np.ndarray) -> tuple[np.ndarray, dict[str, Any]]:
    """保留 MC 条件分布 ``p(Delta|s)``，替换成精确 outer ``p(s)``。

    极小 separation 若 MC 某行为空，用最近的非空条件行填充；这些行的外层
    pair probability 会被保留并在 metadata 中明确计数，避免静默丢失。
    """
    counts = np.asarray(joint_counts, dtype="f8")
    outer = np.asarray(outer_counts, dtype="f8")
    if counts.shape[0] != outer.size:
        raise ValueError("joint_counts 与 outer_counts 的 s 维不一致")
    row_sum = np.sum(counts, axis=1)
    conditional = np.zeros_like(counts)
    nonempty = row_sum > 0.0
    conditional[nonempty] = counts[nonempty] / row_sum[nonempty, None]
    outer_nonzero = outer > 0.0
    fill_rows = np.flatnonzero(outer_nonzero & ~nonempty)
    available = np.flatnonzero(nonempty)
    if available.size == 0:
        raise RuntimeError("joint MC histogram 没有任何非空行")
    for row in fill_rows:
        nearest = available[int(np.argmin(np.abs(available - row)))]
        conditional[row] = conditional[nearest]
    outer_prob = outer / np.sum(outer)
    joint_prob = conditional * outer_prob[:, None]
    return joint_prob, {
        "mc_nonempty_rows": int(np.sum(nonempty)),
        "outer_nonzero_rows": int(np.sum(outer_nonzero)),
        "nearest_filled_rows": [int(v) for v in fill_rows],
        "joint_probability_sum": float(np.sum(joint_prob)),
        "outer_probability_sum": float(np.sum(outer_prob)),
    }


def load_kernel(path: Path) -> dict[str, Any]:
    """读取 radial cache，并把 ``meta_json`` 反序列化。"""
    with np.load(path, allow_pickle=False) as data:
        out = {key: np.asarray(data[key]) for key in data.files if key != "meta_json"}
        out["meta"] = json.loads(str(np.asarray(data["meta_json"]).item()))
    return out


def spherical_j0(argument: np.ndarray) -> np.ndarray:
    """稳定计算 j0(x)=sin(x)/x。"""
    argument = np.asarray(argument, dtype="f8")
    out = np.ones_like(argument)
    mask = argument != 0.0
    out[mask] = np.sin(argument[mask]) / argument[mask]
    return out


def shell_average_j0(k_values: np.ndarray, lower: np.ndarray, upper: np.ndarray) -> np.ndarray:
    """返回 j0(k r) 对每个球壳 ``r^2 dr`` 的解析平均。

    输出 shape 为 ``(nk, nshell)``。同一函数也可交换变量使用，从而计算
    observed k-bin 中按 ``k^2 dk`` 平均的 j0(k s)。
    """
    kval = np.asarray(k_values, dtype="f8")[:, None]
    lower = np.asarray(lower, dtype="f8")[None, :]
    upper = np.asarray(upper, dtype="f8")[None, :]
    shell = (upper**3 - lower**3) / 3.0
    out = np.ones((kval.shape[0], lower.shape[1]), dtype="f8")
    nonzero = kval[:, 0] != 0.0
    if np.any(nonzero):
        k = kval[nonzero]
        hi = np.sin(k * upper) - k * upper * np.cos(k * upper)
        lo = np.sin(k * lower) - k * lower * np.cos(k * lower)
        out[nonzero] = (hi - lo) / (k**3 * shell)
    return out


def piecewise_constant_pk_xi_basis(separation: np.ndarray, theory_edges: np.ndarray) -> np.ndarray:
    """每个 theory P(k) band=1 时产生的 xi(s) basis。

    精确计算 ``(2pi^2)^-1 integral_[klo,khi] k^2 j0(ks) dk``，输出 shape
    ``(nseparation, nband)``。当前 jaxpower window 正是 piecewise-constant
    theory-vector 口径，因此无需再猜额外 dk 权重。
    """
    separation = np.asarray(separation, dtype="f8")[:, None]
    edges = np.asarray(theory_edges, dtype="f8")
    klo = edges[:, 0][None, :]
    khi = edges[:, 1][None, :]
    out = np.empty((separation.shape[0], edges.shape[0]), dtype="f8")
    zero = separation[:, 0] == 0.0
    if np.any(zero):
        out[zero] = (khi**3 - klo**3) / (6.0 * np.pi**2)
    nz = ~zero
    if np.any(nz):
        s = separation[nz]
        hi = np.sin(khi * s) - khi * s * np.cos(khi * s)
        lo = np.sin(klo * s) - klo * s * np.cos(klo * s)
        out[nz] = (hi - lo) / (2.0 * np.pi**2 * s**3)
    return out


def aggregate_joint_to_target_s(
    joint_probability: np.ndarray,
    kernel_edges: np.ndarray,
    target_edges: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """按实际 2PCF s edges 聚合 joint probability，并执行 RR 除法。"""
    joint = np.asarray(joint_probability, dtype="f8")
    kedges = np.asarray(kernel_edges, dtype="f8")
    tedges = np.asarray(target_edges, dtype="f8")
    centers = 0.5 * (kedges[:-1] + kedges[1:])
    rows = []
    norms = []
    for lo, hi in zip(tedges[:-1], tedges[1:], strict=True):
        mask = (centers >= lo) & (centers < hi)
        raw = np.sum(joint[mask], axis=0)
        norm = float(np.sum(raw))
        if norm <= 0.0:
            raise RuntimeError(f"target s bin [{lo}, {hi}) 没有 random outer pair")
        rows.append(raw / norm)
        norms.append(norm)
    return np.asarray(rows, dtype="f8"), np.asarray(norms, dtype="f8")


def build_discrete_xi_response(
    *,
    joint_probability: np.ndarray,
    kernel_edges: np.ndarray,
    target_s_edges: np.ndarray,
    k_eff: np.ndarray,
    batch_size: int = 512,
) -> dict[str, np.ndarray]:
    """构造 FullDiscrete P(k_eff) 到 2PCF RIC correction 的无量纲 j0 response。

    返回 ``response`` shape 为 ``(nk_eff, n_s_target)``；实际 correction 是
    ``sum_q g_q P_q response[q,s] / V_box``，与当前 FullDiscrete 主模型完全
    相同的 degeneracy 与 mother-box volume 归一化。
    """
    conditional, rr_probability = aggregate_joint_to_target_s(joint_probability, kernel_edges, target_s_edges)
    delta = 0.5 * (np.asarray(kernel_edges[:-1]) + np.asarray(kernel_edges[1:]))
    k_eff = np.asarray(k_eff, dtype="f8")
    response = np.empty((k_eff.size, conditional.shape[0]), dtype="f8")
    for start in range(0, k_eff.size, int(batch_size)):
        stop = min(k_eff.size, start + int(batch_size))
        # FullDiscrete k_eff 延伸到 5 h/Mpc；若在 2--4 Mpc/h 的 Delta-bin
        # center 直接评估 j0，会给高 k 引入严重 alias。论文的 random-count
        # window 是 counts/壳体积，因此这里与当前 2PCF 主线一样使用解析
        # volume-shell average，global limit 才能稳定回到旧 W2 结果。
        j0 = shell_average_j0(
            k_eff[start:stop],
            np.asarray(kernel_edges[:-1], dtype="f8"),
            np.asarray(kernel_edges[1:], dtype="f8"),
        )
        response[start:stop] = j0 @ conditional.T
    return {
        "response": response,
        "conditional_delta_given_target_s": conditional,
        "target_rr_probability": rr_probability,
        "delta_centers": delta,
    }


def build_pk_ric_matrix(
    *,
    joint_probability: np.ndarray,
    kernel_edges: np.ndarray,
    observed_k_edges: np.ndarray,
    theory_edges_ell0: np.ndarray,
    pair_volume: float,
) -> dict[str, np.ndarray]:
    """从同一 radial joint kernel 构造 P(k) single-term RIC matrix。

    ``direct_matrix`` 实现
    ``V_pair E[j0(k_obs s_outer) xi_band(Delta_inner)]``。
    ``geometry_mc_matrix`` 把 Delta 换回 s_outer，用于和当前 jaxpower
    geometry window 的 ell=0 block 做独立 normalization/reconstruction 审计。
    """
    joint = np.asarray(joint_probability, dtype="f8")
    centers = 0.5 * (np.asarray(kernel_edges[:-1]) + np.asarray(kernel_edges[1:]))
    observed_edges = np.asarray(observed_k_edges, dtype="f8")
    # shell_average_j0(s, klo, khi) 的输出是 (ns,nkobs)，转置后得到 (nkobs,ns)。
    j0_observed = shell_average_j0(centers, observed_edges[:, 0], observed_edges[:, 1]).T
    xi_band = piecewise_constant_pk_xi_basis(centers, theory_edges_ell0)
    outer_probability = np.sum(joint, axis=1)
    direct = float(pair_volume) * ((j0_observed @ joint) @ xi_band)
    geometry_mc = float(pair_volume) * ((j0_observed * outer_probability[None, :]) @ xi_band)
    # 先按 s 聚合 C_rad(s)，再做外层 Fourier transform；它应和 direct 完全闭合。
    conditional = np.zeros_like(joint)
    nonzero = outer_probability > 0.0
    conditional[nonzero] = joint[nonzero] / outer_probability[nonzero, None]
    c_band_of_s = conditional @ xi_band
    hankel_via_c = float(pair_volume) * ((j0_observed * outer_probability[None, :]) @ c_band_of_s)
    return {
        "direct_matrix": direct,
        "hankel_via_c_matrix": hankel_via_c,
        "geometry_mc_matrix": geometry_mc,
        "j0_observed": j0_observed,
        "xi_band": xi_band,
        "outer_probability": outer_probability,
    }


def global_factorised_joint(joint_probability: np.ndarray) -> np.ndarray:
    """把所有 radial bins 合并为一个 bin 时的解析 factorised joint。"""
    joint = np.asarray(joint_probability, dtype="f8")
    outer = np.sum(joint, axis=1)
    outer /= np.sum(outer)
    # 合并为一个 global bin 后，inner 与 outer 都从同一个 W(x) 独立抽样；
    # 两者的 separation marginal 因而严格相同。使用精确 pycorr outer
    # marginal，而不是有限 quadruplet MC 的 noisy inner marginal。
    return outer[:, None] * outer[None, :]


def covariance_sigma(path: Path, *, key: str = "covariance_single_realization") -> np.ndarray:
    """读取固定 covariance 的逐 bin 1-sigma，供 convergence 阈值审计。"""
    with np.load(path, allow_pickle=False) as data:
        if key not in data.files:
            raise KeyError(f"{path} 没有 covariance key={key}")
        covariance = np.asarray(data[key], dtype="f8")
    return np.sqrt(np.diag(covariance))


def positive_octant_cosine_cdf(*, sobol_power: int, histogram_bins: int, seed: int) -> dict[str, np.ndarray]:
    """用 Sobol QMC 计算两个独立正八分体方向夹角 cosine 的 CDF。

    Task4.3 random 的角向生成器是 ``abs(N(0,1)^3)`` 再归一化，即严格均匀
    正八分体。这里用等价参数化 ``z~U(0,1), phi~U(0,pi/2)``，一次高精度
    角向积分即可被所有 radial-bin pair、所有 phase 和 1/2/4 Mpc/h 设置复用。
    """
    from scipy.stats import qmc

    if int(sobol_power) < 10:
        raise ValueError("sobol_power 太小，无法作为正式角向 kernel")
    engine = qmc.Sobol(d=4, scramble=True, seed=int(seed))
    points = engine.random_base2(m=int(sobol_power))
    z1, phi1, z2, phi2 = points.T
    phi1 = 0.5 * np.pi * phi1
    phi2 = 0.5 * np.pi * phi2
    rho1 = np.sqrt(np.maximum(0.0, 1.0 - z1 * z1))
    rho2 = np.sqrt(np.maximum(0.0, 1.0 - z2 * z2))
    cosine = z1 * z2 + rho1 * rho2 * np.cos(phi1 - phi2)
    cosine = np.clip(cosine, 0.0, 1.0)
    edges = np.linspace(0.0, 1.0, int(histogram_bins) + 1, dtype="f8")
    counts = np.histogram(cosine, bins=edges)[0]
    cdf = np.concatenate([[0.0], np.cumsum(counts, dtype="f8") / float(cosine.size)])
    return {
        "cosine_edges": edges,
        "cosine_cdf": cdf,
        "sample_size": np.asarray(cosine.size, dtype="i8"),
        "cosine_mean": np.asarray(float(np.mean(cosine)), dtype="f8"),
        "cosine_std": np.asarray(float(np.std(cosine)), dtype="f8"),
    }


def radial_statistics(random: RandomSubsample, width: float) -> dict[str, np.ndarray]:
    """计算 binned RIC 所需的 radial probability 与每 bin weighted mean chi。"""
    edges = radial_bin_edges(random.chi, float(width))
    labels = np.searchsorted(edges, random.chi, side="right") - 1
    labels = np.clip(labels, 0, edges.size - 2)
    nbin = edges.size - 1
    weight_sum = np.bincount(labels, weights=random.weight, minlength=nbin).astype("f8")
    weighted_chi = np.bincount(labels, weights=random.weight * random.chi, minlength=nbin).astype("f8")
    count = np.bincount(labels, minlength=nbin).astype("i8")
    keep = weight_sum > 0.0
    probability = weight_sum[keep] / np.sum(weight_sum[keep])
    mean_chi = weighted_chi[keep] / weight_sum[keep]
    return {
        "radial_edges": edges,
        "radial_bin_index": np.flatnonzero(keep).astype("i8"),
        "radial_probability": probability,
        "radial_mean_chi": mean_chi,
        "radial_random_count": count[keep],
        "radial_weight_sum": weight_sum[keep],
    }


def radial_pair_components(probability: np.ndarray, mean_chi: np.ndarray) -> dict[str, np.ndarray]:
    """把有序 radial-bin pair 压缩成 a<=b components。

    a<b 的 component 权重为 ``2 p_a p_b``，a=b 为 ``p_a^2``，因此全部
    component 权重严格和为 1，且保留外层/内层 pair 的完整有序积分。
    """
    probability = np.asarray(probability, dtype="f8")
    mean_chi = np.asarray(mean_chi, dtype="f8")
    ia, ib = np.triu_indices(probability.size)
    weight = probability[ia] * probability[ib]
    weight[ia != ib] *= 2.0
    weight /= np.sum(weight)
    return {
        "component_a": ia.astype("i8"),
        "component_b": ib.astype("i8"),
        "component_weight": weight,
        "component_r1": mean_chi[ia],
        "component_r2": mean_chi[ib],
    }


def component_separation_probability(
    r1: np.ndarray,
    r2: np.ndarray,
    *,
    separation_edges: np.ndarray,
    cosine_edges: np.ndarray,
    cosine_cdf: np.ndarray,
) -> np.ndarray:
    """由正八分体 cosine CDF 解析映射出每个 radial pair 的 p(s-bin)。

    返回 shape ``(ncomponent_batch, nseparation)``。radial bin 内按论文
    binned approximation 忽略选择函数变化，使用该 bin 的 weighted mean chi。
    """
    r1 = np.asarray(r1, dtype="f8")[:, None]
    r2 = np.asarray(r2, dtype="f8")[:, None]
    sedges = np.asarray(separation_edges, dtype="f8")[None, :]
    denominator = 2.0 * r1 * r2
    cosine_at_s = (r1 * r1 + r2 * r2 - sedges * sedges) / denominator
    cdf_at_s = np.interp(
        np.clip(cosine_at_s, 0.0, 1.0).ravel(),
        np.asarray(cosine_edges, dtype="f8"),
        np.asarray(cosine_cdf, dtype="f8"),
    ).reshape(cosine_at_s.shape)
    cdf_at_s[cosine_at_s <= 0.0] = 0.0
    cdf_at_s[cosine_at_s >= 1.0] = 1.0
    # s 增大时 cos(s) 减小；P(slo<=s<shi)=CDF(cos(slo))-CDF(cos(shi))。
    probability = np.maximum(0.0, cdf_at_s[:, :-1] - cdf_at_s[:, 1:])
    norm = np.sum(probability, axis=1)
    valid = norm > 0.0
    probability[valid] /= norm[valid, None]
    if not np.all(valid):
        raise RuntimeError("某些 radial pair 在 separation grid 内没有 coverage")
    return probability


def factorized_outer_model_probability(cache: dict[str, Any], *, batch_size: int = 256) -> np.ndarray:
    """从 factorized cache 计算未 reweight 的解析 outer p_model(s)。"""
    weight = np.asarray(cache["component_weight"], dtype="f8")
    r1 = np.asarray(cache["component_r1"], dtype="f8")
    r2 = np.asarray(cache["component_r2"], dtype="f8")
    out = np.zeros(np.asarray(cache["separation_edges"]).size - 1, dtype="f8")
    for start in range(0, weight.size, int(batch_size)):
        stop = min(weight.size, start + int(batch_size))
        h = component_separation_probability(
            r1[start:stop],
            r2[start:stop],
            separation_edges=cache["separation_edges"],
            cosine_edges=cache["cosine_edges"],
            cosine_cdf=cache["cosine_cdf"],
        )
        out += weight[start:stop] @ h
    out /= np.sum(out)
    return out
