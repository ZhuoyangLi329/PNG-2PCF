#!/usr/bin/env python3
"""绘制 Task43 前 7 个 EZmock lightcone 与 25 个 Abacus lightcone 的临时均值对比。

代码结构
--------
1. 只读取已经完成并标记为 full-input 的 EZmock ph100--ph106 FCFC 2PCF。
2. 2PCF 暂与既有 25-phase Abacus cucount 汇总比较；ph000 的独立全量
   FCFC--cucount bridge 必须通过后才允许画图。
3. 读取同一 7 个 EZmock 的 jaxpower survey P(k)，按 Abacus payload 的 13 个
   DESI-PNG bin edge 精确配对，禁止插值。
4. 保存可复核的 NPZ/JSON 与唯一 PDF；7-realization scatter 仅作诊断，绝不
   作为 covariance 产品。
"""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from task43_ezmock_lightcone_common import (
    ABACUS_PK_PAYLOAD,
    ABACUS_XI_CUCOUNT_SUMMARY,
    FIX_AMPLITUDE,
    MANIFEST,
    OUTPUT_ROOT,
    PDF_BASE,
    PK_TAG,
    PLOT_DIR,
    RHO_C,
    RHO_EXP,
    SIGMA_V,
    SUMMARY_DIR,
    atomic_savez,
    read_jsonl,
    write_json,
)
from task43_measure_lightcone_xi_fcfc import ABACUS_XI_DIR
from task43_pk_common import PK_MEASURE_DIR


# 这是用户要求的中间检查，不替代最终 10-realization 汇总。
N_EZMOCK = 7
STEM = "task43_ezmock_fixedamp_x7_vs_abacus_x25_lightcone_mean_2pcf_pk_interim"
OUTPUT_PDF = PLOT_DIR / f"{STEM}.pdf"
OUTPUT_NPZ = SUMMARY_DIR / f"{STEM}.npz"
OUTPUT_JSON = SUMMARY_DIR / f"{STEM}.json"
RAWBOX_SUMMARY = (
    OUTPUT_ROOT.parent
    / "ezmock_rawbox_z0p725_mmin1p4e13/summary"
    / "task43_ezmock_rawbox_z0p725_mmin1p4e13_x25.npz"
)


def sha256(path: Path) -> str:
    """返回文件 SHA256，便于追踪中间图对应的确定输入。"""
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def atomic_savefig(figure: plt.Figure, path: Path) -> None:
    """遵循项目绘图策略，只原子保存 PDF，不生成 PNG。"""
    if path.suffix.lower() != ".pdf":
        raise ValueError("Task43 scientific plots must be PDF")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.stem}.{os.getpid()}.tmp.pdf")
    figure.savefig(temporary, bbox_inches="tight")
    temporary.replace(path)


def match_edges(source: np.ndarray, target: np.ndarray) -> np.ndarray:
    """逐 bin 精确匹配左右边界，明确禁止最近邻匹配或数据插值。"""
    indices: list[int] = []
    for edge in np.asarray(target, dtype="f8"):
        hits = np.flatnonzero(
            np.all(np.isclose(source, edge, rtol=0.0, atol=1.0e-12), axis=1)
        )
        if hits.size != 1:
            raise ValueError(f"target k edge does not have one exact match: {edge}")
        indices.append(int(hits[0]))
    return np.asarray(indices, dtype="i8")


def main() -> None:
    """完成 x7 临时汇总、强口径检查和 PDF 绘制。"""
    rows = read_jsonl(MANIFEST)[:N_EZMOCK]
    if len(rows) != N_EZMOCK or not FIX_AMPLITUDE:
        raise ValueError("interim plot requires exactly seven FIX_AMPLITUDE=T rows")
    expected_phases = [f"ph{index:03d}" for index in range(100, 107)]
    if [row["phase"] for row in rows] != expected_phases:
        raise ValueError("unexpected EZmock phase labels for x7 interim plot")

    # 读取 7 份正式 FCFC 结果，并强制统一 rlist、全量标记和有限性。
    xi_rows, ndata_rows, nrandom_rows, zeff_rows = [], [], [], []
    s = s_edges = None
    xi_paths: list[str] = []
    for row in rows:
        path = Path(row["xi_path"])
        metadata = json.loads(path.with_suffix(".json").read_text(encoding="utf-8"))
        if metadata.get("classification") != "full_input_validation_measurement":
            raise ValueError(f"not a full-input FCFC result: {path}")
        with np.load(path, allow_pickle=False) as data:
            this_s = np.asarray(data["s"], dtype="f8")
            this_edges = np.asarray(data["s_edges"], dtype="f8")
            xi = np.asarray(data["xi0"], dtype="f8")
            if bool(np.asarray(data["debug_subsample"]).item()):
                raise ValueError(f"debug FCFC result is forbidden: {path}")
            if not np.all(np.isfinite(xi)) or not np.all(np.asarray(data["RR"]) > 0):
                raise ValueError(f"invalid xi or RR: {path}")
            if s is None:
                s, s_edges = this_s, this_edges
            elif not np.array_equal(this_s, s) or not np.array_equal(this_edges, s_edges):
                raise ValueError("EZmock FCFC rlist mismatch")
            xi_rows.append(xi)
            ndata_rows.append(int(np.asarray(data["ndata"]).item()))
            nrandom_rows.append(int(np.asarray(data["nrandom"]).item()))
            zeff_rows.append(float(np.asarray(data["zeff"]).item()))
        xi_paths.append(str(path))
    xi_ez = np.vstack(xi_rows)

    # 临时图使用既有 25-phase cucount 平均；独立 ph000 bridge 是必要的安全门。
    with np.load(ABACUS_XI_CUCOUNT_SUMMARY, allow_pickle=False) as data:
        s_ab = np.asarray(data["s"], dtype="f8")
        xi_ab = np.asarray(data["xi0_all"], dtype="f8")
        ndata_ab = np.asarray(data["ndata"], dtype="i8")
        zeff_ab = np.asarray(data["zeff_all"], dtype="f8")
    if xi_ab.shape != (25, 50) or not np.array_equal(s_ab, s):
        raise ValueError("unexpected Abacus x25 cucount payload")

    # 额外加载质量阈值相同的 25 个 Abacus periodic raw boxes。它们没有
    # survey FKP/window，但与 lightcone 的差异也混合了 snapshot/redshift
    # evolution 和 periodic/survey geometry，图中因此只标作 rawbox overlay。
    with np.load(RAWBOX_SUMMARY, allow_pickle=False) as data:
        rawbox_phases = np.asarray(data["phases"])
        s_rawbox = np.asarray(data["s"], dtype="f8")
        xi_rawbox = np.asarray(data["xi0_all"], dtype="f8")
        k_rawbox_fine = np.asarray(data["k"], dtype="f8")
        k_edges_rawbox_fine = np.asarray(data["k_edges"], dtype="f8")
        k_valid_rawbox = np.asarray(data["k_valid_mask"], dtype=bool)
        pk_rawbox_fine = np.asarray(data["pk0_all"], dtype="f8")
        rawbox_redshift = np.asarray(data["redshift"], dtype="f8")
    if (
        rawbox_phases.size != 25
        or xi_rawbox.shape != (25, 50)
        or pk_rawbox_fine.shape != (25, 150)
        or not np.array_equal(s_rawbox, s)
    ):
        raise ValueError("unexpected Abacus rawbox x25 payload")
    bridge_path = ABACUS_XI_DIR / (
        "xi0_AbacusSummit_base_c000_ph000_z0p6_0p8_mmin1p4e13_"
        "x25_s50_550_ds10_fcfc.npz"
    )
    with np.load(bridge_path, allow_pickle=False) as data:
        bridge_xi = np.asarray(data["xi0"], dtype="f8")
    xi_ab_std = np.std(xi_ab, axis=0, ddof=1)
    bridge_delta = bridge_xi - xi_ab[0]
    bridge_max_sigma = float(np.max(np.abs(bridge_delta) / xi_ab_std))
    if bridge_max_sigma >= 0.05:
        raise RuntimeError(f"FCFC--cucount bridge failed: {bridge_max_sigma}")

    # 读取 7 份 survey P(k)，严格映射到 Abacus 的同一 13 个 observed bins。
    with np.load(ABACUS_PK_PAYLOAD, allow_pickle=False) as data:
        k_ab = np.asarray(data["k_obs"], dtype="f8")
        k_edges_ab = np.asarray(data["k_edges"], dtype="f8")
        pk_ab = np.asarray(data["pk_stack"], dtype="f8")
    if pk_ab.shape != (25, 13):
        raise ValueError("unexpected Abacus P(k) stack shape")
    rawbox_selected_indices = match_edges(k_edges_rawbox_fine, k_edges_ab)
    if not np.all(k_valid_rawbox[rawbox_selected_indices]):
        raise ValueError("one or more selected rawbox P(k) bins are invalid")
    k_rawbox = k_rawbox_fine[rawbox_selected_indices]
    pk_rawbox = pk_rawbox_fine[:, rawbox_selected_indices]
    if not np.all(np.isfinite(k_rawbox)) or not np.all(np.isfinite(pk_rawbox)):
        raise ValueError("non-finite selected rawbox P(k)")
    pk_ez_rows, k_ez_rows, pk_paths = [], [], []
    selected_indices = None
    for row in rows:
        path = PK_MEASURE_DIR / PK_TAG / (
            f"task43_pk_{row['phase']}_mesh256_kmax0p300_dk0p002.npz"
        )
        with np.load(path, allow_pickle=False) as data:
            indices = match_edges(np.asarray(data["k_edges"], dtype="f8"), k_edges_ab)
            if selected_indices is None:
                selected_indices = indices
            elif not np.array_equal(indices, selected_indices):
                raise ValueError("selected P(k) indices differ between EZmock realizations")
            pk = np.asarray(data["pk0"], dtype="f8")[indices]
            kval = np.asarray(data["k_obs"], dtype="f8")[indices]
            if not np.all(np.isfinite(pk)):
                raise ValueError(f"non-finite P(k): {path}")
            pk_ez_rows.append(pk)
            k_ez_rows.append(kval)
        pk_paths.append(str(path))
    pk_ez = np.vstack(pk_ez_rows)
    k_ez = np.vstack(k_ez_rows)

    # 这里的 std 是 single-realization scatter，仅用于观察调参后的均值与离散。
    xi_ez_mean, xi_ez_std = np.mean(xi_ez, axis=0), np.std(xi_ez, axis=0, ddof=1)
    xi_ab_mean = np.mean(xi_ab, axis=0)
    xi_rawbox_mean = np.mean(xi_rawbox, axis=0)
    pk_ez_mean, pk_ez_std = np.mean(pk_ez, axis=0), np.std(pk_ez, axis=0, ddof=1)
    pk_ab_mean, pk_ab_std = np.mean(pk_ab, axis=0), np.std(pk_ab, axis=0, ddof=1)
    pk_rawbox_mean = np.mean(pk_rawbox, axis=0)
    xi_residual_sigma = (xi_ez_mean - xi_ab_mean) / xi_ab_std
    xi_rawbox_residual_sigma = (xi_rawbox_mean - xi_ab_mean) / xi_ab_std
    pk_fractional_residual = pk_ez_mean / pk_ab_mean - 1.0
    pk_rawbox_fractional_residual = pk_rawbox_mean / pk_ab_mean - 1.0

    metrics = {
        "xi_delta_over_abacus_scatter_rms": float(np.sqrt(np.mean(xi_residual_sigma**2))),
        "xi_delta_over_abacus_scatter_max_abs": float(np.max(np.abs(xi_residual_sigma))),
        "xi_rawbox_minus_lightcone_over_abacus_scatter_rms": float(
            np.sqrt(np.mean(xi_rawbox_residual_sigma**2))
        ),
        "xi_rawbox_minus_lightcone_over_abacus_scatter_max_abs": float(
            np.max(np.abs(xi_rawbox_residual_sigma))
        ),
        "pk_fractional_delta_rms_13bins": float(np.sqrt(np.mean(pk_fractional_residual**2))),
        "pk_fractional_delta_max_abs_13bins": float(np.max(np.abs(pk_fractional_residual))),
        "pk_rawbox_over_lightcone_fractional_delta_rms_13bins": float(
            np.sqrt(np.mean(pk_rawbox_fractional_residual**2))
        ),
        "pk_rawbox_over_lightcone_fractional_delta_max_abs_13bins": float(
            np.max(np.abs(pk_rawbox_fractional_residual))
        ),
        "ndata_ezmock_mean": float(np.mean(ndata_rows)),
        "ndata_abacus_mean": float(np.mean(ndata_ab)),
        "ndata_fractional_difference": float(np.mean(ndata_rows) / np.mean(ndata_ab) - 1.0),
        "zeff_ezmock_mean": float(np.mean(zeff_rows)),
        "zeff_abacus_mean": float(np.mean(zeff_ab)),
        "fcfc_cucount_bridge_ph000_max_delta_over_abacus_scatter": bridge_max_sigma,
        "rawbox_redshift_mean": float(np.mean(rawbox_redshift)),
    }

    atomic_savez(
        OUTPUT_NPZ,
        ezmock_phases=np.asarray(expected_phases),
        s=s,
        s_edges=s_edges,
        xi_ezmock_all=xi_ez,
        xi_ezmock_mean=xi_ez_mean,
        xi_ezmock_std=xi_ez_std,
        xi_abacus_all=xi_ab,
        xi_abacus_mean=xi_ab_mean,
        xi_abacus_std=xi_ab_std,
        xi_residual_over_abacus_std=xi_residual_sigma,
        xi_rawbox_all=xi_rawbox,
        xi_rawbox_mean=xi_rawbox_mean,
        xi_rawbox_minus_lightcone_over_abacus_std=xi_rawbox_residual_sigma,
        k_abacus=k_ab,
        k_ezmock_all=k_ez,
        k_rawbox=k_rawbox,
        k_edges=k_edges_ab,
        selected_fine_bin_indices=selected_indices,
        rawbox_selected_fine_bin_indices=rawbox_selected_indices,
        pk_ezmock_all=pk_ez,
        pk_ezmock_mean=pk_ez_mean,
        pk_ezmock_std=pk_ez_std,
        pk_abacus_all=pk_ab,
        pk_abacus_mean=pk_ab_mean,
        pk_abacus_std=pk_ab_std,
        pk_fractional_residual=pk_fractional_residual,
        pk_rawbox_all=pk_rawbox,
        pk_rawbox_mean=pk_rawbox_mean,
        pk_rawbox_fractional_residual=pk_rawbox_fractional_residual,
        ndata_ezmock=np.asarray(ndata_rows, dtype="i8"),
        nrandom_ezmock=np.asarray(nrandom_rows, dtype="i8"),
        zeff_ezmock=np.asarray(zeff_rows, dtype="f8"),
        fix_amplitude=np.asarray(True),
    )

    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.size": 11,
            "axes.labelsize": 13,
            "legend.fontsize": 9.5,
            "xtick.direction": "in",
            "ytick.direction": "in",
            "xtick.top": True,
            "ytick.right": True,
        }
    )
    figure, axes = plt.subplots(2, 2, figsize=(12.0, 8.3), constrained_layout=True)

    ax = axes[0, 0]
    ax.fill_between(s, s**2 * (xi_ab_mean - xi_ab_std), s**2 * (xi_ab_mean + xi_ab_std), color="#4C78A8", alpha=0.20, lw=0)
    ax.plot(s, s**2 * xi_ab_mean, color="#2F5D8A", lw=2.1, label="Abacus halo lightcone x25")
    ax.fill_between(s, s**2 * (xi_ez_mean - xi_ez_std), s**2 * (xi_ez_mean + xi_ez_std), color="#E07A2D", alpha=0.18, lw=0)
    ax.plot(s, s**2 * xi_ez_mean, color="#D45D00", lw=2.0, label="EZmock lightcone x7 (fixed amplitude)")
    ax.plot(
        s,
        s**2 * xi_rawbox_mean,
        color="#7A5195",
        lw=1.9,
        ls="--",
        label=r"Abacus periodic rawbox x25 ($z\simeq0.726$)",
    )
    ax.axhline(0.0, color="0.35", lw=0.8, ls="--")
    ax.set_xlim(50.0, 550.0)
    ax.set_ylabel(r"$s^2\xi_0(s)\;[(h^{-1}{\rm Mpc})^2]$")
    ax.legend(frameon=False)
    ax.set_title("2PCF mean and single-realization scatter")

    ax = axes[1, 0]
    ax.plot(s, xi_residual_sigma, color="#8E3B20", marker="o", ms=3.0, lw=1.25, label="EZ lightcone - Abacus lightcone")
    ax.plot(s, xi_rawbox_residual_sigma, color="#7A5195", lw=1.35, ls="--", label="Abacus rawbox - lightcone")
    ax.axhline(0.0, color="0.3", lw=0.8)
    ax.axhline(1.0, color="0.6", lw=0.65, ls=":")
    ax.axhline(-1.0, color="0.6", lw=0.65, ls=":")
    ax.set_xlim(50.0, 550.0)
    ax.set_xlabel(r"$s\;[h^{-1}{\rm Mpc}]$")
    ax.set_ylabel(r"$(\bar\xi_X-\bar\xi_{\rm Aba,LC})/\sigma_{\rm Aba,LC}$")
    ax.legend(frameon=False, fontsize=8.2, loc="lower left")

    ax = axes[0, 1]
    ax.fill_between(k_ab, pk_ab_mean - pk_ab_std, pk_ab_mean + pk_ab_std, color="#4C78A8", alpha=0.20, lw=0)
    ax.plot(k_ab, pk_ab_mean, color="#2F5D8A", marker="o", ms=3.5, lw=1.8, label="Abacus halo lightcone x25")
    k_ez_plot = np.mean(k_ez, axis=0)
    ax.fill_between(k_ez_plot, pk_ez_mean - pk_ez_std, pk_ez_mean + pk_ez_std, color="#59A14F", alpha=0.18, lw=0)
    ax.plot(k_ez_plot, pk_ez_mean, color="#2E7D32", marker="s", ms=3.2, lw=1.8, label="EZmock lightcone x7 (fixed amplitude)")
    ax.plot(
        k_rawbox,
        pk_rawbox_mean,
        color="#7A5195",
        marker="^",
        ms=3.2,
        lw=1.7,
        ls="--",
        label=r"Abacus periodic rawbox x25 ($z\simeq0.726$)",
    )
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlim(0.005, 0.08)
    ax.set_ylabel(r"$P_0(k)\;[(h^{-1}{\rm Mpc})^3]$")
    ax.legend(frameon=False)
    ax.set_title("jaxpower observed/windowed P(k)")

    ax = axes[1, 1]
    ax.plot(k_ab, pk_fractional_residual, color="#276749", marker="o", ms=3.2, lw=1.25, label="EZ lightcone / Abacus lightcone")
    ax.plot(k_ab, pk_rawbox_fractional_residual, color="#7A5195", marker="^", ms=3.0, lw=1.25, ls="--", label="Abacus rawbox / lightcone")
    ax.axhline(0.0, color="0.3", lw=0.8)
    ax.set_xscale("log")
    ax.set_xlim(0.005, 0.08)
    ax.set_xlabel(r"$k\;[h\,{\rm Mpc}^{-1}]$")
    ax.set_ylabel(r"$\bar P_X/\bar P_{\rm Aba,LC}-1$")
    ax.legend(frameon=False, fontsize=8.2, loc="best")

    figure.suptitle(
        rf"Task43 interim validation (7/10): $\rho_c={RHO_C:g}$, "
        rf"$\rho_{{\rm exp}}={RHO_EXP:g}$, $b={PDF_BASE:g}$, "
        rf"$\sigma_v={SIGMA_V:g}$; $P_0^{{\rm FKP}}=10000$",
        fontsize=13,
    )
    atomic_savefig(figure, OUTPUT_PDF)
    plt.close(figure)

    summary = {
        "task": "task43_plot_ezmock_lightcone_interim_x7",
        "status": "done",
        "classification": "interim fixed-amplitude mean validation; x7 scatter is diagnostic only, not covariance",
        "nreal_ezmock": N_EZMOCK,
        "nreal_abacus": 25,
        "ezmock_labels": expected_phases,
        "label_note": "ph100--ph106 are file labels for EZmock realizations 1--7, not physical Abacus phases",
        "fixed_amplitude": True,
        "future_covariance_policy": "production must use FIX_AMPLITUDE=F",
        "xi_abacus_source": "existing cucount x25 summary; ph000 full FCFC--cucount bridge required and passed",
        "rawbox_overlay": (
            "Abacus periodic rawbox x25 at mean z~0.726, no FKP/window; difference from lightcone "
            "also includes snapshot/redshift evolution and periodic/survey geometry, so it is not a pure window-only effect"
        ),
        "pk_comparison": "same 13 observed DESI-PNG bin edges; exact edge matching, no P(k) interpolation",
        "metrics": metrics,
        "inputs": {
            "manifest": str(MANIFEST),
            "ezmock_xi": xi_paths,
            "ezmock_pk": pk_paths,
            "abacus_xi": str(ABACUS_XI_CUCOUNT_SUMMARY),
            "abacus_pk": str(ABACUS_PK_PAYLOAD),
            "abacus_rawbox": str(RAWBOX_SUMMARY),
            "fcfc_cucount_bridge_ph000": str(bridge_path),
        },
        "outputs": {
            "pdf": str(OUTPUT_PDF),
            "npz": str(OUTPUT_NPZ),
            "json": str(OUTPUT_JSON),
        },
    }
    # PDF/NPZ 已完成后再记录哈希，保证 JSON 指向不可含糊的中间结果。
    summary["outputs"]["pdf_sha256"] = sha256(OUTPUT_PDF)
    summary["outputs"]["npz_sha256"] = sha256(OUTPUT_NPZ)
    write_json(OUTPUT_JSON, summary)
    print(f"[write] {OUTPUT_PDF}")
    print(f"[write] {OUTPUT_NPZ}")
    print(f"[write] {OUTPUT_JSON}")
    print(json.dumps(metrics, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
