#!/usr/bin/env python3
"""汇总 Task43 EZmock/Abacus lightcone 2PCF 与 P(k)，并生成唯一 PDF 对比图。

执行大纲
--------
1. 读取 10 个 EZmock 的 full-input FCFC 2PCF，以及用户批准复用的 25-phase
   Abacus cucount 2PCF 汇总；独立 ph000 full-input FCFC bridge 必须通过。
2. 读取 10 个 EZmock jaxpower survey P(k) 和冻结的 25-phase Abacus payload，
   严格按 Abacus 13 个 DESI-PNG k-bin edges 配对，不插值数据值。
3. 计算均值、single-realization scatter、数量/数密度/zeff 和 FCFC--cucount
   ph000 bridge 指标；10-realization scatter 只作诊断，不反演 covariance。
4. 原子写入 NPZ、JSON 和一个 PDF，不生成 PNG。
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np

from task43_ezmock_lightcone_common import (
    ABACUS_FKP_SUMMARY,
    ABACUS_PK_PAYLOAD,
    ABACUS_XI_CUCOUNT_SUMMARY,
    FIX_AMPLITUDE,
    MANIFEST,
    NREAL,
    OUTPUT_ROOT,
    PDF_BASE,
    PHASES,
    PK_TAG,
    PLOT_DIR,
    RHO_C,
    RHO_EXP,
    SEEDS,
    SIGMA_V,
    SUMMARY_DIR,
    atomic_savez,
    read_jsonl,
    write_json,
)
from task43_measure_lightcone_xi_fcfc import ABACUS_XI_DIR
from task43_pk_common import PK_MEASURE_DIR


DEFAULT_PREFIX = PLOT_DIR / "task43_ezmock_fixedamp_x10_vs_abacus_x25_lightcone_mean_2pcf_pk"
DEFAULT_SUMMARY = SUMMARY_DIR / "task43_ezmock_fixedamp_x10_vs_abacus_x25_lightcone_mean_2pcf_pk"


def parse_args() -> argparse.Namespace:
    """解析输入 manifest 和输出前缀。"""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=MANIFEST)
    parser.add_argument("--output-prefix", type=Path, default=DEFAULT_PREFIX)
    parser.add_argument("--summary-prefix", type=Path, default=DEFAULT_SUMMARY)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def sha256(path: Path) -> str:
    """计算文件 SHA256，供最终 provenance 审计使用。"""
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def atomic_savefig(figure: plt.Figure, path: Path) -> None:
    """只以 PDF 原子保存科学图。"""
    if path.suffix.lower() != ".pdf":
        raise ValueError("Task43 plotting policy requires PDF output")
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.stem}.{os.getpid()}.tmp.pdf")
    figure.savefig(tmp, bbox_inches="tight")
    tmp.replace(path)


def load_xi(path: Path) -> dict[str, Any]:
    """读取并验证一份 full-input FCFC 2PCF。"""
    if not path.is_file() or not path.with_suffix(".json").is_file():
        raise FileNotFoundError(path)
    metadata = json.loads(path.with_suffix(".json").read_text(encoding="utf-8"))
    if metadata.get("classification") != "full_input_validation_measurement":
        raise ValueError(f"not a full-input FCFC measurement: {path}")
    with np.load(path, allow_pickle=False) as data:
        out = {
            "path": path,
            "s": np.asarray(data["s"], dtype="f8"),
            "s_edges": np.asarray(data["s_edges"], dtype="f8"),
            "xi0": np.asarray(data["xi0"], dtype="f8"),
            "ndata": int(np.asarray(data["ndata"]).item()),
            "nrandom": int(np.asarray(data["nrandom"]).item()),
            "zeff": float(np.asarray(data["zeff"]).item()),
            "metadata": metadata,
        }
    if not np.all(np.isfinite(out["xi0"])):
        raise ValueError(f"non-finite xi0 in {path}")
    return out


def ez_xi_path(row: dict[str, Any]) -> Path:
    """返回 EZmock manifest 行对应的正式 FCFC 输出。"""
    return Path(row["xi_path"])


def abacus_xi_path(index: int) -> Path:
    """返回 Abacus phase 对应的正式 FCFC 输出。"""
    phase = f"ph{index:03d}"
    return ABACUS_XI_DIR / (
        f"xi0_AbacusSummit_base_c000_{phase}_z0p6_0p8_mmin1p4e13_"
        "x25_s50_550_ds10_fcfc.npz"
    )


def load_pk(path: Path) -> dict[str, Any]:
    """读取一份 jaxpower survey P(k) 测量。"""
    if not path.is_file():
        raise FileNotFoundError(path)
    with np.load(path, allow_pickle=False) as data:
        out = {
            "path": path,
            "k": np.asarray(data["k_obs"], dtype="f8"),
            "k_edges": np.asarray(data["k_edges"], dtype="f8"),
            "pk0": np.asarray(data["pk0"], dtype="f8"),
            "shotnoise": np.asarray(data["shotnoise"], dtype="f8"),
            "ndata": int(np.asarray(data["data_n_used"]).item()),
            "nrandom": int(np.asarray(data["random_n_used"]).item()),
            "phase": str(np.asarray(data["phase"]).item()),
            "has_window": bool(np.asarray(data["has_window"]).item()),
            "window_shape": tuple(np.asarray(data["window_matrix"]).shape) if "window_matrix" in data.files else None,
        }
    return out


def match_edges(source_edges: np.ndarray, target_edges: np.ndarray) -> np.ndarray:
    """按完全相同的左右边界匹配 bin，禁止最近邻或插值。"""
    indices: list[int] = []
    for target in np.asarray(target_edges, dtype="f8"):
        hit = np.flatnonzero(np.all(np.isclose(source_edges, target, rtol=0.0, atol=1.0e-12), axis=1))
        if hit.size != 1:
            raise ValueError(f"could not uniquely match k edge {target}")
        indices.append(int(hit[0]))
    return np.asarray(indices, dtype="i8")


def stack_xi(items: list[dict[str, Any]]) -> dict[str, np.ndarray]:
    """验证统一 rlist 后堆叠 xi 与 metadata 标量。"""
    reference_s = items[0]["s"]
    reference_edges = items[0]["s_edges"]
    for item in items[1:]:
        if not np.array_equal(item["s"], reference_s) or not np.array_equal(item["s_edges"], reference_edges):
            raise ValueError("FCFC rlist/s_edges differ across realizations")
    return {
        "s": reference_s,
        "s_edges": reference_edges,
        "xi": np.vstack([item["xi0"] for item in items]),
        "ndata": np.asarray([item["ndata"] for item in items], dtype="i8"),
        "nrandom": np.asarray([item["nrandom"] for item in items], dtype="i8"),
        "zeff": np.asarray([item["zeff"] for item in items], dtype="f8"),
    }


def main() -> None:
    """完成全量汇总、bridge、图和机器审计。"""
    args = parse_args()
    output_pdf = args.output_prefix.with_suffix(".pdf")
    output_npz = args.summary_prefix.with_suffix(".npz")
    output_json = args.summary_prefix.with_suffix(".json")
    if all(path.exists() for path in (output_pdf, output_npz, output_json)) and not args.overwrite:
        print(f"[skip] {output_pdf}")
        return

    rows = read_jsonl(args.manifest)
    if len(rows) != NREAL or not all(bool(row["fix_amplitude"]) for row in rows):
        raise ValueError("summary requires exactly 10 FIX_AMPLITUDE=T EZmock rows")
    ez_xi_items = [load_xi(ez_xi_path(row)) for row in rows]
    ez_xi = stack_xi(ez_xi_items)
    with np.load(ABACUS_XI_CUCOUNT_SUMMARY, allow_pickle=False) as data:
        ab_xi = {
            "s": np.asarray(data["s"], dtype="f8"),
            "s_edges": np.asarray(data["s_edges"], dtype="f8"),
            "xi": np.asarray(data["xi0_all"], dtype="f8"),
            "ndata": np.asarray(data["ndata"], dtype="i8"),
            "nrandom": np.asarray(data["nrandom"], dtype="i8"),
            "zeff": np.asarray(data["zeff_all"], dtype="f8"),
        }
        abacus_phases = np.asarray(data["phases"])
        cucount_mean_stored = np.asarray(data["xi0_mean"], dtype="f8")
        cucount_std_stored = np.asarray(data["xi0_std"], dtype="f8")
    if (
        ab_xi["xi"].shape != (25, 50)
        or ab_xi["ndata"].shape != (25,)
        or ab_xi["nrandom"].shape != (25,)
        or ab_xi["zeff"].shape != (25,)
        or abacus_phases.shape != (25,)
        or not np.all(np.isfinite(ab_xi["xi"]))
    ):
        raise ValueError("unexpected Abacus x25 cucount payload")
    if not np.allclose(np.mean(ab_xi["xi"], axis=0), cucount_mean_stored):
        raise ValueError("stored Abacus cucount mean does not match xi0_all")
    if not np.allclose(np.std(ab_xi["xi"], axis=0, ddof=1), cucount_std_stored):
        raise ValueError("stored Abacus cucount scatter does not match xi0_all")
    if not np.array_equal(ez_xi["s"], ab_xi["s"]) or not np.array_equal(ez_xi["s_edges"], ab_xi["s_edges"]):
        raise ValueError("EZmock FCFC and Abacus cucount rlist differ")

    # 仅 ph000 进行一次真正的 full-input FCFC 重测，作为 25-phase cucount
    # 参考可复用的 estimator bridge；用户已明确批准不再重算余下 24 phases。
    abacus_bridge_item = load_xi(abacus_xi_path(0))
    if not np.array_equal(abacus_bridge_item["s"], ab_xi["s"]) or not np.array_equal(
        abacus_bridge_item["s_edges"], ab_xi["s_edges"]
    ):
        raise ValueError("Abacus ph000 FCFC--cucount bridge grid mismatch")

    with np.load(ABACUS_PK_PAYLOAD, allow_pickle=False) as data:
        k_ab = np.asarray(data["k_obs"], dtype="f8")
        k_edges_ab = np.asarray(data["k_edges"], dtype="f8")
        pk_ab = np.asarray(data["pk_stack"], dtype="f8")
        pk_ab_mean_stored = np.asarray(data["pk_mean"], dtype="f8")
    if pk_ab.shape != (25, 13) or not np.allclose(np.mean(pk_ab, axis=0), pk_ab_mean_stored):
        raise ValueError("unexpected Abacus 25x13 P(k) payload")

    ez_pk_items = [
        load_pk(PK_MEASURE_DIR / PK_TAG / f"task43_pk_{phase}_mesh256_kmax0p300_dk0p002.npz")
        for phase in PHASES
    ]
    selected_indices = match_edges(ez_pk_items[0]["k_edges"], k_edges_ab)
    pk_ez_rows, k_ez_rows, shot_rows = [], [], []
    for item in ez_pk_items:
        if not np.array_equal(item["k_edges"], ez_pk_items[0]["k_edges"]):
            raise ValueError("EZmock fine k_edges differ across realizations")
        pk_ez_rows.append(item["pk0"][selected_indices])
        k_ez_rows.append(item["k"][selected_indices])
        shot_rows.append(item["shotnoise"][selected_indices])
    pk_ez = np.vstack(pk_ez_rows)
    k_ez = np.vstack(k_ez_rows)
    shot_ez = np.vstack(shot_rows)
    if not np.all(np.isfinite(pk_ez)):
        raise ValueError("non-finite EZmock P(k) in selected 13 bins")
    if not ez_pk_items[0]["has_window"] or ez_pk_items[0]["window_shape"] is None:
        raise ValueError("ph100 EZmock P(k) lacks required smooth window audit")

    s = ez_xi["s"]
    xi_ez_mean, xi_ab_mean = np.mean(ez_xi["xi"], axis=0), np.mean(ab_xi["xi"], axis=0)
    xi_ez_std, xi_ab_std = np.std(ez_xi["xi"], axis=0, ddof=1), np.std(ab_xi["xi"], axis=0, ddof=1)
    pk_ez_mean, pk_ab_mean = np.mean(pk_ez, axis=0), np.mean(pk_ab, axis=0)
    pk_ez_std, pk_ab_std = np.std(pk_ez, axis=0, ddof=1), np.std(pk_ab, axis=0, ddof=1)
    xi_residual_sigma = (xi_ez_mean - xi_ab_mean) / xi_ab_std
    pk_fractional_residual = (pk_ez_mean - pk_ab_mean) / pk_ab_mean

    # 单独量化 ph000 full-input FCFC--cucount bridge，作为 estimator 口径证据。
    cucount_phase0 = ab_xi["xi"][0]
    cucount_std = np.std(ab_xi["xi"], axis=0, ddof=1)
    bridge_delta = abacus_bridge_item["xi0"] - cucount_phase0
    bridge = {
        "grid_exact": True,
        "max_abs_delta": float(np.max(np.abs(bridge_delta))),
        "rms_delta": float(np.sqrt(np.mean(bridge_delta**2))),
        "max_abs_delta_over_abacus_scatter": float(np.max(np.abs(bridge_delta) / cucount_std)),
        "rms_delta_over_abacus_scatter": float(np.sqrt(np.mean((bridge_delta / cucount_std) ** 2))),
        "pass_threshold": "max |FCFC-cucount| / sigma_Abacus < 0.05",
    }
    bridge["pass"] = bool(bridge["max_abs_delta_over_abacus_scatter"] < 0.05)
    if not bridge["pass"]:
        raise RuntimeError(f"FCFC--cucount bridge failed: {bridge}")

    with np.load(ABACUS_FKP_SUMMARY, allow_pickle=False) as data:
        volume = float(np.sum(np.asarray(data["volume_shell"], dtype="f8")))
    metrics = {
        "xi_delta_over_abacus_single_realization_scatter_rms": float(np.sqrt(np.mean(xi_residual_sigma**2))),
        "xi_delta_over_abacus_single_realization_scatter_max_abs": float(np.max(np.abs(xi_residual_sigma))),
        "pk_fractional_delta_rms_13bins": float(np.sqrt(np.mean(pk_fractional_residual**2))),
        "pk_fractional_delta_mean_13bins": float(np.mean(pk_fractional_residual)),
        "pk_fractional_delta_max_abs_13bins": float(np.max(np.abs(pk_fractional_residual))),
        "ndata_ezmock_mean": float(np.mean(ez_xi["ndata"])),
        "ndata_abacus_mean": float(np.mean(ab_xi["ndata"])),
        "ndata_fractional_difference": float(np.mean(ez_xi["ndata"]) / np.mean(ab_xi["ndata"]) - 1.0),
        "nbar_ezmock_mean": float(np.mean(ez_xi["ndata"]) / volume),
        "nbar_abacus_mean": float(np.mean(ab_xi["ndata"]) / volume),
        "zeff_ezmock_mean": float(np.mean(ez_xi["zeff"])),
        "zeff_abacus_mean": float(np.mean(ab_xi["zeff"])),
    }

    atomic_savez(
        output_npz,
        seeds=np.asarray(SEEDS, dtype="i8"),
        ezmock_phases=np.asarray(PHASES),
        abacus_phases=abacus_phases,
        s=s,
        s_edges=ez_xi["s_edges"],
        xi_ezmock_all=ez_xi["xi"],
        xi_ezmock_mean=xi_ez_mean,
        xi_ezmock_std=xi_ez_std,
        xi_abacus_all=ab_xi["xi"],
        xi_abacus_mean=xi_ab_mean,
        xi_abacus_std=xi_ab_std,
        xi_residual_over_abacus_std=xi_residual_sigma,
        k_abacus=k_ab,
        k_ezmock_all=k_ez,
        k_edges=k_edges_ab,
        pk_ezmock_all=pk_ez,
        pk_ezmock_mean=pk_ez_mean,
        pk_ezmock_std=pk_ez_std,
        pk_abacus_all=pk_ab,
        pk_abacus_mean=pk_ab_mean,
        pk_abacus_std=pk_ab_std,
        pk_fractional_residual=pk_fractional_residual,
        shotnoise_ezmock_all=shot_ez,
        ndata_ezmock=ez_xi["ndata"],
        ndata_abacus=ab_xi["ndata"],
        nrandom_ezmock=ez_xi["nrandom"],
        nrandom_abacus=ab_xi["nrandom"],
        zeff_ezmock=ez_xi["zeff"],
        zeff_abacus=ab_xi["zeff"],
        fix_amplitude=np.asarray(FIX_AMPLITUDE),
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
    ax.plot(s, s**2 * xi_ab_mean, color="#2F5D8A", lw=2.1, label="Abacus halo lightcone x25 (cucount)")
    ax.fill_between(s, s**2 * (xi_ez_mean - xi_ez_std), s**2 * (xi_ez_mean + xi_ez_std), color="#E07A2D", alpha=0.18, lw=0)
    ax.plot(s, s**2 * xi_ez_mean, color="#D45D00", lw=2.0, label="EZmock lightcone x10 (fixed amplitude)")
    ax.axhline(0.0, color="0.35", lw=0.8, ls="--")
    ax.set_xlim(50.0, 550.0)
    ax.set_ylabel(r"$s^2\xi_0(s)\;[(h^{-1}{\rm Mpc})^2]$")
    ax.legend(frameon=False)
    ax.set_title("Weighted Landy--Szalay; ph000 FCFC--cucount bridge passed")

    ax = axes[1, 0]
    ax.plot(s, xi_residual_sigma, color="#8E3B20", marker="o", ms=3.0, lw=1.25)
    ax.axhline(0.0, color="0.3", lw=0.8)
    ax.axhline(1.0, color="0.6", lw=0.65, ls=":")
    ax.axhline(-1.0, color="0.6", lw=0.65, ls=":")
    ax.set_xlim(50.0, 550.0)
    ax.set_xlabel(r"$s\;[h^{-1}{\rm Mpc}]$")
    ax.set_ylabel(r"$(\bar\xi_{\rm EZ}-\bar\xi_{\rm Aba})/\sigma_{\rm Aba}$")

    ax = axes[0, 1]
    ax.fill_between(k_ab, pk_ab_mean - pk_ab_std, pk_ab_mean + pk_ab_std, color="#4C78A8", alpha=0.20, lw=0)
    ax.plot(k_ab, pk_ab_mean, color="#2F5D8A", marker="o", ms=3.5, lw=1.8, label="Abacus halo lightcone x25")
    k_ez_plot = np.mean(k_ez, axis=0)
    ax.fill_between(k_ez_plot, pk_ez_mean - pk_ez_std, pk_ez_mean + pk_ez_std, color="#59A14F", alpha=0.18, lw=0)
    ax.plot(k_ez_plot, pk_ez_mean, color="#2E7D32", marker="s", ms=3.2, lw=1.8, label="EZmock lightcone x10 (fixed amplitude)")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlim(0.005, 0.08)
    ax.set_ylabel(r"$P_0(k)\;[(h^{-1}{\rm Mpc})^3]$")
    ax.legend(frameon=False)
    ax.set_title("jaxpower observed/windowed, shot-noise-subtracted")

    ax = axes[1, 1]
    ax.plot(k_ab, pk_fractional_residual, color="#276749", marker="o", ms=3.2, lw=1.25)
    ax.axhline(0.0, color="0.3", lw=0.8)
    ax.set_xscale("log")
    ax.set_xlim(0.005, 0.08)
    ax.set_xlabel(r"$k\;[h\,{\rm Mpc}^{-1}]$")
    ax.set_ylabel(r"$\bar P_{\rm EZ}/\bar P_{\rm Aba}-1$")

    figure.suptitle(
        rf"Task43 lightcone validation: $\rho_c={RHO_C:g}$, $\rho_{{\rm exp}}={RHO_EXP:g}$, "
        rf"$b={PDF_BASE:g}$, $\sigma_v={SIGMA_V:g}$; $P_0^{{\rm FKP}}=10000$",
        fontsize=13,
    )
    atomic_savefig(figure, output_pdf)
    plt.close(figure)

    summary = {
        "task": "task43_summarize_ezmock_lightcone_validation",
        "status": "done",
        "classification": (
            "10-realization fixed-amplitude mean/pipeline validation; EZmock xi uses FCFC, "
            "Abacus x25 xi uses approved cucount reference; scatter is diagnostic only, not covariance"
        ),
        "fixed_amplitude": True,
        "covariance_production_policy": "future production must use FIX_AMPLITUDE=F",
        "nreal_ezmock": 10,
        "nreal_abacus": 25,
        "seeds": list(SEEDS),
        "parameters": {"rho_c": RHO_C, "rho_exp": RHO_EXP, "pdf_base": PDF_BASE, "sigma_v": SIGMA_V},
        "measurement_contract": {
            "2pcf": (
                "EZmock: FCFC_2PT weighted Landy-Szalay; Abacus x25: existing cucount "
                "weighted Landy-Szalay; common aperiodic s_edges=50..550 ds=10"
            ),
            "abacus_xi_reference_decision": (
                "user approved reuse of existing 25-phase cucount mean/scatter after ph000 "
                "full-input FCFC bridge passed; remaining 24 Abacus phases were not remeasured"
            ),
            "pk": "jaxpower survey data-random FKP field, mesh=256, local LOS, observed/windowed, shot-noise-subtracted",
            "random": "per-realization positive-octant data_redshift_resample x25",
            "fkp": "frozen Abacus nbar(z), P0=10000",
            "pk_comparison": "same 13 DESI-PNG bin edges; no interpolation of P(k)",
        },
        "fcfc_cucount_bridge_ph000": bridge,
        "metrics": metrics,
        "window_audit": {
            "ezmock_phase": ez_pk_items[0]["phase"],
            "has_window": ez_pk_items[0]["has_window"],
            "window_matrix_shape": ez_pk_items[0]["window_shape"],
            "note": "observed P(k) already carries survey window; matrix is saved for geometry audit, not applied again to data",
        },
        "outputs": {
            "pdf": str(output_pdf),
            "pdf_sha256": sha256(output_pdf),
            "npz": str(output_npz),
            "npz_sha256": sha256(output_npz),
            "json": str(output_json),
        },
        "inputs": {
            "manifest": str(args.manifest),
            "abacus_pk_payload": str(ABACUS_PK_PAYLOAD),
            "abacus_cucount_summary": str(ABACUS_XI_CUCOUNT_SUMMARY),
            "abacus_fkp_summary": str(ABACUS_FKP_SUMMARY),
            "ezmock_xi": [str(item["path"]) for item in ez_xi_items],
            "abacus_xi_reference": str(ABACUS_XI_CUCOUNT_SUMMARY),
            "abacus_xi_fcfc_bridge_ph000": str(abacus_bridge_item["path"]),
            "ezmock_pk": [str(item["path"]) for item in ez_pk_items],
        },
    }
    write_json(output_json, summary)
    print(f"[write] {output_npz}")
    print(f"[write] {output_json}")
    print(f"[write] {output_pdf}")
    print(json.dumps(metrics, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
