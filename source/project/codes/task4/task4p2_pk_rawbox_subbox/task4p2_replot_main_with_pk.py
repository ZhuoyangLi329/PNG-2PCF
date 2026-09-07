#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Replot the active Task4.2 2PCF forest figure with Task4.2 P(k) results.

The plot intentionally uses each sample's own P(k) fit kmin:
L3000 rawbox -> 2pi/3000, L1500 -> 2pi/1500, L1000 -> 2pi/1000.
L750 is intentionally excluded from this curated figure.

执行逻辑：
1. 读取原有 Task4.2 2PCF 约束，不改写 2PCF-only 主图。
2. 只接受 z=1、free-sn0 的正式 P(k) 长链，防止旧 z=0.725 结果再次混入。
3. 以 Task4.2 原 forest 风格叠加三个 own-kmin P(k) 点，并写出显式 manifest。
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from task4p2_pk_common import FASTPM_SNAPSHOT_Z


PROJECT_ROOT = Path("/pscratch/sd/l/lzy/PNG-2PCF-model-to-observe")

TWOPCF_SUMMARY_ACTIVE = (
    PROJECT_ROOT
    / "outputs"
    / "task4_outputs"
    / "fastpm_fnl100_subbox_emcee_p1p2_task47_r50_free_sn0_formalgic"
    / "task4p2_r50_free_sn0_p1p2_formalgic_emcee_summary.json"
)
TWOPCF_SUMMARY_ARCHIVE = (
    PROJECT_ROOT
    / "old_doc_codes"
    / "task4_task44_cleanup_20260707T061844Z"
    / "moved"
    / "outputs"
    / "task4_outputs"
    / "fastpm_fnl100_subbox_emcee_p1p2_task47_r50_free_sn0_formalgic"
    / "task4p2_r50_free_sn0_p1p2_formalgic_emcee_summary.json"
)

# 2026-07-07 仓库整理后，2PCF 完整 MCMC summary 被带 manifest 地归档，
# 而 curated 2PCF-only PDF 仍保留在 active plot 目录。这里显式使用 fallback，
# 避免通过递归扫描 outputs 猜输入，也避免为了重画一张图复制大批旧链。
TWOPCF_SUMMARY = TWOPCF_SUMMARY_ACTIVE if TWOPCF_SUMMARY_ACTIVE.exists() else TWOPCF_SUMMARY_ARCHIVE

PK_SUMMARIES = {
    "rawbox reference": (
        PROJECT_ROOT
        / "outputs"
        / "task4_outputs"
        / "task4p2_pk_rawbox_subbox"
        / "fits"
        / "rawbox_ownkmin_free_sn0_sigmas0_z1"
        / "task4p2_rawbox_ownkmin_free_sn0_sigmas0_z1_pk_windowed_fit_summary.json"
    ),
    "L1500": (
        PROJECT_ROOT
        / "outputs"
        / "task4_outputs"
        / "task4p2_pk_rawbox_subbox"
        / "fits"
        / "L1500_main750_windowed_free_sn0_sigmas0_z1_wtheoryfitkmin"
        / "task4p2_L1500_main750_windowed_free_sn0_sigmas0_z1_wtheoryfitkmin_pk_windowed_fit_summary.json"
    ),
    "L1000": (
        PROJECT_ROOT
        / "outputs"
        / "task4_outputs"
        / "task4p2_pk_rawbox_subbox"
        / "fits"
        / "L1000_main1500_windowed_free_sn0_sigmas0_z1_wtheoryfitkmin"
        / "task4p2_L1000_main1500_windowed_free_sn0_sigmas0_z1_wtheoryfitkmin_pk_windowed_fit_summary.json"
    ),
}

OUT_PDF = (
    PROJECT_ROOT
    / "plots"
    / "task4"
    / "important_4p1_4p2"
    / "4p2_main_r50_free_sn0_formalgic_with_pk.pdf"
)
OUT_MANIFEST = (
    PROJECT_ROOT
    / "outputs"
    / "task4_outputs"
    / "task4p2_pk_rawbox_subbox"
    / "comparison"
    / "task4p2_4p2_main_r50_free_sn0_formalgic_with_pk_explicit_manifest.json"
)
PK_CHAIN_AUDIT = (
    PROJECT_ROOT
    / "outputs"
    / "task4_outputs"
    / "task4p2_pk_rawbox_subbox"
    / "audits"
    / "task4p2_pk_z1_free_sn0_chain_audit.json"
)

ROW_ORDER = ("rawbox reference", "L1500", "L1000")
YTICK_LABELS = {
    "rawbox reference": "3Gpc/h(original box)",
    "L1500": "1.5Gpc/h",
    "L1000": "1Gpc/h",
}
TWOPCF_KEYS = {
    "rawbox reference": "rawbox",
    "L1500_no_gic": "L1500_no_gic",
    "L1500_formal_gic": "L1500_formal_gic",
    "L1000_no_gic": "L1000_no_gic",
    "L1000_formal_gic": "L1000_formal_gic",
}


def read_json(path: Path) -> dict[str, Any]:
    """Read a JSON file."""
    return json.loads(path.read_text(encoding="utf-8"))


def constraint_from_parameter(param: dict[str, Any], *, prefer_percival: bool = True) -> dict[str, float]:
    """Return median and asymmetric errors from a parameter summary."""
    low_key = "err_low_percival" if prefer_percival and "err_low_percival" in param else "err_low"
    high_key = "err_high_percival" if prefer_percival and "err_high_percival" in param else "err_high"
    return {
        "median": float(param["median"]),
        "err_low": float(param[low_key]),
        "err_high": float(param[high_key]),
        "err_low_raw": float(param.get("err_low", param[low_key])),
        "err_high_raw": float(param.get("err_high", param[high_key])),
    }


def format_constraint(item: dict[str, float]) -> str:
    """Compact plot annotation for one fNL constraint."""
    median = float(item["median"])
    lo = float(item["err_low"])
    hi = float(item["err_high"])
    if np.isclose(lo, hi, rtol=0.06, atol=1.0):
        return f"{median:.1f} +/- {0.5 * (lo + hi):.1f}"
    return f"{median:.1f} -{lo:.1f}/+{hi:.1f}"


def load_twopcf() -> dict[str, dict[str, float]]:
    """Load 2PCF no-GIC/formal-GIC constraints for the curated rows."""
    summary = read_json(TWOPCF_SUMMARY)
    summaries = summary["summaries"]
    out = {
        "rawbox reference": constraint_from_parameter(summaries["rawbox"]["parameters"]["fnl_loc"]),
        "L1500 no-GIC": constraint_from_parameter(summaries["L1500_no_gic"]["parameters"]["fnl_loc"]),
        "L1500 formal-GIC": constraint_from_parameter(summaries["L1500_formal_gic"]["parameters"]["fnl_loc"]),
        "L1000 no-GIC": constraint_from_parameter(summaries["L1000_no_gic"]["parameters"]["fnl_loc"]),
        "L1000 formal-GIC": constraint_from_parameter(summaries["L1000_formal_gic"]["parameters"]["fnl_loc"]),
    }
    return out


def validate_pk_summary(summary: dict[str, Any], path: Path) -> None:
    """拒绝旧红移或 fixed-sn0 P(k) summary，避免静默回归到错误口径。"""
    config = summary.get("config", {})
    extra = summary.get("extra_meta", {})
    template_z = float(config.get("template_z", np.nan))
    free_parameters = tuple(str(name) for name in config.get("free_parameters", ()))
    fit_kmin = float(extra.get("fit_kmin_observed", extra.get("kmin_observed", extra.get("kmin_model", np.nan))))
    theory_kmin = float(extra.get("theory_kmin_model", extra.get("rawbox_theory_kmin", np.nan)))
    if not np.isclose(template_z, FASTPM_SNAPSHOT_Z, rtol=0.0, atol=1.0e-12):
        raise ValueError(f"P(k) summary 必须使用 FastPM snapshot z=1，实际为 z={template_z}: {path}")
    if "sn0" not in free_parameters:
        raise ValueError(f"P(k) summary 必须 marginalize free sn0: {path}")
    if not np.isclose(theory_kmin, fit_kmin, rtol=0.0, atol=1.0e-12):
        raise ValueError(
            f"当前 own-kmin 主图要求 theory kmin == fit kmin，实际为 {theory_kmin} != {fit_kmin}: {path}"
        )


def load_pk() -> dict[str, dict[str, float]]:
    """Load and validate the formal own-kmin P(k) constraints."""
    out = {}
    for sample, path in PK_SUMMARIES.items():
        summary = read_json(path)
        validate_pk_summary(summary, path)
        item = constraint_from_parameter(summary["parameters"]["fnl_loc"])
        item["nmock"] = float(summary.get("covariance", {}).get("nmock", np.nan))
        item["ndata"] = float(summary.get("data", {}).get("ndata", np.nan))
        item["fit_kmin"] = float(
            summary.get("extra_meta", {}).get(
                "fit_kmin_observed",
                summary.get("extra_meta", {}).get("kmin_observed", summary.get("extra_meta", {}).get("kmin_model", np.nan)),
            )
        )
        item["template_z"] = float(summary["config"]["template_z"])
        item["theory_kmin"] = float(
            summary.get("extra_meta", {}).get(
                "theory_kmin_model", summary.get("extra_meta", {}).get("rawbox_theory_kmin", np.nan)
            )
        )
        item["sn0_free"] = True
        out[sample] = item
    return out


def validate_chain_audit() -> dict[str, Any]:
    """要求三个正式 P(k) case 的统一链审计全部通过后才允许画主图。"""
    audit = read_json(PK_CHAIN_AUDIT)
    expected_cases = {"rawbox", "L1500", "L1000"}
    actual_cases = set(audit.get("cases", {}))
    failed = [name for name, item in audit.get("cases", {}).items() if item.get("status") != "pass"]
    if audit.get("status") != "pass" or actual_cases != expected_cases or failed:
        raise ValueError(
            f"P(k) chain audit 未通过或 case 不完整: status={audit.get('status')}, "
            f"actual_cases={sorted(actual_cases)}, failed={failed}"
        )
    return audit


def add_point(
    ax: plt.Axes,
    *,
    item: dict[str, float],
    y: float,
    color: str,
    marker: str,
    label: str | None,
    text_dy: int = 5,
    zorder: int = 3,
) -> tuple[float, float]:
    """Draw one horizontal errorbar and annotation; return interval bounds."""
    x = float(item["median"])
    lo = float(item["err_low"])
    hi = float(item["err_high"])
    ax.errorbar(
        x,
        y,
        xerr=np.array([[lo], [hi]]),
        fmt=marker,
        color=color,
        mfc="white",
        mew=2.0,
        ms=8.0,
        capsize=5,
        capthick=1.8,
        elinewidth=2.0,
        label=label,
        zorder=zorder,
    )
    ax.annotate(
        format_constraint(item),
        xy=(x, y),
        xytext=(9, text_dy),
        textcoords="offset points",
        color=color,
        fontsize=13,
        ha="left",
        va="bottom",
        annotation_clip=False,
        clip_on=False,
        zorder=6,
    )
    return x - lo, x + hi


def make_plot(
    twopcf: dict[str, dict[str, float]],
    pk: dict[str, dict[str, float]],
    *,
    twopcf_descriptor: str = r"2PCF: $r=50$-$350\,h^{-1}{\rm Mpc}$",
) -> dict[str, Any]:
    """Create the curated Task4.2 forest plot with 2PCF and P(k)."""
    plt.rcParams.update(
        {
            "font.size": 18,
            "axes.titlesize": 22,
            "axes.labelsize": 21,
            "xtick.labelsize": 18,
            "ytick.labelsize": 18,
            "legend.fontsize": 13,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "savefig.facecolor": "white",
            "savefig.transparent": False,
        }
    )
    fig, ax = plt.subplots(figsize=(12.8, 5.7))
    fig.patch.set_facecolor("white")
    ax.set_facecolor("white")

    colors = {
        "raw": "#1f4e79",
        "nogic": "#4d4d4d",
        "formal": "#b04a2f",
        "pk": "#2b7a78",
    }
    intervals: list[tuple[float, float]] = []
    y_positions = np.arange(len(ROW_ORDER), dtype="f8")[::-1]

    for y, sample in zip(y_positions, ROW_ORDER, strict=True):
        if sample == "rawbox reference":
            intervals.append(
                add_point(
                    ax,
                    item=twopcf["rawbox reference"],
                    y=y + 0.13,
                    color=colors["raw"],
                    marker="D",
                    label="2PCF rawbox",
                    text_dy=5,
                    zorder=4,
                )
            )
            intervals.append(
                add_point(
                    ax,
                    item=pk[sample],
                    y=y - 0.13,
                    color=colors["pk"],
                    marker="^",
                    label=r"P(k), $k_{\rm th}\geq k_{\min}^{\rm fit}$",
                    text_dy=-17,
                    zorder=4,
                )
            )
            continue

        intervals.append(
            add_point(
                ax,
                item=twopcf[f"{sample} no-GIC"],
                y=y + 0.20,
                color=colors["nogic"],
                marker="o",
                label="2PCF no-GIC" if sample == "L1500" else None,
                text_dy=9,
            )
        )
        intervals.append(
            add_point(
                ax,
                item=twopcf[f"{sample} formal-GIC"],
                y=y,
                color=colors["formal"],
                marker="s",
                label="2PCF formal-GIC" if sample == "L1500" else None,
                text_dy=-18,
            )
        )
        intervals.append(
            add_point(
                ax,
                item=pk[sample],
                y=y - 0.20,
                color=colors["pk"],
                marker="^",
                label=None,
                text_dy=-17,
            )
        )

    raw_med = float(twopcf["rawbox reference"]["median"])
    ax.axvline(raw_med, color="#777777", ls="--", lw=1.3, alpha=0.75)
    ax.axvline(100.0, color="0.55", ls=":", lw=1.3, alpha=0.75)

    finite = np.asarray(intervals, dtype="f8")
    xmin = max(-250.0, float(np.nanmin(finite[:, 0]) - 35.0))
    xmax = min(400.0, float(np.nanmax(finite[:, 1]) + 120.0))
    if xmax - xmin < 190.0:
        mid = 0.5 * (xmin + xmax)
        xmin, xmax = mid - 95.0, mid + 95.0

    ax.set_yticks(y_positions)
    ax.set_yticklabels([YTICK_LABELS[item] for item in ROW_ORDER])
    ax.set_xlabel(r"$f_{\rm NL}^{\rm loc}$")
    ax.set_xlim(xmin, xmax)
    ax.set_ylim(-0.55, len(ROW_ORDER) + 0.25)
    ax.grid(axis="x", color="#d8d8d8", lw=0.8, alpha=0.75)
    ax.tick_params(axis="y", length=0)
    ax.legend(frameon=False, loc="upper right", fontsize=13, ncol=1)
    ax.text(
        0.02,
        0.965,
        r"FastPM $f_{\rm NL}=100$",
        transform=ax.transAxes,
        fontsize=22,
        fontweight="bold",
        color="0.15",
        ha="left",
        va="top",
    )
    ax.text(
        0.02,
        0.900,
        twopcf_descriptor + r";  P(k): $k_{\rm th}\geq k_{\min}^{\rm fit}$, $k_{\max}=0.08$",
        transform=ax.transAxes,
        fontsize=12.5,
        color="0.25",
        ha="left",
        va="top",
    )
    fig.subplots_adjust(left=0.23, right=0.97, top=0.93, bottom=0.18)
    OUT_PDF.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT_PDF, facecolor="white", transparent=False)
    plt.close(fig)
    return {
        "plot_pdf": str(OUT_PDF),
        "xlim": [float(xmin), float(xmax)],
    }


def main() -> None:
    """Main entry point."""
    chain_audit = validate_chain_audit()
    twopcf = load_twopcf()
    pk = load_pk()
    paths = make_plot(twopcf, pk)
    rows = []
    for sample in ROW_ORDER:
        rows.append(
            {
                "sample": sample,
                "twopcf_rawbox_or_formal_gic": twopcf["rawbox reference" if sample == "rawbox reference" else f"{sample} formal-GIC"],
                "twopcf_no_gic": None if sample == "rawbox reference" else twopcf[f"{sample} no-GIC"],
                "pk_own_kmin": pk[sample],
            }
        )
    manifest = {
        "task": "task4p2_replot_4p2_main_with_pk",
        "status": "done",
        "note": (
            "L750 intentionally excluded; P(k) points use each sample's own fit kmin, "
            "the FastPM snapshot redshift z=1, and a free sn0 nuisance parameter."
        ),
        "formal_pk_policy": {
            "template_z": FASTPM_SNAPSHOT_Z,
            "sn0": "free",
            "sigmas": "fixed_zero_for_no_rsd",
            "legacy_z0p725_results_allowed": False,
        },
        "twopcf_summary": str(TWOPCF_SUMMARY),
        "twopcf_summary_location_policy": {
            "active_candidate": str(TWOPCF_SUMMARY_ACTIVE),
            "archive_fallback": str(TWOPCF_SUMMARY_ARCHIVE),
            "used_archive_fallback": bool(TWOPCF_SUMMARY == TWOPCF_SUMMARY_ARCHIVE),
        },
        "pk_summaries": {key: str(value) for key, value in PK_SUMMARIES.items()},
        "pk_chain_audit": {
            "path": str(PK_CHAIN_AUDIT),
            "status": str(chain_audit["status"]),
            "required_cases": ["rawbox", "L1500", "L1000"],
        },
        "twopcf_caveat": (
            "The preserved Task47 2PCF theory was evaluated at bin centers rather than spherical-shell averaged; "
            "the requested overlay is not a final Task4.2 closure claim."
        ),
        "rows": rows,
        "paths": paths,
    }
    OUT_MANIFEST.parent.mkdir(parents=True, exist_ok=True)
    OUT_MANIFEST.write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"[write] {OUT_PDF}")
    print(f"[write] {OUT_MANIFEST}")


if __name__ == "__main__":
    main()
