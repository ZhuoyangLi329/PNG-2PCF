#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""绘制 Task4.3 radial single-term RIC 的最终 PDF 图组。

代码大纲
========
1. correction/convergence 图：2PCF 与 P(k) 的 baseline correction/数据误差，
   以及 dchi=1/4、50k 相对 dchi=2/200k 的逐 bin 差异和 0.05 sigma 门槛。
2. posterior overlay：沿用 Task4.4 的 PDF 字体、红/青/蓝/灰配色与填充
   68/95% contour 风格，左侧比较 2PCF no/global/radial，右侧比较 P(k)
   geometry-only/radial。
3. P(k)-2PCF comparison：统一画五条 fNL 68% interval，直接展示 probe
   consistency 与 RIC 带来的中心移动。
4. 只写 PDF 图和一个小型 JSON plot manifest；不生成 PNG。
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

for _name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_name, "1")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import to_rgba
from matplotlib.lines import Line2D
from scipy.ndimage import gaussian_filter

from task43_ric_singleterm import RIC_PLOT_DIR, write_json


PROJECT_ROOT = Path("/pscratch/sd/l/lzy/PNG-2PCF-model-to-observe")
AUDIT_PATH = PROJECT_ROOT / "outputs/task43_outputs/ric_singleterm/audits/task43_ric_singleterm_audit.json"
XI_PATH = PROJECT_ROOT / "outputs/task43_outputs/summary/task43_mean_xi_mmin1p4e13_x25_s50_350_ds10_fkpP010000.npz"
XI_COV = PROJECT_ROOT / "outputs/task43_outputs/summary/jaxpower_covariance_lightcone_mmin1p4e13_ph000_fitcov_bessel_interp_smoothfftlog_rrdeconv_mesh64_nran100k_ndata50k_pad400_win3600_ds2_k0001_3000_dk002_p1p0_s50_350_ds10.npz"
PK_PAYLOAD = PROJECT_ROOT / "outputs/task43_outputs/pk_lightcone/summary/task43_pk_lightcone_mmin1p4e13_x25_fkpP010000_desi_rebin_kmax0p10_payload.npz"
OPDIR = PROJECT_ROOT / "outputs/task43_outputs/ric_singleterm/operators"
FITROOT = PROJECT_ROOT / "outputs/task43_outputs/ric_singleterm/fits"

XI_SAMPLE_PATHS = {
    "no_gic": FITROOT / "2pcf_ph000_dchi2_nsub200000_mcmc5x/task43_mcmc_no_gic_samples.npz",
    "formal_gic": FITROOT / "2pcf_ph000_dchi2_nsub200000_mcmc5x/task43_mcmc_formal_gic_samples.npz",
    "radial_singleterm": FITROOT / "2pcf_ph000_dchi2_nsub200000_mcmc5x/task43_mcmc_radial_singleterm_samples.npz",
}
PK_SAMPLE_PATHS = {
    "geometry_only": PROJECT_ROOT / "outputs/task43_outputs/pk_lightcone/fits/mmin1p4e13_x25_fkpP010000_desi_rebin_kmax0p10_free_sn0_wtheorykmin_boxL2000_mcmc5x_seed20260712/task43_pk_lightcone_mmin1p4e13_x25_fkpP010000_desi_rebin_kmax0p10_free_sn0_wtheorykmin_boxL2000_mcmc5x_seed20260712_fit_samples.npz",
    "radial_singleterm": FITROOT / "pk/task43_pk_ric_ph000_dchi2_nsub200000_motherbox_mcmc5x/task43_pk_lightcone_task43_pk_ric_ph000_dchi2_nsub200000_motherbox_mcmc5x_fit_samples.npz",
}

COLORS = {
    "no_gic": "#64748b",
    "formal_gic": "#2563eb",
    "radial_singleterm": "#dc2626",
    "geometry_only": "#42949E",
    "pk_radial": "#B64342",
    "truth": "#16a34a",
}


def configure_style() -> None:
    """统一使用 Task4.4 contour 图的字体与线宽口径。"""
    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "DejaVu Sans", "Liberation Sans"],
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "font.size": 10,
            "axes.linewidth": 0.8,
            "legend.frameon": False,
        }
    )


def load_json(path: Path) -> dict[str, Any]:
    """读取小型 summary/audit JSON。"""
    return json.loads(path.read_text(encoding="utf-8"))


def operator_path(width: int, nsub: int) -> Path:
    """返回 ph000 factorized operator 路径。"""
    return OPDIR / (
        f"task43_ric_factorized_operator_ph000_dchi{width}_nsub{nsub}_"
        "sobol2p22_ds2_seed20260712_L2000.npz"
    )


def xi_correction(path: Path) -> np.ndarray:
    """读取 fNL=0,b1=2.5 的 2PCF positive auto response。"""
    with np.load(path, allow_pickle=False) as data:
        return 2.5**2 * np.asarray(data["xi_basis_pk_dd"], dtype="f8")


def pk_correction(path: Path, fiducial_vector: np.ndarray) -> np.ndarray:
    """读取 fNL=0,b1=2.5 的 P(k) positive auto response。"""
    with np.load(path, allow_pickle=False) as data:
        return np.asarray(data["pk_ric_matrix"], dtype="f8") @ fiducial_vector


def load_samples(path: Path, *, probe: str) -> np.ndarray:
    """读取 contour 所需的 (fNL,b1) 两列。"""
    with np.load(path, allow_pickle=False) as data:
        samples = np.asarray(data["samples"], dtype="f8")
        if probe == "pk":
            names = [str(v) for v in np.asarray(data["param_names"])]
            return samples[:, [names.index("fnl_loc"), names.index("b1")]]
        return samples[:, :2]


def density_levels(density: np.ndarray) -> tuple[float, float]:
    """返回包含 95% 与 68% probability mass 的 density thresholds。"""
    flat = np.sort(np.asarray(density, dtype="f8").ravel())[::-1]
    cumulative = np.cumsum(flat)
    cumulative /= cumulative[-1]
    level68 = flat[min(int(np.searchsorted(cumulative, 0.68)), flat.size - 1)]
    level95 = flat[min(int(np.searchsorted(cumulative, 0.95)), flat.size - 1)]
    return float(level95), float(level68)


def draw_contour(
    ax: plt.Axes,
    samples: np.ndarray,
    *,
    color: str,
    x_range: tuple[float, float],
    y_range: tuple[float, float],
) -> None:
    """画一组 Task4.4-style 平滑 68/95% 填充 contour。"""
    hist, xedges, yedges = np.histogram2d(
        samples[:, 0],
        samples[:, 1],
        bins=(180, 160),
        range=(x_range, y_range),
        density=True,
    )
    # 2PCF 只有 120k 样本且三条 contour 高度重叠；稍强平滑可去掉 histogram
    # pixel wiggle，同时远小于 posterior 宽度，不改变 68/95% 面积关系。
    hist = gaussian_filter(hist, sigma=3.0)
    x = 0.5 * (xedges[:-1] + xedges[1:])
    y = 0.5 * (yedges[:-1] + yedges[1:])
    level95, level68 = density_levels(hist)
    top = float(np.max(hist)) * 1.001
    ax.contourf(
        x,
        y,
        hist.T,
        levels=[level95, level68, top],
        colors=[to_rgba(color, 0.14), to_rgba(color, 0.30)],
        antialiased=True,
    )
    ax.contour(x, y, hist.T, levels=[level95, level68], colors=color, linewidths=[1.2, 1.8])


def correction_figure(audit: dict[str, Any], output: Path) -> None:
    """绘制 correction 大小与 model-level convergence 四面板。"""
    with np.load(XI_PATH, allow_pickle=False) as data:
        s = np.asarray(data["s"], dtype="f8")
    with np.load(XI_COV, allow_pickle=False) as data:
        xi_sigma = np.sqrt(np.diag(np.asarray(data["covariance_single_realization"], dtype="f8")))
    with np.load(PK_PAYLOAD, allow_pickle=False) as data:
        k = np.asarray(data["k_obs"], dtype="f8")
        pk_sigma = np.sqrt(np.diag(np.asarray(data["covariance"], dtype="f8")))
        theory_k = np.asarray(data["theory_k"], dtype="f8")
        theory_ell = np.asarray(data["theory_ell"], dtype="i8")
    # audit 已保存 fiducial baseline response；无需在绘图脚本重新建 cosmology。
    xi_ref = np.asarray(audit["baseline_correction"]["xi_values"], dtype="f8")
    pk_ref = np.asarray(audit["baseline_correction"]["pk_values"], dtype="f8")
    # 为 variation P(k) response 恢复 operator 所需的 fiducial theory vector：
    # 从 baseline correction 无法反演，因此调用 summary 脚本的相同 helper。
    from task43_summarize_ric_singleterm import FitData, make_pk_theory_basis

    fit_data = FitData(PK_PAYLOAD)
    theory_basis = make_pk_theory_basis(fit_data)
    fiducial = 2.5**2 * theory_basis["pk_dd"]

    variations = {
        r"$\Delta\chi=1$": (operator_path(1, 200000), "#2563eb", "--"),
        r"$\Delta\chi=4$": (operator_path(4, 200000), "#f59e0b", "-."),
        "50k random": (operator_path(2, 50000), "#64748b", ":"),
    }
    fig, axes = plt.subplots(2, 2, figsize=(10.2, 7.2), sharex="col")
    ax = axes[0, 0]
    ax.plot(s, xi_ref / xi_sigma, color=COLORS["radial_singleterm"], lw=2.0, marker="o", ms=3.0, label=r"baseline radial $IC^{\rm rad,rad}$")
    global_value = float(audit["operator_tests"]["global_limit"]["constant_value_fnl0_b1_2p5"])
    ax.plot(s, np.full_like(s, global_value) / xi_sigma, color="#2563eb", lw=1.5, ls="--", label="merged-bin global limit")
    ax.set_ylabel(r"2PCF correction $/\sigma_{\xi}$")
    ax.legend(loc="upper right", fontsize=8.5)

    ax = axes[0, 1]
    ax.plot(k, pk_ref / pk_sigma, color="#B64342", lw=2.0, marker="o", ms=3.0)
    ax.set_ylabel(r"$P(k)$ correction $/\sigma_P$")

    ax = axes[1, 0]
    for label, (path, color, linestyle) in variations.items():
        delta = (xi_correction(path) - xi_ref) / xi_sigma
        ax.plot(s, delta, color=color, lw=1.5, ls=linestyle, label=label)
    ax.axhline(0.0, color="0.35", lw=0.8)
    ax.set_ylim(-5.0e-4, 5.0e-4)
    ax.text(0.03, 0.94, r"$0.05\sigma$ gate lies outside this zoom", transform=ax.transAxes, va="top", fontsize=8.2, color="0.35")
    ax.set_xlabel(r"$s\,[h^{-1}{\rm Mpc}]$")
    ax.set_ylabel(r"$\Delta IC_{\xi}/\sigma_{\xi}$")
    ax.legend(loc="best", fontsize=8.3)

    ax = axes[1, 1]
    for label, (path, color, linestyle) in variations.items():
        delta = (pk_correction(path, fiducial) - pk_ref) / pk_sigma
        ax.plot(k, delta, color=color, lw=1.5, ls=linestyle, marker="o", ms=2.5, label=label)
    ax.axhline(0.0, color="0.35", lw=0.8)
    ax.set_ylim(-5.0e-4, 5.0e-4)
    ax.text(0.03, 0.94, r"$0.05\sigma$ gate lies outside this zoom", transform=ax.transAxes, va="top", fontsize=8.2, color="0.35")
    ax.set_xlabel(r"$k\,[h\,{\rm Mpc}^{-1}]$")
    ax.set_ylabel(r"$\Delta IC_P/\sigma_P$")
    for axis in axes.ravel():
        axis.grid(color="#d0d0d0", lw=0.55, alpha=0.65)
    fig.suptitle("Task4.3 radial single-term RIC: correction and convergence", fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, bbox_inches="tight")
    plt.close(fig)


def posterior_figure(output: Path) -> None:
    """绘制 2PCF 与 P(k) 的 fNL-b1 contour overlay。"""
    xi_samples = {name: load_samples(path, probe="xi") for name, path in XI_SAMPLE_PATHS.items()}
    pk_samples = {name: load_samples(path, probe="pk") for name, path in PK_SAMPLE_PATHS.items()}
    x_range = (-110.0, 110.0)
    y_range = (2.15, 2.90)
    fig, axes = plt.subplots(1, 2, figsize=(10.4, 4.6), sharex=True, sharey=True)
    xi_labels = {"no_gic": "2PCF: no IC", "formal_gic": "2PCF: global single-term", "radial_singleterm": "2PCF: radial single-term"}
    for name in ("no_gic", "formal_gic", "radial_singleterm"):
        draw_contour(axes[0], xi_samples[name], color=COLORS[name], x_range=x_range, y_range=y_range)
    pk_colors = {"geometry_only": COLORS["geometry_only"], "radial_singleterm": COLORS["pk_radial"]}
    pk_labels = {"geometry_only": r"$P(k)$: geometry only", "radial_singleterm": r"$P(k)$: radial single-term"}
    for name in ("geometry_only", "radial_singleterm"):
        draw_contour(axes[1], pk_samples[name], color=pk_colors[name], x_range=x_range, y_range=y_range)
    axes[0].legend(
        [Line2D([], [], color=COLORS[name], lw=2) for name in ("no_gic", "formal_gic", "radial_singleterm")],
        [xi_labels[name] for name in ("no_gic", "formal_gic", "radial_singleterm")],
        loc="upper right",
        fontsize=8.5,
    )
    axes[1].legend(
        [Line2D([], [], color=pk_colors[name], lw=2) for name in ("geometry_only", "radial_singleterm")],
        [pk_labels[name] for name in ("geometry_only", "radial_singleterm")],
        loc="upper right",
        fontsize=8.5,
    )
    for axis, title in zip(axes, ("2PCF, fixed $sn_0=0$", r"$P(k)$, free $sn_0$"), strict=True):
        axis.axvline(0.0, color=COLORS["truth"], ls="-.", lw=1.3)
        axis.set_xlim(*x_range)
        axis.set_ylim(*y_range)
        axis.set_xlabel(r"$f_{\rm NL}^{\rm loc}$")
        axis.set_title(title)
        axis.grid(color="#d0d0d0", lw=0.5, alpha=0.55)
    axes[0].set_ylabel(r"$b_1$")
    fig.suptitle("Task4.3 posterior response to radial single-term RIC", fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, bbox_inches="tight")
    plt.close(fig)


def comparison_figure(audit: dict[str, Any], output: Path) -> None:
    """用统一坐标展示 2PCF/P(k) 五条 fNL 68% interval。"""
    rows = [
        ("2PCF: no IC", audit["main_posteriors"]["2pcf"]["no_gic"], COLORS["no_gic"], "o"),
        ("2PCF: global single-term", audit["main_posteriors"]["2pcf"]["formal_gic"], COLORS["formal_gic"], "s"),
        ("2PCF: radial single-term", audit["main_posteriors"]["2pcf"]["radial_singleterm"], COLORS["radial_singleterm"], "D"),
        (r"$P(k)$: geometry only", audit["main_posteriors"]["pk"]["geometry_only"], COLORS["geometry_only"], "o"),
        (r"$P(k)$: radial single-term", audit["main_posteriors"]["pk"]["radial_singleterm"], COLORS["pk_radial"], "D"),
    ]
    fig, ax = plt.subplots(figsize=(8.2, 4.8))
    ypos = np.arange(len(rows))[::-1]
    for y, (label, row, color, marker) in zip(ypos, rows, strict=True):
        q16, q50, q84 = row["fnl_q16"], row["fnl_q50"], row["fnl_q84"]
        ax.errorbar(
            q50,
            y,
            xerr=np.asarray([[q50 - q16], [q84 - q50]]),
            fmt=marker,
            ms=7,
            color=color,
            ecolor=color,
            elinewidth=2.0,
            capsize=4,
        )
        ax.text(104, y, rf"${q50:.1f}^{{+{q84-q50:.1f}}}_{{-{q50-q16:.1f}}}$", va="center", ha="right", fontsize=9)
    ax.axvline(0.0, color=COLORS["truth"], ls="-.", lw=1.4, label="truth")
    ax.set_yticks(ypos, [row[0] for row in rows])
    ax.set_xlim(-75, 110)
    ax.set_xlabel(r"$f_{\rm NL}^{\rm loc}$")
    ax.set_title("Task4.3: 2PCF and $P(k)$ constraints with radial single-term RIC")
    ax.grid(axis="x", color="#d0d0d0", lw=0.6, alpha=0.7)
    ax.legend(loc="lower left")
    fig.tight_layout()
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    """生成三张 PDF 并写 plot manifest。"""
    configure_style()
    audit = load_json(AUDIT_PATH)
    outputs = {
        "correction_pdf": RIC_PLOT_DIR / "task43_ric_singleterm_correction_convergence.pdf",
        "posterior_pdf": RIC_PLOT_DIR / "task43_ric_singleterm_posterior_overlay.pdf",
        "comparison_pdf": RIC_PLOT_DIR / "task43_ric_singleterm_pk_vs_2pcf.pdf",
    }
    correction_figure(audit, outputs["correction_pdf"])
    posterior_figure(outputs["posterior_pdf"])
    comparison_figure(audit, outputs["comparison_pdf"])
    manifest = {
        "task": "task43_plot_ric_singleterm",
        "status": "done",
        "audit": str(AUDIT_PATH),
        "style": "Task4.4-style PDF font42, filled 68/95 contours, red/teal/blue/gray palette",
        "outputs": {key: str(value) for key, value in outputs.items()},
        "formats": ["pdf"],
    }
    manifest_path = RIC_PLOT_DIR / "task43_ric_singleterm_plot_manifest.json"
    write_json(manifest_path, manifest)
    for path in outputs.values():
        print(f"[write] {path}")
    print(f"[write] {manifest_path}")


if __name__ == "__main__":
    main()
