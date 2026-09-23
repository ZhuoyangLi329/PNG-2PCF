#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""重画 Task43 EZmock/jaxpower covariance 对比，并补齐全部参数 contour。

代码大纲：
1. 从已经通过收敛审计的 MCMC ``chain_*.npz`` 和 ``summary_*.json`` 读取
   EZmock empirical covariance 与 jaxpower analytic covariance 的六条链。
2. 对 ``fNL, b1, sigma_s, sn0`` 计算两页图共同的坐标范围；xi02 链没有
   ``sn0``，因此 sn0 相关面板只绘制 P02 和 joint，并在审计 JSON 中记录。
3. 每个 covariance 画一张 4x4 corner 图，保留 P02、xi02、joint 三种
   observable 的颜色和 68/95% contour，对角线绘制一维后验。
4. 原子写入 PDF 和机器可读审计 JSON 到新的 ``9.22meeting`` 目录，
   不覆盖 9.18meeting 的旧图。

运行示例（NERSC 登录节点 CPU-only）：
python task43_plot_ezmock_covariance_mcmc_vs_jaxpower_allparams.py \\
  --mcmc-root <project>/outputs/.../mcmc_preliminary_604 \\
  --nmock 604
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
from matplotlib.backends.backend_pdf import PdfPages
import matplotlib.pyplot as plt
from matplotlib.colors import to_rgba
import numpy as np
from scipy.ndimage import gaussian_filter

from task43_plot_rsd_rawbox_pk0_vs_xi0_contours import density_levels
from task43_rsd_common import PROJECT_ROOT, atomic_write_json, sha256_file


MCMC_ROOT_DEFAULT = (
    PROJECT_ROOT
    / "outputs/task43_outputs/ezmock_rsd_lightcone_covariance_z0p4_0p8_x1000_common50_nzmatch_b0.30_z0.60"
    / "mcmc_preliminary_604"
)
MEETING_DIR = PROJECT_ROOT / "9.22meeting/task43_ezmock_covariance_mcmc_vs_jaxpower_allparams"
PARAMETERS = ("fNL", "b1", "sigma_s", "sn0")
LABELS = {
    "fNL": r"$f_{\rm NL}$",
    "b1": r"$b_1$",
    "sigma_s": r"$\sigma_s\ [h^{-1}{\rm Mpc}]$",
    "sn0": r"$s_{n0}$",
}
VARIANTS = (
    ("p02", "#1f77b4", r"$P_0+P_2$"),
    ("xi02", "#2ca02c", r"$\xi_0+\xi_2$"),
    ("joint", "#d62728", "joint"),
)


def load_chain(root: Path, tag: str) -> dict[str, Any]:
    """读取一条链及其摘要，并再次检查已有的收敛门。"""

    chain_path = root / f"chain_{tag}.npz"
    summary_path = root / f"summary_{tag}.json"
    if not chain_path.is_file() or not summary_path.is_file():
        raise FileNotFoundError(f"missing MCMC product: {chain_path} / {summary_path}")
    with np.load(chain_path, allow_pickle=False) as payload:
        raw = np.asarray(payload["chain"], dtype="f8")
        chain = raw.reshape(-1, raw.shape[-1])
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    if not all(bool(value) for value in summary.get("gates", {}).values()):
        raise RuntimeError(f"MCMC convergence gates failed: {tag}")
    return {"chain": chain, "summary": summary, "path": chain_path}


def make_range(values: np.ndarray, parameter: str) -> tuple[float, float]:
    """从所有 covariance/observable 链的后验给出共享坐标范围。"""

    finite = np.asarray(values, dtype="f8")
    finite = finite[np.isfinite(finite)]
    if finite.size < 10:
        raise ValueError(f"too few finite samples for {parameter}")
    lo, hi = np.percentile(finite, [0.5, 99.5])
    span = max(float(hi - lo), 1.0e-6)
    margin = 0.08 * span
    lo, hi = float(lo - margin), float(hi + margin)
    if parameter == "fNL":
        lo = min(lo, -1.0)
        hi = max(hi, 1.0)
    elif parameter == "b1":
        lo = max(0.0, lo)
    elif parameter == "sigma_s":
        lo = max(0.0, lo)
    return lo, hi


def draw_contour(
    axis: plt.Axes,
    x: np.ndarray,
    y: np.ndarray,
    *,
    color: str,
    xlim: tuple[float, float],
    ylim: tuple[float, float],
    zorder: int,
) -> None:
    """在一个 corner 面板绘制平滑的 95%/68% 后验等高线。"""

    hist, xedges, yedges = np.histogram2d(
        np.asarray(x, dtype="f8"),
        np.asarray(y, dtype="f8"),
        bins=(150, 150),
        range=(xlim, ylim),
        density=False,
    )
    hist = gaussian_filter(hist.astype("f8"), sigma=2.2, mode="nearest")
    if float(np.max(hist)) <= 0.0:
        return
    level95, level68 = density_levels(hist)
    xcenter = 0.5 * (xedges[:-1] + xedges[1:])
    ycenter = 0.5 * (yedges[:-1] + yedges[1:])
    axis.contourf(
        xcenter,
        ycenter,
        hist.T,
        levels=[level95, level68, float(np.max(hist)) * 1.001],
        colors=[to_rgba(color, 0.09), to_rgba(color, 0.20)],
        antialiased=True,
        zorder=zorder,
    )
    axis.contour(
        xcenter,
        ycenter,
        hist.T,
        levels=[level95, level68],
        colors=color,
        linewidths=[1.0, 1.5],
        zorder=zorder + 1,
    )


def interval_text(entry: dict[str, Any], parameter: str) -> str:
    """返回审计摘要中的 maximum-likelihood 和 68% 区间文本。"""

    summary = entry["summary"]
    posterior = summary["posterior"][parameter]
    index = PARAMETERS.index(parameter)
    ml = float(np.asarray(summary["map_theta"], dtype="f8")[index])
    q16, q84 = float(posterior["q16"]), float(posterior["q84"])
    return rf"{ml:.2f}_{{-{ml-q16:.2f}}}^{{+{q84-ml:.2f}}}"


def corner_page(
    pdf: PdfPages,
    *,
    covariance: str,
    covariance_display: str,
    chains: dict[str, dict[str, Any]],
    ranges: dict[str, tuple[float, float]],
) -> None:
    """绘制一个 covariance choice 的四参数 corner 页面。"""

    figure, axes = plt.subplots(4, 4, figsize=(10.5, 9.7), squeeze=False)
    for i, parameter in enumerate(PARAMETERS):
        values = []
        for variant, _, _ in VARIANTS:
            entry = chains[f"{covariance}_{variant}"]
            index = PARAMETERS.index(parameter)
            if entry["chain"].shape[1] > index:
                values.append(entry["chain"][:, index])
        if values:
            bins = np.linspace(*ranges[parameter], 55)
            for variant, color, _ in VARIANTS:
                entry = chains[f"{covariance}_{variant}"]
                index = PARAMETERS.index(parameter)
                if entry["chain"].shape[1] <= index:
                    continue
                axes[i, i].hist(
                    entry["chain"][:, index],
                    bins=bins,
                    density=True,
                    histtype="step",
                    lw=1.5,
                    color=color,
                )
        axes[i, i].set_xlim(*ranges[parameter])
        axes[i, i].set_yticks([])
        axes[i, i].tick_params(labelsize=8)
        axes[i, i].set_xlabel(LABELS[parameter], fontsize=10)
        if parameter == "fNL":
            axes[i, i].axvline(0.0, color="0.50", lw=0.8, ls="--", zorder=0)
    for i in range(4):
        for j in range(4):
            if j > i:
                axes[i, j].set_axis_off()
                continue
            if i == j:
                continue
            xpar, ypar = PARAMETERS[j], PARAMETERS[i]
            axis = axes[i, j]
            axis.set_xlim(*ranges[xpar])
            axis.set_ylim(*ranges[ypar])
            zorder = 2
            for variant, color, _ in VARIANTS:
                entry = chains[f"{covariance}_{variant}"]
                ix, iy = PARAMETERS.index(xpar), PARAMETERS.index(ypar)
                if entry["chain"].shape[1] <= max(ix, iy):
                    continue
                draw_contour(
                    axis,
                    entry["chain"][:, ix],
                    entry["chain"][:, iy],
                    color=color,
                    xlim=ranges[xpar],
                    ylim=ranges[ypar],
                    zorder=zorder,
                )
                zorder += 2
            if xpar == "fNL":
                axis.axvline(0.0, color="0.50", lw=0.7, ls="--", zorder=0)
            if ypar == "fNL":
                axis.axhline(0.0, color="0.50", lw=0.7, ls="--", zorder=0)
            axis.tick_params(labelsize=8)
            if i == 3:
                axis.set_xlabel(LABELS[xpar], fontsize=10)
            else:
                axis.set_xticklabels([])
            if j == 0:
                axis.set_ylabel(LABELS[ypar], fontsize=10)
            else:
                axis.set_yticklabels([])
    handles, labels = [], []
    for variant, color, display in VARIANTS:
        entry = chains[f"{covariance}_{variant}"]
        text = display + "\n" + rf"$f_{{\rm NL}}={interval_text(entry, 'fNL')}$"
        handles.append(plt.Line2D([], [], color=color, lw=2.0))
        labels.append(text)
    figure.legend(
        handles,
        labels,
        loc="upper center",
        bbox_to_anchor=(0.54, 0.995),
        ncol=3,
        frameon=False,
        fontsize=11.0,
        handletextpad=0.5,
        columnspacing=1.0,
    )
    figure.suptitle(
        "Task43 RSD lightcone: all-parameter contours "
        f"({covariance_display} covariance)",
        fontsize=12.0,
        y=1.02,
    )
    figure.text(
        0.985,
        0.005,
        "sn0 is unavailable for the xi0+xi2-only chain",
        ha="right",
        va="bottom",
        fontsize=8.2,
        color="0.35",
    )
    figure.subplots_adjust(left=0.08, right=0.985, bottom=0.065, top=0.91, wspace=0.06, hspace=0.06)
    pdf.savefig(figure, bbox_inches="tight", pad_inches=0.08)
    plt.close(figure)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mcmc-root", type=Path, default=MCMC_ROOT_DEFAULT)
    parser.add_argument("--nmock", type=int, required=True)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    root = Path(args.mcmc_root)
    ezlabel = f"ezmock{int(args.nmock)}"
    pages = ((ezlabel, f"EZmock-{int(args.nmock)} empirical"), ("jaxpower", "jaxpower analytic"))
    output = MEETING_DIR / (
        f"task43_{ezlabel}_vs_jaxpower_rsd_P02xi02_joint_kmax0p08_smin50_baomask80_120_allparams_v1.pdf"
    )
    output_json = output.with_suffix(".json")
    if not args.force and (output.exists() or output_json.exists()):
        raise FileExistsError(f"immutable output exists: {output}")
    chains = {
        f"{covariance}_{variant}": load_chain(root, f"{covariance}_{variant}")
        for covariance, _ in pages
        for variant, _, _ in VARIANTS
    }
    ranges: dict[str, tuple[float, float]] = {}
    for index, parameter in enumerate(PARAMETERS):
        values = [
            entry["chain"][:, index]
            for entry in chains.values()
            if entry["chain"].shape[1] > index
        ]
        if not values:
            raise RuntimeError(f"no chain contains parameter {parameter}")
        ranges[parameter] = make_range(np.concatenate(values), parameter)
    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "DejaVu Sans", "Liberation Sans"],
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "font.size": 9.0,
            "axes.linewidth": 1.0,
            "xtick.direction": "in",
            "ytick.direction": "in",
            "xtick.top": True,
            "ytick.right": True,
        }
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(f".{output.name}.{os.getpid()}.tmp.pdf")
    with PdfPages(temporary) as pdf:
        for covariance, covariance_display in pages:
            corner_page(pdf, covariance=covariance, covariance_display=covariance_display, chains=chains, ranges=ranges)
    temporary.replace(output)
    if output.read_bytes()[:5] != b"%PDF-":
        raise RuntimeError("output is not a PDF")
    constraints: dict[str, Any] = {}
    for tag, entry in chains.items():
        posterior = entry["summary"]["posterior"]
        map_theta = np.asarray(entry["summary"]["map_theta"], dtype="f8")
        constraints[tag] = {
            "chain_npz": str(entry["path"]),
            "gates": entry["summary"]["gates"],
            "parameters": {
                parameter: {
                    "maximum_likelihood": float(map_theta[index]),
                    **{key: posterior[parameter][key] for key in ("q16", "q50", "q84", "sigma68")},
                }
                for index, parameter in enumerate(PARAMETERS)
                if index < entry["chain"].shape[1]
            },
        }
    atomic_write_json(
        output_json,
        {
            "task": "task43_plot_ezmock_covariance_mcmc_vs_jaxpower_allparams",
            "status": "pass",
            "mcmc_root": str(root),
            "ezmock_realizations": int(args.nmock),
            "pages": [f"EZmock-{int(args.nmock)} empirical covariance", "jaxpower analytic covariance"],
            "contour_parameters": list(PARAMETERS),
            "missing_parameter_policy": "xi02 chain has no sn0; sn0 panels use P02 and joint only",
            "shared_axis_ranges": {key: list(value) for key, value in ranges.items()},
            "data_contract": "P0 13 + P2 9 + BAO-masked xi0 26 + xi2 26; Abacus lightcone x25 mean",
            "constraints": constraints,
            "output_pdf": str(output),
            "output_pdf_sha256": sha256_file(output),
        },
    )
    print(json.dumps({"status": "pass", "output": str(output), "sha256": sha256_file(output)}, sort_keys=True))


if __name__ == "__main__":
    main()
