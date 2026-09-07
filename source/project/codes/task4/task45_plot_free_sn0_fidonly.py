#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
代码大纲：

1. 读取 Task45 fixed-sn0 主结果 manifest，以及 free-sn0 run 已经完成的
   fid summary JSON。
2. 将 fid 的 P(k) 与 4 个 2PCF r-range 整理成同一张 forest plot：
   - 蓝色点：free-sn0 posterior median +/- Percival 修正后的 1 sigma；
   - 灰色点：fixed-sn0 主结果的参考值，只在 fnl_loc/b1/sigmas 三列显示；
   - 第四列单独显示 free-sn0 的 sn0 posterior，并标出 sn0=0。
3. 写出 fid-only PDF、对比 CSV，以及一个小 manifest，供后续 AI 快速定位。

这个脚本只读现有 Task45 结果，不会覆盖全量 all-tags free-sn0 图。
"""

from __future__ import annotations

import csv
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


PROJECT_ROOT = Path("/pscratch/sd/l/lzy/PNG-2PCF-model-to-observe")
FIXED_MANIFEST = PROJECT_ROOT / "outputs" / "task4_outputs" / "quijote_ultranest" / "task45_quijote_ultranest_manifest.json"
FREE_ROOT = PROJECT_ROOT / "outputs" / "task4_outputs" / "quijote_ultranest_free_sn0"
PLOT_ROOT = PROJECT_ROOT / "plots" / "task4" / "quijote_ultranest_free_sn0"
IMPORTANT_DIR = PROJECT_ROOT / "plots" / "task4" / "important_4p1_4p2"

OUT_PDF = PLOT_ROOT / "task45_quijote_ultranest_free_sn0_fidonly_constraints.pdf"
OUT_CSV = FREE_ROOT / "task45_quijote_ultranest_free_sn0_fidonly_comparison.csv"
OUT_MANIFEST = FREE_ROOT / "task45_quijote_ultranest_free_sn0_fidonly_plot_manifest.json"
IMPORTANT_LINK = IMPORTANT_DIR / "4p1_diagnostic_task45_quijote_ultranest_free_sn0_fidonly_constraints.pdf"

CASE_ORDER = ["pk_binavg", "xi_r50_350", "xi_r60_350", "xi_r80_350", "xi_r100_350"]
CASE_LABELS = {
    "pk_binavg": "P(k) BinAvgFit",
    "xi_r50_350": "2PCF 55-345",
    "xi_r60_350": "2PCF 65-345",
    "xi_r80_350": "2PCF 85-345",
    "xi_r100_350": "2PCF 105-345",
}
PARAMS = ["fnl_loc", "b1", "sigmas", "sn0"]
PARAM_LABELS = {
    "fnl_loc": r"$f_{\mathrm{NL}}^{\mathrm{loc}}$",
    "b1": r"$b_1$",
    "sigmas": r"$\sigma_s\,[h^{-1}{\rm Mpc}]$",
    "sn0": r"$s_{n,0}\,[(h^{-1}{\rm Mpc})^3]$",
}


def load_json(path: Path) -> dict:
    """读取 JSON 文件，并在缺失时给出清楚的报错路径。"""
    if not path.exists():
        raise FileNotFoundError(path)
    return json.loads(path.read_text(encoding="utf-8"))


def load_fixed_fid() -> dict[str, dict]:
    """
    从 fixed-sn0 主 manifest 中读取 fid 的 5 个 case。

    输出：
    - 字典键是 case 名，值是对应 summary 字典。
    """
    manifest = load_json(FIXED_MANIFEST)
    rows = {}
    for item in manifest["summaries"]:
        if item["tag"] == "fid" and item["case"] in CASE_ORDER:
            rows[item["case"]] = item
    missing = [case for case in CASE_ORDER if case not in rows]
    if missing:
        raise RuntimeError(f"fixed fid cases missing: {missing}")
    return rows


def load_free_fid() -> dict[str, dict]:
    """
    读取 free-sn0 run 中已经完成的 fid summary。

    输出：
    - 字典键是 case 名，值是对应 summary 字典。
    """
    rows = {}
    for case in CASE_ORDER:
        rows[case] = load_json(FREE_ROOT / f"task45_free_sn0_fid_{case}_summary.json")
    return rows


def param_interval(summary: dict, name: str) -> tuple[float, float, float]:
    """
    取 posterior median 和 Percival 修正后的误差。

    输出：
    - median, err_low_percival, err_high_percival。
    """
    item = summary["parameters"][name]
    return float(item["median"]), float(item["err_low_percival"]), float(item["err_high_percival"])


def write_comparison_csv(fixed: dict[str, dict], free: dict[str, dict]) -> None:
    """写出 fid-only fixed-vs-free 参数对比表。"""
    rows = []
    for case in CASE_ORDER:
        for name in PARAMS:
            free_med, free_elo, free_ehi = param_interval(free[case], name)
            row = {
                "tag": "fid",
                "case": case,
                "parameter": name,
                "free_sn0_median": free_med,
                "free_sn0_err_low_percival": free_elo,
                "free_sn0_err_high_percival": free_ehi,
                "free_sn0_ncall": free[case]["ultranest"]["ncall"],
            }
            if name in fixed[case]["parameters"]:
                fixed_med, fixed_elo, fixed_ehi = param_interval(fixed[case], name)
                row.update(
                    {
                        "fixed_sn0_median": fixed_med,
                        "fixed_sn0_err_low_percival": fixed_elo,
                        "fixed_sn0_err_high_percival": fixed_ehi,
                        "free_minus_fixed": free_med - fixed_med,
                    }
                )
            else:
                row.update(
                    {
                        "fixed_sn0_median": "",
                        "fixed_sn0_err_low_percival": "",
                        "fixed_sn0_err_high_percival": "",
                        "free_minus_fixed": "",
                    }
                )
            rows.append(row)

    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    with OUT_CSV.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def plot_fidonly(fixed: dict[str, dict], free: dict[str, dict]) -> None:
    """生成 fid-only fixed-sn0 vs free-sn0 forest plot。"""
    PLOT_ROOT.mkdir(parents=True, exist_ok=True)
    IMPORTANT_DIR.mkdir(parents=True, exist_ok=True)

    y = np.arange(len(CASE_ORDER))[::-1]
    fig, axes = plt.subplots(1, 4, figsize=(18.5, 5.7), sharey=True, gridspec_kw={"wspace": 0.20})
    fig.patch.set_facecolor("#f7f7f5")

    free_color = "#2f6fba"
    fixed_color = "#7a7a7a"
    marker_by_case = {"pk_binavg": "s"}

    for ax, pname in zip(axes, PARAMS):
        ax.set_facecolor("#fbfbfa")
        lo_all, hi_all = [], []
        for i, case in enumerate(CASE_ORDER):
            yi = y[i]
            if i % 2 == 0:
                ax.axhspan(yi - 0.45, yi + 0.45, color="#ececea", alpha=0.45, zorder=0)

            free_med, free_elo, free_ehi = param_interval(free[case], pname)
            lo_all.append(free_med - free_elo)
            hi_all.append(free_med + free_ehi)
            ax.errorbar(
                free_med,
                yi - 0.10,
                xerr=np.array([[free_elo], [free_ehi]]),
                fmt=marker_by_case.get(case, "o"),
                ms=7.0,
                mfc="white",
                mec=free_color,
                mew=2.0,
                ecolor=free_color,
                elinewidth=1.8,
                capsize=3.2,
                capthick=1.5,
                zorder=3,
            )

            if pname in fixed[case]["parameters"]:
                fixed_med, fixed_elo, fixed_ehi = param_interval(fixed[case], pname)
                lo_all.append(fixed_med - fixed_elo)
                hi_all.append(fixed_med + fixed_ehi)
                ax.errorbar(
                    fixed_med,
                    yi + 0.14,
                    xerr=np.array([[fixed_elo], [fixed_ehi]]),
                    fmt=marker_by_case.get(case, "o"),
                    ms=5.8,
                    mfc="white",
                    mec=fixed_color,
                    mew=1.5,
                    ecolor=fixed_color,
                    elinewidth=1.3,
                    capsize=2.7,
                    capthick=1.2,
                    alpha=0.78,
                    zorder=2,
                )

        if pname == "sn0":
            ax.axvline(0.0, color="#4a4a4a", lw=1.2, ls="--", alpha=0.65)
        ax.set_title(PARAM_LABELS[pname], pad=10)
        ax.set_xlabel("posterior median +/- 1 sigma")
        if lo_all:
            xmin, xmax = float(np.nanmin(lo_all)), float(np.nanmax(hi_all))
            span = max(xmax - xmin, 1.0e-6)
            ax.set_xlim(xmin - 0.10 * span, xmax + 0.10 * span)
        ax.grid(axis="x", color="#c9c9c9", lw=0.8, alpha=0.65)
        ax.tick_params(axis="y", length=0)

    axes[0].set_yticks(y)
    axes[0].set_yticklabels([CASE_LABELS[case] for case in CASE_ORDER])
    for ax in axes[1:]:
        ax.tick_params(labelleft=False)

    handles = [
        axes[0].plot([], [], "o", ms=7, mfc="white", mec=free_color, mew=2, label="free sn0")[0],
        axes[0].plot([], [], "o", ms=6, mfc="white", mec=fixed_color, mew=1.5, label="fixed sn0=0")[0],
        axes[0].plot([], [], "s", ms=7, mfc="white", mec="#555555", mew=1.8, label="P(k)")[0],
        axes[0].plot([], [], "o", ms=7, mfc="white", mec="#555555", mew=1.8, label="2PCF")[0],
    ]
    fig.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.54, 0.965), ncol=4, frameon=False)
    fig.suptitle("fid only: UltraNest posterior constraints, fixed-sn0 vs free-sn0", x=0.54, y=0.995, fontsize=15, fontweight="bold")
    fig.text(
        0.54,
        0.035,
        "P(k): BinAvgFit, kcen <= 0.08 h/Mpc.  2PCF: FullDiscrete, rmax = 350 Mpc/h.  free-sn0 prior [-1, 1].",
        ha="center",
        va="center",
        fontsize=10.5,
        color="#4d4d4d",
    )
    fig.subplots_adjust(left=0.18, right=0.985, top=0.84, bottom=0.16)
    fig.savefig(OUT_PDF)
    plt.close(fig)

    IMPORTANT_LINK.unlink(missing_ok=True)
    IMPORTANT_LINK.symlink_to(Path("../quijote_ultranest_free_sn0") / OUT_PDF.name)


def write_manifest() -> None:
    """写出这个 fid-only 图的机器可读 manifest。"""
    manifest = {
        "task": "task45_quijote_ultranest_free_sn0_fidonly_plot",
        "status": "done",
        "inputs": {
            "fixed_manifest": str(FIXED_MANIFEST),
            "free_summary_root": str(FREE_ROOT),
            "tag": "fid",
            "cases": CASE_ORDER,
        },
        "outputs": {
            "pdf": str(OUT_PDF),
            "comparison_csv": str(OUT_CSV),
            "important_link": str(IMPORTANT_LINK),
        },
    }
    OUT_MANIFEST.write_text(json.dumps(manifest, indent=2), encoding="utf-8")


def main() -> None:
    """主入口：读结果、写 CSV、画 PDF、写 manifest。"""
    fixed = load_fixed_fid()
    free = load_free_fid()
    write_comparison_csv(fixed, free)
    plot_fidonly(fixed, free)
    write_manifest()
    print(f"[write] {OUT_PDF}")
    print(f"[write] {OUT_CSV}")
    print(f"[write] {OUT_MANIFEST}")
    print(f"[link] {IMPORTANT_LINK}")


if __name__ == "__main__":
    main()
