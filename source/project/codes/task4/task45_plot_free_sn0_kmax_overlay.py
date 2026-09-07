#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
代码大纲（执行逻辑关系）：

1. 读取 Task45 free-sn0 主结果 manifest，其中 P(k) 使用默认 `kmax_fit=0.08`，
   2PCF 使用同一张主图里的四个 rlist。
2. 读取两个只重跑 P(k) BinAvgFit 的对照 manifest：
   - `kmax_fit=0.06`
   - `kmax_fit=0.10`
3. 对每个 Quijote tag 组织 forest plot 行：
   - P(k), k<=0.06
   - P(k), k<=0.08
   - P(k), k<=0.10
   - 原来的 2PCF rlist
4. 写出 PDF 与一个轻量 manifest，作为 `sn0` 自由时 P(k) kmax 扫描的可读图。
"""

from __future__ import annotations

import json
import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


PROJECT_ROOT = Path("/pscratch/sd/l/lzy/PNG-2PCF-model-to-observe")
OUTPUT_ROOT = PROJECT_ROOT / "outputs" / "task4_outputs" / "quijote_ultranest_free_sn0_kmax_overlay"
PLOT_ROOT = PROJECT_ROOT / "plots" / "task4" / "quijote_ultranest_free_sn0"

DEFAULT_BASE_MANIFEST = (
    PROJECT_ROOT
    / "outputs"
    / "task4_outputs"
    / "quijote_ultranest_free_sn0"
    / "task45_quijote_ultranest_free_sn0_manifest.json"
)
DEFAULT_KMAX_MANIFESTS = {
    "0.06": (
        PROJECT_ROOT
        / "outputs"
        / "task4_outputs"
        / "quijote_ultranest_free_sn0_kmax0p06"
        / "task45_quijote_ultranest_free_sn0_kmax0p06_manifest.json"
    ),
    "0.10": (
        PROJECT_ROOT
        / "outputs"
        / "task4_outputs"
        / "quijote_ultranest_free_sn0_kmax0p10"
        / "task45_quijote_ultranest_free_sn0_kmax0p10_manifest.json"
    ),
}

DEFAULT_OUT_PDF = PLOT_ROOT / "task45_quijote_ultranest_free_sn0_alltags_constraints_with_pk_kmax_scan.pdf"
DEFAULT_OUT_MANIFEST = OUTPUT_ROOT / "task45_quijote_ultranest_free_sn0_kmax_overlay_manifest.json"

TAG_ORDER = ["fid", "LCp50", "LCp100"]
TAG_FNL_VALUES = {"fid": "0", "LCp50": "50", "LCp100": "100"}
PARAM_NAMES = ["fnl_loc", "b1", "sigmas", "sn0"]
PARAM_LABELS = {
    "fnl_loc": r"$f_{\mathrm{NL}}^{\mathrm{loc}}$",
    "b1": r"$b_1$",
    "sigmas": r"$\sigma_s\,[h^{-1}{\rm Mpc}]$",
    "sn0": r"$s_{n,0}\,[(h^{-1}{\rm Mpc})^3]$",
}
XI_CASE_LABELS = {
    "xi_r50_350": "2PCF 55-345",
    "xi_r60_350": "2PCF 65-345",
    "xi_r80_350": "2PCF 85-345",
    "xi_r100_350": "2PCF 105-345",
}
XI_CASE_LABELS_WITH_UNITS = {
    key: f"{label} Mpc/h" for key, label in XI_CASE_LABELS.items()
}
MEETING_STYLE = {
    "title": 24,
    "axis_label": 20,
    "tick": 20,
    "row_label": 20,
    "annotation": 15,
    "legend": 14,
    "suptitle": 20,
    "tag_label": 22,
}


def load_manifest(path: Path) -> dict:
    """读取 manifest，并在文件不存在时给出明确报错。"""
    if not path.exists():
        raise FileNotFoundError(f"missing manifest: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def index_summaries(manifest: dict) -> dict[tuple[str, str], dict]:
    """
    把 manifest 内的 summaries 按 `(tag, case)` 建索引。

    Task45 manifest 已经内嵌 summary；这里不再额外扫描 outputs 目录。
    """
    out: dict[tuple[str, str], dict] = {}
    for item in manifest["summaries"]:
        out[(item["tag"], item["case"])] = item
    return out


def tagged_row(summary: dict, row_label: str, row_kind: str, kmax_label: str | None = None) -> dict:
    """给一个 summary 附加画图标签。"""
    row = dict(summary)
    row["row_label"] = row_label
    row["row_kind"] = row_kind
    row["kmax_label"] = kmax_label
    return row


def parse_args() -> argparse.Namespace:
    """解析 overlay 输入/输出路径。"""
    parser = argparse.ArgumentParser(description="Plot Task45 free-sn0 P(k) kmax overlay")
    parser.add_argument("--base-manifest", type=Path, default=DEFAULT_BASE_MANIFEST, help="manifest for kmax=0.08 full run")
    parser.add_argument("--kmax006-manifest", type=Path, default=DEFAULT_KMAX_MANIFESTS["0.06"], help="manifest for P(k) kmax=0.06")
    parser.add_argument("--kmax010-manifest", type=Path, default=DEFAULT_KMAX_MANIFESTS["0.10"], help="manifest for P(k) kmax=0.10")
    parser.add_argument("--out-pdf", type=Path, default=DEFAULT_OUT_PDF, help="output PDF path")
    parser.add_argument("--out-manifest", type=Path, default=DEFAULT_OUT_MANIFEST, help="output manifest path")
    parser.add_argument(
        "--params",
        default=",".join(PARAM_NAMES),
        help="comma-separated parameter columns to plot, e.g. fnl_loc,b1",
    )
    parser.add_argument(
        "--pk-kmax",
        default="0.06,0.08,0.10",
        help="comma-separated P(k) kmax rows to keep from 0.06,0.08,0.10",
    )
    parser.add_argument(
        "--tags",
        default=",".join(TAG_ORDER),
        help="comma-separated Quijote tags to plot, e.g. LCp50",
    )
    parser.add_argument(
        "--meeting-labels",
        action="store_true",
        help="omit fid/LCp tag names and annotate each tag block inside the plot as fNL=...",
    )
    return parser.parse_args()


def parse_csv_choice(raw: str, allowed: list[str] | set[str], label: str) -> list[str]:
    """解析逗号分隔的选择，并检查是否属于允许集合。"""
    allowed_set = set(allowed)
    choices = [item.strip() for item in raw.split(",") if item.strip()]
    if not choices:
        raise ValueError(f"{label} cannot be empty")
    bad = [item for item in choices if item not in allowed_set]
    if bad:
        raise ValueError(f"invalid {label}: {bad}; allowed={sorted(allowed_set)}")
    return choices


def build_rows(
    base_manifest_path: Path,
    kmax_manifest_paths: dict[str, Path],
    pk_kmax: list[str],
    tags: list[str],
    meeting_labels: bool = False,
) -> tuple[list[dict], dict]:
    """
    组织最终 forest plot 的行，并返回用于写 manifest 的输入元数据。
    """
    base_manifest = load_manifest(base_manifest_path)
    base = index_summaries(base_manifest)
    kmax_indices = {label: index_summaries(load_manifest(path)) for label, path in kmax_manifest_paths.items()}

    rows: list[dict] = []
    for tag in tags:
        for kmax_label in pk_kmax:
            if kmax_label == "0.08":
                summary = base[(tag, "pk_binavg")]
            else:
                summary = kmax_indices[kmax_label][(tag, "pk_binavg")]
            row_label = f"P(k) k<={kmax_label} h/Mpc" if meeting_labels else f"{tag}  P(k) k<={kmax_label}"
            rows.append(tagged_row(summary, row_label, "pk", kmax_label))
        for case, label in XI_CASE_LABELS.items():
            row_label = XI_CASE_LABELS_WITH_UNITS[case] if meeting_labels else f"{tag}  {label}"
            rows.append(tagged_row(base[(tag, case)], row_label, "xi", None))
        rows.append({"case": "gap", "tag": tag})
    rows.pop()

    meta = {
        "base_manifest": str(base_manifest_path),
        "kmax_manifests": {label: str(path) for label, path in kmax_manifest_paths.items()},
        "pk_kmax": pk_kmax,
        "tags": tags,
        "meeting_labels": meeting_labels,
        "row_count": len([row for row in rows if row.get("case") != "gap"]),
    }
    return rows, meta


def format_constraint_text(pname: str, med: float, elo: float, ehi: float) -> str:
    """Format a compact median/error label for point annotations."""
    if pname == "fnl_loc":
        return rf"$\mathbf{{{med:.1f}}}$ -{elo:.1f}/+{ehi:.1f}"
    if pname == "b1":
        return rf"$\mathbf{{{med:.3f}}}$ -{elo:.3f}/+{ehi:.3f}"
    if pname == "sigmas":
        return rf"$\mathbf{{{med:.2f}}}$ -{elo:.2f}/+{ehi:.2f}"
    return rf"$\mathbf{{{med:.2g}}}$ -{elo:.2g}/+{ehi:.2g}"


def plot_rows(rows: list[dict], out_pdf: Path, param_names: list[str], meeting_labels: bool = False) -> None:
    """画包含 P(k) kmax 扫描的 free-sn0 all-tags forest plot。"""
    y = np.arange(len(rows))[::-1]
    fig_width = 6.0 + 4.6 * len(param_names)
    fig_height = max(6.8 if meeting_labels else 6.2, (0.78 if meeting_labels else 0.48) * len(rows) + 1.8)
    title_size = MEETING_STYLE["title"] if meeting_labels else None
    axis_label_size = MEETING_STYLE["axis_label"] if meeting_labels else None
    tick_size = MEETING_STYLE["tick"] if meeting_labels else None
    row_label_size = MEETING_STYLE["row_label"] if meeting_labels else None
    annotation_size = MEETING_STYLE["annotation"] if meeting_labels else 9
    legend_size = MEETING_STYLE["legend"] if meeting_labels else None
    suptitle_size = MEETING_STYLE["suptitle"] if meeting_labels else 17
    fig, axes = plt.subplots(
        1,
        len(param_names),
        figsize=(fig_width, fig_height),
        sharey=True,
        gridspec_kw={"wspace": 0.16},
    )
    axes = np.atleast_1d(axes)
    fig.patch.set_facecolor("white")
    colors = {"fid": "#3c6e9f", "LCp50": "#bf5b2f", "LCp100": "#417a50"}
    markers = {"0.06": "^", "0.08": "s", "0.10": "D", None: "o"}
    tags_present = [tag for tag in TAG_ORDER if any(row.get("tag") == tag and row.get("case") != "gap" for row in rows)]

    for ax, pname in zip(axes, param_names):
        ax.set_facecolor("white")
        lo_all: list[float] = []
        hi_all: list[float] = []
        for i, row in enumerate(rows):
            yi = y[i]
            if row.get("case") == "gap":
                ax.axhline(yi, color="#9e9e9e", lw=0.9, alpha=0.6, zorder=0)
                continue
            params = row["parameters"][pname]
            med = float(params["median"])
            elo = float(params["err_low_percival"])
            ehi = float(params["err_high_percival"])
            lo_all.append(med - elo)
            hi_all.append(med + ehi)
            ax.errorbar(
                med,
                yi,
                xerr=np.array([[elo], [ehi]]),
                fmt=markers[row.get("kmax_label")],
                ms=7.0,
                mfc="white",
                mec=colors[row["tag"]],
                mew=2.0,
                ecolor=colors[row["tag"]],
                elinewidth=1.8,
                capsize=3.6,
                capthick=1.5,
                alpha=0.95,
                zorder=3,
            )
            if meeting_labels:
                ax.annotate(
                    format_constraint_text(pname, med, elo, ehi),
                    xy=(med, yi),
                    xytext=(8, 7),
                    textcoords="offset points",
                    ha="left",
                    va="bottom",
                    fontsize=annotation_size,
                    color="#111111",
                    annotation_clip=False,
                    clip_on=False,
                    zorder=6,
                )
        ax.set_title(PARAM_LABELS[pname], pad=12, fontsize=title_size)
        if meeting_labels:
            ax.set_xlabel("")
        else:
            ax.set_xlabel("posterior median +/- 1 sigma", fontsize=axis_label_size)
        if lo_all:
            xmin, xmax = float(np.nanmin(lo_all)), float(np.nanmax(hi_all))
            span = xmax - xmin
            if span <= 0:
                span = max(abs(xmax), 1.0)
            left_pad = 0.10 if meeting_labels else 0.08
            right_pad = 0.36 if meeting_labels else 0.08
            ax.set_xlim(xmin - left_pad * span, xmax + right_pad * span)
        if len(rows):
            ax.set_ylim(float(np.min(y)) - 0.70, float(np.max(y)) + (0.95 if meeting_labels else 0.70))
        ax.grid(axis="x", color="#d8d8d8", lw=0.8, alpha=0.75)
        ax.tick_params(axis="x", labelsize=tick_size)
        ax.tick_params(axis="y", length=0, labelsize=tick_size)

    if meeting_labels:
        from matplotlib.transforms import blended_transform_factory

        transform = blended_transform_factory(axes[0].transAxes, axes[0].transData)
        for tag in tags_present:
            tag_y = [y[i] for i, row in enumerate(rows) if row.get("tag") == tag and row.get("case") != "gap"]
            if not tag_y:
                continue
            y_text = max(tag_y) - 0.35 if len(tags_present) == 1 else 0.5 * (min(tag_y) + max(tag_y))
            axes[0].text(
                0.965,
                y_text,
                rf"$f_{{\rm NL}}={TAG_FNL_VALUES[tag]}$",
                transform=transform,
                ha="right",
                va="center",
                fontsize=MEETING_STYLE["tag_label"] if meeting_labels else 19,
                fontweight="bold",
                color=colors[tag],
                bbox={"boxstyle": "round,pad=0.22", "fc": "white", "ec": colors[tag], "lw": 1.0, "alpha": 0.90},
                zorder=5,
            )

    labels = ["" if row.get("case") == "gap" else row["row_label"] for row in rows]
    axes[0].set_yticks(y)
    axes[0].set_yticklabels(labels, fontsize=row_label_size)
    for ax in axes[1:]:
        ax.tick_params(labelleft=False, labelsize=tick_size)

    handles = []
    if not (meeting_labels and len(tags_present) == 1):
        for tag in tags_present:
            color = colors[tag]
            label = rf"$f_{{\rm NL}}={TAG_FNL_VALUES[tag]}$" if meeting_labels else tag
            handles.append(axes[0].plot([], [], "o", ms=8, mfc="white", mec=color, mew=2, label=label)[0])
    pk_kmax_seen = sorted({row.get("kmax_label") for row in rows if row.get("kmax_label") is not None})
    for kmax_label in pk_kmax_seen:
        handles.append(
            axes[0].plot(
                [],
                [],
                markers[kmax_label],
                ms=8,
                mfc="white",
                mec="#555555",
                mew=2,
                label=f"P(k) k<={kmax_label} h/Mpc",
            )[0]
        )
    handles.append(axes[0].plot([], [], "o", ms=8, mfc="white", mec="#555555", mew=2, label="2PCF")[0])
    fig.legend(
        handles=handles,
        loc="upper center",
        bbox_to_anchor=(0.56, 0.965),
        ncol=min(len(handles), 6),
        frameon=False,
        fontsize=legend_size,
    )
    pk_title = f"P(k) k<={pk_kmax_seen[0]}" if len(pk_kmax_seen) == 1 else "P(k) kmax scan"
    if meeting_labels and len(tags_present) == 1:
        title_prefix = rf"Quijote $f_{{\rm NL}}={TAG_FNL_VALUES[tags_present[0]]}$"
    elif meeting_labels:
        title_prefix = "Quijote local-PNG"
    else:
        title_prefix = "Quijote local-PNG tags"
    if not meeting_labels:
        fig.suptitle(
            f"{title_prefix}: UltraNest posterior constraints (free sn0; {pk_title})",
            x=0.56,
            y=0.995,
            fontsize=suptitle_size,
            fontweight="bold",
        )
    pk_note = (
        f"P(k): BinAvgFit with kcen cut {pk_kmax_seen[0]} h/Mpc."
        if len(pk_kmax_seen) == 1
        else f"P(k): BinAvgFit with kcen cuts {'/'.join(pk_kmax_seen)} h/Mpc."
    )
    if not meeting_labels:
        fig.text(
            0.56,
            0.035,
            f"{pk_note}  2PCF: original FullDiscrete rows, rmax = 350 Mpc/h.",
            ha="center",
            va="center",
            fontsize=11,
            color="#4d4d4d",
        )
    top_margin = 0.80 if meeting_labels and len(tags_present) == 1 else (0.89 if meeting_labels else 0.90)
    fig.subplots_adjust(left=0.34 if meeting_labels else 0.27, right=0.975, top=top_margin, bottom=0.18 if meeting_labels else 0.085)
    out_pdf.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_pdf)
    plt.close(fig)


def main() -> None:
    """主程序入口。"""
    args = parse_args()
    param_names = parse_csv_choice(args.params, PARAM_NAMES, "params")
    pk_kmax = parse_csv_choice(args.pk_kmax, {"0.06", "0.08", "0.10"}, "pk-kmax")
    tags = parse_csv_choice(args.tags, TAG_ORDER, "tags")
    args.out_manifest.parent.mkdir(parents=True, exist_ok=True)
    rows, meta = build_rows(
        args.base_manifest,
        {"0.06": args.kmax006_manifest, "0.10": args.kmax010_manifest},
        pk_kmax,
        tags,
        args.meeting_labels,
    )
    plot_rows(rows, args.out_pdf, param_names, args.meeting_labels)
    manifest = {
        "task": "task45_free_sn0_pk_kmax_overlay",
        "status": "done",
        "outputs": {"pdf": str(args.out_pdf), "manifest": str(args.out_manifest)},
        "inputs": {**meta, "params": param_names},
    }
    args.out_manifest.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"[write] {args.out_pdf}")
    print(f"[write] {args.out_manifest}")


if __name__ == "__main__":
    main()
