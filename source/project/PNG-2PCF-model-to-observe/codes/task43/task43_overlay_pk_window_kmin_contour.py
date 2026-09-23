#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""生成 Task43 的两张旧 formal-GIC/geometry-only covariance A/B 图。

2026-07-12 用户指定的 RascalC public path 已改由
``task43_plot_ric_rascalc_pk_vs_2pcf.py`` 维护，并使用两条 radial-RIC 长链。
因此本脚本默认输出加上 ``formalgic_geometryonly_legacy`` 标签，避免以后
误运行时覆盖新的 RIC 主图；历史显式路径参数仍可用于复现旧结果。

代码执行大纲
============
1. 读取唯一的一条 P(k) posterior；它使用 jaxpower Gaussian
   survey-window covariance 和 mother-box theory cutoff。
2. 分别读取两条完全相同 2PCF 模型口径的 posterior：
   一条使用 jaxpower RR-deconvolved covariance，另一条使用
   mock-calibrated FullDiscrete RascalC covariance。
3. 使用三条 posterior 的并集一次性确定 fNL 和 b1 坐标范围，保证两张图
   具有完全相同的坐标轴、颜色和 P(k) 参考链。
4. 分别输出“双方都是 jaxpower covariance”和“2PCF 换成 RascalC
   covariance”两张 PDF。
5. 将输入、posterior 数值、共享坐标范围和输出路径写入 outputs 下的一个
   JSON；plots/task43 中只新增用户要求的 PDF，不写 plot-side JSON。

这两张图只切换 2PCF covariance，因此可直接用于检查 covariance 选择对
P(k)-2PCF posterior 对比的影响。
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


PROJECT_ROOT = Path("/pscratch/sd/l/lzy/PNG-2PCF-model-to-observe")
PLOT_DIR = PROJECT_ROOT / "plots" / "task43"
SUMMARY_DIR = PROJECT_ROOT / "outputs" / "task43_outputs" / "summary"

DEFAULT_2PCF_JAXPOWER = (
    PROJECT_ROOT
    / "outputs/task43_outputs/fits/"
    "full25_mean_s50_350_ds10_shellavg_jaxpowerRRdeconv_k0001_"
    "L2000_p1p0_formalgic_mcmc5x_nsub200000/"
    "task43_minimal_closure_mcmc_summary.json"
)
DEFAULT_2PCF_JAXPOWER_SAMPLES = (
    DEFAULT_2PCF_JAXPOWER.parent / "task43_mcmc_formal_gic_samples.npz"
)
DEFAULT_2PCF_RASCALC = (
    PROJECT_ROOT
    / "outputs/task43_outputs/fits/"
    "full25_mean_s50_350_ds10_shellavg_rascalcFullDiscrete_nran300k_"
    "nloop512_alphaMockCal_L2000_p1p0_formalgic_mcmc5x_nsub200000/"
    "task43_minimal_closure_mcmc_summary.json"
)
DEFAULT_2PCF_RASCALC_SAMPLES = (
    DEFAULT_2PCF_RASCALC.parent / "task43_mcmc_formal_gic_samples.npz"
)
DEFAULT_PK_SAMPLES = (
    PROJECT_ROOT
    / "outputs/task43_outputs/pk_lightcone/fits/"
    "mmin1p4e13_x25_fkpP010000_desi_rebin_kmax0p10_free_sn0_"
    "wtheorykmin_boxL2000_mcmc5x/"
    "task43_pk_lightcone_mmin1p4e13_x25_fkpP010000_desi_rebin_"
    "kmax0p10_free_sn0_wtheorykmin_boxL2000_mcmc5x_fit_samples.npz"
)
DEFAULT_PK_SUMMARY = (
    DEFAULT_PK_SAMPLES.parent
    / "task43_pk_lightcone_mmin1p4e13_x25_fkpP010000_desi_rebin_"
    "kmax0p10_free_sn0_wtheorykmin_boxL2000_mcmc5x_fit_summary.json"
)
DEFAULT_OUTPUT_JAXPOWER = (
    PLOT_DIR
    / "task43_pk_vs_2pcf_s50_350_both_jaxpower_covariance_"
    "formalgic_geometryonly_legacy.pdf"
)
DEFAULT_OUTPUT_RASCALC = (
    PLOT_DIR
    / "task43_pk_vs_2pcf_s50_350_2pcf_rascalc_covariance_"
    "formalgic_geometryonly_legacy.pdf"
)
DEFAULT_AUDIT = (
    SUMMARY_DIR
    / "task43_pk_vs_2pcf_s50_350_covariance_pair_"
    "formalgic_geometryonly_legacy.json"
)
DEFAULT_JAXPOWER_RERUN_MANIFEST = (
    SUMMARY_DIR / "task43_jaxpower_2pcf_mcmc5x_rerun_20260712.json"
)

COLORS = {
    "twopcf": "#B64342",
    "pk": "#42949E",
    "zero": "#767676",
}
GETDIST_SMOOTH_1D = 0.35
GETDIST_SMOOTH_2D = 0.40


def apply_publication_style() -> None:
    """设置统一的 PDF 字体、线宽和刻度风格，不修改任何数值内容。"""
    plt.rcParams["font.family"] = "sans-serif"
    plt.rcParams["font.sans-serif"] = ["Arial", "DejaVu Sans", "Liberation Sans"]
    plt.rcParams["pdf.fonttype"] = 42
    plt.rcParams.update(
        {
            "font.size": 7,
            "axes.linewidth": 0.8,
            "legend.frameon": False,
            "xtick.major.width": 0.7,
            "ytick.major.width": 0.7,
            "xtick.direction": "out",
            "ytick.direction": "out",
        }
    )


def resolve(path: Path) -> Path:
    """把相对路径解释为相对于项目根目录的路径，并返回绝对语义路径。"""
    return path if path.is_absolute() else PROJECT_ROOT / path


def read_json(path: Path) -> dict[str, Any]:
    """读取一个 JSON 文件。

    参数
    ----
    path
        JSON 输入路径，可为绝对路径或相对项目根目录的路径。

    返回
    ----
    dict
        JSON 解码后的字典。
    """
    return json.loads(resolve(path).read_text(encoding="utf-8"))


def load_named_samples(path: Path, wanted: tuple[str, ...]) -> np.ndarray:
    """从带 param_names 的 NPZ 中按名称抽取 posterior 列。

    参数
    ----
    path
        含 samples 与 param_names 的 NPZ。
    wanted
        需要抽取的参数名及其输出顺序。

    返回
    ----
    numpy.ndarray
        shape 为 (nsample, len(wanted)) 的 posterior 数组。
    """
    with np.load(resolve(path), allow_pickle=False) as data:
        samples = np.asarray(data["samples"], dtype="f8")
        names = [str(value) for value in np.asarray(data["param_names"]).tolist()]
    missing = [name for name in wanted if name not in names]
    if missing:
        raise KeyError(f"{path} 缺少参数 {missing}; 当前参数为 {names}")
    return samples[:, [names.index(name) for name in wanted]]


def formal_model(summary: dict[str, Any]) -> dict[str, Any]:
    """从新旧两种 Task43 summary 结构中取得 formal_gic 模型记录。"""
    models = summary["models"]
    if isinstance(models, dict):
        return models["formal_gic"]
    return next(model for model in models if model["model"] == "formal_gic")


def load_2pcf_samples(
    summary: dict[str, Any],
    samples_path: Path,
) -> np.ndarray:
    """读取 2PCF formal-GIC posterior 的 fNL 与 b1 两列。

    2PCF NPZ 本身没有 param_names，因此参数顺序从对应 summary 的
    parameter_names 读取，避免假设固定列序。
    """
    model = formal_model(summary)
    names = [str(name) for name in model.get("parameter_names", ("fnl_loc", "b1"))]
    with np.load(resolve(samples_path), allow_pickle=False) as data:
        samples = np.asarray(data["samples"], dtype="f8")
    missing = [name for name in ("fnl_loc", "b1") if name not in names]
    if missing:
        raise KeyError(f"{samples_path} 对应 summary 缺少参数 {missing}")
    return samples[:, [names.index("fnl_loc"), names.index("b1")]]


def normalized_summary(stats: dict[str, Any]) -> dict[str, float]:
    """把 P(k) 与 2PCF 的两种 quantile 字段统一成同一结构。"""
    q16 = float(stats["q16"])
    q50 = float(stats.get("q50", stats.get("median")))
    q84 = float(stats["q84"])
    return {
        "mean": float(stats["mean"]),
        "std": float(stats["std"]),
        "q16": q16,
        "q50": q50,
        "q84": q84,
        "err_low": float(q50 - q16),
        "err_high": float(q84 - q50),
    }


def twopcf_parameter_summary(model: dict[str, Any], name: str) -> dict[str, float]:
    """读取一项 2PCF 参数统计，并兼容历史 summary 的嵌套方式。"""
    if "posterior" in model and name in model["posterior"]:
        return normalized_summary(model["posterior"][name])
    return normalized_summary(model[name])


def constraint_text(stats: dict[str, Any]) -> str:
    """把一个参数的 16/50/84 分位数格式化为图中的非对称误差文本。"""
    q50 = float(stats.get("q50", stats.get("median")))
    q16 = float(stats["q16"])
    q84 = float(stats["q84"])
    return rf"{q50:.1f}^{{+{q84 - q50:.1f}}}_{{-{q50 - q16:.1f}}}"


def shared_axis_limits(
    sample_sets: list[np.ndarray],
    column: int,
    *,
    include_zero: bool = False,
) -> tuple[float, float]:
    """用所有候选 posterior 共同确定两张图的同一坐标范围。

    使用 0.25%--99.75% 分位数后再增加 8% 边距，既保留尾部又避免旧图中
    过宽的空白范围。
    """
    values = np.concatenate([samples[:, column] for samples in sample_sets])
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        raise ValueError("posterior 中没有有限样本")
    lo, hi = np.quantile(finite, [0.0025, 0.9975])
    if include_zero:
        lo = min(float(lo), 0.0)
        hi = max(float(hi), 0.0)
    width = float(hi - lo)
    if not np.isfinite(width) or width <= 0.0:
        width = 1.0
    return float(lo - 0.08 * width), float(hi + 0.08 * width)


def validate_inputs(
    pk_summary: dict[str, Any],
    jaxpower_summary: dict[str, Any],
    rascalc_summary: dict[str, Any],
) -> None:
    """执行绘图前的口径门槛，防止把不同尺度或 covariance 混入。

    该函数不修改文件；任一检查失败都会停止绘图并给出明确错误。
    """
    pk_cov_source = (
        pk_summary.get("input_summary", {})
        .get("covariance", {})
        .get("source", "")
    )
    if "jaxpower" not in str(pk_cov_source).lower():
        raise ValueError(f"P(k) covariance 不是 jaxpower: {pk_cov_source}")

    checks = (
        ("jaxpower", jaxpower_summary, "jaxpower"),
        ("rascalc", rascalc_summary, "rascalc"),
    )
    for label, summary, expected_covariance in checks:
        fit_range = summary.get("fit_range", {})
        if float(fit_range.get("rmin", np.nan)) != 50.0:
            raise ValueError(f"{label} 2PCF rmin 不是 50: {fit_range}")
        if float(fit_range.get("rmax", np.nan)) != 350.0:
            raise ValueError(f"{label} 2PCF rmax 不是 350: {fit_range}")
        covariance_path = str(summary.get("covariance", {}).get("path", "")).lower()
        if expected_covariance not in covariance_path:
            raise ValueError(
                f"{label} 2PCF covariance provenance 不匹配: {covariance_path}"
            )
        model = formal_model(summary)
        if model.get("model") != "formal_gic":
            raise ValueError(f"{label} 2PCF 不是 formal_gic")


def twopcf_label(covariance_label: str) -> str:
    """生成 2PCF 图例，明确尺度范围与 covariance 来源。"""
    return (
        rf"2PCF formal GIC, {covariance_label}" "\n"
        rf"$s=50$--$350\,h^{{-1}}{{\rm Mpc}}$"
    )


def pk_label(summary: dict[str, Any]) -> str:
    """生成 P(k) 图例，明确 jaxpower covariance 与拟合 k 范围。"""
    data = summary["data"]
    theory_kmin = float(summary["config"]["window_theory_kmin"])
    return (
        r"$P(k)$, jaxpower covariance" "\n"
        + rf"$k_{{\rm obs,min}}={float(data['k_min']):.3f},\ "
        + rf"k_{{\max}}={float(data['kmax_fit']):.2f};\ "
        + rf"k_{{\rm th,min}}={theory_kmin:.4f}\,h\,{{\rm Mpc}}^{{-1}}$"
    )


def plot_getdist_corner(
    samples: dict[str, np.ndarray],
    labels: dict[str, str],
    xlim: tuple[float, float],
    ylim: tuple[float, float],
) -> tuple[plt.Figure, np.ndarray[Any, Any]]:
    """绘制 fNL-b1 triangle plot。

    参数
    ----
    samples
        twopcf 与 pk 两条 shape=(nsample, 2) 的 posterior。
    labels
        与 samples key 对应的图例文本。
    xlim, ylim
        两张输出图共用的 fNL 和 b1 坐标范围。

    返回
    ----
    figure, axes
        GetDist 创建的 Matplotlib figure 与 2x2 axes 数组。
    """
    from getdist import MCSamples, plots

    names = ["fnl_loc", "b1"]
    parameter_labels = [r"f_{\rm NL}^{\rm loc}", r"b_1"]
    ranges = {"fnl_loc": xlim, "b1": ylim}
    order = ["twopcf", "pk"]
    chains = [
        MCSamples(
            samples=samples[key],
            names=names,
            labels=parameter_labels,
            label=labels[key],
            ranges=ranges,
            settings={
                "ignore_rows": 0,
                "smooth_scale_1D": GETDIST_SMOOTH_1D,
                "smooth_scale_2D": GETDIST_SMOOTH_2D,
            },
        )
        for key in order
    ]

    plotter = plots.get_subplot_plotter(width_inch=7.2)
    plotter.settings.axes_fontsize = 15
    plotter.settings.axes_labelsize = 20
    plotter.settings.legend_fontsize = 9.6
    plotter.settings.legend_frame = False
    plotter.settings.figure_legend_frame = False
    plotter.settings.linewidth_contour = 1.8
    plotter.settings.solid_contour_palefactor = 0.55
    plotter.settings.alpha_filled_add = 0.70
    plotter.triangle_plot(
        chains,
        names,
        filled=True,
        contour_colors=[COLORS[key] for key in order],
        legend_labels=[labels[key] for key in order],
        legend_loc="upper right",
        param_limits=ranges,
        markers={"fnl_loc": 0.0},
        marker_args={
            "color": COLORS["zero"],
            "ls": "--",
            "lw": 0.9,
            "alpha": 0.85,
        },
    )

    axes = plotter.subplots
    axes[0, 0].set_xlim(*xlim)
    axes[1, 0].set_xlim(*xlim)
    axes[1, 0].set_ylim(*ylim)
    axes[1, 1].set_xlim(*ylim)
    return plotter.fig, axes


def add_map_markers(
    axes: np.ndarray[Any, Any],
    twopcf_summary: dict[str, Any],
    pk_summary: dict[str, Any],
) -> None:
    """在联合 posterior panel 上添加两条链各自的 MAP 星号。"""
    axis = axes[1, 0]
    twopcf_map = formal_model(twopcf_summary)["map"]
    axis.scatter(
        twopcf_map["fnl_loc"],
        twopcf_map["b1"],
        marker="*",
        s=62,
        color=COLORS["twopcf"],
        edgecolor="white",
        linewidth=0.35,
        zorder=20,
    )
    pk_map = pk_summary["maximum_posterior_sample"]["point"]
    axis.scatter(
        pk_map["fnl_loc"],
        pk_map["b1"],
        marker="*",
        s=62,
        color=COLORS["pk"],
        edgecolor="white",
        linewidth=0.35,
        zorder=20,
    )


def add_constraint_box(
    figure: plt.Figure,
    twopcf_summary: dict[str, Any],
    pk_summary: dict[str, Any],
    covariance_short: str,
) -> None:
    """在空白 panel 中列出两条链的 fNL 约束，便于快速读取数值。"""
    twopcf_fnl = twopcf_parameter_summary(
        formal_model(twopcf_summary),
        "fnl_loc",
    )
    pk_fnl = normalized_summary(pk_summary["parameters"]["fnl_loc"])
    text = (
        rf"2PCF ({covariance_short}): "
        rf"$f_{{\rm NL}}={constraint_text(twopcf_fnl)}$" "\n"
        rf"$P(k)$ (jaxpower): "
        rf"$f_{{\rm NL}}={constraint_text(pk_fnl)}$"
    )
    figure.text(
        0.64,
        0.65,
        text,
        ha="left",
        va="top",
        fontsize=9.2,
        color="#4D4D4D",
        bbox={
            "boxstyle": "round,pad=0.30",
            "fc": "white",
            "ec": "#D7D7D7",
            "alpha": 0.90,
        },
    )


def jsonable(value: Any) -> Any:
    """把 numpy、Path 与嵌套容器转换为可写入 JSON 的普通对象。"""
    if isinstance(value, dict):
        return {str(key): jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [jsonable(item) for item in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    return value


def posterior_payload(
    twopcf_summary: dict[str, Any],
    pk_summary: dict[str, Any],
) -> dict[str, Any]:
    """提取一张图使用的 fNL、b1 posterior 数值，供审计 JSON 使用。"""
    model = formal_model(twopcf_summary)
    return {
        "twopcf": {
            "fnl_loc": twopcf_parameter_summary(model, "fnl_loc"),
            "b1": twopcf_parameter_summary(model, "b1"),
        },
        "pk": {
            "fnl_loc": normalized_summary(pk_summary["parameters"]["fnl_loc"]),
            "b1": normalized_summary(pk_summary["parameters"]["b1"]),
        },
    }


def main() -> None:
    """读取三条 posterior，执行口径检查并生成两张受控对比 PDF。"""
    parser = argparse.ArgumentParser()
    parser.add_argument("--twopcf-jaxpower-json", type=Path, default=DEFAULT_2PCF_JAXPOWER)
    parser.add_argument(
        "--twopcf-jaxpower-samples",
        type=Path,
        default=DEFAULT_2PCF_JAXPOWER_SAMPLES,
    )
    parser.add_argument("--twopcf-rascalc-json", type=Path, default=DEFAULT_2PCF_RASCALC)
    parser.add_argument(
        "--twopcf-rascalc-samples",
        type=Path,
        default=DEFAULT_2PCF_RASCALC_SAMPLES,
    )
    parser.add_argument("--pk-samples", type=Path, default=DEFAULT_PK_SAMPLES)
    parser.add_argument("--pk-summary", type=Path, default=DEFAULT_PK_SUMMARY)
    parser.add_argument("--output-jaxpower", type=Path, default=DEFAULT_OUTPUT_JAXPOWER)
    parser.add_argument("--output-rascalc", type=Path, default=DEFAULT_OUTPUT_RASCALC)
    parser.add_argument("--audit-json", type=Path, default=DEFAULT_AUDIT)
    args = parser.parse_args()

    apply_publication_style()
    pk_summary = read_json(args.pk_summary)
    jaxpower_summary = read_json(args.twopcf_jaxpower_json)
    rascalc_summary = read_json(args.twopcf_rascalc_json)
    validate_inputs(pk_summary, jaxpower_summary, rascalc_summary)

    pk_samples = load_named_samples(args.pk_samples, ("fnl_loc", "b1"))
    jaxpower_samples = load_2pcf_samples(
        jaxpower_summary,
        args.twopcf_jaxpower_samples,
    )
    rascalc_samples = load_2pcf_samples(
        rascalc_summary,
        args.twopcf_rascalc_samples,
    )
    all_samples = [pk_samples, jaxpower_samples, rascalc_samples]
    xlim = shared_axis_limits(all_samples, 0, include_zero=True)
    ylim = shared_axis_limits(all_samples, 1)

    variants = (
        {
            "key": "both_jaxpower",
            "summary": jaxpower_summary,
            "samples": jaxpower_samples,
            "covariance_label": "jaxpower covariance",
            "covariance_short": "jaxpower",
            "output": resolve(args.output_jaxpower),
            "summary_path": resolve(args.twopcf_jaxpower_json),
            "samples_path": resolve(args.twopcf_jaxpower_samples),
        },
        {
            "key": "twopcf_rascalc",
            "summary": rascalc_summary,
            "samples": rascalc_samples,
            "covariance_label": "RascalC covariance",
            "covariance_short": "RascalC",
            "output": resolve(args.output_rascalc),
            "summary_path": resolve(args.twopcf_rascalc_json),
            "samples_path": resolve(args.twopcf_rascalc_samples),
        },
    )

    products: list[dict[str, Any]] = []
    for variant in variants:
        labels = {
            "twopcf": twopcf_label(str(variant["covariance_label"])),
            "pk": pk_label(pk_summary),
        }
        samples = {
            "twopcf": np.asarray(variant["samples"], dtype="f8"),
            "pk": pk_samples,
        }
        figure, axes = plot_getdist_corner(samples, labels, xlim, ylim)
        add_map_markers(axes, variant["summary"], pk_summary)
        add_constraint_box(
            figure,
            variant["summary"],
            pk_summary,
            str(variant["covariance_short"]),
        )
        output = Path(variant["output"])
        output.parent.mkdir(parents=True, exist_ok=True)
        figure.savefig(output, bbox_inches="tight")
        plt.close(figure)
        products.append(
            {
                "key": variant["key"],
                "pdf": str(output),
                "twopcf_summary": str(variant["summary_path"]),
                "twopcf_samples": str(variant["samples_path"]),
                "twopcf_covariance": variant["summary"]["covariance"],
                "posterior": posterior_payload(variant["summary"], pk_summary),
            }
        )
        print(f"[write] {output}")

    audit_path = resolve(args.audit_json)
    audit_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "task": "task43_pk_vs_2pcf_s50_350_covariance_pair",
        "status": "done",
        "figure_contract": {
            "number_of_pdfs": 2,
            "fit_range_2pcf": [50.0, 350.0],
            "same_pk_chain": True,
            "same_axis_ranges": True,
            "only_changed_quantity": "2PCF covariance: jaxpower versus RascalC",
            "pk_covariance": "jaxpower Gaussian survey-window single-lightcone",
            "pk_model_caveat": (
                "P(k) uses a continuous mother-box hard cutoff before window "
                "convolution, not a FullDiscrete integer-lattice window model."
            ),
            "jaxpower_2pcf_caveat": (
                "RR-deconvolved jaxpower covariance remains a diagnostic "
                "baseline relative to the mock-calibrated RascalC result."
            ),
            "parameter_basis_caveat": (
                "P(k) marginalizes over its bandpower sn0 nuisance while "
                "2PCF uses the established contact/fixed sn0=0 convention; "
                "this shared model-basis difference is unchanged between "
                "the two covariance figures."
            ),
        },
        "shared_inputs": {
            "pk_summary": str(resolve(args.pk_summary)),
            "pk_samples": str(resolve(args.pk_samples)),
            "jaxpower_2pcf_rerun_manifest": str(
                DEFAULT_JAXPOWER_RERUN_MANIFEST
            ),
        },
        "shared_plot_ranges": {
            "fnl_loc": xlim,
            "b1": ylim,
        },
        "plot_settings": {
            "backend": "matplotlib + getdist",
            "smooth_scale_1d": GETDIST_SMOOTH_1D,
            "smooth_scale_2d": GETDIST_SMOOTH_2D,
        },
        "products": products,
    }
    audit_path.write_text(
        json.dumps(jsonable(payload), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(f"[write] {audit_path}")


if __name__ == "__main__":
    main()
