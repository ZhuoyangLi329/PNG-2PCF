#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""叠画 Task43 五个 ``smax`` 的 jaxpower 2PCF ``fNL-b1`` contours。

代码大纲
========
1. 读取 rmax-scan 总科学审计，并定位 ``smax=350/400/450/500/550`` 的
   五份 summary 与五份 burn-in 后 MCMC samples。
2. 对每一档硬检查 bin 数、径向范围、共同 50-bin data/covariance/operator、
   ``covariance_single_realization``、FullDiscrete/full-PNG/fixed-sn0=0 和
   ``64x20000, burnin=5000, seed=20260720`` 的 matched-chain contract。
3. 从 raw samples 统一重算 quantile、tau 与 split-chain gate，并与总审计
   逐值桥接；任一档不一致就停止，不输出看似正常的错误 contour。
4. 使用同一 fNL/b1 坐标、同一 GetDist smoothing，按 ``smax`` 递增叠画
   五组 68/95% 线轮廓和一维边缘分布；不用大面积填充，避免 nested
   posterior 互相遮住，并用颜色、线型和线宽三重编码。
5. 原子写一张 PDF-only 主图和一份 JSON audit，记录输入哈希、posterior、
   收敛诊断、坐标、颜色及输出哈希，便于下一位 AI 无需扫描大目录复核。

科学边界
========
这是同一数据向量和固定 RR-deconvolved jaxpower diagnostic covariance 的
相关嵌套裁剪比较，mean model 只含 radial ``IC^(rad,rad)`` single term；
图用于比较相对信息增益，不是 science-ready 的绝对 ``fNL`` 约束。
"""

from __future__ import annotations

import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


PROJECT_ROOT = Path("/pscratch/sd/l/lzy/PNG-2PCF-model-to-observe")
CODE_DIR = PROJECT_ROOT / "codes/task43"
if str(CODE_DIR) not in sys.path:
    sys.path.insert(0, str(CODE_DIR))

import task43_overlay_pk_window_kmin_contour as contour_base  # noqa: E402
import task43_plot_ric_rascalc_pk_vs_2pcf as chain_tools  # noqa: E402


RMAX_VALUES = (350, 400, 450, 500, 550)
NBINS_VALUES = (30, 35, 40, 45, 50)
# 颜色取自色觉友好的灰色 + Okabe-Ito 调色板；再叠加线型和线宽编码，
# 保证黑白打印时也能辨认。最终的 smax=550 用粗红色实线突出。
COLORS = ("#6B6B6B", "#56B4E9", "#009E73", "#E69F00", "#C44E52")
LINESTYLES = ("--", "-.", ":", (0, (5, 1.5)), "-")
LINEWIDTHS = (1.55, 1.65, 1.75, 2.00, 2.35)

RMAX_ROOT = PROJECT_ROOT / "outputs/task43_outputs/rmax_scan"
SCIENCE_AUDIT = RMAX_ROOT / "audits/task43_jaxpower_2pcf_rmax_scan_longchain.json"
COMMON_XI = RMAX_ROOT / (
    "summary/task43_mean_xi_mmin1p4e13_x25_s50_550_ds10_fkpP010000.npz"
)
COMMON_COVARIANCE = RMAX_ROOT / "covariance" / (
    "jaxpower_covariance_lightcone_mmin1p4e13_ph000_fitcov_bessel_interp_"
    "smoothfftlog_rrdeconv_fkpNorm4p8925e10_mesh64_nran100k_ndata50k_"
    "pad400_win3600_ds2_k0001_3000_dk002_p1p0_s50_550_ds10.npz"
)
COMMON_OPERATOR = RMAX_ROOT / "operators" / (
    "task43_ric_factorized_operator_ph000_dchi2_nsub200000_sobol2p22_"
    "ds2_seed20260712_L2000_s50_550_ds10.npz"
)
OUTPUT_PDF = PROJECT_ROOT / "plots/task43/rmax_scan" / (
    "task43_jaxpower_2pcf_rmax_scan_fnl_b1_contours_smax350_550.pdf"
)
OUTPUT_AUDIT = RMAX_ROOT / "audits" / (
    "task43_jaxpower_2pcf_rmax_scan_fnl_b1_contours.json"
)


def require(condition: bool, message: str) -> None:
    """把输入、统计或 provenance 不一致升级为硬失败。"""
    if not bool(condition):
        raise RuntimeError(message)


def read_json(path: Path) -> dict[str, Any]:
    """读取一个必须存在且非空的 JSON object。"""
    require(path.is_file() and path.stat().st_size > 0, f"缺少 JSON：{path}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    require(isinstance(payload, dict), f"JSON 顶层不是 object：{path}")
    return payload


def canonical_recorded_path(value: str | Path) -> Path:
    """把 summary 中可能为相对路径的记录统一解析到项目根目录。"""
    path = Path(value).expanduser()
    if not path.is_absolute():
        path = PROJECT_ROOT / path
    return path.resolve(strict=False)


def sha256_file(path: Path) -> str:
    """分块计算 SHA256，避免为 MCMC samples 创建任何副本。"""
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while True:
            block = stream.read(8 * 1024 * 1024)
            if not block:
                break
            digest.update(block)
    return digest.hexdigest()


def close(left: float, right: float, *, atol: float = 1.0e-12) -> bool:
    """对审计中应相同的 scalar 使用严格绝对误差比较。"""
    return bool(np.isclose(float(left), float(right), rtol=0.0, atol=atol))


def validate_and_load_fit(
    *,
    rmax: int,
    nbins: int,
    public: dict[str, Any],
) -> tuple[dict[str, Any], np.ndarray, dict[str, Any], dict[str, Any]]:
    """验证一档 fit、读取两参数样本，并桥接总审计中的统计量。

    Returns
    -------
    summary, samples, posterior, diagnostics
        summary 是原 fit JSON；samples 列顺序固定为 ``fnl_loc,b1``；
        posterior 和 diagnostics 均从 raw samples 重新计算。
    """
    fit_dir = RMAX_ROOT / f"fits/smax{rmax}"
    summary_path = fit_dir / "task43_minimal_closure_mcmc_summary.json"
    samples_path = fit_dir / "task43_mcmc_radial_singleterm_samples.npz"
    require(
        canonical_recorded_path(public["paths"]["summary"]) == summary_path.resolve()
        and canonical_recorded_path(public["paths"]["samples"]) == samples_path.resolve(),
        f"smax={rmax} 总审计路径与冻结 fit 目录不一致",
    )
    summary = read_json(summary_path)
    require(summary.get("status") == "done", f"smax={rmax} summary 未完成")
    fit = summary.get("fit_range", {})
    expected_centers = np.arange(55.0, float(rmax), 10.0)
    require(
        float(fit.get("rmin", np.nan)) == 50.0
        and float(fit.get("rmax", np.nan)) == float(rmax)
        and int(fit.get("nbins", -1)) == nbins
        and int(fit.get("data_vector_size", -1)) == nbins,
        f"smax={rmax} fit range/bin contract 不匹配：{fit}",
    )
    require(
        int(fit.get("nreal", -1)) == 25
        and np.array_equal(np.asarray(fit.get("s_centers", []), dtype="f8"), expected_centers),
        f"smax={rmax} 不是共同 50--550、ds=10 数据向量的正确前缀",
    )
    require(
        summary.get("fit_target") == "mean"
        and summary.get("phase") == "mean"
        and summary.get("sim_name") == "AbacusSummit_base_c000_ph000-ph024_mean"
        and close(summary.get("zeff", np.nan), 0.7030110803750067)
        and canonical_recorded_path(summary.get("xi_path", "")) == COMMON_XI.resolve(),
        f"smax={rmax} mean-data provenance 不匹配",
    )
    covariance = summary.get("covariance", {})
    require(
        covariance.get("key") == "covariance_single_realization"
        and covariance.get("fit_target") == "mean"
        and canonical_recorded_path(covariance.get("path", ""))
        == COMMON_COVARIANCE.resolve(),
        f"smax={rmax} covariance contract 不匹配",
    )
    theory = summary.get("theory", {})
    require(
        float(theory.get("boxsize", np.nan)) == 2000.0
        and theory.get("cosmology") == "abacus_c000"
        and theory.get("png_order") == "full"
        and float(theory.get("p_fixed", np.nan)) == 1.0
        and theory.get("sn0_policy") == "fixed"
        and float(theory.get("sn0_fixed", np.nan)) == 0.0
        and theory.get("xi_kernel") == "shell-averaged",
        f"smax={rmax} theory contract 不匹配",
    )
    model = chain_tools.radial_model(summary)
    require(model.get("parameter_names") == ["fnl_loc", "b1"], f"smax={rmax} 参数顺序错误")
    radial = model.get("radial_singleterm", {})
    require(
        canonical_recorded_path(radial.get("operator", {}).get("path", ""))
        == COMMON_OPERATOR.resolve()
        and radial.get("extra_global_sigma_w2") is False,
        f"smax={rmax} operator/extra-global contract 不匹配",
    )
    # ``selected_indices`` 属于 operator 子记录；它描述从共同 50-bin
    # 算子中实际取出的前缀，不能误读 radial_singleterm 的同级字段。
    selected = radial.get("operator", {}).get("selected_indices", [])
    require(selected == list(range(nbins)), f"smax={rmax} operator crop indices 不匹配")
    require(
        np.array_equal(
            np.asarray(radial.get("operator", {}).get("selected_s", []), dtype="f8"),
            expected_centers,
        ),
        f"smax={rmax} operator 的径向中心不匹配",
    )
    mcmc = model.get("mcmc", {})
    require(
        int(mcmc.get("nwalkers", -1)) == 64
        and int(mcmc.get("nsteps", -1)) == 20000
        and int(mcmc.get("burnin", -1)) == 5000
        and int(mcmc.get("seed", -1)) == 20260720
        and int(mcmc.get("nsamples", -1)) == 960000,
        f"smax={rmax} MCMC contract 不匹配：{mcmc}",
    )

    # 除了绘图使用的 samples，还硬检完整 NPZ contract 与 log-probability；
    # prediction/residual 的长度同时提供另一条独立的 bin-count 证据。
    with np.load(samples_path, allow_pickle=False) as payload:
        require(
            set(payload.files)
            == {"samples", "log_prob", "prediction_map", "residual_map"},
            f"smax={rmax} samples NPZ keys 不匹配：{payload.files}",
        )
        raw_samples = np.asarray(payload["samples"], dtype="f8")
        log_prob = np.asarray(payload["log_prob"], dtype="f8")
        prediction_map = np.asarray(payload["prediction_map"], dtype="f8")
        residual_map = np.asarray(payload["residual_map"], dtype="f8")
    require(
        raw_samples.shape == (960000, 2)
        and log_prob.shape == (960000,)
        and prediction_map.shape == (nbins,)
        and residual_map.shape == (nbins,),
        f"smax={rmax} samples/log_prob/map shape contract 不匹配",
    )
    require(
        np.all(np.isfinite(raw_samples))
        and np.all(np.isfinite(log_prob))
        and np.all(np.isfinite(prediction_map))
        and np.all(np.isfinite(residual_map)),
        f"smax={rmax} samples NPZ 含非有限值",
    )
    samples = chain_tools.load_xi_samples(summary, samples_path)
    require(samples.shape == (960000, 2), f"smax={rmax} samples shape={samples.shape}")
    require(np.all(np.isfinite(samples)), f"smax={rmax} samples 含非有限值")
    posterior = {
        "fnl_loc": chain_tools.summarize(samples[:, 0]),
        "b1": chain_tools.summarize(samples[:, 1]),
        "corr_fnl_b1": float(np.corrcoef(samples.T)[0, 1]),
    }
    diagnostics = chain_tools.chain_diagnostics(samples, nwalkers=64)
    require(diagnostics["pass"], f"smax={rmax} raw-chain gate 失败：{diagnostics}")

    # 总审计使用完全相同的 Gaussian-equivalent 16/84% quantiles；逐 scalar
    # 桥接可以阻止错误地混入 summary 自带的普通 0.16/0.84 近似。
    for parameter in ("fnl_loc", "b1"):
        expected = public["posterior"][parameter]
        actual = posterior[parameter]
        for key in ("q16", "q50", "q84", "mean", "std", "err_low", "err_high"):
            require(
                close(actual[key], expected[key], atol=1.0e-12),
                f"smax={rmax} {parameter}.{key} 与总审计不一致",
            )
    require(
        close(posterior["corr_fnl_b1"], public["posterior"]["corr_fnl_b1"]),
        f"smax={rmax} corr(fNL,b1) 与总审计不一致",
    )
    expected_diag = public["chain_diagnostics"]
    require(
        close(diagnostics["length_over_tau_min"], expected_diag["length_over_tau_min"])
        and close(
            diagnostics["split_median_shift_sigma_max"],
            expected_diag["split_median_shift_sigma_max"],
        ),
        f"smax={rmax} chain diagnostics 与总审计不一致",
    )
    return summary, samples, posterior, diagnostics


def plot_contours(
    *,
    samples: list[np.ndarray],
    xlim: tuple[float, float],
    ylim: tuple[float, float],
) -> dict[str, Any]:
    """绘制五组 1D/2D posterior，并返回 legend 的几何位置。"""
    from getdist import MCSamples, plots

    names = ["fnl_loc", "b1"]
    parameter_labels = [r"f_{\rm NL}", r"b_1"]
    ranges = {"fnl_loc": xlim, "b1": ylim}
    legend_labels = [
        rf"$s_{{\rm max}}={rmax}\,h^{{-1}}{{\rm Mpc}}$" for rmax in RMAX_VALUES
    ]
    chains = [
        MCSamples(
            samples=sample,
            names=names,
            labels=parameter_labels,
            label=legend,
            ranges=ranges,
            settings={
                "ignore_rows": 0,
                "smooth_scale_1D": contour_base.GETDIST_SMOOTH_1D,
                "smooth_scale_2D": contour_base.GETDIST_SMOOTH_2D,
            },
        )
        for sample, legend in zip(samples, legend_labels, strict=True)
    ]

    contour_base.apply_publication_style()
    plt.rcParams.update(
        {
            "font.size": 12,
            "axes.linewidth": 1.0,
            "legend.frameon": False,
        }
    )
    plotter = plots.get_subplot_plotter(width_inch=8.8)
    plotter.settings.axes_fontsize = 13.5
    plotter.settings.lab_fontsize = 17.0
    plotter.settings.legend_fontsize = 13.0
    plotter.settings.legend_frame = False
    plotter.settings.figure_legend_frame = False
    plotter.settings.linewidth = 1.8
    plotter.settings.linewidth_contour = 1.8
    plotter.settings.num_plot_contours = 2
    line_args = [
        {"color": color, "ls": linestyle, "lw": linewidth}
        for color, linestyle, linewidth in zip(
            COLORS, LINESTYLES, LINEWIDTHS, strict=True
        )
    ]
    plotter.triangle_plot(
        chains,
        names,
        filled=False,
        contour_colors=list(COLORS),
        contour_ls=list(LINESTYLES),
        contour_lws=list(LINEWIDTHS),
        line_args=line_args,
        legend_labels=legend_labels,
        legend_loc="lower left",
        param_limits=ranges,
    )
    figure = plotter.fig
    axes = plotter.subplots
    axes[0, 0].set_xlim(*xlim)
    axes[1, 0].set_xlim(*xlim)
    axes[1, 0].set_ylim(*ylim)
    axes[1, 1].set_xlim(*ylim)

    # 把五项 legend 固定在 triangle 的空白第一象限，避免压住任何 posterior。
    blank_corner_x = float(axes[0, 0].get_position().x1)
    blank_corner_y = float(axes[1, 1].get_position().y1)
    legend_anchor = (blank_corner_x + 0.018, blank_corner_y + 0.018)
    require(bool(figure.legends), "GetDist 没有创建 figure legend")
    legend = figure.legends[-1]
    legend.set_bbox_to_anchor(legend_anchor, transform=figure.transFigure)
    legend.set_title(
        r"$\xi_0(s)$, radial single-term RIC" "\n" "jaxpower covariance",
        prop={"size": 12.5},
    )

    OUTPUT_PDF.parent.mkdir(parents=True, exist_ok=True)
    temporary = OUTPUT_PDF.with_name(f".{OUTPUT_PDF.name}.tmp.pdf")
    try:
        figure.savefig(
            temporary,
            format="pdf",
            bbox_inches="tight",
            pad_inches=0.08,
        )
        temporary.replace(OUTPUT_PDF)
    finally:
        plt.close(figure)
        if temporary.exists():
            temporary.unlink()
    return {
        "legend_anchor_figure_fraction": list(legend_anchor),
        "legend_labels": legend_labels,
    }


def main() -> None:
    """完成全部 bridge/gate 后生成 PDF 与机器审计。"""
    science = read_json(SCIENCE_AUDIT)
    require(
        science.get("status") == "done"
        and science.get("hard_gates", {}).get("pass") is True,
        "rmax-scan 总科学审计没有通过 hard gates",
    )
    require(
        canonical_recorded_path(science.get("inputs", {}).get("covariance_50bin", ""))
        == COMMON_COVARIANCE.resolve(),
        "总审计使用的 covariance 不是本脚本冻结的 50-bin 文件",
    )
    require(
        canonical_recorded_path(science.get("inputs", {}).get("xi_50bin", ""))
        == COMMON_XI.resolve()
        and canonical_recorded_path(science.get("inputs", {}).get("operator_50bin", ""))
        == COMMON_OPERATOR.resolve(),
        "总审计使用的 xi/operator 不是本脚本冻结的共同 50-bin 文件",
    )
    public_by_rmax = {int(row["rmax"]): row for row in science.get("fits", [])}
    require(tuple(sorted(public_by_rmax)) == RMAX_VALUES, "总审计没有恰好五档 smax")

    summaries: dict[str, dict[str, Any]] = {}
    sample_sets: list[np.ndarray] = []
    posterior: dict[str, dict[str, Any]] = {}
    diagnostics: dict[str, dict[str, Any]] = {}
    input_paths: list[Path] = [
        SCIENCE_AUDIT,
        COMMON_XI,
        COMMON_COVARIANCE,
        COMMON_OPERATOR,
    ]
    for rmax, nbins in zip(RMAX_VALUES, NBINS_VALUES, strict=True):
        summary, samples, fit_posterior, fit_diagnostics = validate_and_load_fit(
            rmax=rmax,
            nbins=nbins,
            public=public_by_rmax[rmax],
        )
        key = str(rmax)
        summaries[key] = summary
        sample_sets.append(samples)
        posterior[key] = fit_posterior
        diagnostics[key] = fit_diagnostics
        input_paths.extend(
            [
                RMAX_ROOT / f"fits/smax{rmax}/task43_minimal_closure_mcmc_summary.json",
                RMAX_ROOT / f"fits/smax{rmax}/task43_mcmc_radial_singleterm_samples.npz",
            ]
        )

    xlim = contour_base.shared_axis_limits(sample_sets, 0, include_zero=True)
    ylim = contour_base.shared_axis_limits(sample_sets, 1)
    plot_meta = plot_contours(samples=sample_sets, xlim=xlim, ylim=ylim)
    require(OUTPUT_PDF.is_file() and OUTPUT_PDF.stat().st_size > 0, "contour PDF 未生成")
    with OUTPUT_PDF.open("rb") as stream:
        require(stream.read(5) == b"%PDF-", "输出没有 PDF magic header")

    audit = {
        "task": "task43_plot_rmax_scan_fnl_b1_contours",
        "status": "done",
        "created_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "figure_contract": {
            "probe": "2PCF xi0",
            "covariance": "fixed RR-deconvolved jaxpower covariance_single_realization",
            "model": "xi_noIC - IC^(rad,rad)",
            "rmax_edges_mpc_h": list(RMAX_VALUES),
            "nbins": list(NBINS_VALUES),
            "same_50bin_data_covariance_operator": True,
            "nested_correlated_crops": True,
            "density_ric_cross_terms": False,
            "science_ready": False,
        },
        "inputs": {
            "science_audit": str(SCIENCE_AUDIT),
            "common_xi": str(COMMON_XI),
            "common_covariance": str(COMMON_COVARIANCE),
            "common_operator": str(COMMON_OPERATOR),
            "fit_summaries": {
                key: str(RMAX_ROOT / f"fits/smax{key}/task43_minimal_closure_mcmc_summary.json")
                for key in summaries
            },
            "fit_samples": {
                key: str(RMAX_ROOT / f"fits/smax{key}/task43_mcmc_radial_singleterm_samples.npz")
                for key in summaries
            },
            "sha256": {str(path): sha256_file(path) for path in input_paths},
        },
        "posterior": posterior,
        "chain_diagnostics": diagnostics,
        "plot_ranges": {"fnl_loc": list(xlim), "b1": list(ylim)},
        "plot_settings": {
            "backend": "matplotlib + getdist",
            "smooth_scale_1d": contour_base.GETDIST_SMOOTH_1D,
            "smooth_scale_2d": contour_base.GETDIST_SMOOTH_2D,
            "credible_contours": [0.68, 0.95],
            "filled": False,
            "colors": list(COLORS),
            "linestyles": list(LINESTYLES),
            "linewidths": list(LINEWIDTHS),
            "format": "pdf only",
            **plot_meta,
        },
        "scope_warning": (
            "The five contours are correlated nested crops under one fixed diagnostic "
            "jaxpower covariance and the radial single IC^(rad,rad) term; not a "
            "science-ready absolute fNL comparison."
        ),
        "output": {
            "pdf": str(OUTPUT_PDF),
            "size_bytes": int(OUTPUT_PDF.stat().st_size),
            "sha256": sha256_file(OUTPUT_PDF),
        },
    }
    OUTPUT_AUDIT.parent.mkdir(parents=True, exist_ok=True)
    temporary = OUTPUT_AUDIT.with_name(f".{OUTPUT_AUDIT.name}.tmp")
    temporary.write_text(
        json.dumps(contour_base.jsonable(audit), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(OUTPUT_AUDIT)
    print(f"[write] {OUTPUT_PDF}")
    print(f"[write] {OUTPUT_AUDIT}")


if __name__ == "__main__":
    main()
