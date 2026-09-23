#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""叠画 P(k) 与 preliminary-xN 三档 2PCF rmin posterior。

执行逻辑大纲：
1. 读取冻结 ph000--ph(N-1) 的 xN 总审计、三条 ``fnl_loc,b1`` 原始链，
   以及 Task43 当前权威 P(k) ``fnl_loc,b1,sn0`` 长链。
2. 硬检查两边都使用 radial single-term RIC 和 jaxpower single-lightcone
   covariance，同时保留 nuisance 差异：P(k) free ``sn0``，2PCF fixed
   contact ``sn0=0``。
3. 从 raw chain 重算 posterior 与收敛诊断，并逐值桥接 xN 总审计；任何
   不一致都停止，不生成可能误标的图。
4. 在冻结的 Task43 公共 fNL/b1 坐标上叠画 P(k)、rmin=50/40/30 四组
   68%（1 sigma）无填充 contour，标出 posterior 中位数中心点，并在图例
   写出中位数与 16--84% 非对称误差。
5. 原子输出 PDF-only 图与 JSON audit，记录输入/输出 SHA256、posterior、
   chain gate 和图例口径；本图明确标注为 xN preliminary，不替代 x25。
"""

from __future__ import annotations

import hashlib
import json
import os
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


ROOT = Path("/pscratch/sd/l/lzy/PNG-2PCF-model-to-observe")
CODE = ROOT / "codes/task43"
if str(CODE) not in sys.path:
    sys.path.insert(0, str(CODE))

import task43_overlay_pk_window_kmin_contour as contour_base  # noqa: E402
import task43_plot_ric_rascalc_pk_vs_2pcf as chain_tools  # noqa: E402


FINAL_MODE = os.environ.get("TASK43_RMIN_FINAL", "0") == "1"
NREAL = 25 if FINAL_MODE else int(os.environ.get("TASK43_PRELIM_NREAL", "12"))
if not 1 <= NREAL <= 25:
    raise ValueError(f"TASK43_PRELIM_NREAL 必须在 1..25：{NREAL}")
TAG = "final_x25" if FINAL_MODE else f"preliminary_x{NREAL}"
SHORT = "finalx25" if FINAL_MODE else f"prelimx{NREAL}"
SCAN = ROOT / "outputs/task43_outputs/rmin_scan"
PRE = SCAN if FINAL_MODE else SCAN / TAG
MOVED = ROOT / "plots/outputs/task43_outputs"
SCIENCE_AUDIT = (
    PRE / "audits/task43_jaxpower_2pcf_rmin_scan_longchain.json"
    if FINAL_MODE
    else PRE / f"audits/task43_jaxpower_2pcf_rmin_scan_{TAG}.json"
)
OLD_PAIR_AUDIT = ROOT / "plots/outputs/task43_outputs/ric_singleterm/audits" / (
    "task43_ric_pk_vs_2pcf_covariance_pair_longchain.json"
)
OUTPUT_PDF = (
    ROOT / "plots/task43/rmin_scan/task43_pk_vs_2pcf_rmin_scan_contours.pdf"
    if FINAL_MODE
    else ROOT / "plots/task43/rmin_scan" / TAG / f"task43_pk_vs_2pcf_rmin_scan_{TAG}_contours.pdf"
)
OUTPUT_AUDIT = (
    PRE / "audits/task43_pk_vs_2pcf_rmin_scan_contours.json"
    if FINAL_MODE
    else PRE / f"audits/task43_pk_vs_2pcf_rmin_scan_{TAG}_contours.json"
)
PK_PREFIX = MOVED / "ric_singleterm/fits/pk" / (
    "task43_pk_ric_ph000_dchi2_nsub200000_motherbox_long_mcmc50k/"
    "task43_pk_ric_ph000_dchi2_nsub200000_motherbox_long_mcmc50k/"
    "task43_pk_lightcone_task43_pk_ric_ph000_dchi2_nsub200000_motherbox_long_mcmc50k"
)
PK_SUMMARY = PK_PREFIX.with_name(f"{PK_PREFIX.name}_fit_summary.json")
PK_SAMPLES = PK_PREFIX.with_name(f"{PK_PREFIX.name}_fit_samples.npz")
RMINS = (50, 40, 30)
COLORS = ("#2F2F2F", "#4C72B0", "#DD8452", "#C44E52")
LINESTYLES = ("-", "--", "-.", "-")
LINEWIDTHS = (2.35, 1.65, 1.85, 2.35)
MARKERS = ("o", "s", "^", "D")


def require(condition: bool, message: str) -> None:
    """把 provenance、数值或绘图 contract 不一致升级为硬失败。"""

    if not bool(condition):
        raise RuntimeError(message)


def read_json(path: Path) -> dict[str, Any]:
    """读取必须存在且非空的 JSON object。"""

    require(path.is_file() and path.stat().st_size > 0, f"缺少 JSON：{path}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    require(isinstance(payload, dict), f"JSON 顶层不是 object：{path}")
    return payload


def sha256(path: Path) -> str:
    """流式计算输入/输出 SHA256。"""

    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while True:
            block = stream.read(8 * 1024 * 1024)
            if not block:
                break
            digest.update(block)
    return digest.hexdigest()


def atomic_json(path: Path, payload: dict[str, Any]) -> None:
    """原子写入小型 JSON audit。"""

    path.parent.mkdir(parents=True, exist_ok=True)
    handle, temporary = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(handle, "w", encoding="utf-8") as stream:
            json.dump(payload, stream, indent=2, sort_keys=True)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    except Exception:
        Path(temporary).unlink(missing_ok=True)
        raise


def canonical(value: str | Path) -> Path:
    """把 summary 中可能的相对路径统一解析到项目根。"""

    path = Path(value)
    return (path if path.is_absolute() else ROOT / path).resolve(strict=False)


def close(left: float, right: float, *, atol: float = 1.0e-12) -> bool:
    """严格比较 audit 中应完全桥接的 scalar。"""

    return bool(np.isclose(float(left), float(right), rtol=0.0, atol=atol))


def validate_xi_case(
    rmin: int,
    public: dict[str, Any],
) -> tuple[Path, Path, np.ndarray, dict[str, Any], dict[str, Any]]:
    """验证一档 xN fit，读取 raw samples，并桥接总审计统计量。"""

    fit_dir = PRE / f"fits/rmin{rmin}"
    summary_path = fit_dir / "task43_minimal_closure_mcmc_summary.json"
    samples_path = fit_dir / "task43_mcmc_radial_singleterm_samples.npz"
    summary = read_json(summary_path)
    require(summary.get("status") == "done", f"rmin={rmin} summary 未完成")
    fit = summary.get("fit_range", {})
    require(
        float(fit.get("rmin", np.nan)) == float(rmin)
        and float(fit.get("rmax", np.nan)) == 350.0
        and int(fit.get("nbins", -1)) == (350 - rmin) // 10
        and int(fit.get("nreal", -1)) == NREAL,
        f"rmin={rmin} fit-range/x{NREAL} contract 错误：{fit}",
    )
    covariance = summary.get("covariance", {})
    covariance_path = str(covariance.get("path", ""))
    covariance_scope_ok = (
        ("/rmin_scan/covariance/" in covariance_path and "preliminary_" not in covariance_path)
        if FINAL_MODE
        else TAG in covariance_path
    )
    require(
        covariance.get("key") == "covariance_single_realization"
        and covariance_scope_ok
        and "rrdeconv" in Path(covariance_path).name,
        f"rmin={rmin} 没有使用 x{NREAL} RR-deconvolved jaxpower covariance",
    )
    theory = summary.get("theory", {})
    require(
        theory.get("cosmology") == "abacus_c000"
        and float(theory.get("boxsize", np.nan)) == 2000.0
        and theory.get("png_order") == "full"
        and float(theory.get("p_fixed", np.nan)) == 1.0
        and theory.get("sn0_policy") == "fixed"
        and float(theory.get("sn0_fixed", np.nan)) == 0.0
        and theory.get("xi_kernel") == "shell-averaged",
        f"rmin={rmin} theory contract 错误",
    )
    model = chain_tools.radial_model(summary)
    require(model.get("parameter_names") == ["fnl_loc", "b1"], f"rmin={rmin} 参数顺序错误")
    radial = model.get("radial_singleterm", {})
    operator_path = str(radial.get("operator", {}).get("path", ""))
    operator_scope_ok = (
        ("/rmin_scan/operators/" in operator_path and "prelimx" not in operator_path)
        if FINAL_MODE
        else SHORT in operator_path
    )
    require(
        operator_scope_ok
        and radial.get("extra_global_sigma_w2") is False,
        f"rmin={rmin} radial operator contract 错误",
    )
    mcmc = model.get("mcmc", {})
    expected_nsteps = 20000 if FINAL_MODE else 10000
    expected_burnin = 5000 if FINAL_MODE else 2000
    expected_seed = 20260803 if FINAL_MODE else 20260804
    expected_nsamples = 960000 if FINAL_MODE else 512000
    require(
        int(mcmc.get("nwalkers", -1)) == 64
        and int(mcmc.get("nsteps", -1)) == expected_nsteps
        and int(mcmc.get("burnin", -1)) == expected_burnin
        and int(mcmc.get("seed", -1)) == expected_seed
        and int(mcmc.get("nsamples", -1)) == expected_nsamples,
        f"rmin={rmin} MCMC contract 错误：{mcmc}",
    )
    with np.load(samples_path, allow_pickle=False) as data:
        raw = np.asarray(data["samples"], dtype="f8")
        log_prob = np.asarray(data["log_prob"], dtype="f8")
    require(
        raw.shape == (expected_nsamples, 2) and log_prob.shape == (expected_nsamples,),
        f"rmin={rmin} chain shape 错误",
    )
    require(np.all(np.isfinite(raw)) and np.all(np.isfinite(log_prob)), f"rmin={rmin} chain 非有限")
    samples = chain_tools.load_xi_samples(summary, samples_path)
    posterior = {
        "fnl_loc": chain_tools.summarize(samples[:, 0]),
        "b1": chain_tools.summarize(samples[:, 1]),
        "corr_fnl_b1": float(np.corrcoef(samples.T)[0, 1]),
    }
    diagnostics = chain_tools.chain_diagnostics(samples, nwalkers=64)
    require(diagnostics["pass"], f"rmin={rmin} chain gate 失败：{diagnostics}")
    for parameter in ("fnl_loc", "b1"):
        for key in ("q16", "q50", "q84", "mean", "std", "err_low", "err_high"):
            require(
                close(posterior[parameter][key], public["posterior"][parameter][key]),
                f"rmin={rmin} {parameter}.{key} 与 x{NREAL} 总审计不一致",
            )
    require(close(posterior["corr_fnl_b1"], public["posterior"]["corr_fnl_b1"]), f"rmin={rmin} corr bridge 失败")
    return summary_path, samples_path, samples, posterior, diagnostics


def validate_pk() -> tuple[np.ndarray, dict[str, Any], dict[str, Any]]:
    """验证并读取不变的 P(k) 正式长链，保留 free-sn0 口径。"""

    summary = read_json(PK_SUMMARY)
    require(summary.get("status") == "done", "P(k) summary 未完成")
    data = summary.get("data", {})
    config = summary.get("config", {})
    require(
        close(data.get("kmax_fit", np.nan), 0.10)
        and config.get("cosmology") == "abacus_c000"
        and float(config.get("p_fixed", np.nan)) == 1.0
        and config.get("sn0_policy") == "free"
        and config.get("standalone_gic") is False
        and close(config.get("window_theory_kmin", np.nan), 2.0 * np.pi / 2000.0, atol=1.0e-15),
        "P(k) kmax/cosmology/free-sn0/mother-box contract 错误",
    )
    full = contour_base.load_named_samples(PK_SAMPLES, ("fnl_loc", "b1", "sn0"))
    require(full.shape == (1080000, 3) and np.all(np.isfinite(full)), f"P(k) chain shape/finite 错误：{full.shape}")
    diagnostics = chain_tools.chain_diagnostics(full, nwalkers=int(config["nwalkers"]))
    require(diagnostics["pass"], f"P(k) chain gate 失败：{diagnostics}")
    posterior = {
        "fnl_loc": chain_tools.summarize(full[:, 0]),
        "b1": chain_tools.summarize(full[:, 1]),
        "sn0": chain_tools.summarize(full[:, 2]),
        "corr_fnl_b1": float(np.corrcoef(full[:, :2].T)[0, 1]),
    }
    return full, posterior, diagnostics


def posterior_legend_label(name: str, stats: dict[str, Any]) -> str:
    """把中位数和 16--84% 非对称误差压缩进图例。"""

    fnl = stats["fnl_loc"]
    b1 = stats["b1"]
    return (
        f"{name}\n"
        rf"$f_{{\rm NL}}={fnl['q50']:.1f}^{{+{fnl['err_high']:.1f}}}_{{-{fnl['err_low']:.1f}}},\ "
        rf"b_1={b1['q50']:.3f}^{{+{b1['err_high']:.3f}}}_{{-{b1['err_low']:.3f}}}$"
    )


def plot_contours(
    samples: list[np.ndarray],
    posterior_rows: list[dict[str, Any]],
    xlim: tuple[float, float],
    ylim: tuple[float, float],
) -> None:
    """用统一 GetDist smoothing 绘制四组 1D/2D posterior。"""

    from getdist import MCSamples, plots

    names = ["fnl_loc", "b1"]
    labels = [r"f_{\rm NL}", r"b_1"]
    ranges = {"fnl_loc": xlim, "b1": ylim}
    probe_names = [
        r"$P_0(k),\ k<0.10\,h\,{\rm Mpc}^{-1}\ [x25]$",
        rf"$\xi_0(s),\ 50<s<350\,h^{{-1}}{{\rm Mpc}}\ [x{NREAL}]$",
        rf"$\xi_0(s),\ 40<s<350\,h^{{-1}}{{\rm Mpc}}\ [x{NREAL}]$",
        rf"$\xi_0(s),\ 30<s<350\,h^{{-1}}{{\rm Mpc}}\ [x{NREAL}]$",
    ]
    legend = [
        posterior_legend_label(name, stats)
        for name, stats in zip(probe_names, posterior_rows, strict=True)
    ]
    chains = [
        MCSamples(
            samples=value,
            names=names,
            labels=labels,
            label=name,
            ranges=ranges,
            settings={
                "ignore_rows": 0,
                "smooth_scale_1D": contour_base.GETDIST_SMOOTH_1D,
                "smooth_scale_2D": contour_base.GETDIST_SMOOTH_2D,
            },
        )
        for value, name in zip(samples, legend, strict=True)
    ]
    contour_base.apply_publication_style()
    plt.rcParams.update({"font.size": 11.5, "legend.frameon": False})
    plotter = plots.get_subplot_plotter(width_inch=9.2)
    plotter.settings.axes_fontsize = 13.5
    plotter.settings.lab_fontsize = 17.0
    plotter.settings.legend_fontsize = 9.3
    plotter.settings.legend_frame = False
    plotter.settings.figure_legend_frame = False
    plotter.settings.linewidth = 1.9
    plotter.settings.linewidth_contour = 1.9
    plotter.settings.num_plot_contours = 1
    line_args = [
        {"color": color, "ls": linestyle, "lw": linewidth}
        for color, linestyle, linewidth in zip(COLORS, LINESTYLES, LINEWIDTHS, strict=True)
    ]
    plotter.triangle_plot(
        chains,
        names,
        filled=False,
        contour_colors=list(COLORS),
        contour_ls=list(LINESTYLES),
        contour_lws=list(LINEWIDTHS),
        line_args=line_args,
        legend_labels=legend,
        legend_loc="upper right",
    )
    joint_axis = plotter.subplots[1, 0]
    for stats, color, marker in zip(posterior_rows, COLORS, MARKERS, strict=True):
        joint_axis.scatter(
            stats["fnl_loc"]["q50"],
            stats["b1"]["q50"],
            marker=marker,
            s=31,
            facecolor="white",
            edgecolor=color,
            linewidth=1.5,
            zorder=20,
        )
    for row in plotter.subplots:
        for axis in row:
            if axis is not None:
                axis.grid(alpha=0.10, lw=0.5)
    OUTPUT_PDF.parent.mkdir(parents=True, exist_ok=True)
    temporary = OUTPUT_PDF.with_name(f".{OUTPUT_PDF.name}.{os.getpid()}.tmp.pdf")
    plotter.export(str(temporary))
    plt.close(plotter.fig)
    os.replace(temporary, OUTPUT_PDF)


def main() -> None:
    """完成全部 gate 后原子生成 contour PDF 与机器审计。"""

    science = read_json(SCIENCE_AUDIT)
    old = read_json(OLD_PAIR_AUDIT)
    require(
        science.get("status") == "pass" and int(science.get("nreal", -1)) == NREAL,
        f"x{NREAL} 总审计无效",
    )
    public = {int(row["rmin"]): row for row in science.get("cases", [])}
    require(set(public) == set(RMINS), f"x{NREAL} case 集合错误：{set(public)}")

    input_paths: list[Path] = [SCIENCE_AUDIT, OLD_PAIR_AUDIT, PK_SUMMARY, PK_SAMPLES]
    xi_samples: dict[int, np.ndarray] = {}
    posterior: dict[str, Any] = {}
    diagnostics: dict[str, Any] = {}
    for rmin in RMINS:
        summary_path, samples_path, samples, stats, diag = validate_xi_case(rmin, public[rmin])
        input_paths.extend([summary_path, samples_path])
        xi_samples[rmin] = samples
        posterior[f"2pcf_rmin{rmin}_{TAG}"] = stats
        diagnostics[f"2pcf_rmin{rmin}_{TAG}"] = diag

    pk_full, pk_posterior, pk_diagnostics = validate_pk()
    posterior["pk_jaxpower_reference"] = pk_posterior
    diagnostics["pk_jaxpower_reference"] = pk_diagnostics

    ranges = old.get("shared_plot_ranges", {})
    xlim = tuple(float(value) for value in ranges.get("fnl_loc", ()))
    ylim = tuple(float(value) for value in ranges.get("b1", ()))
    require(len(xlim) == 2 and xlim[0] < xlim[1], "冻结 fNL 坐标范围无效")
    require(len(ylim) == 2 and ylim[0] < ylim[1], "冻结 b1 坐标范围无效")
    plot_contours(
        [pk_full[:, :2], xi_samples[50], xi_samples[40], xi_samples[30]],
        [
            pk_posterior,
            posterior[f"2pcf_rmin50_{TAG}"],
            posterior[f"2pcf_rmin40_{TAG}"],
            posterior[f"2pcf_rmin30_{TAG}"],
        ],
        xlim,
        ylim,
    )
    require(OUTPUT_PDF.is_file() and OUTPUT_PDF.stat().st_size > 0, "contour PDF 未生成")
    with OUTPUT_PDF.open("rb") as stream:
        require(stream.read(5) == b"%PDF-", "输出不是 PDF")
    require(not list(OUTPUT_PDF.parent.glob("*.png")), "contour 目录出现 PNG")

    payload = {
        "status": "pass",
        "task": "task43_plot_rmin_scan_final_pk_contours" if FINAL_MODE else f"task43_plot_rmin_scan_{TAG}_pk_contours",
        "created_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "scope": (
            "canonical final ph000--ph024 x25 2PCF comparison against frozen x25 P(k)"
            if FINAL_MODE
            else (
                f"preliminary ph000--ph{NREAL - 1:03d} x{NREAL} 2PCF comparison against frozen x25 P(k); "
                "not final x25 and not science-ready"
            )
        ),
        "figure_contract": {
            "probes": ["P0(k)", "xi0 rmin50", "xi0 rmin40", "xi0 rmin30"],
            "pk_nuisance": "free sn0",
            "2pcf_nuisance": "fixed contact sn0=0",
            "covariance": "jaxpower single-lightcone on both probes",
            "ric": "radial single-term on both probes",
            "nested_2pcf_cases_are_correlated": True,
            "data_targets": {"pk": "x25 mean", "2pcf": f"x{NREAL} mean"},
            "matched_realization_count": NREAL == 25,
            "format": "pdf only",
            "center_statistic": "posterior median",
            "uncertainty": "q16--q84 asymmetric errors",
            "center_markers": True,
        },
        "plot_ranges": {"fnl_loc": list(xlim), "b1": list(ylim)},
        "plot_settings": {
            "smooth_scale_1d": contour_base.GETDIST_SMOOTH_1D,
            "smooth_scale_2d": contour_base.GETDIST_SMOOTH_2D,
            "colors": list(COLORS),
            "linestyles": [str(value) for value in LINESTYLES],
            "markers": list(MARKERS),
            "filled": False,
            "contour_probability": 0.68,
            "num_plot_contours": 1,
        },
        "posterior": posterior,
        "chain_diagnostics": diagnostics,
        "inputs": {str(path): sha256(path) for path in input_paths},
        "output": {"path": str(OUTPUT_PDF), "bytes": OUTPUT_PDF.stat().st_size, "sha256": sha256(OUTPUT_PDF)},
    }
    atomic_json(OUTPUT_AUDIT, payload)
    print(json.dumps({"status": "pass", "output": str(OUTPUT_PDF), "posterior": posterior}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
