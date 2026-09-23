#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""重画用户指定的 Task4.3 RascalC-2PCF / jaxpower-P(k) RIC 主图。

代码大纲
========
1. 只读取重新采样的两条长链：RascalC covariance 的 2PCF radial-RIC，
   以及 jaxpower covariance 的 P(k) radial-RIC。
2. 对 fit range、mother-box cutoff、sn0 口径、RIC operator 和 covariance
   provenance 做硬检查，防止把旧 formal-GIC/geometry-only 链混入新图。
3. 从 flat samples 恢复 walker 维，计算 integrated autocorrelation time 和
   前后半链中位数漂移；任一长链不通过门槛就拒绝输出 PDF。
4. 复用原主图的 GetDist 配色、68/95% 填充轮廓、MAP 星号和约束文本，覆盖
   用户指定的 ``task43_pk_vs_2pcf_s50_350_2pcf_rascalc_covariance.pdf``。
5. 在 outputs 下写独立 audit JSON，记录完整输入、链长、收敛量、posterior
   和旧 P(k)-RIC 链的 matched stability；plots 目录仍只保存 PDF。

科学边界
========
图中两条模型都只包含 ``model = noIC - IC^(rad,rad)``。两个 density--RIC
cross terms 没有加入，radial normalization 之外也没有再减 global sigma_W2。
本图固定既有 covariance，只比较 mean-model 的 radial single-term response。
"""

from __future__ import annotations

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
CODE_DIR = PROJECT_ROOT / "codes" / "task43"
if str(CODE_DIR) not in sys.path:
    sys.path.insert(0, str(CODE_DIR))

import task43_overlay_pk_window_kmin_contour as base  # noqa: E402


XI_DIR = (
    PROJECT_ROOT
    / "outputs/task43_outputs/ric_singleterm/fits/"
    "2pcf_rascalc_ph000_dchi2_nsub200000_long_mcmc20k"
)
XI_SUMMARY = XI_DIR / "task43_minimal_closure_mcmc_summary.json"
XI_SAMPLES = XI_DIR / "task43_mcmc_radial_singleterm_samples.npz"

PK_LABEL = "task43_pk_ric_ph000_dchi2_nsub200000_motherbox_long_mcmc50k"
# task43_fit_pk_lightcone.py 会在 --output-dir 下再创建一层 label；本次命令
# 为了把所有 long-chain 产物聚在独立根目录，显式传入了同名 output-dir，
# 因此这里忠实记录两层 label，而不是事后搬动并破坏 summary 的 paths provenance。
PK_DIR = (
    PROJECT_ROOT
    / "outputs/task43_outputs/ric_singleterm/fits/pk"
    / PK_LABEL
    / PK_LABEL
)
PK_PREFIX = PK_DIR / f"task43_pk_lightcone_{PK_LABEL}"
PK_SUMMARY = PK_PREFIX.with_name(f"{PK_PREFIX.name}_fit_summary.json")
PK_SAMPLES = PK_PREFIX.with_name(f"{PK_PREFIX.name}_fit_samples.npz")

OLD_PK_DIR = (
    PROJECT_ROOT
    / "outputs/task43_outputs/ric_singleterm/fits/pk/"
    "task43_pk_ric_ph000_dchi2_nsub200000_motherbox_mcmc5x"
)
OLD_PK_PREFIX = (
    OLD_PK_DIR
    / "task43_pk_lightcone_task43_pk_ric_ph000_dchi2_nsub200000_motherbox_mcmc5x"
)
OLD_PK_SUMMARY = OLD_PK_PREFIX.with_name(f"{OLD_PK_PREFIX.name}_fit_summary.json")

OUTPUT = (
    PROJECT_ROOT
    / "plots/task43/task43_pk_vs_2pcf_s50_350_2pcf_rascalc_covariance.pdf"
)
AUDIT = (
    PROJECT_ROOT
    / "outputs/task43_outputs/ric_singleterm/audits/"
    "task43_ric_rascalc_pk_vs_2pcf_longchain.json"
)
RASCALC_COVARIANCE = (
    PROJECT_ROOT / "outputs/task43_outputs/summary/task43_rascalc_alpha_mock_calibration.npz"
)
BASE_OPERATOR_NAME = (
    "task43_ric_factorized_operator_ph000_dchi2_nsub200000_"
    "sobol2p22_ds2_seed20260712_L2000.npz"
)


def read_json(path: Path) -> dict[str, Any]:
    """读取并返回一个必须存在的 JSON。"""
    return json.loads(Path(path).read_text(encoding="utf-8"))


def radial_model(summary: dict[str, Any]) -> dict[str, Any]:
    """从 2PCF summary 中取得唯一 radial_singleterm 记录。"""
    rows = [row for row in summary["models"] if row["model"] == "radial_singleterm"]
    if len(rows) != 1:
        raise ValueError(f"2PCF summary 中 radial_singleterm 行数为 {len(rows)}")
    return rows[0]


def load_xi_samples(summary: dict[str, Any], path: Path) -> np.ndarray:
    """按 summary 的参数名读取 2PCF fNL,b1 两列。"""
    model = radial_model(summary)
    names = [str(name) for name in model["parameter_names"]]
    with np.load(path, allow_pickle=False) as data:
        samples = np.asarray(data["samples"], dtype="f8")
    return samples[:, [names.index("fnl_loc"), names.index("b1")]]


def chain_diagnostics(samples: np.ndarray, *, nwalkers: int) -> dict[str, Any]:
    """恢复 walker 维并计算 tau、链长/tau 和 split-chain 稳定性。"""
    import emcee

    samples = np.asarray(samples, dtype="f8")
    if samples.shape[0] % int(nwalkers):
        raise ValueError("flat chain 无法按 nwalkers 整形")
    chain = samples.reshape((-1, int(nwalkers), samples.shape[1]))
    tau = np.asarray(emcee.autocorr.integrated_time(chain, quiet=True), dtype="f8")
    half = chain.shape[0] // 2
    first = chain[:half].reshape((-1, chain.shape[-1]))
    second = chain[half:].reshape((-1, chain.shape[-1]))
    sigma = np.std(samples, axis=0, ddof=1)
    split = np.abs(np.median(first, axis=0) - np.median(second, axis=0)) / sigma
    length_over_tau = chain.shape[0] / tau
    passed = bool(np.min(length_over_tau) > 100.0 and np.max(split) < 0.05)
    return {
        "post_burn_steps_per_walker": int(chain.shape[0]),
        "tau": [float(value) for value in tau],
        "length_over_tau": [float(value) for value in length_over_tau],
        "length_over_tau_min": float(np.min(length_over_tau)),
        "split_median_shift_sigma": [float(value) for value in split],
        "split_median_shift_sigma_max": float(np.max(split)),
        "finite": bool(np.all(np.isfinite(samples))),
        "gate": "length/tau > 100 and split median shift < 0.05 sigma",
        "pass": passed and bool(np.all(np.isfinite(samples))),
    }


def summarize(values: np.ndarray) -> dict[str, float]:
    """返回统一的 16/50/84 posterior summary。"""
    q16, q50, q84 = np.quantile(np.asarray(values, dtype="f8"), [0.1586552539, 0.5, 0.8413447461])
    return {
        "mean": float(np.mean(values)),
        "std": float(np.std(values, ddof=1)),
        "q16": float(q16),
        "q50": float(q50),
        "q84": float(q84),
        "err_low": float(q50 - q16),
        "err_high": float(q84 - q50),
    }


def validate_inputs(xi_summary: dict[str, Any], pk_summary: dict[str, Any]) -> None:
    """对新主图的关键物理口径做不可绕过的硬检查。"""
    fit = xi_summary["fit_range"]
    if (float(fit["rmin"]), float(fit["rmax"])) != (50.0, 350.0):
        raise ValueError(f"2PCF fit range 不匹配：{fit}")
    if Path(xi_summary["covariance"]["path"]).name != RASCALC_COVARIANCE.name:
        raise ValueError("2PCF 没有使用指定的 mock-calibrated RascalC covariance")
    theory = xi_summary["theory"]
    required_xi = (
        float(theory["boxsize"]) == 2000.0
        and float(theory["p_fixed"]) == 1.0
        and theory["sn0_policy"] == "fixed"
        and float(theory["sn0_fixed"]) == 0.0
        and theory["xi_kernel"] == "shell-averaged"
    )
    if not required_xi:
        raise ValueError("2PCF theory/sn0/shell-average 口径不匹配")

    model = radial_model(xi_summary)
    xi_operator = Path(model["radial_singleterm"]["operator"]["path"]).name
    if xi_operator != BASE_OPERATOR_NAME:
        raise ValueError(f"2PCF RIC operator 不匹配：{xi_operator}")
    if model["radial_singleterm"]["extra_global_sigma_w2"] is not False:
        raise ValueError("2PCF radial branch 意外重复减了 global sigmaW2")

    config = pk_summary["config"]
    pk_operator = Path(config["radial_singleterm_ric"]["path"]).name
    if pk_operator != BASE_OPERATOR_NAME or pk_operator != xi_operator:
        raise ValueError("P(k) 与 2PCF 没有使用同一个 baseline radial operator")
    required_pk = (
        float(pk_summary["data"]["kmax_fit"]) == 0.1
        and config["sn0_policy"] == "free"
        and np.isclose(float(config["window_theory_kmin"]), 2.0 * np.pi / 2000.0)
        and config["standalone_gic"] is False
    )
    if not required_pk:
        raise ValueError("P(k) kmax/free-sn0/mother-box/RIC 口径不匹配")


def add_map_markers(
    axes: np.ndarray,
    xi_summary: dict[str, Any],
    pk_summary: dict[str, Any],
) -> None:
    """在联合 panel 标记两条长链各自的 MAP。"""
    axis = axes[1, 0]
    xi_map = radial_model(xi_summary)["map"]
    pk_map = pk_summary["maximum_posterior_sample"]["point"]
    for point, color in ((xi_map, base.COLORS["twopcf"]), (pk_map, base.COLORS["pk"])):
        axis.scatter(
            point["fnl_loc"],
            point["b1"],
            marker="*",
            s=62,
            color=color,
            edgecolor="white",
            linewidth=0.35,
            zorder=20,
        )


def add_constraint_box(
    figure: plt.Figure,
    xi_fnl: dict[str, float],
    pk_fnl: dict[str, float],
) -> None:
    """在空白 panel 标注两条 radial-RIC 的 fNL 约束和固定-covariance 边界。"""
    label = (
        rf"2PCF RIC (RascalC): $f_{{\rm NL}}={base.constraint_text(xi_fnl)}$" "\n"
        rf"$P(k)$ RIC (jaxpower): $f_{{\rm NL}}={base.constraint_text(pk_fnl)}$" "\n"
        r"single $IC^{\rm rad,rad}$ term; fixed covariance"
    )
    figure.text(
        0.64,
        0.66,
        label,
        ha="left",
        va="top",
        fontsize=8.9,
        color="#4D4D4D",
        bbox={"boxstyle": "round,pad=0.30", "fc": "white", "ec": "#D7D7D7", "alpha": 0.90},
    )


def main() -> None:
    """执行口径/收敛门槛、绘图并写审计。"""
    base.apply_publication_style()
    xi_summary = read_json(XI_SUMMARY)
    pk_summary = read_json(PK_SUMMARY)
    validate_inputs(xi_summary, pk_summary)

    xi_samples = load_xi_samples(xi_summary, XI_SAMPLES)
    # 绘图只显示 fNL,b1，但收敛 gate 必须包含实际自由的 sn0，不能因为它
    # 不出现在 triangle panel 就漏掉最慢的 nuisance 方向。
    pk_samples_full = base.load_named_samples(PK_SAMPLES, ("fnl_loc", "b1", "sn0"))
    pk_samples = pk_samples_full[:, :2]
    xi_model = radial_model(xi_summary)
    xi_diag = chain_diagnostics(xi_samples, nwalkers=int(xi_model["mcmc"]["nwalkers"]))
    pk_diag = chain_diagnostics(pk_samples_full, nwalkers=int(pk_summary["config"]["nwalkers"]))
    if not xi_diag["pass"] or not pk_diag["pass"]:
        raise RuntimeError(f"长链没有通过绘图 gate：2PCF={xi_diag}, P(k)={pk_diag}")

    xlim = base.shared_axis_limits([xi_samples, pk_samples], 0, include_zero=True)
    ylim = base.shared_axis_limits([xi_samples, pk_samples], 1)
    labels = {
        "twopcf": (
            r"2PCF radial RIC, RascalC covariance" "\n"
            r"$s=50$--$350\,h^{-1}{\rm Mpc}$"
        ),
        "pk": (
            r"$P(k)$ radial RIC, jaxpower covariance" "\n"
            + rf"$k_{{\rm obs}}={float(pk_summary['data']['k_min']):.3f}\,--\,"
            + rf"{float(pk_summary['data']['k_max']):.3f},\ "
            + rf"k_{{\rm th,min}}={float(pk_summary['config']['window_theory_kmin']):.4f}"
            + r"\,h\,{\rm Mpc}^{-1}$"
        ),
    }
    figure, axes = base.plot_getdist_corner(
        {"twopcf": xi_samples, "pk": pk_samples},
        labels,
        xlim,
        ylim,
    )
    # GetDist 把 legend 放在空白 panel；对所有可能的 figure/axes legend
    # 同时设小一号字体，保证 bbox_inches='tight' 后没有右侧文字裁切。
    legends = list(figure.legends)
    legends.extend(
        axis.get_legend()
        for axis in np.asarray(axes).ravel()
        if axis is not None and axis.get_legend() is not None
    )
    for legend in legends:
        for text in legend.get_texts():
            text.set_fontsize(8.8)
    add_map_markers(axes, xi_summary, pk_summary)
    xi_fnl = summarize(xi_samples[:, 0])
    pk_fnl = summarize(pk_samples[:, 0])
    add_constraint_box(figure, xi_fnl, pk_fnl)
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(OUTPUT, bbox_inches="tight")
    plt.close(figure)

    old_pk = read_json(OLD_PK_SUMMARY)
    old_pk_fnl = base.normalized_summary(old_pk["parameters"]["fnl_loc"])
    old_sigma = 0.5 * (old_pk_fnl["q84"] - old_pk_fnl["q16"])
    stability = {
        "comparison": "new 24x50000 independent-seed P(k)-RIC versus old 18x33334 P(k)-RIC",
        "old_fnl": old_pk_fnl,
        "new_fnl": pk_fnl,
        "median_shift_over_old_sigma": float((pk_fnl["q50"] - old_pk_fnl["q50"]) / old_sigma),
        "sigma68_ratio": float(
            (0.5 * (pk_fnl["q84"] - pk_fnl["q16"])) / old_sigma
        ),
    }
    audit = {
        "task": "task43_plot_ric_rascalc_pk_vs_2pcf",
        "status": "done",
        "created_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "figure_contract": {
            "model": "radial single-term RIC on both probes",
            "fit_sign": "model = noIC - IC^(rad,rad)",
            "density_ric_cross_terms": False,
            "extra_global_sigma_w2": False,
            "covariances_changed": False,
            "twopcf_covariance": "mock-calibrated FullDiscrete RascalC single-lightcone",
            "pk_covariance": "jaxpower Gaussian survey-window single-lightcone",
            "same_radial_operator": True,
        },
        "inputs": {
            "twopcf_summary": str(XI_SUMMARY),
            "twopcf_samples": str(XI_SAMPLES),
            "pk_summary": str(PK_SUMMARY),
            "pk_samples": str(PK_SAMPLES),
            "rascalc_covariance": str(RASCALC_COVARIANCE),
            "radial_operator_name": BASE_OPERATOR_NAME,
        },
        "chain_diagnostics": {"2pcf": xi_diag, "pk": pk_diag},
        "posterior": {
            "2pcf": {"fnl_loc": xi_fnl, "b1": summarize(xi_samples[:, 1])},
            "pk": {"fnl_loc": pk_fnl, "b1": summarize(pk_samples[:, 1])},
        },
        "pk_long_chain_stability": stability,
        "plot_ranges": {"fnl_loc": xlim, "b1": ylim},
        "plot_settings": {
            "backend": "matplotlib + getdist",
            "smooth_scale_1d": base.GETDIST_SMOOTH_1D,
            "smooth_scale_2d": base.GETDIST_SMOOTH_2D,
            "format": "pdf only",
        },
        "output_pdf": str(OUTPUT),
    }
    AUDIT.parent.mkdir(parents=True, exist_ok=True)
    AUDIT.write_text(
        json.dumps(base.jsonable(audit), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(f"[write] {OUTPUT}")
    print(f"[write] {AUDIT}")
    print(
        "[posterior] 2PCF fNL={:.2f} -{:.2f} +{:.2f}; "
        "P(k) fNL={:.2f} -{:.2f} +{:.2f}".format(
            xi_fnl["q50"], xi_fnl["err_low"], xi_fnl["err_high"],
            pk_fnl["q50"], pk_fnl["err_low"], pk_fnl["err_high"],
        )
    )


if __name__ == "__main__":
    main()
