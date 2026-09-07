#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
代码大纲（执行逻辑关系）：

1. 读取 Quijote `fid / LCp50 / LCp100` 的 P(k) 与 2PCF mocks。
2. 复用 task42 的 `BinAvgFit + FullDiscrete` 理论口径：
   - P(k) case: 对理论 P0(k_q) 做离散壳层 bin-average；
   - 2PCF case: 用 FullDiscrete/CachedRebin 计算 xi0(s)。
3. 对每个 tag 和 case 用 UltraNest 采样参数：
   默认采样 `fnl_loc, b1, sigmas` 并固定 `p=1.2, sn0=0`；
   如果命令行加 `--free-sn0`，则额外采样 `sn0`。
4. 用 Hartlap precision 写入 likelihood，并在报告误差时乘 Percival factor。
5. 输出：
   - 每个 case 的 UltraNest run 目录；
   - 每个 case 的 posterior samples npz；
   - 汇总 JSON/CSV；
   - 类似 task42 profiler 图的 all-tag forest PDF；
   - 每个 tag/case 的 corner contour PDF。

说明：
- 本脚本不覆盖 task42 profiler 结果；所有产物放在新的
  `outputs/task4_outputs/quijote_ultranest/` 与
  `plots/task4/quijote_ultranest/` 下。
- 运行时请显式限制线程数，避免登录节点超过 8 核。
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import time
from dataclasses import dataclass
from pathlib import Path

for _name in (
    "OMP_NUM_THREADS",
    "MKL_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
    "JAX_NUM_THREADS",
):
    os.environ.setdefault(_name, "1")
os.environ.setdefault("XLA_FLAGS", "--xla_cpu_multi_thread_eigen=false intra_op_parallelism_threads=1")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import ultranest

from task42_quijote_lcp50_profiler import (
    DATA_DIR,
    K_FUND,
    KMAX_DISCRETE,
    KMAX_FIT,
    P_FIXED,
    SN0_FIXED,
    VOLUME,
    RSDModel,
    bin_shell_indices,
    binavg_pk,
    build_cosmology,
    build_rlist_selections,
    covariance_corrections,
    gq_enumerate,
    gq_fft,
    j0_matrix,
    load_pcf_stack,
    load_pk_stack,
    precompute_rebin_cache,
    shell_arrays,
)


PROJECT_ROOT = Path("/pscratch/sd/l/lzy/PNG-2PCF-model-to-observe")
OUTPUT_ROOT = PROJECT_ROOT / "outputs" / "task4_outputs" / "quijote_ultranest"
PLOT_ROOT = PROJECT_ROOT / "plots" / "task4" / "quijote_ultranest"
RUN_STEM = "task45_quijote_ultranest"
CASE_STEM = "task45"
SN0_POLICY = "fixed"

BASE_PARAM_NAMES = ["fnl_loc", "b1", "sigmas"]
PARAM_NAMES = list(BASE_PARAM_NAMES)
PARAM_LABELS = {
    "fnl_loc": r"$f_{\mathrm{NL}}^{\mathrm{loc}}$",
    "b1": r"$b_1$",
    "sigmas": r"$\sigma_s\,[h^{-1}{\rm Mpc}]$",
    "sn0": r"$s_{n,0}\,[(h^{-1}{\rm Mpc})^3]$",
}
DEFAULT_PARAM_PRIORS = {
    "fnl_loc": (-500.0, 500.0),
    "b1": (0.5, 5.0),
    "sigmas": (0.0, 30.0),
    "sn0": (-1.0, 1.0),
}
PARAM_PRIORS = {name: DEFAULT_PARAM_PRIORS[name] for name in PARAM_NAMES}
TAG_ORDER = ["fid", "LCp50", "LCp100"]
DEFAULT_CASES = ["pk_binavg", "xi_r50_350", "xi_r60_350", "xi_r80_350", "xi_r100_350"]


@dataclass
class LikelihoodContext:
    """
    单个 UltraNest likelihood 所需的全部缓存。

    参数：
    - tag: Quijote tag，如 fid/LCp50/LCp100。
    - case: `pk_binavg` 或某个 `xi_rXX_350`。
    - data: 数据向量。
    - precision: Hartlap 修正后的 inverse covariance。
    - corrections: Hartlap/Percival 等协方差信息。
    - model: desilike RSD P0(k) evaluator。
    - extra: P(k) bin 或 2PCF kernel 所需的额外数组。
    """

    tag: str
    case: str
    data: np.ndarray
    precision: np.ndarray
    corrections: dict[str, float]
    model: RSDModel
    extra: dict[str, np.ndarray]


def ensure_dirs() -> None:
    """创建 task45 需要的输出目录。"""
    for path in [OUTPUT_ROOT, OUTPUT_ROOT / "samples", OUTPUT_ROOT / "ultranest_runs", PLOT_ROOT, PLOT_ROOT / "corners"]:
        path.mkdir(parents=True, exist_ok=True)


def parse_csv(value: str) -> list[str]:
    """解析逗号分隔命令行列表。"""
    return [item.strip() for item in value.split(",") if item.strip()]


def parse_pair(value: str) -> tuple[float, float]:
    """
    解析命令行里的二元浮点数区间。

    输入：
    - value: 形如 `lo,hi` 的字符串。

    输出：
    - `(lo, hi)` 浮点数元组；若 `lo >= hi` 则报错。
    """
    items = parse_csv(value)
    if len(items) != 2:
        raise argparse.ArgumentTypeError(f"expected two comma-separated values, got {value!r}")
    lo, hi = float(items[0]), float(items[1])
    if not lo < hi:
        raise argparse.ArgumentTypeError(f"expected lo < hi, got {value!r}")
    return lo, hi


def configure_run(args: argparse.Namespace) -> None:
    """
    根据命令行配置本轮 UltraNest run 的参数空间和输出目录。

    设计原则：
    - 默认 fixed-sn0 行为完全保持原来的输出路径和文件名；
    - `--free-sn0` 默认写入 `quijote_ultranest_free_sn0`，避免覆盖主线；
    - 如需额外实验，可用 `--output-label` 生成更明确的独立目录。
    """
    global OUTPUT_ROOT, PLOT_ROOT, RUN_STEM, CASE_STEM, SN0_POLICY, PARAM_NAMES, PARAM_PRIORS, KMAX_FIT

    KMAX_FIT = float(args.kmax_fit)
    label = args.output_label.strip()
    if args.free_sn0:
        SN0_POLICY = "free"
        PARAM_NAMES = [*BASE_PARAM_NAMES, "sn0"]
        PARAM_PRIORS = {name: DEFAULT_PARAM_PRIORS[name] for name in PARAM_NAMES}
        PARAM_PRIORS["sn0"] = tuple(args.sn0_prior)
        if not label:
            label = "free_sn0"
    else:
        SN0_POLICY = "fixed"
        PARAM_NAMES = list(BASE_PARAM_NAMES)
        PARAM_PRIORS = {name: DEFAULT_PARAM_PRIORS[name] for name in PARAM_NAMES}

    if label:
        clean_label = label.replace("/", "_")
        OUTPUT_ROOT = PROJECT_ROOT / "outputs" / "task4_outputs" / f"quijote_ultranest_{clean_label}"
        PLOT_ROOT = PROJECT_ROOT / "plots" / "task4" / f"quijote_ultranest_{clean_label}"
        RUN_STEM = f"task45_quijote_ultranest_{clean_label}"
        CASE_STEM = f"task45_{clean_label}"
    else:
        OUTPUT_ROOT = PROJECT_ROOT / "outputs" / "task4_outputs" / "quijote_ultranest"
        PLOT_ROOT = PROJECT_ROOT / "plots" / "task4" / "quijote_ultranest"
        RUN_STEM = "task45_quijote_ultranest"
        CASE_STEM = "task45"


def hartlap_precision(cov: np.ndarray, nmock: int, nparams: int) -> tuple[np.ndarray, dict[str, float]]:
    """
    对 mock covariance 施加 Hartlap correction，返回 precision 与元数据。

    输入：
    - cov: sample covariance。
    - nmock: mock 数量。
    - nparams: likelihood 中自由参数个数。

    输出：
    - precision: Hartlap 修正后的逆协方差。
    - corrections: 包含 Hartlap/Percival 因子的字典。
    """
    ndata = int(cov.shape[0])
    corrections = dict(covariance_corrections(nmock=nmock, ndata=ndata, nparams=nparams))
    corrections.update(
        {
            "nmock": int(nmock),
            "ndata": ndata,
            "nparams": int(nparams),
            "covariance_estimator": "C_sample",
            "used_covariance_divided_by_nmock": False,
            "condition_number": float(np.linalg.cond(cov)),
        }
    )
    return corrections["hartlap"] * np.linalg.pinv(cov, rcond=1.0e-10), corrections


def weighted_quantile(values: np.ndarray, weights: np.ndarray, quantiles: list[float]) -> np.ndarray:
    """
    计算 weighted posterior quantile。

    UltraNest 的 weighted samples 不是等权样本，因此 forest 图和 summary
    都用这个函数计算 median/16/84 分位数。
    """
    values = np.asarray(values, dtype="f8")
    weights = np.asarray(weights, dtype="f8")
    mask = np.isfinite(values) & np.isfinite(weights) & (weights >= 0.0)
    values = values[mask]
    weights = weights[mask]
    if values.size == 0 or np.sum(weights) <= 0:
        return np.full(len(quantiles), np.nan)
    order = np.argsort(values)
    values = values[order]
    weights = weights[order]
    cdf = np.cumsum(weights)
    cdf /= cdf[-1]
    return np.interp(quantiles, cdf, values)


def resample_equal_weight(points: np.ndarray, weights: np.ndarray, size: int, seed: int) -> np.ndarray:
    """
    从 weighted posterior 中重采样等权样本，供 corner 图使用。

    参数：
    - points: shape=(Nsample, Nparam) 的 posterior 点。
    - weights: 对应权重。
    - size: 最大重采样数量。
    - seed: 随机种子。
    """
    points = np.asarray(points, dtype="f8")
    weights = np.asarray(weights, dtype="f8")
    mask = np.all(np.isfinite(points), axis=1) & np.isfinite(weights) & (weights >= 0.0)
    points = points[mask]
    weights = weights[mask]
    if points.size == 0 or np.sum(weights) <= 0:
        raise ValueError("weighted posterior is empty")
    weights = weights / np.sum(weights)
    rng = np.random.default_rng(seed)
    n = min(int(size), points.shape[0])
    idx = rng.choice(points.shape[0], size=n, replace=True, p=weights)
    return points[idx]


def make_pk_context(tag: str, cosmo) -> LikelihoodContext:
    """构建 P(k) BinAvgFit UltraNest likelihood 缓存。"""
    pk = load_pk_stack(tag)
    fit_mask = pk["kcen"] <= KMAX_FIT
    data = pk["mean"][fit_mask]
    cov = pk["cov"][np.ix_(fit_mask, fit_mask)]
    precision, corrections = hartlap_precision(cov, nmock=int(pk["mocks"].shape[0]), nparams=len(PARAM_NAMES))

    kmin = pk["kmin"][fit_mask]
    kmax = pk["kmax"][fit_mask]
    qmax = int(np.floor((float(kmax[-1]) / K_FUND) ** 2)) + 1
    gq = gq_enumerate(qmax)
    k_shell, g_shell = shell_arrays(gq, kmax=float(kmax[-1]))
    indices = bin_shell_indices(k_shell, kmin, kmax)
    model = RSDModel(cosmo, k_shell, sigmas_init=5.0)
    extra = {
        "g_shell": g_shell,
        "indices_object": np.array(indices, dtype=object),
        "kmin": kmin,
        "kmax": kmax,
    }
    return LikelihoodContext(tag=tag, case="pk_binavg", data=data, precision=precision, corrections=corrections, model=model, extra=extra)


def make_xi_context(tag: str, case: str, pcf: dict[str, np.ndarray], cosmo, g_cache: np.ndarray, k_eff: np.ndarray) -> LikelihoodContext:
    """构建某个 2PCF rlist 的 UltraNest likelihood 缓存。"""
    rmin = float(case.split("_r", 1)[1].split("_", 1)[0])
    selections = build_rlist_selections(pcf, [rmin], 350.0, ["contiguous"])
    indices = selections[f"r{int(rmin)}_350"]
    data = pcf["mean"][indices]
    cov = pcf["cov"][np.ix_(indices, indices)]
    precision, corrections = hartlap_precision(cov, nmock=int(pcf["mocks"].shape[0]), nparams=len(PARAM_NAMES))
    model = RSDModel(cosmo, k_eff, sigmas_init=10.0)
    extra = {
        "indices": indices,
        "s": pcf["s"][indices],
        "g_cache": g_cache,
        "kernel": j0_matrix(k_eff, pcf["s"][indices]),
    }
    return LikelihoodContext(tag=tag, case=case, data=data, precision=precision, corrections=corrections, model=model, extra=extra)


def build_contexts(tags: list[str], cases: list[str]) -> dict[tuple[str, str], LikelihoodContext]:
    """
    为所有 tag/case 构建 likelihood 缓存。

    FullDiscrete 的 `g_cache/k_eff` 与 tag 无关，只构建一次。
    """
    cosmo = build_cosmology()
    qmax = int((KMAX_DISCRETE / K_FUND) ** 2)
    nmax = int(KMAX_DISCRETE / K_FUND)
    gq = gq_fft(qmax, nmax)
    g_cache, k_eff = precompute_rebin_cache(gq, KMAX_DISCRETE)

    contexts: dict[tuple[str, str], LikelihoodContext] = {}
    for tag in tags:
        pcf = load_pcf_stack(tag)
        if "pk_binavg" in cases:
            contexts[(tag, "pk_binavg")] = make_pk_context(tag, cosmo)
        for case in cases:
            if case.startswith("xi_"):
                contexts[(tag, case)] = make_xi_context(tag, case, pcf, cosmo, g_cache, k_eff)
    return contexts


def prior_transform(unit_cube: np.ndarray) -> np.ndarray:
    """
    UltraNest prior transform: [0,1]^N -> physical parameters.

    当前固定先验：
    - fnl_loc: [-500, 500]
    - b1: [0.5, 5.0]
    - sigmas: [0, 30]
    - sn0: 默认只在 `--free-sn0` 时启用，区间由 `--sn0-prior` 控制。
    """
    u = np.asarray(unit_cube, dtype="f8")
    out = np.empty_like(u)
    for i, name in enumerate(PARAM_NAMES):
        lo, hi = PARAM_PRIORS[name]
        out[i] = lo + (hi - lo) * u[i]
    return out


def loglike_from_context(context: LikelihoodContext):
    """把 `LikelihoodContext` 转成 UltraNest loglike callable。"""

    def unpack_theta(theta) -> dict[str, float]:
        """把 UltraNest 参数数组转成命名参数，并为 fixed-sn0 补上默认值。"""
        values = {name: float(theta[i]) for i, name in enumerate(PARAM_NAMES)}
        values.setdefault("sn0", SN0_FIXED)
        return values

    if context.case == "pk_binavg":
        g_shell = context.extra["g_shell"]
        indices = list(context.extra["indices_object"])

        def loglike(theta):
            pars = unpack_theta(theta)
            pk_shell = context.model.pk0(
                fnl_loc=pars["fnl_loc"],
                b1=pars["b1"],
                sigmas=pars["sigmas"],
                sn0=pars["sn0"],
            )
            theory = binavg_pk(pk_shell, g_shell, indices)
            diff = context.data - theory
            return -0.5 * float(diff @ context.precision @ diff)

    else:
        g_cache = context.extra["g_cache"]
        kernel = context.extra["kernel"]

        def loglike(theta):
            pars = unpack_theta(theta)
            pk = context.model.pk0(
                fnl_loc=pars["fnl_loc"],
                b1=pars["b1"],
                sigmas=pars["sigmas"],
                sn0=pars["sn0"],
            )
            theory = (g_cache * pk) @ kernel / VOLUME
            diff = context.data - theory
            return -0.5 * float(diff @ context.precision @ diff)

    return loglike


def summarize_result(tag: str, case: str, result: dict, context: LikelihoodContext, elapsed: float) -> dict[str, object]:
    """从 UltraNest result 生成机器可读 summary。"""
    points = np.asarray(result["weighted_samples"]["points"], dtype="f8")
    weights = np.asarray(result["weighted_samples"]["weights"], dtype="f8")
    q16, q50, q84 = 0.15865525393145707, 0.5, 0.8413447460685429
    percival = float(context.corrections["percival_error_factor"])
    params = {}
    for i, name in enumerate(PARAM_NAMES):
        lo, med, hi = weighted_quantile(points[:, i], weights, [q16, q50, q84])
        mean = float(np.average(points[:, i], weights=weights / np.sum(weights)))
        variance = float(np.average((points[:, i] - mean) ** 2, weights=weights / np.sum(weights)))
        params[name] = {
            "mean": mean,
            "std": float(np.sqrt(max(variance, 0.0))),
            "median": float(med),
            "q16": float(lo),
            "q84": float(hi),
            "err_low": float(med - lo),
            "err_high": float(hi - med),
            "err_low_percival": float((med - lo) * percival),
            "err_high_percival": float((hi - med) * percival),
        }
    ml = result.get("maximum_likelihood", {})
    return {
        "tag": tag,
        "case": case,
        "parameters": params,
        "maximum_likelihood": {
            "logl": float(ml.get("logl", np.nan)),
            "point": {name: float(ml.get("point", [np.nan] * len(PARAM_NAMES))[i]) for i, name in enumerate(PARAM_NAMES)},
        },
        "ultranest": {
            "logz": float(result.get("logz", np.nan)),
            "logzerr": float(result.get("logzerr", np.nan)),
            "ncall": int(result.get("ncall", -1)),
            "niter": int(result.get("niter", -1)),
            "ess": float(result.get("ess", np.nan)),
        },
        "covariance_corrections": context.corrections,
        "elapsed_sec": elapsed,
    }


def save_samples(tag: str, case: str, result: dict, summary: dict[str, object]) -> str:
    """保存 weighted posterior samples 到 npz。"""
    path = OUTPUT_ROOT / "samples" / f"{CASE_STEM}_{tag}_{case}_weighted_samples.npz"
    ws = result["weighted_samples"]
    np.savez_compressed(
        path,
        param_names=np.array(PARAM_NAMES),
        points=np.asarray(ws["points"], dtype="f8"),
        weights=np.asarray(ws["weights"], dtype="f8"),
        logl=np.asarray(ws["logl"], dtype="f8"),
        logw=np.asarray(ws["logw"], dtype="f8"),
        maximum_likelihood_point=np.asarray(result["maximum_likelihood"]["point"], dtype="f8"),
        summary_json=json.dumps(summary),
    )
    return str(path)


def run_one(context: LikelihoodContext, args: argparse.Namespace) -> dict[str, object]:
    """运行单个 tag/case 的 UltraNest 采样。"""
    run_dir = OUTPUT_ROOT / "ultranest_runs" / context.tag / context.case
    loglike = loglike_from_context(context)
    sampler = ultranest.ReactiveNestedSampler(
        PARAM_NAMES,
        loglike,
        prior_transform,
        log_dir=str(run_dir),
        resume="overwrite" if args.overwrite else "subfolder",
        vectorized=False,
        num_test_samples=2,
    )
    t0 = time.time()
    result = sampler.run(
        min_num_live_points=args.min_live,
        dlogz=args.dlogz,
        min_ess=args.min_ess,
        max_ncalls=args.max_ncalls,
        show_status=not args.quiet,
        viz_callback=False,
    )
    elapsed = time.time() - t0
    summary = summarize_result(context.tag, context.case, result, context, elapsed)
    summary["paths"] = {
        "run_dir": str(run_dir),
        "samples_npz": save_samples(context.tag, context.case, result, summary),
    }
    return summary


def load_sample_npz(path: str) -> tuple[np.ndarray, np.ndarray]:
    """读取保存的 weighted sample npz。"""
    data = np.load(path)
    return data["points"], data["weights"]


def plot_corner_for_summary(summary: dict[str, object], max_points: int, seed: int) -> str:
    """为一个 tag/case 画 corner contour。"""
    import corner

    points, weights = load_sample_npz(summary["paths"]["samples_npz"])
    samples = resample_equal_weight(points, weights, size=max_points, seed=seed)
    fig = corner.corner(
        samples,
        labels=[PARAM_LABELS[name] for name in PARAM_NAMES],
        color="#315f9f",
        plot_datapoints=False,
        fill_contours=True,
        levels=(0.393, 0.865),
        hist_kwargs={"density": True, "histtype": "stepfilled", "alpha": 0.35},
        contour_kwargs={"linewidths": 1.4},
        label_kwargs={"fontsize": 12},
    )
    fig.suptitle(f"{summary['tag']} {summary['case']} UltraNest posterior ({SN0_POLICY} sn0)", fontsize=13, fontweight="bold")
    out = PLOT_ROOT / "corners" / f"{CASE_STEM}_{summary['tag']}_{summary['case']}_corner.pdf"
    fig.savefig(out)
    plt.close(fig)
    return str(out)


def plot_forest(summaries: list[dict[str, object]], out_pdf: Path) -> str:
    """画类似 task42 profiler 的 all-tag forest/errorbar 图。"""
    case_label = {
        "pk_binavg": "P(k) BinAvgFit",
        "xi_r50_350": "2PCF 55-345",
        "xi_r60_350": "2PCF 65-345",
        "xi_r80_350": "2PCF 85-345",
        "xi_r100_350": "2PCF 105-345",
    }
    rows = []
    by_key = {(item["tag"], item["case"]): item for item in summaries}
    for tag in TAG_ORDER:
        tag_cases = [case for case in DEFAULT_CASES if (tag, case) in by_key]
        for case in tag_cases:
            rows.append(by_key[(tag, case)])
        if tag_cases:
            rows.append({"tag": tag, "case": "gap"})
    if rows and rows[-1].get("case") == "gap":
        rows.pop()

    y = np.arange(len(rows))[::-1]
    ncols = len(PARAM_NAMES)
    fig_width = 19.0 if ncols == 3 else 22.5
    fig, axes = plt.subplots(1, ncols, figsize=(fig_width, 10.5), sharey=True, gridspec_kw={"wspace": 0.17})
    axes = np.atleast_1d(axes)
    fig.patch.set_facecolor("#f7f7f5")
    colors = {"fid": "#3c6e9f", "LCp50": "#bf5b2f", "LCp100": "#417a50"}
    markers = {"pk_binavg": "s"}

    for ax, pname in zip(axes, PARAM_NAMES):
        ax.set_facecolor("#fbfbfa")
        lo_all, hi_all = [], []
        for i, row in enumerate(rows):
            yi = y[i]
            if row.get("case") == "gap":
                ax.axhline(yi, color="#a6a6a6", lw=0.9, alpha=0.55, zorder=0)
                continue
            item = row["parameters"][pname]
            med = item["median"]
            elo = item["err_low_percival"]
            ehi = item["err_high_percival"]
            lo_all.append(med - elo)
            hi_all.append(med + ehi)
            if i % 2 == 0:
                ax.axhspan(yi - 0.42, yi + 0.42, color="#ececea", alpha=0.45, zorder=0)
            ax.errorbar(
                med,
                yi,
                xerr=np.array([[elo], [ehi]]),
                fmt=markers.get(row["case"], "o"),
                ms=7.0,
                mfc="white",
                mec=colors[row["tag"]],
                mew=2.0,
                ecolor=colors[row["tag"]],
                elinewidth=1.9,
                capsize=3.8,
                capthick=1.7,
                alpha=0.95,
                zorder=3,
            )
        ax.set_title(PARAM_LABELS[pname], pad=12)
        ax.set_xlabel("posterior median +/- 1 sigma")
        if lo_all:
            xmin, xmax = float(np.nanmin(lo_all)), float(np.nanmax(hi_all))
            span = xmax - xmin
            ax.set_xlim(xmin - 0.08 * span, xmax + 0.08 * span)
        ax.grid(axis="x", color="#c9c9c9", lw=0.8, alpha=0.65)
        ax.tick_params(axis="y", length=0)

    labels = []
    for row in rows:
        if row.get("case") == "gap":
            labels.append("")
        else:
            labels.append(f"{row['tag']}  {case_label.get(row['case'], row['case'])}")
    axes[0].set_yticks(y)
    axes[0].set_yticklabels(labels)
    for ax in axes[1:]:
        ax.tick_params(labelleft=False)

    handles = []
    for tag, color in colors.items():
        handles.append(axes[0].plot([], [], "o", ms=8, mfc="white", mec=color, mew=2, label=tag)[0])
    handles.extend(
        [
            axes[0].plot([], [], "s", ms=8, mfc="white", mec="#555555", mew=2, label="P(k)")[0],
            axes[0].plot([], [], "o", ms=8, mfc="white", mec="#555555", mew=2, label="2PCF")[0],
        ]
    )
    fig.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.54, 0.965), ncol=5, frameon=False)
    fig.suptitle(
        f"Quijote local-PNG tags: UltraNest posterior constraints ({SN0_POLICY} sn0)",
        x=0.54,
        y=0.995,
        fontsize=17,
        fontweight="bold",
    )
    sn0_note = (
        f"sn0 sampled with uniform prior [{PARAM_PRIORS['sn0'][0]:.3g}, {PARAM_PRIORS['sn0'][1]:.3g}]."
        if "sn0" in PARAM_NAMES
        else "sn0 fixed to 0."
    )
    fig.text(
        0.54,
        0.035,
        f"P(k): BinAvgFit, kcen <= {KMAX_FIT:.3g} h/Mpc.  2PCF: FullDiscrete, rmax = 350 Mpc/h.  {sn0_note}",
        ha="center",
        va="center",
        fontsize=11,
        color="#4d4d4d",
    )
    fig.subplots_adjust(left=0.29, right=0.985, top=0.90, bottom=0.085)
    fig.savefig(out_pdf)
    plt.close(fig)
    return str(out_pdf)


def write_tables(summaries: list[dict[str, object]]) -> dict[str, str]:
    """写出 CSV/Markdown 简表。"""
    csv_path = OUTPUT_ROOT / f"{RUN_STEM}_summary_table.csv"
    md_path = OUTPUT_ROOT / f"{RUN_STEM}_summary_table.md"
    rows = []
    for item in summaries:
        for pname in PARAM_NAMES:
            p = item["parameters"][pname]
            rows.append(
                {
                    "tag": item["tag"],
                    "case": item["case"],
                    "parameter": pname,
                    "median": p["median"],
                    "err_low_percival": p["err_low_percival"],
                    "err_high_percival": p["err_high_percival"],
                    "ml": item["maximum_likelihood"]["point"][pname],
                    "ncall": item["ultranest"]["ncall"],
                    "logz": item["ultranest"]["logz"],
                    "logzerr": item["ultranest"]["logzerr"],
                }
            )
    with csv_path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    with md_path.open("w", encoding="utf-8") as file:
        file.write("| tag | case | parameter | median | -1sigma | +1sigma | ML | ncall |\n")
        file.write("|---|---|---:|---:|---:|---:|---:|---:|\n")
        for row in rows:
            file.write(
                f"| {row['tag']} | {row['case']} | {row['parameter']} | "
                f"{row['median']:.6g} | {row['err_low_percival']:.6g} | "
                f"{row['err_high_percival']:.6g} | {row['ml']:.6g} | {row['ncall']} |\n"
            )
    return {"csv": str(csv_path), "markdown": str(md_path)}


def parse_args() -> argparse.Namespace:
    """解析命令行参数。"""
    parser = argparse.ArgumentParser(description="Task45 Quijote UltraNest MCMC/nested posterior")
    parser.add_argument("--tags", default="fid,LCp50,LCp100", help="comma separated tags")
    parser.add_argument("--cases", default=",".join(DEFAULT_CASES), help="comma separated cases")
    parser.add_argument("--free-sn0", action="store_true", help="sample sn0 instead of fixing sn0=0")
    parser.add_argument(
        "--sn0-prior",
        type=parse_pair,
        default=DEFAULT_PARAM_PRIORS["sn0"],
        help="uniform sn0 prior for --free-sn0, formatted as lo,hi",
    )
    parser.add_argument(
        "--output-label",
        default="",
        help="optional suffix for output/plot roots; --free-sn0 defaults to free_sn0",
    )
    parser.add_argument("--min-live", type=int, default=120, help="UltraNest min_num_live_points")
    parser.add_argument("--dlogz", type=float, default=0.5, help="UltraNest evidence stopping threshold")
    parser.add_argument("--min-ess", type=int, default=200, help="UltraNest min effective posterior samples")
    parser.add_argument("--max-ncalls", type=int, default=5000, help="UltraNest max likelihood calls per case")
    parser.add_argument("--kmax-fit", type=float, default=KMAX_FIT, help="maximum P(k) bin center used in BinAvgFit, in h/Mpc")
    parser.add_argument("--corner-max-points", type=int, default=6000, help="max resampled points per corner plot")
    parser.add_argument("--seed", type=int, default=12345, help="resampling seed for plots")
    parser.add_argument("--overwrite", action="store_true", help="overwrite UltraNest run subfolders")
    parser.add_argument("--quiet", action="store_true", help="suppress UltraNest live status")
    parser.add_argument("--skip-existing", action="store_true", help="reuse existing sample npz and summary JSON if present")
    parser.add_argument(
        "--skip-final-products",
        action="store_true",
        help="worker mode: write per-case summaries/corners only, skipping shared tables/forest/manifest",
    )
    return parser.parse_args()


def main() -> None:
    """主程序入口。"""
    args = parse_args()
    configure_run(args)
    ensure_dirs()
    tags = parse_csv(args.tags)
    cases = parse_csv(args.cases)
    invalid = sorted(set(cases) - set(DEFAULT_CASES))
    if invalid:
        raise ValueError(f"unknown cases: {invalid}; allowed={DEFAULT_CASES}")

    contexts = build_contexts(tags, cases)
    summaries: list[dict[str, object]] = []
    for tag in tags:
        for case in cases:
            key = (tag, case)
            if key not in contexts:
                continue
            summary_json = OUTPUT_ROOT / f"{CASE_STEM}_{tag}_{case}_summary.json"
            if args.skip_existing and summary_json.exists():
                print(f"[reuse] {summary_json}", flush=True)
                summary = json.loads(summary_json.read_text())
            else:
                print(f"[run] tag={tag} case={case}", flush=True)
                summary = run_one(contexts[key], args)
                summary_json.write_text(json.dumps(summary, indent=2), encoding="utf-8")
            corner_pdf = plot_corner_for_summary(summary, max_points=args.corner_max_points, seed=args.seed)
            summary["paths"]["corner_pdf"] = corner_pdf
            summary_json.write_text(json.dumps(summary, indent=2), encoding="utf-8")
            summaries.append(summary)

    if args.skip_final_products:
        print("[skip] final tables/forest/manifest", flush=True)
        return

    forest_pdf = plot_forest(summaries, PLOT_ROOT / f"{RUN_STEM}_alltags_constraints.pdf")
    table_paths = write_tables(summaries)
    manifest = {
        "task": "task45_quijote_ultranest",
        "run_stem": RUN_STEM,
        "status": "done",
        "inputs": {
            "data_dir": str(DATA_DIR),
            "tags": tags,
            "cases": cases,
            "parameter_names": PARAM_NAMES,
            "priors": PARAM_PRIORS,
            "fixed": {"p": P_FIXED, "kmax_fit": KMAX_FIT, "kmax_discrete": KMAX_DISCRETE, **({} if "sn0" in PARAM_NAMES else {"sn0": SN0_FIXED})},
            "sn0_policy": SN0_POLICY,
            "thread_limits": {
                name: os.environ.get(name)
                for name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS", "JAX_NUM_THREADS")
            },
            "ultranest_settings": {
                "min_live": args.min_live,
                "dlogz": args.dlogz,
                "min_ess": args.min_ess,
                "max_ncalls": args.max_ncalls,
            },
        },
        "outputs": {
            "output_root": str(OUTPUT_ROOT),
            "plot_root": str(PLOT_ROOT),
            "forest_pdf": forest_pdf,
            "tables": table_paths,
            "summaries": [str(OUTPUT_ROOT / f"{CASE_STEM}_{item['tag']}_{item['case']}_summary.json") for item in summaries],
            "corner_pdfs": [item["paths"]["corner_pdf"] for item in summaries],
        },
        "summaries": summaries,
    }
    manifest_path = OUTPUT_ROOT / f"{RUN_STEM}_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"[write] {manifest_path}", flush=True)
    print(f"[write] {forest_pdf}", flush=True)


if __name__ == "__main__":
    main()
