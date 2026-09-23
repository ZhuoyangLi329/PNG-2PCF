#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""用 Task43 ``smax=550`` 长链替换公开 jaxpower P(k)-2PCF 图中的红色 contour。

代码大纲
========
1. 读取已经通过总审计的 ``smax=550`` 2PCF 长链，以及原 350 图使用的
   同一条 P(k) 长链；两个 NPZ 都只含 burn-in 后的等权样本。
2. 硬检查 2PCF 的 50--550 范围、50 bins、single-realization jaxpower
   covariance、FullDiscrete/full-PNG/shell-average/fixed-sn0=0 和正式 MCMC
   设置，同时检查 P(k) 的 jaxpower covariance、free sn0 与 mother-box
   cutoff 口径没有变化。
3. 新旧 2PCF compiled operator 文件名不同，因此不伪称它们是同一个文件；
   本脚本要求二者解析到同一个 factorized kernel，并逐数组验证 P(k)
   response 等价、旧 30-bin xi basis 与新 operator 前 30 bins 等价。
4. 复用 ``task43_plot_ric_covariance_pair_longchain.py`` 的定稿 PPT 样式，
   冻结原 350 图的 fNL/b1 坐标，只把红色 2PCF posterior、约束文字和
   legend 范围替换为 ``50<s<550``；P(k) contour 和样本完全不变。
5. 原子写一张 PDF，并写独立 JSON audit，记录输入 SHA256、链诊断、
   posterior、operator bridge、冻结坐标与“旧 350 图保留”的 provenance。

本图仍是 ``IC^(rad,rad)`` single-term approximation；2PCF 使用尚未由
realization scatter 校准的固定 Gaussian jaxpower covariance，因此只是
当前近似下的对比图，不应解释为 science-ready 的绝对 PNG 约束。
"""

from __future__ import annotations

import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np


PROJECT_ROOT = Path("/pscratch/sd/l/lzy/PNG-2PCF-model-to-observe")
CODE_DIR = PROJECT_ROOT / "codes/task43"
if str(CODE_DIR) not in sys.path:
    sys.path.insert(0, str(CODE_DIR))

import task43_overlay_pk_window_kmin_contour as contour_base  # noqa: E402
import task43_plot_ric_covariance_pair_longchain as pair_plot  # noqa: E402
import task43_plot_ric_rascalc_pk_vs_2pcf as chain_tools  # noqa: E402


RMAX_ROOT = PROJECT_ROOT / "outputs/task43_outputs/rmax_scan"
XI_DIR = RMAX_ROOT / "fits/smax550"
XI_SUMMARY = XI_DIR / "task43_minimal_closure_mcmc_summary.json"
XI_SAMPLES = XI_DIR / "task43_mcmc_radial_singleterm_samples.npz"
XI_COVARIANCE = RMAX_ROOT / "covariance" / (
    "jaxpower_covariance_lightcone_mmin1p4e13_ph000_fitcov_bessel_interp_"
    "smoothfftlog_rrdeconv_fkpNorm4p8925e10_mesh64_nran100k_ndata50k_"
    "pad400_win3600_ds2_k0001_3000_dk002_p1p0_s50_550_ds10.npz"
)
XI_OPERATOR = RMAX_ROOT / "operators" / (
    "task43_ric_factorized_operator_ph000_dchi2_nsub200000_sobol2p22_"
    "ds2_seed20260712_L2000_s50_550_ds10.npz"
)
PK_OPERATOR = PROJECT_ROOT / "outputs/task43_outputs/ric_singleterm/operators" / (
    "task43_ric_factorized_operator_ph000_dchi2_nsub200000_sobol2p22_"
    "ds2_seed20260712_L2000.npz"
)
OLD_PAIR_AUDIT = PROJECT_ROOT / "outputs/task43_outputs/ric_singleterm/audits" / (
    "task43_ric_pk_vs_2pcf_covariance_pair_longchain.json"
)
OLD_JAXPOWER_PDF = PROJECT_ROOT / "plots/task43" / (
    "task43_pk_vs_2pcf_s50_350_both_jaxpower_covariance.pdf"
)
OUTPUT_PDF = PROJECT_ROOT / "plots/task43" / (
    "task43_pk_vs_2pcf_s50_550_both_jaxpower_covariance.pdf"
)
OUTPUT_AUDIT = PROJECT_ROOT / "outputs/task43_outputs/ric_singleterm/audits" / (
    "task43_ric_pk_vs_2pcf_s50_550_both_jaxpower_longchain.json"
)


def require(condition: bool, message: str) -> None:
    """把 provenance 或数值不一致升级为硬失败，禁止画出误标 contour。"""
    if not bool(condition):
        raise RuntimeError(message)


def read_json(path: Path) -> dict[str, Any]:
    """读取非空 JSON object；路径不存在或结构错误时立即失败。"""
    require(path.is_file() and path.stat().st_size > 0, f"缺少 JSON：{path}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    require(isinstance(payload, dict), f"JSON 顶层不是 object：{path}")
    return payload


def sha256_file(path: Path) -> str:
    """流式计算输入与输出哈希，不把较大的 MCMC 文件再复制一份。"""
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while True:
            block = stream.read(8 * 1024 * 1024)
            if not block:
                break
            digest.update(block)
    return digest.hexdigest()


def canonical_recorded_path(value: str | Path) -> Path:
    """把 summary 中可能为相对路径的 provenance 统一解析到项目根目录。"""
    path = Path(value).expanduser()
    if not path.is_absolute():
        path = PROJECT_ROOT / path
    return path.resolve(strict=False)


def validate_xi_summary(summary: dict[str, Any]) -> dict[str, Any]:
    """检查新红色 contour 确实来自冻结的 50--550 jaxpower likelihood。"""
    require(summary.get("status") == "done", "smax550 2PCF summary 未完成")
    fit = summary.get("fit_range", {})
    require(
        float(fit.get("rmin", np.nan)) == 50.0
        and float(fit.get("rmax", np.nan)) == 550.0
        and int(fit.get("nbins", -1)) == 50
        and int(fit.get("data_vector_size", -1)) == 50,
        f"2PCF fit range/bins 不是 50--550 / 50 bins：{fit}",
    )
    covariance = summary.get("covariance", {})
    require(
        covariance.get("key") == "covariance_single_realization"
        and covariance.get("fit_target") == "mean"
        and canonical_recorded_path(covariance.get("path", "")) == XI_COVARIANCE.resolve(),
        "2PCF 没有读取指定的 50-bin single-realization jaxpower covariance",
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
        "2PCF theory/cosmology/PNG/sn0/shell-average 口径不匹配",
    )
    model = chain_tools.radial_model(summary)
    require(model.get("parameter_names") == ["fnl_loc", "b1"], "2PCF 参数顺序不匹配")
    radial = model.get("radial_singleterm", {})
    require(
        canonical_recorded_path(radial.get("operator", {}).get("path", ""))
        == XI_OPERATOR.resolve(),
        "2PCF 没有使用指定的扩展 50-bin operator",
    )
    require(radial.get("extra_global_sigma_w2") is False, "2PCF 重复减了 global sigmaW2")
    mcmc = model.get("mcmc", {})
    require(
        int(mcmc.get("nwalkers", -1)) == 64
        and int(mcmc.get("nsteps", -1)) == 20000
        and int(mcmc.get("burnin", -1)) == 5000
        and int(mcmc.get("seed", -1)) == 20260720
        and int(mcmc.get("nsamples", -1)) == 960000,
        f"2PCF MCMC contract 不匹配：{mcmc}",
    )
    return model


def validate_pk_summary(summary: dict[str, Any]) -> None:
    """确认黑色 contour 与原 350 PDF 使用的是同一条正式 P(k) 链。"""
    require(summary.get("status") == "done", "P(k) summary 未完成")
    data = summary.get("data", {})
    config = summary.get("config", {})
    require(
        np.isclose(float(data.get("kmax_fit", np.nan)), 0.1)
        and config.get("cosmology") == "abacus_c000"
        and float(config.get("p_fixed", np.nan)) == 1.0
        and config.get("sn0_policy") == "free"
        and config.get("standalone_gic") is False
        and np.isclose(
            float(config.get("window_theory_kmin", np.nan)),
            2.0 * np.pi / 2000.0,
            rtol=0.0,
            atol=1.0e-15,
        ),
        "P(k) kmax/cosmology/free-sn0/mother-box/RIC 口径改变",
    )
    require(
        int(config.get("nwalkers", -1)) == 24
        and int(config.get("nsteps", -1)) == 50000
        and int(config.get("burnin", -1)) == 5000
        and int(config.get("seed", -1)) == 20260713,
        "P(k) 长链设置不再是原图使用的 24x50000 正式链",
    )
    operator = config.get("radial_singleterm_ric", {}).get("path", "")
    require(
        canonical_recorded_path(operator) == PK_OPERATOR.resolve(),
        "P(k) contour 没有沿用原 350 图的 compiled operator",
    )
    require("jaxpower" in str(summary.get("covariance", {}).get("precision", "")).casefold(),
            "P(k) covariance 不再是 jaxpower")


def _relative_l2(left: np.ndarray, right: np.ndarray) -> float:
    """计算稳定的相对 L2 差，零 reference 时退化为绝对 L2。"""
    denominator = float(np.linalg.norm(right))
    numerator = float(np.linalg.norm(left - right))
    return numerator / denominator if denominator else numerator


def audit_operator_compatibility() -> dict[str, Any]:
    """证明扩展 xi operator 与原 P(k) operator 共享 kernel 且 response 等价。"""
    require(XI_OPERATOR.is_file() and PK_OPERATOR.is_file(), "缺少新/旧 radial operator")
    with np.load(XI_OPERATOR, allow_pickle=False) as new, np.load(
        PK_OPERATOR, allow_pickle=False
    ) as old:
        new_meta = json.loads(str(np.asarray(new["meta_json"]).item()))
        old_meta = json.loads(str(np.asarray(old["meta_json"]).item()))
        new_kernel = canonical_recorded_path(str(np.asarray(new["kernel_path"]).item()))
        old_kernel = canonical_recorded_path(str(np.asarray(old["kernel_path"]).item()))
        require(new_kernel == old_kernel, "新旧 operator 没有解析到同一个 factorized kernel")
        new_pk_payload = canonical_recorded_path(new_meta["inputs"]["pk_payload"])
        old_pk_payload = canonical_recorded_path(old_meta["inputs"]["pk_payload"])
        require(new_pk_payload == old_pk_payload, "新旧 operator 的 P(k) payload 不同")

        old_edges = np.asarray(old["target_s_edges"], dtype="f8")
        new_edges = np.asarray(new["target_s_edges"], dtype="f8")
        require(np.array_equal(new_edges[: old_edges.size], old_edges), "新 operator 前 30 bins 坐标未 bridge")

        xi_keys = (
            "xi_basis_pk_dd",
            "xi_basis_alpha_pk_dd",
            "xi_basis_alpha2_pk_dd",
            "xi_global_basis_pk_dd",
            "xi_global_basis_alpha_pk_dd",
            "xi_global_basis_alpha2_pk_dd",
        )
        xi_bridge: dict[str, Any] = {}
        for key in xi_keys:
            old_value = np.asarray(old[key], dtype="f8")
            new_value = np.asarray(new[key], dtype="f8")[: old_value.size]
            passed = bool(np.allclose(new_value, old_value, rtol=1.0e-10, atol=1.0e-14))
            require(passed, f"operator xi bridge 失败：{key}")
            xi_bridge[key] = {
                "max_abs_delta": float(np.max(np.abs(new_value - old_value))),
                "relative_l2_delta": _relative_l2(new_value, old_value),
                "pass": passed,
            }

        pk_keys = (
            "k_eff",
            "g_nz",
            "volume",
            "pk_theory_k",
            "pk_theory_ell",
            "pk_ric_matrix",
            "pk_ric_matrix_hankel",
            "pk_ric_matrix_continuum_raw",
            "pk_ric_matrix_hankel_continuum_raw",
            "pk_geometry_factorized_matrix",
            "pk_estimator_transfer",
            "outer_exact_probability",
            "outer_model_probability",
            "outer_row_ratio",
        )
        pk_bridge: dict[str, Any] = {}
        for key in pk_keys:
            new_value = np.asarray(new[key])
            old_value = np.asarray(old[key])
            require(new_value.shape == old_value.shape, f"operator P(k) shape 不同：{key}")
            if np.issubdtype(new_value.dtype, np.integer):
                passed = bool(np.array_equal(new_value, old_value))
            else:
                passed = bool(np.allclose(new_value, old_value, rtol=1.0e-12, atol=1.0e-18))
            require(passed, f"operator P(k) response bridge 失败：{key}")
            delta = np.asarray(new_value, dtype="f8") - np.asarray(old_value, dtype="f8")
            pk_bridge[key] = {
                "max_abs_delta": float(np.max(np.abs(delta))) if delta.size else 0.0,
                "relative_l2_delta": _relative_l2(
                    np.asarray(new_value, dtype="f8"), np.asarray(old_value, dtype="f8")
                ),
                "bitwise_exact": bool(np.array_equal(new_value, old_value)),
                "pass": passed,
            }

    return {
        "pass": True,
        "relation": (
            "distinct compiled xi-support files; identical factorized kernel and "
            "numerically equivalent P(k) response"
        ),
        "new_operator": str(XI_OPERATOR),
        "pk_chain_operator": str(PK_OPERATOR),
        "shared_kernel": str(new_kernel),
        "shared_pk_payload": str(new_pk_payload),
        "xi_first30": xi_bridge,
        "pk_response": pk_bridge,
    }


def main() -> None:
    """完成全部 gate 后生成新的 550 PDF 和独立 provenance audit。"""
    for path in (
        XI_SUMMARY,
        XI_SAMPLES,
        XI_COVARIANCE,
        XI_OPERATOR,
        PK_OPERATOR,
        chain_tools.PK_SUMMARY,
        chain_tools.PK_SAMPLES,
        OLD_PAIR_AUDIT,
        OLD_JAXPOWER_PDF,
    ):
        require(path.is_file() and path.stat().st_size > 0, f"缺少输入：{path}")

    xi_summary = read_json(XI_SUMMARY)
    pk_summary = read_json(chain_tools.PK_SUMMARY)
    old_audit = read_json(OLD_PAIR_AUDIT)
    xi_model = validate_xi_summary(xi_summary)
    validate_pk_summary(pk_summary)
    operator_bridge = audit_operator_compatibility()

    xi_samples = chain_tools.load_xi_samples(xi_summary, XI_SAMPLES)
    pk_samples_full = contour_base.load_named_samples(
        chain_tools.PK_SAMPLES,
        ("fnl_loc", "b1", "sn0"),
    )
    pk_samples = pk_samples_full[:, :2]
    require(xi_samples.shape == (960000, 2), f"2PCF samples shape 错误：{xi_samples.shape}")
    require(pk_samples_full.shape[1] == 3, f"P(k) samples 参数维数错误：{pk_samples_full.shape}")
    require(np.all(np.isfinite(xi_samples)), "2PCF samples 含非有限值")
    require(np.all(np.isfinite(pk_samples_full)), "P(k) samples 含非有限值")

    diagnostics = {
        "2pcf_jaxpower_s50_550": chain_tools.chain_diagnostics(
            xi_samples,
            nwalkers=int(xi_model["mcmc"]["nwalkers"]),
        ),
        "pk_jaxpower_unchanged": chain_tools.chain_diagnostics(
            pk_samples_full,
            nwalkers=int(pk_summary["config"]["nwalkers"]),
        ),
    }
    require(all(row["pass"] for row in diagnostics.values()), f"长链 gate 失败：{diagnostics}")

    old_ranges = old_audit.get("shared_plot_ranges", {})
    xlim = tuple(float(value) for value in old_ranges.get("fnl_loc", ()))
    ylim = tuple(float(value) for value in old_ranges.get("b1", ()))
    require(len(xlim) == 2 and xlim[0] < xlim[1], "旧图 fNL 坐标范围无效")
    require(len(ylim) == 2 and ylim[0] < ylim[1], "旧图 b1 坐标范围无效")
    require(old_audit.get("inputs", {}).get("pk_samples") == str(chain_tools.PK_SAMPLES),
            "冻结坐标 audit 对应的 P(k) 链不是当前原图链")

    contour_base.apply_publication_style()
    xi_posterior = pair_plot.plot_variant(
        xi_summary=xi_summary,
        xi_samples=xi_samples,
        pk_summary=pk_summary,
        pk_samples=pk_samples,
        covariance_short="jaxpower",
        output=OUTPUT_PDF,
        xlim=xlim,
        ylim=ylim,
    )
    pk_posterior = {
        "fnl_loc": chain_tools.summarize(pk_samples[:, 0]),
        "b1": chain_tools.summarize(pk_samples[:, 1]),
        "sn0": chain_tools.summarize(pk_samples_full[:, 2]),
    }
    require(OUTPUT_PDF.is_file() and OUTPUT_PDF.stat().st_size > 0, "PDF 未生成")
    with OUTPUT_PDF.open("rb") as stream:
        require(stream.read(5) == b"%PDF-", "输出没有 PDF magic header")

    hashed_inputs = (
        XI_SUMMARY,
        XI_SAMPLES,
        XI_COVARIANCE,
        XI_OPERATOR,
        chain_tools.PK_SUMMARY,
        chain_tools.PK_SAMPLES,
        PK_OPERATOR,
        OLD_PAIR_AUDIT,
        OLD_JAXPOWER_PDF,
    )
    audit = {
        "task": "task43_plot_ric_jaxpower_pk_vs_2pcf_smax550",
        "status": "done",
        "created_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "figure_contract": {
            "replacement": "red 2PCF contour only: s50_350 -> s50_550",
            "pk_chain_unchanged_from_old_jaxpower_pdf": True,
            "axis_ranges_unchanged_from_old_jaxpower_pdf": True,
            "style_unchanged_from_old_jaxpower_pdf": True,
            "old_s50_350_pdf_retained_for_provenance": True,
            "model": "radial single-term RIC on both probes",
            "density_ric_cross_terms": False,
            "extra_global_sigma_w2": False,
            "covariances_changed_by_ric": False,
            "science_ready": False,
        },
        "inputs": {
            "2pcf_summary": str(XI_SUMMARY),
            "2pcf_samples": str(XI_SAMPLES),
            "2pcf_covariance": str(XI_COVARIANCE),
            "2pcf_operator": str(XI_OPERATOR),
            "pk_summary": str(chain_tools.PK_SUMMARY),
            "pk_samples": str(chain_tools.PK_SAMPLES),
            "pk_operator": str(PK_OPERATOR),
            "old_plot_audit_for_frozen_axes": str(OLD_PAIR_AUDIT),
            "old_s50_350_pdf": str(OLD_JAXPOWER_PDF),
            "sha256": {str(path): sha256_file(path) for path in hashed_inputs},
        },
        "fit_ranges": {
            "2pcf_s_edges_mpc_h": [50.0, 550.0],
            "2pcf_nbins": 50,
            "pk_k_centers_h_mpc": [
                float(pk_summary["data"]["k_min"]),
                float(pk_summary["data"]["k_max"]),
            ],
        },
        "operator_compatibility": operator_bridge,
        "chain_diagnostics": diagnostics,
        "posterior": {
            "2pcf_jaxpower_s50_550": xi_posterior,
            "pk_jaxpower_unchanged": pk_posterior,
        },
        "plot_ranges": {"fnl_loc": list(xlim), "b1": list(ylim)},
        "plot_settings": {
            "backend": "matplotlib + getdist",
            "smooth_scale_1d": contour_base.GETDIST_SMOOTH_1D,
            "smooth_scale_2d": contour_base.GETDIST_SMOOTH_2D,
            "format": "pdf only",
            "style_reference": "existing Task43/Task44 PPT contour style",
            "title_shown": False,
            "truth_marker_shown": False,
            "map_markers_shown": False,
            "contour_and_legend_order": ["pk", "xi0_2pcf"],
            "pk_color": pair_plot.PPT_COLORS["pk"],
            "twopcf_color": pair_plot.PPT_COLORS["twopcf"],
            "legend_fontsize": pair_plot.PPT_LEGEND_FONTSIZE,
            "fnl_annotation_fontsize": pair_plot.PPT_FNL_FONTSIZE,
            "2pcf_legend_range": "50<s<550 h^-1 Mpc",
        },
        "scope_warning": (
            "Fixed diagnostic jaxpower covariance and single IC^(rad,rad) term only; "
            "not a science-ready absolute fNL constraint."
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
    print(f"[write] {OUTPUT_AUDIT}")


if __name__ == "__main__":
    main()
