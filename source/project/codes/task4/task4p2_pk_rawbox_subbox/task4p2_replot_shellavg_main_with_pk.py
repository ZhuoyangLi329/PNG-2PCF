#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
重画 Task4.2 全 shell-average 2PCF + 正式 P(k) forest 图。

执行大纲：
1. 读取新的 z=1、fixed-sn0、volume-shell-averaged 2PCF 五条收敛链。
2. 检查统一 parent-cluster covariance audit 为 ``pass_with_caveats``，并检查
   每条 2PCF summary 的 shell kernel、fixed sn0 与 convergence gate。
3. 复用现有主图入口读取 z=1/free-sn0 P(k) 三条正式长链及其 audit。
4. 复用 Task4.2 forest 样式生成一个新 PDF；旧 center-kernel PDF 保留不覆盖。
5. 写显式 manifest，列出每个点、输入路径和 covariance caveat。
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

import task4p2_replot_main_with_pk as base


PROJECT_ROOT = Path("/pscratch/sd/l/lzy/PNG-2PCF-model-to-observe")
TWOPCF_SUMMARY = (
    PROJECT_ROOT
    / "outputs/task4_outputs/task4p2_2pcf_shellavg_fixed_sn0_z1"
    / "task4p2_2pcf_shellavg_fixed_sn0_z1_summary.json"
)
PARENT_AUDIT = (
    PROJECT_ROOT
    / "outputs/task4_outputs/task4p2_pk_rawbox_subbox/audits"
    / "task4p2_subbox_parent_dependence_covariance_audit.json"
)
OUT_PDF = (
    PROJECT_ROOT
    / "plots/task4/important_4p1_4p2"
    / "4p2_main_r50_shellavg_fixed_sn0_formalgic_with_pk.pdf"
)
OUT_MANIFEST = (
    PROJECT_ROOT
    / "outputs/task4_outputs/task4p2_2pcf_shellavg_fixed_sn0_z1"
    / "task4p2_shellavg_fixed_sn0_with_pk_manifest.json"
)


def read_json(path: Path) -> dict[str, Any]:
    """读取 UTF-8 JSON。"""
    return json.loads(path.read_text(encoding="utf-8"))


def validate_inputs() -> tuple[dict[str, Any], dict[str, Any]]:
    """验证新 2PCF summary 与 parent-cluster covariance audit。"""
    summary = read_json(TWOPCF_SUMMARY)
    if summary.get("status") != "pass":
        raise ValueError(f"2PCF shell-average combined summary 未通过: {TWOPCF_SUMMARY}")
    scope = summary.get("scientific_scope", {})
    if scope.get("xi_kernel") != "volume-shell-averaged-j0":
        raise ValueError("2PCF combined summary 不是 volume shell-average kernel")
    fixed = scope.get("fixed_parameters", {})
    if not np.isclose(float(fixed.get("sn0", np.nan)), 0.0):
        raise ValueError("2PCF shell-average 正式口径要求 fixed sn0=0")

    expected = {"rawbox", "L1500_no_gic", "L1500_formal_gic", "L1000_no_gic", "L1000_formal_gic"}
    actual = set(summary.get("summaries", {}))
    if actual != expected:
        raise ValueError(f"2PCF shell-average cases 不完整: actual={sorted(actual)}")
    for label, item in summary["summaries"].items():
        config = item.get("config", {})
        if item.get("status") != "pass" or config.get("xi_kernel") != "volume-shell-averaged-j0":
            raise ValueError(f"2PCF case 未通过或 kernel 不正确: {label}")
        if not np.isclose(float(config.get("fixed_parameters", {}).get("sn0", np.nan)), 0.0):
            raise ValueError(f"2PCF case 不是 fixed sn0=0: {label}")

    parent_audit = read_json(PARENT_AUDIT)
    if parent_audit.get("status") != "pass_with_caveats":
        raise ValueError(f"parent-cluster covariance audit 未通过: {PARENT_AUDIT}")
    return summary, parent_audit


def constraint(item: dict[str, Any]) -> dict[str, float]:
    """抽取一个参数的 Percival-scaled 68% constraint。"""
    return {
        "median": float(item["median"]),
        "err_low": float(item["err_low_percival"]),
        "err_high": float(item["err_high_percival"]),
        "err_low_raw": float(item["err_low"]),
        "err_high_raw": float(item["err_high"]),
    }


def load_twopcf(summary: dict[str, Any]) -> dict[str, dict[str, float]]:
    """把新的五条 2PCF summaries 映射成现有 forest 绘图键。"""
    cases = summary["summaries"]
    return {
        "rawbox reference": constraint(cases["rawbox"]["parameters"]["fnl_loc"]),
        "L1500 no-GIC": constraint(cases["L1500_no_gic"]["parameters"]["fnl_loc"]),
        "L1500 formal-GIC": constraint(cases["L1500_formal_gic"]["parameters"]["fnl_loc"]),
        "L1000 no-GIC": constraint(cases["L1000_no_gic"]["parameters"]["fnl_loc"]),
        "L1000 formal-GIC": constraint(cases["L1000_formal_gic"]["parameters"]["fnl_loc"]),
    }


def main() -> None:
    """验证、绘图并写 manifest。"""
    summary, parent_audit = validate_inputs()
    pk_chain_audit = base.validate_chain_audit()
    twopcf = load_twopcf(summary)
    pk = base.load_pk()

    # ``base.make_plot`` 只通过该全局路径决定输出；这里显式切换到新文件名，
    # 保证旧 center-kernel comparison PDF 完整保留。
    base.OUT_PDF = OUT_PDF
    paths = base.make_plot(
        twopcf,
        pk,
        twopcf_descriptor=r"2PCF shell-avg: $r=50$-$350\,h^{-1}{\rm Mpc}$",
    )
    rows = []
    for sample in base.ROW_ORDER:
        rows.append(
            {
                "sample": sample,
                "twopcf_shellavg_rawbox_or_formal_gic": twopcf[
                    "rawbox reference" if sample == "rawbox reference" else f"{sample} formal-GIC"
                ],
                "twopcf_shellavg_no_gic": None if sample == "rawbox reference" else twopcf[f"{sample} no-GIC"],
                "pk_own_kmin_z1_free_sn0": pk[sample],
            }
        )
    manifest = {
        "task": "task4p2_replot_shellavg_main_with_pk",
        "status": "done",
        "twopcf_policy": {
            "summary": str(TWOPCF_SUMMARY),
            "template_z": 1.0,
            "xi_kernel": "volume-shell-averaged-j0",
            "sn0": "fixed_zero_contact_term",
            "sigmas": "fixed_zero_no_rsd",
            "p_fixed": 1.2,
            "display_edges_mpc_h": [50.0, 350.0],
        },
        "parent_cluster_covariance_audit": {
            "path": str(PARENT_AUDIT),
            "status": str(parent_audit["status"]),
            "caveat": parent_audit["global_conclusion"]["required_practice"],
        },
        "pk_policy": {
            "summaries": {name: str(path) for name, path in base.PK_SUMMARIES.items()},
            "chain_audit": str(base.PK_CHAIN_AUDIT),
            "chain_audit_status": str(pk_chain_audit["status"]),
            "template_z": 1.0,
            "sn0": "free",
        },
        "rows": rows,
        "paths": paths,
    }
    OUT_MANIFEST.parent.mkdir(parents=True, exist_ok=True)
    temporary = OUT_MANIFEST.with_suffix(OUT_MANIFEST.suffix + ".tmp")
    temporary.write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    temporary.replace(OUT_MANIFEST)
    print(f"[write] {OUT_PDF}")
    print(f"[write] {OUT_MANIFEST}")


if __name__ == "__main__":
    main()
