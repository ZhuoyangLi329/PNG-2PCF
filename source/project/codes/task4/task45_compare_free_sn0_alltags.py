#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
代码大纲：

1. 读取 Task45 fixed-sn0 主结果 manifest 与 free-sn0 对照 manifest。
2. 对每个 tag/case 提取 `b1`、`fnl_loc`、`sigmas`、free-sn0 的 `sn0`。
3. 计算 free-sn0 相对 fixed-sn0 的参数偏移，尤其是 `b1` 偏移。
4. 分别汇总 P(k) 与 2PCF 的平均 `b1` 偏移，判断 shot-noise 常数项主要影响哪一类统计量。
5. 写出 CSV、Markdown 与 JSON summary，作为这个 free-sn0 对照 run 的轻量审计文件。
"""

from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np


PROJECT_ROOT = Path("/pscratch/sd/l/lzy/PNG-2PCF-model-to-observe")
FIXED_MANIFEST = PROJECT_ROOT / "outputs" / "task4_outputs" / "quijote_ultranest" / "task45_quijote_ultranest_manifest.json"
FREE_MANIFEST = PROJECT_ROOT / "outputs" / "task4_outputs" / "quijote_ultranest_free_sn0" / "task45_quijote_ultranest_free_sn0_manifest.json"
OUT_ROOT = PROJECT_ROOT / "outputs" / "task4_outputs" / "quijote_ultranest_free_sn0"

OUT_CSV = OUT_ROOT / "task45_quijote_ultranest_free_sn0_vs_fixed_b1_summary.csv"
OUT_MD = OUT_ROOT / "task45_quijote_ultranest_free_sn0_vs_fixed_b1_summary.md"
OUT_JSON = OUT_ROOT / "task45_quijote_ultranest_free_sn0_vs_fixed_b1_summary.json"

TAG_ORDER = ["fid", "LCp50", "LCp100"]
CASE_ORDER = ["pk_binavg", "xi_r50_350", "xi_r60_350", "xi_r80_350", "xi_r100_350"]


def load_json(path: Path) -> dict:
    """读取 JSON 文件。"""
    if not path.exists():
        raise FileNotFoundError(path)
    return json.loads(path.read_text(encoding="utf-8"))


def index_manifest(manifest: dict) -> dict[tuple[str, str], dict]:
    """把 manifest 的 summaries 转成 `(tag, case) -> summary` 字典。"""
    return {(item["tag"], item["case"]): item for item in manifest["summaries"]}


def median(summary: dict, name: str) -> float:
    """读取某个参数的 posterior median。"""
    return float(summary["parameters"][name]["median"])


def err_text(summary: dict, name: str) -> str:
    """生成 `median -err +err` 的简短字符串，误差使用 Percival 修正。"""
    p = summary["parameters"][name]
    med = float(p["median"])
    elo = float(p["err_low_percival"])
    ehi = float(p["err_high_percival"])
    return f"{med:.6g} -{elo:.3g} +{ehi:.3g}"


def build_rows(fixed_by_key: dict[tuple[str, str], dict], free_by_key: dict[tuple[str, str], dict]) -> list[dict[str, object]]:
    """构造逐 tag/case 的对比行。"""
    rows: list[dict[str, object]] = []
    for tag in TAG_ORDER:
        for case in CASE_ORDER:
            key = (tag, case)
            fixed = fixed_by_key[key]
            free = free_by_key[key]
            row = {
                "tag": tag,
                "case": case,
                "statistic": "pk" if case == "pk_binavg" else "xi",
                "fixed_fnl_median": median(fixed, "fnl_loc"),
                "free_fnl_median": median(free, "fnl_loc"),
                "delta_fnl_free_minus_fixed": median(free, "fnl_loc") - median(fixed, "fnl_loc"),
                "fixed_b1_median": median(fixed, "b1"),
                "free_b1_median": median(free, "b1"),
                "delta_b1_free_minus_fixed": median(free, "b1") - median(fixed, "b1"),
                "fixed_sigmas_median": median(fixed, "sigmas"),
                "free_sigmas_median": median(free, "sigmas"),
                "delta_sigmas_free_minus_fixed": median(free, "sigmas") - median(fixed, "sigmas"),
                "free_sn0_median": median(free, "sn0"),
                "free_sn0_err_text": err_text(free, "sn0"),
                "free_ncall": int(free["ultranest"]["ncall"]),
                "free_logzerr": float(free["ultranest"]["logzerr"]),
            }
            rows.append(row)
    return rows


def aggregate(rows: list[dict[str, object]]) -> dict[str, object]:
    """汇总 P(k) 与 2PCF 的 `b1` 偏移和 free-sn0 大小。"""
    out: dict[str, object] = {}
    for statistic in ["pk", "xi"]:
        subset = [row for row in rows if row["statistic"] == statistic]
        b1_delta = np.array([float(row["delta_b1_free_minus_fixed"]) for row in subset], dtype="f8")
        sn0 = np.array([float(row["free_sn0_median"]) for row in subset], dtype="f8")
        out[statistic] = {
            "n_cases": len(subset),
            "delta_b1_mean": float(np.mean(b1_delta)),
            "delta_b1_min": float(np.min(b1_delta)),
            "delta_b1_max": float(np.max(b1_delta)),
            "abs_sn0_median_mean": float(np.mean(np.abs(sn0))),
            "sn0_median_min": float(np.min(sn0)),
            "sn0_median_max": float(np.max(sn0)),
        }
    out["interpretation"] = (
        "free sn0 substantially lowers P(k) b1, while 2PCF b1 and sn0 are nearly unchanged; "
        "the constant shot-noise term absorbs part of the original P(k)-2PCF b1 offset mainly on the P(k) side."
    )
    return out


def write_csv(rows: list[dict[str, object]]) -> None:
    """写出逐 case CSV。"""
    with OUT_CSV.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def write_markdown(rows: list[dict[str, object]], summary: dict[str, object]) -> None:
    """写出便于阅读的 Markdown 汇总。"""
    with OUT_MD.open("w", encoding="utf-8") as file:
        file.write("# Task45 free-sn0 vs fixed-sn0 b1 summary\n\n")
        file.write("## Aggregate\n\n")
        file.write("| statistic | n_cases | mean delta b1 | delta b1 min | delta b1 max | mean abs(sn0) |\n")
        file.write("|---|---:|---:|---:|---:|---:|\n")
        for statistic in ["pk", "xi"]:
            item = summary[statistic]
            file.write(
                f"| {statistic} | {item['n_cases']} | {item['delta_b1_mean']:.6g} | "
                f"{item['delta_b1_min']:.6g} | {item['delta_b1_max']:.6g} | "
                f"{item['abs_sn0_median_mean']:.6g} |\n"
            )
        file.write("\n## Per Case\n\n")
        file.write("| tag | case | fixed b1 | free b1 | delta b1 | free sn0 median |\n")
        file.write("|---|---|---:|---:|---:|---:|\n")
        for row in rows:
            file.write(
                f"| {row['tag']} | {row['case']} | {row['fixed_b1_median']:.6g} | "
                f"{row['free_b1_median']:.6g} | {row['delta_b1_free_minus_fixed']:.6g} | "
                f"{row['free_sn0_median']:.6g} |\n"
            )
        file.write("\n## Interpretation\n\n")
        file.write(str(summary["interpretation"]) + "\n")


def main() -> None:
    """主入口。"""
    fixed = load_json(FIXED_MANIFEST)
    free = load_json(FREE_MANIFEST)
    fixed_by_key = index_manifest(fixed)
    free_by_key = index_manifest(free)
    missing = [(tag, case) for tag in TAG_ORDER for case in CASE_ORDER if (tag, case) not in free_by_key]
    if missing:
        raise RuntimeError(f"missing free-sn0 summaries: {missing}")
    rows = build_rows(fixed_by_key, free_by_key)
    summary = aggregate(rows)
    payload = {
        "task": "task45_quijote_ultranest_free_sn0_vs_fixed_b1_summary",
        "status": "done",
        "inputs": {
            "fixed_manifest": str(FIXED_MANIFEST),
            "free_manifest": str(FREE_MANIFEST),
        },
        "aggregate": summary,
        "rows": rows,
        "outputs": {
            "csv": str(OUT_CSV),
            "markdown": str(OUT_MD),
            "json": str(OUT_JSON),
        },
    }
    write_csv(rows)
    write_markdown(rows, summary)
    OUT_JSON.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"[write] {OUT_CSV}")
    print(f"[write] {OUT_MD}")
    print(f"[write] {OUT_JSON}")


if __name__ == "__main__":
    main()
