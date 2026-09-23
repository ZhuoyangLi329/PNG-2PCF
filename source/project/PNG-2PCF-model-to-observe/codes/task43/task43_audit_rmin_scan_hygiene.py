#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""审计 Task43 2PCF rmin 扫描的最终文件、状态与 provenance。

执行逻辑大纲：
1. 只读取 ``outputs/task43_outputs/rmin_scan`` 与对应 PDF 目录，不扫描或
   改写已归档的 Task43 大型历史树。
2. 要求 25-phase 测量、32-bin mean/covariance/operator、两份 Fisher、
   三条长链和所有科学 audit 均存在且状态通过。
3. 检查最终绘图树只含 PDF、没有 PNG；正式目录必须恰含汇总图与 contour，
   允许独立子目录保留已审计的 preliminary PDF。
4. 对核心文件计算 SHA256，原子写出一个可独立复核的 hygiene JSON；本脚本
   只做只读验收，不重新运行任何科学计算。
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import tempfile
from pathlib import Path
from typing import Any


ROOT = Path("/pscratch/sd/l/lzy/PNG-2PCF-model-to-observe")
SCAN = ROOT / "outputs/task43_outputs/rmin_scan"
PLOT = ROOT / "plots/task43/rmin_scan"
OUTPUT = SCAN / "audits/task43_rmin_scan_hygiene.json"


def atomic_json(path: Path, payload: dict[str, Any]) -> None:
    """把小型 hygiene JSON 原子写入目标目录。"""

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


def sha256(path: Path) -> str:
    """流式计算文件 SHA256，避免把长链一次性读入内存。"""

    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while True:
            block = stream.read(8 * 1024 * 1024)
            if not block:
                break
            digest.update(block)
    return digest.hexdigest()


def load_json(path: Path) -> dict[str, Any]:
    """读取 JSON 并要求顶层为字典。"""

    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise RuntimeError(f"JSON 顶层不是字典：{path}")
    return value


def require_file(path: Path, *, root: Path | None = None) -> None:
    """要求普通非空文件，并可附加 lexical containment 检查。"""

    if path.is_symlink() or not path.is_file() or path.stat().st_size <= 0:
        raise RuntimeError(f"核心文件缺失、为空或为 symlink：{path}")
    if root is not None:
        try:
            path.relative_to(root)
        except ValueError as error:
            raise RuntimeError(f"核心文件越出任务根目录：{path} not under {root}") from error


def main() -> None:
    """执行最终只读 hygiene gate 并写 SHA256 manifest。"""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scan-root", type=Path, default=SCAN)
    parser.add_argument("--plot-root", type=Path, default=PLOT)
    parser.add_argument("--output", type=Path, default=OUTPUT)
    args = parser.parse_args()

    scan = args.scan_root.absolute()
    plot = args.plot_root.absolute()
    output = args.output.absolute()
    if scan.is_symlink() or plot.is_symlink() or output.is_symlink():
        raise RuntimeError("scan/plot/output 根路径不得是 symlink")
    output.relative_to(scan)

    manifest_stem = scan / "manifests/task43_rmin_scan_mmin1p4e13_x25"
    mean_stem = scan / "summary/task43_mean_xi_mmin1p4e13_x25_s30_350_ds10_fkpP010000"
    rr_stem = scan / "covariance/task43_rr_smu_ph000_nran100k_seed20260702_s30_350_ds10_nmu20"
    raw3_stem = scan / "covariance/jaxpower_covariance_lightcone_mmin1p4e13_ph000_fitcov_bessel_interp_smoothfftlog_mesh64_nran100k_ndata50k_pad400_win3600_ds2_k0001_3000_dk002_p1p0_s30_350_ds10"
    final3_stem = scan / "covariance/jaxpower_covariance_lightcone_mmin1p4e13_ph000_fitcov_bessel_interp_smoothfftlog_rrdeconv_fkpNorm4p8925e10_mesh64_nran100k_ndata50k_pad400_win3600_ds2_k0001_3000_dk002_p1p0_s30_350_ds10"
    raw4_stem = scan / "covariance/jaxpower_covariance_lightcone_mmin1p4e13_ph000_fitcov_bessel_interp_smoothfftlog_mesh64_nran100k_ndata50k_pad400_win3600_ds2_k0001_4000_dk002_p1p0_s30_350_ds10"
    final4_stem = scan / "covariance/jaxpower_covariance_lightcone_mmin1p4e13_ph000_fitcov_bessel_interp_smoothfftlog_rrdeconv_fkpNorm4p8925e10_mesh64_nran100k_ndata50k_pad400_win3600_ds2_k0001_4000_dk002_p1p0_s30_350_ds10"
    operator_stem = scan / "operators/task43_ric_factorized_operator_ph000_dchi2_nsub200000_sobol2p22_ds2_seed20260712_L2000_s30_350_ds10"
    fisher_stem = scan / "fisher/task43_rmin_scan_fisher_information"
    fisher4_stem = scan / "fisher/task43_rmin_scan_fisher_information_kmax4"
    final_summary_stem = scan / "audits/task43_jaxpower_2pcf_rmin_scan_longchain"
    contour_audit = scan / "audits/task43_pk_vs_2pcf_rmin_scan_contours.json"

    core: list[Path] = [
        manifest_stem.with_suffix(".jsonl"),
        manifest_stem.with_suffix(".json"),
        mean_stem.with_suffix(".npz"),
        mean_stem.with_suffix(".json"),
        rr_stem.with_suffix(".npz"),
        rr_stem.with_suffix(".json"),
        raw3_stem.with_suffix(".npz"),
        raw3_stem.with_suffix(".json"),
        final3_stem.with_suffix(".npz"),
        final3_stem.with_suffix(".json"),
        raw4_stem.with_suffix(".npz"),
        raw4_stem.with_suffix(".json"),
        final4_stem.with_suffix(".npz"),
        final4_stem.with_suffix(".json"),
        operator_stem.with_suffix(".npz"),
        operator_stem.with_name(f"{operator_stem.name}_audit.json"),
        fisher_stem.with_suffix(".json"),
        fisher_stem.with_suffix(".csv"),
        fisher4_stem.with_suffix(".json"),
        fisher4_stem.with_suffix(".csv"),
        scan / "audits/task43_rmin_scan_measurement_audit.json",
        scan / "audits/task43_rmin_scan_covariance_audit.json",
        scan / "audits/task43_rmin_scan_covariance_kmax3_vs4.json",
        final_summary_stem.with_suffix(".json"),
        final_summary_stem.with_suffix(".csv"),
        contour_audit,
    ]
    for rmin in (30, 40, 50):
        fit = scan / f"fits/rmin{rmin}"
        core.extend(
            [
                fit / "task43_minimal_closure_mcmc_summary.json",
                fit / "task43_mcmc_radial_singleterm_samples.npz",
            ]
        )

    # measurement 文件名由新任务目录和 phase tag 唯一确定；精确 25 份要求
    # 可防止残留测试文件或漏 phase 被最终汇总静默吸收。
    measurements = sorted((scan / "xi_cucount").glob("*.npz"))
    if len(measurements) != 25:
        raise RuntimeError(f"2PCF measurement 数量不是 25：{len(measurements)}")
    expected_phase_tags = {f"ph{index:03d}" for index in range(25)}
    observed_phase_tags = {
        next((part for part in path.stem.split("_") if part.startswith("ph") and len(part) == 5), "")
        for path in measurements
    }
    if observed_phase_tags != expected_phase_tags:
        raise RuntimeError(f"phase 文件集合错误：{sorted(observed_phase_tags)}")
    core.extend(measurements)

    summary_pdf = plot / "task43_jaxpower_2pcf_rmin_scan_b1_fnl.pdf"
    contour_pdf = plot / "task43_pk_vs_2pcf_rmin_scan_contours.pdf"
    expected_canonical_pdfs = [summary_pdf, contour_pdf]
    for pdf in expected_canonical_pdfs:
        require_file(pdf, root=plot)
    plot_files = sorted(path for path in plot.rglob("*") if path.is_file())
    canonical_plot_files = sorted(path for path in plot.glob("*") if path.is_file())
    png_files = [path for path in plot_files if path.suffix.lower() == ".png"]
    non_pdf_files = [path for path in plot_files if path.suffix.lower() != ".pdf"]
    if png_files or non_pdf_files or canonical_plot_files != sorted(expected_canonical_pdfs):
        raise RuntimeError(
            "PDF-only plot gate 失败："
            f"canonical={canonical_plot_files}, all={plot_files}, png={png_files}, non_pdf={non_pdf_files}"
        )
    core.extend(expected_canonical_pdfs)

    for path in core:
        require_file(path, root=plot if path in expected_canonical_pdfs else scan)

    status_contract = {
        scan / "audits/task43_rmin_scan_measurement_audit.json": "pass",
        scan / "audits/task43_rmin_scan_covariance_audit.json": "pass",
        scan / "audits/task43_rmin_scan_covariance_kmax3_vs4.json": "pass",
        operator_stem.with_name(f"{operator_stem.name}_audit.json"): "done",
        fisher_stem.with_suffix(".json"): "pass",
        fisher4_stem.with_suffix(".json"): "pass",
        final_summary_stem.with_suffix(".json"): "pass",
        contour_audit: "pass",
        mean_stem.with_suffix(".json"): "done",
    }
    observed_status: dict[str, str] = {}
    for path, expected in status_contract.items():
        status = str(load_json(path).get("status"))
        observed_status[str(path)] = status
        if status != expected:
            raise RuntimeError(f"status gate 失败：{path}, expected={expected}, observed={status}")

    measurement_audit = load_json(scan / "audits/task43_rmin_scan_measurement_audit.json")
    if measurement_audit.get("nrows_audited") != 25 or measurement_audit.get("missing_indices") != []:
        raise RuntimeError("25-phase measurement audit 未覆盖完整集合")
    final_summary = load_json(final_summary_stem.with_suffix(".json"))
    if len(final_summary.get("cases", [])) != 3:
        raise RuntimeError("最终 long-chain summary 不含三档 rmin case")

    hashes = {
        str(path.relative_to(ROOT)): {"bytes": path.stat().st_size, "sha256": sha256(path)}
        for path in sorted(set(core))
    }
    payload = {
        "status": "pass",
        "task": "task43_audit_rmin_scan_hygiene",
        "scope": "Task43 jaxpower-only 2PCF rmin=30/40/50 final artifact/provenance hygiene",
        "core_file_count": len(hashes),
        "measurement_count": len(measurements),
        "plot_policy": {
            "pdf_only": True,
            "canonical_files": [str(path.relative_to(ROOT)) for path in expected_canonical_pdfs],
            "all_pdf_count_including_preliminary": len(plot_files),
        },
        "legacy_policy": "plots/outputs historical products are read-only inputs and are not hygiene write targets",
        "status_contract": observed_status,
        "files": hashes,
    }
    atomic_json(output, payload)
    print(json.dumps({key: payload[key] for key in ("status", "core_file_count", "measurement_count", "plot_policy")}, indent=2))


if __name__ == "__main__":
    main()
