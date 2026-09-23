#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Task 4.3.2 baseline contract builder.

代码执行大纲
------------
1. 固定 Task 4.3.2 的当前 BAO-mask、尺度、covariance 和参数合同。
2. 记录 rawbox、box-safe lightcone 的主要 audit/summary 文件。
3. 对当前生产代码和关键结果计算 SHA256，防止后续诊断混用旧文件。
4. 将机器可读合同写入独立的 task432_model_repair/audits 子目录。

这个脚本只读取已有文件并写入新的 Task 4.3.2 audit，不覆盖任何 4.3
生产结果、链文件、协方差或会议图。
"""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path("/pscratch/sd/l/lzy/PNG-2PCF-model-to-observe")
OUTPUT_ROOT = PROJECT_ROOT / "outputs" / "task43_outputs" / "rsd_validation" / "task432_model_repair"
AUDIT_PATH = OUTPUT_ROOT / "audits" / "task432_baseline_contract.json"


def sha256_file(path: Path) -> str:
    """分块计算文件 SHA256，避免把大文件一次性读入内存。"""

    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while block := stream.read(8 * 1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def file_record(path: Path) -> dict[str, Any]:
    """记录输入文件的存在性、大小、修改时间和哈希。"""

    record: dict[str, Any] = {"path": str(path), "exists": path.is_file()}
    if path.is_file():
        stat = path.stat()
        record.update(
            {
                "size_bytes": int(stat.st_size),
                "mtime_ns": int(stat.st_mtime_ns),
                "sha256": sha256_file(path),
            }
        )
    return record


def main() -> None:
    """构造 Task 4.3.2 的只读 baseline 合同审计。"""

    code_root = PROJECT_ROOT / "codes" / "task43"
    source_files = [
        code_root / "task43_rsd_model.py",
        code_root / "task43_fit_rsd_rawbox_x25.py",
        code_root / "task43_rsd_rawbox_joint_4way.py",
        code_root / "task43_rsd_joint_p02xi02_fit.py",
        code_root / "task43_measure_rsd_rawbox_p02_jaxpower.py",
        code_root / "task43_measure_rsd_rawbox_xi_fcfc.py",
        code_root / "task43_measure_rsd_rawbox_pk0_jaxpower.py",
        code_root / "task43_xi_linear_rsd.py",
        code_root / "task43_xi_gsm.py",
    ]
    result_files = [
        PROJECT_ROOT
        / "outputs/task43_outputs/rsd_validation/rawbox/standard_joint_baomask80_120_v1/task43_rawbox_standard_joint_baomask80_120_v1.json",
        PROJECT_ROOT
        / "outputs/task43_outputs/rsd_validation/lightcone_standard_joint_baomask80_120_v1/task43_lightcone_standard_joint_baomask80_120_v1.json",
        PROJECT_ROOT
        / "outputs/task43_outputs/rsd_validation/rawbox/joint_p02xi02/audits/task43_rsd_rawbox_joint_4way_summary.json",
        PROJECT_ROOT
        / "9.11meeting/task43_standard_kmax0p08_smin50/task43_rawbox_real_Pxi_rsd_P02xi02_joint_kmax0p08_smin50_baomask80_120_v1.json",
        PROJECT_ROOT
        / "9.11meeting/task43_standard_kmax0p08_smin50/task43_lightcone_real_Pxi_rsd_P02xi02_joint_kmax0p08_smin50_baomask80_120_v1.json",
    ]

    contract = {
        "task": "Task 4.3.2 RSD model repair and closure",
        "status": "baseline_frozen",
        "created_by": "task432_contract.py",
        "created_pid": os.getpid(),
        "scientific_scope": {
            "primary_question": (
                "Determine whether the P-versus-xi RSD sigma_s tension and lightcone joint b1 shift "
                "come from the 2PCF RSD model, finite-lattice angular projection, multipole measurement "
                "operator, covariance, or a genuinely different effective velocity nuisance."
            ),
            "rawbox_first": True,
            "lightcone_after_rawbox": True,
            "shared_sigma_is_baseline": True,
            "split_sigma_is_diagnostic_only_initially": True,
        },
        "promoted_data_contract": {
            "xi_range_mpc_h": [50.0, 350.0],
            "xi_interval_convention": "50 <= s < 350",
            "bao_exclusion_mpc_h": [80.0, 120.0],
            "bao_exclusion_convention": "80 <= s < 120",
            "removed_xi_centers_mpc_h": [85.0, 95.0, 105.0, 115.0],
            "lightcone_p2_kmin_h_mpc": 0.015,
            "p_fixed": 1.0,
            "sn0_policy": "P ell=0 only; xi contact term fixed to zero",
            "observed_curve": "arithmetic mean of 25 realizations",
            "likelihood_covariance": "C_single; never divide by 25",
            "plot_errorbar": "sqrt(diag(C_single))",
            "plot_policy": "PDF only",
        },
        "historical_control_contracts": {
            "rawbox_fourway_unmasked": (
                "xi0 s>=50 and xi2 s>=80; retained only as provenance/control, "
                "not the promoted BAO-masked standard"
            ),
            "meeting_pdfs": "9.11/9.18 figures are immutable provenance products and may have older bin contracts",
        },
        "source_files": [file_record(path) for path in source_files],
        "result_files": [file_record(path) for path in result_files],
        "planned_variants": [
            "V0 shared-sigma current continuous-angle FullDiscrete model",
            "V1 rawbox exact-lattice low-k angular correction",
            "V2 FCFC-consistent xi(s,mu) and multipole projection",
            "V3 lightcone xi theory including required higher multipoles/window operator",
            "V4 phenomenological split sigma_s_P and sigma_s_xi diagnostic",
            "V5 optional scale-dependent velocity/GSM xi-only diagnostic",
        ],
        "resource_contract": {
            "cpu_only_total_threads": 8,
            "login_node_cpu_tasks": True,
            "gpu_only_for_required_measurement_or_raw_covariance": True,
            "overwrite_existing_production": False,
            "new_output_root": str(OUTPUT_ROOT),
        },
        "ssh_policy": {
            "keep_connection_when_possible": True,
            "after_disconnect_wait_minutes": 15,
            "max_consecutive_reconnect_failures": 5,
        },
    }
    AUDIT_PATH.parent.mkdir(parents=True, exist_ok=True)
    temporary = AUDIT_PATH.with_name(f".{AUDIT_PATH.name}.{os.getpid()}.tmp")
    temporary.write_text(json.dumps(contract, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(AUDIT_PATH)
    print(json.dumps({"status": "pass", "output": str(AUDIT_PATH)}, sort_keys=True))


if __name__ == "__main__":
    main()
