#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""审计 Task4.3 jaxpower-only ``smax`` scan 的最终文件与 provenance 卫生。

代码大纲
========
1. 从隔离 manifest 恢复 25 个 phase 的 catalog 引用与实际 FKP xi 文件；
   大 catalog 只 ``stat``，明确记录 ``copied=false``，绝不计算其哈希。
2. 严格要求 manifest、measurement bridge、25 个 xi、mean xi、RR、raw/final
   jaxpower covariance、radial operator、五条长链及最终 JSON/CSV/PDF 全部非空。
3. 为每个核心小/中等产物记录字节数与 SHA256，并检查 JSON 的成功状态。
4. 盘点 scan/plot/log 三棵隔离目录；plot 目录只允许 PDF，并扫描新产物是否
   泄漏到旧 Task43 输出目录，同时禁止新增 RascalC 文件名或 provenance 引用。
5. 记录可重跑脚本、命令和日志清单。旧 Task43 bridge 只保留路径字符串，默认
   不 ``stat``、不打开、不哈希，因此卫生审计本身不重新依赖旧结果。
6. 无论成功或失败均原子写 hygiene JSON；成功为 ``status=pass``，任何失败为
   ``status=fail`` 且进程返回非零。脚本不删除、移动或修改任何科学产物。
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import traceback
import zipfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import numpy as np


DEFAULT_PROJECT_ROOT = Path("/pscratch/sd/l/lzy/PNG-2PCF-model-to-observe")
RMAX_VALUES = (350, 400, 450, 500, 550)
EXPECTED_PHASES = tuple(f"ph{index:03d}" for index in range(25))
TEXT_SUFFIXES = {".csv", ".err", ".json", ".jsonl", ".log", ".md", ".out", ".txt"}
RASCALC_TOKEN = b"rascalc"


def utc_now() -> str:
    """返回无本地时区歧义的创建时间。"""
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def canonical(path: str | Path) -> Path:
    """解析路径但不要求目标已经存在。"""
    return Path(path).expanduser().resolve(strict=False)


def lexical_absolute(path: str | Path) -> Path:
    """Return an absolute path without resolving a symlink at the target."""
    return Path(os.path.abspath(os.fspath(Path(path).expanduser())))


def is_within(path: str | Path, root: str | Path) -> bool:
    """判断 canonical path 是否位于 canonical root 内。"""
    try:
        canonical(path).relative_to(canonical(root))
    except ValueError:
        return False
    return True


def jsonable(value: Any) -> Any:
    """递归转换 Path 与 NumPy scalar，保证 audit 可序列化。"""
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.generic):
        return jsonable(value.item())
    if isinstance(value, np.ndarray):
        return jsonable(value.tolist())
    if isinstance(value, dict):
        return {str(key): jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [jsonable(item) for item in value]
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value


def atomic_write_json(path: Path, payload: dict[str, Any]) -> None:
    """原子落盘，避免中断后留下看似完整的半个审计。"""
    path = lexical_absolute(path)
    if path.is_symlink():
        raise RuntimeError(f"refusing to follow symlink hygiene output: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(
        json.dumps(jsonable(payload), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def sha256_file(path: Path) -> str:
    """流式计算文件哈希，不把中等大小 MCMC samples 整体读入内存。"""
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while True:
            chunk = stream.read(8 * 1024 * 1024)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


class HygieneAudit:
    """累积全部问题，使一次失败审计尽可能给出完整修复清单。"""

    def __init__(self) -> None:
        self.errors: list[str] = []
        self.warnings: list[str] = []
        self.checks: list[dict[str, Any]] = []
        self.core_files: list[dict[str, Any]] = []

    def require(self, condition: bool, message: str) -> bool:
        """记录硬失败但不中断后续只读盘点。"""
        passed = bool(condition)
        self.checks.append({"pass": passed, "message": message})
        if not passed:
            self.errors.append(message)
        return passed

    def warn(self, message: str) -> None:
        """记录不改变退出码的提示。"""
        self.warnings.append(message)

    def core_file(
        self,
        label: str,
        path: Path,
        *,
        required_root: Path,
        expected_json_status: Iterable[str] | None = None,
    ) -> dict[str, Any]:
        """检查、哈希一个核心小/中等文件，并可验证 JSON 成功状态。"""
        lexical_path = lexical_absolute(path)
        self.require(
            not lexical_path.is_symlink(),
            f"core file must not be a symlink: {label}: {lexical_path}",
        )
        path = canonical(lexical_path)
        allowed_status = None if expected_json_status is None else tuple(expected_json_status)
        record: dict[str, Any] = {
            "label": label,
            "path": str(path),
            "required_root": str(canonical(required_root)),
            "exists": path.exists(),
            "is_file": path.is_file(),
            "is_symlink": lexical_path.is_symlink(),
            "size_bytes": None,
            "sha256": None,
        }
        self.require(is_within(path, required_root), f"core file escaped required root: {label}: {path}")
        usable = path.is_file() and not lexical_path.is_symlink()
        self.require(usable, f"missing core file: {label}: {path}")
        if usable:
            try:
                size = int(path.stat().st_size)
                record["size_bytes"] = size
                self.require(size > 0, f"empty core file: {label}: {path}")
                if size > 0:
                    record["sha256"] = sha256_file(path)
            except OSError as error:
                self.require(False, f"cannot stat/hash core file {label}: {path}: {error}")

        if usable and record["size_bytes"] and allowed_status is not None:
            try:
                payload = json.loads(path.read_text(encoding="utf-8"))
                status = payload.get("status") if isinstance(payload, dict) else None
                record["json_status"] = status
                record["allowed_json_status"] = list(allowed_status)
                self.require(
                    status in allowed_status,
                    f"unexpected JSON status for {label}: {status!r}; expected {allowed_status}",
                )
                if label == "measurement_audit":
                    self.require(
                        payload.get("complete_and_all_pass") is True
                        and int(payload.get("n_checked", -1)) == 25,
                        "measurement audit is not complete_and_all_pass for all 25 phases",
                    )
            except Exception as error:  # noqa: BLE001 - 错误必须进入机器审计
                self.require(False, f"cannot parse core JSON {label}: {path}: {error}")
        self.core_files.append(record)
        return record


def iter_tree(root: Path) -> list[Path]:
    """稳定列出目录树；不跟随目录 symlink。"""
    root = canonical(root)
    if not root.is_dir():
        return []
    output: list[Path] = []
    for current, directories, filenames in os.walk(root, followlinks=False):
        current_path = Path(current)
        for name in sorted(directories):
            candidate = current_path / name
            if candidate.is_symlink():
                output.append(candidate)
        for name in sorted(filenames):
            output.append(current_path / name)
    return sorted(output, key=lambda item: str(item))


def tree_inventory(root: Path, *, exclude: set[Path] | None = None) -> dict[str, Any]:
    """记录文件/链接大小；tree inventory 不重复计算核心文件哈希。"""
    root = canonical(root)
    excluded = {canonical(path) for path in (exclude or set())}
    entries: list[dict[str, Any]] = []
    total_bytes = 0
    for path in iter_tree(root):
        resolved = canonical(path)
        if resolved in excluded:
            continue
        is_link = path.is_symlink()
        is_file = path.is_file() and not is_link
        size = int(path.stat().st_size) if is_file else None
        if size is not None:
            total_bytes += size
        entries.append(
            {
                "path": str(path),
                "relative_path": str(path.relative_to(root)),
                "type": "symlink" if is_link else ("file" if is_file else "other"),
                "size_bytes": size,
                "suffix": path.suffix.lower() if is_file else None,
            }
        )
    return {
        "root": str(root),
        "exists": root.is_dir(),
        "n_entries": len(entries),
        "n_files": sum(row["type"] == "file" for row in entries),
        "n_symlinks": sum(row["type"] == "symlink" for row in entries),
        "total_file_bytes": total_bytes,
        "entries": entries,
    }


def stream_contains_token(path: Path, token: bytes) -> bool:
    """以跨 chunk 安全的方式扫描普通文本文件。"""
    overlap = max(len(token) - 1, 0)
    tail = b""
    with path.open("rb") as stream:
        while True:
            chunk = stream.read(1024 * 1024)
            if not chunk:
                return False
            combined = tail + chunk.lower()
            if token in combined:
                return True
            tail = combined[-overlap:] if overlap else b""


def npz_string_hits(path: Path, token: str) -> list[str]:
    """只展开 NPZ 内字符串数组，跳过大型数值 samples/covariance payload。"""
    hits: list[str] = []
    needle = token.casefold()
    with zipfile.ZipFile(path, "r") as archive:
        for member in archive.namelist():
            if needle in member.casefold():
                hits.append(f"member:{member}")
            if not member.endswith(".npy"):
                continue
            # np.load 对 ZipExtFile 只在字符串 member 上执行；数值大数组不会解压。
            with archive.open(member, "r") as stream:
                version = np.lib.format.read_magic(stream)
                if version == (1, 0):
                    _shape, _fortran, dtype = np.lib.format.read_array_header_1_0(stream)
                elif version in {(2, 0), (3, 0)}:
                    _shape, _fortran, dtype = np.lib.format.read_array_header_2_0(stream)
                else:
                    raise ValueError(f"unsupported NPY version {version} in {path}:{member}")
            if dtype.hasobject:
                # allow_pickle=False 的科学产物不应含不可安全审计的 object provenance。
                raise ValueError(f"object array cannot be safely provenance-scanned: {path}:{member}")
            if dtype.kind not in {"S", "U"}:
                continue
            with archive.open(member, "r") as stream:
                array = np.load(stream, allow_pickle=False)
            for index, value in enumerate(np.asarray(array).ravel()):
                text = value.decode("utf-8", errors="replace") if isinstance(value, bytes) else str(value)
                if needle in text.casefold():
                    hits.append(f"{member}[{index}]")
    return hits


def scan_for_rascalc(roots: Iterable[Path], *, exclude: set[Path]) -> dict[str, Any]:
    """扫描新结果的文件名、文本 provenance 和 NPZ 字符串 metadata。"""
    excluded = {canonical(path) for path in exclude}
    scanned: list[str] = []
    hits: list[dict[str, str]] = []
    errors: list[str] = []
    for root in roots:
        for path in iter_tree(root):
            if path.is_symlink() or not path.is_file() or canonical(path) in excluded:
                continue
            scanned.append(str(path))
            if "rascalc" in path.name.casefold():
                hits.append({"path": str(path), "location": "filename"})
            try:
                if path.suffix.lower() in TEXT_SUFFIXES and stream_contains_token(path, RASCALC_TOKEN):
                    hits.append({"path": str(path), "location": "text-content"})
                elif path.suffix.lower() == ".npz":
                    for location in npz_string_hits(path, "rascalc"):
                        hits.append({"path": str(path), "location": location})
            except Exception as error:  # noqa: BLE001 - 无法扫描本身就是卫生失败
                errors.append(f"{path}: {type(error).__name__}: {error}")
    return {"n_files_scanned": len(scanned), "hits": hits, "scan_errors": errors}


def find_spills(
    root: Path,
    isolated_root: Path,
    tokens: Iterable[str],
    *,
    allowed_paths: Iterable[Path] = (),
) -> list[str]:
    """寻找带本实验唯一标签、却落在隔离根之外的文件或链接。

    ``allowed_paths`` 只接受逐文件白名单，用于用户随后明确要求的跨 probe
    展示图；不能用目录级豁免绕过原 ``rmax_scan`` 隔离 contract。
    """
    root = canonical(root)
    isolated_root = canonical(isolated_root)
    allowed = {canonical(path) for path in allowed_paths}
    lowered = tuple(token.casefold() for token in tokens)
    spills: list[str] = []
    for path in iter_tree(root):
        if is_within(path, isolated_root):
            continue
        if canonical(path) in allowed:
            continue
        relative = str(path.relative_to(root)).casefold()
        if any(token in relative for token in lowered):
            spills.append(str(path))
    return spills


def parse_manifest(path: Path, audit: HygieneAudit) -> list[dict[str, Any]]:
    """解析 JSONL；逐行错误会成为硬失败。"""
    rows: list[dict[str, Any]] = []
    if not path.is_file() or path.stat().st_size == 0:
        return rows
    try:
        for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
            if not line.strip():
                continue
            row = json.loads(line)
            if not isinstance(row, dict):
                raise TypeError(f"line {line_number} is not a JSON object")
            rows.append(row)
    except Exception as error:  # noqa: BLE001 - 继续输出失败清单
        audit.require(False, f"cannot parse manifest {path}: {error}")
    return rows


def tagged_xi_path(base_path: Path) -> Path:
    """复现 Task43 ``path_with_weight_tag(..., P0=10000)`` 的固定命名。"""
    return base_path.with_name(f"{base_path.stem}_fkpP010000{base_path.suffix}")


def build_paths(project_root: Path, scan_root: Path, plot_root: Path) -> dict[str, Any]:
    """集中定义实验 contract 的唯一核心文件名。"""
    covariance_dir = scan_root / "covariance"
    raw_covariance_stem = (
        "jaxpower_covariance_lightcone_mmin1p4e13_ph000_fitcov_bessel_interp_"
        "smoothfftlog_mesh64_nran100k_ndata50k_pad400_win3600_ds2_k0001_"
        "3000_dk002_p1p0_s50_550_ds10"
    )
    final_covariance_stem = (
        "jaxpower_covariance_lightcone_mmin1p4e13_ph000_fitcov_bessel_interp_"
        "smoothfftlog_rrdeconv_fkpNorm4p8925e10_mesh64_nran100k_ndata50k_"
        "pad400_win3600_ds2_k0001_3000_dk002_p1p0_s50_550_ds10"
    )
    operator_stem = (
        "task43_ric_factorized_operator_ph000_dchi2_nsub200000_sobol2p22_"
        "ds2_seed20260712_L2000_s50_550_ds10"
    )
    final_stem = "task43_jaxpower_2pcf_rmax_scan_longchain"
    return {
        "manifest": scan_root / "manifests/task43_rmax_scan_mmin1p4e13_x25.jsonl",
        "manifest_audit": scan_root / "manifests/task43_rmax_scan_mmin1p4e13_x25.json",
        "measurement_audit": scan_root / "audits/task43_rmax_scan_measurement_bridge.json",
        "summary_npz": scan_root / "summary/task43_mean_xi_mmin1p4e13_x25_s50_550_ds10_fkpP010000.npz",
        "summary_json": scan_root / "summary/task43_mean_xi_mmin1p4e13_x25_s50_550_ds10_fkpP010000.json",
        "rr_npz": covariance_dir / "task43_rr_smu_ph000_nran100k_seed20260702_s50_550_ds10_nmu20.npz",
        "rr_json": covariance_dir / "task43_rr_smu_ph000_nran100k_seed20260702_s50_550_ds10_nmu20.json",
        "raw_covariance_npz": covariance_dir / f"{raw_covariance_stem}.npz",
        "raw_covariance_json": covariance_dir / f"{raw_covariance_stem}.json",
        "final_covariance_npz": covariance_dir / f"{final_covariance_stem}.npz",
        "final_covariance_json": covariance_dir / f"{final_covariance_stem}.json",
        "operator_npz": scan_root / f"operators/{operator_stem}.npz",
        "operator_json": scan_root / f"operators/{operator_stem}_audit.json",
        "final_json": scan_root / f"audits/{final_stem}.json",
        "final_csv": scan_root / f"audits/{final_stem}.csv",
        "final_pdf": plot_root / "task43_jaxpower_2pcf_rmax_scan_s50_350_550.pdf",
        "contour_pdf": plot_root
        / "task43_jaxpower_2pcf_rmax_scan_fnl_b1_contours_smax350_550.pdf",
        "contour_audit": scan_root
        / "audits/task43_jaxpower_2pcf_rmax_scan_fnl_b1_contours.json",
        "crossprobe_pdf": project_root / (
            "plots/task43/task43_pk_vs_2pcf_s50_550_both_jaxpower_covariance.pdf"
        ),
        "crossprobe_audit": project_root / (
            "outputs/task43_outputs/ric_singleterm/audits/"
            "task43_ric_pk_vs_2pcf_s50_550_both_jaxpower_longchain.json"
        ),
        "fit_root": scan_root / "fits",
        "project_root": project_root,
    }


def diagnostic_manifest(project_root: Path) -> list[dict[str, str]]:
    """给出从零重跑/复核各阶段的脚本与命令，不在卫生审计中执行。"""
    code = project_root / "codes/task43"
    rows = [
        ("manifest", code / "task43_make_rmax_scan_manifest.py", "/global/homes/l/lzy/anaconda3/envs/desilike/bin/python codes/task43/task43_make_rmax_scan_manifest.py  # fresh isolated root only"),
        ("ph000_gpu_smoke", code / "run_task43_rmax_scan_xi_gpu_single.sbatch", "sbatch --export=ALL,TASK43_INDEX=0 codes/task43/run_task43_rmax_scan_xi_gpu_single.sbatch"),
        ("25_phase_gpu_array", code / "run_task43_rmax_scan_xi_gpu_array.sbatch", "sbatch codes/task43/run_task43_rmax_scan_xi_gpu_array.sbatch"),
        ("measurement_bridge", code / "task43_audit_rmax_scan_measurements.py", "/global/homes/l/lzy/anaconda3/envs/desilike/bin/python codes/task43/task43_audit_rmax_scan_measurements.py --require-all"),
        ("xi_summary", code / "run_task43_rmax_scan_summarize_login.sh", "bash codes/task43/run_task43_rmax_scan_summarize_login.sh"),
        ("rr_builder", code / "task43_build_rmax_scan_rr_smu.py", "bash -lc 'set +u; source /global/common/software/desi/users/adematti/cosmodesi_environment.sh main; set -u; taskset -c 6-13 python codes/task43/task43_build_rmax_scan_rr_smu.py'"),
        ("covariance", code / "run_task43_rmax_scan_covariance_login.sh", "bash codes/task43/run_task43_rmax_scan_covariance_login.sh"),
        ("operator_and_chains", code / "run_task43_rmax_scan_inference_login.sh", "bash codes/task43/run_task43_rmax_scan_inference_login.sh"),
        ("final_science_audit", code / "task43_summarize_rmax_scan.py", "/global/homes/l/lzy/anaconda3/envs/desilike/bin/python codes/task43/task43_summarize_rmax_scan.py"),
        ("five_smax_contour", code / "task43_plot_rmax_scan_fnl_b1_contours.py", "/global/homes/l/lzy/anaconda3/envs/desilike/bin/python codes/task43/task43_plot_rmax_scan_fnl_b1_contours.py"),
        ("pk_vs_2pcf_smax550_contour", code / "task43_plot_ric_jaxpower_pk_vs_2pcf_smax550.py", "/global/homes/l/lzy/anaconda3/envs/desilike/bin/python codes/task43/task43_plot_ric_jaxpower_pk_vs_2pcf_smax550.py"),
        ("final_hygiene", code / "task43_audit_rmax_scan_hygiene.py", "/global/homes/l/lzy/anaconda3/envs/desilike/bin/python codes/task43/task43_audit_rmax_scan_hygiene.py"),
    ]
    return [
        {"stage": stage, "script": str(canonical(script)), "command_from_project_root": command}
        for stage, script, command in rows
    ]


def bridge_paths(project_root: Path, archive_root: Path) -> dict[str, Any]:
    """只构造旧 bridge 路径；调用者承诺不查询这些目标。"""
    old = project_root / "outputs/task43_outputs"
    old_fit = old / "ric_singleterm/fits/2pcf_jaxpower_ph000_dchi2_nsub200000_long_mcmc20k"
    return {
        "access_policy": "path strings retained only; not stat'ed, opened, parsed, or hashed by default",
        "accessed": False,
        "paths": {
            "archived_measurement_directory": str(archive_root / "xi_cucount"),
            "xi_summary": str(old / "summary/task43_mean_xi_mmin1p4e13_x25_s50_350_ds10_fkpP010000.npz"),
            "rr_smu": str(old / "summary/task43_rr_smu_window_smoke_ph000_nran100k_s50_350_ds10_nmu20_midpoint.npz"),
            "raw_covariance": str(
                old
                / "summary/jaxpower_covariance_lightcone_mmin1p4e13_ph000_fitcov_bessel_interp_smoothfftlog_mesh64_nran100k_ndata50k_pad400_win3600_ds2_k0001_3000_dk002_p1p0_s50_350_ds10.npz"
            ),
            "final_covariance": str(
                old
                / "summary/jaxpower_covariance_lightcone_mmin1p4e13_ph000_fitcov_bessel_interp_smoothfftlog_rrdeconv_mesh64_nran100k_ndata50k_pad400_win3600_ds2_k0001_3000_dk002_p1p0_s50_350_ds10.npz"
            ),
            "radial_operator": str(
                old
                / "ric_singleterm/operators/task43_ric_factorized_operator_ph000_dchi2_nsub200000_sobol2p22_ds2_seed20260712_L2000.npz"
            ),
            "fit_summary": str(old_fit / "task43_minimal_closure_mcmc_summary.json"),
            "fit_samples": str(old_fit / "task43_mcmc_radial_singleterm_samples.npz"),
            "authoritative_longchain_audit": str(
                old / "ric_singleterm/audits/task43_ric_pk_vs_2pcf_covariance_pair_longchain.json"
            ),
        },
    }


def parse_args() -> argparse.Namespace:
    """根路径均可覆盖；其余核心命名保持 frozen experiment contract。"""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, default=DEFAULT_PROJECT_ROOT)
    parser.add_argument("--scan-root", type=Path, default=None)
    parser.add_argument("--plots-root", type=Path, default=None)
    parser.add_argument("--logs-root", type=Path, default=None)
    parser.add_argument("--archive-root", type=Path, default=None)
    parser.add_argument("--output", type=Path, default=None)
    return parser.parse_args()


def run_audit(args: argparse.Namespace) -> tuple[dict[str, Any], int, Path]:
    """执行只读检查并返回 payload、退出码与待写 output path。"""
    raw_project_root = lexical_absolute(args.project_root)
    raw_scan_root = lexical_absolute(
        args.scan_root or raw_project_root / "outputs/task43_outputs/rmax_scan"
    )
    raw_plot_root = lexical_absolute(args.plots_root or raw_project_root / "plots/task43/rmax_scan")
    raw_logs_root = lexical_absolute(args.logs_root or raw_project_root / "codes/logs/task43/rmax_scan")
    raw_archive_root = lexical_absolute(
        args.archive_root
        or raw_project_root
        / "old_doc_codes/task4_task44_cleanup_20260707T061844Z/moved/outputs/task43_outputs"
    )
    raw_output = lexical_absolute(args.output or raw_scan_root / "audits/task43_rmax_scan_hygiene.json")
    project_root = canonical(raw_project_root)
    scan_root = canonical(raw_scan_root)
    plot_root = canonical(raw_plot_root)
    logs_root = canonical(raw_logs_root)
    archive_root = canonical(raw_archive_root)
    output = raw_output
    audit = HygieneAudit()
    roots = {
        "project": project_root,
        "scan": scan_root,
        "plots": plot_root,
        "logs": logs_root,
        "archive_catalog_inputs": archive_root,
    }
    audit.require(not raw_project_root.is_symlink(), f"project root must not be a symlink: {raw_project_root}")
    audit.require(not raw_archive_root.is_symlink(), f"archive root must not be a symlink: {raw_archive_root}")
    audit.require(project_root.is_dir(), f"project root is missing: {project_root}")
    audit.require(is_within(output, scan_root), f"hygiene output must remain under scan root: {output}")
    for label, raw_root, root in (
        ("scan", raw_scan_root, scan_root),
        ("plots", raw_plot_root, plot_root),
        ("logs", raw_logs_root, logs_root),
    ):
        audit.require(not raw_root.is_symlink(), f"{label} root must not be a symlink: {raw_root}")
        audit.require(root.is_dir(), f"{label} root is missing: {root}")
    audit.require(not raw_output.is_symlink(), f"hygiene output must not be a symlink: {raw_output}")

    paths = build_paths(project_root, scan_root, plot_root)
    # 核心静态 pair；所有文件都 hash，只有 catalog 引用例外。
    core_specs = [
        ("manifest_jsonl", paths["manifest"], scan_root, None),
        ("manifest_audit", paths["manifest_audit"], scan_root, ("done",)),
        ("measurement_audit", paths["measurement_audit"], scan_root, ("pass",)),
        ("xi_summary_npz", paths["summary_npz"], scan_root, None),
        ("xi_summary_json", paths["summary_json"], scan_root, ("done",)),
        ("rr_npz", paths["rr_npz"], scan_root, None),
        ("rr_json", paths["rr_json"], scan_root, ("done",)),
        ("raw_covariance_npz", paths["raw_covariance_npz"], scan_root, None),
        ("raw_covariance_json", paths["raw_covariance_json"], scan_root, ("done",)),
        ("final_covariance_npz", paths["final_covariance_npz"], scan_root, None),
        ("final_covariance_json", paths["final_covariance_json"], scan_root, ("done",)),
        ("operator_npz", paths["operator_npz"], scan_root, None),
        ("operator_audit_json", paths["operator_json"], scan_root, ("done",)),
        (
            "crossprobe_smax550_pdf",
            paths["crossprobe_pdf"],
            project_root / "plots/task43",
            None,
        ),
        (
            "crossprobe_smax550_audit",
            paths["crossprobe_audit"],
            project_root / "outputs/task43_outputs/ric_singleterm/audits",
            ("done",),
        ),
        ("five_smax_contour_pdf", paths["contour_pdf"], plot_root, None),
        (
            "five_smax_contour_audit",
            paths["contour_audit"],
            scan_root,
            ("done",),
        ),
    ]
    for label, path, required_root, statuses in core_specs:
        audit.core_file(
            label,
            path,
            required_root=required_root,
            expected_json_status=statuses,
        )

    # Enforce the backend contract from independent metadata, rather than only
    # recording the word "jaxpower" in the final audit.
    def load_core_json(label: str, path: Path) -> dict[str, Any]:
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except Exception as error:  # noqa: BLE001 - accumulate every gate
            audit.require(False, f"cannot load {label} for backend gate: {path}: {error}")
            return {}
        audit.require(isinstance(payload, dict), f"{label} must be a JSON object: {path}")
        return payload if isinstance(payload, dict) else {}

    manifest_meta = load_core_json("manifest audit", paths["manifest_audit"])
    measurement_meta = load_core_json("measurement audit", paths["measurement_audit"])
    raw_meta = load_core_json("raw covariance metadata", paths["raw_covariance_json"])
    final_meta = load_core_json("final covariance metadata", paths["final_covariance_json"])
    science_meta = load_core_json("final science audit", paths["final_json"])
    final_input_covariance = final_meta.get("input_covariance")
    final_input_rr = final_meta.get("input_rr_smu")
    science_covariance = science_meta.get("inputs", {}).get("covariance_50bin")
    audit.require(
        manifest_meta.get("measurement", {}).get("backend") == "cucount.jax",
        "manifest audit does not require measurement backend cucount.jax",
    )
    audit.require(
        measurement_meta.get("expected_measurement", {}).get("backend") == "jax"
        and measurement_meta.get("expected_measurement", {}).get("engine") == "cucount_jax",
        "measurement audit does not certify backend=jax and engine=cucount_jax",
    )
    audit.require(
        raw_meta.get("task") == "task43_diagnose_jaxpower_lightcone_covariance"
        and raw_meta.get("window", {}).get("interface")
        == "jaxpower.compute_fkp2_covariance_window",
        "raw covariance metadata does not certify the frozen jaxpower interface",
    )
    audit.require(
        final_meta.get("task") == "task43_apply_rr_smu_deconvolution_to_covariance"
        and isinstance(final_input_covariance, str)
        and canonical(final_input_covariance)
        == canonical(paths["raw_covariance_npz"])
        and isinstance(final_input_rr, str)
        and canonical(final_input_rr)
        == canonical(paths["rr_npz"]),
        "final covariance metadata does not point to this scan's raw jaxpower covariance and RR window",
    )
    audit.require(
        science_meta.get("task") == "task43_summarize_rmax_scan"
        and "jaxpower" in str(science_meta.get("scope", {}).get("covariance", "")).casefold()
        and isinstance(science_covariance, str)
        and canonical(science_covariance)
        == canonical(paths["final_covariance_npz"]),
        "final science audit does not identify this scan's fixed jaxpower covariance",
    )

    rows = parse_manifest(paths["manifest"], audit)
    phases = [str(row.get("phase")) for row in rows]
    audit.require(len(rows) == 25, f"manifest must contain 25 rows, found {len(rows)}")
    audit.require(tuple(phases) == EXPECTED_PHASES, f"manifest phase order mismatch: {phases}")
    audit.require(len(set(phases)) == len(phases), "manifest contains duplicate phases")
    catalog_records: list[dict[str, Any]] = []
    xi_paths: list[Path] = []
    for index, row in enumerate(rows):
        phase = str(row.get("phase", f"row{index}"))
        copied_flag = row.get("input_location", {}).get("catalogs_copied")
        audit.require(copied_flag is False, f"manifest {phase} must set catalogs_copied=false")
        audit.require(
            row.get("experiment") == "task43_jaxpower_only_rmax_scan",
            f"manifest {phase} experiment tag mismatch",
        )
        for kind, key in (("halo", "halo_catalog_path"), ("random", "random_catalog_path")):
            value = row.get(key)
            lexical_path = lexical_absolute(value) if isinstance(value, str) and value else Path("/__missing__")
            audit.require(not lexical_path.is_symlink(), f"archived {kind} catalog must not be a symlink for {phase}: {lexical_path}")
            path = canonical(lexical_path)
            exists = path.is_file() and not lexical_path.is_symlink()
            size = int(path.stat().st_size) if exists else None
            audit.require(exists, f"missing archived {kind} catalog for {phase}: {path}")
            audit.require(is_within(path, archive_root), f"{phase} {kind} catalog is outside controlled archive: {path}")
            audit.require(not is_within(path, scan_root), f"{phase} {kind} catalog was copied into scan root: {path}")
            audit.require(size is not None and size > 0, f"empty archived {kind} catalog for {phase}: {path}")
            catalog_records.append(
                {
                    "phase": phase,
                    "kind": kind,
                    "path": str(path),
                    "size_bytes": size,
                    "copied": False,
                    "sha256": None,
                    "hash_policy": "intentionally skipped for large archived catalog",
                }
            )
        base_value = row.get("xi_path")
        base = canonical(base_value) if isinstance(base_value, str) and base_value else Path("/__missing__")
        actual = tagged_xi_path(base)
        audit.require(is_within(base, scan_root / "xi_cucount"), f"manifest xi base escaped isolated xi root for {phase}: {base}")
        audit.require(base.name.endswith("_s50_550_ds10.npz"), f"manifest xi base name mismatch for {phase}: {base.name}")
        xi_paths.append(actual)
        audit.core_file(f"xi_{phase}", actual, required_root=scan_root)
    audit.require(len({str(path) for path in xi_paths}) == len(xi_paths), "manifest maps multiple phases to one xi file")

    # 五档链必须是 summary+samples 完整 pair。
    fit_files: dict[str, dict[str, str]] = {}
    for rmax in RMAX_VALUES:
        fit_dir = scan_root / f"fits/smax{rmax}"
        summary = fit_dir / "task43_minimal_closure_mcmc_summary.json"
        samples = fit_dir / "task43_mcmc_radial_singleterm_samples.npz"
        audit.core_file(
            f"fit_smax{rmax}_summary",
            summary,
            required_root=scan_root,
            expected_json_status=("done",),
        )
        audit.core_file(f"fit_smax{rmax}_samples", samples, required_root=scan_root)
        fit_files[str(rmax)] = {"summary": str(summary), "samples": str(samples)}

    # 最终 science JSON/CSV/PDF 在最后加入，PDF 只能位于专用 plot root。
    audit.core_file(
        "final_science_json",
        paths["final_json"],
        required_root=scan_root,
        expected_json_status=("done",),
    )
    audit.core_file("final_science_csv", paths["final_csv"], required_root=scan_root)
    audit.core_file("final_science_pdf", paths["final_pdf"], required_root=plot_root)

    diagnostics = diagnostic_manifest(project_root)
    for row in diagnostics:
        script = Path(row["script"])
        usable = script.is_file() and not script.is_symlink() and script.stat().st_size > 0
        audit.require(usable, f"missing rerunnable diagnostic script: {script}")
        row["exists_nonempty"] = usable
        row["size_bytes"] = int(script.stat().st_size) if usable else None
        row["sha256"] = sha256_file(script) if usable else None

    scan_inventory = tree_inventory(scan_root, exclude={output})
    plot_inventory = tree_inventory(plot_root)
    log_inventory = tree_inventory(logs_root)
    audit.require(scan_inventory["n_symlinks"] == 0, "scan root contains symlinks")
    audit.require(plot_inventory["n_symlinks"] == 0, "plot root contains symlinks")
    audit.require(log_inventory["n_symlinks"] == 0, "log root contains symlinks")
    non_pdf_plots = [
        row["path"]
        for row in plot_inventory["entries"]
        if row["type"] == "file" and row["suffix"] != ".pdf"
    ]
    audit.require(not non_pdf_plots, f"plots/rmax_scan contains non-PDF files: {non_pdf_plots}")
    invalid_pdf_headers: list[str] = []
    for row in plot_inventory["entries"]:
        if row["type"] != "file" or row["suffix"] != ".pdf":
            continue
        plot_path = Path(row["path"])
        try:
            with plot_path.open("rb") as stream:
                magic = stream.read(5)
            if magic != b"%PDF-":
                invalid_pdf_headers.append(str(plot_path))
        except OSError:
            invalid_pdf_headers.append(str(plot_path))
    audit.require(not invalid_pdf_headers, f"plot files lack a PDF magic header: {invalid_pdf_headers}")

    # 唯一实验标签不允许散落到旧 output/plot/log 位置。
    spill_tokens = (
        "rmax_scan",
        "s50_550_ds10",
        "smax350",
        "smax400",
        "smax450",
        "smax500",
        "smax550",
        "rmax_scan_longchain",
    )
    science_output_root = project_root / "outputs/task43_outputs"
    plot_parent = project_root / "plots/task43"
    log_parent = project_root / "codes/logs/task43"
    allowed_crossprobe = {
        "science_outputs": [paths["crossprobe_audit"]],
        "plots": [paths["crossprobe_pdf"]],
        "logs": [],
    }
    spills = {
        "science_outputs": find_spills(
            science_output_root,
            scan_root,
            spill_tokens,
            allowed_paths=allowed_crossprobe["science_outputs"],
        ),
        "plots": find_spills(
            plot_parent,
            plot_root,
            spill_tokens,
            allowed_paths=allowed_crossprobe["plots"],
        ),
        "logs": find_spills(
            log_parent,
            logs_root,
            spill_tokens,
            allowed_paths=allowed_crossprobe["logs"],
        ),
    }
    audit.require(not any(spills.values()), f"rmax-scan artifacts leaked outside isolated roots: {spills}")

    # 白名单仍必须严格保持“一份 JSON + 一份 PDF”，且不得混入 RascalC。
    crossprobe_pdf_ok = paths["crossprobe_pdf"].is_file()
    audit.require(crossprobe_pdf_ok, "allowed smax550 cross-probe PDF is missing")
    if crossprobe_pdf_ok:
        with paths["crossprobe_pdf"].open("rb") as stream:
            audit.require(
                paths["crossprobe_pdf"].suffix.lower() == ".pdf"
                and stream.read(5) == b"%PDF-",
                "allowed smax550 cross-probe plot is not a valid PDF",
            )
    crossprobe_audit_ok = paths["crossprobe_audit"].is_file()
    audit.require(crossprobe_audit_ok, "allowed smax550 cross-probe audit is missing")
    if crossprobe_audit_ok:
        audit.require(
            "rascalc" not in paths["crossprobe_pdf"].name.casefold()
            and "rascalc"
            not in paths["crossprobe_audit"].read_text(encoding="utf-8").casefold(),
            "allowed smax550 cross-probe deliverable contains RascalC provenance",
        )

    rascalc = scan_for_rascalc(
        (scan_root, plot_root, logs_root),
        exclude={output},
    )
    audit.require(not rascalc["hits"], f"new rmax-scan artifact contains RascalC reference: {rascalc['hits']}")
    audit.require(not rascalc["scan_errors"], f"RascalC provenance scan could not inspect files: {rascalc['scan_errors']}")

    status = "pass" if not audit.errors else "fail"
    payload = {
        "task": "task43_audit_rmax_scan_hygiene",
        "status": status,
        "created_utc": utc_now(),
        "scope": "Task43 jaxpower-only 2PCF rmax scan final artifact/provenance hygiene",
        "roots": roots,
        "output": str(output),
        "self_output_policy": "excluded from its own pre-write inventory and SHA256 to avoid a self-referential hash",
        "contract": {
            "phases": list(EXPECTED_PHASES),
            "rmax_edges_mpc_h": list(RMAX_VALUES),
            "required_core_file_count": 55,
            "catalog_policy": "25 halo/random pairs referenced in controlled archive; never copied or hashed",
            "plot_policy": "every file below plots/task43/rmax_scan must be a PDF",
            "covariance_backend": "jaxpower only",
            "rascalc_policy": "no new artifact name or provenance/content reference",
        },
        "core_files": audit.core_files,
        "core_file_count": len(audit.core_files),
        "catalog_references": {
            "n_records": len(catalog_records),
            "n_pairs": len(catalog_records) // 2,
            "records": catalog_records,
        },
        "fit_files": fit_files,
        "directory_inventory": {
            "scan": scan_inventory,
            "plots": plot_inventory,
            "logs": log_inventory,
        },
        "isolation": {
            "spill_tokens": list(spill_tokens),
            "allowed_crossprobe_files": {
                key: [str(canonical(path)) for path in values]
                for key, values in allowed_crossprobe.items()
            },
            "spills": spills,
        },
        "rascalc_scan": rascalc,
        "rerunnable_diagnostics": {
            "working_directory": str(project_root),
            "policy": "listed and hashed only; no diagnostic/science command executed by hygiene audit",
            "stages": diagnostics,
        },
        "old_task43_bridge_references": bridge_paths(project_root, archive_root),
        "checks": audit.checks,
        "n_checks": len(audit.checks),
        "errors": audit.errors,
        "warnings": audit.warnings,
    }
    audit.require(
        len(audit.core_files) == 55,
        f"internal core-file enumeration mismatch: found {len(audit.core_files)}, expected 55",
    )
    # 上一条 internal check 必须反映到最终 status/payload 中。
    payload["checks"] = audit.checks
    payload["n_checks"] = len(audit.checks)
    payload["errors"] = audit.errors
    payload["status"] = "pass" if not audit.errors else "fail"
    return payload, (0 if payload["status"] == "pass" else 1), output


def main() -> int:
    """失败也写机器可读 JSON；永不删除、移动或覆盖科学文件。"""
    args = parse_args()
    fallback_project = canonical(args.project_root)
    fallback_scan = canonical(args.scan_root or fallback_project / "outputs/task43_outputs/rmax_scan")
    output = lexical_absolute(args.output or fallback_scan / "audits/task43_rmax_scan_hygiene.json")
    try:
        payload, return_code, output = run_audit(args)
    except Exception as error:  # noqa: BLE001 - 顶层异常也必须有 failure audit
        payload = {
            "task": "task43_audit_rmax_scan_hygiene",
            "status": "fail",
            "created_utc": utc_now(),
            "output": str(output),
            "error": {
                "type": type(error).__name__,
                "message": str(error),
                "traceback": traceback.format_exc(),
            },
        }
        return_code = 1
    try:
        atomic_write_json(output, payload)
    except Exception as error:  # noqa: BLE001 - 输出失败只能 stderr 报告
        print(f"[fatal] cannot write hygiene audit {output}: {type(error).__name__}: {error}", file=sys.stderr)
        return 2
    print(f"[{payload['status']}] wrote {output}")
    if return_code:
        for message in payload.get("errors", []):
            print(f"[fail] {message}", file=sys.stderr)
    return return_code


if __name__ == "__main__":
    raise SystemExit(main())
