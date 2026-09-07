#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
代码大纲（执行逻辑关系）：

1. 定义 overlay 图里全部需要重跑的 21 个 UltraNest case：
   - kmax_fit=0.08：3 个 tag × 5 个 case（P(k)+4 个 2PCF rlist）
   - kmax_fit=0.06：3 个 tag × P(k)
   - kmax_fit=0.10：3 个 tag × P(k)
2. 在登录节点并发启动 worker；每个 worker 调用 `task45_quijote_ultranest.py`
   只跑一个 tag/case，并打开 `--skip-final-products`，避免并发覆盖总表和总图。
3. 每轮 worker 完成后读取 summary，检查 `ultranest.ncall >= min_ncall`。
   若不足 30000，则提高 `min_live` 后重跑该 case。
4. 全部 case 达标后，分别用 `--skip-existing` collect 三个 run root，
   重画所有 corner、summary table 和各自 forest 图。
5. 用 `task45_plot_free_sn0_kmax_overlay.py` 生成最终 overlay PDF，
   并写出 steps/audit 表，证明每个 plotted result 都达到步数要求。
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path


PROJECT_ROOT = Path("/pscratch/sd/l/lzy/PNG-2PCF-model-to-observe")
TASK45 = PROJECT_ROOT / "codes" / "task4" / "task45_quijote_ultranest.py"
OVERLAY = PROJECT_ROOT / "codes" / "task4" / "task45_plot_free_sn0_kmax_overlay.py"
LOG_ROOT = PROJECT_ROOT / "codes" / "logs" / "task4" / "task45_30k"
OUT_ROOT = PROJECT_ROOT / "outputs" / "task4_outputs"
PLOT_ROOT = PROJECT_ROOT / "plots" / "task4"
IMPORTANT_DIR = PLOT_ROOT / "important_4p1_4p2"

TAGS = ["fid", "LCp50", "LCp100"]
XI_CASES = ["xi_r50_350", "xi_r60_350", "xi_r80_350", "xi_r100_350"]
ALL_CASES = ["pk_binavg", *XI_CASES]


@dataclass(frozen=True)
class RunSpec:
    """一个需要跑的 tag/case/kmax 组合。"""

    kmax_label: str
    kmax_fit: float
    output_label: str
    tag: str
    case: str

    @property
    def output_root(self) -> Path:
        """Task45 对应的输出目录。"""
        return OUT_ROOT / f"quijote_ultranest_{self.output_label}"

    @property
    def plot_root(self) -> Path:
        """Task45 对应的画图目录。"""
        return PLOT_ROOT / f"quijote_ultranest_{self.output_label}"

    @property
    def case_stem(self) -> str:
        """Task45 summary/corner 文件名前缀。"""
        return f"task45_{self.output_label}"

    @property
    def summary_path(self) -> Path:
        """这个 case 的 summary JSON。"""
        return self.output_root / f"{self.case_stem}_{self.tag}_{self.case}_summary.json"

    @property
    def log_path(self) -> Path:
        """这个 case 的 worker 日志。"""
        return LOG_ROOT / f"{self.case_stem}_{self.tag}_{self.case}.log"


def build_specs() -> list[RunSpec]:
    """生成 overlay 图全部 21 个待重跑 case。"""
    specs: list[RunSpec] = []
    for tag in TAGS:
        for case in ALL_CASES:
            specs.append(RunSpec("0.08", 0.08, "free_sn0_30k", tag, case))
        specs.append(RunSpec("0.06", 0.06, "free_sn0_kmax0p06_30k", tag, "pk_binavg"))
        specs.append(RunSpec("0.10", 0.10, "free_sn0_kmax0p10_30k", tag, "pk_binavg"))
    return specs


def task45_env() -> dict[str, str]:
    """返回每个 worker 使用的 1-thread 环境变量。"""
    env = os.environ.copy()
    env.update(
        {
            "PYTHONUNBUFFERED": "1",
            "OMP_NUM_THREADS": "1",
            "MKL_NUM_THREADS": "1",
            "OPENBLAS_NUM_THREADS": "1",
            "NUMEXPR_NUM_THREADS": "1",
            "VECLIB_MAXIMUM_THREADS": "1",
            "JAX_NUM_THREADS": "1",
            "XLA_FLAGS": "--xla_cpu_multi_thread_eigen=false intra_op_parallelism_threads=1",
            "HDF5_USE_FILE_LOCKING": "FALSE",
            "DESILIKE_CONFIG_DIR": str(PROJECT_ROOT / "codes" / "logs" / "task4" / "desilike_config"),
            "MPLCONFIGDIR": str(PROJECT_ROOT / "codes" / "logs" / "task4" / "mplconfig"),
            "XDG_CACHE_HOME": str(PROJECT_ROOT / "codes" / "logs" / "task4" / "xdg_cache"),
        }
    )
    return env


def run_command_for_spec(spec: RunSpec, min_live: int, min_ess: int, max_ncalls: int) -> list[str]:
    """构造一个 worker 的 Task45 命令。"""
    return [
        sys.executable,
        "-u",
        str(TASK45),
        "--tags",
        spec.tag,
        "--cases",
        spec.case,
        "--free-sn0",
        "--output-label",
        spec.output_label,
        "--sn0-prior=-1,1",
        "--kmax-fit",
        f"{spec.kmax_fit:.3f}",
        "--min-live",
        str(min_live),
        "--dlogz",
        "0.5",
        "--min-ess",
        str(min_ess),
        "--max-ncalls",
        str(max_ncalls),
        "--overwrite",
        "--quiet",
        "--skip-final-products",
    ]


def summary_ncall(spec: RunSpec) -> int | None:
    """读取一个 case 的 ncall；不存在或损坏时返回 None。"""
    if not spec.summary_path.exists():
        return None
    try:
        data = json.loads(spec.summary_path.read_text(encoding="utf-8"))
        return int(data["ultranest"]["ncall"])
    except Exception:
        return None


def run_specs_parallel(
    specs: list[RunSpec],
    *,
    max_parallel: int,
    min_live: int,
    min_ess: int,
    min_ncall: int,
    max_ncalls: int,
) -> None:
    """
    并发运行一批 specs。

    每个进程 1 线程；`max_parallel` 因此就是总核数上限。
    """
    LOG_ROOT.mkdir(parents=True, exist_ok=True)
    env = task45_env()
    pending = [spec for spec in specs if (summary_ncall(spec) or 0) < min_ncall]
    skipped = len(specs) - len(pending)
    if skipped:
        print(f"[skip] {skipped} cases already have ncall >= {min_ncall}", flush=True)
    running: list[tuple[RunSpec, subprocess.Popen, object]] = []
    failed: list[RunSpec] = []

    while pending or running:
        while pending and len(running) < max_parallel:
            spec = pending.pop(0)
            spec.log_path.parent.mkdir(parents=True, exist_ok=True)
            log_file = spec.log_path.open("w", encoding="utf-8")
            cmd = run_command_for_spec(spec, min_live=min_live, min_ess=min_ess, max_ncalls=max_ncalls)
            print(
                f"[launch] kmax={spec.kmax_label} tag={spec.tag} case={spec.case} "
                f"min_live={min_live} min_ess={min_ess} log={spec.log_path}",
                flush=True,
            )
            proc = subprocess.Popen(cmd, cwd=str(PROJECT_ROOT), env=env, stdout=log_file, stderr=subprocess.STDOUT)
            running.append((spec, proc, log_file))

        time.sleep(5)
        still_running: list[tuple[RunSpec, subprocess.Popen, object]] = []
        for spec, proc, log_file in running:
            code = proc.poll()
            if code is None:
                still_running.append((spec, proc, log_file))
                continue
            log_file.close()
            ncall = summary_ncall(spec)
            print(
                f"[done] kmax={spec.kmax_label} tag={spec.tag} case={spec.case} "
                f"return={code} ncall={ncall}",
                flush=True,
            )
            if code != 0:
                failed.append(spec)
        running = still_running

    if failed:
        lines = "\n".join(f"{s.kmax_label} {s.tag} {s.case} log={s.log_path}" for s in failed)
        raise RuntimeError("worker failures:\n" + lines)


def collect_root(output_label: str, kmax_fit: float, cases: list[str]) -> Path:
    """
    对一个 root 运行 Task45 collect：复用 existing summary，重画 corner/forest/table/manifest。
    """
    cmd = [
        sys.executable,
        "-u",
        str(TASK45),
        "--tags",
        ",".join(TAGS),
        "--cases",
        ",".join(cases),
        "--free-sn0",
        "--output-label",
        output_label,
        "--sn0-prior=-1,1",
        "--kmax-fit",
        f"{kmax_fit:.3f}",
        "--min-live",
        "720",
        "--dlogz",
        "0.5",
        "--min-ess",
        "2000",
        "--max-ncalls",
        "100000",
        "--skip-existing",
        "--quiet",
    ]
    print(f"[collect] label={output_label}", flush=True)
    subprocess.run(cmd, cwd=str(PROJECT_ROOT), env=task45_env(), check=True)
    return OUT_ROOT / f"quijote_ultranest_{output_label}" / f"task45_quijote_ultranest_{output_label}_manifest.json"


def write_steps_table(manifest_paths: dict[str, Path], min_ncall: int, out_root: Path) -> tuple[Path, Path, list[dict]]:
    """写出 steps CSV/Markdown，并返回所有行。"""
    rows: list[dict] = []
    for kmax_label, path in manifest_paths.items():
        data = json.loads(path.read_text(encoding="utf-8"))
        for item in data["summaries"]:
            if item["case"] != "pk_binavg" and kmax_label != "0.08":
                continue
            pars = item["parameters"]
            rows.append(
                {
                    "kmax_fit": kmax_label,
                    "tag": item["tag"],
                    "case": item["case"],
                    "ncall": int(item["ultranest"]["ncall"]),
                    "niter": int(item["ultranest"]["niter"]),
                    "ess": float(item["ultranest"]["ess"]),
                    "logzerr": float(item["ultranest"]["logzerr"]),
                    "fnl_loc": float(pars["fnl_loc"]["median"]),
                    "b1": float(pars["b1"]["median"]),
                    "sigmas": float(pars["sigmas"]["median"]),
                    "sn0": float(pars["sn0"]["median"]),
                    "passes_min_ncall": int(item["ultranest"]["ncall"]) >= min_ncall,
                }
            )
    rows.sort(key=lambda r: (float(r["kmax_fit"]), r["tag"], r["case"]))

    out_root.mkdir(parents=True, exist_ok=True)
    csv_path = out_root / "task45_free_sn0_pk_kmax_scan_30k_steps.csv"
    md_path = out_root / "task45_free_sn0_pk_kmax_scan_30k_steps.md"
    with csv_path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    with md_path.open("w", encoding="utf-8") as file:
        file.write("| kmax_fit | tag | case | ncall | niter | ESS | logzerr | fnl_loc | b1 | sn0 | pass |\n")
        file.write("|---:|---|---|---:|---:|---:|---:|---:|---:|---:|---|\n")
        for row in rows:
            file.write(
                f"| {row['kmax_fit']} | {row['tag']} | {row['case']} | {row['ncall']} | "
                f"{row['niter']} | {row['ess']:.1f} | {row['logzerr']:.3f} | "
                f"{row['fnl_loc']:.2f} | {row['b1']:.4f} | {row['sn0']:.4f} | "
                f"{'yes' if row['passes_min_ncall'] else 'NO'} |\n"
            )
    return csv_path, md_path, rows


def make_overlay(manifest_paths: dict[str, Path], out_root: Path) -> tuple[Path, Path]:
    """用 30k manifests 重画用户指定的 overlay PDF。"""
    overlay_pdf = PLOT_ROOT / "quijote_ultranest_free_sn0" / "task45_quijote_ultranest_free_sn0_alltags_constraints_with_pk_kmax_scan.pdf"
    overlay_manifest = out_root / "task45_quijote_ultranest_free_sn0_kmax_overlay_30k_manifest.json"
    cmd = [
        sys.executable,
        "-u",
        str(OVERLAY),
        "--base-manifest",
        str(manifest_paths["0.08"]),
        "--kmax006-manifest",
        str(manifest_paths["0.06"]),
        "--kmax010-manifest",
        str(manifest_paths["0.10"]),
        "--out-pdf",
        str(overlay_pdf),
        "--out-manifest",
        str(overlay_manifest),
    ]
    subprocess.run(cmd, cwd=str(PROJECT_ROOT), env=task45_env(), check=True)
    IMPORTANT_DIR.mkdir(parents=True, exist_ok=True)
    important_link = IMPORTANT_DIR / "4p1_diagnostic_task45_quijote_ultranest_free_sn0_pk_kmax_scan.pdf"
    rel_target = Path("../quijote_ultranest_free_sn0") / overlay_pdf.name
    if important_link.exists() or important_link.is_symlink():
        important_link.unlink()
    important_link.symlink_to(rel_target)
    return overlay_pdf, overlay_manifest


def parse_args() -> argparse.Namespace:
    """解析命令行参数。"""
    parser = argparse.ArgumentParser(description="Run Task45 free-sn0 kmax overlay with >=30k ncall")
    parser.add_argument("--max-parallel", type=int, default=8, help="maximum concurrent one-thread workers; must be <=12")
    parser.add_argument("--min-ncall", type=int, default=30000, help="required minimum UltraNest likelihood calls")
    parser.add_argument("--initial-min-live", type=int, default=120, help="first-pass UltraNest min live points")
    parser.add_argument("--initial-min-ess", type=int, default=4000, help="first-pass UltraNest minimum effective samples")
    parser.add_argument("--retry-min-live", type=int, default=120, help="retry UltraNest min live points if ncall is too low")
    parser.add_argument("--retry-min-ess", type=int, default=6000, help="retry UltraNest minimum effective samples if ncall is too low")
    parser.add_argument("--max-ncalls", type=int, default=100000, help="UltraNest max likelihood calls per worker")
    return parser.parse_args()


def main() -> None:
    """主程序入口。"""
    args = parse_args()
    if args.max_parallel > 12:
        raise ValueError("--max-parallel must be <= 12 on the login node")

    specs = build_specs()
    run_specs_parallel(
        specs,
        max_parallel=args.max_parallel,
        min_live=args.initial_min_live,
        min_ess=args.initial_min_ess,
        min_ncall=args.min_ncall,
        max_ncalls=args.max_ncalls,
    )
    short = [spec for spec in specs if (summary_ncall(spec) or 0) < args.min_ncall]
    if short:
        print(
            f"[retry] {len(short)} cases below {args.min_ncall}; "
            f"rerun with min_live={args.retry_min_live}, min_ess={args.retry_min_ess}",
            flush=True,
        )
        run_specs_parallel(
            short,
            max_parallel=args.max_parallel,
            min_live=args.retry_min_live,
            min_ess=args.retry_min_ess,
            min_ncall=args.min_ncall,
            max_ncalls=args.max_ncalls,
        )

    still_short = [spec for spec in specs if (summary_ncall(spec) or 0) < args.min_ncall]
    if still_short:
        lines = "\n".join(f"{s.kmax_label} {s.tag} {s.case} ncall={summary_ncall(s)}" for s in still_short)
        raise RuntimeError("cases still below min_ncall:\n" + lines)

    manifest_paths = {
        "0.08": collect_root("free_sn0_30k", 0.08, ALL_CASES),
        "0.06": collect_root("free_sn0_kmax0p06_30k", 0.06, ["pk_binavg"]),
        "0.10": collect_root("free_sn0_kmax0p10_30k", 0.10, ["pk_binavg"]),
    }
    out_root = OUT_ROOT / "quijote_ultranest_free_sn0_kmax_overlay_30k"
    steps_csv, steps_md, rows = write_steps_table(manifest_paths, args.min_ncall, out_root)
    overlay_pdf, overlay_manifest = make_overlay(manifest_paths, out_root)

    audit = {
        "task": "task45_free_sn0_kmax_scan_30k_parallel",
        "status": "done",
        "min_ncall": args.min_ncall,
        "max_parallel": args.max_parallel,
        "all_pass_min_ncall": all(row["passes_min_ncall"] for row in rows),
        "outputs": {
            "steps_csv": str(steps_csv),
            "steps_markdown": str(steps_md),
            "overlay_pdf": str(overlay_pdf),
            "overlay_manifest": str(overlay_manifest),
            "manifest_paths": {k: str(v) for k, v in manifest_paths.items()},
            "log_root": str(LOG_ROOT),
        },
    }
    audit_path = out_root / "task45_free_sn0_kmax_scan_30k_audit.json"
    audit_path.write_text(json.dumps(audit, indent=2), encoding="utf-8")
    print(f"[write] {steps_md}", flush=True)
    print(f"[write] {overlay_pdf}", flush=True)
    print(f"[write] {audit_path}", flush=True)


if __name__ == "__main__":
    main()
