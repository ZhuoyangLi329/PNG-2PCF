#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""构造 Task43 rmin-scan manifest，并原位引用既有大 catalog。

执行逻辑大纲：
1. 读取已经通过审计的 rmax-scan manifest，复用其中 25 个 phase 的输入路径。
2. 严格检查 phase 顺序、halo/random 文件和不可变的科学口径。
3. 只改写新实验标签、径向网格和输出路径，不复制或移动任何 catalog。
4. 原子写出 JSONL manifest 与小型 JSON audit，供 GPU array 和后续审计共用。
"""

from __future__ import annotations

import argparse
import json
import os
import tempfile
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path("/pscratch/sd/l/lzy/PNG-2PCF-model-to-observe")
MOVED_OUTPUT_ROOT = PROJECT_ROOT / "plots/outputs/task43_outputs"
SOURCE_MANIFEST = MOVED_OUTPUT_ROOT / "rmax_scan/manifests/task43_rmax_scan_mmin1p4e13_x25.jsonl"
SCAN_ROOT = PROJECT_ROOT / "outputs/task43_outputs/rmin_scan"
DEFAULT_OUTPUT = SCAN_ROOT / "manifests/task43_rmin_scan_mmin1p4e13_x25.jsonl"


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    """读取 JSONL manifest。

    参数：path 为输入 JSONL 路径。
    返回：按文件顺序排列的字典列表。
    """

    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def atomic_write_text(path: Path, text: str) -> None:
    """在目标目录内临时写入后原子替换文件，避免中断留下半文件。

    参数：path 为最终路径，text 为完整文本。
    返回：无。
    """

    path.parent.mkdir(parents=True, exist_ok=True)
    handle, temporary = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(handle, "w", encoding="utf-8") as stream:
            stream.write(text)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    except Exception:
        Path(temporary).unlink(missing_ok=True)
        raise


def output_base(row: dict[str, Any]) -> Path:
    """根据 phase 返回未附加 FKP tag 的新 2PCF 输出路径。

    参数：row 为单个 source-manifest 条目。
    返回：位于隔离 rmin_scan 目录下的 NPZ 路径。
    """

    phase = str(row["phase"])
    return SCAN_ROOT / "xi_cucount" / (
        f"xi0_AbacusSummit_base_c000_{phase}_z0p6_0p8_mmin1p4e13_x25_s30_350_ds10.npz"
    )


def main() -> None:
    """解析参数、验证 25 个输入 phase，并写出新 manifest 与审计。"""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-manifest", type=Path, default=SOURCE_MANIFEST)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    if not args.source_manifest.is_file():
        raise FileNotFoundError(args.source_manifest)
    if args.output.exists() and not args.force:
        raise FileExistsError(f"目标已存在；如需重建必须显式传入 --force：{args.output}")

    source_rows = read_jsonl(args.source_manifest)
    expected_phases = [f"ph{index:03d}" for index in range(25)]
    phases = [str(row.get("phase")) for row in source_rows]
    if phases != expected_phases:
        raise RuntimeError(f"source manifest 必须严格为 ph000..ph024，实际为 {phases}")

    rows: list[dict[str, Any]] = []
    for source in source_rows:
        for key in ("halo_catalog_path", "random_catalog_path"):
            path = Path(str(source[key]))
            if not path.is_file() or path.stat().st_size == 0:
                raise FileNotFoundError(f"{source['phase']} 的 {key} 缺失或为空：{path}")
        if (
            float(source.get("zmin", -1.0)) != 0.6
            or float(source.get("zmax", -1.0)) != 0.8
            or str(source.get("space_mode")) != "real"
            or float(source.get("mass_threshold_hmsun", -1.0)) != 1.4e13
        ):
            raise RuntimeError(f"{source['phase']} 的科学口径与 Task43 baseline 不一致")

        row = dict(source)
        row.update(
            {
                "experiment": "task43_jaxpower_only_rmin_scan",
                "xi_path": str(output_base(source)),
                "measurement_edges_mpc_h": {"min": 30.0, "max": 350.0, "step": 10.0},
                "fit_rmin_edges_mpc_h": [30.0, 40.0, 50.0],
                "fit_rmax_edge_mpc_h": 350.0,
                "source_manifest": str(args.source_manifest.resolve()),
                "catalog_policy": "reference immutable controlled archive in place; never copy or move",
            }
        )
        rows.append(row)

    manifest_text = "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows)
    atomic_write_text(args.output, manifest_text)
    audit = {
        "status": "done",
        "task": "task43_make_rmin_scan_manifest",
        "source_manifest": str(args.source_manifest.resolve()),
        "output_manifest": str(args.output.resolve()),
        "nrows": 25,
        "phases": phases,
        "catalogs_copied": False,
        "measurement": {
            "backend": "cucount.jax",
            "estimator": "weighted Landy-Szalay",
            "p0": 10000.0,
            "s_edges": [float(value) for value in range(30, 351, 10)],
            "nbins": 32,
        },
    }
    audit_path = args.output.with_suffix(".json")
    atomic_write_text(audit_path, json.dumps(audit, indent=2, sort_keys=True) + "\n")
    print(json.dumps(audit, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
