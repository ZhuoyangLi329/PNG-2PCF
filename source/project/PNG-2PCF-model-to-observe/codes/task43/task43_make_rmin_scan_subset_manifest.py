#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""冻结 Task43 rmin-scan 已完成 phase 的前 N 个子集 manifest。

执行逻辑大纲：
1. 读取正式 25-phase manifest，但只选择严格连续的 ``ph000..ph(N-1)``。
2. 验证每个被选 phase 的 32-bin FKP 2PCF 已经完整落盘；不查看或吸收
   manifest 之外稍后完成的 phase。
3. 原子写 JSONL 与 JSON audit，供 preliminary mean/covariance/fit 复现。
4. 只引用既有 catalog/measurement，不复制、不移动任何大型文件。
"""

from __future__ import annotations

import argparse
import json
import os
import tempfile
from pathlib import Path
from typing import Any

import numpy as np

from task43_fkp_zeff import path_with_weight_tag


ROOT = Path("/pscratch/sd/l/lzy/PNG-2PCF-model-to-observe")
SCAN = ROOT / "outputs/task43_outputs/rmin_scan"
DEFAULT_SOURCE = SCAN / "manifests/task43_rmin_scan_mmin1p4e13_x25.jsonl"
DEFAULT_OUTPUT = SCAN / "preliminary_x12/manifests/task43_rmin_scan_mmin1p4e13_x12.jsonl"


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    """读取非空 JSONL 行。"""

    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def atomic_text(path: Path, value: str) -> None:
    """在目标目录内原子写入小型文本。"""

    path.parent.mkdir(parents=True, exist_ok=True)
    handle, temporary = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(handle, "w", encoding="utf-8") as stream:
            stream.write(value)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    except Exception:
        Path(temporary).unlink(missing_ok=True)
        raise


def main() -> None:
    """验证并冻结连续 phase 子集。"""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--nreal", type=int, default=12)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    if args.nreal < 1 or args.nreal > 25:
        raise ValueError("--nreal 必须在 1..25")
    if not args.source.is_file():
        raise FileNotFoundError(args.source)
    if args.output.exists() and not args.force:
        raise FileExistsError(f"subset manifest 已存在：{args.output}")

    rows = read_jsonl(args.source)
    if len(rows) != 25:
        raise RuntimeError(f"正式 manifest 不是 25 行：{len(rows)}")
    subset = rows[: args.nreal]
    expected_phases = [f"ph{index:03d}" for index in range(args.nreal)]
    observed_phases = [str(row.get("phase")) for row in subset]
    if observed_phases != expected_phases:
        raise RuntimeError(f"subset phase 不连续：{observed_phases}")

    expected_edges = np.arange(30.0, 351.0, 10.0)
    measurements: list[str] = []
    for row in subset:
        path = path_with_weight_tag(Path(str(row["xi_path"])), p0=10000.0, output_tag="fkpP010000")
        if not path.is_file() or path.stat().st_size <= 0:
            raise FileNotFoundError(path)
        with np.load(path, allow_pickle=False) as data:
            edges = np.asarray(data["s_edges"], dtype="f8")
            xi = np.asarray(data["xi0"], dtype="f8")
            phase = str(np.asarray(data["phase"]).item())
        if not np.array_equal(edges, expected_edges) or xi.shape != (32,) or not np.all(np.isfinite(xi)):
            raise RuntimeError(f"subset measurement contract 错误：{path}")
        if phase != str(row["phase"]):
            raise RuntimeError(f"measurement/manifest phase 不匹配：{path}")
        measurements.append(str(path))

    text = "".join(json.dumps(row, sort_keys=True) + "\n" for row in subset)
    atomic_text(args.output, text)
    audit = {
        "status": "done",
        "task": "task43_make_rmin_scan_subset_manifest",
        "scope": "frozen preliminary subset; not final x25 product",
        "source_manifest": str(args.source),
        "output_manifest": str(args.output),
        "nreal": args.nreal,
        "phases": expected_phases,
        "measurements": measurements,
        "selection": "first N ordered phases, frozen at manifest creation",
    }
    atomic_text(args.output.with_suffix(".json"), json.dumps(audit, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"status": "done", "nreal": args.nreal, "phases": expected_phases}, indent=2))


if __name__ == "__main__":
    main()
