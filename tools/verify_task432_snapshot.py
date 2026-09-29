#!/usr/bin/env python3
"""只读核验9.30归档：哈希/语法/依赖 -> 可选NPZ算子与链一致性。

本工具不导入科学pipeline、不重跑模型、不写入任何文件。
默认仅依赖标准库；--arrays要求调用环境已经提供NumPy。
"""
from __future__ import annotations

import argparse
import ast
import hashlib
import json
from pathlib import Path
import sys

PREFIX = Path("source/project/PNG-2PCF-model-to-observe")
MEETING = PREFIX / "9.30meeting/task432_velocileptors_fullric"
OPERATOR = PREFIX / (
    "outputs/task43_outputs/rsd_validation/task432_model_repair/"
    "velocileptors_fullric_v1/operator"
)
MANIFEST = Path("provenance/task432_20260929_sources.json")


def require(condition: bool, message: str) -> None:
    """条件不满足时终止；使用显式检查，避免python -O关闭assert。"""
    if not condition:
        raise ValueError(message)


def read_json(path: Path):
    """以UTF-8读JSON，返回解析对象；不接受缺失或损坏的输入。"""
    return json.loads(path.read_text(encoding="utf-8"))


def member(root: Path, relative: str | Path) -> Path:
    """将清单路径约束在仓库内，返回存在的普通文件路径。"""
    path = (root / relative).resolve()
    require(path.is_relative_to(root), f"Path outside repository: {relative}")
    require(path.is_file(), f"Missing file: {relative}")
    return path


def verify_sources(root: Path) -> dict:
    """检查19个源产物及既有直接依赖；返回校验计数和科学状态。"""
    manifest = read_json(member(root, MANIFEST))
    require(manifest["schema_version"] == 1, "Unsupported manifest schema")
    require(manifest["scientific_status"] == "diagnostic_only", "Unexpected scientific status")
    seen: set[str] = set()
    modules: set[str] = set()
    trees = []
    count_json = 0
    for row in manifest["files"]:
        require(row["path"] not in seen, f"Duplicate path: {row['path']}")
        seen.add(row["path"])
        path = member(root, row["path"])
        content = path.read_bytes()
        require(len(content) == row["size_bytes"], f"Size mismatch: {row['path']}")
        require(hashlib.sha256(content).hexdigest() == row["sha256"], f"SHA256 mismatch: {row['path']}")
        if path.suffix == ".py":
            trees.append((path.name, ast.parse(content, filename=row["path"])))
            modules.add(path.stem)
        elif path.suffix == ".json":
            read_json(path)
            count_json += 1
    for row in manifest["existing_direct_dependencies"]:
        path = member(root, row["path"])
        require(hashlib.sha256(path.read_bytes()).hexdigest() == row["sha256"], f"Dependency changed: {row['path']}")
        modules.add(path.stem)
    for filename, tree in trees:
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom):
                names = [node.module] if node.module else []
            elif isinstance(node, ast.Import):
                names = [alias.name for alias in node.names]
            else:
                names = []
            for name in names:
                base = name.split(".")[0]
                if base.startswith("task"):
                    require(base in modules, f"Unrecorded direct project dependency: {filename}: {base}")
    direct = read_json(member(root, MEETING / "diagnostics/direct_map.json"))
    require(direct["status"] == "diagnostic_only", "Direct MAP must remain diagnostic")
    expected = {"xi_only", "joint_lambda0", "joint_lambda0.5", "joint_lambda1"}
    require(set(direct["results"]) == expected, "Incomplete direct MAP variants")
    smoke = read_json(member(root, MEETING / "diagnostics/smoke.json"))
    require(smoke["formal_status"] == "diagnostic_only", "Smoke must remain diagnostic")
    return {"source_files": len(seen), "python_sources": len(trees), "json_sources": count_json,
            "existing_direct_dependencies": len(manifest["existing_direct_dependencies"]),
            "scientific_status": "diagnostic_only"}


def verify_arrays(root: Path) -> dict:
    """核对全部数组结构，独立重组RIC投影，并从原链重算报告分位数。"""
    try:
        import numpy as np
    except ImportError as error:
        raise ValueError("--arrays needs NumPy in the selected Python environment") from error
    inventory = read_json(member(root, "provenance/task432_20260929_array_index.json"))
    for row in inventory["files"]:
        with np.load(member(root, row["path"]), allow_pickle=False) as data:
            require(set(data.files) == set(row["arrays"]), f"Array keys changed: {row['path']}")
            for name, spec in row["arrays"].items():
                value = data[name]
                require(list(value.shape) == spec["shape"], f"Shape mismatch: {row['path']}: {name}")
                require(str(value.dtype) == spec["dtype"], f"Dtype mismatch: {row['path']}: {name}")
                require(bool(np.all(np.isfinite(value))), f"Nonfinite array: {row['path']}: {name}")
    grid_results = []
    for folder in (OPERATOR, OPERATOR / "local"):
        metadata = read_json(member(root, folder / "operator_manifest.json"))
        path = member(root, folder / "velo_fullric_grid.npz")
        require(hashlib.sha256(path.read_bytes()).hexdigest() == metadata["output_sha256"], f"Compiler hash mismatch: {folder}")
        with np.load(path, allow_pickle=False) as data:
            f, b, edges = data["f_grid"], data["b_grid"], data["separation_edges"]
            require(bool(np.all(np.diff(f) > 0) and np.all(np.diff(b) > 0) and np.all(np.diff(edges) > 0)), f"Nonmonotonic grid: {folder}")
            require(np.array_equal(f, metadata["f_grid"]) and np.array_equal(b, metadata["b_grid"]), f"Grid axes disagree with manifest: {folder}")
            require(np.array_equal(edges, metadata["shell_edges"]), f"Shell edges disagree: {folder}")
            require(list(data["raw_xi"].shape) == metadata["grid_shape"], f"Compiled shape mismatch: {folder}")
            require(data["raw_xi"].shape == (len(f), len(b), 52), f"Unexpected output dimension: {folder}")
            require(data["poles"].shape == (len(f), len(b), 3, len(edges)-1), f"Unexpected pole dimensions: {folder}")
            require(data["xi_response"].shape == (52, 3, len(edges)-1), f"Unexpected response dimensions: {folder}")
            require(data["fixed_shot"].shape == data["sn0_ric_response"].shape == (52,), f"Unexpected stochastic response: {folder}")
            require(int(data["nint"]) == metadata["nint"], f"Integration metadata mismatch: {folder}")
            rebuilt = np.einsum("ild,abld->abi", data["xi_response"], data["poles"])
            delta = float(np.max(np.abs(rebuilt-data["ric_clustering"])))
            require(np.allclose(rebuilt, data["ric_clustering"], rtol=1e-11, atol=1e-13), f"RIC recomposition failed: {folder}: {delta}")
            grid_results.append({"path": str(folder), "shape": list(data["raw_xi"].shape), "ric_recomposition_max_abs": delta})
    summaries = read_json(member(root, MEETING / "diagnostic_chain_summary.json"))
    chain_results = {}
    with np.load(member(root, MEETING / "diagnostic_chains.npz"), allow_pickle=False) as data:
        for variant, key in (("diag_xi02", "xi_chain"), ("diag_joint", "joint_chain")):
            summary = summaries[variant]
            samples = data[key]
            expected_count = summary["nwalkers"] * (summary["nsteps"]-summary["burnin"])
            require(samples.shape == (expected_count, len(summary["parameter_names"])), f"Chain shape mismatch: {variant}")
            quantiles = np.percentile(samples, [16, 50, 84], axis=0)
            target = np.asarray(summary["summary"]["q16_q50_q84"])
            require(np.allclose(quantiles, target, rtol=1e-12, atol=1e-12), f"Chain/summary quantiles disagree: {variant}")
            best = np.asarray(summary["summary"]["map_theta"])
            require(bool(np.any(np.all(samples == best, axis=1))), f"Reported sample MAP absent from chain: {variant}")
            chain_results[variant] = {"samples": len(samples), "quantile_max_abs": float(np.max(np.abs(quantiles-target)))}
    return {"array_files": len(inventory["files"]), "operator_grids": grid_results, "diagnostic_chains": chain_results}


def main() -> int:
    """解析根目录与可选数组检查开关，以JSON返回可复查结果或清晰失败。"""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1], help="Repository root (default: parent of tools/)")
    parser.add_argument("--arrays", action="store_true", help="Also check archived NPZ arrays using NumPy")
    args = parser.parse_args()
    try:
        root = args.root.resolve()
        report = verify_sources(root)
        if args.arrays:
            report.update(verify_arrays(root))
        print(json.dumps({"status": "pass", "check_scope": "archive_consistency_only", **report}, indent=2))
    except (ValueError, OSError, KeyError, TypeError, SyntaxError) as error:
        print(json.dumps({"status": "failed", "error": str(error)}), file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
