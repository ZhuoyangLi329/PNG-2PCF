#!/usr/bin/env python3
"""Task 4.3.2 box-safe lightcone：从 wide(0.4<zobs<1.1) 原始 catalog 切出 0.4<zobs<0.8 并生成新 manifest。

大纲（本脚本做什么、为什么）：
1. 背景：wide 0.4-1.1 lightcone 的高红移端（chi>2000 Mpc/h, z>0.834）超出了 AbacusSummit
   母盒 L=2000 的单盒几何，catalog 在轴向角落区域不完整（data 在高 z 的角覆盖只有 random
   期望的 ~30%），导致 xi0/P0 出现 ~0.05 的假大尺度平台、拟合塌缩到 fNL=500 等先验边界。
2. 方案：保守取 zmax=0.8（cosmoprimo AbacusSummit(0) 给 chi(0.8)=1936.43 Mpc/h，距离
   2000 边界还有 ~64 Mpc/h 余量），把已验证完整的 z<0.8 部分切出来作为 box-safe 样本；
   randoms/fkp/xi 全部用官方脚本基于切割后的 catalog 重建，保证 25 个 random block
   每个 == ndata 的严格约束。
3. 执行逻辑：
   main()
     ├── read_jsonl(wide manifest) 读入 25 个 phase 的行
     ├── cut_one_phase(row) 对每个 phase：
     │     ├── 读 wide 原始 catalog npz（含全部 provenance 列）
     │     ├── 掩膜 (Z > zmin) & (Z < zmax)（f64 比较吸收 f32 边界）
     │     ├── 原子写 cut npz + json 元数据（记录来源路径/sha256/前后行数）
     │     └── 打印该 phase 的前后计数
     └── write_boxsafe_manifest() 由 wide 行做字符串手术生成新 manifest：
           - lightcone_wide_zobs0p4_1p1 → lightcone_boxsafe_zobs0p4_0p8
           - _zobs0p4_1p1 → _zobs0p4_0p8
           - zmin_observed/zmax_observed 更新为 0.4/0.8
           - zero-velocity bridge 路径置 None（本线不需要）
           - 附加 provenance_boxsafe 说明块
4. 输入：--manifest wide jsonl；输出：cut catalogs + boxsafe jsonl manifest。
5. 注意：只做数据切割，不重建 ASDF；不覆盖任何 wide/narrow/Task44 产物。
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np

from task43_rsd_common import PHASES, atomic_savez, atomic_write_json, sha256_file

# 保守的 box-safe 红移上限：chi(0.8)=1936.43 < 2000（母盒半宽），留 ~64 Mpc/h 余量
BOXSAFE_ZMAX_DEFAULT = 0.8
# 与 wide 相同的红移下限
BOXSAFE_ZMIN = 0.4


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    """读 jsonl 文件，每行一个 json dict；跳过空行。"""
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def retag(value: str) -> str:
    """把任意 wide 命名片段转换成 boxsafe 命名：目录前缀和 zobs 标签两处替换。"""
    return (
        value.replace("lightcone_wide_zobs0p4_1p1", "lightcone_boxsafe_zobs0p4_0p8")
        .replace("zobs0p4_1p1", "zobs0p4_0p8")
    )


def cut_one_phase(row: dict[str, Any], zmin: float, zmax: float) -> dict[str, Any]:
    """切单个 phase 的 wide 原始 catalog。

    参数：
        row  —— wide manifest 中该 phase 的行（提供 lightcone_catalog_path 等路径）
        zmin/zmax —— 观测红移开区间的下/上限
    返回：
        dict：ndata_before/ndata_after/输出路径/来源 sha256 等审计信息
    """
    source_path = Path(row["lightcone_catalog_path"])
    output_path = Path(retag(str(row["lightcone_catalog_path"])))
    metadata_path = Path(retag(str(row["lightcone_metadata_path"])))
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # 幂等复用：如果切割产物已存在且 json 校验通过（sha 匹配 + 红移窗口一致），
    # 直接复用，不重写文件（npz 压缩字节不保证可复现，重切会破坏下游 provenance）
    if output_path.is_file() and metadata_path.is_file():
        existing = json.loads(metadata_path.read_text(encoding="utf-8"))
        if (
            existing.get("status") == "pass"
            and existing.get("zmin_observed") == zmin
            and existing.get("zmax_observed") == zmax
            and existing.get("output_sha256") == sha256_file(output_path)
        ):
            print(f"[skip] validated cut catalog exists for {row['phase']}")
            return {
                "phase": row["phase"],
                "ndata_before": int(existing["ndata_before"]),
                "ndata_after": int(existing["ndata_after"]),
                "output": str(output_path),
            }

    with np.load(source_path, allow_pickle=False) as payload:
        arrays = {key: payload[key] for key in payload.files}

    redshift = np.asarray(arrays["Z"], dtype="f8")
    keep = (redshift > zmin) & (redshift < zmax)

    cut_arrays = {key: value[keep] if (value.ndim == 1 and value.shape[:1] == redshift.shape) else value for key, value in arrays.items()}
    ndata_before = int(redshift.size)
    ndata_after = int(np.count_nonzero(keep))
    if ndata_after == 0:
        raise RuntimeError(f"empty cut catalog for {row['phase']}")

    cut_ra = np.asarray(cut_arrays["RA"], dtype="f8")
    cut_dec = np.asarray(cut_arrays["DEC"], dtype="f8")
    cut_x = np.asarray(cut_arrays["X"], dtype="f8")
    cut_y = np.asarray(cut_arrays["Y"], dtype="f8")
    cut_zcart = np.asarray(cut_arrays["Zcart"], dtype="f8")
    # 对切割后的行重新验证正八分体几何（RA/DEC 在 [0,90]、笛卡尔坐标非负），
    # 通过后才允许下游 random 生成器消费（它的门禁检查 positive_octant_gate 字段）。
    # 注意边界用闭区间：个别对象因浮点舍入恰好落在 RA=90/X=0 上（如 ph006），
    # 与 wide 原始构建的容许口径一致
    positive_octant_gate = bool(
        np.all(cut_ra >= 0.0) and np.all(cut_ra <= 90.0)
        and np.all(cut_dec >= 0.0) and np.all(cut_dec <= 90.0)
        and np.all(cut_x >= 0.0) and np.all(cut_y >= 0.0) and np.all(cut_zcart >= 0.0)
        and np.all(np.isfinite(cut_x)) and np.all(np.isfinite(cut_y)) and np.all(np.isfinite(cut_zcart))
    )

    atomic_savez(output_path, **cut_arrays)

    # 继承源 catalog 的元数据（保留几何门禁/宇宙学等字段），再覆盖切割相关字段
    source_metadata_path = Path(str(row["lightcone_metadata_path"]))
    base_metadata = json.loads(source_metadata_path.read_text(encoding="utf-8"))
    # 刷新随切割变化的字段：z_observed 范围（random 生成器的严格红移门禁会读取）、
    # selection 描述字符串、行数计数
    cut_z = redshift[keep]
    z_observed = dict(base_metadata.get("z_observed", {}))
    z_observed["min"] = float(cut_z.min())
    z_observed["max"] = float(cut_z.max())
    metadata = {
        **base_metadata,
        "task": "task43_build_rsd_boxsafe_catalogs",
        "phase": row["phase"],
        "sim_name": row["sim_name"],
        "status": "pass",
        "source_catalog_path": str(source_path),
        "source_catalog_sha256": sha256_file(source_path),
        "output_path": str(output_path),
        "output_sha256": sha256_file(output_path),
        "cut_definition": f"observed Z in open interval ({zmin}, {zmax}); float64 comparison on stored Z",
        "zmin_observed": zmin,
        "zmax_observed": zmax,
        "ndata_before": ndata_before,
        "ndata_after": ndata_after,
        "n_selected": ndata_after,
        "selection": f"{zmin} < z_observed < {zmax}",
        "z_observed": z_observed,
        "positive_octant_gate": positive_octant_gate,
        "boxsafe_rationale": (
            "wide 0.4-1.1 catalog is geometrically incomplete beyond mother-box chi=2000 "
            "(z=0.83403); conservative zmax=0.8 with chi(0.8)=1936.43 Mpc/h keeps ~64 Mpc/h margin"
        ),
    }
    atomic_write_json(metadata_path, metadata)
    return {
        "phase": row["phase"],
        "ndata_before": ndata_before,
        "ndata_after": ndata_after,
        "output": str(output_path),
    }


def write_boxsafe_manifest(
    rows: list[dict[str, Any]],
    source_manifest: Path,
    zmin: float,
    zmax: float,
    output: Path,
    cut_records: list[dict[str, Any]],
) -> None:
    """由 wide manifest 行生成 boxsafe manifest。

    做法：对每行做浅拷贝，路径字符串经 retag() 替换，再覆盖若干语义字段；
    zero-velocity bridge 路径置 None；shells 只保留 <= z0.800 的条目。
    """
    new_rows: list[dict[str, Any]] = []
    for row in rows:
        new_row: dict[str, Any] = {}
        for key, value in row.items():
            if isinstance(value, str):
                new_row[key] = retag(value)
            elif isinstance(value, list):
                new_row[key] = [retag(v) if isinstance(v, str) else v for v in value]
            else:
                new_row[key] = value
        new_row["analysis_scope"] = "task43.2_boxsafe_lightcone_within_mother_box"
        new_row["analysis_tag"] = "zobs0p4_0p8"
        new_row["task"] = "task43_rsd_validation_lightcone_boxsafe"
        new_row["zmin_observed"] = zmin
        new_row["zmax_observed"] = zmax
        new_row["immutability"] = "independent box-safe products; never overwrite narrow/wide/Task43 or Task44 products"
        # zero-velocity bridge 属于 wide 线的审计产物，box-safe 线不需要
        new_row["lightcone_zero_velocity_catalog_path"] = None
        new_row["lightcone_zero_velocity_metadata_path"] = None
        # 只保留 z<=0.8 的壳层来源（信息性字段）
        shells = new_row.get("lightcone_shells") or []
        if isinstance(shells, list):
            kept = []
            for shell, source in zip(shells, new_row.get("lightcone_source_paths") or []):
                try:
                    z_shell = float(str(shell).replace("z", ""))
                except ValueError:
                    continue
                if z_shell <= zmax + 1e-9:
                    kept.append((shell, source))
            if kept:
                new_row["lightcone_shells"] = [item[0] for item in kept]
                new_row["lightcone_source_paths"] = [item[1] for item in kept if item[1] is not None]
        new_row["provenance_boxsafe"] = {
            "derived_from_manifest": str(source_manifest),
            "derived_from_tag": "zobs0p4_1p1",
            "reason": (
                "wide catalog incomplete where chi>2000 Mpc/h (z>0.834) requires box replicas; "
                "cut to conservative zmax=0.8 (chi=1936.43) verified complete by angular-cell tests"
            ),
            "cut_records": {record["phase"]: record for record in cut_records},
        }
        new_rows.append(new_row)

    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as handle:
        for new_row in new_rows:
            handle.write(json.dumps(new_row, sort_keys=True) + "\n")
    print(f"[manifest] wrote {len(new_rows)} rows -> {output}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Build box-safe 0.4<zobs<0.8 catalogs and manifest from the wide run")
    parser.add_argument("--manifest", type=Path, required=True, help="wide zobs0p4_1p1 jsonl manifest")
    parser.add_argument("--zmax", type=float, default=BOXSAFE_ZMAX_DEFAULT)
    parser.add_argument("--phases", default=",".join(PHASES), help="comma-separated subset, e.g. ph000 for smoke")
    parser.add_argument("--manifest-out", type=Path, default=None, help="default: manifests/task43_rsd_validation_lightcone_boxsafe_zobs0p4_0p8_x25.jsonl")
    args = parser.parse_args()

    zmin = BOXSAFE_ZMIN
    requested = [token.strip() for token in str(args.phases).split(",") if token.strip()]
    rows = read_jsonl(args.manifest)
    selected = [row for row in rows if row["phase"] in requested]
    if len(selected) != len(requested):
        raise ValueError(f"manifest missing some requested phases: {[p for p in requested if p not in {r['phase'] for r in rows}]}")

    cut_records: list[dict[str, Any]] = []
    for row in selected:
        record = cut_one_phase(row, zmin, args.zmax)
        cut_records.append(record)
        print(f"[cut] {row['phase']}: {record['ndata_before']} -> {record['ndata_after']} rows")

    manifest_out = args.manifest_out
    if manifest_out is None:
        manifest_out = args.manifest.parent / "task43_rsd_validation_lightcone_boxsafe_zobs0p4_0p8_x25.jsonl"
    # manifest 只在全部 25 个 phase 都切割完成时才写完整版；smoke 子集也允许写（供试跑），但会少行
    write_boxsafe_manifest(selected, args.manifest, zmin, args.zmax, manifest_out, cut_records)


if __name__ == "__main__":
    main()
