#!/usr/bin/env python3
"""生成一个 EZmock rawbox，并从盒子角点切出 Task43 正八分体 lightcone。

执行大纲
--------
1. 读取 manifest 行并写出完全展开的 EZmock 配置文件。
2. 在不超过八核的 CPU affinity 下运行 EZmock，输出六列 ASCII catalog。
3. 把位置归一到 [0,L)，以 (0,0,0) 为 observer，截取 0.6<z<0.8
   对应的正八分体径向壳层。
4. 用 AbacusSummit_base_c000 距离关系把半径反解为几何红移，保存与
   Task43 data/random/P(k) loader 兼容的 NPZ catalog。
5. 写出几何、数量、n(z)、参数和输入输出 provenance 审计 JSON。

这里的几何红移只用于 survey 坐标和 FKP 权重，不引入 tracer 演化或 RSD。
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import time
from pathlib import Path
from typing import Any

import numpy as np

from task43_ezmock_lightcone_common import (
    ABACUS_FKP_SUMMARY,
    ATTACH_PARTICLE,
    BAO_ENHANCE,
    BOX_SIZE,
    EZMOCK_BINARY,
    FIX_AMPLITUDE,
    INVERT_PHASE,
    LINEAR_PK,
    MANIFEST,
    NGRID,
    NTRACER,
    OMEGA_M_NON_NEUTRINO,
    OMEGA_NU,
    PDF_BASE,
    PK_INTERP_LOG,
    RAND_GENERATOR,
    REDSHIFT,
    RHO_C,
    RHO_EXP,
    SIGMA_V,
    ZMAX,
    ZMIN,
    atomic_savez,
    read_jsonl,
    write_json,
)
from task43_rawbox_ezmock_common import set_cpu_affinity


def parse_args() -> argparse.Namespace:
    """解析 manifest 选择、线程数和覆盖策略。"""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=MANIFEST)
    parser.add_argument("--index", type=int, required=True)
    parser.add_argument("--threads", type=int, default=8)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def bool_token(value: bool) -> str:
    """把 Python bool 转为 EZmock 的 T/F token。"""
    return "T" if bool(value) else "F"


def make_ezmock_config(row: dict[str, Any]) -> str:
    """从冻结 manifest 行构造 EZmock 配置文本。"""
    if not bool(row["fix_amplitude"]) or not bool(row["attach_particle"]):
        raise ValueError("current validation requires FIX_AMPLITUDE=T and ATTACH_PARTICLE=T")
    return f"""BOX_SIZE = {BOX_SIZE:.17g}
NUM_GRID = {NGRID}
NUM_TRACER = {NTRACER}
LINEAR_PK = '{LINEAR_PK}'
REDSHIFT_PK = {REDSHIFT:.17g}
PK_INTERP_LOG = {bool_token(PK_INTERP_LOG)}
RAND_GENERATOR = {RAND_GENERATOR}
RAND_SEED = {int(row['seed'])}
FIX_AMPLITUDE = {bool_token(FIX_AMPLITUDE)}
INVERT_PHASE = {bool_token(INVERT_PHASE)}
OMEGA_M = {OMEGA_M_NON_NEUTRINO:.17g}
OMEGA_NU = {OMEGA_NU:.17g}
DE_EOS_W = -1
REDSHIFT = {REDSHIFT:.17g}
BAO_ENHANCE = {BAO_ENHANCE:.17g}
RHO_CRITICAL = {RHO_C:.17g}
RHO_EXP = {RHO_EXP:.17g}
PDF_BASE = {PDF_BASE:.17g}
SIGMA_VELOCITY = {SIGMA_V:.17g}
ATTACH_PARTICLE = {bool_token(ATTACH_PARTICLE)}
OUTPUT = '{row['rawbox_catalog_path']}'
OUTPUT_FORMAT = 0
OUTPUT_HEADER = T
OVERWRITE = 2
VERBOSE = T
"""


def distance_table() -> tuple[np.ndarray, np.ndarray]:
    """返回 AbacusSummit(0) 的稠密 z--chi 表，距离单位为 Mpc/h。"""
    from cosmoprimo.fiducial import AbacusSummit

    zgrid = np.linspace(ZMIN, ZMAX, 100_001, dtype="f8")
    cosmo = AbacusSummit(0)
    chigrid = np.asarray(cosmo.comoving_radial_distance(zgrid), dtype="f8")
    if not np.all(np.diff(chigrid) > 0.0):
        raise RuntimeError("AbacusSummit comoving distance is not strictly increasing")
    return zgrid, chigrid


def load_positions(path: Path) -> np.ndarray:
    """读取 EZmock 六列 ASCII 中前三列位置，并归一到 [0,L)。"""
    position = np.loadtxt(path, comments="#", usecols=(0, 1, 2), dtype="f8")
    if position.ndim != 2 or position.shape[1] != 3:
        raise ValueError(f"bad EZmock position shape in {path}: {position.shape}")
    return np.mod(position, BOX_SIZE)


def build_lightcone(row: dict[str, Any], *, overwrite: bool) -> dict[str, Any]:
    """运行 EZmock 并构造一个 Task43 几何 lightcone catalog。"""
    config_path = Path(row["ezmock_config_path"])
    rawbox_path = Path(row["rawbox_catalog_path"])
    output_path = Path(row["halo_catalog_path"])
    metadata_path = Path(row["halo_metadata_path"])
    log_path = Path(row["ezmock_log_path"])
    for path in (config_path, rawbox_path, output_path, metadata_path, log_path):
        path.parent.mkdir(parents=True, exist_ok=True)

    if output_path.exists() and metadata_path.exists() and not overwrite:
        print(f"[skip] {output_path}")
        return json.loads(metadata_path.read_text(encoding="utf-8"))

    if not EZMOCK_BINARY.is_file() or not LINEAR_PK.is_file() or not ABACUS_FKP_SUMMARY.is_file():
        raise FileNotFoundError("missing EZmock binary, linear P(k), or frozen Abacus FKP summary")

    config_path.write_text(make_ezmock_config(row), encoding="utf-8")
    env = os.environ.copy()
    env.update(
        {
            "OMP_NUM_THREADS": env.get("OMP_NUM_THREADS", "8"),
            "OMP_DYNAMIC": "FALSE",
            "OMP_PROC_BIND": "close",
            "MKL_NUM_THREADS": "1",
            "OPENBLAS_NUM_THREADS": "1",
            "NUMEXPR_NUM_THREADS": "1",
        }
    )
    started = time.perf_counter()
    ez_started = time.perf_counter()
    with log_path.open("w", encoding="utf-8") as log:
        subprocess.run(
            [str(EZMOCK_BINARY), "-c", str(config_path)],
            check=True,
            stdout=log,
            stderr=subprocess.STDOUT,
            env=env,
        )
    ezmock_sec = time.perf_counter() - ez_started

    position = load_positions(rawbox_path)
    zgrid, chigrid = distance_table()
    chi_min, chi_max = float(chigrid[0]), float(chigrid[-1])
    radius = np.linalg.norm(position, axis=1)
    keep = (radius > chi_min) & (radius < chi_max)
    selected = np.asarray(position[keep], dtype="f4")
    selected_radius = np.asarray(radius[keep], dtype="f8")
    if selected.shape[0] < 100_000:
        raise RuntimeError(f"unexpectedly sparse EZmock lightcone: {selected.shape[0]}")
    redshift = np.interp(selected_radius, chigrid, zgrid).astype("f4")
    ra = (np.degrees(np.arctan2(selected[:, 1], selected[:, 0])) % 360.0).astype("f8")
    dec = np.degrees(
        np.arcsin(np.clip(selected[:, 2] / selected_radius, -1.0, 1.0))
    ).astype("f8")
    direction = selected.astype("f8") / selected_radius[:, None]

    # 用与 4.3 FKP 表完全相同的 z bins 记录 n(z) 诊断。
    with np.load(ABACUS_FKP_SUMMARY, allow_pickle=False) as fkp:
        z_edges = np.asarray(fkp["z_edges"], dtype="f8")
        volume_shell = np.asarray(fkp["volume_shell"], dtype="f8")
    z_counts = np.histogram(redshift, bins=z_edges)[0].astype("i8")
    nbar_z = z_counts.astype("f8") / volume_shell

    atomic_savez(
        output_path,
        RA=ra,
        DEC=dec,
        Z=redshift,
        X=selected[:, 0],
        Y=selected[:, 1],
        Zcart=selected[:, 2],
        WEIGHT=np.ones(selected.shape[0], dtype="f4"),
        phase=np.asarray(row["phase"]),
        seed=np.asarray(int(row["seed"]), dtype="i8"),
        realization=np.full(selected.shape[0], int(row["validation_index"]), dtype="i2"),
        observer_origin=np.zeros(3, dtype="f8"),
        snapshot_redshift=np.asarray(REDSHIFT, dtype="f8"),
    )

    radius_from_z = np.interp(redshift.astype("f8"), zgrid, chigrid)
    frac_error = selected_radius / radius_from_z - 1.0
    metadata = {
        "task": "task43_build_ezmock_lightcone",
        "status": "done",
        "phase": row["phase"],
        "seed": int(row["seed"]),
        "sim_name": row["sim_name"],
        "classification": "fixed-amplitude mean/pipeline validation; not covariance production",
        "fix_amplitude": True,
        "attach_particle": True,
        "space_mode": "real",
        "redshift_evolution": False,
        "snapshot_redshift": REDSHIFT,
        "boxsize": BOX_SIZE,
        "ngrid": NGRID,
        "ntracer_requested": NTRACER,
        "nraw_actual": int(position.shape[0]),
        "selected_count": int(selected.shape[0]),
        "selected_fraction": float(selected.shape[0] / position.shape[0]),
        "rawbox_nbar": float(position.shape[0] / BOX_SIZE**3),
        "zmin": ZMIN,
        "zmax": ZMAX,
        "chi_min": chi_min,
        "chi_max": chi_max,
        "observer_origin": [0.0, 0.0, 0.0],
        "observer_policy": "box corner after modulo [0,L)",
        "geometry": "positive octant radial shell; no box tiling",
        "positive_octant_gate": bool(np.all(direction >= -1.0e-12)),
        "coordinate_min": np.min(selected, axis=0),
        "coordinate_max": np.max(selected, axis=0),
        "direction_min": np.min(direction, axis=0),
        "direction_max": np.max(direction, axis=0),
        "ra_range": [float(np.min(ra)), float(np.max(ra))],
        "dec_range": [float(np.min(dec)), float(np.max(dec))],
        "z_range": [float(np.min(redshift)), float(np.max(redshift))],
        "radius_range": [float(np.min(selected_radius)), float(np.max(selected_radius))],
        "coord_check_radius_over_chi_minus_one": {
            "mean": float(np.mean(frac_error)),
            "p99_abs": float(np.quantile(np.abs(frac_error), 0.99)),
            "max_abs": float(np.max(np.abs(frac_error))),
        },
        "z_edges": z_edges,
        "z_counts": z_counts,
        "nbar_z": nbar_z,
        "nbar_volume_weighted": float(np.sum(z_counts) / np.sum(volume_shell)),
        "parameters": {
            "rho_c": RHO_C,
            "rho_exp": RHO_EXP,
            "pdf_base": PDF_BASE,
            "sigma_v": SIGMA_V,
        },
        "paths": {
            "manifest": str(MANIFEST),
            "config": str(config_path),
            "linear_pk": str(LINEAR_PK),
            "rawbox_catalog": str(rawbox_path),
            "lightcone_catalog": str(output_path),
            "log": str(log_path),
        },
        "runtime": {
            "ezmock_sec": float(ezmock_sec),
            "total_sec": float(time.perf_counter() - started),
        },
    }
    if not metadata["positive_octant_gate"]:
        raise RuntimeError("EZmock lightcone failed positive-octant gate")
    if metadata["coord_check_radius_over_chi_minus_one"]["max_abs"] > 2.0e-6:
        raise RuntimeError("EZmock lightcone failed radius/chi(z) interpolation gate")
    write_json(metadata_path, metadata)
    print(
        f"[done] {row['phase']} seed={row['seed']} raw={position.shape[0]} "
        f"lightcone={selected.shape[0]} nbar={metadata['nbar_volume_weighted']:.8e}"
    )
    return metadata


def main() -> None:
    """选择 manifest 中一个 realization 并完成生成与切割。"""
    args = parse_args()
    cpus = set_cpu_affinity(int(args.threads))
    rows = read_jsonl(args.manifest)
    if not 0 <= int(args.index) < len(rows):
        raise IndexError(f"index {args.index} outside manifest of length {len(rows)}")
    print(f"[cpu] affinity={cpus}")
    build_lightcone(rows[int(args.index)], overwrite=bool(args.overwrite))


if __name__ == "__main__":
    main()

