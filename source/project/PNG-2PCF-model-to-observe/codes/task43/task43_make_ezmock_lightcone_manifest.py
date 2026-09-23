#!/usr/bin/env python3
"""建立 Task43 十个 fixed-amplitude EZmock lightcone 的冻结 manifest。

执行大纲
--------
1. 从共享常量构造 seeds=431001..431010 的十行 manifest。
2. 每行记录生成参数、lightcone 几何、matching-random 策略和所有路径。
3. 同时写一个便于人工/机器审计的 JSON sidecar。
"""

from __future__ import annotations

from datetime import datetime, timezone

from task43_ezmock_lightcone_common import (
    ATTACH_PARTICLE,
    AUDIT_MANIFEST,
    BAO_ENHANCE,
    BOX_SIZE,
    DK,
    FIX_AMPLITUDE,
    INVERT_PHASE,
    KMAX,
    KMIN,
    MANIFEST,
    NGRID,
    NREAL,
    NTRACER,
    P0,
    PDF_BASE,
    PHASES,
    PK_INTERP_LOG,
    PK_MESH_PAD,
    PK_MESHSIZE,
    RAND_GENERATOR,
    RANDOM_MULTIPLIER,
    REDSHIFT,
    RHO_C,
    RHO_EXP,
    SEEDS,
    SIGMA_V,
    ZMAX,
    ZMIN,
    ensure_dirs,
    row_paths,
    write_json,
    write_jsonl,
)


def build_rows() -> list[dict[str, object]]:
    """构造十个 realization 的 manifest 行。"""
    rows: list[dict[str, object]] = []
    for index, (phase, seed) in enumerate(zip(PHASES, SEEDS, strict=True)):
        rows.append(
            {
                "task": "task43_ezmock_lightcone_fixedamp_validation",
                "experiment": "mean_clustering_and_measurement_pipeline_validation",
                "phase": phase,
                "phase_index": 100 + index,
                "validation_index": index,
                "seed": int(seed),
                "sim_name": f"EZmock_fixedamp_seed{seed}",
                "boxsize": BOX_SIZE,
                "ngrid": NGRID,
                "ntracer": NTRACER,
                "redshift_snapshot": REDSHIFT,
                "space_mode": "real",
                "fix_amplitude": FIX_AMPLITUDE,
                "attach_particle": ATTACH_PARTICLE,
                "rho_c": RHO_C,
                "rho_exp": RHO_EXP,
                "pdf_base": PDF_BASE,
                "sigma_v": SIGMA_V,
                "rand_generator": RAND_GENERATOR,
                "pk_interp_log": PK_INTERP_LOG,
                "invert_phase": INVERT_PHASE,
                "bao_enhance": BAO_ENHANCE,
                "zmin": ZMIN,
                "zmax": ZMAX,
                "observer": [0.0, 0.0, 0.0],
                "geometry": "positive_octant_radial_shell_from_box_corner",
                "selection_mode": "geometric_shell_no_redshift_evolution",
                "selection_tag": "mmin1p4e13_density_calibrated_rawbox",
                "mass_threshold_hmsun": None,
                "target_count": None,
                "random_multiplier": RANDOM_MULTIPLIER,
                "random_radial_policy": "data_redshift_resample",
                "fkp_p0": P0,
                "pk_meshsize": PK_MESHSIZE,
                "pk_mesh_pad": PK_MESH_PAD,
                "pk_kmin": KMIN,
                "pk_kmax": KMAX,
                "pk_dk": DK,
                **row_paths(phase, seed),
            }
        )
    return rows


def main() -> None:
    """写 manifest，并执行不可变口径的基本断言。"""
    ensure_dirs()
    rows = build_rows()
    if len(rows) != NREAL or len({row["seed"] for row in rows}) != NREAL:
        raise RuntimeError("EZmock validation manifest must contain 10 unique seeds")
    if not all(row["fix_amplitude"] is True and row["attach_particle"] is True for row in rows):
        raise RuntimeError("validation contract requires FIX_AMPLITUDE=T and ATTACH_PARTICLE=T")
    write_jsonl(MANIFEST, rows)
    write_json(
        AUDIT_MANIFEST,
        {
            "task": "task43_make_ezmock_lightcone_manifest",
            "status": "done",
            "created_utc": datetime.now(timezone.utc).isoformat(),
            "manifest": str(MANIFEST),
            "nreal": len(rows),
            "seeds": [row["seed"] for row in rows],
            "phases": [row["phase"] for row in rows],
            "fixed_amplitude": True,
            "fixed_amplitude_scope": "validation only; covariance production must use a separate F workflow",
            "attach_particle": True,
            "parameters": {
                "rho_c": RHO_C,
                "rho_exp": RHO_EXP,
                "pdf_base": PDF_BASE,
                "sigma_v": SIGMA_V,
                "boxsize": BOX_SIZE,
                "ngrid": NGRID,
                "ntracer": NTRACER,
                "redshift": REDSHIFT,
            },
        },
    )
    print(f"[write] {MANIFEST}")
    print(f"[write] {AUDIT_MANIFEST}")


if __name__ == "__main__":
    main()

