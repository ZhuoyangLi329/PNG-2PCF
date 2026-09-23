#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Task43 joint P(k)+xi0 covariance builder (s50-350, jaxpower, radial-RIC era).

目的
====
为 4.3 实空间的 P(k)+xi0 联合拟合构造内部一致的 45x45 Gaussian
survey-window 联合协方差：

    C_joint = [[C_pp(15x15), C_px(15x30)],
               [C_px^T,      C_xx(30x30)]]

两个 stage
----------
run
    用与已审计 jaxpower xi 协方差（diagnose 脚本, mesh64, 50k data /
    100k random, cic, interlacing 1, bessel+interp window, smooth+fftlog）
    完全相同的 ph000 FKP 配置重新做一次 jaxpower Gaussian covariance；
    唯一科学差异是 k 网格对齐到 P(k) 测量网格（kmin=0.001 而不是
    0.0001），使 C_pp 的 bin 与 payload 的 15 个拟合 bin 严格一致。
    保存细网格 WW/WS/SS、corrected 总矩阵、Hankel 投影 J 与 xi 投影块。

assemble
    从 run 产物 + 已审计 RR-deconv 逆窗口 R^{-1} + P(k) payload 的
    15 个拟合 bin 组装 45x45 联合协方差；执行 PSD gate、两个边缘块
    对已审计协方差的 bridge、cross-block 相关系数统计，写 joint npz
    和 audit json。

预注册决策规则
--------------
mesh64 的 P-block bridge 满足
    sigma_ratio_median in [0.95, 1.05] 且 relative_frobenius < 0.10
时接受 mesh64 为主口径；否则必须再跑 mesh128 做 A/B。

与已审计口径的对应关系
----------------------
- C_xx(新) = nearest_spd(R^{-1} J C_fine J^T R^{-T})，与已审计
  rrdeconv 30x30 的唯一差异来自 kmin 0.0001->0.001 的网格变化；
  bridge 必须量化该差异。
- C_pp(新) 来自 mesh64/cic/子采样 recipe，与已审计 mesh128/tsc/全
  catalog P(k) 协方方的差异同样由 bridge 量化（这就是上面的决策规则）。
- C_px = C_fine[sel15, :] @ J^T @ R^{-T}，xi 侧与 C_xx 完全一致的
  去卷积基；WS/SS 子采样修正与两个边缘块同一套（对 data thinning
  的 1/s、1/s^2 缩放逐部分作用后再求和）。
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Any

for _name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_name, "1")

import numpy as np

PROJECT_ROOT = Path("/pscratch/sd/l/lzy/PNG-2PCF-model-to-observe")
CODE_DIR = PROJECT_ROOT / "codes" / "task43"
for _path in (CODE_DIR,):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

from task43_config import read_jsonl  # noqa: E402
from task43_fkp_zeff import load_fkp_summary  # noqa: E402
from task43_make_jaxpower_lightcone_covariance import (  # noqa: E402
    build_theory_poles,
    covariance_diagnostics,
    infer_mesh_attrs,
    load_positions_weights,
    make_edges,
)
from task43_make_rawbox_z0p725_mmin1p3e13_gaussian_covariance import correlation_from_covariance, nearest_spd  # noqa: E402
from task43_theory_template import COSMOLOGY_CHOICES, DEFAULT_COSMOLOGY  # noqa: E402

# 2026-08 的仓库重组把旧的 outputs/task43_outputs/{summary,manifests,...}
# 移到了 plots/outputs/task43_outputs/；catalog 则在 2026-07-07 的清理中
# 移入 old_doc_codes 归档。这里集中定义当前真实路径与解析规则。
ACTIVE_T43_ROOT = PROJECT_ROOT / "plots" / "outputs" / "task43_outputs"
ARCHIVED_T43_ROOT = (
    PROJECT_ROOT
    / "old_doc_codes"
    / "task4_task44_cleanup_20260707T061844Z"
    / "moved"
    / "outputs"
    / "task43_outputs"
)
JOINT_ROOT = PROJECT_ROOT / "outputs" / "task43_outputs" / "joint_pkxi_s50_350"
JOINT_COV_DIR = JOINT_ROOT / "covariance"

MANIFEST_PATH = ACTIVE_T43_ROOT / "manifests" / "task43_mmin1p4e13_x25.jsonl"
FKP_SUMMARY_PATH = ACTIVE_T43_ROOT / "summary" / "task43_fkp_zeff_mmin1p4e13_x25.npz"
MEAN_XI_PATH = ACTIVE_T43_ROOT / "summary" / "task43_mean_xi_mmin1p4e13_x25_s50_350_ds10_fkpP010000.npz"
PK_PAYLOAD_PATH = (
    ACTIVE_T43_ROOT
    / "pk_lightcone"
    / "summary"
    / "task43_pk_lightcone_mmin1p4e13_x25_fkpP010000_desi_rebin_kmax0p10_payload.npz"
)
PK_COV_MESH128_PATH = (
    ACTIVE_T43_ROOT
    / "pk_lightcone"
    / "covariance"
    / "mmin1p4e13_x25_fkpP010000"
    / "task43_pk_cov_ph000_mesh128_kmax0p300_dk0p002.npz"
)
RRDECONV_PATH = (
    ACTIVE_T43_ROOT
    / "summary"
    / "jaxpower_covariance_lightcone_mmin1p4e13_ph000_fitcov_bessel_interp_smoothfftlog_rrdeconv_mesh64_nran100k_ndata50k_pad400_win3600_ds2_k0001_3000_dk002_p1p0_s50_350_ds10.npz"
)

# 与已审计 diagnose 协方差运行完全一致的固定配置（kmin 除外）。
AUDITED_DATA_SUBSAMPLE_SEED = 20260703
AUDITED_RANDOM_SUBSAMPLE_SEED = 20260704


def resolve_task43_input(path: str | Path) -> Path:
    """按 原路径 -> old_doc_codes 归档 的顺序解析被搬家的 catalog 路径。"""
    original = Path(path)
    if original.exists():
        return original
    for prefix, replacement in (
        (PROJECT_ROOT / "outputs" / "task43_outputs", ARCHIVED_T43_ROOT),
        (PROJECT_ROOT / "outputs" / "task43_outputs", ACTIVE_T43_ROOT),
        (ACTIVE_T43_ROOT, ARCHIVED_T43_ROOT),
        (ARCHIVED_T43_ROOT, ACTIVE_T43_ROOT),
    ):
        try:
            relative = original.relative_to(prefix)
        except ValueError:
            continue
        candidate = replacement / relative
        if candidate.exists():
            return candidate
    raise FileNotFoundError(f"cannot resolve moved Task43 input: {original}")


def to_jsonable(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): to_jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [to_jsonable(item) for item in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, Path):
        return str(value)
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    return str(value)


def atomic_write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    tmp.write_text(json.dumps(to_jsonable(payload), indent=2, sort_keys=True) + "\n", encoding="utf-8")
    tmp.replace(path)


def stage_run(args: argparse.Namespace) -> None:
    """跑一次 k 网格对齐到 P(k) 测量网格的 jaxpower Gaussian covariance。"""
    import jax

    jax.config.update("jax_enable_x64", True)
    from jaxpower import (
        BinMesh2CorrelationPoles,
        FKPField,
        ParticleField,
        compute_fkp2_covariance_window,
        compute_spectrum2_covariance,
        interpolate_window_function,
    )
    from jaxpower.cov2 import matrix_project_to_correlation

    t0 = time.perf_counter()
    rows = read_jsonl(args.manifest)
    matches = [row for row in rows if str(row.get("phase")) == str(args.phase)]
    if len(matches) != 1:
        raise ValueError(f"manifest 中 phase={args.phase!r} 的行数不是 1")
    row = matches[0]
    halo_path = resolve_task43_input(row["halo_catalog_path"])
    random_path = resolve_task43_input(row["random_catalog_path"])

    fkp_summary = load_fkp_summary(args.fkp_summary)
    p0_values = np.asarray(fkp_summary["p0_values"], dtype="f8")
    zeff_values = np.asarray(fkp_summary["zeff_random_auto"], dtype="f8")
    p0_match = np.flatnonzero(np.isclose(p0_values, float(args.p0), rtol=0.0, atol=1.0e-10))
    if p0_match.size != 1:
        raise ValueError(f"P0={args.p0} not found in {args.fkp_summary}")
    zeff = float(zeff_values[int(p0_match[0])])
    nbar_shot = float(np.mean(np.asarray(fkp_summary["nbar"], dtype="f8")))

    # 与已审计运行一致的子采样 seed（data 20260703 / random 20260704），
    # 保证 mesh64 桥接时子采样实现完全相同。
    data_xyz, data_weight, data_meta = load_positions_weights(
        halo_path,
        fkp_summary=fkp_summary,
        p0=float(args.p0),
        max_rows=args.max_data,
        seed=int(args.data_subsample_seed),
        rescale_subsample=True,
    )
    random_xyz, random_weight, random_meta = load_positions_weights(
        random_path,
        fkp_summary=fkp_summary,
        p0=float(args.p0),
        max_rows=args.max_random,
        seed=int(args.random_subsample_seed),
        rescale_subsample=True,
    )

    mattrs, mesh_meta = infer_mesh_attrs([data_xyz, random_xyz], meshsize=int(args.meshsize), pad=float(args.mesh_pad))
    data_field = ParticleField(data_xyz, weights=data_weight, attrs=mattrs, exchange=False)
    random_field = ParticleField(random_xyz, weights=random_weight, attrs=mattrs, exchange=False)
    fkp = FKPField(data_field, random_field)

    window_s_edges = make_edges(0.0, float(args.window_s_max), float(args.window_ds))
    window_bin = BinMesh2CorrelationPoles(mattrs, edges=window_s_edges, ells=(0,), basis="bessel")
    window2 = compute_fkp2_covariance_window(
        fkp,
        bin=window_bin,
        los="local",
        resampler="cic",
        interlacing=1,
    )
    interp_coords = np.logspace(
        float(args.interpolate_window_log10_min),
        float(args.interpolate_window_log10_max),
        int(args.interpolate_window_ncoords),
    )
    window2 = window2.map(lambda window: interpolate_window_function(window, coords=interp_coords), level=1)

    k_edges = make_edges(float(args.kmin), float(args.kmax), float(args.dk))
    poles, theory_meta = build_theory_poles(
        k_edges=k_edges,
        zeff=zeff,
        b1_cov=float(args.b1_cov),
        fnl_cov=float(args.fnl_cov),
        p_fixed=float(args.p_fixed),
        sn0_fixed=float(args.sn0_fixed),
        nbar_shot=nbar_shot,
        cosmology=str(args.cosmology),
    )
    cov_parts = compute_spectrum2_covariance(window2, poles, flags=("smooth", "fftlog"), return_type="list")
    part_values = {
        name: np.asarray(part.value(), dtype="f8")
        for name, part in zip(("WW", "WS", "SS"), cov_parts, strict=True)
    }
    data_count_scale = float(data_meta["n_total"]) / float(data_meta["n_used"])
    factors = {"WW": 1.0, "WS": 1.0 / data_count_scale, "SS": 1.0 / data_count_scale**2}
    fine_corrected = sum(part_values[name] * factors[name] for name in ("WW", "WS", "SS"))

    s_edges = make_edges(float(args.s_edge_min), float(args.s_edge_max), float(args.s_edge_step))
    project = np.asarray(matrix_project_to_correlation(s_edges, poles), dtype="f8")
    cov_xi_parts = {name: project @ (part_values[name] * factors[name]) @ project.T for name in ("WW", "WS", "SS")}
    cov_xi_raw_new = sum(cov_xi_parts.values())

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    kmin_tag = f"{float(args.kmin):g}".replace(".", "p")
    out_npz = out_dir / f"task43_joint_finecov_{args.phase}_mesh{int(args.meshsize)}_kmin{kmin_tag}.npz"
    out_json = out_npz.with_suffix(".json")
    meta = {
        "task": "task43_joint_pkxi_build_cov",
        "stage": "run",
        "status": "done",
        "phase": row["phase"],
        "purpose": (
            "k-grid aligned (kmin=0.001) replica of the audited diagnose jaxpower covariance; "
            "saves the fine-grid WW/WS/SS parts needed for the joint C_pp/C_px/C_xx blocks"
        ),
        "halo_catalog_path": str(halo_path),
        "random_catalog_path": str(random_path),
        "subsample": {
            "data": data_meta,
            "random": random_meta,
            "data_subsample_seed": int(args.data_subsample_seed),
            "random_subsample_seed": int(args.random_subsample_seed),
            "rescale_subsample": True,
            "note": "seeds replicate the audited mesh64 run so the xi-block bridge is not dominated by subsample noise",
        },
        "mesh": mesh_meta,
        "window": {
            "interface": "jaxpower.compute_fkp2_covariance_window",
            "basis": "bessel",
            "ells": [0],
            "los": "local",
            "resampler": "cic",
            "interlacing": 1,
            "s_edges_summary": [float(window_s_edges[0]), float(window_s_edges[-1]), float(args.window_ds)],
            "interpolation": {
                "applied": True,
                "log10_min": float(args.interpolate_window_log10_min),
                "log10_max": float(args.interpolate_window_log10_max),
                "ncoords": int(args.interpolate_window_ncoords),
            },
        },
        "covariance_flags": ["smooth", "fftlog"],
        "k_grid": {
            "kmin": float(args.kmin),
            "kmax": float(args.kmax),
            "dk": float(args.dk),
            "nk": int(k_edges.size - 1),
            "alignment_note": (
                "edges 0.001+0.002*i replicate the P(k) measurement grid so payload "
                "fit bins map to exact fine-grid indices"
            ),
        },
        "subsample_correction": {
            "data_count_scale": data_count_scale,
            "applied_to": {"WW": 1.0, "WS": factors["WS"], "SS": factors["SS"]},
        },
        "theory": theory_meta,
        "xi_projection": {
            "interface": "jaxpower.cov2.matrix_project_to_correlation",
            "s_edges": [float(args.s_edge_min), float(args.s_edge_max), float(args.s_edge_step)],
        },
        "output_npz": str(out_npz),
        "elapsed_sec": float(time.perf_counter() - t0),
    }
    np.savez_compressed(
        out_npz,
        k_edges=k_edges,
        k_centers=0.5 * (k_edges[:-1] + k_edges[1:]),
        fine_cov_WW=part_values["WW"],
        fine_cov_WS=part_values["WS"],
        fine_cov_SS=part_values["SS"],
        fine_cov_corrected=fine_corrected,
        data_count_scale=np.asarray(data_count_scale, dtype="f8"),
        s=0.5 * (s_edges[:-1] + s_edges[1:]),
        s_edges=s_edges,
        projection_matrix=project,
        cov_xi_raw_new=cov_xi_raw_new,
        cov_xi_WW=cov_xi_parts["WW"],
        cov_xi_WS=cov_xi_parts["WS"],
        cov_xi_SS=cov_xi_parts["SS"],
        zeff=np.asarray(zeff, dtype="f8"),
        p0=np.asarray(float(args.p0), dtype="f8"),
        meta_json=np.asarray(json.dumps(to_jsonable(meta), sort_keys=True)),
    )
    atomic_write_json(out_json, meta)
    print(f"[done] wrote {out_npz} elapsed={time.perf_counter() - t0:.1f}s")


def covariance_bridge(cov_new: np.ndarray, cov_ref: np.ndarray) -> dict[str, Any]:
    """两个同形状协方差的 bridge 统计（sigma 比值 + 相对 Frobenius 差）。"""
    if cov_new.shape != cov_ref.shape:
        raise ValueError(f"bridge shape mismatch: {cov_new.shape} vs {cov_ref.shape}")
    sigma_new = np.sqrt(np.diag(cov_new))
    sigma_ref = np.sqrt(np.diag(cov_ref))
    ratio = sigma_new / sigma_ref
    diff = cov_new - cov_ref
    return {
        "shape": [int(v) for v in cov_new.shape],
        "sigma_ratio_min": float(np.min(ratio)),
        "sigma_ratio_median": float(np.median(ratio)),
        "sigma_ratio_max": float(np.max(ratio)),
        "relative_frobenius_delta": float(np.linalg.norm(diff) / np.linalg.norm(cov_ref)),
        "corr_delta_absmax": float(
            np.max(
                np.abs(
                    correlation_from_covariance(cov_new)
                    - correlation_from_covariance(cov_ref)
                )
            )
        ),
    }


def stage_assemble(args: argparse.Namespace) -> None:
    fine = np.load(args.fine_npz, allow_pickle=False)
    payload = np.load(args.pk_payload, allow_pickle=False)
    rrdeconv = np.load(args.rrdeconv_path, allow_pickle=False)
    k_centers = np.asarray(fine["k_centers"], dtype="f8")
    fine_cov = np.asarray(fine["fine_cov_corrected"], dtype="f8")
    project = np.asarray(fine["projection_matrix"], dtype="f8")
    cov_xi_raw_new = np.asarray(fine["cov_xi_raw_new"], dtype="f8")

    k_obs = np.asarray(payload["k_obs"], dtype="f8")
    pk_cov_payload = np.asarray(payload["covariance"], dtype="f8")
    rinv = np.asarray(rrdeconv["rr_window_inverse"], dtype="f8")
    cov_xx_audited = np.asarray(rrdeconv["covariance_single_realization"], dtype="f8")

    # 1) payload 15 个拟合 bin -> 细网格索引（容差 0.001，远小于 bin 宽 0.002）。
    sel = np.array([int(np.argmin(np.abs(k_centers - value))) for value in k_obs], dtype="i8")
    k_deltas = np.abs(k_centers[sel] - k_obs)
    if float(np.max(k_deltas)) > 1.0e-3:
        raise ValueError(f"payload k_obs 到细网格中心最大偏差 {np.max(k_deltas):.3e} 超过容差 1e-3")
    if len(set(int(v) for v in sel)) != k_obs.size:
        raise ValueError("payload k_obs 映射到重复的细网格索引")

    # 2) 三个块。C_xx 与已审计 rrdeconv 同一处理：R^{-1} C R^{-T} 后 nearest_spd。
    c_pp = fine_cov[np.ix_(sel, sel)]
    c_px = fine_cov[sel, :] @ project.T @ rinv.T
    c_xx_new, spd_xx = nearest_spd(rinv @ cov_xi_raw_new @ rinv.T, floor_fraction=1.0e-10)

    joint = np.zeros((45, 45), dtype="f8")
    joint[:15, :15] = c_pp
    joint[:15, 15:] = c_px
    joint[15:, :15] = c_px.T
    joint[15:, 15:] = c_xx_new
    joint = 0.5 * (joint + joint.T)
    # 注意：这里不能用相对 floor 的 nearest_spd。联合矩阵的谱跨度极大
    # （P(k) 块特征值 ~4e8，xi 块 ~1e-9），1e-10 的相对 floor 会把 xi 侧
    # 特征值压到 0.04 量级，摧毁 cross-block 结构。raw 联合矩阵本身已经
    # 正定（eigmin ~ 8.7e-9 > 0），因此只在确实出现负特征值时才用绝对
    # 极小 floor 修复，并完整记录。
    raw_eig = np.linalg.eigvalsh(joint)
    if float(np.min(raw_eig)) > 0.0:
        joint_final = joint
        spd_joint = {
            "repaired": False,
            "raw_min_eigenvalue": float(np.min(raw_eig)),
            "raw_max_eigenvalue": float(np.max(raw_eig)),
            "floor_eigenvalue": 0.0,
            "n_floored": 0,
            "condition_number_after_floor": float(np.linalg.cond(joint_final)),
        }
    else:
        floor = 1.0e-30
        evals, evecs = np.linalg.eigh(joint)
        floored = np.maximum(evals, floor)
        joint_final = (evecs * floored[None, :]) @ evecs.T
        joint_final = 0.5 * (joint_final + joint_final.T)
        spd_joint = {
            "repaired": True,
            "raw_min_eigenvalue": float(np.min(raw_eig)),
            "raw_max_eigenvalue": float(np.max(raw_eig)),
            "floor_eigenvalue": floor,
            "n_floored": int(np.count_nonzero(evals < floor)),
            "condition_number_after_floor": float(np.linalg.cond(joint_final)),
        }

    corr_joint = correlation_from_covariance(joint_final)
    cross_corr = corr_joint[:15, 15:]
    diag_product = np.sqrt(np.outer(np.diag(c_pp), np.diag(c_xx_new)))
    cross_amp = c_px / diag_product

    # 3) bridges：xi 块 vs 已审计 rrdeconv；P 块 vs payload mesh128 15x15。
    bridge_xx = covariance_bridge(c_xx_new, cov_xx_audited)
    bridge_pp = covariance_bridge(c_pp, pk_cov_payload)
    pp_gate_pass = bool(0.95 <= bridge_pp["sigma_ratio_median"] <= 1.05 and bridge_pp["relative_frobenius_delta"] < 0.10)
    xx_gate_pass = bool(0.95 <= bridge_xx["sigma_ratio_median"] <= 1.05 and bridge_xx["relative_frobenius_delta"] < 0.10)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    mesh_tag = Path(args.fine_npz).stem.split("_mesh")[-1].split("_")[0]
    out_npz = out_dir / f"task43_joint_cov_{mesh_tag}_s50_350.npz"
    meta = {
        "task": "task43_joint_pkxi_build_cov",
        "stage": "assemble",
        "status": "done",
        "mesh_tag": mesh_tag,
        "inputs": {
            "fine_npz": str(args.fine_npz),
            "pk_payload": str(args.pk_payload),
            "rrdeconv": str(args.rrdeconv_path),
        },
        "k_selection": {
            "selected_fine_indices": [int(v) for v in sel],
            "max_k_center_delta": float(np.max(k_deltas)),
            "k_obs": [float(v) for v in k_obs],
            "k_centers_selected": [float(v) for v in k_centers[sel]],
        },
        "blocks": {
            "c_pp": covariance_diagnostics(c_pp),
            "c_px": {
                "shape": [15, 30],
                "amp_absmax": float(np.max(np.abs(cross_amp))),
                "amp_absmedian": float(np.median(np.abs(cross_amp))),
                "corr_absmax": float(np.max(np.abs(cross_corr))),
                "corr_absmedian": float(np.median(np.abs(cross_corr))),
            },
            "c_xx_new": covariance_diagnostics(c_xx_new),
        },
        "spd": {
            "xx_new": spd_xx,
            "joint": spd_joint,
            "joint_min_eigenvalue_before_repair": float(np.min(np.linalg.eigvalsh(joint))),
        },
        "bridges": {
            "xx_new_vs_audited_rrdeconv": bridge_xx,
            "pp_vs_payload_mesh128": bridge_pp,
        },
        "gates": {
            "pp_bridge_pass": pp_gate_pass,
            "xx_bridge_pass": xx_gate_pass,
            "rule": "mesh64 accepted as primary iff pp sigma_ratio_median in [0.95,1.05] and rel_frobenius < 0.10; otherwise mesh128 A/B is mandatory",
            "joint_psd": bool(np.min(np.linalg.eigvalsh(joint_final)) > 0.0),
        },
        "output_npz": str(out_npz),
    }
    np.savez_compressed(
        out_npz,
        joint_covariance=joint_final,
        joint_covariance_raw=joint,
        c_pp=c_pp,
        c_px=c_px,
        c_xx_new=c_xx_new,
        pk_k_obs=k_obs,
        xi_s=np.asarray(fine["s"], dtype="f8"),
        xi_s_edges=np.asarray(fine["s_edges"], dtype="f8"),
        selected_fine_indices=sel,
        meta_json=np.asarray(json.dumps(to_jsonable(meta), sort_keys=True)),
    )
    atomic_write_json(out_npz.with_suffix(".json"), meta)
    print(f"[done] wrote {out_npz}")
    print(f"[gate] pp_bridge_pass={pp_gate_pass} xx_bridge_pass={xx_gate_pass}")
    print(f"[bridge pp] sigma_ratio_med={bridge_pp['sigma_ratio_median']:.4f} relFro={bridge_pp['relative_frobenius_delta']:.4f}")
    print(f"[bridge xx] sigma_ratio_med={bridge_xx['sigma_ratio_median']:.4f} relFro={bridge_xx['relative_frobenius_delta']:.4f}")
    print(f"[cross] |corr|max={meta['blocks']['c_px']['corr_absmax']:.3f} |amp|max={meta['blocks']['c_px']['amp_absmax']:.3f}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("run", "assemble"))
    parser.add_argument("--manifest", type=Path, default=MANIFEST_PATH)
    parser.add_argument("--fkp-summary", type=Path, default=FKP_SUMMARY_PATH)
    parser.add_argument("--phase", type=str, default="ph000")
    parser.add_argument("--p0", type=float, default=10000.0)
    parser.add_argument("--meshsize", type=int, default=64)
    parser.add_argument("--mesh-pad", type=float, default=400.0)
    parser.add_argument("--max-data", type=int, default=50000)
    parser.add_argument("--max-random", type=int, default=100000)
    parser.add_argument("--data-subsample-seed", type=int, default=AUDITED_DATA_SUBSAMPLE_SEED)
    parser.add_argument("--random-subsample-seed", type=int, default=AUDITED_RANDOM_SUBSAMPLE_SEED)
    parser.add_argument("--window-s-max", type=float, default=3600.0)
    parser.add_argument("--window-ds", type=float, default=2.0)
    parser.add_argument("--interpolate-window-log10-min", type=float, default=-2.0)
    parser.add_argument("--interpolate-window-log10-max", type=float, default=8.0)
    parser.add_argument("--interpolate-window-ncoords", type=int, default=8192)
    parser.add_argument("--kmin", type=float, default=0.001)
    parser.add_argument("--kmax", type=float, default=3.0001)
    parser.add_argument("--dk", type=float, default=0.002)
    parser.add_argument("--s-edge-min", type=float, default=50.0)
    parser.add_argument("--s-edge-max", type=float, default=350.0)
    parser.add_argument("--s-edge-step", type=float, default=10.0)
    parser.add_argument("--b1-cov", type=float, default=2.5)
    parser.add_argument("--fnl-cov", type=float, default=0.0)
    parser.add_argument("--p-fixed", type=float, default=1.0)
    parser.add_argument("--sn0-fixed", type=float, default=0.0)
    parser.add_argument("--cosmology", choices=COSMOLOGY_CHOICES, default=DEFAULT_COSMOLOGY)
    parser.add_argument("--out-dir", type=Path, default=None, help="run stage 输出目录（默认 covariance/fine）")
    parser.add_argument(
        "--assemble-out-dir",
        type=Path,
        default=JOINT_COV_DIR,
        help="assemble stage 输出目录（默认 covariance/）",
    )
    # assemble-only
    parser.add_argument("--fine-npz", type=Path, default=None)
    parser.add_argument("--pk-payload", type=Path, default=PK_PAYLOAD_PATH)
    parser.add_argument("--rrdeconv-path", type=Path, default=RRDECONV_PATH)
    args = parser.parse_args()

    if args.stage == "run":
        if args.out_dir is None:
            args.out_dir = JOINT_COV_DIR / "fine"
        stage_run(args)
    else:
        if args.fine_npz is None:
            raise ValueError("assemble 需要 --fine-npz")
        args.out_dir = args.assemble_out_dir
        stage_assemble(args)


if __name__ == "__main__":
    main()
