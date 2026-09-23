#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
4.3 lightcone 实空间 P0/xi0/joint：把协方差缩放到等效体积 V = 27 (Gpc/h)^3 后重跑 MCMC。

代码大纲：
1. 读入冻结的 4.3 实空间几何常数 TASK43_REALSPACE_VOLUME（= 1.908169922664481 Gpc^3）与目标体积 27。
2. 把源协方差 joint_covariance 按 C_new = C_old * (V_old / V_new) 缩放并原子落盘，
   同时写一份与源审计同构的审计 JSON（status=done，并记录缩放因子）。
3. 直接复用生产驱动 task43_run_lightcone_joint_baomask_v1.py 的
   load_real_specs / fit_maximum_likelihood / run_chain，
   保证模型、数据向量、先验、起点、种子、门限与生产完全一致；唯一变化是协方差体积。
4. 只跑 real_p0 / real_xi0 / real_joint_p0xi0 三条链（跳过 RSD 半边），
   用与生产相同的 seed = 20260923 + 100*index（index = 0,1,2）以保持可复现。
5. 输出 samples.npz + summary.json 到独立 out-root，不覆盖任何既有审计产物。

资源：登录节点，线程数 <= 8。
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_v, "1")
os.environ.setdefault("MPLBACKEND", "Agg")

import numpy as np

RECON = Path("/pscratch/sd/l/lzy/PNG-2PCF-model-to-observe")
sys.path.insert(0, str(RECON / "codes" / "task43"))

import task43_run_lightcone_joint_baomask_v1 as drv
from task43_rsd_lightcone_pk0_contract import TASK43_REALSPACE_VOLUME
from task43_rsd_common import OUTPUT_ROOT, atomic_savez, atomic_write_json, sha256_file

V_OLD = float(TASK43_REALSPACE_VOLUME)   # (Mpc/h)^3
V_NEW = 27.0e9                           # (Mpc/h)^3 = 27 (Gpc/h)^3，与 recon 面板一致
SCALE = V_OLD / V_NEW                    # 协方差缩放因子（C ∝ 1/V）


def build_scaled_covariance(out_root: Path) -> tuple[Path, Path]:
    """把源协方差按体积比缩放，原子写出新 npz 与同构审计 JSON。"""
    cov_dir = out_root / "covariance"
    cov_dir.mkdir(parents=True, exist_ok=True)
    source = drv.REAL_COVARIANCE
    target = cov_dir / "task43_joint_cov_64_s50_350_vequiv27.npz"
    with np.load(source, allow_pickle=False) as payload:
        data = {key: np.asarray(payload[key]) for key in payload.files}
    raw = data["joint_covariance"]
    data["joint_covariance"] = raw * SCALE
    data["joint_covariance_unscaled"] = raw
    tmp = target.with_name(target.stem + ".tmp.npz")
    np.savez_compressed(tmp, **data)
    tmp.replace(target)

    audit_src = Path(str(source).replace(".npz", ".json"))
    audit = json.loads(audit_src.read_text(encoding="utf-8"))
    audit["volume_equivalent"] = {
        "V_old_mpc_h3": V_OLD,
        "V_new_mpc_h3": V_NEW,
        "scale_C_new_over_C_old": SCALE,
        "L_eff_old_mpc_h": V_OLD ** (1.0 / 3.0),
        "L_eff_new_mpc_h": V_NEW ** (1.0 / 3.0),
        "rule": "Gaussian survey-window covariance scales as C propto 1/V; data vector unchanged",
        "source_covariance": str(source),
        "source_covariance_sha256": sha256_file(source),
    }
    audit_target = target.with_suffix(".json")
    atomic_write_json(audit_target, audit)
    return target, audit_target


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--threads", type=int, default=8)
    ap.add_argument("--nwalkers", type=int, default=64)
    ap.add_argument("--nsteps", type=int, default=30000)
    ap.add_argument("--burnin", type=int, default=5000)
    ap.add_argument("--seed", type=int, default=20260923)
    ap.add_argument("--pk-kmax", type=float, default=0.08)
    ap.add_argument("--smin", type=float, default=50.0)
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--out-root", type=Path,
                    default=OUTPUT_ROOT.parent / "lightcone" / "kmax0p08_smin50_vequiv27_v1")
    args = ap.parse_args()
    if not 1 <= args.threads <= 8:
        raise ValueError("--threads 必须在 1..8")

    nsteps = 400 if args.smoke else int(args.nsteps)
    burnin = 100 if args.smoke else int(args.burnin)
    out_root = Path(args.out_root) / ("smoke" if args.smoke else "")
    out_root.mkdir(parents=True, exist_ok=True)
    cpus = drv.set_affinity(int(args.threads))
    print(f"[info] V_old={V_OLD:.6e} Mpc^3, V_new={V_NEW:.6e} Mpc^3, scale={SCALE:.10f}", flush=True)

    cov_path, cov_audit = build_scaled_covariance(out_root)
    print(f"[info] scaled covariance -> {cov_path}", flush=True)

    specs, metadata, arrays = drv.load_real_specs(
        smin=float(args.smin), pk_kmax=float(args.pk_kmax),
        covariance_path=cov_path, covariance_audit_path=cov_audit,
    )
    metadata["volume_equivalent"] = {
        "V_old_mpc_h3": V_OLD, "V_new_mpc_h3": V_NEW, "scale_C_new_over_C_old": SCALE,
    }
    atomic_write_json(out_root / "task43_lightcone_joint_vequiv27_metadata.json", metadata)

    results = {}
    gates = {}
    for index, spec in enumerate(specs):
        metric = drv.ScaledGaussianMetric(spec.covariance)
        gates[f"{spec.name}_strict_spd"] = bool(metric.eigenvalues[0] > 1.0e-12)
        map_summary, theta_ml = drv.fit_maximum_likelihood(spec, metric)
        prediction = np.asarray(spec.evaluate(theta_ml), dtype="f8")
        print(json.dumps({"variant": spec.name, "map": map_summary.get("theta"),
                          "map_summary_keys": sorted(map_summary.keys())}, sort_keys=True), flush=True)
        t0 = time.perf_counter()
        chain_summary, chain, logp = drv.run_chain(
            spec, metric, theta_ml,
            nwalkers=int(args.nwalkers), nsteps=nsteps, burnin=burnin,
            seed=int(args.seed) + 100 * index, nworkers=int(args.threads),
        )
        result = {
            "group": spec.group,
            "parameter_names": list(spec.parameter_names),
            "map": map_summary,
            "mcmc": chain_summary,
            "phase_diagnostics": drv.phase_diagnostics(spec, metric, theta_ml),
            "correlation_eigenvalue_min": float(metric.eigenvalues[0]),
            "volume_equivalent": {"V_old_mpc_h3": V_OLD, "V_new_mpc_h3": V_NEW,
                                  "scale_C_new_over_C_old": SCALE},
        }
        fit_root = out_root / "fits" / spec.name
        fit_root.mkdir(parents=True, exist_ok=True)
        fit_npz, fit_json = fit_root / "samples.npz", fit_root / "summary.json"
        atomic_savez(fit_npz, parameter_names=np.asarray(spec.parameter_names),
                     chain_by_step=chain, log_probability_by_step=logp,
                     theta_maximum_likelihood=theta_ml, prediction_maximum_likelihood=prediction,
                     data=np.asarray(spec.data), phase_data=np.asarray(spec.phase_data),
                     covariance_single=np.asarray(spec.covariance))
        status = "pass" if args.smoke or all(chain_summary["gates"].values()) else "validation_failed"
        atomic_write_json(fit_json, {
            "task": "task43_run_lightcone_joint_vequiv27", "variant": spec.name,
            "status": status, "result": result, "nsteps": nsteps, "burnin": burnin,
            "output_npz": str(fit_npz), "output_npz_sha256": sha256_file(fit_npz),
        })
        results[spec.name] = result
        if not args.smoke:
            for gate, passed in chain_summary["gates"].items():
                gates[f"{spec.name}_{gate}"] = bool(passed)
        print(json.dumps({"variant": spec.name,
                          "fNL": chain_summary["posterior"]["fNL"],
                          "elapsed_sec": time.perf_counter() - t0,
                          "gates": chain_summary["gates"]}, sort_keys=True), flush=True)

    atomic_write_json(out_root / "task43_lightcone_joint_vequiv27_audit.json", {
        "task": "task43_run_lightcone_joint_vequiv27", "status": "pass" if all(gates.values()) else "validation_failed",
        "cpu_affinity": cpus, "volume_equivalent": {"V_old_mpc_h3": V_OLD, "V_new_mpc_h3": V_NEW,
                                                    "scale_C_new_over_C_old": SCALE},
        "gates": gates, "results": {k: v["mcmc"]["posterior"]["fNL"] for k, v in results.items()},
    })
    print("[done]", flush=True)


if __name__ == "__main__":
    main()
