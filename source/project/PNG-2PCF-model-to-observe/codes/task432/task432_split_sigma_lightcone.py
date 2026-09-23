#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Task 4.3.2 box-safe lightcone shared-vs-split sigma diagnostic.

本脚本复用 Task43 已冻结的 box-safe lightcone P02/xi02 数据和 covariance，
只把 joint likelihood 的 nuisance 从共享 ``sigma_s`` 改成
``sigma_s_P`` 与 ``sigma_s_xi``，用于判断 lightcone joint b1 下移是否只是
shared-FoG 约束造成的参数折中。

所有结果写入 task432_model_repair 独立目录，不覆盖 Task43 production chains。
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any

for _name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_name, "1")

import numpy as np

PROJECT_ROOT = Path("/pscratch/sd/l/lzy/PNG-2PCF-model-to-observe")
CODE_DIR = PROJECT_ROOT / "codes" / "task43"
TASK432_DIR = PROJECT_ROOT / "codes" / "task432"
for _path in (CODE_DIR, TASK432_DIR):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

from task432_split_sigma_rawbox import atomic_json, precision_from_covariance, run_fit  # noqa: E402
from task43_fit_rsd_lightcone_pk0_vs_xi0_smin50 import load_pk  # noqa: E402
from task43_fit_rsd_lightcone_x25 import FastWindowRSDModel, load_window  # noqa: E402
from task43_fit_rsd_rawbox_pk0_vs_xi0_smin50 import set_affinity  # noqa: E402
from task43_rsd_boxsafe_p02_increment import COV_NPZ, MEASURE_DIR, PAYLOAD_NPZ, WindowConvolvedP02Model  # noqa: E402
from task43_rsd_common import OUTPUT_ROOT, PHASES  # noqa: E402
from task43_rsd_joint_p02xi02_fit import (  # noqa: E402
    BOXSAFE_ROOT,
    ELL02_DECONV_NPZ,
    ELL2_SUMMARY,
    MEAN_WINDOW_NPZ,
    RAW_NPZ,
    P2_KMIN,
)
from task43_rsd_model import FullDiscreteRSDModel, build_cache  # noqa: E402


def build_products() -> dict[str, Any]:
    """读取当前 box-safe lightcone 的 BAO-masked data/cov/model。"""

    payload = load_pk(PAYLOAD_NPZ)
    k_obs = np.asarray(payload["k_obs"], dtype="f8")
    fit_indices = np.asarray(payload["fit_bin_indices"], dtype="i8")
    zeff = float(np.asarray(payload["zeff"]).item())
    keep_p2 = np.flatnonzero(k_obs >= float(P2_KMIN) - 1.0e-12)

    with np.load(RAW_NPZ, allow_pickle=False) as data:
        covariance_pk = np.asarray(data["covariance_pk"], dtype="f8")
        projection = np.asarray(data["projection_matrix"], dtype="f8")
        covariance_xi_multi = np.asarray(data["covariance_xi_multipoles"], dtype="f8")
    with np.load(ELL02_DECONV_NPZ, allow_pickle=False) as data:
        deconv64 = np.asarray(data["covariance_single_realization"], dtype="f8")
        rr_inverse = np.asarray(data["rr_window_inverse"], dtype="f8")
        s = np.asarray(data["s"], dtype="f8")
    with np.load(ELL2_SUMMARY, allow_pickle=False) as data:
        xi_mean = np.asarray(data["xi_multipoles_mean"], dtype="f8")

    # Current promoted BAO-mask: both xi poles retain the same 26 bins.
    mask_xi = (s >= 50.0) & (s < 350.0) & ~((s >= 80.0) & (s < 120.0))

    stacks: list[np.ndarray] = []
    window_rows = None
    theory = None
    for phase in PHASES:
        path = MEASURE_DIR / f"task43_rsd_lightcone_p02_{phase}_mesh256_kmax0p300_dk0p002.npz"
        with np.load(path, allow_pickle=False) as data:
            p0 = np.asarray(data["pk0"], dtype="f8")
            p2 = np.asarray(data["pk2"], dtype="f8")
            stacks.append(np.concatenate([p0[fit_indices], p2[fit_indices]]))
            if phase == "ph000":
                window_rows = np.asarray(data["window_matrix"], dtype="f8")
                theory = (np.asarray(data["theory_k"], dtype="f8"), np.asarray(data["theory_ell"], dtype="i8"))
    mean_p_full = np.mean(np.stack(stacks), axis=0)
    p0_n = int(fit_indices.size)
    p2_n = int(keep_p2.size)
    data_p = np.concatenate([mean_p_full[:p0_n], mean_p_full[p0_n:][keep_p2]])
    data_x = np.concatenate([xi_mean[0][mask_xi], xi_mean[1][mask_xi]])
    data_joint = np.concatenate([data_p, data_x])

    # Assemble the same frozen covariance family used by the standard four-way fit.
    nper = int(np.asarray(covariance_pk).shape[0] / 3)
    ids_p = np.concatenate([fit_indices, nper + fit_indices[keep_p2]])
    c_pp = covariance_pk[np.ix_(ids_p, ids_p)]
    proj64 = projection[: 2 * s.size, :]
    c_px_full = rr_inverse @ (proj64 @ covariance_pk[:, ids_p])
    row_ids = np.concatenate([np.flatnonzero(mask_xi), s.size + np.flatnonzero(mask_xi)])
    c_px = c_px_full[row_ids, :]
    c_xx = deconv64[np.ix_(row_ids, row_ids)]
    c_joint = np.block([[c_pp, c_px.T], [c_px, c_xx]])
    c_joint = 0.5 * (c_joint + c_joint.T)
    eigenvalues, eigenvectors = np.linalg.eigh(c_joint)
    floor = max(float(eigenvalues[-1]) * 1.0e-14, 1.0e-300)
    if float(eigenvalues[0]) < floor:
        c_joint = (eigenvectors * np.maximum(eigenvalues, floor)[None, :]) @ eigenvectors.T
        c_joint = 0.5 * (c_joint + c_joint.T)

    # window_matrix 的观测行按 P0/P2 排列，各自 150 行；这与 theory_ell
    # 的 327 个理论网格点不能混用，后者是输入列而不是观测 row offset。
    measurement_rows_per_ell = int(window_rows.shape[0] // 2)
    model_p = WindowConvolvedP02Model(
        window_rows[np.concatenate([fit_indices, measurement_rows_per_ell + fit_indices]), :],
        theory[0],
        theory[1],
        zeff=zeff,
    )
    cache = build_cache(zeff=zeff, boxsize=2000.0, kmax=5.0, ells=(0, 2), cosmology="abacus_c000")
    model_x = FastWindowRSDModel(
        FullDiscreteRSDModel(cache, nmu=64),
        {"mean": load_window(MEAN_WINDOW_NPZ)},
        sigma_step=0.05,
    )

    def eval_p(theta: np.ndarray) -> np.ndarray:
        values = model_p.evaluate(np.asarray([theta[0], theta[1], theta[2], theta[4]], dtype="f8"))
        return np.concatenate([values[:p0_n], values[p0_n:][keep_p2]])

    def eval_x(theta: np.ndarray) -> np.ndarray:
        values = model_x.evaluate(
            np.asarray([theta[0], theta[1], theta[3]], dtype="f8"),
            model="formal_gic",
            window_key="mean",
        )
        return np.concatenate([np.asarray(values[0])[mask_xi], np.asarray(values[2])[mask_xi]])

    return {
        "data": {"p": data_p, "x": data_x, "joint": data_joint},
        "covariance": {"p": c_pp, "x": c_xx, "joint": c_joint},
        "evaluate": {
            "p": eval_p,
            "x": eval_x,
            "joint": lambda theta: np.concatenate([eval_p(theta), eval_x(theta)]),
        },
        "metadata": {
            "raw_covariance": str(RAW_NPZ),
            "deconv_covariance": str(ELL02_DECONV_NPZ),
            "p_bins_ell0": p0_n,
            "p_bins_ell2": p2_n,
            "xi_bins_ell0": int(np.count_nonzero(mask_xi)),
            "xi_bins_ell2": int(np.count_nonzero(mask_xi)),
            "xi_mask": "50 <= s < 350; 80 <= s < 120 excluded",
            "p2_kmin": float(P2_KMIN),
            "covariance_policy": "C_single, no /25",
        },
    }


def main() -> None:
    """运行 lightcone split-sigma 诊断。"""

    parser = argparse.ArgumentParser()
    parser.add_argument("--threads", type=int, default=6)
    parser.add_argument("--nwalkers", type=int, default=32)
    parser.add_argument("--nsteps", type=int, default=1000)
    parser.add_argument("--burnin", type=int, default=200)
    parser.add_argument("--seed", type=int, default=20260925)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument(
        "--output-root",
        type=Path,
        default=OUTPUT_ROOT / "task432_model_repair" / "lightcone_baomask_split_sigma",
    )
    args = parser.parse_args()
    if int(args.threads) > 8:
        raise ValueError("Task432 login-node fit must use <= 8 threads")
    set_affinity(int(args.threads))
    if args.smoke:
        args.nsteps, args.burnin = 300, 80

    products = build_products()
    data = products["data"]
    covariance = products["covariance"]
    evaluate = products["evaluate"]
    output_root = Path(args.output_root)
    lo5 = np.asarray([-500.0, 0.5, 0.0, 0.0, -1.0], dtype="f8")
    hi5 = np.asarray([500.0, 5.0, 30.0, 30.0, 1.0], dtype="f8")

    def shared(theta: np.ndarray) -> np.ndarray:
        full = np.asarray([theta[0], theta[1], theta[2], theta[2], theta[3]], dtype="f8")
        return evaluate["joint"](full)

    def p_compact(theta: np.ndarray) -> np.ndarray:
        return evaluate["p"](np.asarray([theta[0], theta[1], theta[2], 8.0, theta[3]], dtype="f8"))

    def x_compact(theta: np.ndarray) -> np.ndarray:
        return evaluate["x"](np.asarray([theta[0], theta[1], 1.0, theta[2], 0.0], dtype="f8"))

    fits: dict[str, Any] = {}
    fits["p_marginal"] = run_fit(
        "p_marginal", data["p"], covariance["p"], p_compact,
        (lo5[[0, 1, 2, 4]], hi5[[0, 1, 2, 4]]), ("fNL", "b1", "sigma_s_P", "sn0"),
        nwalkers=args.nwalkers, nsteps=args.nsteps, burnin=args.burnin, seed=args.seed, output_root=output_root,
    )
    fits["xi_marginal"] = run_fit(
        "xi_marginal", data["x"], covariance["x"], x_compact,
        (lo5[[0, 1, 3]], hi5[[0, 1, 3]]), ("fNL", "b1", "sigma_s_xi"),
        nwalkers=args.nwalkers, nsteps=args.nsteps, burnin=args.burnin, seed=args.seed + 1, output_root=output_root,
    )
    fits["joint_shared"] = run_fit(
        "joint_shared", data["joint"], covariance["joint"], shared,
        (lo5[[0, 1, 2, 4]], hi5[[0, 1, 2, 4]]), ("fNL", "b1", "sigma_s", "sn0"),
        nwalkers=args.nwalkers, nsteps=args.nsteps, burnin=args.burnin, seed=args.seed + 2, output_root=output_root,
    )
    fits["joint_split"] = run_fit(
        "joint_split", data["joint"], covariance["joint"], evaluate["joint"],
        (lo5, hi5), ("fNL", "b1", "sigma_s_P", "sigma_s_xi", "sn0"),
        nwalkers=args.nwalkers, nsteps=args.nsteps, burnin=args.burnin, seed=args.seed + 3, output_root=output_root,
    )

    audit = {
        "task": "Task 4.3.2 box-safe lightcone BAO-masked split sigma diagnostic",
        "status": "smoke_complete" if args.smoke else "complete",
        "data_contract": products["metadata"],
        "fit_contract": {
            "shared_baseline": "sigma_s shared by P02 and xi02",
            "split_variant": "sigma_s_P on P02; sigma_s_xi on xi02; sn0 P-side only",
            "covariance_policy": "frozen C_single; no /25; cross block retained",
        },
        "fits": fits,
        "resource": {"threads": int(args.threads), "nwalkers": int(args.nwalkers), "nsteps": int(args.nsteps), "burnin": int(args.burnin)},
    }
    output_root.joinpath("audits").mkdir(parents=True, exist_ok=True)
    atomic_json(output_root / "audits" / ("task432_lightcone_split_sigma_smoke.json" if args.smoke else "task432_lightcone_split_sigma_summary.json"), audit)
    print(json.dumps({"status": audit["status"], "output": str(output_root / "audits")}, sort_keys=True))


if __name__ == "__main__":
    main()
