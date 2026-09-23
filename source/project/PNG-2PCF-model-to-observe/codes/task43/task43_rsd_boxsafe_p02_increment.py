#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Quadrupole-increment test on the P(k) side: does adding P2 to P0 help?

Data: x25 boxsafe measurements with ells=(0,2) on the frozen 150-bin grid
(the P0 block bridges the audited monopole bitwise); the fit selects the same
15 DESI-PNG bins as the payload for both poles.  Model: the frozen
WindowConvolvedPk0 surrogate extended to the (30, 981) window whose rows are
[P0 bins, P2 bins]; sn0 enters only the ell=0 theory so window leakage into
P2 is handled automatically.  Covariance: the ell0/ell2 blocks of the frozen
450x450 jaxpower matrix at the same fit bins (no new jaxpower run).

Two fits: P0-only control (15 bins) and P0+P2 (30 bins), free
(fNL, b1, sigma_s, sn0), priors and MCMC identical to the joint experiment.
"""

from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path
from typing import Any

for _name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_name, "1")

import numpy as np
from scipy.interpolate import CubicSpline
from scipy.stats import chi2 as chi2_distribution

from task43_fit_rsd_lightcone_pk0_vs_xi0_smin50 import load_pk  # noqa: E402
from task43_joint_rsd_pkxi_fit import (  # noqa: E402
    BOUNDS_HI,
    BOUNDS_LO,
    JOINT_PARAMS,
    OPTIMIZER_STARTS,
    build_precision,
    fisher_covariance,
    run_chain,
    summarize_chain,
)
from task43_rsd_common import OUTPUT_ROOT, PHASES, atomic_savez, atomic_write_json, sha256_file  # noqa: E402
from task43_rsd_lightcone_pk0_contract import WINDOW_THEORY_KMIN  # noqa: E402
from task43_rsd_model import DELTA_C, build_cache  # noqa: E402
from task43_theory_template import build_template_arrays, load_task41  # noqa: E402


PROJECT_ROOT = Path("/pscratch/sd/l/lzy/PNG-2PCF-model-to-observe")
BOXSAFE_ROOT = OUTPUT_ROOT / "lightcone_boxsafe_zobs0p4_0p8"
MEASURE_DIR = (
    PROJECT_ROOT / "outputs/task43_outputs/rsd_validation/lightcone/pk"
    / "boxsafe_zobs0p4_0p8_p02_x25_fkpP010000"
)
COV_NPZ = (
    PROJECT_ROOT / "outputs/task43_outputs/rsd_validation/lightcone/pk_covariance"
    / "boxsafe_zobs0p4_0p8_x25_fkpP010000_fnlcov0_b1cov2p50_sigmas7p50"
    / "task43_rsd_lightcone_pk0_cov_ph000_mesh128_kmax0p300_dk0p002.npz"
)
PAYLOAD_NPZ = (
    BOXSAFE_ROOT / "pk_summary"
    / "task43_rsd_boxsafe_lightcone_pk0_x25_kmin0p004291_kmax0p10_l0only_15bin.npz"
)
OUT_DIR = BOXSAFE_ROOT / "p02_increment"
AUDIT_JSON = OUT_DIR / "audits" / "task43_rsd_boxsafe_p02_increment_summary.json"
SN0_SCALE = 1.0e4


def interp_logk(k: np.ndarray, base_k: np.ndarray, values: np.ndarray) -> np.ndarray:
    return np.interp(np.log(np.asarray(k, dtype="f8")), np.log(base_k), values)


class WindowConvolvedP02Model:
    """Cubic-in-sigma surrogate after the ell=0,2,4 window convolution (P0+P2 rows)."""

    def __init__(self, window: np.ndarray, theory_k: np.ndarray, theory_ell: np.ndarray, *, zeff: float, sigma_step: float = 0.05, nmu: int = 96) -> None:
        self.window = np.asarray(window, dtype="f8")
        self.theory_k = np.asarray(theory_k, dtype="f8")
        self.theory_ell = np.asarray(theory_ell, dtype="i8")
        self.zeff = float(zeff)
        self.theory_kmin = WINDOW_THEORY_KMIN
        if not np.array_equal(np.unique(self.theory_ell), [0, 2, 4]):
            raise RuntimeError(f"window theory multipoles are {np.unique(self.theory_ell)}")
        self.nmu = int(nmu)
        self.mu, self.wmu = np.polynomial.legendre.leggauss(self.nmu)
        self.mu2 = self.mu**2
        self.task41 = load_task41()
        k_template = np.geomspace(1.0e-5, 20.0, 20000)
        self.template, self.cosmology_meta = build_template_arrays(
            self.task41, k_template, z=self.zeff, cosmology="abacus_c000"
        )
        self.alpha = interp_logk(self.theory_k, self.template["k"], self.template["alpha"])
        self.pk_dd = interp_logk(self.theory_k, self.template["k"], self.template["pk_dd"])
        cache = build_cache(zeff=self.zeff, boxsize=2000.0, kmax=3.0, ells=(0, 2), cosmology="abacus_c000")
        self.theory_cache = cache
        with np.load(cache, allow_pickle=False) as data:
            self.f_growth = float(np.asarray(data["f_growth"]).item())
        self.support = self.theory_k >= self.theory_kmin - 1.0e-15
        self.sigma_grid = np.arange(0.0, 30.0 + 0.5 * float(sigma_step), float(sigma_step), dtype="f8")
        convolved = np.empty((self.sigma_grid.size, self.window.shape[0], 6), dtype="f8")
        for index, sigma_s in enumerate(self.sigma_grid):
            convolved[index] = self.window @ self._theory_basis(float(sigma_s))
        self.spline = CubicSpline(self.sigma_grid, convolved, axis=0)
        shot_vector = np.zeros(self.theory_k.size, dtype="f8")
        shot_vector[(self.theory_ell == 0) & self.support] = SN0_SCALE
        self.shot_response = self.window @ shot_vector

    def _theory_basis(self, sigma_s: float) -> np.ndarray:
        basis = np.zeros((self.theory_k.size, 6), dtype="f8")
        for ell in (0, 2, 4):
            selected = self.theory_ell == ell
            k = self.theory_k[selected]
            alpha = self.alpha[selected]
            pk_dd = self.pk_dd[selected]
            coeff = np.zeros(ell + 1, dtype="f8")
            coeff[ell] = 1.0
            legendre = np.polynomial.legendre.legval(self.mu, coeff)
            damping = 1.0 / (1.0 + 0.5 * (k[:, None] * self.mu[None, :] * float(sigma_s)) ** 2) ** 2
            prefactor = 0.5 * (2 * ell + 1)
            moment0 = prefactor * np.sum(self.wmu[None, :] * legendre[None, :] * damping, axis=1)
            moment2 = prefactor * np.sum(self.wmu[None, :] * legendre[None, :] * damping * self.mu2[None, :], axis=1)
            moment4 = prefactor * np.sum(self.wmu[None, :] * legendre[None, :] * damping * self.mu2[None, :] ** 2, axis=1)
            basis[selected, 0] = pk_dd * moment0
            basis[selected, 1] = pk_dd * alpha * moment0
            basis[selected, 2] = pk_dd * alpha**2 * moment0
            basis[selected, 3] = pk_dd * moment2
            basis[selected, 4] = pk_dd * alpha * moment2
            basis[selected, 5] = pk_dd * moment4
        basis[~self.support] = 0.0
        return basis

    def evaluate(self, theta: np.ndarray) -> np.ndarray:
        fnl, b1, sigma_s, sn0 = map(float, np.asarray(theta, dtype="f8")[:4])
        q = fnl * 2.0 * DELTA_C * (b1 - 1.0)
        f = self.f_growth
        coefficients = np.asarray([b1 * b1, 2.0 * b1 * q, q * q, 2.0 * b1 * f, 2.0 * q * f, f * f])
        return np.asarray(self.spline(sigma_s), dtype="f8") @ coefficients + sn0 * self.shot_response

    def direct(self, theta: np.ndarray, *, nmu: int | None = None) -> np.ndarray:
        fnl, b1, sigma_s, sn0 = map(float, np.asarray(theta, dtype="f8")[:4])
        nquad = self.nmu if nmu is None else int(nmu)
        mu, wmu = np.polynomial.legendre.leggauss(nquad)
        mu2 = mu**2
        bphi = 2.0 * DELTA_C * (b1 - 1.0)
        amplitude = b1 + fnl * bphi * self.alpha
        theory = np.zeros(self.theory_k.size, dtype="f8")
        for ell in (0, 2, 4):
            selected = self.theory_ell == ell
            k = self.theory_k[selected]
            coeff = np.zeros(ell + 1, dtype="f8")
            coeff[ell] = 1.0
            legendre = np.polynomial.legendre.legval(mu, coeff)
            damping = 1.0 / (1.0 + 0.5 * (k[:, None] * mu[None, :] * sigma_s) ** 2) ** 2
            pkmu = self.pk_dd[selected, None] * (amplitude[selected, None] + self.f_growth * mu2[None, :]) ** 2 * damping
            theory[selected] = 0.5 * (2 * ell + 1) * np.sum(wmu[None, :] * pkmu * legendre[None, :], axis=1)
        theory[(self.theory_ell == 0)] += sn0 * SN0_SCALE
        theory[~self.support] = 0.0
        return self.window @ theory

    def validate(self) -> dict[str, Any]:
        trials = (
            np.asarray([0.0, 2.55, 8.0, 0.0]),
            np.asarray([-75.0, 2.2, 3.37, 0.2]),
            np.asarray([80.0, 2.8, 12.43, -0.3]),
            np.asarray([15.0, 2.5, 0.07, 0.1]),
            np.asarray([-20.0, 2.6, 29.93, -0.1]),
        )
        rows = []
        for theta in trials:
            fast = self.evaluate(theta)
            direct = self.direct(theta)
            direct_hi = self.direct(theta, nmu=192)
            scale = max(1.0, float(np.linalg.norm(direct)))
            rows.append(
                {
                    "theta": theta.tolist(),
                    "surrogate_relative_l2": float(np.linalg.norm(fast - direct) / scale),
                    "nmu96_vs_192_relative_l2": float(np.linalg.norm(direct - direct_hi) / max(1.0, np.linalg.norm(direct_hi))),
                }
            )
        maximum = max(max(row["surrogate_relative_l2"], row["nmu96_vs_192_relative_l2"]) for row in rows)
        return {"status": "pass" if maximum < 1.0e-8 else "fail", "max_relative_l2": float(maximum), "trials": rows}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--threads", type=int, default=8)
    parser.add_argument("--nwalkers", type=int, default=64)
    parser.add_argument("--nsteps", type=int, default=30000)
    parser.add_argument("--burnin", type=int, default=5000)
    parser.add_argument("--seed", type=int, default=20260908)
    args = parser.parse_args()
    if AUDIT_JSON.exists():
        raise FileExistsError(f"immutable increment summary exists: {AUDIT_JSON}")

    started = time.perf_counter()
    from task43_fit_rsd_rawbox_pk0_vs_xi0_smin50 import set_affinity

    cpus = set_affinity(int(args.threads))
    payload = load_pk(PAYLOAD_NPZ)
    fit_indices = np.asarray(payload["fit_bin_indices"], dtype="i8")
    if fit_indices.size != 15:
        raise RuntimeError("payload fit bins are not the frozen 15-bin selection")
    zeff = float(np.asarray(payload["zeff"]).item())

    stacks, window_rows, theory = [], None, None
    for phase in PHASES:
        path = MEASURE_DIR / (
            f"task43_rsd_lightcone_p02_{phase}_mesh256_"
            + "kmax0.300_dk0.002".replace(".", "p") + ".npz"
        )
        if not path.is_file():
            raise FileNotFoundError(path)
        with np.load(path, allow_pickle=False) as d:
            pk0 = np.asarray(d["pk0"], dtype="f8")[fit_indices]
            pk2 = np.asarray(d["pk2"], dtype="f8")[fit_indices]
            if phase == "ph000":
                window_rows = np.asarray(d["window_matrix"], dtype="f8")
                theory = {
                    "k": np.asarray(d["theory_k"], dtype="f8"),
                    "ell": np.asarray(d["theory_ell"], dtype="i8"),
                }
                zeff_meas = float(np.asarray(d["zeff"]).item())
        stacks.append(np.concatenate([pk0, pk2]))
    data = np.stack(stacks)  # (25, 30)
    if abs(zeff_meas - zeff) > 1.0e-3:
        raise RuntimeError(f"zeff mismatch beyond tolerance: measurement {zeff_meas} vs payload {zeff}")
    # Both zeff values are audited coexisting conventions (fkp-summary vs
    # payload/window-model); the window model follows the payload, as in the
    # audited P0 surrogate.
    zeff_note = {"measurement_fkf": zeff_meas, "payload_window_model": zeff}
    mean = np.mean(data, axis=0)
    window = window_rows[np.concatenate([fit_indices, 150 + fit_indices]), :]
    if window.shape != (30, theory["k"].size):
        raise RuntimeError(f"unexpected fit-bin window shape {window.shape}")

    with np.load(COV_NPZ, allow_pickle=False) as d:
        cov_full = np.asarray(d["covariance_full"], dtype="f8")
        k_edges_cov = np.asarray(d["k_edges"], dtype="f8")
    ids = np.concatenate([fit_indices, 150 + fit_indices])
    cov30 = cov_full[np.ix_(ids, ids)]
    cov30 = 0.5 * (cov30 + cov30.T)
    eig = np.linalg.eigvalsh(cov30)
    if float(eig[0]) <= 0.0:
        raise RuntimeError(f"P02 covariance not SPD: eigmin={eig[0]:.3e}")
    # Bridge: the ell0-ell0 block must reproduce the payload covariance.
    payload_cov = np.asarray(payload["covariance_single_realization"], dtype="f8")
    bridge_relfro = float(np.linalg.norm(cov30[:15, :15] - payload_cov) / np.linalg.norm(payload_cov))
    if bridge_relfro > 1.0e-10:
        raise RuntimeError(f"ell0 covariance bridge failed: relFro={bridge_relfro:.3e}")

    model = WindowConvolvedP02Model(window, theory["k"], theory["ell"], zeff=zeff)
    validation = model.validate()
    if validation["status"] != "pass":
        raise RuntimeError(f"P02 surrogate failed: {validation}")

    from scipy.optimize import least_squares

    results: dict[str, Any] = {}
    chains: dict[str, np.ndarray] = {}
    variants = {"p0_control": slice(0, 15), "p02": slice(0, 30)}
    for index, (variant, sl) in enumerate(variants.items()):
        data_v = mean[sl]
        cov_v = cov30[sl, sl]
        precision, precision_meta = build_precision(cov_v)
        chol = np.linalg.cholesky(cov_v)

        def residual(theta: np.ndarray) -> np.ndarray:
            return np.linalg.solve(chol, data_v - model.evaluate(theta)[sl])

        solutions = [
            least_squares(residual, start, bounds=(BOUNDS_LO, BOUNDS_HI), max_nfev=3000,
                          xtol=1.0e-12, ftol=1.0e-12, gtol=1.0e-12)
            for start in OPTIMIZER_STARTS
        ]
        best = min(solutions, key=lambda r: float(r.fun @ r.fun))
        theta_map = np.asarray(best.x, dtype="f8")
        summary, chain, logp = run_chain(
            lambda theta, sl=sl: model.evaluate(theta)[sl],
            data_v,
            precision,
            theta_map,
            4,
            nwalkers=int(args.nwalkers),
            nsteps=int(args.nsteps),
            burnin=int(args.burnin),
            seed=int(args.seed) + 100 * index,
        )
        fisher = fisher_covariance(lambda theta, sl=sl: model.evaluate(theta)[sl], precision, theta_map)
        summary["nominal"] = {
            "theta": [float(v) for v in theta_map],
            "chi2": float(best.fun @ best.fun),
            "dof": int(data_v.size - 4),
        }
        summary["precision_meta"] = precision_meta
        summary["fisher_sigma"] = {
            name: float(np.sqrt(max(np.diag(fisher)[i], 0.0))) for i, name in enumerate(JOINT_PARAMS)
        }
        summary["mcmc_over_fisher_sigma_fNL"] = float(
            summary["posterior"]["fNL"]["sigma68"] / max(summary["fisher_sigma"]["fNL"], 1.0e-300)
        )
        # Per-pole mean-shape PTE at MAP with Cmean = Csingle/25.
        prediction = model.evaluate(theta_map)[sl]
        pte: dict[str, Any] = {}
        for pole, name in ((0, "ell0"), (2, "ell2")):
            if pole == 2 and variant == "p0_control":
                continue
            offset = 0 if pole == 0 else 15
            block = cov30[sl, sl][offset:offset + 15, offset:offset + 15] / len(PHASES)
            r = data_v[offset:offset + 15] - prediction[offset:offset + 15]
            c2 = float(r @ np.linalg.solve(block, r))
            pte[name] = {"chi2_Cmean": c2, "dof": 11, "pte": float(chi2_distribution.sf(c2, 11))}
        summary["per_pole_mean_goodness"] = pte
        results[variant] = summary
        chains[variant] = chain
        out_npz = OUT_DIR / "fits" / variant / "samples.npz"
        if out_npz.exists():
            raise FileExistsError(f"immutable chain output exists: {out_npz}")
        atomic_savez(out_npz, chain_by_step=chain, log_probability_by_step=logp, prediction_map=prediction, data=data_v)
        print(json.dumps({"variant": variant, "fNL": summary["posterior"]["fNL"], "gates": summary["gates"], "pte": pte}, sort_keys=True), flush=True)
        if not all(summary["gates"].values()):
            raise SystemExit(f"MCMC gates failed for {variant}")

    control = results["p0_control"]
    increment = results["p02"]
    metrics = {
        "sigma_fNL": {"p0": control["posterior"]["fNL"]["sigma68"], "p02": increment["posterior"]["fNL"]["sigma68"]},
        "sigma_b1": {"p0": control["posterior"]["b1"]["sigma68"], "p02": increment["posterior"]["b1"]["sigma68"]},
        "sigma_sigma_s": {"p0": control["posterior"]["sigma_s"]["sigma68"], "p02": increment["posterior"]["sigma_s"]["sigma68"]},
        "improvement_fNL": float(1.0 - increment["posterior"]["fNL"]["sigma68"] / control["posterior"]["fNL"]["sigma68"]),
        "improvement_sigma_s": float(1.0 - increment["posterior"]["sigma_s"]["sigma68"] / control["posterior"]["sigma_s"]["sigma68"]),
        "crosscheck_pk_marginal_joint_experiment": {
            "note": "p0_control should reproduce the joint-experiment pk_marginal (-16.41 +- 36.44)",
            "p0_control": control["posterior"]["fNL"],
        },
    }
    audit = {
        "task": "task43_rsd_boxsafe_p02_increment",
        "status": "complete",
        "scope": (
            "boxsafe 0.4<zobs<0.8 RSD lightcone; x25 measured P0+P2 on the frozen 150-bin grid, "
            "15 DESI-PNG fit bins per pole; ph000 smooth geometry window (30, 981) = [P0 rows, P2 rows]; "
            "covariance ell0/ell2 blocks of the frozen 450x450 jaxpower matrix at the same bins"
        ),
        "caveats": [
            "diagnostic Gaussian covariance; the frozen RSD closure remains validation_failed",
            "P2 estimator contains window-leaked shot noise from the ell0 theory term (handled by the window rows)",
            "sn0 free on the ell0 theory only, as in the audited P0 model",
        ],
        "mcmc": {"nwalkers": int(args.nwalkers), "nsteps": int(args.nsteps), "burnin": int(args.burnin), "seed": int(args.seed)},
        "surrogate_validation": validation,
        "covariance": {
            "source": str(COV_NPZ),
            "source_sha256": sha256_file(COV_NPZ),
            "shape": [30, 30],
            "eigenvalue_min": float(eig[0]),
            "ell0_block_bridge_relfro": bridge_relfro,
        },
        "inputs": {
            "measure_dir": str(MEASURE_DIR),
            "payload": {"path": str(PAYLOAD_NPZ), "sha256": sha256_file(PAYLOAD_NPZ)},
            "zeff_conventions": zeff_note,
        },
        "results": results,
        "metrics": metrics,
        "cpu_affinity": cpus,
        "elapsed_sec": float(time.perf_counter() - started),
    }
    atomic_write_json(AUDIT_JSON, audit)
    print(json.dumps({"status": "complete", "metrics": metrics, "output": str(AUDIT_JSON)}, sort_keys=True))


if __name__ == "__main__":
    main()
