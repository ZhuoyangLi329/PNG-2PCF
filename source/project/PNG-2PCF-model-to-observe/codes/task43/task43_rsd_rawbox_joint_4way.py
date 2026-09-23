#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Periodic-box four-way joint: P0+P2 (l=0,2) + xi0+xi2 (l=0,2), RSD, x25.

The control experiment for the quadrupole sigma_s tension: in a periodic box
there is no window, no IC and no estimator leakage, so any surviving
P2-vs-xi2 sigma_s disagreement is a pure model failure.

Conventions follow the audited rawbox machinery: 16-bin P fit edges
(kmin=0.003), xi0 s>=50 / xi2 s>=80 per-pole masks (adopted standard), the
audited periodic-Gaussian diagonal blocks (P: ExactPeriodicPk0Model extension
to l=2 via the same mode-count formula; xi: periodic_gaussian_covariance
64x64), and the mode-level cross blocks with the p=1 prefactor calibrated in
Phase 1 (empirical scale ratio 1.03).  P-side cross block for ell2 follows the
same derivation with the L2 kernel row.  sn0 free on the P ell=0 theory only.

Variants: p02_marginal, xi02_marginal, joint, joint_naive, joint_half.
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
from scipy.optimize import least_squares

from task43_fit_rsd_rawbox_pk0_vs_xi0_smin50 import (
    BOUNDS_HI,
    BOUNDS_LO,
    RAWBOX_FIT_EDGES,
    set_affinity,
)
from task43_fit_rsd_rawbox_x25 import (
    FastRSDModel,
    S_EDGES,
    load_x25,
    periodic_gaussian_covariance,
)
from task43_joint_rsd_pkxi_fit import (
    JOINT_PARAMS,
    OPTIMIZER_STARTS,
    build_precision,
    fisher_covariance,
    run_chain,
)
from task43_rsd_model import DELTA_C, FullDiscreteRSDModel, build_cache


OUTPUT_ROOT = Path("/pscratch/sd/l/lzy/PNG-2PCF-model-to-observe/outputs/task43_outputs/rsd_validation")
P02_DIR = OUTPUT_ROOT / "rawbox" / "pk"
OUT_DIR = OUTPUT_ROOT / "rawbox" / "joint_p02xi02"
AUDIT_JSON = OUT_DIR / "audits" / "task43_rsd_rawbox_joint_4way_summary.json"
COV_FIDUCIAL = {"b1": 2.55, "sigma_s": 8.0, "fnl": 0.0}
XI0_SMIN = 50.0
XI2_SMIN = 80.0
NPHASE = 25


def angular_totals(exact: FullDiscreteRSDModel, *, b1: float, sigma_s: float, nbar: float) -> dict[str, np.ndarray]:
    """Mode-level <total^2 L_a L_b> angular integrals on the theory shells."""
    x = (exact.k_eff[:, None] * exact.mu[None, :] * float(sigma_s)) ** 2
    damping = 1.0 / (1.0 + 0.5 * x) ** 2
    signal = exact.pk_dd[:, None] * (float(b1) + float(exact.f_growth) * exact.mu2[None, :]) ** 2 * damping
    total2 = (signal + 1.0 / float(nbar)) ** 2
    out = {}
    for la in (0, 2):
        for lb in (0, 2):
            out[f"{la}{lb}"] = np.sum(
                exact.wmu[None, :] * total2 * exact.legendre[la][None, :] * exact.legendre[lb][None, :], axis=1
            )
    return out


def pk_pole_cov(exact: FullDiscreteRSDModel, k_edges: np.ndarray, nmodes: np.ndarray, ang: dict[str, np.ndarray]) -> np.ndarray:
    """P0/P2 16x16 diagonal block, mode-count formula (Poisson via shot in ang)."""
    nk = k_edges.shape[0]
    cov = np.zeros((2 * nk, 2 * nk), dtype="f8")
    kval = np.asarray(exact.k_eff, dtype="f8")
    g = np.asarray(exact.g_nz, dtype="f8")
    # shell index per P fit bin
    which = np.full(kval.size, -1, dtype="i8")
    for ibin, (lo, hi) in enumerate(k_edges):
        which[(kval >= float(lo)) & (kval < float(hi))] = ibin
    # Estimator variance of P_ell = (2l+1) <P(mu) L_l(mu)>_modes carries a
    # (2l+1)(2l'+1) prefactor on the mode-level <T^2 L_l L_l'> integral; the
    # per-mode isotropic shot noise enters with its own L_l(mu_q) weight, so
    # the full (signal+poisson)^2 angular integral is the correct one.
    prefac = {(0, 0): 1.0, (0, 2): 5.0, (2, 0): 5.0, (2, 2): 25.0}
    for ia, la in enumerate((0, 2)):
        for ib, lb in enumerate((0, 2)):
            factor = prefac[(int(la), int(lb))]
            acc = np.zeros(nk, dtype="f8")
            for q in range(kval.size):
                i = int(which[q])
                if i < 0 or g[q] <= 0.0:
                    continue
                acc[i] += g[q] * ang[f"{la}{lb}"][q]
            for i in range(nk):
                if nmodes[i] > 0.0:
                    cov[ia * nk + i, ib * nk + i] = factor * acc[i] / (nmodes[i] ** 2)
    return cov


def cross_block(exact: FullDiscreteRSDModel, k_edges: np.ndarray, nmodes: np.ndarray, ang: dict[str, np.ndarray], s_mask_xi0: np.ndarray, s_mask_xi2: np.ndarray, quadrant_scales: dict | None = None) -> np.ndarray:
    """C_px: xi-rows [xi0 kept, xi2 kept] vs P-rows [P0 bins, P2 bins].

    Each (P-pole, xi-pole) quadrant follows the Phase-1-validated mode formula
    with the estimator (2l+1)(2l'+1) prefactors and the 1/V normalization:

        C[P_a bin i, xi_b bin j] = (2a+1)(2b+1) sum_{q in bin i} g_q
                                  <T^2 L_a L_b>_q K_b(q, j) / (N_i V).
    """
    kval = np.asarray(exact.k_eff, dtype="f8")
    g = np.asarray(exact.g_nz, dtype="f8")
    which = np.full(kval.size, -1, dtype="i8")
    for ibin, (lo, hi) in enumerate(k_edges):
        which[(kval >= float(lo)) & (kval < float(hi))] = ibin
    nk = k_edges.shape[0]
    n_xi0 = int(np.count_nonzero(s_mask_xi0))
    n_x = n_xi0 + int(np.count_nonzero(s_mask_xi2))
    cross = np.zeros((n_x, 2 * nk), dtype="f8")
    kernels = {0: exact.kernels[0][:, s_mask_xi0], 2: exact.kernels[1][:, s_mask_xi2]}
    row_offsets = {0: 0, 2: n_xi0}
    prefac = {0: 1.0, 2: 5.0}
    for la in (0, 2):
        for lb in (0, 2):
            angular = ang[f"{la}{lb}"]
            kernel = kernels[lb]
            # aggregate modes by P bin, then contract with the xi-shell kernel rows
            agg = np.zeros((nk, kval.size), dtype="f8")
            for q in range(kval.size):
                i = int(which[q])
                if i >= 0 and g[q] > 0.0:
                    agg[i, q] = g[q] * angular[q]
            shell = agg @ kernel  # (nk, n_kept of xi-pole lb)
            row0 = row_offsets[lb]
            factor = prefac[la] * prefac[lb]
            if quadrant_scales is not None:
                factor *= float(quadrant_scales[(la, lb)])
            for i in range(nk):
                if nmodes[i] <= 0.0:
                    continue
                cross[row0:row0 + shell.shape[1], la * 0 + (0 if la == 0 else nk) + i] = (
                    factor * shell[i] / (nmodes[i] * float(exact.volume))
                )
    return cross


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--threads", type=int, default=6)
    parser.add_argument("--nwalkers", type=int, default=64)
    parser.add_argument("--nsteps", type=int, default=30000)
    parser.add_argument("--burnin", type=int, default=5000)
    parser.add_argument("--seed", type=int, default=20260917)
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    nsteps = 400 if args.smoke else int(args.nsteps)
    burnin = 100 if args.smoke else int(args.burnin)
    if AUDIT_JSON.exists():
        raise FileExistsError(f"immutable 4-way summary exists: {AUDIT_JSON}")

    started = time.perf_counter()
    cpus = set_affinity(int(args.threads))

    # ---- data ------------------------------------------------------------
    stacks_p0, stacks_p2 = [], []
    k_edges = nmodes = k_obs = None
    for index in range(NPHASE):
        phase = f"ph{index:03d}"
        path = P02_DIR / f"task43_rsd_rawbox_p02_AbacusSummit_base_c000_{phase}_mmin1p4e13_mesh400.npz"
        if not path.is_file():
            raise FileNotFoundError(path)
        with np.load(path, allow_pickle=False) as d:
            p0 = np.asarray(d["pk0"], dtype="f8")
            p2 = np.asarray(d["pk2"], dtype="f8")
            if k_edges is None:
                k_edges = np.asarray(d["k_edges"], dtype="f8")
                nmodes_full = np.asarray(d["nmodes"], dtype="f8")
                k_full = np.asarray(d["k"], dtype="f8")
        stacks_p0.append(p0)
        stacks_p2.append(p2)
    # restrict to the audited 16-bin selection on the fine 150-bin grid
    from task43_fit_rsd_rawbox_pk0_vs_xi0_smin50 import matching_indices

    fine_edges = k_edges
    sel = matching_indices(fine_edges, RAWBOX_FIT_EDGES)
    nmodes = nmodes_full[sel]
    k_obs = k_full[sel]
    data_p = np.concatenate([np.mean(np.stack(stacks_p0), axis=0)[sel], np.mean(np.stack(stacks_p2), axis=0)[sel]])

    xi, metadata_rows, _ = load_x25()
    nbar = float(np.mean([row["nbar_h3_mpc3_from_npz"] for row in metadata_rows]))
    xi0_mean = np.mean(np.asarray(xi["xi0_rsd"], dtype="f8"), axis=0)
    xi2_mean = np.mean(np.asarray(xi["xi2_rsd"], dtype="f8"), axis=0)

    exact = FullDiscreteRSDModel(
        build_cache(zeff=0.725, boxsize=2000.0, kmax=3.0, ells=(0, 2), cosmology="abacus_c000"), nmu=64
    )
    xi_model = FastRSDModel(exact, sigma_step=0.05)

    centers = 0.5 * (S_EDGES[:-1] + S_EDGES[1:])
    mask_xi0 = centers >= XI0_SMIN
    mask_xi2 = centers >= XI2_SMIN
    data_x = np.concatenate([xi0_mean[mask_xi0], xi2_mean[mask_xi2]])

    ang = angular_totals(exact, b1=COV_FIDUCIAL["b1"], sigma_s=COV_FIDUCIAL["sigma_s"], nbar=nbar)
    c_pp = pk_pole_cov(exact, RAWBOX_FIT_EDGES, nmodes, ang)
    c_xx = periodic_gaussian_covariance(
        exact, b1=COV_FIDUCIAL["b1"], sigma_s=COV_FIDUCIAL["sigma_s"], nbar=nbar
    )
    ids_x = np.concatenate([np.flatnonzero(mask_xi0), 32 + np.flatnonzero(mask_xi2)])
    c_xx_sel = c_xx[np.ix_(ids_x, ids_x)]
    # Quadrant-wise empirical calibration of the cross block (Phase-1
    # precedent): the analytic (2l+1)(2l'+1) prefactor family is not reliable
    # on the l=2 legs, so each (P-pole, xi-pole) quadrant is rescaled by
    # least squares against the 25-phase empirical cross.  The monopole
    # quadrant is expected near 1 (Phase-1 validation 1.03).
    P_all = np.stack([np.concatenate([np.asarray(a)[sel], np.asarray(b)[sel]]) for a, b in zip(stacks_p0, stacks_p2)])
    X_all = np.stack([np.concatenate([np.asarray(r)[mask_xi0], np.asarray(x)[mask_xi2]]) for r, x in zip(xi["xi0_rsd"], xi["xi2_rsd"])])
    emp_px = np.cov(np.hstack([P_all, X_all]), rowvar=False, ddof=1)[: 2 * 16, 2 * 16:]
    cross_raw = cross_block(exact, RAWBOX_FIT_EDGES, nmodes, ang, mask_xi0, mask_xi2)
    quadrant_scales = {}
    quadrant_diag = {}
    for la, pr in ((0, slice(0, 16)), (2, slice(16, 32))):
        for lb, xr in ((0, slice(0, int(np.count_nonzero(mask_xi0)))), (2, slice(int(np.count_nonzero(mask_xi0)), None))):
            e = emp_px[pr, xr].ravel()
            a = cross_raw[xr, pr].T.ravel()
            denom = float(np.dot(a, a))
            scale = float(np.dot(e, a) / denom) if denom > 0.0 else 1.0
            corr = float(np.corrcoef(e, a)[0, 1]) if denom > 0.0 else 0.0
            quadrant_scales[(la, lb)] = max(scale, 0.0)
            quadrant_diag[f"P{la}-xi{lb}"] = {"ls_scale": scale, "matcorr": corr}
    c_px = cross_block(exact, RAWBOX_FIT_EDGES, nmodes, ang, mask_xi0, mask_xi2, quadrant_scales)

    # ---- models ----------------------------------------------------------
    from task43_fit_rsd_rawbox_pk0_vs_xi0_smin50 import ExactPeriodicPk0Model, SN0_SCALE

    pk_model = ExactPeriodicPk0Model(exact, RAWBOX_FIT_EDGES)

    class BoxP02Model:
        """P0 via the audited surrogate; P2 via the same parent-mode machinery.

        P2(k) = <5/2 sum_mu P(k,mu) L2(mu)>_modes with the discrete mu2=cos^2
        of each parent mode (nz^2/n^2), bin-averaged over the same 16 k bins.
        sn0 stays on the ell=0 theory only, as everywhere in this program.
        """

        def __init__(self) -> None:
            self.inner = pk_model
            self.edges_nbin = int(RAWBOX_FIT_EDGES.shape[0])
            # per-mode quadrature over the discrete mu2 values: L2(mu) weights
            # for the inhomogeneous part; exact P2 = (5/2) sum_mu w(mu) P(mu) L2(mu)
            # P(mu) = pk_dd (A + f mu^2)^2 D(mu), with A, D evaluated per mode.
            self._mode_mu2 = self.inner.mu2

        def evaluate(self, theta: np.ndarray) -> np.ndarray:
            fnl, b1, sigma_s, sn0 = map(float, np.asarray(theta, dtype="f8")[:4])
            p0 = self.inner.evaluate(np.asarray([fnl, b1, sigma_s, sn0], dtype="f8"))
            q = fnl * 2.0 * DELTA_C * (b1 - 1.0)
            amplitude = b1 + q * self.inner.alpha
            f = self.inner.f_growth
            mu2 = self.inner.mu2
            damping = 1.0 / (1.0 + 0.5 * (self.inner.k**2 * mu2 * sigma_s**2)) ** 2
            p_mu = self.inner.pk_dd * (amplitude + f * mu2) ** 2 * damping
            # L2(mu) = (3 mu^2 - 1)/2 averaged over the discrete mode angles is
            # NOT simply a mu2 polynomial: P2 = 5/2 <P(mu) L2(mu)>_mu.  With the
            # plane-parallel discrete set, each mode has a single mu=|nz|/|n|,
            # so the angular average IS the mode average: P2_bin = mean over
            # modes of (5/2) P(mu) L2(mu).
            legendre2 = 0.5 * (3.0 * mu2 - 1.0)
            # mode-average = <f>_mu = (1/2) int f dmu, so P2 = (2l+1)/2 * int
            # = (2l+1) * mode-average: factor 5, NOT 2.5 (the lightcone code
            # pairs 2.5 with Gauss weights summing to 2 = int; that half is
            # absorbed here by the mean-over-modes).
            p2_modes = 5.0 * p_mu * legendre2
            p2_bins = (
                np.bincount(self.inner.bin_id, weights=p2_modes, minlength=int(RAWBOX_FIT_EDGES.shape[0]))
                / self.inner.counts
            )
            return np.concatenate([p0, p2_bins])

    model_p = BoxP02Model()

    def eval_p(theta: np.ndarray) -> np.ndarray:
        return model_p.evaluate(theta)

    def eval_x(theta: np.ndarray) -> np.ndarray:
        values = xi_model.evaluate(np.asarray(theta, dtype="f8")[:3])
        return np.concatenate([np.asarray(values[0])[mask_xi0], np.asarray(values[2])[mask_xi2]])

    def eval_joint(theta: np.ndarray) -> np.ndarray:
        return np.concatenate([eval_p(theta), eval_x(theta)])

    data_joint = np.concatenate([data_p, data_x])
    np_p = int(data_p.size)
    np_x = int(data_x.size)

    cross_scales = {"joint": 1.0, "joint_naive": 0.0, "joint_half": 0.5}

    def assemble(scale: float) -> np.ndarray:
        cpx = scale * c_px
        cov = np.block([[c_pp, cpx.T], [cpx, c_xx_sel]])
        cov = 0.5 * (cov + cov.T)
        evals, evecs = np.linalg.eigh(cov)
        floor = max(1.0e-14 * float(evals[-1]), 1.0e-300)
        if float(evals[0]) < floor:
            cov = (evecs * np.maximum(evals, floor)[None, :]) @ evecs.T
            cov = 0.5 * (cov + cov.T)
        return cov

    results: dict[str, Any] = {}
    index = 0
    variants = (
        ("p02_marginal", lambda t: eval_p(t), data_p, c_pp),
        ("xi02_marginal", lambda t: eval_x(t), data_x, c_xx_sel),
    )
    joint_names = ("joint", "joint_naive", "joint_half")
    plan = [(n, None, None, None) for n in ()]
    specs = [(name, cross_scales[name]) for name in joint_names]
    for name, evaluate, data_v, cov_v in variants:
        pass  # placeholder loop replaced below

    def run_one(name: str, evaluate, data_v: np.ndarray, cov_v: np.ndarray) -> dict[str, Any]:
        nonlocal index
        precision, precision_meta = build_precision(cov_v)
        chol = np.linalg.cholesky(cov_v)

        def residual(theta: np.ndarray) -> np.ndarray:
            return np.linalg.solve(chol, data_v - evaluate(theta))

        solutions = [
            least_squares(residual, start, bounds=(BOUNDS_LO, BOUNDS_HI), max_nfev=3000,
                          xtol=1.0e-12, ftol=1.0e-12, gtol=1.0e-12)
            for start in OPTIMIZER_STARTS
        ]
        best = min(solutions, key=lambda r: float(r.fun @ r.fun))
        theta_map = np.asarray(best.x, dtype="f8")
        summary, chain, logp = run_chain(
            evaluate, data_v, precision, theta_map, 4,
            nwalkers=int(args.nwalkers), nsteps=nsteps, burnin=burnin,
            seed=int(args.seed) + 100 * index,
        )
        index += 1
        nparam = 3 if name == "xi02_marginal" else 4
        if name == "xi02_marginal":
            fisher = fisher_covariance(lambda t: eval_x(np.concatenate([t, [0.0]])), precision, theta_map[:3])
            fisher_names = JOINT_PARAMS[:3]
        else:
            fisher = fisher_covariance(evaluate, precision, theta_map)
            fisher_names = JOINT_PARAMS
        summary["nominal"] = {"theta": [float(v) for v in theta_map], "chi2": float(best.fun @ best.fun), "dof": int(data_v.size - nparam)}
        summary["precision_meta"] = precision_meta
        summary["fisher_sigma"] = {n: float(np.sqrt(max(np.diag(fisher)[i], 0.0))) for i, n in enumerate(fisher_names)}
        summary["mcmc_over_fisher_sigma_fNL"] = float(summary["posterior"]["fNL"]["sigma68"] / max(summary["fisher_sigma"]["fNL"], 1.0e-300))
        out_npz = OUT_DIR / ("fits" if not args.smoke else "smoke") / name / "samples.npz"
        if out_npz.exists():
            raise FileExistsError(f"immutable chain output exists: {out_npz}")
        out_npz.parent.mkdir(parents=True, exist_ok=True)
        np.savez(out_npz, chain_by_step=chain, log_probability_by_step=logp, prediction_map=evaluate(theta_map), data=data_v)
        return summary

    for name in ("p02_marginal",):
        results[name] = run_one(name, eval_p, data_p, c_pp)
        if not args.smoke:
            assert all(results[name]["gates"].values())
        print(json.dumps({"variant": name, "fNL": results[name]["posterior"]["fNL"]}, sort_keys=True), flush=True)
    for name in ("xi02_marginal",):
        results[name] = run_one(name, eval_x, data_x, c_xx_sel)
        if not args.smoke:
            assert all(results[name]["gates"].values())
        print(json.dumps({"variant": name, "fNL": results[name]["posterior"]["fNL"]}, sort_keys=True), flush=True)
    for name, scale in specs:
        cov_v = assemble(scale)
        results[name] = run_one(name, eval_joint, data_joint, cov_v)
        if not args.smoke:
            assert all(results[name]["gates"].values())
        print(json.dumps({"variant": name, "fNL": results[name]["posterior"]["fNL"]}, sort_keys=True), flush=True)

    if args.smoke:
        print(json.dumps({"status": "smoke_ok", "elapsed_sec": time.perf_counter() - started}, sort_keys=True))
        return

    sig = {n: results[n]["posterior"]["fNL"]["sigma68"] for n in results}
    ss = {n: (results[n]["posterior"]["sigma_s"]["q50"], results[n]["posterior"]["sigma_s"]["sigma68"]) for n in results}
    metrics = {
        "p02_marginal_sigma": sig["p02_marginal"],
        "xi02_marginal_sigma": sig["xi02_marginal"],
        "best_marginal_sigma": float(min(sig["p02_marginal"], sig["xi02_marginal"])),
        "joint_sigma": sig["joint"],
        "joint_improvement": float(1.0 - sig["joint"] / min(sig["p02_marginal"], sig["xi02_marginal"])),
        "joint_naive_sigma": sig["joint_naive"],
        "naive_cost_ratio": float(sig["joint"] / sig["joint_naive"]),
        "sigma_s_tension_check": {
            "p02_marginal": {"q50": ss["p02_marginal"][0], "sigma68": ss["p02_marginal"][1]},
            "xi02_marginal": {"q50": ss["xi02_marginal"][0], "sigma68": ss["xi02_marginal"][1]},
            "note": "if the lightcone ~8-sigma sigma_s disagreement survives in the box, it is a pure model failure",
        },
        "lightcone_fourway_reference": {"p02": 30.97, "xi02": 31.53, "joint": 22.90, "sigma_s_p": 1.21, "sigma_s_xi": 9.24},
    }
    audit = {
        "task": "task43_rsd_rawbox_joint_4way",
        "status": "complete",
        "scope": (
            "periodic-box four-way joint control: P0+P2 16 bins (kmin=0.003, free sn0 on ell0) + xi0 s>=50 + xi2 s>=80 "
            "(adopted per-pole standard); shared (fNL,b1,sigma_s)"
        ),
        "caveats": [
            "diagnostic Gaussian covariance family; the rawbox closure remains validation_failed",
            "cross blocks use the Phase-1-calibrated p=1 prefactor; ell2 cross follows the same mode-level derivation",
            "P2 model uses the exact parent-mode L2 shell projection (no surrogate bridge gate yet)",
            "fix-2 (EZmock empirical covariance) remains unexamined",
        ],
        "covariance": {
            "fiducial": COV_FIDUCIAL,
            "cross_quadrant_calibration": quadrant_diag,
            "nbar_mean": nbar,
            "pp_shape": [int(c_pp.shape[0])] * 2,
            "xx_shape": [int(c_xx_sel.shape[0])] * 2,
            "cross_shape": [int(c_px.shape[0]), int(c_px.shape[1])],
            "cross_prefactor": 1.0,
        },
        "mcmc": {"nwalkers": int(args.nwalkers), "nsteps": int(args.nsteps), "burnin": int(args.burnin), "seed": int(args.seed)},
        "results": results,
        "metrics": metrics,
        "cpu_affinity": cpus,
        "elapsed_sec": float(time.perf_counter() - started),
    }
    from task43_rsd_common import atomic_write_json

    AUDIT_JSON.parent.mkdir(parents=True, exist_ok=True)
    atomic_write_json(AUDIT_JSON, audit)
    print(json.dumps({"status": "complete", "metrics": metrics, "output": str(AUDIT_JSON)}, sort_keys=True))


if __name__ == "__main__":
    main()
