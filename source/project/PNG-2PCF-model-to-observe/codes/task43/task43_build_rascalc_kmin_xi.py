#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Build matched low-k correlation-function inputs for Task43 RascalC.

This file deliberately separates three notions that were mixed in earlier
Task43/Task44 diagnostics:

``full_discrete``
    The primary mock-closure target.  It sums the non-zero Fourier modes of the
    2 Gpc/h Abacus mother box, including their integer-lattice degeneracies.
``continuous_boxcut``
    A continuous-Hankel approximation with k >= 2 pi / L_box.
``continuous_lowk``
    An intentionally infinite-volume-like control with k >= 1e-4 h/Mpc.

RascalC has no standalone ``kmin`` option.  Its clustering-mode support enters
through the supplied xi(s, mu), so every covariance run must record exactly
which of these tables it used.  The output uses theoretical point samples
(the RascalC three-array interface), not coarse 50--350 Mpc/h bin averages.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any

for _name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_name, "1")

import numpy as np

from task43_config import BOX_SIZE, K_FUND, SUMMARY_DIR
from task43_fkp_zeff import load_fkp_summary
from task43_theory_template import COSMOLOGY_CHOICES, DEFAULT_COSMOLOGY, build_template_arrays, load_task41


DEFAULT_OUTPUT = SUMMARY_DIR.parent / "rascalc_covariance" / "inputs" / "task43_rascalc_xi_kmin_L2000_fnl0_b1cov2p5"


def _jsonable(value: Any) -> Any:
    """Convert numpy/path values to plain JSON values."""
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, Path):
        return str(value)
    return value


def zeff_for_p0(path: Path, p0: float) -> float:
    """Read the Task43 all-random-auto effective redshift for one FKP P0."""
    summary = load_fkp_summary(path)
    p0_values = np.asarray(summary["p0_values"], dtype="f8")
    match = np.flatnonzero(np.isclose(p0_values, float(p0), rtol=0.0, atol=1.0e-10))
    if match.size != 1:
        raise ValueError(f"P0={p0} is not uniquely present in {path}")
    return float(np.asarray(summary["zeff_random_auto"], dtype="f8")[int(match[0])])


def radial_samples(rmax: float, dr: float) -> np.ndarray:
    """Return a near-zero plus linearly spaced theoretical xi interpolation grid."""
    # RascalC recommends that its correlation table reach very close to zero.
    # The small logarithmic prefix avoids asking the interpolator to invent the
    # 0--1 Mpc/h behavior from a first sample at several Mpc/h.
    prefix = np.array([1.0e-3, 2.0e-3, 5.0e-3, 1.0e-2, 2.0e-2, 5.0e-2, 0.1, 0.2, 0.5], dtype="f8")
    linear = np.arange(1.0, float(rmax) + 0.5 * float(dr), float(dr), dtype="f8")
    r = np.unique(np.concatenate([prefix[prefix < linear[0]], linear]))
    if r[-1] < float(rmax):
        r = np.append(r, float(rmax))
    return r


def j0(x: np.ndarray) -> np.ndarray:
    """Stable spherical Bessel j0(x) = sin(x) / x."""
    return np.sinc(np.asarray(x, dtype="f8") / np.pi)


def trapz_weights(x: np.ndarray) -> np.ndarray:
    """One-dimensional trapezoid-rule weights for a strictly increasing grid."""
    x = np.asarray(x, dtype="f8")
    if x.ndim != 1 or x.size < 2 or np.any(np.diff(x) <= 0.0):
        raise ValueError("integration grid must be a strictly increasing 1D array")
    weight = np.empty_like(x)
    weight[0] = 0.5 * (x[1] - x[0])
    weight[-1] = 0.5 * (x[-1] - x[-2])
    weight[1:-1] = 0.5 * (x[2:] - x[:-2])
    return weight


def xi_from_weighted_modes(k: np.ndarray, mode_weight: np.ndarray, r: np.ndarray, *, batch: int = 64) -> np.ndarray:
    """Evaluate sum_k mode_weight(k) j0(k r) without a large permanent matrix."""
    k = np.asarray(k, dtype="f8")
    weight = np.asarray(mode_weight, dtype="f8")
    r = np.asarray(r, dtype="f8")
    if k.shape != weight.shape:
        raise ValueError(f"k and mode_weight shapes differ: {k.shape} vs {weight.shape}")
    out = np.empty_like(r)
    for start in range(0, r.size, int(batch)):
        stop = min(start + int(batch), r.size)
        out[start:stop] = weight @ j0(np.outer(k, r[start:stop]))
    return out


def theory_pk(
    k: np.ndarray,
    *,
    zeff: float,
    b1_cov: float,
    fnl_cov: float,
    p_fixed: float,
    sn0_fixed: float,
    cosmology: str,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Evaluate the same real-space PNG halo P(k) used by Task43 fits."""
    task41 = load_task41()
    k = np.asarray(k, dtype="f8")
    template_k = np.geomspace(min(1.0e-5, float(np.min(k)) / 2.0), max(1.0, float(np.max(k)) * 1.2), 5000)
    template, cosmology_meta = build_template_arrays(task41, template_k, z=float(zeff), cosmology=str(cosmology))
    pk = task41.evaluate_realspace_png_pk(
        k,
        template,
        fnl_loc=float(fnl_cov),
        b1=float(b1_cov),
        p_fixed=float(p_fixed),
        sn0=float(sn0_fixed),
    )
    return np.asarray(pk, dtype="f8"), cosmology_meta


def build_full_discrete(
    r: np.ndarray,
    *,
    boxsize: float,
    kmax: float,
    ndense: int,
    zeff: float,
    b1_cov: float,
    fnl_cov: float,
    p_fixed: float,
    sn0_fixed: float,
    cosmology: str,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Sum the non-zero integer-lattice modes of a cubic mother box."""
    task41 = load_task41()
    kfund = 2.0 * np.pi / float(boxsize)
    qmax = int(np.ceil((float(kmax) / kfund) ** 2))
    nmax = int(np.ceil(np.sqrt(qmax)))
    gq = task41.gq_fft(qmax, nmax)
    degeneracy, k_eff = task41.precompute_rebin_cache(gq, kfund, float(kmax), dk_factor=0.1)
    k_dense = np.geomspace(max(1.0e-5, kfund / 5.0), float(kmax), int(ndense))
    pk_dense, cosmology_meta = theory_pk(
        k_dense,
        zeff=zeff,
        b1_cov=b1_cov,
        fnl_cov=fnl_cov,
        p_fixed=p_fixed,
        sn0_fixed=sn0_fixed,
        cosmology=cosmology,
    )
    pk_eff = task41.interp_logk(k_eff, k_dense, pk_dense)
    volume = float(boxsize) ** 3
    xi = xi_from_weighted_modes(k_eff, degeneracy * pk_eff / volume, r)
    meta = {
        "mode": "full_discrete",
        "formula": "xi(r)=V_box^-1 sum_{n in Z^3, n!=0} P(k_n) j0(k_n r)",
        "boxsize": float(boxsize),
        "volume": volume,
        "kfund": kfund,
        "k_first": float(k_eff[0]),
        "k_last": float(k_eff[-1]),
        "n_unique_mode_radii": int(k_eff.size),
        "first_mode_degeneracy": float(degeneracy[0]),
        "sum_mode_degeneracy": float(np.sum(degeneracy)),
        "zero_mode_included": False,
        "kmax": float(kmax),
        "ndense": int(ndense),
        "cosmology_meta": cosmology_meta,
    }
    return xi, meta


def build_continuous(
    r: np.ndarray,
    *,
    kmin: float,
    kmax: float,
    nk: int,
    zeff: float,
    b1_cov: float,
    fnl_cov: float,
    p_fixed: float,
    sn0_fixed: float,
    cosmology: str,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Evaluate a continuous Hankel transform with an explicit lower limit."""
    k = np.linspace(float(kmin), float(kmax), int(nk), dtype="f8")
    pk, cosmology_meta = theory_pk(
        k,
        zeff=zeff,
        b1_cov=b1_cov,
        fnl_cov=fnl_cov,
        p_fixed=p_fixed,
        sn0_fixed=sn0_fixed,
        cosmology=cosmology,
    )
    weight = trapz_weights(k) * k**2 * pk / (2.0 * np.pi**2)
    xi = xi_from_weighted_modes(k, weight, r)
    meta = {
        "mode": "continuous_hankel",
        "formula": "xi(r)=int_kmin^kmax dk k^2 P(k) j0(kr)/(2 pi^2)",
        "kmin": float(kmin),
        "kmax": float(kmax),
        "nk": int(nk),
        "cosmology_meta": cosmology_meta,
    }
    return xi, meta


def range_diagnostics(reference: np.ndarray, other: np.ndarray, r: np.ndarray, lo: float, hi: float) -> dict[str, float]:
    """Summarize absolute and scale-aware differences on one radial interval."""
    mask = (r >= float(lo)) & (r <= float(hi))
    delta = np.asarray(other)[mask] - np.asarray(reference)[mask]
    scale = max(float(np.max(np.abs(np.asarray(reference)[mask]))), 1.0e-30)
    return {
        "rmin": float(lo),
        "rmax": float(hi),
        "max_abs_difference": float(np.max(np.abs(delta))),
        "rms_difference": float(np.sqrt(np.mean(delta**2))),
        "max_abs_difference_over_reference_peak": float(np.max(np.abs(delta)) / scale),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--fkp-summary", type=Path, default=SUMMARY_DIR / "task43_fkp_zeff_mmin1p4e13_x25.npz")
    parser.add_argument("--output-prefix", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--p0", type=float, default=10000.0)
    parser.add_argument("--boxsize", type=float, default=BOX_SIZE)
    parser.add_argument("--rmax", type=float, default=600.0)
    parser.add_argument("--dr", type=float, default=1.0)
    parser.add_argument("--nmu", type=int, default=10)
    parser.add_argument("--kmax", type=float, default=5.0)
    parser.add_argument("--nk-continuous", type=int, default=120000)
    parser.add_argument("--ndense", type=int, default=60000)
    parser.add_argument("--low-kmin", type=float, default=1.0e-4)
    parser.add_argument("--b1-cov", type=float, default=2.5)
    parser.add_argument("--fnl-cov", type=float, default=0.0)
    parser.add_argument("--p-fixed", type=float, default=1.0)
    parser.add_argument("--sn0-fixed", type=float, default=0.0)
    parser.add_argument("--cosmology", choices=COSMOLOGY_CHOICES, default=DEFAULT_COSMOLOGY)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    if not args.fkp_summary.exists():
        raise FileNotFoundError(f"missing FKP summary: {args.fkp_summary}")
    if int(args.nmu) < 1:
        raise ValueError("--nmu must be positive")
    out_npz = args.output_prefix.with_suffix(".npz")
    out_json = args.output_prefix.with_suffix(".json")
    if out_npz.exists() and out_json.exists() and not args.overwrite:
        print(f"[skip] existing {out_npz}")
        return

    zeff = zeff_for_p0(args.fkp_summary, float(args.p0))
    r = radial_samples(float(args.rmax), float(args.dr))
    mu = (np.arange(int(args.nmu), dtype="f8") + 0.5) / float(args.nmu)
    kfund = 2.0 * np.pi / float(args.boxsize)

    print(f"[setup] Lbox={args.boxsize:g} kfund={kfund:.9f} zeff={zeff:.9f} nr={r.size}", flush=True)
    xi_discrete, meta_discrete = build_full_discrete(
        r,
        boxsize=float(args.boxsize),
        kmax=float(args.kmax),
        ndense=int(args.ndense),
        zeff=zeff,
        b1_cov=float(args.b1_cov),
        fnl_cov=float(args.fnl_cov),
        p_fixed=float(args.p_fixed),
        sn0_fixed=float(args.sn0_fixed),
        cosmology=str(args.cosmology),
    )
    print("[xi] full_discrete done", flush=True)
    xi_boxcut, meta_boxcut = build_continuous(
        r,
        kmin=kfund,
        kmax=float(args.kmax),
        nk=int(args.nk_continuous),
        zeff=zeff,
        b1_cov=float(args.b1_cov),
        fnl_cov=float(args.fnl_cov),
        p_fixed=float(args.p_fixed),
        sn0_fixed=float(args.sn0_fixed),
        cosmology=str(args.cosmology),
    )
    print("[xi] continuous_boxcut done", flush=True)
    xi_lowk, meta_lowk = build_continuous(
        r,
        kmin=float(args.low_kmin),
        kmax=float(args.kmax),
        nk=int(args.nk_continuous),
        zeff=zeff,
        b1_cov=float(args.b1_cov),
        fnl_cov=float(args.fnl_cov),
        p_fixed=float(args.p_fixed),
        sn0_fixed=float(args.sn0_fixed),
        cosmology=str(args.cosmology),
    )
    print("[xi] continuous_lowk done", flush=True)

    diagnostics = {
        "full_discrete_vs_continuous_boxcut_50_350": range_diagnostics(xi_discrete, xi_boxcut, r, 50.0, 350.0),
        "full_discrete_vs_continuous_lowk_50_350": range_diagnostics(xi_discrete, xi_lowk, r, 50.0, 350.0),
        "continuous_boxcut_vs_lowk_50_350": range_diagnostics(xi_boxcut, xi_lowk, r, 50.0, 350.0),
    }
    meta = {
        "status": "done",
        "task": "task43_build_rascalc_kmin_xi",
        "purpose": "matched FullDiscrete/continuous-boxcut/low-k xi inputs for RascalC covariance closure",
        "primary_mock_closure_model": "full_discrete",
        "warning": "These are linear-theory halo xi tables; a measured/hybrid small-scale xi input remains a separate RascalC systematic A/B.",
        "output_npz": str(out_npz),
        "fkp_summary": str(args.fkp_summary),
        "p0": float(args.p0),
        "zeff": zeff,
        "theory": {
            "b1_cov": float(args.b1_cov),
            "fnl_cov": float(args.fnl_cov),
            "p_fixed": float(args.p_fixed),
            "sn0_fixed": float(args.sn0_fixed),
            "cosmology": str(args.cosmology),
        },
        "radial_grid": {"rmin": float(r[0]), "rmax": float(r[-1]), "dr_linear": float(args.dr), "nr": int(r.size)},
        "mu_grid": {"nmu": int(mu.size), "mu_min": float(mu[0]), "mu_max": float(mu[-1])},
        "models": {
            "full_discrete": meta_discrete,
            "continuous_boxcut": meta_boxcut,
            "continuous_lowk": meta_lowk,
        },
        "diagnostics": diagnostics,
    }
    out_npz.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        out_npz,
        r=r,
        mu=mu,
        xi_full_discrete=xi_discrete,
        xi_continuous_boxcut=xi_boxcut,
        xi_continuous_lowk=xi_lowk,
        boxsize=np.array(float(args.boxsize), dtype="f8"),
        kfund=np.array(kfund, dtype="f8"),
        zeff=np.array(zeff, dtype="f8"),
        p0=np.array(float(args.p0), dtype="f8"),
        meta_json=np.asarray(json.dumps(_jsonable(meta), sort_keys=True)),
    )
    out_json.write_text(json.dumps(_jsonable(meta), indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"[done] wrote {out_npz}")
    print(f"[check] kfund_config={K_FUND:.9f} kfund_run={kfund:.9f}")


if __name__ == "__main__":
    main()
