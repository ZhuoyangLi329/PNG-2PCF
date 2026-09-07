#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Quijote LCp50 1Gpc halo validation trial.

This script fits:
1. P(k) BinAvgFit reference using kcen <= 0.08 h/Mpc.
2. 2PCF profiler using r-bin edge cuts rmin-350 Mpc/h.
   The default 2PCF covariance is C_sample for a single-box constraint test;
   C_sample / Nmock is available only as an ensemble-mean residual diagnostic.

Main convention follows the old Quijote Mission10 validation:
- Lbox = 1000 Mpc/h
- z = 1
- PNGTracerPowerSpectrumMultipoles, mode="b-p"
- P(k) reference free parameters: fnl_loc, b1, sigmas, optionally sn0
- 2PCF profiler free parameters: fnl_loc, b1, optionally sigmas/sn0
- fixed parameter: p = 1.2
"""

from __future__ import annotations

import argparse
import json
import math
import re
import time
from pathlib import Path

import numpy as np
from iminuit import Minuit
from scipy.fft import irfft, next_fast_len, rfft

from cosmoprimo import Cosmology
from desilike.theories.galaxy_clustering import FixedPowerSpectrumTemplate, PNGTracerPowerSpectrumMultipoles


PROJECT_ROOT = Path("/pscratch/sd/l/lzy/PNG-2PCF-model-to-observe")
DATA_DIR = Path("/pscratch/sd/l/lzy/pks_2pcfs")
OUTPUT_DIR = PROJECT_ROOT / "outputs" / "task4_outputs" / "quijote_lcp50_profiler"

LBOX = 1000.0
VOLUME = LBOX**3
K_FUND = 2.0 * np.pi / LBOX
Z = 1.0
P_FIXED = 1.2
SN0_FIXED = 0.0
SN0_LIMIT = (-1.0e6, 1.0e6)
KMAX_FIT = 0.08
KMAX_DISCRETE = 15.0
REBIN_DK_FACTOR = 0.1
N_PK_PARAMS = 3
N_XI_PARAMS_FIXED_SIGMAS = 2
N_XI_PARAMS_FREE_SIGMAS = 3
N_XI_PARAMS_FREE_SIGMAS_SN0 = 4


def rid_from_path(path: Path) -> int:
    """Extract trailing realization id."""
    match = re.search(r"_(\d+)\.[^.]+$", path.name)
    if not match:
        raise ValueError(f"cannot parse realization id from {path}")
    return int(match.group(1))


def sorted_files(pattern: str) -> list[Path]:
    """Return files sorted by realization id."""
    files = sorted(DATA_DIR.glob(pattern), key=rid_from_path)
    if not files:
        raise FileNotFoundError(f"no files match {DATA_DIR / pattern}")
    return files


def load_pk_stack(tag: str) -> dict[str, np.ndarray]:
    """Load Quijote P(k) stack for one tag."""
    files = sorted_files(f"pk_{tag}_*.txt")
    ref = np.loadtxt(files[0], comments="#")
    valid = ref[:, 4] > 0
    kcen = ref[valid, 0].astype("f8")
    kmin = ref[valid, 1].astype("f8")
    kmax = ref[valid, 2].astype("f8")
    nmod = ref[valid, 4].astype("f8")
    mocks = []
    realizations = []
    for path in files:
        data = np.loadtxt(path, comments="#")
        mocks.append(data[valid, 5].astype("f8"))
        realizations.append(rid_from_path(path))
    mocks_arr = np.asarray(mocks, dtype="f8")
    return {
        "files": np.array([str(path) for path in files]),
        "realizations": np.asarray(realizations, dtype="i8"),
        "kcen": kcen,
        "kmin": kmin,
        "kmax": kmax,
        "nmod": nmod,
        "mocks": mocks_arr,
        "mean": mocks_arr.mean(axis=0),
        "cov": np.cov(mocks_arr, rowvar=False, ddof=1),
    }


def load_pcf_stack(tag: str) -> dict[str, np.ndarray]:
    """Load Quijote 2PCF stack for one tag."""
    files = sorted_files(f"pcf_{tag}_*.dat")
    ref = np.loadtxt(files[0], comments="#")
    s = ref[:, 0].astype("f8")
    smin = ref[:, 1].astype("f8")
    smax = ref[:, 2].astype("f8")
    mocks = []
    realizations = []
    for path in files:
        data = np.loadtxt(path, comments="#")
        if data.shape != ref.shape or not np.allclose(data[:, :3], ref[:, :3], rtol=0.0, atol=0.0):
            raise ValueError(f"PCF grid mismatch in {path}")
        mocks.append(data[:, 3].astype("f8"))
        realizations.append(rid_from_path(path))
    mocks_arr = np.asarray(mocks, dtype="f8")
    return {
        "files": np.array([str(path) for path in files]),
        "realizations": np.asarray(realizations, dtype="i8"),
        "s": s,
        "smin": smin,
        "smax": smax,
        "mocks": mocks_arr,
        "mean": mocks_arr.mean(axis=0),
        "cov": np.cov(mocks_arr, rowvar=False, ddof=1),
    }


def build_cosmology() -> Cosmology:
    """Cosmology used in the old validation scripts."""
    return Cosmology(
        h=0.6711,
        Omega_b=0.049,
        Omega_cdm=0.3175 - 0.049,
        sigma8=0.834,
        n_s=0.9624,
        engine="class",
    )


def gq_enumerate(qmax: int) -> np.ndarray:
    """Direct shell degeneracy enumeration for small q."""
    nmax = int(np.ceil(np.sqrt(qmax))) + 1
    gq = np.zeros(qmax + 1, dtype=np.int64)
    for nx in range(-nmax, nmax + 1):
        for ny in range(-nmax, nmax + 1):
            for nz in range(-nmax, nmax + 1):
                if nx == 0 and ny == 0 and nz == 0:
                    continue
                q = nx * nx + ny * ny + nz * nz
                if q <= qmax:
                    gq[q] += 1
    return gq


def gq_fft(qmax: int, nmax: int) -> np.ndarray:
    """FFT shell degeneracy computation for large q."""
    a = np.zeros(qmax + 1, dtype=np.float32)
    a[0] = 1.0
    sq = np.arange(1, nmax + 1, dtype=np.int64) ** 2
    a[sq[sq <= qmax]] = 2.0
    nfft = next_fast_len(3 * qmax + 1)
    fa = rfft(a, n=nfft)
    return np.rint(irfft(fa * fa * fa, n=nfft)[: qmax + 1]).astype(np.int64)


def shell_arrays(gq: np.ndarray, kmax: float) -> tuple[np.ndarray, np.ndarray]:
    """Return shell k and degeneracy up to kmax."""
    qnz = np.nonzero(gq[1:])[0] + 1
    k = K_FUND * np.sqrt(qnz.astype("f8"))
    mask = k <= kmax
    q = qnz[mask]
    return k[mask], gq[q].astype("f8")


def bin_shell_indices(k_shell: np.ndarray, kmin: np.ndarray, kmax: np.ndarray) -> list[np.ndarray]:
    """Map each measured P(k) bin to discrete shell indices."""
    return [np.nonzero((k_shell >= lo) & (k_shell < hi))[0] for lo, hi in zip(kmin, kmax)]


def binavg_pk(pk_shell: np.ndarray, g_shell: np.ndarray, indices: list[np.ndarray]) -> np.ndarray:
    """Compute shell-degeneracy weighted bin-average P(k)."""
    out = np.zeros(len(indices), dtype="f8")
    for i, idx in enumerate(indices):
        if idx.size == 0:
            out[i] = np.nan
            continue
        weights = g_shell[idx]
        out[i] = float(np.sum(weights * pk_shell[idx]) / np.sum(weights))
    return out


def covariance_corrections(nmock: int, ndata: int, nparams: int) -> dict[str, float]:
    """Hartlap and Percival corrections."""
    hartlap = (nmock - ndata - 2.0) / (nmock - 1.0)
    a = 2.0 / ((nmock - ndata - 1.0) * (nmock - ndata - 4.0))
    b = (nmock - ndata - 2.0) / ((nmock - ndata - 1.0) * (nmock - ndata - 4.0))
    m1 = (1.0 + b * (ndata - nparams)) / (1.0 + a + b * (nparams + 1.0))
    return {
        "hartlap": float(hartlap),
        "percival_error_factor": float(math.sqrt(m1)),
        "percival_m1_variance": float(m1),
        "A": float(a),
        "B": float(b),
    }


class RSDModel:
    """Reusable desilike RSD PNG monopole evaluator."""

    def __init__(self, cosmo: Cosmology, k: np.ndarray, *, sigmas_init: float = 0.0):
        template = FixedPowerSpectrumTemplate(z=Z, fiducial=cosmo)
        self.theory = PNGTracerPowerSpectrumMultipoles(k=k, template=template, mode="b-p")
        self.theory.init.params["p"].update(fixed=True, value=P_FIXED)
        self.theory.init.params["sn0"].update(fixed=False, value=SN0_FIXED)
        self.theory.init.params["sigmas"].update(fixed=False, value=float(sigmas_init))

    def pk0(self, *, fnl_loc: float, b1: float, sigmas: float, sn0: float = SN0_FIXED) -> np.ndarray:
        """Evaluate P0(k)."""
        self.theory(
            fnl_loc=float(fnl_loc),
            b1=float(b1),
            sigmas=float(sigmas),
            p=P_FIXED,
            sn0=float(sn0),
        )
        return np.asarray(self.theory.power[0], dtype="f8")


def fit_pk_reference(
    pk_data: dict[str, np.ndarray],
    pk_cov: dict[str, np.ndarray],
    cosmo: Cosmology,
    *,
    pk_sn0_policy: str,
) -> dict[str, object]:
    """Fit center and BinAvg P(k) references with kcen <= KMAX_FIT."""
    if pk_sn0_policy not in {"fixed", "free"}:
        raise ValueError(f"unknown pk_sn0_policy={pk_sn0_policy}")

    fit_mask = pk_data["kcen"] <= KMAX_FIT
    kcen = pk_data["kcen"][fit_mask]
    kmin = pk_data["kmin"][fit_mask]
    kmax = pk_data["kmax"][fit_mask]
    y = pk_data["mean"][fit_mask]
    cov = pk_cov["cov"][np.ix_(fit_mask, fit_mask)]
    cov_inv = np.linalg.pinv(cov, rcond=1e-10)
    n_pk_params = 4 if pk_sn0_policy == "free" else 3

    center_model = RSDModel(cosmo, kcen, sigmas_init=0.1)

    if pk_sn0_policy == "fixed":
        def chi2_center(fnl_loc: float, b1: float, sigmas: float) -> float:
            diff = y - center_model.pk0(fnl_loc=fnl_loc, b1=b1, sigmas=sigmas, sn0=SN0_FIXED)
            return float(diff @ cov_inv @ diff)

        m_center = Minuit(chi2_center, fnl_loc=50.0, b1=2.72, sigmas=0.1)
    else:
        def chi2_center(fnl_loc: float, b1: float, sigmas: float, sn0: float) -> float:
            diff = y - center_model.pk0(fnl_loc=fnl_loc, b1=b1, sigmas=sigmas, sn0=sn0)
            return float(diff @ cov_inv @ diff)

        m_center = Minuit(chi2_center, fnl_loc=50.0, b1=2.72, sigmas=0.1, sn0=SN0_FIXED)
    m_center.errordef = 1.0
    m_center.limits["fnl_loc"] = (-2000.0, 2000.0)
    m_center.limits["b1"] = (0.0, None)
    m_center.limits["sigmas"] = (0.0, None)
    if "sn0" in m_center.parameters:
        m_center.limits["sn0"] = SN0_LIMIT
    m_center.migrad()
    m_center.hesse()

    qmax = int(np.floor((float(kmax[-1]) / K_FUND) ** 2)) + 1
    gq = gq_enumerate(qmax)
    k_shell, g_shell = shell_arrays(gq, kmax=float(kmax[-1]))
    idx = bin_shell_indices(k_shell, kmin, kmax)
    bin_model = RSDModel(cosmo, k_shell, sigmas_init=float(m_center.values["sigmas"]))

    if pk_sn0_policy == "fixed":
        def chi2_bin(fnl_loc: float, b1: float, sigmas: float) -> float:
            pk_shell = bin_model.pk0(fnl_loc=fnl_loc, b1=b1, sigmas=sigmas, sn0=SN0_FIXED)
            model = binavg_pk(pk_shell, g_shell, idx)
            diff = y - model
            return float(diff @ cov_inv @ diff)

        m_bin = Minuit(
            chi2_bin,
            fnl_loc=float(m_center.values["fnl_loc"]),
            b1=float(m_center.values["b1"]),
            sigmas=float(m_center.values["sigmas"]),
        )
    else:
        def chi2_bin(fnl_loc: float, b1: float, sigmas: float, sn0: float) -> float:
            pk_shell = bin_model.pk0(fnl_loc=fnl_loc, b1=b1, sigmas=sigmas, sn0=sn0)
            model = binavg_pk(pk_shell, g_shell, idx)
            diff = y - model
            return float(diff @ cov_inv @ diff)

        m_bin = Minuit(
            chi2_bin,
            fnl_loc=float(m_center.values["fnl_loc"]),
            b1=float(m_center.values["b1"]),
            sigmas=float(m_center.values["sigmas"]),
            sn0=float(m_center.values["sn0"]),
        )
    m_bin.errordef = 1.0
    m_bin.limits["fnl_loc"] = (-2000.0, 2000.0)
    m_bin.limits["b1"] = (0.0, None)
    m_bin.limits["sigmas"] = (0.0, None)
    if "sn0" in m_bin.parameters:
        m_bin.limits["sn0"] = SN0_LIMIT
    m_bin.migrad()
    m_bin.hesse()

    def pack(m: Minuit, ndof: int) -> dict[str, object]:
        sn0_free = "sn0" in m.parameters
        return {
            "fnl_loc": float(m.values["fnl_loc"]),
            "b1": float(m.values["b1"]),
            "sigmas": float(m.values["sigmas"]),
            "sn0": float(m.values["sn0"]) if sn0_free else SN0_FIXED,
            "sn0_fixed": not sn0_free,
            "errors_raw": {
                "fnl_loc": float(m.errors["fnl_loc"]),
                "b1": float(m.errors["b1"]),
                "sigmas": float(m.errors["sigmas"]),
                "sn0": float(m.errors["sn0"]) if sn0_free else 0.0,
            },
            "chi2": float(m.fval),
            "ndof": int(ndof),
            "chi2_per_dof": float(m.fval / ndof),
            "valid": bool(m.fmin.is_valid),
            "has_accurate_covar": bool(m.fmin.has_accurate_covar),
        }

    return {
        "fit_mask_indices": np.nonzero(fit_mask)[0].astype(int).tolist(),
        "n_fit_bins": int(fit_mask.sum()),
        "k_min_first": float(kmin[0]),
        "k_max_last_edge": float(kmax[-1]),
        "k_max_last_center": float(kcen[-1]),
        "pk_sn0_policy": pk_sn0_policy,
        "center": pack(m_center, ndof=int(fit_mask.sum() - n_pk_params)),
        "binavg": pack(m_bin, ndof=int(fit_mask.sum() - n_pk_params)),
    }


def precompute_rebin_cache(gq: np.ndarray, kmax: float) -> tuple[np.ndarray, np.ndarray]:
    """Precompute cached rebin weights and k_eff."""
    qnz = np.nonzero(gq[1:])[0] + 1
    k = K_FUND * np.sqrt(qnz.astype("f8"))
    g = gq[qnz].astype("f8")
    mask = k <= kmax
    k = k[mask]
    g = g[mask]
    dk = REBIN_DK_FACTOR * K_FUND
    nbins = int(np.ceil(kmax / dk)) + 1
    ibin = np.clip((k / dk).astype(np.int64), 0, nbins - 1)
    gbin = np.bincount(ibin, weights=g, minlength=nbins)
    gkbin = np.bincount(ibin, weights=g * k, minlength=nbins)
    nonzero = gbin > 0
    return gbin[nonzero].astype("f8"), gkbin[nonzero] / gbin[nonzero]


def j0_matrix(k: np.ndarray, s: np.ndarray) -> np.ndarray:
    """sin(ks)/(ks) matrix."""
    arg = np.outer(k, s)
    out = np.ones_like(arg)
    mask = arg != 0.0
    out[mask] = np.sin(arg[mask]) / arg[mask]
    return out


def fit_xi_profiler(
    *,
    name: str,
    indices: np.ndarray,
    pcf_data: dict[str, np.ndarray],
    pcf_cov: dict[str, np.ndarray],
    xi_model: RSDModel,
    g_cache: np.ndarray,
    k_eff: np.ndarray,
    pk_reference: dict[str, object],
    covariance_policy: str,
    xi_sigmas_policy: str,
    xi_sn0_policy: str,
    pk_reference_label: str,
) -> dict[str, object]:
    """Run one 2PCF profiler for a chosen rlist."""
    if xi_sn0_policy not in {"fixed", "free"}:
        raise ValueError(f"unknown xi_sn0_policy={xi_sn0_policy}")

    s = pcf_data["s"][indices]
    y = pcf_data["mean"][indices]
    cov_sample = pcf_cov["cov"][np.ix_(indices, indices)]
    nmock = int(pcf_cov["mocks"].shape[0])
    ndata = int(indices.size)
    if xi_sigmas_policy == "fixed":
        nparams = N_XI_PARAMS_FIXED_SIGMAS
    elif xi_sigmas_policy == "free":
        nparams = N_XI_PARAMS_FREE_SIGMAS
    else:
        raise ValueError(f"unknown xi_sigmas_policy={xi_sigmas_policy}")
    if xi_sn0_policy == "free":
        nparams += 1
    corrections = covariance_corrections(nmock, ndata, nparams)
    if covariance_policy == "single":
        cov = cov_sample
        policy_description = "C_sample for single-box parameter-constraint test"
    elif covariance_policy == "mean":
        cov = cov_sample / float(nmock)
        policy_description = "C_sample / Nmock for ensemble-mean residual diagnostic"
    else:
        raise ValueError(f"unknown covariance_policy={covariance_policy}")
    cov_inv = corrections["hartlap"] * np.linalg.pinv(cov, rcond=1e-10)
    kernel = j0_matrix(k_eff, s)
    ref = pk_reference["binavg"]
    ref_sigmas = float(ref["sigmas"])
    ref_sn0 = float(ref.get("sn0", SN0_FIXED))

    def model(fnl_loc: float, b1: float, sigmas: float, sn0: float) -> np.ndarray:
        pk = xi_model.pk0(fnl_loc=fnl_loc, b1=b1, sigmas=sigmas, sn0=sn0)
        return (g_cache * pk) @ kernel / VOLUME

    ref_all = {
        "fnl_loc": float(ref["fnl_loc"]),
        "b1": float(ref["b1"]),
        "sigmas": ref_sigmas,
        "sn0": ref_sn0,
    }
    varied_parameters = ["fnl_loc", "b1"]
    if xi_sigmas_policy == "free":
        varied_parameters.append("sigmas")
    if xi_sn0_policy == "free":
        varied_parameters.append("sn0")
    ref_values = {par: ref_all[par] for par in varied_parameters}
    ref_diff = y - model(**ref_all)
    initial_values = dict(ref_values)
    if xi_sigmas_policy == "free":
        initial_values["sigmas"] = max(ref_sigmas, 10.0)
    if xi_sn0_policy == "free":
        initial_values["sn0"] = SN0_FIXED

    if xi_sigmas_policy == "fixed" and xi_sn0_policy == "fixed":
        def chi2(fnl_loc: float, b1: float) -> float:
            diff = y - model(fnl_loc=fnl_loc, b1=b1, sigmas=ref_sigmas, sn0=ref_sn0)
            return float(diff @ cov_inv @ diff)

        m = Minuit(chi2, **initial_values)
    elif xi_sigmas_policy == "free" and xi_sn0_policy == "fixed":
        def chi2(fnl_loc: float, b1: float, sigmas: float) -> float:
            diff = y - model(fnl_loc=fnl_loc, b1=b1, sigmas=sigmas, sn0=ref_sn0)
            return float(diff @ cov_inv @ diff)

        m = Minuit(chi2, **initial_values)
        m.limits["sigmas"] = (0.0, None)
    elif xi_sigmas_policy == "fixed" and xi_sn0_policy == "free":
        def chi2(fnl_loc: float, b1: float, sn0: float) -> float:
            diff = y - model(fnl_loc=fnl_loc, b1=b1, sigmas=ref_sigmas, sn0=sn0)
            return float(diff @ cov_inv @ diff)

        m = Minuit(chi2, **initial_values)
        m.limits["sn0"] = SN0_LIMIT
    else:
        def chi2(fnl_loc: float, b1: float, sigmas: float, sn0: float) -> float:
            diff = y - model(fnl_loc=fnl_loc, b1=b1, sigmas=sigmas, sn0=sn0)
            return float(diff @ cov_inv @ diff)

        m = Minuit(chi2, **initial_values)
        m.limits["sigmas"] = (0.0, None)
        m.limits["sn0"] = SN0_LIMIT
    ref_chi2 = float(ref_diff @ cov_inv @ ref_diff)

    def configure_minuit(minuit: Minuit) -> None:
        """Apply common limits and Minuit settings."""
        minuit.errordef = 1.0
        minuit.limits["fnl_loc"] = (-2000.0, 2000.0)
        minuit.limits["b1"] = (0.0, None)
        if "sigmas" in minuit.parameters:
            minuit.limits["sigmas"] = (0.0, None)
        if "sn0" in minuit.parameters:
            minuit.limits["sn0"] = SN0_LIMIT

    configure_minuit(m)
    t0 = time.time()
    m.migrad()
    m.hesse()
    fallback_tried = False
    fallback_used = False
    if (not m.fmin.is_valid or float(m.fval) > ref_chi2) and initial_values != ref_values:
        fallback_tried = True
        # A free sn0 dimension can make Minuit sensitive to starting values.
        # If the main start lands above the reference chi2, retry from the
        # literal P(k) reference and keep the lower-chi2 solution.
        m_alt = Minuit(chi2, **ref_values)
        configure_minuit(m_alt)
        m_alt.migrad()
        m_alt.hesse()
        if float(m_alt.fval) < float(m.fval):
            m = m_alt
            fallback_used = True
    elapsed = time.time() - t0

    err_fac = corrections["percival_error_factor"]
    best = {par: float(m.values[par]) for par in varied_parameters}
    errors_raw = {par: float(m.errors[par]) for par in varied_parameters}
    errors = {par: float(errors_raw[par] * err_fac) for par in errors_raw}
    delta = {par: float(best[par] - ref_values[par]) for par in best}
    pulls = {par: float(delta[par] / errors[par]) if errors[par] > 0 else float("nan") for par in best}
    ndof = ndata - nparams
    sigmas_best = float(best["sigmas"]) if xi_sigmas_policy == "free" else ref_sigmas
    sn0_best = float(best["sn0"]) if xi_sn0_policy == "free" else ref_sn0
    return {
        "selection": name,
        "indices": indices.astype(int).tolist(),
        "s_min_edge": float(pcf_data["smin"][indices].min()),
        "s_max_edge": float(pcf_data["smax"][indices].max()),
        "s_center_min": float(s.min()),
        "s_center_max": float(s.max()),
        "ndata": ndata,
        "nmock": nmock,
        "covariance": {
            "policy": covariance_policy,
            "policy_description": policy_description,
            "condition_number": float(np.linalg.cond(cov)),
            "nparams": nparams,
            **corrections,
        },
        "bestfit": {
            **best,
            "sigmas": sigmas_best,
            "sn0": sn0_best,
            "sigmas_fixed": xi_sigmas_policy == "fixed",
            "sn0_fixed": xi_sn0_policy == "fixed",
            "xi_sigmas_policy": xi_sigmas_policy,
            "xi_sn0_policy": xi_sn0_policy,
            "p": P_FIXED,
            "chi2": float(m.fval),
            "ndof": int(ndof),
            "chi2_per_dof": float(m.fval / ndof),
            "valid": bool(m.fmin.is_valid),
            "has_accurate_covar": bool(m.fmin.has_accurate_covar),
        },
        "errors_raw": errors_raw,
        "errors_percival": errors,
        "reference_delta": {
            "reference": pk_reference_label,
            "reference_values": ref_all,
            "delta": delta,
            "pull": pulls,
            "chi2_at_reference": ref_chi2,
            "chi2_per_dof_at_reference": float(ref_chi2 / ndof),
        },
        "timing_sec": elapsed,
        "optimizer": {
            "initial_values": initial_values,
            "fallback_reference_start_tried": fallback_tried,
            "fallback_reference_start_used": fallback_used,
        },
    }


def build_rlist_selections(
    pcf: dict[str, np.ndarray],
    rmins: list[float],
    rmax: float,
    selection_modes: list[str],
) -> dict[str, np.ndarray]:
    """Build rlists by edge containment, with optional stride-2 thinning."""
    selections = {}
    for rmin in rmins:
        indices = np.nonzero((pcf["smin"] >= rmin) & (pcf["smax"] <= rmax))[0]
        if not indices.size:
            continue
        base_name = f"r{int(rmin)}_{int(rmax)}"
        if "contiguous" in selection_modes:
            name = base_name if selection_modes == ["contiguous"] else f"{base_name}_all"
            selections[name] = indices.astype("i8")
        if "stride2" in selection_modes:
            for phase in (0, 1):
                thinned = indices[phase::2]
                if thinned.size > N_XI_PARAMS_FREE_SIGMAS:
                    selections[f"{base_name}_stride2p{phase}"] = thinned.astype("i8")
        if "stride3" in selection_modes:
            for phase in (0, 1, 2):
                thinned = indices[phase::3]
                if thinned.size > N_XI_PARAMS_FREE_SIGMAS:
                    selections[f"{base_name}_stride3p{phase}"] = thinned.astype("i8")
        if not any(mode in {"contiguous", "stride2", "stride3"} for mode in selection_modes):
            selections[f"r{int(rmin)}_{int(rmax)}"] = indices.astype("i8")
    return selections


def parse_selection_modes(value: str) -> list[str]:
    """Parse comma-separated rlist selection modes."""
    modes = [item.strip() for item in value.split(",") if item.strip()]
    allowed = {"contiguous", "stride2", "stride3"}
    unknown = sorted(set(modes) - allowed)
    if unknown:
        raise ValueError(f"unknown selection modes: {unknown}; allowed={sorted(allowed)}")
    return modes or ["contiguous"]


def main() -> None:
    parser = argparse.ArgumentParser(description="Quijote LCp50 P(k)+2PCF profiler trial")
    parser.add_argument("--data-tag", default="LCp50", help="P(k) and 2PCF data stack tag")
    parser.add_argument("--pk-cov-tag", default="", help="P(k) covariance stack tag; default=data-tag")
    parser.add_argument("--pcf-cov-tag", default="", help="2PCF covariance stack tag; default=data-tag")
    parser.add_argument(
        "--pcf-covariance-policy",
        default="single",
        choices=("single", "mean"),
        help="single uses C_sample; mean uses C_sample/Nmock as residual diagnostic",
    )
    parser.add_argument(
        "--xi-sigmas-policy",
        default="fixed",
        choices=("fixed", "free"),
        help="fixed uses P(k) BinAvgFit sigmas in 2PCF; free profiles sigmas in 2PCF",
    )
    parser.add_argument(
        "--pk-sn0-policy",
        default="fixed",
        choices=("fixed", "free"),
        help="fixed keeps P(k) reference sn0=0; free profiles sn0 in P(k) reference",
    )
    parser.add_argument(
        "--xi-sn0-policy",
        default="fixed",
        choices=("fixed", "free"),
        help="fixed uses P(k) reference sn0 in 2PCF; free profiles sn0 in 2PCF",
    )
    parser.add_argument("--rmins", default="60,80,100,120,140,150", help="comma separated rmin edge cuts")
    parser.add_argument("--rmax", default=350.0, type=float, help="rmax edge cut")
    parser.add_argument(
        "--selection-modes",
        default="contiguous",
        help="comma separated choices from contiguous,stride2,stride3",
    )
    parser.add_argument("--output-label", default="", help="optional output suffix")
    args = parser.parse_args()
    pk_cov_tag = args.pk_cov_tag or args.data_tag
    pcf_cov_tag = args.pcf_cov_tag or args.data_tag

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    label = args.output_label or (
        f"{args.xi_sigmas_policy}sigmas_{args.xi_sn0_policy}sn0_{args.pcf_covariance_policy}_"
        f"data_{args.data_tag}_pkcov_{pk_cov_tag}_pcfcov_{pcf_cov_tag}"
    )
    out_json = OUTPUT_DIR / f"task42_quijote_lcp50_profiler_{label}.json"

    print("[read] P(k) and 2PCF stacks")
    pk_data = load_pk_stack(args.data_tag)
    pk_cov = load_pk_stack(pk_cov_tag)
    pcf_data = load_pcf_stack(args.data_tag)
    pcf_cov = load_pcf_stack(pcf_cov_tag)
    print(f"[read] tag={args.data_tag}, pk mocks={pk_data['mocks'].shape}, pcf mocks={pcf_data['mocks'].shape}")

    print("[cosmo] build")
    cosmo = build_cosmology()

    print("[fit] P(k) reference")
    pk_reference = fit_pk_reference(pk_data, pk_cov, cosmo, pk_sn0_policy=args.pk_sn0_policy)
    print("[fit] center", pk_reference["center"])
    print("[fit] binavg", pk_reference["binavg"])

    print("[full-discrete] gq/cache")
    t0 = time.time()
    qmax = int((KMAX_DISCRETE / K_FUND) ** 2)
    nmax = int(KMAX_DISCRETE / K_FUND)
    gq = gq_fft(qmax, nmax)
    g_cache, k_eff = precompute_rebin_cache(gq, KMAX_DISCRETE)
    cache_sec = time.time() - t0
    print(f"[full-discrete] cache bins={len(k_eff)}, sec={cache_sec:.2f}")
    xi_model = RSDModel(cosmo, k_eff, sigmas_init=float(pk_reference["binavg"]["sigmas"]))

    rmins = [float(item.strip()) for item in args.rmins.split(",") if item.strip()]
    selection_modes = parse_selection_modes(args.selection_modes)
    selections = build_rlist_selections(pcf_data, rmins, args.rmax, selection_modes)
    pk_reference_label = f"pk_binavg_{args.data_tag}_kcen_le_0p08"

    rows = []
    for name, indices in selections.items():
        print(
            f"[profiler] {name}: N={indices.size}, "
            f"edges=[{pcf_data['smin'][indices].min():.0f},{pcf_data['smax'][indices].max():.0f}], "
            f"centers=[{pcf_data['s'][indices].min():.0f},{pcf_data['s'][indices].max():.0f}]"
        )
        row = fit_xi_profiler(
            name=name,
            indices=indices,
            pcf_data=pcf_data,
            pcf_cov=pcf_cov,
            xi_model=xi_model,
            g_cache=g_cache,
            k_eff=k_eff,
            pk_reference=pk_reference,
            covariance_policy=args.pcf_covariance_policy,
            xi_sigmas_policy=args.xi_sigmas_policy,
            xi_sn0_policy=args.xi_sn0_policy,
            pk_reference_label=pk_reference_label,
        )
        sigmas_error = row["errors_percival"].get("sigmas")
        sn0_error = row["errors_percival"].get("sn0")
        sigmas_text = (
            f"sigmas={row['bestfit']['sigmas']:.3f} +/- {sigmas_error:.3f}"
            if sigmas_error is not None
            else f"sigmas_fixed={row['bestfit']['sigmas']:.4f}"
        )
        sn0_text = (
            f"sn0={row['bestfit']['sn0']:.3g} +/- {sn0_error:.3g}"
            if sn0_error is not None
            else f"sn0_fixed={row['bestfit']['sn0']:.3g}"
        )
        print(
            "  "
            f"fnl={row['bestfit']['fnl_loc']:.3f} +/- {row['errors_percival']['fnl_loc']:.3f}, "
            f"b1={row['bestfit']['b1']:.5f} +/- {row['errors_percival']['b1']:.5f}, "
            f"{sigmas_text}, {sn0_text}, "
            f"pull_fnl={row['reference_delta']['pull']['fnl_loc']:.3f}, "
            f"chi2/dof={row['bestfit']['chi2_per_dof']:.3f}"
        )
        rows.append(row)

    summary = {
        "task": "task42_quijote_lcp50_profiler",
        "status": "done",
        "inputs": {
            "data_dir": str(DATA_DIR),
            "pk_data_tag": args.data_tag,
            "pk_cov_tag": pk_cov_tag,
            "pcf_data_tag": args.data_tag,
            "pcf_cov_tag": pcf_cov_tag,
            "pcf_covariance_policy": args.pcf_covariance_policy,
            "xi_sigmas_policy": args.xi_sigmas_policy,
            "pk_sn0_policy": args.pk_sn0_policy,
            "xi_sn0_policy": args.xi_sn0_policy,
            "selection_modes": selection_modes,
            "n_pk_mocks": int(pk_data["mocks"].shape[0]),
            "n_pcf_mocks": int(pcf_data["mocks"].shape[0]),
        },
        "fixed_parameters": {
            "Lbox": LBOX,
            "z": Z,
            "p": P_FIXED,
            "sn0_fixed_default": SN0_FIXED,
            "sn0_limit": list(SN0_LIMIT),
            "kmax_fit_center_cut": KMAX_FIT,
            "kmax_discrete": KMAX_DISCRETE,
            "rmax_edge": args.rmax,
            "xi_sigmas_source": "pk_reference.binavg.sigmas" if args.xi_sigmas_policy == "fixed" else None,
            "xi_sigmas_fixed": float(pk_reference["binavg"]["sigmas"]) if args.xi_sigmas_policy == "fixed" else None,
            "xi_sn0_source": "pk_reference.binavg.sn0" if args.xi_sn0_policy == "fixed" else None,
            "xi_sn0_fixed": float(pk_reference["binavg"]["sn0"]) if args.xi_sn0_policy == "fixed" else None,
        },
        "free_parameters": {
            "pk_reference": ["fnl_loc", "b1", "sigmas"] + (["sn0"] if args.pk_sn0_policy == "free" else []),
            "xi_profiler": (
                ["fnl_loc", "b1"]
                + (["sigmas"] if args.xi_sigmas_policy == "free" else [])
                + (["sn0"] if args.xi_sn0_policy == "free" else [])
            ),
        },
        "pk_reference": pk_reference,
        "profiler_results": rows,
        "timing": {"cache_sec": cache_sec},
        "outputs": {"json": str(out_json)},
    }
    out_json.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"[write] {out_json}")


if __name__ == "__main__":
    main()
