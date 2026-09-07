#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""RSD monopole theory helpers for Task44 LRG2 fNL fits."""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

import numpy as np

from task44_config import OMEGA_L, OMEGA_M, PROJECT_ROOT


TASK43_DIR = PROJECT_ROOT / "codes" / "task43"
if str(TASK43_DIR) not in sys.path:
    sys.path.insert(0, str(TASK43_DIR))

from task43_theory_template import DEFAULT_COSMOLOGY, build_template_arrays, load_task41  # noqa: E402


DELTA_C = 1.686
FOG_MODELS = ("lorentzian", "gaussian")


def omega_m_z(z: float) -> float:
    zp1 = 1.0 + float(z)
    return float(OMEGA_M * zp1**3 / (OMEGA_M * zp1**3 + OMEGA_L))


def growth_rate_approx(z: float) -> float:
    """Approximate f=dlnD/dlna used for the first Task44 RSD monopole pass."""
    return omega_m_z(float(z)) ** 0.55


def gaussian_mu_moments(k: np.ndarray, sigma_s: float, *, nmu: int = 96) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return <mu^{0,2,4} exp[-(k mu sigma_s)^2]> over mu in [-1,1].

    The Gaussian FoG moments have closed forms.  ``nmu`` is accepted for API
    compatibility with the first numerical-quadrature implementation.
    """
    from scipy.special import erf

    a = np.abs(np.asarray(k, dtype="f8") * float(sigma_s))
    i0 = np.ones_like(a)
    i2 = np.full_like(a, 1.0 / 3.0)
    i4 = np.full_like(a, 1.0 / 5.0)
    small = a <= 1.0e-2
    if np.any(small):
        x2 = a[small] ** 2
        i0[small] = 1.0 - x2 / 3.0 + x2**2 / 10.0
        i2[small] = 1.0 / 3.0 - x2 / 5.0 + x2**2 / 14.0
        i4[small] = 1.0 / 5.0 - x2 / 7.0 + x2**2 / 18.0
    mask = ~small
    if np.any(mask):
        x = a[mask]
        expx = np.exp(-(x * x))
        erfx = erf(x)
        sqrt_pi = np.sqrt(np.pi)
        i0[mask] = sqrt_pi * erfx / (2.0 * x)
        i2[mask] = sqrt_pi * erfx / (4.0 * x**3) - expx / (2.0 * x**2)
        i4[mask] = 3.0 * sqrt_pi * erfx / (8.0 * x**5) - expx * (2.0 * x**2 + 3.0) / (4.0 * x**4)
    return i0, i2, i4


def lorentzian_mu_moments(k: np.ndarray, sigma_s: float, *, nmu: int = 96) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return <mu^{0,2,4} / (1 + (k mu sigma_s)^2 / 2)^2> over mu in [-1,1]."""
    del nmu
    a = np.abs(np.asarray(k, dtype="f8") * float(sigma_s)) / np.sqrt(2.0)
    i0 = np.ones_like(a)
    i2 = np.full_like(a, 1.0 / 3.0)
    i4 = np.full_like(a, 1.0 / 5.0)
    small = a <= 5.0e-2
    if np.any(small):
        x2 = a[small] ** 2
        i0[small] = 1.0 - 2.0 * x2 / 3.0 + 3.0 * x2**2 / 5.0 - 4.0 * x2**3 / 7.0
        i2[small] = 1.0 / 3.0 - 2.0 * x2 / 5.0 + 3.0 * x2**2 / 7.0 - 4.0 * x2**3 / 9.0
        i4[small] = 1.0 / 5.0 - 2.0 * x2 / 7.0 + x2**2 / 3.0 - 4.0 * x2**3 / 11.0
    mask = ~small
    if np.any(mask):
        x = a[mask]
        atanx = np.arctan(x)
        x2 = x * x
        one_plus_x2 = 1.0 + x2
        i0[mask] = 1.0 / (2.0 * one_plus_x2) + atanx / (2.0 * x)
        i2[mask] = atanx / (2.0 * x**3) - 1.0 / (2.0 * x2 * one_plus_x2)
        i4[mask] = 1.0 / x**4 + 1.0 / (2.0 * x**4 * one_plus_x2) - 3.0 * atanx / (2.0 * x**5)
    return i0, i2, i4


def mu_moments(k: np.ndarray, sigma_s: float, *, nmu: int = 96, fog_model: str = "lorentzian") -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return even-mu FoG moments for the requested damping model."""
    model = str(fog_model).lower()
    if model == "lorentzian":
        return lorentzian_mu_moments(k, sigma_s, nmu=nmu)
    if model == "gaussian":
        return gaussian_mu_moments(k, sigma_s, nmu=nmu)
    raise ValueError(f"unknown fog_model {fog_model!r}; choices are {FOG_MODELS}")


def fog_description(fog_model: str) -> str:
    model = str(fog_model).lower()
    if model == "lorentzian":
        return "Lorentzian desilike PNG auto-power: [1 + (k mu sigma_s)^2 / 2]^-2"
    if model == "gaussian":
        return "Gaussian exp[-(k mu sigma_s)^2]"
    raise ValueError(f"unknown fog_model {fog_model!r}; choices are {FOG_MODELS}")


def evaluate_rsd_png_monopole_pk(
    k_query: np.ndarray,
    template_arrays: dict[str, np.ndarray],
    *,
    fnl_loc: float,
    b1: float,
    p_fixed: float,
    f_growth: float,
    sigma_s: float,
    sn0: float = 0.0,
    nmu: int = 96,
    fog_model: str = "lorentzian",
) -> np.ndarray:
    """Evaluate the redshift-space PNG monopole with FoG damping."""
    task41 = load_task41()
    k = np.asarray(k_query, dtype="f8")
    alpha = task41.interp_logk(k, template_arrays["k"], template_arrays["alpha"])
    pk_dd = task41.interp_logk(k, template_arrays["k"], template_arrays["pk_dd"])
    bphi = 2.0 * DELTA_C * (float(b1) - float(p_fixed))
    amp = float(b1) + float(fnl_loc) * bphi * alpha
    i0, i2, i4 = mu_moments(k, float(sigma_s), nmu=int(nmu), fog_model=str(fog_model))
    pk0 = pk_dd * (amp * amp * i0 + 2.0 * amp * float(f_growth) * i2 + float(f_growth) ** 2 * i4)
    return pk0 + float(sn0)


def evaluate_rsd_png_multipole_pk(
    k_query: np.ndarray,
    template_arrays: dict[str, np.ndarray],
    *,
    fnl_loc: float,
    b1: float,
    p_fixed: float,
    f_growth: float,
    sigma_s: float,
    ells: tuple[int, ...] = (0, 2, 4),
    sn0: float = 0.0,
    nmu: int = 256,
    fog_model: str = "lorentzian",
) -> dict[int, np.ndarray]:
    """Evaluate redshift-space PNG power-spectrum multipoles."""
    task41 = load_task41()
    k = np.asarray(k_query, dtype="f8")
    alpha = task41.interp_logk(k, template_arrays["k"], template_arrays["alpha"])
    pk_dd = task41.interp_logk(k, template_arrays["k"], template_arrays["pk_dd"])
    bphi = 2.0 * DELTA_C * (float(b1) - float(p_fixed))
    amp = float(b1) + float(fnl_loc) * bphi * alpha
    mu, wmu = np.polynomial.legendre.leggauss(int(nmu))
    mu2 = mu * mu
    k_mu_sigma2 = (k[:, None] * mu[None, :] * float(sigma_s)) ** 2
    if str(fog_model).lower() == "lorentzian":
        damp = 1.0 / (1.0 + 0.5 * k_mu_sigma2) ** 2
    elif str(fog_model).lower() == "gaussian":
        damp = np.exp(-k_mu_sigma2)
    else:
        raise ValueError(f"unknown fog_model {fog_model!r}; choices are {FOG_MODELS}")
    pk_mu = pk_dd[:, None] * (amp[:, None] + float(f_growth) * mu2[None, :]) ** 2 * damp
    out: dict[int, np.ndarray] = {}
    for ell in tuple(int(v) for v in ells):
        coeff = np.zeros(ell + 1, dtype="f8")
        coeff[ell] = 1.0
        leg = np.polynomial.legendre.legval(mu, coeff)
        pole = 0.5 * (2 * ell + 1) * np.sum(wmu[None, :] * pk_mu * leg[None, :], axis=1)
        if ell == 0:
            pole = pole + float(sn0)
        out[ell] = np.asarray(pole, dtype="f8")
    return out


def build_theory_context(
    zeff: float,
    *,
    kmax: float,
    ndense: int,
    boxsize: float,
    cosmology: str = DEFAULT_COSMOLOGY,
) -> dict[str, Any]:
    """Build the FullDiscrete context shared by Task44 RSD fits."""
    task41 = load_task41()
    kfund = 2.0 * np.pi / float(boxsize)
    volume = float(boxsize) ** 3
    k_template = np.geomspace(min(1.0e-4, kfund / 2.0), max(1.0, kmax * 1.2), 4000)
    template, cosmology_meta = build_template_arrays(task41, k_template, z=float(zeff), cosmology=str(cosmology))
    qmax = int(np.ceil((float(kmax) / kfund) ** 2))
    nmax = int(np.ceil(np.sqrt(qmax)))
    gq = task41.gq_fft(qmax, nmax)
    g_nz, k_eff = task41.precompute_rebin_cache(gq, kfund, float(kmax), dk_factor=0.1)
    k_dense = np.geomspace(max(1.0e-4, kfund / 5.0), float(kmax), int(ndense))
    f_growth = growth_rate_approx(float(zeff))
    return {
        "task41": task41,
        "template": template,
        "cosmology": str(cosmology),
        "cosmology_meta": cosmology_meta,
        "g_nz": g_nz,
        "k_eff": k_eff,
        "k_dense": k_dense,
        "boxsize": float(boxsize),
        "volume": volume,
        "kfund": kfund,
        "zeff": float(zeff),
        "f_growth": float(f_growth),
        "growth_rate_method": "Omega_m(z)^0.55 using CUTSKY Omega_m",
    }
