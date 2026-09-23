#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Shared helpers for the exploratory Task44 Gaussian-GSM hybrid model.

The hybrid deliberately keeps the existing FullDiscrete linear PNG response
and adds only a Gaussian-initial-condition streaming correction.  It does not
implement nonlinear PNG operators in the CLPT cumulants.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from scipy.interpolate import CubicSpline


MODEL_NAME = "gaussian_gsm_plus_fulldiscrete_lowk_png"
MODEL_VERSION = 2


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def edge_pairs(edges: np.ndarray) -> np.ndarray:
    arr = np.asarray(edges, dtype="f8")
    if arr.ndim == 2 and arr.shape[1] == 2:
        return arr
    if arr.ndim != 1 or arr.size < 2:
        raise ValueError(f"invalid radial edges shape {arr.shape}")
    return np.column_stack([arr[:-1], arr[1:]])


def cosine_lowpass(k: np.ndarray, start: float, end: float) -> np.ndarray:
    """Unity below ``start``, cosine roll-off, and zero at/above ``end``."""

    kval = np.asarray(k, dtype="f8")
    start = float(start)
    end = float(end)
    if not (0.0 < start < end):
        raise ValueError(f"low-k taper requires 0 < start < end; got {start}, {end}")
    out = np.ones_like(kval)
    out[kval >= end] = 0.0
    middle = (kval > start) & (kval < end)
    out[middle] = 0.5 * (1.0 + np.cos(np.pi * (kval[middle] - start) / (end - start)))
    return out


def smooth_radial_window(
    radius: np.ndarray,
    *,
    low_zero: float,
    low_full: float,
    high_full: float,
    high_zero: float,
) -> np.ndarray:
    """Smoothly localize the GSM-minus-Kaiser correction in separation."""

    r = np.asarray(radius, dtype="f8")
    if not (0.0 <= low_zero < low_full < high_full < high_zero):
        raise ValueError("radial correction window edges must be strictly ordered")
    out = np.ones_like(r)
    out[r <= low_zero] = 0.0
    rise = (r > low_zero) & (r < low_full)
    out[rise] = 0.5 * (1.0 - np.cos(np.pi * (r[rise] - low_zero) / (low_full - low_zero)))
    fall = (r > high_full) & (r < high_zero)
    out[fall] = 0.5 * (1.0 + np.cos(np.pi * (r[fall] - high_full) / (high_zero - high_full)))
    out[r >= high_zero] = 0.0
    return out


def shell_quadrature(edges: np.ndarray, nquad: int) -> tuple[np.ndarray, np.ndarray]:
    """Return radii and normalized weights for volume-averaged radial shells."""

    pairs = edge_pairs(edges)
    if np.any(pairs[:, 0] < 0.0) or np.any(pairs[:, 1] <= pairs[:, 0]):
        raise ValueError("radial shell edges must be non-negative and ordered")
    nodes, weights = np.polynomial.legendre.leggauss(int(nquad))
    lo = pairs[:, 0, None]
    hi = pairs[:, 1, None]
    radius = 0.5 * (hi - lo) * nodes[None, :] + 0.5 * (hi + lo)
    raw = 0.5 * (hi - lo) * weights[None, :] * radius**2
    norm = (hi**3 - lo**3) / 3.0
    return radius, raw / norm


def _load_gsm_class():
    """Import velocileptors with a compatibility alias for recent SciPy."""

    import scipy.signal
    from scipy.signal.windows import tukey

    if not hasattr(scipy.signal, "tukey"):
        scipy.signal.tukey = tukey
    from velocileptors.LPT.gaussian_streaming_model_fftw import GaussianStreamingModel

    return GaussianStreamingModel


def _evaluate_gsm_monopole(
    gsm: Any,
    radii: np.ndarray,
    *,
    f_growth: float,
    b1_eulerian: float,
    b2_lagrangian: float,
    bs_lagrangian: float,
    b3_lagrangian: float,
    nint: int,
    ngauss: int,
    rwidth: float,
) -> np.ndarray:
    """Evaluate xi0 one radius at a time; the upstream method is scalar-only."""

    b1_lagrangian = float(b1_eulerian) - 1.0
    output = np.empty(np.asarray(radii).size, dtype="f8")
    for index, radius in enumerate(np.asarray(radii, dtype="f8").ravel()):
        xi0, _, _ = gsm.compute_xi_ell(
            float(radius),
            float(f_growth),
            b1_lagrangian,
            float(b2_lagrangian),
            float(bs_lagrangian),
            float(b3_lagrangian),
            0.0,  # alpha
            0.0,  # alpha_v
            0.0,  # alpha_s0
            0.0,  # alpha_s2
            0.0,  # s2fog [(Mpc/h)^2]
            rwidth=float(rwidth),
            Nint=int(nint),
            ngauss=int(ngauss),
            update_cumulants=(index == 0),
        )
        output[index] = float(xi0)
    return output


def fixed_lagrangian_biases(b1_eulerian: float, prescription: str) -> tuple[float, float, float]:
    """Return fixed (b2^L, bs^L, b3^L) for a named no-new-parameter rule."""

    name = str(prescription).lower()
    b1_e = float(b1_eulerian)
    if name == "zero":
        return 0.0, 0.0, 0.0
    if name == "lazeyras-coevolution":
        # Lazeyras et al. halo b2^E(b1^E), converted with
        # b2^E = b2^L + 8 b1^L / 21.  Local-Lagrangian tidal and catch-all
        # third-order biases are kept at zero for this controlled variant.
        b2_e = 0.412 - 2.143 * b1_e + 0.929 * b1_e**2 + 0.008 * b1_e**3
        b2_l = b2_e - (8.0 / 21.0) * (b1_e - 1.0)
        return float(b2_l), 0.0, 0.0
    raise ValueError(f"unknown fixed Gaussian bias prescription {prescription!r}")


def build_gsm_emulator_arrays(
    *,
    k_linear: np.ndarray,
    pk_linear: np.ndarray,
    f_growth: float,
    b1_grid: np.ndarray,
    data_edges: np.ndarray,
    rr_edges: np.ndarray,
    rr_pair_probability: np.ndarray,
    nquad: int = 8,
    nint: int = 1000,
    ngauss: int = 4,
    rwidth: float = 100.0,
    gsm_kmin: float = 3.0e-3,
    gsm_kmax: float = 0.5,
    gsm_nk: int = 100,
    fft_n: int = 1600,
    fft_cutoff: float = 10.0,
    radial_window: tuple[float, float, float, float] = (10.0, 20.0, 400.0, 500.0),
    bias_prescription: str = "zero",
    threads: int = 1,
) -> dict[str, np.ndarray]:
    """Precompute shell corrections and their configuration-space IC scalar."""

    if int(threads) < 1 or int(threads) > 8:
        raise ValueError("GSM emulator construction requires 1 <= threads <= 8")
    bgrid = np.asarray(b1_grid, dtype="f8")
    if bgrid.ndim != 1 or bgrid.size < 4 or np.any(np.diff(bgrid) <= 0.0):
        raise ValueError("b1_grid must be a strictly increasing one-dimensional grid")
    k = np.asarray(k_linear, dtype="f8")
    pk = np.asarray(pk_linear, dtype="f8")
    if k.shape != pk.shape or k.ndim != 1 or np.any(k <= 0.0) or np.any(pk <= 0.0):
        raise ValueError("linear k/P input must be positive one-dimensional arrays of equal shape")

    data_radii, data_weights = shell_quadrature(data_edges, int(nquad))
    rr_radii, rr_weights = shell_quadrature(rr_edges, int(nquad))
    pair_probability = np.asarray(rr_pair_probability, dtype="f8")
    if pair_probability.shape != (rr_radii.shape[0],) or np.any(pair_probability < 0.0):
        raise ValueError("RR pair probability does not match RR radial shells")
    pair_probability = pair_probability / np.sum(pair_probability)

    all_radii = np.concatenate([data_radii.ravel(), rr_radii.ravel()])
    unique_radii, inverse = np.unique(all_radii, return_inverse=True)
    data_inverse = inverse[: data_radii.size].reshape(data_radii.shape)
    rr_inverse = inverse[data_radii.size :].reshape(rr_radii.shape)
    low_zero, low_full, high_full, high_zero = map(float, radial_window)
    correction_window = smooth_radial_window(
        unique_radii,
        low_zero=low_zero,
        low_full=low_full,
        high_full=high_full,
        high_zero=high_zero,
    )
    active = correction_window > 0.0

    GaussianStreamingModel = _load_gsm_class()
    gsm = GaussianStreamingModel(
        k,
        pk,
        kmin=float(gsm_kmin),
        kmax=float(gsm_kmax),
        nk=int(gsm_nk),
        N=int(fft_n),
        threads=int(threads),
        cutoff=float(fft_cutoff),
    )
    q_linear, xi_linear = gsm.sph_gsm.sph(0, gsm.plin * gsm.window)
    xi_matter = np.interp(unique_radii, q_linear, xi_linear)

    ndata = data_radii.shape[0]
    delta_shell = np.empty((bgrid.size, ndata), dtype="f8")
    gsm_shell = np.empty_like(delta_shell)
    kaiser_shell = np.empty_like(delta_shell)
    delta_gic = np.empty(bgrid.size, dtype="f8")
    max_abs_delta_high = np.empty(bgrid.size, dtype="f8")
    fixed_bias_grid = np.empty((bgrid.size, 3), dtype="f8")
    for ibias, b1_eulerian in enumerate(bgrid):
        b2_lagrangian, bs_lagrangian, b3_lagrangian = fixed_lagrangian_biases(
            float(b1_eulerian), str(bias_prescription)
        )
        fixed_bias_grid[ibias] = (b2_lagrangian, bs_lagrangian, b3_lagrangian)
        gsm_values = np.zeros_like(unique_radii)
        gsm_values[active] = _evaluate_gsm_monopole(
            gsm,
            unique_radii[active],
            f_growth=float(f_growth),
            b1_eulerian=float(b1_eulerian),
            b2_lagrangian=b2_lagrangian,
            bs_lagrangian=bs_lagrangian,
            b3_lagrangian=b3_lagrangian,
            nint=int(nint),
            ngauss=int(ngauss),
            rwidth=float(rwidth),
        )
        kaiser_amplitude = (
            float(b1_eulerian) ** 2
            + (2.0 / 3.0) * float(b1_eulerian) * float(f_growth)
            + (1.0 / 5.0) * float(f_growth) ** 2
        )
        kaiser_values = kaiser_amplitude * xi_matter
        delta_values = (gsm_values - kaiser_values) * correction_window
        delta_data_nodes = delta_values[data_inverse]
        gsm_data_nodes = gsm_values[data_inverse]
        kaiser_data_nodes = kaiser_values[data_inverse]
        delta_shell[ibias] = np.sum(data_weights * delta_data_nodes, axis=1)
        gsm_shell[ibias] = np.sum(data_weights * gsm_data_nodes, axis=1)
        kaiser_shell[ibias] = np.sum(data_weights * kaiser_data_nodes, axis=1)
        delta_rr_shell = np.sum(rr_weights * delta_values[rr_inverse], axis=1)
        delta_gic[ibias] = float(pair_probability @ delta_rr_shell)
        high = unique_radii >= float(high_full)
        max_abs_delta_high[ibias] = float(np.max(np.abs(delta_values[high]))) if np.any(high) else 0.0

    if not all(np.all(np.isfinite(item)) for item in (delta_shell, gsm_shell, kaiser_shell, delta_gic)):
        raise RuntimeError("non-finite GSM emulator output")
    return {
        "b1_grid": bgrid,
        "data_edges": edge_pairs(data_edges),
        "data_s": np.mean(edge_pairs(data_edges), axis=1),
        "delta_xi0_shell": delta_shell,
        "gsm_xi0_shell": gsm_shell,
        "kaiser_continuous_xi0_shell": kaiser_shell,
        "delta_gic": delta_gic,
        "rr_edges": edge_pairs(rr_edges),
        "rr_pair_probability": pair_probability,
        "max_abs_windowed_delta_at_or_above_high_full": max_abs_delta_high,
        "fixed_lagrangian_bias_grid_b2_bs_b3": fixed_bias_grid,
    }


@dataclass
class GSMCorrectionEmulator:
    path: Path
    b1_grid: np.ndarray
    data_edges: np.ndarray
    data_s: np.ndarray
    delta_xi0_shell_grid: np.ndarray
    delta_gic_grid: np.ndarray
    metadata: dict[str, Any]

    def __post_init__(self) -> None:
        self._shell_spline = CubicSpline(self.b1_grid, self.delta_xi0_shell_grid, axis=0)
        self._gic_spline = CubicSpline(self.b1_grid, self.delta_gic_grid)

    @property
    def b1_min(self) -> float:
        return float(self.b1_grid[0])

    @property
    def b1_max(self) -> float:
        return float(self.b1_grid[-1])

    def evaluate(self, b1_eulerian: float) -> tuple[np.ndarray, float]:
        b1 = float(b1_eulerian)
        if not (self.b1_min <= b1 <= self.b1_max):
            raise ValueError(f"b1={b1} lies outside GSM emulator [{self.b1_min}, {self.b1_max}]")
        return np.asarray(self._shell_spline(b1), dtype="f8"), float(self._gic_spline(b1))


def load_gsm_emulator(path: Path) -> GSMCorrectionEmulator:
    path = Path(path)
    with np.load(path, allow_pickle=False) as data:
        metadata = json.loads(str(np.asarray(data["meta_json"]).item()))
        emulator = GSMCorrectionEmulator(
            path=path,
            b1_grid=np.asarray(data["b1_grid"], dtype="f8"),
            data_edges=np.asarray(data["data_edges"], dtype="f8"),
            data_s=np.asarray(data["data_s"], dtype="f8"),
            delta_xi0_shell_grid=np.asarray(data["delta_xi0_shell"], dtype="f8"),
            delta_gic_grid=np.asarray(data["delta_gic"], dtype="f8"),
            metadata=metadata,
        )
    if metadata.get("model_name") != MODEL_NAME or int(metadata.get("model_version", -1)) != MODEL_VERSION:
        raise ValueError(f"unsupported GSM emulator contract in {path}")
    expected = (emulator.b1_grid.size, emulator.data_s.size)
    if emulator.delta_xi0_shell_grid.shape != expected:
        raise ValueError(f"GSM emulator correction shape {emulator.delta_xi0_shell_grid.shape}, expected {expected}")
    return emulator
