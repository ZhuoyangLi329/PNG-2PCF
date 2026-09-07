#!/usr/bin/env python3
"""Cached FullDiscrete Kaiser x FoG model for Task4.3.2."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
from scipy.special import spherical_jn

from task43_rsd_common import OUTPUT_ROOT, S_EDGES, atomic_savez, atomic_write_json, sha256_file


DELTA_C = 1.686
TASK44_DIR = Path(__file__).resolve().parents[1] / "task44"
if str(TASK44_DIR) not in sys.path:
    sys.path.insert(0, str(TASK44_DIR))


def shell_jell_kernel(k: np.ndarray, edges: np.ndarray, ell: int, *, nquad: int = 24) -> np.ndarray:
    """Return volume-averaged i^ell*j_ell(kr) for every k and shell."""
    kval = np.asarray(k, dtype="f8")
    sedges = np.asarray(edges, dtype="f8")
    lo, hi = sedges[:-1], sedges[1:]
    shell_volume_no4pi = (hi**3 - lo**3) / 3.0
    if int(ell) == 0:
        kk = kval[:, None]
        upper = np.sin(kk * hi[None, :]) - kk * hi[None, :] * np.cos(kk * hi[None, :])
        lower = np.sin(kk * lo[None, :]) - kk * lo[None, :] * np.cos(kk * lo[None, :])
        result = (upper - lower) / (kk**3 * shell_volume_no4pi[None, :])
        small = np.abs(kval) * float(np.max(hi)) < 1.0e-3
        if np.any(small):
            mean_r2 = 3.0 / 5.0 * (hi**5 - lo**5) / (hi**3 - lo**3)
            mean_r4 = 3.0 / 7.0 * (hi**7 - lo**7) / (hi**3 - lo**3)
            ks = kval[small, None]
            result[small] = 1.0 - ks**2 * mean_r2[None, :] / 6.0 + ks**4 * mean_r4[None, :] / 120.0
    else:
        nodes, weights = np.polynomial.legendre.leggauss(int(nquad))
        radius = 0.5 * (hi - lo)[:, None] * nodes[None, :] + 0.5 * (hi + lo)[:, None]
        radial_weight = 0.5 * (hi - lo)[:, None] * weights[None, :] * radius**2
        result = np.empty((kval.size, lo.size), dtype="f8")
        for start in range(0, kval.size, 1024):
            stop = min(start + 1024, kval.size)
            values = spherical_jn(int(ell), kval[start:stop, None, None] * radius[None, :, :])
            result[start:stop] = np.sum(values * radial_weight[None, :, :], axis=2) / shell_volume_no4pi
    return ((-1.0) ** (int(ell) // 2)) * result


def cache_path(*, zeff: float, boxsize: float, kmax: float, ells: tuple[int, ...]) -> Path:
    ztag = f"{float(zeff):.6f}".replace(".", "p")
    ktag = f"{float(kmax):g}".replace(".", "p")
    etag = "".join(str(int(ell)) for ell in ells)
    return OUTPUT_ROOT / "theory" / f"task43_rsd_fulldiscrete_z{ztag}_box{boxsize:g}_kmax{ktag}_ell{etag}.npz"


def build_cache(
    *,
    zeff: float,
    boxsize: float = 2000.0,
    kmax: float = 3.0,
    ells: tuple[int, ...] = (0, 2),
    cosmology: str = "abacus_c000",
) -> Path:
    path = cache_path(zeff=zeff, boxsize=boxsize, kmax=kmax, ells=ells)
    metadata_path = path.with_suffix(".json")
    if path.is_file() and metadata_path.is_file():
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        if metadata.get("status") == "pass" and metadata.get("output_sha256") == sha256_file(path):
            return path
        raise FileExistsError(f"unvalidated theory cache: {path}")
    if path.exists() or metadata_path.exists():
        raise FileExistsError(f"partial theory cache: {path} / {metadata_path}")
    from task44_rsd_theory import build_theory_context

    context = build_theory_context(
        float(zeff), kmax=float(kmax), ndense=30_000, boxsize=float(boxsize), cosmology=str(cosmology)
    )
    task41 = context["task41"]
    k_eff = np.asarray(context["k_eff"], dtype="f8")
    pk_dd = task41.interp_logk(k_eff, context["template"]["k"], context["template"]["pk_dd"])
    alpha = task41.interp_logk(k_eff, context["template"]["k"], context["template"]["alpha"])
    kernels = {int(ell): shell_jell_kernel(k_eff, S_EDGES, int(ell)) for ell in ells}
    atomic_savez(
        path,
        k_eff=k_eff,
        g_nz=np.asarray(context["g_nz"], dtype="f8"),
        pk_dd=np.asarray(pk_dd, dtype="f8"),
        alpha=np.asarray(alpha, dtype="f8"),
        s_edges=np.asarray(S_EDGES, dtype="f8"),
        ells=np.asarray(ells, dtype="i4"),
        kernels=np.stack([kernels[int(ell)] for ell in ells]),
        zeff=np.asarray(float(zeff), dtype="f8"),
        f_growth=np.asarray(float(context["f_growth"]), dtype="f8"),
        boxsize=np.asarray(float(boxsize), dtype="f8"),
        volume=np.asarray(float(context["volume"]), dtype="f8"),
        kfund=np.asarray(float(context["kfund"]), dtype="f8"),
        kmax=np.asarray(float(kmax), dtype="f8"),
    )
    metadata = {
        "task": "task43_rsd_model_cache",
        "status": "pass",
        "definition": "shell-averaged FullDiscrete Kaiser x Lorentzian/Gaussian FoG PNG multipoles",
        "cosmology": cosmology,
        "zeff": float(zeff),
        "boxsize_mpc_h": float(boxsize),
        "kmax_h_mpc": float(kmax),
        "kfund_h_mpc": float(context["kfund"]),
        "nmodes_rebinned": int(k_eff.size),
        "ells": list(ells),
        "s_edges_mpc_h": S_EDGES.tolist(),
        "growth_rate": float(context["f_growth"]),
        "growth_rate_method": context["growth_rate_method"],
        "output_path": str(path),
        "output_sha256": sha256_file(path),
    }
    atomic_write_json(metadata_path, metadata)
    return path


class FullDiscreteRSDModel:
    def __init__(self, path: Path, *, nmu: int = 96) -> None:
        with np.load(path, allow_pickle=False) as data:
            for key in data.files:
                setattr(self, key, np.asarray(data[key]))
        self.path = Path(path)
        self.nmu = int(nmu)
        self.mu, self.wmu = np.polynomial.legendre.leggauss(self.nmu)
        self.mu2 = self.mu**2
        self.ell_values = tuple(int(value) for value in np.asarray(self.ells).ravel())
        self.legendre: dict[int, np.ndarray] = {}
        for ell in self.ell_values:
            coeff = np.zeros(ell + 1, dtype="f8")
            coeff[ell] = 1.0
            self.legendre[ell] = np.polynomial.legendre.legval(self.mu, coeff)

    def evaluate(
        self,
        *,
        fnl: float,
        b1: float,
        sigma_s: float,
        p_fixed: float = 1.0,
        fog_model: str = "lorentzian",
    ) -> dict[int, np.ndarray]:
        bphi = 2.0 * DELTA_C * (float(b1) - float(p_fixed))
        amplitude = float(b1) + float(fnl) * bphi * self.alpha
        kmusigma2 = (self.k_eff[:, None] * self.mu[None, :] * float(sigma_s)) ** 2
        if fog_model == "lorentzian":
            damping = 1.0 / (1.0 + 0.5 * kmusigma2) ** 2
        elif fog_model == "gaussian":
            damping = np.exp(-kmusigma2)
        else:
            raise ValueError(f"unknown FoG model {fog_model!r}")
        pkmu = self.pk_dd[:, None] * (amplitude[:, None] + float(self.f_growth) * self.mu2[None, :]) ** 2 * damping
        result: dict[int, np.ndarray] = {}
        for index, ell in enumerate(self.ell_values):
            pole = 0.5 * (2 * ell + 1) * np.sum(
                self.wmu[None, :] * pkmu * self.legendre[ell][None, :], axis=1
            )
            result[ell] = ((self.g_nz * pole) @ self.kernels[index]) / float(self.volume)
        return result

    def vector(self, theta: np.ndarray, *, ells: tuple[int, ...] = (0, 2), p_fixed: float = 1.0) -> np.ndarray:
        fnl, b1, sigma_s = np.asarray(theta, dtype="f8")[:3]
        values = self.evaluate(fnl=fnl, b1=b1, sigma_s=sigma_s, p_fixed=p_fixed)
        return np.concatenate([values[int(ell)] for ell in ells])


def load_or_build(**kwargs: Any) -> FullDiscreteRSDModel:
    return FullDiscreteRSDModel(build_cache(**kwargs))
