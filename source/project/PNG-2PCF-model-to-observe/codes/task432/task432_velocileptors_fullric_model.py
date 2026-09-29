#!/usr/bin/env python3
"""Velocileptors/GSM xi multipoles for the Task432 full-RIC operator."""
from __future__ import annotations
import os
for key in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(key, "1")
if not hasattr(__import__("numpy"), "trapezoid"):
    import numpy as _np
    _np.trapezoid = _np.trapz
from pathlib import Path
import sys
import numpy as np
ROOT = Path("/pscratch/sd/l/lzy/PNG-2PCF-model-to-observe")
for folder in (ROOT / "codes/task432", ROOT / "codes/task43", ROOT / "codes/task44"):
    if str(folder) not in sys.path:
        sys.path.insert(0, str(folder))
from task432_lightcone_png_velocileptors import LightconePNGVelocileptors, build_cache, RSD_P_PAYLOAD

class VeloPoles:
    """Direct GSM ell=(0,2,4) evaluator with explicit state reset per call."""
    def __init__(self, s: np.ndarray, *, zeff: float | None = None, kmax: float = 3.0):
        if zeff is None:
            with np.load(RSD_P_PAYLOAD, allow_pickle=False) as payload:
                zeff = float(np.asarray(payload["zeff"]).item())
        cache = build_cache(zeff=float(zeff), boxsize=2000.0, kmax=float(kmax), ells=(0,2), cosmology="abacus_c000")
        with np.load(cache, allow_pickle=False) as payload:
            self.k = np.asarray(payload["k_eff"], dtype="f8")
            self.pk = np.asarray(payload["pk_dd"], dtype="f8")
            self.alpha = np.asarray(payload["alpha"], dtype="f8")
            self.f_growth = float(np.asarray(payload["f_growth"]).item())
        self.s = np.asarray(s, dtype="f8")
        self.model = LightconePNGVelocileptors(self.k, self.pk, self.alpha, self.f_growth, self.s)

    def evaluate(self, fnl: float, b1: float, *, nint: int = 600) -> np.ndarray:
        self.model._set_cumulants(float(fnl), float(b1))
        b1_l = float(b1) - 1.0
        out = np.empty((3, self.s.size), dtype="f8")
        for index, radius in enumerate(self.s):
            poles = self.model.gsm.compute_xi_ell(
                float(radius), self.f_growth, b1_l, 0.0, 0.0, 0.0, 0.0, 0.0,
                0.0, 0.0, 0.0, rwidth=100.0, Nint=int(nint), ngauss=4,
                update_cumulants=False,
            )
            out[:, index] = np.asarray(poles[:3], dtype="f8")
        return out

    def evaluate_volume_average(self, edges: np.ndarray, fnl: float, b1: float, *, nint: int = 600, order: int = 4) -> np.ndarray:
        edges = np.asarray(edges, dtype="f8")
        nodes, weights = np.polynomial.legendre.leggauss(int(order))
        lo, hi = edges[:-1], edges[1:]
        radii = 0.5 * (hi-lo)[:, None] * nodes[None, :] + 0.5 * (hi+lo)[:, None]
        flat = self.evaluate(fnl, b1, nint=nint) if radii.shape[1] == 1 else VeloPoles(radii.ravel(), zeff=None).evaluate(fnl, b1, nint=nint)
        flat = flat.reshape(3, radii.shape[0], radii.shape[1])
        weights_r = 0.5 * (hi-lo)[None, :] * weights[None, :] * radii.T**2
        denom = (hi**3-lo**3)/3.0
        return np.sum(flat * weights_r[None, :, :], axis=2) / denom[None, :]
