#!/usr/bin/env python3
"""Temporary diagnostic for the linear GSM normalization check."""

from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from task432_linear_gsm_rawbox import (  # noqa: E402
    LinearGSMModel,
    LinearRadialMomentProvider,
    LinearRadialMoments,
)


class FixedScaleBasis:
    def __init__(self, base: LinearRadialMomentProvider, velocity_scale: float, variance_scale: float) -> None:
        self.base = base
        self.velocity_scale = float(velocity_scale)
        self.variance_scale = float(variance_scale)
        self.radial_grid = base.radial_grid
        self.sigma_r2 = base.sigma_r2 * max(self.variance_scale, 1.0e-12)
        self.sigma_t2 = base.sigma_t2 * max(self.variance_scale, 1.0e-12)

    def provider(self, *, fnl: float, b1: float) -> LinearRadialMoments:
        value = self.base.provider(fnl=fnl, b1=b1)
        return LinearRadialMoments(
            radial_grid=value.radial_grid,
            xi_values=value.xi_values,
            v12_values=value.v12_values * self.velocity_scale,
            sigma_r2_values=value.sigma_r2_values * max(self.variance_scale, 1.0e-12),
            sigma_t2_values=value.sigma_t2_values * max(self.variance_scale, 1.0e-12),
            metadata=value._metadata,
        )


def main() -> None:
    cache = Path("/pscratch/sd/l/lzy/PNG-2PCF-model-to-observe/outputs/task43_outputs/rsd_validation/theory/task43_rsd_fulldiscrete_z0p725000_box2000_kmax3_ell02.npz")
    with np.load(cache, allow_pickle=False) as payload:
        edges = np.asarray(payload["s_edges"], dtype="f8")
    base = LinearRadialMomentProvider(cache, kmax_gsm=0.5, n_radial=2800)
    for label, velocity_scale, variance_scale in (
        ("no_rsd", 0.0, 1.0e-8),
        ("v_only", 1.0, 1.0e-8),
        ("sigma_only", 0.0, 1.0),
        ("v_half", 0.5, 1.0),
        ("full", 1.0, 1.0),
    ):
        basis = FixedScaleBasis(base, velocity_scale, variance_scale)
        model = LinearGSMModel(basis, edges, shell_order=4, angular_order=12, stream_order=8, zmax=8.0)
        result = model.evaluate(fnl=0.0, b1=2.55, sigma_fog=0.0)
        print(label, result[0][2:8].tolist(), result[2][2:8].tolist())

    # Check the first-order streaming expansion directly:
    # xi_s = xi_r - d[mu*v12]/dy + 0.5*d2[sigma12^2]/dy2.
    provider = base.provider(fnl=0.0, b1=2.55)
    mu_nodes, mu_weights = np.polynomial.legendre.leggauss(16)
    radial_nodes, radial_weights = np.polynomial.legendre.leggauss(6)
    h = 1.0e-2
    rows0, rows2 = [], []
    for lo, hi in zip(edges[:-1], edges[1:]):
        radii = 0.5 * (hi - lo) * radial_nodes + 0.5 * (hi + lo)
        weights = 0.5 * (hi - lo) * radial_weights * radii**2 / ((hi**3 - lo**3) / 3.0)
        vals0, vals2 = [], []
        for radius in radii:
            transverse = radius * np.sqrt(1.0 - mu_nodes**2)
            parallel = radius * mu_nodes
            xi, mean, variance = provider.at_los_array(transverse, parallel)
            def moments(y: float) -> tuple[float, float, float]:
                xx, mm, vv = provider.at_los_array(np.asarray([transverse[0]]), np.asarray([y]))
                return float(xx[0]), float(mm[0]), float(vv[0])
            # The vectorized provider is used for xi; scalar finite differences
            # use each mu node's fixed transverse separation below.
            dmean = np.empty_like(mu_nodes)
            d2var = np.empty_like(mu_nodes)
            for index, (trans, par) in enumerate(zip(transverse, parallel)):
                def one(y: float) -> tuple[float, float, float]:
                    xx, mm, vv = provider.at_los_array(np.asarray([trans]), np.asarray([y]))
                    return float(xx[0]), float(mm[0]), float(vv[0])
                _, mean_p, var_p = one(par + h)
                _, mean_m, var_m = one(par - h)
                _, _, var_0 = one(par)
                dmean[index] = (mean_p - mean_m) / (2.0 * h)
                d2var[index] = (var_p - 2.0 * var_0 + var_m) / h**2
            xi_linear = xi - dmean + 0.5 * d2var
            vals0.append(0.5 * (xi_linear @ mu_weights))
            vals2.append(xi_linear @ (2.5 * mu_weights * (3.0 * mu_nodes**2 - 1.0)))
        rows0.append(float(np.sum(weights * np.asarray(vals0))))
        rows2.append(float(np.sum(weights * np.asarray(vals2))))
    print("first_order", rows0[2:8], rows2[2:8])


if __name__ == "__main__":
    main()
