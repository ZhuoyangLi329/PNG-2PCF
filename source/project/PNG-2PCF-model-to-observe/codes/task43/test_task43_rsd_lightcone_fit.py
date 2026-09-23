#!/usr/bin/env python3
"""Fast algebra tests for the Task 4.3.2 lightcone likelihood surrogate."""

from __future__ import annotations

import numpy as np

from task43_fit_rsd_lightcone_x25 import FastWindowRSDModel
from task43_apply_rr_smu_deconvolution_to_covariance import scatter_comparison
from task43_rsd_common import S_EDGES
from task43_rsd_model import DELTA_C


class TinyExactModel:
    def __init__(self) -> None:
        self.k_eff = np.linspace(0.002, 0.05, 13)
        self.g_nz = np.linspace(1.0, 2.0, self.k_eff.size)
        self.pk_dd = 1000.0 / (1.0 + 20.0 * self.k_eff)
        self.alpha = 2.0e-5 / self.k_eff**2
        self.volume = np.asarray(8.0e9)
        self.f_growth = np.asarray(0.82)
        self.zeff = np.asarray(0.703)
        self.ell_values = (0, 2)
        self.mu, self.wmu = np.polynomial.legendre.leggauss(24)
        self.mu2 = self.mu**2
        self.legendre = {0: np.ones_like(self.mu), 2: 0.5 * (3.0 * self.mu2 - 1.0)}
        centers = 0.5 * (S_EDGES[:-1] + S_EDGES[1:])
        base = np.exp(-self.k_eff[:, None] * centers[None, :] / 25.0)
        self.kernels = np.stack([base, -0.35 * base])

    def evaluate(
        self,
        *,
        fnl: float,
        b1: float,
        sigma_s: float,
        p_fixed: float,
        fog_model: str = "lorentzian",
    ) -> dict[int, np.ndarray]:
        assert fog_model == "lorentzian"
        q = float(fnl) * 2.0 * DELTA_C * (float(b1) - float(p_fixed))
        amplitude = float(b1) + q * self.alpha
        damping = 1.0 / (
            1.0 + 0.5 * (self.k_eff[:, None] * self.mu[None, :] * float(sigma_s)) ** 2
        ) ** 2
        pk_mu = self.pk_dd[:, None] * (
            amplitude[:, None] + float(self.f_growth) * self.mu2[None, :]
        ) ** 2 * damping
        result: dict[int, np.ndarray] = {}
        for index, ell in enumerate(self.ell_values):
            pole = 0.5 * (2 * ell + 1) * np.sum(
                self.wmu[None, :] * pk_mu * self.legendre[ell][None, :], axis=1
            )
            result[ell] = ((self.g_nz * pole) @ self.kernels[index]) / float(self.volume)
        return result


def test_fast_window_model_matches_exact_formal_gic() -> None:
    exact = TinyExactModel()
    windows = {
        "mean": {
            "k_eff": exact.k_eff.copy(),
            "w2": np.linspace(0.03, 0.12, exact.k_eff.size),
            "meta": {"test": True},
        },
        "phase": {
            "k_eff": exact.k_eff.copy(),
            "w2": np.linspace(0.08, 0.01, exact.k_eff.size),
            "meta": {"test": True},
        },
    }
    model = FastWindowRSDModel(exact, windows, sigma_step=0.05)
    validation = model.validate()
    assert validation["status"] == "pass"
    assert validation["formal_gic_max_relative_l2"] < 1.0e-7

    theta = np.asarray([37.0, 2.4, 7.13])
    no_gic = model.evaluate(theta, model="no_gic", window_key="phase")
    formal = model.evaluate(theta, model="formal_gic", window_key="phase")
    correction = model.gic_value(theta, "phase")
    np.testing.assert_allclose(formal[0], no_gic[0] - correction, rtol=0.0, atol=1.0e-16)
    np.testing.assert_array_equal(formal[2], no_gic[2])


def test_scatter_comparison_accepts_xi0_xi2_summary(tmp_path) -> None:
    summary = tmp_path / "summary.npz"
    s = np.asarray([35.0, 45.0])
    xi = np.asarray(
        [
            [[1.0, 2.0], [0.1, 0.2]],
            [[1.2, 1.8], [0.0, 0.3]],
            [[0.8, 2.1], [0.2, 0.1]],
        ],
        dtype="f8",
    )
    np.savez(summary, s=s, ells=np.asarray([0, 2]), xi_multipoles_by_phase=xi)
    result = scatter_comparison(summary, np.asarray([30.0, 40.0, 50.0]), np.eye(4))
    assert result is not None
    assert result["available"] is True
    assert result["ells"] == [0, 2]
    assert result["nreal"] == 3
    assert result["nbins"] == 4
