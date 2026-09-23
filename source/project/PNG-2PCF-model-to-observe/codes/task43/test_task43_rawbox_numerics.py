#!/usr/bin/env python3
"""Regression tests for strict Task 4.3 rawbox numerical contracts."""

from __future__ import annotations

import numpy as np
import pytest
from scipy.integrate import quad
from scipy.optimize import least_squares
from scipy.special import spherical_jn

from task43_rawbox_numerics import (
    GaussianMetric,
    RAWBOX_FIT_EDGES,
    angular_total_integrals,
    assemble_covariance,
    canonical_correlations,
    conditional_xi_residual,
    continuous_rsd_poles,
    exact_lattice_modes,
    frozen_covariance_refit,
    gaussian_mode_covariance,
    lorentzian_moments,
    mode_operators,
    shell_kernel,
)

EXPECTED_COUNTS = np.asarray(
    [18, 14, 60, 86, 156, 234, 410, 672, 1262, 1656, 2570, 2994, 4190, 5070, 6000, 7170]
)


def test_gaussian_metric_is_unit_invariant() -> None:
    rng = np.random.default_rng(20260907)
    matrix = rng.normal(size=(7, 7))
    covariance = matrix @ matrix.T + np.diag(np.linspace(0.2, 0.8, 7))
    residual = rng.normal(size=7)
    units = np.asarray([1.0e6, 1.0e-6, 1.0e3, 1.0e-3, 1.0e2, 1.0e-2, 1.0])
    reference = GaussianMetric(covariance).chi2(residual)
    changed = GaussianMetric(covariance * np.outer(units, units)).chi2(residual * units)
    np.testing.assert_allclose(changed, reference, rtol=2.0e-12, atol=0.0)


def test_cross_zero_preserves_blocks_and_chi2_adds() -> None:
    c_pp = np.asarray([[2.0e9, 3.0e8], [3.0e8, 4.0e9]])
    c_xx = np.asarray([[3.0e-8, 1.0e-8], [1.0e-8, 2.0e-8]])
    residual_p = np.asarray([1.0e4, -2.0e4])
    residual_x = np.asarray([1.0e-4, -2.0e-4])
    joint = assemble_covariance(c_pp, c_xx)
    np.testing.assert_array_equal(joint[:2, :2], c_pp)
    np.testing.assert_array_equal(joint[2:, 2:], c_xx)
    expected = GaussianMetric(c_pp).chi2(residual_p) + GaussianMetric(c_xx).chi2(residual_x)
    actual = GaussianMetric(joint).chi2(np.concatenate([residual_p, residual_x]))
    np.testing.assert_allclose(actual, expected, rtol=0.0, atol=1.0e-12)


def test_invalid_cross_is_rejected() -> None:
    with pytest.raises(ValueError, match="automatic repair is forbidden"):
        assemble_covariance(np.eye(2), np.eye(2), 1.1 * np.eye(2))


def test_conditional_chi2_identity() -> None:
    c_pp = np.asarray([[2.0, 0.2], [0.2, 1.5]])
    c_xx = np.asarray([[1.2, 0.1], [0.1, 1.8]])
    c_xp = np.asarray([[0.1, -0.05], [0.02, 0.08]])
    residual_p = np.asarray([0.3, -0.4])
    residual_x = np.asarray([0.2, 0.7])
    _, _, conditional_chi2 = conditional_xi_residual(residual_p, residual_x, c_pp, c_xx, c_xp)
    joint = assemble_covariance(c_pp, c_xx, c_xp)
    joint_chi2 = GaussianMetric(joint).chi2(np.concatenate([residual_p, residual_x]))
    expected = GaussianMetric(c_pp).chi2(residual_p) + conditional_chi2
    np.testing.assert_allclose(joint_chi2, expected, rtol=0.0, atol=1.0e-12)
    assert canonical_correlations(c_pp, c_xx, c_xp)[0] < 1.0


def test_exact_lattice_counts_match_rawbox_measurement_contract() -> None:
    modes = exact_lattice_modes(2000.0, RAWBOX_FIT_EDGES)
    np.testing.assert_array_equal(modes.counts, EXPECTED_COUNTS)
    first = modes.bin_id == 0
    moments = [np.mean(modes.mu[first] ** power) for power in (2, 4, 6)]
    np.testing.assert_allclose(moments, [1.0 / 3.0, 2.0 / 9.0, 1.0 / 6.0], atol=1.0e-14)


def test_common_mode_covariance_blocks_are_self_consistent() -> None:
    modes = exact_lattice_modes(600.0, np.asarray([[0.01, 0.017], [0.017, 0.028]]))
    operator_p, operator_x = mode_operators(modes, np.asarray([20.0, 40.0, 65.0]))
    total = 2000.0 * (2.0 + 0.8 * modes.mu**2) ** 2 + 6000.0
    all_blocks = gaussian_mode_covariance(np.vstack([operator_p, operator_x]), total)
    n_p = operator_p.shape[0]
    np.testing.assert_allclose(all_blocks[:n_p, :n_p], gaussian_mode_covariance(operator_p, total))
    np.testing.assert_allclose(all_blocks[n_p:, n_p:], gaussian_mode_covariance(operator_x, total))
    rho = canonical_correlations(
        all_blocks[:n_p, :n_p], all_blocks[n_p:, n_p:], all_blocks[n_p:, :n_p]
    )
    assert rho[0] <= 1.0 + 1.0e-12


def test_lorentzian_moments_match_adaptive_quadrature() -> None:
    for exponent in (2, 4):
        for x in (0.0, 0.2, 8.0, 24.0, 90.0):
            moments = lorentzian_moments(np.asarray(x), max_power=6, exponent=exponent)
            for power in range(7):
                reference = quad(
                    lambda mu: mu ** (2 * power) / (1.0 + 0.5 * (x * mu) ** 2) ** exponent,
                    0.0,
                    1.0,
                    epsabs=1.0e-27,
                    epsrel=2.0e-12,
                    limit=300,
                )[0]
                np.testing.assert_allclose(moments[power], reference, rtol=1.0e-9, atol=1.0e-25)


def test_continuous_poles_recover_kaiser_limit() -> None:
    b1 = 2.55
    growth = 0.81
    poles = continuous_rsd_poles(np.asarray([0.003, 0.1]), 1.0, b1, growth, 0.0)
    np.testing.assert_allclose(poles[0], b1**2 + 2.0 * b1 * growth / 3.0 + growth**2 / 5.0)
    np.testing.assert_allclose(poles[2], 4.0 * b1 * growth / 3.0 + 4.0 * growth**2 / 7.0)


def test_angular_total_integrals_match_direct_quadrature() -> None:
    k = 0.19
    power = 3500.0
    bias = 2.6
    growth = 0.81
    sigma_s = 8.0
    shot = 6100.0
    actual = angular_total_integrals(k, power, bias, growth, sigma_s, shot)
    for (ell_a, ell_b), value in actual.items():
        reference = quad(
            lambda mu: (
                power
                * (bias + growth * mu**2) ** 2
                / (1.0 + 0.5 * (k * sigma_s * mu) ** 2) ** 2
                + shot
            )
            ** 2
            * np.polynomial.legendre.legval(mu, [0.0] * ell_a + [1.0])
            * np.polynomial.legendre.legval(mu, [0.0] * ell_b + [1.0]),
            -1.0,
            1.0,
            epsabs=1.0e-5,
            epsrel=1.0e-11,
        )[0]
        np.testing.assert_allclose(value, reference, rtol=1.0e-10, atol=1.0e-5)


def test_shell_kernel_matches_direct_quadrature() -> None:
    edges = np.asarray([30.0, 40.0, 110.0, 120.0, 350.0])
    kval = np.asarray([0.0, 1.0e-5, 0.003, 0.03, 0.3, 3.0])
    for ell in (0, 2):
        actual = shell_kernel(kval, edges, ell)
        expected = np.empty_like(actual)
        for ik, k in enumerate(kval):
            for ibin, (lower, upper) in enumerate(zip(edges[:-1], edges[1:])):
                expected[ik, ibin] = (
                    (-1) ** (ell // 2)
                    * 3.0
                    / (upper**3 - lower**3)
                    * quad(
                        lambda radius: radius**2 * spherical_jn(ell, k * radius),
                        lower,
                        upper,
                        epsabs=1.0e-8,
                        epsrel=2.0e-11,
                        limit=500,
                    )[0]
                )
        np.testing.assert_allclose(actual, expected, rtol=3.0e-9, atol=3.0e-13)


def test_frozen_covariance_refit_uses_final_metric() -> None:
    data = np.asarray([1.0, 2.0, 3.0])
    design = np.asarray([0.7, 1.0, 1.4])

    def covariance_builder(b1: float) -> np.ndarray:
        return np.diag([1.0, b1**2, 4.0])

    def fit(covariance: np.ndarray):
        cholesky = np.linalg.cholesky(covariance)
        return least_squares(lambda theta: np.linalg.solve(cholesky, data - theta[0] * design), [2.0])

    result = frozen_covariance_refit(covariance_builder, fit, initial_b1=2.5, b1_index=0)
    independent = fit(result["covariance"])
    np.testing.assert_allclose(result["final_theta"], independent.x, rtol=0.0, atol=1.0e-13)
    assert not np.array_equal(result["prefit_theta"], result["final_theta"])
