#!/usr/bin/env python3
"""Tests for the independent xi-only linear RSD implementation."""

from __future__ import annotations

import numpy as np

from task43_rawbox_numerics import shell_kernel
from task43_xi_linear_rsd import XiLinearRSDModel


def make_cache(tmp_path, *, growth=0.81):
    path = tmp_path / "linear_xi_cache.npz"
    k = np.geomspace(2.0 * np.pi / 400.0, 0.3, 80)
    edges = np.asarray([20.0, 35.0, 50.0, 80.0])
    np.savez(
        path,
        k_eff=k,
        g_nz=np.linspace(1.0, 100.0, k.size),
        pk_dd=1.0e4 * np.exp(-(k / 0.2) ** 2),
        alpha=1.0 / (1.0 + (k / 0.03) ** 2),
        s_edges=edges,
        ells=np.asarray([0, 2]),
        kernels=np.stack([shell_kernel(k, edges, ell) for ell in (0, 2)]),
        volume=np.asarray(400.0**3),
        boxsize=np.asarray(400.0),
        f_growth=np.asarray(float(growth)),
    )
    return path


def test_analytic_angles_match_converged_gauss_legendre(tmp_path) -> None:
    model = XiLinearRSDModel(make_cache(tmp_path))
    analytic = model.evaluate(fnl=35.0, b1=2.5, sigma_s=8.0)
    numerical = model.evaluate_gauss_legendre(
        fnl=35.0,
        b1=2.5,
        sigma_s=8.0,
        nmu=512,
        use_cached_kernels=False,
    )
    for ell in (0, 2):
        np.testing.assert_allclose(analytic[ell], numerical[ell], rtol=2.0e-12, atol=1.0e-14)


def test_zero_fog_recovers_kaiser_multipoles(tmp_path) -> None:
    model = XiLinearRSDModel(make_cache(tmp_path))
    actual = model.evaluate(fnl=0.0, b1=2.5, sigma_s=0.0)
    power = model.pk_dd
    growth = model.f_growth
    expected_poles = {
        0: power * (2.5**2 + 2.0 * 2.5 * growth / 3.0 + growth**2 / 5.0),
        2: power * (4.0 * 2.5 * growth / 3.0 + 4.0 * growth**2 / 7.0),
    }
    for index, ell in enumerate((0, 2)):
        expected = ((model.g_nz * expected_poles[ell]) @ model.analytic_kernels[index]) / model.volume
        np.testing.assert_allclose(actual[ell], expected, rtol=2.0e-13, atol=1.0e-15)


def test_covariance_is_unfloored_and_valid(tmp_path) -> None:
    model = XiLinearRSDModel(make_cache(tmp_path))
    covariance = model.covariance(b1=2.5, sigma_s=8.0, nbar=2.0e-4)
    np.linalg.cholesky(covariance)
    assert covariance.shape == (6, 6)


def test_lowk_delta_vanishes_without_rsd_for_constant_radial_signal(tmp_path) -> None:
    model = XiLinearRSDModel(make_cache(tmp_path, growth=0.0))
    delta = model.lowk_angular_delta(fnl=0.0, b1=2.5, sigma_s=0.0, k_switch=0.08)
    np.testing.assert_allclose(delta[0], 0.0, atol=1.0e-14)
    np.testing.assert_allclose(delta[2], 0.0, atol=1.0e-14)


def test_realspace_pair_moments_have_signed_infall_and_positive_variances(tmp_path) -> None:
    model = XiLinearRSDModel(make_cache(tmp_path))
    edges = np.asarray([0.0, 20.0, 30.0, 50.0, 80.0, 120.0])
    moments = model.realspace_pair_moments(fnl=0.0, b1=2.5, radial_edges=edges)
    assert moments["xi_real"].shape == (edges.size - 1,)
    assert np.all(moments["v12_radial"] < 0.0)
    assert np.all(moments["sigma_r2_central"] > 0.0)
    assert np.all(moments["sigma_t2_one_component"] > 0.0)


def test_zero_growth_removes_all_linear_velocity_moments(tmp_path) -> None:
    model = XiLinearRSDModel(make_cache(tmp_path, growth=0.0))
    # The physical-variance guard is intentionally strict, so verify the
    # zero-growth boundary through a vanishingly small positive growth value.
    model.f_growth = 1.0e-12
    moments = model.realspace_pair_moments(
        fnl=0.0,
        b1=2.5,
        radial_edges=np.asarray([0.0, 20.0, 40.0, 80.0, 120.0]),
    )
    assert np.max(np.abs(moments["v12_radial"])) < 1.0e-10
    assert np.max(moments["sigma_r2_central"]) < 1.0e-20
    assert np.max(moments["sigma_t2_one_component"]) < 1.0e-20
