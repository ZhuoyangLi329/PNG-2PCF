#!/usr/bin/env python3
"""Synthetic unit tests for the independent xi-only rawbox GSM."""

from __future__ import annotations

import math

import numpy as np
import pytest

from task43_xi_gsm import (
    EmpiricalRadialMoments,
    fcfc_project_smu,
    fcfc_project_smu_vectorized,
    gsm_point,
    gsm_points_fixed,
    shell_quadrature,
)


def example_provider() -> EmpiricalRadialMoments:
    edges = np.asarray([0.0, 20.0, 30.0, 40.0, 60.0, 100.0])
    radius = 0.75 * (edges[1:] ** 4 - edges[:-1] ** 4) / (edges[1:] ** 3 - edges[:-1] ** 3)
    return EmpiricalRadialMoments.from_arrays(
        edges,
        0.2 * np.exp(-radius / 40.0),
        -2.0 * np.exp(-radius / 50.0),
        20.0 + 0.02 * radius,
        12.0 + 0.01 * radius,
    )


def test_provider_uses_real_mu_and_one_transverse_component() -> None:
    provider = EmpiricalRadialMoments.from_arrays(
        np.asarray([0.0, 10.0, 20.0]),
        np.asarray([0.1, 0.1]),
        np.asarray([-3.0, -3.0]),
        np.asarray([25.0, 25.0]),
        np.asarray([9.0, 9.0]),
    )
    xi, mean, variance = provider.at_los(3.0, 4.0)
    assert xi == pytest.approx(0.1)
    assert mean == pytest.approx(-2.4)
    assert variance == pytest.approx(0.64 * 25.0 + 0.36 * 9.0)
    assert provider.at_los(3.0, -4.0)[1] == pytest.approx(2.4)


def test_provider_rejects_extrapolation_and_invalid_variance() -> None:
    provider = example_provider()
    with pytest.raises(ValueError):
        provider.evaluate_radial(100.1)
    with pytest.raises(ValueError):
        EmpiricalRadialMoments.from_arrays(
            [0.0, 10.0, 20.0], [0.1, 0.1], [-1.0, -1.0], [4.0, -1.0], [3.0, 3.0]
        )


def test_gsm_homogeneous_unclustered_and_periodic_images() -> None:
    moments = lambda transverse, parallel: (0.0, 0.0, 36.0)
    base, _ = gsm_point(50.0, 20.0, moments, integration_scale=6.0, zmax=12.0)
    periodized, _ = gsm_point(
        50.0,
        20.0,
        moments,
        integration_scale=6.0,
        zmax=12.0,
        boxsize=2000.0,
        periodic_images=1,
    )
    assert base == pytest.approx(0.0, abs=1.0e-12)
    assert periodized == pytest.approx(base, abs=1.0e-14)


def test_gsm_parity_for_isotropic_radial_provider() -> None:
    provider = example_provider()
    scale = provider.integration_scale
    positive, _ = gsm_point(35.0, 30.0, provider.at_los, integration_scale=scale, zmax=10.0)
    negative, _ = gsm_point(35.0, -30.0, provider.at_los, integration_scale=scale, zmax=10.0)
    assert positive == pytest.approx(negative, abs=2.0e-11)


def test_fixed_quadrature_matches_adaptive_reference() -> None:
    provider = example_provider()
    transverse = np.asarray([35.0, 45.0, 65.0])
    parallel = np.asarray([30.0, 10.0, 20.0])
    fixed = gsm_points_fixed(
        transverse,
        parallel,
        provider,
        integration_scale=provider.integration_scale,
        zmax=10.0,
        quadrature_order=32,
        boxsize=2000.0,
        periodic_images=1,
    )
    adaptive = np.asarray(
        [
            gsm_point(
                float(perpendicular),
                float(los),
                provider.at_los,
                integration_scale=provider.integration_scale,
                zmax=10.0,
                boxsize=2000.0,
                periodic_images=1,
            )[0]
            for perpendicular, los in zip(transverse, parallel)
        ]
    )
    np.testing.assert_allclose(fixed, adaptive, rtol=2.0e-8, atol=2.0e-10)


def test_zero_velocity_requires_all_velocity_terms_to_vanish() -> None:
    moments = lambda transverse, parallel: (0.1 * math.exp(-(transverse**2 + parallel**2) / 1.0e4), 0.0, 0.0)
    value, error = gsm_point(30.0, 40.0, moments, integration_scale=1.0, zero_velocity=True)
    assert value == pytest.approx(0.1 * math.exp(-0.25))
    assert error == 0.0
    with pytest.raises(ValueError):
        gsm_point(30.0, 40.0, lambda p, y: (0.1, 1.0, 0.0), integration_scale=1.0, zero_velocity=True)


def test_shell_quadrature_preserves_volume_average() -> None:
    edges = np.asarray([1.0, 2.0, 4.0])
    radius, weight = shell_quadrature(edges, 8)
    np.testing.assert_allclose(weight.sum(axis=1), 1.0, atol=1.0e-14)
    measured = np.sum(weight * radius**2, axis=1)
    expected = (3.0 / 5.0) * (edges[1:] ** 5 - edges[:-1] ** 5) / (edges[1:] ** 3 - edges[:-1] ** 3)
    np.testing.assert_allclose(measured, expected, rtol=1.0e-13)


def test_fcfc_projection_keeps_finite_mu_midpoint_definition() -> None:
    radial_edges = np.asarray([30.0, 40.0, 50.0])
    mu_edges = np.linspace(0.0, 1.0, 121)
    constant = 0.125
    smu, poles = fcfc_project_smu(
        lambda transverse, parallel: constant,
        radial_edges,
        mu_edges,
        nradial=3,
        nmu_per_bin=2,
    )
    np.testing.assert_allclose(smu, constant, atol=1.0e-15)
    mu_midpoint = 0.5 * (mu_edges[:-1] + mu_edges[1:])
    np.testing.assert_allclose(poles[0], constant, atol=1.0e-15)
    np.testing.assert_allclose(poles[2], 5.0 * constant * np.mean(0.5 * (3.0 * mu_midpoint**2 - 1.0)))

    vector_smu, vector_poles = fcfc_project_smu_vectorized(
        lambda transverse, parallel: np.full(np.broadcast_shapes(transverse.shape, parallel.shape), constant),
        radial_edges,
        mu_edges,
        nradial=3,
        nmu_per_bin=2,
    )
    np.testing.assert_allclose(vector_smu, smu, atol=1.0e-15)
    for ell in (0, 2, 4):
        np.testing.assert_allclose(vector_poles[ell], poles[ell], atol=1.0e-15)
