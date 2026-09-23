"""Focused tests for the standard Task 4.3 lightcone BAO-mask runner."""

from __future__ import annotations

import numpy as np

from task43_run_lightcone_joint_baomask_v1 import ScaledGaussianMetric, lightcone_xi_primary_mask


def test_lightcone_primary_mask_matches_rawbox_policy() -> None:
    centers = np.arange(35.0, 350.0, 10.0)
    mask = lightcone_xi_primary_mask(centers)
    assert np.count_nonzero(mask) == 26
    np.testing.assert_array_equal(centers[(centers >= 50.0) & (centers < 350.0) & ~mask], [85.0, 95.0, 105.0, 115.0])


def test_lightcone_smin40_adds_only_the_45_bin() -> None:
    centers = np.arange(35.0, 350.0, 10.0)
    mask50 = lightcone_xi_primary_mask(centers)
    mask40 = lightcone_xi_primary_mask(centers, smin=40.0)
    assert np.count_nonzero(mask40) == 27
    np.testing.assert_array_equal(centers[mask40 & ~mask50], [45.0])
    np.testing.assert_array_equal(centers[(centers >= 40.0) & (centers < 350.0) & ~mask40], [85.0, 95.0, 105.0, 115.0])


def test_scaled_metric_preserves_chi2_across_block_units() -> None:
    covariance = np.asarray([[4.0e8, 1.2e-1], [1.2e-1, 9.0e-10]])
    difference = np.asarray([2.5e3, -1.0e-5])
    metric = ScaledGaussianMetric(covariance)
    expected = float(difference @ np.linalg.solve(covariance, difference))
    np.testing.assert_allclose(metric.chi2(difference), expected, rtol=1.0e-13, atol=1.0e-13)
    assert metric.eigenvalues[0] > 1.0e-12
