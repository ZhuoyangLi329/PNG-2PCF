#!/usr/bin/env python3
"""Unit tests for the Task 4.3 free-growth lightcone diagnostic."""

from __future__ import annotations

import numpy as np

from task43_lightcone_free_growth_test import FREE_NAMES, expand_fixed_theta, growth_coefficients


def test_growth_coefficients_match_direct_square() -> None:
    fnl, b1, f_growth = 17.0, 2.35, 0.79
    alpha = np.asarray([0.002, 0.01, 0.04], dtype="f8")
    mu2 = np.asarray([0.0, 0.3, 0.9], dtype="f8")
    q = fnl * 2.0 * 1.686 * (b1 - 1.0)
    basis = np.stack(
        (
            np.ones_like(alpha),
            alpha,
            alpha**2,
            mu2,
            alpha * mu2,
            mu2**2,
        ),
        axis=1,
    )
    expanded = basis @ growth_coefficients(fnl, b1, f_growth)
    direct = (b1 + q * alpha + f_growth * mu2) ** 2
    np.testing.assert_allclose(expanded, direct, rtol=2.0e-15, atol=0.0)


def test_fixed_theta_expansion_preserves_standard_order() -> None:
    fiducial_growth = 0.7968804024919754
    p = expand_fixed_theta("rsd_p02", np.asarray([1.0, 2.0, 3.0, 4.0]), fiducial_growth)
    x = expand_fixed_theta("rsd_xi02", np.asarray([1.0, 2.0, 3.0]), fiducial_growth)
    np.testing.assert_array_equal(p, [1.0, 2.0, 3.0, fiducial_growth, 4.0])
    np.testing.assert_array_equal(x, [1.0, 2.0, 3.0, fiducial_growth])
    assert FREE_NAMES["rsd_p02"] == ("fNL", "b1", "sigma_s", "f_growth", "sn0")
    assert FREE_NAMES["rsd_xi02"] == ("fNL", "b1", "sigma_s", "f_growth")
