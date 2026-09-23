#!/usr/bin/env python3

from __future__ import annotations

import numpy as np
import pytest

from task43_xi_realspace_ir import FiniteBoxIRXiModel, bao_displacement_sigma, ir_resummed_power


def test_ir_sigma_zero_recovers_linear_power_exactly() -> None:
    k = np.geomspace(1.0e-3, 2.0, 128)
    plin = 100.0 * k ** -0.7 * (1.0 + 0.04 * np.sin(90.0 * k))
    pnow = 100.0 * k ** -0.7
    assert np.array_equal(ir_resummed_power(k, plin, pnow, 0.0), plin)


def test_ir_large_sigma_approaches_smooth_power() -> None:
    k = np.geomspace(0.05, 2.0, 128)
    plin = 100.0 * k ** -0.7 * (1.0 + 0.04 * np.sin(90.0 * k))
    pnow = 100.0 * k ** -0.7
    result = ir_resummed_power(k, plin, pnow, 1.0e4)
    assert np.array_equal(result, pnow)


def test_bao_displacement_requires_covered_ir_cutoff() -> None:
    q = np.geomspace(1.0e-4, 0.1, 128)
    with pytest.raises(ValueError, match="cover k_ir"):
        bao_displacement_sigma(q, np.ones_like(q), r_bao=100.0, k_ir=0.2)


def test_finite_box_model_matches_direct_linear_sum_at_sigma_zero() -> None:
    rng = np.random.default_rng(42)
    k = np.geomspace(0.003, 2.0, 50)
    degeneracy = rng.integers(1, 100, k.size).astype("f8")
    plin = 1000.0 / (1.0 + k) ** 2
    pnow = plin * (1.0 + 0.01 * np.cos(60.0 * k))
    alpha = 0.01 / (1.0 + k) ** 2
    kernel = rng.normal(size=(k.size, 7))
    model = FiniteBoxIRXiModel(k, degeneracy, plin, pnow, alpha, kernel, 8.0e9)
    fnl, b1 = -7.0, 2.6
    amplitude = b1 + fnl * 2.0 * 1.686 * (b1 - 1.0) * alpha
    expected = ((degeneracy * plin * amplitude**2) @ kernel) / 8.0e9
    assert np.allclose(model.evaluate(fnl=fnl, b1=b1, sigma_bao=0.0), expected, rtol=2.0e-15, atol=0.0)

