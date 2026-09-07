#!/usr/bin/env python3
"""Fast tests for Task4.3.2 RSD model algebra."""

from __future__ import annotations

import numpy as np

from task43_rsd_model import shell_jell_kernel


def test_shell_j0_low_k_limit() -> None:
    kernel = shell_jell_kernel(np.array([1.0e-8]), np.array([30.0, 40.0]), 0)
    np.testing.assert_allclose(kernel, 1.0, rtol=0.0, atol=2.0e-5)


def test_shell_j2_has_fourier_phase() -> None:
    k = np.array([0.01, 0.02])
    edges = np.array([30.0, 40.0])
    kernel = shell_jell_kernel(k, edges, 2, nquad=48)
    assert np.all(kernel < 0.0)
