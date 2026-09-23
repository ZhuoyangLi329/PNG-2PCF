#!/usr/bin/env python3
"""Tests for target-blind pair-moment smoothing selection."""

from __future__ import annotations

import numpy as np
import pytest

from task43_pair_moment_smoothing import (
    apply_smoothing_factor,
    cross_validated_smoothing,
    shell_volume_centers,
)


def test_shell_volume_centers_match_analytic_r_mean() -> None:
    edges = np.asarray([0.0, 2.0, 4.0, 6.0, 8.0])
    centers = shell_volume_centers(edges)
    expected = 0.75 * (edges[1:] ** 4 - edges[:-1] ** 4) / (edges[1:] ** 3 - edges[:-1] ** 3)
    np.testing.assert_allclose(centers, expected)


def test_cross_validation_smooths_noise_without_using_a_target_vector() -> None:
    rng = np.random.default_rng(4302)
    edges = np.concatenate(([0.0], np.arange(20.0, 105.0, 5.0)))
    radius = shell_volume_centers(edges)
    truth = -3.0 * np.exp(-radius / 50.0)
    blocks = truth[None, :] + rng.normal(scale=0.25, size=(8, radius.size))
    smoothed, audit = cross_validated_smoothing(edges, blocks)
    pooled = np.mean(blocks, axis=0)
    assert np.sqrt(np.mean((smoothed - truth) ** 2)) < np.sqrt(np.mean((pooled - truth) ** 2))
    assert audit["selection"].startswith("leave-one-anchor-block-out")
    assert audit["selected_factor"] in [row["factor"] for row in audit["candidates"]]


def test_positive_constraint_rejects_invalid_inputs_or_candidates() -> None:
    edges = np.asarray([0.0, 20.0, 25.0, 30.0, 35.0, 40.0])
    blocks = np.ones((8, 5))
    blocks[:, 2] = -1.0
    with pytest.raises(RuntimeError):
        cross_validated_smoothing(edges, blocks, factors=(0.0,), require_positive=True)


def test_zero_factor_reproduces_pooled_values_at_knots() -> None:
    edges = np.asarray([0.0, 20.0, 25.0, 30.0, 35.0, 40.0])
    values = np.asarray([-3.0, -2.0, -1.5, -1.0, -0.8])
    actual = apply_smoothing_factor(edges, values, np.full(values.size, 0.1), 0.0)
    np.testing.assert_allclose(actual, values, atol=1.0e-14)
