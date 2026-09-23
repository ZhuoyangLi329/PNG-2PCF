#!/usr/bin/env python3
"""Small deterministic tests for rawbox pair-moment accumulation."""

from __future__ import annotations

import numpy as np

from task43_measure_rawbox_pair_moments_v2 import (
    accumulate_anchor_pairs,
    finalize_mapping_counts,
    finalize_pair_sums,
    pair_moment_edges,
)


def test_origin_bin_is_merged_before_the_fine_grid() -> None:
    edges = pair_moment_edges(0.0, 40.0, 5.0, 20.0)
    np.testing.assert_array_equal(edges, [0.0, 20.0, 25.0, 30.0, 35.0, 40.0])


def test_pair_moments_use_real_separation_and_central_variance() -> None:
    positions = np.asarray(
        [
            [1.0, 1.0, 1.0],
            [3.0, 1.0, 1.0],
            [1.0, 4.0, 1.0],
            [1.0, 1.0, 5.0],
        ]
    )
    displacement = np.asarray(
        [
            [0.0, 0.0, 0.0],
            [-2.0, 0.0, 0.0],
            [0.0, -3.0, 0.0],
            [0.0, 0.0, -4.0],
        ]
    )
    edges = np.asarray([1.0, 5.0])
    sums = accumulate_anchor_pairs(
        positions,
        displacement,
        np.asarray([0]),
        edges,
        np.asarray([0.0, 0.5, 1.0]),
        boxsize=20.0,
        threads=1,
        query_chunk=1,
    )
    result = finalize_pair_sums(
        sums,
        nanchors=1,
        catalog_size=4,
        boxsize=20.0,
        radial_edges=edges,
    )
    np.testing.assert_allclose(result["v12_radial"], [-3.0])
    np.testing.assert_allclose(result["sigma_r2_central"], [2.0 / 3.0])
    np.testing.assert_allclose(result["sigma_t2_one_component"], [0.0])
    assert result["count"][0] == 3


def test_periodic_minimum_image_and_pair_orientation_are_consistent() -> None:
    positions = np.asarray([[0.5, 1.0, 1.0], [9.5, 1.0, 1.0]])
    displacement = np.asarray([[1.0, 0.0, 0.0], [3.0, 0.0, 0.0]])
    sums = accumulate_anchor_pairs(
        positions,
        displacement,
        np.asarray([0, 1]),
        np.asarray([0.5, 1.5]),
        np.asarray([0.0, 0.5, 1.0]),
        boxsize=10.0,
        threads=1,
        query_chunk=1,
    )
    result = finalize_pair_sums(
        sums,
        nanchors=2,
        catalog_size=2,
        boxsize=10.0,
        radial_edges=np.asarray([0.5, 1.5]),
    )
    assert result["count"][0] == 2
    np.testing.assert_allclose(result["v12_radial"], [-2.0])
    np.testing.assert_allclose(result["sigma_r2_central"], [0.0])


def test_direct_pair_mapping_uses_los_displacement() -> None:
    positions = np.asarray([[10.0, 0.0, 10.0], [10.0, 0.0, 20.0]])
    displacements = np.asarray([[0.0, 0.0, 0.0], [0.0, 0.0, 5.0]])
    mapped_edges = np.asarray([5.0, 12.0, 20.0])
    sums = accumulate_anchor_pairs(
        positions,
        displacements,
        np.asarray([0, 1]),
        np.asarray([0.0, 30.0]),
        np.asarray([0.0, 1.0]),
        boxsize=100.0,
        threads=1,
        mapping_edges=mapped_edges,
        mapping_nmu=2,
    )
    assert sums.mapping_real_count[0, 1] == 2.0
    assert sums.mapping_rsd_count[1, 1] == 2.0
    finalized = finalize_mapping_counts(
        sums,
        nanchors=2,
        catalog_size=2,
        boxsize=100.0,
        radial_edges=mapped_edges,
        nmu=2,
    )
    assert finalized["xi_smu_real"].shape == (2, 2)
