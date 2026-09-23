#!/usr/bin/env python3
"""Unit tests for the Task 4.3.2 coordinate and atomic-I/O contract."""

from __future__ import annotations

import json

import numpy as np

from task43_rsd_common import (
    apply_official_lightcone_fallback,
    apply_plane_parallel_rsd,
    apply_radial_rsd,
    atomic_savez,
    atomic_write_json,
    rawbox_xi_primary_mask,
)


def test_plane_parallel_zero_velocity_is_exact() -> None:
    position = np.array([[1.0, 2.0, 3.0], [1999.0, 5.0, 1998.0]])
    shifted, displacement = apply_plane_parallel_rsd(
        position,
        np.zeros_like(position),
        velocity_kms_per_mpc_h_value=100.0,
        boxsize_mpc_h=2000.0,
    )
    np.testing.assert_array_equal(shifted, position)
    np.testing.assert_array_equal(displacement, np.zeros(2))


def test_plane_parallel_shift_and_wrap() -> None:
    position = np.array([[1.0, 2.0, 1999.0], [4.0, 5.0, 1.0]])
    velocity = np.array([[0.0, 0.0, 200.0], [0.0, 0.0, -300.0]])
    shifted, displacement = apply_plane_parallel_rsd(
        position,
        velocity,
        velocity_kms_per_mpc_h_value=100.0,
        boxsize_mpc_h=2000.0,
    )
    np.testing.assert_allclose(displacement, [2.0, -3.0])
    np.testing.assert_allclose(shifted[:, 2], [1.0, 1998.0])
    np.testing.assert_array_equal(shifted[:, :2], position[:, :2])


def test_radial_mapping_preserves_angles_and_uses_origin_modulo() -> None:
    origins = np.array([[0.0, 0.0, 0.0], [10.0, 0.0, 0.0]])
    position = np.array([[3.0, 4.0, 0.0], [13.0, 4.0, 0.0]])
    velocity = np.array([[60.0, 80.0, 0.0], [60.0, 80.0, 0.0]])
    result = apply_radial_rsd(
        position,
        velocity,
        origins,
        np.array([0, 3]),
        np.array([100.0, 100.0]),
    )
    np.testing.assert_array_equal(result["origin_index"], [0, 1])
    np.testing.assert_allclose(result["velocity_los"], [100.0, 100.0])
    np.testing.assert_allclose(result["displacement"], [1.0, 1.0])
    np.testing.assert_allclose(result["radius_real"], [5.0, 5.0])
    np.testing.assert_allclose(result["radius_rsd"], [6.0, 6.0])
    direction_after = result["position_rsd_relative"] / result["radius_rsd"][:, None]
    np.testing.assert_allclose(direction_after, result["direction"], atol=1.0e-15)


def test_atomic_writers(tmp_path) -> None:
    json_path = tmp_path / "payload.json"
    npz_path = tmp_path / "payload.npz"
    atomic_write_json(json_path, {"status": "pass", "value": np.float64(2.0)})
    atomic_savez(npz_path, value=np.arange(4), label=np.asarray("test"))
    assert json.loads(json_path.read_text())["status"] == "pass"
    with np.load(npz_path, allow_pickle=False) as data:
        np.testing.assert_array_equal(data["value"], np.arange(4))
        assert str(data["label"]) == "test"
    assert not list(tmp_path.glob(".*.tmp"))


def test_official_lightcone_fallback_replaces_only_available_averages() -> None:
    position_interp = np.arange(12, dtype="f8").reshape(4, 3)
    velocity_interp = position_interp + 100.0
    position_average = np.zeros_like(position_interp)
    velocity_average = np.zeros_like(velocity_interp)
    position_average[2:] = position_interp[2:] + 1000.0
    velocity_average[2:] = velocity_interp[2:] + 2000.0
    position, velocity, mask = apply_official_lightcone_fallback(
        position_interp,
        velocity_interp,
        position_average,
        velocity_average,
        np.array([0, 2, 3, 5]),
    )
    np.testing.assert_array_equal(mask, [False, False, True, True])
    np.testing.assert_array_equal(position[:2], position_interp[:2])
    np.testing.assert_array_equal(velocity[:2], velocity_interp[:2])
    np.testing.assert_array_equal(position[2:], position_average[2:])
    np.testing.assert_array_equal(velocity[2:], velocity_average[2:])


def test_rawbox_primary_mask_removes_exactly_four_bao_centers() -> None:
    centers = np.arange(35.0, 350.0, 10.0)
    mask = rawbox_xi_primary_mask(centers)
    np.testing.assert_array_equal(centers[~mask & (centers >= 50.0)], [85.0, 95.0, 105.0, 115.0])
    np.testing.assert_array_equal(centers[mask][:4], [55.0, 65.0, 75.0, 125.0])
    assert int(np.count_nonzero(mask)) == 26
