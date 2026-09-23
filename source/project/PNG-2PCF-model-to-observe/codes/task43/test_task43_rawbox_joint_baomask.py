"""Focused tests for the strict BAO-masked rawbox joint construction."""

from __future__ import annotations

import numpy as np

from task43_rawbox_numerics import RAWBOX_FIT_EDGES, exact_lattice_modes
from task43_plot_rawbox_joint_baomask_v1 import interval_text
from task43_rsd_common import S_EDGES, rawbox_xi_primary_mask
from task43_run_rawbox_joint_baomask_v1 import common_mode_covariance
from task43_rsd_model import FullDiscreteRSDModel, build_cache


def _small_model() -> tuple[FullDiscreteRSDModel, object, np.ndarray]:
    exact = FullDiscreteRSDModel(
        build_cache(zeff=0.725, boxsize=2000.0, kmax=3.0, ells=(0, 2), cosmology="abacus_c000"),
        nmu=32,
    )
    modes = exact_lattice_modes(2000.0, RAWBOX_FIT_EDGES)
    centers = 0.5 * (S_EDGES[:-1] + S_EDGES[1:])
    return exact, modes, rawbox_xi_primary_mask(centers)


def test_real_common_mode_covariance_preserves_blocks_without_repair() -> None:
    exact, modes, mask = _small_model()
    result = common_mode_covariance(exact, modes, mask, nbar=1.62131295e-4, b1=2.64, sigma_s=None)
    n_p = RAWBOX_FIT_EDGES.shape[0]
    np.testing.assert_array_equal(result["joint"][:n_p, :n_p], result["pp"])
    np.testing.assert_array_equal(result["joint"][n_p:, n_p:], result["xx"])
    np.testing.assert_array_equal(result["joint"][n_p:, :n_p], result["xp"])
    assert result["joint"].shape == (42, 42)
    assert result["correlation_eigenvalue_min"]["joint"] > 1.0e-12
    assert np.max(result["canonical_correlations"]) < 1.0


def test_rsd_common_mode_covariance_is_strict_spd() -> None:
    exact, modes, mask = _small_model()
    result = common_mode_covariance(exact, modes, mask, nbar=1.62131295e-4, b1=2.55, sigma_s=8.0)
    assert result["pp"].shape == (32, 32)
    assert result["xx"].shape == (52, 52)
    assert result["xp"].shape == (52, 32)
    assert result["joint"].shape == (84, 84)
    assert result["correlation_eigenvalue_min"]["joint"] > 1.0e-12
    assert np.max(result["canonical_correlations"]) < 1.0
    assert np.all(np.diag(result["xx_tail"]) > 0.0)


def test_interval_text_uses_explicit_interval_when_ml_is_outside_marginal_68_percent() -> None:
    variant = {
        "parameter_names": np.asarray(["sigma_s"]),
        "chain_by_step": np.linspace(0.3, 1.9, 1000).reshape(100, 10, 1),
        "theta_maximum_likelihood": np.asarray([0.0]),
    }
    label = interval_text(variant, "sigma_s")
    assert label.startswith(r"0.00;\ 68\%=[")
    assert "--" not in label
