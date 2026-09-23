from __future__ import annotations

import numpy as np

from task43_make_rsd_lightcone_pk0_covariance_jaxpower import extract_p0_block, rsd_multipoles
from task43_rsd_lightcone_pk0_contract import (
    MEASUREMENT_DK,
    MEASUREMENT_KMAX,
    MEASUREMENT_KMIN,
    TASK43_REALSPACE_FIT_EDGES,
    TASK43_REALSPACE_OBSERVED_KMIN,
    TASK43_REALSPACE_VOLUME,
    WINDOW_THEORY_KMIN,
    load_task43_realspace_reference,
    observed_fit_kmin,
)


def test_frozen_task43_realspace_contract() -> None:
    reference = load_task43_realspace_reference()
    assert reference["checks"] and all(reference["checks"].values())
    assert np.allclose(reference["fit_edges"], TASK43_REALSPACE_FIT_EDGES, rtol=0.0, atol=1.0e-15)
    assert np.isclose(
        observed_fit_kmin(TASK43_REALSPACE_VOLUME), TASK43_REALSPACE_OBSERVED_KMIN, rtol=0.0, atol=1.0e-15
    )
    assert MEASUREMENT_KMIN < WINDOW_THEORY_KMIN < TASK43_REALSPACE_OBSERVED_KMIN
    assert (MEASUREMENT_KMIN, MEASUREMENT_KMAX, MEASUREMENT_DK) == (0.001, 0.3001, 0.002)


def test_rsd_multipoles_reduce_to_analytic_kaiser_limit() -> None:
    k = np.asarray([0.01, 0.05, 0.09])
    pk_dd = np.asarray([1000.0, 500.0, 250.0])
    template = {
        "k": k,
        "pk_dd": pk_dd,
        "alpha": np.zeros_like(k),
    }
    b1, growth = 2.2, 0.8
    poles = rsd_multipoles(
        k,
        template=template,
        f_growth=growth,
        fnl=0.0,
        b1=b1,
        sigma_s=0.0,
        p_fixed=1.0,
        nmu=64,
    )
    expected = {
        0: pk_dd * (b1**2 + 2.0 * b1 * growth / 3.0 + growth**2 / 5.0),
        2: pk_dd * (4.0 * b1 * growth / 3.0 + 4.0 * growth**2 / 7.0),
        4: pk_dd * (8.0 * growth**2 / 35.0),
    }
    for ell in (0, 2, 4):
        assert np.allclose(poles[ell], expected[ell], rtol=2.0e-13, atol=2.0e-10)


def test_extract_p0_block_from_ordered_ell024_covariance() -> None:
    full = np.arange(36, dtype="f8").reshape(6, 6)
    full = full + full.T
    block, metadata = extract_p0_block(full, nk=2, theory_ells=(0, 2, 4))
    assert np.array_equal(block, full[:2, :2])
    assert metadata["source_shape"] == [6, 6]
    assert metadata["selected_shape"] == [2, 2]
    assert metadata["theory_ells_order"] == [0, 2, 4]
