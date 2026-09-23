from __future__ import annotations

import numpy as np
import pytest

from task43_build_rsd_lightcone_random import repair_float32_redshift_boundaries


def test_repair_float32_upper_boundary_rounding() -> None:
    stored = np.asarray([np.float32(0.6000003), np.float32(0.7), np.float32(0.8)])
    repaired, metadata = repair_float32_redshift_boundaries(stored)
    assert np.all(repaired > 0.6)
    assert np.all(repaired < 0.8)
    assert metadata["n_upper_repaired"] == 1
    assert metadata["n_boundary_repaired"] == 1


def test_repair_float32_rejects_values_beyond_serialization_ulp() -> None:
    stored = np.asarray([np.nextafter(np.float32(0.8), np.float32(np.inf))])
    with pytest.raises(RuntimeError, match="float32 image"):
        repair_float32_redshift_boundaries(stored)
