#!/usr/bin/env python3
"""Frozen Task 4.3 real-space P0 contract reused by Task 4.3.2 lightcone RSD.

There are three intentionally distinct low-k scales:

* the estimator/covariance grid starts at 0.001 h/Mpc;
* observed bins are fitted above 2 pi / V_eff^(1/3);
* the theory vector entering the window is cut at the parent-box fundamental
  mode, 2 pi / 2000 h/Mpc.

Keeping these values in one module prevents the generic word ``kmin`` from
silently changing one of the other two policies.
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any

import numpy as np


PROJECT_ROOT = Path("/pscratch/sd/l/lzy/PNG-2PCF-model-to-observe")
TASK43_REALSPACE_REFERENCE = (
    PROJECT_ROOT
    / "plots/outputs/task43_outputs/pk_lightcone/summary/"
    "task43_pk_lightcone_mmin1p4e13_x25_fkpP010000_desi_rebin_kmax0p10_payload.npz"
)

MEASUREMENT_KMIN = 0.001
MEASUREMENT_KMAX = 0.3001
MEASUREMENT_DK = 0.002
FIT_KMAX = 0.10
PARENT_BOX_SIZE = 2000.0
WINDOW_THEORY_KMIN = 2.0 * math.pi / PARENT_BOX_SIZE

TASK43_REALSPACE_VOLUME = 1908169922.664481
TASK43_REALSPACE_LEFF = 1240.3350470500761
TASK43_REALSPACE_OBSERVED_KMIN = 0.005065716172515695

TASK43_REALSPACE_FIT_EDGES = np.asarray(
    [
        [0.005, 0.007],
        [0.007, 0.009],
        [0.009, 0.011],
        [0.013, 0.015],
        [0.017, 0.019],
        [0.021, 0.023],
        [0.029, 0.031],
        [0.037, 0.039],
        [0.045, 0.047],
        [0.053, 0.055],
        [0.061, 0.063],
        [0.069, 0.071],
        [0.077, 0.079],
        [0.085, 0.087],
        [0.093, 0.095],
    ],
    dtype="f8",
)

# The wide 0.4 < zobs < 1.1 positive-octant volume lowers the observed
# fundamental-mode cut enough to admit exactly one additional measured bin.
# The DESI-PNG thinning policy above 0.005 h/Mpc is otherwise unchanged.
TASK43_WIDE_LRGALL_FIT_EDGES = np.vstack(
    [np.asarray([[0.003, 0.005]], dtype="f8"), TASK43_REALSPACE_FIT_EDGES]
)


def observed_fit_kmin(volume: float) -> float:
    """Return the Task 4.3 observed-bin cutoff for a survey volume."""

    return 2.0 * math.pi / float(volume) ** (1.0 / 3.0)


def load_task43_realspace_reference(path: Path = TASK43_REALSPACE_REFERENCE) -> dict[str, Any]:
    """Load and validate the archived, frozen Task 4.3 real-space payload."""

    if not path.is_file():
        raise FileNotFoundError(f"missing frozen Task 4.3 real-space P0 reference: {path}")
    with np.load(path, allow_pickle=False) as payload:
        result = {
            "path": path,
            "fit_edges": np.asarray(payload["k_edges"], dtype="f8"),
            "kmin_fit_observed": float(np.asarray(payload["kmin_fit_observed"]).item()),
            "volume": float(np.asarray(payload["lightcone_volume_eff"]).item()),
            "leff": float(np.asarray(payload["lightcone_leff_volume"]).item()),
            "window_shape": tuple(np.asarray(payload["window_matrix"]).shape),
            "theory_k": np.asarray(payload["theory_k"], dtype="f8"),
            "theory_ell": np.asarray(payload["theory_ell"], dtype="i8"),
        }
    checks = {
        "fit_edges": np.allclose(result["fit_edges"], TASK43_REALSPACE_FIT_EDGES, rtol=0.0, atol=1.0e-15),
        "volume": np.isclose(result["volume"], TASK43_REALSPACE_VOLUME, rtol=0.0, atol=1.0e-6),
        "leff": np.isclose(result["leff"], TASK43_REALSPACE_LEFF, rtol=0.0, atol=1.0e-12),
        "observed_kmin": np.isclose(
            result["kmin_fit_observed"], TASK43_REALSPACE_OBSERVED_KMIN, rtol=0.0, atol=1.0e-15
        ),
        "observed_kmin_formula": np.isclose(
            observed_fit_kmin(result["volume"]), result["kmin_fit_observed"], rtol=0.0, atol=1.0e-15
        ),
        "window_shape": result["window_shape"] == (15, 981),
        "theory_ells": np.array_equal(np.unique(result["theory_ell"]), [0, 2, 4]),
        "theory_ell_counts": np.array_equal(
            np.unique(result["theory_ell"], return_counts=True)[1], [327, 327, 327]
        ),
    }
    if not all(bool(value) for value in checks.values()):
        raise RuntimeError(f"frozen Task 4.3 real-space P0 contract changed: {checks}")
    result["checks"] = {key: bool(value) for key, value in checks.items()}
    return result
