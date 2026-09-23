#!/usr/bin/env python3
"""Finite-mock covariance corrections used by the Task43 interim fits.

Hartlap corrects the inverse sample covariance inside the likelihood.  The
Percival ``m1`` factor is deliberately kept separate: it inflates reported
parameter variances/errors and must not be multiplied into the data
covariance or the likelihood precision.
"""

from __future__ import annotations

import math
from typing import Any


def covariance_corrections(*, nmock: int, ndata: int, nparams: int) -> dict[str, Any]:
    """Return Hartlap and Percival (2014-style) finite-mock corrections."""
    ns, nb, np_ = int(nmock), int(ndata), int(nparams)
    if ns <= nb + 4:
        raise ValueError(
            f"Nmock={ns} must exceed Ndata+4={nb + 4} for the Percival correction"
        )
    if not 0 < np_ < nb:
        raise ValueError(f"expected 0 < Nparams < Ndata, got {np_}, {nb}")
    hartlap = (ns - nb - 2.0) / (ns - 1.0)
    a = 2.0 / ((ns - nb - 1.0) * (ns - nb - 4.0))
    b = (ns - nb - 2.0) / ((ns - nb - 1.0) * (ns - nb - 4.0))
    m1 = (1.0 + b * (nb - np_)) / (1.0 + a + b * (np_ + 1.0))
    return {
        "nmock": ns,
        "ndata": nb,
        "nparams": np_,
        "hartlap": float(hartlap),
        "hartlap_formula": "(Nmock-Ndata-2)/(Nmock-1)",
        "hartlap_application": "precision = hartlap * inverse(C_sample)",
        "percival_A": float(a),
        "percival_B": float(b),
        "percival_m1_variance": float(m1),
        "percival_error_factor": float(math.sqrt(m1)),
        "percival_formula": (
            "A=2/[(Nmock-Ndata-1)(Nmock-Ndata-4)]; "
            "B=(Nmock-Ndata-2)/[(Nmock-Ndata-1)(Nmock-Ndata-4)]; "
            "m1=[1+B(Ndata-Nparams)]/[1+A+B(Nparams+1)]"
        ),
        "percival_application": (
            "reported parameter covariance = m1 * raw posterior covariance; "
            "reported errors = sqrt(m1) * raw posterior errors"
        ),
    }
