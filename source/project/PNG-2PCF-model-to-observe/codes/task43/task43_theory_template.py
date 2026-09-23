#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Shared Task43 theory-template helpers.

This module keeps the cosmology used for the no-RSD PNG theory explicit.  The
Task43 Abacus rawbox/lightcone branch should use ``abacus_c000`` by default;
``task41_fastpm`` is kept only for reproducing older FastPM/Task4.2 diagnostics.
``desi`` is available for comparisons to DESI/desilike lightcone P(k)
likelihoods that build their template from ``cosmoprimo.fiducial.DESI``.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path
from typing import Any

import numpy as np


PROJECT_ROOT = Path("/pscratch/sd/l/lzy/PNG-2PCF-model-to-observe")
TASK41_SCRIPT = PROJECT_ROOT / "codes" / "task4" / "task41_rawbox_norsd_fnl100_profiler.py"

DEFAULT_COSMOLOGY = "abacus_c000"
COSMOLOGY_CHOICES = ("abacus_c000", "task41_fastpm", "desi")


def load_task41() -> Any:
    """Import the Task41 module for shared FullDiscrete and PNG-P(k) helpers."""
    spec = importlib.util.spec_from_file_location("task41_rawbox_norsd_fnl100_profiler", TASK41_SCRIPT)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot import {TASK41_SCRIPT}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _json_value(value: Any) -> Any:
    """Convert cosmoprimo scalar/array parameters to JSON-friendly values."""
    if isinstance(value, np.ndarray):
        return [float(v) for v in np.ravel(value)]
    if isinstance(value, (list, tuple)):
        return [float(v) if isinstance(v, (int, float, np.floating, np.integer)) else v for v in value]
    if isinstance(value, (np.floating, np.integer)):
        return float(value)
    return value


def cosmology_metadata(cosmo: Any, *, name: str, source: str) -> dict[str, Any]:
    """Return the key cosmology parameters that affect Pdd and alpha(k)."""
    keys = (
        "h",
        "H0",
        "Omega_m",
        "Omega_b",
        "Omega_cdm",
        "omega_b",
        "omega_cdm",
        "omega_ncdm",
        "N_ur",
        "N_ncdm",
        "n_s",
        "A_s",
        "sigma8",
    )
    params: dict[str, Any] = {}
    for key in keys:
        try:
            params[key] = _json_value(cosmo[key])
        except Exception:
            if key == "h":
                try:
                    params[key] = float(cosmo.h)
                except Exception:
                    pass
    return {"name": name, "source": source, "parameters": params}


def build_template_arrays(
    task41: Any,
    k_template: np.ndarray,
    *,
    z: float,
    cosmology: str = DEFAULT_COSMOLOGY,
) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    """Build Pdd(k) and alpha(k) for the requested cosmology.

    Parameters
    ----------
    task41
        Imported Task41 module.  It supplies the interpolation convention and,
        for ``task41_fastpm``, the historical FastPM template builder.
    k_template
        Wavenumber grid in h/Mpc.
    z
        Effective redshift.
    cosmology
        ``abacus_c000`` for AbacusSummit base_c000, ``desi`` for the DESI
        fiducial cosmology used by desilike lightcone P(k) likelihoods, or
        ``task41_fastpm`` for reproducing older Task41/Task4.2 FastPM outputs.
    """
    cosmology = str(cosmology)
    if cosmology == "task41_fastpm":
        template = task41.build_template_arrays(k_template, z=float(z))
        cosmo = task41.build_cosmology()
        return template, cosmology_metadata(
            cosmo,
            name=cosmology,
            source="codes/task4/task41_rawbox_norsd_fnl100_profiler.py::build_cosmology",
        )
    if cosmology not in {"abacus_c000", "desi"}:
        raise ValueError(f"unknown cosmology {cosmology!r}; choices={COSMOLOGY_CHOICES}")

    from cosmoprimo.fiducial import AbacusSummit, DESI

    if cosmology == "abacus_c000":
        cosmo = AbacusSummit(0)
        source = "cosmoprimo.fiducial.AbacusSummit(0); AbacusSummit_base_c000 Planck2018"
    else:
        cosmo = DESI(engine="camb")
        source = "cosmoprimo.fiducial.DESI(engine='camb'); matches local_png desilike P(k) template"
    kin = np.asarray(k_template, dtype="f8")
    pk_dd = np.asarray(cosmo.get_fourier().pk_interpolator(non_linear=False, of="delta_m")(kin, z=float(z)), dtype="f8")
    pk_prim = cosmo.get_primordial(mode="scalar").pk_interpolator()(kin)
    pphi_prim = 9.0 / 25.0 * 2.0 * np.pi**2 / kin**3 * pk_prim / cosmo.h**3
    alpha = 1.0 / np.sqrt(pk_dd / pphi_prim)
    template = {"k": kin, "pk_dd": pk_dd, "alpha": alpha}
    return template, cosmology_metadata(
        cosmo,
        name=cosmology,
        source=source,
    )
