#!/usr/bin/env python3
"""Create the AbacusSummit c000 linear matter P(k) used by EZmock."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path

import numpy as np


PROJECT_ROOT = Path("/pscratch/sd/l/lzy/PNG-2PCF-model-to-observe")
DEFAULT_OUTPUT = (
    PROJECT_ROOT
    / "outputs/task43_outputs/ezmock_calibration_rawbox_z0p725_mmin1p4e13/linear_pk"
    / "abacus_c000_linear_matter_pk_z0p725_desilike_cosmoprimo.dat"
)


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json(path: Path, payload: dict[str, object]) -> None:
    tmp = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    tmp.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    tmp.replace(path)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--redshift", type=float, default=0.725)
    parser.add_argument("--kmin", type=float, default=1.0e-5)
    parser.add_argument("--kmax", type=float, default=10.0)
    parser.add_argument("--nk", type=int, default=8192)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    if args.output.exists() and not args.overwrite:
        print(f"[skip] {args.output}")
        return
    if not (0.0 <= args.redshift and 0.0 < args.kmin < args.kmax and args.nk >= 128):
        raise ValueError("invalid redshift or k grid")

    # cosmoprimo is the Boltzmann/cosmology backend shipped with the desilike
    # environment used by this project.  AbacusSummit(0) is base_c000.
    from cosmoprimo.fiducial import AbacusSummit

    cosmo = AbacusSummit(0)
    kval = np.geomspace(args.kmin, args.kmax, args.nk, dtype="f8")
    pk = np.asarray(
        cosmo.get_fourier().pk_interpolator(non_linear=False, of="delta_m")(
            kval, z=float(args.redshift)
        ),
        dtype="f8",
    )
    if not np.all(np.isfinite(pk)) or np.any(pk <= 0.0):
        raise RuntimeError("cosmoprimo returned a non-positive or non-finite linear P(k)")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    tmp = args.output.with_name(f".{args.output.name}.{os.getpid()}.tmp")
    np.savetxt(
        tmp,
        np.column_stack([kval, pk]),
        fmt="%.12e",
        header=(
            "AbacusSummit base_c000 linear matter spectrum; "
            f"z={args.redshift:.8g}; k[h/Mpc] P_lin[(Mpc/h)^3]"
        ),
    )
    tmp.replace(args.output)

    omega_nu = float(cosmo.Omega0_ncdm_tot)
    omega_non_nu = float(cosmo.Omega0_b + cosmo.Omega0_cdm)
    metadata = {
        "task": "task43_make_ezmock_abacus_c000_linear_pk",
        "status": "done",
        "generator": "desilike environment with cosmoprimo.fiducial.AbacusSummit(0)",
        "cosmology": "AbacusSummit_base_c000",
        "spectrum": "linear delta_m delta_m",
        "redshift": float(args.redshift),
        "k_unit": "h/Mpc",
        "pk_unit": "(Mpc/h)^3",
        "kmin": float(kval[0]),
        "kmax": float(kval[-1]),
        "nk": int(kval.size),
        "h": float(cosmo["h"]),
        "omega_m_non_neutrino": omega_non_nu,
        "omega_nu": omega_nu,
        "omega_m_total": float(cosmo.Omega0_m),
        "output": str(args.output),
        "output_sha256": file_sha256(args.output),
    }
    write_json(args.output.with_suffix(".json"), metadata)
    print(f"[done] {args.output} nk={kval.size} z={args.redshift:g}")


if __name__ == "__main__":
    main()
