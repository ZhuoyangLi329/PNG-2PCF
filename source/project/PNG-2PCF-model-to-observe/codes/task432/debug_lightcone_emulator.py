#!/usr/bin/env python3
"""Cross-validate the lightcone xi emulator against direct velocileptors."""
from __future__ import annotations

import json
from pathlib import Path
import numpy as np

from task432_lightcone_png_velocileptors import LightconePNGVelocileptors, RSD_X_SUMMARY, RSD_P_PAYLOAD, build_cache
from task432_lightcone_png_velocileptors_emulated_longchain import XiEmulator


def main() -> None:
    root = Path("/pscratch/sd/l/lzy/PNG-2PCF-model-to-observe")
    emulator_path = root / "outputs/task43_outputs/rsd_validation/task432_model_repair/lightcone_png_velocileptors_gsm_longchain/lightcone_xi_emulator.npz"
    with np.load(emulator_path, allow_pickle=False) as payload:
        emulator = XiEmulator(np.asarray(payload["f_grid"], dtype="f8"), np.asarray(payload["b_grid"], dtype="f8"), np.asarray(payload["values"], dtype="f8"))
    with np.load(RSD_P_PAYLOAD, allow_pickle=False) as payload:
        zeff = float(np.asarray(payload["zeff"]).item())
    cache = build_cache(zeff=zeff, boxsize=2000.0, kmax=3.0, ells=(0, 2), cosmology="abacus_c000")
    with np.load(cache, allow_pickle=False) as payload:
        k, pk, alpha = payload["k_eff"], payload["pk_dd"], payload["alpha"]
        f = float(np.asarray(payload["f_growth"]).item())
    with np.load(RSD_X_SUMMARY, allow_pickle=False) as payload:
        s = np.asarray(payload["s"], dtype="f8")
    mask = (s >= 50.0) & (s < 350.0) & ~((s >= 80.0) & (s < 120.0))
    direct = LightconePNGVelocileptors(k, pk, alpha, f, s)
    rows = []
    for fnl in (-30.0, 0.0, 30.0):
        for b1 in (2.30, 2.40, 2.50):
            values = direct.evaluate(fnl=fnl, b1=b1, nint=500)
            truth = np.concatenate((values[0][mask], values[2][mask]))
            interp = emulator.evaluate(fnl, b1)
            delta = interp - truth
            rows.append({"fNL": fnl, "b1": b1, "max_abs": float(np.max(np.abs(delta))), "relative_l2": float(np.linalg.norm(delta) / max(np.linalg.norm(truth), 1e-30))})
    print(json.dumps({"rows": rows, "max_relative_l2": max(x["relative_l2"] for x in rows)}, sort_keys=True))


if __name__ == "__main__":
    main()
