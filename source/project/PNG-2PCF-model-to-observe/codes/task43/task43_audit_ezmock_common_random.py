#!/usr/bin/env python3
"""Independent geometry, selection, FKP, and HDF5 audit of common50 random."""

from __future__ import annotations

import json

import h5py
import numpy as np

from task43_ezmock_covariance_common import (
    ABACUS_FKP_SUMMARY, COMMON_RANDOM, COMMON_RANDOM_HDF5,
    COMMON_RANDOM_HDF5_META, COMMON_RANDOM_META, COMMON_RANDOM_SIZE, P0,
    SUMMARY_DIR, sha256, write_json,
)
from task43_fkp_zeff import fkp_bin_weights


OUTPUT = SUMMARY_DIR / "task43_ezmock_common50_random_independent_audit.json"


def main() -> None:
    source_meta = json.loads(COMMON_RANDOM_META.read_text(encoding="utf-8")); h5_meta = json.loads(COMMON_RANDOM_HDF5_META.read_text(encoding="utf-8"))
    source_hash, h5_hash = sha256(COMMON_RANDOM), sha256(COMMON_RANDOM_HDF5)
    with np.load(ABACUS_FKP_SUMMARY, allow_pickle=False) as fkp:
        z_edges = np.asarray(fkp["z_edges"], dtype="f8"); nbar = np.asarray(fkp["nbar"], dtype="f8"); volume = np.asarray(fkp["volume_shell"], dtype="f8")
    with np.load(COMMON_RANDOM, allow_pickle=False) as source:
        x = np.asarray(source["X"], dtype="f8"); y = np.asarray(source["Y"], dtype="f8"); zcart = np.asarray(source["Zcart"], dtype="f8"); redshift = np.asarray(source["Z"], dtype="f8")
    radius = np.sqrt(x*x + y*y + zcart*zcart); phi = np.arctan2(y, x); mu = zcart / radius
    probability = volume / volume.sum(); counts = np.histogram(redshift, bins=z_edges)[0]
    expected = COMMON_RANDOM_SIZE * probability; standardized = (counts - expected) / np.sqrt(expected * (1.0 - probability))
    hdf_equal = {}
    with h5py.File(COMMON_RANDOM_HDF5, "r") as handle, np.load(COMMON_RANDOM, allow_pickle=False) as source:
        for name in ("X", "Y", "Zcart", "Z"):
            hdf_equal[name] = bool(np.array_equal(np.asarray(handle[name]), np.asarray(source[name])))
        ibin = np.clip(np.searchsorted(z_edges, redshift, side="right") - 1, 0, nbar.size - 1)
        expected_weight = np.asarray(fkp_bin_weights(nbar, P0)[ibin], dtype="f4")
        hdf_equal["WEIGHT_FKP"] = bool(np.array_equal(np.asarray(handle["WEIGHT_FKP"]), expected_weight))
    gates = {
        "exact_size": redshift.size == COMMON_RANDOM_SIZE,
        "metadata_hashes": source_meta.get("sha256") == source_hash and h5_meta.get("source_npz_sha256") == source_hash and h5_meta.get("sha256") == h5_hash,
        "positive_octant": bool(np.all(x >= 0) and np.all(y >= 0) and np.all(zcart >= 0)),
        "finite": bool(all(np.all(np.isfinite(array)) for array in (x, y, zcart, redshift, phi, mu))),
        "angular_support": bool(phi.min() >= 0 and phi.max() <= np.pi / 2 and mu.min() >= 0 and mu.max() <= 1),
        "angular_means": bool(abs(phi.mean() - np.pi / 4) < 1e-3 and abs(mu.mean() - 0.5) < 5e-4),
        "selection_multinomial_6sigma": bool(np.max(np.abs(standardized)) < 6.0),
        "hdf_exact_transport": bool(all(hdf_equal.values())),
    }
    payload = {
        "task": "task43_audit_ezmock_common_random", "status": "pass" if all(gates.values()) else "fail",
        "gates": gates, "nrandom": int(redshift.size), "source_npz_sha256": source_hash, "hdf5_sha256": h5_hash,
        "phi": {"min": float(phi.min()), "max": float(phi.max()), "mean": float(phi.mean()), "target_mean": float(np.pi/4)},
        "mu_cos_theta": {"min": float(mu.min()), "max": float(mu.max()), "mean": float(mu.mean()), "target_mean": 0.5},
        "selection": {"z_edges": z_edges, "counts": counts, "target_probability": probability, "standardized_residual": standardized, "max_abs_standardized_residual": float(np.max(np.abs(standardized)))},
        "hdf_dataset_exact_equality": hdf_equal,
    }
    write_json(OUTPUT, payload)
    if payload["status"] != "pass": raise RuntimeError(f"common random audit failed: {gates}")
    print(f"[pass] {OUTPUT} phi_mean={phi.mean():.6f} mu_mean={mu.mean():.6f} max_selection_z={np.max(np.abs(standardized)):.2f}")


if __name__ == "__main__": main()
