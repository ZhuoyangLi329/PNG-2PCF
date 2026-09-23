#!/usr/bin/env python3
"""Fit only the two-phase c000/c302 pseudo-lightcone observable means."""

from __future__ import annotations

import argparse
from dataclasses import replace
import json
from pathlib import Path
import time
from typing import Any

import numpy as np

from task43_pngbase_pseudolc_common import (
    COSMOLOGIES,
    FNL_BY_COSMOLOGY,
    PHASES,
    SEED_BY_PHASE,
    fit_root,
    fit_root_png_cov,
    pk_path,
    png_covariance_path,
    xi_path,
)
from task43_rsd_common import atomic_savez, atomic_write_json, sha256_file
from task43_run_lightcone_joint_baomask_v1 import (
    REAL_P_PAYLOAD,
    RSD_P_PAYLOAD,
    FitSpec,
    ScaledGaussianMetric,
    fit_maximum_likelihood,
    independent_tension,
    lightcone_xi_primary_mask,
    load_real_specs,
    load_rsd_specs,
    phase_diagnostics,
    run_chain,
    select_pk_bins,
)
from task43_fit_rsd_rawbox_pk0_vs_xi0_smin50 import set_affinity


VARIANT_ORDER = (
    "real_p0",
    "real_xi0",
    "real_joint_p0xi0",
    "rsd_p0",
    "rsd_xi0",
    "rsd_joint_p0xi0",
    "rsd_p02",
    "rsd_xi02",
    "rsd_joint_p02xi02",
)


def validated_npz(path: Path) -> dict[str, Any]:
    metadata_path = path.with_suffix(".json")
    if not (path.is_file() and metadata_path.is_file()):
        raise FileNotFoundError(f"missing product/metadata: {path} / {metadata_path}")
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    if metadata.get("status") != "pass" or metadata.get("output_sha256") != sha256_file(path):
        raise RuntimeError(f"unvalidated product: {path}")
    return metadata


def load_pk_stack(cosmology: str, space: str, indices: np.ndarray, *, include_p2: bool) -> tuple[np.ndarray, dict[str, Any]]:
    stacks: list[np.ndarray] = []
    hashes: list[str] = []
    edges_reference: np.ndarray | None = None
    k_reference: np.ndarray | None = None
    k_max_delta = 0.0
    for phase in PHASES:
        path = pk_path(cosmology, phase, space)
        metadata = validated_npz(path)
        with np.load(path, allow_pickle=False) as payload:
            if str(np.asarray(payload["phase"]).item()) != phase:
                raise RuntimeError(f"phase label mismatch in {path}")
            k = np.asarray(payload["k_obs"], dtype="f8")
            edges = np.asarray(payload["k_edges"], dtype="f8")
            if edges_reference is None:
                edges_reference, k_reference = edges, k
            else:
                if not np.allclose(edges, edges_reference, rtol=0.0, atol=1.0e-13):
                    raise RuntimeError(f"P(k) edges changed in {path}")
                k_max_delta = max(k_max_delta, float(np.max(np.abs(k - k_reference))))
            values = [np.asarray(payload["pk0"], dtype="f8")[indices]]
            if include_p2:
                values.append(np.asarray(payload["pk2"], dtype="f8")[indices])
            stacks.append(np.concatenate(values))
        hashes.append(metadata["output_sha256"])
    return np.stack(stacks), {
        "paths": [str(pk_path(cosmology, phase, space)) for phase in PHASES],
        "sha256": hashes,
        "phase_k_coordinate_max_abs_delta": k_max_delta,
    }


def load_xi_stack(cosmology: str, space: str, target_s: np.ndarray) -> tuple[np.ndarray, dict[str, Any]]:
    stacks: list[np.ndarray] = []
    hashes: list[str] = []
    zeff: list[float] = []
    for phase in PHASES:
        path = xi_path(cosmology, phase, space)
        metadata = validated_npz(path)
        with np.load(path, allow_pickle=False) as payload:
            if str(np.asarray(payload["phase"]).item()) != phase:
                raise RuntimeError(f"phase label mismatch in {path}")
            s = np.asarray(payload["s"], dtype="f8")
            mask = lightcone_xi_primary_mask(s, smin=50.0)
            if not np.array_equal(s[mask], np.asarray(target_s, dtype="f8")):
                raise RuntimeError(f"selected xi grid differs from Task4.3 reference in {path}")
            ells = np.asarray(payload["ells"], dtype="i8")
            if not np.array_equal(ells, [0, 2]):
                raise RuntimeError(f"xi multipoles changed in {path}: {ells}")
            values = np.asarray(payload["xi_multipoles"], dtype="f8")
            stacks.append(np.concatenate((values[0, mask], values[1, mask])))
            zeff.append(float(np.asarray(payload["zeff"]).item()))
        hashes.append(metadata["output_sha256"])
    return np.stack(stacks), {
        "paths": [str(xi_path(cosmology, phase, space)) for phase in PHASES],
        "sha256": hashes,
        "zeff_by_phase": zeff,
        "zeff_equal_phase_mean": float(np.mean(zeff)),
    }


def observed_specs(cosmology: str) -> tuple[list[FitSpec], dict[str, Any], dict[str, np.ndarray]]:
    reference_real, real_meta, real_arrays = load_real_specs(smin=50.0, pk_kmax=0.08)
    reference_rsd, rsd_meta, rsd_arrays = load_rsd_specs(smin=50.0, pk_kmax=0.08)
    references = {spec.name: spec for spec in reference_real + reference_rsd}

    with np.load(REAL_P_PAYLOAD, allow_pickle=False) as payload:
        real_indices, _, _, _ = select_pk_bins({key: np.asarray(payload[key]) for key in payload.files}, 0.08)
    with np.load(RSD_P_PAYLOAD, allow_pickle=False) as payload:
        rsd_indices, _, rsd_k, _ = select_pk_bins({key: np.asarray(payload[key]) for key in payload.files}, 0.08)
    keep_p2 = np.flatnonzero(rsd_k >= 0.015 - 1.0e-12)

    real_p, real_p_meta = load_pk_stack(cosmology, "real", real_indices, include_p2=False)
    rsd_p_both, rsd_p_meta = load_pk_stack(cosmology, "rsd", rsd_indices, include_p2=True)
    n_rsd_p0 = int(rsd_indices.size)
    rsd_p = np.concatenate(
        (rsd_p_both[:, :n_rsd_p0], rsd_p_both[:, n_rsd_p0:][:, keep_p2]), axis=1
    )
    real_s = real_arrays["real_s"][real_arrays["real_xi_mask"].astype(bool)]
    rsd_s = rsd_arrays["rsd_s"][rsd_arrays["rsd_xi_mask"].astype(bool)]
    real_x_both, real_x_meta = load_xi_stack(cosmology, "real", real_s)
    rsd_x, rsd_x_meta = load_xi_stack(cosmology, "rsd", rsd_s)
    n_real_x = int(real_s.size)
    real_x = real_x_both[:, :n_real_x]
    n_rsd_x = int(rsd_s.size)

    phase_data = {
        "real_p0": real_p,
        "real_xi0": real_x,
        "real_joint_p0xi0": np.hstack((real_p, real_x)),
        "rsd_p0": rsd_p[:, :n_rsd_p0],
        "rsd_xi0": rsd_x[:, :n_rsd_x],
        "rsd_joint_p0xi0": np.hstack((rsd_p[:, :n_rsd_p0], rsd_x[:, :n_rsd_x])),
        "rsd_p02": rsd_p,
        "rsd_xi02": rsd_x,
        "rsd_joint_p02xi02": np.hstack((rsd_p, rsd_x)),
    }
    specs = [
        replace(references[name], data=np.mean(phase_data[name], axis=0), phase_data=phase_data[name])
        for name in VARIANT_ORDER
    ]
    source = {
        "real_p": real_p_meta,
        "real_xi": real_x_meta,
        "rsd_p02": rsd_p_meta,
        "rsd_xi02": rsd_x_meta,
    }
    metadata = {
        "real": real_meta,
        "rsd": rsd_meta,
        "sources": source,
        "arithmetic_mean_gate": all(
            np.array_equal(spec.data, np.mean(spec.phase_data, axis=0)) for spec in specs
        ),
    }
    return specs, metadata, {**real_arrays, **rsd_arrays}


def metrics(results: dict[str, Any]) -> dict[str, Any]:
    groups = {
        "real": ("real_p0", "real_xi0", "real_joint_p0xi0", ("fNL", "b1")),
        "rsd_monopole": ("rsd_p0", "rsd_xi0", "rsd_joint_p0xi0", ("fNL", "b1", "sigma_s")),
        "rsd_multipole": ("rsd_p02", "rsd_xi02", "rsd_joint_p02xi02", ("fNL", "b1", "sigma_s")),
    }
    output: dict[str, Any] = {}
    for group, (left, right, joint, names) in groups.items():
        sigma_left = results[left]["mcmc"]["posterior"]["fNL"]["sigma68"]
        sigma_right = results[right]["mcmc"]["posterior"]["fNL"]["sigma68"]
        sigma_joint = results[joint]["mcmc"]["posterior"]["fNL"]["sigma68"]
        output[group] = {
            "joint_fNL_improvement_over_best_marginal_fraction": 1.0 - sigma_joint / min(sigma_left, sigma_right),
            "marginal_tensions": {
                name: independent_tension(results[left], results[right], name) for name in names
            },
        }
    return output


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Fit a cosmology-level ph000/ph001 mean; this command intentionally has no --phase option."
    )
    parser.add_argument("--cosmology", choices=COSMOLOGIES, required=True)
    parser.add_argument("--covariance-mode", choices=("task43_fnl0", "c302_fnl100"), default="task43_fnl0")
    parser.add_argument("--threads", type=int, default=8)
    parser.add_argument("--nwalkers", type=int, default=64)
    parser.add_argument("--nsteps", type=int, default=30000)
    parser.add_argument("--burnin", type=int, default=5000)
    parser.add_argument("--seed", type=int, default=20261009)
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    if args.covariance_mode == "c302_fnl100" and args.cosmology != "c302":
        raise ValueError("c302_fnl100 covariance mode is defined only for c302")
    if not 1 <= int(args.threads) <= 8:
        raise ValueError("--threads must be in [1,8]")
    if int(args.burnin) >= int(args.nsteps):
        raise ValueError("--burnin must be smaller than --nsteps")
    nsteps = 400 if args.smoke else int(args.nsteps)
    burnin = 100 if args.smoke else int(args.burnin)
    base_root = fit_root_png_cov() if args.covariance_mode == "c302_fnl100" else fit_root(args.cosmology)
    out_root = base_root / ("smoke" if args.smoke else "")
    audit_path = out_root / "task43_pngbase_pseudolc_joint_baomask80_120.json"
    covariance_path = out_root / "task43_pngbase_pseudolc_joint_baomask80_120_covariance.npz"
    if audit_path.exists():
        raise FileExistsError(f"immutable audit exists: {audit_path}")

    started = time.perf_counter()
    cpus = set_affinity(int(args.threads))
    code_hash = sha256_file(Path(__file__))
    specs, metadata, arrays = observed_specs(args.cosmology)
    covariance_source: dict[str, Any]
    if args.covariance_mode == "c302_fnl100":
        covariance_file = png_covariance_path()
        covariance_metadata_path = covariance_file.with_suffix(".json")
        covariance_metadata = json.loads(covariance_metadata_path.read_text(encoding="utf-8"))
        if (
            covariance_metadata.get("status") != "pass"
            or covariance_metadata.get("cosmology") != "c302"
            or float(covariance_metadata.get("fnl_cov", np.nan)) != 100.0
            or float(covariance_metadata.get("covariance_divisor", np.nan)) != 1.0
            or covariance_metadata.get("output_sha256") != sha256_file(covariance_file)
        ):
            raise RuntimeError("c302 fNL_cov=100 covariance provenance gate failed")
        with np.load(covariance_file, allow_pickle=False) as payload:
            replacement = {key: np.asarray(payload[key]) for key in payload.files}
        covariance_by_variant = {
            "real_p0": replacement["real_pp"], "real_xi0": replacement["real_xx"],
            "real_joint_p0xi0": replacement["real_joint"], "rsd_p0": replacement["rsd_pp0"],
            "rsd_xi0": replacement["rsd_xx0"], "rsd_joint_p0xi0": replacement["rsd_joint0"],
            "rsd_p02": replacement["rsd_pp"], "rsd_xi02": replacement["rsd_xx"],
            "rsd_joint_p02xi02": replacement["rsd_joint"],
        }
        specs = [replace(spec, covariance=covariance_by_variant[spec.name]) for spec in specs]
        for key in (
            "real_k", "real_s", "real_xi_mask", "real_pp", "real_xx", "real_px", "real_joint",
            "rsd_k", "rsd_p2_keep_indices", "rsd_s", "rsd_xi_mask", "rsd_pp", "rsd_xx", "rsd_xp",
            "rsd_joint", "rsd_pp0", "rsd_xx0", "rsd_xp0", "rsd_joint0",
        ):
            arrays[key] = replacement[key]
        covariance_source = {
            "mode": args.covariance_mode, "path": str(covariance_file), "sha256": sha256_file(covariance_file),
            "metadata_path": str(covariance_metadata_path), "metadata_sha256": sha256_file(covariance_metadata_path),
            "fnl_cov": 100.0, "phase_covariance_combination": covariance_metadata["phase_covariance_combination"],
        }
    else:
        covariance_source = {"mode": args.covariance_mode, "fnl_cov": 0.0, "source": "frozen Task4.3 covariance"}
    if not metadata["arithmetic_mean_gate"]:
        raise RuntimeError("two-realization arithmetic-mean gate failed")
    results: dict[str, Any] = {}
    numerical_gates: dict[str, bool] = {
        "arithmetic_ph000_ph001_mean": True,
        "single_phase_fnl_fit_count_is_zero": True,
    }
    for index, spec in enumerate(specs):
        metric = ScaledGaussianMetric(spec.covariance)
        numerical_gates[f"{spec.name}_strict_spd"] = bool(metric.eigenvalues[0] > 1.0e-12)
        map_summary, theta_ml = fit_maximum_likelihood(spec, metric)
        map_summary["chi2_observed_two_phase_mean_with_single_realization_covariance"] = map_summary.pop(
            "chi2_observed_x25_mean_with_single_realization_covariance"
        )
        map_summary["pte_observed_two_phase_mean_with_single_realization_covariance"] = map_summary.pop(
            "pte_observed_x25_mean_with_single_realization_covariance"
        )
        prediction = np.asarray(spec.evaluate(theta_ml), dtype="f8")
        fit_dir = out_root / "fits" / spec.name
        fit_npz, fit_json = fit_dir / "samples.npz", fit_dir / "summary.json"
        if fit_npz.exists() != fit_json.exists():
            raise RuntimeError(f"partial resumable output for {spec.name}")
        if fit_npz.exists():
            existing = json.loads(fit_json.read_text(encoding="utf-8"))
            if (
                existing.get("status") != "pass"
                or existing.get("code_sha256") != code_hash
                or existing.get("output_npz_sha256") != sha256_file(fit_npz)
                or existing.get("nsteps") != nsteps
                or existing.get("burnin") != burnin
            ):
                raise RuntimeError(f"resumable output contract failed for {spec.name}")
            results[spec.name] = existing["result"]
            if not args.smoke:
                for gate, passed in results[spec.name]["mcmc"]["gates"].items():
                    numerical_gates[f"{spec.name}_{gate}"] = bool(passed)
            print(json.dumps({"variant": spec.name, "status": "resumed"}), flush=True)
            continue
        chain_summary, chain, logp = run_chain(
            spec,
            metric,
            theta_ml,
            nwalkers=int(args.nwalkers),
            nsteps=nsteps,
            burnin=burnin,
            seed=int(args.seed) + 1000 * COSMOLOGIES.index(args.cosmology) + 100 * index,
            nworkers=int(args.threads),
        )
        result = {
            "group": spec.group,
            "parameter_names": list(spec.parameter_names),
            "map": map_summary,
            "mcmc": chain_summary,
            "phase_diagnostics": phase_diagnostics(spec, metric, theta_ml),
            "correlation_eigenvalue_min": float(metric.eigenvalues[0]),
        }
        result["phase_diagnostics"]["definition"] = (
            "ph000 and ph001 evaluated at the two-realization-mean ML model using C_single; diagnostic only"
        )
        atomic_savez(
            fit_npz,
            parameter_names=np.asarray(spec.parameter_names),
            chain_by_step=chain,
            log_probability_by_step=logp,
            theta_maximum_likelihood=theta_ml,
            prediction_maximum_likelihood=prediction,
            data=np.asarray(spec.data),
            phase_data=np.asarray(spec.phase_data),
            covariance_single=np.asarray(spec.covariance),
            phases=np.asarray(PHASES),
            injected_fnl=np.asarray(FNL_BY_COSMOLOGY[args.cosmology]),
        )
        fit_status = "pass" if args.smoke or all(chain_summary["gates"].values()) else "validation_failed"
        atomic_write_json(
            fit_json,
            {
                "task": "task43_run_pngbase_pseudolc_joint",
                "variant": spec.name,
                "cosmology": args.cosmology,
                "covariance_mode": args.covariance_mode,
                "fit_unit": "ph000/ph001 arithmetic observable mean",
                "single_phase_fnl_fit": False,
                "status": fit_status,
                "result": result,
                "nsteps": nsteps,
                "burnin": burnin,
                "code_sha256": code_hash,
                "output_npz": str(fit_npz),
                "output_npz_sha256": sha256_file(fit_npz),
            },
        )
        results[spec.name] = result
        if not args.smoke:
            for gate, passed in chain_summary["gates"].items():
                numerical_gates[f"{spec.name}_{gate}"] = bool(passed)
        print(
            json.dumps(
                {
                    "cosmology": args.cosmology,
                    "variant": spec.name,
                    "fNL": chain_summary["posterior"]["fNL"],
                    "gates": chain_summary["gates"],
                },
                sort_keys=True,
            ),
            flush=True,
        )

    if args.smoke:
        print(json.dumps({"status": "smoke_ok", "cosmology": args.cosmology}, sort_keys=True))
        return
    atomic_savez(
        covariance_path,
        **arrays,
        phases=np.asarray(PHASES),
        mean_realization_count=np.asarray(2, dtype="i8"),
        covariance_divisor=np.asarray(1.0, dtype="f8"),
    )
    status = "pass" if all(numerical_gates.values()) else "validation_failed"
    summary = {
        "task": "task43_run_pngbase_pseudolc_joint",
        "status": status,
        "classification": "paired pngbase snapshot-shell pseudo-lightcone P/xi fits",
        "cosmology": args.cosmology,
        "covariance_mode": args.covariance_mode,
        "injected_fnl": FNL_BY_COSMOLOGY[args.cosmology],
        "initial_condition_seeds": [SEED_BY_PHASE[phase] for phase in PHASES],
        "phases": list(PHASES),
        "mean_realization_count": 2,
        "observed_curve_label": "ph000/ph001 arithmetic mean; single-realization error",
        "plot_label": f"pngbase {args.cosmology} snapshot-shell pseudo-lightcone",
        "fit_unit": "one arithmetic observable mean per cosmology",
        "single_phase_fnl_fits": [],
        "fit_contract": {
            "pk_kmax_h_mpc": 0.08,
            "xi_smin_mpc_h": 50.0,
            "xi_smax_mpc_h": 350.0,
            "xi_bao_excluded_half_open_mpc_h": [80.0, 120.0],
            "real_redshift": "0.6 < z_geom < 0.8",
            "rsd_redshift": "0.4 < z_observed < 0.8",
        },
        "mask_policy": {
            "full_range_mpc_h": [50.0, 350.0],
            "excluded_half_open_range_mpc_h": [80.0, 120.0],
            "selected_centers_mpc_h": arrays["real_s"][arrays["real_xi_mask"].astype(bool)].tolist(),
            "excluded_centers_mpc_h": [85.0, 95.0, 105.0, 115.0],
            "same_mask_for_rsd_xi0_and_xi2": True,
        },
        "covariance_contract": {
            "observed_curve": "arithmetic mean of ph000 and ph001",
            "likelihood": f"{args.covariance_mode} representative single-realization covariance; never divided by two",
            "posterior": f"{args.covariance_mode} representative single-realization covariance; never divided by two",
            "goodness_of_fit": "two-phase mean and phase diagnostics use C_single",
            "plot_errorbars": "sqrt(diag(C_single)); never divided by sqrt(2)",
            "covariance_divisor": 1.0,
            "source": covariance_source,
        },
        "reference_models_and_covariance": {"real": metadata["real"], "rsd": metadata["rsd"]},
        "measurements": metadata["sources"],
        "results": results,
        "metrics": metrics(results),
        "numerical_gates": numerical_gates,
        "mcmc": {
            "nwalkers": int(args.nwalkers),
            "nsteps": nsteps,
            "burnin": burnin,
            "parallel_workers": int(args.threads),
            "seed_base": int(args.seed),
        },
        "cpu_affinity": cpus,
        "outputs": {
            "covariance_npz": str(covariance_path),
            "covariance_npz_sha256": sha256_file(covariance_path),
        },
        "code_sha256": code_hash,
        "elapsed_sec": time.perf_counter() - started,
    }
    atomic_write_json(audit_path, summary)
    print(json.dumps({"status": status, "cosmology": args.cosmology, "output": str(audit_path)}, sort_keys=True))
    if status != "pass":
        raise SystemExit(2)


if __name__ == "__main__":
    main()
