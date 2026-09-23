#!/usr/bin/env python3
"""Monopole-only Task 4.3.2 closure for the wide 0.4 < zobs < 1.1 lightcone."""

from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path
from typing import Any

for _name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_name, "1")

import numpy as np
from scipy.stats import chi2 as chi2_distribution

from task43_fit_rsd_lightcone_x25 import (
    FastWindowRSDModel,
    compact_fit,
    convergence_diagnostics,
    fit_one,
    jsonable,
    load_window,
    profile_record,
    set_affinity,
)
from task43_rsd_common import OUTPUT_ROOT, PHASES, PLOT_ROOT, atomic_savez, atomic_write_json, sha256_file
from task43_rsd_model import FullDiscreteRSDModel, build_cache


WIDE_ROOT = OUTPUT_ROOT / "lightcone_wide_zobs0p4_1p1"
DEFAULT_MANIFEST = OUTPUT_ROOT / "manifests" / "task43_rsd_validation_lightcone_wide_zobs0p4_1p1_x25.jsonl"


def edge_pairs(edges: np.ndarray) -> np.ndarray:
    values = np.asarray(edges, dtype="f8")
    return values if values.ndim == 2 else np.column_stack([values[:-1], values[1:]])


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def load_inputs(summary_path: Path, covariance_path: Path) -> dict[str, Any]:
    with np.load(summary_path, allow_pickle=False) as payload:
        phases = tuple(str(value) for value in np.asarray(payload["phases"]).tolist())
        ells = tuple(int(value) for value in np.asarray(payload["ells"]).ravel())
        if phases != PHASES or ells != (0,):
            raise RuntimeError(f"wide xi summary contract changed: phases={phases}, ells={ells}")
        data = {
            "s": np.asarray(payload["s"], dtype="f8"),
            "s_edges": edge_pairs(payload["s_edges"]),
            "xi": np.asarray(payload["xi_multipoles_by_phase"], dtype="f8"),
            "rr": np.asarray(payload["RR_by_phase"], dtype="f8"),
            "zeff": np.asarray(payload["zeff_by_phase"], dtype="f8"),
        }
    with np.load(covariance_path, allow_pickle=False) as payload:
        cov_s = np.asarray(payload["s"], dtype="f8")
        cov_ells = tuple(int(value) for value in np.asarray(payload["ells"]).ravel())
        covariance = np.asarray(payload["covariance_single_realization"], dtype="f8")
        covariance_meta = (
            json.loads(str(np.asarray(payload["meta_json"]).item()))
            if "meta_json" in payload.files and str(np.asarray(payload["meta_json"]).item())
            else {}
        )
    if cov_ells != (0,) or not np.array_equal(data["s"], cov_s):
        raise RuntimeError(f"wide xi0 covariance contract changed: ells={cov_ells}")
    if data["xi"].shape != (25, 1, data["s"].size):
        raise RuntimeError(f"unexpected xi stack shape {data['xi'].shape}")
    if covariance.shape != (data["s"].size, data["s"].size):
        raise RuntimeError(f"unexpected xi0 covariance shape {covariance.shape}")
    covariance = 0.5 * (covariance + covariance.T)
    eigenvalues = np.linalg.eigvalsh(covariance)
    if not np.all(np.isfinite(covariance)) or float(eigenvalues[0]) <= 0.0:
        raise RuntimeError(f"xi0 covariance is not finite SPD: eigmin={eigenvalues[0]}")
    data.update(
        {
            "covariance": covariance,
            "covariance_ells": cov_ells,
            "covariance_meta": covariance_meta,
            "covariance_eigenvalues": eigenvalues,
        }
    )
    return data


def scatter_diagnostics(inputs: dict[str, Any], *, smin: float = 50.0) -> dict[str, Any]:
    mask = np.asarray(inputs["s"], dtype="f8") >= float(smin)
    values = np.asarray(inputs["xi"], dtype="f8")[:, 0, mask]
    empirical = np.cov(values, rowvar=False, ddof=1)
    analytic = np.asarray(inputs["covariance"], dtype="f8")[np.ix_(mask, mask)]
    empirical_sigma = np.sqrt(np.diag(empirical))
    analytic_sigma = np.sqrt(np.diag(analytic))
    empirical_corr = empirical / np.outer(empirical_sigma, empirical_sigma)
    analytic_corr = analytic / np.outer(analytic_sigma, analytic_sigma)
    centered = values - np.mean(values, axis=0)
    precision = np.linalg.inv(analytic)
    phase_chi2 = np.einsum("ij,jk,ik->i", centered, precision, centered)
    return {
        "smin_mpc_h": float(smin),
        "sample_std_over_cov_sigma": (empirical_sigma / analytic_sigma).tolist(),
        "sample_std_over_cov_sigma_median": float(np.median(empirical_sigma / analytic_sigma)),
        "empirical_vs_cov_correlation_absmax": float(np.max(np.abs(empirical_corr - analytic_corr))),
        "phase_chi2_about_empirical_mean": phase_chi2.tolist(),
        "phase_chi2_mean_about_empirical_mean": float(np.mean(phase_chi2)),
        "expected_phase_chi2_mean_about_empirical_mean": float(
            (len(PHASES) - 1) / len(PHASES) * values.shape[1]
        ),
    }


def make_plot(
    path: Path,
    *,
    inputs: dict[str, Any],
    mean_xi0: np.ndarray,
    formal_map: np.ndarray,
    chains: dict[str, np.ndarray],
    phase_profiles: list[dict[str, Any]],
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.backends.backend_pdf import PdfPages

    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp.pdf")
    s = np.asarray(inputs["s"], dtype="f8")
    mask = s >= 50.0
    sigma_mean = np.sqrt(np.diag(inputs["covariance"])) / np.sqrt(len(PHASES))
    with PdfPages(temporary) as pdf:
        figure, axes = plt.subplots(2, 1, figsize=(8.6, 6.6), sharex=True, gridspec_kw={"height_ratios": [2.2, 1.0]})
        axes[0].errorbar(s, s**2 * mean_xi0, yerr=s**2 * sigma_mean, fmt="o", ms=3.5, color="#1f4e79", label="wide-lightcone x25 mean")
        axes[0].plot(s[mask], s[mask] ** 2 * formal_map, color="#b22222", lw=1.6, label="formal-GIC best fit")
        axes[0].axvline(50.0, color="0.4", ls="--", lw=0.8)
        axes[0].set_ylabel(r"$s^2\xi_0(s)$")
        axes[0].legend(frameon=False)
        residual = (mean_xi0[mask] - formal_map) / sigma_mean[mask]
        axes[1].axhline(0.0, color="0.4", lw=0.8)
        axes[1].plot(s[mask], residual, "o-", ms=3.5, lw=0.8, color="#1f4e79")
        axes[1].set(xlabel=r"$s\,[h^{-1}{\rm Mpc}]$", ylabel=r"residual / $\sigma_{\rm mean}$")
        figure.tight_layout()
        pdf.savefig(figure)
        plt.close(figure)

        names = ("fnl_loc", "b1", "sigma_s")
        labels = (r"$f_{\rm NL}^{\rm loc}$", r"$b_1$", r"$\sigma_s\,[h^{-1}{\rm Mpc}]$")
        figure, axes = plt.subplots(1, 3, figsize=(11.2, 3.4))
        colors = {"no_gic": "0.45", "formal_gic": "#1f4e79"}
        for index, (name, label) in enumerate(zip(names, labels, strict=True)):
            combined = np.concatenate([chains[model][:, index] for model in ("no_gic", "formal_gic")])
            lo, hi = np.percentile(combined, [0.2, 99.8])
            bins = np.linspace(lo, hi, 80)
            for model, text in (("no_gic", "no GIC"), ("formal_gic", "formal GIC")):
                axes[index].hist(chains[model][:, index], bins=bins, density=True, histtype="step", lw=1.8, color=colors[model], label=text)
            if index == 0:
                axes[index].axvline(0.0, color="0.5", ls="--", lw=0.8)
            axes[index].set_xlabel(label)
        axes[0].set_ylabel("posterior density")
        axes[0].legend(frameon=False)
        figure.tight_layout()
        pdf.savefig(figure)
        plt.close(figure)

        figure, axes = plt.subplots(1, 3, figsize=(11.2, 3.4))
        theta = np.asarray(
            [[row["theta"][name] for name in names] for row in phase_profiles], dtype="f8"
        )
        for index, label in enumerate(labels):
            axes[index].hist(theta[:, index], bins=10, color="#4c78a8", alpha=0.75)
            if index == 0:
                axes[index].axvline(0.0, color="0.5", ls="--", lw=0.8)
            axes[index].set_xlabel(label)
        axes[0].set_ylabel("phase count")
        figure.tight_layout()
        pdf.savefig(figure)
        plt.close(figure)
    temporary.replace(path)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--summary-path", type=Path, required=True)
    parser.add_argument("--covariance-path", type=Path, required=True)
    parser.add_argument("--mean-window-path", type=Path, required=True)
    parser.add_argument("--phase-window-root", type=Path, required=True)
    parser.add_argument("--output-prefix", type=Path, required=True)
    parser.add_argument("--plot", type=Path, required=True)
    parser.add_argument("--threads", type=int, default=8)
    parser.add_argument("--nsub-window", type=int, default=200000)
    parser.add_argument("--window-seed-base", type=int, default=430340)
    parser.add_argument("--nwalkers", type=int, default=48)
    parser.add_argument("--nsteps", type=int, default=8000)
    parser.add_argument("--burnin", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=430450)
    parser.add_argument("--nmu", type=int, default=64)
    parser.add_argument("--sigma-grid-step", type=float, default=0.05)
    args = parser.parse_args()
    if args.burnin >= args.nsteps:
        raise ValueError("--burnin must be smaller than --nsteps")
    output_npz, output_json = args.output_prefix.with_suffix(".npz"), args.output_prefix.with_suffix(".json")
    if any(path.exists() for path in (output_npz, output_json, args.plot)):
        raise FileExistsError("immutable wide-lightcone xi0 closure output exists")

    started = time.perf_counter()
    cpus = set_affinity(args.threads)
    rows = read_jsonl(args.manifest)
    if tuple(row["phase"] for row in rows) != PHASES:
        raise RuntimeError("manifest phase order is not ph000..ph024")
    analysis_scopes = {str(row.get("analysis_scope", "")) for row in rows}
    observed_intervals = {
        (float(row["zmin_observed"]), float(row["zmax_observed"]))
        for row in rows
    }
    if len(analysis_scopes) != 1 or len(observed_intervals) != 1:
        raise RuntimeError(
            "manifest mixes analysis scopes or observed-redshift intervals: "
            f"scopes={sorted(analysis_scopes)}, intervals={sorted(observed_intervals)}"
        )
    analysis_scope = analysis_scopes.pop()
    observed_interval = observed_intervals.pop()
    is_boxsafe = "boxsafe" in analysis_scope
    inputs = load_inputs(args.summary_path, args.covariance_path)
    mean_xi = np.mean(inputs["xi"], axis=0)
    mean_rr = np.mean(inputs["rr"], axis=0)
    zeff_mean = float(np.mean(inputs["zeff"]))
    windows = {"mean": load_window(args.mean_window_path)}
    from task43_build_rsd_formal_gic_window import output_path as phase_window_path

    windows.update(
        {
            phase: load_window(
                phase_window_path(
                    phase,
                    args.nsub_window,
                    args.window_seed_base,
                    root=args.phase_window_root,
                )
            )
            for phase in PHASES
        }
    )
    theory_cache_path = build_cache(
        zeff=zeff_mean,
        boxsize=2000.0,
        kmax=5.0,
        ells=(0, 2),
        cosmology="abacus_c000",
    )
    exact_model = FullDiscreteRSDModel(theory_cache_path, nmu=args.nmu)
    model = FastWindowRSDModel(exact_model, windows, sigma_step=args.sigma_grid_step)
    surrogate = model.validate()
    if surrogate["status"] != "pass":
        raise RuntimeError(f"wide xi0 likelihood surrogate failed: {surrogate}")

    nominal: dict[str, dict[str, Any]] = {}
    chains: dict[str, np.ndarray] = {}
    logp: dict[str, np.ndarray] = {}
    maps: dict[str, np.ndarray] = {}
    convergence: dict[str, Any] = {}
    for index, model_name in enumerate(("no_gic", "formal_gic")):
        fit, chain, probability, model_map = fit_one(
            inputs=inputs,
            xi_by_ell=mean_xi,
            rr=mean_rr,
            fast_model=model,
            window_key="mean",
            model=model_name,
            ells=(0,),
            smin=50.0,
            nwalkers=args.nwalkers,
            nsteps=args.nsteps,
            burnin=args.burnin,
            seed=args.seed + 100 * index,
            optimizer_only=False,
        )
        nominal[model_name] = fit
        chains[model_name] = chain
        logp[model_name] = probability
        maps[model_name] = model_map
        convergence[model_name] = convergence_diagnostics(
            chain,
            names=list(fit["parameter_names"]),
            nwalkers=args.nwalkers,
            nsteps=args.nsteps,
            burnin=args.burnin,
        )
        print(json.dumps({"stage": "nominal", "model": model_name, "fnl": fit["fnl_loc"]["q50"]}), flush=True)

    phase_profiles: list[dict[str, Any]] = []
    for index, phase in enumerate(PHASES):
        fit, _, _, _ = fit_one(
            inputs=inputs,
            xi_by_ell=inputs["xi"][index],
            rr=inputs["rr"][index],
            fast_model=model,
            window_key=phase,
            model="formal_gic",
            ells=(0,),
            smin=50.0,
            nwalkers=8,
            nsteps=2,
            burnin=0,
            seed=args.seed + 1000 + index,
            optimizer_only=True,
        )
        phase_profiles.append(profile_record(phase, "formal_gic", fit))

    primary = nominal["formal_gic"]
    fnl = primary["fnl_loc"]
    sigma68 = 0.5 * (float(fnl["q84"]) - float(fnl["q16"]))
    null_ratio = abs(float(fnl["q50"])) / sigma68
    context = primary["task43_fit_context"]
    mean_chi2 = len(PHASES) * float(context["chi2_map_single_covariance"])
    mean_dof = int(context["dof_nominal"])
    mean_pte = float(chi2_distribution.sf(mean_chi2, mean_dof))
    ensemble_chi2 = float(sum(row["chi2"] for row in phase_profiles))
    ensemble_dof = int(sum(row["dof"] for row in phase_profiles))
    ensemble_pte = float(chi2_distribution.sf(ensemble_chi2, ensemble_dof))
    convergence_gates = {
        f"{model_name}_{key}": bool(value)
        for model_name in ("no_gic", "formal_gic")
        for key, value in convergence[model_name]["gates"].items()
    }
    gates = {
        "null_abs_median_over_sigma68_single_below_0p3": bool(null_ratio < 0.3),
        "mean_pte_Cmean_above_0p05": bool(mean_pte > 0.05),
        "phase_ensemble_profile_pte_above_0p05": bool(ensemble_pte > 0.05),
        **convergence_gates,
    }
    diagnostic_status = "pass" if all(gates.values()) else "validation_failed"
    scatter = scatter_diagnostics(inputs)
    no_formal_shift = (
        float(primary["fnl_loc"]["q50"]) - float(nominal["no_gic"]["fnl_loc"]["q50"])
    ) / sigma68

    make_plot(
        args.plot,
        inputs=inputs,
        mean_xi0=mean_xi[0],
        formal_map=maps["formal_gic"],
        chains=chains,
        phase_profiles=phase_profiles,
    )
    save_payload: dict[str, Any] = {
        "s": inputs["s"],
        "s_edges": inputs["s_edges"],
        "ells": np.asarray([0], dtype="i8"),
        "phases": np.asarray(PHASES),
        "zeff_by_phase": inputs["zeff"],
        "zeff_mean": np.asarray(zeff_mean),
        "xi_multipoles_by_phase": inputs["xi"],
        "xi_multipoles_mean": mean_xi,
        "RR_by_phase": inputs["rr"],
        "RR_mean": mean_rr,
        "covariance_single_realization": inputs["covariance"],
        "covariance_of_mean": inputs["covariance"] / len(PHASES),
        "formal_gic_chain_flat": chains["formal_gic"],
        "formal_gic_log_probability_flat": logp["formal_gic"],
        "formal_gic_model_map": maps["formal_gic"],
        "no_gic_chain_flat": chains["no_gic"],
        "no_gic_log_probability_flat": logp["no_gic"],
        "no_gic_model_map": maps["no_gic"],
        "phase_profile_formalgic_theta": np.asarray(
            [[row["theta"][name] for name in ("fnl_loc", "b1", "sigma_s")] for row in phase_profiles]
        ),
        "phase_profile_formalgic_chi2": np.asarray([row["chi2"] for row in phase_profiles]),
    }
    atomic_savez(output_npz, **save_payload)
    payload = {
        "task": "task43_fit_rsd_wide_lightcone_xi0_smin50",
        "status": "validation_failed",
        "diagnostic_execution_status": "pass",
        "diagnostic_validation_status": diagnostic_status,
        "classification": (
            "box-safe monopole-only Task 4.3.2 diagnostic; rawbox gate remains failed"
            if is_boxsafe
            else "wide-redshift monopole-only Task 4.3.2 diagnostic; rawbox gate remains failed"
        ),
        "analysis_scope": analysis_scope,
        "observed_redshift_open_interval": list(observed_interval),
        "phases": list(PHASES),
        "nphase": len(PHASES),
        "cpu_affinity": cpus,
        "model": {
            "name": "FullDiscrete shell-averaged Kaiser x squared-Lorentzian FoG",
            "fit_ells_primary": [0],
            "p_fixed": 1.0,
            "zeff_mean": zeff_mean,
            "redshift_policy": (
                "pair-weighted box-safe-sample zeff; phase-specific formal-GIC windows"
                if is_boxsafe
                else "pair-weighted wide-sample zeff; phase-specific formal-GIC windows"
            ),
            "theory_cache_path": str(theory_cache_path),
            "theory_cache_sha256": sha256_file(theory_cache_path),
            "likelihood_surrogate_validation": surrogate,
        },
        "covariance_policy": {
            "path": str(args.covariance_path),
            "sha256": sha256_file(args.covariance_path),
            "quoted_posterior": "C_single, never divided by 25",
            "mean_goodness_of_fit": "C_mean=C_single/25",
            "phase_profiles": "C_single",
            "eigenvalue_min": float(inputs["covariance_eigenvalues"][0]),
            "condition_number": float(inputs["covariance_eigenvalues"][-1] / inputs["covariance_eigenvalues"][0]),
            "source_meta": inputs["covariance_meta"],
            "x25_scatter_comparison_xi0_smin50": scatter,
        },
        "formal_gic_window_policy": {
            "mean_window_path": str(args.mean_window_path),
            "mean_window_sha256": sha256_file(args.mean_window_path),
            "phase_window_root": str(args.phase_window_root),
            "phase_nsub": args.nsub_window,
            "phase_seed_base": args.window_seed_base,
        },
        "nominal_mean_smin50_xi0": {name: compact_fit(fit) for name, fit in nominal.items()},
        "mcmc_convergence": convergence,
        "no_gic_to_formal_gic_fnl_shift_sigma_formal": float(no_formal_shift),
        "phase_profiles": phase_profiles,
        "phase_ensemble_primary": {"model": "formal_gic", "chi2": ensemble_chi2, "dof": ensemble_dof, "pte": ensemble_pte},
        "mean_goodness_primary": {"model": "formal_gic", "chi2_Cmean": mean_chi2, "dof": mean_dof, "pte": mean_pte},
        "primary_null_abs_median_over_sigma68_single": float(null_ratio),
        "primary_gates": gates,
        "rawbox_dependency": {"status": "validation_failed", "task44_greenlight": False},
        "inputs": {
            "manifest": str(args.manifest),
            "manifest_sha256": sha256_file(args.manifest),
            "summary": str(args.summary_path),
            "summary_sha256": sha256_file(args.summary_path),
            "measurements": [str(row["lightcone_xi_path"]) for row in rows],
        },
        "output_npz": str(output_npz),
        "output_npz_sha256": sha256_file(output_npz),
        "output_plot_pdf": str(args.plot),
        "output_plot_pdf_sha256": sha256_file(args.plot),
        "elapsed_sec": time.perf_counter() - started,
    }
    atomic_write_json(output_json, jsonable(payload))
    print(json.dumps({"status": payload["status"], "diagnostic_validation_status": diagnostic_status, "gates": gates, "output": str(output_json)}, sort_keys=True))


if __name__ == "__main__":
    main()
