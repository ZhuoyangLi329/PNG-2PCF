#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Run a controlled Task43 lightcone RascalC covariance calculation.

The runner fixes two failure modes in the historical Task43/Task44 ports:

1. Changing ``--max-random`` changes only Monte-Carlo geometry resolution; it
   never changes the physical target number of data objects used for shot
   noise normalization.
2. The covariance bins (50--350 Mpc/h, 10 Mpc/h wide) and the fine xi table
   (near zero through 600 Mpc/h) are distinct inputs.  The xi table also carries
   an explicit mother-box mode policy produced by
   ``task43_build_rascalc_kmin_xi.py``.

This is an aperiodic lightcone geometry calculation.  ``boxsize=None`` is
therefore passed to RascalC; the 2 Gpc/h mother box enters through the chosen
FullDiscrete xi table, not by pretending the cut sky itself is periodic.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

for _name in ("MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_name, "1")

import numpy as np
from pycorr import TwoPointCorrelationFunction

from task43_config import BOX_SIZE, PROJECT_ROOT, S_EDGES, SUMMARY_DIR
from task43_fkp_zeff import load_fkp_summary, total_weight_from_summary


RASCALC_SOURCE = Path("/global/common/software/desi/users/mrash/RascalC")
if str(RASCALC_SOURCE) not in sys.path:
    sys.path.insert(0, str(RASCALC_SOURCE))

ARCHIVE_ROOT = PROJECT_ROOT / "old_doc_codes" / "task4_task44_cleanup_20260707T061844Z" / "moved" / "outputs" / "task43_outputs"
DEFAULT_ALLCOUNTS = ARCHIVE_ROOT / "rascalc_covariance" / "allcounts_AbacusSummit_base_c000_ph000_z0p6_0p8_mmin1p4e13_x25_s50_350_ds10_nmu20_fkpP010000.npy"
DEFAULT_HALO = ARCHIVE_ROOT / "halo_catalogs" / "halo_lightcone_AbacusSummit_base_c000_ph000_z0p6_0p8_mmin1p4e13.npz"
DEFAULT_RANDOM = ARCHIVE_ROOT / "randoms" / "random_AbacusSummit_base_c000_ph000_z0p6_0p8_mmin1p4e13_x25.npz"
DEFAULT_XI = SUMMARY_DIR.parent / "rascalc_covariance" / "inputs" / "task43_rascalc_xi_kmin_L2000_fnl0_b1cov2p5.npz"
DEFAULT_SCATTER = SUMMARY_DIR / "task43_mean_xi_mmin1p4e13_x25_s50_350_ds10_fkpP010000.npz"
DEFAULT_OUTPUT_ROOT = SUMMARY_DIR.parent / "rascalc_covariance" / "lightcone_kmin"

XI_KEYS = {
    "full_discrete": "xi_full_discrete",
    "continuous_boxcut": "xi_continuous_boxcut",
    "continuous_lowk": "xi_continuous_lowk",
}


def _jsonable(value: Any) -> Any:
    """Convert numpy/path values to plain JSON values."""
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, Path):
        return str(value)
    return value


def format_tag(value: float) -> str:
    """Format a compact filesystem-safe floating-point tag."""
    return f"{float(value):g}".replace("-", "m").replace(".", "p")


def covariance_summary(covariance: np.ndarray) -> dict[str, Any]:
    """Return the core positive-definiteness and conditioning diagnostics."""
    cov = 0.5 * (np.asarray(covariance, dtype="f8") + np.asarray(covariance, dtype="f8").T)
    eig = np.linalg.eigvalsh(cov)
    sigma = np.sqrt(np.diag(cov))
    corr = cov / np.outer(sigma, sigma)
    corr_eig = np.linalg.eigvalsh(0.5 * (corr + corr.T))
    return {
        "shape": [int(v) for v in cov.shape],
        "min_eigenvalue": float(eig[0]),
        "max_eigenvalue": float(eig[-1]),
        "condition_number": float(np.linalg.cond(cov)),
        "sigma_min": float(np.min(sigma)),
        "sigma_max": float(np.max(sigma)),
        "corr_min_eigenvalue": float(corr_eig[0]),
        "corr_max_eigenvalue": float(corr_eig[-1]),
        "corr_offdiag_max_abs": float(np.max(np.abs(corr - np.eye(corr.shape[0])))),
    }


def weight_moments(weight: np.ndarray) -> dict[str, float | int]:
    """Return moments needed to audit simple versus effective-number norms."""
    weight = np.asarray(weight, dtype="f8")
    sumw = float(np.sum(weight))
    sumw2 = float(np.sum(weight * weight))
    return {
        "n": int(weight.size),
        "sum_w": sumw,
        "sum_w2": sumw2,
        "effective_n": float(sumw**2 / sumw2),
        "min_w": float(np.min(weight)),
        "max_w": float(np.max(weight)),
        "mean_w": float(np.mean(weight)),
        "std_w": float(np.std(weight)),
    }


def catalog_arrays(
    path: Path,
    *,
    fkp_summary: dict[str, np.ndarray],
    p0: float,
    max_rows: int | None,
    seed: int,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    """Load Cartesian positions and FKP weights, with an unbiased row subset."""
    data = np.load(path, allow_pickle=False)
    n_total = int(np.asarray(data["Z"]).size)
    if max_rows is not None and 0 < int(max_rows) < n_total:
        rng = np.random.default_rng(int(seed))
        selection: slice | np.ndarray = np.sort(rng.choice(n_total, size=int(max_rows), replace=False))
    else:
        selection = slice(None)
    z = np.asarray(data["Z"][selection], dtype="f8")
    base = np.asarray(data["WEIGHT"][selection], dtype="f8") if "WEIGHT" in data.files else np.ones(z.size, dtype="f8")
    weight = total_weight_from_summary(z, base, fkp_summary, p0=float(p0))
    xyz = np.column_stack(
        [
            np.asarray(data["X"][selection], dtype="f8"),
            np.asarray(data["Y"][selection], dtype="f8"),
            np.asarray(data["Zcart"][selection], dtype="f8"),
        ]
    )
    meta = {
        "path": str(path),
        "n_total": n_total,
        "n_used": int(z.size),
        "subsampled": bool(z.size != n_total),
        "subsample_seed": int(seed),
        "weights": weight_moments(weight),
        "z_min": float(np.min(z)),
        "z_max": float(np.max(z)),
    }
    return xyz, weight, meta


def full_data_weights(path: Path, *, fkp_summary: dict[str, np.ndarray], p0: float) -> tuple[np.ndarray, dict[str, Any]]:
    """Load all data weights used to define the fixed physical shot-noise target."""
    data = np.load(path, allow_pickle=False)
    z = np.asarray(data["Z"], dtype="f8")
    base = np.asarray(data["WEIGHT"], dtype="f8") if "WEIGHT" in data.files else np.ones(z.size, dtype="f8")
    weight = total_weight_from_summary(z, base, fkp_summary, p0=float(p0))
    return weight, {"path": str(path), "weights": weight_moments(weight)}


def load_xi_table(path: Path, model: str) -> tuple[tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray], dict[str, Any]]:
    """Load one isotropic xi model with explicit interpolation-bin edges.

    RascalC commit ``ff0c8f7`` currently constructs the edges for a three-array
    non-uniform theory table with numpy/list addition, which can preserve the
    wrong length.  Supplying explicit midpoint edges avoids that wrapper bug.
    ``xi_refinement_iterations=0`` below keeps the point-sampled model unchanged.
    """
    key = XI_KEYS[str(model)]
    data = np.load(path, allow_pickle=False)
    for required in ("r", "mu", key, "meta_json"):
        if required not in data.files:
            raise KeyError(f"{path} is missing {required}; keys={data.files}")
    r = np.asarray(data["r"], dtype="f8")
    mu = np.asarray(data["mu"], dtype="f8")
    xi0 = np.asarray(data[key], dtype="f8")
    if r.ndim != 1 or mu.ndim != 1 or xi0.shape != r.shape:
        raise ValueError(f"invalid xi arrays in {path}: r={r.shape}, mu={mu.shape}, xi={xi0.shape}")
    if r[0] > 0.01:
        raise ValueError(f"RascalC xi table must begin near zero; first r={r[0]}")
    xi_smu = np.repeat(xi0[:, None], mu.size, axis=1)
    r_edges = np.empty(r.size + 1, dtype="f8")
    r_edges[0] = 1.0e-4
    r_edges[1:-1] = 0.5 * (r[:-1] + r[1:])
    r_edges[-1] = 2.0 * r[-1] - r_edges[-2]
    if np.any(np.diff(r_edges) <= 0.0):
        raise ValueError("explicit xi radial edges are not strictly increasing")
    source_meta = json.loads(str(np.asarray(data["meta_json"]).item()))
    return (r, mu, xi_smu, r_edges), source_meta


def parse_r_inv(log_path: Path) -> list[float]:
    """Extract all RascalC R_inv half-estimate values from its text log."""
    if not log_path.exists():
        return []
    text = log_path.read_text(encoding="utf-8", errors="replace")
    values: list[float] = []
    for match in re.finditer(r"R_inv .*? are ([0-9.eE+\-]+) and ([0-9.eE+\-]+)", text):
        values.extend([float(match.group(1)), float(match.group(2))])
    return values


def scatter_comparison(path: Path, s_edges: np.ndarray, covariance: np.ndarray) -> dict[str, Any] | None:
    """Compare one single-realization covariance to the 25-phase xi scatter."""
    if not path.exists():
        return None
    data = np.load(path, allow_pickle=False)
    s_all = np.asarray(data["s"], dtype="f8")
    xi_all = np.asarray(data["xi0_all"], dtype="f8")
    centers = 0.5 * (s_edges[:-1] + s_edges[1:])
    indices = []
    for center in centers:
        found = np.flatnonzero(np.isclose(s_all, center, rtol=0.0, atol=1.0e-8))
        if found.size != 1:
            raise ValueError(f"scatter summary does not contain exactly one s={center} bin")
        indices.append(int(found[0]))
    xi = xi_all[:, indices]
    residual = xi - np.mean(xi, axis=0)
    cov = np.asarray(covariance, dtype="f8")
    precision = np.linalg.inv(cov)
    chi2 = np.einsum("ij,jk,ik->i", residual, precision, residual)
    sample_cov = np.cov(xi, rowvar=False, ddof=1)
    ratio = np.sqrt(np.diag(sample_cov) / np.diag(cov))
    return {
        "path": str(path),
        "nreal": int(xi.shape[0]),
        "nbins": int(xi.shape[1]),
        "expected_chi2_mean_about_sample_mean": float((xi.shape[0] - 1) / xi.shape[0] * xi.shape[1]),
        "chi2_mean": float(np.mean(chi2)),
        "chi2_min": float(np.min(chi2)),
        "chi2_max": float(np.max(chi2)),
        "sample_std_over_cov_sigma_median": float(np.median(ratio)),
        "sample_std_over_cov_sigma_min": float(np.min(ratio)),
        "sample_std_over_cov_sigma_max": float(np.max(ratio)),
    }


def rascalc_revision() -> dict[str, str | None]:
    """Record the exact local RascalC source revision when available."""
    try:
        safe = f"safe.directory={RASCALC_SOURCE}"
        commit = subprocess.check_output(["git", "-c", safe, "-C", str(RASCALC_SOURCE), "rev-parse", "HEAD"], text=True).strip()
        date = subprocess.check_output(
            ["git", "-c", safe, "-C", str(RASCALC_SOURCE), "show", "-s", "--format=%cI", "HEAD"],
            text=True,
        ).strip()
        return {"source": str(RASCALC_SOURCE), "commit": commit, "commit_date": date}
    except Exception:
        return {"source": str(RASCALC_SOURCE), "commit": None, "commit_date": None}


def output_label(args: argparse.Namespace, nrandom_used: int) -> str:
    """Construct a label whose physics and convergence settings are visible."""
    base = (
        f"xi{args.xi_model}_ndata{args.ndata_definition}_nran{int(nrandom_used)}_"
        f"nloop{int(args.n_loops)}_lps{int(args.loops_per_sample)}_"
        f"n2{int(args.n2)}n3{int(args.n3)}n4{int(args.n4)}_"
        f"alpha{format_tag(args.shot_noise_rescaling)}_xiedges_v2_seed{int(args.seed)}"
    )
    return str(args.label) if args.label else base


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--allcounts", type=Path, default=DEFAULT_ALLCOUNTS)
    parser.add_argument("--halo-catalog", type=Path, default=DEFAULT_HALO)
    parser.add_argument("--random-catalog", type=Path, default=DEFAULT_RANDOM)
    parser.add_argument("--fkp-summary", type=Path, default=SUMMARY_DIR / "task43_fkp_zeff_mmin1p4e13_x25.npz")
    parser.add_argument("--xi-input", type=Path, default=DEFAULT_XI)
    parser.add_argument("--xi-model", choices=tuple(XI_KEYS), default="full_discrete")
    parser.add_argument("--scatter-summary", type=Path, default=DEFAULT_SCATTER)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--label", type=str, default=None)
    parser.add_argument("--p0", type=float, default=10000.0)
    parser.add_argument("--max-random", type=int, default=300000, help="0 means the complete random catalog")
    parser.add_argument("--random-seed", type=int, default=20260709)
    parser.add_argument("--ndata-definition", choices=("effective", "simple"), default="effective")
    parser.add_argument("--nthreads", type=int, default=8)
    parser.add_argument("--n2", type=int, default=5)
    parser.add_argument("--n3", type=int, default=10)
    parser.add_argument("--n4", type=int, default=20)
    parser.add_argument("--n-loops", type=int, default=128)
    parser.add_argument("--loops-per-sample", type=int, default=8)
    parser.add_argument("--sampling-grid-size", type=int, default=301)
    parser.add_argument("--xi-cut-s", type=float, default=600.0)
    parser.add_argument("--shot-noise-rescaling", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=20260709)
    args = parser.parse_args()

    for path in (args.allcounts, args.halo_catalog, args.random_catalog, args.fkp_summary, args.xi_input):
        if not path.exists():
            raise FileNotFoundError(f"missing required input: {path}")
    if not RASCALC_SOURCE.exists():
        raise FileNotFoundError(f"missing RascalC source: {RASCALC_SOURCE}")
    if int(args.nthreads) < 1 or int(args.nthreads) > 8:
        raise ValueError("--nthreads must be between 1 and 8 on the login node")
    if int(args.n_loops) % int(args.nthreads):
        raise ValueError("--n-loops must be divisible by --nthreads")
    if int(args.n_loops) % int(args.loops_per_sample):
        raise ValueError("--n-loops must be divisible by --loops-per-sample")
    nsamples = int(args.n_loops) // int(args.loops_per_sample)
    if not 10 <= nsamples <= 30:
        raise ValueError(f"RascalC recommends 10--30 output subsamples; current value is {nsamples}")
    if min(int(args.n2), int(args.n3), int(args.n4)) < 5:
        raise ValueError("N2/N3/N4 below 5 are intentionally rejected by this controlled runner")

    allcounts = TwoPointCorrelationFunction.load(args.allcounts)
    if getattr(allcounts, "name", "") != "landyszalay":
        raise ValueError(f"survey allcounts must be Landy-Szalay, got {getattr(allcounts, 'name', None)}")
    s_edges = np.asarray(allcounts.edges[0], dtype="f8")
    mu_edges = np.asarray(allcounts.edges[1], dtype="f8")
    if s_edges.shape != S_EDGES.shape or not np.allclose(s_edges, S_EDGES):
        raise ValueError(f"allcounts covariance edges are not Task43 50--350 ds10: {s_edges}")
    if not (np.isclose(mu_edges[0], -1.0) and np.isclose(mu_edges[-1], 1.0) and (mu_edges.size - 1) % 2 == 0):
        raise ValueError("allcounts must use an even number of unwrapped -1..1 mu bins")

    fkp_summary = load_fkp_summary(args.fkp_summary)
    full_weight, data_target_meta = full_data_weights(args.halo_catalog, fkp_summary=fkp_summary, p0=float(args.p0))
    data_simple = int(full_weight.size)
    data_effective = float(np.sum(full_weight) ** 2 / np.sum(full_weight**2))
    if args.ndata_definition == "effective":
        ndata_target = data_effective
        effective_no_def = True
    else:
        ndata_target = float(data_simple)
        effective_no_def = False

    max_random = None if int(args.max_random) <= 0 else int(args.max_random)
    random_xyz, random_weight, random_meta = catalog_arrays(
        args.random_catalog,
        fkp_summary=fkp_summary,
        p0=float(args.p0),
        max_rows=max_random,
        seed=int(args.random_seed),
    )
    xi_table, xi_meta = load_xi_table(args.xi_input, str(args.xi_model))

    label = output_label(args, random_xyz.shape[0])
    run_dir = Path(args.output_root) / label
    summary_npz = run_dir / "task43_rascalc_lightcone_covariance.npz"
    summary_json = summary_npz.with_suffix(".json")
    if summary_npz.exists() and summary_json.exists():
        print(f"[skip] existing {summary_npz}")
        return
    if run_dir.exists() and any(run_dir.iterdir()):
        raise FileExistsError(f"non-empty partial run directory; use a new --label after auditing it: {run_dir}")
    run_dir.mkdir(parents=True, exist_ok=True)

    from RascalC import run_cov

    print(
        f"[setup] xi={args.xi_model} ndata_target={ndata_target:.3f} "
        f"nrandom_mc={random_xyz.shape[0]} loops={args.n_loops} samples={nsamples}",
        flush=True,
    )
    started = time.perf_counter()
    results = run_cov(
        mode="legendre_projected",
        max_l=0,
        boxsize=None,
        nthread=int(args.nthreads),
        N2=int(args.n2),
        N3=int(args.n3),
        N4=int(args.n4),
        n_loops=int(args.n_loops),
        loops_per_sample=int(args.loops_per_sample),
        allcounts_11=allcounts,
        xi_table_11=xi_table,
        no_data_galaxies1=float(ndata_target),
        effective_no_def=bool(effective_no_def),
        position_type="pos",
        randoms_positions1=random_xyz,
        randoms_weights1=random_weight.copy(),
        normalize_wcounts=True,
        out_dir=str(run_dir / "rascalc_out"),
        tmp_dir=str(run_dir / "rascalc_tmp"),
        xi_cut_s=float(args.xi_cut_s),
        xi_refinement_iterations=0,
        sampling_grid_size=int(args.sampling_grid_size),
        shot_noise_rescaling1=float(args.shot_noise_rescaling),
        seed=int(args.seed),
    )
    elapsed = float(time.perf_counter() - started)
    if "full_theory_covariance" not in results:
        raise KeyError(f"RascalC result has no full_theory_covariance; keys={list(results)}")
    covariance = np.asarray(results["full_theory_covariance"], dtype="f8")
    if covariance.shape != (s_edges.size - 1, s_edges.size - 1):
        raise ValueError(f"unexpected RascalC covariance shape {covariance.shape}")
    quality = covariance_summary(covariance)
    if quality["min_eigenvalue"] <= 0.0:
        raise ValueError(f"RascalC covariance is not positive definite: {quality}")

    log_path = run_dir / "rascalc_out" / "log.txt"
    r_inv = parse_r_inv(log_path)
    scatter = scatter_comparison(args.scatter_summary, s_edges, covariance)
    meta = {
        "status": "done",
        "task": "task43_run_rascalc_lightcone_kmin",
        "purpose": "controlled FKP-weighted Task43 lightcone RascalC covariance with explicit mother-box xi mode policy",
        "output_npz": str(summary_npz),
        "run_dir": str(run_dir),
        "rascalc": rascalc_revision(),
        "inputs": {
            "allcounts": str(args.allcounts),
            "halo_catalog": str(args.halo_catalog),
            "random_catalog": str(args.random_catalog),
            "fkp_summary": str(args.fkp_summary),
            "xi_input": str(args.xi_input),
            "scatter_summary": str(args.scatter_summary),
        },
        "physics": {
            "geometry": "aperiodic lightcone; RascalC boxsize=None",
            "mother_boxsize": float(xi_meta["models"][str(args.xi_model)].get("boxsize", BOX_SIZE)),
            "xi_model": str(args.xi_model),
            "xi_model_meta": xi_meta["models"][str(args.xi_model)],
            "p0": float(args.p0),
            "weighting": "WEIGHT_TOTAL=WEIGHT*WEIGHT_FKP",
            "ndata_definition": str(args.ndata_definition),
            "ndata_target": float(ndata_target),
            "ndata_simple": data_simple,
            "ndata_effective": data_effective,
            "effective_no_def": bool(effective_no_def),
            "shot_noise_rescaling": float(args.shot_noise_rescaling),
            "alpha_sn_warning": "alpha_SN=1 is a diagnostic default; production requires jackknife or explicitly labeled mock calibration.",
        },
        "data_target": data_target_meta,
        "random_monte_carlo": random_meta,
        "allcounts": {
            "estimator": str(allcounts.name),
            "shape_unwrapped": [int(v) for v in allcounts.shape],
            "s_edges": [float(v) for v in s_edges],
            "mu_edges_unwrapped": [float(v) for v in mu_edges],
            "normalization": "normalize_wcounts=True; independent of random MC subset size",
        },
        "settings": {
            "mode": "legendre_projected",
            "max_l": 0,
            "nthreads": int(args.nthreads),
            "N2": int(args.n2),
            "N3": int(args.n3),
            "N4": int(args.n4),
            "n_loops": int(args.n_loops),
            "loops_per_sample": int(args.loops_per_sample),
            "n_output_subsamples": nsamples,
            "sampling_grid_size": int(args.sampling_grid_size),
            "xi_cut_s": float(args.xi_cut_s),
            "xi_refinement_iterations": 0,
            "xi_table_interface": "four arrays with explicit midpoint edges; point samples and refinement disabled",
            "xi_table_interface_reason": "work around RascalC ff0c8f7 non-uniform three-array edge-length bug",
            "seed": int(args.seed),
            "random_seed": int(args.random_seed),
        },
        "quality": {
            "covariance": quality,
            "r_inv_values": r_inv,
            "r_inv_max": None if not r_inv else float(max(r_inv)),
            "r_inv_goal": 0.05,
            "r_inv_pass": bool(r_inv and max(r_inv) < 0.05),
            "scatter": scatter,
        },
        "elapsed_sec": elapsed,
        "result_keys": {
            key: ([int(v) for v in value.shape] if isinstance(value, np.ndarray) else _jsonable(value))
            for key, value in results.items()
        },
    }
    save: dict[str, Any] = {
        "s": 0.5 * (s_edges[:-1] + s_edges[1:]),
        "s_edges": s_edges,
        "covariance_single_realization": covariance,
        "correlation": covariance / np.outer(np.sqrt(np.diag(covariance)), np.sqrt(np.diag(covariance))),
        "xi_model": np.asarray(str(args.xi_model)),
        "ndata_target": np.array(float(ndata_target), dtype="f8"),
        "nrandom_monte_carlo": np.array(int(random_xyz.shape[0]), dtype="i8"),
        "shot_noise_rescaling": np.array(float(args.shot_noise_rescaling), dtype="f8"),
        "meta_json": np.asarray(json.dumps(_jsonable(meta), sort_keys=True)),
    }
    for key, value in results.items():
        if isinstance(value, np.ndarray) and value.dtype != object:
            save[f"rascalc_{key}"] = value
    np.savez_compressed(summary_npz, **save)
    summary_json.write_text(json.dumps(_jsonable(meta), indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"[done] wrote {summary_npz}")
    print(
        f"[quality] eigmin={quality['min_eigenvalue']:.3e} "
        f"R_inv_max={meta['quality']['r_inv_max']} chi2_mean={None if scatter is None else scatter['chi2_mean']}",
        flush=True,
    )


if __name__ == "__main__":
    main()
