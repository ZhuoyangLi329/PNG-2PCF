#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Summarize Task43 lightcone P(k) measurements into a fit payload.

The payload separates three k-ranges:

1. The measured/window/covariance grid, normally starting at k=0.001.
2. The observed-bin fit cut, inferred from the Task43 lightcone effective
   volume by default.
3. The window theory input grid stored in the jaxpower window matrix.

Do not pass the observed-bin fit kmin into the window theory input.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np

PROJECT_ROOT = Path("/pscratch/sd/l/lzy/PNG-2PCF-model-to-observe")
CODE_DIR = PROJECT_ROOT / "codes" / "task43"
if str(CODE_DIR) not in sys.path:
    sys.path.insert(0, str(CODE_DIR))

from task43_pk_common import (  # noqa: E402
    DEFAULT_FKP_SUMMARY,
    K_MAX_FIT,
    PK_COV_DIR,
    PK_MEASURE_DIR,
    PK_SUMMARY_DIR,
    SN0_SCALE,
    atomic_savez,
    covariance_diagnostics,
    ensure_pk_dirs,
    lightcone_fit_kmin_from_fkp_summary,
    load_fkp_summary,
    p0_output_tag,
    phase_index,
    to_jsonable,
    write_json,
)


def load_summary_json(npz: np.lib.npyio.NpzFile) -> dict[str, Any]:
    if "summary_json" not in npz.files:
        return {}
    return json.loads(str(np.asarray(npz["summary_json"]).item()))


def measurement_files(tag: str, glob_pattern: str | None) -> list[Path]:
    if glob_pattern:
        files = sorted(Path(".").glob(glob_pattern) if not str(glob_pattern).startswith("/") else Path("/").glob(str(glob_pattern)[1:]))
    else:
        files = sorted((PK_MEASURE_DIR / tag).glob("task43_pk_ph*_mesh*_kmax*_dk*.npz"))
    files = [Path(path) for path in files if path.exists()]
    if not files:
        raise FileNotFoundError(f"no Task43 P(k) measurement files found for tag={tag!r}")
    return sorted(files, key=lambda path: phase_index(str(np.load(path, allow_pickle=False)["phase"].item())))


def read_measurement_stack(files: list[Path]) -> dict[str, Any]:
    stack = []
    shot_stack = []
    k_ref = None
    edges_ref = None
    phases = []
    summaries = []
    data_n_used = []
    random_n_used = []
    for path in files:
        with np.load(path, allow_pickle=False) as data:
            k = np.asarray(data["k_obs"], dtype="f8")
            edges = np.asarray(data["k_edges"], dtype="f8")
            if k_ref is None:
                k_ref = k
                edges_ref = edges
            else:
                if not np.allclose(edges, edges_ref, rtol=0.0, atol=1.0e-12):
                    raise ValueError(f"k_edges mismatch in {path}")
            stack.append(np.asarray(data["pk0"], dtype="f8"))
            if "shotnoise" in data.files:
                shot_stack.append(np.asarray(data["shotnoise"], dtype="f8"))
            else:
                shot_stack.append(np.asarray(data["num_shotnoise"], dtype="f8") / np.asarray(data["norm"], dtype="f8"))
            phase = str(np.asarray(data["phase"]).item())
            phases.append(phase)
            summaries.append(load_summary_json(data))
            data_n_used.append(int(np.asarray(data["data_n_used"]).item()))
            random_n_used.append(int(np.asarray(data["random_n_used"]).item()))
    if k_ref is None or edges_ref is None:
        raise RuntimeError("empty measurement stack")
    return {
        "k_obs": k_ref,
        "k_edges": edges_ref,
        "pk_stack": np.vstack(stack).astype("f8"),
        "shotnoise_stack": np.vstack(shot_stack).astype("f8"),
        "phases": np.asarray(phases),
        "summaries": summaries,
        "data_n_used": np.asarray(data_n_used, dtype="i8"),
        "random_n_used": np.asarray(random_n_used, dtype="i8"),
    }


def default_covariance_file(tag: str) -> Path | None:
    files = sorted((PK_COV_DIR / tag).glob("task43_pk_cov_ph*_mesh*_kmax*_dk*.npz"))
    return files[0] if files else None


def read_covariance(path: Path | None) -> dict[str, Any] | None:
    if path is None:
        return None
    with np.load(path, allow_pickle=False) as data:
        cov = np.asarray(data["covariance"], dtype="f8")
        k = np.asarray(data["k_obs"], dtype="f8")
        edges = np.asarray(data["k_edges"], dtype="f8")
        summary = load_summary_json(data)
    return {"path": path, "covariance": cov, "k_obs": k, "k_edges": edges, "summary": summary}


def find_window_file(files: list[Path], explicit: Path | None) -> Path | None:
    if explicit is not None:
        if not explicit.exists():
            raise FileNotFoundError(explicit)
        return explicit
    for path in files:
        with np.load(path, allow_pickle=False) as data:
            if "window_matrix" in data.files:
                return path
    return None


def read_window(path: Path | None) -> dict[str, np.ndarray] | None:
    if path is None:
        return None
    with np.load(path, allow_pickle=False) as data:
        required = ("window_matrix", "theory_k", "theory_ell")
        missing = [name for name in required if name not in data.files]
        if missing:
            raise KeyError(f"{path} misses window arrays: {missing}")
        out = {
            "window_matrix": np.asarray(data["window_matrix"], dtype="f8"),
            "theory_k": np.asarray(data["theory_k"], dtype="f8"),
            "theory_ell": np.asarray(data["theory_ell"], dtype="i8"),
        }
        for name in (
            "window_observable_k",
            "window_observable_edges",
            "theory_edges",
            "theory_slice_start",
            "theory_slice_stop",
        ):
            if name in data.files:
                out[name] = np.asarray(data[name])
    return out


def fit_mask(k_obs: np.ndarray, k_edges: np.ndarray, pk_stack: np.ndarray, *, kmin_fit: float, kmax_fit: float) -> np.ndarray:
    k_obs = np.asarray(k_obs, dtype="f8")
    k_edges = np.asarray(k_edges, dtype="f8")
    pk_stack = np.asarray(pk_stack, dtype="f8")
    finite = np.isfinite(k_obs) & np.all(np.isfinite(pk_stack), axis=0)
    cut = (k_obs >= float(kmin_fit) - 1.0e-12) & (k_edges[:, 1] <= float(kmax_fit) + 1.0e-12)
    return finite & cut


def parse_float_list(text: str) -> list[float]:
    return [float(item) for item in str(text).split(",") if item.strip()]


def parse_int_list(text: str) -> list[int]:
    return [int(item) for item in str(text).split(",") if item.strip()]


def apply_fit_bin_policy(
    mask: np.ndarray,
    k_obs: np.ndarray,
    *,
    policy: str,
    stride: int,
    pivots: list[float],
    factors: list[int],
) -> tuple[np.ndarray, dict[str, Any]]:
    base = np.asarray(mask, dtype=bool)
    selected = base.copy()
    policy = str(policy)
    if policy == "all":
        pass
    elif policy == "stride":
        if int(stride) < 1:
            raise ValueError("fit-bin-stride must be >= 1")
        idx = np.flatnonzero(selected)
        selected[:] = False
        selected[idx[:: int(stride)]] = True
    elif policy == "desi_png":
        if len(pivots) != len(factors):
            raise ValueError(f"DESI PNG pivots and factors differ: {pivots} vs {factors}")
        for pivot, factor in zip(pivots, factors, strict=True):
            factor = int(factor)
            if factor < 1:
                raise ValueError("DESI PNG factors must be >= 1")
            if factor == 1:
                continue
            idx = np.flatnonzero(selected & (np.asarray(k_obs) >= float(pivot) - 1.0e-12))
            selected[idx] = False
            selected[idx[::factor]] = True
    else:
        raise ValueError(f"unknown fit bin policy {policy!r}")
    if int(np.count_nonzero(selected)) < 3:
        raise RuntimeError(f"too few fit bins after {policy} policy: {int(np.count_nonzero(selected))}")
    return selected, {
        "policy": policy,
        "stride": int(stride),
        "desi_png_pivots": [float(value) for value in pivots],
        "desi_png_factors": [int(value) for value in factors],
        "n_before_policy": int(np.count_nonzero(base)),
        "n_after_policy": int(np.count_nonzero(selected)),
        "selected_indices": [int(value) for value in np.flatnonzero(selected)],
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Summarize Task43 lightcone P(k) measurements.")
    parser.add_argument("--p0", type=float, default=10000.0)
    parser.add_argument("--tag", type=str, default=None)
    parser.add_argument("--measurement-glob", type=str, default=None)
    parser.add_argument("--covariance", type=Path, default=None)
    parser.add_argument("--window-file", type=Path, default=None)
    parser.add_argument("--fkp-summary", type=Path, default=DEFAULT_FKP_SUMMARY)
    parser.add_argument("--output-tag", type=str, default=None)
    parser.add_argument("--fit-kmin", type=float, default=None, help="Observed-bin fit kmin; default is 2pi/Veff^(1/3).")
    parser.add_argument("--kmax-fit", type=float, default=K_MAX_FIT)
    parser.add_argument("--fit-bin-policy", choices=("all", "desi_png", "stride"), default="all")
    parser.add_argument("--fit-bin-stride", type=int, default=2)
    parser.add_argument("--desi-png-pivots", type=str, default="0.01,0.02")
    parser.add_argument("--desi-png-factors", type=str, default="2,2")
    parser.add_argument("--allow-no-window", action="store_true")
    parser.add_argument("--allow-no-covariance", action="store_true")
    parser.add_argument("--allow-scatter-covariance", action="store_true", help="Debug only; not for final Task43 P(k) constraints.")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    ensure_pk_dirs()
    tag = p0_output_tag(float(args.p0), args.tag)
    out_tag = args.output_tag or tag
    payload_path = PK_SUMMARY_DIR / f"task43_pk_lightcone_{out_tag}_payload.npz"
    summary_path = PK_SUMMARY_DIR / f"task43_pk_lightcone_{out_tag}_summary.json"
    if payload_path.exists() and summary_path.exists() and not args.overwrite:
        print(f"[skip] {payload_path}")
        return

    fkp_summary = load_fkp_summary(args.fkp_summary)
    inferred_kmin, kmin_meta = lightcone_fit_kmin_from_fkp_summary(fkp_summary)
    kmin_fit = float(args.fit_kmin) if args.fit_kmin is not None else float(inferred_kmin)

    files = measurement_files(tag, args.measurement_glob)
    measured = read_measurement_stack(files)
    mask = fit_mask(measured["k_obs"], measured["k_edges"], measured["pk_stack"], kmin_fit=kmin_fit, kmax_fit=float(args.kmax_fit))
    if int(np.count_nonzero(mask)) < 3:
        raise RuntimeError(f"too few fit bins after k cuts: {int(np.count_nonzero(mask))}")
    mask, bin_policy_meta = apply_fit_bin_policy(
        mask,
        measured["k_obs"],
        policy=str(args.fit_bin_policy),
        stride=int(args.fit_bin_stride),
        pivots=parse_float_list(args.desi_png_pivots),
        factors=parse_int_list(args.desi_png_factors),
    )

    window_path = find_window_file(files, args.window_file)
    window = read_window(window_path)
    if window is None and not args.allow_no_window:
        raise FileNotFoundError("no window matrix found; rerun measurement with --window-method smooth/exact or pass --allow-no-window")

    cov_path = args.covariance if args.covariance is not None else default_covariance_file(tag)
    cov_data = read_covariance(cov_path)
    if cov_data is None:
        if not args.allow_no_covariance and not args.allow_scatter_covariance:
            raise FileNotFoundError("no jaxpower covariance found; pass --covariance or --allow-no-covariance for measurement-only summary")
        nfit = int(np.count_nonzero(mask))
        if args.allow_scatter_covariance and measured["pk_stack"].shape[0] > 1:
            covariance = np.cov(measured["pk_stack"][:, mask], rowvar=False, ddof=1)
            covariance_source = "phase_scatter_debug"
        else:
            covariance = np.full((nfit, nfit), np.nan, dtype="f8")
            covariance_source = "none_measurement_only"
        cov_path_str = None
    else:
        if not np.allclose(cov_data["k_edges"], measured["k_edges"], rtol=0.0, atol=1.0e-12):
            raise ValueError(f"covariance k_edges do not match measurement k_edges: {cov_data['path']}")
        covariance = np.asarray(cov_data["covariance"], dtype="f8")[np.ix_(mask, mask)]
        covariance_source = "jaxpower_gaussian_survey_window_single_lightcone"
        cov_path_str = str(cov_data["path"])

    pk_stack_fit = measured["pk_stack"][:, mask]
    shot_fit = measured["shotnoise_stack"][:, mask]
    pk_mean = np.mean(pk_stack_fit, axis=0)
    shot_mean = np.mean(shot_fit, axis=0)
    scatter_cov = np.cov(pk_stack_fit, rowvar=False, ddof=1) if pk_stack_fit.shape[0] > 1 else np.full_like(covariance, np.nan)
    if window is not None:
        wmat = np.asarray(window["window_matrix"], dtype="f8")
        if wmat.shape[0] != measured["k_obs"].size:
            raise ValueError(f"window rows={wmat.shape[0]} but measured bins={measured['k_obs'].size}")
        window_fit = wmat[mask, :]
        theory_k = np.asarray(window["theory_k"], dtype="f8")
        theory_ell = np.asarray(window["theory_ell"], dtype="i8")
        theory_edges = np.asarray(window.get("theory_edges", np.empty((0, 2))), dtype="f8")
        theory_slice_start = np.asarray(window.get("theory_slice_start", np.empty(0)), dtype="i8")
        theory_slice_stop = np.asarray(window.get("theory_slice_stop", np.empty(0)), dtype="i8")
    else:
        window_fit = np.empty((pk_mean.size, 0), dtype="f8")
        theory_k = np.empty(0, dtype="f8")
        theory_ell = np.empty(0, dtype="i8")
        theory_edges = np.empty((0, 2), dtype="f8")
        theory_slice_start = np.empty(0, dtype="i8")
        theory_slice_stop = np.empty(0, dtype="i8")

    p0_values = np.asarray(fkp_summary["p0_values"], dtype="f8")
    ip0 = np.flatnonzero(np.isclose(p0_values, float(args.p0), rtol=0.0, atol=1.0e-10))
    zeff = float(np.asarray(fkp_summary["zeff_random_auto"], dtype="f8")[int(ip0[0])]) if ip0.size == 1 else np.nan
    summary = {
        "task": "task43_summarize_pk_lightcone",
        "status": "done",
        "tag": tag,
        "output_tag": out_tag,
        "p0": float(args.p0),
        "zeff": float(zeff),
        "nphase": int(pk_stack_fit.shape[0]),
        "phases": [str(x) for x in measured["phases"].tolist()],
        "measurement_files": [str(path) for path in files],
        "fit_bins": {
            "ndata": int(np.count_nonzero(mask)),
            "k_min_data": float(np.min(measured["k_obs"][mask])),
            "k_max_data": float(np.max(measured["k_obs"][mask])),
            "kmin_fit_observed": float(kmin_fit),
            "kmin_fit_policy": "2pi/Veff^(1/3) from Task43 lightcone volume_shell unless overridden",
            "kmax_fit": float(args.kmax_fit),
            "selection": "finite observed bins with k_obs >= kmin_fit_observed and k_high <= kmax_fit, followed by fit_bin_policy",
            "bin_policy": bin_policy_meta,
        },
        "lightcone_effective_volume": kmin_meta,
        "window": {
            "path": None if window_path is None else str(window_path),
            "matrix_shape_fit": [int(v) for v in window_fit.shape],
            "theory_size": int(theory_k.size),
            "theory_k_min": None if theory_k.size == 0 else float(np.min(theory_k)),
            "theory_k_max": None if theory_k.size == 0 else float(np.max(theory_k)),
            "policy": "geometry-only jaxpower window; no RIC, no AMR, no GIC; observed-bin fit kmin is not applied to theory input",
        },
        "covariance": {
            "path": cov_path_str,
            "source": covariance_source,
            "used_for_fit": bool(covariance_source.startswith("jaxpower") or args.allow_scatter_covariance),
            "not_divided_by_nphase": True,
            "diagnostics": covariance_diagnostics(covariance) if np.all(np.isfinite(covariance)) else None,
            "scatter_covariance_diagnostic_only": True,
        },
        "shotnoise": {
            "pk0_convention": "Mesh2SpectrumPole.value(), i.e. shot-noise-subtracted P0",
            "forward_model": "W(P_theory), with free residual sn0*1e4 added to theory monopole; jaxpower covariance carries Poisson terms through WS/SW/SS.",
            "shotnoise_mean_scalar_fit": float(np.mean(shot_mean)),
            "shotnoise_mean_min_fit": float(np.min(shot_mean)),
            "shotnoise_mean_max_fit": float(np.max(shot_mean)),
            "sn0_scale": float(SN0_SCALE),
            "fit_sn0_policy_main": "free, matching Task4.2 P(k); fixed sn0=0 is not used for the current Task43 P(k) result",
        },
        "paths": {"payload_npz": str(payload_path), "summary_json": str(summary_path)},
    }
    atomic_savez(
        payload_path,
        k_obs=np.asarray(measured["k_obs"][mask], dtype="f8"),
        k_edges=np.asarray(measured["k_edges"][mask], dtype="f8"),
        pk_mean=np.asarray(pk_mean, dtype="f8"),
        pk_stack=np.asarray(pk_stack_fit, dtype="f8"),
        pk_scatter_cov=np.asarray(scatter_cov, dtype="f8"),
        covariance=np.asarray(covariance, dtype="f8"),
        shotnoise_mean=np.asarray(shot_mean, dtype="f8"),
        shotnoise_stack=np.asarray(shot_fit, dtype="f8"),
        shotnoise_mean_scalar=np.asarray(float(np.mean(shot_mean)), dtype="f8"),
        fit_bin_mask=np.asarray(mask, dtype=bool),
        fit_bin_indices=np.asarray(np.flatnonzero(mask), dtype="i8"),
        kmin_fit_observed=np.asarray(float(kmin_fit), dtype="f8"),
        kmax_fit=np.asarray(float(args.kmax_fit), dtype="f8"),
        lightcone_volume_eff=np.asarray(float(kmin_meta["volume_eff"]), dtype="f8"),
        lightcone_leff_volume=np.asarray(float(kmin_meta["leff_volume"]), dtype="f8"),
        window_matrix=np.asarray(window_fit, dtype="f8"),
        theory_k=np.asarray(theory_k, dtype="f8"),
        theory_ell=np.asarray(theory_ell, dtype="i8"),
        theory_edges=np.asarray(theory_edges, dtype="f8"),
        theory_slice_start=np.asarray(theory_slice_start, dtype="i8"),
        theory_slice_stop=np.asarray(theory_slice_stop, dtype="i8"),
        phases=np.asarray(measured["phases"]),
        measurement_files=np.asarray([str(path) for path in files]),
        covariance_file=np.asarray("" if cov_path_str is None else cov_path_str),
        window_file=np.asarray("" if window_path is None else str(window_path)),
        p0=np.asarray(float(args.p0), dtype="f8"),
        zeff=np.asarray(float(zeff), dtype="f8"),
        sn0_scale=np.asarray(float(SN0_SCALE), dtype="f8"),
        summary_json=np.asarray(json.dumps(to_jsonable(summary), sort_keys=True)),
    )
    write_json(summary_path, summary)
    print(f"[write] {payload_path}")
    print(f"[write] {summary_path}")


if __name__ == "__main__":
    main()
