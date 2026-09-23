#!/usr/bin/env python3
"""Aggregate the 25 Task43 raw-box P0(k) and xi0(s) measurements."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from task43_rawbox_ezmock_common import PHASES, SUMMARY_DIR, atomic_savez, pk_path, write_json, xi_path


def correlation(covariance: np.ndarray) -> np.ndarray:
    sigma = np.sqrt(np.clip(np.diag(covariance), 0.0, np.inf))
    denominator = np.outer(sigma, sigma)
    result = np.zeros_like(covariance)
    np.divide(covariance, denominator, out=result, where=denominator > 0.0)
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--meshsize", type=int, default=400)
    parser.add_argument("--allow-partial", action="store_true")
    parser.add_argument("--output-prefix", type=Path, default=SUMMARY_DIR / "task43_ezmock_rawbox_z0p725_mmin1p4e13_x25")
    args = parser.parse_args()

    phases = [phase for phase in PHASES if xi_path(phase).exists() and pk_path(phase, args.meshsize).exists()]
    missing = [phase for phase in PHASES if phase not in phases]
    if missing and not args.allow_partial:
        raise FileNotFoundError(f"missing xi or Pk measurements for: {missing}")
    if not phases:
        raise RuntimeError("no complete phase measurements found")

    xi_rows, pk_rows, nmodes_rows, ndata, nbar, redshift = [], [], [], [], [], []
    s = s_edges = k = k_edges = None
    for phase in phases:
        with np.load(xi_path(phase), allow_pickle=False) as data:
            this_s = np.asarray(data["s"], dtype="f8")
            this_s_edges = np.asarray(data["s_edges"], dtype="f8")
            if s is None:
                s, s_edges = this_s, this_s_edges
            elif not np.array_equal(s, this_s) or not np.array_equal(s_edges, this_s_edges):
                raise ValueError(f"inconsistent xi grid for {phase}")
            xi_rows.append(np.asarray(data["xi0"], dtype="f8"))
            ndata.append(int(np.asarray(data["ndata"]).item()))
            nbar.append(float(np.asarray(data["nbar"]).item()))
            redshift.append(float(np.asarray(data["redshift"]).item()))
        with np.load(pk_path(phase, args.meshsize), allow_pickle=False) as data:
            this_k = np.asarray(data["k"], dtype="f8")
            this_k_edges = np.asarray(data["k_edges"], dtype="f8")
            if k is None:
                k, k_edges = this_k, this_k_edges
            elif not np.array_equal(k, this_k, equal_nan=True) or not np.array_equal(k_edges, this_k_edges, equal_nan=True):
                raise ValueError(f"inconsistent Pk grid for {phase}")
            pk_rows.append(np.asarray(data["pk0"], dtype="f8"))
            nmodes_rows.append(np.asarray(data["nmodes"], dtype="f8"))

    xi_all = np.vstack(xi_rows)
    pk_all = np.vstack(pk_rows)
    nmodes_all = np.vstack(nmodes_rows)
    valid_k = np.all(nmodes_all > 0, axis=0) & np.all(np.isfinite(pk_all), axis=0)
    nreal = len(phases)
    xi_cov = np.cov(xi_all, rowvar=False, ddof=1) if nreal > 1 else np.zeros((xi_all.shape[1], xi_all.shape[1]))
    pk_cov = np.cov(pk_all[:, valid_k], rowvar=False, ddof=1) if nreal > 1 else np.zeros((np.count_nonzero(valid_k), np.count_nonzero(valid_k)))
    pk_mean = np.full(pk_all.shape[1], np.nan, dtype="f8")
    pk_std = np.full(pk_all.shape[1], np.nan, dtype="f8")
    pk_mean[valid_k] = np.mean(pk_all[:, valid_k], axis=0)
    pk_std[valid_k] = np.std(pk_all[:, valid_k], axis=0, ddof=1) if nreal > 1 else 0.0
    output_npz = args.output_prefix.with_suffix(".npz")
    output_json = args.output_prefix.with_suffix(".json")
    output_xi_csv = args.output_prefix.with_name(args.output_prefix.name + "_xi.csv")
    output_pk_csv = args.output_prefix.with_name(args.output_prefix.name + "_pk.csv")
    output_phase_csv = args.output_prefix.with_name(args.output_prefix.name + "_phases.csv")
    atomic_savez(
        output_npz,
        phases=np.asarray(phases),
        s=s,
        s_edges=s_edges,
        xi0_all=xi_all,
        xi0_mean=np.mean(xi_all, axis=0),
        xi0_std=np.std(xi_all, axis=0, ddof=1) if nreal > 1 else np.zeros(xi_all.shape[1]),
        xi0_covariance_single_box=xi_cov,
        xi0_correlation_single_box=correlation(xi_cov),
        k=k,
        k_edges=k_edges,
        k_valid_mask=valid_k,
        pk0_all=pk_all,
        nmodes_all=nmodes_all,
        pk0_mean=pk_mean,
        pk0_std=pk_std,
        pk0_covariance_single_box_valid=pk_cov,
        pk0_correlation_single_box_valid=correlation(pk_cov),
        ndata=np.asarray(ndata, dtype="i8"),
        nbar=np.asarray(nbar, dtype="f8"),
        redshift=np.asarray(redshift, dtype="f8"),
        nreal=np.asarray(nreal, dtype="i8"),
    )
    np.savetxt(
        output_xi_csv,
        np.column_stack([s, s_edges[:-1], s_edges[1:], np.mean(xi_all, axis=0), np.std(xi_all, axis=0, ddof=1) if nreal > 1 else 0.0]),
        delimiter=",",
        header="s,s_lower,s_upper,xi0_mean,xi0_std_single_box",
        comments="",
    )
    np.savetxt(
        output_pk_csv,
        np.column_stack([
            k[valid_k],
            k_edges[valid_k, 0],
            k_edges[valid_k, 1],
            np.mean(nmodes_all[:, valid_k], axis=0),
            pk_mean[valid_k],
            pk_std[valid_k],
        ]),
        delimiter=",",
        header="k,k_lower,k_upper,nmodes,pk0_mean,pk0_std_single_box",
        comments="",
    )
    with output_phase_csv.open("w", encoding="utf-8") as stream:
        stream.write("phase,ndata,nbar,redshift\n")
        for phase, count, density, zvalue in zip(phases, ndata, nbar, redshift, strict=True):
            stream.write(f"{phase},{count},{density:.17g},{zvalue:.17g}\n")
    summary = {
        "task": "task43_summarize_ezmock_rawbox_clustering",
        "status": "done",
        "output_npz": str(output_npz),
        "output_xi_csv": str(output_xi_csv),
        "output_pk_csv": str(output_pk_csv),
        "output_phase_csv": str(output_phase_csv),
        "nreal": nreal,
        "phases": phases,
        "missing_phases": missing,
        "meshsize": int(args.meshsize),
        "xi_nbins": int(xi_all.shape[1]),
        "pk_nbins": int(pk_all.shape[1]),
        "pk_valid_nbins": int(np.count_nonzero(valid_k)),
        "ndata_min": int(np.min(ndata)),
        "ndata_max": int(np.max(ndata)),
        "nbar_mean": float(np.mean(nbar)),
        "nbar_std": float(np.std(nbar, ddof=1)) if nreal > 1 else 0.0,
        "redshift_mean": float(np.mean(redshift)),
        "xi_engine": "FCFC_2PT_BOX",
        "pk_engine": "jaxpower CPU",
    }
    write_json(output_json, summary)
    print(f"[done] summary nreal={nreal} output={output_npz}", flush=True)


if __name__ == "__main__":
    main()
