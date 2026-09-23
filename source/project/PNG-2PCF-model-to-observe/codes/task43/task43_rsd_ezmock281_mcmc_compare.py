#!/usr/bin/env python3
"""Task43 RSD lightcone: EZmock-281 versus analytic-jaxpower MCMC.

The data vector and forward model follow the audited Task43 lightcone RSD
contract: P0 has 13 bins, P2 keeps bins 4..12 (9 bins), and both xi0 and xi2
use 50 <= s < 350 with 80 <= s < 120 removed (26 bins per pole).
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
from scipy.optimize import least_squares

PROJECT_ROOT = Path("/pscratch/sd/l/lzy/PNG-2PCF-model-to-observe")
CODE_DIR = PROJECT_ROOT / "codes/task43"
sys.path.insert(0, str(CODE_DIR))

from task43_joint_rsd_pkxi_fit import (  # noqa: E402
    BOUNDS_HI,
    BOUNDS_LO,
    JOINT_PARAMS,
    OPTIMIZER_STARTS,
    run_chain,
)
from task43_rsd_joint_p02xi02_fit import (  # noqa: E402
    ELL2_SUMMARY,
    MEAN_WINDOW_NPZ,
    MEASURE_DIR,
    PAYLOAD_NPZ,
    WindowConvolvedP02Model,
    FastWindowRSDModel,
    FullDiscreteRSDModel,
    build_cache,
    load_window,
)
from task43_fit_rsd_rawbox_pk0_vs_xi0_smin50 import LIGHTCONE_FIT_EDGES  # noqa: E402

BASE = PROJECT_ROOT / "outputs/task43_outputs/ezmock_rsd_lightcone_covariance_z0p4_0p8_x1000_common50"
MANIFEST = BASE / "manifests/task43_ezmock_rsd_covariance_x1000_fixampF_common50.jsonl"
ANALYTIC_COV = PROJECT_ROOT / "outputs/task43_outputs/rsd_validation/lightcone/kmax0p08_smin50_v1/task43_lightcone_standard_joint_baomask80_120_covariance_v1.npz"
OUT_ROOT = BASE / "mcmc_preliminary_281"

P_EDGES = np.asarray(LIGHTCONE_FIT_EDGES[:13], dtype="f8")
P2_KEEP = np.arange(4, 13, dtype="i8")
XI_CENTERS = np.arange(35.0, 350.0, 10.0)
XI_MASK = (XI_CENTERS >= 50.0) & ~((XI_CENTERS >= 80.0) & (XI_CENTERS < 120.0))
N_P0, N_P2, N_XI = 13, 9, int(np.count_nonzero(XI_MASK))
N_DATA = N_P0 + N_P2 + 2 * N_XI
RR_SHA = "61e6eb2bf493d0a2b8042b065c2434b94b7ec6bc2b503e9dc9e6ea5e6274d574"

# gaussian_kde cost scales as n_samples x n_grid_points; the saved chains hold
# ~1e6 samples each, so render contours from a random subsample.
KDE_MAX_SAMPLES = 60000
KDE_GRID = 140


def jsonable(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [jsonable(v) for v in value]
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    return value


def read_manifest() -> list[dict[str, Any]]:
    return [json.loads(line) for line in MANIFEST.read_text().splitlines() if line.strip()]


def matching_edges(source: np.ndarray, target: np.ndarray) -> np.ndarray:
    out = []
    for edge in target:
        found = np.flatnonzero(np.all(np.isclose(source, edge[None, :], rtol=0.0, atol=1e-12), axis=1))
        if found.size != 1:
            raise RuntimeError(f"target edge {edge.tolist()} has {found.size} matches")
        out.append(int(found[0]))
    return np.asarray(out, dtype="i8")


def load_ezmock_stack(nmock: int) -> tuple[np.ndarray, list[dict[str, Any]]]:
    rows = sorted(read_manifest(), key=lambda row: int(row["production_index"]))
    selected: list[dict[str, Any]] = []
    for row in rows:
        if int(row["production_index"]) >= 1000 or row.get("fix_amplitude") is not False:
            continue
        xp, pp = Path(row["xi_path"]), Path(row["pk_path"])
        xm, pm = xp.with_suffix(".json"), pp.with_suffix(".json")
        if not all(p.is_file() and p.stat().st_size > 0 for p in (xp, pp, xm, pm)):
            continue
        try:
            xmeta, pmeta = json.loads(xm.read_text()), json.loads(pm.read_text())
            if xmeta.get("status") != "done" or pmeta.get("status") != "done":
                continue
            if xmeta.get("fix_amplitude") is not False or pmeta.get("fix_amplitude") is not False:
                continue
            if xmeta.get("ells") != [0, 2] or int(xmeta.get("mu_bin_num", -1)) != 120:
                continue
            if xmeta.get("common_rr_smu_sha256") != RR_SHA:
                continue
            with np.load(pp, allow_pickle=False) as p, np.load(xp, allow_pickle=False) as x:
                i0 = matching_edges(np.asarray(p["k_edges0"], dtype="f8"), P_EDGES)
                i2 = matching_edges(np.asarray(p["k_edges2"], dtype="f8"), P_EDGES)
                p0 = np.asarray(p["pk0"], dtype="f8")[i0]
                p2 = np.asarray(p["pk2"], dtype="f8")[i2][P2_KEEP]
                xi0 = np.asarray(x["xi0"], dtype="f8")
                xi2 = np.asarray(x["xi2"], dtype="f8")
                if xi0.shape != (32,) or xi2.shape != (32,):
                    continue
                vector = np.concatenate([p0, p2, xi0[XI_MASK], xi2[XI_MASK]])
                if not np.all(np.isfinite(vector)):
                    continue
            selected.append({"row": row, "vector": vector})
        except Exception:
            continue
        if len(selected) >= int(nmock):
            break
    if len(selected) < int(nmock):
        raise RuntimeError(f"only {len(selected)} valid EZmock vectors; need {nmock}")
    stack = np.stack([item["vector"] for item in selected[:nmock]])
    return stack, [item["row"] for item in selected[:nmock]]


def correction_factors(ns: int, nb: int, npfit: int) -> dict[str, float]:
    hartlap = (ns - nb - 2.0) / (ns - 1.0)
    A = 2.0 / ((ns - nb - 1.0) * (ns - nb - 4.0))
    B = (ns - nb - 2.0) / ((ns - nb - 1.0) * (ns - nb - 4.0))
    m1 = (1.0 + B * (nb - npfit)) / (1.0 + A + B * (npfit + 1.0))
    return {"nmock": float(ns), "ndata": float(nb), "nparams": float(npfit), "hartlap": hartlap, "A": A, "B": B, "percival_m1": m1, "percival_sigma_factor": float(np.sqrt(m1))}


def precision_from_cov(cov: np.ndarray, hartlap: float = 1.0) -> tuple[np.ndarray, dict[str, Any]]:
    cov = 0.5 * (np.asarray(cov, dtype="f8") + np.asarray(cov, dtype="f8").T)
    scale = np.sqrt(np.clip(np.diag(cov), 1e-300, None))
    corr = cov / np.outer(scale, scale)
    evals, evecs = np.linalg.eigh(0.5 * (corr + corr.T))
    floor = 1e-12
    n_floor = int(np.count_nonzero(evals < floor))
    inv_corr = (evecs * (1.0 / np.maximum(evals, floor))[None, :]) @ evecs.T
    precision = hartlap * inv_corr / np.outer(scale, scale)
    return precision, {"corr_eigen_min": float(evals[0]), "corr_eigen_max": float(evals[-1]), "n_eigen_floor": n_floor, "hartlap": float(hartlap)}


def map_fit(evaluate, data: np.ndarray, precision: np.ndarray, nparams: int) -> np.ndarray:
    vals, vecs = np.linalg.eigh(0.5 * (precision + precision.T))
    root = (vecs * np.sqrt(np.maximum(vals, 0.0))[None, :]) @ vecs.T
    def residual(theta):
        return root @ (data - evaluate(np.asarray(theta, dtype="f8")))
    sol = [least_squares(residual, start[:nparams], bounds=(BOUNDS_LO[:nparams], BOUNDS_HI[:nparams]), max_nfev=3000) for start in OPTIMIZER_STARTS]
    return np.asarray(min(sol, key=lambda s: float(s.fun @ s.fun)).x, dtype="f8")


def corrected_summary(summary: dict[str, Any], m1: float) -> dict[str, Any]:
    out = json.loads(json.dumps(summary))
    fac = float(np.sqrt(m1))
    for name, item in out.get("posterior", {}).items():
        q50 = float(item["q50"])
        item["q16_raw"] = float(item["q16"])
        item["q84_raw"] = float(item["q84"])
        item["sigma68_raw"] = float(item["sigma68"])
        item["q16"] = q50 + fac * (float(item["q16"]) - q50)
        item["q84"] = q50 + fac * (float(item["q84"]) - q50)
        item["sigma68"] = fac * float(item["sigma68"])
        item["std"] = fac * float(item["std"])
    out["percival_m1"] = float(m1)
    out["percival_sigma_factor"] = fac
    return out


def build_models_and_data() -> tuple[Any, Any, np.ndarray, np.ndarray, Any, Any]:
    # Reproduce the audited analytic-jaxpower data/model setup, but apply the
    # exact kmax=0.08 and common xi BAO mask used by the 4.3 main result.
    with np.load(PAYLOAD_NPZ, allow_pickle=False) as d:
        payload = {key: np.asarray(d[key]) for key in d.files}
    with np.load(MEASURE_DIR / "task43_rsd_lightcone_p02_ph000_mesh256_kmax0p300_dk0p002.npz", allow_pickle=False) as d:
        window_rows = np.asarray(d["window_matrix"], dtype="f8")
        theory_k = np.asarray(d["theory_k"], dtype="f8")
        theory_ell = np.asarray(d["theory_ell"], dtype="i8")
    # Payload selection is the canonical 15-bin edge list; first 13 are kmax=0.08.
    fit_indices = np.asarray(payload["fit_bin_indices"], dtype="i8")
    rows13 = fit_indices[:13]
    window = window_rows[np.concatenate([rows13, 150 + rows13]), :]
    zeff = float(np.asarray(payload["zeff"]).item())
    pmodel = WindowConvolvedP02Model(window, theory_k, theory_ell, zeff=zeff)
    with np.load(ELL2_SUMMARY, allow_pickle=False) as d:
        mean_xi = np.asarray(d["xi_multipoles_mean"], dtype="f8")
    phase_pk = []
    for phase in sorted(MEASURE_DIR.glob("task43_rsd_lightcone_p02_*_mesh256_kmax0p300_dk0p002.npz")):
        with np.load(phase, allow_pickle=False) as d:
            phase_pk.append(np.concatenate([np.asarray(d["pk0"], dtype="f8")[fit_indices], np.asarray(d["pk2"], dtype="f8")[fit_indices]]))
    mean_pk = np.mean(np.stack(phase_pk), axis=0)
    data_p = np.concatenate([mean_pk[:13], mean_pk[15:15 + 13][P2_KEEP]])
    zeff_xi = float(np.mean(np.asarray(np.load(ELL2_SUMMARY, allow_pickle=False)["zeff_by_phase"])))
    cache = build_cache(zeff=zeff_xi, boxsize=2000.0, kmax=5.0, ells=(0, 2), cosmology="abacus_c000")
    xmodel = FastWindowRSDModel(FullDiscreteRSDModel(cache, nmu=64), {"mean": load_window(MEAN_WINDOW_NPZ)}, sigma_step=0.05)
    data_x = np.concatenate([mean_xi[0][XI_MASK], mean_xi[1][XI_MASK]])
    return pmodel, xmodel, data_p, data_x, payload, mean_xi


def main() -> None:
    global BASE, MANIFEST, OUT_ROOT
    ap = argparse.ArgumentParser()
    ap.add_argument("--nmock", type=int, default=281)
    ap.add_argument("--nwalkers", type=int, default=64)
    ap.add_argument("--nsteps", type=int, default=30000)
    ap.add_argument("--burnin", type=int, default=5000)
    ap.add_argument("--seed", type=int, default=20260915)
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--contour-only", action="store_true", help="re-render the contour PDF from saved chains, no MCMC")
    ap.add_argument("--base", type=Path, default=BASE)
    ap.add_argument("--manifest", type=Path, default=MANIFEST)
    ap.add_argument("--out-root", type=Path, default=OUT_ROOT)
    args = ap.parse_args()
    BASE, MANIFEST, OUT_ROOT = args.base, args.manifest, args.out_root
    ezlabel = f"ezmock{int(args.nmock)}"
    if args.contour_only:
        make_contours(json.loads((OUT_ROOT / "comparison_summary.json").read_text(encoding="utf-8")))
        return
    nsteps, burnin = (500, 100) if args.smoke else (int(args.nsteps), int(args.burnin))
    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    stack, frozen_rows = load_ezmock_stack(int(args.nmock))
    cov_ez = np.cov(stack, rowvar=False, ddof=1)
    np.savez_compressed(OUT_ROOT / f"{ezlabel}_covariance_and_stack.npz", stack=stack, covariance=cov_ez, production_indices=np.asarray([r["production_index"] for r in frozen_rows]))
    pmodel, xmodel, data_p, data_x, payload, mean_xi = build_models_and_data()
    data_joint = np.concatenate([data_p, data_x])
    dims = {"p02": 22, "xi02": 2 * N_XI, "joint": N_DATA}
    with np.load(ANALYTIC_COV, allow_pickle=False) as d:
        cov_an = np.asarray(d["rsd_joint"], dtype="f8")
    if cov_an.shape != (N_DATA, N_DATA):
        raise RuntimeError(f"analytic covariance shape {cov_an.shape} != {(N_DATA, N_DATA)}")
    covs = {
        ezlabel: cov_ez,
        "jaxpower": cov_an,
    }
    results: dict[str, Any] = {}
    for cov_name, cov in covs.items():
        blocks = {"p02": cov[:22, :22], "xi02": cov[22:, 22:], "joint": cov}
        for variant in ("p02", "xi02", "joint"):
            nparams = 3 if variant == "xi02" else 4
            data = data_p if variant == "p02" else data_x if variant == "xi02" else data_joint
            if variant == "p02":
                evaluate = lambda t, pm=pmodel: np.concatenate([pm.evaluate(np.asarray(t))[:13], pm.evaluate(np.asarray(t))[13:26][P2_KEEP]])
            elif variant == "xi02":
                evaluate = lambda t, xm=xmodel: np.concatenate([xm.evaluate(np.asarray(t)[:3], model="formal_gic", window_key="mean")[0][XI_MASK], xm.evaluate(np.asarray(t)[:3], model="formal_gic", window_key="mean")[2][XI_MASK]])
            else:
                evaluate = lambda t, pm=pmodel, xm=xmodel: np.concatenate([pm.evaluate(np.asarray(t))[:13], pm.evaluate(np.asarray(t))[13:26][P2_KEEP], xm.evaluate(np.asarray(t)[:3], model="formal_gic", window_key="mean")[0][XI_MASK], xm.evaluate(np.asarray(t)[:3], model="formal_gic", window_key="mean")[2][XI_MASK]])
            corr = correction_factors(int(args.nmock), dims[variant], nparams) if cov_name == ezlabel else {"hartlap": 1.0, "percival_m1": 1.0, "percival_sigma_factor": 1.0, "ndata": dims[variant], "nparams": nparams, "nmock": 0}
            precision, pmeta = precision_from_cov(blocks[variant], float(corr["hartlap"]))
            theta_map = map_fit(evaluate, data, precision, nparams)
            summary, chain, logp = run_chain(evaluate, data, precision, theta_map, nparams, nwalkers=int(args.nwalkers), nsteps=nsteps, burnin=burnin, seed=int(args.seed) + hash((cov_name, variant)) % 100000)
            summary["posterior_raw"] = summary.get("posterior")
            summary = corrected_summary(summary, float(corr["percival_m1"]))
            summary.update({"covariance": cov_name, "variant": variant, "ndata": dims[variant], "nparams": nparams, "hartlap_percival": corr, "precision_meta": pmeta, "map_theta": theta_map.tolist(), "data_contract": {"p0": N_P0, "p2": N_P2, "xi0": N_XI, "xi2": N_XI}})
            tag = f"{cov_name}_{variant}"
            np.savez_compressed(OUT_ROOT / f"chain_{tag}.npz", chain=chain, logp=logp, data=data)
            (OUT_ROOT / f"summary_{tag}.json").write_text(json.dumps(jsonable(summary), indent=2, sort_keys=True))
            results[tag] = summary
            print(json.dumps({"tag": tag, "fNL": summary["posterior"]["fNL"], "gates": summary["gates"]}, sort_keys=True), flush=True)
    (OUT_ROOT / "comparison_summary.json").write_text(json.dumps(jsonable(results), indent=2, sort_keys=True))
    if not args.smoke:
        make_contours(results)


def make_contours(results: dict[str, Any]) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from scipy.stats import gaussian_kde
    rng = np.random.default_rng(20260917)
    fig, ax = plt.subplots(figsize=(7.5, 6.5))
    ez_tag = next((t for t in results if t.startswith("ezmock") and t.endswith("_p02")), "ezmock_p02")
    ezlabel = ez_tag[: -len("_p02")]
    variant_labels = {"p02": "P0+P2", "xi02": "ξ0+ξ2", "joint": "joint"}
    styles = {}
    for cov_name, colors in ((ezlabel, ("#1f77b4", "#2ca02c", "#d62728")), ("jaxpower", ("#6baed6", "#74c476", "#fb6a4a"))):
        for (variant, vlabel), color in zip(variant_labels.items(), colors):
            styles[f"{cov_name}_{variant}"] = (color, f"{vlabel} · {cov_name}")
    for tag, (color, label) in styles.items():
        path = OUT_ROOT / f"chain_{tag}.npz"
        if not path.is_file():
            continue
        with np.load(path, allow_pickle=False) as d:
            chain = np.asarray(d["chain"], dtype="f8").reshape(-1, d["chain"].shape[-1])
        if chain.shape[0] > KDE_MAX_SAMPLES:
            sel = rng.choice(chain.shape[0], size=KDE_MAX_SAMPLES, replace=False)
            chain = chain[sel]
        x, y = chain[:, 0], chain[:, 1]
        kde = gaussian_kde(np.vstack([x, y]))
        xx = np.linspace(np.percentile(x, 0.2), np.percentile(x, 99.8), KDE_GRID)
        yy = np.linspace(np.percentile(y, 0.2), np.percentile(y, 99.8), KDE_GRID)
        X, Y = np.meshgrid(xx, yy)
        Z = kde(np.vstack([X.ravel(), Y.ravel()])).reshape(X.shape)
        levels = np.sort([np.quantile(Z, 0.50), np.quantile(Z, 0.10)])
        ax.contour(X, Y, Z, levels=levels, colors=[color], linewidths=[1.2, 2.0], alpha=0.85)
        ax.plot(np.median(x), np.median(y), marker="o", ms=3.5, color=color, label=label)
    ax.set(xlabel=r"$f_{\rm NL}$", ylabel=r"$b_1$", title="Task43 RSD lightcone: covariance comparison")
    ax.legend(fontsize=8, loc="best", frameon=False)
    ax.grid(alpha=0.18)
    fig.tight_layout()
    fig.savefig(OUT_ROOT / f"fNL_b1_contours_{ezlabel}_vs_jaxpower.pdf")
    plt.close(fig)


if __name__ == "__main__":
    main()
