#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""No-BAO-mask xi0+xi2-only lightcone fit for original vs hybrid GSM."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys
from typing import Any

for _name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_name, "1")

import emcee
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.colors import to_rgba
from matplotlib.ticker import MaxNLocator
import numpy as np
from scipy.ndimage import gaussian_filter
from scipy.interpolate import RegularGridInterpolator

if not hasattr(np, "trapezoid"):
    np.trapezoid = np.trapz  # type: ignore[attr-defined]

PROJECT_ROOT = Path("/pscratch/sd/l/lzy/PNG-2PCF-model-to-observe")
TASK43_DIR = PROJECT_ROOT / "codes" / "task43"
TASK432_DIR = PROJECT_ROOT / "codes" / "task432"
OUT_ROOT = PROJECT_ROOT / "outputs" / "task43_outputs" / "rsd_validation" / "task432_model_repair" / "lightcone_no_bao_2pcf"
MEETING_ROOT = PROJECT_ROOT / "9.22meeting" / "task432_lightcone_png_comparison"
PLOT_PATH = MEETING_ROOT / "task432_lightcone_no_bao_2pcf_9.22style.pdf"

for path in (TASK43_DIR, TASK432_DIR):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from task43_fit_rsd_lightcone_x25 import FastWindowRSDModel, load_window  # noqa: E402
from task43_rsd_joint_p02xi02_fit import ELL02_DECONV_NPZ, MEAN_WINDOW_NPZ, RAW_NPZ  # noqa: E402
from task43_rsd_model import FullDiscreteRSDModel, build_cache  # noqa: E402
from task432_lightcone_png_velocileptors import LightconePNGVelocileptors, RSD_P_PAYLOAD, RSD_X_SUMMARY  # noqa: E402
from task43_run_lightcone_joint_baomask_v1 import ScaledGaussianMetric  # noqa: E402


def jsonable(value: Any) -> Any:
    if isinstance(value, dict): return {str(k): jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)): return [jsonable(v) for v in value]
    if isinstance(value, np.ndarray): return value.tolist()
    if isinstance(value, np.generic): return value.item()
    if isinstance(value, Path): return str(value)
    return value


def atomic_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    tmp.write_text(json.dumps(jsonable(payload), indent=2, sort_keys=True) + "\n", encoding="utf-8")
    tmp.replace(path)


def contour_levels(hist: np.ndarray) -> tuple[float, float]:
    values = np.sort(hist.ravel())[::-1]
    cumulative = np.cumsum(values) / np.sum(values)
    return float(values[np.searchsorted(cumulative, .95)]), float(values[np.searchsorted(cumulative, .68)])


def draw_contour(ax, x, y, color, xlim, ylim, zorder):
    hist, xe, ye = np.histogram2d(x, y, bins=(150, 140), range=(xlim, ylim))
    hist = gaussian_filter(hist.astype("f8"), sigma=2.5, mode="nearest")
    l95, l68 = contour_levels(hist); top = hist.max() * 1.001
    xc, yc = .5 * (xe[:-1] + xe[1:]), .5 * (ye[:-1] + ye[1:])
    ax.contourf(xc, yc, hist.T, levels=[l95, l68, top], colors=[to_rgba(color,.10), to_rgba(color,.22)], zorder=zorder)
    ax.contour(xc, yc, hist.T, levels=[l95, l68], colors=color, linewidths=[1.2,1.8], zorder=zorder+1)


def load_data() -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, float, Path]:
    with np.load(RSD_X_SUMMARY, allow_pickle=False) as payload:
        s = np.asarray(payload["s"], dtype="f8")
        phase = np.asarray(payload["xi_multipoles_by_phase"], dtype="f8")
        zeff = float(np.asarray(payload["zeff_mean"]).item())
    mask = (s >= 50.0) & (s < 350.0)
    data = np.concatenate((np.mean(phase[:, 0, mask], axis=0), np.mean(phase[:, 1, mask], axis=0)))
    with np.load(ELL02_DECONV_NPZ, allow_pickle=False) as payload:
        cov_full = np.asarray(payload["covariance_single_realization"], dtype="f8")
    ids = np.concatenate((np.flatnonzero(mask), s.size + np.flatnonzero(mask)))
    return s, mask, data, cov_full[np.ix_(ids, ids)], zeff, ids


def build_models(s: np.ndarray, mask: np.ndarray, zeff: float) -> tuple[Any, Any, Any]:
    cache = build_cache(zeff=zeff, boxsize=2000.0, kmax=5.0, ells=(0,2), cosmology="abacus_c000")
    with np.load(cache, allow_pickle=False) as payload:
        k = np.asarray(payload["k_eff"], dtype="f8")
        pk = np.asarray(payload["pk_dd"], dtype="f8")
        alpha = np.asarray(payload["alpha"], dtype="f8")
        growth = float(np.asarray(payload["f_growth"]).item())
    original = FastWindowRSDModel(FullDiscreteRSDModel(cache, nmu=64), {"mean": load_window(MEAN_WINDOW_NPZ)}, sigma_step=0.05)
    hybrid = LightconePNGVelocileptors(k, pk, alpha, growth, s)
    return original, hybrid, cache


def make_hybrid_emulator(path: Path, model: LightconePNGVelocileptors, mask: np.ndarray, nint: int) -> RegularGridInterpolator:
    f_grid = np.linspace(-500., 500., 41)
    b_grid = np.linspace(.5, 5., 46)
    if path.is_file():
        with np.load(path, allow_pickle=False) as d:
            if np.array_equal(d["f_grid"], f_grid) and np.array_equal(d["b_grid"], b_grid):
                return RegularGridInterpolator((f_grid, b_grid), d["values"], bounds_error=True)
    values = np.empty((f_grid.size, b_grid.size, int(np.count_nonzero(mask))*2), dtype="f8")
    for i, f in enumerate(f_grid):
        for j, b in enumerate(b_grid):
            y = model.evaluate(fnl=float(f), b1=float(b), nint=int(nint))
            values[i,j] = np.concatenate((y[0][mask], y[2][mask]))
        print(json.dumps({"emulator_row": i+1, "total": f_grid.size}), flush=True)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, f_grid=f_grid, b_grid=b_grid, values=values)
    return RegularGridInterpolator((f_grid,b_grid), values, bounds_error=True)


def run_chain(name, data, cov, evaluate, start, lower, upper, *, nwalkers, nsteps, burnin, seed):
    metric = ScaledGaussianMetric(cov)
    def lp(theta):
        t=np.asarray(theta,dtype="f8")
        if np.any(t<=lower) or np.any(t>=upper): return -np.inf
        d=data-evaluate(t); return -.5*metric.chi2(d)
    rng=np.random.default_rng(seed); spread=np.asarray([4.,.04,.5] if len(start)==3 else [4.,.04],dtype="f8")
    init=np.asarray(start)[None,:]+rng.normal(size=(nwalkers,len(start)))*spread[None,:]
    init=np.maximum(init,lower[None,:]+1e-7); init=np.minimum(init,upper[None,:]-1e-7)
    np.random.seed(seed); sampler=emcee.EnsembleSampler(nwalkers,len(start),lp); sampler.run_mcmc(init,nsteps,progress=False,skip_initial_state_check=True)
    chain=sampler.get_chain(discard=burnin); flat=chain.reshape(-1,chain.shape[-1]); logp=sampler.get_log_prob(discard=burnin).reshape(-1)
    names=["fNL","b1","sigma_s"] if len(start)==3 else ["fNL","b1"]
    q=np.percentile(flat,[16,50,84],axis=0); tau=np.asarray(emcee.autocorr.integrated_time(chain,quiet=True,tol=0),dtype="f8")
    return {"name":name,"chain":flat,"logp":logp,"summary":{"posterior":{n:{"q16":float(q[0,i]),"q50":float(q[1,i]),"q84":float(q[2,i]),"sigma68":float(.5*(q[2,i]-q[0,i]))} for i,n in enumerate(names)},"parameter_names":names,"nwalkers":nwalkers,"nsteps":nsteps,"burnin":burnin,"seed":seed,"acceptance_fraction_mean":float(np.mean(sampler.acceptance_fraction)),"tau":tau.tolist(),"postburn_length_over_tau":(chain.shape[0]/tau).tolist()}}


def main():
    parser=argparse.ArgumentParser(); parser.add_argument("--nwalkers",type=int,default=64); parser.add_argument("--nsteps",type=int,default=30000); parser.add_argument("--burnin",type=int,default=5000); parser.add_argument("--nint",type=int,default=300); args=parser.parse_args()
    s,mask,data,cov,zeff,ids=load_data(); original,hybrid,cache=build_models(s,mask,zeff); emulator=make_hybrid_emulator(OUT_ROOT/"lightcone_no_bao_xi_emulator.npz",hybrid,mask,args.nint)
    def original_eval(t):
        v=original.evaluate(np.asarray(t,dtype="f8"),model="formal_gic",window_key="mean"); return np.concatenate((v[0][mask],v[2][mask]))
    def hybrid_eval(t): return emulator(np.asarray([[float(t[0]),float(t[1])]],dtype="f8"))[0]
    lo3=np.asarray([-500.,.5,0.]); hi3=np.asarray([500.,5.,30.]); lo2=np.asarray([-500.,.5]); hi2=np.asarray([500.,5.])
    original_result=run_chain("original_xi02_no_bao",data,cov,original_eval,np.asarray([-4.,2.41,5.6]),lo3,hi3,nwalkers=args.nwalkers,nsteps=args.nsteps,burnin=args.burnin,seed=20260928)
    hybrid_result=run_chain("hybrid_xi02_no_bao",data,cov,hybrid_eval,np.asarray([-5.,2.43]),lo2,hi2,nwalkers=args.nwalkers,nsteps=args.nsteps,burnin=args.burnin,seed=20260929)
    OUT_ROOT.mkdir(parents=True,exist_ok=True); np.savez_compressed(OUT_ROOT/"original_xi02_no_bao_samples.npz",samples=original_result["chain"],logp=original_result["logp"]); np.savez_compressed(OUT_ROOT/"hybrid_xi02_no_bao_samples.npz",samples=hybrid_result["chain"],logp=hybrid_result["logp"])
    atomic_json(OUT_ROOT/"task432_no_bao_xi02_summary.json",{"contract":{"bao_mask":False,"s_range":"50 <= s < 350","covariance":"C_single"},"original":original_result["summary"],"hybrid":hybrid_result["summary"]})
    print(json.dumps({"status":"complete","original":original_result["summary"],"hybrid":hybrid_result["summary"]},sort_keys=True))


if __name__=="__main__": main()
