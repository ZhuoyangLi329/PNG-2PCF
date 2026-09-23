#!/usr/bin/env python3
"""Frozen EZmock-281 covariance comparison against the September 11 RSD result.

Uses the existing, validated main-result model and data.  The historical
15-bin P grid is a sparse selection of 0.002-wide bins, not a wide-bin
average.  kmax=0.08 and P2 kmin=0.015 leave 13+9 P bins; the shared xi BAO
mask leaves 26+26 xi bins.  All covariance units are single lightcone.
"""
from __future__ import annotations

import argparse
import fcntl
import hashlib
import json
import os
from dataclasses import replace
from datetime import datetime, timezone
from pathlib import Path
import sys
import time

# Set these before importing numpy or any model dependency.
for var in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[var] = "1"
os.environ["JAX_PLATFORMS"] = "cpu"
os.environ["CUDA_VISIBLE_DEVICES"] = ""

import numpy as np
from scipy.linalg import solve_triangular

PROJECT = Path("/pscratch/sd/l/lzy/PNG-2PCF-model-to-observe")
sys.path.insert(0, str(PROJECT / "codes/task43"))
PRODUCTION = PROJECT / "outputs/task43_outputs/ezmock_rsd_lightcone_covariance_z0p4_0p8_x1000_common50"
MANIFEST = PRODUCTION / "manifests/task43_ezmock_rsd_covariance_x1000_fixampF_common50.jsonl"
REFERENCE = PROJECT / "outputs/task43_outputs/rsd_validation/lightcone/kmax0p08_smin50_v1"
REFERENCE_COV = REFERENCE / "task43_lightcone_standard_joint_baomask80_120_covariance_v1.npz"
REFERENCE_AUDIT = REFERENCE / "task43_lightcone_standard_joint_baomask80_120_v1.json"
DEFAULT_OUTPUT = PRODUCTION / "mcmc_preliminary_281_v2"
VARIANTS = {"p02": "rsd_p02", "xi02": "rsd_xi02", "joint": "rsd_joint_p02xi02"}
SLICES = {"p02": slice(0, 22), "xi02": slice(22, 74), "joint": slice(0, 74)}
NAMES = {"p02": ("fNL", "b1", "sigma_s", "sn0"), "xi02": ("fNL", "b1", "sigma_s"), "joint": ("fNL", "b1", "sigma_s", "sn0")}


def digest(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as src:
        for chunk in iter(lambda: src.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def plain(value):
    if isinstance(value, dict):
        return {str(k): plain(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [plain(v) for v in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    return value


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(plain(value), indent=2, sort_keys=True), encoding="utf-8")
    tmp.replace(path)


def save_npz(path, **arrays):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name("." + path.name + ".tmp.npz")
    np.savez_compressed(tmp, **arrays)
    tmp.replace(path)


def emit(**value):
    print(json.dumps(plain(value), sort_keys=True), flush=True)


def corrections(ns, nd, npfit):
    if ns <= nd + 4:
        raise ValueError("insufficient mocks for Hartlap/Percival")
    A = 2.0 / ((ns-nd-1) * (ns-nd-4))
    B = (ns-nd-2.0) / ((ns-nd-1) * (ns-nd-4))
    m1 = (1.0+B*(nd-npfit)) / (1.0+A+B*(npfit+1))
    return dict(nmock=ns, ndata=nd, nparams=npfit, hartlap=(ns-nd-2.0)/(ns-1),
                A=A, B=B, percival_m1=m1, sigma_factor=float(np.sqrt(m1)))


def covariance_diagnostics(cov):
    cov = np.asarray(cov, dtype="f8")
    sig = np.sqrt(np.diag(cov))
    if not np.isfinite(sig).all() or np.any(sig <= 0):
        raise ValueError("non-positive covariance diagonal")
    corr = cov / np.outer(sig, sig)
    eig = np.linalg.eigvalsh((corr+corr.T)*0.5)
    if eig[0] <= 1e-12:
        raise ValueError(f"strict SPD failed; min correlation eigenvalue={eig[0]}")
    np.linalg.cholesky(corr)
    return dict(shape=list(cov.shape), correlation_eigen_min=float(eig[0]),
                correlation_eigen_max=float(eig[-1]), correlation_condition=float(eig[-1]/eig[0]),
                eigenvalue_repair=False)


def matches(edges, target):
    indices = []
    for pair in target:
        found = np.flatnonzero(np.all(np.isclose(edges, pair[None, :], rtol=0, atol=1e-12), axis=1))
        if len(found) != 1:
            raise ValueError(f"edge {pair} has {len(found)} matches")
        indices.append(int(found[0]))
    return np.asarray(indices)


def prepare(root, nmock):
    audit_path, payload_path = root/"freeze.json", root/"covariance.npz"
    if audit_path.exists():
        audit = json.loads(audit_path.read_text())
        if audit["nmock"] != nmock or digest(payload_path) != audit["payload_sha256"]:
            raise RuntimeError("frozen snapshot contract changed")
        with np.load(payload_path, allow_pickle=False) as src:
            payload = {key: np.asarray(src[key]) for key in src.files}
        emit(stage="freeze", status="reused", nmock=nmock, path=payload_path)
        return audit, payload
    if payload_path.exists():
        raise RuntimeError("partial freeze exists; inspect before replacing")
    baseline = json.loads(REFERENCE_AUDIT.read_text())
    with np.load(REFERENCE_COV, allow_pickle=False) as src:
        analytic = np.asarray(src["rsd_joint"])
        s = np.asarray(src["rsd_s"])
        mask = np.asarray(src["rsd_xi_mask"], dtype=bool)
        p2keep = np.asarray(src["rsd_p2_keep_indices"], dtype=int)
        k = np.asarray(src["rsd_k"])
    edges = np.asarray(baseline["rsd"]["pk_fit_edges_h_mpc"], dtype="f8")
    assert edges.shape == (13,2) and len(p2keep)==9 and mask.sum()==26
    expected_mask = (s>=50)&(s<350)&~((s>=80)&(s<120))
    assert np.array_equal(mask, expected_mask) and analytic.shape==(74,74)
    rr = PRODUCTION/"fcfc_pairs/RR_common50_zobs0p4_0p8_s30_350_ds10_mu120.bin"
    rr_hash = digest(rr)
    rows = sorted((json.loads(line) for line in MANIFEST.read_text().splitlines() if line.strip()), key=lambda row:row["production_index"])
    frozen, stack = [], []
    for row in rows:
        xp, pp = Path(row["xi_path"]), Path(row["pk_path"])
        files = (xp, pp, xp.with_suffix(".json"), pp.with_suffix(".json"))
        if not all(p.is_file() and p.stat().st_size for p in files):
            continue
        xm, pm = json.loads(files[2].read_text()), json.loads(files[3].read_text())
        if xm.get("status") != "done" or pm.get("status") != "done":
            continue
        idx, seed = int(row["production_index"]), int(row["seed"])
        assert seed==600001+idx and row["fix_amplitude"] is False
        assert all(m["fix_amplitude"] is False and m["seed"]==seed for m in (xm,pm))
        assert xm["ells"]==[0,2] and xm["mu_bin_num"]==120
        assert xm["common_rr_smu_sha256"]==rr_hash
        with np.load(pp, allow_pickle=False) as p, np.load(xp, allow_pickle=False) as x:
            assert int(x["seed"])==seed and int(p["seed"])==seed
            assert np.array_equal(x["s"], s)
            assert not bool(x["fix_amplitude"]) and not bool(p["fix_amplitude"])
            i0, i2 = matches(p["k_edges0"],edges), matches(p["k_edges2"],edges)
            assert x["xi0"].shape==(32,) and x["xi2"].shape==(32,)
            assert str(x["common_rr_smu_sha256"])==rr_hash
            common = str(p["common_random_sha256"])
            assert common==str(x["common_random_npz_sha256"])
            vector = np.concatenate([p["pk0"][i0],p["pk2"][i2][p2keep],x["xi0"][mask],x["xi2"][mask]])
            assert vector.shape==(74,) and np.isfinite(vector).all()
        frozen.append(dict(production_index=idx,seed=seed,common_random_sha256=common,
                           paths={str(p):digest(p) for p in files}))
        stack.append(vector)
        if len(stack)==nmock:
            break
    if len(stack)!=nmock:
        raise RuntimeError(f"{len(stack)} complete paired mocks; expected {nmock}")
    if len({r["seed"] for r in frozen})!=nmock or len({r["common_random_sha256"] for r in frozen})!=1:
        raise RuntimeError("seed uniqueness/common random invariant failed")
    stack=np.asarray(stack,dtype="f8")
    cov=np.cov(stack,rowvar=False,ddof=1)
    payload=dict(stack=stack,covariance_ezmock=cov,covariance_jaxpower=analytic,
                 edges=edges,p2_keep=p2keep,s=s,xi_mask=mask,k_reference=k,
                 indices=np.asarray([r["production_index"] for r in frozen]),seeds=np.asarray([r["seed"] for r in frozen]))
    source_hashes={str(REFERENCE_COV):digest(REFERENCE_COV),str(REFERENCE_AUDIT):digest(REFERENCE_AUDIT)}
    for variant,specname in VARIANTS.items():
        source=REFERENCE/"fits"/specname/"samples.npz"
        with np.load(source,allow_pickle=False) as data:
            payload[f"data_{variant}"]=np.asarray(data["data"])
            payload[f"phase_data_{variant}"]=np.asarray(data["phase_data"])
            assert np.allclose(data["covariance_single"],analytic[SLICES[variant],SLICES[variant]],rtol=1e-12,atol=1e-18)
        source_hashes[str(source)]=digest(source)
    assert np.allclose(payload["data_joint"],np.r_[payload["data_p02"],payload["data_xi02"]],rtol=0,atol=1e-12)
    diag={family:{v:covariance_diagnostics(C[sl,sl]) for v,sl in SLICES.items()} for family,C in (("ezmock281",cov),("jaxpower",analytic))}
    save_npz(payload_path,**payload)
    audit=dict(status="frozen",nmock=nmock,created_utc=datetime.now(timezone.utc).isoformat(),
               selection="first 281 completed valid paired rows in production-index order at freeze time; fixed thereafter",
               rows=frozen,source_hashes=source_hashes,manifest_sha256=digest(MANIFEST),
               rr_sha256=rr_hash,payload_sha256=digest(payload_path),data_dimensions=dict(p0=13,p2=9,xi0=26,xi2=26,joint=74),
               bin_policy="exact sparse fine-bin edge selection; no interpolation or wide-bin averaging",
               covariance_policy="single-realization sample covariance ddof=1; not divided by 281 or 25",
               corrections={v:corrections(nmock,len(payload[f"data_{v}"]),len(NAMES[v])) for v in VARIANTS},diagnostics=diag)
    write_json(audit_path,audit)
    emit(stage="freeze",status="done",nmock=nmock,dimensions=audit["data_dimensions"],corrections=audit["corrections"],diagnostics=diag)
    return audit,payload


class FastMetric:
    """Strict correlation-normalized Gaussian metric; no eigenvalue repair."""
    def __init__(self,cov,hartlap):
        self.scale=np.sqrt(np.diag(cov))
        self.correlation=cov/np.outer(self.scale,self.scale)
        self.chol=np.linalg.cholesky((self.correlation+self.correlation.T)/2)
        self.sqrt_h=np.sqrt(hartlap)
    def residual(self,difference):
        return self.sqrt_h*solve_triangular(self.chol,np.asarray(difference)/self.scale,lower=True,check_finite=False)
    def chi2(self,difference):
        r=self.residual(difference)
        return float(r@r)


def posterior(flat,names):
    q=np.percentile(flat,[2.5,16,50,84,97.5],axis=0)
    return {name:dict(q025=float(q[0,i]),q16=float(q[1,i]),q50=float(q[2,i]),q84=float(q[3,i]),q975=float(q[4,i]),
                     sigma68=float((q[3,i]-q[1,i])/2),std=float(flat[:,i].std(ddof=1)),mean=float(flat[:,i].mean())) for i,name in enumerate(names)}


def corrected_posterior(raw,factor):
    corrected=json.loads(json.dumps(raw))
    for p in corrected.values():
        for key in ("q025","q16","q84","q975"):
            p[key]=p["q50"]+factor*(p[key]-p["q50"])
        p["std"]*=factor
        p["sigma68"]*=factor
    return corrected


def fit_all(root,audit,payload,args):
    from task43_run_lightcone_joint_baomask_v1 import load_rsd_specs,fit_maximum_likelihood,run_chain
    started=time.perf_counter()
    emit(stage="canonical_model",status="loading")
    specs,meta,arrays=load_rsd_specs(smin=50.0,pk_kmax=0.08)
    specmap={s.name:s for s in specs}
    assert np.allclose(arrays["rsd_joint"],payload["covariance_jaxpower"],rtol=1e-12,atol=1e-18)
    assert np.array_equal(arrays["rsd_xi_mask"],payload["xi_mask"])
    assert np.array_equal(arrays["rsd_p2_keep_indices"],payload["p2_keep"])
    emit(stage="canonical_model",status="validated",seconds=time.perf_counter()-started)
    runroot=root/("smoke" if args.stage=="smoke" else "fits")
    write_json(runroot/"model_audit.json",meta)
    nsteps,burnin,nruns=(300,100,1) if args.stage=="smoke" else (args.nsteps,args.burnin,args.ensembles)
    results={}
    families = tuple(x.strip() for x in args.families.split(",") if x.strip())
    for family_i,family in enumerate(families):
        fullcov=payload["covariance_ezmock" if family=="ezmock281" else "covariance_jaxpower"]
        for vi,(variant,specname) in enumerate(VARIANTS.items()):
            base=specmap[specname]
            assert base.parameter_names==NAMES[variant]
            assert np.allclose(base.data,payload[f"data_{variant}"],rtol=1e-12,atol=1e-12)
            cov=fullcov[SLICES[variant],SLICES[variant]]
            covariance_diagnostics(cov)
            factors=audit["corrections"][variant] if family=="ezmock281" else dict(hartlap=1.0,percival_m1=1.0,sigma_factor=1.0)
            spec=replace(base,covariance=cov)
            metric=FastMetric(cov,factors["hartlap"])
            nominal,theta=fit_maximum_likelihood(spec,metric)
            tag=f"{family}_{variant}"
            pieces,runinfo=[],[]
            for ri in range(nruns):
                seed=args.seed+family_i*10000+vi*1000+ri*100
                config=dict(nwalkers=args.nwalkers,nsteps=nsteps,burnin=burnin,seed=seed,freeze_sha256=audit["payload_sha256"])
                out=runroot/tag/f"run{ri}"
                jp,npz=out.with_suffix(".json"),out.with_suffix(".npz")
                if jp.exists():
                    info=json.loads(jp.read_text())
                    if info["config"]!=config or info["chain_sha256"]!=digest(npz):
                        raise RuntimeError(f"resume mismatch: {jp}")
                    with np.load(npz,allow_pickle=False) as z:
                        chain=np.asarray(z["chain"])
                    sm=info["mcmc"]
                    emit(stage="mcmc",tag=tag,run=ri,status="resumed")
                else:
                    emit(stage="mcmc",tag=tag,run=ri,status="started",config=config)
                    ts=time.perf_counter()
                    sm,chain,logp=run_chain(spec,metric,theta,nwalkers=args.nwalkers,nsteps=nsteps,burnin=burnin,seed=seed,nworkers=args.workers)
                    save_npz(npz,chain=chain,logp=logp,parameter_names=np.asarray(spec.parameter_names),data=spec.data,theta_map=theta)
                    info=dict(config=config,mcmc=sm,seconds=time.perf_counter()-ts,chain_sha256=digest(npz))
                    write_json(jp,info)
                    emit(stage="mcmc",tag=tag,run=ri,status="saved",seconds=info["seconds"],gates=sm["gates"])
                pieces.append(chain.reshape(-1,len(spec.parameter_names)))
                runinfo.append(sm)
            flat=np.vstack(pieces)
            raw=posterior(flat,spec.parameter_names)
            corrected=corrected_posterior(raw,factors["sigma_factor"])
            result=dict(parameter_names=list(spec.parameter_names),ndata=len(spec.data),map=nominal,factors=factors,
                        posterior_hartlap=raw,posterior_corrected=corrected,
                        parameter_covariance_raw=np.cov(flat,rowvar=False),
                        parameter_covariance_corrected=factors["percival_m1"]*np.cov(flat,rowvar=False),
                        chains_converged=all(all(s["gates"].values()) for s in runinfo),runs=runinfo,
                        correction_policy="Hartlap in likelihood for sample covariance only; Percival m1 on parameter covariance, sqrt(m1) width approximation. No finite-mock correction on analytic covariance.",
                        width_caveat="Affine width correction is approximate for non-Gaussian or prior-truncated parameters; raw chains retained.")
            results[tag]=result
            write_json(runroot/tag/"summary.json",result)
            emit(stage="result",tag=tag,fNL=corrected["fNL"],converged=result["chains_converged"])
    metrics={}
    for fam in families:
        widths={v:results[f"{fam}_{v}"]["posterior_corrected"]["fNL"]["sigma68"] for v in VARIANTS}
        metrics[fam]=dict(sigma68_fNL=widths,joint_gain_over_p_fraction=1-widths["joint"]/widths["p02"],
                          joint_gain_over_xi_fraction=1-widths["joint"]/widths["xi02"],
                          joint_gain_over_best_fraction=1-widths["joint"]/min(widths["p02"],widths["xi02"]))
    write_json(runroot/"comparison.json",dict(status="smoke_only" if args.stage=="smoke" else "pass" if all(r["chains_converged"] for r in results.values()) else "needs_longer_chains",results=results,metrics=metrics,nmock=args.nmock))
    if args.stage!="smoke":
        plot(root)


def plot(root):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from scipy.ndimage import gaussian_filter
    comparison=json.loads((root/"fits/comparison.json").read_text())
    nmock=int(comparison["results"]["ezmock281_joint"]["factors"]["nmock"])
    fig,axes=plt.subplots(1,3,figsize=(16.5,5.3),constrained_layout=True)
    colors=dict(p02="#2763a3",xi02="#329457",joint="#bf2f3b")
    labels=dict(p02=r"$P_0+P_2$",xi02=r"$\xi_0+\xi_2$",joint="Joint")
    families = tuple(f for f in ("ezmock281","jaxpower") if any(k.startswith(f+"_") for k in comparison["results"]))
    for family,linestyle in (("ezmock281","-"),("jaxpower","--")):
        if family not in families: continue
        for variant in VARIANTS:
            tag=f"{family}_{variant}"
            sm=comparison["results"][tag]
            pieces=[]
            for file in sorted((root/"fits"/tag).glob("run*.npz")):
                with np.load(file,allow_pickle=False) as data:
                    pieces.append(np.asarray(data["chain"]).reshape(-1,len(sm["parameter_names"]))[:, :2])
            points=np.vstack(pieces)
            center=np.median(points,axis=0)
            points=center+(points-center)*sm["factors"]["sigma_factor"]
            limits=np.percentile(points,[0.05,99.95],axis=0)
            hist,xe,ye=np.histogram2d(points[:,0],points[:,1],bins=110,range=limits.T.tolist())
            density=gaussian_filter(hist,1.2)
            ordered=np.sort(density.ravel())[::-1]
            cumulative=np.cumsum(ordered)/ordered.sum()
            levels=np.sort([ordered[np.searchsorted(cumulative,p)] for p in (0.6827,0.9545)])
            x=(xe[1:]+xe[:-1])/2;y=(ye[1:]+ye[:-1])/2
            targets=[axes[0 if family=="ezmock281" else 1]]
            if variant=="joint":targets.append(axes[2])
            for ax in targets:
                ax.contour(x,y,density.T,levels=levels,colors=colors[variant],linestyles=linestyle,linewidths=[1.1,1.8])
                pf=sm["posterior_corrected"]["fNL"]
                value=(rf"$f_{{\rm NL}}={pf['q50']:+.1f}"
                       rf"^{{+{pf['q84']-pf['q50']:.1f}}}_{{-{pf['q50']-pf['q16']:.1f}}}$")
                family_label = f"EZmock N={int(sm['factors']['nmock'])}" if family == "ezmock281" else "analytic jaxpower"
                label=(f"{labels[variant]} ({value})" if ax is not axes[2]
                       else f"{family_label} ({value})")
                ax.plot([],[],color=colors[variant],ls=linestyle,label=label)
    for ax,title in zip(axes,(f"EZmock N={nmock}: Hartlap + Percival","Analytic jaxpower covariance (reused)","Joint contour comparison")):
        ax.set(xlabel=r"$f_{\rm NL}$",ylabel=r"$b_1$",title=title)
        ax.legend(frameon=False,fontsize=9); ax.grid(alpha=.15)
    # Common ranges for the first two panels permit direct visual comparison.
    xmin=min(ax.get_xlim()[0] for ax in axes[:2]);xmax=max(ax.get_xlim()[1] for ax in axes[:2])
    ymin=min(ax.get_ylim()[0] for ax in axes[:2]);ymax=max(ax.get_ylim()[1] for ax in axes[:2])
    for ax in axes:ax.set_xlim(xmin,xmax);ax.set_ylim(ymin,ymax)
    gains=[]
    for family,label in (("ezmock281",f"EZmock N={nmock}"),("jaxpower","Analytic")):
        if family not in families: continue
        m=comparison["metrics"][family]
        gains.append(f"{label}: joint error reduction vs P = {100*m['joint_gain_over_p_fraction']:.1f}%, "
                     f"vs xi = {100*m['joint_gain_over_xi_fraction']:.1f}%")
    fig.suptitle("Task43 RSD lightcone · preliminary covariance · 68% / 95% contours\n"
                 "Median and 16th-84th percentile errors; marginalized nuisances\n"+" | ".join(gains),fontsize=11)
    target=root/f"contours_ezmock{nmock}_vs_jaxpower.pdf"
    fig.savefig(target);plt.close(fig)
    emit(stage="plot",path=target,status="done")


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage",choices=("prepare","smoke","run","plot"),default="prepare")
    parser.add_argument("--output-root",type=Path,default=DEFAULT_OUTPUT)
    parser.add_argument("--nmock",type=int,default=281)
    parser.add_argument("--threads",type=int,default=8)
    parser.add_argument("--workers",type=int,default=1)
    parser.add_argument("--nwalkers",type=int,default=48)
    parser.add_argument("--nsteps",type=int,default=8000)
    parser.add_argument("--burnin",type=int,default=2000)
    parser.add_argument("--ensembles",type=int,default=4)
    parser.add_argument("--seed",type=int,default=20260915)
    parser.add_argument("--families",default="ezmock281,jaxpower",help="comma-separated covariance families to sample")
    args=parser.parse_args()
    if args.nmock < 281 or args.nmock > 1000 or not 1<=args.threads<=8 or not 1<=args.workers<=args.threads:
        raise ValueError("this production snapshot requires 281<=nmock<=1000 and at most 8 CPU cores")
    allowed=sorted(os.sched_getaffinity(0))[:args.threads]
    os.sched_setaffinity(0,allowed)
    args.output_root.mkdir(parents=True,exist_ok=True)
    with (args.output_root/"run.lock").open("a") as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        emit(stage=args.stage,cpu_affinity=allowed,pid=os.getpid(),host=os.uname().nodename)
        if args.stage=="plot":plot(args.output_root);return
        audit,payload=prepare(args.output_root,args.nmock)
        if args.stage!="prepare":fit_all(args.output_root,audit,payload,args)


if __name__=="__main__":main()
