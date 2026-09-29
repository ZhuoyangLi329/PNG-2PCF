#!/usr/bin/env python3
"""Compile an independent velocileptors xi full-RIC diagnostic grid."""
from __future__ import annotations
import argparse, json, os, sys, time, hashlib
from pathlib import Path
for key in ("OMP_NUM_THREADS","MKL_NUM_THREADS","OPENBLAS_NUM_THREADS","NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(key,"1")
import numpy as np
ROOT=Path("/pscratch/sd/l/lzy/PNG-2PCF-model-to-observe")
for folder in (ROOT/"codes/task432",ROOT/"codes/task43",ROOT/"codes/task44"):
    if str(folder) not in sys.path: sys.path.insert(0,str(folder))
from task432_velocileptors_fullric_model import VeloPoles
from task432_lightcone_png_velocileptors import RSD_P_PAYLOAD

BASE=ROOT/"outputs/task43_outputs/rsd_validation/task432_model_repair/full_ric_p02xi02_v1"
COMPILED=BASE/"pilot/mean4xi_mean2P_prodrr_stochastic/engine.npz"
XI_KERNEL=BASE/"geometry/validated_mean4_midpoint.npz"
OUT=ROOT/"outputs/task43_outputs/rsd_validation/task432_model_repair/velocileptors_fullric_v1/operator"

def sha(path):
    h=hashlib.sha256();
    with Path(path).open("rb") as f:
        for chunk in iter(lambda:f.read(8*1024*1024),b""): h.update(chunk)
    return h.hexdigest()

def main():
    ap=argparse.ArgumentParser(); ap.add_argument("--nint",type=int,default=600); ap.add_argument("--f-step",type=float,default=25.0); ap.add_argument("--b-step",type=float,default=0.1); ap.add_argument("--resume",action="store_true"); args=ap.parse_args()
    OUT.mkdir(parents=True,exist_ok=True); rows_dir=OUT/"rows"; rows_dir.mkdir(exist_ok=True)
    with np.load(COMPILED,allow_pickle=False) as a:
        fgrid=np.asarray(a["f_grid"],dtype="f8"); bgrid=np.asarray(a["b_grid"],dtype="f8")
        fixed_shot=np.asarray(a["fixed_shot_vector"][22:],dtype="f8")
        sn_resp=np.asarray(a["sn0_ric_response"][22:],dtype="f8")
        xi_response=np.asarray(a["xi_response"],dtype="f8")
    # Preserve the validated full grid unless an explicit reduced grid is requested.
    if args.f_step != 25.0: fgrid=np.arange(-500.,500.+0.5*args.f_step,args.f_step)
    if args.b_step != 0.1: bgrid=np.arange(0.5,5.0+0.5*args.b_step,args.b_step)
    with np.load(XI_KERNEL,allow_pickle=False) as a:
        edges=np.asarray(a["separation_edges"],dtype="f8")
    with np.load(ROOT/"outputs/task43_outputs/rsd_validation/task432_model_repair/hybrid_ezmock1000_0918_contract_v1/frozen_inputs.npz",allow_pickle=False) as a:
        fit_centers=np.asarray(a["xi_centers"],dtype="f8")
    centers=0.75*(edges[1:]**4-edges[:-1]**4)/(edges[1:]**3-edges[:-1]**3)
    # Two-node r^2 shell average matches the existing hybrid physical cache.
    nodes,weights=np.polynomial.legendre.leggauss(2); lo,hi=edges[:-1],edges[1:]
    radii=0.5*(hi-lo)[:,None]*nodes[None,:]+0.5*(hi+lo)[:,None]
    shell_weights=0.5*(hi-lo)[:,None]*weights[None,:]*radii**2
    shell_den=(hi**3-lo**3)/3.0
    # Each model evaluation receives shell nodes followed by the 52 fit centers.
    all_s=np.r_[fit_centers,radii.ravel()]
    nf,nb=len(fgrid),len(bgrid)
    manifest={"status":"running","f_grid":fgrid.tolist(),"b_grid":bgrid.tolist(),"nint":int(args.nint),"shell_edges":edges.tolist(),"shell_average":"2-point Gauss-Legendre r^2 volume average","pole_order":[0,2,4],"xi_response_shape":list(xi_response.shape),"fixed_shot_shape":list(fixed_shot.shape),"sn0_response_shape":list(sn_resp.shape),"source_hashes":{"compiled":sha(COMPILED),"xi_kernel":sha(XI_KERNEL)},"output":str(OUT)}
    (OUT/"operator_manifest.json").write_text(json.dumps(manifest,indent=2)+"\n")
    for i,f in enumerate(fgrid):
        target=rows_dir/f"row_{i:03d}.npz"
        if args.resume and target.is_file():
            print(json.dumps({"event":"skip_row","i":i}),flush=True); continue
        raw=np.empty((nb,52),dtype="f8"); ric=np.empty((nb,52),dtype="f8"); poles_store=np.empty((nb,3,len(edges)-1),dtype="f8")
        started=time.time()
        for j,b in enumerate(bgrid):
            model=VeloPoles(all_s)
            poles_all=model.evaluate(float(f),float(b),nint=int(args.nint))
            nfit=len(fit_centers)
            shell=poles_all[:,nfit:].reshape(3,len(edges)-1,2)
            poles=np.sum(shell*shell_weights[None,:,:],axis=2)/shell_den[None,:]
            raw_fit=np.concatenate((poles_all[0,:nfit],poles_all[1,:nfit]))
            ric_fit=np.einsum("ild,ld->i",xi_response,poles)
            raw[j]=raw_fit; ric[j]=ric_fit; poles_store[j]=poles
        np.savez_compressed(target, fNL=np.asarray(float(f)), raw_xi=raw, ric_clustering=ric, poles=poles_store)
        print(json.dumps({"event":"row_done","i":i,"fNL":float(f),"elapsed":time.time()-started}),flush=True)
    rows=[]
    for i,f in enumerate(fgrid):
        p=rows_dir/f"row_{i:03d}.npz"
        if not p.is_file(): raise RuntimeError(f"missing row {p}")
        with np.load(p,allow_pickle=False) as a: rows.append((np.asarray(a["raw_xi"]),np.asarray(a["ric_clustering"]),np.asarray(a["poles"])))
    raw=np.stack([r[0] for r in rows]); ric=np.stack([r[1] for r in rows]); poles=np.stack([r[2] for r in rows])
    output=OUT/"velo_fullric_grid.npz"
    np.savez_compressed(output,f_grid=fgrid,b_grid=bgrid,raw_xi=raw,ric_clustering=ric,poles=poles,fixed_shot=fixed_shot,sn0_ric_response=sn_resp,xi_response=xi_response,xi_centers=centers,separation_edges=edges,nint=np.asarray(args.nint))
    manifest.update({"status":"pass","output_sha256":sha(output),"output":str(output),"grid_shape":list(raw.shape)})
    (OUT/"operator_manifest.json").write_text(json.dumps(manifest,indent=2)+"\n")
    print(json.dumps({"event":"compile_done","output":str(output),"shape":list(raw.shape)}),flush=True)
if __name__=="__main__": main()




