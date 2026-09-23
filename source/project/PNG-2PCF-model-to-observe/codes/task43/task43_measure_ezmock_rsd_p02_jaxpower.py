#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Task43 EZmock RSD rawbox P0/P2 jaxpower helper。"""
from __future__ import annotations
import argparse, json, os, sys, time
from pathlib import Path
import numpy as np
for name in ("OMP_NUM_THREADS","MKL_NUM_THREADS","OPENBLAS_NUM_THREADS","NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(name,"1")
os.environ.setdefault("JAX_PLATFORMS","cpu")
os.environ.setdefault("JAX_PLATFORM_NAME","cpu")
os.environ.setdefault("CUDA_VISIBLE_DEVICES","")
CODE_DIR=Path(__file__).resolve().parent
DESI_CLUSTERING_ROOT=Path("/pscratch/sd/l/lzy/desi-clustering")
# 让 helper 无论从 notebook 还是 shell 启动，都优先使用同一份官方
# cosmodesiconda site-packages；用户 ~/.local 中的 mpi4py ABI 可能不兼容。
_python_site=Path(sys.prefix) / "lib" / f"python{sys.version_info.major}.{sys.version_info.minor}" / "site-packages"
_cosmo_root=Path(sys.prefix).parent
_extra_sites=[_python_site]
# 递归加入当前 cosmodesiconda bundle 的全部内部 site-packages，覆盖
# lsstypes、mpytools、mockfactory 等 clustering_statistics 的间接依赖。
_code_python = f"python{sys.version_info.major}.{sys.version_info.minor}"
_extra_sites.extend((_cosmo_root / "code").glob(f"*/main/lib/{_code_python}/site-packages"))
_extra_sites.extend((_cosmo_root / "code").glob(f"*/mpi/lib/{_code_python}/site-packages"))
_bad_user_sites=[item for item in sys.path if ".local/lib/python" in str(item)]
for item in _bad_user_sites:
    sys.path.remove(item)
for item in reversed(_extra_sites + [CODE_DIR, DESI_CLUSTERING_ROOT]):
    if item.is_dir() and str(item) not in sys.path:
        sys.path.insert(0, str(item))
# 先缓存官方 MPI 模块，再恢复用户路径以便其他 DESI 依赖正常寻找。
try:
    from mpi4py import MPI as _task43_mpi
finally:
    for item in _bad_user_sites:
        if item not in sys.path:
            sys.path.append(item)
from task43_rawbox_ezmock_common import atomic_savez, set_cpu_affinity, write_json

def edges(kmin:float,kmax:float,dk:float)->np.ndarray:
    """构造等宽 k bin 边界。"""
    e=np.arange(float(kmin),float(kmax)+0.5*float(dk),float(dk),dtype="f8")
    if e[-1] < float(kmax): e=np.append(e,float(kmax))
    e[0],e[-1]=float(kmin),float(kmax)
    return np.column_stack([e[:-1],e[1:]])

def main()->None:
    """测量一个 RSD catalog 的 P0 和 P2，并原子保存结果。"""
    p=argparse.ArgumentParser()
    p.add_argument("--catalog",type=Path,required=True); p.add_argument("--output",type=Path,required=True)
    p.add_argument("--boxsize",type=float,default=2000.0); p.add_argument("--meshsize",type=int,default=400)
    p.add_argument("--kmin",type=float,default=0.001); p.add_argument("--kmax",type=float,default=0.3001); p.add_argument("--dk",type=float,default=0.002)
    p.add_argument("--threads",type=int,default=8); p.add_argument("--seed",type=int,required=True)
    a=p.parse_args(); cpus=set_cpu_affinity(int(a.threads)); meta=a.output.with_suffix(".json")
    if not a.catalog.is_file(): raise FileNotFoundError(a.catalog)
    import jax
    jax.config.update("jax_enable_x64",True)
    from clustering_statistics import spectrum2_tools
    started=time.perf_counter(); pos=np.loadtxt(a.catalog,comments="#",usecols=(0,1,2),dtype="f8")
    if pos.ndim!=2 or pos.shape[1]!=3: raise ValueError(f"invalid catalog shape {pos.shape}")
    pos=np.mod(pos,float(a.boxsize)); data={"POSITION":pos,"INDWEIGHT":np.ones(pos.shape[0],dtype="f8")}
    spectrum=spectrum2_tools.compute_box_mesh2_spectrum(lambda:{"data":data}, mattrs={"boxsize":float(a.boxsize),"boxcenter":float(a.boxsize)/2.0,"meshsize":int(a.meshsize)}, edges=edges(a.kmin,a.kmax,a.dk), ells=(0,2), los="z")
    arrays={"seed":np.asarray(a.seed,dtype="i8"),"ndata":np.asarray(pos.shape[0],dtype="i8"),"boxsize":np.asarray(a.boxsize,dtype="f8"),"meshsize":np.asarray(a.meshsize,dtype="i8")}
    for ell in (0,2):
        pole=spectrum.get(ell); arrays[f"k{ell}"]=np.asarray(pole.coords("k"),dtype="f8"); arrays[f"k_edges{ell}"]=np.asarray(pole.edges("k"),dtype="f8"); arrays[f"pk{ell}"]=np.asarray(pole.value(),dtype="f8"); arrays[f"nmodes{ell}"]=np.asarray(pole.values("nmodes"),dtype="f8"); arrays[f"norm{ell}"]=np.asarray(pole.values("norm"),dtype="f8"); arrays[f"num_shotnoise{ell}"]=np.asarray(pole.values("num_shotnoise"),dtype="f8"); arrays[f"shotnoise{ell}"]=np.asarray(pole.values("shotnoise"),dtype="f8")
    atomic_savez(a.output,**arrays)
    write_json(meta,{"task":"task43_measure_ezmock_rsd_p02_jaxpower","status":"done","catalog":str(a.catalog),"output":str(a.output),"seed":int(a.seed),"ndata":int(pos.shape[0]),"boxsize":float(a.boxsize),"meshsize":int(a.meshsize),"kmin":float(a.kmin),"kmax":float(a.kmax),"dk":float(a.dk),"ells":[0,2],"engine":"clustering_statistics.spectrum2_tools","jax_backend":jax.default_backend(),"cpu_affinity":cpus,"elapsed_sec":float(time.perf_counter()-started)})
    print(f"[done] seed={a.seed} P0/P2 bins={arrays['pk0'].size} elapsed={time.perf_counter()-started:.1f}s",flush=True)
if __name__=="__main__": main()