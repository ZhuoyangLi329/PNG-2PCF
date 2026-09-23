#!/usr/bin/env python3
import argparse, json, os, shutil
from pathlib import Path
import numpy as np
import task43_rsd_ezmock426_monopole as m
from task43_run_lightcone_joint_baomask_v1 import load_rsd_specs

def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--root',type=Path,required=True); ap.add_argument('--nsteps',type=int,default=24000); ap.add_argument('--burnin',type=int,default=6000); ap.add_argument('--nwalkers',type=int,default=48); ap.add_argument('--ensembles',type=int,default=4); ap.add_argument('--seed',type=int,default=20260915); args=ap.parse_args()
    os.sched_setaffinity(0,sorted(os.sched_getaffinity(0))[:8]); root=args.root
    old=root/'fits'; backup=root/'superseded_shortchains'; backup.mkdir(exist_ok=True)
    audit,payload=m.prepare(root,426)
    for v in ('p0','joint'):
        src=old/f'ezmock_{v}'; dst=backup/f'ezmock_{v}'
        if src.exists():
            if dst.exists(): shutil.rmtree(dst)
            src.rename(dst)
    specs,_,_=load_rsd_specs(smin=50.,pk_kmax=.08); smap={x.name:x for x in specs}
    args.output_root=root
    for v in ('p0','joint'):
        m.posterior_result(smap[m.VARIANTS[v]],m.base.FastMetric,payload,audit,root,'ezmock',v,args)
    results={}
    for family in ('ezmock','jaxpower'):
        for v in m.VARIANTS:
            results[f'{family}_{v}']=json.loads((old/f'{family}_{v}/summary.json').read_text())
    metrics={}
    for family in ('ezmock','jaxpower'):
        w={v:results[f'{family}_{v}']['posterior_corrected']['fNL']['sigma68'] for v in m.VARIANTS}
        metrics[family]={'sigma68_fNL':w,'gain_p':1-w['joint']/w['p0'],'gain_xi':1-w['joint']/w['xi0']}
    comp={'status':'pass' if all(x['chains_converged'] for x in results.values()) else 'needs_longer_chains','nmock':426,'results':results,'metrics':metrics}
    m.write(root/'fits/comparison.json',comp); m.plot(root,comp); print(json.dumps(metrics,indent=2),flush=True)
if __name__=='__main__': main()
