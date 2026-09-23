#!/usr/bin/env python3
import json, os, shutil
from pathlib import Path
import task43_rsd_ezmock426_monopole as m
from task43_run_lightcone_joint_baomask_v1 import load_rsd_specs

root=Path('/pscratch/sd/l/lzy/PNG-2PCF-model-to-observe/outputs/task43_outputs/ezmock_rsd_lightcone_covariance_z0p4_0p8_x1000_common50/mcmc_monopole_426')
os.sched_setaffinity(0, sorted(os.sched_getaffinity(0))[:8])
audit,payload=m.prepare(root,426)
backup=root/'superseded_shortchains_analytic'; backup.mkdir(exist_ok=True)
for v in ('p0','joint'):
    src=root/'fits'/f'jaxpower_{v}'; dst=backup/f'jaxpower_{v}'
    if src.exists():
        if dst.exists(): shutil.rmtree(dst)
        src.rename(dst)
specs,_,_=load_rsd_specs(smin=50.,pk_kmax=.08); smap={x.name:x for x in specs}
class A: pass
args=A(); args.nsteps=24000; args.burnin=6000; args.nwalkers=48; args.ensembles=4; args.seed=20260915; args.output_root=root
for v in ('p0','joint'):
    m.posterior_result(smap[m.VARIANTS[v]],m.base.FastMetric,payload,audit,root,'jaxpower',v,args)
results={}
for family in ('ezmock','jaxpower'):
  for v in m.VARIANTS: results[f'{family}_{v}']=json.loads((root/'fits'/f'{family}_{v}/summary.json').read_text())
metrics={}
for family in ('ezmock','jaxpower'):
  w={v:results[f'{family}_{v}']['posterior_corrected']['fNL']['sigma68'] for v in m.VARIANTS}; metrics[family]={'sigma68_fNL':w,'gain_p':1-w['joint']/w['p0'],'gain_xi':1-w['joint']/w['xi0']}
comp={'status':'pass' if all(x['chains_converged'] for x in results.values()) else 'needs_longer_chains','nmock':426,'results':results,'metrics':metrics}; m.write(root/'fits/comparison.json',comp); m.plot(root,comp); print(json.dumps(metrics,indent=2),flush=True)
