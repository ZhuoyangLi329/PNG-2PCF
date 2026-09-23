#!/usr/bin/env python3
"""Task43 RSD P02/xi02/joint test with xi smax=200, using 1000 mocks."""
from __future__ import annotations
import argparse, fcntl, hashlib, json, os, sys, time
from dataclasses import replace
from datetime import datetime, timezone
from pathlib import Path
import numpy as np

for v in ("OMP_NUM_THREADS","MKL_NUM_THREADS","OPENBLAS_NUM_THREADS","NUMEXPR_NUM_THREADS"):
    os.environ[v]="1"
os.environ.update(JAX_PLATFORMS="cpu",CUDA_VISIBLE_DEVICES="")
PROJECT=Path('/pscratch/sd/l/lzy/PNG-2PCF-model-to-observe'); sys.path.insert(0,str(PROJECT/'codes/task43'))
import task43_rsd_ezmock281_inference as base
PRODUCTION=base.PRODUCTION; REFERENCE=base.REFERENCE; MANIFEST=base.MANIFEST
SMAX=200.; VARIANTS={'p02':'rsd_p02','xi02':'rsd_xi02','joint':'rsd_joint_p02xi02'}
SLICES={'p02':slice(0,22),'xi02':slice(22,44),'joint':slice(0,44)}
NAMES=base.NAMES

def digest(p): return base.digest(p)
def write(p,x): base.write_json(p,x)

def prepare(root,nmock):
    fp,cp=root/'freeze.json',root/'covariance.npz'
    if fp.exists():
        a=json.loads(fp.read_text()); assert a['nmock']==nmock and digest(cp)==a['payload_sha256']
        with np.load(cp,allow_pickle=False) as z: return a,{k:np.asarray(z[k]) for k in z.files}
    aud=json.loads(base.REFERENCE_AUDIT.read_text())
    with np.load(base.REFERENCE_COV,allow_pickle=False) as z:
        analytic_full=np.asarray(z['rsd_joint']); s=np.asarray(z['rsd_s']); fullmask=np.asarray(z['rsd_xi_mask'],bool); p2keep=np.asarray(z['rsd_p2_keep_indices'],int)
    mask=(s>=50)&(s<SMAX)&~((s>=80)&(s<120)); assert mask.sum()==11
    xi_idx=np.r_[np.flatnonzero(mask),26+np.flatnonzero(mask)]; vector_keep=np.r_[np.arange(22),22+xi_idx]
    edges=np.asarray(aud['rsd']['pk_fit_edges_h_mpc'],float); rr=PRODUCTION/'fcfc_pairs/RR_common50_zobs0p4_0p8_s30_350_ds10_mu120.bin'; rrhash=digest(rr)
    rows=sorted((json.loads(x) for x in MANIFEST.read_text().splitlines() if x.strip()),key=lambda x:x['production_index']); frozen=[]; stack=[]
    for row in rows:
        xp,pp=Path(row['xi_path']),Path(row['pk_path']); files=(xp,pp,xp.with_suffix('.json'),pp.with_suffix('.json'))
        if not all(p.is_file() and p.stat().st_size for p in files): continue
        xm,pm=json.loads(files[2].read_text()),json.loads(files[3].read_text())
        if xm.get('status')!='done' or pm.get('status')!='done' or xm.get('ells')!=[0,2] or xm.get('mu_bin_num')!=120 or xm.get('common_rr_smu_sha256')!=rrhash: continue
        seed=int(row['seed']); assert seed==600001+int(row['production_index']) and row['fix_amplitude'] is False
        with np.load(pp,allow_pickle=False) as p,np.load(xp,allow_pickle=False) as x:
            ids=[]
            for e in edges:
                h=np.flatnonzero(np.all(np.isclose(p['k_edges0'],e[None,:],rtol=0,atol=1e-12),axis=1)); assert len(h)==1; ids.append(int(h[0]))
            ids=np.asarray(ids); vec=np.r_[p['pk0'][ids],p['pk2'][ids][p2keep],x['xi0'][mask],x['xi2'][mask]]; assert vec.shape==(44,) and np.isfinite(vec).all()
            common=str(p['common_random_sha256']); assert common==str(x['common_random_npz_sha256'])
        frozen.append({'production_index':int(row['production_index']),'seed':seed,'common_random_sha256':common,'paths':{str(p):digest(p) for p in files}}); stack.append(vec)
        if len(stack)==nmock: break
    assert len(stack)==nmock,f'only {len(stack)} valid mocks'
    stack=np.asarray(stack,float); cov=np.cov(stack,rowvar=False,ddof=1); payload={'stack':stack,'covariance_ezmock':cov,'covariance_jaxpower':analytic_full[np.ix_(vector_keep,vector_keep)],'s':s,'xi_mask':mask,'p2_keep':p2keep,'indices':np.asarray([x['production_index'] for x in frozen]),'seeds':np.asarray([x['seed'] for x in frozen])}
    for v,spec in VARIANTS.items():
        f=REFERENCE/'fits'/spec/'samples.npz'
        with np.load(f,allow_pickle=False) as z:
            d=np.asarray(z['data']); ph=np.asarray(z['phase_data'])
        if v=='xi02': d=d[xi_idx]; ph=ph[xi_idx] if ph.ndim==1 else ph[...,xi_idx]
        if v=='joint': d=d[vector_keep]; ph=ph[vector_keep] if ph.ndim==1 else ph[...,vector_keep]
        payload[f'data_{v}']=d; payload[f'phase_data_{v}']=ph
    assert payload['data_joint'].shape==(44,)
    audit={'status':'frozen','nmock':nmock,'smin':50.,'smax':SMAX,'xi_bins_per_ell':11,'data_dimensions':{'p02':22,'xi02':22,'joint':44},'rows':frozen,'manifest_sha256':digest(MANIFEST),'rr_sha256':rrhash,'payload_sha256':None,'corrections':{v:base.corrections(nmock,len(payload[f'data_{v}']),len(NAMES[v])) for v in VARIANTS},'created_utc':datetime.now(timezone.utc).isoformat()}
    base.save_npz(cp,**payload); audit['payload_sha256']=digest(cp); write(fp,audit); return audit,payload

def fit_one(root,audit,payload,args,family,v,spec):
    from task43_run_lightcone_joint_baomask_v1 import fit_maximum_likelihood,run_chain,FitSpec
    full=payload['covariance_ezmock' if family=='ezmock281' else 'covariance_jaxpower']; cov=full[SLICES[v],SLICES[v]]; fac=audit['corrections'][v] if family=='ezmock281' else {'hartlap':1.,'percival_m1':1.,'sigma_factor':1.}
    if v=='xi02':
        keep=np.r_[np.flatnonzero(payload['xi_mask']),26+np.flatnonzero(payload['xi_mask'])]
        spec=replace(spec,data=spec.data[keep],phase_data=spec.phase_data[...,keep] if spec.phase_data.ndim>1 else spec.phase_data[keep],evaluate=lambda t,sp=spec,k=keep: np.asarray(sp.evaluate(t))[k],covariance=cov)
    elif v=='joint':
        xkeep=np.r_[np.flatnonzero(payload['xi_mask']),26+np.flatnonzero(payload['xi_mask'])]; keep=np.r_[np.arange(22),22+xkeep]
        spec=replace(spec,data=spec.data[keep],phase_data=spec.phase_data[...,keep] if spec.phase_data.ndim>1 else spec.phase_data[keep],evaluate=lambda t,sp=spec,k=keep: np.asarray(sp.evaluate(t))[k],covariance=cov)
    else: spec=replace(spec,covariance=cov)
    metric=base.FastMetric(cov,fac['hartlap']); nominal,theta=fit_maximum_likelihood(spec,metric); tag=f'{family}_{v}'; pieces=[]; infos=[]
    for ri in range(args.ensembles):
        seed=args.seed+(0 if family=='ezmock281' else 10000)+({'p02':0,'xi02':1000,'joint':2000}[v])+ri*100; out=root/'fits'/tag/f'run{ri}'; jp,npz=out.with_suffix('.json'),out.with_suffix('.npz'); cfg={'nwalkers':args.nwalkers,'nsteps':args.nsteps,'burnin':args.burnin,'seed':seed,'freeze_sha256':audit['payload_sha256']}
        if jp.exists(): info=json.loads(jp.read_text()); assert info['config']==cfg and info['chain_sha256']==digest(npz); chain=np.load(npz,allow_pickle=False)['chain']; info=info
        else:
            sm,chain,logp=run_chain(spec,metric,theta,nwalkers=args.nwalkers,nsteps=args.nsteps,burnin=args.burnin,seed=seed,nworkers=1); base.save_npz(npz,chain=chain,logp=logp,parameter_names=np.asarray(spec.parameter_names),data=spec.data,theta_map=theta); info={'config':cfg,'mcmc':sm,'chain_sha256':digest(npz)}; write(jp,info)
        pieces.append(np.asarray(chain).reshape(-1,len(spec.parameter_names))); infos.append(info['mcmc'])
    flat=np.vstack(pieces); raw=base.posterior(flat,spec.parameter_names); cor=base.corrected_posterior(raw,fac['sigma_factor']); result={'parameter_names':list(spec.parameter_names),'ndata':len(spec.data),'map':nominal,'factors':fac,'posterior_hartlap':raw,'posterior_corrected':cor,'parameter_covariance_raw':np.cov(flat,rowvar=False),'parameter_covariance_corrected':fac['percival_m1']*np.cov(flat,rowvar=False),'chains_converged':all(all(x['gates'].values()) for x in infos),'runs':infos}; write(root/'fits'/tag/'summary.json',result); return result

def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--output-root',type=Path,required=True); ap.add_argument('--nmock',type=int,default=1000); ap.add_argument('--families',default='ezmock281,jaxpower'); ap.add_argument('--nwalkers',type=int,default=48); ap.add_argument('--nsteps',type=int,default=16000); ap.add_argument('--burnin',type=int,default=4000); ap.add_argument('--ensembles',type=int,default=4); ap.add_argument('--seed',type=int,default=20260915); args=ap.parse_args(); os.sched_setaffinity(0,sorted(os.sched_getaffinity(0))[:8]); root=args.output_root; root.mkdir(parents=True,exist_ok=True)
    with (root/'run.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB); audit,payload=prepare(root,args.nmock); specs,_,_=__import__('task43_run_lightcone_joint_baomask_v1',fromlist=['load_rsd_specs']).load_rsd_specs(smin=50.,pk_kmax=.08); smap={s.name:s for s in specs}; results={}
        for fam in [x.strip() for x in args.families.split(',') if x.strip()]:
            for v,sn in VARIANTS.items(): results[f'{fam}_{v}']=fit_one(root,audit,payload,args,fam,v,smap[sn])
        metrics={}
        for fam in [x.strip() for x in args.families.split(',') if x.strip()]:
            w={v:results[f'{fam}_{v}']['posterior_corrected']['fNL']['sigma68'] for v in VARIANTS}; metrics[fam]={'sigma68_fNL':w,'gain_p':1-w['joint']/w['p02'],'gain_xi':1-w['joint']/w['xi02']}
        write(root/'fits/comparison.json',{'status':'pass' if all(r['chains_converged'] for r in results.values()) else 'needs_longer_chains','nmock':args.nmock,'smax':SMAX,'results':results,'metrics':metrics}); print(json.dumps(metrics,indent=2),flush=True)
if __name__=='__main__': main()
