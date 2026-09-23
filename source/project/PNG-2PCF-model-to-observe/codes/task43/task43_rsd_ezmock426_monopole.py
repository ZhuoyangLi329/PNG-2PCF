#!/usr/bin/env python3
"""426-realization RSD monopole-only MCMC: P0, xi0, and P0+xi0."""
from __future__ import annotations
import argparse, fcntl, json, os, sys, time
from datetime import datetime, timezone
from pathlib import Path
import numpy as np

for v in ("OMP_NUM_THREADS","MKL_NUM_THREADS","OPENBLAS_NUM_THREADS","NUMEXPR_NUM_THREADS"):
    os.environ[v] = "1"
os.environ.update(JAX_PLATFORMS="cpu", CUDA_VISIBLE_DEVICES="")
PROJECT=Path('/pscratch/sd/l/lzy/PNG-2PCF-model-to-observe')
sys.path.insert(0,str(PROJECT/'codes/task43'))
import task43_rsd_ezmock281_inference as base
PRODUCTION=base.PRODUCTION
REFERENCE=base.REFERENCE
MANIFEST=PRODUCTION/'manifests/task43_ezmock_rsd_covariance_x1000_fixampF_common50.jsonl'
VARIANTS={'p0':'rsd_p0','xi0':'rsd_xi0','joint':'rsd_joint_p0xi0'}
SLICES={'p0':slice(0,13),'xi0':slice(13,39),'joint':slice(0,39)}
NAMES={'p0':('fNL','b1','sigma_s','sn0'),'xi0':('fNL','b1','sigma_s'),'joint':('fNL','b1','sigma_s','sn0')}

def write(path,obj): base.write_json(path,obj)
def digest(path): return base.digest(path)

def prepare(root,nmock):
    payload_path=root/'covariance.npz'; audit_path=root/'freeze.json'
    if audit_path.exists():
        audit=json.loads(audit_path.read_text())
        with np.load(payload_path,allow_pickle=False) as z: payload={k:np.asarray(z[k]) for k in z.files}
        if audit['nmock']!=nmock or digest(payload_path)!=audit['payload_sha256']: raise RuntimeError('freeze mismatch')
        return audit,payload
    refcov=REFERENCE/'task43_lightcone_standard_joint_baomask80_120_covariance_v1.npz'
    refaudit=REFERENCE/'task43_lightcone_standard_joint_baomask80_120_v1.json'
    with np.load(refcov,allow_pickle=False) as z:
        analytic=np.asarray(z['rsd_joint0']); s=np.asarray(z['rsd_s']); mask=np.asarray(z['rsd_xi_mask'])
    edges=np.asarray(json.loads(refaudit.read_text())['rsd']['pk_fit_edges_h_mpc'],dtype='f8')
    assert analytic.shape==(39,39) and mask.sum()==26 and edges.shape==(13,2)
    rr=PRODUCTION/'fcfc_pairs/RR_common50_zobs0p4_0p8_s30_350_ds10_mu120.bin'; rr_hash=digest(rr)
    rows=sorted((json.loads(x) for x in MANIFEST.read_text().splitlines() if x.strip()),key=lambda x:x['production_index'])
    frozen=[]; stack=[]
    for row in rows:
        xp,pp=Path(row['xi_path']),Path(row['pk_path']); files=(xp,pp,xp.with_suffix('.json'),pp.with_suffix('.json'))
        if not all(p.is_file() and p.stat().st_size for p in files): continue
        xm,pm=json.loads(files[2].read_text()),json.loads(files[3].read_text())
        if xm.get('status')!='done' or pm.get('status')!='done': continue
        idx,seed=int(row['production_index']),int(row['seed']); assert seed==600001+idx and row['fix_amplitude'] is False
        assert xm['ells']==[0,2] and xm['mu_bin_num']==120 and xm['common_rr_smu_sha256']==rr_hash
        with np.load(pp,allow_pickle=False) as p,np.load(xp,allow_pickle=False) as x:
            assert np.array_equal(x['s'],s) and str(x['common_rr_smu_sha256'])==rr_hash
            found=[]
            for e in edges:
                hits=np.flatnonzero(np.all(np.isclose(p['k_edges0'],e[None,:],rtol=0,atol=1e-12),axis=1)); assert len(hits)==1; found.append(int(hits[0]))
            vec=np.r_[p['pk0'][found],x['xi0'][mask]]; assert vec.shape==(39,) and np.isfinite(vec).all()
            common=str(p['common_random_sha256']); assert common==str(x['common_random_npz_sha256'])
        frozen.append(dict(production_index=idx,seed=seed,common_random_sha256=common,paths={str(p):digest(p) for p in files})); stack.append(vec)
        if len(stack)==nmock: break
    if len(stack)!=nmock: raise RuntimeError(f'only {len(stack)} valid paired mocks')
    stack=np.asarray(stack,float); cov=np.cov(stack,rowvar=False,ddof=1)
    payload={'stack':stack,'covariance_ezmock':cov,'covariance_jaxpower':analytic,'s':s,'xi_mask':mask,'edges':edges,
             'indices':np.asarray([r['production_index'] for r in frozen]),'seeds':np.asarray([r['seed'] for r in frozen])}
    source_hashes={str(refcov):digest(refcov),str(refaudit):digest(refaudit)}
    for v,spec in VARIANTS.items():
        f=REFERENCE/'fits'/spec/'samples.npz'
        with np.load(f,allow_pickle=False) as z:
            payload[f'data_{v}']=np.asarray(z['data']); payload[f'phase_data_{v}']=np.asarray(z['phase_data'])
        source_hashes[str(f)]=digest(f)
    assert np.allclose(payload['data_joint'],np.r_[payload['data_p0'],payload['data_xi0']],rtol=0,atol=1e-12)
    audit={'status':'frozen','nmock':nmock,'created_utc':datetime.now(timezone.utc).isoformat(),'rows':frozen,
           'manifest_sha256':digest(MANIFEST),'rr_sha256':rr_hash,'source_hashes':source_hashes,
           'data_dimensions':{'p0':13,'xi0':26,'joint':39},'payload_sha256':None,
           'corrections':{v:base.corrections(nmock,len(payload[f'data_{v}']),len(NAMES[v])) for v in VARIANTS}}
    base.save_npz(payload_path,**payload); audit['payload_sha256']=digest(payload_path); write(audit_path,audit)
    return audit,payload

def posterior_result(spec,metric,payload,audit,root,family,v, args):
    from task43_run_lightcone_joint_baomask_v1 import fit_maximum_likelihood,run_chain
    cov=payload['covariance_ezmock' if family=='ezmock' else 'covariance_jaxpower'][SLICES[v],SLICES[v]]
    factor=audit['corrections'][v] if family=='ezmock' else {'hartlap':1.,'percival_m1':1.,'sigma_factor':1.}
    spec=spec.__class__(spec.name,spec.group,spec.data,spec.phase_data,spec.evaluate,cov,spec.parameter_names,spec.starts)
    nominal,theta_map=fit_maximum_likelihood(spec,metric(cov,factor['hartlap']))
    pieces=[]; infos=[]; tag=f'{family}_{v}'
    for ri in range(args.ensembles):
        seed=args.seed+(0 if family=='ezmock' else 10000)+({'p0':0,'xi0':1000,'joint':2000}[v])+ri*100
        out=root/'fits'/tag/f'run{ri}'; jp=out.with_suffix('.json'); npz=out.with_suffix('.npz')
        cfg={'nwalkers':args.nwalkers,'nsteps':args.nsteps,'burnin':args.burnin,'seed':seed,'freeze_sha256':audit['payload_sha256']}
        if jp.exists():
            info=json.loads(jp.read_text()); assert info['config']==cfg and info['chain_sha256']==digest(npz)
            with np.load(npz,allow_pickle=False) as z: chain=np.asarray(z['chain'])
        else:
            sm,chain,logp=run_chain(spec,metric(cov,factor['hartlap']),theta_map,nwalkers=args.nwalkers,nsteps=args.nsteps,burnin=args.burnin,seed=seed,nworkers=1)
            base.save_npz(npz,chain=chain,logp=logp,parameter_names=np.asarray(spec.parameter_names),data=spec.data,theta_map=theta_map)
            info={'config':cfg,'mcmc':sm,'chain_sha256':digest(npz)}; write(jp,info)
        pieces.append(chain.reshape(-1,len(spec.parameter_names))); infos.append(info['mcmc'])
    flat=np.vstack(pieces); raw=base.posterior(flat,spec.parameter_names); corrected=base.corrected_posterior(raw,factor['sigma_factor'])
    result={'parameter_names':list(spec.parameter_names),'ndata':len(spec.data),'map':nominal,'factors':factor,'posterior_hartlap':raw,'posterior_corrected':corrected,
            'parameter_covariance_raw':np.cov(flat,rowvar=False),'parameter_covariance_corrected':factor['percival_m1']*np.cov(flat,rowvar=False),
            'chains_converged':all(all(x['gates'].values()) for x in infos),'runs':infos}
    write(root/'fits'/tag/'summary.json',result); return result

def plot(root,comparison):
    import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
    from scipy.ndimage import gaussian_filter
    cols={'p0':'#2763a3','xi0':'#329457','joint':'#bf2f3b'}; labels={'p0':r'$P_0$', 'xi0':r'$\xi_0$', 'joint':'Joint'}
    nmock=int(comparison['nmock']); fig,axes=plt.subplots(1,3,figsize=(16.5,5.3),constrained_layout=True)
    for family,ls in (('ezmock','-'),('jaxpower','--')):
      for v in VARIANTS:
        tag=f'{family}_{v}'; sm=comparison['results'][tag]; pieces=[]
        for f in sorted((root/'fits'/tag).glob('run*.npz')):
          with np.load(f,allow_pickle=False) as z: pieces.append(np.asarray(z['chain']).reshape(-1,len(sm['parameter_names']))[:,:2])
        points=np.vstack(pieces); center=np.median(points,axis=0); points=center+(points-center)*sm['factors']['sigma_factor']; lim=np.percentile(points,[.05,99.95],axis=0)
        h,xe,ye=np.histogram2d(points[:,0],points[:,1],bins=110,range=lim.T.tolist()); d=gaussian_filter(h,1.2); o=np.sort(d.ravel())[::-1]; c=np.cumsum(o)/o.sum(); lev=np.sort([o[np.searchsorted(c,p)] for p in (.6827,.9545)]); x=(xe[1:]+xe[:-1])/2; y=(ye[1:]+ye[:-1])/2
        pf=sm['posterior_corrected']['fNL']; val=rf'$f_{{\rm NL}}={pf["q50"]:+.1f}^{{+{pf["q84"]-pf["q50"]:.1f}}}_{{-{pf["q50"]-pf["q16"]:.1f}}}$'; fam=f'EZmock N={nmock}' if family=='ezmock' else 'analytic jaxpower'
        for ax in ([axes[0] if family=='ezmock' else axes[1]] + ([axes[2]] if v=='joint' else [])):
          ax.contour(x,y,d.T,levels=lev,colors=cols[v],linestyles=ls,linewidths=[1.1,1.8]); ax.plot([],[],color=cols[v],ls=ls,label=f'{labels[v]} ({val})' if ax is not axes[2] else f'{fam} ({val})')
    for ax,title in zip(axes,(f'EZmock N={nmock}: Hartlap + Percival','Analytic jaxpower covariance','Joint contour comparison')):
      ax.set(xlabel=r'$f_{\rm NL}$',ylabel=r'$b_1$',title=title); ax.legend(frameon=False,fontsize=9); ax.grid(alpha=.15)
    xmin=min(a.get_xlim()[0] for a in axes[:2]); xmax=max(a.get_xlim()[1] for a in axes[:2]); ymin=min(a.get_ylim()[0] for a in axes[:2]); ymax=max(a.get_ylim()[1] for a in axes[:2]); [a.set(xlim=(xmin,xmax),ylim=(ymin,ymax)) for a in axes]
    m=comparison['metrics']; fig.suptitle(f'Task43 RSD lightcone · monopole-only covariance · 68% / 95% contours\nMedian and 16th-84th percentile errors; marginalized nuisances\nEZmock: joint reduction vs P0={100*m["ezmock"]["gain_p"]:.1f}%, vs xi0={100*m["ezmock"]["gain_xi"]:.1f}% | Analytic: {100*m["jaxpower"]["gain_p"]:.1f}%, {100*m["jaxpower"]["gain_xi"]:.1f}%',fontsize=11)
    out=root/f'contours_ezmock{nmock}_monopole_vs_jaxpower.pdf'; fig.savefig(out); plt.close(fig); print(out)

def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--output-root',type=Path,default=PRODUCTION/'mcmc_monopole_426'); ap.add_argument('--nmock',type=int,default=426); ap.add_argument('--nwalkers',type=int,default=48); ap.add_argument('--nsteps',type=int,default=12000); ap.add_argument('--burnin',type=int,default=3000); ap.add_argument('--ensembles',type=int,default=4); ap.add_argument('--seed',type=int,default=20260915); args=ap.parse_args(); args.output_root.mkdir(parents=True,exist_ok=True); os.sched_setaffinity(0,sorted(os.sched_getaffinity(0))[:8])
    with (args.output_root/'run.lock').open('a') as lock:
      fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB); audit,payload=prepare(args.output_root,args.nmock)
      from task43_run_lightcone_joint_baomask_v1 import load_rsd_specs
      specs,meta,arr=load_rsd_specs(smin=50.,pk_kmax=.08); smap={x.name:x for x in specs}; results={}
      for fam in ('ezmock','jaxpower'):
       for v,sn in VARIANTS.items(): results[f'{fam}_{v}']=posterior_result(smap[sn],base.FastMetric,payload,audit,args.output_root,fam,v,args)
      metrics={}
      for fam in ('ezmock','jaxpower'):
       w={v:results[f'{fam}_{v}']['posterior_corrected']['fNL']['sigma68'] for v in VARIANTS}; metrics[fam]={'sigma68_fNL':w,'gain_p':1-w['joint']/w['p0'],'gain_xi':1-w['joint']/w['xi0']}
      comp={'status':'pass' if all(x['chains_converged'] for x in results.values()) else 'needs_longer_chains','nmock':args.nmock,'results':results,'metrics':metrics}; write(args.output_root/'fits/comparison.json',comp); plot(args.output_root,comp); print(json.dumps(metrics,indent=2))
if __name__=='__main__': main()
