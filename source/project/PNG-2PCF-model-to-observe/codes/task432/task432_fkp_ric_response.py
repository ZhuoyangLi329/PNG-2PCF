#!/usr/bin/env python3
"""FKP配对的权重相关full-RIC模型变化；只在固定fiducial作诊断。

大纲：同一Sobol几何两种权重 -> cross+auto核 -> 既有P/GSM动力学 ->
更新data/random Poisson与既有sn0投影 -> 两个scramble检查差分精度。
这不是新MCMC模型；每分支实际I2与RR来自对应的真实配对测量。
"""
import os
for k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):os.environ[k]='1'
import argparse,json,sys,time
from pathlib import Path
import numpy as np
from scipy.stats import qmc
from scipy.interpolate import RectBivariateSpline
ROOT=Path('/pscratch/sd/l/lzy/PNG-2PCF-model-to-observe')
for folder in ('codes/task43','codes/task432','codes/task44'):sys.path.insert(0,str(ROOT/folder))
from task432_fkp_conditional_modes import OUT,THETA
from task432_fkp_paired import prepare,table_weight
from task432_full_ric_backend import OUT as RIC,counts
from task432_full_ric_geometry import BASE,separation_edges,sha
from task432_full_ric_response import Response
from task432_full_ric_shot import shot_counts
from task432_hybrid_jaxpower_0918_contract import save
from task43_rsd_boxsafe_p02_increment import WindowConvolvedP02Model,PAYLOAD_NPZ

def samples(a,n,seed):
    """同data红移分布、三独立Sobol取样；两FKP策略严格使用共同随机数。"""
    from cosmoprimo.fiducial import AbacusSummit
    z=np.sort(a['data_redshift'].copy());ze=a['z_edges']
    z=np.clip(z,np.nextafter(ze[0],np.inf),np.nextafter(ze[-1],-np.inf))
    zg=np.linspace(ze[0],ze[-1],200001);cg=AbacusSummit(0).comoving_radial_distance(zg)
    cats=[];bins=[]
    for i in range(3):
        u=qmc.Sobol(3,scramble=True,seed=seed+i*104729).random_base2(int(np.log2(n)))
        zz=z[np.minimum((u[:,0]*len(z)).astype(int),len(z)-1)];r=np.interp(zz,zg,cg)
        phi=u[:,1]*np.pi/2;mu=u[:,2];tr=np.sqrt(1-mu*mu)
        xyz=r[:,None]*np.column_stack([tr*np.cos(phi),tr*np.sin(phi),mu]);labels=np.floor(r/2).astype(int)
        cats.append((xyz,zz,labels));bins.append(np.unique(labels))
    assert all(np.array_equal(bins[0],b) for b in bins)
    return [(p,z,np.searchsorted(bins[0],labels)) for p,z,labels in cats]

def run(phase,n=16384,seed=4322407):
    """计算单phase、单scramble、两分支和两LOS的固定参数IC响应。"""
    dest=OUT/'paired'/phase;target=dest/f'ric_response_n{n}_seed{seed}.npz'
    if target.exists():return
    with np.load(prepare(phase)) as z:a={k:z[k] for k in z.files}
    with np.load(BASE/'frozen_inputs.npz') as z:pe=z['p_edges'];xc=z['xi_centers']
    with np.load(PAYLOAD_NPZ) as z:zeff=float(z['zeff'])
    with np.load(RIC/'geometry/hybrid_inner_refine1_q2_n600.npz') as z:
        ff,bb,poles=z['f_grid'],z['b_grid'],z['poles'];edges=z['separation_edges']
    # 和正式RIC相同的双三次插值，固定点无须重做整套动力学网格。
    xpol=np.array([RectBivariateSpline(ff,bb,poles[:,:,ell,d],s=0).ev(THETA[0],THETA[1])
                   for ell in range(3) for d in range(len(edges)-1)]).reshape(3,-1)
    assert np.array_equal(edges,separation_edges(1))
    cats=samples(a,n,seed);responses={};checks=[];kept=np.r_[np.arange(13),np.arange(17,26)]
    for branch in ('own','independent'):
        table=a[branch+'_table'];current=[(p,table_weight(z,a['z_edges'],table),labels) for p,z,labels in cats]
        with np.load(dest/f'{branch}_pk.npz') as z:
            norm=float(z['norms'][0].ravel()[0]);tk=z['theory_k'];te=z['theory_ell'];tedges=z['theory_edges']
        dw=a['data_base']*table_weight(a['data_redshift'],a['z_edges'],table)
        rw=a['random_base']*table_weight(a['random_redshift'],a['z_edges'],table)
        pairvol=dw.sum()**2/norm;alpha=dw.sum()/rw.sum();da=dw@dw/norm;rp=alpha*alpha*(rw@rw)/norm
        rr=[];rx=[];nd=len(dw)
        for p in sorted(dest.glob(f'{branch}_xi_block*.npz')):
            block=int(p.stem.split('block')[-1]);w=rw[block*nd:(block+1)*nd]
            with np.load(p) as z:
                ids=[np.flatnonzero(np.isclose(z['s'],s,rtol=0,atol=1e-8))[0] for s in xc]
                rr.append(z['RR'][ids]);rx.append(pairvol*(w@w)/w.sum()**2)
        rr=np.mean(rr,axis=0);rx=float(np.mean(rx))
        pmodel=WindowConvolvedP02Model(np.zeros((1,len(tk))),tk,te,zeff=zeff,sigma_step=30.)
        f,b,sig,sn=THETA;q=f*2*1.686*(b-1);g=pmodel.f_growth
        theory=pmodel._theory_basis(sig)@np.array([b*b,2*b*q,q*q,2*b*g,2*q*g,g*g])
        components=[]
        for los,maxell in (('endpoint',2),('midpoint',8)):
            started=time.monotonic();ells=tuple(range(0,maxell+1,2))
            path=dest/f'kernel_{branch}_n{n}_seed{seed}_{los}.npz'
            if not path.exists():
                r=counts(*current[0],edges,ells_out=ells,los_out=los,los_in='midpoint',nthreads=8,progress=False,independent_projection=current[1:])
                error=max(np.max(abs(r['cross'][:,0].sum(-1)-2*r['rr'])),np.max(abs(r['auto'][:,0].sum(-1)-r['rr'])))/np.max(abs(r['rr']))
                assert error<1e-10
                meta={k:v for k,v in r.items() if k not in ('rr','cross','auto')};meta.update(pair_volume=pairvol,phase=phase,branch=branch,seed=seed,nsub=n,constant_mode_error=float(error))
                np.savez_compressed(path,separation_edges=edges,rr=r['rr'],cross=r['cross'],auto=r['auto'],normalization=r['normalization'],meta_json=np.array(json.dumps(meta)))
            response=Response(path)
            projector=(lambda m:response.xi_project(m,xc,rr_smu=rr)) if los=='midpoint' else (lambda m:response.p_project(m,pe,nquad=16).reshape(26)[kept])
            if los=='midpoint':clustering=np.einsum('ild,ld->i',response.xi_matrix(xc,rr_smu=rr),xpol)
            else:clustering=response.pk_theory_matrix(pe,tedges,te,nquad=16,inner_nquad=16).reshape(26,-1)[kept]@theory
            # 白随机项rho=nbar；选择函数不变，仅FKP变，故两分支使用同一own_nbar。
            noise=[table_weight(z,a['z_edges'],a['own_nbar']) for _,z,_ in cats[:2]]
            shotparts=[]
            for label,width,density in (('data_radial',2,None),('random_global',0,None),('sn_radial',2,noise)):
                shotpath=dest/f'shot_{branch}_{label}_n{n}_seed{seed}_{los}.npz'
                if shotpath.exists():
                    with np.load(shotpath) as z:moment=z['moment']
                else:
                    ab=[(p,w,l if width else np.zeros(len(l),dtype=int)) for p,w,l in current[:2]]
                    r=shot_counts(*ab,edges,ells,los,nthreads=8,noise_density=density)
                    moment=r['auto']-r['cross'];assert abs(1+moment[0].sum())<1e-10
                    np.savez_compressed(shotpath,moment=moment)
                shotparts.append(projector(moment/pairvol))
            # 连续白噪声积分/I2：random按nbar抽样，因此W_D×E_R[nbar*w]。
            nb=table_weight(a['random_redshift'],a['z_edges'],a['own_nbar'])
            I2cont=alpha*np.sum(nb*rw**2)
            result=clustering+da*shotparts[0]+(rx if los=='midpoint' else rp)*shotparts[1]+sn*1e4*(I2cont/norm)*shotparts[2]
            components.append(result);checks.append({'branch':branch,'los':los,'I2':norm,'pair_volume':pairvol,'I2_cont_ratio':float(I2cont/norm),'elapsed_s':time.monotonic()-started})
            print(json.dumps({'event':'FKP_IC_response','phase':phase,'branch':branch,'los':los,'seed':seed,'elapsed_s':time.monotonic()-started}),flush=True)
        responses[branch]=np.concatenate(components)
    np.savez_compressed(target,own=responses['own'],independent=responses['independent'],difference=responses['independent']-responses['own'],theta=THETA)
    save(target.with_suffix('.json'),{'status':'diagnostic_response','phase':phase,'seed':seed,'nsub':n,'checks':checks,'source_sha256':sha(__file__),
        'scope':'Full clustering cross+auto, deterministic Poisson and existing sn0 IC at fixed fiducial. Compare paired differences across independent scrambles before interpretation.'})

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--phase',default='ph000');p.add_argument('--nsub',type=int,default=16384);p.add_argument('--seed',type=int,default=4322407);a=p.parse_args();run(a.phase,a.nsub,a.seed)
