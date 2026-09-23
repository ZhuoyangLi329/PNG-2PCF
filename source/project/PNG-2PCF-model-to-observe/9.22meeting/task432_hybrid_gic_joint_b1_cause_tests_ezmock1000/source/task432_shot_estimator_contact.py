#!/usr/bin/env python3
"""Deterministic contact injection through the production P02 estimator.

Generate an independent weighted Poisson catalog in an octant shell. Split
each data particle into two coincident half-weight particles: the painted
field and normalization remain identical, while the subtracted self-pair
term drops by half the data contribution. This isolates the estimator's
contact response without interpreting a noisy Poisson realization as zero.
"""
import os,json,time,hashlib
from pathlib import Path
import numpy as np
import jax
jax.config.update('jax_enable_x64',True)
from clustering_statistics import spectrum2_tools as st
from task43_pk_common import column_edges,make_k_edges,extract_spectrum_arrays

ROOT=Path('/pscratch/sd/l/lzy/PNG-2PCF-model-to-observe')
OUT=ROOT/'outputs/task43_outputs/rsd_validation/task432_model_repair/hybrid_gic_joint_b1_cause_tests_v1'

def catalog(n,rng):
    r=(1100.**3+rng.random(n)*(1900.**3-1100.**3))**(1/3)
    direction=abs(rng.normal(size=(n,3)));direction/=np.linalg.norm(direction,axis=1)[:,None]
    return {'POSITION':r[:,None]*direction,'INDWEIGHT':.3+.1*(r-1100)/800,'Z':.4+.4*(r-1100)/800,'TARGETID':np.arange(n,dtype='i8')}

def main():
    start=time.time();rng=np.random.default_rng(4320922);D=catalog(60000,rng);R=catalog(300000,rng)
    E={k:np.concatenate([v,v],axis=0) for k,v in D.items()};E['INDWEIGHT']*=.5;E['TARGETID']=np.arange(len(E['Z']),dtype='i8')
    mattrs={'boxsize':np.full(3,2700.),'boxcenter':np.full(3,950.),'meshsize':np.full(3,128)}
    outputs=[]
    for data in (D,E):
        spectrum=st.compute_mesh2_spectrum(lambda:{'data':data,'randoms':R},mattrs=mattrs,edges=column_edges(make_k_edges(.001,.081,.002)),ells=(0,2),los='local',optimal_weights=None,norm={'cellsize':10.})
        outputs.append([extract_spectrum_arrays(spectrum,ell=ell) for ell in (0,2)])
    expected=.5*np.sum(D['INDWEIGHT']**2)/outputs[0][0]['norm'];d0=outputs[1][0]['pk0']-outputs[0][0]['pk0'];d2=outputs[1][1]['pk0']-outputs[0][1]['pk0']
    rel=float(np.max(abs(d0-expected))/np.max(abs(expected)));quad=float(np.max(abs(d2))/np.max(abs(expected)))
    normerr=float(np.max(abs(outputs[1][0]['norm']/outputs[0][0]['norm']-1)))
    result={'scope':'Deterministic coincident-pair contact injection on a synthetic weighted Poisson catalog; not a test that halo stochasticity is white, nor a zero-mean Poisson/GIC test.',
        'seed':4320922,'data_n':len(D['Z']),'random_n':len(R['Z']),'mesh':128,'affinity':sorted(os.sched_getaffinity(0)),
        'monopole_relative_error':rel,'quadrupole_relative_leakage':quad,'normalization_relative_error':normerr,
        'gate':rel<1e-8 and quad<1e-8 and normerr<1e-10,'elapsed_sec':time.time()-start,
        'runner_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),'estimator_source':st.__file__,'estimator_sha256':hashlib.sha256(Path(st.__file__).read_bytes()).hexdigest()}
    np.savez_compressed(OUT/'shot_contact_injection.npz',k=outputs[0][0]['k_obs'],expected_P0=expected,delta_P0=d0,delta_P2=d2)
    (OUT/'shot_contact_injection.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result),flush=True)
    assert result['gate'],result

if __name__=='__main__':main()
