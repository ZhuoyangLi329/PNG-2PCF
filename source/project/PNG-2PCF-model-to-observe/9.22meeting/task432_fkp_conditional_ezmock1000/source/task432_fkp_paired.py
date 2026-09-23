#!/usr/bin/env python3
"""同目录同random，仅替换FKP的实际P02/xi02测量，独立诊断不覆盖生产。

执行大纲：锁定phase和random子集 -> 自身/其余24phase平均nbar权重 ->
P使用各分支实际I2/shot和重算窗口 -> 相同N_R=N_D blocks的LS -> 输出小向量。
固定实际位置、mesh、k/s/mu选择；不能把配对差分的原C_single度量当显著性。
"""
import os
for key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS'):
    os.environ[key]='1'
os.environ['JAX_PLATFORMS']='cpu';os.environ['CUDA_VISIBLE_DEVICES']=''
import argparse, json, sys, time, hashlib
from pathlib import Path
import numpy as np
ROOT=Path('/pscratch/sd/l/lzy/PNG-2PCF-model-to-observe')
for folder in ('codes/task43','codes/task432'):sys.path.insert(0,str(ROOT/folder))
sys.path.insert(0,'/pscratch/sd/l/lzy/desi-clustering')
from task432_full_ric_geometry import HALO_MANIFEST,PK_DIR,BASE,sha
from task432_full_ric_backend import OUT as RIC
from task432_full_ric_pair_checks import count,project,EDGES
from task43_pk_common import mesh_attrs_for_jaxpower,extract_window_arrays
from task432_fkp_conditional_modes import OUT,THETA
from task432_hybrid_jaxpower_0918_contract import save

def table_weight(z,edges,table):
    """生产分段常数FKP，不作插值；参数z为观测红移数组。"""
    return table[np.clip(np.searchsorted(edges,z,side='right')-1,0,len(table)-1)]

def prepare(phase):
    """缓存4N random的共同位置；两个权重策略只保存40-bin表，不复制目录。"""
    dest=OUT/'paired'/phase;dest.mkdir(parents=True,exist_ok=True)
    path=dest/'catalogs.npz'
    if path.exists():return path
    rows=[json.loads(x) for x in HALO_MANIFEST.read_text().splitlines() if x.strip()]
    row=next(r for r in rows if r['phase']==phase)
    nbar=[];sources=[]
    for r in rows:
        with np.load(r['lightcone_fkp_path']) as a:
            nbar.append(a['nbar']);edges=a['z_edges'];p0=float(a['p0'])
            if r['phase']==phase:own=a['fkp_weights'];own_nbar=a['nbar']
        sources.append({'phase':r['phase'],'path':r['lightcone_fkp_path'],'sha256':sha(r['lightcone_fkp_path'])})
    independent_nbar=np.mean([n for r,n in zip(rows,nbar) if r['phase']!=phase],axis=0)
    independent=1/(1+p0*independent_nbar)
    with np.load(row['lightcone_catalog_path']) as a:
        dz=a['Z'].astype('f8');dp=np.column_stack([a[k] for k in ('X','Y','Zcart')]).astype('f4')
        db=a['WEIGHT'].astype('f8') if 'WEIGHT' in a.files else np.ones(len(dz))
    n=len(dz)
    # ph000已有完全相同的前4N位置缓存，直接复用以减少对大原始文件的读取。
    cache=RIC/'paired/halo_ph000/paired_catalogs.npz'
    if phase=='ph000' and cache.exists():
        with np.load(cache) as a:
            rp=a['shuffled_position'];rz=a['shuffled_redshift']
        rb=np.ones(len(rz))
    else:
        with np.load(row['lightcone_random_path']) as a:
            rp=np.column_stack([a[k][:4*n] for k in ('X','Y','Zcart')]).astype('f4')
            rz=a['Z'][:4*n].astype('f8');rb=a['WEIGHT'][:4*n].astype('f8')
            assert np.array_equal(a['RANDOM_INDEX'][:4*n],np.repeat(np.arange(4),n))
    assert len(rz)==4*n and np.all(db==1) and np.all(rb==1)
    pfile=PK_DIR/f'task43_rsd_lightcone_p02_{phase}_mesh256_kmax0p300_dk0p002.npz'
    pmeta=json.loads(pfile.with_suffix('.json').read_text())
    with np.load(pfile) as a:pe=a['k_edges']
    meta={'phase':phase,'ndata':n,'nrandom_P':4*n,'random_positions':'first4productionblocks; identical between weights',
          'data_path':row['lightcone_catalog_path'],'random_path':row['lightcone_random_path'],
          'xi_production_path':row['lightcone_xi_path'],'source_P':str(pfile),'mesh':pmeta['mesh'],
          'tables':sources,'independent_definition':'Unsmoothed binwise mean nbar from the other24 independent phases; same dz=.01; finite-selection error retained',
          'weights_changed_only':True,'normalization':'Re-estimated per weighting with identical data/random positions; no forced common I2',
          'position_hash_data':hashlib.sha256(dp.tobytes()).hexdigest(),'position_hash_random':hashlib.sha256(rp.tobytes()).hexdigest(),
          'weight_ratio_minmax':[float((independent/own).min()),float((independent/own).max())],
          'source_sha256':sha(__file__)}
    np.savez_compressed(path,data_position=dp,data_redshift=dz,data_base=db,random_position=rp,random_redshift=rz,random_base=rb,
        z_edges=edges,own_table=own,independent_table=independent,own_nbar=own_nbar,independent_nbar=independent_nbar,k_edges=pe,meta_json=np.array(json.dumps(meta)))
    save(dest/'catalogs.json',meta)
    print(json.dumps({'event':'prepared_FKP','phase':phase,'weight_ratio_minmax':meta['weight_ratio_minmax']}),flush=True)
    return path

def measure_pk(phase):
    """重测两分支P及当前smooth window；I2和shot取实际spectrum对象。"""
    path=prepare(phase);dest=path.parent
    import jax
    jax.config.update('jax_enable_x64',True)
    from clustering_statistics import spectrum2_tools
    with np.load(path) as a:arr={k:a[k] for k in a.files}
    meta=json.loads(str(arr['meta_json']));ze=arr['z_edges']
    for branch in ('own','independent'):
        output=dest/f'{branch}_pk.npz'
        if output.exists():continue
        table=arr[branch+'_table'];started=time.monotonic()
        dw=arr['data_base']*table_weight(arr['data_redshift'],ze,table)
        rw=arr['random_base']*table_weight(arr['random_redshift'],ze,table)
        data={'POSITION':arr['data_position'].astype('f8'),'INDWEIGHT':dw,'Z':arr['data_redshift']}
        random={'POSITION':arr['random_position'].astype('f8'),'INDWEIGHT':rw,'Z':arr['random_redshift'],'TARGETID':np.arange(len(rw))}
        def catalogs():
            """返回当前分支加权的同一目录供估计器和窗口共同使用。"""
            return {'data':data,'randoms':random}
        spectrum=spectrum2_tools.compute_mesh2_spectrum(catalogs,mattrs=mesh_attrs_for_jaxpower(meta['mesh']),
            edges=arr['k_edges'],ells=(0,2),los='local',optimal_weights=None,norm={'cellsize':10.})
        poles=np.array([np.asarray(spectrum.get(ell).value()) for ell in (0,2)])
        norms=np.array([np.asarray(spectrum.get(ell).values('norm')) for ell in (0,2)])
        shots=np.array([np.asarray(spectrum.get(ell).values('shotnoise')) for ell in (0,2)])
        alpha=dw.sum()/rw.sum();analytic_shot=(dw@dw+alpha*alpha*(rw@rw))/norms[0]
        assert np.max(abs(analytic_shot-shots[0])/analytic_shot)<1e-10
        window=spectrum2_tools.compute_window_mesh2_spectrum(catalogs,spectrum=spectrum,optimal_weights=None,method='smooth')
        window_arrays=extract_window_arrays(window)
        np.savez_compressed(output,multipoles=poles,k_edges=arr['k_edges'],norms=norms,shotnoise=shots,
            analytic_shot=analytic_shot,data_weight_sum=dw.sum(),random_weight_sum=rw.sum(),**window_arrays)
        save(output.with_suffix('.json'),{'phase':phase,'branch':branch,'elapsed_s':time.monotonic()-started,
            'source_sha256':sha(__file__),'normalization':float(norms[0].ravel()[0]),'cpu_affinity':sorted(os.sched_getaffinity(0)),
            'shot_relative_error':float(np.max(abs(analytic_shot-shots[0])/analytic_shot))})
        print(json.dumps({'event':'FKP_P_window','phase':phase,'branch':branch,'elapsed_s':time.monotonic()-started}),flush=True)

def measure_xi(phase,nblocks=2):
    """真实midpoint40mu LS，D/R随权重同步更新；每个block先做ratio再平均。"""
    path=prepare(phase);dest=path.parent
    with np.load(path) as a:arr={k:a[k] for k in a.files}
    ze=arr['z_edges'];n=len(arr['data_redshift'])
    assert nblocks<=4
    for branch in ('own','independent'):
        table=arr[branch+'_table']
        dw=(arr['data_base']*table_weight(arr['data_redshift'],ze,table)).astype('f4')
        rw=(arr['random_base']*table_weight(arr['random_redshift'],ze,table)).astype('f4')
        data=(arr['data_position'],dw);wd=dw.sum(dtype='f8')
        ddpath=dest/f'{branch}_DD.npz'
        if ddpath.exists():
            with np.load(ddpath) as a:dd=a['DD']
        else:
            # 保留原CPU桥接的同一基准，减少已验证重复计算。
            cached=RIC/'paired/halo_ph000/bridge_block0_DD.npz'
            if phase=='ph000' and branch=='own' and cached.exists():
                with np.load(cached) as a:dd=a['normalized_counts']
            else:dd=count(data)/(wd*wd)
            np.savez_compressed(ddpath,DD=dd)
        for block in range(nblocks):
            output=dest/f'{branch}_xi_block{block}.npz'
            if output.exists():continue
            started=time.monotonic();sl=slice(block*n,(block+1)*n)
            random=(arr['random_position'][sl],rw[sl]);wr=random[1].sum(dtype='f8')
            cached=RIC/f'paired/halo_ph000/shuffled_xi_block{block}.npz'
            if phase=='ph000' and branch=='own' and cached.exists():
                with np.load(cached) as a:dr=a['DR'];rr=a['RR']
            else:
                dr=count(data,random)/(wd*wr);rr=count(random)/(wr*wr)
            assert np.all(rr>0)
            xi=(dd-2*dr+rr)/rr
            np.savez_compressed(output,xi_smu=xi,xi_multipoles=project(xi),DD=dd,DR=dr,RR=rr,s=(EDGES[:-1]+EDGES[1:])/2)
            print(json.dumps({'event':'FKP_xi','phase':phase,'branch':branch,'block':block,'elapsed_s':time.monotonic()-started}),flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('action',choices=('prepare','pk','xi'))
    p.add_argument('--phase',default='ph000');p.add_argument('--blocks',type=int,default=2);a=p.parse_args()
    if a.action=='xi':measure_xi(a.phase,a.blocks)
    else:{'prepare':prepare,'pk':measure_pk}[a.action](a.phase)
