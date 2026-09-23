#!/usr/bin/env python3
"""对一个 halo 和真实 covariance 中的少量 EZ 做 shuffled/reference 配对测量。

执行大纲：共同角度/CDF分位数的径向配对 -> 固定 FKP、mesh、I2 ->
midpoint 40mu LS 与 CPU mesh256 P02 -> 保存四分量差分，不改任何生产文件。
配对使用4倍 random 的 P 和前2个等N LS blocks，旨在最小规模闭合，
不是重建正式25block数据或EZmock1000 covariance。
"""
import os
for key in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ[key] = '8' if key == 'OMP_NUM_THREADS' else '1'
os.environ['JAX_PLATFORMS'] = 'cpu'
os.environ['CUDA_VISIBLE_DEVICES'] = ''
import argparse
import json
import sys
import time
from pathlib import Path
import numpy as np
from task432_full_ric_backend import ROOT, OUT
from task432_full_ric_geometry import BASE, HALO_MANIFEST, PK_DIR, save_json, sha
from task432_full_ric_pair_checks import count, project, EDGES
for folder in ('codes/task43', 'codes/task432'):
    sys.path.insert(0, str(ROOT/folder))
sys.path.insert(0, '/pscratch/sd/l/lzy/desi-clustering')
from task43_measure_ezmock_rsd_lightcone_p02 import mean_fkp
from task43_pk_common import mesh_attrs_for_jaxpower


def root_for(name):
    """将每个实际 realization 的独立配对产物放入新目录。"""
    return OUT/'paired'/name


def fkp_weight(z, edges, table):
    """固定生产 FKP 表，只根据每个点的红移选 bin。"""
    return table[np.clip(np.searchsorted(edges, z, side='right')-1, 0, len(table)-1)]


def quantile(z, sorted_source, rng):
    """随机化 empirical-CDF ties，保持共同随机数并复现源分布抽样。"""
    low = np.searchsorted(sorted_source, z, side='left')
    high = np.searchsorted(sorted_source, z, side='right')
    return np.clip((low+rng.random(len(z))*np.maximum(high-low, 1))/len(sorted_source), 0, np.nextafter(1., 0.))


def prepare(name):
    """准备有限数量的成对目录，原始数据和协方差始终只读。"""
    dest = root_for(name); dest.mkdir(parents=True, exist_ok=True)
    target = dest/'paired_catalogs.npz'
    if target.exists():
        return target
    rows = [json.loads(line) for line in HALO_MANIFEST.read_text().splitlines() if line.strip()]
    halo = name == 'halo_ph000'
    if halo:
        row = next(r for r in rows if r['phase'] == 'ph000')
        pfile = PK_DIR/'task43_rsd_lightcone_p02_ph000_mesh256_kmax0p300_dk0p002.npz'
        with np.load(row['lightcone_fkp_path']) as a:
            zedges, table = a['z_edges'], a['fkp_weights']
        rfile = row['lightcone_random_path']
    else:
        index = int(name.removeprefix('ezmock_'))
        selection = json.loads((BASE.parent/'hybrid_ezmock1000_0918_contract_v1/mock_selection_manifest.json').read_text())
        row = next(r for r in selection['rows'] if int(r['production_index']) == index)
        pfile = Path(row['pk_path']); rfile = row['random_catalog_path']
        zedges, table, _, _ = mean_fkp()
    with np.load(row['lightcone_catalog_path']) as a:
        dz = a['Z'].astype('f8')
        dp = np.column_stack([a[k] for k in ('X', 'Y', 'Zcart')]).astype('f4')
        base = a['WEIGHT'].astype('f8') if 'WEIGHT' in a.files else np.ones(len(dz))
    dw = base*fkp_weight(dz, zedges, table)
    n, nr = len(dz), 4*len(dz)
    rng = np.random.default_rng(4322251+(0 if halo else index+1))
    with np.load(rfile) as a:
        total = len(a['Z'])
        ids = np.arange(nr) if halo else rng.choice(total, nr, replace=False)
        rz_all = a['Z'].astype('f8')
        rz = rz_all[ids]
        rp = np.column_stack([a[k][ids] for k in ('X', 'Y', 'Zcart')]).astype('f4')
        source_random_weight = a['WEIGHT_FKP'][ids].astype('f8') if 'WEIGHT_FKP' in a.files else fkp_weight(rz, zedges, table)
    sources = []
    if halo:
        independent = []
        for other in rows:
            if other['phase'] == 'ph000':
                continue
            with np.load(other['lightcone_catalog_path']) as a:
                z = a['Z'].astype('f8')
                independent.append(z)
                sources.append({'phase': other['phase'], 'path': other['lightcone_catalog_path'],
                                'z_sha256': __import__('hashlib').sha256(z.tobytes()).hexdigest()})
        reference_cdf = np.sort(np.concatenate(independent))
        u = quantile(rz, np.sort(dz), rng)
        other_z = reference_cdf[np.minimum((u*len(reference_cdf)).astype(int), len(reference_cdf)-1)]
        reference_note = 'Independent pooled observed-z CDF from the other 24 halo phases; finite-selection uncertainty retained, not asserted to be noiseless truth.'
    else:
        source_cdf = np.sort(rz_all)
        u = quantile(rz, source_cdf, rng)
        source_data = np.sort(dz)
        other_z = source_data[np.minimum((u*n).astype(int), n-1)]
        reference_note = 'Immutable common-random reference used by the actual frozen covariance mocks; paired shuffled branch draws the current mock observed-z distribution.'
    from cosmoprimo.fiducial import AbacusSummit
    cosmo = AbacusSummit(0)
    zg = np.linspace(.4, .8, 200001)
    cg = cosmo.comoving_radial_distance(zg)
    unit = rp.astype('f8')/np.linalg.norm(rp.astype('f8'), axis=1)[:, None]
    other_p = (unit*np.interp(other_z, zg, cg)[:, None]).astype('f4')
    if halo:
        catalogs = {'shuffled': (rp, rz, fkp_weight(rz, zedges, table)),
                    'reference': (other_p, other_z, fkp_weight(other_z, zedges, table))}
    else:
        catalogs = {'reference': (rp, rz, source_random_weight),
                    'shuffled': (other_p, other_z, fkp_weight(other_z, zedges, table))}
    with np.load(pfile) as a:
        edges = a['k_edges']
        norm = float(np.ravel(a['norm_ell0'] if halo else a['norm0'])[0])
    pmeta = json.loads(pfile.with_suffix('.json').read_text())
    meta = {'name': name, 'data_path': row['lightcone_catalog_path'], 'random_source': str(rfile),
            'source_pk': str(pfile), 'source_pk_sha256': sha(pfile), 'reference_definition': reference_note,
            'independent_reference_sources': sources, 'ndata': n, 'nrandom_P': nr, 'nrandom_per_xi_block': n,
            'xi_blocks': 2, 'P_mesh': pmeta['mesh'], 'shared_P_I2': norm, 'source_sha256': sha(__file__),
            'covariance_modified': False, 'seed': 4322251+(0 if halo else index+1),
            'fixed_weight_policy': 'Production phase FKP for halo; production mean FKP for EZ; no FKP re-estimation after shuffle.'}
    arrays = {'data_position': dp, 'data_redshift': dz, 'data_weight': dw, 'k_edges': edges,
              'meta_json': np.array(json.dumps(meta))}
    for label, (p, z, w) in catalogs.items():
        arrays.update({label+'_position': p, label+'_redshift': z, label+'_weight': w})
    np.savez_compressed(target, **arrays)
    save_json(dest/'paired_catalogs.json', meta)
    print(json.dumps({'event': 'paired_catalogs', 'name': name, 'ndata': n, 'nrandom': nr}), flush=True)
    return target


def measure_xi(name):
    """同一 data 的两个径向分支，各测两个真实 N_R=N_D 的 signed40mu LS blocks。"""
    target = prepare(name); dest = root_for(name)
    if (dest/'paired_xi.npz').exists():
        return
    with np.load(target) as a:
        n = len(a['data_weight'])
        data = (a['data_position'].astype('f4'), a['data_weight'].astype('f4'))
        randoms = {label: (a[label+'_position'].astype('f4'), a[label+'_weight'].astype('f4'))
                   for label in ('reference', 'shuffled')}
    ddpath = dest/'bridge_block0_DD.npz'
    if ddpath.exists():
        with np.load(ddpath) as a:
            dd = a['normalized_counts']
    else:
        norm = np.sum(data[1], dtype='f8')**2
        dd = count(data)/norm
        np.savez_compressed(dest/'paired_DD.npz', normalized_counts=dd, norm=norm)
    vectors = {}
    for label in ('reference', 'shuffled'):
        blocks = []
        for block in range(2):
            path = dest/f'{label}_xi_block{block}.npz'
            if path.exists():
                with np.load(path) as a:
                    blocks.append(a['xi_multipoles'])
                continue
            started = time.monotonic()
            random = tuple(x[block*n:(block+1)*n] for x in randoms[label])
            wd, wr = np.sum(data[1], dtype='f8'), np.sum(random[1], dtype='f8')
            if name == 'halo_ph000' and label == 'shuffled' and block == 0:
                with np.load(dest/'bridge_block0_DR.npz') as a:
                    dr = a['normalized_counts']
                with np.load(dest/'bridge_block0_RR.npz') as a:
                    rr = a['normalized_counts']
            else:
                dr = count(data, random)/(wd*wr)
                rr = count(random)/(wr*wr)
            xi_smu = (dd-2*dr+rr)/rr
            multipoles = project(xi_smu)
            np.savez_compressed(path, xi_smu=xi_smu, xi_multipoles=multipoles, DR=dr, RR=rr)
            blocks.append(multipoles)
            print(json.dumps({'event': 'paired_xi_block', 'name': name, 'branch': label,
                              'block': block, 'elapsed_s': time.monotonic()-started}), flush=True)
        vectors[label] = np.mean(blocks, axis=0)
    np.savez_compressed(dest/'paired_xi.npz', reference=vectors['reference'], shuffled=vectors['shuffled'],
                        difference=vectors['shuffled']-vectors['reference'], s=(EDGES[:-1]+EDGES[1:])/2)


def measure_pk(name):
    """固定共同 mesh256 和生产 I2，避免重估归一化或 mesh 边界污染差分。"""
    target = prepare(name); dest = root_for(name)
    import jax
    jax.config.update('jax_enable_x64', True)
    from clustering_statistics import spectrum2_tools
    with np.load(target) as a:
        meta = json.loads(str(a['meta_json'].item())); edges = a['k_edges']
        data = {'POSITION': a['data_position'].astype('f8'), 'INDWEIGHT': a['data_weight'], 'Z': a['data_redshift']}
    outputs = {}
    for label in ('reference', 'shuffled'):
        path = dest/f'{label}_pk.npz'
        if path.exists():
            with np.load(path) as a:
                outputs[label] = a['multipoles']
            continue
        with np.load(target) as a:
            random = {'POSITION': a[label+'_position'].astype('f8'), 'INDWEIGHT': a[label+'_weight'],
                      'Z': a[label+'_redshift'], 'TARGETID': np.arange(meta['nrandom_P'], dtype='i8')}
        def get_data_randoms():
            return {'data': data, 'randoms': random}
        started = time.monotonic()
        spectrum = spectrum2_tools.compute_mesh2_spectrum(get_data_randoms,
                       mattrs=mesh_attrs_for_jaxpower(meta['P_mesh']), edges=edges,
                       ells=(0, 2), los='local', optimal_weights=None, norm={'cellsize': 10.})
        poles, measured_norms, measured_shot = [], [], []
        for ell in (0, 2):
            pole = spectrum.get(ell)
            norm = np.asarray(pole.values('norm'))
            poles.append(np.asarray(pole.value())*norm/meta['shared_P_I2'])
            measured_norms.append(norm)
            measured_shot.append(np.asarray(pole.values('shotnoise'))*norm/meta['shared_P_I2'])
        outputs[label] = np.asarray(poles)
        if not np.all(np.isfinite(outputs[label])):
            raise RuntimeError('Nonfinite paired P')
        np.savez_compressed(path, multipoles=outputs[label], k_edges=edges, measured_norms=measured_norms,
                            measured_shot=measured_shot, shared_norm=meta['shared_P_I2'])
        print(json.dumps({'event': 'paired_P', 'name': name, 'branch': label,
                          'elapsed_s': time.monotonic()-started}), flush=True)
    np.savez_compressed(dest/'paired_pk.npz', reference=outputs['reference'], shuffled=outputs['shuffled'],
                        difference=outputs['shuffled']-outputs['reference'], k_edges=edges)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=('prepare', 'xi', 'pk'))
    parser.add_argument('--name', choices=('halo_ph000', 'ezmock_0', 'ezmock_1'), required=True)
    args = parser.parse_args()
    {'prepare': prepare, 'xi': measure_xi, 'pk': measure_pk}[args.action](args.name)
