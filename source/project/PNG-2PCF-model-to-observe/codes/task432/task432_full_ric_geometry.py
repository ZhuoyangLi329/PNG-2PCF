#!/usr/bin/env python3
"""冻结实际 halo random 权重，构建 eBOSS 全项 RIC 几何核并保存可恢复产物。

执行大纲：读取正式 manifest/测量元数据 -> 缓存 random 子集 -> 分箱径向投影
-> 分别计算 midpoint 与 endpoint 输出 -> 保存原始核和版本/耗时审计。
首轮粗核仅作资源与收敛 pilot，未经收敛和闭合检查不得用于正式 posterior。
"""
from __future__ import annotations
import os
for _key in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):
    os.environ[_key] = '1'
if hasattr(os, 'sched_getaffinity'):
    os.sched_setaffinity(0, sorted(os.sched_getaffinity(0))[:8])
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
from task432_full_ric_backend import ROOT, OUT, counts

HALO_MANIFEST = ROOT/'outputs/task43_outputs/rsd_validation/manifests/task43_rsd_validation_lightcone_boxsafe_zobs0p4_0p8_x25.jsonl'
PK_DIR = ROOT/'outputs/task43_outputs/rsd_validation/lightcone/pk/boxsafe_zobs0p4_0p8_p02_x25_fkpP010000'
BASE = ROOT/'outputs/task43_outputs/rsd_validation/task432_model_repair/hybrid_ezmock1000_gic_0918_contract'


def sha(path):
    """对小输入或源码计算 hash；大 catalog 使用已经冻结的生产元数据 hash。"""
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(8*1024*1024), b''):
            h.update(chunk)
    return h.hexdigest()


def save_json(path, data):
    """原子保存只含 JSON 类型的审计。"""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix('.tmp.json')
    temporary.write_text(json.dumps(data, indent=2)+'\n')
    temporary.replace(path)


def prepare_pool(phase='ph000', size=131072, seed=4322203):
    """缓存均匀抽取的真实 random；FKP 使用生产的分段常数表，避免旧 helper 的插值差异。"""
    destination = OUT/f'geometry/random_pool_{phase}.npz'
    if destination.exists():
        with np.load(destination, allow_pickle=False) as a:
            return {k: a[k] for k in a.files}
    rows = [json.loads(line) for line in HALO_MANIFEST.read_text().splitlines() if line.strip()]
    row = next(r for r in rows if r['phase'] == phase)
    measurement = PK_DIR/f'task43_rsd_lightcone_p02_{phase}_mesh256_kmax0p300_dk0p002.npz'
    metadata_path = measurement.with_suffix('.json')
    metadata = json.loads(metadata_path.read_text())
    fkp_path = Path(metadata['fkp_summary'])
    with np.load(fkp_path, allow_pickle=False) as a:
        fkp = {k: a[k] for k in a.files}
    random_path = Path(row['lightcone_random_path'])
    with np.load(random_path, allow_pickle=False) as a:
        redshift = a['Z']; total = len(redshift)
        ids = np.random.default_rng(seed).choice(total, min(size, total), replace=False)
        redshift = np.asarray(redshift[ids], dtype='f8')
        position = np.column_stack([a[k][ids] for k in ('X', 'Y', 'Zcart')]).astype('f8')
        base = np.asarray(a['WEIGHT'][ids], dtype='f8')
        stored = np.asarray(a['WEIGHT_TOTAL'][ids], dtype='f8')
        block = a['RANDOM_INDEX'][ids]
    ibin = np.clip(np.searchsorted(fkp['z_edges'], redshift, side='right')-1, 0, len(fkp['fkp_weights'])-1)
    weight = base*fkp['fkp_weights'][ibin]
    if not np.allclose(weight, stored, rtol=1e-7, atol=1e-8):
        raise RuntimeError('FKP table does not reproduce production WEIGHT_TOTAL')
    with np.load(measurement, allow_pickle=False) as a:
        norm = float(np.ravel(a['norm_ell0'])[0])
        if not np.allclose(a['norm_ell0'], norm, rtol=1e-12):
            raise RuntimeError('Unexpected k-dependent P normalization')
    wdata = metadata['data']['weight_sum']
    pair_volume = wdata*wdata/norm
    manifest = {'phase': phase, 'row': row, 'pool_n': len(ids), 'full_random_n': total, 'seed': seed,
                'fkp_summary': str(fkp_path), 'fkp_sha256': sha(fkp_path),
                'random_hash_recorded_in_production': metadata['random']['sha256'],
                'measurement': str(measurement), 'measurement_metadata_sha256': sha(metadata_path),
                'weight_rule': 'production piecewise-constant fkp_weights[z_bin] times WEIGHT',
                'max_stored_weight_error': float(np.max(abs(weight-stored))),
                'P_pair_volume': pair_volume, 'P_I2': norm, 'P_data_weight_sum': wdata,
                'source_sha256': sha(__file__)}
    destination.parent.mkdir(parents=True, exist_ok=True)
    arrays = dict(position=position, weight=weight, redshift=redshift, random_block=block, source_index=ids,
                  pair_volume=np.array(pair_volume), meta_json=np.array(json.dumps(manifest)))
    np.savez_compressed(destination, **arrays)
    save_json(destination.with_suffix('.json'), manifest)
    frozen = {str(BASE/name): sha(BASE/name) for name in ('frozen_inputs.npz', 'xi_emulator.npz', 'input_audit.json')}
    save_json(OUT/'input_manifest.json', {'status': 'geometry_development', 'baseline': str(BASE), 'frozen': frozen,
                                       'formal_inputs_modified': False, 'covariance': 'EZmock1000 C_single full-cross',
                                       'geometry_pool': str(destination)})
    return arrays


def separation_edges(refinement=1):
    """在拟合小尺度用细网格，远离拟合区间使用较粗网格；全 survey 距离始终保留。"""
    f = int(refinement)
    return np.unique(np.r_[.001, 1., 2., np.arange(5., 355., 5./f),
                            np.arange(350., 1000., 10./f), np.arange(1000., 3400.1, 25./f)])


def build(phase, nsub, radial_width, refinement, los, maxell):
    """构造一套可独立重现的原始核；参数编码进目录，成功产物不覆盖。"""
    pool = prepare_pool(phase)
    if 3*nsub > len(pool['weight']):
        raise ValueError('Requested sample exceeds frozen pool')
    position, weight = pool['position'][:nsub], pool['weight'][:nsub]
    radial = np.linalg.norm(position, axis=1)
    # 空径向 bin 对积分无贡献，压缩标签供后端调用，并保存原编号以便审计。
    labels_raw = np.floor(radial/radial_width).astype('i8') if radial_width > 0 else np.zeros(nsub, dtype='i8')
    occupied, labels = np.unique(labels_raw, return_inverse=True)
    references = []
    for index in (1, 2):
        pp = pool['position'][index*nsub:(index+1)*nsub]
        ww = pool['weight'][index*nsub:(index+1)*nsub]
        raw = np.floor(np.linalg.norm(pp, axis=1)/radial_width).astype('i8') if radial_width > 0 else np.zeros(nsub, dtype='i8')
        if not np.array_equal(np.unique(raw), occupied):
            raise RuntimeError('Increase catalogue size: independent radial support is incomplete')
        ll = np.searchsorted(occupied, raw)
        references.append((pp, ww, ll))
    edges = separation_edges(refinement)
    width_tag = f'{radial_width:g}'.replace('.', 'p')
    stem = f'{phase}_independent_n{nsub}_dchi{width_tag}_refine{refinement}_{los}_ell{maxell}'
    target = OUT/f'geometry/{stem}.npz'
    if target.exists():
        print(json.dumps({'event': 'kernel_exists', 'path': str(target)}), flush=True)
        return
    for name in ('reference_algebra_poisson.json', 'eboss_backend_bridge.json'):
        assert json.loads((OUT/'smoke'/name).read_text())['status'] == 'pass'
    result = counts(position, weight, labels, edges, ells_out=tuple(range(0, maxell+1, 2)),
                    los_out=los, nthreads=8, progress=True, independent_projection=references)
    # 对常数 Xi，两个 cross 应积成 2 RR，auto 应积成 RR；独立目录无重复点偏差。
    const_cross = np.sum(result['cross'][:, 0], axis=-1)
    const_auto = np.sum(result['auto'][:, 0], axis=-1)
    scale = np.max(abs(result['rr']))
    constant_error = max(np.max(abs(const_cross-2*result['rr'])), np.max(abs(const_auto-result['rr'])))/scale
    if constant_error > 1e-10:
        raise RuntimeError(f'Constant field closure failed: {constant_error}')
    meta = {k: v for k, v in result.items() if k not in ('rr', 'cross', 'auto')}
    meta.update({'status': 'pilot_unvalidated', 'phase': phase, 'nsub': nsub, 'radial_width': radial_width,
                 'refinement': refinement, 'occupied_bins': len(occupied),
                 'radial_count_range': [int(np.bincount(labels).min()), int(np.bincount(labels).max())],
                 'pair_volume': float(pool['pair_volume']), 'source_sha256': sha(__file__),
                 'backend_adapter_sha256': sha(Path(__file__).with_name('task432_full_ric_backend.py')),
                 'scope': 'Independent outer/A/B projection samples; full cross and symmetrized auto. Numerical convergence remains mandatory.',
                 'constant_field_relative_error': float(constant_error),
                 'input_los': 'midpoint', 'cpu_affinity': sorted(os.sched_getaffinity(0))})
    temporary = target.with_suffix('.tmp.npz')
    np.savez_compressed(temporary, separation_edges=edges, radial_labels=occupied,
                        rr=result['rr'], cross=result['cross'], auto=result['auto'],
                        normalization=result['normalization'], meta_json=np.array(json.dumps(meta)))
    temporary.replace(target)
    save_json(target.with_suffix('.json'), meta)
    print(json.dumps({'event': 'kernel_done', 'path': str(target), 'elapsed_s': result['elapsed_s']}), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--phase', default='ph000')
    parser.add_argument('--nsub', type=int, default=1024)
    parser.add_argument('--radial-width', type=float, default=20.)
    parser.add_argument('--refinement', type=int, default=1)
    parser.add_argument('--los', choices=('midpoint', 'endpoint'), default='midpoint')
    parser.add_argument('--maxell', type=int, default=8)
    args = parser.parse_args()
    build(args.phase, args.nsub, args.radial_width, args.refinement, args.los, args.maxell)
