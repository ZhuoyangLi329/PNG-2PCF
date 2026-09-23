#!/usr/bin/env python3
"""有界只读审计：查找现成halo-matter谱和公开snapshot粒子入口，不下载大文件。"""
import json,os
from pathlib import Path
import numpy as np
ROOT=Path('/pscratch/sd/l/lzy/PNG-2PCF-model-to-observe')
OUT=ROOT/'outputs/task43_outputs/rsd_validation/task432_model_repair/fkp_conditional_diagnostics'
report={'precomputed_cross_candidates':[],'snapshot_entries':[],'scope':'Bounded directory/name inventory only; particle content/completeness not certified.'}
root=ROOT/'outputs/task43_outputs'
for current,dirs,files in os.walk(root):
    depth=len(Path(current).relative_to(root).parts)
    if depth>=4:dirs[:]=[]
    dirs[:]=[d for d in dirs if d not in ('chains','catalogs','randoms','fkp_catalogs','logs','geometry','paired')]
    for f in files:
        if any(t in f.lower() for t in ('halo_matter','halo-matter','phm','p_hm','matter_cross')):
            report['precomputed_cross_candidates'].append(str(Path(current)/f))
public=Path('/global/cfs/cdirs/desi/public/cosmosim/AbacusSummit/AbacusSummit_base_c000_ph000/halos')
for redshift in ('z0.400','z0.500','z0.575','z0.650','z0.725','z0.800'):
    path=public/redshift
    item={'path':str(path),'exists':path.exists(),'subdirectories':[]}
    if path.exists():
        for sub in sorted(path.iterdir()):
            if sub.is_dir():
                names=list(sub.iterdir());sample=next((p for p in names if p.is_file()),None)
                item['subdirectories'].append({'name':sub.name,'n_entries':len(names),'sample':None if sample is None else str(sample),
                                               'sample_size_bytes':None if sample is None else sample.stat().st_size})
    report['snapshot_entries'].append(item)
report['status']='inventory_complete'
# 按FKP权重本身（不看b1残差）检查三个预选phase是否覆盖典型权重扰动。
manifest=ROOT/'outputs/task43_outputs/rsd_validation/manifests/task43_rsd_validation_lightcone_boxsafe_zobs0p4_0p8_x25.jsonl'
rows=[json.loads(line) for line in manifest.read_text().splitlines() if line.strip()]
tables=[]
for row in rows:
    with np.load(row['lightcone_fkp_path']) as a:
        tables.append({k:a[k] for k in ('nbar','fkp_weights','data_counts','p0')})
all_nbar=np.array([t['nbar'] for t in tables]);fkp_stats=[]
for i,(row,t) in enumerate(zip(rows,tables)):
    ref=(all_nbar.sum(0)-all_nbar[i])/24
    ratio=(1/(1+float(t['p0'])*ref))/t['fkp_weights']-1
    rms=float(np.sqrt(np.sum(t['data_counts']*ratio**2)/t['data_counts'].sum()))
    fkp_stats.append({'phase':row['phase'],'count_weighted_relative_FKP_RMS':rms,'max_abs_FKP_change':float(abs(ratio).max()),'pilot':row['phase'] in ('ph000','ph012','ph024')})
report['fkp_weight_amplitudes_all25']=fkp_stats
(OUT/'external_bias_inventory.json').write_text(json.dumps(report,indent=2)+'\n')
print(json.dumps({'status':report['status'],'precomputed_cross_candidates':report['precomputed_cross_candidates'],'FKP_RMS_range':[min(r['count_weighted_relative_FKP_RMS'] for r in fkp_stats),max(r['count_weighted_relative_FKP_RMS'] for r in fkp_stats)],'pilot_FKP':[r for r in fkp_stats if r['pilot']]}))
