import pathlib,json,datetime,hashlib,h5py,numpy as np
root=pathlib.Path('/pscratch/sd/l/lzy/PNG-2PCF-model-to-observe')
p=root/'outputs/task43_outputs/rsd_validation/lightcone/pk'
out=root/'outputs/task43_outputs/rsd_validation/task432_model_repair/hybrid_gic_joint_b1_cause_tests_v1'
name='task43_rsd_lightcone_p02_ph000_mesh256_kmax0p300_dk0p002'
folders=['boxsafe_zobs0p4_0p8_p02_x25_fkpP010000','boxsafe_p02_window_cause_pilot_v1','boxsafe_p02_window_cause_repeat_v1']
metas=[json.loads((p/f/(name+'.json')).read_text()) for f in folders]
keys=['data','random','mesh','measurement_k_grid','fkp_summary_sha256','window_method']
checks={k:all(m[k]==metas[0][k] for m in metas[1:]) for k in keys}
assert all(checks.values()),checks
files=[p/f/(name+'.npz') for f in folders]+[pathlib.Path('/global/common/software/desi/users/adematti/perlmutter/cosmodesiconda/20260321-1.0.0/code/jaxpower/main/lib/python3.12/site-packages/jaxpower/mesh2.py')]
records={str(q):{'sha256':hashlib.sha256(q.read_bytes()).hexdigest(),'mtime_utc':datetime.datetime.fromtimestamp(q.stat().st_mtime,datetime.timezone.utc).isoformat()} for q in files}
bridges={}
for folder in folders:
    with np.load(p/folder/(name+'.npz')) as a,h5py.File(p/folder/(name+'_window_smooth.h5')) as h:
        bridges[folder]=float(np.max(abs(a['window_matrix']-h['value'][()])))
assert max(bridges.values())==0
report={'input_metadata_identical':checks,'npz_vs_h5_maxabs':bridges,'files':records,'source_of_drift':'Unresolved at commit level: current library source was modified after the historical measurement, and no historical source snapshot was located. Current repeated calculations agree; do not label historical drift as phase variation.'}
(out/'window_provenance_audit.json').write_text(json.dumps(report,indent=2)+'\n')
print(json.dumps(report))
