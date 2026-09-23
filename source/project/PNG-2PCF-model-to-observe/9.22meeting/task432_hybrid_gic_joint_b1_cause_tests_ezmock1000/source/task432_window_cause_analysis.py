#!/usr/bin/env python3
"""Compare full-catalog phase-window pilots at the frozen 9.18 fit contract.

Ph000 reproduction checks precede any interpretation. The 3-phase mean is
explicitly a pilot, not the missing full 25-phase average. Cosmology, zeff,
data, covariance and xi model remain fixed to isolate the P window response.
"""
import task432_joint_b1_cause_tests as c
from task432_joint_b1_cause_tests import np,Path,json,save,sha,log

def main():
    c.initialize();old,pm,source=c.load_p()
    with np.load(old.PAYLOAD_NPZ,allow_pickle=False) as z:idx=z['fit_bin_indices'][:13]
    rows=np.r_[idx,150+idx[4:]];pilot=old.MEASURE_DIR.parent/'boxsafe_p02_window_cause_pilot_v1'
    matrices=[];records=[]
    for phase in (0,12,24):
        name=f'task43_rsd_lightcone_p02_ph{phase:03d}_mesh256_kmax0p300_dk0p002.npz';p=pilot/name
        with np.load(p,allow_pickle=False) as a, np.load(old.MEASURE_DIR/name,allow_pickle=False) as b:
            assert np.array_equal(a['theory_k'],pm.theory_k)
            assert np.array_equal(a['theory_ell'],pm.theory_ell)
            win=a['window_matrix'][rows];matrices.append(win)
            record={'phase':phase,'source':str(p),'sha256':sha(p),'measurement_relative_l2':{q:np.linalg.norm(a[q]-b[q])/np.linalg.norm(b[q]) for q in ('pk0','pk2')},'window_relative_l2_to_ph000':np.linalg.norm(win-pm.window)/np.linalg.norm(pm.window)}
            assert max(record['measurement_relative_l2'].values())<1e-9,record
        records.append(record)
    repeat=old.MEASURE_DIR.parent/'boxsafe_p02_window_cause_repeat_v1'/'task43_rsd_lightcone_p02_ph000_mesh256_kmax0p300_dk0p002.npz'
    with np.load(repeat,allow_pickle=False) as z:
        repeat_error=np.linalg.norm(z['window_matrix'][rows]-matrices[0])/np.linalg.norm(matrices[0])
    assert repeat_error<1e-9,repeat_error
    variants={f'ph{phase:03d}':matrices[i] for i,phase in enumerate((0,12,24))}
    variants['three_phase_pilot_mean']=np.mean(matrices,axis=0)
    report={'scope':'Full data and x25 random catalogs, mesh256, local-LOS estimator. Only phases 0,12,24; their mean is NOT the full 25-phase window. Same data and EZmock1000 C_single in all fits. Historical ph000 window fails exact reproduction; quantify that drift separately from within-current-implementation phase differences. No formal estimator or covariance replacement.','reproduction_gates':records,'current_repeat_source':str(repeat),'current_repeat_relative_error':repeat_error,'historical_window_reproduced':bool(records[0]['window_relative_l2_to_ph000']<1e-9),'variants':{}}
    for name,win in variants.items():
        model=old.WindowConvolvedP02Model(win,pm.theory_k,pm.theory_ell,zeff=pm.zeff)
        def predict(t):
            ts=np.atleast_2d(t);out=c.d.prediction(ts)
            out[:,:22]=np.array([model.evaluate(x) for x in ts]);return out
        report['variants'][name]=c.model_change(predict,name)
        log('window_refit',name=name,theta=report['variants'][name]['fits']['joint']['theta'])
    report['within_current_implementation_joint_b1_shifts']={name:v['fits']['joint']['theta'][1]-report['variants']['ph000']['fits']['joint']['theta'][1] for name,v in report['variants'].items()}
    save(c.OUT/'window_pilot.json',report)
    np.savez_compressed(c.OUT/'window_pilot_matrices.npz',phases=[0,12,24],selected_matrices=matrices,pilot_mean=variants['three_phase_pilot_mean'],original_ph000=pm.window,theory_k=pm.theory_k,theory_ell=pm.theory_ell)

if __name__=='__main__':main()
