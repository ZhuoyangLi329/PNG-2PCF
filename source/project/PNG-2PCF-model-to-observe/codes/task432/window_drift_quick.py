import task432_joint_b1_cause_tests as c
from task432_joint_b1_cause_tests import np,json,save
c.initialize();old,pm,source=c.load_p()
with np.load(old.PAYLOAD_NPZ) as z:idx=z['fit_bin_indices'][:13]
rows=np.r_[idx,150+idx[4:]];report={}
for phase in (0,12):
    p=old.MEASURE_DIR.parent/'boxsafe_p02_window_cause_pilot_v1'/f'task43_rsd_lightcone_p02_ph{phase:03d}_mesh256_kmax0p300_dk0p002.npz'
    with np.load(p) as z:win=z['window_matrix'][rows]
    model=old.WindowConvolvedP02Model(win,pm.theory_k,pm.theory_ell,zeff=pm.zeff)
    def predict(t):
        ts=np.atleast_2d(t);y=c.d.prediction(ts);y[:,:22]=[model.evaluate(x) for x in ts];return y
    rr=c.model_change(predict,f'ph{phase:03d}')
    report[str(phase)]=rr
    print(phase,rr['changes_at_same_theta']['joint']['delta_chi2_Csingle'],{v:r['theta'] for v,r in rr['fits'].items()},flush=True)
save(c.OUT/'window_recompute_drift_preliminary.json',report)
