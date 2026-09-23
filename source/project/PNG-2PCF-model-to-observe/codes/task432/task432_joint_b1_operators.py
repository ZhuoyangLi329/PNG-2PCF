#!/usr/bin/env python3
"""观测算子诊断：P 的 ph000/25-phase mean window，xi 的中心值/球壳体积平均。

只使用固定的原 hybrid+GIC；先复现基准，再校验局部差值插值，最后重优化。
不改写正式模型；shell 平均是球壳 r² 权重诊断，不冒充完整 RR(s,mu) 算子。
"""
import task432_joint_b1_diagnostics as d
from task432_joint_b1_diagnostics import np,Path,json,save,sha,log,Metric
from task432_hybrid_gic import HybridGIC
from scipy.interpolate import CubicSpline,RectBivariateSpline
import copy,time

def run():
    """核验 ph000 复现和小网格精度后，对两套协方差分别拟合四种算子。"""
    d.initialize();import task43_rsd_ezmock281_mcmc_compare as old
    with np.load(old.PAYLOAD_NPZ,allow_pickle=False) as z:rows=z['fit_bin_indices'][:13];zeff=float(z['zeff'])
    selected=np.r_[rows,150+rows[4:]];windows=[];meta=[];missing=[]
    for i in range(25):
        p=old.MEASURE_DIR/f'task43_rsd_lightcone_p02_ph{i:03d}_mesh256_kmax0p300_dk0p002.npz'
        with np.load(p,allow_pickle=False) as z:
            if 'window_matrix' not in z:
                missing.append(i);meta.append({'path':str(p),'sha256':sha(p),'window_present':False});continue
            if i==0:k=z['theory_k'];ell=z['theory_ell']
            else:assert np.array_equal(k,z['theory_k']) and np.array_equal(ell,z['theory_ell'])
            windows.append(z['window_matrix'][selected]);meta.append({'path':str(p),'sha256':sha(p),'window_present':True})
    w=np.asarray(windows);pm=old.WindowConvolvedP02Model(w[0],k,ell,zeff=zeff)
    meanpm=copy.copy(pm);meanpm.window=w.mean(axis=0)
    bases=np.array([pm._theory_basis(float(s)) for s in pm.sigma_grid])
    meanpm.spline=CubicSpline(pm.sigma_grid,np.einsum('ij,sjk->sik',meanpm.window,bases),axis=0)
    shot=np.where((ell==0)&pm.support,1e4,0.);meanpm.shot_response=meanpm.window@shot
    pri=json.loads((d.OUT/'primary.json').read_text())
    probes=[np.array(pri['covariances'][c]['baseline'][v]['theta']) for c in d.COVS for v in d.IDS]
    reproduction=max(np.max(abs(pm.evaluate(t)-d.prediction(t)[0,:22])) for t in probes)
    assert reproduction<1e-7,reproduction
    model=HybridGIC();s=model.s;xq,wq=np.polynomial.legendre.leggauss(8);lo=s[:,None]-5;hi=s[:,None]+5
    rr=(hi-lo)/2*xq+(hi+lo)/2;rw=(hi-lo)/2*wq*rr**2/((hi**3-lo**3)/3)
    def shell_delta(f,b,nint=600,nquad=None):
        """相同累积量和 GIC 下的体积壳平均减中心值，GIC 常数自动抵消。"""
        model._set_cumulants(float(f),float(b))
        yy=model.poles(np.r_[s,rr.ravel()],nint=nint)[:2]
        cen=yy[:,:len(s)];ave=np.sum(yy[:,len(s):].reshape(2,len(s),8)*rw,axis=2)
        return (ave-cen).ravel()
    fg=np.linspace(-60,60,5);bg=np.linspace(2.2,2.6,5);grid=np.empty((5,5,52))
    for i,f in enumerate(fg):
        for j,b in enumerate(bg):grid[i,j]=shell_delta(f,b)
        log('shell_grid_row',index=i)
    spl=[RectBivariateSpline(fg,bg,grid[:,:,i],kx=3,ky=3,s=0) for i in range(52)]
    def delta(t):
        """局部算子差值插值，仅用于中心附近 ML；报告会检查最终点落在网格内。"""
        t=np.atleast_2d(t);return np.column_stack([sp.ev(t[:,0],t[:,1]) for sp in spl])
    errors=[]
    for t in probes:
        direct=shell_delta(*t[:2],nint=1200);e=np.r_[np.zeros(22),delta(t)[0]-direct]
        errors.append(max(float(Metric(c).chi2(e))*25 for c in d.COVS.values()))
    assert max(errors)<1e-3,errors
    result={'p_window_source_files':meta,'missing_window_phases':missing,'mean_P_window_status':'unavailable: only ph000 window was measured' if missing else 'complete','p_ph000_reproduction_max_abs':reproduction,'shell_grid':{'fNL':fg,'b1':bg,'interpolation_error_chi2_Cmean25':errors},'scope':'xi shell-volume average, not full RR(s,mu) averaging. Fixed common zeff and LOS conventions retained. GIC remains parameter-dependent and identical. Mean-P-window refit is performed only if all 25 windows exist.','fits':{}}
    for c,cov in d.COVS.items():
        result['fits'][c]={};t=np.array(pri['covariances'][c]['baseline']['joint']['theta']);metric=Metric(cov)
        dp=np.r_[meanpm.evaluate(t)-pm.evaluate(t),np.zeros(52)];dx=np.r_[np.zeros(22),delta(t)[0]]
        result['fits'][c]['prediction_deltas_at_baseline']={'mean_window_chi2_Csingle':None if missing else metric.chi2(dp),'shell_average_chi2_Csingle':metric.chi2(dx),'max_window_relative_frobenius':None if missing else np.linalg.norm(w.mean(axis=0)-w[0])/np.linalg.norm(w[0])}
        variants=[('shell_xi',False,True)] if missing else [('mean_P_window',True,False),('shell_xi',False,True),('both',True,True)]
        for label,usep,usex in variants:
            def predict(t):
                """向冻结基准添加已验证的算子差异，动力学与 GIC 不变。"""
                ts=np.atleast_2d(t);y=d.prediction(ts)
                if usep:y[:,:22]=np.stack([meanpm.evaluate(tt) for tt in ts])
                if usex:y[:,22:]+=delta(ts)
                return y
            fits={v:d.fit(cov,ids,start=np.array(pri['covariances'][c]['baseline'][v]['theta']),predict=predict,multi=True) for v,ids in d.IDS.items()}
            for v,r in fits.items():
                tt=r['theta'];assert fg.min()<tt[0]<fg.max() and bg.min()<tt[1]<bg.max(),(label,v,tt)
                if usex:
                    err=np.r_[np.zeros(22),delta(tt)[0]-shell_delta(*tt[:2],nint=1200)]
                    r['verified_delta_chi2_error_Cmean25']=metric.chi2(err)*25
                    assert r['verified_delta_chi2_error_Cmean25']<1e-3
            result['fits'][c][label]=fits
        save(d.OUT/'operators.json',result);log('operators_cov_done',covariance=c)
    payload={'f_grid':fg,'b_grid':bg,'shell_minus_center':grid,'p_window_ph000':w[0],'missing_window_phases':np.array(missing,dtype=int)}
    if not missing:payload['p_window_mean']=w.mean(axis=0)
    np.savez_compressed(d.OUT/'operator_delta_grid.npz',**payload)
    save(d.OUT/'operators.json',result)

if __name__=='__main__':run()
