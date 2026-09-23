#!/usr/bin/env python3
"""只审计 EZ1000 分析的输入观测口径与低 k 支持，不改变任何正式预测。

核对 25 phase 的 zeff/FKP/目录；记录不同 LOS 与缺失 P 窗口；估算移除
k<kbox 线性 Kaiser 模式并同步更新 scalar GIC 的局部响应。后者不是完整 GSM 重算。
"""
import task432_joint_b1_diagnostics as d
from task432_joint_b1_diagnostics import np,json,save,sha,Path,Metric
from task432_hybrid_gic import HybridGIC
from scipy.special import spherical_jn
from scipy.integrate import simpson

def run():
    """先用实际文件核对约定，再做带明确近似标签的 IR 灵敏度估计。"""
    d.initialize();import task43_rsd_ezmock281_mcmc_compare as old
    import task43_rsd_boxsafe_build_ell02_products as xp
    from task43_rsd_lightcone_pk0_contract import WINDOW_THEORY_KMIN
    pairs=[]
    with np.load(old.PAYLOAD_NPZ,allow_pickle=False) as z:model_z=float(z['zeff'])
    for i in range(25):
        pp=old.MEASURE_DIR/f'task43_rsd_lightcone_p02_ph{i:03d}_mesh256_kmax0p300_dk0p002.npz'
        xx=xp.XI_DIR/f'task43_rsd_xi0_AbacusSummit_base_c000_ph{i:03d}_mmin1p4e13_zobs0p4_0p8_x25_s30_350_ds10.npz'
        pm=json.loads(pp.with_suffix('.json').read_text());xm=json.loads(xx.with_suffix('.json').read_text())
        with np.load(pp,allow_pickle=False) as z:pz=float(z['zeff']);p0=float(z['p0']);hasw='window_matrix' in z
        with np.load(xx,allow_pickle=False) as z:xz=float(z['zeff']);x0=float(z['p0'])
        pairs.append({'phase':i,'P_zeff':pz,'xi_zeff':xz,'P_fkp_P0':p0,'xi_fkp_P0':x0,'P_los':pm['los'],'xi_los':xm['los'],'same_fkp_file':pm['fkp_summary']==xm['fkp_summary_path'],'P_window_exists':hasw,'P_metadata':str(pp.with_suffix('.json')),'xi_metadata':str(xx.with_suffix('.json')),'P_fkp_sha256':pm['fkp_summary_sha256'],'xi_fkp_sha256':sha(xm['fkp_summary_path'])})
    assert all(x['same_fkp_file'] and x['P_fkp_sha256']==x['xi_fkp_sha256'] and x['P_zeff']==x['xi_zeff'] and x['P_fkp_P0']==x['xi_fkp_P0']==10000 for x in pairs)
    pri=json.loads((d.OUT/'primary.json').read_text())['covariances']['ezmock1000'];t=np.array(pri['baseline']['joint']['theta']);model=HybridGIC()
    k=model.gsm.kint;mask=k<WINDOW_THEORY_KMIN;kk=np.r_[k[mask],WINDOW_THEORY_KMIN]
    pl=np.interp(np.log(kk),np.log(k),model.gsm.plin);aa=np.interp(np.log(kk),np.log(model.k),model.alpha)
    r,w=model.window.quadrature(8);w2=spherical_jn(0,kk[:,None]*r[None,:])@w
    response=np.array(pri['conditional_at_joint_ML']['response']);cov=d.COVS['ezmock1000'];estimates=[]
    for fnl in (0.,float(t[0])):
        b=t[1];fg=model.f_growth;B=b+fnl*2*1.686*(b-1)*aa
        p0=(B*B+2/3*fg*B+fg*fg/5)*pl;p2=(4/3*fg*B+4/7*fg*fg)*pl
        pref=kk**2/(2*np.pi**2);s=model.s
        x0=simpson((pref*p0)[:,None]*spherical_jn(0,kk[:,None]*s),x=kk,axis=0)
        x2=-simpson((pref*p2)[:,None]*spherical_jn(2,kk[:,None]*s),x=kk,axis=0)
        cw=simpson(pref*p0*w2,x=kk);dm=np.r_[np.zeros(22),-x0+cw,-x2]
        estimates.append({'fNL':fnl,'delta_model_remove_lowk':dm,'chi2_delta_Csingle':Metric(cov).chi2(dm),'linear_parameter_response':-response@dm,'lowk_GIC_constant':cw})
    save(d.OUT/'contract_audit.json',{'status':'pass_for_verified_fields','phase_contracts':pairs,'common_model_zeff':model_z,'mean_phase_zeff':np.mean([x['P_zeff'] for x in pairs]),'phase_zeff_range':[min(x['P_zeff'] for x in pairs),max(x['P_zeff'] for x in pairs)],'P_window_theory_kmin':WINDOW_THEORY_KMIN,'hybrid_internal_k_range':[k[0],k[-1]],'hybrid_cached_alpha_k_range':[model.k[0],model.k[-1]],'lowk_estimates':estimates,'limitations':['Only ph000 P window exists; no 25-window average is asserted.','P LOS=local and xi LOS=midpoint; this documents conventions, not proof of a bias.','Low-k estimate removes linear Kaiser contributions with matched scalar GIC only; nonlinear GSM cumulants and the exact finite-box operator are not recomputed.','No independent halo-matter b1 measurement was created in this diagnostic.']})

if __name__=='__main__':run()
