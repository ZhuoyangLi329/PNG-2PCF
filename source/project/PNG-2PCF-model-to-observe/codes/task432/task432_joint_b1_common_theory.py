#!/usr/bin/env python3
"""共同 Kaiser+FoG 理论的连续 Hankel/GIC 预测及跨模型注入，仅作归因控制。

执行大纲：相同 P_L/alpha → 宽 k FFTLog → 完整 RR GIC → 去除低 k 的可控分支 →
插值/积分审计 → 无噪声自闭合与交叉注入 → 真实数据诊断拟合。
共享 sigma 是控制模型已有的 P nuisance；不改变正式 hybrid 的参数或动力学。
"""
import task432_joint_b1_cause_tests as c
from task432_joint_b1_cause_tests import np,Path,json,save,sha,log,Metric
from scipy.interpolate import CubicSpline
from scipy.signal.windows import tukey
from scipy.special import spherical_jn
from scipy.integrate import simpson
from velocileptors.Utils.spherical_bessel_transform import SphericalBesselTransform
from task432_hybrid_gic import PairWindow
import time

class CommonXi:
    def __init__(self,pm,nk=8192,nmu=96):
        """理论 P_L、alpha 与 P 模型来自同一密集模板；外推只用于高 k 数值尾部。"""
        self.pm=pm;self.k=np.geomspace(1e-5,1e3,nk);self.nk=nk;self.nmu=nmu;self.s=c.d.FROZEN['ezmock1000']['xi_centers']
        self.mu,self.w=np.polynomial.legendre.leggauss(nmu);self.mu2=self.mu**2;self.taper=tukey(nk)
        self.pl,self.alpha=self.inputs(self.k);self.sbt=SphericalBesselTransform(self.k,L=3,fourier=True)
        window=PairWindow();self.rr,self.rw=window.quadrature(8)
        self.lowk=np.geomspace(1e-5,pm.theory_kmin,2049);self.lowpl,self.lowalpha=self.inputs(self.lowk)
        # 低 k 用平滑直接积分剔除，避免在 FFTLog 输入中人为制造阶跃振铃。
        w2=np.empty(len(self.lowk))
        for start in range(0,len(self.lowk),128):w2[start:start+128]=spherical_jn(0,self.lowk[start:start+128,None]*self.rr)@self.rw
        self.lowkernels=[spherical_jn(0,self.lowk[:,None]*self.s)-w2[:,None],-spherical_jn(2,self.lowk[:,None]*self.s)]

    def inputs(self,k):
        """模板区间内严格使用 P 侧的 log-k 线性插值，区间外沿端点幂律延伸。"""
        tk=self.pm.template['k'];pl=self.pm.template['pk_dd'];a=self.pm.template['alpha'];lk=np.log(k);base=np.log(tk)
        pp=np.interp(lk,base,pl);aa=np.interp(lk,base,a)
        for side in (0,-1):
            sel=k<tk[0] if side==0 else k>tk[-1];i,j=(0,1) if side==0 else (-2,-1)
            slope=np.log(pl[j]/pl[i])/np.log(tk[j]/tk[i]);pp[sel]=pl[side]*(k[sel]/tk[side])**slope
        return pp,aa

    def spectral_basis(self,k,pl,alpha,sig,ell):
        """Kaiser×Lorentzian² 多极矩分成六个幅度基底，与 P 侧系数顺序完全一致。"""
        damp=(1+.5*(k[:,None]*self.mu*sig)**2)**-2
        leg=np.ones_like(self.mu) if ell==0 else (3*self.mu2-1)/2
        moments=np.array([.5*(2*ell+1)*np.sum(damp*(self.w*leg*self.mu2**p),axis=1) for p in (0,1,2)])
        return pl*np.array([moments[0],alpha*moments[0],alpha**2*moments[0],moments[1],alpha*moments[1],moments[2]])

    def basis(self,sig):
        """返回连续和 kbox 截断的 52x6 预测基底，两个分支各自同步计算 GIC。"""
        output=[];cutoutput=[]
        for ell in (0,2):
            spectral=self.spectral_basis(self.k,self.pl,self.alpha,sig,ell)*self.taper
            vals=[]
            for b in spectral:
                r,y=self.sbt.sph(ell,b);y=np.ravel(y)*(-1)**(ell//2)
                interp=CubicSpline(np.log(r),y);v=interp(np.log(self.s))
                if ell==0:v=v-self.rw@interp(np.log(self.rr))
                vals.append(v)
            val=np.array(vals).T;output.append(val)
            low=self.spectral_basis(self.lowk,self.lowpl,self.lowalpha,sig,ell)*np.interp(np.log(self.lowk),np.log(self.k),self.taper)
            missing=simpson(low[:,:,None]*(self.lowk**2/(2*np.pi**2))[None,:,None]*self.lowkernels[ell//2][None,:,:],x=self.lowk,axis=1).T
            cutoutput.append(val-missing)
        return np.vstack(output),np.vstack(cutoutput)

    def coefficients(self,t):
        """从 fNL、b1 生成原 P 侧六项多项式系数。"""
        t=np.atleast_2d(t);fn,b=t[:,:2].T;q=fn*2*1.686*(b-1);f=self.pm.f_growth
        return np.column_stack([b*b,2*b*q,q*q,2*b*f,2*q*f,np.full(len(t),f*f)])

def run():
    """自闭合必须通过，跨模型注入才可用于判断理论不一致是否足以造成观察到的偏移。"""
    c.initialize();_,pm,_=c.load_p();base=CommonXi(pm);hi=CommonXi(pm,nk=16384,nmu=192);grid=np.arange(0,30.0001,.1)
    path=c.OUT/'common_kaiser_grid.npz'
    if not path.exists():
        raw=[];cut=[]
        for i,sig in enumerate(grid):
            a,b=base.basis(sig);raw.append(a);cut.append(b)
            if i%50==0:log('common_theory_grid',row=i,total=len(grid))
        np.savez_compressed(path,sigma_grid=grid,continuous_basis=raw,boxcut_basis=cut)
    with np.load(path,allow_pickle=False) as z:sp={key:CubicSpline(z['sigma_grid'],z[key],axis=0) for key in ('continuous_basis','boxcut_basis')}
    def predictor(kind):
        """共同理论保留现有 P 预测，只用同一物理谱变换得到 xi；连续/boxcut 分支分开记录。"""
        def predict(t):
            ts=np.atleast_2d(t);y=c.d.prediction(ts);y[:,22:]=np.einsum('nij,nj->ni',sp[kind](ts[:,2]),base.coefficients(ts));return y
        return predict
    probes=[[0,2.42,2.15,.08],[0,2.35,5,.02],[-30,2.49,2.517,.08],[30,2.42,8.337,.08],[0,2.42,.05,.08],[0,2.42,29.95,.08]]
    audit=[];metric=Metric(c.COV)
    for t in probes:
        bases=hi.basis(t[2]);coef=base.coefficients(t)[0]
        for j,kind in enumerate(sp):
            fast=predictor(kind)(t)[0,22:];exact=bases[j]@coef;err=metric.chi2(np.r_[np.zeros(22),fast-exact])*25
            audit.append({'theta':t,'kind':kind,'error_chi2_Cmean25':err})
    save(c.OUT/'common_theory_numerical_audit.json',{'checks':audit,'gate':max(a['error_chi2_Cmean25'] for a in audit)<.01,'nk':8192,'nmu':96,'reference_nk':16384,'reference_nmu':192,'sigma_step':.1})
    assert max(a['error_chi2_Cmean25'] for a in audit)<.01,audit
    report={'scope':'Matched Kaiser+FoG clustering controls, not a replacement for hybrid. Joint parameter count remains four; coherent xi-only has three physical parameters (fNL,b1,sigma), versus two for hybrid. Stochastic sn0 remains an independent P contact template.','numerical_checks':audit,'variants':{}}
    truths=[[0,2.35,2.15,.08],[0,2.42,2.15,.08],[0,2.49,2.15,.08],[0,2.42,5.,.08],[-30,2.42,2.15,.08],[30,2.42,2.15,.08]]
    for kind in sp:
        predict=predictor(kind);rr=c.model_change(predict,kind,xi_sigma=True);rr['injections']=[]
        for t in truths:
            y=predict(t)[0];current=c.d.prediction(t)[0]
            # Start away from the injected truth so closure also exercises the optimizer.
            ownstart=np.array(t)+np.array([17.,.05,1.1,.025])
            own={v:c.solve(predict,ids,data=y,start=ownstart,xi_sigma=True) for v,ids in c.d.IDS.items()}
            fit_mixed={v:c.solve(c.d.prediction,ids,data=y,start=t) for v,ids in c.d.IDS.items()}
            fit_reverse={v:c.solve(predict,ids,data=current,start=t,xi_sigma=True) for v,ids in c.d.IDS.items()}
            assert max(abs(r['theta'][1]-t[1]) for r in own.values())<1e-4
            assert all(r['success'] and r['chi2_Csingle']<1e-8 for r in own.values())
            rr['injections'].append({'truth':t,'self_recovery_start':ownstart,'same_model_recovery':own,'coherent_truth_fitted_by_current':fit_mixed,'current_composite_fitted_by_coherent':fit_reverse,'mixed_fit_joint_minus_marginals':[fit_mixed['joint']['theta'][1]-fit_mixed[v]['theta'][1] for v in ('p02','xi02')]})
        report['variants'][kind]=rr;save(c.OUT/'common_theory_cross_injection.json',report)
        log('cross_injection_done',kind=kind,actual_b1={v:r['theta'][1] for v,r in rr['fits'].items()},central_fake_b1={v:r['theta'][1] for v,r in rr['injections'][1]['coherent_truth_fitted_by_current'].items()})

if __name__=='__main__':run()
