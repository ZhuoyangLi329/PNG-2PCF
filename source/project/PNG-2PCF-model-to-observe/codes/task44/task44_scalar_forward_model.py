#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Task44 scalar-LS前向模型：完整多极投影、实际RR权重与可验证的sigma插值。

大纲：
1. lorentz_moments 稳定解析计算 squared-Lorentzian 的mu偶次矩。
2. ScalarReference 保持母盒k支持、cosmology和原PNG二次项，建立同一
   (s,mu)观测算子；legacy、RR混合及各向异性formal-GIC明确分开。
3. build_cache 只在sigma_s轴插值，将b与PNG幅度的多项式关系精确保留。
   缓存必须通过原模型自重现和随机参数点的C^{-1}范数误差门。
4. FastScalar/FastPk提供批量预测，供独立ensemble和成对mock复用。
这些类不改变数据尺度，不做Fourier band matching，不自动推广新科学主线。
"""
from __future__ import annotations
import argparse,json,os,sys,time
from pathlib import Path
for key in ['OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS','NUMEXPR_NUM_THREADS']:os.environ[key]='1'
import numpy as np
from scipy.interpolate import CubicSpline
from scipy.special import spherical_jn
R=Path('/pscratch/sd/l/lzy/PNG-2PCF-model-to-observe');sys.path.insert(0,str(R/'codes/task44'))
import task44_fit_lrg2_rsd as xi
import task44_fit_pk_lrg2 as pk
from task44_rsd_theory import build_theory_context,mu_moments
from task43_theory_template import build_template_arrays,load_task41
from task44_fit_validation import file_digest
from task44_diagnose_pk_xi_transfer import read,save,whiten_matrix
from task44_measure_random75x_xi import atomic_savez
O=R/'outputs/task44_outputs/method_refinement_20260905';ELLS=(0,2,4,6,8)
_BASE=None


def lorentz_moments(k,sigma,nmax=6):
    """返回 I_n=积分_0^1 mu^(2n)/(1+(k*sigma)^2*mu^2/2)^2 dmu。

    小a使用收敛幂级数，避免递推相消；大a使用解析I0及J_n递推。
    nmax=6足以构造RSD的ell<=8和Kaiser mu^4项。
    """
    a=.5*(np.asarray(k,dtype='f8')*float(sigma))**2
    out=np.empty((nmax+1,a.size));small=a<.3
    if np.any(small):
        x=a[small];powers=np.ones_like(x);vals=np.zeros((nmax+1,len(x)))
        for m in range(40):
            vals+=((-1.)**m*(m+1)*powers)[None,:]/(2*np.arange(nmax+1)[:,None]+2*m+1)
            powers*=x
        out[:,small]=vals
    if np.any(~small):
        x=a[~small];u=np.sqrt(x);j=np.arctan(u)/u
        last=.5*(1/(1+x)+j);out[0,~small]=last
        for n in range(1,nmax+1):
            current=(j-last)/x;out[n,~small]=current
            j=(1/(2*n-1)-j)/x;last=current
    return out


def legendre_damping(moments,ell):
    """由偶次mu矩构造积分D(mu)Lell(mu)mu^(0,2,4)的三个因子。"""
    coeff=np.polynomial.legendre.leg2poly(np.r_[np.zeros(ell),1.])[::2]
    return np.array([(2*ell+1)*sum(c*moments[n+p] for n,c in enumerate(coeff)) for p in range(3)])


def coefficients(theta,p_fixed,f_growth):
    """对单点或批量theta=[fNL,b1,sigma_s,...]给出精确PNG多项式系数。"""
    t=np.atleast_2d(theta);b=t[:,1];amp=3.372*t[:,0]*(b-float(p_fixed));f=float(f_growth)
    return np.stack([b*b,2*b*amp,amp*amp,2*b*f,2*amp*f,np.full(len(t),f*f)],axis=1)


def kernel_chunked(k,edges,ell):
    """按k分块调用原投影核，控制高阶shell积分的临时内存。"""
    result=np.empty((len(k),len(edges)))
    for start in range(0,len(k),256):
        sl=slice(start,start+256)
        result[sl]=xi.build_xi_kernel(k[sl],edges.mean(axis=1),edges,'shell-averaged',ell=ell)[0]
    return result


class ScalarReference:
    """给定样本建立直接求值基准；不依赖sigma插值，保留独立原P0自重现。"""
    def __init__(self,sample):
        global _BASE
        self.sample=sample;entry=read(O/'inventory.json')['samples'][sample]
        self.case=entry['historical_results'];self.summary=read(self.case['xi']['provenance']['summary']);cfg=self.summary['theory']
        self.config=cfg;self.p=self.summary['p_fixed'];self.f=cfg['f_growth'];self.task41=load_task41()
        assert cfg['fog_model']=='lorentzian' and cfg['kmax']==5 and cfg['ndense']==8000 and cfg['boxsize']==2000
        if _BASE is None:_BASE=build_theory_context(self.summary['zeff'],kmax=5.,ndense=8000,boxsize=2000.,cosmology=cfg['cosmology'])
        self.context=dict(_BASE);self.kd=_BASE['k_dense'];self.ke=_BASE['k_eff'];self.g=_BASE['g_nz'];self.volume=_BASE['volume']
        kt=np.geomspace(min(1e-4,_BASE['kfund']/2),6.,4000)
        template,_=build_template_arrays(self.task41,kt,z=self.summary['zeff'],cosmology=cfg['cosmology'])
        self.P=self.task41.interp_logk(self.kd,template['k'],template['pk_dd']);self.alpha=self.task41.interp_logk(self.kd,template['k'],template['alpha'])
        self.edges=np.column_stack([np.arange(30.,350.,10.),np.arange(40.,360.,10.)]);self.s=self.edges.mean(axis=1)
        self.K0=kernel_chunked(self.ke,self.edges,0)
        prov=self.case['xi']['provenance'];self.legacy_window=Path(prov.get('formal_gic_cache',prov.get('formal_gic_window')))
        with np.load(self.legacy_window) as z:
            assert np.array_equal(self.ke,z['k_eff']);self.w2=z['w2']
        self.window_path=O/'windows'/sample/'scalar_rr_window.npz'
        with np.load(self.window_path) as z:
            self.fine_edges=np.column_stack([z['s_edges_fine'][:-1],z['s_edges_fine'][1:]])
            self.rrfine=z['rrmu_mean'].sum(axis=1);self.afine=z['a_ell_fine'];self.gic_edges=np.column_stack([z['gic_s_edges'][:-1],z['gic_s_edges'][1:]])
            self.gic_blocks=z['gic_pair_moments_blocks'];self.rr_blocks=z['rrmu_blocks']
        nsub=len(self.fine_edges)//len(self.edges);assert nsub*len(self.edges)==len(self.fine_edges)
        assert np.array_equal(self.fine_edges[::nsub,0],self.edges[:,0]) and np.array_equal(self.fine_edges[nsub-1::nsub,1],self.edges[:,1])
        denominator=self.rrfine.reshape(len(self.edges),nsub).sum(axis=1)
        self.Kgeom={};self.Kgic={};self.Wgic={};self.gic_active=self.ke<=xi.K_CUT_WINDOW
        for j,ell in enumerate(ELLS):
            fine=kernel_chunked(self.ke,self.fine_edges,ell)
            weighted=fine*(self.rrfine*self.afine[j])[None,:]
            self.Kgeom[ell]=weighted.reshape(len(self.ke),len(self.edges),nsub).sum(axis=2)/denominator[None,:]
            self.Kgic[ell]=kernel_chunked(self.ke[self.gic_active],self.gic_edges,ell)
            self.Wgic[ell]=np.zeros_like(self.ke)
            self.Wgic[ell][self.gic_active]=self.Kgic[ell]@self.gic_blocks[:,:,j,:].mean(axis=(0,1))
        # 使用已有物理矩阵作为数值误差度量；不在缓存构造时决定最终C。
        inv=read(R/'outputs/task44_outputs/pipeline_audit_20260905/artifact_inventory.json')['rsd'][sample]
        with np.load(inv['physical']) as z:self.cov=z['covariance_single_realization'][:len(self.s),:len(self.s)]
        self.whitener=whiten_matrix(self.cov)

    def basis(self,sigma,ell):
        """直接给出六个参数系数的Pell离散求和权重，保持原log-k线性插值。"""
        damping=legendre_damping(lorentz_moments(self.kd,sigma),ell)
        a=self.alpha;P=self.P
        dense=P[None,:]*np.array([damping[0],a*damping[0],a*a*damping[0],damping[1],a*damping[1],damping[2]])
        return np.array([self.task41.interp_logk(self.ke,self.kd,row) for row in dense])*(self.g/self.volume)[None,:]

    def tables(self,sigma):
        """返回某sigma的几何、legacy、GIC系数表，可用于精确求值或缓存。"""
        geom=[];gic=[];gic_batches=[]
        for j,ell in enumerate(ELLS):
            weight=self.basis(sigma,ell)
            geom.append(weight@self.Kgeom[ell]);gic.append(weight@self.Wgic[ell])
            radial=weight[:,self.gic_active]@self.Kgic[ell]
            gic_batches.append(np.einsum('br,ijr->ijb',radial,self.gic_blocks[:,:,j,:]))
            if ell==0:legacy=weight@self.K0;legacy_gic=weight@self.w2
        return {'geometry':np.array(geom),'gic':np.array(gic),'legacy':legacy,'legacy_gic':legacy_gic,'gic_batches':np.array(gic_batches)}

    def predict(self,theta,mode='rr_anisotropic_gic',ellmax=8):
        """直接模型；formal-GIC仍为全距离配对均值，未冒充完整三项IC。"""
        tables=self.tables(float(theta[2]));c=coefficients(theta,self.p,self.f)[0]
        return combine_tables(tables,c,mode,ellmax)

    def legacy_original(self,theta):
        """直接调用原mu_moments公式，用于独立验证新解析矩与旧P0结果一致。"""
        fnl,b,sigma=theta[:3];amp=b+3.372*fnl*(b-self.p)*self.alpha
        i0,i2,i4=mu_moments(self.kd,sigma,fog_model='lorentzian')
        power=self.P*(amp*amp*i0+2*amp*self.f*i2+self.f*self.f*i4)
        weight=self.g*self.task41.interp_logk(self.ke,self.kd,power)/self.volume
        return weight@self.K0-weight@self.w2


def combine_tables(tables,c,mode,ellmax):
    """显式选择观测/IC合同，避免用不同含义的xi0标签混合结果。"""
    ne=ELLS.index(ellmax)+1
    if mode=='legacy':return c@tables['legacy']-c@tables['legacy_gic']
    if mode=='pure_no_gic':return c@tables['legacy']
    out=np.einsum('ebs,b->s',tables['geometry'][:ne],c)
    if mode=='rr_legacy_gic':out-=c@tables['legacy_gic']
    elif mode=='rr_anisotropic_gic':out-=np.einsum('eb,b->',tables['gic'][:ne],c)
    elif mode!='rr_no_gic':raise ValueError(mode)
    return out


class FastScalar:
    """从已验证缓存提供单点/批量预测；超出sigma表范围返回错误而不外推。"""
    def __init__(self,path,mode='rr_anisotropic_gic',ellmax=8):
        self.path=Path(path);self.mode=mode;self.ellmax=ellmax
        with np.load(path) as z:
            self.s=z['s'];self.p=float(z['p_fixed']);self.f=float(z['f_growth']);self.sigma=z['sigma_grid']
            self.splines={k:CubicSpline(self.sigma,z['table_'+k],axis=0,extrapolate=False) for k in ['geometry','gic','legacy','legacy_gic','gic_batches']}
    def __call__(self,theta):
        scalar=np.asarray(theta).ndim==1;t=np.atleast_2d(theta);c=coefficients(t,self.p,self.f);sig=t[:,2]
        ne=ELLS.index(self.ellmax)+1
        if self.mode in ['legacy','pure_no_gic']:
            out=np.einsum('nbs,nb->ns',self.splines['legacy'](sig),c)
            if self.mode=='legacy':out-=np.einsum('nb,nb->n',self.splines['legacy_gic'](sig),c)[:,None]
        else:
            out=np.einsum('nebs,nb->ns',self.splines['geometry'](sig)[:,:ne],c)
            if self.mode=='rr_legacy_gic':out-=np.einsum('nb,nb->n',self.splines['legacy_gic'](sig),c)[:,None]
            elif self.mode=='rr_anisotropic_gic':out-=np.einsum('neb,nb->n',self.splines['gic'](sig)[:,:ne],c)[:,None]
            elif self.mode!='rr_no_gic':raise ValueError(self.mode)
        if not np.all(np.isfinite(out)):raise ValueError('sigma outside validated interpolation domain')
        return out[0] if scalar else out


def sigma_grid(refinement=1):
    """在sigma≈0加密，以解析矩在无FoG极限的快速变化为依据。"""
    return np.unique(np.r_[np.linspace(0,.5,21*refinement),np.linspace(.5,2.,13*refinement),np.linspace(2.,30.,113*refinement)])


def build_cache(sample):
    """先重现旧MAP，再生成插值表并以C^{-1}误差范数验证；失败则自动加密一次。"""
    t=time.time();root=O/'model_cache'/sample;root.mkdir(parents=True,exist_ok=True)
    out=root/'scalar_forward_cache.npz';js=out.with_suffix('.json')
    window=O/'windows'/sample/'scalar_rr_window.npz'
    if js.exists():
        old=read(js)
        assert old['script_sha256']==file_digest(__file__) and old['window_sha256']==file_digest(window) and old['output_sha256']==file_digest(out)
        print('[verified scalar cache]',sample,flush=True);return
    ref=ScalarReference(sample)
    theta=np.array([ref.summary['models']['formal_gic']['map'][k] for k in ['fnl_loc','b1','sigma_s']])
    original=ref.legacy_original(theta);new=ref.predict(theta,'legacy')
    with np.load(ref.case['xi']['provenance']['chains']) as z:
        old_s=z['s'];saved=z['formal_gic_model_map'];ids=np.array([np.flatnonzero(ref.s==x)[0] for x in old_s])
    repro={'new_vs_original_max_abs':float(np.max(abs(new-original))),'original_vs_saved_max_abs':float(np.max(abs(original[ids]-saved)))}
    if max(repro.values())>1e-10:raise RuntimeError(f'legacy self reproduction failed: {repro}')
    rng=np.random.default_rng(20260910)
    test=np.column_stack([rng.uniform(-500,500,60),rng.uniform(.2,10.,60),rng.uniform(0,30,60)])
    test=np.vstack([theta,test,np.array([[100,2,s] for s in [0.,.013,.071,.113,.375,.73,1.13,2.13,29.99,30.]])])
    # 直接模型在固定测试点只算一次，不能用插值表检验自身。
    truth={mode:[] for mode in ['legacy','rr_legacy_gic','rr_anisotropic_gic']}
    for point in test:
        direct_tables=ref.tables(point[2]);c=coefficients(point,ref.p,ref.f)[0]
        for mode in truth:truth[mode].append(combine_tables(direct_tables,c,mode,8))
    truth={mode:np.array(rows) for mode,rows in truth.items()}
    attempts=[]
    for refinement in [1,2]:
        sig=sigma_grid(refinement);tables={k:[] for k in ref.tables(0.)}
        for index,sigma in enumerate(sig):
            now=ref.tables(sigma)
            for key in tables:tables[key].append(now[key])
            if index%30==0:print('[cache grid]',sample,index,len(sig),flush=True)
        atomic_savez(out,s=ref.s,s_edges=ref.edges,p_fixed=np.array(ref.p),f_growth=np.array(ref.f),sigma_grid=sig,
                     **{'table_'+k:np.array(v) for k,v in tables.items()})
        errors={}
        for mode in truth:
            model=FastScalar(out,mode);difference=model(test)-truth[mode]
            errors[mode]={'max_abs':float(np.max(abs(difference))),'max_covariance_norm':float(np.max(np.linalg.norm(difference@ref.whitener.T,axis=1)))}
        attempts.append({'sigma_nodes':len(sig),'errors':errors})
        if max(v['max_covariance_norm'] for v in errors.values())<.05:break
    else:raise RuntimeError(f'sigma interpolation error budget failed: {attempts}')
    result={'status':'numerically_validated','sample':sample,'script_sha256':file_digest(__file__),'window_sha256':file_digest(window),
            'output_sha256':file_digest(out),'self_reproduction':repro,'interpolation_validation':attempts,
            'numerical_error_budget':'C^-1 norm(delta_model)<0.05 on fixed random test set; local parameter bias bound0.05sigma',
            'modes':['legacy','rr_legacy_gic','rr_anisotropic_gic','rr_no_gic','pure_no_gic'],
            'formal_GIC_only':True,'full_density_IC_cross_terms':False,'ellmax_available':8,
            'scientific_status':'operator-informed diagnostic model; full IC/covariance/mock validation tracked separately',
            'elapsed_seconds':time.time()-t}
    save(js,result);print('[done scalar cache]',sample,'elapsed',result['elapsed_seconds'],flush=True)


def main():
    p=argparse.ArgumentParser();p.add_argument('--samples',nargs='+',default=['lrg2','qso1','qsoall','lrgall','lrg1','lrg3','qso2','qso3']);a=p.parse_args()
    assert len(os.sched_getaffinity(0))<=8
    for sample in a.samples:build_cache(sample)

if __name__=='__main__':main()
