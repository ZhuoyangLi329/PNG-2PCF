#!/usr/bin/env python3
"""Task432 hybrid 的固定窗口 GIC：扩展距离表、体积壳积分、25 phase 等权平均。

执行大纲
--------
1. 读取 9.18 每个 phase 的正式随机点对窗口，分别归一化后等权平均。
2. 沿用 Task432 的 Gaussian CLPT/GSM + 最小 PNG 响应，仅延长内部 r 表。
3. 用与上游等价的批量 streaming 积分求完整窗口上的 xi0，并做壳平均。
4. C_W(fNL,b1)=<xi0>_RR；正式预测 xi0-C_W、xi2，物理自由参数不变。

这是项目既定的各向同性 formal-GIC 常数处方，不宣称实现一般各向异性
三点窗口的全部交叉项。窗口几何固定；C_W 与 hybrid 使用同一底层谱和低 k 约定。
"""
from __future__ import annotations

import task432_hybrid_jaxpower_0918_contract as common
from task432_hybrid_jaxpower_0918_contract import np, Path, json, sha
from task432_lightcone_png_velocileptors import LightconePNGVelocileptors, build_cache, RSD_P_PAYLOAD

ROOT = common.ROOT
WINDOW_ROOT = ROOT/'outputs/task43_outputs/rsd_validation/lightcone_boxsafe_zobs0p4_0p8/formal_gic_windows'
MEAN_WINDOW = WINDOW_ROOT/'task43_rsd_boxsafe_formal_gic_x25_equal_phase_mean_nsub200000_seedbase430340.npz'
SHARED = ROOT/'outputs/task43_outputs/rsd_validation/task432_model_repair/hybrid_gic_window'


class PairWindow:
    def __init__(self):
        """读取固定的 25 个 phase；输出共有边界、各 phase 概率及其算术均值。"""
        with np.load(MEAN_WINDOW, allow_pickle=False) as a:
            meta = json.loads(str(a['meta_json'].item()))
            self.k = a['k_eff']; self.w2 = a['w2']
        rows=[]; probabilities=[]; edges=None
        for i, (name, expected) in enumerate(zip(meta['input_paths'], meta['input_sha256'])):
            path=Path(name)
            if sha(path)!=expected: raise RuntimeError(f'Window hash changed: {path}')
            with np.load(path, allow_pickle=False) as a:
                ee=a['s_edges']; prob=a['pair_prob']
                if edges is None: edges=ee
                if not np.array_equal(edges,ee): raise RuntimeError('Phase radial edges differ')
                coverage=float(prob.sum())
                if coverage<.999 or np.any(prob<0): raise RuntimeError('Invalid RR coverage')
                probabilities.append(prob/coverage)
                rows.append({'phase':f'ph{i:03d}','path':str(path),'sha256':expected,'coverage':coverage})
        if len(rows)!=25: raise RuntimeError('Expected 25 phase windows')
        self.edges=edges; self.phase_prob=np.asarray(probabilities); self.prob=self.phase_prob.mean(axis=0)
        self.meta={'mean_window':str(MEAN_WINDOW),'mean_window_sha256':sha(MEAN_WINDOW),'phases':rows,
                   'normalization':'Each phase RR normalized independently, then arithmetic mean of 25 phases',
                   's_range':[float(edges[0]),float(edges[-1])],
                   'pair_weight_ge600':float(self.prob[edges[:-1]>=600].sum())}

    def quadrature(self,nquad=4):
        """输入每壳 Gauss 点数，返回全窗口积分节点及归一化 RR×r² 权重。"""
        x,w=np.polynomial.legendre.leggauss(int(nquad));lo=self.edges[:-1,None];hi=self.edges[1:,None]
        r=(hi-lo)*x[None,:]/2+(hi+lo)/2
        rw=(hi-lo)*w[None,:]/2*r*r/((hi**3-lo**3)/3)
        weights=rw*self.prob[:,None]
        active=weights.ravel()>0
        return r.ravel()[active],weights.ravel()[active]


def theory_inputs():
    """返回原 hybrid 已验证缓存的 k、P_L、alpha、增长率；不生成新的宇宙学模板。"""
    with np.load(RSD_P_PAYLOAD,allow_pickle=False) as a: z=float(a['zeff'])
    path=build_cache(zeff=z,boxsize=2000.,kmax=3.,ells=(0,2),cosmology='abacus_c000')
    with np.load(path,allow_pickle=False) as a:
        arrays={key:a[key] for key in ('k_eff','pk_dd','alpha','f_growth')}
    return arrays,path


class HybridGIC(LightconePNGVelocileptors):
    def __init__(self,window=None,rmax=4000.,radial_refinement=1):
        """输入固定窗口和数值距离上限；保留原 r 节点，在 600 之外延长插值表。"""
        self.window=PairWindow() if window is None else window
        a,self.theory_path=theory_inputs()
        with np.load(common.OUT/'frozen_inputs.npz',allow_pickle=False) as z: s=z['xi_centers']
        super().__init__(a['k_eff'],a['pk_dd'],a['alpha'],float(a['f_growth']),s)
        original=self.gsm.rint.copy()
        rr=np.logspace(-3,5,4000)
        rr=rr[(rr>.1)&(rr<float(rmax))]
        if int(radial_refinement)>1:
            lognodes=np.log(rr)
            rr=np.exp(np.interp(np.arange((len(rr)-1)*int(radial_refinement)+1)/int(radial_refinement),np.arange(len(rr)),lognodes))
        self.gsm.rint=rr
        self.radial_meta={'rmax_requested':rmax,'rmax_actual':float(rr[-1]),'n_r':len(rr),
                         'refinement':radial_refinement,
                         'original_nodes_preserved':bool(np.array_equal(rr[rr<600],original))}
        if rr[-1] < self.window.edges[-1]+150: raise RuntimeError('GSM r table does not cover RR plus streaming support')

    def poles(self,radii,*,nint=600,ngauss=4,rwidth=100.,chunk=64):
        """对已设置的累积量批量求 xi0、xi2、xi4；公式逐项对应上游 compute_xi_rsd。

        radii 是一维距离数组；nint/ngauss/rwidth 只控制积分精度，不是拟合参数。
        方差下限沿每个半径单独计算，严格复现上游标量调用的处理。
        """
        radii=np.asarray(radii,dtype='f8'); result=np.empty((3,len(radii)))
        nu,ww=np.polynomial.legendre.leggauss(2*int(ngauss))
        leg=np.array([np.ones_like(nu),5*(3*nu**2-1)/2,9*(35*nu**4-30*nu**2+3)/8])
        yy=np.linspace(-float(rwidth),float(rwidth),int(nint))[None,:]
        gsm=self.gsm;f=self.f_growth
        for start in range(0,len(radii),int(chunk)):
            ss=radii[start:start+int(chunk),None];out=np.zeros((3,len(ss)))
            for i,mu in enumerate(nu[:int(ngauss)]):
                spar=ss*mu; sperp=ss*np.sqrt(1-mu*mu)
                rr=np.sqrt((spar-yy)**2+sperp**2);mm=(spar-yy)/rr
                if rr.max()>gsm.rint[-1]: raise RuntimeError('Streaming interpolation exceeds computed r range')
                xi=1+np.interp(rr,gsm.rint,gsm.xieft)
                vv=f*np.interp(rr,gsm.rint,gsm.veft)*mm/xi
                variance=f*f*(np.interp(rr,gsm.rint,gsm.s0eft)+(3*mm*mm-1)/2*np.interp(rr,gsm.rint,gsm.s2eft))/xi-vv*vv
                floor=1e-5*np.max(variance,axis=1,keepdims=True)
                variance=np.maximum(variance,floor)
                with np.errstate(invalid='ignore',divide='ignore',over='ignore'):
                    integrand=xi*np.exp(-.5*(yy-vv)**2/variance)/np.sqrt(2*np.pi*variance)
                integrand[np.isnan(integrand)]=0.
                values=np.trapezoid(integrand,x=yy,axis=1)-1
                out+=leg[:,i,None]*ww[i]*values[None,:]
            result[:,start:start+len(ss)]=out
        if not np.all(np.isfinite(result)): raise RuntimeError('Nonfinite hybrid prediction')
        return result

    def correction(self,*,fnl,b1,nquad=4,nint=600,ngauss=4,rwidth=100.):
        """返回该参数点的确定性 GIC 常数，积分覆盖完整 RR 窗口，无拟合尺度 mask。"""
        self._set_cumulants(float(fnl),float(b1))
        r,w=self.window.quadrature(nquad)
        xi=self.poles(r,nint=nint,ngauss=ngauss,rwidth=rwidth)[0]
        return float(w@xi)

    def evaluate(self,*,fnl,b1,nquad=8,nint=1200,ngauss=4):
        """正式多极接口始终包含 GIC，避免继承父类无 GIC 的 evaluate 导致漏修正。"""
        y,_=self.prediction([fnl,b1],nquad=nquad,nint=nint,ngauss=ngauss)
        return {0:y[:len(self.s)],2:y[len(self.s):]}

    def prediction(self,theta,*,nquad=8,nint=1200,ngauss=4):
        """返回 52 维正式预测及 GIC；xi bin 中心及原 GSM 角积分口径保持一致。"""
        fnl,b1=map(float,np.asarray(theta)[:2])
        c=self.correction(fnl=fnl,b1=b1,nquad=nquad,nint=nint,ngauss=ngauss)
        y=self.poles(self.s,nint=nint,ngauss=4)
        return np.r_[y[0]-c,y[1]],c
