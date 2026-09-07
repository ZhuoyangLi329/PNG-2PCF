"""Read-only numerical diagnostics for PNG-2PCF; no production imports.

The Gaussian mode formula requires BOTH +/- members and even mode weights.
The finite-band xi operator is not automatically the unfiltered FCFC estimator.
Run --help for the optional original-cache audit. No silent covariance repair.
"""
from __future__ import annotations
import argparse
import hashlib
import json
import math
from pathlib import Path
import numpy as np
from scipy.linalg import solve_triangular
from scipy.special import eval_legendre, factorial2, hyp2f1, sici

FIT_BINS = np.array([[.003,.005],[.005,.007],[.007,.009],[.009,.011],
 [.013,.015],[.017,.019],[.021,.023],[.029,.031],[.037,.039],[.045,.047],
 [.053,.055],[.061,.063],[.069,.071],[.077,.079],[.085,.087],[.093,.095]])


def finite(value, name):
    result = np.asarray(value, dtype=float)
    if not np.all(np.isfinite(result)):
        raise ValueError(f'{name} contains non-finite entries')
    return result


class GaussianMetric:
    """One fixed covariance for optimization, MCMC and diagnostics.

    Standardize before factorization. Reject unresolved/indefinite matrices;
    never change physical marginal blocks or silently drop eigenvectors.
    """
    def __init__(self, covariance, tolerance=1e-12):
        c = finite(covariance, 'covariance').copy()
        if c.ndim != 2 or c.shape[0] != c.shape[1] or not c.size:
            raise ValueError('covariance must be a nonempty square matrix')
        if np.any(np.diag(c) <= 0):
            raise ValueError('nonpositive marginal variance')
        self.scale = np.sqrt(np.diag(c))
        r = c / np.outer(self.scale, self.scale)
        if not np.allclose(r, r.T, atol=1e-12, rtol=0):
            raise ValueError('covariance is not symmetric in normalized units')
        r = (r+r.T)/2
        self.eigenvalues = np.linalg.eigvalsh(r)
        if self.eigenvalues[0] <= tolerance:
            raise ValueError(f'unresolved correlation matrix: min eigenvalue {self.eigenvalues[0]:g}')
        self.chol = np.linalg.cholesky(r)
        self.covariance = (c+c.T)/2

    def residual(self, residual):
        r = finite(residual, 'residual')
        if r.shape != self.scale.shape:
            raise ValueError('residual shape mismatch')
        return solve_triangular(self.chol, r/self.scale, lower=True)

    def chi2(self, residual):
        w = self.residual(residual)
        return float(w@w)

    def solve(self, rhs):
        b = finite(rhs, 'right hand side')
        if b.ndim not in (1, 2) or b.shape[0] != self.scale.size:
            raise ValueError('right hand side shape mismatch')
        scale = self.scale if b.ndim == 1 else self.scale[:,None]
        z = solve_triangular(self.chol, b/scale, lower=True)
        return solve_triangular(self.chol.T, z, lower=False)/scale


def assemble_covariance(cpp, cxx, cxp=None):
    p, x = GaussianMetric(cpp), GaussianMetric(cxx)
    cross = np.zeros((len(x.scale),len(p.scale))) if cxp is None else finite(cxp,'cross block')
    if cross.shape != (len(x.scale),len(p.scale)):
        raise ValueError('cross block must have xi rows and P columns')
    c = np.block([[p.covariance,cross.T],[cross,x.covariance]])
    GaussianMetric(c)
    return c


def canonical_correlations(cpp, cxx, cxp):
    p, x = GaussianMetric(cpp), GaussianMetric(cxx)
    a = finite(cxp,'cross block')
    if a.shape != (len(x.scale),len(p.scale)):
        raise ValueError('cross block shape mismatch')
    a = a/np.outer(x.scale,p.scale)
    a = solve_triangular(x.chol,a,lower=True)
    a = solve_triangular(p.chol,a.T,lower=True).T
    return np.linalg.svd(a,compute_uv=False)


def conditional_xi(rp, rx, cpp, cxx, cxp):
    assemble_covariance(cpp,cxx,cxp)
    metric = GaussianMetric(cpp)
    a = np.asarray(cxp)
    residual = np.asarray(rx)-a@metric.solve(rp)
    covariance = np.asarray(cxx)-a@metric.solve(a.T)
    return residual,covariance,GaussianMetric(covariance).chi2(residual)


def lorentzian_moments(ksigma, max_power=6, exponent=2):
    """Average mu^(2n)/(1+(k*sigma*mu)^2/2)^exponent, n=0..max_power."""
    x = finite(ksigma,'k sigma')
    if max_power < 0 or exponent < 0:
        raise ValueError('negative powers')
    return np.stack([hyp2f1(exponent,n+.5,n+1.5,-.5*x*x)/(2*n+1)
                     for n in range(max_power+1)])


def continuous_poles(k, plin, amplitude, growth, sigma):
    k,p,a = np.broadcast_arrays(finite(k,'k'),finite(plin,'P'),finite(amplitude,'amplitude'))
    if np.any(k < 0) or sigma < 0:
        raise ValueError('negative k or sigma')
    m = lorentzian_moments(k*sigma,3)
    z = a*a*m[0]+2*a*growth*m[1]+growth*growth*m[2]
    z2 = a*a*m[1]+2*a*growth*m[2]+growth*growth*m[3]
    return {0:p*z,2:2.5*p*(3*z2-z)}


def angular_total_integrals(k, plin, amplitude, growth, sigma, shot):
    """FULL integral_-1^1 [P(signal)+shot]^2 L_a L_b dmu, a,b=0,2.

    Not an angular average: do not add a second Gaussian factor of two
    when inserting these integrals into the radial g-weighted formula.
    """
    k,p,a = np.broadcast_arrays(finite(k,'k'),finite(plin,'P'),finite(amplitude,'amplitude'))
    if np.any(k < 0) or sigma < 0:
        raise ValueError('negative k or sigma')
    m2,m4 = lorentzian_moments(k*sigma,6,2),lorentzian_moments(k*sigma,6,4)
    poly = {0:[1.],2:[-.5,1.5]}
    result = {}
    for ell in (0,2):
        for ell2 in (0,2):
            value = np.zeros_like(k)
            for j,coef in enumerate(np.polynomial.polynomial.polymul(poly[ell],poly[ell2])):
                t = p*p*sum(math.comb(4,n)*a**(4-n)*growth**n*m4[j+n] for n in range(5))
                t += 2*shot*p*sum(math.comb(2,n)*a**(2-n)*growth**n*m2[j+n] for n in range(3))
                value += 2*coef*(t+shot*shot/(2*j+1))
            result[ell,ell2] = value
    return result


def shell_kernel(k, edges, ell):
    """i^ell times volume-averaged j_ell, ell=0,2; analytic primitives."""
    k,edges = finite(k,'k').reshape(-1),finite(edges,'edges')
    if ell not in (0,2) or np.any(k < 0):
        raise ValueError('requires ell=0,2 and k>=0')
    if edges.ndim != 1 or len(edges)<2 or edges[0]<0 or np.any(np.diff(edges)<=0):
        raise ValueError('invalid separation edges')
    lo,hi = edges[:-1][None,:],edges[1:][None,:]
    kk = k[:,None]
    primitive = (lambda x: np.sin(x)-x*np.cos(x)) if ell==0 else (lambda x: 3*sici(x)[0]+x*np.cos(x)-4*np.sin(x))
    with np.errstate(divide='ignore',invalid='ignore'):
        answer = 3*(primitive(kk*hi)-primitive(kk*lo))/(kk**3*(hi**3-lo**3))
    small = kk*hi < .5
    series = np.zeros_like(answer)
    for m in range(7):
        power = ell+2*m
        coef = (-1.)**m/(2**m*math.factorial(m)*factorial2(2*ell+2*m+1))
        series += coef*kk**power*3*(hi**(power+3)-lo**(power+3))/((power+3)*(hi**3-lo**3))
    answer[small] = series[small]
    return (-1)**(ell//2)*answer


def lattice_modes(boxsize, kmax, max_cells=3_000_000):
    """Small/low-k FULL +/- lattice test only. Explicitly excludes k=0."""
    if boxsize<=0 or kmax<=0:
        raise ValueError('boxsize and kmax must be positive')
    kf = 2*np.pi/boxsize
    n = int(np.ceil(kmax/kf))
    if (2*n+1)**3 > max_cells:
        raise ValueError('large 3D enumeration rejected; use a converged low-k correction')
    v = np.arange(-n,n+1,dtype=np.int32)
    modes = np.stack(np.meshgrid(v,v,v,indexing='ij'),axis=-1).reshape(-1,3)
    q = np.sum(modes.astype(np.int64)**2,axis=1)
    keep = (q>0)&(kf*np.sqrt(q)<=kmax)
    modes,q = modes[keep],q[keep]
    return modes,kf*np.sqrt(q),modes[:,2]/np.sqrt(q)


def mode_operators(k, mu, bins, edges, volume):
    """Common finite-band P02/xi02 operators, not raw FCFC pair counting."""
    k,mu,bins = finite(k,'k'),finite(mu,'mu'),finite(bins,'bins')
    if k.ndim!=1 or mu.shape!=k.shape or np.any(k<=0) or np.any(abs(mu)>1) or volume<=0:
        raise ValueError('invalid modes/volume')
    if bins.ndim!=2 or bins.shape[1]!=2 or np.any(bins[:,1]<=bins[:,0]):
        raise ValueError('invalid bins')
    selected = (k[None,:]>=bins[:,0,None])&(k[None,:]<bins[:,1,None])
    counts = selected.sum(axis=1)
    if np.any(counts==0):
        raise ValueError('empty P bin')
    wp,wx = [],[]
    for ell in (0,2):
        weight = (2*ell+1)*eval_legendre(ell,mu)
        wp.append(selected*weight/counts[:,None])
        wx.append(shell_kernel(k,edges,ell).T*weight/volume)
    return np.vstack(wp),np.vstack(wx),counts


def mode_covariance(operator, total_power):
    w,t = finite(operator,'operator'),finite(total_power,'total power')
    if w.ndim!=2 or t.shape!=(w.shape[1],) or np.any(t<=0):
        raise ValueError('invalid operator/total power')
    a = np.sqrt(2)*w*t
    return a@a.T


def rebin_geometry(boxsize=2000., bins=FIT_BINS):
    _,k,_ = lattice_modes(boxsize,float(np.max(bins))+.005)
    ku,g = np.unique(k,return_counts=True)
    idx = (ku/(.1*(2*np.pi/boxsize))).astype(np.int64)
    gg = np.bincount(idx,weights=g)
    gk = np.bincount(idx,weights=g*ku)
    mask = gg>0
    keff,gg = gk[mask]/gg[mask],gg[mask]
    exact = np.array([np.sum((k>=lo)&(k<hi)) for lo,hi in bins])
    grouped = np.array([np.sum(gg[(keff>=lo)&(keff<hi)]) for lo,hi in bins])
    return {'bins':bins.tolist(),'exact_counts':exact.tolist(),
            'grouped_counts':grouped.tolist(),'relative_error':(grouped/exact-1).tolist()}


def legacy_floor(covariance):
    """Diagnostic reproduction ONLY. Never use this function for fitting."""
    e,q = np.linalg.eigh(covariance)
    floor = max(1e-14*float(e[-1]),1e-300)
    return (q*np.maximum(e,floor))@q.T, {'min_before':float(e[0]),
        'max_before':float(e[-1]),'floor':floor,'n_floored':int(np.sum(e<floor))}


def audit_cache(cache_path, output_path):
    """Read an existing c000 cache; reproduce old blocks without refitting data.

    Audits cross=0 covariance units, narrow-bin counts, GL64 vs analytic
    moments and closed-form kernels. Does NOT validate a production pipeline.
    """
    path,out = Path(cache_path),Path(output_path)
    if out.suffix!='.json' or out.exists() or out.with_suffix('.npz').exists():
        raise ValueError('output must be a NEW .json path (and companion .npz)')
    meta = json.loads(path.with_suffix('.json').read_text())
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    if meta.get('status')!='pass' or meta.get('output_sha256')!=digest or meta.get('cosmology')!='abacus_c000':
        raise ValueError('cache status, SHA256 or cosmology validation failed')
    with np.load(path,allow_pickle=False) as data:
        d = {key:finite(data[key],key) for key in ('k_eff','g_nz','pk_dd','kernels','s_edges','ells','volume','boxsize','f_growth','zeff','kmax')}
    k,g,p = d['k_eff'],d['g_nz'],d['pk_dd']
    if k.ndim!=1 or g.shape!=k.shape or p.shape!=k.shape or np.any(k<=0) or np.any(np.diff(k)<=0) or np.any(g<=0) or np.any(p<0):
        raise ValueError('invalid cache grid')
    if not np.array_equal(d['ells'],[0,2]) or not np.array_equal(d['s_edges'],np.arange(30.,360.,10.)) or d['kernels'].shape!=(2,len(k),32):
        raise ValueError('unexpected multipoles/shells/kernel shape')
    if not np.isclose(d['boxsize'],2000.) or not np.isclose(d['volume'],2000.**3) or not np.isclose(d['zeff'],.725) or not np.isclose(d['kmax'],3.):
        raise ValueError('unexpected box geometry')
    b,f,sigma,shot,v = 2.55,float(d['f_growth']),8.,1/.000162131295,float(d['volume'])
    u,w = np.polynomial.legendre.leggauss(64)
    signal = p[:,None]*(b+f*u*u)**2/(1+.5*(k[:,None]*u*sigma)**2)**2
    angles = {(a,c):np.sum(w*(signal+shot)**2*eval_legendre(a,u)*eval_legendre(c,u),axis=1) for a in (0,2) for c in (0,2)}
    centers = (d['s_edges'][1:]+d['s_edges'][:-1])/2
    masks = {0:centers>=50,2:centers>=80}
    kernels = {ell:d['kernels'][ell//2][:,masks[ell]] for ell in (0,2)}
    cxx = np.block([[(2*a+1)*(2*c+1)*kernels[a].T@((g*angles[a,c]/v**2)[:,None]*kernels[c]) for c in (0,2)] for a in (0,2)])
    geometry = rebin_geometry()
    counts = np.asarray(geometry['exact_counts'])
    cpp = np.zeros((32,32))
    cxp = np.zeros((57,32))
    cached_counts = []
    for i,(lo,hi) in enumerate(FIT_BINS):
        keep = (k>=lo)&(k<hi)
        cached_counts.append(float(g[keep].sum()))
        for a in (0,2):
            for c in (0,2):
                ww = (2*a+1)*(2*c+1)*g[keep]*angles[a,c][keep]
                cpp[(a//2)*16+i,(c//2)*16+i] = ww.sum()/counts[i]**2
                rows = slice(0,30) if c==0 else slice(30,57)
                cxp[rows,(a//2)*16+i] = ww@kernels[c][keep]/(counts[i]*v)
    fixed = assemble_covariance(cpp,cxx)
    old,floor_meta = legacy_floor(fixed)
    scale = np.r_[np.full(32,1e-6),np.ones(57)]
    old_rescaled = legacy_floor(fixed*np.outer(scale,scale))[0]/np.outer(scale,scale)
    poles64 = {ell:.5*(2*ell+1)*np.sum(w*signal*eval_legendre(ell,u),axis=1) for ell in (0,2)}
    poles_exact = continuous_poles(k,p,b,f,sigma)
    def project(poles,ks):
        return np.concatenate([(g*poles[ell])@ks[ell]/v for ell in (0,2)])
    xi64,xi_exact = project(poles64,kernels),project(poles_exact,kernels)
    closed = {ell:shell_kernel(k,d['s_edges'],ell)[:,masks[ell]] for ell in (0,2)}
    xi_closed = project(poles_exact,closed)
    metric = GaussianMetric(cxx)
    rho = canonical_correlations(cpp,cxx,cxp)
    scaled = cxp.copy()
    for a,c,z in ((0,0,1.0304495708031909),(0,2,2.5061267644581964),(2,0,2.336048060830107),(2,2,1.7207911170155552)):
        scaled[(slice(0,30) if c==0 else slice(30,57)),(slice(0,16) if a==0 else slice(16,32))] *= z
    report = {'scope':'original-cache numerical audit only; no data fit or estimator closure',
      'cache_sha256':digest,'fiducial':{'b1':b,'sigma_s':sigma,'fNL':0.,'nbar':1/shot},
      'legacy_floor':floor_meta,'xi_sigma_ratio_median':float(np.median(np.sqrt(np.diag(old)[32:]/np.diag(cxx)))),
      'legacy_unit_dependence':float(np.linalg.norm(old[32:,32:]-old_rescaled[32:,32:])/np.linalg.norm(old[32:,32:])),
      'geometry':geometry,'original_cache_bin_counts':cached_counts,
      'measurement_nmodes_check':'NOT RUN: exact lattice counts are not a measurement-file check',
      'rho_max_analytic':float(rho[0]),'rho_max_quadrant_scaled':float(canonical_correlations(cpp,cxx,scaled)[0]),
      'prediction_change_chi2_Cmean':{'GL64_to_analytic_angles':25*metric.chi2(xi_exact-xi64),
      'cached_to_closed_shell_kernel':25*metric.chi2(xi_closed-xi_exact)}}
    out.parent.mkdir(parents=True,exist_ok=True)
    with out.with_suffix('.npz').open('xb') as stream:
        np.savez_compressed(stream,Cpp=cpp,Cxx=cxx,Cxp=cxp,xi_GL64=xi64,xi_analytic=xi_exact,xi_closed_kernel=xi_closed)
    with out.open('x') as stream:
        json.dump(report,stream,indent=2); stream.write('\n')
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--cache',type=Path,required=True,help='original theory NPZ with validated JSON sidecar')
    parser.add_argument('--out',type=Path,required=True,help='NEW output .json, also creates .npz')
    args = parser.parse_args()
    try:
        print(json.dumps(audit_cache(args.cache,args.out),indent=2))
    except (OSError,ValueError,KeyError) as exc:
        parser.exit(2,f'audit failed, with no fallback or cache rebuild: {exc}\n')


if __name__=='__main__':
    main()
