"""Read-only audit of an ORIGINAL rawbox theory cache. No MCMC/catalog jobs.

python GPT5.6-respose/audit_rawbox.py . --out /path/to/NEW/audit.json
A checksum sidecar is mandatory. Output JSON and NPZ must not already exist.
Use --measurement existing_P02.npz to additionally check measured mode counts.
"""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
from scipy.special import eval_legendre
from rawbox_numerics import (GaussianMetric, assemble_covariance,
    canonical_correlations, angular_total_integrals, continuous_poles,
    shell_kernel, lattice_modes)

BINS = np.array([[.003,.005],[.005,.007],[.007,.009],[.009,.011],
    [.013,.015],[.017,.019],[.021,.023],[.029,.031],[.037,.039],
    [.045,.047],[.053,.055],[.061,.063],[.069,.071],[.077,.079],
    [.085,.087],[.093,.095]])
CACHE = ('source/project/outputs/task43_outputs/rsd_validation/theory/'
         'task43_rsd_fulldiscrete_z0p725000_box2000_kmax3_ell02.npz')
SCALES = {(0,0):1.0304495708031909, (0,2):2.5061267644581964,
          (2,0):2.336048060830107, (2,2):1.7207911170155552}


def checksum(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b''):
            h.update(chunk)
    return h.hexdigest()


def load_cache(path):
    path = Path(path)
    meta = json.loads(path.with_suffix('.json').read_text())
    digest = checksum(path)
    if meta.get('status') != 'pass' or meta.get('output_sha256') != digest:
        raise ValueError('Cache status/checksum failed; no automatic rebuild')
    if meta.get('cosmology') != 'abacus_c000':
        raise ValueError('This audit is scoped to abacus_c000')
    with np.load(path, allow_pickle=False) as data:
        d = {key: np.asarray(data[key]) for key in data.files}
    for key in ('k_eff','g_nz','pk_dd','alpha','kernels','s_edges','ells','volume','boxsize','f_growth'):
        if key not in d or not np.all(np.isfinite(d[key])):
            raise ValueError(f'Missing/non-finite cache field: {key}')
    k = d['k_eff']
    if k.ndim != 1 or np.any(k <= 0) or np.any(np.diff(k) <= 0):
        raise ValueError('Invalid cache k grid')
    for key in ('g_nz','pk_dd','alpha'):
        if d[key].shape != k.shape:
            raise ValueError(f'Shape mismatch: {key}')
    if np.any(d['g_nz'] <= 0) or np.any(d['pk_dd'] < 0):
        raise ValueError('Invalid mode weights or matter spectrum')
    if not np.array_equal(d['ells'], [0,2]) or not np.array_equal(d['s_edges'], np.arange(30.,360.,10.)):
        raise ValueError('Unexpected multipoles/separation edges')
    if d['kernels'].shape != (2,k.size,32):
        raise ValueError('Wrong kernel shape')
    if not np.isclose(float(d['boxsize']),2000.) or not np.isclose(float(d['volume']),2000.**3):
        raise ValueError('Unexpected box geometry')
    return d, digest


def angular_gauss(d, b1, sigma, nbar, nmu=64):
    u, w = np.polynomial.legendre.leggauss(nmu)
    k, p, f = d['k_eff'], d['pk_dd'], float(d['f_growth'])
    signal = p[:,None]*(b1+f*u[None,:]**2)**2/(1+.5*(k[:,None]*u[None,:]*sigma)**2)**2
    total2 = (signal+1/nbar)**2
    return {(a,b): np.sum(w[None,:]*total2*eval_legendre(a,u)[None,:]*eval_legendre(b,u)[None,:],axis=1)
            for a in (0,2) for b in (0,2)}


def covariance_blocks(d, counts, angular):
    """Reproduce legacy radial-bin membership; NOT the proposed correction."""
    k, g, v = d['k_eff'], d['g_nz'], float(d['volume'])
    centers = (d['s_edges'][1:]+d['s_edges'][:-1])/2
    kernels = {0:d['kernels'][0][:,centers>=50], 2:d['kernels'][1][:,centers>=80]}
    offsets = {0:0, 2:kernels[0].shape[1]}
    nx, nk = sum(t.shape[1] for t in kernels.values()), len(BINS)
    cpp, cxx, cxp = np.zeros((2*nk,2*nk)), np.zeros((nx,nx)), np.zeros((nx,2*nk))
    for a in (0,2):
        for b in (0,2):
            prefactor = (2*a+1)*(2*b+1)
            ka, kb = kernels[a], kernels[b]
            cxx[offsets[a]:offsets[a]+ka.shape[1],offsets[b]:offsets[b]+kb.shape[1]] = (
                ka.T @ ((g*angular[a,b]*prefactor/v**2)[:,None]*kb))
            for i,(lo,hi) in enumerate(BINS):
                keep = (k>=lo)&(k<hi)
                weights = g[keep]*angular[a,b][keep]*prefactor
                cpp[(a//2)*nk+i,(b//2)*nk+i] = np.sum(weights)/counts[i]**2
                cxp[offsets[b]:offsets[b]+kb.shape[1],(a//2)*nk+i] = weights@kb[keep]/(counts[i]*v)
    return cpp, (cxx+cxx.T)/2, cxp


def old_floor(c):
    e, q = np.linalg.eigh((c+c.T)/2)
    floor = max(1e-14*float(e[-1]),1e-300)
    return (q*np.maximum(e,floor))@q.T, {'min_before':float(e[0]),'max_before':float(e[-1]),
                                      'floor':floor,'n_floored':int(np.sum(e<floor))}


def vector(poles, kernels, g, v, centers):
    return np.r_[(g*poles[0])@kernels[0][:,centers>=50]/v,
                 (g*poles[2])@kernels[1][:,centers>=80]/v]


def run_audit(root, out, cache=None, measurement=None, nbar=.000162131295, b1=2.55, sigma=8.):
    root, out = Path(root).resolve(), Path(out)
    npz_out = out.with_suffix('.npz')
    if out.exists() or npz_out.exists():
        raise FileExistsError('Refusing to overwrite audit output')
    if out.suffix.lower() != '.json':
        raise ValueError('--out must end in .json')
    if not np.all(np.isfinite([nbar,b1,sigma])) or nbar <= 0 or sigma < 0:
        raise ValueError('Invalid fiducial parameters')
    path = Path(cache) if cache is not None else root/CACHE
    d, digest = load_cache(path)
    v, f = float(d['volume']), float(d['f_growth'])
    _, km, mu = lattice_modes(float(d['boxsize']), float(BINS.max()))
    keep_bins = [(km>=lo)&(km<hi) for lo,hi in BINS]
    counts = np.array([int(s.sum()) for s in keep_bins])
    measurement_check = 'not_requested: enumerated counts, not checked against measured nmodes'
    if measurement is not None:
        with np.load(measurement, allow_pickle=False) as m:
            edges, measured = np.asarray(m['k_edges']), np.asarray(m['nmodes'])
        if edges.ndim == 1:
            edges = np.column_stack([edges[:-1], edges[1:]])
        if edges.ndim != 2 or edges.shape[1] != 2:
            raise ValueError('Invalid measured Fourier edges')
        indices = []
        for row in BINS:
            match = np.flatnonzero(np.all(np.isclose(edges,row,rtol=0,atol=1e-12),axis=1))
            if match.size != 1:
                raise ValueError('Measured Fourier edges do not uniquely match')
            indices.append(int(match[0]))
        if not np.allclose(measured[indices],counts,rtol=0,atol=1e-8):
            raise ValueError('Measured and full +/- enumerated mode counts differ')
        measurement_check = 'pass'
    k,g,p = d['k_eff'],d['g_nz'],d['pk_dd']
    rebinned_counts = np.array([np.sum(g[(k>=lo)&(k<hi)]) for lo,hi in BINS])
    old_ang = angular_gauss(d,b1,sigma,nbar)
    exact_ang = angular_total_integrals(k,p,b1,f,sigma,1/nbar)
    cpp,cxx,cxp = covariance_blocks(d,counts,old_ang)
    _,cxx_a,_ = covariance_blocks(d,counts,exact_ang)
    fixed = assemble_covariance(cpp,cxx)
    floored,floor_meta = old_floor(fixed)
    nk, nx = cpp.shape[0], cxx.shape[0]
    units = np.r_[np.full(nk,1e-6),np.ones(nx)]
    altered = old_floor(fixed*np.outer(units,units))[0]/np.outer(units,units)
    trial = np.sin(np.arange(nk+nx)+1)*np.sqrt(np.diag(fixed))
    stable = GaussianMetric(fixed).chi2(trial)
    stable_units = GaussianMetric(fixed*np.outer(units,units)).chi2(trial*units)
    scaled_cross = cxp.copy()
    for a in (0,2):
        for b in (0,2):
            rows = slice(0,30) if b==0 else slice(30,nx)
            cols = slice(0,16) if a==0 else slice(16,32)
            scaled_cross[rows,cols] *= SCALES[a,b]
    rho = canonical_correlations(cpp,cxx,cxp)
    rho_scaled = canonical_correlations(cpp,cxx,scaled_cross)
    pm = np.interp(np.log(km),np.log(k),p)
    tm = pm*(b1+f*mu*mu)**2/(1+.5*(km*mu*sigma)**2)**2+1/nbar
    cpp_discrete = np.zeros_like(cpp)
    for a in (0,2):
        for b in (0,2):
            weight = 2*(2*a+1)*(2*b+1)*tm*tm*eval_legendre(a,mu)*eval_legendre(b,mu)
            for i,select in enumerate(keep_bins):
                cpp_discrete[(a//2)*16+i,(b//2)*16+i] = weight[select].sum()/counts[i]**2
    centers = (d['s_edges'][1:]+d['s_edges'][:-1])/2
    u,w = np.polynomial.legendre.leggauss(64)
    smu = p[:,None]*(b1+f*u[None,:]**2)**2/(1+.5*(k[:,None]*u[None,:]*sigma)**2)**2
    poles64 = {a:.5*(2*a+1)*np.sum(w[None,:]*smu*eval_legendre(a,u)[None,:],axis=1) for a in (0,2)}
    poles_a = continuous_poles(k,p,b1,f,sigma)
    xi64 = vector(poles64,d['kernels'],g,v,centers)
    xia = vector(poles_a,d['kernels'],g,v,centers)
    new_kernels = np.stack([shell_kernel(k,d['s_edges'],a) for a in (0,2)])
    xia_kernel = vector(poles_a,new_kernels,g,v,centers)
    # Identical uncompressed radial nodes isolate ANGULAR discreteness.
    ku, inverse, ng = np.unique(km,return_inverse=True,return_counts=True)
    pu = np.interp(np.log(ku),np.log(k),p)
    cont_low = continuous_poles(ku,pu,b1,f,sigma)
    signal_modes = tm-1/nbar
    dpoles = {a:(2*a+1)*np.bincount(inverse,weights=signal_modes*eval_legendre(a,mu))/ng-cont_low[a] for a in (0,2)}
    lowkernels = np.stack([shell_kernel(ku,d['s_edges'],a) for a in (0,2)])
    delta_angular = vector(dpoles,lowkernels,ng,v,centers)
    mx = GaussianMetric(cxx)
    report = {
        'scope':'cache/geometry audit only; not a data fit or production validation',
        'inputs':{'cache':str(path),'sha256':digest,'nbar':nbar,'b1':b1,'sigma_s':sigma,'fNL':0.,
                  'p_fixed':1.,'growth':f,'nmu_legacy':64,'measurement_count_check':measurement_check},
        'bin_counts':{'exact':counts.tolist(),'rebinned':rebinned_counts.tolist(),
                      'fractional_rebinned_error':(rebinned_counts/counts-1).tolist()},
        'legacy_cross_zero_floor':floor_meta,
        'legacy_xi_sigma_ratio_median':float(np.median(np.sqrt(np.diag(floored)[nk:]/np.diag(cxx)))),
        'legacy_unit_change_xi_relative_frobenius':float(np.linalg.norm(floored[nk:,nk:]-altered[nk:,nk:])/np.linalg.norm(floored[nk:,nk:])),
        'strict_metric_unit_relative_chi2_error':float(abs(stable_units/stable-1)),
        'cross_canonical_rho_max':float(rho[0]),
        'scaled_cross_canonical_rho_max':float(rho_scaled[0]),
        'scaled_cross_is_psd_necessary_condition':bool(rho_scaled[0]<=1+1e-10),
        'quadrant_scales':{f'P{a}_xi{b}':value for (a,b),value in SCALES.items()},
        'P02_exact_mode_over_legacy_covariance_diagonal':(np.diag(cpp_discrete)/np.diag(cpp)).tolist(),
        'continuous_exact_over_GL64_xi_covariance_diagonal':(np.diag(cxx_a)/np.diag(cxx)).tolist(),
        'delta_chi2_in_fixed_Cmean_metric':{
            'continuous_angular_quadrature_only':25*mx.chi2(xia-xi64),
            'radial_kernel_quadrature_only':25*mx.chi2(xia_kernel-xia),
            'low_k_angular_discreteness_only_kmax_0p095':25*mx.chi2(delta_angular)},
        'warnings':['Prediction changes are not fitted parameter shifts.',
                    'Low-k correction is not applied automatically; test k-switch convergence.',
                    'Connected/non-Poisson covariance terms are not included.',
                    'Empirical cross-quadrant scales are audited, not endorsed.']}
    arrays = dict(C_PP_legacy=cpp,C_XX_legacy=cxx,C_XP_analytic64=cxp,
        C_XP_scaled=scaled_cross,C_PP_exact_modes=cpp_discrete,C_XX_exact_angle=cxx_a,
        xi_GL64=xi64,xi_exact_angle=xia,xi_exact_kernel=xia_kernel,
        delta_xi_low_k_angular=delta_angular,k_edges=BINS,nmodes=counts)
    out.parent.mkdir(parents=True,exist_ok=True)
    with npz_out.open('xb') as stream:
        np.savez_compressed(stream,**arrays)
    with out.open('x') as stream:
        json.dump(report,stream,indent=2); stream.write('\n')
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('root',type=Path)
    parser.add_argument('--out',required=True,type=Path)
    parser.add_argument('--cache',type=Path)
    parser.add_argument('--measurement',type=Path)
    parser.add_argument('--nbar',type=float,default=.000162131295)
    parser.add_argument('--b1',type=float,default=2.55)
    parser.add_argument('--sigma',type=float,default=8.)
    args = parser.parse_args()
    try:
        report = run_audit(args.root,args.out,args.cache,args.measurement,args.nbar,args.b1,args.sigma)
    except (OSError,ValueError,KeyError) as exc:
        parser.exit(2,f'audit failed (no fallback/rebuild): {exc}\n')
    print(json.dumps({k:report[k] for k in ('scope','legacy_cross_zero_floor',
        'legacy_xi_sigma_ratio_median','cross_canonical_rho_max','scaled_cross_canonical_rho_max',
        'delta_chi2_in_fixed_Cmean_metric')},indent=2))


if __name__ == '__main__':
    main()
