"""Deterministic geometry and synthetic tests. NO original rawbox arrays."""
import json
from pathlib import Path
import tempfile
import unittest
import numpy as np
from scipy.integrate import quad
from scipy.special import eval_legendre, spherical_jn
from rawbox_numerics import (GaussianMetric, assemble_covariance,
    canonical_correlations, conditional_xi_residual, lorentzian_moments,
    continuous_poles, angular_total_integrals, shell_kernel, lattice_modes,
    projectors, gaussian_mode_covariance)

RESULTS = {'scope': 'synthetic and exact geometry tests, not a rawbox-data refit'}


class NumericsTests(unittest.TestCase):
    def test_unit_invariance(self):
        rng = np.random.default_rng(43)
        a = rng.normal(size=(7,7)); c = a@a.T + np.eye(7)
        r = rng.normal(size=7)
        scale = np.array([1e6,1e-6,1e3,1e-3,1e2,1e-2,1.])
        chi = GaussianMetric(c).chi2(r)
        changed = GaussianMetric(c*np.outer(scale,scale)).chi2(r*scale)
        self.assertAlmostEqual(changed,chi,places=11)
        RESULTS['unit_invariance_relative_chi2_error'] = abs(changed/chi-1)

    def test_block_identity_and_conditional(self):
        p = np.array([[2e9,3e8],[3e8,4e9]])
        x = np.array([[3e-8,1e-8],[1e-8,2e-8]])
        rp,rx = np.array([1e4,-2e4]),np.array([1e-4,-2e-4])
        c = assemble_covariance(p,x)
        np.testing.assert_array_equal(c[:2,:2],p)
        np.testing.assert_array_equal(c[2:,2:],x)
        expected = GaussianMetric(p).chi2(rp)+GaussianMetric(x).chi2(rx)
        self.assertAlmostEqual(GaussianMetric(c).chi2(np.r_[rp,rx]),expected,places=12)
        cross = np.array([[.1,-.05],[.02,.08]])
        c = assemble_covariance(p,x,cross)
        _,_,conditional = conditional_xi_residual(rp,rx,p,x,cross)
        joint = GaussianMetric(c).chi2(np.r_[rp,rx])
        self.assertAlmostEqual(joint,GaussianMetric(p).chi2(rp)+conditional,places=12)
        self.assertLess(canonical_correlations(p,x,cross)[0],1)
        RESULTS['conditional_chi2_identity_error'] = abs(joint-GaussianMetric(p).chi2(rp)-conditional)

    def test_bad_covariance_rejected(self):
        for c in ([[1.,2.],[2.,1.]],[[1.,1.],[1.,1.]], [[-1.,0.],[0.,1.]],
                  [[1.,np.nan],[np.nan,1.]], [[1.,.2],[.1,1.]]):
            with self.assertRaises(ValueError):
                GaussianMetric(c)
        with self.assertRaises(ValueError):
            assemble_covariance(np.eye(2),np.eye(2),1.1*np.eye(2))

    def test_legacy_floor_is_not_unit_invariant(self):
        from audit_rawbox import old_floor
        c = np.diag([4.269509520005424e9,2e8,1.0042022457664528e-9,8e-6])
        old = old_floor(c)[0]
        units = np.array([1e-6,1e-6,1.,1.])
        changed = old_floor(c*np.outer(units,units))[0]/np.outer(units,units)
        difference = np.linalg.norm(old[2:,2:]-changed[2:,2:])/np.linalg.norm(old[2:,2:])
        self.assertGreater(difference,.5)
        RESULTS['synthetic_legacy_floor_unit_dependence'] = difference

    def test_lorentzian_moments_against_adaptive_quadrature(self):
        worst = 0.
        for exponent in (2,4):
            for x in (0.,.01,.2,1.,8.,24.,90.):
                moments = lorentzian_moments(x,6,exponent)
                for n in range(7):
                    reference = quad(lambda u: u**(2*n)/(1+.5*(x*u)**2)**exponent,
                                     0,1,epsabs=1e-27,epsrel=2e-12,limit=300)[0]
                    error = abs(moments[n]-reference)/max(abs(reference),1e-300)
                    worst = max(worst,error)
                    np.testing.assert_allclose(moments[n],reference,rtol=1e-9,atol=1e-25)
        RESULTS['moment_max_relative_error_vs_adaptive_quad'] = worst

    def test_kaiser_limit_and_quadrature_tail(self):
        b,f = 2.55,.81
        p = continuous_poles(np.array([.003,.1]),1.,b,f,0.)
        np.testing.assert_allclose(p[0],b*b+2*b*f/3+f*f/5,rtol=1e-13)
        np.testing.assert_allclose(p[2],4*b*f/3+4*f*f/7,rtol=1e-13)
        errors = {}
        for x in (24.,90.):
            exact = float(lorentzian_moments(x,0,2)[0])
            for nmu in (64,128,256):
                u,w = np.polynomial.legendre.leggauss(nmu)
                approx = .5*np.sum(w/(1+.5*(x*u)**2)**2)
                errors[f'ksigma={x:g},nmu={nmu}'] = float(approx/exact-1)
        RESULTS['I0_quadrature_relative_errors'] = errors

    def test_covariance_angular_integral_normalization(self):
        pure = angular_total_integrals(np.array([.01]),0.,2.,.8,8.,1.)
        self.assertAlmostEqual(pure[0,0][0],2.,places=12)
        self.assertAlmostEqual(pure[2,2][0],2./5.,places=12)
        self.assertAlmostEqual(pure[0,2][0],0.,places=12)
        k,p,a,f,s,shot = .19,3500.,2.6,.81,8.,6100.
        result = angular_total_integrals(np.array([k]),p,a,f,s,shot)
        for ell,ell2 in result:
            def integrand(u):
                total = p*(a+f*u*u)**2/(1+.5*(k*s*u)**2)**2+shot
                return total*total*eval_legendre(ell,u)*eval_legendre(ell2,u)
            ref = quad(integrand,-1,1,epsabs=1e-5,epsrel=1e-11)[0]
            np.testing.assert_allclose(result[ell,ell2],ref,rtol=1e-10,atol=1e-5)

    def test_shell_antiderivatives(self):
        edges = np.array([30.,40.,50.,110.,120.,340.,350.])
        ks = np.array([0.,1e-5,.003,.03,.3,3.])
        largest = 0.
        for ell in (0,2):
            calculated = shell_kernel(ks,edges,ell)
            reference = np.empty_like(calculated)
            for i,k in enumerate(ks):
                for j,(lo,hi) in enumerate(zip(edges[:-1],edges[1:])):
                    reference[i,j] = ((-1)**(ell//2)*3/(hi**3-lo**3) *
                        quad(lambda r:r*r*spherical_jn(ell,k*r),lo,hi,
                             epsabs=1e-8,epsrel=2e-11,limit=500)[0])
            largest = max(largest,float(np.max(np.abs(calculated-reference))))
            np.testing.assert_allclose(calculated,reference,rtol=3e-9,atol=3e-13)
        RESULTS['shell_kernel_max_absolute_error_vs_quad'] = largest

    def test_first_rawbox_bin_geometry(self):
        _,k,mu = lattice_modes(2000.,.005)
        select = (k>=.003)&(k<.005)
        self.assertEqual(int(select.sum()),18)
        moments = [float(np.mean(mu[select]**(2*n))) for n in (1,2,3)]
        np.testing.assert_allclose(moments,[1/3,2/9,1/6],atol=1e-14)
        b,f = 2.55,.81
        actual = float(np.mean(5*(b+f*mu[select]**2)**2*eval_legendre(2,mu[select])))
        expected = 5*b*f/3+25*f*f/36
        self.assertAlmostEqual(actual,expected,places=12)
        RESULTS['first_bin'] = {'nmodes':18,'mean_k':float(np.mean(k[select])),
            'mu2_mu4_mu6':moments,'P2_constant_Plin_toy':actual,
            'P2_continuum_toy':4*b*f/3+4*f*f/7,
            'note':'constant radial power toy, not the measured spectrum'}

    def test_mode_covariance_monte_carlo(self):
        vectors,k,mu = lattice_modes(600.,.028)
        wp,wx,_ = projectors(k,mu,[[.01,.017],[.017,.028]],[20.,40.,65.],600.**3)
        w = np.vstack([wp,wx])
        total = 2000*(2+.8*mu*mu)**2/(1+.5*(k*mu*8)**2)**2+6000
        theory = gaussian_mode_covariance(w,total)
        v = vectors
        half = (v[:,2]>0)|((v[:,2]==0)&(v[:,1]>0))|((v[:,2]==0)&(v[:,1]==0)&(v[:,0]>0))
        rng = np.random.default_rng(20260907)
        nmc = 50000
        powers = rng.exponential(size=(nmc,int(half.sum())))*total[half]
        mock = 2*powers@w[:,half].T
        empirical = np.cov(mock,rowvar=False)
        se = np.sqrt((np.outer(np.diag(theory),np.diag(theory))+theory*theory)/(nmc-1))
        zmax = float(np.max(np.abs(empirical-theory)/se))
        self.assertLess(zmax,6.)
        RESULTS['mode_covariance_mc'] = {'draws':nmc,'independent_complex_modes':int(half.sum()),
            'max_element_discrepancy_in_gaussian_covariance_standard_errors':zmax,
            'note':'SE is approximate diagnostic; theory uses exponential mode powers'}

    def test_zero_mode_and_large_enumeration_rejected(self):
        v,k,_ = lattice_modes(2000.,.01)
        self.assertTrue(np.all(k>0))
        self.assertFalse(np.any(np.all(v==0,axis=1)))
        with self.assertRaises(ValueError):
            lattice_modes(2000.,3.)
        with self.assertRaises(ValueError):
            projectors(k,k*0,[[1.,2.]],[30.,40.],2000.**3)

    def test_audit_driver_synthetic_fixture(self):
        from audit_rawbox import run_audit,checksum,BINS
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory); path = root/'synthetic_cache.npz'
            kf = 2*np.pi/2000.; k = np.geomspace(kf,3.,1800)
            edges = np.arange(30.,360.,10.)
            g = 4*np.pi*k*k*np.gradient(k)/kf**3
            np.savez(path,k_eff=k,g_nz=g,pk_dd=10000/(1+(k/.03)**2),
                alpha=1/(1+k*k),kernels=np.stack([shell_kernel(k,edges,l) for l in (0,2)]),
                s_edges=edges,ells=np.array([0,2]),volume=2000.**3,boxsize=2000.,f_growth=.81)
            path.with_suffix('.json').write_text(json.dumps({'status':'pass',
                'output_sha256':checksum(path),'cosmology':'abacus_c000','synthetic_fixture':True}))
            _,km,_ = lattice_modes(2000.,float(BINS.max()))
            counts = np.array([np.sum((km>=lo)&(km<hi)) for lo,hi in BINS])
            measurement = root/'synthetic_P02.npz'
            np.savez(measurement,k_edges=BINS,nmodes=counts)
            out = root/'audit.json'
            report = run_audit(root,out,path,measurement)
            self.assertEqual(report['inputs']['measurement_count_check'],'pass')
            self.assertLess(report['strict_metric_unit_relative_chi2_error'],1e-10)
            self.assertTrue(out.with_suffix('.npz').is_file())
            with self.assertRaises(FileExistsError):
                run_audit(root,out,path)
            RESULTS['audit_driver_synthetic_fixture'] = 'pass; no original cache was used'

    def test_cached_rebin_does_not_preserve_pk_bin_membership(self):
        from audit_rawbox import BINS
        _,km,_ = lattice_modes(2000.,.10)
        ku,multiplicity = np.unique(km,return_counts=True)
        index = (ku/(.1*(2*np.pi/2000.))).astype(np.int64)
        g = np.bincount(index,weights=multiplicity)
        gk = np.bincount(index,weights=multiplicity*ku)
        nonzero = g>0; effective_k = gk[nonzero]/g[nonzero]; g = g[nonzero]
        actual = np.array([np.sum((km>=lo)&(km<hi)) for lo,hi in BINS])
        rebinned = np.array([np.sum(g[(effective_k>=lo)&(effective_k<hi)]) for lo,hi in BINS])
        self.assertEqual(actual[8],1262); self.assertEqual(rebinned[8],1094)
        RESULTS['radial_rebin_membership_geometry'] = {'dk_factor':.1,
            'exact_counts':actual.tolist(),'rebinned_counts':rebinned.tolist(),
            'relative_count_errors':(rebinned/actual-1).tolist(),
            'scope':'source-equivalent geometry reproduction; not an original NPZ read'}

    def test_sigma_boundary_derivative(self):
        h,k = 1e-4,.1
        model = lambda s: 1/(1+.5*(k*s)**2)**2
        self.assertEqual((model(h)-model(-h))/(2*h),0.)
        lambda_derivative = (1/(1+.5*k*k*h)**2-1)/h
        self.assertAlmostEqual(lambda_derivative,-k*k,places=7)


if __name__ == '__main__':
    result = unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.loadTestsFromTestCase(NumericsTests))
    RESULTS.update(tests_run=result.testsRun,failures=len(result.failures),errors=len(result.errors),
                   passed=result.wasSuccessful(),numpy_version=np.__version__)
    Path(__file__).with_name('local_test_results.json').write_text(json.dumps(RESULTS,indent=2)+'\n')
    raise SystemExit(0 if result.wasSuccessful() else 1)
