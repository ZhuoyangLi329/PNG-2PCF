"""Deterministic/synthetic tests only: NOT a refit of original rawbox data."""
import hashlib
import json
from pathlib import Path
import tempfile
import unittest
import numpy as np
import scipy
from scipy.integrate import quad
from scipy.special import eval_legendre, spherical_jn
from rawbox_diagnostics import (GaussianMetric, assemble_covariance,
 canonical_correlations, conditional_xi, lorentzian_moments, continuous_poles,
 angular_total_integrals, shell_kernel, lattice_modes, mode_operators,
 mode_covariance, rebin_geometry, legacy_floor, audit_cache)

RESULTS = {'scope':'synthetic and geometry tests; original NPZ not used'}

class Tests(unittest.TestCase):
    def test_01_unit_invariance(self):
        rng=np.random.default_rng(43)
        a=rng.normal(size=(7,7)); c=a@a.T+np.eye(7); r=rng.normal(size=7)
        scale=np.array([1e6,1e-6,1e3,1e-3,1e2,1e-2,1.])
        original=GaussianMetric(c).chi2(r)
        changed=GaussianMetric(c*np.outer(scale,scale)).chi2(r*scale)
        self.assertAlmostEqual(original,changed,places=11)
        RESULTS['chi2_unit_relative_error']=float(abs(changed/original-1))

    def test_02_block_identity(self):
        p=np.array([[2e9,3e8],[3e8,4e9]])
        x=np.array([[3e-8,1e-8],[1e-8,2e-8]])
        rp,rx=np.array([1e4,-2e4]),np.array([1e-4,-2e-4])
        c=assemble_covariance(p,x)
        np.testing.assert_array_equal(c[:2,:2],p)
        np.testing.assert_array_equal(c[2:,2:],x)
        self.assertAlmostEqual(GaussianMetric(c).chi2(np.r_[rp,rx]),
             GaussianMetric(p).chi2(rp)+GaussianMetric(x).chi2(rx),places=12)

    def test_03_conditional_identity(self):
        p=np.array([[2e9,3e8],[3e8,4e9]])
        x=np.array([[3e-8,1e-8],[1e-8,2e-8]])
        cross=np.array([[.1,-.05],[.02,.08]])
        rp,rx=np.array([1e4,-2e4]),np.array([1e-4,-2e-4])
        c=assemble_covariance(p,x,cross)
        _,_,cond=conditional_xi(rp,rx,p,x,cross)
        self.assertLess(canonical_correlations(p,x,cross)[0],1)
        delta=abs(GaussianMetric(c).chi2(np.r_[rp,rx])-GaussianMetric(p).chi2(rp)-cond)
        self.assertLess(delta,1e-12)
        RESULTS['conditional_chi2_identity_abs_error']=delta

    def test_04_bad_covariance_rejected(self):
        for c in ([[1,2],[2,1]],[[1,1],[1,1]],[[-1,0],[0,1]],
                  [[1,np.nan],[np.nan,1]],[[1,.2],[.1,1]]):
            with self.assertRaises(ValueError): GaussianMetric(c)
        with self.assertRaises(ValueError): assemble_covariance(np.eye(2),np.eye(2),1.1*np.eye(2))

    def test_05_legacy_floor_unit_error(self):
        c=np.diag([4.269509520005424e9,2e8,1.0042022457664528e-9,8e-6])
        old,_=legacy_floor(c)
        unit=np.array([1e-6,1e-6,1.,1.])
        changed=legacy_floor(c*np.outer(unit,unit))[0]/np.outer(unit,unit)
        self.assertGreater(np.linalg.norm(old-changed)/np.linalg.norm(old[2:,2:]),.5)

    def test_06_moments(self):
        worst=0.
        for power in (2,4):
            for x in (0.,.01,.2,1.,8.,24.,90.):
                exact=lorentzian_moments(x,6,power)
                for n in range(7):
                    reference=quad(lambda u:u**(2*n)/(1+.5*(x*u)**2)**power,
                                   0,1,epsabs=1e-27,epsrel=2e-12,limit=300)[0]
                    worst=max(worst,abs(exact[n]/reference-1))
                    np.testing.assert_allclose(exact[n],reference,rtol=1e-9,atol=1e-25)
        RESULTS['moments_max_relative_error_vs_quad']=float(worst)

    def test_07_kaiser_and_GL64_tail(self):
        b,f=2.55,.81
        poles=continuous_poles(np.array([.003,.1]),1.,b,f,0.)
        np.testing.assert_allclose(poles[0],b*b+2*b*f/3+f*f/5,rtol=1e-13)
        np.testing.assert_allclose(poles[2],4*b*f/3+4*f*f/7,rtol=1e-13)
        errors={}
        for x in (24.,90.):
            for nmu in (64,128,256):
                u,w=np.polynomial.legendre.leggauss(nmu)
                value=.5*np.sum(w/(1+.5*(x*u)**2)**2)
                errors[f'ksigma={x:g},nmu={nmu}']=float(value/lorentzian_moments(x,0)[0]-1)
        RESULTS['I0_GL_relative_errors']=errors

    def test_08_angular_covariance_normalization(self):
        out=angular_total_integrals(np.array([.01]),0.,2.,.8,8.,1.)
        self.assertAlmostEqual(out[0,0][0],2.)
        self.assertAlmostEqual(out[2,2][0],2/5)
        self.assertAlmostEqual(out[0,2][0],0.)
        k,p,b,f,s,shot=.19,3500.,2.6,.81,8.,6100.
        out=angular_total_integrals(np.array([k]),p,b,f,s,shot)
        for ell,ell2 in out:
            def fn(u):
                t=p*(b+f*u*u)**2/(1+.5*(k*s*u)**2)**2+shot
                return t*t*eval_legendre(ell,u)*eval_legendre(ell2,u)
            reference=quad(fn,-1,1,epsabs=1e-5,epsrel=1e-11)[0]
            np.testing.assert_allclose(out[ell,ell2],reference,rtol=1e-10,atol=1e-5)

    def test_09_shell_kernels(self):
        edges=np.array([30.,40.,50.,110.,120.,340.,350.])
        ks=np.array([0.,1e-5,.003,.03,.3,3.]); worst=0.
        for ell in (0,2):
            actual=shell_kernel(ks,edges,ell)
            reference=np.empty_like(actual)
            for i,k in enumerate(ks):
                for j,(lo,hi) in enumerate(zip(edges[:-1],edges[1:])):
                    reference[i,j]=(-1)**(ell//2)*3/(hi**3-lo**3)*quad(
                        lambda r:r*r*spherical_jn(ell,k*r),lo,hi,
                        epsabs=1e-8,epsrel=2e-11,limit=500)[0]
            worst=max(worst,float(np.max(abs(actual-reference))))
            np.testing.assert_allclose(actual,reference,rtol=3e-9,atol=3e-13)
        RESULTS['shell_max_abs_error_vs_quad']=worst

    def test_10_first_bin_angles(self):
        _,k,mu=lattice_modes(2000.,.005)
        use=(k>=.003)&(k<.005)
        self.assertEqual(use.sum(),18)
        m=[float(np.mean(mu[use]**(2*n))) for n in (1,2,3)]
        np.testing.assert_allclose(m,[1/3,2/9,1/6],atol=1e-14)
        b,f=2.55,.81
        actual=float(np.mean(5*(b+f*mu[use]**2)**2*eval_legendre(2,mu[use])))
        self.assertAlmostEqual(actual,5*b*f/3+25*f*f/36,places=12)
        RESULTS['first_bin']={'nmodes':18,'mu2_mu4_mu6':m,
          'constant_Plin_P2_discrete':actual,'constant_Plin_P2_continuous':4*b*f/3+4*f*f/7}

    def test_11_mode_covariance_MC(self):
        v,k,mu=lattice_modes(600.,.028)
        wp,wx,_=mode_operators(k,mu,[[.01,.017],[.017,.028]],[20.,40.,65.],600.**3)
        w=np.vstack([wp,wx]); t=2000*(2+.8*mu*mu)**2/(1+.5*(k*mu*8)**2)**2+6000
        theory=mode_covariance(w,t)
        half=(v[:,2]>0)|((v[:,2]==0)&(v[:,1]>0))|((v[:,2]==0)&(v[:,1]==0)&(v[:,0]>0))
        rng=np.random.default_rng(20260908); nmc=50000
        powers=rng.exponential(size=(nmc,int(half.sum())))*t[half]
        empirical=np.cov(2*powers@w[:,half].T,rowvar=False)
        se=np.sqrt((np.outer(np.diag(theory),np.diag(theory))+theory*theory)/(nmc-1))
        worst=float(np.max(abs(empirical-theory)/se))
        self.assertLess(worst,6.)
        RESULTS['mode_covariance_MC']={'draws':nmc,'complex_modes':int(half.sum()),'max_approx_SE_discrepancy':worst}

    def test_12_rebin_membership(self):
        result=rebin_geometry()
        self.assertEqual(result['exact_counts'][8],1262)
        self.assertEqual(result['grouped_counts'][8],1094)
        RESULTS['source_equivalent_rebin_geometry']=result

    def test_13_boundary_and_guards(self):
        h,k=1e-4,.1
        fn=lambda s:1/(1+.5*(k*s)**2)**2
        self.assertEqual((fn(h)-fn(-h))/(2*h),0.)
        self.assertAlmostEqual((1/(1+.5*k*k*h)**2-1)/h,-k*k,places=7)
        _,k,_=lattice_modes(2000.,.01)
        self.assertTrue(np.all(k>0))
        with self.assertRaises(ValueError): lattice_modes(2000.,3.)
        with self.assertRaises(ValueError): mode_operators(k,k*0,[[1.,2.]],[30.,40.],2000.**3)

    def test_14_audit_synthetic_fixture(self):
        with tempfile.TemporaryDirectory() as folder:
            path=Path(folder)/'synthetic.npz'; out=Path(folder)/'audit.json'
            kf=2*np.pi/2000.; k=np.geomspace(kf,3.,1800); edges=np.arange(30.,360.,10.)
            np.savez(path,k_eff=k,g_nz=4*np.pi*k*k*np.gradient(k)/kf**3,
                pk_dd=10000/(1+(k/.03)**2),s_edges=edges,ells=[0,2],
                kernels=np.stack([shell_kernel(k,edges,l) for l in (0,2)]),
                volume=2000.**3,boxsize=2000.,f_growth=.81,zeff=.725,kmax=3.)
            path.with_suffix('.json').write_text(json.dumps({'status':'pass',
                'output_sha256':hashlib.sha256(path.read_bytes()).hexdigest(),
                'cosmology':'abacus_c000','synthetic_fixture':True}))
            report=audit_cache(path,out)
            self.assertTrue(out.with_suffix('.npz').exists())
            self.assertEqual(report['legacy_floor']['n_floored'],57)
            with self.assertRaises(ValueError): audit_cache(path,out)
        RESULTS['audit_driver_test']='synthetic-cache flow test passed; NOT an original-cache audit'

if __name__=='__main__':
    result=unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.loadTestsFromTestCase(Tests))
    RESULTS.update(tests_run=result.testsRun,failures=len(result.failures),errors=len(result.errors),
                   passed=result.wasSuccessful(),numpy_version=np.__version__,scipy_version=scipy.__version__)
    Path(__file__).with_name('local_test_results.json').write_text(json.dumps(RESULTS,indent=2)+'\n')
    raise SystemExit(0 if result.wasSuccessful() else 1)
