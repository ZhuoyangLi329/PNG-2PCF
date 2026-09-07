"""Synthetic mathematical tests only. No original rawbox measurements or fits."""
import hashlib
import json
import math
from pathlib import Path
import unittest
import numpy as np
import scipy
from scipy.integrate import quad
from scipy.special import eval_legendre
from xi_rsd_reference import (LinearModeMoments, gsm_point,
    linear_spectrum_coefficients, lorentzian_los_pdf, radial_moment_adapter,
    shell_multipoles)

RESULTS = {"scope": "xi-only synthetic identities; no original catalog/cache or fit"}


def make_modes(amplitude=1., growth=.8):
    box = 400.
    n = np.stack(np.meshgrid(*([np.arange(-2, 3)]*3), indexing='ij'), axis=-1).reshape(-1, 3)
    n = n[np.any(n != 0, axis=1)]
    kvec = 2*np.pi/box*n
    k = np.linalg.norm(kvec, axis=1)
    p = amplitude*1e4*np.exp(-(k/.05)**2)
    b = 2.5+.3*((2*np.pi/box)/k)**2
    return LinearModeMoments(kvec, p, b, growth, box**3)


class RSDTests(unittest.TestCase):
    def test_continuous_kaiser_coefficients_with_png_shaped_bias(self):
        k = np.array([.003, .01, .08]); p = np.array([3000., 1e4, 5000.])
        b = 2.5+.02*(.01/k)**2; f = .81
        calculated = linear_spectrum_coefficients(p, b, f)
        mu, w = np.polynomial.legendre.leggauss(32)
        for ell in (0, 2, 4):
            ref = .5*(2*ell+1)*np.sum(w*p[:, None]*(b[:, None]+f*mu**2)**2*eval_legendre(ell, mu), axis=1)
            np.testing.assert_allclose(calculated[ell], ref, rtol=2e-12)
        # A constant beta applied to the full PNG galaxy spectrum is NOT equivalent.
        wrong = b*b*p*(1+2*(f/2.5)/3+(f/2.5)**2/5)
        RESULTS['wrong_constant_beta_monopole_fractional_error'] = (wrong/calculated[0]-1).tolist()
        self.assertGreater(np.max(np.abs(wrong/calculated[0]-1)), .01)

    def test_mode_streaming_derivative_identity(self):
        model = make_modes()
        worst = 0.
        for r in ([35., 0., 45.], [75., 10., 15.], [20., -25., 90.]):
            r = np.asarray(r); h = .01; dz = np.array([0., 0., h])
            x0, _, var0 = model.at(r)
            _, meanp, varp = model.at(r+dz)
            _, meanm, varm = model.at(r-dz)
            streaming = x0-(meanp-meanm)/(2*h)+.5*(varp-2*var0+varm)/h**2
            target = model.linear_parts(r)['direct_kaiser']
            worst = max(worst, abs(streaming-target))
            self.assertAlmostEqual(streaming, target, delta=1e-8)
        RESULTS['finite_mode_derivative_identity_max_abs_error'] = worst

    def test_gsm_has_correct_linear_limit(self):
        s = [35., 0., 45.]
        target = make_modes().linear_parts(s)['direct_kaiser']
        rows = []
        for eps in (.1, .05, .025, .0125):
            model = make_modes(eps)
            scale = math.sqrt(model.at(s)[2])
            value, err = gsm_point(s[0], s[2], model.as_los_function(), integration_scale=scale)
            rows.append(dict(amplitude=eps, xi_gsm=value, xi_linear=eps*target,
                             scaled_error=abs(value/eps-target), quadrature_error=err))
        for before, after in zip(rows, rows[1:]):
            self.assertLess(after['scaled_error'], .6*before['scaled_error'])
        RESULTS['gsm_linear_limit'] = rows

    def test_streaming_xi_only_loses_both_velocity_terms(self):
        eps = .001; s = [35., 0., 45.]
        model = make_modes(eps); parts = make_modes().linear_parts(s)
        scale = math.sqrt(model.at(s)[2])
        good, _ = gsm_point(s[0], s[2], model.as_los_function(), integration_scale=scale)
        def wrong(t):
            y = s[2]+scale*t
            xi, mean, variance = model.at([s[0], 0, y])
            return scale*xi/math.sqrt(2*np.pi*variance)*np.exp(-(s[2]-y-mean)**2/(2*variance))
        bad = quad(wrong, -12, 12, epsabs=1e-12)[0]
        # Streaming xi only loses BOTH infall and variance terms at O(P).
        self.assertAlmostEqual(bad/eps, parts['real'], delta=2e-6)
        self.assertAlmostEqual(good/eps, parts['direct_kaiser'], delta=2e-6)
        RESULTS['wrong_stream_xi_only'] = {'correct_over_amplitude': good/eps,
            'wrong_over_amplitude': bad/eps, 'expected_kaiser': parts['direct_kaiser'],
            'expected_real': parts['real']}

    def test_constant_dispersion_loses_f_squared(self):
        eps = .001; s = [35., 0., 45.]
        model = make_modes(eps); var0 = model.at(s)[2]
        def constant_variance(transverse, y):
            xi, mean, _ = model.at([transverse, 0, y])
            return xi, mean, var0
        value, _ = gsm_point(s[0], s[2], constant_variance, integration_scale=math.sqrt(var0))
        parts = make_modes().linear_parts(s)
        expected = parts['real']+parts['infall']
        self.assertAlmostEqual(value/eps, expected, delta=2e-6)
        self.assertGreater(abs(value/eps-parts['direct_kaiser']), 1e-5)
        RESULTS['wrong_constant_variance'] = {'result_over_amplitude': value/eps,
            'expected_without_f_squared': expected, 'missing_f_squared': parts['dispersion']}

    def test_fog_pdf_normalization_variance_and_fourier(self):
        sigma = 8.; a = sigma/np.sqrt(2)
        norm = 2*quad(lambda x: lorentzian_los_pdf(x, sigma), 0, 100*a, epsabs=1e-12)[0]
        variance = 2*quad(lambda x: x*x*lorentzian_los_pdf(x, sigma), 0, 100*a, epsabs=1e-9)[0]
        self.assertAlmostEqual(norm, 1., places=12)
        self.assertAlmostEqual(variance, 2*sigma*sigma, places=9)
        errors = []
        for kz in (0., .03, .1, .3):
            actual = 2*quad(lambda w: np.cos(kz*w)*lorentzian_los_pdf(w, sigma), 0, 100*a, epsabs=1e-12)[0]
            target = (1+.5*(kz*sigma)**2)**-2
            errors.append(abs(actual-target))
            self.assertAlmostEqual(actual, target, places=11)
        RESULTS['fog_kernel'] = {'normalization': norm, 'pair_variance': variance,
            'expected_pair_variance': 2*sigma*sigma, 'max_fourier_abs_error': max(errors)}

    def test_gsm_homogeneous_unclustered(self):
        value, _ = gsm_point(50., 20., lambda p,y: (0., 0., 36.), integration_scale=6.)
        self.assertAlmostEqual(value, 0., places=12)

    def test_gsm_parity_and_tail_convergence(self):
        model = make_modes(.5); s = [35., 0., 45.]
        scale = np.sqrt(model.at(s)[2]); callback = model.as_los_function()
        a, _ = gsm_point(35., 45., callback, integration_scale=scale, zmax=10.)
        b, _ = gsm_point(35., -45., callback, integration_scale=scale, zmax=14.)
        self.assertAlmostEqual(a, b, places=10)
        RESULTS['gsm_parity_and_tail_abs_difference'] = abs(a-b)

    def test_radial_adapter_uses_real_mu_and_one_transverse_component(self):
        fn = radial_moment_adapter(lambda r:.1, lambda r:-3., lambda r:25., lambda r:9.)
        xi, mean, var = fn(3., 4.)
        self.assertAlmostEqual(mean, -2.4)
        self.assertAlmostEqual(var, .64*25+.36*9)
        self.assertAlmostEqual(fn(3., -4.)[1], 2.4)

    def test_zero_velocity_and_shell_average(self):
        fn = lambda p,y: (.1*np.exp(-(p*p+y*y)/10000), 0., 0.)
        evaluator = lambda p,y: gsm_point(p,y,fn,integration_scale=1.,zero_velocity=True)[0]
        out = shell_multipoles(evaluator, [30., 40., 50.], ells=(0,2,4))
        for i, (lo, hi) in enumerate(((30.,40.),(40.,50.))):
            target = 3/(hi**3-lo**3)*quad(lambda r: .1*r*r*np.exp(-r*r/10000), lo, hi)[0]
            self.assertAlmostEqual(out[0][i], target, places=13)
        np.testing.assert_allclose(out[2], 0., atol=1e-14)
        np.testing.assert_allclose(out[4], 0., atol=1e-14)

    def test_invalid_moments_fail_instead_of_clipping(self):
        for fn in (lambda p,y:(0.,0.,-1.), lambda p,y:(-2.,0.,1.), lambda p,y:(0.,0.,0.)):
            with self.assertRaises(ValueError):
                gsm_point(30.,40.,fn,integration_scale=1.)
        with self.assertRaises(ValueError):
            LinearModeMoments(np.array([[.1,0,0]]), np.array([1.]), np.array([2.]), .8, 1000.)


if __name__ == '__main__':
    result = unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.loadTestsFromTestCase(RSDTests))
    root = Path(__file__).parent
    RESULTS.update(tests_run=result.testsRun, failures=len(result.failures), errors=len(result.errors),
        passed=result.wasSuccessful(), numpy_version=np.__version__, scipy_version=scipy.__version__,
        code_sha256={p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(root.glob('*.py'))})
    (root/'validation.json').write_text(json.dumps(RESULTS, indent=2)+'\n')
    raise SystemExit(0 if result.wasSuccessful() else 1)
