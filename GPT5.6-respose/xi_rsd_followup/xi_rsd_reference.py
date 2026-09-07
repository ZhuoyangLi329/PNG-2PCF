"""Xi-only RSD reference tests; never imports/modifies the production P(k) pipeline.

This is NOT a calibrated rawbox predictor. All distances/LOS displacements are
in the SAME length unit. Pair variances are CENTRAL variances in length**2.
The generic GSM is an infinite-LOS numerical integral; periodic images and
input-moment calibration must be supplied/validated before production use.
"""
from __future__ import annotations
from dataclasses import dataclass
from typing import Callable
import math
import numpy as np
from scipy.integrate import quad
from scipy.special import eval_legendre

MomentFunction = Callable[[float, float], tuple[float, float, float]]


def _array(value, name: str) -> np.ndarray:
    out = np.asarray(value, dtype=float)
    if not np.all(np.isfinite(out)):
        raise ValueError(f"{name} must be finite")
    return out


def linear_spectrum_coefficients(plin, bias, growth: float):
    """Internal xi construction/verification only, not a replacement P(k) fit.

    Scale-dependent bias is INSIDE each k-dependent coefficient. theta is
    normalized so that theta=delta_m at linear order (no velocity bias).
    """
    p, b = np.broadcast_arrays(_array(plin, "plin"), _array(bias, "bias"))
    f = float(growth)
    if not np.isfinite(f) or np.any(p < 0):
        raise ValueError("nonfinite growth or negative spectrum")
    return {0: (b*b+2*b*f/3+f*f/5)*p,
            2: (4*b*f/3+4*f*f/7)*p,
            4: (8*f*f/35)*p}


def lorentzian_los_pdf(w, sigma_s: float):
    """Inverse FT of [1+(kz*sigma_s)**2/2]**-2.

    This extra, separation-independent smoothing kernel has variance
    2*sigma_s**2. It is NOT the full pair-weighted streaming velocity PDF.
    """
    if not np.isfinite(sigma_s) or sigma_s <= 0:
        raise ValueError("sigma_s must be positive; zero is a Dirac distribution")
    a = sigma_s / math.sqrt(2)
    z = np.abs(_array(w, "w")) / a
    return (1+z)*np.exp(-z)/(4*a)


@dataclass
class LinearModeMoments:
    """Exact leading-order real-space moments on a finite +/- mode set.

    Density contrast is B(k)*delta_m; u_i(k)=i*f*k_i/k**2*delta_m.
    Complete +/- pairs, even P and even B are required. No shot noise is used
    in velocity moments. mean_los is O(P), NOT an all-orders calibrated mean;
    variance_los is O(P). Inserting them in a GSM defines a diagnostic partial
    resummation, not a full nonlinear or primordial-non-Gaussian GSM.
    """
    kvec: np.ndarray
    plin: np.ndarray
    bias: np.ndarray
    growth: float
    volume: float

    def __post_init__(self):
        self.kvec = _array(self.kvec, "kvec")
        if self.kvec.ndim != 2 or self.kvec.shape[1] != 3:
            raise ValueError("kvec must have shape (N,3)")
        n = len(self.kvec)
        self.plin = np.broadcast_to(_array(self.plin, "plin"), (n,)).copy()
        self.bias = np.broadcast_to(_array(self.bias, "bias"), (n,)).copy()
        self.k2 = np.sum(self.kvec**2, axis=1)
        if n == 0 or np.any(self.k2 <= 0) or np.any(self.plin < 0):
            raise ValueError("nonzero modes and nonnegative power required")
        if not np.isfinite(self.volume) or self.volume <= 0 or not np.isfinite(self.growth):
            raise ValueError("invalid volume/growth")
        lookup = {tuple(row): i for i, row in enumerate(self.kvec)}
        if len(lookup) != n:
            raise ValueError("duplicate modes")
        for i, row in enumerate(self.kvec):
            j = lookup.get(tuple(-row))
            if j is None or not np.isclose(self.plin[i], self.plin[j], rtol=1e-13, atol=0) or not np.isclose(self.bias[i], self.bias[j], rtol=1e-13, atol=0):
                raise ValueError("complete +/- pairs with even power/bias required")

    def at(self, separation) -> tuple[float, float, float]:
        """Return (xi_real, mean relative u_z, central variance at O(P))."""
        r = _array(separation, "separation")
        if r.shape != (3,):
            raise ValueError("separation must have 3 components")
        phase = self.kvec @ r
        c, s = np.cos(phase), np.sin(phase)
        p, b, f, v = self.plin, self.bias, self.growth, self.volume
        kz = self.kvec[:, 2]
        xi = np.sum(b*b*p*c)/v
        mean = -2*f*np.sum(b*p*kz/self.k2*s)/v
        # 1-cos(x) evaluated as 2*sin(x/2)**2 avoids loss near coincident points.
        variance = 4*f*f*np.sum(p*kz*kz/self.k2**2*np.sin(phase/2)**2)/v
        return float(xi), float(mean), float(variance)

    def linear_parts(self, separation) -> dict[str, float]:
        """Independent analytic terms in xi_r - d_z m1 + .5*d_z^2 m2."""
        r = _array(separation, "separation")
        if r.shape != (3,):
            raise ValueError("separation must have 3 components")
        c = np.cos(self.kvec @ r)
        mu2 = self.kvec[:, 2]**2/self.k2
        p, b, f, v = self.plin, self.bias, self.growth, self.volume
        return {"real": float(np.sum(p*b*b*c)/v),
                "infall": float(np.sum(p*2*b*f*mu2*c)/v),
                "dispersion": float(np.sum(p*f*f*mu2*mu2*c)/v),
                "direct_kaiser": float(np.sum(p*(b+f*mu2)**2*c)/v)}

    def as_los_function(self) -> MomentFunction:
        """Use the x-z slice. General finite-box moments can depend on azimuth."""
        return lambda transverse, y: self.at([transverse, 0., y])


def radial_moment_adapter(xi: Callable, infall: Callable,
                          variance_parallel: Callable,
                          variance_transverse_one_component: Callable) -> MomentFunction:
    """Convert isotropic radial moments into a LOS callback.

    mu is the REAL-space y/r, not observed s_parallel/s. Parallel/transverse
    refer to the PAIR SEPARATION, not a fixed LOS. Transverse is ONE component.
    Input infall is signed: negative for approaching pairs.
    """
    def moments(transverse: float, y: float):
        r = math.hypot(transverse, y)
        if r == 0:
            raise ValueError("coincident pairs require a separately defined limit")
        mu = y/r
        vr, vt = float(variance_parallel(r)), float(variance_transverse_one_component(r))
        if vr < 0 or vt < 0:
            raise ValueError("central variances cannot be negative")
        return float(xi(r)), mu*float(infall(r)), mu*mu*vr+(1-mu*mu)*vt
    return moments


def gsm_point(transverse: float, parallel: float, moments: MomentFunction,
              *, integration_scale: float, zmax: float = 12.,
              extra_pair_variance: float = 0., epsabs: float = 2e-10,
              zero_velocity: bool = False) -> tuple[float, float]:
    """Reference pair-conserving Gaussian streaming integral at one point.

    Integral is over REAL LOS y. The callback returns xi_r, signed mean u12_z,
    and central pair variance, all at (transverse,y). Kaiser/FoG must NOT have
    been applied to xi_r. No division by integral(kernel) is performed.

    integration_scale is a quadrature coordinate scale, NOT a fitted parameter.
    Increase both zmax and numerical accuracy to demonstrate tail convergence.
    Returned error is quadrature error only, NOT tail or physical model error.
    zero_velocity is an explicit global assertion of a zero-displacement model.
    """
    args = [transverse, parallel, integration_scale, zmax, extra_pair_variance, epsabs]
    if not np.all(np.isfinite(args)) or transverse < 0 or integration_scale <= 0 or zmax <= 0 or extra_pair_variance < 0 or epsabs <= 0:
        raise ValueError("invalid separation, integration settings or variance")
    if zero_velocity:
        xi, mean, variance = moments(transverse, parallel)
        if not np.all(np.isfinite([xi, mean, variance])) or xi < -1:
            raise ValueError("invalid zero-velocity moments")
        if mean != 0 or variance != 0 or extra_pair_variance != 0:
            raise ValueError("zero_velocity contradicts supplied moments")
        return float(xi), 0.

    def integrand(t):
        y = parallel + integration_scale*t
        xi, mean, variance = moments(transverse, y)
        if not np.all(np.isfinite([xi, mean, variance])) or xi < -1 or variance < 0:
            raise ValueError("invalid real-space pair moments")
        variance += extra_pair_variance
        if variance <= 0:
            raise ValueError("zero local variance needs a Dirac-limit treatment")
        residual = parallel-y-mean
        density = integration_scale/math.sqrt(2*math.pi*variance)*math.exp(-residual*residual/(2*variance))
        # Subtract an analytically normalized reference, not the physical PDF.
        # This retains the background contribution from spatial variance gradients.
        reference = math.exp(-t*t/2)/math.sqrt(2*math.pi)
        return (1+xi)*density-reference

    breaks = [t for t in (-8., -4., -2., 0., 2., 4., 8.) if -zmax < t < zmax]
    value, error = quad(integrand, -zmax, zmax, points=breaks,
                        epsabs=epsabs, epsrel=2e-8, limit=250)
    return float(value), float(error)


def shell_multipoles(evaluate_xi: Callable[[float, float], float], edges,
                     ells=(0, 2), *, nradial=12, nmu=32):
    """Project the ALREADY-MAPPED xi(s_perp,s_parallel), then shell average.

    This uses continuous mu_s. Production FCFC finite-mu bins need their exact
    published projection weights. Do not stream already shell-averaged xi.
    """
    e = _array(edges, "edges")
    if e.ndim != 1 or len(e) < 2 or np.any(np.diff(e) <= 0) or e[0] < 0:
        raise ValueError("invalid shell edges")
    if any(l not in (0, 2, 4) for l in ells):
        raise ValueError("only even ell=0,2,4 supported")
    u, wu = np.polynomial.legendre.leggauss(nmu)
    t, wt = np.polynomial.legendre.leggauss(nradial)
    out = {ell: [] for ell in ells}
    for lo, hi in zip(e[:-1], e[1:]):
        s = (hi+lo)/2+(hi-lo)*t/2
        values = np.array([[evaluate_xi(float(rr*np.sqrt(1-mu*mu)), float(rr*mu))
                            for mu in u] for rr in s])
        for ell in ells:
            angular = (2*ell+1)/2*(values @ (wu*eval_legendre(ell, u)))
            result = 3/(hi**3-lo**3)*(hi-lo)/2*np.sum(wt*s*s*angular)
            out[ell].append(float(result))
    return {ell: np.asarray(values) for ell, values in out.items()}
