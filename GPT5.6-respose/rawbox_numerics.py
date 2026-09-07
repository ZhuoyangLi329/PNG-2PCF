"""Read-only numerical building blocks for the PNG-2PCF rawbox review.

All lattice sums contain BOTH members of every nonzero +/- Fourier pair.
Mode projectors describe a band-limited field, not automatically FCFC's
unfiltered point-pair estimator. Gaussian covariance omits connected terms.
"""
from __future__ import annotations
from dataclasses import dataclass
import math
import numpy as np
from scipy.linalg import solve_triangular
from scipy.special import eval_legendre, factorial2, hyp2f1, sici


def _finite(value, name):
    value = np.asarray(value, dtype=float)
    if not np.all(np.isfinite(value)):
        raise ValueError(f'{name} contains non-finite entries')
    return value


@dataclass
class GaussianMetric:
    """Fixed-covariance metric: normalize, validate, Cholesky; NEVER repair.

    Use the same residual() in least_squares and chi2() in MCMC. An unresolved
    eigenvalue is rejected, not floored or silently removed. True linear
    redundancy needs an explicitly defined supported-subspace likelihood.
    """
    covariance: np.ndarray
    min_correlation_eigenvalue: float = 1e-12

    def __post_init__(self):
        c = _finite(self.covariance, 'covariance').copy()
        if c.ndim != 2 or c.shape[0] != c.shape[1] or not c.size:
            raise ValueError('covariance must be a nonempty square matrix')
        if np.any(np.diag(c) <= 0):
            raise ValueError('non-positive marginal variance')
        self.scale = np.sqrt(np.diag(c))
        r = c / np.outer(self.scale, self.scale)
        if not np.allclose(r, r.T, rtol=0, atol=1e-12):
            raise ValueError('covariance is not symmetric in normalized units')
        r = (r + r.T) / 2
        self.eigenvalues = np.linalg.eigvalsh(r)
        if self.eigenvalues[0] <= self.min_correlation_eigenvalue:
            raise ValueError('indefinite/unresolved covariance; no automatic repair: '
                             f'lambda_min(R)={self.eigenvalues[0]:.6g}')
        self.cholesky = np.linalg.cholesky(r)
        self.covariance = (c + c.T) / 2

    def residual(self, residual):
        r = _finite(residual, 'residual')
        if r.shape != self.scale.shape:
            raise ValueError('residual shape differs from covariance dimension')
        return solve_triangular(self.cholesky, r / self.scale, lower=True)

    def chi2(self, residual):
        w = self.residual(residual)
        return float(w @ w)

    def solve(self, value):
        b = _finite(value, 'right hand side')
        if b.ndim not in (1, 2) or b.shape[0] != self.scale.size:
            raise ValueError('right hand side has the wrong shape')
        scale = self.scale if b.ndim == 1 else self.scale[:, None]
        z = solve_triangular(self.cholesky, b / scale, lower=True)
        return solve_triangular(self.cholesky.T, z, lower=False) / scale


def assemble_covariance(cpp, cxx, cxp=None):
    """Preserve marginal blocks; cross has xi rows and P columns."""
    p, x = GaussianMetric(cpp), GaussianMetric(cxx)
    cross = (np.zeros((x.scale.size, p.scale.size)) if cxp is None
             else _finite(cxp, 'cross covariance'))
    if cross.shape != (x.scale.size, p.scale.size):
        raise ValueError('cxp must have xi rows and P columns')
    c = np.block([[p.covariance, cross.T], [cross, x.covariance]])
    GaussianMetric(c)
    return c


def canonical_correlations(cpp, cxx, cxp):
    """Singular values of Lx^-1 Cxp Lp^-T in standardized coordinates."""
    p, x = GaussianMetric(cpp), GaussianMetric(cxx)
    cross = _finite(cxp, 'cross covariance')
    if cross.shape != (x.scale.size, p.scale.size):
        raise ValueError('wrong cross-block shape')
    a = cross / np.outer(x.scale, p.scale)
    a = solve_triangular(x.cholesky, a, lower=True)
    a = solve_triangular(p.cholesky, a.T, lower=True).T
    return np.linalg.svd(a, compute_uv=False)


def conditional_xi_residual(rp, rx, cpp, cxx, cxp):
    """Return conditional residual, fixed Schur covariance, and chi2."""
    assemble_covariance(cpp, cxx, cxp)
    p = GaussianMetric(cpp)
    a = np.asarray(cxp, dtype=float)
    residual = np.asarray(rx, dtype=float) - a @ p.solve(rp)
    schur = np.asarray(cxx, dtype=float) - a @ p.solve(a.T)
    metric = GaussianMetric(schur)
    return residual, metric.covariance, metric.chi2(residual)


def lorentzian_moments(ksigma, max_power=6, exponent=2):
    """<mu^(2n) [1+(k sigma mu)^2/2]^-exponent>, n=0..max_power.

    Brackets mean HALF the integral on [-1,1]. exponent=2 for the signal,
    exponent=4 for its square. Formula is analytical; evaluate via hyp2f1.
    """
    x = _finite(ksigma, 'k sigma')
    if max_power < 0 or exponent < 0:
        raise ValueError('powers must be non-negative')
    return np.stack([hyp2f1(exponent, n+.5, n+1.5, -.5*x*x)/(2*n+1)
                     for n in range(max_power+1)])


def continuous_poles(k, plin, amplitude, growth, sigma):
    """Continuous-angle ell=0,2 PNG/Kaiser/Lorentzian signal, no shot."""
    k, p, a = np.broadcast_arrays(_finite(k, 'k'), _finite(plin, 'Plin'),
                                  _finite(amplitude, 'amplitude'))
    if np.any(k < 0) or sigma < 0:
        raise ValueError('negative k or sigma')
    m = lorentzian_moments(k*sigma, 3, 2)
    z = a*a*m[0] + 2*a*growth*m[1] + growth*growth*m[2]
    zmu2 = a*a*m[1] + 2*a*growth*m[2] + growth*growth*m[3]
    return {0: p*z, 2: 2.5*p*(3*zmu2-z)}


def angular_total_integrals(k, plin, amplitude, growth, sigma, shot):
    """INTEGRAL_-1^1 [signal+shot]^2 L_a L_b dmu for a,b=0,2.

    This is not an angular average: do not multiply the radial covariance
    formula by an extra Gaussian factor of two. shot is externally fixed;
    a low-k stochastic constant is not automatically valid at all k.
    """
    k, p, a = np.broadcast_arrays(_finite(k, 'k'), _finite(plin, 'Plin'),
                                  _finite(amplitude, 'amplitude'))
    if sigma < 0 or np.any(k < 0):
        raise ValueError('negative scale')
    m2, m4 = lorentzian_moments(k*sigma, 6, 2), lorentzian_moments(k*sigma, 6, 4)
    pol = {0: np.array([1.]), 2: np.array([-.5, 1.5])}
    out = {}
    for ell in (0, 2):
        for ell2 in (0, 2):
            lp = np.polynomial.polynomial.polymul(pol[ell], pol[ell2])
            result = np.zeros_like(k)
            for j, coefficient in enumerate(lp):
                term = p*p*sum(math.comb(4,n)*a**(4-n)*growth**n*m4[j+n] for n in range(5))
                term += 2*shot*p*sum(math.comb(2,n)*a**(2-n)*growth**n*m2[j+n] for n in range(3))
                term += shot*shot/(2*j+1)
                result += 2*coefficient*term
            out[ell, ell2] = result
    return out


def shell_kernel(k, edges, ell):
    """i^ell times VOLUME-averaged j_ell(kr), ell=0,2, analytically.

    Uses exact antiderivatives (Si for ell=2) and a small-argument series.
    """
    k = _finite(k, 'k').reshape(-1)
    edges = _finite(edges, 'separation edges')
    if ell not in (0, 2) or np.any(k < 0):
        raise ValueError('requires non-negative k and ell=0 or 2')
    if edges.ndim != 1 or edges.size < 2 or edges[0] < 0 or np.any(np.diff(edges) <= 0):
        raise ValueError('invalid separation edges')
    lo, hi = edges[:-1][None, :], edges[1:][None, :]
    kk = k[:, None]
    xl, xh = kk*lo, kk*hi
    if ell == 0:
        primitive = lambda x: np.sin(x)-x*np.cos(x)
    else:
        primitive = lambda x: 3*sici(x)[0]+x*np.cos(x)-4*np.sin(x)
    denominator = kk**3*(hi**3-lo**3)/3
    with np.errstate(divide='ignore', invalid='ignore'):
        answer = (primitive(xh)-primitive(xl))/denominator
    small = xh < .5
    if np.any(small):
        series = np.zeros_like(answer)
        for m in range(7):
            power = ell+2*m
            coefficient = (-1.)**m/(2**m*math.factorial(m)*factorial2(2*ell+2*m+1))
            r_moment = 3*(hi**(power+3)-lo**(power+3))/((power+3)*(hi**3-lo**3))
            series += coefficient*kk**power*r_moment
        answer[small] = series[small]
    return (-1)**(ell//2)*answer


def lattice_modes(boxsize, kmax, max_cube_cells=3_000_000):
    """Enumerate full +/- lattice for LOW-k tests; explicitly exclude k=0."""
    if boxsize <= 0 or kmax <= 0:
        raise ValueError('boxsize and kmax must be positive')
    kf = 2*np.pi/boxsize
    n = int(np.ceil(kmax/kf))
    if (2*n+1)**3 > max_cube_cells:
        raise ValueError('large 3-D enumeration rejected; use converged low-k corrections')
    v = np.arange(-n, n+1, dtype=np.int32)
    modes = np.stack(np.meshgrid(v, v, v, indexing='ij'), axis=-1).reshape(-1,3)
    q = np.sum(modes.astype(np.int64)**2, axis=1)
    keep = (q > 0) & (kf*np.sqrt(q) <= kmax)
    modes, q = modes[keep], q[keep]
    return modes, kf*np.sqrt(q), modes[:,2]/np.sqrt(q)


def projectors(k, mu, k_bins, s_edges, volume, ells=(0,2)):
    """P and xi operators from the SAME full +/- mode-power vector.

    xi is the angular shell projection of the band-limited field. Assignment,
    aliasing, finite-N pair normalization and FCFC mu bins are separate tests.
    """
    k, mu, bins = _finite(k,'k'), _finite(mu,'mu'), _finite(k_bins,'k bins')
    if k.ndim != 1 or mu.shape != k.shape or volume <= 0 or np.any(k <= 0) or np.any(np.abs(mu) > 1):
        raise ValueError('invalid modes or volume')
    if bins.ndim != 2 or bins.shape[1] != 2 or np.any(bins[:,1] <= bins[:,0]):
        raise ValueError('k_bins must have shape (n,2), lo < hi')
    if any(ell not in (0,2) for ell in ells):
        raise ValueError('only ell=0,2 supported')
    selection = (k[None,:] >= bins[:,0,None]) & (k[None,:] < bins[:,1,None])
    counts = selection.sum(axis=1)
    if np.any(counts == 0):
        raise ValueError('empty Fourier bin')
    wp, wx = [], []
    for ell in ells:
        weight = (2*ell+1)*eval_legendre(ell,mu)
        wp.append(selection*weight[None,:]/counts[:,None])
        wx.append(shell_kernel(k,s_edges,ell).T*weight[None,:]/volume)
    return np.vstack(wp), np.vstack(wx), counts


def gaussian_mode_covariance(operator, total_power):
    """2 W diag(T^2) W.T, for full +/- modes and even rows only."""
    w, t = _finite(operator,'operator'), _finite(total_power,'total power')
    if w.ndim != 2 or t.shape != (w.shape[1],) or np.any(t <= 0):
        raise ValueError('invalid operator or non-positive total power')
    a = np.sqrt(2)*w*t[None,:]
    return a@a.T
