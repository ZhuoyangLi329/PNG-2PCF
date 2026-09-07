# REVIEW EXCERPT: general numerical helpers only; original source ranges are in provenance/curation.json.
# No scientific formulas in the retained definitions were edited. Survey driver and imports omitted.
from __future__ import annotations
from typing import Any, Callable
import numpy as np
import emcee
from task43_fit_rsd_rawbox_pk0_vs_xi0_smin50 import split_rhat

JOINT_PARAMS = ("fNL", "b1", "sigma_s", "sn0")


BOUNDS_LO = np.asarray([-500.0, 0.5, 0.0, -1.0], dtype="f8")


BOUNDS_HI = np.asarray([500.0, 5.0, 30.0, 1.0], dtype="f8")


INIT_SCALE = np.asarray([4.0, 0.015, 0.12, 0.025], dtype="f8")


OPTIMIZER_STARTS = (
    np.asarray([0.0, 2.55, 8.0, 0.0], dtype="f8"),
    np.asarray([-80.0, 2.4, 4.0, 0.2], dtype="f8"),
    np.asarray([80.0, 2.7, 12.0, -0.2], dtype="f8"),
    np.asarray([0.0, 2.5, 15.0, 0.5], dtype="f8"),
)


FISHER_STEPS = np.asarray([1.0, 0.01, 0.05, 0.01], dtype="f8")


def build_precision(covariance: np.ndarray) -> tuple[np.ndarray, dict[str, Any]]:
    """Correlation-normalized inversion with an absolute eigenvalue floor only."""
    cov = 0.5 * (np.asarray(covariance, dtype="f8") + np.asarray(covariance, dtype="f8").T)
    scale = np.sqrt(np.clip(np.diag(cov), 1.0e-300, None))
    scaled = cov / np.outer(scale, scale)
    evals, evecs = np.linalg.eigh(scaled)
    floor = 1.0e-14
    inv_evals = np.where(evals >= floor, 1.0 / np.maximum(evals, floor), 0.0)
    inv_scaled = (evecs * inv_evals[None, :]) @ evecs.T
    meta = {
        "scaled_eigenvalues": {
            "min": float(evals[0]),
            "max": float(evals[-1]),
            "n_below_floor": int(np.count_nonzero(evals < floor)),
        }
    }
    return inv_scaled / np.outer(scale, scale), meta


def summarize_chain(chain: np.ndarray, logp: np.ndarray, names: tuple[str, ...]) -> dict[str, Any]:
    flat = np.asarray(chain, dtype="f8").reshape(-1, len(names))
    flat_logp = np.asarray(logp, dtype="f8").reshape(-1)
    quantiles = np.percentile(flat, [16.0, 50.0, 84.0], axis=0)
    sigma68 = 0.5 * (quantiles[2] - quantiles[0])
    try:
        tau = np.asarray(emcee.autocorr.integrated_time(chain, quiet=True, tol=0), dtype="f8")
    except Exception:
        tau = np.full(chain.shape[-1], np.inf)
    rhat = split_rhat(chain)
    half = chain.shape[0] // 2
    first = chain[:half].reshape(-1, chain.shape[-1])
    second = chain[-half:].reshape(-1, chain.shape[-1])
    half_shift = np.abs(np.median(first, axis=0) - np.median(second, axis=0)) / sigma68
    imax = int(np.argmax(flat_logp))
    return {
        "posterior": {
            name: {
                "q16": float(quantiles[0, index]),
                "q50": float(quantiles[1, index]),
                "q84": float(quantiles[2, index]),
                "sigma68": float(sigma68[index]),
                "mean": float(np.mean(flat[:, index])),
                "std": float(np.std(flat[:, index], ddof=1)),
            }
            for index, name in enumerate(names)
        },
        "map_chain": {name: float(flat[imax, index]) for index, name in enumerate(names)},
        "map_chain_log_probability": float(flat_logp[imax]),
        "tau": {name: float(tau[index]) for index, name in enumerate(names)},
        "postburn_length_over_tau": {name: float(chain.shape[0] / tau[index]) for index, name in enumerate(names)},
        "split_rhat": {name: float(rhat[index]) for index, name in enumerate(names)},
        "half_chain_shift_sigma": {name: float(half_shift[index]) for index, name in enumerate(names)},
        "gates": {
            "split_rhat_max_below_1p01": bool(np.max(rhat) < 1.01),
            "postburn_length_min_above_50tau": bool(np.min(chain.shape[0] / tau) > 50.0),
            "half_chain_shift_max_below_0p1sigma": bool(np.max(half_shift) < 0.1),
        },
    }


def run_chain(
    evaluate: Callable[[np.ndarray], np.ndarray],
    data: np.ndarray,
    precision: np.ndarray,
    nominal_theta: np.ndarray,
    nparams: int,
    *,
    nwalkers: int,
    nsteps: int,
    burnin: int,
    seed: int,
) -> tuple[dict[str, Any], np.ndarray, np.ndarray]:
    names = JOINT_PARAMS[:nparams]

    def log_probability(theta: np.ndarray) -> float:
        values = np.asarray(theta, dtype="f8")
        if np.any(values < BOUNDS_LO[:nparams]) or np.any(values > BOUNDS_HI[:nparams]):
            return -np.inf
        diff = np.asarray(data) - evaluate(values)
        return -0.5 * float(diff @ precision @ diff)

    rng = np.random.default_rng(int(seed))
    initial = np.asarray(nominal_theta, dtype="f8")[None, :] + rng.normal(
        size=(int(nwalkers), nparams)
    ) * INIT_SCALE[:nparams][None, :]
    initial = np.clip(initial, BOUNDS_LO[:nparams] + 1.0e-7, BOUNDS_HI[:nparams] - 1.0e-7)
    sampler = emcee.EnsembleSampler(int(nwalkers), nparams, log_probability)
    sampler.run_mcmc(initial, int(nsteps), progress=False)
    chain = np.asarray(sampler.get_chain(discard=int(burnin)), dtype="f8")
    logp = np.asarray(sampler.get_log_prob(discard=int(burnin)), dtype="f8")
    summary = summarize_chain(chain, logp, names)
    summary.update(
        {
            "nwalkers": int(nwalkers),
            "nsteps": int(nsteps),
            "burnin": int(burnin),
            "acceptance_fraction_mean": float(np.mean(sampler.acceptance_fraction)),
        }
    )
    return summary, chain, logp


def fisher_covariance(
    evaluate: Callable[[np.ndarray], np.ndarray],
    precision: np.ndarray,
    theta: np.ndarray,
) -> np.ndarray:
    nparams = int(theta.size)
    columns = []
    for index in range(nparams):
        step = float(FISHER_STEPS[index])
        plus, minus = np.asarray(theta, dtype="f8").copy(), np.asarray(theta, dtype="f8").copy()
        plus[index] += step
        minus[index] -= step
        columns.append((evaluate(plus) - evaluate(minus)) / (2.0 * step))
    jacobian = np.column_stack(columns)
    return np.linalg.inv(jacobian.T @ precision @ jacobian)

