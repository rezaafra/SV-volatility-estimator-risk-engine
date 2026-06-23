"""
Stochastic-volatility bootstrap particle filter (v1).

Model
-----
Latent log-variance follows a discretized OU / AR(1) process:

    h_t = mu + phi * (h_{t-1} - mu) + sigma_eta * eta_t,    eta_t ~ N(0, 1)

Observation is a return with unit-variance Student-t innovations:

    r_t = exp(h_t / 2) * eps_t,    eps_t ~ standardized-t(nu),  Var(eps_t) = 1

so that exp(h_t) is exactly the conditional variance of r_t. The Student-t
innovation is what lets the *risk engine* cover the tails; Gaussian eps_t
systematically under-covers 1% VaR.

This module provides:
  - SVParams        : parameter container with OU-style diagnostics
  - simulate_sv     : generate (h, r) from known params  -> ground truth
  - bootstrap_pf    : SIR filter with systematic adaptive resampling,
                      returning filtered moments and the exact marginal
                      log-likelihood estimator (used later for MLE/PMMH).

No scipy dependency. Vectorized over particles; the time axis is sequential
by nature (numba can wrap the per-step kernel later if profiling demands it).
"""

from __future__ import annotations

from dataclasses import dataclass
from math import lgamma, log, pi

import numpy as np

LOG_2PI = log(2.0 * pi)


# --------------------------------------------------------------------------- #
# Parameters
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class SVParams:
    mu: float          # long-run mean of log-variance
    phi: float         # AR(1) persistence,  |phi| < 1
    sigma_eta: float   # vol-of-vol (std of log-variance shock)
    nu: float          # Student-t degrees of freedom,  nu > 2

    def __post_init__(self) -> None:
        if not (-1.0 < self.phi < 1.0):
            raise ValueError("phi must satisfy |phi| < 1 for stationarity")
        if self.sigma_eta <= 0:
            raise ValueError("sigma_eta must be positive")
        if self.nu <= 2.0:
            raise ValueError("nu must exceed 2 for finite variance")

    @property
    def stationary_var(self) -> float:
        """Stationary variance of the log-variance state."""
        return self.sigma_eta**2 / (1.0 - self.phi**2)

    @property
    def mean_reversion_half_life(self) -> float:
        """Steps for a log-variance deviation to decay by half (OU intuition)."""
        return log(2.0) / (-log(self.phi))


# --------------------------------------------------------------------------- #
# Standardized Student-t (unit variance) log density
# --------------------------------------------------------------------------- #
def _std_t_logpdf(x: np.ndarray, nu: float) -> np.ndarray:
    """
    log pdf of a unit-variance Student-t with nu dof.

    Derived from t_nu rescaled by sqrt((nu-2)/nu); the x^2 term collapses
    neatly to x^2 / (nu - 2).
    """
    const = (
        lgamma(0.5 * (nu + 1.0))
        - lgamma(0.5 * nu)
        - 0.5 * log((nu - 2.0) * pi)   # log(nu*pi) + log((nu-2)/nu) combined
    )
    return const - 0.5 * (nu + 1.0) * np.log1p(x * x / (nu - 2.0))


def _obs_loglik(r_t: float, h: np.ndarray, nu: float) -> np.ndarray:
    """
    log g(r_t | h^(i)) for every particle h^(i).

    r = exp(h/2) * eps  =>  eps = r * exp(-h/2),  Jacobian d eps/d r = exp(-h/2).
    """
    eps = r_t * np.exp(-0.5 * h)
    return _std_t_logpdf(eps, nu) - 0.5 * h


# --------------------------------------------------------------------------- #
# Synthetic data (ground truth for validation)
# --------------------------------------------------------------------------- #
def simulate_sv(params: SVParams, n: int, seed: int | None = None):
    """Return (h, r): latent log-variance path and observed returns."""
    rng = np.random.default_rng(seed)
    h = np.empty(n)
    # start in the stationary distribution
    h[0] = params.mu + np.sqrt(params.stationary_var) * rng.standard_normal()
    for t in range(1, n):
        h[t] = (
            params.mu
            + params.phi * (h[t - 1] - params.mu)
            + params.sigma_eta * rng.standard_normal()
        )
    scale = np.sqrt((params.nu - 2.0) / params.nu)          # -> unit variance
    eps = rng.standard_t(params.nu, size=n) * scale
    r = np.exp(0.5 * h) * eps
    return h, r


# --------------------------------------------------------------------------- #
# Resampling helpers
# --------------------------------------------------------------------------- #
def _systematic_resample(weights: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Low-variance systematic resampling; returns parent indices."""
    n = weights.size
    positions = (rng.random() + np.arange(n)) / n
    cumulative = np.cumsum(weights)
    cumulative[-1] = 1.0                                     # guard fp drift
    return np.searchsorted(cumulative, positions)


def _logsumexp(a: np.ndarray) -> float:
    m = np.max(a)
    return m + log(np.sum(np.exp(a - m)))


# --------------------------------------------------------------------------- #
# Bootstrap particle filter
# --------------------------------------------------------------------------- #
def bootstrap_pf(
    r: np.ndarray,
    params: SVParams,
    n_particles: int = 2000,
    resample_frac: float = 0.5,
    seed: int | None = None,
    quantiles=(0.05, 0.5, 0.95),
):
    """
    SIR filter for the SV model.

    Returns a dict with:
      loglik          : exact marginal log-likelihood estimate  log p(r_{1:T})
                        (this is the objective for offline MLE / PMMH)
      filt_logvar     : E[h_t | r_{1:t}]
      filt_var        : E[exp(h_t) | r_{1:t}]   (conditional variance)
      var_quantiles   : weighted quantiles of exp(h_t)  (shape: len(q) x T)
      ess             : effective sample size at each step
    """
    r = np.asarray(r, dtype=float)
    T = r.size
    N = n_particles
    rng = np.random.default_rng(seed)
    resample_thresh = resample_frac * N

    # initialize from the stationary distribution
    sd0 = np.sqrt(params.stationary_var)
    h = params.mu + sd0 * rng.standard_normal(N)
    log_w = np.full(N, -log(N))                              # uniform weights

    filt_logvar = np.empty(T)
    filt_var = np.empty(T)
    ess = np.empty(T)
    qs = np.empty((len(quantiles), T))
    loglik = 0.0

    for t in range(T):
        # propagate (bootstrap proposal = the prior transition)
        h = (
            params.mu
            + params.phi * (h - params.mu)
            + params.sigma_eta * rng.standard_normal(N)
        )

        # incremental weight update in log space
        log_w = log_w + _obs_loglik(r[t], h, params.nu)
        inc = _logsumexp(log_w)                              # log sum of unnorm w
        loglik += inc
        log_w -= inc                                         # normalize

        w = np.exp(log_w)
        # filtered moments (post-update, pre-resample)
        filt_logvar[t] = np.sum(w * h)
        var_i = np.exp(h)
        filt_var[t] = np.sum(w * var_i)
        qs[:, t] = _weighted_quantile(var_i, w, quantiles)
        ess[t] = 1.0 / np.sum(w * w)

        # adaptive resampling
        if ess[t] < resample_thresh:
            idx = _systematic_resample(w, rng)
            h = h[idx]
            log_w = np.full(N, -log(N))

    return {
        "loglik": loglik,
        "filt_logvar": filt_logvar,
        "filt_var": filt_var,
        "var_quantiles": qs,
        "ess": ess,
        "quantile_levels": np.asarray(quantiles),
    }


def _weighted_quantile(values: np.ndarray, weights: np.ndarray, qs) -> np.ndarray:
    order = np.argsort(values)
    v = values[order]
    cw = np.cumsum(weights[order])
    cw /= cw[-1]
    return np.interp(qs, cw, v)


# --------------------------------------------------------------------------- #
# Self-test: simulate known params, filter, check recovery
# --------------------------------------------------------------------------- #
if __name__ == "__main__":
    true = SVParams(mu=-9.0, phi=0.98, sigma_eta=0.15, nu=6.0)
    print("Stationary var of log-var :", round(true.stationary_var, 4))
    print("Vol mean-reversion half-life:", round(true.mean_reversion_half_life, 1), "steps")

    h_true, r = simulate_sv(true, n=4000, seed=1)
    out = bootstrap_pf(r, true, n_particles=4000, seed=2)

    var_true = np.exp(h_true)
    corr = np.corrcoef(out["filt_var"], var_true)[0, 1]
    rel_err = np.abs(out["filt_var"] - var_true) / var_true

    print("\n--- filter recovery (params known) ---")
    print(f"marginal log-likelihood : {out['loglik']:.1f}")
    print(f"corr(filtered var, true var) : {corr:.3f}")
    print(f"median relative error on variance : {np.median(rel_err):.2%}")
    print(f"mean ESS / N : {out['ess'].mean() / 4000:.2f}")

    # likelihood should peak near the true params -> quick sanity slice on phi
    print("\n--- log-lik vs phi (should peak near 0.98) ---")
    for phi in (0.90, 0.95, 0.98, 0.99):
        p = SVParams(mu=true.mu, phi=phi, sigma_eta=true.sigma_eta, nu=true.nu)
        ll = bootstrap_pf(r, p, n_particles=4000, seed=3)["loglik"]
        print(f"  phi={phi:.2f} : loglik={ll:.1f}")
