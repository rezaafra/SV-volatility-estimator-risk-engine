"""
Particle Marginal Metropolis-Hastings (PMMH) for the SV model.

Andrieu, Doucet & Holenstein (2010): the bootstrap PF returns an *unbiased*
estimate of the marginal likelihood p(r_{1:T} | theta). Using that estimate
inside Metropolis-Hastings still leaves the exact parameter posterior
invariant (pseudo-marginal MCMC). This module wraps `bootstrap_pf` from
sv_filter to sample the posterior over (mu, phi, sigma_eta, nu).

Design
------
* Sampling happens in an UNCONSTRAINED space so proposals are always valid:
      xi = (mu, arctanh(phi), log(sigma_eta), log(nu - 2))
  The log-Jacobian of xi -> theta is carried into the target.
* Adaptive random-walk proposal: Haario empirical-covariance adaptation plus
  a Robbins-Monro global scale targeting 0.234 acceptance, ACTIVE ONLY during
  burn-in and then frozen, so the stationary chain is a valid MH chain.
* Pseudo-marginal correctness: the current state's likelihood estimate is
  carried forward (NOT refreshed); only the proposal is re-evaluated.

Particle count note: PMMH mixes well when Var[log p_hat] ~ 1-2 at the mode
(Pitt et al. 2012; Doucet et al. 2015). Too few particles -> sticky chain.
Increase n_particles if acceptance collapses or the chain gets stuck.
"""

from __future__ import annotations

from dataclasses import dataclass
from math import lgamma, log, pi

import numpy as np

from sv_filter import SVParams, bootstrap_pf, simulate_sv

LOG_2PI = log(2.0 * pi)


# --------------------------------------------------------------------------- #
# Priors (defined on the natural / constrained parameters)
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class Priors:
    # mu ~ Normal(mu_loc, mu_scale)
    mu_loc: float = -9.0
    mu_scale: float = 5.0
    # (phi + 1) / 2 ~ Beta(phi_a, phi_b)   -- favors high persistence
    phi_a: float = 25.0
    phi_b: float = 1.5
    # sigma_eta ~ HalfNormal(sigma_scale)
    sigma_scale: float = 0.5
    # (nu - 2) ~ Exponential(rate = nu_rate)   -- mean 1/rate
    nu_rate: float = 0.1

    def log_density(self, p: SVParams) -> float:
        # mu : Normal
        lp = -0.5 * LOG_2PI - log(self.mu_scale) \
            - 0.5 * ((p.mu - self.mu_loc) / self.mu_scale) ** 2
        # phi : Beta on (phi+1)/2  (+ constant Jacobian log(1/2), kept for cleanliness)
        y = 0.5 * (p.phi + 1.0)
        lp += (
            (self.phi_a - 1.0) * log(y)
            + (self.phi_b - 1.0) * log(1.0 - y)
            - (lgamma(self.phi_a) + lgamma(self.phi_b) - lgamma(self.phi_a + self.phi_b))
            + log(0.5)
        )
        # sigma_eta : HalfNormal
        lp += log(2.0) - 0.5 * LOG_2PI - log(self.sigma_scale) \
            - 0.5 * (p.sigma_eta / self.sigma_scale) ** 2
        # nu - 2 : Exponential
        lp += log(self.nu_rate) - self.nu_rate * (p.nu - 2.0)
        return lp


# --------------------------------------------------------------------------- #
# Constrained <-> unconstrained transforms (+ log-Jacobian)
# --------------------------------------------------------------------------- #
def to_unconstrained(p: SVParams) -> np.ndarray:
    return np.array([p.mu, np.arctanh(p.phi), log(p.sigma_eta), log(p.nu - 2.0)])


def to_constrained(xi: np.ndarray) -> SVParams:
    return SVParams(
        mu=float(xi[0]),
        phi=float(np.tanh(xi[1])),
        sigma_eta=float(np.exp(xi[2])),
        nu=float(2.0 + np.exp(xi[3])),
    )


def log_jacobian(p: SVParams, xi: np.ndarray) -> float:
    """log |d theta / d xi| for the four transforms."""
    return (
        log(1.0 - p.phi**2)   # phi = tanh(.)
        + xi[2]               # sigma = exp(.)  -> log(sigma)
        + xi[3]               # nu-2  = exp(.)  -> log(nu-2)
    )                          # mu identity contributes 0


# --------------------------------------------------------------------------- #
# PMMH sampler
# --------------------------------------------------------------------------- #
def pmmh(
    r: np.ndarray,
    init: SVParams,
    n_iter: int = 3000,
    burn_in: int = 1000,
    n_particles: int = 1000,
    priors: Priors | None = None,
    seed: int | None = None,
    verbose: bool = True,
    loglik_fn=None,
):
    priors = priors or Priors()
    rng = np.random.default_rng(seed)
    d = 4

    if loglik_fn is None:                       # default: the NumPy filter
        def loglik_fn(rr, params, n_particles, seed):
            return bootstrap_pf(rr, params, n_particles=n_particles, seed=seed)["loglik"]

    def log_target(p: SVParams, xi: np.ndarray, pf_seed: int):
        ll = loglik_fn(r, p, n_particles=n_particles, seed=pf_seed)
        return ll + priors.log_density(p) + log_jacobian(p, xi), ll

    # --- initial state ---
    xi = to_unconstrained(init)
    p = init
    lt, ll = log_target(p, xi, int(rng.integers(1 << 30)))

    chain = np.empty((n_iter, d))
    loglik_trace = np.empty(n_iter)
    accepted = np.zeros(n_iter, dtype=bool)

    # --- adaptive proposal state ---
    scale = 2.38 / np.sqrt(d)                 # global RW scale (Robbins-Monro)
    L = 0.1 * np.eye(d)                       # initial proposal chol factor
    emp_mean = xi.copy()
    emp_cov = 0.1 * np.eye(d)
    target_acc = 0.234

    for i in range(n_iter):
        prop = xi + scale * (L @ rng.standard_normal(d))
        p_prop = to_constrained(prop)
        lt_prop, ll_prop = log_target(p_prop, prop, int(rng.integers(1 << 30)))

        log_alpha = lt_prop - lt
        if log(rng.random()) < log_alpha:
            xi, p, lt, ll = prop, p_prop, lt_prop, ll_prop
            accepted[i] = True

        chain[i] = xi
        loglik_trace[i] = ll

        # --- adaptation during burn-in only ---
        if i < burn_in:
            a = min(1.0, np.exp(log_alpha))
            gamma = 1.0 / (i + 1) ** 0.6
            scale *= np.exp(gamma * (a - target_acc))         # Robbins-Monro
            delta = xi - emp_mean
            emp_mean = emp_mean + gamma * delta
            emp_cov = emp_cov + gamma * (np.outer(delta, delta) - emp_cov)
            if i > 50 and i % 50 == 0:
                try:
                    L = np.linalg.cholesky(emp_cov + 1e-8 * np.eye(d))
                except np.linalg.LinAlgError:
                    pass
        if verbose and (i + 1) % max(1, n_iter // 10) == 0:
            print(f"  iter {i+1}/{n_iter}  acc={accepted[:i+1].mean():.2f}  "
                  f"loglik={ll:.1f}")

    # map chain back to constrained parameters
    post = np.empty((n_iter, d))
    post[:, 0] = chain[:, 0]
    post[:, 1] = np.tanh(chain[:, 1])
    post[:, 2] = np.exp(chain[:, 2])
    post[:, 3] = 2.0 + np.exp(chain[:, 3])

    return {
        "samples": post[burn_in:],          # mu, phi, sigma_eta, nu
        "samples_all": post,
        "loglik_trace": loglik_trace,
        "accept_rate": accepted[burn_in:].mean(),
        "param_names": ["mu", "phi", "sigma_eta", "nu"],
    }


def _ess(x: np.ndarray) -> float:
    """Rough autocorrelation-based effective sample size (initial positive seq)."""
    n = x.size
    x = x - x.mean()
    var = np.dot(x, x) / n
    if var == 0:
        return float(n)
    acf_sum = 0.0
    for k in range(1, min(n - 1, 1000)):
        rho = np.dot(x[:-k], x[k:]) / (n * var)
        if rho < 0.05:
            break
        acf_sum += rho
    return n / (1.0 + 2.0 * acf_sum)


def summarize(result: dict) -> None:
    s = result["samples"]
    print(f"\nposterior summary  (acceptance {result['accept_rate']:.2f}, "
          f"{s.shape[0]} post-burn-in draws)")
    print(f"{'param':>10} {'mean':>9} {'sd':>8} {'q05':>9} {'q50':>9} {'q95':>9} {'ESS':>7}")
    for j, name in enumerate(result["param_names"]):
        col = s[:, j]
        q = np.quantile(col, [0.05, 0.5, 0.95])
        print(f"{name:>10} {col.mean():>9.4f} {col.std():>8.4f} "
              f"{q[0]:>9.4f} {q[1]:>9.4f} {q[2]:>9.4f} {_ess(col):>7.0f}")


# --------------------------------------------------------------------------- #
# Validation: recover known params from synthetic data
# --------------------------------------------------------------------------- #
if __name__ == "__main__":
    import time

    truth = SVParams(mu=-9.0, phi=0.98, sigma_eta=0.15, nu=6.0)
    h, r = simulate_sv(truth, n=1200, seed=7)

    # time one PF to calibrate the run length
    t0 = time.time()
    _ = bootstrap_pf(r, truth, n_particles=1000, seed=0)["loglik"]
    print(f"single PF over {r.size} steps, N=1000: {time.time()-t0:.3f}s")

    start = SVParams(mu=-8.0, phi=0.90, sigma_eta=0.30, nu=10.0)
    print("\nrunning PMMH (deliberately starting away from the truth)...")
    t0 = time.time()
    res = pmmh(r, init=start, n_iter=2500, burn_in=800,
               n_particles=1000, seed=11)
    print(f"PMMH wall-clock: {time.time()-t0:.1f}s")

    summarize(res)
    print(f"\ntruth: mu={truth.mu}, phi={truth.phi}, "
          f"sigma_eta={truth.sigma_eta}, nu={truth.nu}")
