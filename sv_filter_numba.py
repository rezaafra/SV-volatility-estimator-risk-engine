"""
Numba-compiled marginal log-likelihood for the SV bootstrap particle filter.

PMMH evaluates the PF likelihood thousands of times; that inner loop is the
whole cost of a fit. The likelihood path only needs to (a) propagate the
log-variance state with Gaussian noise and (b) evaluate the Student-t
observation density -- no t-sampling -- so it compiles cleanly under numba's
nopython mode and runs at C speed.

`pf_loglik` is a drop-in replacement for bootstrap_pf(...)["loglik"], matching
its SIR scheme (carried normalized weights, systematic adaptive resampling,
exact marginal-likelihood accumulation). Values agree with the NumPy filter up
to Monte-Carlo noise; use it as the `loglik_fn` argument to pmmh().
"""

from __future__ import annotations

import math

import numpy as np
from numba import njit

from sv_filter import SVParams


@njit(cache=True, fastmath=True)
def _pf_loglik(r, mu, phi, sigma_eta, nu, N, resample_frac, seed):
    np.random.seed(seed)
    T = r.shape[0]
    logN = math.log(N)
    c = (math.lgamma(0.5 * (nu + 1.0)) - math.lgamma(0.5 * nu)
         - 0.5 * math.log((nu - 2.0) * math.pi))
    half_nu1 = 0.5 * (nu + 1.0)
    nu_m2 = nu - 2.0
    sd0 = sigma_eta / math.sqrt(1.0 - phi * phi)
    thresh = resample_frac * N

    h = np.empty(N)
    logw = np.empty(N)
    w = np.empty(N)
    cum = np.empty(N)
    newh = np.empty(N)
    for i in range(N):
        h[i] = mu + sd0 * np.random.normal()
        logw[i] = -logN

    loglik = 0.0
    for t in range(T):
        rt = r[t]
        maxlw = -1.0e300
        for i in range(N):
            h[i] = mu + phi * (h[i] - mu) + sigma_eta * np.random.normal()
            eps = rt * math.exp(-0.5 * h[i])
            ll_obs = c - half_nu1 * math.log1p(eps * eps / nu_m2) - 0.5 * h[i]
            logw[i] += ll_obs
            if logw[i] > maxlw:
                maxlw = logw[i]

        s = 0.0
        for i in range(N):
            s += math.exp(logw[i] - maxlw)
        inc = maxlw + math.log(s)
        loglik += inc

        ss = 0.0
        for i in range(N):
            logw[i] -= inc                 # normalize
            w[i] = math.exp(logw[i])
            ss += w[i] * w[i]
        ess = 1.0 / ss

        if ess < thresh:
            acc = 0.0
            for i in range(N):
                acc += w[i]
                cum[i] = acc
            cum[N - 1] = 1.0
            u0 = np.random.random() / N
            j = 0
            for i in range(N):
                pos = u0 + i / N
                while pos > cum[j] and j < N - 1:
                    j += 1
                newh[i] = h[j]
            for i in range(N):
                h[i] = newh[i]
                logw[i] = -logN
    return loglik


def pf_loglik(r, params: SVParams, n_particles: int = 1000,
              resample_frac: float = 0.5, seed: int = 0) -> float:
    """Python wrapper; signature matches what pmmh(loglik_fn=...) expects."""
    return _pf_loglik(np.ascontiguousarray(r, dtype=np.float64),
                      params.mu, params.phi, params.sigma_eta, params.nu,
                      int(n_particles), float(resample_frac), int(seed))


# --------------------------------------------------------------------------- #
# Benchmark + agreement check vs the NumPy filter
# --------------------------------------------------------------------------- #
if __name__ == "__main__":
    import time
    from sv_filter import simulate_sv, bootstrap_pf

    truth = SVParams(mu=-9.0, phi=0.98, sigma_eta=0.15, nu=6.0)
    _, r = simulate_sv(truth, n=4000, seed=1)
    N = 2000

    # warm up the JIT (first call compiles)
    _ = pf_loglik(r[:50], truth, n_particles=64, seed=0)

    t0 = time.time()
    ll_nb = pf_loglik(r, truth, n_particles=N, seed=7)
    t_nb = time.time() - t0

    t0 = time.time()
    ll_np = bootstrap_pf(r, truth, n_particles=N, seed=7)["loglik"]
    t_np = time.time() - t0

    print(f"numba loglik = {ll_nb:.1f}   ({t_nb*1000:.0f} ms)")
    print(f"numpy loglik = {ll_np:.1f}   ({t_np*1000:.0f} ms)")
    print(f"agreement: |diff| = {abs(ll_nb - ll_np):.2f}  "
          f"(MC noise; both estimate the same quantity)")
    print(f"SPEEDUP: {t_np / t_nb:.1f}x")

    print("\nlog-lik vs phi (numba) -- should peak near 0.98:")
    for phi in (0.90, 0.95, 0.98, 0.99):
        p = SVParams(truth.mu, phi, truth.sigma_eta, truth.nu)
        print(f"  phi={phi:.2f}: {pf_loglik(r, p, n_particles=N, seed=11):.1f}")
