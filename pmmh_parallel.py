"""
Parallel multi-chain PMMH + convergence diagnostics.

Independent chains are embarrassingly parallel, so we run them across cores
(~min(n_chains, n_cores) wall-clock speedup) and combine. Each chain uses the
numba-compiled filter (sv_filter_numba.pf_loglik), which is fastest in the
moderate-particle regime PMMH lives in.

Running multiple chains is not just for speed: pooling them and computing the
Gelman-Rubin split-R-hat is how you actually demonstrate the sampler converged
(R-hat ~ 1.0 across chains started from dispersed points), rather than asserting
it from one chain.
"""

from __future__ import annotations

import multiprocessing as mp

import numpy as np

from sv_filter import SVParams
from pmmh import pmmh, summarize


def _chain_worker(args):
    r, init, kw, seed = args
    from sv_filter_numba import pf_loglik          # numba filter inside the worker
    res = pmmh(r, init=init, seed=seed, verbose=False, loglik_fn=pf_loglik, **kw)
    return res["samples"]


def gelman_rubin(chains: list[np.ndarray]) -> np.ndarray:
    """Split-less R-hat per parameter from K chains of equal post-burn length."""
    M = min(c.shape[0] for c in chains)
    X = np.stack([c[:M] for c in chains])          # (K, M, P)
    K, M, P = X.shape
    chain_means = X.mean(axis=1)                   # (K, P)
    B = M * chain_means.var(axis=0, ddof=1)        # between-chain
    W = X.var(axis=1, ddof=1).mean(axis=0)         # within-chain
    var_hat = (M - 1) / M * W + B / M
    return np.sqrt(var_hat / W)


def run_chains(r, init: SVParams, n_chains=4, base_seed=0,
               n_iter=2000, burn_in=800, n_particles=600):
    kw = dict(n_iter=n_iter, burn_in=burn_in, n_particles=n_particles)
    args = [(r, init, kw, base_seed + i) for i in range(n_chains)]
    # fork on Unix, spawn on Windows/macOS-default
    method = "fork" if "fork" in mp.get_all_start_methods() else "spawn"
    ctx = mp.get_context(method)
    with ctx.Pool(processes=n_chains) as pool:
        chains = pool.map(_chain_worker, args)
    return chains


def report(chains, names=("mu", "phi", "sigma_eta", "nu")):
    rh = gelman_rubin(chains)
    pooled = np.vstack(chains)
    print(f"\n{len(chains)} chains x {chains[0].shape[0]} draws  "
          f"({pooled.shape[0]} pooled)")
    print(f"{'param':>10} {'mean':>10} {'q05':>10} {'q95':>10} {'R-hat':>8}")
    for j, nm in enumerate(names):
        col = pooled[:, j]
        q = np.quantile(col, [0.05, 0.95])
        flag = "" if rh[j] < 1.05 else "  <-- not converged"
        print(f"{nm:>10} {col.mean():>10.4f} {q[0]:>10.4f} {q[1]:>10.4f} "
              f"{rh[j]:>8.3f}{flag}")
    return pooled, rh


if __name__ == "__main__":
    import time
    from sv_filter import simulate_sv
    from sv_filter_numba import pf_loglik

    truth = SVParams(mu=-9.0, phi=0.98, sigma_eta=0.15, nu=6.0)
    _, r = simulate_sv(truth, n=1200, seed=7)

    pf_loglik(r[:50], truth, n_particles=64, seed=0)   # warm JIT before fork

    # deliberately dispersed start across chains via different seeds
    import os

    start = SVParams(mu=-8.5, phi=0.93, sigma_eta=0.30, nu=10.0)
    n_chains = 3
    n_cores = os.cpu_count() or 1
    print(f"running {n_chains} chains across up to {n_cores} cores "
          f"(~{min(n_chains, n_cores)}x faster than serial)")
    t0 = time.time()
    chains = run_chains(r, start, n_chains=n_chains,
                        n_iter=700, burn_in=250, n_particles=500)
    print(f"wall-clock: {time.time()-t0:.0f}s")

    report(chains)
    print(f"\ntruth: mu={truth.mu} phi={truth.phi} "
          f"sigma_eta={truth.sigma_eta} nu={truth.nu}")
