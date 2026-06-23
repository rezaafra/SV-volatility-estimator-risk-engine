"""
Risk engine + backtest harness (v1).

Produces one-step-ahead predictive VaR and Expected Shortfall and scores them
with the standard VaR/ES backtests:

  * Kupiec POF              -- unconditional coverage  (breach rate == p ?)
  * Christoffersen          -- independence of breaches (no clustering)
  * Conditional coverage    -- the two combined
  * Acerbi-Szekely Z2       -- whether ES is the right *size* on breach days

Two forecasters share one harness so the SV-t model and an EWMA-Gaussian
baseline are scored identically:

  sv_risk_forecast    -- SV particle filter, Student-t predictive (the model)
  ewma_risk_forecast  -- RiskMetrics EWMA variance, Gaussian predictive (baseline)

Convention: r_t is a return. The level-p predictive quantile q_t is negative;
VaR_t = -q_t > 0; a breach is r_t < q_t; ES_t = -E[r | r < q_t] > 0.

(For production these two would consume a single streaming filter core; kept
as separate loops here for readability. The math kernels are imported from
sv_filter so nothing is duplicated.)
"""

from __future__ import annotations

from math import lgamma, log, pi, sqrt

import numpy as np

from sv_filter import (
    SVParams,
    simulate_sv,
    _obs_loglik,
    _systematic_resample,
    _weighted_quantile,
)

# --------------------------------------------------------------------------- #
# Gaussian helpers (no scipy)
# --------------------------------------------------------------------------- #
def _norm_pdf(z: float) -> float:
    return np.exp(-0.5 * z * z) / sqrt(2.0 * pi)


def _norm_ppf(p: float) -> float:
    """Acklam rational approximation to the standard-normal quantile."""
    a = [-3.969683028665376e+01, 2.209460984245205e+02, -2.759285104469687e+02,
         1.383577518672690e+02, -3.066479806614716e+01, 2.506628277459239e+00]
    b = [-5.447609879822406e+01, 1.615858368580409e+02, -1.556989798598866e+02,
         6.680131188771972e+01, -1.328068155288572e+01]
    c = [-7.784894002430293e-03, -3.223964580411365e-01, -2.400758277161838e+00,
         -2.549732539343734e+00, 4.374664141464968e+00, 2.938163982698783e+00]
    d = [7.784695709041462e-03, 3.224671290700398e-01, 2.445134137142996e+00,
         3.754408661907416e+00]
    plow, phigh = 0.02425, 1 - 0.02425
    if p < plow:
        q = sqrt(-2 * log(p))
        return (((((c[0]*q+c[1])*q+c[2])*q+c[3])*q+c[4])*q+c[5]) / \
               ((((d[0]*q+d[1])*q+d[2])*q+d[3])*q+1)
    if p <= phigh:
        q = p - 0.5
        r = q*q
        return (((((a[0]*r+a[1])*r+a[2])*r+a[3])*r+a[4])*r+a[5])*q / \
               (((((b[0]*r+b[1])*r+b[2])*r+b[3])*r+b[4])*r+1)
    q = sqrt(-2 * log(1 - p))
    return -(((((c[0]*q+c[1])*q+c[2])*q+c[3])*q+c[4])*q+c[5]) / \
            ((((d[0]*q+d[1])*q+d[2])*q+d[3])*q+1)


# --------------------------------------------------------------------------- #
# SV particle-filter risk forecaster
# --------------------------------------------------------------------------- #
def sv_risk_forecast(
    r: np.ndarray,
    params: SVParams,
    levels=(0.01, 0.05),
    n_particles: int = 1000,
    n_forward: int = 40,
    resample_frac: float = 0.5,
    seed: int | None = None,
):
    """
    Online predict -> forecast -> update loop.

    Returns {'var': {p: array(T)}, 'es': {p: array(T)}}, each forecast made
    ex ante (using r_{0:t-1}) for the realized return r_t.
    """
    r = np.asarray(r, float)
    T = r.size
    N = n_particles
    rng = np.random.default_rng(seed)
    thresh = resample_frac * N
    t_scale = sqrt((params.nu - 2.0) / params.nu)

    sd0 = sqrt(params.stationary_var)
    h = params.mu + sd0 * rng.standard_normal(N)
    w = np.full(N, 1.0 / N)

    var = {p: np.empty(T) for p in levels}
    es = {p: np.empty(T) for p in levels}

    for t in range(T):
        if t > 0:                                            # predict h_t
            h = params.mu + params.phi * (h - params.mu) \
                + params.sigma_eta * rng.standard_normal(N)

        # --- one-step predictive sample of r_t (mixture of scaled-t) ---
        eps = rng.standard_t(params.nu, size=(N, n_forward)) * t_scale
        r_sim = (np.exp(0.5 * h)[:, None] * eps).ravel()
        wsim = np.repeat(w / n_forward, n_forward)
        order = np.argsort(r_sim)
        rs, ws = r_sim[order], wsim[order]
        cw = np.cumsum(ws)
        for p in levels:
            k = np.searchsorted(cw, p)                       # predictive p-quantile
            k = min(k, rs.size - 1)
            q = rs[k]
            var[p][t] = -q
            tail = rs[:k + 1]
            tw = ws[:k + 1]
            es[p][t] = -(np.sum(tail * tw) / np.sum(tw)) if tw.sum() > 0 else -q

        # --- update with realized r_t ---
        logw = np.log(w) + _obs_loglik(r[t], h, params.nu)
        m = logw.max()
        w = np.exp(logw - m)
        w /= w.sum()
        if 1.0 / np.sum(w * w) < thresh:
            idx = _systematic_resample(w, rng)
            h = h[idx]
            w = np.full(N, 1.0 / N)

    return {"var": var, "es": es}


# --------------------------------------------------------------------------- #
# EWMA-Gaussian baseline (RiskMetrics)
# --------------------------------------------------------------------------- #
def ewma_risk_forecast(r, levels=(0.01, 0.05), lam: float = 0.94, warmup: int = 50):
    r = np.asarray(r, float)
    T = r.size
    s2 = np.var(r[:warmup]) if T >= warmup else np.var(r)
    var = {p: np.empty(T) for p in levels}
    es = {p: np.empty(T) for p in levels}
    z = {p: _norm_ppf(p) for p in levels}                    # negative
    for t in range(T):
        sigma = sqrt(s2)
        for p in levels:
            var[p][t] = -z[p] * sigma                        # positive VaR
            es[p][t] = sigma * _norm_pdf(z[p]) / p           # Gaussian ES
        s2 = lam * s2 + (1.0 - lam) * r[t] ** 2              # update for t+1
    return {"var": var, "es": es}


# --------------------------------------------------------------------------- #
# Backtests
# --------------------------------------------------------------------------- #
def _chi2_sf_1df(x: float) -> float:
    """Survival function of chi-square(1) = erfc(sqrt(x/2))."""
    from math import erfc
    return erfc(sqrt(x / 2.0))


def kupiec_pof(breaches: np.ndarray, p: float):
    n = breaches.size
    x = int(breaches.sum())
    pi = x / n
    if x == 0 or x == n:
        # unrestricted MLE is degenerate (pi=0 or 1); LR vs the null only
        lr = -2.0 * ((n - x) * log(1 - p) + x * log(p))
    else:
        ll0 = (n - x) * log(1 - p) + x * log(p)
        ll1 = (n - x) * log(1 - pi) + x * log(pi)
        lr = -2.0 * (ll0 - ll1)
    return {"breaches": x, "rate": pi, "expected": p, "LR": lr,
            "pvalue": _chi2_sf_1df(lr)}


def christoffersen_independence(breaches: np.ndarray):
    b = breaches.astype(int)
    n00 = n01 = n10 = n11 = 0
    for prev, cur in zip(b[:-1], b[1:]):
        if prev == 0 and cur == 0: n00 += 1
        elif prev == 0 and cur == 1: n01 += 1
        elif prev == 1 and cur == 0: n10 += 1
        else: n11 += 1
    pi01 = n01 / (n00 + n01) if (n00 + n01) else 0.0
    pi11 = n11 / (n10 + n11) if (n10 + n11) else 0.0
    pi = (n01 + n11) / (n00 + n01 + n10 + n11)
    def term(num, k):
        return num * log(k) if (num > 0 and k > 0) else 0.0
    ll_null = term(n00 + n10, 1 - pi) + term(n01 + n11, pi)
    ll_alt = (term(n00, 1 - pi01) + term(n01, pi01)
              + term(n10, 1 - pi11) + term(n11, pi11))
    lr = -2.0 * (ll_null - ll_alt)
    return {"LR": lr, "pvalue": _chi2_sf_1df(lr), "pi01": pi01, "pi11": pi11}


def acerbi_szekely_z2(returns, var, es, p: float):
    """
    Z2 = 1 + (1/(N p)) * sum_t I_t * r_t / ES_t,   I_t = 1{r_t < -VaR_t}.
    E[Z2] ~ 0 under a correct model; Z2 < 0 => ES underestimates tail loss.
    """
    r = np.asarray(returns, float)
    q = -np.asarray(var, float)                              # negative quantile
    breach = r < q
    n = r.size
    contrib = np.where(breach, r / np.asarray(es, float), 0.0)
    return 1.0 + contrib.sum() / (n * p)


def backtest_report(returns, fc, p: float, label: str):
    var, es = fc["var"][p], fc["es"][p]
    breaches = np.asarray(returns) < (-np.asarray(var))
    kp = kupiec_pof(breaches, p)
    ci = christoffersen_independence(breaches)
    z2 = acerbi_szekely_z2(returns, var, es, p)
    cc_lr = kp["LR"] + ci["LR"]
    from math import erfc
    cc_p = erfc(sqrt(cc_lr / 2.0)) if cc_lr < 50 else 0.0    # crude chi2(2) tail
    print(f"\n[{label}]  p={p:.0%}   breaches {kp['breaches']}/{len(breaches)} "
          f"= {kp['rate']:.2%}  (expected {p:.2%})")
    print(f"    Kupiec POF        LR={kp['LR']:6.2f}  p={kp['pvalue']:.3f}"
          f"   {'PASS' if kp['pvalue']>0.05 else 'FAIL'}")
    print(f"    Christoffersen    LR={ci['LR']:6.2f}  p={ci['pvalue']:.3f}"
          f"   {'PASS' if ci['pvalue']>0.05 else 'FAIL'}")
    print(f"    Acerbi-Szekely Z2 = {z2:+.3f}   "
          f"({'ES OK' if z2 > -0.10 else 'ES UNDERESTIMATES tail loss'})")


# --------------------------------------------------------------------------- #
# Validation on synthetic data (known fat tails, nu=6)
# --------------------------------------------------------------------------- #
if __name__ == "__main__":
    truth = SVParams(mu=-9.0, phi=0.98, sigma_eta=0.15, nu=6.0)
    _, r = simulate_sv(truth, n=6000, seed=21)
    print(f"synthetic returns: {r.size}  (true nu=6 -> genuine fat tails)")

    sv = sv_risk_forecast(r, truth, levels=(0.01, 0.05),
                          n_particles=1000, n_forward=40, seed=3)
    ew = ewma_risk_forecast(r, levels=(0.01, 0.05), lam=0.94)

    for p in (0.01, 0.05):
        backtest_report(r, sv, p, "SV particle filter (t)")
        backtest_report(r, ew, p, "EWMA-Gaussian baseline")
