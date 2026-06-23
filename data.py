"""
Data pipeline + end-to-end out-of-sample evaluation (v1).

Flow:  fetch -> clean -> chronological split -> PMMH fit on TRAIN only
       -> freeze params -> online filter across all data -> backtest TEST slice.

Anti-lookahead guarantees:
  * parameters come from the training window only;
  * online filtering and EWMA use only past returns;
  * the test mean is never used (returns are demeaned by the TRAIN mean);
  * we score the test slice exclusively.

The Binance fetch runs wherever you have open network access. If it fails
(e.g. restricted environment), main() falls back to a synthetic hourly series
so the rest of the pipeline still runs and can be validated.
"""

from __future__ import annotations

import time

import numpy as np

from sv_filter import SVParams, simulate_sv
from pmmh import pmmh, summarize
from risk_engine import sv_risk_forecast, ewma_risk_forecast, backtest_report

BINANCE_URL = "https://api.binance.com/api/v3/klines"
_INTERVAL_MS = {"1h": 3_600_000, "1m": 60_000, "1d": 86_400_000}


# --------------------------------------------------------------------------- #
# Fetch (runs on a machine with network access)
# --------------------------------------------------------------------------- #
def fetch_binance_klines(symbol="BTCUSDT", interval="1h",
                         start_ms=None, end_ms=None, pause=0.25):
    """
    Paginated public klines pull (no API key). Returns (close_times_ms, close).
    Each request returns <= 1000 candles; we advance startTime until end_ms.
    """
    import requests

    step = _INTERVAL_MS[interval]
    if end_ms is None:
        end_ms = int(time.time() * 1000)
    if start_ms is None:
        start_ms = end_ms - 365 * 24 * step          # ~1 year by default

    times, closes = [], []
    cursor = start_ms
    while cursor < end_ms:
        params = {"symbol": symbol, "interval": interval,
                  "startTime": cursor, "endTime": end_ms, "limit": 1000}
        resp = requests.get(BINANCE_URL, params=params, timeout=15)
        resp.raise_for_status()
        batch = resp.json()
        if not batch:
            break
        for k in batch:
            times.append(int(k[6]))                  # close time
            closes.append(float(k[4]))               # close price
        cursor = int(batch[-1][0]) + step            # next open time
        if len(batch) < 1000:
            break
        time.sleep(pause)                            # be polite to the API
    return np.asarray(times), np.asarray(closes)


# --------------------------------------------------------------------------- #
# Cleaning
# --------------------------------------------------------------------------- #
def clean_to_returns(close_times_ms, close, interval="1h"):
    """
    Log returns, dropping any return that spans a data gap so we never
    fabricate a move across missing candles.
    """
    step = _INTERVAL_MS[interval]
    close = np.asarray(close, float)
    t = np.asarray(close_times_ms, float)
    raw = np.diff(np.log(close))
    dt = np.diff(t)
    ok = dt <= 1.5 * step                            # contiguous bars only
    r = raw[ok]
    n_dropped = (~ok).sum()
    return r, int(n_dropped)


# --------------------------------------------------------------------------- #
# End-to-end run
# --------------------------------------------------------------------------- #
def run(r, train_frac=0.6, pmmh_iter=1000, pmmh_burn=400,
        n_particles=800, levels=(0.01, 0.05), seed=0):
    n = r.size
    n_train = int(train_frac * n)

    # demean by TRAIN mean only (no lookahead)
    train_mean = r[:n_train].mean()
    r = r - train_mean
    print(f"returns: {n}  train {n_train} / test {n - n_train}  "
          f"(train mean removed: {train_mean:.2e})")

    # --- fit parameters on TRAIN ---
    start = SVParams(mu=float(np.log(np.var(r[:n_train]) + 1e-12)),
                     phi=0.95, sigma_eta=0.2, nu=8.0)
    print(f"\nPMMH fit on train (start mu={start.mu:.2f})...")
    t0 = time.time()
    fit = pmmh(r[:n_train], init=start, n_iter=pmmh_iter, burn_in=pmmh_burn,
               n_particles=n_particles, seed=seed, verbose=False)
    print(f"  done in {time.time()-t0:.0f}s")
    summarize(fit)

    s = fit["samples"]
    est = SVParams(mu=float(np.median(s[:, 0])), phi=float(np.median(s[:, 1])),
                   sigma_eta=float(np.median(s[:, 2])), nu=float(np.median(s[:, 3])))
    print(f"\npoint estimate (posterior median): mu={est.mu:.3f} phi={est.phi:.4f} "
          f"sigma_eta={est.sigma_eta:.3f} nu={est.nu:.2f}")

    # --- online forecasts across all data, frozen params ---
    sv = sv_risk_forecast(r, est, levels=levels, n_particles=n_particles,
                          n_forward=40, seed=seed + 1)
    ew = ewma_risk_forecast(r, levels=levels, lam=0.94)

    # --- score TEST slice only ---
    test = slice(n_train, n)
    r_test = r[test]
    print("\n================  OUT-OF-SAMPLE (test slice)  ================")
    for p in levels:
        sv_test = {"var": {p: sv["var"][p][test]}, "es": {p: sv["es"][p][test]}}
        ew_test = {"var": {p: ew["var"][p][test]}, "es": {p: ew["es"][p][test]}}
        backtest_report(r_test, sv_test, p, "SV-PF (t)  OOS")
        backtest_report(r_test, ew_test, p, "EWMA-Gaussian OOS")


# --------------------------------------------------------------------------- #
def main():
    try:
        print("fetching BTCUSDT 1h from Binance...")
        t, c = fetch_binance_klines("BTCUSDT", "1h")
        r, dropped = clean_to_returns(t, c, "1h")
        print(f"got {r.size} hourly returns ({dropped} dropped across gaps)")
    except Exception as e:           # restricted network / offline
        print(f"  fetch unavailable ({type(e).__name__}); using synthetic hourly series")
        truth = SVParams(mu=-9.5, phi=0.97, sigma_eta=0.22, nu=5.0)
        _, r = simulate_sv(truth, n=2500, seed=99)
        print(f"  synthetic truth: {truth}")

    run(r)


if __name__ == "__main__":
    main()
