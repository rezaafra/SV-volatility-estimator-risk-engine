"""
Streaming online volatility / VaR / signal estimator (v1).

The productionizable core: a stateful estimator that ingests one price at a
time and, on each tick, emits

    * vol           -- current filtered conditional volatility  sqrt(E[exp(h_t)|data])
    * vol_forecast  -- one-step-ahead forecast volatility
    * var / es      -- one-step-ahead forward VaR & ES for the NEXT period
    * signal        -- vol-targeted position size  (target_vol / vol_forecast)
    * z             -- standardized innovation  r_t / vol_t  (diagnostic)

Each tick runs predict -> update (current state) then propagates once more to
produce the forward risk numbers. State (the particle cloud) persists across
calls, so this is true online filtering at O(N) per tick.

    est = StreamingVolEstimator(params, target_vol=...)
    for price in feed:
        out = est.update(price)        # None on the very first price
        if out: act_on(out)

`estimate(prices, params)` is the batch convenience wrapper named in the brief.
"""

from __future__ import annotations

from dataclasses import dataclass
from math import exp, log, sqrt

import numpy as np

from sv_filter import SVParams, _obs_loglik, _systematic_resample


@dataclass
class Estimate:
    n_obs: int
    ret: float
    vol: float                 # current conditional vol (per period)
    vol_forecast: float        # next-period forecast vol
    var: dict                  # {level: forward VaR > 0}
    es: dict                   # {level: forward ES  > 0}
    signal: float              # vol-targeted position scalar
    z: float                   # standardized innovation
    ess: float                 # effective sample size

    def as_dict(self, ann: float = 1.0) -> dict:
        """JSON-friendly view; `ann` = periods/year to annualize vols."""
        a = sqrt(ann)
        return {
            "n_obs": self.n_obs,
            "ret": self.ret,
            "vol": self.vol * a,
            "vol_forecast": self.vol_forecast * a,
            "var": {f"{p:.4f}": v for p, v in self.var.items()},
            "es": {f"{p:.4f}": v for p, v in self.es.items()},
            "signal": self.signal,
            "z": self.z,
            "ess": self.ess,
        }


class StreamingVolEstimator:
    def __init__(
        self,
        params: SVParams,
        n_particles: int = 1000,
        n_forward: int = 40,
        levels=(0.01, 0.05),
        resample_frac: float = 0.5,
        target_vol: float | None = None,
        max_position: float = 3.0,
        seed: int | None = None,
    ):
        self.p = params
        self.N = n_particles
        self.n_forward = n_forward
        self.levels = tuple(levels)
        self.thresh = resample_frac * n_particles
        self.max_position = max_position
        self.target_vol = target_vol if target_vol is not None else exp(0.5 * params.mu)
        self.t_scale = sqrt((params.nu - 2.0) / params.nu)

        self.rng = np.random.default_rng(seed)
        sd0 = sqrt(params.stationary_var)
        self.h = params.mu + sd0 * self.rng.standard_normal(self.N)   # ~ p(h_0)
        self.w = np.full(self.N, 1.0 / self.N)
        self.prev_price: float | None = None
        self.n = 0

    # -- public API ------------------------------------------------------- #
    def update(self, price) -> Estimate | None:
        price = float(price)
        if self.prev_price is None:                  # need two prices for a return
            self.prev_price = price
            return None
        r = log(price / self.prev_price)
        self.prev_price = price
        return self._step(r)

    def update_return(self, r) -> Estimate:
        return self._step(float(r))

    # -- core ------------------------------------------------------------- #
    def _propagate(self, h: np.ndarray) -> np.ndarray:
        return (self.p.mu + self.p.phi * (h - self.p.mu)
                + self.p.sigma_eta * self.rng.standard_normal(h.size))

    def _step(self, r: float) -> Estimate:
        self.n += 1
        p = self.p

        # PREDICT h_t  (propagate posterior of h_{t-1}; from prior on first call)
        self.h = self._propagate(self.h)

        # UPDATE with realized r_t  ->  posterior p(h_t | r_{1:t})
        logw = np.log(self.w) + _obs_loglik(r, self.h, p.nu)
        m = logw.max()
        w = np.exp(logw - m)
        w /= w.sum()

        vol = sqrt(np.sum(w * np.exp(self.h)))       # current conditional vol
        z = r / vol
        ess = 1.0 / np.sum(w * w)

        if ess < self.thresh:                        # resample, keep posterior
            idx = _systematic_resample(w, self.rng)
            self.h = self.h[idx]
            w = np.full(self.N, 1.0 / self.N)
        self.w = w

        # FORWARD: propagate once more for the next-period predictive
        h_next = self._propagate(self.h)
        vol_forecast = sqrt(np.sum(self.w * np.exp(h_next)))
        var, es = self._forward_risk(h_next, self.w)

        signal = float(np.clip(self.target_vol / vol_forecast, 0.0, self.max_position))
        return Estimate(self.n, r, vol, vol_forecast, var, es, signal, z, ess)

    def _forward_risk(self, h_next: np.ndarray, w: np.ndarray):
        eps = self.rng.standard_t(self.p.nu, size=(self.N, self.n_forward)) * self.t_scale
        r_sim = (np.exp(0.5 * h_next)[:, None] * eps).ravel()
        wsim = np.repeat(w / self.n_forward, self.n_forward)
        order = np.argsort(r_sim)
        rs, ws = r_sim[order], wsim[order]
        cw = np.cumsum(ws)
        var, es = {}, {}
        for p in self.levels:
            k = min(int(np.searchsorted(cw, p)), rs.size - 1)
            q = rs[k]
            var[p] = -q
            tw = ws[:k + 1]
            es[p] = -(np.sum(rs[:k + 1] * tw) / tw.sum()) if tw.sum() > 0 else -q
        return var, es


# --------------------------------------------------------------------------- #
# Batch convenience API named in the brief
# --------------------------------------------------------------------------- #
def estimate(prices, params: SVParams, **kwargs):
    """Feed an iterable of prices; return a list of Estimate (one per return)."""
    est = StreamingVolEstimator(params, **kwargs)
    out = []
    for px in prices:
        e = est.update(px)
        if e is not None:
            out.append(e)
    return out


def estimates_to_arrays(estimates, levels=(0.01, 0.05)) -> dict:
    """Stack a list of Estimate into numpy arrays for plotting / backtesting."""
    return {
        "ret": np.array([e.ret for e in estimates]),
        "vol": np.array([e.vol for e in estimates]),
        "vol_forecast": np.array([e.vol_forecast for e in estimates]),
        "signal": np.array([e.signal for e in estimates]),
        "var": {p: np.array([e.var[p] for e in estimates]) for p in levels},
        "es": {p: np.array([e.es[p] for e in estimates]) for p in levels},
    }


# --------------------------------------------------------------------------- #
# Validation: stream a synthetic price path, backtest the forward forecasts
# --------------------------------------------------------------------------- #
if __name__ == "__main__":
    from sv_filter import simulate_sv
    from risk_engine import backtest_report

    truth = SVParams(mu=-9.0, phi=0.98, sigma_eta=0.15, nu=6.0)
    h, r = simulate_sv(truth, n=6000, seed=21)
    prices = 100.0 * np.exp(np.cumsum(r))            # returns -> price path

    ests = estimate(prices, truth, n_particles=1000, n_forward=40, seed=5)
    arr = estimates_to_arrays(ests, levels=(0.01, 0.05))
    print(f"streamed {len(prices)} prices -> {len(ests)} estimates")

    # current-vol tracking vs truth
    true_vol = np.exp(0.5 * h[1:])                    # aligned to returns r_1..
    corr = np.corrcoef(arr["vol"], true_vol[:arr["vol"].size])[0, 1]
    print(f"corr(streamed current vol, true vol) = {corr:.3f}")
    print(f"signal: mean={arr['signal'].mean():.2f}  "
          f"corr(signal, 1/true_vol)="
          f"{np.corrcoef(arr['signal'], 1/true_vol[:arr['signal'].size])[0,1]:.2f}")

    # forward VaR made at step t is for the NEXT return -> align forecast[:-1] with r[1:]
    realized_next = arr["ret"][1:]
    for p in (0.01, 0.05):
        fc = {"var": {p: arr["var"][p][:-1]}, "es": {p: arr["es"][p][:-1]}}
        backtest_report(realized_next, fc, p, f"streaming forward VaR (known params)")
