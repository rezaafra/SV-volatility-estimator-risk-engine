# volest — real-time stochastic-volatility estimator & risk engine

An online volatility estimator and VaR/ES risk engine built on a **stochastic-volatility
state-space model** filtered with a **bootstrap particle filter**, with Bayesian parameter
estimation via **particle-marginal Metropolis–Hastings (PMMH)**, proper VaR/ES backtesting,
a streaming `estimate(prices) → (vol, VaR, signal)` API, and a containerized FastAPI service.

> **What it claims, honestly:** out-of-sample, the SV–Student-*t* engine gives *correctly
> calibrated tail coverage* where a RiskMetrics EWMA–Gaussian baseline does not — it passes
> 1% VaR coverage tests the baseline fails, and produces better-sized Expected Shortfall.
> It does **not** claim to dominate GARCH on point volatility RMSE (a famously hard bar);
> the edge is tail calibration and adaptivity, which is where SV filtering actually helps.

---

## Model

Latent log-variance follows a discretized Ornstein–Uhlenbeck / AR(1) process; the return is
observed with **unit-variance Student-*t*** innovations so the conditional variance is `exp(h_t)`:

```
h_t = mu + phi (h_{t-1} - mu) + sigma_eta * eta_t      eta_t ~ N(0, 1)
r_t = exp(h_t / 2) * eps_t                             eps_t ~ standardized-t(nu),  Var = 1
```

- `phi` is the vol persistence; the **mean-reversion half-life** is `ln 2 / (-ln phi)`.
- The **Student-*t*** observation is what lets the risk engine cover fat tails — Gaussian
  innovations systematically under-cover 1% VaR.
- The **leverage effect** (correlated `eps_t`, `eta_{t+1}`) is deferred to v2; it is weak/ambiguous
  in crypto and matters mainly for equities.

The latent state is non-linear and non-Gaussian in the returns, so it is filtered with a
bootstrap particle filter rather than a Kalman filter. The PF likelihood path never *samples*
the *t* distribution (it only propagates Gaussian state noise and *evaluates* the *t* density),
which is why it compiles cleanly under numba.

---

## Repository layout

| file | role |
|------|------|
| `sv_filter.py`        | particle filter, simulator, marginal log-likelihood |
| `sv_filter_numba.py`  | numba-compiled likelihood for fast PMMH |
| `pmmh.py`             | PMMH sampler (unconstrained reparam, adaptive proposal, pluggable likelihood) |
| `pmmh_parallel.py`    | parallel multi-chain PMMH + Gelman–Rubin R-hat |
| `risk_engine.py`      | predictive VaR/ES + Kupiec / Christoffersen / Acerbi–Székely backtests |
| `online.py`           | streaming `StreamingVolEstimator` + `estimate()` + vol-target signal |
| `data.py`             | Binance hourly fetch, cleaning, train/test fit + out-of-sample backtest |
| `app.py` / `Dockerfile` | FastAPI service, containerized |

---

## Results (synthetic validation)

All numbers below are reproducible by running the module `__main__` blocks. They are on
**synthetic data**, where the data-generating process is known — this validates the *machinery*.
Real-market numbers are produced by `data.py` on your own network (see Quickstart) and should
be the only numbers ever quoted as performance.

**Filter recovery** (`sv_filter.py`) — known parameters, 4000 steps:
- correlation of filtered conditional variance with truth ≈ **0.71** (single returns are a noisy
  observation of instantaneous variance — this is near the information-theoretic ceiling, not a bug);
- the marginal log-likelihood surface peaks at the true `phi`, confirming it can drive estimation.

**Parameter estimation** (`pmmh_parallel.py`) — 3 dispersed chains, Gelman–Rubin R-hat:

| param | truth | post. mean | R-hat |
|-------|-------|-----------|-------|
| `mu`        | −9.0 | −9.47 | 1.01 ✓ |
| `phi`       | 0.98 | 0.973 | 1.02 ✓ |
| `sigma_eta` | 0.15 | 0.223 | 1.00 ✓ |
| `nu`        | 6.0  | 7.06  | 1.10 — **not converged** |

R-hat correctly flags `nu` (tail thickness) as the hardest parameter to identify; it is informed
only by rare large moves. The fix is mechanical (more iterations / particles concentrated on `nu`).

**Risk backtest, known parameters** (`risk_engine.py`) — 6000 fat-tailed (`nu`=6) returns:

| forecaster | level | breach rate | Kupiec | Acerbi–Székely Z2 |
|------------|-------|-------------|--------|-------------------|
| SV–PF (*t*)        | 1% | 0.95% | PASS | +0.01 (ES OK) |
| EWMA–Gaussian      | 1% | **2.15%** | **FAIL** | −1.53 (ES underestimates) |
| SV–PF (*t*)        | 5% | 5.05% | PASS | −0.04 (ES OK) |
| EWMA–Gaussian      | 5% | 5.05% | PASS | −0.20 (ES underestimates) |

The 5% rows are the subtle point: both pass the *frequency* (VaR) test, yet the ES backtest still
catches EWMA–Gaussian under-sizing tail losses — which is exactly why ES is backtested separately.

**Out-of-sample, estimated parameters** (`data.py`) — synthetic hourly, train 1500 / test 1000:

| forecaster | level | breach rate | Kupiec | Z2 |
|------------|-------|-------------|--------|----|
| SV–PF (*t*) OOS   | 1% | 1.20% | PASS | −0.19 |
| EWMA–Gaussian OOS | 1% | **2.10%** | **FAIL** (p=0.002) | −1.54 |

With *estimated* parameters the edge is real but modest (as it should be): SV–PF passes 1% coverage
where EWMA fails, with consistently less-biased ES. SV–PF's own ES is slightly low because `nu` was
estimated at 7.25 vs a true 5.0 — overestimated degrees of freedom ⇒ thinner fitted tails ⇒ ES too
small. The miscalibration traces directly to the `nu` uncertainty above.

**numba crossover** (`sv_filter_numba.py`) — numba wins only at moderate particle counts, because
numpy already vectorizes over the particle dimension:

| N | numpy | numba | speedup |
|---|-------|-------|---------|
| 128 | 187 ms | 40 ms | 4.6x |
| 512 | 273 ms | 158 ms | 1.7x |
| 1000 | 385 ms | 308 ms | 1.3x |
| 2000 | 589 ms | 616 ms | 1.0x |

Combined with parallel chains (≈min(chains, cores)×), a fit that took ~15 min single-threaded
runs in a few minutes *and* returns convergence diagnostics.

---

## Quickstart

```bash
pip install -r requirements.txt

python sv_filter.py        # filter recovery on synthetic data
python risk_engine.py      # SV-t vs EWMA-Gaussian backtest
python online.py           # streaming estimator + forward-VaR backtest
python pmmh_parallel.py    # multi-chain PMMH + R-hat
python data.py             # full pipeline: fetch -> fit -> out-of-sample
```

`data.py` pulls real BTCUSDT hourly bars from Binance where network access allows, and falls
back to a synthetic series otherwise. Run it on a machine with open network access to produce
real-market results.

### Service

```bash
docker build -t volest .
docker run -p 8000:8000 volest
# interactive Swagger UI at http://localhost:8000/docs
```

```bash
SID=$(curl -s -XPOST localhost:8000/sessions -H 'content-type: application/json' \
  -d '{"params":{"mu":-9,"phi":0.98,"sigma_eta":0.15,"nu":6},"ann":8760}' | jq -r .session_id)

curl -s -XPOST localhost:8000/sessions/$SID/tick -d '{"price":42000}' -H 'content-type: application/json'
curl -s -XPOST localhost:8000/sessions/$SID/tick -d '{"price":42150}' -H 'content-type: application/json'
# -> {"vol":..., "vol_forecast":..., "var":{"0.0100":...}, "es":{...}, "signal":..., "z":...}
```

---

## Evaluation methodology

Volatility has no observable ground truth, so point-RMSE of "realized vol" is a weak, circular
metric at high frequency (the realized-vol proxy is itself a contested estimand under microstructure
noise). Evaluation therefore leads with:

- **Predictive log-likelihood** of returns — a proper scoring rule, model-agnostic, for the filter;
- **Kupiec POF** (unconditional coverage), **Christoffersen** (independence / no breach clustering),
  and their conditional-coverage combination — for VaR;
- **Acerbi–Székely Z2** — for whether ES is the right *size* on breach days.

All forecasts are one-step-ahead and strictly ex ante: parameters are estimated on the training
window only, online filtering uses only past returns, the test mean is never used (returns are
demeaned by the train mean), and only the test slice is scored.

---

## Design decisions (defensible choices)

- **OU log-variance state** — guarantees positive variance, gives an interpretable persistence /
  mean-reversion half-life, and is the standard SV parameterization.
- **Student-*t* innovations** — the load-bearing choice for the risk engine; Gaussian tails fail
  1% VaR on real returns.
- **PMMH, not "online MLE"** — joint online state-and-parameter learning in particle filters is
  degenerate (weight collapse). PMMH estimates parameters offline on a training window (with
  scheduled refits) and freezes them for online filtering — production-sane and exact.
- **Backtesting ES separately from VaR** — VaR frequency can pass while ES size fails; both are reported.
- **Multi-chain + R-hat** — convergence is demonstrated across dispersed starts, not asserted.

---

## Limitations & honest caveats

- Synthetic results validate the machinery only; real-market edges are smaller and must come from `data.py`.
- `nu` (tail thickness) is the hardest parameter to pin down; under-resolving it biases ES. Run longer
  chains / more particles to resolve it.
- GARCH is hard to beat on point volatility forecasting; the contribution here is tail calibration and
  adaptivity, not RMSE dominance.
- numba speeds up only moderate particle counts (see crossover table); use the numpy filter at large N.
- FastAPI sessions live in process memory — fine for a single instance / demo; externalize to Redis
  (pickling the particle cloud) for horizontal scaling.

---

## Roadmap

- **v2 model:** leverage effect (correlated return/vol shocks); minute bars with an explicit intraday
  volatility-seasonality layer (the open/close U-shape) so the filter does not mistake the daily cycle
  for regime shifts.
- **Sampling:** correlated PMMH (Deligiannidis et al.) to cut iterations; posterior-averaged VaR
  (Bayesian predictive) rather than plug-in point estimates.
- **Demo:** Jupyter notebook walking theory → code → live plots on real BTC data.

---

## References

- Kim, Shephard & Chib (1998), *Stochastic Volatility: Likelihood Inference and Comparison with ARCH Models*, Review of Economic Studies.
- Andrieu, Doucet & Holenstein (2010), *Particle Markov chain Monte Carlo methods*, JRSS-B.
- Pitt, Silva, Giordani & Kohn (2012), *On some properties of Markov chain Monte Carlo simulation methods based on the particle filter*, Journal of Econometrics.
- Deligiannidis, Doucet & Pitt (2018), *The correlated pseudo-marginal method*, JRSS-B.
- Kupiec (1995), *Techniques for verifying the accuracy of risk measurement models*, Journal of Derivatives.
- Christoffersen (1998), *Evaluating interval forecasts*, International Economic Review.
- Acerbi & Székely (2014), *Back-testing Expected Shortfall*, Risk.
- Haario, Saksman & Tamminen (2001), *An adaptive Metropolis algorithm*, Bernoulli.
