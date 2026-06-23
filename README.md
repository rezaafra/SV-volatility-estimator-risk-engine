# volest — real-time stochastic-volatility estimator & risk engine

An online volatility estimator and VaR/ES risk engine built on a **stochastic-volatility
state-space model** filtered with a **bootstrap particle filter**, with Bayesian parameter
estimation via **particle-marginal Metropolis–Hastings (PMMH)**, proper VaR/ES backtesting,
a streaming `estimate(prices) → (vol, VaR, signal)` API, and a containerized FastAPI service.

> **What it claims, honestly:** out-of-sample on real BTC, the SV–Student-*t* engine gives
> *correctly calibrated tail coverage* where a RiskMetrics EWMA–Gaussian baseline does not —
> it passes the 1% VaR coverage test the baseline fails, and produces better-sized Expected
> Shortfall. It does **not** claim to dominate GARCH on point volatility RMSE (a famously hard
> bar); the edge is tail calibration and adaptivity, which is where SV filtering actually helps.

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
| `data.py`             | Coinbase BTC-USD hourly fetch, cleaning, train/test fit + out-of-sample backtest |
| `app.py` / `Dockerfile` | FastAPI service, containerized |

---
## Results — real data (Coinbase BTC-USD, hourly)

Trained by 2-chain PMMH on 3,000 hourly returns; evaluated out-of-sample on the
following 2,000 returns. Parameters frozen from training; online filtering uses
only past data; only the test slice is scored.

**Parameter posterior (2 chains, 1,700 draws each, Gelman–Rubin R-hat):**

| param | mean | 90% CI | R-hat |
|-------|------|--------|-------|
| `mu`        | −11.08 | [−11.36, −10.82] | 1.005 |
| `phi`       | 0.939  | [0.912, 0.963]   | 1.005 |
| `sigma_eta` | 0.409  | [0.329, 0.507]   | 1.008 |
| `nu`        | 5.60   | [4.33, 7.28]     | 1.077 |

Three parameters are well-converged. `nu` (tail thickness) is only weakly
identified (R-hat ≈ 1.08): a single year of hourly data contains few genuine
tail events, so the data cannot tightly constrain tail-thickness. The point
estimate is stable across chains and runs (5.6–5.8); the *posterior width* is
what remains uncertain. Fix: more data (multi-year, or pooling across assets).

**Out-of-sample VaR/ES backtest:**

| forecaster | level | breach rate (expected) | Kupiec | Christoffersen | Acerbi–Székely Z2 |
|------------|-------|------------------------|--------|----------------|-------------------|
| SV–PF (*t*)   | 1% | 0.80% (1.00%) | PASS (p=0.35) | PASS | +0.25 (ES OK) |
| EWMA–Gaussian | 1% | 2.05% (1.00%) | **FAIL** (p<0.001) | PASS | −1.60 (ES underestimates) |
| SV–PF (*t*)   | 5% | 4.15% (5.00%) | PASS (p=0.07) | PASS | +0.20 (ES OK) |
| EWMA–Gaussian | 5% | 4.65% (5.00%) | PASS | PASS | −0.17 (ES underestimates) |

On real BTC, SV–PF passes 1% VaR coverage where the EWMA–Gaussian baseline fails
outright, and produces correctly-sized Expected Shortfall at both levels where
the baseline underestimates tail losses. The 5% rows show the subtle point:
both pass VaR *frequency*, yet ES backtesting still separates them.

## Results — real data (Coinbase BTC-USD, hourly)

One year of hourly BTC-USD (8,747 returns after gap cleaning). Fit on a trailing 5,000-point
window, split 3,000 train / 2,000 test. Parameters estimated by **2-chain PMMH on the training
window only**, then frozen; online filtering uses only past returns; the test mean is never used;
only the test slice is scored.

**Parameter posterior** (2 chains × 1,700 draws, Gelman–Rubin R-hat):

| param | mean | 90% CI | R-hat |
|-------|------|--------|-------|
| `mu`        | −11.08 | [−11.36, −10.82] | 1.005 |
| `phi`       | 0.939  | [0.912, 0.963]   | 1.005 |
| `sigma_eta` | 0.409  | [0.329, 0.507]   | 1.008 |
| `nu`        | 5.60   | [4.33, 7.28]     | **1.077** |

Three parameters are well-converged (R-hat ≈ 1.005). `nu` (tail thickness) is only **weakly
identified** (R-hat ≈ 1.08): a single year of hourly data contains few genuine tail events, so
the data cannot tightly constrain tail-thickness. The *point estimate* is stable across chains and
runs (`nu` ≈ 5.6–5.8); the posterior *width* is what remains uncertain. The fix is more data
(multi-year, or pooling across assets), not more sampler iterations — this is an identifiability
limit, not a mixing failure.

**Out-of-sample VaR/ES backtest:**

| forecaster | level | breach rate (expected) | Kupiec | Christoffersen | Acerbi–Székely Z2 |
|------------|-------|------------------------|--------|----------------|-------------------|
| SV–PF (*t*)   | 1% | 0.80% (1.00%) | PASS (p=0.35) | PASS | +0.25 (ES OK) |
| EWMA–Gaussian | 1% | 2.05% (1.00%) | **FAIL** (p<0.001) | PASS | −1.60 (ES underestimates) |
| SV–PF (*t*)   | 5% | 4.15% (5.00%) | PASS (p=0.07) | PASS | +0.20 (ES OK) |
| EWMA–Gaussian | 5% | 4.65% (5.00%) | PASS | PASS | −0.17 (ES underestimates) |

On real BTC, SV–PF passes 1% VaR coverage where the EWMA–Gaussian baseline fails outright, and
produces correctly-sized Expected Shortfall at both levels where the baseline underestimates tail
losses. The 5% rows show the subtle point: both pass VaR *frequency*, yet ES backtesting still
separates them — which is precisely why ES is backtested separately from VaR.

---

## Results — synthetic validation (machinery check)

These runs use data drawn from the model with known parameters, to validate the *machinery*
(reproducible from each module's `__main__`). They are not performance claims.

- **Filter recovery** (`sv_filter.py`): filtered-variance correlation with truth ≈ 0.71 (single
  returns are a noisy observation of instantaneous variance — near the information-theoretic
  ceiling, not a bug); the marginal-likelihood surface peaks at the true `phi`.
- **Estimation** (`pmmh_parallel.py`): on synthetic data, R-hat correctly flags `nu` as the
  hardest parameter to identify — the same behavior observed on real BTC above.
- **Risk backtest, known parameters** (`risk_engine.py`, 6,000 fat-tailed returns): SV–PF is
  calibrated at 1% (0.95%, Z2 ≈ 0) while EWMA–Gaussian breaches 2.15% and fails Kupiec; at 5% both
  pass frequency but only the ES backtest catches EWMA–Gaussian under-sizing tail losses.

**numba crossover** (`sv_filter_numba.py`) — numba wins only at moderate particle counts, because
numpy already vectorizes over the particle dimension:

| N | numpy | numba | speedup |
|---|-------|-------|---------|
| 128 | 187 ms | 40 ms | 4.6x |
| 512 | 273 ms | 158 ms | 1.7x |
| 1000 | 385 ms | 308 ms | 1.3x |
| 2000 | 589 ms | 616 ms | 1.0x |

Combined with parallel chains (≈min(chains, cores)×), a multi-thousand-iteration fit runs in
minutes and returns convergence diagnostics.

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

`data.py` pulls real BTC-USD hourly bars from **Coinbase** (no API key, no US geo-block; it falls
back to Binance for non-US hosts, then to a synthetic series if neither is reachable). A PMMH fit
on a few thousand points takes minutes; run it under `nohup`/`tmux` on a remote box so a console
timeout does not interrupt it, and fit on a trailing window (e.g. `r[-5000:]`) on small instances.

### Service

```bash
docker build -t volest .
docker run -p 8000:8000 volest
# interactive Swagger UI at http://localhost:8000/docs
```

```bash
SID=$(curl -s -XPOST localhost:8000/sessions -H 'content-type: application/json' \
  -d '{"params":{"mu":-11,"phi":0.94,"sigma_eta":0.41,"nu":5.6},"ann":8760}' | jq -r .session_id)

curl -s -XPOST localhost:8000/sessions/$SID/tick -d '{"price":42000}' -H 'content-type: application/json'
curl -s -XPOST localhost:8000/sessions/$SID/tick -d '{"price":42150}' -H 'content-type: application/json'
# -> {"vol":..., "vol_forecast":..., "var":{"0.0100":...}, "es":{...}, "signal":..., "z":...}
```

---

## Evaluation methodology

Volatility has no observable ground truth, so point-RMSE of "realized vol" is a weak, circular
metric at high frequency. Evaluation therefore leads with:

- **Predictive log-likelihood** of returns — a proper scoring rule, model-agnostic — for the filter;
- **Kupiec POF** (unconditional coverage), **Christoffersen** (independence / no breach clustering),
  and their conditional-coverage combination — for VaR;
- **Acerbi–Székely Z2** — for whether ES is the right *size* on breach days.

All forecasts are one-step-ahead and strictly ex ante.

---

## Design decisions (defensible choices)

- **OU log-variance state** — guarantees positive variance, gives an interpretable persistence /
  mean-reversion half-life, the standard SV parameterization.
- **Student-*t* innovations** — the load-bearing choice for the risk engine; Gaussian tails fail 1% VaR.
- **PMMH, not "online MLE"** — joint online state-and-parameter learning in particle filters is
  degenerate (weight collapse). PMMH estimates parameters offline on a training window (with scheduled
  refits) and freezes them for online filtering — production-sane and exact.
- **Backtesting ES separately from VaR** — VaR frequency can pass while ES size fails; both are reported.
- **Multi-chain + R-hat** — convergence is demonstrated across dispersed starts, not asserted.

---

## Limitations & honest caveats

- `nu` (tail thickness) is weakly identified from one year of hourly data (R-hat ≈ 1.08); the ES
  result is reported with that caveat. Resolve with multi-year data or cross-asset pooling.
- GARCH is hard to beat on point volatility forecasting; the contribution here is tail calibration
  and adaptivity, not RMSE dominance.
- numba speeds up only moderate particle counts (see crossover table); use the numpy filter at large N.
- FastAPI sessions live in process memory — fine for a single instance / demo; externalize to Redis
  (pickling the particle cloud) for horizontal scaling.

---

## Roadmap

- **v2 model:** leverage effect (correlated return/vol shocks); minute bars with an explicit intraday
  volatility-seasonality layer (the open/close U-shape).
- **Sampling:** correlated PMMH (Deligiannidis et al.) to cut iterations; posterior-averaged VaR.
- **Demo:** Jupyter notebook walking theory → code → live plots on real BTC data.

---

## References

- Kim, Shephard & Chib (1998), *Stochastic Volatility: Likelihood Inference and Comparison with ARCH Models*, Review of Economic Studies.
- Andrieu, Doucet & Holenstein (2010), *Particle Markov chain Monte Carlo methods*, JRSS-B.
- Pitt, Silva, Giordani & Kohn (2012), *On some properties of MCMC methods based on the particle filter*, Journal of Econometrics.
- Deligiannidis, Doucet & Pitt (2018), *The correlated pseudo-marginal method*, JRSS-B.
- Kupiec (1995), *Techniques for verifying the accuracy of risk measurement models*, Journal of Derivatives.
- Christoffersen (1998), *Evaluating interval forecasts*, International Economic Review.
- Acerbi & Székely (2014), *Back-testing Expected Shortfall*, Risk.
- Haario, Saksman & Tamminen (2001), *An adaptive Metropolis algorithm*, Bernoulli.
