"""
FastAPI service exposing the streaming SV volatility / VaR / signal estimator.

The estimator is STATEFUL (an online particle filter), so the service is
session-oriented: create a session, then stream ticks into it. Each tick
returns current vol, one-step-ahead forward VaR/ES, and a vol-targeted signal.

Endpoints
---------
  GET  /health
  POST /sessions                 -> {session_id}        (configure a filter)
  POST /sessions/{id}/tick       -> Estimate            (feed one price or return)
  POST /sessions/{id}/batch      -> {n_estimates, last} (feed many prices)
  GET  /sessions/{id}            -> session status
  DELETE /sessions/{id}
  POST /backtest                 -> per-level VaR/ES backtest on supplied prices

Run:  uvicorn app:app --reload      (or via the Dockerfile)

Sessions live in-memory (fine for a demo / single instance). For horizontal
scaling you'd externalize state to Redis and pickle the particle cloud.
"""

from __future__ import annotations

import threading
import time
import uuid

import numpy as np
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from sv_filter import SVParams
from online import StreamingVolEstimator
from risk_engine import (
    kupiec_pof,
    christoffersen_independence,
    acerbi_szekely_z2,
)

app = FastAPI(
    title="volest",
    version="0.1.0",
    description="Real-time stochastic-volatility estimator & risk engine.",
)


# --------------------------------------------------------------------------- #
# Schemas
# --------------------------------------------------------------------------- #
class ParamsIn(BaseModel):
    mu: float = -9.0
    phi: float = 0.97
    sigma_eta: float = 0.2
    nu: float = 6.0

    def to_params(self) -> SVParams:
        return SVParams(self.mu, self.phi, self.sigma_eta, self.nu)


class SessionConfig(BaseModel):
    params: ParamsIn = Field(default_factory=ParamsIn)
    n_particles: int = 1000
    n_forward: int = 40
    levels: list[float] = Field(default_factory=lambda: [0.01, 0.05])
    target_vol: float | None = None
    seed: int | None = None
    ann: float = 8760.0          # hourly -> annualized vol


class TickIn(BaseModel):
    price: float | None = None
    ret: float | None = None     # supply a return directly instead of a price


class BatchIn(BaseModel):
    prices: list[float]


class BacktestIn(BaseModel):
    prices: list[float]
    params: ParamsIn = Field(default_factory=ParamsIn)
    levels: list[float] = Field(default_factory=lambda: [0.01, 0.05])
    n_particles: int = 1000
    seed: int | None = 0


# --------------------------------------------------------------------------- #
# Session registry
# --------------------------------------------------------------------------- #
class Session:
    def __init__(self, cfg: SessionConfig):
        self.est = StreamingVolEstimator(
            cfg.params.to_params(),
            n_particles=cfg.n_particles,
            n_forward=cfg.n_forward,
            levels=tuple(cfg.levels),
            target_vol=cfg.target_vol,
            seed=cfg.seed,
        )
        self.ann = cfg.ann
        self.created = time.time()
        self.last: dict | None = None
        self.lock = threading.Lock()


_sessions: dict[str, Session] = {}
_registry_lock = threading.Lock()


def _get(session_id: str) -> Session:
    s = _sessions.get(session_id)
    if s is None:
        raise HTTPException(status_code=404, detail="unknown session_id")
    return s


# --------------------------------------------------------------------------- #
# Routes
# --------------------------------------------------------------------------- #
@app.get("/health")
def health():
    return {"status": "ok", "sessions": len(_sessions)}


@app.post("/sessions")
def create_session(cfg: SessionConfig):
    sid = uuid.uuid4().hex[:12]
    with _registry_lock:
        _sessions[sid] = Session(cfg)
    return {"session_id": sid}


@app.post("/sessions/{session_id}/tick")
def tick(session_id: str, t: TickIn):
    s = _get(session_id)
    if (t.price is None) == (t.ret is None):
        raise HTTPException(status_code=422, detail="supply exactly one of price, ret")
    with s.lock:
        if t.ret is not None:
            est = s.est.update_return(t.ret)
        else:
            est = s.est.update(t.price)
        if est is None:                          # first price -> warmup
            return {"status": "warmup", "detail": "first price stored; send another"}
        s.last = est.as_dict(ann=s.ann)
        return s.last


@app.post("/sessions/{session_id}/batch")
def batch(session_id: str, b: BatchIn):
    s = _get(session_id)
    with s.lock:
        n = 0
        for px in b.prices:
            e = s.est.update(px)
            if e is not None:
                s.last = e.as_dict(ann=s.ann)
                n += 1
        return {"n_estimates": n, "last": s.last}


@app.get("/sessions/{session_id}")
def status(session_id: str):
    s = _get(session_id)
    return {"n_obs": s.est.n, "created": s.created, "last": s.last}


@app.delete("/sessions/{session_id}")
def drop(session_id: str):
    with _registry_lock:
        _sessions.pop(session_id, None)
    return {"deleted": session_id}


@app.post("/backtest")
def backtest(b: BacktestIn):
    """One-shot: stream supplied prices through a fresh filter and score the
    one-step-ahead forward VaR/ES at each level."""
    est = StreamingVolEstimator(
        b.params.to_params(), n_particles=b.n_particles,
        levels=tuple(b.levels), seed=b.seed,
    )
    rets, var, es = [], {p: [] for p in b.levels}, {p: [] for p in b.levels}
    for px in b.prices:
        e = est.update(px)
        if e is None:
            continue
        rets.append(e.ret)
        for p in b.levels:
            var[p].append(e.var[p])
            es[p].append(e.es[p])
    rets = np.asarray(rets)
    if rets.size < 30:
        raise HTTPException(status_code=422, detail="need >= ~30 returns to backtest")

    # forecast made at t is for return t+1
    realized = rets[1:]
    out = {"n_forecasts": int(realized.size), "levels": {}}
    for p in b.levels:
        v = np.asarray(var[p][:-1])
        e_ = np.asarray(es[p][:-1])
        breaches = realized < (-v)
        kp = kupiec_pof(breaches, p)
        ci = christoffersen_independence(breaches)
        z2 = acerbi_szekely_z2(realized, v, e_, p)
        out["levels"][f"{p:.4f}"] = {
            "breaches": kp["breaches"],
            "breach_rate": kp["rate"],
            "expected": p,
            "kupiec_p": kp["pvalue"],
            "christoffersen_p": ci["pvalue"],
            "as_z2": z2,
            "pass_coverage": bool(kp["pvalue"] > 0.05),
            "es_ok": bool(z2 > -0.10),
        }
    return out


@app.get("/")
def root():
    return {
        "service": "volest",
        "docs": "/docs",
        "endpoints": ["/health", "/sessions", "/sessions/{id}/tick",
                      "/sessions/{id}/batch", "/backtest"],
    }
