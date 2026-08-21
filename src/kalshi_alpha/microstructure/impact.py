"""Liquidity and price-impact estimators.

These answer the question every sizing decision depends on: *what does it cost
to move this market, and how much of that cost is permanent?*

* **Kyle's lambda** -- the slope of price on signed order flow. The permanent,
  information-bearing component of impact. High lambda means the book is thin
  relative to informed flow, so an arbitrage that needs size will not survive
  its own execution.
* **Amihud illiquidity** -- absolute return per dollar traded. Cruder than
  lambda but robust on sparse tapes, which prediction markets usually have.
* **Roll's implied spread** -- backs the effective spread out of the negative
  serial covariance of transaction-price changes, using no quote data at all.
  A useful cross-check when the book feed and the tape disagree.
* **VPIN** -- volume-synchronised probability of informed trading. Spikes ahead
  of scheduled releases; used here as the trigger to widen quotes and stand
  down the passive strategies before an event.
* **Effective vs realised spread** -- decomposes the round-trip cost actually
  paid into the part the market maker keeps (realised) and the part lost to
  adverse selection (impact). Their difference is the true cost of being run
  over by informed flow.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass(frozen=True, slots=True)
class Regression:
    """Minimal OLS result with heteroskedasticity-consistent (HC0) errors."""

    beta: float
    stderr: float
    tstat: float
    r2: float
    n: int
    intercept: float = 0.0

    @property
    def significant(self) -> bool:
        return abs(self.tstat) > 1.96


def ols_slope(x: np.ndarray, y: np.ndarray, add_intercept: bool = True) -> Regression:
    """Univariate OLS with White standard errors.

    Robust errors matter here: order-flow data is violently heteroskedastic,
    and classical standard errors would overstate significance in exactly the
    high-volume regime we care about.
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    mask = np.isfinite(x) & np.isfinite(y)
    x, y = x[mask], y[mask]
    n = x.size
    if n < 3 or np.allclose(x, x[0]):
        return Regression(np.nan, np.nan, np.nan, np.nan, n)

    X = np.column_stack([np.ones(n), x]) if add_intercept else x.reshape(-1, 1)
    xtx_inv = np.linalg.pinv(X.T @ X)
    beta_vec = xtx_inv @ X.T @ y
    resid = y - X @ beta_vec

    # HC0: (X'X)^-1 X' diag(e^2) X (X'X)^-1
    meat = X.T @ (X * (resid**2)[:, None])
    cov = xtx_inv @ meat @ xtx_inv
    slope_idx = 1 if add_intercept else 0
    slope = float(beta_vec[slope_idx])
    se = float(np.sqrt(max(cov[slope_idx, slope_idx], 0.0)))

    ss_tot = float(np.sum((y - y.mean()) ** 2))
    r2 = 1.0 - float(np.sum(resid**2)) / ss_tot if ss_tot > 0 else np.nan
    return Regression(
        beta=slope,
        stderr=se,
        tstat=slope / se if se > 0 else np.nan,
        r2=r2,
        n=n,
        intercept=float(beta_vec[0]) if add_intercept else 0.0,
    )


def kyle_lambda(price_changes: Sequence[float], signed_volume: Sequence[float]) -> Regression:
    """Estimate lambda in ``dP = lambda * Q + e``.

    Units are cents of permanent price move per contract of net signed flow.
    """
    return ols_slope(np.asarray(signed_volume, dtype=float),
                     np.asarray(price_changes, dtype=float))


def amihud_illiquidity(returns: Sequence[float], notional: Sequence[float]) -> float:
    """Mean of ``|return| / notional``; higher means thinner.

    Intervals with no volume are dropped rather than treated as infinitely
    illiquid, which would let a single quiet minute dominate the average.
    """
    r = np.abs(np.asarray(returns, dtype=float))
    v = np.asarray(notional, dtype=float)
    mask = np.isfinite(r) & np.isfinite(v) & (v > 0)
    return float(np.mean(r[mask] / v[mask])) if mask.any() else np.nan


def roll_spread(prices: Sequence[float]) -> float:
    """Roll (1984) effective spread implied by first-order serial covariance.

    ``c = 2 * sqrt(-cov(dP_t, dP_{t-1}))``. Returns NaN when the covariance is
    positive, which is Roll's model failing rather than a zero spread -- most
    often because trending information flow swamps bid-ask bounce.
    """
    p = np.asarray(prices, dtype=float)
    dp = np.diff(p[np.isfinite(p)])
    if dp.size < 3:
        return np.nan
    cov = float(np.cov(dp[:-1], dp[1:])[0, 1])
    return 2.0 * float(np.sqrt(-cov)) if cov < 0 else np.nan


def vpin(
    signed_volume: Sequence[float],
    volume: Sequence[float],
    bucket_size: float | None = None,
    n_buckets: int = 50,
) -> float:
    """Volume-synchronised probability of informed trading.

    Trades are bucketed into equal-*volume* bins rather than equal-time bins,
    which is the whole point: information arrives on a volume clock, and a
    calendar clock badly under-samples exactly the bursts that matter.
    """
    q = np.asarray(signed_volume, dtype=float)
    v = np.asarray(volume, dtype=float)
    mask = np.isfinite(q) & np.isfinite(v) & (v > 0)
    q, v = q[mask], v[mask]
    if v.size == 0:
        return np.nan

    total = float(v.sum())
    if bucket_size is None:
        bucket_size = total / max(n_buckets, 1)
    if bucket_size <= 0:
        return np.nan

    imbalances: list[float] = []
    acc_v = 0.0
    acc_q = 0.0
    for qi, vi in zip(q, v, strict=True):
        acc_v += vi
        acc_q += qi
        while acc_v >= bucket_size:
            # Attribute the imbalance proportionally to the completed bucket.
            frac = bucket_size / acc_v
            imbalances.append(abs(acc_q * frac) / bucket_size)
            acc_q *= 1.0 - frac
            acc_v -= bucket_size
    return float(np.mean(imbalances)) if imbalances else np.nan


def effective_spread(trade_price: float, mid_before: float, taker_bought: bool) -> float:
    """Twice the signed distance from the prevailing mid: what the taker paid."""
    sign = 1.0 if taker_bought else -1.0
    return 2.0 * sign * (trade_price - mid_before)


def realized_spread(trade_price: float, mid_after: float, taker_bought: bool) -> float:
    """Twice the signed distance to the *post-trade* mid: what the maker kept."""
    sign = 1.0 if taker_bought else -1.0
    return 2.0 * sign * (trade_price - mid_after)


def price_impact(trade_price: float, mid_before: float, mid_after: float,
                 taker_bought: bool) -> float:
    """Permanent component = effective - realised: the adverse-selection cost."""
    return effective_spread(trade_price, mid_before, taker_bought) - realized_spread(
        trade_price, mid_after, taker_bought
    )


def spread_decomposition(
    trades: pd.DataFrame, mids: pd.DataFrame, horizon_s: float = 60.0
) -> pd.DataFrame:
    """Effective / realised / impact for every print on the tape.

    ``trades`` needs ``ts``, ``price``, ``signed_size``; ``mids`` needs ``ts``
    and ``mid``. The post-trade mid is taken ``horizon_s`` seconds later, which
    is the standard convention and long enough on this asset class for the
    information content of a print to be absorbed.
    """
    if trades.empty or mids.empty:
        return pd.DataFrame(columns=["ts", "effective", "realized", "impact"])

    m = mids[["ts", "mid"]].dropna().sort_values("ts")
    t = trades.sort_values("ts")

    before = pd.merge_asof(t, m, on="ts", direction="backward")
    after_ref = t.copy()
    after_ref["ts"] = after_ref["ts"] + horizon_s
    after = pd.merge_asof(after_ref, m, on="ts", direction="backward", suffixes=("", "_after"))

    bought = t["signed_size"].to_numpy() > 0
    mid_b = before["mid"].to_numpy()
    mid_a = after["mid"].to_numpy()
    px = t["price"].to_numpy()
    sign = np.where(bought, 1.0, -1.0)

    eff = 2.0 * sign * (px - mid_b)
    real = 2.0 * sign * (px - mid_a)
    return pd.DataFrame(
        {"ts": t["ts"].to_numpy(), "effective": eff, "realized": real, "impact": eff - real}
    )


def liquidity_report(features: pd.DataFrame, trades: pd.DataFrame) -> dict[str, float]:
    """One-line liquidity summary used by the CLI and the HTML report."""
    out: dict[str, float] = {}
    if not features.empty:
        out["mean_spread_cents"] = float(features["spread"].mean(skipna=True))
        out["median_spread_cents"] = float(features["spread"].median(skipna=True))
        out["mean_depth_cents"] = float(features["notional_depth"].mean(skipna=True))
        out["roll_spread_cents"] = roll_spread(features["mid"].to_numpy())
    if not trades.empty and not features.empty:
        merged = pd.merge_asof(
            trades.sort_values("ts"),
            features[["ts", "mid"]].dropna().sort_values("ts"),
            on="ts",
            direction="backward",
        )
        dmid = merged["mid"].diff().to_numpy()
        reg = kyle_lambda(dmid[1:], merged["signed_size"].to_numpy()[1:])
        out["kyle_lambda_cents_per_contract"] = reg.beta
        out["kyle_lambda_t"] = reg.tstat
        out["kyle_r2"] = reg.r2
        out["amihud"] = amihud_illiquidity(dmid[1:], merged["notional"].to_numpy()[1:])
        out["vpin"] = vpin(merged["signed_size"].to_numpy(), merged["size"].to_numpy())
    return out
