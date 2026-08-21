"""Performance measurement, including the statistics that expose overfitting.

A Sharpe ratio from a backtest is not an estimate of future performance -- it is
the **maximum** of however many Sharpe ratios were computed along the way, and
the maximum of many noisy draws is biased upward by a lot. Search twenty
parameter settings on pure noise and the best one reliably looks good.

Two corrections are reported alongside the headline numbers and both should be
read before the headline:

* **Deflated Sharpe ratio** (Bailey and Lopez de Prado, 2014). Given the number
  of configurations actually tried, it returns the probability that the true
  Sharpe is positive, adjusting for selection bias, for return skew, and for
  fat tails. A strategy with SR 2.0 found on the fiftieth attempt over a short
  sample can easily deflate to a coin flip.
* **Stationary bootstrap confidence interval.** Resampling in blocks preserves
  serial dependence, so the interval reflects the fact that trades cluster.
  An i.i.d. bootstrap would produce a spuriously narrow band.

``n_trials`` must be the honest count of every variant evaluated, not the count
of variants reported. Understating it is how the deflation gets defeated.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from scipy import stats

EULER_MASCHERONI = 0.5772156649015329


def sharpe_ratio(returns, periods_per_year: float = 252.0, rf: float = 0.0) -> float:
    """Annualised Sharpe ratio of a per-period return series."""
    r = np.asarray(returns, dtype=float)
    r = r[np.isfinite(r)] - rf / periods_per_year
    if r.size < 2:
        return np.nan
    sd = float(r.std(ddof=1))
    return float(r.mean() / sd * np.sqrt(periods_per_year)) if sd > 0 else np.nan


def sortino_ratio(returns, periods_per_year: float = 252.0, target: float = 0.0) -> float:
    """Like Sharpe, but penalising only downside deviation."""
    r = np.asarray(returns, dtype=float)
    r = r[np.isfinite(r)]
    if r.size < 2:
        return np.nan
    downside = r[r < target] - target
    dd = float(np.sqrt(np.mean(downside**2))) if downside.size else 0.0
    return float((r.mean() - target) / dd * np.sqrt(periods_per_year)) if dd > 0 else np.inf


def max_drawdown(equity) -> tuple[float, int, int]:
    """Largest peak-to-trough decline; returns ``(depth, peak_idx, trough_idx)``."""
    e = np.asarray(equity, dtype=float)
    if e.size == 0:
        return np.nan, -1, -1
    running_max = np.maximum.accumulate(e)
    dd = e - running_max
    trough = int(np.argmin(dd))
    peak = int(np.argmax(e[: trough + 1])) if trough > 0 else 0
    return float(dd[trough]), peak, trough


def calmar_ratio(equity, periods_per_year: float = 252.0) -> float:
    e = np.asarray(equity, dtype=float)
    if e.size < 2:
        return np.nan
    depth, _, _ = max_drawdown(e)
    total = e[-1] - e[0]
    years = e.size / periods_per_year
    ann = total / years if years > 0 else np.nan
    return float(ann / abs(depth)) if depth < 0 else np.inf


def probabilistic_sharpe_ratio(returns, benchmark_sr: float = 0.0,
                               periods_per_year: float = 252.0) -> float:
    """P(true Sharpe > benchmark), correcting for skew and kurtosis.

    Non-normal returns break the classical Sharpe standard error. Negative skew
    and fat tails -- both endemic to arbitrage strategies, which win small and
    often and lose large and rarely -- inflate the naive statistic, and this
    corrects for exactly that.
    """
    r = np.asarray(returns, dtype=float)
    r = r[np.isfinite(r)]
    n = r.size
    if n < 3:
        return np.nan
    sr = sharpe_ratio(r, periods_per_year)
    if not np.isfinite(sr):
        return np.nan
    sr_p = sr / np.sqrt(periods_per_year)  # per-period
    bench_p = benchmark_sr / np.sqrt(periods_per_year)
    skew = float(stats.skew(r))
    kurt = float(stats.kurtosis(r, fisher=False))
    denom = np.sqrt(max(1.0 - skew * sr_p + (kurt - 1.0) / 4.0 * sr_p**2, 1e-12))
    z = (sr_p - bench_p) * np.sqrt(n - 1) / denom
    return float(stats.norm.cdf(z))


def expected_max_sharpe(n_trials: int, sharpe_variance: float) -> float:
    """Sharpe you expect from the *best* of ``n_trials`` strategies with zero edge.

    The benchmark the deflated Sharpe measures against. It grows like
    ``sqrt(2 log N)``, so trying a hundred variants raises the bar substantially
    even before any of them has an edge.
    """
    if n_trials < 2 or sharpe_variance <= 0:
        return 0.0
    g = EULER_MASCHERONI
    z1 = stats.norm.ppf(1.0 - 1.0 / n_trials)
    z2 = stats.norm.ppf(1.0 - 1.0 / (n_trials * np.e))
    return float(np.sqrt(sharpe_variance) * ((1.0 - g) * z1 + g * z2))


def deflated_sharpe_ratio(
    returns,
    n_trials: int = 1,
    sharpe_variance: float | None = None,
    periods_per_year: float = 252.0,
) -> float:
    """Probability the true Sharpe is positive, after correcting for selection.

    ``sharpe_variance`` is the variance of the Sharpe ratios across the trials
    that were run. When it is unknown, the conservative fallback assumes each
    trial's Sharpe has the sampling variance implied by the sample length,
    which understates the spread of a real parameter sweep and therefore makes
    this an *optimistic* bound. Supply the real number when you have it.
    """
    r = np.asarray(returns, dtype=float)
    r = r[np.isfinite(r)]
    if r.size < 3:
        return np.nan
    if sharpe_variance is None:
        sharpe_variance = 1.0 / max(r.size - 1, 1)
    bench = expected_max_sharpe(n_trials, sharpe_variance) * np.sqrt(periods_per_year)
    return probabilistic_sharpe_ratio(r, bench, periods_per_year)


def stationary_bootstrap_sharpe(
    returns,
    draws: int = 1_000,
    mean_block: float = 20.0,
    periods_per_year: float = 252.0,
    seed: int = 0,
) -> dict[str, float]:
    """Bootstrap confidence interval for the Sharpe ratio.

    Politis-Romano stationary bootstrap: block lengths are geometric, so the
    resampled series is stationary and the serial dependence in the original is
    preserved on average. Fixed-length blocks would leave artefacts at the
    joins.
    """
    r = np.asarray(returns, dtype=float)
    r = r[np.isfinite(r)]
    n = r.size
    if n < 20:
        return {"mean": np.nan, "ci_low": np.nan, "ci_high": np.nan, "p_positive": np.nan}

    rng = np.random.default_rng(seed)
    p = 1.0 / max(mean_block, 1.0)
    out = np.empty(draws)
    for d in range(draws):
        idx = np.empty(n, dtype=int)
        i = int(rng.integers(0, n))
        for k in range(n):
            idx[k] = i
            i = int(rng.integers(0, n)) if rng.random() < p else (i + 1) % n
        out[d] = sharpe_ratio(r[idx], periods_per_year)
    finite = out[np.isfinite(out)]
    if finite.size == 0:
        return {"mean": np.nan, "ci_low": np.nan, "ci_high": np.nan, "p_positive": np.nan}
    return {
        "mean": float(finite.mean()),
        "ci_low": float(np.percentile(finite, 2.5)),
        "ci_high": float(np.percentile(finite, 97.5)),
        "p_positive": float((finite > 0).mean()),
    }


@dataclass
class PerformanceStats:
    """Everything the report needs about one equity curve. Cents throughout."""

    total_pnl: float = np.nan
    gross_pnl: float = np.nan
    fees: float = np.nan
    n_periods: int = 0
    n_trades: int = 0
    n_contracts: int = 0
    sharpe: float = np.nan
    sortino: float = np.nan
    calmar: float = np.nan
    max_drawdown: float = np.nan
    hit_rate: float = np.nan
    profit_factor: float = np.nan
    avg_win: float = np.nan
    avg_loss: float = np.nan
    turnover_cents: float = np.nan
    fee_drag: float = np.nan
    psr: float = np.nan
    dsr: float = np.nan
    bootstrap: dict[str, float] = field(default_factory=dict)

    def to_dict(self) -> dict[str, float]:
        d = {k: v for k, v in self.__dict__.items() if k != "bootstrap"}
        d.update({f"boot_{k}": v for k, v in self.bootstrap.items()})
        return d

    def summary(self) -> str:
        return (
            f"P&L {self.total_pnl / 100:>10,.2f} USD  "
            f"(gross {self.gross_pnl / 100:,.2f}, fees {self.fees / 100:,.2f} = "
            f"{self.fee_drag:.1%} of gross)\n"
            f"  Sharpe {self.sharpe:6.2f}   Sortino {self.sortino:6.2f}   "
            f"Calmar {self.calmar:6.2f}   maxDD {self.max_drawdown / 100:,.2f}\n"
            f"  trades {self.n_trades:6d}   contracts {self.n_contracts:7d}   "
            f"hit {self.hit_rate:5.1%}   profit factor {self.profit_factor:5.2f}\n"
            f"  PSR {self.psr:5.3f}   deflated SR {self.dsr:5.3f}   "
            f"bootstrap SR 95% CI "
            f"[{self.bootstrap.get('ci_low', float('nan')):.2f}, "
            f"{self.bootstrap.get('ci_high', float('nan')):.2f}]"
        )


def performance(
    equity_cents,
    fills: pd.DataFrame | None = None,
    periods_per_year: float = 252.0,
    n_trials: int = 1,
    bootstrap_draws: int = 500,
) -> PerformanceStats:
    """Compute the full statistics block from an equity curve and a fill log."""
    e = np.asarray(equity_cents, dtype=float)
    e = e[np.isfinite(e)]
    stats_out = PerformanceStats(n_periods=int(e.size))
    if e.size < 3:
        return stats_out

    pnl = np.diff(e)
    stats_out.total_pnl = float(e[-1] - e[0])
    stats_out.sharpe = sharpe_ratio(pnl, periods_per_year)
    stats_out.sortino = sortino_ratio(pnl, periods_per_year)
    stats_out.calmar = calmar_ratio(e, periods_per_year)
    stats_out.max_drawdown = max_drawdown(e)[0]

    wins = pnl[pnl > 0]
    losses = pnl[pnl < 0]
    active = pnl[pnl != 0]
    stats_out.hit_rate = float(wins.size / active.size) if active.size else np.nan
    stats_out.avg_win = float(wins.mean()) if wins.size else 0.0
    stats_out.avg_loss = float(losses.mean()) if losses.size else 0.0
    gross_win = float(wins.sum())
    gross_loss = float(-losses.sum())
    stats_out.profit_factor = gross_win / gross_loss if gross_loss > 0 else np.inf

    stats_out.psr = probabilistic_sharpe_ratio(pnl, 0.0, periods_per_year)
    stats_out.dsr = deflated_sharpe_ratio(pnl, n_trials, None, periods_per_year)
    if bootstrap_draws > 0:
        stats_out.bootstrap = stationary_bootstrap_sharpe(
            pnl, bootstrap_draws, 20.0, periods_per_year
        )

    if fills is not None and not fills.empty:
        stats_out.n_trades = int(len(fills))
        stats_out.n_contracts = int(fills["qty"].sum())
        stats_out.fees = float(fills["fee_cents"].sum())
        stats_out.turnover_cents = float((fills["qty"] * fills["price"]).sum())
        stats_out.gross_pnl = stats_out.total_pnl + stats_out.fees
        stats_out.fee_drag = (
            stats_out.fees / stats_out.gross_pnl if stats_out.gross_pnl > 0 else np.nan
        )
    return stats_out


def walk_forward_splits(n: int, n_folds: int = 4, min_train: int = 100) -> list[tuple[range, range]]:
    """Expanding-window train/test splits, in time order.

    Anchored rather than rolling: every test fold is evaluated on a model fitted
    only on data that preceded it, which is the only split that does not leak.
    K-fold cross-validation on time series does leak, badly, and is the second
    most common way a backtest lies.
    """
    if n < min_train + n_folds:
        return []
    fold = (n - min_train) // n_folds
    if fold < 1:
        return []
    return [
        (range(0, min_train + i * fold), range(min_train + i * fold,
                                                min(min_train + (i + 1) * fold, n)))
        for i in range(n_folds)
    ]
