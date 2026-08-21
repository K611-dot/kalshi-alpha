"""How long does it take for news to finish being priced?

Around a scheduled release the price makes a jump and then keeps drifting for a
while. The size of the jump is not the interesting number -- the *shape of the
tail* is. Three complementary readings, because each fails differently:

* **Adjustment profile** (:func:`adjustment_profile`). Non-parametric. Rescale
  the path so the pre-event price is 0 and the settled post-event price is 1,
  then read off when it first crosses 50% and 90%. Assumes nothing about
  functional form; needs a trustworthy terminal price.
* **Exponential decay** (:func:`exponential_half_life`). Regress
  ``log|p_t - p_inf|`` on time. Smooth and gives a single interpretable
  ``tau``; badly behaved if the gap crosses zero, which is why the fit is
  weighted and the :math:`R^2` is always reported alongside.
* **AR(1) on the gap** (:func:`ar1_half_life`). Discrete-time cousin of the
  above, robust to noise around zero and directly comparable across sampling
  frequencies once converted to seconds.

Agreement between the three is the evidence that the number means something.
Disagreement usually means the terminal price was taken too early, and
:func:`terminal_price_stability` exists to check exactly that before any of the
estimates are trusted.

The reason to care in trading terms: a half-life materially longer than the
round-trip fee hurdle is post-event drift you can actually capture, and
:class:`~kalshi_alpha.backtest.strategies.DriftStrategy` sizes its holding
period straight off ``t90``.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

EPS = 1e-9


@dataclass(frozen=True, slots=True)
class HalfLifeFit:
    method: str
    half_life_s: float
    tau_s: float
    r2: float
    n: int
    phi: float = np.nan

    @property
    def valid(self) -> bool:
        return bool(np.isfinite(self.half_life_s) and self.half_life_s > 0)

    def summary(self) -> str:
        return (
            f"{self.method:14s} half-life={self.half_life_s:8.1f}s  "
            f"tau={self.tau_s:8.1f}s  R2={self.r2:5.3f}  n={self.n}"
        )


def exponential_half_life(
    times_s: np.ndarray, gap: np.ndarray, min_gap: float = 1e-3
) -> HalfLifeFit:
    """Fit ``|gap_t| = A exp(-t / tau)`` by weighted log-linear regression.

    Observations are weighted by the gap itself. Without that weighting the
    tail -- where the gap is pure measurement noise around zero -- dominates
    the log-space fit and biases ``tau`` upward, which would make every market
    look slower than it is.
    """
    t = np.asarray(times_s, dtype=float)
    g = np.abs(np.asarray(gap, dtype=float))
    mask = np.isfinite(t) & np.isfinite(g) & (g > min_gap)
    t, g = t[mask], g[mask]
    if t.size < 5:
        return HalfLifeFit("exponential", np.nan, np.nan, np.nan, int(t.size))

    y = np.log(g)
    w = g / g.sum()
    X = np.column_stack([np.ones(t.size), t])
    W = np.diag(w)
    try:
        beta = np.linalg.solve(X.T @ W @ X, X.T @ W @ y)
    except np.linalg.LinAlgError:
        return HalfLifeFit("exponential", np.nan, np.nan, np.nan, int(t.size))

    slope = float(beta[1])
    resid = y - X @ beta
    ss_tot = float(np.sum(w * (y - np.average(y, weights=w)) ** 2))
    r2 = 1.0 - float(np.sum(w * resid**2)) / ss_tot if ss_tot > 0 else np.nan

    if slope >= 0:  # gap widening: no decay to measure
        return HalfLifeFit("exponential", np.nan, np.nan, r2, int(t.size))
    tau = -1.0 / slope
    return HalfLifeFit("exponential", float(tau * np.log(2.0)), float(tau), r2, int(t.size))


def ar1_half_life(gap: np.ndarray, dt_s: float = 1.0) -> HalfLifeFit:
    """Fit ``gap_t = phi * gap_{t-1} + e`` and convert ``phi`` to a half-life.

    No intercept: the gap is defined relative to the terminal price, so its
    unconditional mean is zero by construction and fitting a constant would
    absorb part of the decay.
    """
    g = np.asarray(gap, dtype=float)
    g = g[np.isfinite(g)]
    if g.size < 10:
        return HalfLifeFit("ar1", np.nan, np.nan, np.nan, int(g.size))

    x, y = g[:-1], g[1:]
    denom = float(x @ x)
    if denom <= 0:
        return HalfLifeFit("ar1", np.nan, np.nan, np.nan, int(g.size))
    phi = float((x @ y) / denom)
    resid = y - phi * x
    ss_tot = float(np.sum((y - y.mean()) ** 2))
    r2 = 1.0 - float(resid @ resid) / ss_tot if ss_tot > 0 else np.nan

    if not 0.0 < phi < 1.0:
        return HalfLifeFit("ar1", np.nan, np.nan, r2, int(g.size), phi)
    hl_bars = np.log(0.5) / np.log(phi)
    tau_bars = -1.0 / np.log(phi)
    return HalfLifeFit(
        "ar1", float(hl_bars * dt_s), float(tau_bars * dt_s), r2, int(g.size), phi
    )


def adjustment_profile(
    times_s: np.ndarray,
    prices: np.ndarray,
    pre_price: float | None = None,
    terminal_price: float | None = None,
    tail_frac: float = 0.2,
) -> pd.DataFrame:
    """Rescale a post-event path to the fraction of the total move completed.

    ``phi(t) = (p_t - p_pre) / (p_inf - p_pre)`` runs from 0 at the event to 1
    once the move is fully priced. The terminal price defaults to the mean of
    the last ``tail_frac`` of the window rather than the final tick, so a single
    noisy print at the end cannot rescale the entire profile.
    """
    t = np.asarray(times_s, dtype=float)
    p = np.asarray(prices, dtype=float)
    mask = np.isfinite(t) & np.isfinite(p)
    t, p = t[mask], p[mask]
    if t.size < 3:
        return pd.DataFrame(columns=["t", "price", "phi"])

    p0 = float(p[0]) if pre_price is None else float(pre_price)
    if terminal_price is None:
        k = max(1, int(len(p) * tail_frac))
        p_inf = float(np.mean(p[-k:]))
    else:
        p_inf = float(terminal_price)

    move = p_inf - p0
    phi = (p - p0) / move if abs(move) > EPS else np.full_like(p, np.nan)
    return pd.DataFrame({"t": t, "price": p, "phi": phi, "gap": p_inf - p})


def crossing_times(profile: pd.DataFrame, levels: tuple[float, ...] = (0.5, 0.9)) -> dict[str, float]:
    """First time the adjustment profile reaches each level, by linear interpolation."""
    out: dict[str, float] = {}
    if profile.empty or profile["phi"].isna().all():
        return {f"t{int(lv * 100)}": np.nan for lv in levels}
    t = profile["t"].to_numpy()
    phi = profile["phi"].to_numpy()
    for lv in levels:
        key = f"t{int(lv * 100)}"
        idx = np.argmax(phi >= lv) if (phi >= lv).any() else -1
        if idx <= 0:
            out[key] = float(t[0]) if idx == 0 else np.nan
            continue
        # interpolate between the straddling samples
        x0, x1 = phi[idx - 1], phi[idx]
        t0, t1 = t[idx - 1], t[idx]
        out[key] = float(t0 + (lv - x0) / (x1 - x0) * (t1 - t0)) if x1 != x0 else float(t1)
    return out


def terminal_price_stability(prices: np.ndarray, tail_frac: float = 0.25) -> dict[str, float]:
    """Diagnostic: has the price actually settled by the end of the window?

    Compares the mean of the final ``tail_frac`` with the mean of the segment
    before it. A large gap relative to the tail's own noise means the window is
    too short and every half-life estimate built on it is biased *downward* --
    the drift has not finished, so the estimator only sees the fast part.
    """
    p = np.asarray(prices, dtype=float)
    p = p[np.isfinite(p)]
    k = max(2, int(p.size * tail_frac))
    if p.size < 2 * k:
        return {"drift": np.nan, "tail_sd": np.nan, "z": np.nan, "settled": float("nan")}
    tail = p[-k:]
    prev = p[-2 * k : -k]
    drift = float(tail.mean() - prev.mean())
    sd = float(tail.std(ddof=1))
    z = drift / (sd / np.sqrt(k)) if sd > 0 else np.nan
    return {"drift": drift, "tail_sd": sd, "z": z, "settled": float(abs(z) < 2.0)}


def information_lag_half_life(
    follower: np.ndarray, leader: np.ndarray, dt_s: float = 1.0
) -> HalfLifeFit:
    """Half-life of the gap between a lagging series and a leading one.

    **This is the estimator to prefer.** Measuring adjustment against a fixed
    terminal price assumes the efficient price stops moving, which it does not:
    after the release it resumes being a martingale, and over a long window that
    random walk swamps the decay and drags the estimated half-life upward
    without bound.

    The gap between two cointegrated series has no such problem. It is
    stationary by construction -- arbitrage bounds it -- so its AR(1)
    coefficient measures information transmission speed cleanly, and the result
    is insensitive to how long a window you use.

    ``leader`` can be another venue, the underlying asset mapped to probability
    space, or a model's fair value; all that is required is that the two series
    settle on the same event.
    """
    f = np.asarray(follower, dtype=float)
    lead = np.asarray(leader, dtype=float)
    n = min(f.size, lead.size)
    gap = f[:n] - lead[:n]
    fit = ar1_half_life(gap, dt_s)
    return HalfLifeFit("spread_ar1", fit.half_life_s, fit.tau_s, fit.r2, fit.n, fit.phi)


def window_scan(
    times_s: np.ndarray,
    prices: np.ndarray,
    dt_s: float,
    min_bars: int = 40,
    max_bars: int | None = None,
    growth: float = 1.5,
) -> pd.DataFrame:
    """Estimated half-life as a function of how much data you feed it.

    Diagnostic, and the input to :func:`select_window`. Two regimes are visible
    in the output and they are easy to tell apart:

    * **Plateau.** Once the window covers several half-lives the estimate stops
      changing, because the remaining gap is negligible. This is the answer.
    * **Runaway.** Beyond that, the efficient price's own random walk starts to
      dominate the residual and the fitted AR(1) coefficient drifts toward one,
      so the estimate grows roughly linearly in the window with no upper bound.

    Reporting a single number without looking at this curve is how a
    post-release half-life of two minutes gets published as forty.
    """
    t = np.asarray(times_s, dtype=float)
    p = np.asarray(prices, dtype=float)
    n = min(t.size, p.size)
    cap = n if max_bars is None else min(n, max_bars)
    if cap < min_bars:
        return pd.DataFrame(columns=["bars", "seconds", "half_life_s", "r2", "phi"])

    candidates: list[int] = []
    w = float(min_bars)
    while int(w) <= cap:
        candidates.append(int(w))
        w = w * growth + 1
    if not candidates or candidates[-1] != cap:
        candidates.append(cap)

    rows = []
    for width in candidates:
        prof = adjustment_profile(t[:width], p[:width])
        if prof.empty:
            continue
        fit = ar1_half_life(prof["gap"].to_numpy(), dt_s)
        rows.append(
            {
                "bars": width,
                "seconds": width * dt_s,
                "half_life_s": fit.half_life_s if fit.valid else np.nan,
                "r2": fit.r2,
                "phi": fit.phi,
            }
        )
    return pd.DataFrame(rows)


@dataclass(frozen=True, slots=True)
class WindowChoice:
    bars: int
    seconds: float
    reliable: bool
    rel_change: float
    coverage: float  # window length in units of the estimated half-life

    def summary(self) -> str:
        flag = "plateau found" if self.reliable else "NO PLATEAU - estimate unidentified"
        return (
            f"window={self.seconds:.0f}s ({self.bars} bars)  "
            f"coverage={self.coverage:.1f} half-lives  "
            f"drift={self.rel_change:.1%}  {flag}"
        )


def select_spread_window(
    gap: np.ndarray,
    dt_s: float,
    multiple: float = 10.0,
    min_bars: int = 40,
    passes: int = 5,
) -> int:
    """Window for the spread estimator, by fixed-point iteration.

    The spread is stationary, so -- unlike the single-series case -- widening
    the window does not inflate the estimate and the iteration converges
    instead of running away. Estimate, resize to ``multiple`` half-lives,
    repeat. A window much longer than that adds only noise, because the gap has
    already decayed to nothing.
    """
    g = np.asarray(gap, dtype=float)
    n = g.size
    if n < min_bars:
        return n
    win = min(n, max(min_bars, int(round(300.0 / max(dt_s, 1e-9)))))
    for _ in range(passes):
        fit = ar1_half_life(g[:win], dt_s)
        if not fit.valid:
            return win
        target = int(np.clip(multiple * fit.half_life_s / dt_s, min_bars, n))
        if abs(target - win) <= max(2, win // 20):
            return target
        win = target
    return int(np.clip(win, min_bars, n))


def select_window(
    times_s: np.ndarray,
    prices: np.ndarray,
    dt_s: float,
    min_bars: int = 40,
    min_half_lives: float = 6.0,
    max_bars: int | None = None,
) -> WindowChoice:
    """Pick the estimation window at the plateau of :func:`window_scan`.

    Among candidate windows that span at least ``min_half_lives`` of their own
    estimate -- so a window too short to contain the decay cannot win by being
    trivially self-consistent -- choose the one whose estimate changes least
    when the window is widened. That knee sits between "not enough data" and
    "contaminated by the random walk", and locating it needs no prior knowledge
    of the half-life being measured.

    When no candidate qualifies, the half-life is simply **not identified from
    this series alone**: the decay is slow enough relative to residual
    volatility that no window separates the two. The result is returned with
    ``reliable=False`` rather than dressed up as an answer, and the caller
    should fall back on :func:`information_lag_half_life`, which does not have
    this failure mode.
    """
    scan = window_scan(times_s, prices, dt_s, min_bars, max_bars)
    n = min(len(times_s), len(prices))
    fallback_bars = int(min(n, max(min_bars, n // 4)))
    if scan.empty or scan["half_life_s"].notna().sum() < 2:
        return WindowChoice(fallback_bars, fallback_bars * dt_s, False, np.nan, np.nan)

    bars = scan["bars"].to_numpy()
    hl = scan["half_life_s"].to_numpy()

    best_idx, best_rel, best_cov = None, np.inf, np.nan
    for i in range(len(bars) - 1):
        a, b = hl[i], hl[i + 1]
        if not (np.isfinite(a) and np.isfinite(b) and a > 0):
            continue
        coverage = bars[i] * dt_s / a
        if coverage < min_half_lives:
            continue
        rel = abs(b - a) / a
        if rel < best_rel:
            best_rel, best_idx, best_cov = rel, i, coverage

    if best_idx is not None:
        w = int(bars[best_idx])
        return WindowChoice(w, w * dt_s, True, float(best_rel), float(best_cov))

    # No plateau: report the widest window that at least covers a few
    # half-lives, and mark the whole estimate as unidentified.
    coverages = np.array(
        [bars[i] * dt_s / hl[i] if np.isfinite(hl[i]) and hl[i] > 0 else np.nan
         for i in range(len(bars))]
    )
    if np.isfinite(coverages).any():
        pick = int(np.nanargmax(coverages))
        w = int(bars[pick])
        return WindowChoice(w, w * dt_s, False, np.nan, float(coverages[pick]))
    return WindowChoice(fallback_bars, fallback_bars * dt_s, False, np.nan, np.nan)


def impulse_response(
    prices: np.ndarray, event_index: int, horizon: int, pre: int = 10
) -> np.ndarray:
    """Path of cumulative price change relative to the pre-event level."""
    p = np.asarray(prices, dtype=float)
    lo = max(0, event_index - pre)
    hi = min(p.size, event_index + horizon + 1)
    if hi - lo < 2:
        return np.zeros(0)
    base = float(np.nanmean(p[lo:event_index])) if event_index > lo else float(p[lo])
    return p[event_index:hi] - base


@dataclass
class HalfLifeResult:
    """Everything the ensemble produced, typed.

    A dict of heterogeneous values would be one line shorter and would silently
    accept ``result["consnesus_half_life_s"]`` forever.
    """

    consensus_half_life_s: float
    consensus_source: str
    identified: bool
    fits: list[HalfLifeFit] = field(default_factory=list)
    crossings: dict[str, float] = field(default_factory=dict)
    stability: dict[str, float] = field(default_factory=dict)
    profile: pd.DataFrame = field(default_factory=pd.DataFrame)
    dt_s: float = 1.0
    window: WindowChoice | None = None
    window_bars: int = 0
    window_s: float = 0.0
    spread_window_s: float = 0.0

    @property
    def valid(self) -> bool:
        return bool(np.isfinite(self.consensus_half_life_s) and self.consensus_half_life_s > 0)

    def summary(self) -> str:
        lines = [
            f"consensus half-life = {self.consensus_half_life_s:.1f}s "
            f"(source: {self.consensus_source}, "
            f"{'identified' if self.identified else 'NOT identified'})"
        ]
        lines.extend("  " + f.summary() for f in self.fits)
        if self.window is not None:
            lines.append("  " + self.window.summary())
        return "\n".join(lines)


def half_life_ensemble(
    times_s: np.ndarray,
    prices: np.ndarray,
    dt_s: float | None = None,
    terminal_price: float | None = None,
    leader: np.ndarray | None = None,
    auto_window: bool = True,
    window_bars: int | None = None,
) -> HalfLifeResult:
    """Run every estimator plus the diagnostics and return them together.

    Pass ``leader`` -- another venue, or the underlying mapped into probability
    space -- whenever one is available. The spread-based estimator it enables is
    the only one immune to the efficient price continuing to move after the
    release, and when it is present it is weighted into the consensus twice.

    Without a leader the window matters enormously, so ``auto_window`` sizes it
    at roughly twelve half-lives by iteration rather than leaving it to whatever
    range the caller happened to slice.

    The consensus is the median of the estimators that returned a finite,
    positive value -- median rather than mean because the exponential fit
    occasionally diverges when the gap changes sign, and one bad draw should not
    move the answer.
    """
    t_all = np.asarray(times_s, dtype=float)
    p_all = np.asarray(prices, dtype=float)
    if t_all.size < 3:
        return HalfLifeResult(np.nan, "insufficient_data", False)

    if dt_s is None:
        diffs = np.diff(t_all)
        dt_s = float(np.median(diffs)) if diffs.size else 1.0

    if window_bars is not None:
        choice = WindowChoice(int(np.clip(window_bars, 3, t_all.size)),
                              float(window_bars * dt_s), True, np.nan, np.nan)
    elif auto_window:
        choice = select_window(t_all, p_all, dt_s)
    else:
        choice = WindowChoice(t_all.size, float(t_all.size * dt_s), False, np.nan, np.nan)

    win = choice.bars
    t, p = t_all[:win], p_all[:win]
    profile = adjustment_profile(t, p, terminal_price=terminal_price)
    if profile.empty:
        return HalfLifeResult(np.nan, "insufficient_data", False)

    fits = [
        exponential_half_life(profile["t"].to_numpy(), profile["gap"].to_numpy()),
        ar1_half_life(profile["gap"].to_numpy(), dt_s),
    ]

    spread_fit: HalfLifeFit | None = None
    spread_window = win
    if leader is not None:
        # The spread gets its own window. It is stationary, so the plateau
        # logic used for the single-series estimators -- which exists purely to
        # bound random-walk contamination -- would only hand it a window far
        # longer than it needs and add noise.
        lead_full = np.asarray(leader, dtype=float)[: p_all.size]
        gap_full = p_all[: lead_full.size] - lead_full
        spread_window = select_spread_window(gap_full, dt_s)
        spread_fit = information_lag_half_life(
            p_all[:spread_window], lead_full[:spread_window], dt_s
        )
        fits.append(spread_fit)

    # The spread estimator wins outright when it is available. It is the only
    # one that does not assume the efficient price stops moving after the
    # release, so averaging it with the single-series estimates would import
    # their bias for no gain. The others are still reported for comparison.
    if spread_fit is not None and spread_fit.valid:
        consensus = spread_fit.half_life_s
        source = "spread_ar1"
    else:
        single = [f.half_life_s for f in fits if f.valid]
        consensus = float(np.median(single)) if single else np.nan
        source = "single_series_median"

    return HalfLifeResult(
        consensus_half_life_s=consensus,
        consensus_source=source,
        identified=bool(choice.reliable or (spread_fit is not None and spread_fit.valid)),
        fits=fits,
        crossings=crossing_times(profile),
        stability=terminal_price_stability(profile["price"].to_numpy()),
        profile=profile,
        dt_s=dt_s,
        window=choice,
        window_bars=int(win),
        window_s=float(win * dt_s),
        spread_window_s=float(spread_window * dt_s),
    )
