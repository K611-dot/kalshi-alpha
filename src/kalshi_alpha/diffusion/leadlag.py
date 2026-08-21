"""Lead-lag between an event contract and its underlying.

If the S&P prints a move at 14:00:00.100 and the "S&P above 5000" contract only
reprices at 14:00:02.400, that 2.3-second gap is the diffusion latency, and it
is the single most tradeable number in this package.

Measuring it correctly is harder than it looks, because the two series are
**asynchronous**: the underlying ticks many times a second while the contract
may sit unchanged for a minute. Sampling both onto a common grid to compute a
correlation induces the Epps effect -- measured correlation collapses toward
zero as the sampling interval shrinks, purely as an artefact of
non-synchronous observation, and the bias grows exactly as you zoom into the
horizon you care about.

:func:`hayashi_yoshida` avoids the grid entirely. It sums the products of
returns over *overlapping* time intervals only, and is unbiased for the
integrated covariance without any synchronisation step. Shifting one clock and
re-estimating traces out a lead-lag curve whose peak is the propagation delay.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


def cross_correlation(x, y, max_lag: int = 20) -> tuple[np.ndarray, np.ndarray]:
    """Cross-correlation of two evenly-sampled return series.

    Returns ``(lags, corr)`` where a **positive** lag means ``x`` leads ``y``:
    ``corr[k]`` is ``corr(x_t, y_{t+k})``.
    """
    a = np.asarray(x, dtype=float)
    b = np.asarray(y, dtype=float)
    n = min(a.size, b.size)
    a, b = a[:n], b[:n]
    mask = np.isfinite(a) & np.isfinite(b)
    a, b = a[mask], b[mask]
    if a.size < max_lag + 5:
        return np.arange(-max_lag, max_lag + 1), np.full(2 * max_lag + 1, np.nan)

    a = a - a.mean()
    b = b - b.mean()
    denom = np.sqrt(float(a @ a) * float(b @ b))
    if denom <= 0:
        return np.arange(-max_lag, max_lag + 1), np.full(2 * max_lag + 1, np.nan)

    lags = np.arange(-max_lag, max_lag + 1)
    out = np.empty(lags.size)
    for i, k in enumerate(lags):
        if k >= 0:
            out[i] = float(a[: a.size - k] @ b[k:]) / denom
        else:
            out[i] = float(a[-k:] @ b[: b.size + k]) / denom
    return lags, out


def lead_lag_ratio(x, y, max_lag: int = 20) -> dict[str, float]:
    """Summarise a cross-correlation function into a single directional number.

    ``LLR = sum(corr at positive lags^2) / sum(corr at negative lags^2)``.
    Values above 1 mean ``x`` leads ``y``. Squaring makes the measure
    sign-agnostic, so a lead shows up whether the relationship is positive or
    inverse -- which matters when the contract is a "below" strike.
    """
    lags, corr = cross_correlation(x, y, max_lag)
    finite = np.isfinite(corr)
    if not finite.any():
        return {"llr": np.nan, "peak_lag": np.nan, "peak_corr": np.nan}
    pos = float(np.nansum(corr[finite & (lags > 0)] ** 2))
    neg = float(np.nansum(corr[finite & (lags < 0)] ** 2))
    peak = int(np.nanargmax(np.abs(np.where(finite, corr, np.nan))))
    return {
        "llr": pos / neg if neg > 0 else np.inf,
        "peak_lag": float(lags[peak]),
        "peak_corr": float(corr[peak]),
    }


def hayashi_yoshida(
    t1: np.ndarray, p1: np.ndarray, t2: np.ndarray, p2: np.ndarray, shift_s: float = 0.0
) -> float:
    """Hayashi-Yoshida covariance estimator for asynchronous series.

    Sums ``dp1_i * dp2_j`` over every pair of return intervals that overlap in
    time. ``shift_s`` moves the second clock forward, so a positive shift that
    maximises the estimate means series 2 *lags* series 1 by that much.

    The two-pointer sweep is O(n + m): both interval lists are sorted, so once
    an interval from series 2 ends before the current interval from series 1
    begins, it can never overlap any later one either.
    """
    t1 = np.asarray(t1, dtype=float)
    t2 = np.asarray(t2, dtype=float) + float(shift_s)
    p1 = np.asarray(p1, dtype=float)
    p2 = np.asarray(p2, dtype=float)
    if t1.size < 2 or t2.size < 2:
        return np.nan

    d1 = np.diff(p1)
    d2 = np.diff(p2)
    a0, a1 = t1[:-1], t1[1:]
    b0, b1 = t2[:-1], t2[1:]

    total = 0.0
    j_start = 0
    for i in range(d1.size):
        if not np.isfinite(d1[i]):
            continue
        while j_start < d2.size and b1[j_start] <= a0[i]:
            j_start += 1
        j = j_start
        while j < d2.size and b0[j] < a1[i]:
            if np.isfinite(d2[j]):
                total += d1[i] * d2[j]
            j += 1
    return float(total)


def hy_correlation(t1, p1, t2, p2, shift_s: float = 0.0) -> float:
    """Hayashi-Yoshida covariance normalised by each series' realised variance."""
    cov = hayashi_yoshida(t1, p1, t2, p2, shift_s)
    v1 = float(np.nansum(np.diff(np.asarray(p1, dtype=float)) ** 2))
    v2 = float(np.nansum(np.diff(np.asarray(p2, dtype=float)) ** 2))
    denom = np.sqrt(v1 * v2)
    return cov / denom if denom > 0 else np.nan


@dataclass(frozen=True)
class LeadLagEstimate:
    delay_s: float
    peak_corr: float
    shifts: np.ndarray
    curve: np.ndarray

    @property
    def leader(self) -> str:
        if not np.isfinite(self.delay_s):
            return "undetermined"
        if abs(self.delay_s) < 1e-9:
            return "simultaneous"
        return "series_1" if self.delay_s > 0 else "series_2"

    def summary(self) -> str:
        return (
            f"lead-lag: {self.leader} leads by {abs(self.delay_s):.2f}s "
            f"(HY corr at peak = {self.peak_corr:+.3f})"
        )


def estimate_delay(
    t1, p1, t2, p2, max_shift_s: float = 30.0, step_s: float = 0.5
) -> LeadLagEstimate:
    """Scan the clock shift that maximises Hayashi-Yoshida correlation.

    The returned ``delay_s`` is how far series 2 trails series 1 in seconds; a
    negative value means series 2 is in front.
    """
    shifts = np.arange(-max_shift_s, max_shift_s + step_s, step_s)
    curve = np.array([hy_correlation(t1, p1, t2, p2, float(s)) for s in shifts])
    if not np.isfinite(curve).any():
        return LeadLagEstimate(np.nan, np.nan, shifts, curve)
    best = int(np.nanargmax(curve))
    return LeadLagEstimate(float(shifts[best]), float(curve[best]), shifts, curve)


def epps_curve(t1, p1, t2, p2, intervals=(1.0, 2.0, 5.0, 15.0, 30.0, 60.0)) -> dict[float, float]:
    """Measured correlation as a function of sampling interval.

    A curve that rises steeply with the interval is the Epps effect and is
    evidence that any synchronised-grid correlation at fine resolution is
    understated. Compare against the grid-free Hayashi-Yoshida number to see
    how much bias the grid was introducing.
    """
    t1 = np.asarray(t1, dtype=float)
    t2 = np.asarray(t2, dtype=float)
    p1 = np.asarray(p1, dtype=float)
    p2 = np.asarray(p2, dtype=float)
    if t1.size < 2 or t2.size < 2:
        return dict.fromkeys(intervals, np.nan)

    lo = max(t1[0], t2[0])
    hi = min(t1[-1], t2[-1])
    out: dict[float, float] = {}
    for dt in intervals:
        if hi - lo < 5 * dt:
            out[dt] = np.nan
            continue
        grid = np.arange(lo, hi, dt)
        s1 = np.interp(grid, t1, p1)
        s2 = np.interp(grid, t2, p2)
        r1, r2 = np.diff(s1), np.diff(s2)
        if r1.size < 5 or np.allclose(r1, 0) or np.allclose(r2, 0):
            out[dt] = np.nan
            continue
        out[dt] = float(np.corrcoef(r1, r2)[0, 1])
    return out
