"""Event study on scheduled releases.

Scheduled events -- CPI, non-farm payrolls, an FOMC statement, an earnings
print -- are the cleanest natural experiment available in finance. The timing
is known in advance to the second, so any price move afterwards is
unambiguously attributable to the release rather than to something you failed
to control for. That removes the hardest problem in ordinary event studies:
you never have to guess when the information arrived.

The pipeline:

1. **Align.** Resample each event's path onto a common event-relative grid so
   paths of different tick densities are comparable.
2. **Abnormalise.** Subtract the drift estimated on the pre-event window from
   the same market. Prediction-market prices have no market beta to strip out,
   but they do have their own drift as the deadline approaches, and it is
   exactly the horizon we are measuring over.
3. **Aggregate.** Average the cumulative abnormal revisions across events into
   a CAAR path with cross-sectional standard errors.
4. **Falsify.** Re-run the whole thing on placebo timestamps drawn from
   non-event periods. With a handful of events and fat-tailed returns the
   parametric t-statistic is not credible; the placebo distribution gives an
   empirical p-value that makes no distributional assumption at all.

Step 4 is the one that keeps the result honest. It is very easy to produce an
impressive-looking CAAR chart from eight events and pure noise.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field

import numpy as np
import pandas as pd


@dataclass(frozen=True, slots=True)
class ScheduledEvent:
    """One release: when it happened and what it was."""

    ts: float
    name: str
    category: str = ""
    surprise: float | None = None  # actual minus consensus, if known


def align_event_windows(
    times: np.ndarray,
    prices: np.ndarray,
    event_ts: Sequence[float] | np.ndarray,
    pre_s: float = 900.0,
    post_s: float = 3600.0,
    bar_s: float = 5.0,
) -> tuple[np.ndarray, np.ndarray]:
    """Resample every event window onto one event-relative grid.

    Returns ``(grid, matrix)`` with ``matrix[i, j]`` the price of event ``i``
    at offset ``grid[j]`` seconds. Sampling is *previous-tick* interpolation --
    the last price at or before the grid point -- because forward-filling would
    leak information backwards across the event boundary and manufacture
    exactly the instant repricing we are trying to measure.
    """
    t = np.asarray(times, dtype=float)
    p = np.asarray(prices, dtype=float)
    mask = np.isfinite(t) & np.isfinite(p)
    t, p = t[mask], p[mask]
    grid = np.arange(-pre_s, post_s + bar_s, bar_s)
    if t.size < 2 or not len(event_ts):
        return grid, np.full((0, grid.size), np.nan)

    rows: list[np.ndarray] = []
    for ets in event_ts:
        want = ets + grid
        if want[0] < t[0] or want[-1] > t[-1]:
            continue
        idx = np.searchsorted(t, want, side="right") - 1
        idx = np.clip(idx, 0, t.size - 1)
        rows.append(p[idx])
    matrix = np.vstack(rows) if rows else np.full((0, grid.size), np.nan)
    return grid, matrix


@dataclass
class EventStudyResult:
    grid: np.ndarray
    caar: np.ndarray  # cumulative average abnormal revision, in cents
    stderr: np.ndarray
    tstat: np.ndarray
    n_events: int
    car_terminal: float = np.nan
    tstat_terminal: float = np.nan
    placebo_p: float = np.nan
    placebo_draws: int = 0
    per_event_car: np.ndarray = field(default_factory=lambda: np.zeros(0))

    @property
    def significant(self) -> bool:
        return bool(np.isfinite(self.placebo_p) and self.placebo_p < 0.05)

    def to_frame(self) -> pd.DataFrame:
        return pd.DataFrame(
            {"t": self.grid, "caar": self.caar, "stderr": self.stderr, "tstat": self.tstat}
        )

    def summary(self) -> str:
        return (
            f"event study over {self.n_events} events\n"
            f"  terminal CAAR = {self.car_terminal:+.2f}c  "
            f"(cross-sectional t={self.tstat_terminal:+.2f})\n"
            f"  placebo p-value = {self.placebo_p:.4f} over {self.placebo_draws} draws -> "
            f"{'significant' if self.significant else 'not distinguishable from noise'}"
        )


def _abnormal_paths(grid: np.ndarray, matrix: np.ndarray) -> np.ndarray:
    """Convert price paths to cumulative abnormal revisions, in cents.

    Each path is re-based to the **last pre-event price**, then the pre-event
    drift rate measured on the same path is projected forward and removed.

    Basing on the last observation strictly *before* the release, rather than
    on the price at the release instant, is essential and easy to get wrong: on
    a scheduled release most of the repricing happens in the first tick, so
    basing at ``t = 0`` subtracts out precisely the jump the study exists to
    measure and reports a large, real reaction as approximately zero.
    """
    if matrix.size == 0:
        return matrix

    pre_indices = np.flatnonzero(grid < 0)
    base_idx = int(pre_indices[-1]) if pre_indices.size else 0
    base = matrix[:, base_idx][:, None]
    raw = matrix - base

    pre_mask = grid < 0
    out = np.empty_like(raw)
    for i in range(raw.shape[0]):
        pre_vals = raw[i, pre_mask]
        pre_t = grid[pre_mask]
        good = np.isfinite(pre_vals)
        if good.sum() >= 3 and np.ptp(pre_t[good]) > 0:
            slope = np.polyfit(pre_t[good], pre_vals[good], 1)[0]
        else:
            slope = 0.0
        out[i] = raw[i] - slope * grid
    return out


def event_study(
    times: np.ndarray,
    prices: np.ndarray,
    event_ts: Sequence[float] | np.ndarray,
    pre_s: float = 900.0,
    post_s: float = 3600.0,
    bar_s: float = 5.0,
    placebo_draws: int = 500,
    seed: int = 0,
) -> EventStudyResult:
    """Full event study with a placebo-based empirical p-value."""
    grid, matrix = align_event_windows(times, prices, event_ts, pre_s, post_s, bar_s)
    n_events = matrix.shape[0]
    if n_events == 0:
        return EventStudyResult(grid, np.full(grid.size, np.nan), np.full(grid.size, np.nan),
                                np.full(grid.size, np.nan), 0)

    car = _abnormal_paths(grid, matrix)
    caar = np.nanmean(car, axis=0)
    sd = np.nanstd(car, axis=0, ddof=1) if n_events > 1 else np.zeros(grid.size)
    se = sd / np.sqrt(n_events)
    with np.errstate(divide="ignore", invalid="ignore"):
        tstat = np.where(se > 0, caar / se, np.nan)

    terminal = float(caar[-1])
    t_terminal = float(tstat[-1])
    per_event = car[:, -1]

    p_val, draws = _placebo_pvalue(
        times, prices, event_ts, pre_s, post_s, bar_s, terminal, placebo_draws, seed
    )

    return EventStudyResult(
        grid=grid,
        caar=caar,
        stderr=se,
        tstat=tstat,
        n_events=n_events,
        car_terminal=terminal,
        tstat_terminal=t_terminal,
        placebo_p=p_val,
        placebo_draws=draws,
        per_event_car=per_event,
    )


def _placebo_pvalue(
    times: np.ndarray,
    prices: np.ndarray,
    event_ts: Sequence[float] | np.ndarray,
    pre_s: float,
    post_s: float,
    bar_s: float,
    observed: float,
    draws: int,
    seed: int,
    exclusion_s: float | None = None,
) -> tuple[float, int]:
    """Empirical two-sided p-value from randomly-timed pseudo-events.

    Pseudo-event times are drawn away from every real event by ``exclusion_s``
    so the null distribution is not contaminated by the effect being tested.

    The exclusion radius defaults to one event window and is then shrunk if the
    sample is too short to supply enough clean candidates. A fixed radius wide
    enough for a month of data silently yields *zero* draws on a single session,
    and a p-value computed from zero draws is not a conservative result -- it is
    a missing one.
    """
    t = np.asarray(times, dtype=float)
    if t.size < 10 or draws <= 0 or not np.isfinite(observed):
        return np.nan, 0

    rng = np.random.default_rng(seed)
    lo, hi = t[0] + pre_s, t[-1] - post_s
    if hi <= lo:
        return np.nan, 0

    real = np.asarray(list(event_ts), dtype=float)
    n_events = max(1, len(event_ts))
    span = hi - lo
    if exclusion_s is None:
        exclusion_s = min(pre_s + post_s, span / (2.0 * max(real.size, 1)))

    stats_out: list[float] = []
    attempts = 0
    while len(stats_out) < draws and attempts < draws * 10:
        attempts += 1
        cand = rng.uniform(lo, hi, size=n_events)
        if real.size and np.any(np.abs(cand[:, None] - real[None, :]) < exclusion_s):
            continue
        grid, mat = align_event_windows(t, prices, cand, pre_s, post_s, bar_s)
        if mat.shape[0] == 0:
            continue
        car = _abnormal_paths(grid, mat)
        stats_out.append(float(np.nanmean(car[:, -1])))

    if not stats_out:
        return np.nan, 0
    null = np.abs(np.asarray(stats_out))
    p = float((np.sum(null >= abs(observed)) + 1) / (null.size + 1))
    return p, int(null.size)


def surprise_regression(
    result: EventStudyResult, events: Sequence[ScheduledEvent]
) -> dict[str, float]:
    """Regress each event's terminal CAR on the size of its surprise.

    A slope significantly different from zero is the strongest evidence that
    the measured move is information rather than noise: bigger surprises should
    move the contract more, and the intercept should be near zero.
    """
    surprises = [e.surprise for e in events if e.surprise is not None]
    if len(surprises) != result.per_event_car.size or len(surprises) < 4:
        return {"slope": np.nan, "tstat": np.nan, "r2": np.nan, "n": float(len(surprises))}

    from kalshi_alpha.microstructure.impact import ols_slope

    reg = ols_slope(np.asarray(surprises, dtype=float), result.per_event_car)
    return {"slope": reg.beta, "tstat": reg.tstat, "r2": reg.r2, "n": float(reg.n)}
