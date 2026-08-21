"""Probability scoring and calibration.

A prediction market quote *is* a probability forecast, so the natural way to
ask "is this market efficient?" is to score it the way you would score a
forecaster. Two questions matter and they are different:

* **Calibration** -- when the market says 30c, does it happen 30% of the time?
  A miscalibrated market is a standing, systematic bet.
* **Resolution** -- does the market discriminate at all, or does it quote
  everything near the base rate? A perfectly calibrated but useless forecaster
  quotes the unconditional frequency every time.

Murphy's decomposition separates the two exactly:

.. math::

    BS = \\underbrace{REL}_{\\text{miscalibration}}
       - \\underbrace{RES}_{\\text{discrimination}}
       + \\underbrace{UNC}_{\\text{irreducible}}

Only ``REL`` is tradeable. ``RES`` is skill you cannot capture and ``UNC`` is
the variance of the world. The favourite-longshot bias familiar from betting
markets shows up here as a specific, exploitable shape in the reliability
curve, and :func:`isotonic_fit` recovers the monotone mapping that corrects it
without imposing a functional form.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from numpy.typing import ArrayLike

EPS = 1e-12

# Every scoring function accepts lists, tuples, pandas Series and numpy arrays
# alike; the boundary converts once with np.asarray rather than forcing callers
# to marshal their data into a particular container.
Floats = ArrayLike


def _clean(p: Floats, y: Floats) -> tuple[np.ndarray, np.ndarray]:
    p_arr = np.asarray(p, dtype=float)
    y_arr = np.asarray(y, dtype=float)
    if p_arr.shape != y_arr.shape:
        raise ValueError("forecasts and outcomes must have the same shape")
    mask = np.isfinite(p_arr) & np.isfinite(y_arr)
    return p_arr[mask], y_arr[mask]


def brier_score(p: Floats, y: Floats) -> float:
    """Mean squared error of a probability forecast. Lower is better."""
    p_arr, y_arr = _clean(p, y)
    return float(np.mean((p_arr - y_arr) ** 2)) if p_arr.size else np.nan


def log_loss(p: Floats, y: Floats, clip: float = 1e-6) -> float:
    """Negative log-likelihood, clipped to keep a single 0/1 miss from exploding."""
    p_arr, y_arr = _clean(p, y)
    if not p_arr.size:
        return np.nan
    q = np.clip(p_arr, clip, 1.0 - clip)
    return float(-np.mean(y_arr * np.log(q) + (1.0 - y_arr) * np.log(1.0 - q)))


def brier_skill_score(p: Floats, y: Floats) -> float:
    """Brier score relative to always quoting the base rate. Positive = skill."""
    p_arr, y_arr = _clean(p, y)
    if not p_arr.size:
        return np.nan
    base = float(y_arr.mean())
    ref = float(np.mean((base - y_arr) ** 2))
    return 1.0 - brier_score(p_arr, y_arr) / ref if ref > 0 else np.nan


@dataclass(frozen=True, slots=True)
class BrierDecomposition:
    reliability: float
    resolution: float
    uncertainty: float
    brier: float
    n: int
    n_bins: int

    @property
    def residual(self) -> float:
        """Decomposition error from binning; should be ~0 up to bin coarseness."""
        return self.brier - (self.reliability - self.resolution + self.uncertainty)


def brier_decomposition(
    p: Floats, y: Floats, n_bins: int = 10
) -> BrierDecomposition:
    """Murphy's reliability / resolution / uncertainty decomposition."""
    p_arr, y_arr = _clean(p, y)
    n = p_arr.size
    if n == 0:
        return BrierDecomposition(np.nan, np.nan, np.nan, np.nan, 0, n_bins)

    base = float(y_arr.mean())
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    idx = np.clip(np.digitize(p_arr, edges[1:-1], right=False), 0, n_bins - 1)

    rel = 0.0
    res = 0.0
    for k in range(n_bins):
        sel = idx == k
        nk = int(sel.sum())
        if nk == 0:
            continue
        p_bar = float(p_arr[sel].mean())
        o_bar = float(y_arr[sel].mean())
        rel += nk * (p_bar - o_bar) ** 2
        res += nk * (o_bar - base) ** 2
    rel /= n
    res /= n
    unc = base * (1.0 - base)
    return BrierDecomposition(rel, res, unc, brier_score(p_arr, y_arr), n, n_bins)


def reliability_curve(
    p: Floats, y: Floats, n_bins: int = 10, strategy: str = "quantile"
) -> pd.DataFrame:
    """Binned forecast-vs-frequency table with Wilson confidence intervals.

    Quantile binning is the default because equal-width bins leave the tails
    almost empty on real market data, and the tails are exactly where the
    favourite-longshot bias lives.
    """
    p_arr, y_arr = _clean(p, y)
    if not p_arr.size:
        return pd.DataFrame(columns=["bin", "n", "mean_pred", "freq", "lo", "hi"])

    if strategy == "quantile":
        qs = np.linspace(0, 1, n_bins + 1)
        edges = np.unique(np.quantile(p_arr, qs))
        if edges.size < 2:
            edges = np.array([p_arr.min(), p_arr.max() + 1e-9])
    else:
        edges = np.linspace(0.0, 1.0, n_bins + 1)

    idx = np.clip(np.digitize(p_arr, edges[1:-1], right=False), 0, len(edges) - 2)
    rows = []
    z = 1.96
    for k in range(len(edges) - 1):
        sel = idx == k
        nk = int(sel.sum())
        if nk == 0:
            continue
        freq = float(y_arr[sel].mean())
        # Wilson interval: correct near 0 and 1, where the normal approximation
        # produces impossible bounds and the tails are the whole story.
        denom = 1.0 + z**2 / nk
        centre = (freq + z**2 / (2 * nk)) / denom
        half = z * np.sqrt(freq * (1 - freq) / nk + z**2 / (4 * nk**2)) / denom
        rows.append(
            {
                "bin": k,
                "lo_edge": float(edges[k]),
                "hi_edge": float(edges[k + 1]),
                "n": nk,
                "mean_pred": float(p_arr[sel].mean()),
                "freq": freq,
                "lo": max(0.0, centre - half),
                "hi": min(1.0, centre + half),
            }
        )
    return pd.DataFrame(rows)


def expected_calibration_error(
    p: Floats, y: Floats, n_bins: int = 10
) -> tuple[float, float]:
    """``(ECE, MCE)``: size-weighted mean and worst-case bin miscalibration."""
    curve = reliability_curve(p, y, n_bins)
    if curve.empty:
        return np.nan, np.nan
    gaps = (curve["mean_pred"] - curve["freq"]).abs()
    weights = curve["n"] / curve["n"].sum()
    return float((gaps * weights).sum()), float(gaps.max())


# --------------------------------------------------------------------------
# recalibration maps
# --------------------------------------------------------------------------
def pava(y: np.ndarray, w: np.ndarray | None = None, increasing: bool = True) -> np.ndarray:
    """Pool-adjacent-violators: the exact isotonic regression, in O(n).

    Implemented directly rather than pulled from a dependency because the
    weighted, decreasing variant is needed for ladder projection in
    :mod:`kalshi_alpha.probability.constraints` and the block structure it
    produces is used there too.
    """
    y = np.asarray(y, dtype=float)
    n = y.size
    if n == 0:
        return y.copy()
    w = np.ones(n) if w is None else np.asarray(w, dtype=float)
    if not increasing:
        return pava(y[::-1], w[::-1], increasing=True)[::-1]

    values: list[float] = []
    weights: list[float] = []
    counts: list[int] = []
    for i in range(n):
        v, ww, c = float(y[i]), float(w[i]), 1
        while values and values[-1] > v:
            pv, pw, pc = values.pop(), weights.pop(), counts.pop()
            v = (pv * pw + v * ww) / (pw + ww)
            ww += pw
            c += pc
        values.append(v)
        weights.append(ww)
        counts.append(c)

    out = np.empty(n, dtype=float)
    pos = 0
    for v, c in zip(values, counts, strict=True):
        out[pos : pos + c] = v
        pos += c
    return out


@dataclass
class IsotonicCalibrator:
    """Monotone, non-parametric recalibration map fitted by PAVA."""

    x: np.ndarray = field(default_factory=lambda: np.zeros(0))
    y: np.ndarray = field(default_factory=lambda: np.zeros(0))

    def fit(self, p: Floats, outcomes: Floats) -> IsotonicCalibrator:
        p_arr, y_arr = _clean(p, outcomes)
        order = np.argsort(p_arr, kind="mergesort")
        self.x = p_arr[order]
        self.y = pava(y_arr[order])
        return self

    def predict(self, p: Floats) -> np.ndarray:
        q = np.asarray(p, dtype=float)
        if self.x.size == 0:
            return q
        return np.interp(q, self.x, self.y, left=self.y[0], right=self.y[-1])


def isotonic_fit(p: Floats, y: Floats) -> IsotonicCalibrator:
    return IsotonicCalibrator().fit(p, y)


def platt_scale(
    p: Floats, y: Floats, max_iter: int = 100, tol: float = 1e-9
) -> tuple[float, float]:
    """Fit ``sigma(a * logit(p) + b)`` by Newton-Raphson.

    Two parameters instead of isotonic's ``n``: ``a`` measures how *sharp* the
    market is (``a < 1`` means it is overconfident and should be shrunk toward
    50c, ``a > 1`` means underconfident) and ``b`` is a constant directional
    tilt. That interpretability is why it is worth keeping alongside isotonic --
    it says *what* is wrong, not just how to fix it.
    """
    p_arr, y_arr = _clean(p, y)
    if p_arr.size < 3:
        return 1.0, 0.0
    z = np.log(np.clip(p_arr, EPS, 1 - EPS) / np.clip(1 - p_arr, EPS, 1 - EPS))
    a, b = 1.0, 0.0
    for _ in range(max_iter):
        eta = a * z + b
        mu = 1.0 / (1.0 + np.exp(-eta))
        wgt = np.clip(mu * (1.0 - mu), EPS, None)
        resid = y_arr - mu
        grad = np.array([float(np.sum(resid * z)), float(np.sum(resid))])
        hess = np.array(
            [
                [float(np.sum(wgt * z * z)), float(np.sum(wgt * z))],
                [float(np.sum(wgt * z)), float(np.sum(wgt))],
            ]
        )
        try:
            step = np.linalg.solve(hess, grad)
        except np.linalg.LinAlgError:
            break
        a += float(step[0])
        b += float(step[1])
        if np.max(np.abs(step)) < tol:
            break
    return a, b


def apply_platt(p: Floats, a: float, b: float) -> np.ndarray:
    q = np.clip(np.asarray(p, dtype=float), EPS, 1 - EPS)
    z = np.log(q / (1 - q))
    return 1.0 / (1.0 + np.exp(-(a * z + b)))


@dataclass(frozen=True)
class CalibrationReport:
    brier: float
    log_loss: float
    brier_skill: float
    ece: float
    mce: float
    decomposition: BrierDecomposition
    curve: pd.DataFrame
    platt_a: float
    platt_b: float
    n: int

    @property
    def overconfident(self) -> bool:
        """``a < 1`` means quoted probabilities are too extreme for the outcomes."""
        return bool(np.isfinite(self.platt_a) and self.platt_a < 1.0)

    def summary(self) -> str:
        d = self.decomposition
        tilt = "over" if self.overconfident else "under"
        return (
            f"n={self.n}  brier={self.brier:.4f}  logloss={self.log_loss:.4f}  "
            f"skill={self.brier_skill:+.4f}\n"
            f"  reliability={d.reliability:.5f}  resolution={d.resolution:.5f}  "
            f"uncertainty={d.uncertainty:.5f}\n"
            f"  ECE={self.ece:.4f}  MCE={self.mce:.4f}  "
            f"platt a={self.platt_a:.3f} b={self.platt_b:+.3f} ({tilt}confident)"
        )


def calibration_report(
    p: Floats, y: Floats, n_bins: int = 10
) -> CalibrationReport:
    p_arr, y_arr = _clean(p, y)
    ece, mce = expected_calibration_error(p_arr, y_arr, n_bins)
    a, b = platt_scale(p_arr, y_arr)
    return CalibrationReport(
        brier=brier_score(p_arr, y_arr),
        log_loss=log_loss(p_arr, y_arr),
        brier_skill=brier_skill_score(p_arr, y_arr),
        ece=ece,
        mce=mce,
        decomposition=brier_decomposition(p_arr, y_arr, n_bins),
        curve=reliability_curve(p_arr, y_arr, n_bins),
        platt_a=a,
        platt_b=b,
        n=int(p_arr.size),
    )
