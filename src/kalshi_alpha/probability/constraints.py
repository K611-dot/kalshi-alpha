"""Projecting raw quotes onto a coherent probability surface.

Reading each market's mid independently gives a set of numbers that usually
does not describe any probability distribution: an exhaustive group whose mids
sum to 1.04, a strike ladder whose survival function ticks *up* between two
strikes. Those incoherences are the same objects the arbitrage detectors hunt,
seen from the modelling side.

This module supplies the projection that makes them coherent, and the residual
of that projection is a clean, continuous mispricing signal -- available even
when the dislocation is too small for fees to permit an actual arbitrage. Most
of the time the violation is one or two cents, which the exchange fee schedule
puts firmly out of reach; the projection residual still tells you which side of
the ladder the market is leaning on, which is tradeable as a directional view
even when the riskless version is not.

Both projections are Euclidean, so both are the *closest* coherent surface to
what the market is actually quoting -- nothing is imposed beyond coherence.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence

import numpy as np
from numpy.typing import ArrayLike

from kalshi_alpha.probability.calibration import pava


def project_to_simplex(p: ArrayLike, total: float = 1.0) -> np.ndarray:
    """Euclidean projection onto ``{q >= 0, sum q = total}``.

    Uses the standard sort-and-threshold algorithm (Duchi et al., 2008): sort
    descending, find the largest ``rho`` whose running average keeps the shifted
    coordinate positive, and subtract that threshold from every entry.
    """
    v = np.asarray(p, dtype=float)
    if v.size == 0:
        return v.copy()
    u = np.sort(v)[::-1]
    css = np.cumsum(u) - total
    idx = np.arange(1, v.size + 1)
    cond = u - css / idx > 0
    if not cond.any():
        return np.full_like(v, total / v.size)
    rho = int(idx[cond][-1])
    theta = css[rho - 1] / rho
    return np.maximum(v - theta, 0.0)


def project_to_monotone(
    p: ArrayLike, decreasing: bool = True, weights: ArrayLike | None = None
) -> np.ndarray:
    """Euclidean projection onto the monotone cone, then clipped to [0, 1].

    This is isotonic regression: the closest non-increasing (or non-decreasing)
    sequence to the quoted survival function. Clipping after PAVA is safe
    because clipping a monotone sequence to an interval leaves it monotone.
    """
    w = None if weights is None else np.asarray(weights, dtype=float)
    fitted = pava(np.asarray(p, dtype=float), w, increasing=not decreasing)
    return np.clip(fitted, 0.0, 1.0)


def ladder_to_bucket_pmf(survival: ArrayLike, decreasing: bool = True) -> np.ndarray:
    """Convert a monotone survival ladder into the implied bucket distribution.

    For a ``>= k`` ladder with ``k`` strikes there are ``k + 1`` buckets. The
    probabilities are successive differences of the survival function, which are
    non-negative precisely when the ladder is monotone -- so a negative bucket
    is a direct readout of the arbitrage the ladder detector would fire on.
    """
    s = np.asarray(survival, dtype=float)
    if not decreasing:
        s = 1.0 - s
    padded = np.concatenate([[1.0], s, [0.0]])
    return np.diff(-padded)


def repair_probabilities(
    values: Mapping[str, float],
    kind: str = "exclusive",
    decreasing: bool = True,
    order: Sequence[str] | None = None,
) -> dict[str, float]:
    """Project a quoted probability map onto the coherent set.

    ``kind='exclusive'`` projects onto the simplex (exhaustive groups);
    ``kind='ladder'`` projects onto the monotone cone. ``order`` fixes the
    strike ordering for ladders and defaults to insertion order.
    """
    keys = list(order) if order is not None else list(values)
    vec = np.array([values[k] for k in keys], dtype=float)
    if kind == "exclusive":
        fixed = project_to_simplex(vec)
    elif kind == "ladder":
        fixed = project_to_monotone(vec, decreasing=decreasing)
    else:
        raise ValueError(f"unknown kind {kind!r}")
    return dict(zip(keys, (float(x) for x in fixed), strict=True))


def incoherence(
    values: Mapping[str, float],
    kind: str = "exclusive",
    decreasing: bool = True,
    order: Sequence[str] | None = None,
) -> dict[str, float]:
    """Measure how far a quoted surface is from coherent.

    Returns the L1 and L-infinity distance to the projection, plus the signed
    over-round for exclusive groups. The L-infinity figure is the one to compare
    against the fee hurdle: it is the largest single-market dislocation, and it
    is what an arbitrage would have to monetise.
    """
    keys = list(order) if order is not None else list(values)
    raw = np.array([values[k] for k in keys], dtype=float)
    fixed = np.array(
        [repair_probabilities(values, kind, decreasing, keys)[k] for k in keys], dtype=float
    )
    diff = raw - fixed
    out = {
        "l1": float(np.abs(diff).sum()),
        "linf": float(np.abs(diff).max()) if diff.size else 0.0,
        "l2": float(np.sqrt(np.sum(diff**2))),
    }
    if kind == "exclusive":
        out["overround"] = float(raw.sum() - 1.0)
    else:
        out["max_inversion"] = float(
            np.max(np.diff(raw)) if decreasing and raw.size > 1 else
            (-np.min(np.diff(raw)) if raw.size > 1 else 0.0)
        )
    return out


def coherent_mispricing_signal(
    values: Mapping[str, float],
    kind: str = "exclusive",
    decreasing: bool = True,
    order: Sequence[str] | None = None,
) -> dict[str, float]:
    """Per-market signed distance from the coherent surface.

    Positive means the market is quoted *rich* relative to what the rest of the
    surface implies, so the trade is to sell it (buy NO); negative means cheap.
    This is the continuous cousin of an arbitrage signal and is what
    :class:`~kalshi_alpha.backtest.strategies.CoherenceStrategy` trades on.
    """
    keys = list(order) if order is not None else list(values)
    fixed = repair_probabilities(values, kind, decreasing, keys)
    return {k: float(values[k] - fixed[k]) for k in keys}
