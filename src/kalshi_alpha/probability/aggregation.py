"""Combining probability forecasts.

The market price is one forecast. A model is another. Whether to trade is a
question about how to *combine* them, and the answer is not a simple average.

Averaging probabilities linearly is provably underconfident: the mean of two
independent, well-calibrated forecasts is less extreme than either, so the
pooled forecast is systematically too close to 50c. Pooling in **log-odds**
space fixes the direction of the bias, and **extremization** -- raising the
pooled odds to a power ``a > 1`` -- corrects the remaining shrinkage that comes
from the forecasters sharing information. Extremized log-odds pooling is the
technique that produced the largest measured gains in the IARPA forecasting
tournaments, and it is directly applicable here because a market quote and a
model output are exactly two correlated forecasts of the same binary event.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np

EPS = 1e-9


def logit(p: float | np.ndarray) -> np.ndarray:
    q = np.clip(np.asarray(p, dtype=float), EPS, 1.0 - EPS)
    return np.log(q / (1.0 - q))


def expit(x: float | np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.asarray(x, dtype=float)))


def linear_pool(probs: Sequence[float], weights: Sequence[float] | None = None) -> float:
    """Weighted arithmetic mean. Included as the baseline it usually loses to."""
    p = np.asarray(probs, dtype=float)
    w = np.ones_like(p) if weights is None else np.asarray(weights, dtype=float)
    w = w / w.sum()
    return float(np.dot(w, p))


def log_odds_pool(
    probs: Sequence[float], weights: Sequence[float] | None = None, a: float = 1.0
) -> float:
    """Weighted geometric mean of the odds, optionally extremized by ``a``.

    ``a = 1`` is plain log-odds pooling. ``a > 1`` pushes the result away from
    50c; the value that is optimal under the standard shared-information model
    grows with the number of correlated forecasters, and 1.5-2.5 is the range
    that survives out-of-sample in practice.
    """
    p = np.asarray(probs, dtype=float)
    w = np.ones_like(p) if weights is None else np.asarray(weights, dtype=float)
    total = w.sum()
    if total <= 0:
        return float("nan")
    w = w / total
    return float(expit(a * float(np.dot(w, logit(p)))))


def extremize(p: float, a: float = 1.5) -> float:
    """Sharpen a single forecast in odds space: ``odds -> odds^a``."""
    return float(expit(a * logit(p)))


def shrink_to_prior(p: float, prior: float, weight: float) -> float:
    """Shrink toward a prior in log-odds space.

    ``weight`` is the effective number of prior observations relative to the
    evidence behind ``p``; 0 keeps the forecast, large values collapse it onto
    the prior. Shrinking in log-odds rather than probability space keeps the
    result well-behaved in the tails, where probability-space shrinkage would
    destroy the very extremity that carries the information.
    """
    w = max(0.0, float(weight))
    return float(expit((logit(p) + w * logit(prior)) / (1.0 + w)))


def beta_posterior_mean(successes: float, trials: float, alpha: float = 1.0,
                        beta: float = 1.0) -> float:
    """Beta-binomial posterior mean; the standard small-sample base rate."""
    if trials < 0 or successes < 0 or successes > trials:
        raise ValueError("require 0 <= successes <= trials")
    return float((successes + alpha) / (trials + alpha + beta))


def disagreement(probs: Sequence[float]) -> float:
    """Spread of a forecast panel in log-odds units.

    A useful trade filter in its own right: when a model and the market disagree
    by a lot in log-odds, either the model has found something or it is broken,
    and the correct response is to size down rather than up.
    """
    z = logit(np.asarray(probs, dtype=float))
    return float(np.std(z)) if z.size > 1 else 0.0


def edge_vs_market(model_p: float, market_p: float) -> dict[str, float]:
    """Decompose a model-versus-market view into probability and odds terms."""
    return {
        "prob_edge": float(model_p - market_p),
        "logodds_edge": float(logit(model_p) - logit(market_p)),
        "relative_odds": float(
            (model_p / max(1 - model_p, EPS)) / max(market_p / max(1 - market_p, EPS), EPS)
        ),
    }


def blend_with_market(
    model_p: float,
    market_p: float,
    model_weight: float = 0.35,
    extremization: float = 1.0,
) -> float:
    """Default combiner used by the model-driven strategies.

    The market gets most of the weight by default. That is a deliberate prior:
    on a liquid contract the quote already aggregates every participant's
    information, and a model that overrides it needs to earn the right to.
    """
    w = float(np.clip(model_weight, 0.0, 1.0))
    return log_odds_pool([model_p, market_p], [w, 1.0 - w], a=extremization)
