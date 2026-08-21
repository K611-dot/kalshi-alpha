"""Forecast scoring, calibration and no-arbitrage probability projection."""

from kalshi_alpha.probability.aggregation import (
    extremize,
    linear_pool,
    log_odds_pool,
    shrink_to_prior,
)
from kalshi_alpha.probability.calibration import (
    CalibrationReport,
    brier_decomposition,
    brier_score,
    expected_calibration_error,
    isotonic_fit,
    log_loss,
    platt_scale,
    reliability_curve,
)
from kalshi_alpha.probability.constraints import (
    project_to_monotone,
    project_to_simplex,
    repair_probabilities,
)

__all__ = [
    "CalibrationReport",
    "brier_decomposition",
    "brier_score",
    "expected_calibration_error",
    "extremize",
    "isotonic_fit",
    "linear_pool",
    "log_loss",
    "log_odds_pool",
    "platt_scale",
    "project_to_monotone",
    "project_to_simplex",
    "reliability_curve",
    "repair_probabilities",
    "shrink_to_prior",
]
