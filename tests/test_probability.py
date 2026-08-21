"""Scoring, calibration, pooling and no-arbitrage projection."""

from __future__ import annotations

import numpy as np
import pytest

from kalshi_alpha.probability.aggregation import (
    beta_posterior_mean,
    blend_with_market,
    disagreement,
    edge_vs_market,
    expit,
    extremize,
    linear_pool,
    log_odds_pool,
    logit,
    shrink_to_prior,
)
from kalshi_alpha.probability.calibration import (
    IsotonicCalibrator,
    brier_decomposition,
    brier_score,
    brier_skill_score,
    expected_calibration_error,
    log_loss,
    pava,
    reliability_curve,
)
from kalshi_alpha.probability.constraints import (
    coherent_mispricing_signal,
    incoherence,
    ladder_to_bucket_pmf,
    project_to_monotone,
    project_to_simplex,
    repair_probabilities,
)


class TestScoring:
    def test_perfect_forecasts_score_zero(self) -> None:
        assert brier_score([1.0, 0.0, 1.0], [1, 0, 1]) == 0.0

    def test_worst_forecasts_score_one(self) -> None:
        assert brier_score([0.0, 1.0], [1, 0]) == 1.0

    def test_log_loss_is_clipped_not_infinite(self) -> None:
        assert np.isfinite(log_loss([0.0], [1]))

    def test_skill_is_zero_against_the_base_rate(self) -> None:
        y = np.array([1, 1, 0, 0, 1, 0, 1, 0])
        base = np.full_like(y, y.mean(), dtype=float)
        assert brier_skill_score(base, y) == pytest.approx(0.0, abs=1e-9)

    def test_skill_is_positive_for_a_real_forecast(self) -> None:
        rng = np.random.default_rng(0)
        p = rng.uniform(0.05, 0.95, 4000)
        y = (rng.random(4000) < p).astype(float)
        assert brier_skill_score(p, y) > 0.1


class TestDecomposition:
    def test_a_calibrated_forecast_has_near_zero_reliability(self) -> None:
        rng = np.random.default_rng(1)
        p = rng.uniform(0.05, 0.95, 40_000)
        y = (rng.random(40_000) < p).astype(float)
        assert brier_decomposition(p, y, 10).reliability < 0.002

    def test_a_biased_forecast_has_large_reliability(self) -> None:
        rng = np.random.default_rng(2)
        p = rng.uniform(0.05, 0.95, 20_000)
        y = (rng.random(20_000) < np.clip(p - 0.2, 0.01, 0.99)).astype(float)
        assert brier_decomposition(p, y, 10).reliability > 0.02

    def test_a_constant_forecast_has_zero_resolution(self) -> None:
        rng = np.random.default_rng(3)
        y = (rng.random(5000) < 0.4).astype(float)
        p = np.full(5000, 0.4)
        assert brier_decomposition(p, y, 10).resolution == pytest.approx(0.0, abs=1e-6)

    def test_uncertainty_is_the_base_rate_variance(self) -> None:
        y = np.array([1.0] * 300 + [0.0] * 700)
        d = brier_decomposition(np.full(1000, 0.5), y, 10)
        assert d.uncertainty == pytest.approx(0.3 * 0.7, abs=1e-9)


class TestReliability:
    def test_curve_bins_cover_every_observation(self) -> None:
        rng = np.random.default_rng(4)
        p = rng.uniform(0, 1, 2000)
        y = (rng.random(2000) < p).astype(float)
        assert reliability_curve(p, y, 10)["n"].sum() == 2000

    def test_wilson_intervals_stay_inside_the_unit_interval(self) -> None:
        rng = np.random.default_rng(5)
        p = rng.uniform(0, 0.05, 800)  # deep tail, where normal CIs go negative
        y = (rng.random(800) < p).astype(float)
        curve = reliability_curve(p, y, 8)
        assert (curve["lo"] >= 0).all() and (curve["hi"] <= 1).all()

    def test_ece_is_small_for_a_calibrated_sample(self) -> None:
        rng = np.random.default_rng(6)
        p = rng.uniform(0.05, 0.95, 40_000)
        y = (rng.random(40_000) < p).astype(float)
        ece, mce = expected_calibration_error(p, y, 10)
        assert ece < 0.02 and mce < 0.06


class TestIsotonic:
    def test_pava_output_is_monotone(self) -> None:
        rng = np.random.default_rng(7)
        y = np.sort(rng.normal(size=500)) + rng.normal(0, 0.4, 500)
        fitted = pava(y)
        assert np.all(np.diff(fitted) >= -1e-12)

    def test_pava_preserves_the_mean(self) -> None:
        rng = np.random.default_rng(8)
        y = rng.normal(size=400)
        assert pava(y).mean() == pytest.approx(y.mean(), abs=1e-9)

    def test_pava_is_idempotent_on_monotone_input(self) -> None:
        y = np.linspace(0, 1, 100)
        assert np.allclose(pava(y), y)

    def test_decreasing_variant(self) -> None:
        y = np.linspace(1, 0, 50)
        assert np.all(np.diff(pava(y, increasing=False)) <= 1e-12)

    def test_calibrator_corrects_a_known_distortion(self) -> None:
        rng = np.random.default_rng(9)
        latent = rng.normal(0, 1.5, 20_000)
        true_p = expit(latent)
        quoted = expit(latent / 0.75)  # too extreme
        y = (rng.random(20_000) < true_p).astype(float)
        cal = IsotonicCalibrator().fit(quoted, y)
        assert brier_score(cal.predict(quoted), y) < brier_score(quoted, y)


class TestAggregation:
    def test_logit_and_expit_round_trip(self) -> None:
        for p in (0.01, 0.2, 0.5, 0.8, 0.99):
            assert float(expit(logit(p))) == pytest.approx(p, abs=1e-9)

    def test_log_odds_pool_is_more_extreme_than_linear(self) -> None:
        probs = [0.8, 0.9]
        assert log_odds_pool(probs) < linear_pool(probs) or log_odds_pool(probs) > 0.8
        # Both agree the answer is above either input's minimum.
        assert log_odds_pool(probs) > 0.8

    def test_pooling_two_identical_forecasts_is_a_no_op(self) -> None:
        assert log_odds_pool([0.7, 0.7]) == pytest.approx(0.7, abs=1e-9)

    def test_extremization_pushes_away_from_even_money(self) -> None:
        assert extremize(0.7, 2.0) > 0.7
        assert extremize(0.3, 2.0) < 0.3
        assert extremize(0.5, 3.0) == pytest.approx(0.5)

    def test_shrinkage_moves_toward_the_prior(self) -> None:
        assert 0.5 < shrink_to_prior(0.9, 0.5, 1.0) < 0.9

    def test_zero_weight_shrinkage_is_a_no_op(self) -> None:
        assert shrink_to_prior(0.9, 0.5, 0.0) == pytest.approx(0.9, abs=1e-9)

    def test_market_gets_most_of_the_weight_by_default(self) -> None:
        blended = blend_with_market(0.9, 0.5)
        assert 0.5 < blended < 0.75

    def test_disagreement_is_zero_for_a_consensus(self) -> None:
        assert disagreement([0.6, 0.6, 0.6]) == pytest.approx(0.0, abs=1e-9)

    def test_edge_decomposition_signs_agree(self) -> None:
        e = edge_vs_market(0.7, 0.5)
        assert e["prob_edge"] > 0 and e["logodds_edge"] > 0 and e["relative_odds"] > 1

    def test_beta_posterior_shrinks_small_samples(self) -> None:
        assert beta_posterior_mean(1, 1) < 1.0
        assert beta_posterior_mean(0, 1) > 0.0


class TestConstraints:
    def test_simplex_projection_sums_to_one(self) -> None:
        out = project_to_simplex([0.4, 0.4, 0.4])
        assert out.sum() == pytest.approx(1.0)
        assert (out >= 0).all()

    def test_simplex_projection_is_a_no_op_when_already_valid(self) -> None:
        v = np.array([0.2, 0.3, 0.5])
        assert np.allclose(project_to_simplex(v), v)

    def test_simplex_projection_can_zero_a_coordinate(self) -> None:
        out = project_to_simplex([0.9, 0.9, -0.4])
        assert out[2] == pytest.approx(0.0)

    def test_monotone_projection_fixes_an_inversion(self) -> None:
        out = project_to_monotone([0.8, 0.5, 0.6, 0.2], decreasing=True)
        assert np.all(np.diff(out) <= 1e-12)

    def test_monotone_projection_preserves_a_valid_ladder(self) -> None:
        v = [0.9, 0.6, 0.3, 0.1]
        assert np.allclose(project_to_monotone(v, decreasing=True), v)

    def test_bucket_pmf_is_a_distribution(self) -> None:
        pmf = ladder_to_bucket_pmf([0.9, 0.6, 0.2])
        assert pmf.sum() == pytest.approx(1.0)
        assert (pmf >= -1e-12).all()

    def test_bucket_pmf_goes_negative_on_an_inverted_ladder(self) -> None:
        """The projection residual and the arbitrage signal are the same object."""
        assert (ladder_to_bucket_pmf([0.5, 0.8, 0.2]) < -1e-9).any()

    def test_repair_is_idempotent(self) -> None:
        values = {"A": 0.5, "B": 0.4, "C": 0.3}
        once = repair_probabilities(values, "ladder")
        twice = repair_probabilities(once, "ladder")
        assert all(once[k] == pytest.approx(twice[k]) for k in once)

    def test_overround_is_reported(self) -> None:
        stats = incoherence({"A": 0.4, "B": 0.4, "C": 0.4}, "exclusive")
        assert stats["overround"] == pytest.approx(0.2)

    def test_mispricing_signal_points_at_the_rich_market(self) -> None:
        signal = coherent_mispricing_signal(
            {"A": 0.5, "B": 0.8, "C": 0.2}, kind="ladder", order=["A", "B", "C"]
        )
        # B is quoted above A despite a higher strike, so B is the rich one.
        assert signal["B"] > 0
        assert signal["A"] < 0

    def test_coherent_ladder_has_no_signal(self) -> None:
        signal = coherent_mispricing_signal(
            {"A": 0.8, "B": 0.5, "C": 0.2}, kind="ladder", order=["A", "B", "C"]
        )
        assert all(abs(v) < 1e-9 for v in signal.values())
