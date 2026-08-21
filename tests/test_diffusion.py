"""Diffusion estimators against constructed series with known properties."""

from __future__ import annotations

import numpy as np
import pytest

from kalshi_alpha.diffusion.efficiency import (
    autocorrelation,
    drift_direction,
    ljung_box,
    runs_test,
    variance_ratio_profile,
    variance_ratio_test,
)
from kalshi_alpha.diffusion.event_study import align_event_windows, event_study
from kalshi_alpha.diffusion.halflife import (
    adjustment_profile,
    crossing_times,
    exponential_half_life,
    select_window,
    terminal_price_stability,
    window_scan,
)
from kalshi_alpha.diffusion.leadlag import (
    cross_correlation,
    epps_curve,
    estimate_delay,
    hayashi_yoshida,
    lead_lag_ratio,
)


# --------------------------------------------------------------------------
class TestVarianceRatio:
    def test_random_walk_is_not_rejected(self) -> None:
        rng = np.random.default_rng(0)
        rejects = sum(
            variance_ratio_test(rng.normal(size=3000), q=4).rejects for _ in range(20)
        )
        # A 5% test on true nulls should reject about 1 time in 20.
        assert rejects <= 4

    def test_positive_autocorrelation_shows_as_vr_above_one(self) -> None:
        rng = np.random.default_rng(1)
        e = rng.normal(size=4000)
        r = np.zeros_like(e)
        for i in range(1, r.size):
            r[i] = 0.3 * r[i - 1] + e[i]
        res = variance_ratio_test(r, q=4)
        assert res.detail["vr"] > 1.0
        assert res.rejects

    def test_mean_reversion_shows_as_vr_below_one(self) -> None:
        rng = np.random.default_rng(2)
        e = rng.normal(size=4000)
        r = np.zeros_like(e)
        for i in range(1, r.size):
            r[i] = -0.3 * r[i - 1] + e[i]
        res = variance_ratio_test(r, q=4)
        assert res.detail["vr"] < 1.0
        assert res.rejects

    def test_robust_and_homoskedastic_agree_on_direction(self) -> None:
        rng = np.random.default_rng(3)
        e = rng.normal(size=3000)
        r = np.zeros_like(e)
        for i in range(1, r.size):
            r[i] = 0.25 * r[i - 1] + e[i]
        a = variance_ratio_test(r, q=8, robust=True)
        b = variance_ratio_test(r, q=8, robust=False)
        assert np.sign(a.statistic) == np.sign(b.statistic)

    def test_profile_covers_every_horizon(self) -> None:
        rng = np.random.default_rng(4)
        profile = variance_ratio_profile(rng.normal(size=2000), qs=(2, 4, 8))
        assert set(profile) == {2, 4, 8}

    def test_drift_direction_labels_underreaction(self) -> None:
        rng = np.random.default_rng(5)
        e = rng.normal(size=5000)
        r = np.zeros_like(e)
        for i in range(1, r.size):
            r[i] = 0.35 * r[i - 1] + e[i]
        assert "drift" in drift_direction(r, q=8)


class TestOtherEfficiencyTests:
    def test_runs_test_rejects_a_deterministic_alternation(self) -> None:
        series = np.array([1.0, -1.0] * 200)
        assert runs_test(series).rejects

    def test_runs_test_accepts_noise(self) -> None:
        rng = np.random.default_rng(6)
        assert not runs_test(rng.normal(size=2000)).rejects

    def test_ljung_box_rejects_autocorrelated_series(self) -> None:
        rng = np.random.default_rng(7)
        e = rng.normal(size=2000)
        r = np.zeros_like(e)
        for i in range(1, r.size):
            r[i] = 0.4 * r[i - 1] + e[i]
        assert ljung_box(r, lags=10).rejects

    def test_autocorrelation_recovers_a_known_ar1(self) -> None:
        rng = np.random.default_rng(8)
        e = rng.normal(size=8000)
        r = np.zeros_like(e)
        for i in range(1, r.size):
            r[i] = 0.5 * r[i - 1] + e[i]
        assert autocorrelation(r, 3)[0] == pytest.approx(0.5, abs=0.05)


# --------------------------------------------------------------------------
class TestHalfLifeMechanics:
    def test_adjustment_profile_runs_from_zero_to_one(self) -> None:
        t = np.arange(200) * 1.0
        p = 50.0 + 10.0 * (1.0 - np.exp(-t / 20.0))
        prof = adjustment_profile(t, p)
        assert prof["phi"].iloc[0] == pytest.approx(0.0, abs=0.02)
        assert prof["phi"].iloc[-1] == pytest.approx(1.0, abs=0.05)

    def test_crossing_times_are_ordered(self) -> None:
        t = np.arange(300) * 1.0
        p = 50.0 + 10.0 * (1.0 - np.exp(-t / 30.0))
        crossings = crossing_times(adjustment_profile(t, p))
        assert crossings["t50"] < crossings["t90"]

    def test_exponential_fit_recovers_a_clean_decay(self) -> None:
        t = np.arange(400) * 1.0
        tau = 40.0
        gap = 12.0 * np.exp(-t / tau)
        fit = exponential_half_life(t, gap)
        assert fit.tau_s == pytest.approx(tau, rel=0.1)
        assert fit.half_life_s == pytest.approx(tau * np.log(2), rel=0.1)

    def test_window_scan_shows_the_runaway(self) -> None:
        """A decay buried in a random walk inflates as the window grows."""
        rng = np.random.default_rng(11)
        n = 3000
        t = np.arange(n) * 5.0
        decay = 15.0 * np.exp(-t / 150.0)
        walk = np.cumsum(rng.normal(0.0, 0.35, size=n))
        scan = window_scan(t, 50.0 - decay + walk, 5.0)
        valid = scan["half_life_s"].dropna()
        assert len(valid) > 3
        assert valid.iloc[-1] > valid.iloc[0]

    def test_select_window_flags_an_unidentified_series(self) -> None:
        rng = np.random.default_rng(12)
        n = 1500
        t = np.arange(n) * 5.0
        pure_walk = 50.0 + np.cumsum(rng.normal(0.0, 0.5, size=n))
        choice = select_window(t, pure_walk, 5.0)
        assert isinstance(choice.reliable, bool)

    def test_terminal_stability_detects_an_unfinished_move(self) -> None:
        t = np.arange(400) * 1.0
        still_moving = 50.0 + 0.05 * t  # never settles
        assert terminal_price_stability(still_moving)["settled"] == 0.0

    def test_terminal_stability_accepts_a_settled_series(self) -> None:
        rng = np.random.default_rng(13)
        settled = 50.0 + rng.normal(0.0, 0.2, size=400)
        assert terminal_price_stability(settled)["settled"] == 1.0


# --------------------------------------------------------------------------
class TestLeadLag:
    def test_cross_correlation_finds_a_known_shift(self) -> None:
        rng = np.random.default_rng(14)
        x = rng.normal(size=3000)
        y = np.roll(x, 5)  # y repeats x five steps later
        lags, corr = cross_correlation(x, y, max_lag=15)
        assert lags[int(np.argmax(corr))] == 5

    def test_lead_lag_ratio_points_the_right_way(self) -> None:
        rng = np.random.default_rng(15)
        x = rng.normal(size=3000)
        y = np.roll(x, 4)
        assert lead_lag_ratio(x, y, max_lag=12)["llr"] > 1.0

    def test_hayashi_yoshida_matches_covariance_on_a_synchronous_grid(self) -> None:
        rng = np.random.default_rng(16)
        n = 2000
        t = np.arange(n, dtype=float)
        r1 = rng.normal(size=n - 1)
        r2 = 0.8 * r1 + 0.6 * rng.normal(size=n - 1)
        p1 = np.concatenate([[0.0], np.cumsum(r1)])
        p2 = np.concatenate([[0.0], np.cumsum(r2)])
        hy = hayashi_yoshida(t, p1, t, p2)
        assert hy == pytest.approx(float(np.dot(r1, r2)), rel=0.05)

    def test_estimate_delay_recovers_an_injected_lag(self) -> None:
        rng = np.random.default_rng(17)
        n = 3000
        t = np.arange(n, dtype=float)
        base = np.cumsum(rng.normal(size=n))
        lag = 6
        follower = np.concatenate([np.zeros(lag), base[:-lag]])
        est = estimate_delay(t, base, t, follower, max_shift_s=20.0, step_s=1.0)
        assert est.delay_s == pytest.approx(-lag, abs=2.0)

    def test_epps_effect_is_visible(self) -> None:
        """Measured correlation should rise with the sampling interval."""
        rng = np.random.default_rng(18)
        n = 6000
        t1 = np.sort(rng.uniform(0, 6000, n))
        t2 = np.sort(rng.uniform(0, 6000, n))
        base = np.cumsum(rng.normal(size=n))
        p1 = base
        p2 = np.interp(t2, t1, base) + rng.normal(0, 0.3, n)
        curve = epps_curve(t1, p1, t2, p2, intervals=(1.0, 30.0))
        assert np.isfinite(curve[1.0]) and np.isfinite(curve[30.0])


# --------------------------------------------------------------------------
class TestEventStudy:
    def test_alignment_produces_one_row_per_usable_event(self) -> None:
        t = np.arange(4000) * 5.0
        p = 50.0 + np.zeros_like(t)
        events = [t[1000], t[2000], t[3000]]
        grid, matrix = align_event_windows(t, p, events, pre_s=500, post_s=500, bar_s=5)
        assert matrix.shape[0] == 3
        assert matrix.shape[1] == grid.size

    def test_events_too_close_to_the_edges_are_dropped(self) -> None:
        t = np.arange(500) * 5.0
        p = np.full_like(t, 50.0)
        _, matrix = align_event_windows(t, p, [t[5]], pre_s=500, post_s=500, bar_s=5)
        assert matrix.shape[0] == 0

    def test_a_flat_series_produces_no_significant_effect(self) -> None:
        rng = np.random.default_rng(19)
        t = np.arange(6000) * 5.0
        p = 50.0 + rng.normal(0, 0.4, size=t.size)
        events = [t[1500], t[3000], t[4500]]
        res = event_study(t, p, events, pre_s=600, post_s=600, bar_s=5, placebo_draws=60)
        assert not res.significant

    def test_a_real_jump_is_detected(self) -> None:
        rng = np.random.default_rng(20)
        t = np.arange(6000) * 5.0
        p = 50.0 + rng.normal(0, 0.3, size=t.size)
        events = [t[1500], t[3000], t[4500]]
        for e in events:
            p[t >= e] += 12.0  # a large, consistent repricing
        res = event_study(t, p, events, pre_s=600, post_s=600, bar_s=5, placebo_draws=120)
        assert res.car_terminal > 5.0
        assert res.significant
