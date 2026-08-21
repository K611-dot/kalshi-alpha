"""Ground-truth recovery: the tests that turn "it runs" into "it is correct".

Every other test checks that the code does what it says. These check that what
it says is *true*, by generating data whose answer is known by construction and
asserting the estimator finds it. An estimator that has never been scored
against a known answer is an opinion with a docstring.
"""

from __future__ import annotations

import numpy as np
import pytest

from kalshi_alpha.data.synthetic import (
    SimConfig,
    digital_probability,
    simulate_calibration_sample,
    simulate_ladder,
    simulate_two_venue,
)
from kalshi_alpha.diffusion.halflife import ar1_half_life, half_life_ensemble
from kalshi_alpha.diffusion.price_discovery import fit_vecm, information_share, price_discovery
from kalshi_alpha.probability.calibration import calibration_report, platt_scale


def _ladder_with_half_life(true_hl: float, seed: int = 5):
    dt = 5.0
    post = int(np.clip(40.0 * true_hl / dt, 400, 6000))
    cfg = SimConfig(
        n_steps=600 + post, dt_s=dt, event_step=600,
        adjustment_half_life_s=true_hl, event_jump_cents=18.0,
        tau_floor_frac=0.5, seed=seed,
    )
    return simulate_ladder(cfg=cfg), cfg


class TestHalfLifeRecovery:
    @pytest.mark.parametrize("true_hl", [30.0, 60.0, 120.0, 300.0])
    def test_spread_estimator_recovers_the_true_half_life(self, true_hl: float) -> None:
        sim, cfg = _ladder_with_half_life(true_hl)
        tk = sim.tickers[len(sim.tickers) // 2]
        ev = int(cfg.event_step)
        t = sim.times[ev:] - sim.times[ev]
        quoted = sim.quoted[tk][ev:] * 100
        fair = sim.fair[tk][ev:] * 100

        res = half_life_ensemble(t, quoted, cfg.dt_s, leader=fair)
        est = res.consensus_half_life_s
        assert res.consensus_source == "spread_ar1"
        assert est == pytest.approx(true_hl, rel=0.25)

    def test_ar1_inverts_a_known_phi(self) -> None:
        rng = np.random.default_rng(0)
        dt, true_hl = 5.0, 120.0
        phi = 0.5 ** (dt / true_hl)
        gap = np.zeros(4000)
        for i in range(1, gap.size):
            gap[i] = phi * gap[i - 1] + rng.normal(0.0, 1.0)
        assert ar1_half_life(gap, dt).half_life_s == pytest.approx(true_hl, rel=0.15)

    def test_faster_markets_measure_as_faster(self) -> None:
        """Monotonicity is a weaker but far more robust claim than point accuracy."""
        estimates = []
        for hl in (30.0, 120.0, 300.0):
            sim, cfg = _ladder_with_half_life(hl)
            tk = sim.tickers[len(sim.tickers) // 2]
            ev = int(cfg.event_step)
            res = half_life_ensemble(
                sim.times[ev:] - sim.times[ev],
                sim.quoted[tk][ev:] * 100,
                cfg.dt_s,
                leader=sim.fair[tk][ev:] * 100,
            )
            estimates.append(res.consensus_half_life_s)
        assert estimates[0] < estimates[1] < estimates[2]

    def test_an_efficient_market_shows_no_persistent_gap(self) -> None:
        """With no lag injected, the quote tracks fair value and there is no decay."""
        cfg = SimConfig(n_steps=1200, dt_s=5.0, event_step=400,
                        adjustment_half_life_s=0.0, event_jump_cents=18.0, seed=9)
        sim = simulate_ladder(cfg=cfg)
        tk = sim.tickers[2]
        gap = (sim.quoted[tk] - sim.fair[tk]) * 100
        assert np.allclose(gap, 0.0, atol=1e-9)


class TestInformationShareRecovery:
    @pytest.mark.parametrize(
        ("sigma_a", "sigma_b", "expect_leader"),
        [(0.9, 0.3, 0), (0.3, 0.9, 1), (0.8, 0.2, 0)],
    )
    def test_leader_is_identified(self, sigma_a, sigma_b, expect_leader) -> None:
        tv = simulate_two_venue(n=6000, sigma_a=sigma_a, sigma_b=sigma_b,
                                kappa_a=0.04, kappa_b=0.30)
        res = price_discovery(tv.venue_a, tv.venue_b, ("a", "b"), lags=4)
        assert res is not None
        assert int(np.argmax(res.is_mid)) == expect_leader

    def test_shares_sum_to_one(self) -> None:
        tv = simulate_two_venue(n=5000)
        fit = fit_vecm(tv.venue_a, tv.venue_b, lags=4)
        assert fit is not None
        shares = information_share(fit)
        assert float(np.sum(shares["mid"])) == pytest.approx(1.0, abs=0.05)
        assert float(np.sum(fit.psi)) == pytest.approx(1.0, abs=1e-9)

    def test_share_is_monotone_in_the_true_share(self) -> None:
        estimates = []
        for sa, sb in ((0.3, 0.9), (0.5, 0.5), (0.9, 0.3)):
            tv = simulate_two_venue(n=6000, sigma_a=sa, sigma_b=sb,
                                    kappa_a=0.04, kappa_b=0.30)
            res = price_discovery(tv.venue_a, tv.venue_b, ("a", "b"), lags=4)
            assert res is not None
            estimates.append(float(res.is_mid[0]))
        assert estimates[0] < estimates[1] < estimates[2]

    def test_cholesky_bounds_bracket_the_midpoint(self) -> None:
        tv = simulate_two_venue(n=5000)
        res = price_discovery(tv.venue_a, tv.venue_b, ("a", "b"), lags=4)
        assert res is not None
        assert (res.is_lower <= res.is_mid + 1e-9).all()
        assert (res.is_mid <= res.is_upper + 1e-9).all()

    def test_spread_is_stationary(self) -> None:
        """Cointegration is the licence to interpret the VECM at all."""
        tv = simulate_two_venue(n=5000)
        res = price_discovery(tv.venue_a, tv.venue_b, ("a", "b"), lags=4)
        assert res is not None
        assert res.adf["stat"] < res.adf["crit_5pct"]


class TestCalibrationRecovery:
    @pytest.mark.parametrize("bias", [0.7, 0.85, 1.0, 1.2])
    def test_platt_recovers_the_injected_distortion(self, bias: float) -> None:
        quoted, outcomes = simulate_calibration_sample(n=20_000, bias_a=bias, seed=4)
        a, _ = platt_scale(quoted, outcomes)
        assert a == pytest.approx(bias, abs=0.12)

    def test_a_calibrated_market_scores_as_calibrated(self) -> None:
        quoted, outcomes = simulate_calibration_sample(n=20_000, bias_a=1.0, seed=6)
        rep = calibration_report(quoted, outcomes)
        assert rep.ece < 0.02
        assert rep.decomposition.reliability < 0.002

    def test_murphy_decomposition_reconstructs_the_brier_score(self) -> None:
        quoted, outcomes = simulate_calibration_sample(n=10_000, bias_a=0.85)
        rep = calibration_report(quoted, outcomes, n_bins=20)
        d = rep.decomposition
        assert d.reliability - d.resolution + d.uncertainty == pytest.approx(d.brier, abs=0.01)


class TestSimulatorInvariants:
    def test_digital_price_is_monotone_in_the_strike(self) -> None:
        probs = [digital_probability(100.0, k, 0.35, 0.02) for k in range(90, 111)]
        assert all(a >= b for a, b in zip(probs, probs[1:], strict=False))

    def test_digital_price_converges_at_expiry(self) -> None:
        assert digital_probability(105.0, 100.0, 0.35, 0.0) == 1.0
        assert digital_probability(95.0, 100.0, 0.35, 0.0) == 0.0

    def test_simulated_ladder_is_internally_coherent(self, sim) -> None:
        """Fair values must be monotone at every timestep, by construction."""
        for i in range(0, len(sim.times), 37):
            values = [sim.fair[t][i] for t in sim.tickers]
            assert all(a >= b - 1e-12 for a, b in zip(values, values[1:], strict=False))

    def test_simulated_books_are_never_crossed(self, sim) -> None:
        for snapshot in sim.books[::29]:
            for b in snapshot.values():
                assert not b.is_crossed
                assert b.is_two_sided

    def test_settlement_is_consistent_with_the_final_underlying(self, sim) -> None:
        settle = sim.settle()
        final = float(sim.underlying[-1])
        for tk, strike in zip(sim.tickers, sim.ladder.strikes, strict=True):
            assert settle[tk] == int(final >= strike)
