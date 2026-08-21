"""Microstructure features and liquidity estimators."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from kalshi_alpha.microstructure.features import (
    book_features,
    feature_frame,
    liquidity_profile,
    merge_books_trades,
    order_flow_imbalance,
    quote_slippage,
    trade_frame,
)
from kalshi_alpha.microstructure.impact import (
    amihud_illiquidity,
    effective_spread,
    kyle_lambda,
    liquidity_report,
    ols_slope,
    price_impact,
    realized_spread,
    roll_spread,
    spread_decomposition,
    vpin,
)
from kalshi_alpha.types import OrderBook


class TestFeatures:
    def test_book_features_are_finite_on_a_normal_book(self, mk_book) -> None:
        f = book_features(mk_book("T", 40, 42))
        for key in ("bid", "ask", "mid", "microprice", "spread", "imbalance"):
            assert np.isfinite(f[key])

    def test_features_degrade_gracefully_on_an_empty_book(self) -> None:
        f = book_features(OrderBook.from_levels("T", 0.0, [], []))
        assert np.isnan(f["mid"])

    def test_ofi_is_positive_when_the_bid_thickens(self) -> None:
        prev = OrderBook.from_levels("T", 0.0, [(40, 100)], [(58, 100)])
        curr = OrderBook.from_levels("T", 1.0, [(40, 300)], [(58, 100)])
        assert order_flow_imbalance(prev, curr) > 0

    def test_ofi_is_negative_when_the_bid_is_pulled(self) -> None:
        prev = OrderBook.from_levels("T", 0.0, [(40, 300)], [(58, 100)])
        curr = OrderBook.from_levels("T", 1.0, [(40, 50)], [(58, 100)])
        assert order_flow_imbalance(prev, curr) < 0

    def test_ofi_is_positive_when_the_offer_is_lifted(self) -> None:
        prev = OrderBook.from_levels("T", 0.0, [(40, 100)], [(58, 300)])
        curr = OrderBook.from_levels("T", 1.0, [(40, 100)], [(58, 50)])
        assert order_flow_imbalance(prev, curr) > 0

    def test_feature_frame_shape_and_columns(self, sim) -> None:
        books = [snap[sim.tickers[0]] for snap in sim.books[:200]]
        df = feature_frame(books)
        assert len(df) == 200
        for col in ("mid", "microprice", "ofi", "ret", "prob", "prob_var"):
            assert col in df.columns

    def test_bernoulli_variance_peaks_at_even_money(self, sim) -> None:
        books = [snap[sim.tickers[0]] for snap in sim.books[:200]]
        df = feature_frame(books).dropna(subset=["prob"])
        near_half = (df["prob"] - 0.5).abs().idxmin()
        assert df.loc[near_half, "prob_var"] == df["prob_var"].max()

    def test_empty_input_returns_empty_frame(self) -> None:
        assert feature_frame([]).empty

    def test_slippage_grows_with_size(self, mk_book) -> None:
        book = mk_book("T", 40, 42, size=10, depth=4)
        assert quote_slippage(book, 30) > quote_slippage(book, 5)

    def test_liquidity_profile_is_monotone(self, mk_book) -> None:
        book = mk_book("T", 40, 42, size=20, depth=4)
        prof = liquidity_profile(book, sizes=(5, 20, 60))
        vals = prof["yes_slippage_cents"].dropna().tolist()
        assert all(a <= b + 1e-9 for a, b in zip(vals, vals[1:], strict=False))

    def test_merge_is_backward_looking_only(self, sim) -> None:
        books = [snap[sim.tickers[0]] for snap in sim.books[:200]]
        bdf = feature_frame(books)
        tdf = trade_frame(sim.trades[sim.tickers[0]])
        merged = merge_books_trades(bdf, tdf)
        assert len(merged) == len(bdf)


class TestRegression:
    def test_ols_recovers_a_known_slope(self) -> None:
        rng = np.random.default_rng(0)
        x = rng.normal(size=3000)
        y = 2.5 * x + rng.normal(0, 0.3, size=3000)
        reg = ols_slope(x, y)
        assert reg.beta == pytest.approx(2.5, abs=0.05)
        assert reg.r2 > 0.9
        assert reg.significant

    def test_ols_finds_nothing_in_noise(self) -> None:
        rng = np.random.default_rng(1)
        reg = ols_slope(rng.normal(size=2000), rng.normal(size=2000))
        assert not reg.significant

    def test_degenerate_input_returns_nan(self) -> None:
        assert np.isnan(ols_slope(np.ones(50), np.arange(50.0)).beta)


class TestImpact:
    def test_kyle_lambda_recovers_a_known_impact(self) -> None:
        rng = np.random.default_rng(2)
        flow = rng.normal(0, 100, 3000)
        dp = 0.004 * flow + rng.normal(0, 0.1, 3000)
        assert kyle_lambda(dp, flow).beta == pytest.approx(0.004, abs=0.0005)

    def test_amihud_rises_as_liquidity_falls(self) -> None:
        rng = np.random.default_rng(3)
        returns = rng.normal(0, 1, 1000)
        thick = amihud_illiquidity(returns, np.full(1000, 10_000.0))
        thin = amihud_illiquidity(returns, np.full(1000, 100.0))
        assert thin > thick

    def test_amihud_ignores_zero_volume_intervals(self) -> None:
        assert np.isfinite(amihud_illiquidity([1.0, 2.0], [0.0, 100.0]))

    def test_roll_spread_recovers_bid_ask_bounce(self) -> None:
        rng = np.random.default_rng(4)
        n = 6000
        efficient = np.cumsum(rng.normal(0, 0.05, n))
        half_spread = 1.0
        observed = efficient + half_spread * rng.choice([-1.0, 1.0], n)
        assert roll_spread(observed) == pytest.approx(2 * half_spread, rel=0.2)

    def test_roll_spread_is_nan_when_the_model_fails(self) -> None:
        trending = np.arange(200, dtype=float)
        assert np.isnan(roll_spread(trending))

    def test_vpin_is_one_for_perfectly_directional_flow(self) -> None:
        signed = np.full(1000, 10.0)
        volume = np.full(1000, 10.0)
        assert vpin(signed, volume, n_buckets=20) == pytest.approx(1.0, abs=0.05)

    def test_vpin_is_near_zero_for_balanced_flow(self) -> None:
        signed = np.tile([10.0, -10.0], 500)
        volume = np.full(1000, 10.0)
        assert vpin(signed, volume, n_buckets=20) < 0.2

    def test_effective_and_realized_spread_decompose(self) -> None:
        eff = effective_spread(trade_price=51, mid_before=50, taker_bought=True)
        real = realized_spread(trade_price=51, mid_after=50.5, taker_bought=True)
        impact = price_impact(51, 50, 50.5, True)
        assert eff == pytest.approx(2.0)
        assert real == pytest.approx(1.0)
        assert impact == pytest.approx(eff - real)

    def test_selling_flips_the_sign_convention(self) -> None:
        assert effective_spread(49, 50, taker_bought=False) > 0

    def test_spread_decomposition_runs_on_a_tape(self, sim) -> None:
        tk = sim.tickers[0]
        books = [snap[tk] for snap in sim.books]
        feats = feature_frame(books)
        tdf = trade_frame(sim.trades[tk])
        out = spread_decomposition(tdf, feats[["ts", "mid"]], horizon_s=60.0)
        assert {"ts", "effective", "realized", "impact"}.issubset(out.columns)

    def test_liquidity_report_populates(self, sim) -> None:
        tk = sim.tickers[0]
        feats = feature_frame([snap[tk] for snap in sim.books])
        report = liquidity_report(feats, trade_frame(sim.trades[tk]))
        assert "mean_spread_cents" in report
        assert report["mean_spread_cents"] > 0

    def test_liquidity_report_handles_no_trades(self) -> None:
        assert liquidity_report(pd.DataFrame(), pd.DataFrame()) == {}
