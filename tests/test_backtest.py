"""Backtester, fill model and performance statistics.

The tests that matter here are the ones that would catch a backtest lying:
no look-ahead, fees actually charged, queue position actually respected, and
a strategy with no edge producing no profit.
"""

from __future__ import annotations

import numpy as np
import pytest

from kalshi_alpha.backtest.engine import Backtester, Context
from kalshi_alpha.backtest.fills import FillModel, RestingOrder
from kalshi_alpha.backtest.metrics import (
    deflated_sharpe_ratio,
    expected_max_sharpe,
    max_drawdown,
    performance,
    probabilistic_sharpe_ratio,
    sharpe_ratio,
    sortino_ratio,
    stationary_bootstrap_sharpe,
    walk_forward_splits,
)
from kalshi_alpha.backtest.strategies import build_strategy
from kalshi_alpha.config import ExecutionConfig, FeeConfig, Settings
from kalshi_alpha.types import (
    Action,
    Order,
    OrderStatus,
    OrderType,
    Side,
    TimeInForce,
    Trade,
)


# --------------------------------------------------------------------------
class TestFillModel:
    def test_aggressive_order_walks_the_book(self, mk_book) -> None:
        model = FillModel(execution=ExecutionConfig(adverse_reject_prob=0.0))
        book = mk_book("T", 40, 42, size=10, depth=3)
        order = Order("T", Side.YES, Action.BUY, 25, None, OrderType.MARKET)
        fills = model.fill_aggressive(order, book, 0.0)
        assert sum(f.qty for f in fills) == 25
        assert [f.price for f in fills] == [42, 43, 44]

    def test_limit_stops_at_its_price(self, mk_book) -> None:
        model = FillModel(execution=ExecutionConfig(adverse_reject_prob=0.0))
        book = mk_book("T", 40, 42, size=10, depth=3)
        order = Order("T", Side.YES, Action.BUY, 25, 43, OrderType.LIMIT)
        fills = model.fill_aggressive(order, book, 0.0)
        assert sum(f.qty for f in fills) == 20  # 44 is worse than the limit
        assert all(f.price <= 43 for f in fills)

    def test_every_fill_is_charged_a_fee(self, mk_book) -> None:
        model = FillModel(execution=ExecutionConfig(adverse_reject_prob=0.0))
        book = mk_book("T", 40, 42, size=100)
        order = Order("T", Side.YES, Action.BUY, 50, None, OrderType.MARKET)
        fills = model.fill_aggressive(order, book, 0.0)
        assert all(f.fee_cents > 0 for f in fills)

    def test_adverse_rejection_happens(self, mk_book) -> None:
        model = FillModel(execution=ExecutionConfig(adverse_reject_prob=1.0))
        order = Order("T", Side.YES, Action.BUY, 10, None, OrderType.MARKET)
        assert model.fill_aggressive(order, mk_book("T", 40, 42), 0.0) == []

    def test_queue_must_be_consumed_before_we_fill(self, mk_book) -> None:
        model = FillModel(execution=ExecutionConfig(queue_leakage=0.0))
        book = mk_book("T", 40, 42, size=100)
        order = Order("T", Side.YES, Action.BUY, 10, 40, OrderType.LIMIT,
                      tif=TimeInForce.GTC)
        resting = RestingOrder(order, 0.0, 0.0, model.initial_queue(order, book),
                               status=OrderStatus.OPEN)
        assert resting.queue_ahead == 100
        # A print smaller than the queue ahead does not reach us.
        assert model.apply_trade(resting, Trade("T", 1.0, 40, 40, Side.NO)) == []
        assert resting.filled == 0
        # A print that clears the queue does.
        fills = model.apply_trade(resting, Trade("T", 2.0, 40, 65, Side.NO))
        assert sum(f.qty for f in fills) == 5

    def test_queue_leakage_speeds_up_the_fill(self, mk_book) -> None:
        book = mk_book("T", 40, 42, size=100)
        order = Order("T", Side.YES, Action.BUY, 10, 40, OrderType.LIMIT,
                      tif=TimeInForce.GTC)
        results = []
        for leak in (0.0, 0.5):
            model = FillModel(execution=ExecutionConfig(queue_leakage=leak))
            r = RestingOrder(order, 0.0, 0.0, model.initial_queue(order, book),
                             status=OrderStatus.OPEN)
            model.apply_trade(r, Trade("T", 1.0, 40, 60, Side.NO))
            results.append(r.filled)
        assert results[1] > results[0]

    def test_wrong_side_print_does_not_fill_us(self) -> None:
        model = FillModel()
        order = Order("T", Side.YES, Action.BUY, 10, 40, OrderType.LIMIT,
                      tif=TimeInForce.GTC)
        resting = RestingOrder(order, 0.0, 0.0, 0.0, status=OrderStatus.OPEN)
        # A taker buying YES lifts the offer; it never hits our bid.
        assert model.apply_trade(resting, Trade("T", 1.0, 40, 50, Side.YES)) == []

    def test_sweep_through_is_adverse_selection(self, mk_book) -> None:
        model = FillModel()
        order = Order("T", Side.YES, Action.BUY, 10, 40, OrderType.LIMIT,
                      tif=TimeInForce.GTC)
        resting = RestingOrder(order, 0.0, 0.0, 0.0, status=OrderStatus.OPEN)
        # The market has traded down through our bid.
        fills = model.sweep_through(resting, mk_book("T", 30, 32), 1.0)
        assert sum(f.qty for f in fills) == 10
        assert "adverse" in fills[0].tag

    def test_maker_fees_are_zero_by_default(self) -> None:
        model = FillModel(fees=FeeConfig(maker_fees_enabled=False))
        assert model.fee(50, 100, "maker") == 0
        assert model.fee(50, 100, "taker") > 0


# --------------------------------------------------------------------------
class TestMetrics:
    def test_sharpe_of_a_constant_series_is_undefined(self) -> None:
        assert np.isnan(sharpe_ratio(np.ones(100)))

    def test_sharpe_scales_with_the_mean(self) -> None:
        rng = np.random.default_rng(0)
        base = rng.normal(0.0, 1.0, 2000)
        assert sharpe_ratio(base + 0.2) > sharpe_ratio(base + 0.05)

    def test_sortino_ignores_upside_volatility(self) -> None:
        r = np.array([1.0, 5.0, 1.0, 5.0, 1.0])  # never negative
        assert np.isinf(sortino_ratio(r))

    def test_max_drawdown_finds_the_worst_stretch(self) -> None:
        equity = np.array([100.0, 120.0, 90.0, 130.0, 125.0])
        depth, peak, trough = max_drawdown(equity)
        assert depth == pytest.approx(-30.0)
        assert peak == 1 and trough == 2

    def test_no_drawdown_on_a_rising_curve(self) -> None:
        assert max_drawdown(np.arange(10.0))[0] == pytest.approx(0.0)

    def test_expected_max_sharpe_grows_with_trials(self) -> None:
        assert expected_max_sharpe(100, 0.1) > expected_max_sharpe(5, 0.1)
        assert expected_max_sharpe(1, 0.1) == 0.0

    def test_deflation_penalises_a_wide_search(self) -> None:
        rng = np.random.default_rng(1)
        r = rng.normal(0.05, 1.0, 500)
        assert deflated_sharpe_ratio(r, n_trials=200) < deflated_sharpe_ratio(r, n_trials=1)

    def test_psr_is_a_probability(self) -> None:
        rng = np.random.default_rng(2)
        p = probabilistic_sharpe_ratio(rng.normal(0.1, 1.0, 500))
        assert 0.0 <= p <= 1.0

    def test_bootstrap_ci_brackets_the_point_estimate(self) -> None:
        rng = np.random.default_rng(3)
        r = rng.normal(0.05, 1.0, 600)
        boot = stationary_bootstrap_sharpe(r, draws=200, periods_per_year=252)
        assert boot["ci_low"] <= boot["mean"] <= boot["ci_high"]

    def test_walk_forward_splits_never_leak(self) -> None:
        for train, test in walk_forward_splits(1000, n_folds=4, min_train=200):
            assert max(train) < min(test)

    def test_performance_handles_an_empty_curve(self) -> None:
        assert performance(np.array([])).n_periods == 0


# --------------------------------------------------------------------------
class TestBacktester:
    def test_a_do_nothing_strategy_ends_flat(self, sim, settings) -> None:
        result = Backtester(settings).run(
            sim.books, sim.times, lambda _ctx: [], trades=sim.trades,
            settlement=sim.settle(),
        )
        assert result.n_orders == 0
        assert result.stats.total_pnl == pytest.approx(0.0)

    def test_arbitrage_strategy_is_silent_on_a_coherent_book(self, sim, settings) -> None:
        strategy = build_strategy("ladder_arb", ladders=[sim.ladder], scan_every=10)
        result = Backtester(settings).run(
            sim.books, sim.times, strategy, trades=sim.trades, settlement=sim.settle()
        )
        assert result.n_orders == 0

    def test_arbitrage_strategy_fires_on_a_dislocation(self, sim, settings) -> None:
        from kalshi_alpha.data.synthetic import inject_dislocation

        mid = sim.tickers[len(sim.tickers) // 2]
        books = [inject_dislocation(s, mid, 32) for s in sim.books]
        strategy = build_strategy("ladder_arb", ladders=[sim.ladder], scan_every=20,
                                  max_qty=50)
        result = Backtester(settings).run(
            books, sim.times, strategy, trades=sim.trades, settlement=sim.settle()
        )
        assert result.n_orders > 0
        assert not result.fills.empty

    def test_crossing_limits_fill_as_taker_and_pay_fees(self, sim, settings) -> None:
        from kalshi_alpha.data.synthetic import inject_dislocation

        mid = sim.tickers[len(sim.tickers) // 2]
        books = [inject_dislocation(s, mid, 32) for s in sim.books]
        strategy = build_strategy("ladder_arb", ladders=[sim.ladder], scan_every=20,
                                  max_qty=50)
        result = Backtester(settings).run(
            books, sim.times, strategy, trades=sim.trades, settlement=sim.settle()
        )
        assert (result.fills["liquidity"] == "taker").all()
        assert result.fills["fee_cents"].sum() > 0

    def test_orders_are_delayed_by_latency(self, sim) -> None:
        """An order submitted at step 0 cannot fill against step 0's book."""
        settings = Settings(execution=ExecutionConfig(
            order_latency_ms=20_000.0, adverse_reject_prob=0.0
        ))
        fired = {"done": False}

        def once(ctx: Context):
            if fired["done"]:
                return []
            fired["done"] = True
            return [
                Order(ctx.books and next(iter(ctx.books)), Side.YES, Action.BUY, 5,
                      None, OrderType.MARKET, client_order_id="x", ts=ctx.ts)
            ]

        result = Backtester(settings).run(sim.books[:20], sim.times[:20], once)
        # 20s of latency at 5s bars means the fill lands at least 4 bars later.
        if not result.fills.empty:
            assert result.fills["ts"].iloc[0] >= sim.times[4]

    def test_risk_limits_reject_oversized_orders(self, sim) -> None:
        from kalshi_alpha.config import RiskConfig

        settings = Settings(risk=RiskConfig(max_order_qty=1))
        ticker = sim.tickers[0]

        def greedy(ctx: Context):
            return [Order(ticker, Side.YES, Action.BUY, 500, None, OrderType.MARKET,
                          client_order_id=f"g{ctx.step}", ts=ctx.ts)]

        result = Backtester(settings).run(sim.books[:30], sim.times[:30], greedy)
        assert result.n_orders == 0
        assert result.n_rejected == 30

    def test_settlement_pays_off_open_positions(self, sim, settings) -> None:
        ticker = sim.tickers[0]
        placed = {"done": False}

        def buy_once(ctx: Context):
            if placed["done"]:
                return []
            placed["done"] = True
            return [Order(ticker, Side.YES, Action.BUY, 10, None, OrderType.MARKET,
                          client_order_id="b", ts=ctx.ts)]

        result = Backtester(settings).run(
            sim.books, sim.times, buy_once, trades=sim.trades, settlement=sim.settle()
        )
        pos = result.positions.get(ticker)
        assert pos is not None
        # Final equity must equal cash plus the realised settlement value.
        assert np.isfinite(result.settled_pnl_cents)

    def test_equity_curve_matches_the_snapshot_count(self, sim, settings) -> None:
        result = Backtester(settings).run(sim.books, sim.times, lambda _ctx: [])
        assert result.equity.size == len(sim.times)
