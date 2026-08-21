"""Book mechanics and position accounting."""

from __future__ import annotations

import pytest

from kalshi_alpha.types import (
    PAYOUT,
    Action,
    Fill,
    Level,
    OrderBook,
    Position,
    Side,
    Trade,
    clamp_price,
)


class TestOrderBook:
    def test_yes_ask_mirrors_the_no_bid(self, mk_book) -> None:
        b = mk_book("T", 40, 42)
        assert b.best_yes_bid == 40
        assert b.best_no_bid == 58
        assert b.best_yes_ask == 42
        assert b.spread == 2

    def test_buying_yes_and_no_always_costs_more_than_a_dollar(self, mk_book) -> None:
        """The structural fact that rules out naive within-market arbitrage."""
        for bid in range(5, 95, 7):
            b = mk_book("T", bid, bid + 2)
            assert b.best_yes_ask + b.best_no_bid == PAYOUT
            cost = b.best_yes_ask + (PAYOUT - b.best_yes_bid)
            assert cost == PAYOUT + b.spread > PAYOUT

    def test_crossed_book_is_flagged(self) -> None:
        crossed = OrderBook.from_levels("T", 0.0, [(60, 10)], [(45, 10)])
        assert crossed.best_yes_bid == 60
        assert crossed.best_yes_ask == 55
        assert crossed.is_crossed

    def test_mid_and_microprice(self) -> None:
        b = OrderBook.from_levels("T", 0.0, [(40, 100)], [(58, 300)])
        assert b.mid == pytest.approx(41.0)
        # Heavier size on the ask pulls fair value toward the bid.
        assert b.microprice < b.mid

    def test_microprice_leans_toward_the_thin_side(self) -> None:
        heavy_bid = OrderBook.from_levels("T", 0.0, [(40, 900)], [(58, 100)])
        assert heavy_bid.microprice > heavy_bid.mid

    def test_imbalance_bounds(self, mk_book) -> None:
        b = mk_book("T", 40, 42)
        assert -1.0 <= b.imbalance <= 1.0

    def test_walk_respects_depth(self) -> None:
        b = OrderBook.from_levels("T", 0.0, [(40, 10), (39, 10)], [(58, 5)])
        filled, cash = b.walk(Side.YES, Action.BUY, 12)
        assert filled == 5  # only 5 offered
        assert cash == 5 * 42

    def test_walk_prices_worse_levels_last(self) -> None:
        b = OrderBook.from_levels("T", 0.0, [], [(58, 10), (57, 10)])
        filled, cash = b.walk(Side.YES, Action.BUY, 15)
        assert filled == 15
        assert cash == 10 * 42 + 5 * 43  # cheapest offer consumed first

    def test_sweep_cost_is_none_when_book_is_thin(self, mk_book) -> None:
        b = mk_book("T", 40, 42, size=5, depth=1)
        assert b.sweep_cost(Side.YES, Action.BUY, 100) is None

    def test_empty_book_is_safe(self) -> None:
        b = OrderBook.from_levels("T", 0.0, [], [])
        assert b.best_yes_bid is None
        assert b.mid is None
        assert not b.is_two_sided
        assert b.walk(Side.YES, Action.BUY, 10) == (0, 0)

    def test_level_rejects_impossible_prices(self) -> None:
        with pytest.raises(ValueError):
            Level(0, 10)
        with pytest.raises(ValueError):
            Level(100, 10)
        with pytest.raises(ValueError):
            Level(50, -1)


class TestPosition:
    """The netting rule that a naive YES-equivalent representation gets wrong."""

    def test_hedged_pair_pays_a_dollar_in_every_state(self) -> None:
        p = Position("T")
        p.apply(Fill("T", 0.0, Side.YES, Action.BUY, 10, 40, fee_cents=17))
        p.apply(Fill("T", 0.0, Side.NO, Action.BUY, 10, 55, fee_cents=17))
        assert p.yes_qty == 0  # no directional exposure
        # Paid 400 + 550 + 34 = 984 for a guaranteed 1000.
        assert p.settle(1) == pytest.approx(16)
        assert p.settle(0) == pytest.approx(16)

    def test_long_no_collects_when_the_event_fails(self) -> None:
        p = Position("T")
        p.apply(Fill("T", 0.0, Side.NO, Action.BUY, 10, 30, fee_cents=15))
        assert p.yes_qty == -10
        assert p.settle(0) == pytest.approx(1000 - 300 - 15)
        assert p.settle(1) == pytest.approx(-315)

    def test_naked_short_yes_owes_the_payout(self) -> None:
        p = Position("T")
        p.apply(Fill("T", 0.0, Side.YES, Action.SELL, 10, 60, fee_cents=17))
        assert p.settle(1) == pytest.approx(600 - 17 - 1000)
        assert p.settle(0) == pytest.approx(600 - 17)

    def test_mark_to_market_agrees_with_settlement_at_the_extremes(self) -> None:
        p = Position("T")
        p.apply(Fill("T", 0.0, Side.YES, Action.BUY, 25, 45, fee_cents=43))
        assert p.mark_to_market(PAYOUT) == pytest.approx(p.settle(1))
        assert p.mark_to_market(0) == pytest.approx(p.settle(0))

    def test_round_trip_loses_exactly_the_fees(self) -> None:
        p = Position("T")
        p.apply(Fill("T", 0.0, Side.YES, Action.BUY, 10, 50, fee_cents=18))
        p.apply(Fill("T", 0.0, Side.YES, Action.SELL, 10, 50, fee_cents=18))
        assert p.yes_qty == 0
        assert p.settle(1) == pytest.approx(-36)
        assert p.settle(0) == pytest.approx(-36)


def test_trade_signing() -> None:
    buy = Trade("T", 0.0, 40, 10, Side.YES)
    sell = Trade("T", 0.0, 40, 10, Side.NO)
    assert buy.signed_size == 10
    assert sell.signed_size == -10


def test_clamp_price_stays_on_the_grid() -> None:
    assert clamp_price(-5) == 1
    assert clamp_price(0.4) == 1
    assert clamp_price(150) == 99
    assert clamp_price(49.6) == 50
