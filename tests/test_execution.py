"""Order management, risk gating, book reconstruction and the paper broker."""

from __future__ import annotations

import pytest

from kalshi_alpha.config import ExecutionConfig, RiskConfig, Settings
from kalshi_alpha.data.kalshi_client import parse_market_meta, parse_orderbook, parse_trade
from kalshi_alpha.execution.oms import OrderManager, OrderRecord, OrderStateError
from kalshi_alpha.execution.paper import PaperBroker
from kalshi_alpha.execution.risk import RiskEngine, RiskViolation, concentration_by_event
from kalshi_alpha.microstructure.book import (
    BookBuilder,
    BookCorruption,
    BookSet,
    SequenceGap,
    apply_delta,
)
from kalshi_alpha.types import (
    Action,
    ArbOpportunity,
    Fill,
    Leg,
    Order,
    OrderStatus,
    OrderType,
    Position,
    Side,
    TimeInForce,
)


# --------------------------------------------------------------------------
class TestBookBuilder:
    def test_snapshot_then_deltas(self) -> None:
        b = BookBuilder("T")
        b.snapshot([(40, 100)], [(58, 100)], seq=1, ts=0.0)
        b.delta(Side.YES, 39, 50, seq=2, ts=1.0)
        book = b.book()
        assert book.best_yes_bid == 40
        assert len(book.yes_bids) == 2

    def test_sequence_gap_raises(self) -> None:
        b = BookBuilder("T")
        b.snapshot([(40, 100)], [(58, 100)], seq=1, ts=0.0)
        with pytest.raises(SequenceGap):
            b.delta(Side.YES, 39, 10, seq=5, ts=1.0)

    def test_negative_size_raises(self) -> None:
        with pytest.raises(BookCorruption):
            apply_delta({40: 10}, 40, -20)

    def test_delta_to_zero_removes_the_level(self) -> None:
        assert apply_delta({40: 10}, 40, -10) == {}

    def test_resync_is_counted(self) -> None:
        b = BookBuilder("T")
        b.snapshot([(40, 100)], [(58, 100)], seq=1, ts=0.0)
        b.snapshot([(41, 100)], [(57, 100)], seq=99, ts=2.0)
        assert b.resyncs == 1
        assert b.seq == 99

    def test_health_reflects_a_one_sided_book(self) -> None:
        b = BookBuilder("T")
        b.snapshot([(40, 100)], [], seq=1, ts=0.0)
        assert not b.healthy

    def test_bookset_aggregates(self) -> None:
        s = BookSet()
        s.snapshot("A", [(40, 10)], [(58, 10)], 1, 0.0)
        s.snapshot("B", [(30, 10)], [(68, 10)], 1, 0.0)
        s.delta("A", Side.YES, 39, 5, 2, 1.0)
        assert len(s) == 2
        assert s.health()["updates"] == 1
        assert set(s.books()) == {"A", "B"}


# --------------------------------------------------------------------------
class TestRiskEngine:
    def test_oversized_order_is_rejected(self) -> None:
        engine = RiskEngine(RiskConfig(max_order_qty=10))
        order = Order("T", Side.YES, Action.BUY, 100, 50)
        decision = engine.check_order(order, {})
        assert not decision
        assert decision.limit == "max_order_qty"

    def test_position_cap_is_enforced(self) -> None:
        engine = RiskEngine(RiskConfig(max_contracts_per_market=50))
        positions = {"T": Position("T", yes_qty=45)}
        assert not engine.check_order(Order("T", Side.YES, Action.BUY, 10, 50), positions)

    def test_kill_switch_blocks_everything(self) -> None:
        engine = RiskEngine()
        engine.kill("manual")
        assert not engine.check_order(Order("T", Side.YES, Action.BUY, 1, 50), {})

    def test_kill_switch_trips_on_drawdown(self) -> None:
        engine = RiskEngine(RiskConfig(kill_switch_drawdown_cents=1000))
        engine.update_equity(10_000)
        engine.update_equity(8_500)
        assert engine.killed

    def test_kill_switch_latches_until_cleared(self) -> None:
        engine = RiskEngine(RiskConfig(kill_switch_drawdown_cents=1000))
        engine.update_equity(10_000)
        engine.update_equity(8_500)
        engine.update_equity(10_000)  # recovery must NOT re-arm it
        assert engine.killed
        engine.reset_kill_switch()
        assert not engine.killed

    def test_a_non_arbitrage_is_refused(self) -> None:
        engine = RiskEngine()
        fake = ArbOpportunity(
            kind="fake", event_ticker="E",
            legs=(Leg("T", Side.YES, Action.BUY, 10, 50),),
            cost_cents=500, worst_case_pnl_cents=-10, best_case_pnl_cents=100,
        )
        decision = engine.check_arbitrage(fake, {})
        assert not decision
        assert decision.limit == "not_arbitrage"

    def test_a_real_arbitrage_is_approved(self) -> None:
        engine = RiskEngine()
        real = ArbOpportunity(
            kind="real", event_ticker="E",
            legs=(Leg("T", Side.YES, Action.BUY, 10, 40),
                  Leg("T", Side.NO, Action.BUY, 10, 55)),
            cost_cents=950, worst_case_pnl_cents=50, best_case_pnl_cents=50,
        )
        assert engine.check_arbitrage(real, {})

    def test_rejections_are_counted_by_limit(self) -> None:
        engine = RiskEngine(RiskConfig(max_order_qty=1))
        for _ in range(3):
            engine.check_order(Order("T", Side.YES, Action.BUY, 50, 50), {})
        assert engine.report()["rejections"]["max_order_qty"] == 3

    def test_approve_or_raise(self) -> None:
        engine = RiskEngine(RiskConfig(max_order_qty=1))
        with pytest.raises(RiskViolation):
            engine.approve_or_raise(Order("T", Side.YES, Action.BUY, 50, 50), {})

    def test_concentration_groups_correlated_markets(self) -> None:
        positions = {"A": Position("A", yes_qty=10), "B": Position("B", yes_qty=20)}
        out = concentration_by_event(positions, {}, {"A": "EVT", "B": "EVT"})
        assert set(out) == {"EVT"}
        assert out["EVT"] == pytest.approx(30 * 50.0)


# --------------------------------------------------------------------------
class TestOrderManager:
    def test_illegal_transition_raises(self) -> None:
        rec = OrderRecord(order=Order("T", Side.YES, Action.BUY, 10, 50))
        rec.transition(OrderStatus.PENDING)
        rec.transition(OrderStatus.CANCELED)
        with pytest.raises(OrderStateError):
            rec.transition(OrderStatus.OPEN)

    def test_overfill_raises(self) -> None:
        rec = OrderRecord(order=Order("T", Side.YES, Action.BUY, 10, 50))
        rec.transition(OrderStatus.PENDING)
        with pytest.raises(OrderStateError):
            rec.apply_fill(Fill("T", 0.0, Side.YES, Action.BUY, 50, 50))

    def test_partial_then_complete(self) -> None:
        rec = OrderRecord(order=Order("T", Side.YES, Action.BUY, 10, 50))
        rec.transition(OrderStatus.PENDING)
        rec.apply_fill(Fill("T", 0.0, Side.YES, Action.BUY, 4, 50))
        assert rec.status is OrderStatus.PARTIALLY_FILLED
        rec.apply_fill(Fill("T", 0.0, Side.YES, Action.BUY, 6, 51))
        assert rec.status is OrderStatus.FILLED
        assert rec.avg_price == pytest.approx((4 * 50 + 6 * 51) / 10)

    def test_client_ids_are_unique(self) -> None:
        oms = OrderManager()
        ids = {oms.new_client_id() for _ in range(500)}
        assert len(ids) == 500

    def test_submit_registers_and_counts(self) -> None:
        oms = OrderManager()
        order = oms.prepare("T", Side.YES, Action.BUY, 10, 50)
        assert oms.submit(order) is not None
        assert oms.risk.open_order_count == 1

    def test_rejected_order_is_recorded(self) -> None:
        oms = OrderManager(RiskEngine(RiskConfig(max_order_qty=1)))
        assert oms.submit(oms.prepare("T", Side.YES, Action.BUY, 100, 50)) is None
        assert len(oms.rejected) == 1

    def test_fill_updates_position_and_cash(self) -> None:
        oms = OrderManager()
        order = oms.prepare("T", Side.YES, Action.BUY, 10, 50)
        oms.submit(order)
        oms.on_ack(order.client_order_id, "x1")
        oms.on_fill(Fill("T", 0.0, Side.YES, Action.BUY, 10, 50, fee_cents=18,
                         order_id=order.client_order_id))
        assert oms.positions["T"].yes_qty == 10
        assert oms.cash_cents == pytest.approx(-518)

    def test_arbitrage_submission_merges_legs_per_market(self) -> None:
        oms = OrderManager()
        opp = ArbOpportunity(
            kind="k", event_ticker="E",
            legs=(
                Leg("A", Side.YES, Action.BUY, 10, 40),
                Leg("A", Side.YES, Action.BUY, 10, 41),  # second price level
                Leg("B", Side.NO, Action.BUY, 20, 55),
            ),
            cost_cents=1910, worst_case_pnl_cents=90, best_case_pnl_cents=90,
        )
        records = oms.submit_arbitrage(opp)
        assert len(records) == 2  # merged into one order per (ticker, side)
        assert {r.order.ticker for r in records} == {"A", "B"}
        a = next(r for r in records if r.order.ticker == "A")
        assert a.order.qty == 20
        assert a.order.price == 41  # the worst price the proof allowed
        assert a.order.tif is TimeInForce.IOC

    def test_reconcile_detects_a_missing_order(self) -> None:
        oms = OrderManager()
        order = oms.prepare("T", Side.YES, Action.BUY, 10, 50)
        oms.submit(order)
        oms.on_ack(order.client_order_id, "x1")
        issues = oms.reconcile([], [])
        assert any("absent at exchange" in msg for msg in issues["orders"])

    def test_reconcile_detects_a_position_mismatch(self) -> None:
        oms = OrderManager()
        oms.positions["T"] = Position("T", yes_qty=10)
        issues = oms.reconcile([], [{"ticker": "T", "position": 4}])
        assert any("local 10 vs exchange 4" in msg for msg in issues["positions"])

    def test_clean_reconciliation_is_empty(self) -> None:
        oms = OrderManager()
        oms.positions["T"] = Position("T", yes_qty=10)
        issues = oms.reconcile([], [{"ticker": "T", "position": 10}])
        assert not issues["orders"] and not issues["positions"]

    def test_unwind_flattens_filled_legs(self) -> None:
        oms = OrderManager()
        order = oms.prepare("A", Side.YES, Action.BUY, 10, 40)
        rec = oms.submit(order)
        assert rec is not None
        oms.on_ack(order.client_order_id, "x")
        oms.on_fill(Fill("A", 0.0, Side.YES, Action.BUY, 10, 40,
                         order_id=order.client_order_id))
        unwind = oms.unwind([rec])
        assert len(unwind) == 1
        assert unwind[0].side is Side.NO  # opposite side flattens the exposure
        assert unwind[0].qty == 10


# --------------------------------------------------------------------------
class TestPaperBroker:
    def test_fills_against_the_book(self, mk_book) -> None:
        broker = PaperBroker(
            settings=Settings(execution=ExecutionConfig(adverse_reject_prob=0.0))
        )
        books = {"T": mk_book("T", 40, 42, size=100)}
        order = broker.oms.prepare("T", Side.YES, Action.BUY, 20, None,
                                   OrderType.MARKET)
        fills = broker.place(order, books)
        assert sum(f.qty for f in fills) == 20
        assert broker.positions()["T"].yes_qty == 20
        assert broker.cash < broker.starting_cash_cents

    def test_risk_rejection_places_nothing(self, mk_book) -> None:
        broker = PaperBroker()
        broker.oms.risk.kill("test")
        order = broker.oms.prepare("T", Side.YES, Action.BUY, 20, None,
                                   OrderType.MARKET)
        assert broker.place(order, {"T": mk_book("T", 40, 42)}) == []

    def test_equity_accounts_for_a_hedged_pair(self, mk_book) -> None:
        broker = PaperBroker(
            settings=Settings(execution=ExecutionConfig(adverse_reject_prob=0.0))
        )
        books = {"T": mk_book("T", 40, 42, size=500)}
        broker.place(broker.oms.prepare("T", Side.YES, Action.BUY, 10, None,
                                        OrderType.MARKET), books)
        broker.place(broker.oms.prepare("T", Side.NO, Action.BUY, 10, None,
                                        OrderType.MARKET), books)
        # Flat exposure, but the pair still holds $1 of guaranteed settlement value.
        assert broker.positions()["T"].yes_qty == 0
        assert broker.equity(books) > broker.cash


# --------------------------------------------------------------------------
class TestClientParsing:
    def test_orderbook_parsing(self) -> None:
        payload = {"yes": [[40, 100], [39, 50]], "no": [[58, 80]]}
        book = parse_orderbook("T", payload, 123.0)
        assert book.best_yes_bid == 40
        assert book.best_yes_ask == 42
        assert book.ts == 123.0

    def test_orderbook_parsing_tolerates_empty_sides(self) -> None:
        book = parse_orderbook("T", {"yes": None, "no": []}, 0.0)
        assert not book.is_two_sided

    def test_trade_parsing(self) -> None:
        t = parse_trade({"ticker": "T", "yes_price": 44, "count": 12,
                         "taker_side": "no", "ts": 5.0})
        assert t.signed_size == -12

    def test_market_meta_greater_or_equal(self) -> None:
        meta = parse_market_meta({
            "ticker": "T", "event_ticker": "E", "strike_type": "greater_or_equal",
            "floor_strike": 3.5,
        })
        assert meta.strike_type == "gte"
        assert meta.payoff(4.0) == 1
        assert meta.payoff(3.0) == 0

    def test_market_meta_between(self) -> None:
        meta = parse_market_meta({
            "ticker": "T", "event_ticker": "E", "strike_type": "between",
            "floor_strike": 2.0, "cap_strike": 3.0,
        })
        assert meta.payoff(2.5) == 1
        assert meta.payoff(3.5) == 0

    def test_unparseable_market_is_excluded_rather_than_guessed(self) -> None:
        meta = parse_market_meta({"ticker": "T", "event_ticker": "E"})
        assert meta.strike_type is None
        with pytest.raises(ValueError):
            meta.payoff(1.0)
